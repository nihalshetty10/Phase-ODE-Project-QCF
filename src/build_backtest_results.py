"""Build the reproducible SPY forecast and trading benchmark used by the site.

The script trains models through 2021, uses 2022 only for early stopping and
signal-threshold selection, then reports the untouched 2023-2024 test period.
Predictions are formed after each close and applied to the following session.

Run from the repository root:
    python src/build_backtest_results.py
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "spy_adjusted_close_2015_2024.csv"
RESULTS_PATH = PROJECT_ROOT / "assets" / "data" / "backtest-results.json"
SEED = 42
WINDOW = 20
INITIAL_CAPITAL = 10_000.0
TRANSACTION_COST = 0.0005  # 5 bps per one-way change in position
TRAIN_END = "2021-12-31"
VALIDATION_END = "2022-12-31"
TEST_END = "2024-12-31"


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def download_data() -> pd.DataFrame:
    import yfinance as yf

    downloaded = yf.download(
        "SPY",
        start="2015-01-01",
        end="2025-01-01",
        auto_adjust=True,
        progress=False,
    )
    if downloaded.empty:
        raise RuntimeError("Yahoo Finance returned no SPY observations")
    close = downloaded["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    frame = close.rename("adjusted_close").to_frame().reset_index()
    frame.columns = ["date", "adjusted_close"]
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PATH, index=False)
    return frame


def load_data(refresh: bool) -> pd.DataFrame:
    frame = download_data() if refresh or not DATA_PATH.exists() else pd.read_csv(DATA_PATH)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna().drop_duplicates("date").sort_values("date")
    frame = frame[frame["date"] <= TEST_END].reset_index(drop=True)
    if frame.empty or frame["date"].max() < pd.Timestamp(TEST_END):
        raise ValueError(f"Data snapshot must include observations through {TEST_END}")
    return frame


@dataclass
class ForecastDataset:
    """Aligned model inputs, next-session targets, prices, and dates."""

    x: np.ndarray
    y: np.ndarray
    origin_price: np.ndarray
    target_price: np.ndarray
    origin_date: np.ndarray
    target_date: np.ndarray


def build_forecast_dataset(frame: pd.DataFrame) -> ForecastDataset:
    """Construct 20-return features without exposing the target session."""
    prices = frame["adjusted_close"].to_numpy(dtype=np.float64)
    dates = frame["date"].dt.strftime("%Y-%m-%d").to_numpy()
    log_returns = np.diff(np.log(prices))
    x, y, origin_price, target_price, origin_date, target_date = [], [], [], [], [], []
    for target_index in range(WINDOW + 1, len(prices)):
        # At origin_index we know returns through that close; y is the next return.
        origin_index = target_index - 1
        x.append(log_returns[origin_index - WINDOW : origin_index])
        y.append(log_returns[origin_index])
        origin_price.append(prices[origin_index])
        target_price.append(prices[target_index])
        origin_date.append(dates[origin_index])
        target_date.append(dates[target_index])
    return ForecastDataset(
        x=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        origin_price=np.asarray(origin_price),
        target_price=np.asarray(target_price),
        origin_date=np.asarray(origin_date),
        target_date=np.asarray(target_date),
    )


class LatentDynamics(nn.Module):
    """Learned derivative function for the Neural ODE latent state."""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.Tanh(),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class RK4ODEBlock(nn.Module):
    """Differentiable fixed-step integration over continuous depth t=[0,1]."""

    def __init__(self, func: LatentDynamics, steps: int = 4):
        super().__init__()
        self.func = func
        self.steps = steps

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        step = 1.0 / self.steps
        for _ in range(self.steps):
            k1 = self.func(state)
            k2 = self.func(state + step * k1 / 2)
            k3 = self.func(state + step * k2 / 2)
            k4 = self.func(state + step * k3)
            state = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return state


class NeuralODEForecaster(nn.Module):
    def __init__(self, window: int = WINDOW, hidden: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(window, hidden), nn.Tanh())
        self.ode = RK4ODEBlock(LatentDynamics(hidden), steps=4)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.ode(self.encoder(x))).squeeze(-1)


class LSTMForecaster(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x.unsqueeze(-1))
        return self.readout(output[:, -1]).squeeze(-1)


def train_torch_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> tuple[nn.Module, dict]:
    x_train = torch.tensor(train_x)
    y_train = torch.tensor(train_y)
    x_validation = torch.tensor(validation_x)
    y_validation = torch.tensor(validation_y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=1.0)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    best_epoch = 0
    patience = 35
    stale = 0

    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(300):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), 128):
            batch = order[start : start + 128]
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(model(x_validation), y_validation).item()
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "validation_huber": best_loss}


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x)).numpy().astype(float)


def signal_positions(predictions: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(predictions > threshold, 1.0, np.where(predictions < -threshold, -1.0, 0.0))


def calculate_net_strategy_returns(realized_log_returns: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Apply forecast positions and charge 5 bps for each unit of turnover."""
    simple_returns = np.expm1(realized_log_returns)
    prior_positions = np.r_[0.0, positions[:-1]]
    costs = TRANSACTION_COST * np.abs(positions - prior_positions)
    return positions * simple_returns - costs


def sharpe(returns: np.ndarray) -> float:
    volatility = np.std(returns, ddof=1)
    return float(np.mean(returns) / volatility * np.sqrt(252)) if volatility > 0 else 0.0


def choose_threshold(predictions: np.ndarray, realized: np.ndarray) -> tuple[float, list[dict]]:
    candidates = [0.0, 0.00025, 0.0005, 0.001, 0.0015, 0.002, 0.003]
    rows = []
    for threshold in candidates:
        positions = signal_positions(predictions, threshold)
        returns = calculate_net_strategy_returns(realized, positions)
        rows.append({"threshold": threshold, "sharpe": sharpe(returns)})
    best = max(rows, key=lambda row: (row["sharpe"], row["threshold"]))
    return float(best["threshold"]), rows


def drawdown(equity: np.ndarray) -> np.ndarray:
    return equity / np.maximum.accumulate(equity) - 1.0


def rolling_sharpe(returns: np.ndarray, window: int = 63) -> np.ndarray:
    values = np.full(len(returns), np.nan)
    for index in range(window - 1, len(returns)):
        values[index] = sharpe(returns[index - window + 1 : index + 1])
    return values


def safe_number(value: float | int) -> float | int | None:
    return value if np.isfinite(value) else None


def calculate_performance_metrics(returns: np.ndarray, positions: np.ndarray | None = None) -> dict:
    """Calculate annualized and path-dependent metrics for one return series."""
    equity = np.cumprod(1.0 + returns)
    years = len(returns) / 252
    total_return = equity[-1] - 1
    cagr = equity[-1] ** (1 / years) - 1 if years > 0 and equity[-1] > 0 else np.nan
    annual_volatility = np.std(returns, ddof=1) * np.sqrt(252)
    downside = returns[returns < 0]
    downside_deviation = np.std(downside, ddof=1) * np.sqrt(252) if len(downside) > 1 else np.nan
    sortino = np.mean(returns) * 252 / downside_deviation if downside_deviation > 0 else np.nan
    max_drawdown = float(np.min(drawdown(equity)))
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan
    result = {
        "total_return": safe_number(float(total_return)),
        "cagr": safe_number(float(cagr)),
        "annual_volatility": safe_number(float(annual_volatility)),
        "sharpe": safe_number(sharpe(returns)),
        "sortino": safe_number(float(sortino)),
        "max_drawdown": safe_number(max_drawdown),
        "calmar": safe_number(float(calmar)),
        "positive_days": safe_number(float(np.mean(returns > 0))),
        "final_value": safe_number(float(INITIAL_CAPITAL * equity[-1])),
    }
    if positions is not None:
        previous = np.r_[0.0, positions[:-1]]
        result.update(
            trades=int(np.sum(positions != previous)),
            gross_exposure=float(np.mean(np.abs(positions))),
            turnover=float(np.sum(np.abs(positions - previous))),
        )
    return result


def serialize_series(values: np.ndarray, digits: int = 6) -> list:
    return [None if not np.isfinite(value) else round(float(value), digits) for value in values]


def build_model_result(
    predictions: np.ndarray,
    realized: np.ndarray,
    origin_prices: np.ndarray,
    threshold: float,
) -> dict:
    positions = signal_positions(predictions, threshold)
    returns = calculate_net_strategy_returns(realized, positions)
    equity = INITIAL_CAPITAL * np.cumprod(1.0 + returns)
    predicted_prices = origin_prices * np.exp(predictions)
    return {
        "predicted_price": serialize_series(predicted_prices, 4),
        "predicted_return": serialize_series(predictions, 8),
        "position": serialize_series(positions, 0),
        "daily_return": serialize_series(returns, 8),
        "equity": serialize_series(equity, 2),
        "drawdown": serialize_series(drawdown(equity), 6),
        "rolling_sharpe": serialize_series(rolling_sharpe(returns), 4),
        "metrics": calculate_performance_metrics(returns, positions),
        "forecast_metrics": {
            "rmse_bps": float(np.sqrt(np.mean((predictions - realized) ** 2)) * 10_000),
            "directional_accuracy": float(np.mean(np.sign(predictions) == np.sign(realized))),
        },
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-data", action="store_true", help="Download the fixed SPY snapshot again")
    args = parser.parse_args()
    seed_everything()
    frame = load_data(args.refresh_data)
    data = build_forecast_dataset(frame)
    target_dates = pd.to_datetime(data.target_date)
    train_mask = target_dates <= TRAIN_END
    validation_mask = (target_dates > TRAIN_END) & (target_dates <= VALIDATION_END)
    test_mask = (target_dates > VALIDATION_END) & (target_dates <= TEST_END)

    feature_mean = data.x[train_mask].mean(axis=0)
    feature_std = data.x[train_mask].std(axis=0)
    feature_std[feature_std == 0] = 1
    target_mean = float(data.y[train_mask].mean())
    target_std = float(data.y[train_mask].std())
    x_scaled = ((data.x - feature_mean) / feature_std).astype(np.float32)
    y_scaled = ((data.y - target_mean) / target_std).astype(np.float32)

    model_specs = {
        "neural_ode": NeuralODEForecaster(),
        "lstm": LSTMForecaster(),
    }
    forecasts: dict[str, dict] = {}
    training_details: dict[str, dict] = {}
    for name, model in model_specs.items():
        model, details = train_torch_model(
            model,
            x_scaled[train_mask],
            y_scaled[train_mask],
            x_scaled[validation_mask],
            y_scaled[validation_mask],
        )
        validation_prediction = predict(model, x_scaled[validation_mask]) * target_std + target_mean
        test_prediction = predict(model, x_scaled[test_mask]) * target_std + target_mean
        threshold, threshold_scores = choose_threshold(validation_prediction, data.y[validation_mask])
        forecasts[name] = {"test": test_prediction, "threshold": threshold}
        training_details[name] = {**details, "threshold_search": threshold_scores}

    ridge_candidates = [0.01, 0.1, 1.0, 10.0, 100.0]
    ridge_rows = []
    for alpha in ridge_candidates:
        ridge = Ridge(alpha=alpha).fit(x_scaled[train_mask], y_scaled[train_mask])
        prediction = ridge.predict(x_scaled[validation_mask]) * target_std + target_mean
        mse = float(np.mean((prediction - data.y[validation_mask]) ** 2))
        ridge_rows.append((mse, alpha, ridge, prediction))
    _, best_alpha, ridge, ridge_validation_prediction = min(ridge_rows, key=lambda row: row[0])
    ridge_test_prediction = ridge.predict(x_scaled[test_mask]) * target_std + target_mean
    ridge_threshold, ridge_threshold_scores = choose_threshold(ridge_validation_prediction, data.y[validation_mask])
    forecasts["ridge_ar"] = {"test": ridge_test_prediction, "threshold": ridge_threshold}
    training_details["ridge_ar"] = {
        "alpha": best_alpha,
        "threshold_search": ridge_threshold_scores,
    }

    test_realized = data.y[test_mask].astype(float)
    test_prices = data.target_price[test_mask]
    test_origin_prices = data.origin_price[test_mask]
    result_models = {
        name: build_model_result(spec["test"], test_realized, test_origin_prices, spec["threshold"])
        for name, spec in forecasts.items()
    }

    buy_hold_returns = np.expm1(test_realized)
    buy_hold_equity = INITIAL_CAPITAL * np.cumprod(1.0 + buy_hold_returns)
    result_models["buy_hold"] = {
        "daily_return": serialize_series(buy_hold_returns, 8),
        "equity": serialize_series(buy_hold_equity, 2),
        "drawdown": serialize_series(drawdown(buy_hold_equity), 6),
        "rolling_sharpe": serialize_series(rolling_sharpe(buy_hold_returns), 4),
        "metrics": calculate_performance_metrics(buy_hold_returns),
    }
    cash_returns = np.zeros_like(test_realized)
    result_models["cash"] = {
        "daily_return": serialize_series(cash_returns, 8),
        "equity": serialize_series(np.full(len(cash_returns), INITIAL_CAPITAL), 2),
        "drawdown": serialize_series(np.zeros(len(cash_returns)), 6),
        "rolling_sharpe": serialize_series(np.zeros(len(cash_returns)), 4),
        "metrics": calculate_performance_metrics(cash_returns),
    }

    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "symbol": "SPY",
            "instrument": "SPDR S&P 500 ETF Trust",
            "data_source": "Yahoo Finance via yfinance; auto-adjusted close",
            "data_snapshot_start": frame["date"].min().strftime("%Y-%m-%d"),
            "data_snapshot_end": frame["date"].max().strftime("%Y-%m-%d"),
            "train_period": f"{data.target_date[train_mask][0]} to {data.target_date[train_mask][-1]}",
            "validation_period": f"{data.target_date[validation_mask][0]} to {data.target_date[validation_mask][-1]}",
            "test_period": f"{data.target_date[test_mask][0]} to {data.target_date[test_mask][-1]}",
            "test_observations": int(test_mask.sum()),
            "window": WINDOW,
            "transaction_cost_bps": TRANSACTION_COST * 10_000,
            "initial_capital": INITIAL_CAPITAL,
            "seed": SEED,
            "timing": "Signals use data through close t and are applied to close-to-close return t+1.",
            "selection": "2022 validation data selects early stopping and each model's neutral-zone threshold; 2023-2024 is untouched until final evaluation.",
        },
        "dates": data.target_date[test_mask].tolist(),
        "actual_price": serialize_series(test_prices, 4),
        "models": result_models,
        "training": training_details,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    for name, result in result_models.items():
        summary = result["metrics"]
        print(
            f"{name:12s} return={summary['total_return']:.2%} "
            f"sharpe={summary['sharpe']:.2f} drawdown={summary['max_drawdown']:.2%}"
        )


if __name__ == "__main__":
    main()
