# Phase ODE: reproducible SPY benchmark

This repository evaluates whether a small Neural ODE can forecast SPY's next-session return. The project site is generated from a checked-in, deterministic backtest artifact—not browser-side sample data.

## Honest headline result

For the locked 2023–2024 test period (502 sessions), the Neural ODE returned **13.3%** after modeled trading costs, versus **57.6%** for buy-and-hold. Its maximum drawdown was **-3.7%** versus **-10.0%** for buy-and-hold, but it had only **19% average gross exposure**. The model therefore reduced risk largely by participating less; it did not beat the passive benchmark.

These values come from `results.json`. Re-run `run_backtest.py` to reproduce them.

## Experiment

- Instrument: SPY auto-adjusted close.
- Inputs: trailing 20 daily log returns.
- Training: 2015-02-03 through 2021-12-31.
- Validation: calendar 2022, used for neural-model early stopping and signal threshold selection.
- Locked test: 2023-01-03 through 2024-12-31.
- Execution: a forecast formed with data through close `t` is applied to the close-to-close return from `t` to `t+1`.
- Cost model: 5 basis points per one-way change in position.
- Seed: 42, with deterministic PyTorch operations.
- Benchmarks: LSTM, Ridge AR(20), buy-and-hold, and cash.

The Neural ODE encodes the return window to a 16-dimensional state, evolves it across continuous depth with a learned vector field and four differentiable RK4 steps, then reads out the next log return.

## Reproduce

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run_backtest.py
python -m http.server 8000
```

Open `http://localhost:8000`. The default run uses the checked-in `data/spy_adjusted.csv` snapshot. Use `python run_backtest.py --refresh-data` to replace it from Yahoo Finance before rebuilding.

## Files that power the site

- `run_backtest.py`: data preparation, model training, validation selection, trading simulation, and metrics.
- `data/spy_adjusted.csv`: fixed adjusted-close snapshot used by the published run.
- `results.json`: generated, browser-readable backtest artifact.
- `index.html`, `styles.css`, `script.js`: static GitHub Pages interface.

## Limitations

This is a single instrument and a single two-year test regime. Threshold selection uses one validation year. The cost model omits bid/ask spread and market impact. Adjusted prices account for distributions, but the simulation does not model taxes, borrow constraints, financing costs, or execution slippage beyond the fixed cost. No statistical-significance claim is made.

The older example modules remain for historical context, but they are not the source of the published site results. In particular, the old comparison script used synthetic Neural ODE predictions; the current site does not consume those outputs.
