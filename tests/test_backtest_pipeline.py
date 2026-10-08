"""Regression tests for data alignment and trading-accounting conventions."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import build_backtest_results as backtest  # noqa: E402


class DatasetAlignmentTests(unittest.TestCase):
    def test_features_end_before_the_target_session(self) -> None:
        """The first feature window must stop at the target's prior close."""
        dates = pd.bdate_range("2020-01-01", periods=backtest.WINDOW + 5)
        prices = 100 * np.exp(np.arange(len(dates)) * 0.001)
        frame = pd.DataFrame({"date": dates, "adjusted_close": prices})

        dataset = backtest.build_forecast_dataset(frame)
        log_returns = np.diff(np.log(prices))

        np.testing.assert_allclose(dataset.x[0], log_returns[: backtest.WINDOW])
        self.assertAlmostEqual(float(dataset.y[0]), float(log_returns[backtest.WINDOW]), places=7)
        self.assertAlmostEqual(dataset.origin_price[0], prices[backtest.WINDOW])
        self.assertAlmostEqual(dataset.target_price[0], prices[backtest.WINDOW + 1])


class TradingAccountingTests(unittest.TestCase):
    def test_turnover_cost_is_charged_on_each_position_change(self) -> None:
        realized = np.zeros(4)
        positions = np.array([1.0, 1.0, 0.0, -1.0])

        returns = backtest.calculate_net_strategy_returns(realized, positions)

        expected = np.array(
            [-backtest.TRANSACTION_COST, 0.0, -backtest.TRANSACTION_COST, -backtest.TRANSACTION_COST]
        )
        np.testing.assert_allclose(returns, expected)


class PublishedArtifactTests(unittest.TestCase):
    def test_published_result_has_complete_test_series(self) -> None:
        artifact_path = PROJECT_ROOT / "assets" / "data" / "backtest-results.json"
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))

        observations = payload["metadata"]["test_observations"]
        self.assertEqual(observations, 502)
        self.assertEqual(len(payload["dates"]), observations)
        self.assertEqual(payload["dates"][0], "2023-01-03")
        self.assertEqual(payload["dates"][-1], "2024-12-31")
        for model in payload["models"].values():
            self.assertEqual(len(model["equity"]), observations)


if __name__ == "__main__":
    unittest.main()
