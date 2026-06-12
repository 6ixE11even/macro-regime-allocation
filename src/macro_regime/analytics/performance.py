"""
Backtest performance metrics.

Returns are monthly; the sqrt(12) / x12 factors annualise. Everything that's a
percentage is reported in percent (the original notebook left max-drawdown as a
raw fraction — fixed here so the table reads consistently).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_ANNUALISE = np.sqrt(12)


def sharpe_ratio(returns: np.ndarray) -> float:
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return _ANNUALISE * np.mean(returns) / np.std(returns)


def sortino_ratio(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return np.inf  # no losing months -> undefined downside risk
    return _ANNUALISE * np.mean(returns) / np.std(downside)


def max_drawdown(returns: np.ndarray) -> float:
    """Largest peak-to-trough decline, as a fraction of the peak."""
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak))


def all_metrics(returns: np.ndarray) -> dict[str, float]:
    n = len(returns)
    return {
        "Sharpe Ratio": sharpe_ratio(returns),
        "Sortino Ratio": sortino_ratio(returns),
        "Ann. Return": np.mean(returns) * 12 * 100 if n else 0.0,
        "Ann. Vol": np.std(returns) * _ANNUALISE * 100 if n else 0.0,
        "Max Drawdown": max_drawdown(returns) * 100,
        "Pct Positive": 100 * np.sum(returns > 0) / n if n else 0.0,
    }


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """One row per strategy, sorted by Sharpe. `results[name]['returns']` is an array."""
    rows = []
    for name, result in results.items():
        row = all_metrics(result["returns"])
        row["Strategy"] = name
        rows.append(row)

    cols = ["Strategy", "Sharpe Ratio", "Sortino Ratio", "Ann. Return", "Ann. Vol", "Max Drawdown", "Pct Positive"]
    return pd.DataFrame(rows)[cols].sort_values("Sharpe Ratio", ascending=False).reset_index(drop=True)
