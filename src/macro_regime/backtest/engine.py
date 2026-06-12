"""
Walk-forward backtest.

At each month t we look back `window` months, fit whatever the strategy needs,
size positions, and earn the realised next-month return. Seven strategies share
one rolling loop so they see identical inputs:

    MVO / Naive / Ridge  x  long-only / long-short    +    equal-weight benchmark

Naive and Ridge need at least two regimes in the lookback to be meaningful, so
they sit out (no position recorded) on windows that don't have them — which is
why their track records can be a few months shorter than MVO's.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from macro_regime.models import allocation
from macro_regime.models.forecast import NaiveForecastModel, RidgeRegressionModel

logger = logging.getLogger(__name__)

DEFAULT_WINDOW = 48  # 4 years of monthly data


def create_regime_probs(regime: int, n_regimes: int, confidence: float = 0.8) -> list[float]:
    """Spread a hard regime label into a soft vector: `confidence` on the called
    regime, the rest shared equally. Keeps the optimiser from betting everything
    on a single noisy classification."""
    probs = np.full(n_regimes, (1 - confidence) / (n_regimes - 1))
    probs[int(regime)] = confidence
    return probs.tolist()


def run_backtests(
    merged: pd.DataFrame,
    asset_cols: list[str],
    component_names: list[str] | None = None,
    window: int = DEFAULT_WINDOW,
) -> dict[str, dict]:
    """Run every strategy over `merged` (returns + 'regime' + 'regime_probs', and
    the PCA `component_names` if Ridge is wanted). Returns {name: {returns, dates}}."""
    use_ridge = bool(component_names) and all(c in merged.columns for c in component_names)
    n_assets = len(asset_cols)
    eq_weights = np.ones(n_assets) / n_assets

    # Accumulators — each strategy keeps its own (returns, dates) so they can differ in length.
    series: dict[str, dict[str, list]] = {
        name: {"returns": [], "dates": []}
        for name in ["MVO_LongOnly", "MVO_LongShort", "Naive_LongOnly", "Naive_LongShort",
                     "Ridge_LongOnly", "Ridge_LongShort", "EqualWeight"]
    }

    def record(name: str, ret: float, date) -> None:
        series[name]["returns"].append(ret)
        series[name]["dates"].append(date)

    for t in range(window, len(merged)):
        win = merged.iloc[t - window:t]
        cur = merged.iloc[t]
        date = cur["date"]
        realised = cur[asset_cols].to_numpy(dtype=float)

        train_ret = win[asset_cols].to_numpy(dtype=float)
        sample_mean = train_ret.mean(axis=0)
        sample_cov = np.cov(train_ret.T) + np.eye(n_assets) * 1e-6
        regimes = win["regime"].to_numpy()
        probs = np.asarray(cur["regime_probs"], dtype=float)

        # 1-2) Mean-variance on the rolling sample moments.
        record("MVO_LongOnly", allocation.long_only(sample_mean, sample_cov) @ realised, date)
        record("MVO_LongShort", allocation.long_short(sample_mean, sample_cov) @ realised, date)

        # 7) Equal-weight benchmark — always on.
        record("EqualWeight", eq_weights @ realised, date)

        # Regime-aware strategies need >=2 regimes in the lookback.
        if len(np.unique(regimes)) < 2:
            continue

        # 3-4) Naive regime-mean forecaster, sized with that regime's covariance.
        naive = NaiveForecastModel().fit(win[asset_cols], regimes)
        exp_ret = naive.predict(probs)
        regime_cov = naive.regime_covariance(int(probs.argmax()))
        if regime_cov is None:
            regime_cov = sample_cov
        record("Naive_LongOnly", allocation.long_only(exp_ret, regime_cov) @ realised, date)
        record("Naive_LongShort", allocation.long_short(exp_ret, regime_cov) @ realised, date)

        # 5-6) Ridge factor forecaster (per-regime), sized with the sample covariance.
        if use_ridge:
            ridge = RidgeRegressionModel(alpha=1.0).fit(win[component_names], win[asset_cols], regimes)
            exp_ret = ridge.predict(cur[component_names].to_numpy(dtype=float), probs)
            record("Ridge_LongOnly", allocation.long_only(exp_ret, sample_cov) @ realised, date)
            record("Ridge_LongShort", allocation.long_short(exp_ret, sample_cov) @ realised, date)

    # Drop any strategy that never recorded (e.g. Ridge with no features) and arrayify.
    results = {
        name: {"returns": np.array(d["returns"]), "dates": d["dates"]}
        for name, d in series.items() if d["returns"]
    }
    logger.info("Backtest done: %d strategies, %d months", len(results), len(merged) - window)
    return results


LONG_ONLY = ["MVO_LongOnly", "Naive_LongOnly", "Ridge_LongOnly", "EqualWeight"]
LONG_SHORT = ["MVO_LongShort", "Naive_LongShort", "Ridge_LongShort"]
