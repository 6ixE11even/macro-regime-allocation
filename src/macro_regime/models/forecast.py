"""
Regime-conditional return forecasters.

Both models learn one set of parameters per regime, then blend across regimes at
prediction time using the current month's regime probabilities:

  * NaiveForecastModel   — expected return = the regime's historical mean return.
  * RidgeRegressionModel — expected return = a per-regime ridge fit on PCA factors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class NaiveForecastModel:
    """Per-regime sample mean/cov. Cheap, surprisingly hard to beat."""

    def __init__(self):
        self.regime_stats: dict[int, dict] = {}
        self.n_assets_: int | None = None

    def fit(self, returns: pd.DataFrame, regimes: np.ndarray) -> "NaiveForecastModel":
        r = returns.values
        self.n_assets_ = r.shape[1]

        for regime in np.unique(regimes):
            block = r[regimes == regime]
            if len(block) == 0:
                continue
            mean = block.mean(axis=0)
            std = block.std(axis=0) + 1e-8  # floor so Sharpe never divides by zero
            # A regime seen only once (e.g. the COVID crisis month) has no estimable
            # covariance — leave it None and let the caller fall back to the sample cov.
            cov = np.cov(block.T) + np.eye(self.n_assets_) * 1e-6 if len(block) > 1 else None
            self.regime_stats[int(regime)] = {"mean": mean, "std": std, "sharpe": mean / std, "cov": cov}
        return self

    def predict(self, regime_probs: np.ndarray) -> np.ndarray:
        """Probability-weighted blend of the per-regime mean returns."""
        expected = np.zeros(self.n_assets_)
        for regime, prob in enumerate(regime_probs):
            stats = self.regime_stats.get(regime)
            if stats is not None:
                expected += prob * stats["mean"]
        return expected

    def regime_covariance(self, regime: int) -> np.ndarray | None:
        stats = self.regime_stats.get(int(regime))
        return stats["cov"] if stats else None


class RidgeRegressionModel:
    """One ridge regression per regime, mapping PCA factors -> asset returns."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.models: dict[int, Ridge] = {}
        self.scalers: dict[int, StandardScaler] = {}
        self.n_assets_: int | None = None

    def fit(self, features: pd.DataFrame, returns: pd.DataFrame, regimes: np.ndarray) -> "RidgeRegressionModel":
        x, y = features.values, returns.values
        self.n_assets_ = y.shape[1]

        for regime in np.unique(regimes):
            mask = regimes == regime
            # Need enough months in a regime for a stable fit; 10 is the original floor.
            if mask.sum() <= 10:
                continue
            scaler = StandardScaler().fit(x[mask])
            model = Ridge(alpha=self.alpha).fit(scaler.transform(x[mask]), y[mask])
            self.scalers[int(regime)] = scaler
            self.models[int(regime)] = model
        return self

    def predict(self, features: np.ndarray, regime_probs: np.ndarray) -> np.ndarray:
        preds, weights = [], []
        for regime, prob in enumerate(regime_probs):
            if prob > 0 and regime in self.models:
                x_scaled = self.scalers[regime].transform(features.reshape(1, -1))
                preds.append(self.models[regime].predict(x_scaled)[0])
                weights.append(prob)

        if not preds:
            return np.zeros(self.n_assets_)
        weights = np.array(weights) / np.sum(weights)
        return np.sum(np.array(preds) * weights[:, None], axis=0)
