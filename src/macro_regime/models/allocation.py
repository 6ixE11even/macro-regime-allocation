"""
Mean-variance position sizing.

Both sizers maximise the usual utility  wᵀμ − λ·wᵀΣw  via SLSQP. They differ only
in the feasible set:

  * long_only  — weights in [0, 1], fully invested (Σw = 1).
  * long_short — weights in [-1, 1], net Σw = 1, gross Σ|w| ≤ 2 (i.e. up to 150/50).

If the optimiser fails to converge we fall back to equal weight rather than
returning something degenerate.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _solve(objective, constraints, bounds, n_assets: int) -> np.ndarray:
    w0 = np.ones(n_assets) / n_assets
    result = minimize(objective, w0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 1000})
    return result.x if result.success else w0


def long_only(expected_returns: np.ndarray, cov: np.ndarray, risk_aversion: float = 1.0) -> np.ndarray:
    n = len(expected_returns)

    def neg_utility(w):
        return -(w @ expected_returns - risk_aversion * (w @ cov @ w))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    return _solve(neg_utility, constraints, bounds, n)


def long_short(expected_returns: np.ndarray, cov: np.ndarray, risk_aversion: float = 1.0) -> np.ndarray:
    n = len(expected_returns)

    def neg_utility(w):
        return -(w @ expected_returns - risk_aversion * (w @ cov @ w))

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},          # net exposure = 100%
        {"type": "ineq", "fun": lambda w: 2.0 - np.sum(np.abs(w))},  # gross exposure <= 200%
    ]
    bounds = [(-1.0, 1.0)] * n
    return _solve(neg_utility, constraints, bounds, n)
