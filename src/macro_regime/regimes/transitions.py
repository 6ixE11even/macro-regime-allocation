"""
Regime persistence, as a Markov chain.

The multi-period optimiser needs a forecast for each month over its horizon, not
just the next one. The regime machinery already gives a distribution over states
today; treating the regime sequence as a first-order Markov chain is the cheapest
honest way to roll that distribution forward:

    p_{t+h} = p_t · Pʰ

and then blend the per-regime mean returns under p_{t+h} to get the expected
return h months out. Because macro regimes are persistent (the diagonal of P
dominates), the projected forecast decays smoothly toward the chain's stationary
distribution rather than being assumed to either hold forever or vanish after one
month — and that decay profile is precisely what tells the optimiser how hard to
trade today.

First-order is an assumption, not a finding: regime durations in the data are not
truly geometric, and a semi-Markov/duration-aware model would fit the tails
better. It is enough to give the horizon a defensible shape, which is all the
optimiser needs.
"""
from __future__ import annotations

import numpy as np


def transition_matrix(regimes: np.ndarray, n_regimes: int, smoothing: float = 1.0) -> np.ndarray:
    """Row-stochastic transition matrix P[i, j] = P(next = j | current = i).

    Laplace `smoothing` keeps rows well-defined for regimes that are rare or never
    seen leaving a state in the lookback window — the crisis regime is exactly this
    case, sometimes appearing as a single month. With smoothing=1 an unobserved row
    degrades to uniform rather than producing NaNs.
    """
    labels = np.asarray(regimes, dtype=int).ravel()
    counts = np.full((n_regimes, n_regimes), float(smoothing))

    for current, nxt in zip(labels[:-1], labels[1:], strict=True):
        if 0 <= current < n_regimes and 0 <= nxt < n_regimes:
            counts[current, nxt] += 1.0

    return counts / counts.sum(axis=1, keepdims=True)


def project_probabilities(probs: np.ndarray, matrix: np.ndarray, horizon: int) -> np.ndarray:
    """Roll a regime distribution forward, returning (horizon, n_regimes).

    Row h is the distribution h+1 months ahead, i.e. p·P^(h+1).
    """
    p = np.asarray(probs, dtype=float).ravel()
    p = p / p.sum() if p.sum() > 0 else np.full_like(p, 1.0 / len(p))

    out = np.empty((horizon, len(p)))
    for h in range(horizon):
        p = p @ matrix
        out[h] = p
    return out


def expected_return_path(
    probs: np.ndarray,
    matrix: np.ndarray,
    regime_means: dict[int, np.ndarray],
    horizon: int,
    n_assets: int,
) -> np.ndarray:
    """(horizon, n_assets) forecast path — the direct input to `solve_multi_period`.

    Each row is the regime-probability-weighted blend of per-regime mean returns at
    that horizon. Regimes with no estimated mean in the lookback contribute nothing,
    and their probability mass is renormalised across the regimes that do, so a
    thinly-observed state can't silently drag the whole forecast toward zero.
    """
    projected = project_probabilities(probs, matrix, horizon)
    path = np.zeros((horizon, n_assets))

    for h in range(horizon):
        weight_used = 0.0
        for regime, prob in enumerate(projected[h]):
            mean = regime_means.get(int(regime))
            if mean is None or prob <= 0:
                continue
            path[h] += prob * np.asarray(mean, dtype=float)
            weight_used += prob
        if weight_used > 0:
            path[h] /= weight_used

    return path


def stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    """Long-run regime mix — the left eigenvector of P for eigenvalue 1.

    Not used by the optimiser; handy for sanity-checking that an estimated chain
    isn't degenerate (e.g. all mass collapsing onto one absorbing state).
    """
    vals, vecs = np.linalg.eig(matrix.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    stationary = np.real(vecs[:, idx])
    stationary = np.abs(stationary)
    return stationary / stationary.sum()


def expected_duration(matrix: np.ndarray) -> np.ndarray:
    """Expected months spent in each regime before leaving, 1/(1 − P[i,i]).

    A quick readability check on whether the fitted chain says what the macro
    narrative says: expansions should persist for years, crises for months.
    """
    diag = np.clip(np.diag(matrix), 0.0, 1.0 - 1e-12)
    return 1.0 / (1.0 - diag)
