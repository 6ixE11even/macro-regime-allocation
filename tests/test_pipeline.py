"""Fast sanity checks — the kind of thing I'd want to fail loudly if a refactor
broke the data contract or the optimiser constraints."""
from __future__ import annotations

import numpy as np

from macro_regime import config
from macro_regime.backtest.engine import create_regime_probs
from macro_regime.data.fredmd_loader import FredMDLoader
from macro_regime.models import allocation


def test_fredmd_loader_clean_panel():
    _, stationary, meta = FredMDLoader(config.FRED_MD_CSV).load()
    assert stationary.shape[0] > 700           # decades of monthly data
    assert stationary.isna().sum().sum() == 0  # nothing left unfilled
    assert len(meta) == stationary.shape[1]


def test_long_only_weights_are_a_simplex():
    np.random.seed(0)
    mu = np.random.randn(8)
    cov = np.eye(8)
    w = allocation.long_only(mu, cov)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert (w >= -1e-9).all()  # no shorts


def test_long_short_respects_gross_leverage():
    np.random.seed(1)
    mu = np.random.randn(8)
    cov = np.eye(8)
    w = allocation.long_short(mu, cov)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)       # net 100%
    assert np.sum(np.abs(w)) <= 2.0 + 1e-6           # gross <= 200%


def test_regime_probs_sum_to_one():
    probs = create_regime_probs(regime=2, n_regimes=5, confidence=0.8)
    assert np.isclose(sum(probs), 1.0)
    assert np.isclose(probs[2], 0.8)
