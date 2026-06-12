"""
Human-readable names for the detected regimes.

These labels come from reading the cluster profiles (mean z-scores per indicator)
in the original analysis: regime 0 is the crisis cluster, 1-4 are the "typical"
sub-regimes ordered by KMeans. They're descriptive, not load-bearing — the
backtest only ever sees the integer regime id. If a run produces more clusters
than we have names for, callers fall back to "Regime {i}" via `regime_label`.
"""
from __future__ import annotations

REGIME_NAMES: dict[int, str] = {
    0: "Economic Collapse",      # extreme negative across output/labour/consumption (e.g. Apr-2020)
    1: "Sluggish Growth",        # below-average output, low inflation
    2: "Elevated Inflation",     # high inflation, low rates
    3: "Broad-Based Expansion",  # strong output, housing, consumption
    4: "Uneven Recovery",        # flat output, housing lagging
}


def regime_label(regime_id: int) -> str:
    return REGIME_NAMES.get(int(regime_id), f"Regime {int(regime_id)}")
