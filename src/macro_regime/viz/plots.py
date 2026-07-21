"""
Plots: strategy equity curves and a regime timeline.

Equity curves are volatility-targeted to a common 10% annual vol before plotting,
so the comparison is about *shape* (timing, drawdowns) rather than who happened to
run the most leverage.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from macro_regime.regimes.naming import regime_label

_ANNUALISE = np.sqrt(12)


def plot_cumulative_returns(results: dict[str, dict], title: str, vol_target: float = 0.10) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 6.5))

    for name, result in results.items():
        returns = np.asarray(result["returns"], dtype=float)
        if returns.size == 0:
            continue
        realised_vol = returns.std() * _ANNUALISE
        scaled = returns * (vol_target / realised_vol) if realised_vol > 0 else returns
        ax.plot(result["dates"], np.cumsum(scaled), label=name, linewidth=1.8)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Cumulative log return (scaled to {vol_target:.0%} vol)")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_regime_timeline(regimes: pd.DataFrame, title: str = "Detected macro regimes") -> plt.Figure:
    """Shaded band per month, coloured by regime — a quick read on how the model
    carved up history (crisis spikes, long expansions, etc.). `regimes` needs
    'date' and 'regime' columns."""
    regimes = regimes.sort_values("date")
    ids = sorted(regimes["regime"].unique())
    cmap = plt.get_cmap("tab10")
    colour = {rid: cmap(i % 10) for i, rid in enumerate(ids)}

    fig, ax = plt.subplots(figsize=(13, 2.6))
    dates = pd.to_datetime(regimes["date"]).to_numpy()
    for rid in ids:
        mask = regimes["regime"].to_numpy() == rid
        ax.scatter(dates[mask], np.zeros(mask.sum()), c=[colour[rid]], s=60, marker="|",
                   label=regime_label(rid))

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=min(len(ids), 5), fontsize=8, frameon=False)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_cost_sensitivity(frame: pd.DataFrame, title: str) -> plt.Figure:
    """Net Sharpe against the assumed cost level, one line per optimiser.

    This is the chart that carries the result. The impact parameter is assumed
    rather than fitted, so a single net-Sharpe number would be worth very little —
    what's defensible is the *shape*: where the lines cross, and which of them is
    flat in the assumption it can't verify.

    `frame` wants a 'linear_bps' column plus one column per strategy.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    strategies = [c for c in frame.columns if c != "linear_bps"]

    for name in strategies:
        style = "--" if name == "EqualWeight" else "-"
        ax.plot(frame["linear_bps"], frame[name], style, marker="o", linewidth=1.9,
                markersize=5, label=name)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Assumed one-way linear cost (bps); quadratic impact held at 2x")
    ax.set_ylabel("Net-of-cost Sharpe ratio")
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def save(fig: plt.Figure, path: str | Path, dpi: int = 200) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
