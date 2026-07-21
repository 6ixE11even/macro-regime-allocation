"""
End-to-end entry point:  macro-regime  (or  python -m macro_regime.cli)

    FRED-MD ->  regime detection  ->  for each MSCI universe: backtest -> CSVs + charts

Outputs land in results/<universe>/ plus a shared results/regime_timeline.png.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from macro_regime import config
from macro_regime.analytics.performance import comparison_table
from macro_regime.backtest.cost_aware import cost_summary, run_cost_aware_backtest
from macro_regime.backtest.engine import LONG_ONLY, LONG_SHORT, create_regime_probs, run_backtests
from macro_regime.data.fredmd_loader import FredMDLoader
from macro_regime.data.msci_loader import MSCIReturns
from macro_regime.models.optimization import CostModel
from macro_regime.regimes.detection import RegimeDetector
from macro_regime.viz import plots

logger = logging.getLogger(__name__)

# FX series are dropped before clustering: they're noisy and not the macro state
# we're trying to capture (kept consistent with the original study).
FX_TO_DROP = [
    "Trade_Weighted_U_S_Dollar_Index",
    "Switzerland_U_S_Foreign_Exchange_Rate",
    "Japan_U_S_Foreign_Exchange_Rate",
    "U_S_U_K_Foreign_Exchange_Rate",
    "Canada_U_S_Foreign_Exchange_Rate",
]

UNIVERSES = {
    "developed": config.MSCI_DEVELOPED_XLSX,
    "emerging": config.MSCI_EMERGING_XLSX,
}


def detect_regimes() -> tuple[pd.DataFrame, pd.DataFrame, RegimeDetector]:
    """Load FRED-MD, learn regimes on the train split, classify the rest.

    Returns (regimes_frame[date, regime, regime_probs], pca_features[date, PC*], detector).
    """
    _, stationary, _ = FredMDLoader(config.FRED_MD_CSV).load()
    stationary = stationary.drop(columns=FX_TO_DROP, errors="ignore")

    cutoff = pd.to_datetime(config.TRAIN_END_DATE)
    train = stationary[stationary.index <= cutoff]
    test = stationary[stationary.index > cutoff]

    detector = RegimeDetector()
    regimes_train, pca_train = detector.fit(train)

    frames = [regimes_train.to_frame()]
    pca_frames = [pca_train]
    if not test.empty:
        regimes_test, _, pca_test = detector.predict(test)
        frames.append(regimes_test.to_frame())
        pca_frames.append(pca_test)

    regimes = pd.concat(frames)
    regimes.index.name = "date"
    regimes = regimes.reset_index()
    regimes["date"] = pd.to_datetime(regimes["date"]).dt.to_period("M").dt.to_timestamp()
    regimes["regime_probs"] = regimes["regime"].apply(
        lambda r: create_regime_probs(r, detector.n_regimes_)
    )

    pca = pd.concat(pca_frames)
    pca.index.name = "date"
    pca = pca.reset_index()
    pca["date"] = pd.to_datetime(pca["date"]).dt.to_period("M").dt.to_timestamp()
    return regimes, pca, detector


def run_universe(name: str, xlsx_path, regimes: pd.DataFrame, pca: pd.DataFrame, components: list[str]) -> pd.DataFrame:
    """Backtest one MSCI universe and write its CSVs + charts. Returns the table."""
    logger.info("=== %s ===", name.upper())
    msci = MSCIReturns(xlsx_path)
    msci.load()
    asset_cols = [c for c in msci.returns.columns if c != "date"]

    merged = msci.merge_with_regimes(regimes)
    merged = merged.merge(pca[["date", *components]], on="date", how="left")

    results = run_backtests(merged, asset_cols, components)
    table = comparison_table(results)

    out_dir = config.RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "performance_all.csv", index=False)
    table[table["Strategy"].isin(LONG_ONLY)].to_csv(out_dir / "performance_long_only.csv", index=False)
    table[table["Strategy"].isin(LONG_SHORT)].to_csv(out_dir / "performance_long_short.csv", index=False)

    plots.save(plots.plot_cumulative_returns(results, f"{name.title()} — all strategies"),
               out_dir / "cumulative_all.png")
    plots.save(plots.plot_cumulative_returns({k: results[k] for k in LONG_ONLY if k in results},
                                             f"{name.title()} — long-only"), out_dir / "cumulative_long_only.png")
    plots.save(plots.plot_cumulative_returns({k: results[k] for k in LONG_SHORT if k in results},
                                             f"{name.title()} — long-short"), out_dir / "cumulative_long_short.png")

    logger.info("%s top strategy: %s (Sharpe %.2f)", name, table.iloc[0]["Strategy"], table.iloc[0]["Sharpe Ratio"])
    return table


DEFAULT_BPS_GRID = (0.0, 5.0, 10.0, 15.0, 25.0, 50.0, 100.0)


def run_cost_analysis(
    name: str,
    xlsx_path,
    regimes: pd.DataFrame,
    linear_bps: float,
    impact_bps: float,
    horizon: int,
    long_short: bool,
    sweep: tuple[float, ...] = DEFAULT_BPS_GRID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cost-aware backtest at one cost level, plus a sweep across the grid.

    Returns (headline table at the chosen cost, sensitivity frame across `sweep`).
    """
    logger.info("=== %s (cost-aware) ===", name.upper())
    msci = MSCIReturns(xlsx_path)
    msci.load()
    asset_cols = [c for c in msci.returns.columns if c != "date"]
    merged = msci.merge_with_regimes(regimes)

    headline = cost_summary(run_cost_aware_backtest(
        merged, asset_cols, costs=CostModel(linear_bps, impact_bps),
        horizon=horizon, long_short=long_short,
    ))

    rows = []
    for bps in sweep:
        results = run_cost_aware_backtest(
            merged, asset_cols, costs=CostModel(bps, 2 * bps),
            horizon=horizon, long_short=long_short,
        )
        net = cost_summary(results).set_index("Strategy")["Net Sharpe"]
        rows.append({"linear_bps": bps, **net.to_dict()})
    sensitivity = pd.DataFrame(rows)

    sleeve = "long_short" if long_short else "long_only"
    out_dir = config.RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    headline.to_csv(out_dir / f"cost_aware_{sleeve}.csv", index=False)
    sensitivity.to_csv(out_dir / f"cost_sensitivity_{sleeve}.csv", index=False)

    plots.save(
        plots.plot_cost_sensitivity(sensitivity, f"{name.title()} — net Sharpe vs assumed cost ({sleeve})"),
        out_dir / f"cost_sensitivity_{sleeve}.png",
    )
    return headline, sensitivity


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Macro-regime tactical asset allocation backtest.")
    parser.add_argument("--universe", choices=[*UNIVERSES, "all"], default="all", help="which MSCI universe to run")
    parser.add_argument("--costs", action="store_true",
                        help="run the transaction-cost-aware optimisers and the cost sensitivity sweep")
    parser.add_argument("--linear-bps", type=float, default=10.0,
                        help="one-way linear cost in bps for the headline table (default: 10)")
    parser.add_argument("--impact-bps", type=float, default=20.0,
                        help="impact cost in bps at 100%% single-asset turnover (default: 20)")
    parser.add_argument("--horizon", type=int, default=3,
                        help="planning horizon in months for the multi-period optimiser (default: 3)")
    parser.add_argument("--long-short", action="store_true",
                        help="use the long-short sleeve (net 100%%, gross <= 200%%) instead of long-only")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    config.setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    regimes, pca, detector = detect_regimes()
    components = detector.components_

    # Regime timeline is universe-independent — render it once.
    plots.save(plots.plot_regime_timeline(regimes), config.RESULTS_DIR / "regime_timeline.png")

    todo = UNIVERSES if args.universe == "all" else {args.universe: UNIVERSES[args.universe]}
    for name, path in todo.items():
        if args.costs:
            headline, sensitivity = run_cost_analysis(
                name, path, regimes, args.linear_bps, args.impact_bps, args.horizon, args.long_short,
            )
            print(f"\n=== {name.upper()} — net of {args.linear_bps:.0f}bps linear "
                  f"+ {args.impact_bps:.0f}bps impact ===")
            print(headline.to_string(index=False))
            print(f"\n--- {name.upper()} — net Sharpe vs assumed cost ---")
            print(sensitivity.to_string(index=False))
        else:
            table = run_universe(name, path, regimes, pca, components)
            print(f"\n=== {name.upper()} ===")
            print(table.to_string(index=False))


if __name__ == "__main__":
    main()
