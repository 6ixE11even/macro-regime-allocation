"""
Load FRED-MD and make every series stationary using the Fed's transform codes.

FRED-MD (McCracken & Ng) is a monthly macro database of ~120 US indicators back
to 1959. Two quirks handled here:

  1. Row 0 of the CSV is a row of transformation codes (one per column) — how to
     render that series stationary (level, first diff, log-diff, ...).
  2. Columns are Fed mnemonics (RPI, INDPRO, GS10); renamed via SERIES_NAMES.

Reference: McCracken & Ng (2016), "FRED-MD: A Monthly Database for Macroeconomic
Research", Journal of Business & Economic Statistics 34(4).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from macro_regime.data.series_names import SERIES_NAMES

logger = logging.getLogger(__name__)

# FRED-MD transformation codes (Fed convention). Comment = the usual reason to pick it.
TCODE_LEVEL = 1            # already stationary (e.g. rate spreads)
TCODE_DIFF = 2            # Δx, integrated of order 1
TCODE_DIFF2 = 3           # Δ²x
TCODE_LOG = 4            # log level
TCODE_LOG_DIFF = 5       # Δ log x ≈ % change — the workhorse for prices/output
TCODE_LOG_DIFF2 = 6      # Δ² log x
TCODE_PCT_CHANGE_DIFF = 7  # Δ(x_t / x_{t-1} − 1)

# Observations each transform eats off the front (a Δ costs one, a Δ² two).
_LAG_COST = {1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2, 7: 2}


class FredMDLoader:
    """Read one FRED-MD CSV, transform it, return a clean monthly panel.

    Returns (raw, stationary, meta). `stationary` feeds the regime model; `raw`
    and `meta` are kept for sanity checks and plotting the untransformed levels.

        raw, stationary, meta = FredMDLoader("data/raw/fred_md.csv").load()
    """

    # Drop a column if >50% missing after transforming (short-history series).
    MAX_MISSING_FRACTION = 0.50

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.raw: pd.DataFrame | None = None
        self.transformed: pd.DataFrame | None = None
        self.tcodes: pd.Series | None = None
        self.meta: dict[str, dict] = {}

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"No FRED-MD file at {self.csv_path}")

        table = pd.read_csv(self.csv_path)

        # First row = transform codes, indexed by mnemonic; the rest is data.
        self.tcodes = table.iloc[0, 1:]
        data = table.iloc[1:, :].copy()

        # Date column is normally 'sasdate'; fall back to the first column.
        date_col = "sasdate" if "sasdate" in data.columns else data.columns[0]
        data[date_col] = pd.to_datetime(data[date_col], format="%m/%d/%Y", errors="coerce")
        data = data.set_index(date_col)

        # Coerce everything else to numeric; non-numeric becomes NaN.
        data = data.apply(pd.to_numeric, errors="coerce")

        self.raw = data
        logger.info("Loaded %d months x %d indicators", *data.shape)

        self.transformed = self._make_stationary(data)
        logger.info("Stationary panel: %d months x %d indicators", *self.transformed.shape)
        return self.raw, self.transformed, self.meta

    def _make_stationary(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply each column's transform code, then trim the edges.

        Keeps only SERIES_NAMES columns (the curated ~120-indicator set).
        """
        columns: dict[str, pd.Series] = {}
        worst_lag = 0

        for code_name, nice_name in SERIES_NAMES.items():
            if code_name not in data.columns or code_name not in self.tcodes.index:
                continue

            try:
                code = int(self.tcodes[code_name])
            except (ValueError, TypeError):
                code = TCODE_LEVEL  # junk code -> assume it's already a level

            columns[nice_name] = self._transform_one(data[code_name], code)
            self.meta[nice_name] = {"tcode": code, "mnemonic": code_name, "lag_cost": _LAG_COST.get(code, 0)}
            worst_lag = max(worst_lag, _LAG_COST.get(code, 0))

        out = pd.DataFrame(columns)

        # Differencing leaves leading NaNs; trim all columns to the worst-case lag.
        if len(out) > worst_lag:
            out = out.iloc[worst_lag:]

        # Drop columns that are still mostly empty.
        missing = out.isna().mean()
        sparse = missing[missing > self.MAX_MISSING_FRACTION].index.tolist()
        if sparse:
            logger.info("Dropping %d sparse columns (>50%% missing)", len(sparse))
            out = out.drop(columns=sparse)

        # Median-fill the few remaining isolated gaps (robust to outliers).
        return out.apply(lambda col: col.fillna(col.median()))

    @staticmethod
    def _transform_one(series: pd.Series, code: int) -> pd.Series:
        if code == TCODE_LEVEL:
            return series
        if code == TCODE_DIFF:
            return series.diff()
        if code == TCODE_DIFF2:
            return series.diff().diff()

        # Treat non-positive values as missing — log is undefined there.
        with np.errstate(invalid="ignore", divide="ignore"):
            safe_log = np.log(series.where(series > 0))
        if code == TCODE_LOG:
            return safe_log
        if code == TCODE_LOG_DIFF:
            return safe_log.diff()
        if code == TCODE_LOG_DIFF2:
            return safe_log.diff().diff()
        if code == TCODE_PCT_CHANGE_DIFF:
            return (series / series.shift(1) - 1).diff()
        return series  # unknown code: leave as-is
