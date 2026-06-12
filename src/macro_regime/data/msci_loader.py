"""
Load an MSCI price sheet and turn it into monthly log returns.

The two MSCI files (developed, emerging) have a `Dates` column plus one price
column per sector/region. We snap every date to the first of its month so they
align cleanly with the FRED-MD regime dates on an inner merge.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MSCIReturns:
    """Read one MSCI .xlsx, expose monthly log returns and a regime merge."""

    def __init__(self, xlsx_path: str | Path):
        self.xlsx_path = Path(xlsx_path)
        self.returns: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        if not self.xlsx_path.exists():
            raise FileNotFoundError(f"No MSCI file at {self.xlsx_path}")

        prices = pd.read_excel(self.xlsx_path)
        prices["Dates"] = pd.to_datetime(prices["Dates"]).dt.to_period("M").dt.to_timestamp()
        prices = prices.sort_values("Dates").reset_index(drop=True)

        price_cols = [c for c in prices.columns if c != "Dates"]

        # Monthly log returns: ln(P_t / P_{t-1}). First row drops out.
        out = pd.DataFrame({"date": prices["Dates"].iloc[1:].to_numpy()})
        for col in price_cols:
            out[col] = np.log(prices[col].iloc[1:].to_numpy() / prices[col].iloc[:-1].to_numpy())

        self.returns = out.reset_index(drop=True)
        logger.info("MSCI %s: %d months x %d sectors", self.xlsx_path.stem, len(self.returns), len(price_cols))
        return self.returns

    def merge_with_regimes(self, regimes: pd.DataFrame) -> pd.DataFrame:
        """Inner-join returns with a regime frame on the (month-start) date."""
        if self.returns is None:
            raise RuntimeError("Call load() before merging.")

        regimes = regimes.copy()
        regimes["date"] = pd.to_datetime(regimes["date"]).dt.to_period("M").dt.to_timestamp()

        merged = self.returns.merge(regimes, on="date", how="inner")
        if merged.empty:
            raise ValueError("No overlapping months between MSCI returns and regimes — check date ranges.")

        logger.info("Merged on date: %d overlapping months", len(merged))
        return merged
