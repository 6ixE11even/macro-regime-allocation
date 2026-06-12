"""
One place for paths and a couple of global knobs.

Everything is resolved relative to the repo root (three parents up from this
file: config.py -> macro_regime -> src -> repo). That way the pipeline runs the
same whether you launch it from the repo root, a notebook, or cron.
"""
from __future__ import annotations

import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = REPO_ROOT / "results"

# Raw inputs (renamed on import so they're tidy and lowercase).
FRED_MD_CSV = RAW_DIR / "fred_md.csv"
MSCI_DEVELOPED_XLSX = RAW_DIR / "msci_developed.xlsx"
MSCI_EMERGING_XLSX = RAW_DIR / "msci_emerging.xlsx"

# Reproducibility: one seed, shared by every sklearn call that takes one.
RANDOM_STATE = 42

# Regimes are learned on data up to here; everything after is out-of-sample.
TRAIN_END_DATE = "2023-12-31"


def setup_logging(level: int = logging.INFO) -> None:
    """Plain, readable logs. Call once at the start of a run."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
