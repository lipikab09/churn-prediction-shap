"""Central configuration: paths, seeds and BUSINESS ASSUMPTIONS (edit these!)."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Business assumptions (USD). These are ASSUMPTIONS, not facts: state them in
# your README and show the sensitivity analysis.
# ---------------------------------------------------------------------------
RETENTION_COST = 25.0        # cost of one retention offer (discount, call, ...)
OFFER_SUCCESS_RATE = 0.25    # share of true churners who stay if contacted
EXPECTED_MONTHS = 12         # months of revenue saved when a churner is retained
# Customer value V_i = MonthlyCharges_i * EXPECTED_MONTHS

THRESHOLD_GRID = np.linspace(0.01, 0.95, 95)


def ensure_dirs() -> None:
    for d in (MODELS_DIR, REPORTS_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
