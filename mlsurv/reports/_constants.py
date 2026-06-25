"""Centralized constants for the reports module.

Groups magic numbers that were previously scattered across 10+ section files
into named constants, making it easier to adjust report styling consistently.

Follows the UPPER_SNAKE_CASE convention used by ``mlsurv/constants.py``.
"""

from __future__ import annotations

import math

from mlsurv._calibration_thresholds import (  # noqa: F401  (re-exported)
    CAL_SLOPE_HIGH,
    CAL_SLOPE_LOW,
    P_THRESHOLD_SIGNIFICANT,
)
from mlsurv.constants import (  # noqa: F401  (re-exported)
    CORRELATION_THRESHOLD,
    P_STARS,
    P_THRESHOLD_EXCELLENT,
    P_THRESHOLD_GOOD,
)

# ── A. Statistical thresholds ──────────────────────────────────────────────

# ``P_THRESHOLD_SIGNIFICANT`` is re-exported from
# ``mlsurv._calibration_thresholds`` so the runtime calibration warning
# and this report module share a single source of truth.
# ``P_THRESHOLD_EXCELLENT``/``P_THRESHOLD_GOOD``/``P_STARS`` and
# ``CORRELATION_THRESHOLD`` moved to ``mlsurv.constants`` so the
# evaluation layer can share them without importing reports.


def format_pvalue(p: float | None, *, stars: bool = False) -> str:
    """Reader-friendly p-value formatter shared across report sections.

    - ``p >= 0.01``: 3 decimals (e.g. ``0.169``, ``0.050``)
    - ``p < 0.01``: scientific with 2 decimals in the mantissa (e.g. ``4.44e-04``)
    - NaN / None / out of ``[0, 1]`` / non-finite: em dash (``—``)
    - ``stars=True``: appends ``*`` / ``**`` / ``***`` per ``P_STARS``
    """
    if p is None:
        return "—"
    try:
        pf = float(p)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(pf) or not (0.0 <= pf <= 1.0):
        return "—"
    body = f"{pf:.3f}" if pf >= P_THRESHOLD_GOOD else f"{pf:.2e}"
    if stars:
        for thresh, mark in P_STARS:
            if pf < thresh:
                return body + mark
    return body

# AUC / C-index quality interpretation bands
# Hosmer & Lemeshow (2000), *Applied Logistic Regression*, Table 5.8
QUALITY_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.8, "excellent"),
    (0.7, "good"),
    (0.6, "moderate"),
)
QUALITY_DEFAULT = "poor"  # below lowest threshold

# Discrimination flag boundary — below this C-index is "poor".
# Kept separate from QUALITY_THRESHOLDS so the flag logic is not coupled
# to the label-mapping table.
DISCRIMINATION_THRESHOLD = 0.6

# ── B. Analysis defaults ───────────────────────────────────────────────────

DEFAULT_RISK_GROUPS = 4
DEFAULT_CALIBRATION_BINS = 10
# Fallback percentiles for calibration when eval_times are unavailable
# from the evaluation results (e.g., no auc_t data).
CALIBRATION_PERCENTILES = (25, 50, 75)

# ── D. Table defaults ─────────────────────────────────────────────────────

TABLE_VISIBLE_ROWS = 10          # data-visible-rows in prediction tables
TABLE1_MAX_FEATURES = 15         # max features in dev/validation demographic table
MAX_CORRELATION_FEATURES = 20    # max features shown in correlation heatmap
MAX_PH_VIOLATION_DISPLAY = 5     # PH violations shown before "and N more"
BENCHMARK_SUBPOPULATION_VISIBLE_ROWS = 20  # rows before "Show remaining" toggle

# ── E. Rounding precision ─────────────────────────────────────────────────

RISK_SCORE_PRECISION = 5         # decimal places for risk scores
