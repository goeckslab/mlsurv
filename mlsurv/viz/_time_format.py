"""Time-unit normalization and labeling helpers for survival figures.

Lives in the ``viz`` layer so figure builders (and ``PredictionResult.plot()``)
can label axes and captions without importing from ``learner``. The
report-specific milestone-default table (``_default_clinical_times``) stays in
``learner/_patient_report.py``.
"""

from __future__ import annotations

_CANONICAL_TIME_UNITS = {
    "day": "days", "days": "days", "d": "days",
    "week": "weeks", "weeks": "weeks", "wk": "weeks", "wks": "weeks",
    "month": "months", "months": "months", "mo": "months", "mos": "months",
    "year": "years", "years": "years", "yr": "years", "yrs": "years",
}


def _canonicalize_time_unit(unit):
    """Map user-provided unit to one of {days, weeks, months, years} or None.

    Case-insensitive; accepts singular/plural/common abbreviations. Returns
    ``None`` for unrecognized or missing values so downstream code can fall
    back to unit-free rendering.
    """
    if unit is None:
        return None
    key = str(unit).strip().lower()
    return _CANONICAL_TIME_UNITS.get(key)


def _format_time_with_unit(t, canonical_unit):
    """Format a time value with its unit for prose captions.

    ``30`` + ``"months"`` -> ``"30 months"``; ``1`` + ``"years"`` -> ``"1 year"``
    (singular). Returns just the numeric value when the unit is unknown.
    """
    val = f"{float(t):g}"
    if canonical_unit is None:
        return val
    is_one = abs(float(t) - 1.0) < 1e-9
    label = canonical_unit[:-1] if is_one else canonical_unit
    return f"{val} {label}"


def _axis_title_time(canonical_unit):
    """X-axis title string; includes the unit when known."""
    if canonical_unit is None:
        return "Time"
    return f"Time ({canonical_unit})"
