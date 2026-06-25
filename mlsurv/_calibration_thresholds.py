"""Shared calibration thresholds.

Single source of truth for the slope/p-value cutoffs used by both the
runtime ``MethodologicalWarning`` (``mlsurv.evaluation.evaluate``) and the
HTML report's poor-calibration flag (``mlsurv.reports.sections.limitations``).
Lives at the package root to avoid an upward dependency from ``evaluation``
into ``reports``.
"""

from __future__ import annotations

# Calibration slope is the sole drift trigger. The band is deliberately wide:
# a slope inside [0.5, 1.5] (|slope-1| <= 0.5) is treated as adequately
# calibrated for deployment, so the flag fires only on genuinely large
# miscalibration that calibrating S(t|x) would meaningfully fix. Both
# CAL_SLOPE_LOW/HIGH and SEVERE_DRIFT_THRESHOLD describe the same |slope-1| > 0.5
# cutoff (kept as separate names for the two call sites that read them).
CAL_SLOPE_LOW = 0.5
CAL_SLOPE_HIGH = 1.5
SEVERE_DRIFT_THRESHOLD = 0.5

# GND (Greenwood-Nam-D'Agostino) p-value: reported as context in the drift
# message text only. It is NOT a standalone trigger -- it over-rejects on
# well-powered cohorts, so the slope band alone decides whether to flag.
P_THRESHOLD_SIGNIFICANT = 0.05
