"""Formatting and interpretation utilities for the evaluation section.

Functions shared across multiple report sections live in
``mlsurv.reports._helpers`` and are re-exported here for convenience.
"""

import warnings

import numpy as np
import pandas as pd

# Re-exported from shared module for backwards compatibility.
from mlsurv.reports._helpers import (  # noqa: F401 — re-exported
    collect_calibration_rows,
    empty_calibration_figure,
    extract_eval_times_from_auc_t,
    format_eval_time,
)

from mlsurv.reports._constants import (
    P_THRESHOLD_EXCELLENT,
    P_THRESHOLD_GOOD,
    P_THRESHOLD_SIGNIFICANT,
)


def format_metric_with_ci(test_val: float, boot_summary, metric_name: str) -> str:
    """Format a metric value with optional bootstrap CI."""
    if boot_summary is not None:
        metric_row = boot_summary[boot_summary['metric'] == metric_name]
        if len(metric_row) > 0 and not np.isnan(metric_row['ci_lower'].iloc[0]):
            ci_lower = metric_row['ci_lower'].iloc[0]
            ci_upper = metric_row['ci_upper'].iloc[0]
            return f"{test_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
    return f"{test_val:.3f}"


def interpret_auct_quality(mean_auc: float) -> str:
    """Map mean AUC(t) to a quality label via BaseSection._interpret_quality."""
    from mlsurv.reports.sections.base import BaseSection
    return BaseSection._interpret_quality(mean_auc)


def compute_separation_quality(p_value: float) -> str:
    """Categorize log-rank test p-value into quality labels."""
    if pd.isna(p_value):
        return "Unknown"
    if p_value < P_THRESHOLD_EXCELLENT:
        return "Excellent"
    if p_value < P_THRESHOLD_GOOD:
        return "Good"
    if p_value < P_THRESHOLD_SIGNIFICANT:
        return "Fair"
    return "Poor"


def warn_eval_times_mismatch(auc_df: pd.DataFrame) -> None:
    """Emit SuggestionWarning if models in *auc_df* have different eval_time grids.

    Only fires when there are 2+ models with non-identical eval_time sets.
    """
    from mlsurv._warnings import SuggestionWarning
    from mlsurv.viz._palettes import get_model_display_name

    if 'model_type' not in auc_df.columns or 'eval_time' not in auc_df.columns:
        return

    models = auc_df['model_type'].unique()
    if len(models) < 2:
        return

    per_model = {
        m: sorted(
            t for t in auc_df.loc[auc_df['model_type'] == m, 'eval_time'].unique()
            if np.isfinite(t)
        )
        for m in models
    }

    reference = per_model[models[0]]
    if all(per_model[m] == reference for m in models[1:]):
        return

    detail = "\n".join(
        f"  {get_model_display_name(m)}: "
        f"[{', '.join(format_eval_time(t) for t in times)}]"
        for m, times in per_model.items()
    )
    warnings.warn(
        f"Models were evaluated at different eval_times, causing N/A cells "
        f"in the AUC(t) table. This typically happens when evaluate() is "
        f"called separately for each model without passing eval_times — "
        f"each call auto-computes its own grid. Fix: pass the same "
        f"eval_times to all evaluate() calls, or use "
        f"run(eval_times=[...]) to set them once.\n"
        f"Per-model eval_times:\n{detail}",
        SuggestionWarning,
        stacklevel=2,
    )
