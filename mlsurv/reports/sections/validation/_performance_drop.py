"""Internal vs. external performance drop analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from mlsurv.constants import (
    ALL_METRICS,
    LOWER_IS_BETTER,
    get_concordance_column,
    get_metric_short_name,
)
from mlsurv.reports.sections.base import DATA_EXCEPTIONS
from mlsurv.viz._palettes import get_model_display_name

if TYPE_CHECKING:
    from mlsurv.reports.sections.validation._section import ValidationSection


def _compute_metric_drop(
    int_row,
    ext_row,
    metric_key: str,
) -> dict | None:
    """Compute a single metric drop between internal and external rows.

    Returns a drop dict or None if the metric is unavailable.
    """
    col = get_concordance_column(metric_key)
    if col not in ext_row.index or pd.isna(ext_row[col]):
        return None
    if col not in int_row.index or pd.isna(int_row[col]):
        return None

    int_val = float(int_row[col])
    ext_val = float(ext_row[col])
    delta = ext_val - int_val

    is_lower_better = metric_key in LOWER_IS_BETTER
    is_drop = (delta < 0) if not is_lower_better else (delta > 0)

    metric_display = get_metric_short_name(metric_key, with_asterisk=False)
    delta_str = f"{delta:+.3f}"
    if is_drop:
        delta_str = f"**{delta_str}**"

    return {
        'Metric': metric_display,
        'Internal Test': f"{int_val:.3f}",
        'External': f"{ext_val:.3f}",
        'Delta': delta_str,
    }


def _format_drop_narrative(drop_rows: list, md_lines: list) -> None:
    """Append C-index narrative for the performance drop table."""
    c_index_drops = [r for r in drop_rows if r['Metric'] == "Harrell's C"]
    if c_index_drops:
        for d in c_index_drops:
            raw_delta = d['Delta'].replace('**', '')
            md_lines.append(
                f"C-index for {d['Model']} on {d['Cohort']}: "
                f"{d['Internal Test']} (internal) → {d['External']} (external), "
                f"Δ = {raw_delta}."
            )
        md_lines.append("")


def add_performance_drop(
    section: ValidationSection,
    md_lines: list,
) -> None:
    """Add performance drop analysis comparing internal test vs external validation."""
    eval_result = section.learner.evaluation_result
    if eval_result is None or eval_result.concordance is None:
        md_lines.append("### Internal vs. External Performance")
        md_lines.append("")
        md_lines.append(
            "*Performance comparison unavailable: run evaluate() before validate() "
            "to enable internal vs. external comparison.*"
        )
        md_lines.append("")
        return

    try:
        _build_performance_drop(section, md_lines, eval_result)
    except DATA_EXCEPTIONS as e:
        section._log_recoverable_exception(
            e,
            "validation: performance drop analysis",
            warn_user=True,
            user_message=f"Could not generate performance drop table: {e}",
        )


def _build_performance_drop(
    section: ValidationSection,
    md_lines: list,
    eval_result,
) -> None:
    """Build the performance drop table."""
    internal_concordance = eval_result.concordance
    drop_rows = []

    for name, val_result in section.validation_results.items():
        if val_result.concordance is None:
            continue

        ext_concordance = val_result.concordance
        for _, ext_row in ext_concordance.iterrows():
            model_name = ext_row['model_type']
            display_name = get_model_display_name(model_name)

            internal_match = internal_concordance[
                internal_concordance['model_type'] == model_name
            ]
            if internal_match.empty:
                continue

            int_row = internal_match.iloc[0]

            for metric_key in ALL_METRICS:
                drop = _compute_metric_drop(int_row, ext_row, metric_key)
                if drop is not None:
                    drop_rows.append({
                        'Cohort': name,
                        'Model': display_name,
                        **drop,
                    })

    if not drop_rows:
        return

    md_lines.append("### Internal vs. External Performance")
    md_lines.append("")
    md_lines.append(
        "Performance changes between the internal test set and external validation cohorts. "
        "Bolded deltas indicate performance degradation."
    )
    md_lines.append("")

    drop_df = pd.DataFrame(drop_rows)
    md_lines.append(section._format_table(drop_df))
    md_lines.append("")

    _format_drop_narrative(drop_rows, md_lines)

    section.data['performance_drop'] = drop_df
