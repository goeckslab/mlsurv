"""Table 1 distribution comparison between dev and validation cohorts."""

from __future__ import annotations

import html as html_module
import re
import warnings
from typing import TYPE_CHECKING, Any, Dict, List

import pandas as pd

from mlsurv._warnings import SuggestionWarning

from mlsurv.reports.sections.base import DATA_EXCEPTIONS
from mlsurv.reports._constants import TABLE1_MAX_FEATURES

if TYPE_CHECKING:
    from mlsurv.reports.sections.validation._section import ValidationSection


def _slugify(value: str) -> str:
    """Build a slug safe for HTML attributes."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or "cohort"


def _warn_column_mismatches(
    dev_cols: set,
    cohort_data: dict,
) -> None:
    """Warn about columns present in dev but missing from validation cohorts."""
    for cohort_name, (df, _n) in cohort_data.items():
        if cohort_name == 'Development':
            continue
        missing = dev_cols - set(df.columns)
        if missing:
            warnings.warn(
                f"Validation cohort '{cohort_name}' is missing {len(missing)} "
                f"feature(s) present in development data: {sorted(missing)[:5]}",
                SuggestionWarning,
                stacklevel=3,
            )


def _format_continuous_row(
    col: str,
    cohort_data: dict,
) -> tuple:
    """Format a continuous variable row.

    Returns
    -------
    tuple of (dict, list)
        (display_row, tidy_records)
    """
    row_dict: Dict[str, str] = {'Variable': f"{col}, mean (SD)"}
    tidy_records: List[List[str]] = []

    for cohort_name, (df, _n) in cohort_data.items():
        valid = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
        if len(valid) == 0:
            val = "N/A"
        elif len(valid) == 1:
            val = f"{valid.iloc[0]:.2f} (n=1)"
        else:
            val = f"{valid.mean():.2f} ({valid.std():.2f})"
        row_dict[cohort_name] = val
        tidy_records.append([col, "Mean (SD)", cohort_name, val])

    return row_dict, tidy_records


def _format_binary_row(
    col: str,
    cohort_data: dict,
    pos_level: Any,
) -> tuple:
    """Format a binary categorical variable row.

    Returns
    -------
    tuple of (dict, list)
        (display_row, tidy_records)
    """
    row_dict: Dict[str, str] = {'Variable': f"{col}, n (%)"}
    tidy_records: List[List[str]] = []

    for cohort_name, (df, n) in cohort_data.items():
        if col not in df.columns:
            val = "N/A"
        else:
            count = int((df[col] == pos_level).sum())
            pct = count / n * 100 if n > 0 else 0
            val = f"{count} ({pct:.1f}%)"
        row_dict[cohort_name] = val
        tidy_records.append([col, str(pos_level), cohort_name, val])

    return row_dict, tidy_records


def _format_multi_level_rows(
    col: str,
    cohort_data: dict,
    top_levels: list,
) -> tuple:
    """Format multi-category variable rows (one per level).

    Returns
    -------
    tuple of (list[dict], list)
        (display_rows, tidy_records)
    """
    display_rows: List[Dict[str, str]] = []
    tidy_records: List[List[str]] = []

    for level in top_levels:
        row_dict: Dict[str, str] = {'Variable': f"  {col} = {level}"}
        for cohort_name, (df, n) in cohort_data.items():
            if col not in df.columns:
                val = "N/A"
            else:
                count = int((df[col] == level).sum())
                pct = count / n * 100 if n > 0 else 0
                val = f"{count} ({pct:.1f}%)"
            row_dict[cohort_name] = val
            tidy_records.append([col, str(level), cohort_name, val])
        display_rows.append(row_dict)

    return display_rows, tidy_records


# ------------------------------------------------------------------
# Interactive HTML rendering
# ------------------------------------------------------------------


def _render_table1_html(
    display_df: pd.DataFrame,
    cohort_names: List[str],
    cohort_col_names: Dict[str, str],
    n_visible_rows: int,
    n_total_features: int,
) -> str:
    """Render Table 1 as interactive HTML with scroll, expand/collapse, and cohort filters.

    Parameters
    ----------
    display_df
        Full DataFrame with all features (Variable + renamed cohort columns).
    cohort_names
        Ordered original cohort names (before renaming), used to derive slugs.
    cohort_col_names
        Mapping from original cohort name to renamed column header
        (e.g. ``{"Development": "Development (n=500)"}``)
    n_visible_rows
        Number of display rows from the first TABLE1_MAX_FEATURES features.
    n_total_features
        Total number of raw features (for the expand button label).
    """
    has_overflow = len(display_df) > n_visible_rows > 0
    cohort_slugs = {name: _slugify(name) for name in cohort_names}

    lines: List[str] = ['<!-- MLSURV_RAW_HTML -->']
    lines.append('<div class="feature-table-container">')

    # --- Cohort filter controls ---
    if len(cohort_names) >= 2:
        lines.append('<div class="feature-table-controls">')
        lines.append('<span class="ft-control-label">Cohorts:</span>')
        for name in cohort_names:
            slug = html_module.escape(cohort_slugs[name], quote=True)
            col_header = cohort_col_names.get(name, name)
            label = html_module.escape(col_header)
            lines.append(
                f'<label class="feature-group-toggle">'
                f'<input type="checkbox" data-ft-model="{slug}" checked> {label}'
                f'</label>'
            )
        lines.append('</div>')

    # --- Scrollable table ---
    lines.append('<div class="feature-table-scroll">')
    lines.append('<table class="feature-analysis-table">')

    # thead
    lines.append('<thead><tr>')
    lines.append('<th class="sticky-col">Variable</th>')
    for name in cohort_names:
        slug = html_module.escape(cohort_slugs[name], quote=True)
        col_header = cohort_col_names.get(name, name)
        header = html_module.escape(col_header)
        lines.append(f'<th data-ft-model="{slug}">{header}</th>')
    lines.append('</tr></thead>')

    # tbody
    lines.append('<tbody>')
    for row_idx, (_, row) in enumerate(display_df.iterrows()):
        tr_cls = ' class="ft-overflow-row"' if has_overflow and row_idx >= n_visible_rows else ''
        lines.append(f'<tr data-original-index="{row_idx}"{tr_cls}>')
        # Variable (sticky)
        var_val = html_module.escape(str(row.get('Variable', '')))
        lines.append(f'<td class="sticky-col">{var_val}</td>')
        # Cohort cells
        for name in cohort_names:
            slug = html_module.escape(cohort_slugs[name], quote=True)
            col_header = cohort_col_names.get(name, name)
            cell_val = html_module.escape(str(row.get(col_header, '')))
            lines.append(f'<td data-ft-model="{slug}">{cell_val}</td>')
        lines.append('</tr>')
    lines.append('</tbody>')

    lines.append('</table>')
    lines.append('</div>')  # feature-table-scroll

    # --- Expand/collapse button ---
    if has_overflow:
        lines.append('<div class="feature-table-expand">')
        lines.append(
            f'<button class="ft-expand-btn" '
            f'data-n-visible="{n_visible_rows}" '
            f'data-n-total="{n_total_features}">'
            f'Show all {n_total_features} features</button>'
        )
        lines.append('</div>')

    lines.append('</div>')  # feature-table-container
    lines.append('<!-- /MLSURV_RAW_HTML -->')
    return '\n'.join(lines)


def add_distribution_comparison(
    section: ValidationSection,
    md_lines: list,
) -> None:
    """Add Table 1 comparing predictor distributions between dev and validation cohorts."""
    has_xval = any(
        val_result.X_val is not None
        for val_result in section.validation_results.values()
    )
    if not has_xval:
        return

    X_train = section.learner.X_train
    X_test = section.learner.X_test
    if X_train is None or X_test is None:
        return

    dev_data = pd.concat([X_train, X_test], ignore_index=True)
    n_dev = len(dev_data)

    cohort_data = {'Development': (dev_data, n_dev)}
    for name, val_result in section.validation_results.items():
        if val_result.X_val is not None:
            cohort_data[name] = (val_result.X_val, len(val_result.X_val))

    if len(cohort_data) < 2:
        return

    md_lines.append("### Cohort Characteristics")
    md_lines.append("")
    md_lines.append(
        "Distribution of predictor variables across development and validation cohorts. "
        "Large differences may indicate population shift and could explain performance changes."
    )
    md_lines.append("")

    try:
        _build_table1(section, md_lines, dev_data, cohort_data)
    except DATA_EXCEPTIONS as e:
        section._log_recoverable_exception(
            e,
            "validation: distribution comparison table",
            warn_user=True,
            user_message=f"Could not generate cohort characteristics table: {e}",
        )


def _build_table1(
    section: ValidationSection,
    md_lines: list,
    dev_data: pd.DataFrame,
    cohort_data: dict,
) -> None:
    """Build and render interactive Table 1 distribution comparison."""
    dev_cols = set(dev_data.columns)
    _warn_column_mismatches(dev_cols, cohort_data)

    all_cols = list(dev_data.columns)
    n_total_features = len(all_cols)

    tidy_rows: List[List[str]] = []
    display_rows: List[Dict[str, str]] = []
    n_visible_display_rows = 0

    for feat_idx, col in enumerate(all_cols):
        rows_before = len(display_rows)

        is_continuous = (
            pd.api.types.is_numeric_dtype(dev_data[col])
            and dev_data[col].nunique() > 10
        )
        if is_continuous:
            row_dict, records = _format_continuous_row(col, cohort_data)
            display_rows.append(row_dict)
            tidy_rows.extend(records)
        else:
            combined = pd.concat([
                df[col] for df, _n in cohort_data.values() if col in df.columns
            ])
            unique_levels = sorted(combined.dropna().unique(), key=str)
            is_binary = len(unique_levels) == 2

            if is_binary:
                pos_level = unique_levels[-1]
                row_dict, records = _format_binary_row(col, cohort_data, pos_level)
                display_rows.append(row_dict)
                tidy_rows.extend(records)
            else:
                top_levels = combined.value_counts().head(5).index.tolist()
                rows, records = _format_multi_level_rows(col, cohort_data, top_levels)
                display_rows.extend(rows)
                tidy_rows.extend(records)

        if feat_idx < TABLE1_MAX_FEATURES:
            n_visible_display_rows += len(display_rows) - rows_before

    if not display_rows:
        return

    display_df = pd.DataFrame(display_rows)
    cohort_names = list(cohort_data.keys())
    cohort_col_names = {
        name: f"{name} (n={n})"
        for name, (_, n) in cohort_data.items()
        if name in display_df.columns
    }
    display_df = display_df.rename(columns=cohort_col_names)

    html = _render_table1_html(
        display_df,
        cohort_names,
        cohort_col_names,
        n_visible_display_rows,
        n_total_features,
    )
    md_lines.append(html)
    md_lines.append("")

    section.data['table1'] = pd.DataFrame(
        tidy_rows, columns=["Variable", "Statistic", "Cohort", "Value"]
    )
