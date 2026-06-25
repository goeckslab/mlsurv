"""Console printing utilities for mlsurv evaluation results."""

import sys
import warnings
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from mlsurv._warnings import InternalWarning
from mlsurv.evaluation._incremental import format_p_sig as _format_p_sig

from mlsurv.constants import (
    get_metric_display_name, get_metric_short_name, get_concordance_column,
    ALL_METRICS, LOWER_IS_BETTER, METRIC_REGISTRY,
    _format_ideal,
)

# Note: Console tables display only scorable metrics (ALL_METRICS).
# Diagnostic metrics (cal_slope, cal_intercept, gnd) are
# shown in HTML reports (see reports/sections/evaluation/) but omitted
# from compact console output for readability.


def _detect_best_marker() -> str:
    """Use arrow if stdout supports it, otherwise fall back to >."""
    encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
    try:
        '\u2192'.encode(encoding)
        return '\u2192'
    except (UnicodeEncodeError, LookupError):
        return '>'


_BEST_MARKER = _detect_best_marker()

# Footnote and column-width budget for the optional held-out-test delta
# annotation, e.g. " (Δ +0.043)". validate() passes a baseline concordance so
# each cell shows its change from the development (held-out test) metrics.
_DELTA_NOTE = "Δ = change from held-out test set."
# Column-width budget for the annotation. " (Δ +0.000)" is 11 chars at the
# default precision=3 with metrics in ~[0, 1]; 12 leaves a 1-char margin.
_DELTA_WIDTH = 12


def resolve_display_name(key: str, display_names: Optional[Dict[str, str]]) -> str:
    """Resolve a storage key to its human-readable display name.

    Parameters
    ----------
    key : str
        Internal model storage key (e.g., 'rsf').
    display_names : dict or None
        Mapping from storage keys to display names. If None or empty, returns
        the key unchanged.

    Returns
    -------
    str
        Human-readable name, or the key unchanged when not mapped.
    """
    if not key or not display_names:
        return key
    return display_names.get(key, key)


def _is_valid_numeric(value) -> bool:
    """Check if a value is a valid finite numeric (not None, NaN, inf, pd.NA, or non-numeric)."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    try:
        return bool(np.isfinite(value))
    except (TypeError, ValueError):
        # Non-numeric values (e.g. a stray string in an object-dtype column)
        # are not valid finite numerics, so report them as invalid rather
        # than letting them slip through to a formatter that would crash.
        return False


def _format_mean_std(mean: float, std: float, precision: int = 3) -> str:
    """Format mean and std as '0.723 (0.032)'."""
    if not _is_valid_numeric(mean) or not _is_valid_numeric(std):
        return "N/A"
    return f"{mean:.{precision}f} ({std:.{precision}f})"


def _format_ci(lower: float, upper: float, precision: int = 3) -> str:
    """Format confidence interval as '[0.665, 0.758]'."""
    if not _is_valid_numeric(lower) or not _is_valid_numeric(upper):
        return "[N/A]"
    return f"[{lower:.{precision}f}, {upper:.{precision}f}]"


def _format_value(value: float, precision: int = 3) -> str:
    """Format a single value."""
    if not _is_valid_numeric(value):
        return "N/A"
    return f"{value:.{precision}f}"


_SIGNIFICANCE_FOOTER = "* p < 0.05  ** p < 0.01  *** p < 0.001"


def _format_pvalue(p: float) -> str:
    """Format p-value for display (scientific notation for very small values)."""
    if not _is_valid_numeric(p):
        return "NaN"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.3f}"


def _significance_stars(p: float) -> str:
    """Return significance stars: *** (<0.001), ** (<0.01), * (<0.05), or empty."""
    if not _is_valid_numeric(p):
        return ""
    if p < 0.001:
        return " ***"
    if p < 0.01:
        return " **"
    if p < 0.05:
        return " *"
    return ""


def _make_separator(*widths: int) -> str:
    """Build a table separator line like '------+---------+------'."""
    return "-+-".join("-" * w for w in widths)


def _print_metric_footnotes(has_lower_is_better: bool, has_ideal: bool,
                            delta_note: Optional[str] = None) -> None:
    """Print footnote lines for metric tables (lower-is-better, ideal target, held-out-test delta)."""
    footnotes = []
    if has_lower_is_better:
        footnotes.append("* Lower is better.")
    if has_ideal:
        footnotes.append("(=N) shows ideal target value.")
    if footnotes:
        print(" ".join(footnotes))
    if delta_note:
        print(delta_note)


def _print_single_table(
    header_text: str,
    label: Optional[str],
    col1_label: str,
    col1_width: int,
    col2_label: str,
    col2_width: int,
    rows: List[tuple],
    has_lower_is_better: bool = False,
    has_ideal: bool = False,
    delta_note: Optional[str] = None,
) -> None:
    """Print a 2-column single-model table.

    Renders: open rule, (optionally labeled) header, open rule, column titles,
    separator, data rows, close rule, footnotes. The helper never warns; any
    `_filter_by_model_type` calls or other warning paths must run in the caller
    so per-call-site `_warn_stacklevel` values stay correct.

    Parameters
    ----------
    rows : list of (str, str)
        Pre-formatted ``(col1_value, col2_value)`` tuples.
    delta_note : str, optional
        Footnote printed when value cells carry ``(Δ ...)`` deltas.
    """
    if label:
        header_text = f"{label}: {header_text}"
    total_width = col1_width + col2_width + 3  # +3 for " | "
    print("=" * total_width)
    print(header_text)
    print("=" * total_width)
    print(f"{col1_label:<{col1_width}} | {col2_label:<{col2_width}}")
    print(_make_separator(col1_width, col2_width))
    for c1, c2 in rows:
        print(f"{c1:<{col1_width}} | {c2:<{col2_width}}")
    print("=" * total_width)
    _print_metric_footnotes(has_lower_is_better, has_ideal, delta_note)


def _print_multi_table(
    header_text: str,
    model_names: List[str],
    labels: Dict[str, str],
    metric_keys: List[str],
    model_col_width: int,
    metric_col_width: int,
    cell_builder,
    best_models: Dict[str, str],
    best_overall_check=None,
    delta_note: Optional[str] = None,
) -> None:
    """Print an N-column multi-model table.

    Parameters
    ----------
    cell_builder : callable(model_name) -> list[str]
        Returns formatted cell strings (one per ``metric_key``). Each cell must
        be left-padded to ``metric_col_width`` so that ``_mark_best_in_cells``
        can re-pad with the best-model marker without misaligning the row.
    best_overall_check : callable(model_name) -> bool, optional
        Returns True if model wins all metrics (gets ``" <best"`` suffix).
        Pass ``None`` to disable the suffix entirely (CV table behavior).
    delta_note : str, optional
        Footnote printed after the table when cells carry ``(Δ ...)`` deltas.
    """
    n_metrics = len(metric_keys)
    total_width = model_col_width + 3 + (metric_col_width + 3) * n_metrics - 1

    print("=" * total_width)
    print(header_text)
    print("=" * total_width)

    header_row, has_lower_is_better = _build_header_row(
        'Model', model_col_width, metric_keys, metric_col_width
    )
    print(header_row)
    print(_build_separator_row(model_col_width, n_metrics, metric_col_width))

    multi = len(model_names) > 1
    for name in model_names:
        cells = cell_builder(name)
        cells = _mark_best_in_cells(
            cells, name, metric_keys, best_models, metric_col_width, multi,
        )
        row = _format_model_row(labels[name], cells, model_col_width)
        if multi and best_overall_check is not None and best_overall_check(name):
            row += " <best"
        print(row)

    print("=" * total_width)
    if has_lower_is_better:
        print("* Lower is better.")
    if delta_note:
        print(delta_note)


def _format_stat_and_p(stat, p, precision: int) -> tuple[str, str]:
    """Format a test statistic and p-value for table display."""
    stat_str = _format_value(stat, precision) if _is_valid_numeric(stat) else "NaN"
    p_str = _format_pvalue(p) + _significance_stars(p)
    return stat_str, p_str


def _format_tau(tau, precision: int) -> str:
    """Format tau for RMST display: integer if whole, otherwise to precision."""
    if not _is_valid_numeric(tau):
        return "N/A"
    if tau == int(tau):
        return f"{tau:.0f}"
    return f"{tau:.{precision}f}"


def _print_rmst_group_table(result, group_w: int, val_w: int, precision: int) -> None:
    """Print the per-group RMST (SE) rows shared by two-group and pairwise printers."""
    print(f"{'Group':<{group_w}} | {'RMST (SE)':<{val_w}}")
    print(_make_separator(group_w, val_w))
    for g in result.rmst:
        val_str = f"{result.rmst[g]:.{precision}f} ({result.se[g]:.{precision}f})"
        print(f"{str(g):<{group_w}} | {val_str:<{val_w}}")


def _format_cv_fold_label(n_scores: int, cv_repeats: int = 1) -> str:
    """CV fold descriptor for a results header.

    ``n_scores`` is the total number of CV scores (folds x repeats). With
    ``cv_repeats > 1`` the label splits them, e.g. ``'5-fold x3 repeats'``;
    otherwise it is just ``'5-fold'``. ASCII ``x`` keeps the header safe on
    Windows consoles.
    """
    if cv_repeats > 1 and n_scores % cv_repeats == 0:
        return f"{n_scores // cv_repeats}-fold x{cv_repeats} repeats"
    return f"{n_scores}-fold"


def print_cv_metrics_table(
    cv_metrics: Dict[str, Dict[str, Any]],
    n_folds: int,
    model_name: Optional[str] = None,
    precision: int = 3,
    display_name: Optional[str] = None,
    cv_repeats: int = 1,
) -> None:
    """
    Print CV metrics table for a single model.

    Parameters
    ----------
    cv_metrics : dict
        Dict of {metric_name: {'mean': float, 'std': float, 'scores': array}}.
    n_folds : int
        Total number of CV scores (folds x repeats).
    model_name : str, optional
        Raw model key. Not used for filtering in this function, but
        accepted for API consistency with print_bootstrap_table.
    precision : int, default=3
        Decimal precision for formatting.
    display_name : str, optional
        Human-readable name for the table header (e.g., 'RSF'). Falls back
        to model_name if not provided.
    cv_repeats : int, default=1
        Number of CV repeats. When > 1, the header reads the base folds and
        repeat count (e.g. '5-fold x3 repeats') instead of the inflated total.
    """
    label = display_name or model_name
    metric_col_width = 26
    value_col_width = 21

    rows = []
    has_lower_is_better = False
    for metric in ALL_METRICS:
        if metric not in cv_metrics:
            continue
        data = cv_metrics[metric]
        metric_label = get_metric_display_name(metric)
        if metric in LOWER_IS_BETTER:
            metric_label += " *"
            has_lower_is_better = True
        rows.append((metric_label, _format_mean_std(data['mean'], data['std'], precision)))

    _print_single_table(
        header_text=f"Cross-Validation Results ({_format_cv_fold_label(n_folds, cv_repeats)})",
        label=label,
        col1_label='Metric', col1_width=metric_col_width,
        col2_label='Mean (Std)', col2_width=value_col_width,
        rows=rows,
        has_lower_is_better=has_lower_is_better,
    )


def _get_available_cv_metrics(model_results: Dict[str, Any]) -> List[str]:
    """Determine which CV metrics are available across any model."""
    available = []
    for metric in ALL_METRICS:
        if any(r.cv_metrics and metric in r.cv_metrics for r in model_results.values()):
            available.append(metric)
    return available or ['harrell']


def _is_better_value(val: float, best_val: float, lower_better: bool) -> bool:
    """Return True if val is better than best_val per the metric direction."""
    return val < best_val if lower_better else val > best_val


def _get_cv_metric_value(result, metric: str) -> Optional[float]:
    """Extract mean CV value for a metric, or None if unavailable."""
    if result.cv_metrics and metric in result.cv_metrics:
        return result.cv_metrics[metric]['mean']
    if metric == 'harrell':
        return result.cv_metric_mean
    return None


def _find_best_cv_models(
    model_results: Dict[str, Any], available_metrics: List[str]
) -> Dict[str, str]:
    """Find best model for each CV metric. Returns {metric: model_name}."""
    best_models = {}
    for metric in available_metrics:
        best_name, best_val = None, None
        lower_better = METRIC_REGISTRY[metric].lower_is_better
        for name, result in model_results.items():
            val = _get_cv_metric_value(result, metric)
            if val is None or not _is_valid_numeric(val):
                continue
            if best_val is None or _is_better_value(val, best_val, lower_better):
                best_val, best_name = val, name
        if best_name is not None:
            best_models[metric] = best_name
    return best_models


def _format_cv_cell(result, metric: str, precision: int) -> str:
    """Format a single CV table cell as mean (std) or N/A."""
    if result.cv_metrics and metric in result.cv_metrics:
        data = result.cv_metrics[metric]
        return _format_mean_std(data['mean'], data['std'], precision)
    if metric == 'harrell':
        # Fall back to cv_metric_mean/std
        return _format_mean_std(result.cv_metric_mean, result.cv_metric_std, precision)
    return "N/A"


def _build_header_row(label: str, label_width: int, metrics: list, col_width: int):
    """Build header row and detect lower-is-better metrics. Returns (row_str, has_lower_is_better)."""
    has_lower_is_better = False
    row = f"{label:<{label_width}}"
    for metric in metrics:
        name = get_metric_short_name(metric)
        if metric in LOWER_IS_BETTER:
            has_lower_is_better = True
        row += f" | {name:<{col_width}}"
    return row, has_lower_is_better


def _build_separator_row(label_width: int, n_metrics: int, col_width: int) -> str:
    """Build separator row for a multi-column table."""
    sep = "-" * label_width
    for _ in range(n_metrics):
        sep += "-+-" + "-" * col_width
    return sep


def print_multi_model_cv_table(
    model_results: Dict[str, Any],
    n_folds: int,
    precision: int = 3,
    tuning_status: Optional[str] = None,
    display_names: Optional[Dict[str, str]] = None,
    cv_repeats: int = 1,
) -> None:
    """
    Print CV metrics table for multiple models.

    Cells are formatted as ``mean (Std)`` across CV scores; the header notes
    this so the parenthetical is self-documenting.

    Rows are ordered best-first by ``cv_metric_mean``
    (descending). ``cv_metric_mean`` is always stored in
    higher-is-better space — see
    :func:`mlsurv._shared.cv_sorted_model_keys` for the convention
    rationale.

    Parameters
    ----------
    model_results : dict
        Dict of {model_name: ModelResult}.
    n_folds : int
        Total number of CV scores (folds x repeats).
    precision : int, default=3
        Decimal precision for formatting.
    tuning_status : str, optional
        Header tuning annotation. When omitted, it is derived from the
        results' ``is_tuned`` flags: "with tuning" when any model was tuned,
        nothing when none were. Models that weren't tuned (e.g. coxph, which
        has no hyperparameters to search) just sit in the table; a separate
        warning at tuning time already explains why.
    display_names : dict, optional
        Mapping from model keys to human-readable display names.
    cv_repeats : int, default=1
        Number of CV repeats. When > 1, the header reads the base folds and
        repeat count (e.g. '5-fold x3 repeats') instead of the inflated total.
    """
    # is_tuned is authoritative (set only by the tuning paths). Annotate the
    # header "with tuning" whenever any model was actually tuned, unless the
    # caller passed an explicit override.
    if tuning_status is None and any(
        getattr(v, 'is_tuned', False) for v in model_results.values()
    ):
        tuning_status = "with tuning"

    suffix = f", {tuning_status}" if tuning_status else ""
    fold_label = _format_cv_fold_label(n_folds, cv_repeats)
    header = f"Cross-Validation Results ({fold_label}{suffix}; cells: mean (Std))"

    labels = {k: resolve_display_name(k, display_names) for k in model_results}
    model_col_width = _compute_model_col_width(labels.values())
    metric_col_width = 16

    available_metrics = _get_available_cv_metrics(model_results)

    from mlsurv._shared import cv_sorted_model_keys
    ordered_keys = cv_sorted_model_keys(model_results)

    best_cv_models = _find_best_cv_models(model_results, available_metrics)

    def cell_builder(model_name):
        result = model_results[model_name]
        return [
            f"{_format_cv_cell(result, metric, precision):<{metric_col_width}}"
            for metric in available_metrics
        ]

    # CV table never appends " <best" suffix (unlike eval/bootstrap multi-tables).
    _print_multi_table(
        header_text=header,
        model_names=ordered_keys,
        labels=labels,
        metric_keys=available_metrics,
        model_col_width=model_col_width,
        metric_col_width=metric_col_width,
        cell_builder=cell_builder,
        best_models=best_cv_models,
        best_overall_check=None,
    )


def _filter_by_model_type(df, model_name: Optional[str], _warn_stacklevel=3):
    """Filter a DataFrame to rows where model_type == model_name.

    Returns the original df if model_name is None, the column is absent,
    or the filter yields no rows (fallback to unfiltered to avoid empty output).
    """
    if model_name is None or 'model_type' not in df.columns:
        return df
    filtered = df[df['model_type'] == model_name]
    if filtered.empty:
        available = sorted(df['model_type'].unique())
        warnings.warn(
            f"model_name={model_name!r} not found. "
            f"Available models: {available}. Showing all models.",
            InternalWarning,
            stacklevel=_warn_stacklevel,
        )
        return df
    return filtered


def _annotate_metric_name(metric_key: str) -> tuple:
    """Return (annotated_display_name, adds_lower_flag, adds_ideal_flag)."""
    info = METRIC_REGISTRY.get(metric_key)
    display_name = get_metric_display_name(metric_key)
    if info and info.ideal_value is not None:
        return display_name + f" (={_format_ideal(info.ideal_value)})", False, True
    if metric_key in LOWER_IS_BETTER:
        return display_name + " *", True, False
    return display_name, False, False


def _resolve_point_estimate(row, metric: str, point_estimates) -> float:
    """Return the best available point estimate: explicit dict > 'point' column > 'mean' column."""
    if point_estimates and metric in point_estimates:
        return point_estimates[metric]
    if 'point' in row.index and _is_valid_numeric(row['point']):
        return row['point']
    return row['mean']


def print_bootstrap_table(
    summary,  # DataFrame with columns: [model_type, metric, point, mean, std, ci_lower, ci_upper]
    n_bootstrap: int,
    ci_level: float = 0.95,
    precision: int = 3,
    model_name: Optional[str] = None,
    point_estimates: Optional[Dict[str, float]] = None,
    display_name: Optional[str] = None,
    _warn_stacklevel=3,
    delta_baseline=None,
) -> None:
    """
    Print bootstrap metrics table with point estimates and confidence intervals.

    Following best practices, displays point estimates from the full test set
    with bootstrap confidence intervals for uncertainty quantification.

    Parameters
    ----------
    summary : DataFrame
        Bootstrap summary with columns [model_type, metric, point, mean, std, ci_lower, ci_upper].
    n_bootstrap : int
        Number of bootstrap iterations.
    ci_level : float, default=0.95
        Confidence level.
    precision : int, default=3
        Decimal precision.
    model_name : str, optional
        Raw model key for DataFrame filtering (e.g., 'rsf').
    point_estimates : dict, optional
        Point estimates from full test set: {metric_name: value}.
        Priority: point_estimates dict > summary 'point' column > bootstrap mean.
        Keys match ``ALL_METRICS`` (e.g., 'harrell', 'ipcw', 'integrated_brier', 'ctd').
    display_name : str, optional
        Human-readable name for table header (e.g., 'RSF'). Falls back to
        model_name if not provided.
    delta_baseline : DataFrame, optional
        Held-out-test concordance. When given, each value shows ``(Δ ...)`` =
        value minus the baseline for this model/metric, with a footnote.
    """
    ci_pct = f"{ci_level * 100:g}"
    label = display_name or model_name
    metric_col_width = 26
    value_col_width = 28 + (_DELTA_WIDTH if delta_baseline is not None else 0)  # fit "0.698 [0.654, 0.738]" (+Δ)

    summary = _filter_by_model_type(summary, model_name, _warn_stacklevel=_warn_stacklevel)
    model_baseline = _filter_baseline(delta_baseline, model_name)

    rows = []
    has_lower_is_better = False
    has_ideal = False
    for metric in ALL_METRICS:
        mask = summary['metric'] == metric
        if not mask.any():
            continue
        row = summary.loc[mask].iloc[0]
        metric_display_name, adds_lower, adds_ideal = _annotate_metric_name(metric)
        has_lower_is_better |= adds_lower
        has_ideal |= adds_ideal

        value = _resolve_point_estimate(row, metric, point_estimates)
        if not _is_valid_numeric(value):
            value_with_ci = "N/A"
        else:
            ci = _format_ci(row['ci_lower'], row['ci_upper'], precision)
            delta = _format_delta(value, model_baseline, get_concordance_column(metric), precision)
            value_with_ci = f"{value:.{precision}f} {ci}{delta}"
        rows.append((metric_display_name, value_with_ci))

    _print_single_table(
        header_text=f"Bootstrap Results ({n_bootstrap} iterations, {ci_pct}% CI)",
        label=label,
        col1_label='Metric', col1_width=metric_col_width,
        col2_label=f'Value [{ci_pct}% CI]', col2_width=value_col_width,
        rows=rows,
        has_lower_is_better=has_lower_is_better,
        has_ideal=has_ideal,
        delta_note=_DELTA_NOTE if delta_baseline is not None else None,
    )


def _has_valid_point_column(df) -> bool:
    """Check if a DataFrame has a 'point' column with at least one non-NaN value."""
    return 'point' in df.columns and df['point'].notna().any()


def _find_best_models(summary, concordance, metrics_config):
    """Find best model for each metric using concordance, point estimates, or bootstrap means."""
    best_models = {}
    for metric, col in metrics_config:
        lower_better = METRIC_REGISTRY[metric].lower_is_better
        if col in concordance.columns:
            best_idx = concordance[col].idxmin() if lower_better else concordance[col].idxmax()
            best_models[metric] = concordance.loc[best_idx, 'model_type']
        else:
            metric_summary = summary[summary['metric'] == metric]
            if len(metric_summary) > 0:
                val_col = 'point' if _has_valid_point_column(metric_summary) else 'mean'
                best_idx = metric_summary[val_col].idxmin() if lower_better else metric_summary[val_col].idxmax()
                best_models[metric] = metric_summary.loc[best_idx, 'model_type']
    return best_models


def _resolve_point_value(metric_summary, col, model_concordance):
    """Resolve the best available point value: concordance > point column > bootstrap mean."""
    if col in model_concordance.columns:
        val = model_concordance[col].iloc[0]
        if _is_valid_numeric(val):
            return val
    if 'point' in metric_summary.columns and _is_valid_numeric(metric_summary['point'].iloc[0]):
        return metric_summary['point'].iloc[0]
    return metric_summary['mean'].iloc[0]


def _filter_baseline(delta_baseline, model_name):
    """Slice a baseline concordance to one model's rows.

    Returns ``None`` when there is no usable baseline (none supplied, or no
    ``model_type`` column). Returns an **empty** DataFrame when the baseline
    exists but does not contain this model — callers must test ``len(...)``,
    not ``is None``, to detect the absent-model case (``_format_delta`` does).
    """
    if delta_baseline is None or 'model_type' not in delta_baseline.columns:
        return None
    return delta_baseline[delta_baseline['model_type'] == model_name]


def _format_delta(current, model_baseline, col, precision) -> str:
    """Return ``' (Δ +0.043)'`` for the change from the held-out-test baseline.

    ``model_baseline`` is the baseline concordance rows for one model (already
    filtered by ``model_type``), or ``None`` when no baseline was supplied. The
    delta is the raw signed difference ``current - baseline``; the table's
    existing ``* Lower is better.`` footnote conveys per-metric direction.
    Returns ``''`` on any miss (no baseline, model/metric absent, non-finite),
    which is how reference rows and never-evaluated models skip the annotation.
    """
    if model_baseline is None or not _is_valid_numeric(current):
        return ""
    if col not in model_baseline.columns or len(model_baseline) == 0:
        return ""
    # A concordance table holds one row per model_type; take it. (Recalibration
    # keeps recalibrated point estimates in-place, not as a second row.)
    base = model_baseline[col].iloc[0]
    if not _is_valid_numeric(base):
        return ""
    return f" (Δ {current - base:+.{precision}f})"


def _format_bootstrap_cell(model_summary, metric, col, model_concordance, precision,
                           model_baseline=None):
    """Format a single bootstrap table cell with point estimate and CI."""
    metric_summary = model_summary[model_summary['metric'] == metric]
    if len(metric_summary) == 0:
        return "N/A"
    ci_lower = metric_summary['ci_lower'].iloc[0]
    ci_upper = metric_summary['ci_upper'].iloc[0]
    point_val = _resolve_point_value(metric_summary, col, model_concordance)
    if not _is_valid_numeric(point_val):
        return "N/A"
    ci_str = _format_ci(ci_lower, ci_upper, precision)
    return f"{point_val:.{precision}f} {ci_str}{_format_delta(point_val, model_baseline, col, precision)}"


def _is_best_across_all(model_name, best_models, metrics_config):
    """Check if model_name is the best for every metric that has a best model."""
    relevant = {m: best for m, _ in metrics_config if (best := best_models.get(m)) is not None}
    return relevant and all(best == model_name for best in relevant.values())


def _mark_best_in_cells(
    cells: List[str],
    model_name: str,
    metric_keys: List[str],
    best_models: Dict[str, str],
    col_width: int,
    multi: bool,
) -> List[str]:
    """Insert best-metric marker on cells where this model is best for that metric."""
    if not multi:
        return cells
    marked = []
    for cell, metric in zip(cells, metric_keys):
        value = cell.strip()
        if best_models.get(metric) == model_name:
            marked.append(f"{_BEST_MARKER} {value:<{col_width - 2}}")
        else:
            marked.append(f"  {value:<{col_width - 2}}")
    return marked


def _build_bootstrap_row(model_name, summary, concordance, metrics_config, precision, col_width,
                         delta_baseline=None):
    """Build a single data row for the bootstrap table."""
    model_concordance = concordance[concordance['model_type'] == model_name]
    model_summary = summary[summary['model_type'] == model_name]
    model_baseline = _filter_baseline(delta_baseline, model_name)
    cells = []
    for metric, col in metrics_config:
        value_str = _format_bootstrap_cell(
            model_summary, metric, col, model_concordance, precision, model_baseline,
        )
        cells.append(f"{value_str:<{col_width}}")
    return cells


def _compute_model_col_width(model_names):
    """Compute model column width from the longest model name."""
    names = list(model_names)
    if not names:
        return 8
    return max(8, max(len(str(name)) for name in names) + 1)


def _format_model_row(model_name, cells, model_col_width):
    """Format a model data row from pre-formatted cells."""
    return f"{model_name:<{model_col_width}}" + "".join(f" | {c}" for c in cells)


def print_multi_model_bootstrap_table(
    summary,
    concordance,
    n_bootstrap: int,
    ci_level: float = 0.95,
    precision: int = 3,
    model_order: Optional[List[str]] = None,
    display_names: Optional[Dict[str, str]] = None,
    delta_baseline=None,
) -> None:
    """Print bootstrap comparison table for multiple models.

    Parameters
    ----------
    summary : DataFrame
        BootstrapResult.summary with per-model, per-metric point estimates
        and confidence-interval bounds.
    concordance : DataFrame
        EvaluationResult.concordance with a ``model_type`` column and
        per-metric columns, used to order/label rows.
    n_bootstrap : int
        Number of bootstrap iterations, shown in the table header.
    ci_level : float, default=0.95
        Confidence level used for the interval columns and header label.
    precision : int, default=3
        Decimal precision for formatted values.
    model_order : list of str, optional
        Display order for model rows. Defaults to
        ``concordance['model_type'].unique()``.
    display_names : dict, optional
        Mapping from model keys to human-readable display names.
    delta_baseline : DataFrame, optional
        Held-out-test concordance (``model_type`` + per-metric columns). When
        given, each cell shows ``(Δ ...)`` = value minus the baseline for that
        model/metric, and a ``Δ = ...`` footnote is printed. Used by
        ``validate()`` to show the generalization gap.
    """
    ci_pct = f"{ci_level * 100:g}"
    header = f"Test Set Results ({n_bootstrap} bootstrap iterations, {ci_pct}% CI)"

    model_names = model_order if model_order is not None else concordance['model_type'].unique().tolist()
    metrics_config = [(m, get_concordance_column(m)) for m in ALL_METRICS]
    metric_keys = [m for m, _ in metrics_config]

    labels = {k: resolve_display_name(k, display_names) for k in model_names}
    model_col_width = _compute_model_col_width(labels.values())
    metric_col_width = 24 + (_DELTA_WIDTH if delta_baseline is not None else 0)

    best_models = _find_best_models(summary, concordance, metrics_config)

    def cell_builder(model_name):
        return _build_bootstrap_row(
            model_name, summary, concordance, metrics_config, precision, metric_col_width,
            delta_baseline,
        )

    _print_multi_table(
        header_text=header,
        model_names=model_names,
        labels=labels,
        metric_keys=metric_keys,
        model_col_width=model_col_width,
        metric_col_width=metric_col_width,
        cell_builder=cell_builder,
        best_models=best_models,
        best_overall_check=lambda name: _is_best_across_all(name, best_models, metrics_config),
        delta_note=_DELTA_NOTE if delta_baseline is not None else None,
    )


def _format_evaluation_cell(model_concordance, col, precision, model_baseline=None):
    """Format a single evaluation table cell with the point value from concordance."""
    if col not in model_concordance.columns or len(model_concordance) == 0:
        return "N/A"
    val = model_concordance[col].iloc[0]
    if not _is_valid_numeric(val):
        return "N/A"
    return f"{_format_value(val, precision)}{_format_delta(val, model_baseline, col, precision)}"


def print_multi_model_evaluation_table(
    concordance,
    precision: int = 3,
    model_order: Optional[List[str]] = None,
    display_names: Optional[Dict[str, str]] = None,
    _warn_stacklevel: int = 2,
    delta_baseline=None,
) -> None:
    """Print test-set evaluation comparison table for multiple models (no bootstrap).

    Mirrors :func:`print_multi_model_bootstrap_table` but reports point estimates
    from ``concordance`` only — no confidence intervals.

    Parameters
    ----------
    concordance : DataFrame
        EvaluationResult.concordance with a ``model_type`` column and per-metric
        columns (e.g., ``c_index_test``, ``integrated_brier_test``).
    precision : int, default=3
        Decimal precision for formatted values.
    model_order : list of str, optional
        Display order for model rows. Defaults to ``concordance['model_type'].unique()``.
    display_names : dict, optional
        Mapping from model keys to human-readable display names.
    delta_baseline : DataFrame, optional
        Held-out-test concordance. When given, each cell shows ``(Δ ...)`` =
        value minus the baseline for that model/metric, with a footnote.
    """
    available = (
        concordance['model_type'].unique().tolist()
        if 'model_type' in concordance.columns else []
    )
    if model_order is not None:
        missing = [m for m in model_order if m not in available]
        if missing:
            warnings.warn(
                f"model_order entries not found in concordance: {missing}. "
                f"Available models: {sorted(available)}. "
                f"Missing rows will display as N/A.",
                InternalWarning,
                stacklevel=_warn_stacklevel,
            )
        model_names = model_order
    else:
        model_names = available
    metrics_config = [(m, get_concordance_column(m)) for m in ALL_METRICS]
    # Drop metrics absent from concordance so the header matches the data.
    metrics_config = [(m, c) for m, c in metrics_config if c in concordance.columns]
    metric_keys = [m for m, _ in metrics_config]

    labels = {k: resolve_display_name(k, display_names) for k in model_names}
    model_col_width = _compute_model_col_width(labels.values())
    metric_col_width = 18 + (_DELTA_WIDTH if delta_baseline is not None else 0)

    # _find_best_models works for the no-bootstrap path: when ``summary`` is
    # empty, the concordance branch in _find_best_models selects the winner directly.
    empty_summary = pd.DataFrame(columns=['model_type', 'metric', 'point', 'mean'])
    best_models = _find_best_models(empty_summary, concordance, metrics_config)

    def cell_builder(model_name):
        model_concordance = concordance[concordance['model_type'] == model_name]
        model_baseline = _filter_baseline(delta_baseline, model_name)
        return [
            f"{_format_evaluation_cell(model_concordance, col, precision, model_baseline):<{metric_col_width}}"
            for _, col in metrics_config
        ]

    _print_multi_table(
        header_text="Test Set Evaluation",
        model_names=model_names,
        labels=labels,
        metric_keys=metric_keys,
        model_col_width=model_col_width,
        metric_col_width=metric_col_width,
        cell_builder=cell_builder,
        best_models=best_models,
        best_overall_check=lambda name: _is_best_across_all(name, best_models, metrics_config),
        delta_note=_DELTA_NOTE if delta_baseline is not None else None,
    )


def _format_param_value(value) -> str:
    """Format a hyperparameter value for display."""
    # bool is a subclass of int; both use str() so they can share a branch
    if isinstance(value, (bool, int)):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return repr(value)


# Prefix-to-label mapping for grouped best params display.
_COMPONENT_PREFIXES = {
    'model__': 'Model',
    'feature_selector__': 'Feature Selector',
    'imputer__': 'Imputer',
    'scaler__': 'Scaler',
}


def _group_best_params(best_params: dict) -> Dict[str, Dict[str, str]]:
    """Group prefixed best_params into {component: {clean_param: formatted_value}}.

    Unprefixed keys are model params. The selector's k arrives prefixed as
    feature_selector__k and is grouped via the prefix loop.
    """
    groups: Dict[str, Dict[str, str]] = {}
    for key, value in best_params.items():
        formatted = _format_param_value(value)
        for prefix, label in _COMPONENT_PREFIXES.items():
            if key.startswith(prefix):
                clean = key[len(prefix):]
                groups.setdefault(label, {})[clean] = formatted
                break
        else:
            # No prefix matched -- unprefixed keys are model params
            groups.setdefault('Model', {})[key] = formatted
    return groups


def _format_params_inline(params: Dict[str, str]) -> str:
    """Format {k: v} dict as 'k1=v1, k2=v2' string."""
    return ", ".join(f"{k}={v}" for k, v in params.items())


def print_best_params(best_params: dict, model_name: Optional[str] = None) -> None:
    """Print best hyperparameters for a single model.

    Simple (1 group): Best Parameters (rsf): n_estimators=200, max_depth=15
    Complex (2+ groups): Grouped with component labels on separate lines.
    """
    if not best_params:
        return

    groups = _group_best_params(best_params)
    prefix = f"Best Parameters ({model_name})" if model_name else "Best Parameters"

    if len(groups) == 1:
        _, params = next(iter(groups.items()))
        print(f"{prefix}: {_format_params_inline(params)}")
    else:
        print(f"{prefix}:")
        for component, params in groups.items():
            print(f"  {component}: {_format_params_inline(params)}")


def print_multi_model_best_params(
    model_results: Dict[str, Any],
    display_names: Optional[Dict[str, str]] = None,
) -> None:
    """Print best params for multiple models, one line each.

    Skips models without best_params_. Cleans prefixed keys.

    Parameters
    ----------
    model_results : dict
        Dict of {model_name: ModelResult}.
    display_names : dict, optional
        Mapping from model keys to human-readable display names.
    """
    lines = []
    for name, result in model_results.items():
        bp = getattr(result, 'best_params_', None)
        if not bp:
            continue
        label = resolve_display_name(name, display_names)
        # Flatten all groups into a single params dict for one-line display
        flat_params = {
            k: v
            for params in _group_best_params(bp).values()
            for k, v in params.items()
        }
        lines.append(f"  {label}: {_format_params_inline(flat_params)}")

    if lines:
        print("Best Parameters:")
        for line in lines:
            print(line)


def print_evaluation_table(
    concordance,  # DataFrame with c_index_test column
    model_name: Optional[str] = None,
    precision: int = 3,
    display_name: Optional[str] = None,
    _warn_stacklevel=3,
    delta_baseline=None,
) -> None:
    """
    Print test set evaluation table (point estimates only).

    Parameters
    ----------
    concordance : DataFrame
        DataFrame with c_index_test column.
    model_name : str, optional
        Raw model key used to filter concordance rows by model_type.
    precision : int, default=3
        Decimal precision.
    display_name : str, optional
        Human-readable name for the table header (e.g., 'RSF'). Falls back
        to model_name if not provided.
    delta_baseline : DataFrame, optional
        Held-out-test concordance. When given, each value shows ``(Δ ...)`` =
        value minus the baseline for this model/metric, with a footnote.
    """
    label = display_name or model_name
    metric_col_width = 26
    value_col_width = 21 + (_DELTA_WIDTH if delta_baseline is not None else 0)

    concordance = _filter_by_model_type(concordance, model_name, _warn_stacklevel=_warn_stacklevel)
    model_baseline = _filter_baseline(delta_baseline, model_name)

    rows = []
    has_lower_is_better = False
    has_ideal = False
    for metric_key in ALL_METRICS:
        col_name = get_concordance_column(metric_key)
        if col_name not in concordance.columns:
            continue
        value = concordance[col_name].iloc[0]
        if not _is_valid_numeric(value):
            continue
        annotated_name, adds_lower, adds_ideal = _annotate_metric_name(metric_key)
        has_lower_is_better |= adds_lower
        has_ideal |= adds_ideal
        delta = _format_delta(value, model_baseline, col_name, precision)
        rows.append((annotated_name, f"{_format_value(value, precision)}{delta}"))

    _print_single_table(
        header_text="Test Set Evaluation",
        label=label,
        col1_label='Metric', col1_width=metric_col_width,
        col2_label='Value', col2_width=value_col_width,
        rows=rows,
        has_lower_is_better=has_lower_is_better,
        has_ideal=has_ideal,
        delta_note=_DELTA_NOTE if delta_baseline is not None else None,
    )


# =====================================================================
# Paired model comparison
# =====================================================================

_METHOD_LABEL_MAP = {
    'paired_bootstrap': 'paired bootstrap, Holm correction',
    'kang': 'Kang z-test',
}


def _resolve_method_label(comparison_df, method_label_override: Optional[str]) -> str:
    """Determine the method label from an override or the 'method' column."""
    if method_label_override is not None:
        return method_label_override
    if 'method' not in comparison_df.columns:
        return "paired comparison"
    parts = [_METHOD_LABEL_MAP.get(m, m) for m in comparison_df['method'].unique()]
    return ", ".join(parts)


def _build_comparison_row(row, p_col: str, display_names, precision: int) -> tuple:
    """Build a single paired comparison row: (pair_str, delta_str, p_str)."""
    model_a = resolve_display_name(row['model_a'], display_names)
    model_b = resolve_display_name(row['model_b'], display_names)
    pair = f"{model_a} vs {model_b}"

    delta = row['delta']
    ci_lower = row.get('ci_lower', np.nan)
    ci_upper = row.get('ci_upper', np.nan)

    if _is_valid_numeric(delta):
        sign = "+" if delta >= 0 else ""
        delta_str = f"{sign}{delta:.{precision}f}"
        if _is_valid_numeric(ci_lower) and _is_valid_numeric(ci_upper):
            delta_str += f" [{ci_lower:.{precision}f}, {ci_upper:.{precision}f}]"
    else:
        delta_str = "N/A"

    p_val = row.get(p_col, np.nan)
    p_str = _format_pvalue(p_val) + _significance_stars(p_val)
    return (pair, delta_str, p_str)


def _comparison_col_widths(rows: List[tuple], p_header: str) -> tuple:
    """Compute (pair_w, delta_w, p_w) column widths from pre-built rows."""
    return (
        max(4, max(len(r[0]) for r in rows)),
        max(14, max(len(r[1]) for r in rows)),
        max(len(p_header), max(len(r[2]) for r in rows)),
    )


def print_paired_comparison(
    comparison_df,
    precision: int = 3,
    display_names: Optional[Dict[str, str]] = None,
    method_label: Optional[str] = None,
    ci_level: float = 0.95,
) -> None:
    """Print paired model comparison table.

    Parameters
    ----------
    comparison_df : DataFrame
        Output from ``compare_bootstrap()``, ``compare_kang()``, or
        concatenated results. Expected columns: model_a, model_b,
        delta, ci_lower, ci_upper. Plus either p_adjusted or p_value.
    precision : int, default=3
        Decimal precision for delta and CI values.
    display_names : dict, optional
        Mapping from model keys to human-readable display names.
    method_label : str, optional
        Override label for the table header (e.g., "Kang z-test").
        If None, auto-detected from the ``method`` column.
    ci_level : float, default=0.95
        Confidence level for CI display in the column header.
    """
    if comparison_df is None or comparison_df.empty:
        return

    method_label = _resolve_method_label(comparison_df, method_label)

    # Determine p-value column
    has_adjusted = 'p_adjusted' in comparison_df.columns and comparison_df['p_adjusted'].notna().any()
    p_col = 'p_adjusted' if has_adjusted else 'p_value'
    p_header = 'p-adj' if has_adjusted else 'p-value'

    rows = [
        _build_comparison_row(row, p_col, display_names, precision)
        for _, row in comparison_df.iterrows()
    ]

    if not rows:
        return

    pair_w, delta_w, p_w = _comparison_col_widths(rows, p_header)

    total_width = pair_w + 3 + delta_w + 3 + p_w

    header = f"Paired Model Comparison ({method_label})"
    print("=" * total_width)
    print(header)
    print("=" * total_width)
    ci_pct = f"{ci_level * 100:g}"
    print(f"{'Pair':<{pair_w}} | {f'Delta [{ci_pct}% CI]':<{delta_w}} | {p_header:<{p_w}}")
    print(_make_separator(pair_w, delta_w, p_w))

    for pair, delta_str, p_str in rows:
        print(f"{pair:<{pair_w}} | {delta_str:<{delta_w}} | {p_str:<{p_w}}")

    print("=" * total_width)
    print(_SIGNIFICANCE_FOOTER)


# =====================================================================
# Statistical test print functions
# =====================================================================

def print_ph_test(df, precision: int = 3) -> None:
    """Print formatted Grambsch-Therneau proportional hazards test results.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``ph_test()``. Columns: feature, test_statistic, p_value,
        p_adjusted, significant.
    precision : int, default=3
        Decimal precision for the test statistic.
    """
    if df is None or df.empty:
        return

    # Separate GLOBAL row from feature rows
    global_mask = df['feature'] == 'GLOBAL'
    features = df[~global_mask]
    global_row = df[global_mask]

    # Dynamic column widths
    all_names = df['feature'].tolist()
    feat_w = max(7, max(len(str(n)) for n in all_names) + 1)
    stat_w = 8
    p_w = 14
    padj_w = 14

    total_w = feat_w + 3 + stat_w + 3 + p_w + 3 + padj_w
    sep = _make_separator(feat_w, stat_w, p_w, padj_w)

    print("=" * total_w)
    print("Proportional Hazards Test (Grambsch-Therneau)")
    print("=" * total_w)
    print(f"{'Feature':<{feat_w}} | {'Chi-sq':>{stat_w}} | {'p-value':<{p_w}} | {'p-adj (BH)':<{padj_w}}")
    print(sep)

    def _print_row(name: str, stat, p, p_adj) -> None:
        stat_str, p_str = _format_stat_and_p(stat, p, precision)
        if not _is_valid_numeric(p_adj):
            padj_str = "\u2014"
        else:
            padj_str = _format_pvalue(p_adj) + _significance_stars(p_adj)
        print(f"{name:<{feat_w}} | {stat_str:>{stat_w}} | {p_str:<{p_w}} | {padj_str:<{padj_w}}")

    for _, row in features.iterrows():
        _print_row(str(row['feature']), row['test_statistic'],
                   row['p_value'], row['p_adjusted'])

    if not global_row.empty:
        print(sep)
        g = global_row.iloc[0]
        _print_row('GLOBAL', g['test_statistic'], g['p_value'], g['p_adjusted'])

    print("=" * total_w)
    print(_SIGNIFICANCE_FOOTER)
    print("Note: p-adj (BH) shows Benjamini-Hochberg FDR-corrected p-values.")
    print("      Significance is based on adjusted p < 0.05 (per-feature) or raw p < 0.05 (GLOBAL).")


def print_rmst_test(result, precision: int = 3) -> None:
    """Print formatted RMST comparison results (two groups).

    Parameters
    ----------
    result : RMSTResult
        Output of ``rmst_test()``.
    precision : int, default=3
        Decimal precision.
    """
    if result is None or not result.rmst:
        return

    groups = list(result.rmst.keys())
    group_w = max(12, max(len(str(g)) for g in groups) + 1)
    val_w = 18
    total_w = group_w + 3 + val_w

    print("=" * total_w)
    print(f"RMST Comparison (tau = {_format_tau(result.tau, precision)})")
    print("=" * total_w)
    _print_rmst_group_table(result, group_w, val_w, precision)

    print("-" * total_w)
    ci_str = _format_ci(result.ci_lower, result.ci_upper, precision)
    if _is_valid_numeric(result.diff):
        sign = "+" if result.diff >= 0 else ""
        diff_str = f"{sign}{result.diff:.{precision}f}"
    else:
        diff_str = "NaN"
    print(f"{'Difference':<{group_w}} : {diff_str} {ci_str}")
    z_str = _format_value(result.z_stat, precision)
    print(f"{'z-statistic':<{group_w}} : {z_str}")
    p_str = _format_pvalue(result.p_value) + _significance_stars(result.p_value)
    print(f"{'p-value':<{group_w}} : {p_str}")
    print("=" * total_w)
    print(_SIGNIFICANCE_FOOTER)


def _build_rmst_pairwise_rows(result, precision: int) -> List[tuple]:
    """Build pairwise RMST comparison rows: list of (pair, delta_str, p_str)."""
    rows = []
    for comp in result.pairwise:
        g0, g1 = comp['groups']
        pair = f"{g0} vs {g1}"
        ci_str = _format_ci(comp['ci_lower'], comp['ci_upper'], precision)
        if _is_valid_numeric(comp['diff']):
            sign = "+" if comp['diff'] >= 0 else ""
            delta_str = f"{sign}{comp['diff']:.{precision}f} {ci_str}"
        else:
            delta_str = f"NaN {ci_str}"
        p_val = comp['adjusted_p']
        p_str = _format_pvalue(p_val) + _significance_stars(p_val)
        rows.append((pair, delta_str, p_str))
    return rows


def print_rmst_pairwise(result, precision: int = 3) -> None:
    """Print formatted pairwise RMST comparison results (2+ groups).

    Parameters
    ----------
    result : PairwiseRMSTResult
        Output of ``rmst_test_pairwise()``.
    precision : int, default=3
        Decimal precision.
    """
    if result is None or not result.rmst:
        return

    groups = list(result.rmst.keys())
    group_w = max(12, max(len(str(g)) for g in groups) + 1)
    val_w = 18
    total_w = group_w + 3 + val_w

    print("=" * total_w)
    print(f"RMST Comparison (tau = {_format_tau(result.tau, precision)}, {result.p_adjust_method} correction)")
    print("=" * total_w)
    _print_rmst_group_table(result, group_w, val_w, precision)

    print("=" * total_w)

    if not result.pairwise:
        return

    pair_rows = _build_rmst_pairwise_rows(result, precision)

    pair_w, delta_w, p_w = _comparison_col_widths(pair_rows, 'p-adj')

    pw_total = pair_w + 3 + delta_w + 3 + p_w
    print()
    print(f"{'Pair':<{pair_w}} | {'Delta [95% CI]':<{delta_w}} | {'p-adj':<{p_w}}")
    print(_make_separator(pair_w, delta_w, p_w))
    for pair, delta_str, p_str in pair_rows:
        print(f"{pair:<{pair_w}} | {delta_str:<{delta_w}} | {p_str:<{p_w}}")
    print("=" * pw_total)
    print(_SIGNIFICANCE_FOOTER)


def print_calibration_test(
    chi2: float,
    p_value: float,
    method: str = 'gnd',
    precision: int = 3,
) -> None:
    """Print formatted calibration goodness-of-fit test result.

    Parameters
    ----------
    chi2 : float
        Chi-squared test statistic.
    p_value : float
        P-value from the calibration test.
    method : str, default='gnd'
        Test method: ``'gnd'`` or ``'hl'``.
    precision : int, default=3
        Decimal precision.
    """
    method_labels = {'gnd': 'GND', 'hl': 'Hosmer-Lemeshow'}
    label = method_labels.get(method, method.upper())

    chi2_str, p_str = _format_stat_and_p(chi2, p_value, precision)
    print(f"Calibration GOF Test ({label}): chi2 = {chi2_str}, p = {p_str}")
    print("Note: Significant p-value indicates predicted and observed survival probabilities diverge (poor calibration).")


_INCREMENTAL_DISPLAY = {
    "delta_c_index": "Delta C-index",
    "nri": "NRI",
    "idi": "IDI",
}


def _print_benchmark_univariate_cox(cox, precision: int) -> None:
    """Print the univariate Cox reference table section."""
    if cox.empty:
        return
    print()
    print("-" * 60)
    print("Univariate Cox Regression (References)")
    print("-" * 60)
    ref_w = max(12, max(len(str(b)) for b in cox["reference"]) + 1)
    print(f"{'Reference':<{ref_w}} | {'HR [95% CI]':<24} | {'p-value':<10}")
    print(_make_separator(ref_w, 24, 10))
    for _, row in cox.iterrows():
        if np.isnan(row["hazard_ratio"]):
            hr_str, p_str = "N/A", "N/A"
        else:
            hr_str = (
                f"{row['hazard_ratio']:.{precision}f} "
                f"[{row['hr_lower_95']:.{precision}f}, "
                f"{row['hr_upper_95']:.{precision}f}]"
            )
            p_str = _format_p_sig(row['p_value'])
        print(f"{row['reference']:<{ref_w}} | {hr_str:<24} | {p_str:<10}")


def _format_incremental_cell(row, has_ci: bool, precision: int) -> str:
    """Format a single incremental metric cell for console output."""
    if np.isnan(row["point"]):
        return "\u2014"
    val = f"{row['point']:.{precision}f}"
    if has_ci and not np.isnan(row["ci_lower"]) and not np.isnan(row["ci_upper"]):
        val += f" [{row['ci_lower']:.{precision}f},{row['ci_upper']:.{precision}f}]"
    return val


def _format_incremental_row(grp, metrics_present, has_ci, precision, col_w):
    """Format one (model, reference) pair as a row of metric cells.

    Returns
    -------
    tuple[str, dict]
        The formatted row string and the metric_lookup dict (keyed by
        metric name) for reuse by callers that need per-metric data.
    """
    metric_lookup = {r["metric"]: r for _, r in grp.iterrows()}
    cells = []
    for mk in metrics_present:
        val = (
            _format_incremental_cell(metric_lookup[mk], has_ci, precision)
            if mk in metric_lookup
            else "\u2014"
        )
        cells.append(f"{val:<{col_w}}")
    return " | ".join(cells), metric_lookup


def _print_benchmark_incremental(inc, precision: int) -> None:
    """Print the incremental value (Delta C-index, NRI, IDI) section.

    Pivoted layout: one line per (model, reference) with all metrics inline.
    """
    if inc.empty:
        return

    metrics_present = [
        m for m in ["delta_c_index", "nri", "idi"]
        if (inc["metric"] == m).any()
    ]
    if not metrics_present:
        return

    print()
    print("-" * 60)
    print("Incremental Value (Model vs Reference)")
    print("-" * 60)
    has_ci = inc["ci_lower"].notna().any()
    has_pval = "p_value" in inc.columns and inc["p_value"].notna().any()

    metric_headers = [_INCREMENTAL_DISPLAY.get(m, m) for m in metrics_present]
    min_col_w = 22 if has_ci else 10
    col_w = max(min_col_w, max(len(h) for h in metric_headers) + 1)

    label_w = max(20, max(
        len(f"{r['model']} vs {r['reference']}")
        for _, r in inc.iterrows()
    ) + 1)

    header = f"  {'':>{label_w}} | " + " | ".join(
        f"{mh:<{col_w}}" for mh in metric_headers
    )
    print(header)
    print("  " + _make_separator(label_w, *([col_w] * len(metrics_present))))

    for (model, ref), grp in inc.groupby(["model", "reference"], sort=False):
        label = f"{model} vs {ref}"
        line, metric_lookup = _format_incremental_row(
            grp, metrics_present, has_ci, precision, col_w,
        )
        print(f"  {label:<{label_w}} | {line}")
        if has_pval:
            p_cells = []
            for mk in metrics_present:
                if mk in metric_lookup:
                    p = metric_lookup[mk].get("p_value", np.nan)
                    p_str = f"p={_format_p_sig(p)}" if not np.isnan(p) else ""
                else:
                    p_str = ""
                p_cells.append(f"{p_str:<{col_w}}")
            print(f"  {'':>{label_w}} | {' | '.join(p_cells)}")


def _print_benchmark_correlation(corr, precision: int) -> None:
    """Print the model-reference correlation section."""
    if corr.empty:
        return
    print()
    print("-" * 60)
    print("Model-Reference Correlation")
    print("-" * 60)
    pair_w = max(20, max(
        len(f"{r['model']} ~ {r['reference']}")
        for _, r in corr.iterrows()
    ) + 1)
    print(f"{'Pair':<{pair_w}} | {'Spearman rho':<12} | {'Pearson r':<12}")
    print(_make_separator(pair_w, 12, 12))
    for _, row in corr.iterrows():
        pair = f"{row['model']} ~ {row['reference']}"
        sp = f"{row['spearman_rho']:.{precision}f}"
        pe = f"{row['pearson_r']:.{precision}f}"
        print(f"{pair:<{pair_w}} | {sp:<12} | {pe:<12}")


def _print_sample_size_note(result) -> None:
    """Print a note when models and references use different sample sizes."""
    if not result.sample_sizes:
        return
    model_n = max(
        (result.sample_sizes[m] for m in result.model_names
         if m in result.sample_sizes),
        default=0,
    )
    if model_n == 0:
        return

    _ds = getattr(result, 'dataset', 'test')
    _ds_label = {
        "test": "test set", "train": "training set", "full": "full dataset",
    }.get(_ds, "test set")

    # When dataset != "test", models use test data, references use
    # the selected dataset — note the intentional size difference.
    ref_sizes = {
        r: result.sample_sizes[r]
        for r in result.reference_names
        if r in result.sample_sizes
    }
    if _ds != "test" and ref_sizes:
        max_ref_n = max(ref_sizes.values())
        if max_ref_n != model_n:
            print()
            print(
                f"Note: Models evaluated on test set (N={model_n}); "
                f"references evaluated on {_ds_label} (N={max_ref_n})."
            )
            print(
                "      Incremental metrics and correlations use "
                "test data for both."
            )

    # NaN-filtered references: report any ref with fewer samples than
    # its expected base (model_n when test, max_ref_n otherwise).
    base_n = model_n if _ds == "test" else max(ref_sizes.values(), default=0)
    filtered = {
        r: n for r, n in ref_sizes.items() if n < base_n
    }
    if filtered:
        print()
        for rname, n in filtered.items():
            n_excluded = base_n - n
            print(
                f"Note: Reference '{rname}' metrics computed on N={n} "
                f"({n_excluded} excluded due to missing values)."
            )
        print("      Direct comparison may reflect population differences.")


def print_benchmark_summary(result, ci_level: Optional[float] = None, precision: int = 3) -> None:
    """Print human-readable benchmark comparison tables.

    Parameters
    ----------
    result : BenchmarkResult
        Output of ``SurvivalLearner.benchmark()``.
    ci_level : float, optional
        Confidence level used to label bootstrap intervals. When None,
        ``EvaluationResult.print_metrics`` resolves it from the attached
        bootstrap's actual level.
    precision : int
        Decimal precision.
    """
    print()
    print("=" * 60)
    _ds = getattr(result, 'dataset', 'test')
    if _ds != "test":
        print(f"Benchmark Summary  (dataset: {_ds})")
    else:
        print("Benchmark Summary")
    print("=" * 60)
    result.evaluation.print_metrics(ci_level=ci_level)
    _print_sample_size_note(result)
    _print_benchmark_univariate_cox(result.univariate_cox, precision)
    _print_benchmark_incremental(result.incremental, precision)
    _print_benchmark_correlation(result.correlation, precision)
    print()
