"""Shared helpers for report sections.

Functions here are used by multiple report sections (evaluation,
validation, environment, benchmarking, etc.).  Domain-specific helpers
remain in their respective section packages.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mlsurv.viz._palettes import get_model_display_name


def extract_eval_times_from_auc_t(auc_t) -> list[float]:
    """Extract finite eval_times from an auc_t DataFrame, or empty list."""
    if auc_t is None or auc_t.empty or 'eval_time' not in auc_t.columns:
        return []
    return sorted(
        t for t in auc_t['eval_time'].unique().tolist()
        if np.isfinite(t)
    )


def eval_grid_with_percentile_fallback(
    auc_t_candidates,
    y,
    percentiles=(25, 50, 75),
) -> tuple[list[float], bool]:
    """Resolve a realized eval grid, falling back to event-time percentiles.

    Try each auc_t DataFrame in *auc_t_candidates* in order; the first that
    yields a non-empty finite grid (via :func:`extract_eval_times_from_auc_t`)
    wins. Otherwise fall back to *percentiles* of the cohort's event times
    (all times when there are no events).

    Parameters
    ----------
    auc_t_candidates : iterable
        auc_t DataFrames (or ``None``) to try, in priority order.
    y : structured array
        Survival labels with ``time`` and ``event`` fields, used for the
        percentile fallback.
    percentiles : sequence of float, optional
        Percentiles used for the fallback when no auc_t grid is available —
        of the cohort's event times, or of all times when there are no events.

    Returns
    -------
    (list[float], bool)
        The resolved grid and *from_auc_t* — ``True`` when the grid came from
        an auc_t candidate, ``False`` when the percentile fallback was used, so
        callers can warn or annotate the divergence.
    """
    for auc_t in auc_t_candidates:
        grid = extract_eval_times_from_auc_t(auc_t)
        if grid:
            return grid, True
    event_times = y['time'][y['event']]
    source = event_times if len(event_times) > 0 else y['time']
    grid = sorted(set(np.percentile(source, list(percentiles)).tolist()))
    return grid, False


def format_eval_time(t: float) -> str:
    """Format an evaluation time compactly: integers as int, floats with 1 decimal."""
    if not np.isfinite(t):
        return str(t)
    return str(int(t)) if t == int(t) else f"{t:.1f}"


def empty_calibration_figure(title: str):
    """Placeholder figure when no calibration bins qualify."""
    from mlsurv.viz._theme import create_figure

    fig = create_figure(title=title, width="single", height="medium")
    fig.add_annotation(
        text="No qualifying calibration bins",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color="gray"),
    )
    return fig


def collect_calibration_rows(
    calibration_items,
) -> list[dict[str, Any]]:
    """Build tidy row dicts from calibration data items.

    Parameters
    ----------
    calibration_items : iterable of (model_name, item) tuples
        Each *item* is a dict with keys ``predicted``, ``observed``,
        ``ci_lower``, ``ci_upper``, and ``eval_time``.

    Returns
    -------
    list[dict]
        One dict per calibration bin with keys ``model_type``,
        ``eval_time``, ``bin_predicted``, ``bin_observed``,
        ``bin_ci_lower``, ``bin_ci_upper``.
    """
    rows: list[dict[str, Any]] = []
    for model_name, item in calibration_items:
        base: dict[str, Any] = {
            'model_type': model_name,
            'eval_time': item['eval_time'],
        }
        for p, o, cl, cu in zip(
            item['predicted'], item['observed'],
            item['ci_lower'], item['ci_upper'],
        ):
            rows.append({
                **base,
                'bin_predicted': p,
                'bin_observed': o,
                'bin_ci_lower': cl,
                'bin_ci_upper': cu,
            })
    return rows


# ------------------------------------------------------------------
# AUC(t) table & tidy-data helpers
# ------------------------------------------------------------------


def build_auc_std_lookup(boot_auc) -> dict:
    """Build a (model_type, eval_time) -> std lookup from bootstrap AUC data."""
    if boot_auc is None:
        return {}
    return boot_auc.groupby(['model_type', 'eval_time'])['auc'].std().to_dict()


def time_label(t: float) -> str:
    """Format an evaluation time as a compact column header (e.g. 't=5', 't=12')."""
    return f"t={round(t)}"


def format_auc_cell(time_data, model, t, std_lookup) -> str:
    """Format a single AUC table cell with optional std."""
    if len(time_data) == 0:
        return "N/A"
    auc_val = time_data['auc'].values[0]
    std_val = std_lookup.get((model, t))
    if std_val is not None:
        return f"{auc_val:.2f} ({std_val:.2f})"
    return f"{auc_val:.2f}"


def build_auct_summary_rows(
    auc_df: pd.DataFrame,
    std_lookup: dict,
) -> tuple[list[dict], int, list[float], list[str]]:
    """Build the model-row dicts for an AUC(t) summary table.

    Returns ``(rows, n_model_rows, eval_times, time_labels)`` where
    *rows* is a list of dicts (one per model) ready to pass to
    ``pd.DataFrame()``, *n_model_rows* is the count used by
    ``BaseSection._bold_best_auct_values`` to know which rows to bold,
    and *eval_times* / *time_labels* are the sorted time grid and
    corresponding column headers.

    Callers are responsible for:
    - calling ``warn_eval_times_mismatch()`` beforehand (evaluation only),
    - appending events/at-risk rows after model rows (evaluation only),
    - creating the DataFrame and calling ``_bold_best_auct_values()``.
    """
    models = auc_df['model_type'].unique()
    eval_times = sorted(auc_df['eval_time'].unique())
    time_labels = [time_label(t) for t in eval_times]

    rows: list[dict] = []
    for model in models:
        row: dict = {'': get_model_display_name(model)}
        model_data = auc_df[auc_df['model_type'] == model]
        for t, label in zip(eval_times, time_labels):
            time_data = model_data[model_data['eval_time'] == t]
            row[label] = format_auc_cell(time_data, model, t, std_lookup)
        rows.append(row)

    return rows, len(models), eval_times, time_labels


def build_auct_tidy_data(
    auc_df: pd.DataFrame,
    boot_auc: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build a tidy DataFrame from AUC(t) results with optional std.

    Includes ``n_events`` and ``n_at_risk`` columns when they exist in
    *auc_df* (evaluation path).  The ``model`` column is inserted as a
    human-readable display name derived from ``model_type``.
    """
    cols = ['model_type', 'eval_time', 'auc']
    for optional in ('n_events', 'n_at_risk'):
        if optional in auc_df.columns:
            cols.append(optional)

    tidy = auc_df[cols].copy()
    tidy.insert(1, 'model', tidy['model_type'].map(get_model_display_name))

    if boot_auc is not None:
        std_lookup = build_auc_std_lookup(boot_auc)
        tidy['auc_std'] = [
            std_lookup.get((mt, et))
            for mt, et in zip(tidy['model_type'], tidy['eval_time'])
        ]

    return tidy
