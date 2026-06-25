"""Bootstrap forest and raincloud plots for validation cohorts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from mlsurv.constants import ALL_METRICS, get_metric_display_name
from mlsurv.reports.sections.base import ALL_EXCEPTIONS, FigureInfo
from mlsurv.reports.sections.validation._helpers import cohort_key as _cohort_key

if TYPE_CHECKING:
    from mlsurv.reports.sections.validation._section import ValidationSection


def _build_point_estimates(summary, plot_metrics):
    """Build point estimates dict from bootstrap summary for raincloud overlay."""
    point_estimates = {}
    if summary is not None and 'point' in summary.columns:
        for metric in plot_metrics:
            rows = summary[summary['metric'] == metric]
            if not rows.empty:
                point_estimates[metric] = dict(
                    zip(rows['model_type'], rows['point'])
                )
    return point_estimates


def add_combined_bootstrap_summary(
    section: ValidationSection,
    md_lines: list,
    figures: list,
    figure_info: list,
) -> None:
    """Add bootstrap forest plot and raincloud for all cohorts."""
    for name, val_result in section.validation_results.items():
        bootstrap = val_result.bootstrap
        if bootstrap is None or bootstrap.summary is None or bootstrap.summary.empty:
            continue

        ck = _cohort_key(name)
        plot_metrics = [
            m for m in bootstrap.summary['metric'].unique()
            if m in ALL_METRICS
        ]
        if not plot_metrics:
            continue

        n_bootstrap = bootstrap.iterations['iteration'].nunique()
        md_lines.append(f"*Bootstrap performed with {n_bootstrap} iterations ({name}).*")
        md_lines.append("")

        _add_bootstrap_forest(
            section, md_lines, figures, figure_info,
            bootstrap, plot_metrics, name, ck,
        )
        _add_bootstrap_raincloud(
            section, md_lines, figures, figure_info,
            bootstrap, plot_metrics, name, ck,
        )


def _add_bootstrap_forest(
    section: ValidationSection,
    md_lines: list,
    figures: list,
    figure_info: list,
    bootstrap,
    plot_metrics: list,
    cohort_name: str,
    ck: str,
) -> None:
    """Add multi-panel forest plot from bootstrap summary."""
    from mlsurv.viz.metrics import plot_metric as _plot_metric

    try:
        summary = bootstrap.summary
        rows = summary[summary['metric'].isin(plot_metrics)].copy()
        if rows.empty:
            return

        point_wide = rows.pivot(
            index="model_type", columns="metric", values="point",
        )
        std_wide = rows.pivot(
            index="model_type", columns="metric", values="std",
        )
        std_wide.columns = [f"{c}_std" for c in std_wide.columns]
        df = pd.concat([point_wide, std_wide], axis=1).reset_index()

        ordered_cols = ["model_type"]
        for m in plot_metrics:
            if m in df.columns:
                ordered_cols.append(m)
                if f"{m}_std" in df.columns:
                    ordered_cols.append(f"{m}_std")
        df = df[[c for c in ordered_cols if c in df.columns]]

        fig = _plot_metric(
            df, model_col="model_type", metric=plot_metrics,
            style="forest",
            title=f"Bootstrap Metric Estimates: {cohort_name}",
        )

        fig_id = f'validation_forest_{ck}'
        data_key = f'forest_{ck}'
        figures.append((f'{fig_id}.png', fig))
        md_lines.append(f"![Bootstrap Forest Plot](images/{fig_id}.png)")
        md_lines.append("")

        metric_list = ', '.join(
            get_metric_display_name(m) for m in plot_metrics
        )
        figure_info.append(FigureInfo(
            figure_id=fig_id,
            title=f'Bootstrap Metric Estimates: {cohort_name}',
            data_description=f'Bootstrap summary for {metric_list} ({cohort_name})',
            data_key=data_key,
            learner_code=(
                "fig = learner.plot_metric(\n"
                "    style='forest', source='bootstrap',\n"
                f"    metric={plot_metrics!r},\n"
                ")"
            ),
        ))

        section.data[data_key] = df

    except ALL_EXCEPTIONS as e:
        section._log_recoverable_exception(
            e,
            f"validation: bootstrap forest plot for {cohort_name}",
            warn_user=True,
            user_message=f"Could not generate bootstrap forest plot for {cohort_name}: {e}",
        )


def _add_bootstrap_raincloud(
    section: ValidationSection,
    md_lines: list,
    figures: list,
    figure_info: list,
    bootstrap,
    plot_metrics: list,
    cohort_name: str,
    ck: str,
) -> None:
    """Add multi-panel raincloud plot from bootstrap iterations."""
    from mlsurv.viz.metrics import plot_metric as _plot_metric

    try:
        iterations = bootstrap.iterations
        if iterations is None or iterations.empty:
            return

        df = iterations[iterations['metric_name'].isin(plot_metrics)].copy()
        if df.empty:
            return

        point_estimates = _build_point_estimates(bootstrap.summary, plot_metrics)

        fig = _plot_metric(
            df, model_col="model_type", metric=plot_metrics,
            style="raincloud",
            title=f"Bootstrap Distributions: {cohort_name}",
            point_estimates=point_estimates or None,
        )

        fig_id = f'validation_raincloud_{ck}'
        data_key = f'raincloud_{ck}'
        figures.append((f'{fig_id}.png', fig))
        md_lines.append(f"![Bootstrap Distributions](images/{fig_id}.png)")
        md_lines.append("")

        metric_list = ', '.join(
            get_metric_display_name(m) for m in plot_metrics
        )
        figure_info.append(FigureInfo(
            figure_id=fig_id,
            title=f'Bootstrap Distributions: {cohort_name}',
            data_description=f'Bootstrap iteration data for {metric_list} ({cohort_name})',
            data_key=data_key,
            learner_code=(
                "fig = learner.plot_metric(\n"
                "    style='raincloud', source='bootstrap',\n"
                f"    metric={plot_metrics!r},\n"
                ")"
            ),
        ))

        section.data[data_key] = df

    except ALL_EXCEPTIONS as e:
        section._log_recoverable_exception(
            e,
            f"validation: bootstrap raincloud for {cohort_name}",
            warn_user=True,
            user_message=f"Could not generate bootstrap raincloud for {cohort_name}: {e}",
        )
