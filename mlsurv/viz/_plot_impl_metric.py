"""Per-family plot dispatch for single-time-point metric (split from ``_plot_impl.py``)."""
from __future__ import annotations

from typing import Optional

from mlsurv._shared import tidy_model_col as _tidy_model_col
from mlsurv.constants import get_metric_display_name
from mlsurv.viz._context import PlotContext, SubpopulationAxis
from mlsurv.viz._data_metrics import (
    _metric_title_parts,
    _resolve_metric_title,
    build_multi_metric_plot,
    dispatch_single_metric_data,
    find_bootstrap,
    resolve_metric_source,
    resolve_metric_uncertainty,
)
from mlsurv.viz._metrics_helpers import normalize_metric_param
from mlsurv.viz._resolve import (
    _swap_benchmark_eval,
    compose_models_with_refs,
    resolve_eval_result,
    resolve_references_param,
)
from mlsurv.viz._plot_impl_shared import (
    _available_model_names,
    _benchmark_reference_names,
    _dispatch_by_source,
    _iter_subpopulation_levels,
    _require_group_axis,
    _resolve_models_arg,
)


def _plot_metric_subpopulation(
    ctx: PlotContext,
    *,
    group_axis: Optional[SubpopulationAxis],
    models=None,
    style: str = "auto",
    metric=None,
    source: str = "auto",
    uncertainty="auto",
    orientation: str = "horizontal",
    level: Optional[str] = None,
    return_data: bool = False,
    references=None,
    **kwargs,
):
    """Subpopulation branch of :func:`_plot_metric_impl`.

    Only ``style='forest'`` (and ``'auto'`` → forest) is supported in v1.
    Delegates to :func:`mlsurv.viz._metrics_helpers.plot_grouped_forest`.
    """
    axis = _require_group_axis(ctx, group_axis, "_plot_metric_impl")

    if references is not None:
        raise NotImplementedError(
            "references= is not supported on SubpopulationView.plot_metric in v1"
        )
    single_metric, metric_list = normalize_metric_param(metric)
    if metric_list is not None:
        raise NotImplementedError(
            "multi-metric mode (metric=[...] with 2+ names) is not supported "
            "on SubpopulationView.plot_metric in v1. Call plot_metric() "
            "once per metric."
        )
    if source not in ("auto", "bootstrap", "test"):
        raise NotImplementedError(
            f"source={source!r} is not supported on "
            f"SubpopulationView.plot_metric; only 'auto' / 'bootstrap' "
            f"/ 'test' (pre-aggregated in SubpopulationResult) are valid."
        )
    if uncertainty != "auto":
        raise NotImplementedError(
            "Custom uncertainty= is not supported on "
            "SubpopulationView.plot_metric in v1; CI levels are fixed "
            "by the stored SubpopulationResult.bootstrap_summary."
        )
    if orientation != "horizontal":
        raise NotImplementedError(
            "orientation='vertical' is not supported on "
            "SubpopulationView.plot_metric in v1."
        )

    resolved_style = style if style != "auto" else "forest"
    if resolved_style != "forest":
        raise NotImplementedError(
            f"style={style!r} is not supported on "
            f"SubpopulationView.plot_metric in v1. Only "
            f"style='forest' (the default) is implemented."
        )

    model_names = _resolve_models_arg(ctx, models)
    if model_names is not None:
        metrics_long = axis.metrics_long
        boot_summary = axis.bootstrap_summary
        filtered_metrics = (
            metrics_long[metrics_long["model_type"].isin(model_names)]
            if metrics_long is not None
            else metrics_long
        )
        # Match _plot_metric_time_subpopulation: surface an explicit
        # ValueError when the post-filter frame is empty. Without this
        # guard, plot_grouped_forest silently returns None (its
        # documented contract for empty input) and direct-view callers
        # lose the actionable error they'd get from the sibling path.
        if filtered_metrics is None or filtered_metrics.empty:
            raise ValueError(
                f"No metrics rows for models {model_names} on "
                f"subpopulation {axis.group_variable!r}. Available: "
                f"{sorted(_available_model_names(ctx))}"
            )
        axis = SubpopulationAxis(
            group_variable=axis.group_variable,
            levels=axis.levels,
            metrics_long=filtered_metrics,
            bootstrap_summary=(
                boot_summary[boot_summary["model_type"].isin(model_names)]
                if boot_summary is not None
                else None
            ),
            auc_t=axis.auc_t,
            auc_t_bootstrap=axis.auc_t_bootstrap,
        )

    if level is not None:
        # Validate via the shared helper so callers get the same
        # "Unknown subpopulation level …" ValueError as plot_calibration
        # and plot_roc.
        _iter_subpopulation_levels(ctx, axis.group_variable, level=level)
        metrics_long = axis.metrics_long
        boot_summary = axis.bootstrap_summary
        filt_metrics = (
            metrics_long[metrics_long["subpopulation"] == level]
            if metrics_long is not None
            else metrics_long
        )
        if filt_metrics is None or filt_metrics.empty:
            raise ValueError(
                f"No metrics rows for subpopulation level {level!r} on "
                f"{axis.group_variable!r}."
            )
        axis = SubpopulationAxis(
            group_variable=axis.group_variable,
            levels=[level],
            metrics_long=filt_metrics,
            bootstrap_summary=(
                boot_summary[boot_summary["subpopulation"] == level]
                if boot_summary is not None
                else None
            ),
            auc_t=axis.auc_t,
            auc_t_bootstrap=axis.auc_t_bootstrap,
        )

    if return_data:
        # Filter to the metric the figure would plot so return_data matches
        # the rendered forest. plot_grouped_forest resolves
        # and filters to a single metric at render time; without this the frame
        # leaked every computed metric.
        from mlsurv.viz._metrics_helpers import (
            _resolve_grouped_forest_metric,
        )

        metrics_long = axis.metrics_long
        if metrics_long is not None and not metrics_long.empty:
            plot_metric_name = _resolve_grouped_forest_metric(
                metrics_long, single_metric or ctx.scoring,
            )
            metrics_long = metrics_long[
                metrics_long["metric"] == plot_metric_name
            ]
        return _tidy_model_col(metrics_long)

    if kwargs:
        raise NotImplementedError(
            f"Unsupported kwargs for SubpopulationView.plot_metric: "
            f"{sorted(kwargs)}. Only the baseline grouped forest layout "
            f"is supported in v1."
        )

    # Resolve CI percentage from the stored bootstrap summary if we can
    # find one. The stored frame does not carry a ci_level column, so
    # fall back to 95 — matching the report section default.
    ci_pct = "95"

    from mlsurv.viz._metrics_helpers import plot_grouped_forest

    return plot_grouped_forest(
        axis,
        scoring=single_metric or ctx.scoring,
        ref_names=_benchmark_reference_names(ctx),
        ci_pct=ci_pct,
    )


def _plot_metric_impl(
    ctx: PlotContext,
    *,
    models=None,
    style: str = "auto",
    metric=None,
    source: str = "auto",
    uncertainty="auto",
    orientation: str = "horizontal",
    return_data: bool = False,
    references=None,
    group_axis: Optional[SubpopulationAxis] = None,
    level: Optional[str] = None,
    **kwargs,
):
    """Shared implementation for ``plot_metric``.

    On main contexts, delegates to the existing metric helpers
    (``dispatch_single_metric_data``, ``build_multi_metric_plot``)
    reading from ``ctx.state_ref``.  On validation contexts, routes
    through the same ``dispatch_single_metric_data`` callback hook
    with ``state=None`` (the CV branch is rejected -- external
    validation cohorts have no CV folds).  On subpopulation contexts,
    dispatches to :func:`_plot_metric_subpopulation`, which routes
    through :func:`mlsurv.viz._metrics_helpers.plot_grouped_forest`
    using the pre-aggregated ``SubpopulationAxis`` frames.

    Parameters
    ----------
    ctx : PlotContext
        Main, validation, or subpopulation context.
    models : None, str, or list[str]
        Filter to specific model names. ``None`` uses all models.
    style : str
        ``"auto"``, ``"forest"``, ``"bar"``, ``"raincloud"``, ``"boxplot"``.
        On subpopulation contexts only ``"forest"`` / ``"auto"`` are valid.
    metric : str or list of str, optional
        Metric name(s). A scalar (or 1-element list) defaults to
        ``ctx.scoring`` (or ``"harrell"``) and draws one panel. A list of 2+
        names draws a multi-panel forest — not supported on validation /
        subpopulation cohorts in v1 (``build_multi_metric_plot`` reads learner
        state directly).
    source : str
        ``"auto"`` (default), ``"bootstrap"``, or ``"test"``.  ``"cv"`` is
        rejected — validation cohorts have no CV folds.
    uncertainty : str or float
        ``"auto"`` (default) / ``"sd"`` / CI level as float.
    orientation : str
        ``"horizontal"`` (default) or ``"vertical"``.
    return_data : bool, default=False
        Return the underlying summary/iteration DataFrame.
    references : anything, optional
        Not supported on validation / subpopulation cohorts in v1.
    group_axis : SubpopulationAxis, optional
        When set (and ``ctx.source == "subpopulation"``), routes to the
        grouped forest renderer instead of the validation path. Passed
        in by :class:`~mlsurv.learner._views.SubpopulationView`.
    **kwargs
        Forwarded to :func:`mlsurv.viz.metrics.plot_metric`.

    Returns
    -------
    plotly.graph_objects.Figure or pandas.DataFrame
    """
    return _dispatch_by_source(
        "_plot_metric_impl",
        ctx,
        main=_plot_metric_impl_main,
        validation=_plot_metric_impl_validation,
        subpopulation=_plot_metric_subpopulation,
        models=models,
        style=style,
        metric=metric,
        source=source,
        uncertainty=uncertainty,
        orientation=orientation,
        return_data=return_data,
        references=references,
        group_axis=group_axis,
        level=level,
        **kwargs,
    )


def _reject_level_for_non_subpop(level) -> None:
    if level is not None:
        raise NotImplementedError(
            "level= is only supported on subpopulation plot contexts "
            "(SubpopulationView.plot_metric)."
        )


def _plot_metric_impl_main(
    ctx: PlotContext,
    *,
    models=None,
    style: str = "auto",
    metric=None,
    source: str = "auto",
    uncertainty="auto",
    orientation: str = "horizontal",
    return_data: bool = False,
    references=None,
    level: Optional[str] = None,
    group_axis: Optional[SubpopulationAxis] = None,
    **kwargs,
):
    """Main-source branch of :func:`_plot_metric_impl`.

    Delegates to ``dispatch_single_metric_data`` /
    ``build_multi_metric_plot`` reading from ``ctx.state_ref``. Honors
    ``references=`` by recursing through ``_swap_benchmark_eval`` to
    splice benchmark reference rows into the evaluation result.

    The deployed (calibrated) metrics already live in the normal bootstrap /
    eval result because evaluate() and bootstrap() apply the attached calibrator
    inline, so this surface needs no calibrate/raw switch.
    """
    del group_axis  # subpopulation-only; ignored on this branch
    _reject_level_for_non_subpop(level)

    from mlsurv.viz.metrics import plot_metric as _plot_metric_viz

    state = ctx.state_ref

    def _find_bootstrap():
        return find_bootstrap(state)

    def _resolve_eval():
        return resolve_eval_result(state)

    single_metric, metric_list = normalize_metric_param(metric)
    if metric_list is None:
        single_metric = single_metric or ctx.scoring or "harrell"

    # Resolve model subset (None = all models)
    model_names = _resolve_models_arg(ctx, models)

    # -- Reference inclusion: swap in benchmark evaluation data --
    ref_names = resolve_references_param(
        state.benchmark_result, references,
    )
    if ref_names is not None:
        with _swap_benchmark_eval(
            state, state.benchmark_result, ref_names, kwargs,
        ):
            fresh_ctx = PlotContext.from_state(state)
            return _plot_metric_impl(
                fresh_ctx,
                models=compose_models_with_refs(models, ref_names),
                style=style, metric=metric,
                source=source, uncertainty=uncertainty,
                orientation=orientation,
                return_data=return_data, references=None,
                **kwargs,
            )

    # -- Multi-metric dispatch --
    if metric_list is not None:
        return build_multi_metric_plot(
            state, _find_bootstrap,
            metrics=metric_list, models=model_names,
            style=style, source=source,
            uncertainty=uncertainty,
            return_data=return_data,
            orientation=orientation, **kwargs,
        )

    source = resolve_metric_source(
        source, state, _find_bootstrap,
    )
    use_sd, ci_level = resolve_metric_uncertainty(uncertainty, source)

    df, extra_kwargs, resolved_style = dispatch_single_metric_data(
        state,
        _find_bootstrap,
        _resolve_eval,
        source, single_metric, model_names, use_sd, ci_level, style,
    )

    if return_data:
        return _tidy_model_col(df)

    u_label = "SD" if use_sd else f"{ci_level * 100:g}% CI"
    # A caller-supplied title replaces only the header line; the band/source
    # subtitle is still appended so the plot keeps describing its error bars.
    header, subtitle = _metric_title_parts(
        state, _find_bootstrap,
        single_metric, source, use_sd, ci_level,
        resolved_style if style == "auto" else style,
    )
    title = _resolve_metric_title(kwargs.pop("title", None), header, subtitle)

    return _plot_metric_viz(
        df, style=resolved_style,
        metric=single_metric, title=title, ci_level=ci_level,
        orientation=orientation, uncertainty_label=u_label,
        **extra_kwargs, **kwargs,
    )


def _plot_metric_impl_validation(
    ctx: PlotContext,
    *,
    models=None,
    style: str = "auto",
    metric=None,
    source: str = "auto",
    uncertainty="auto",
    orientation: str = "horizontal",
    return_data: bool = False,
    references=None,
    level: Optional[str] = None,
    group_axis: Optional[SubpopulationAxis] = None,
    **kwargs,
):
    """Validation-source branch of :func:`_plot_metric_impl`.

    Routes through ``dispatch_single_metric_data`` with ``state=None``
    so the CV branch is unreachable (external cohorts have no folds).
    Auto-resolves ``source`` to ``'bootstrap'`` when the cohort has a
    bootstrap result, else ``'test'``. Rejects ``references=``,
    multi-metric mode (``metric=[...]`` with 2+ names), and
    ``source='cv'``.

    The cohort's deployed (calibrated) metrics already flow through
    validate()/bootstrap(), so this surface needs no calibrate/raw switch.
    """
    del group_axis  # subpopulation-only; ignored on this branch
    _reject_level_for_non_subpop(level)

    if references is not None:
        raise NotImplementedError(
            "references= is not supported on ValidationView.plot_metric in v1"
        )
    single_metric, metric_list = normalize_metric_param(metric)
    if metric_list is not None:
        raise NotImplementedError(
            "multi-metric mode (metric=[...] with 2+ names) is not supported "
            "on ValidationView.plot_metric in v1. Call plot_metric() once "
            "per metric."
        )

    eval_result = ctx.eval_result
    if eval_result is None or eval_result.concordance is None:
        raise ValueError(
            f"Validation cohort {ctx.name!r} has no evaluation data."
        )

    # Auto-resolve source: prefer bootstrap when the cohort has one.
    # CV is intentionally rejected — external cohorts have no folds.
    if source == "auto":
        source = "bootstrap" if ctx.bootstrap_result is not None else "test"
    if source == "cv":
        raise NotImplementedError(
            "source='cv' is not applicable to external validation cohorts "
            "— there are no CV folds on a held-out dataset. Use "
            "source='bootstrap' or source='test'."
        )
    if source == "train":
        raise NotImplementedError(
            "source='train' is not applicable to external validation cohorts "
            "— apparent train-set metrics belong to the model's own training "
            "cohort. Use source='bootstrap' or source='test'."
        )

    single_metric = single_metric or ctx.scoring or "harrell"
    model_names = _resolve_models_arg(ctx, models)
    use_sd, ci_level = resolve_metric_uncertainty(uncertainty, source)

    # NOTE: state=None is safe here because dispatch_single_metric_data
    # only consumes `state` inside the "cv" branch, which is rejected
    # above. DO NOT "fix" this to pass ctx.state_ref — doing so would
    # silently re-enable the CV path and produce main-cohort CV data
    # mislabelled as the validation cohort.
    df, extra_kwargs, default_style = dispatch_single_metric_data(
        state=None,
        find_bootstrap_fn=lambda: ctx.bootstrap_result,
        resolve_eval_fn=lambda: eval_result,
        source=source,
        metric=single_metric,
        model_names=model_names,
        use_sd=use_sd,
        ci_level=ci_level,
        style=style,
    )

    if return_data:
        return _tidy_model_col(df)

    from mlsurv.viz.metrics import plot_metric as _plot_metric_viz

    resolved_style = style if style != "auto" else default_style
    u_label = "SD" if use_sd else f"{ci_level * 100:g}% CI"
    title = kwargs.pop("title", None)
    if title is None:
        title = f"{get_metric_display_name(single_metric)} on {ctx.name}"

    return _plot_metric_viz(
        df,
        style=resolved_style,
        metric=single_metric,
        title=title,
        ci_level=ci_level,
        orientation=orientation,
        uncertainty_label=u_label,
        **extra_kwargs,
        **kwargs,
    )


# ----------------------------------------------------------------------
# plot_survival
# ----------------------------------------------------------------------


