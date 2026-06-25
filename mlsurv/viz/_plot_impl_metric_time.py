"""Per-family plot dispatch for metric-over-time (split from ``_plot_impl.py``)."""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from mlsurv._shared import tidy_model_col as _tidy_model_col
from mlsurv._warnings import (
    SuggestionWarning,
    warn_missing_time_resolved_ci,
)
from mlsurv.viz import _data as _vi_main
from mlsurv.viz._context import PlotContext, SubpopulationAxis
from mlsurv.viz._metrics_helpers import normalize_metric_param
from mlsurv.viz._resolve import (
    build_reference_palette,
    filter_eval_result,
    resolve_eval_result,
    resolve_references_param,
)
from mlsurv.viz._plot_impl_shared import (
    _benchmark_reference_names,
    _calibrators_for,
    _dispatch_by_source,
    _inject_time_unit,
    _iter_subpopulation_levels,
    _require_group_axis,
    _resolve_models_arg,
    _subpop_auc_long_from_axis,
)


def _resolve_time_metric_routing(metric):
    """Map the unified ``metric: str | list`` onto plot_metric_time's paths.

    Returns the list of metrics for the multi-panel builder, or ``None`` to use
    the dedicated single-panel AUC path. ``None`` / ``"auc"`` / ``["auc"]`` use
    the AUC path (preserving the historical default output); any other single
    metric (e.g. ``"brier"``) routes through the multi-panel builder as a
    1-facet plot, and a list of 2+ names is a true multi-panel facet.
    """
    single_metric, metric_list = normalize_metric_param(metric)
    if metric_list is not None:
        return metric_list
    if (single_metric or "auc") == "auc":
        return None
    return [single_metric]


def _inject_report_horizons(
    state, point_df, std_df, *, show_ci=True,
    time_col="eval_time", model_col="model_type", group_cols=(),
    stacklevel=6,
):
    """Outer-align metric-time frames onto the registered report-horizon grid.

    Implements the NA-honest half of ``add_eval_times``: the point frame (and
    the CI std frame, when present) are reindexed onto
    ``union(computed times, report horizons)`` so horizons the user registered
    but has not recomputed — or that fall outside the censoring-stable window —
    appear as honest NA rows rather than silently vanishing. Emits a one-time
    ``SuggestionWarning`` naming the horizons that still lack a value (or, when
    ``show_ci``, a CI band) with a re-run hint.

    Returns ``(point_df, std_df)`` unchanged when no report horizons are
    registered, so non-NA plots are byte-identical to before.
    """
    extras = getattr(state, "user_extra_eval_times", None) if state is not None else None
    if extras is None or point_df is None or len(point_df) == 0:
        return point_df, std_df
    extras = np.asarray(extras, dtype=float)
    if extras.size == 0:
        return point_df, std_df

    computed = np.unique(point_df[time_col].to_numpy(dtype=float))
    render_grid = np.union1d(computed, extras)
    aligned_point = _vi_main.align_metric_time_to_grid(
        point_df, render_grid,
        model_col=model_col, time_col=time_col, group_cols=group_cols,
    )
    aligned_std = std_df
    std_times = np.array([], dtype=float)
    if std_df is not None and len(std_df) > 0:
        std_times = np.unique(std_df[time_col].to_numpy(dtype=float))
        aligned_std = _vi_main.align_metric_time_to_grid(
            std_df, render_grid,
            model_col=model_col, time_col=time_col, group_cols=group_cols,
        )

    if not getattr(state, "extra_eval_times_render_warned", True):
        missing_value = extras[~np.isin(extras, computed)]
        missing_ci = np.array([], dtype=float)
        if show_ci and std_df is not None and len(std_df) > 0:
            on_curve = extras[np.isin(extras, computed)]
            missing_ci = on_curve[~np.isin(on_curve, std_times)]
        missing = np.unique(np.concatenate([missing_value, missing_ci]))
        if missing.size:
            listed = ", ".join(f"{t:g}" for t in missing)
            warnings.warn(
                f"plot_metric_time: registered report horizon(s) [{listed}] "
                f"have no computed value/CI yet and render as NA. Re-run "
                f"evaluate() (and bootstrap() for CI bands) to fill them.",
                SuggestionWarning, stacklevel=stacklevel,
            )
            state.extra_eval_times_render_warned = True
    return aligned_point, aligned_std


def _metric_ribbon_present(combined_std, metric_name):
    """True when the multi-panel ``combined_std`` carries ribbon rows for
    ``metric_name`` (``"auc"`` or ``"brier"``).

    The multi-panel std frame tags each row with ``metric_name``. AUC(t) is
    collected on every bootstrap (cheap, risk scores only); Brier(t) is opt-in
    via ``bootstrap(include_time_resolved=True)``. Either ribbon is data-driven:
    present iff its time-resolved bootstrap distribution was collected.
    """
    return (
        combined_std is not None
        and "metric_name" in combined_std.columns
        and (combined_std["metric_name"] == metric_name).any()
    )


# Requested-metric -> warning label for the missing-ribbon notice. Order
# fixes a stable warning sequence when more than one ribbon is absent.
_TIME_METRIC_CI_LABELS = (("auc", "AUC"), ("brier", "Brier"))


def _warn_missing_metric_ribbons(multi_metrics, combined_std, *, stacklevel):
    """Warn once per requested metric whose CI ribbon has no bootstrap data.

    Fired only when the caller set ``show_ci=True``. AUC(t) normally has data
    (always collected), so in practice this names Brier(t) when
    ``include_time_resolved`` was not set.

    ``stacklevel`` is the depth the caller would pass to
    ``warn_missing_time_resolved_ci`` directly; this helper adds ``+1`` to
    account for its own frame so the warning still attributes to user code.
    """
    for metric, label in _TIME_METRIC_CI_LABELS:
        if metric in multi_metrics and not _metric_ribbon_present(combined_std, metric):
            warn_missing_time_resolved_ci(label, stacklevel=stacklevel + 1)


def _plot_metric_time_subpopulation_multi(
    ctx: PlotContext,
    *,
    axis: SubpopulationAxis,
    models,
    metrics,
    return_data: bool,
    show_ci="auto",
    raw: bool = False,
    **kwargs,
):
    """Multi-metric path for :class:`SubpopulationView.plot_metric_time`.

    Assembles a long-format DataFrame with ``model_type, subpopulation,
    eval_time, value, metric_name`` across the requested metrics. AUC is
    read from ``axis.auc_t`` (preserves the bootstrap-derived ribbons the
    single-metric path already renders); Brier is computed per subgroup
    on-the-fly via :func:`compute_brier_curves`.

    A single metric renders through :func:`plot_grouped_single_metric_time`
    so its facet grid and dimensions match the AUC layout; two or more
    metrics use the wider :func:`plot_grouped_multi_metric_time` grid.
    """
    from mlsurv.viz._metrics_time import (
        plot_grouped_multi_metric_time,
        plot_grouped_single_metric_time,
    )

    if kwargs:
        raise NotImplementedError(
            f"Unsupported kwargs for SubpopulationView.plot_metric_time "
            f"multi-metric: {sorted(kwargs)}."
        )

    _vi_main._validate_multi_metric_time_args(metrics)

    state = ctx.state_ref
    if state is None:
        raise ValueError(
            "Subpopulation multi-metric plot_metric_time requires "
            "ctx.state_ref to access y_train / subpopulations_test."
        )
    if axis.auc_t is None or axis.auc_t.empty:
        raise ValueError(
            f"Subpopulation {axis.group_variable!r} has no AUC(t) data. "
            f"Re-run evaluate() / benchmark() with eval_times= set."
        )

    model_names = _resolve_models_arg(ctx, models)
    fitted_models = ctx.fitted_models or {}
    if model_names is not None:
        fitted_models = {
            k: v for k, v in fitted_models.items() if k in model_names
        }

    eval_times = np.sort(axis.auc_t["eval_time"].unique())

    chunks = []
    std_chunks = []

    auc_ribbon_present = False
    if "auc" in metrics:
        auc_chunk, auc_std_chunk = _subpop_auc_long_from_axis(axis, model_names)
        chunks.append(auc_chunk)
        if auc_std_chunk is not None:
            std_chunks.append(auc_std_chunk)
            auc_ribbon_present = True

    if "brier" in metrics:
        calibrators = _calibrators_for(ctx, fitted_models.keys(), raw)
        levels = _iter_subpopulation_levels(ctx, axis.group_variable)
        dropped_levels = []
        for level_name, mask in levels:
            if hasattr(ctx.X_data, "loc") and not isinstance(mask, slice):
                X_sub = ctx.X_data.loc[mask]
            else:
                X_sub = ctx.X_data[mask]
            y_sub = ctx.y_data[mask]
            brier_chunk = _vi_main.gather_brier_time_data_generic(
                fitted_models=fitted_models,
                X_train=state.X_train, y_train=state.y_train,
                X_test=X_sub, y_test=y_sub,
                eval_times=eval_times,
                calibrators=calibrators,
            )
            if brier_chunk is None or brier_chunk.empty:
                dropped_levels.append(level_name)
                continue
            brier_chunk["subpopulation"] = level_name
            chunks.append(brier_chunk)
        if dropped_levels:
            # stacklevel=6: here → _plot_metric_time_subpopulation →
            # _dispatch_by_source → _plot_metric_time_impl →
            # SubpopulationView.plot_metric_time → user.
            warnings.warn(
                f"plot_metric_time: Brier(t) could not be computed for "
                f"subpopulation {axis.group_variable!r} level(s) "
                f"{dropped_levels}; those panels will be blank. "
                f"See earlier warnings for the cause.",
                SuggestionWarning, stacklevel=6,
            )

    if not chunks:
        raise ValueError(
            f"No valid metric data could be gathered for subpopulation "
            f"{axis.group_variable!r}."
        )

    combined = pd.concat(chunks, ignore_index=True)
    combined_std = (
        pd.concat(std_chunks, ignore_index=True) if std_chunks else None
    )

    # Surface registered report horizons (add_eval_times) as NA-honest rows in
    # the faceted view, mirroring the cohort multi-metric path. group_cols adds
    # the subpopulation facet to the metric_name dimension so each
    # (model, subpopulation, metric) series is aligned independently.
    combined, combined_std = _inject_report_horizons(
        state, combined, combined_std,
        group_cols=("subpopulation", "metric_name"),
    )

    if return_data:
        return _tidy_model_col(combined)

    if show_ci is True and "auc" in metrics and not auc_ribbon_present:
        # stacklevel=7: warn_missing_time_resolved_ci →
        # _plot_metric_time_subpopulation_multi →
        # _plot_metric_time_subpopulation → _dispatch_by_source →
        # _plot_metric_time_impl → SubpopulationView.plot_metric_time → user.
        warn_missing_time_resolved_ci("AUC", stacklevel=7)

    if len(metrics) == 1:
        # A single (non-AUC) metric uses the AUC facet grid so its layout and
        # dimensions match metric="auc"; the multi-metric grid is wider and
        # un-wrapped, which would differ for the same call shape. A single
        # non-AUC metric (e.g. brier) carries no subpop CI ribbon, so show_ci
        # has no effect here and is not threaded.
        fig = plot_grouped_single_metric_time(
            combined, combined_std, metrics[0],
            group_variable=axis.group_variable,
            subpopulations=axis.levels or None,
            time_unit=ctx.time_unit,
            ref_names=_benchmark_reference_names(ctx),
        )
    else:
        fig = plot_grouped_multi_metric_time(
            combined, combined_std, metrics,
            group_variable=axis.group_variable,
            subpopulations=axis.levels or None,
            time_unit=ctx.time_unit,
            ref_names=_benchmark_reference_names(ctx),
            show_ci=show_ci,
        )
    if fig is None:
        # The grouped renderers return None on empty inputs; surface a clear
        # error rather than handing the user None.
        raise ValueError(
            f"plot_metric_time produced no traces for subpopulation "
            f"{axis.group_variable!r} with metrics={metrics}."
        )
    return fig


def _plot_metric_time_subpopulation(
    ctx: PlotContext,
    *,
    group_axis: Optional[SubpopulationAxis],
    models=None,
    metric=None,
    level: Optional[str] = None,
    return_data: bool = False,
    references=None,
    raw: bool = False,
    **kwargs,
):
    """Subpopulation branch of :func:`_plot_metric_time_impl`.

    Delegates to :func:`mlsurv.viz._metrics_time.plot_grouped_metric_time`
    using the pre-aggregated ``axis.auc_t`` / ``axis.auc_t_bootstrap``
    frames from the stored :class:`~mlsurv.results.SubpopulationResult`.

    Per-level Brier(t) (multi-metric mode) is calibrated by default for models
    with an attached calibrator (``train(calibrate=...)``); ``raw=True`` scores
    it raw. AUC(t) is rank-based and unaffected.
    """
    axis = _require_group_axis(ctx, group_axis, "_plot_metric_time_impl")

    # Pop show_ci so it does not trip the "unsupported kwargs" guards below;
    # the subpopulation AUC(t) ribbon comes from axis.auc_t_bootstrap.
    show_ci = kwargs.pop("show_ci", "auto")

    if references is not None:
        raise NotImplementedError(
            "references= is not supported on SubpopulationView.plot_metric_time in v1"
        )
    multi_metrics = _resolve_time_metric_routing(metric)
    if multi_metrics is not None and level is not None:
        # The multi-metric facet path computes Brier per-level on the
        # fly via its own iteration; threading level= through cleanly
        # is out of scope for v1.
        raise NotImplementedError(
            "level= is not supported with multi-metric mode (metric=[...] or "
            "a non-AUC metric) in v1. Call plot_metric_time(metric='auc', "
            "level=...) instead."
        )
    if multi_metrics is not None:
        return _plot_metric_time_subpopulation_multi(
            ctx,
            axis=axis,
            models=models,
            metrics=multi_metrics,
            return_data=return_data,
            show_ci=show_ci,
            raw=raw,
            **kwargs,
        )
    if kwargs:
        raise NotImplementedError(
            f"Unsupported kwargs for SubpopulationView.plot_metric_time: "
            f"{sorted(kwargs)}. Only the baseline faceted AUC(t) layout "
            f"is supported in v1."
        )

    auc_t = axis.auc_t
    if auc_t is None or auc_t.empty:
        raise ValueError(
            f"Subpopulation {axis.group_variable!r} has no AUC(t) data. "
            f"Re-run evaluate() / benchmark() with eval_times= set."
        )

    model_names = _resolve_models_arg(ctx, models)
    if model_names is not None:
        # Build a filtered axis so both the metrics frame and the
        # ribbon frame are consistently scoped to the requested models.
        metrics_long = axis.metrics_long
        boot_summary = axis.bootstrap_summary
        auc_t_boot = axis.auc_t_bootstrap

        auc_t_filt = auc_t[auc_t["model_type"].isin(model_names)]
        if auc_t_filt.empty:
            raise ValueError(
                f"No AUC(t) rows for models {model_names} on "
                f"subpopulation {axis.group_variable!r}"
            )

        axis = SubpopulationAxis(
            group_variable=axis.group_variable,
            levels=axis.levels,
            metrics_long=(
                metrics_long[metrics_long["model_type"].isin(model_names)]
                if metrics_long is not None
                else metrics_long
            ),
            bootstrap_summary=(
                boot_summary[boot_summary["model_type"].isin(model_names)]
                if boot_summary is not None
                else None
            ),
            auc_t=auc_t_filt,
            auc_t_bootstrap=(
                auc_t_boot[auc_t_boot["model_type"].isin(model_names)]
                if auc_t_boot is not None
                else None
            ),
        )

    if level is not None:
        # Validate via the shared helper for uniform error messages
        # across all SubpopulationView.plot_* methods.
        _iter_subpopulation_levels(ctx, axis.group_variable, level=level)
        auc_t_filt = axis.auc_t[axis.auc_t["subpopulation"] == level]
        if auc_t_filt.empty:
            raise ValueError(
                f"No AUC(t) rows for level {level!r} on subpopulation "
                f"{axis.group_variable!r}."
            )
        axis = SubpopulationAxis(
            group_variable=axis.group_variable,
            levels=[level],
            metrics_long=(
                axis.metrics_long[axis.metrics_long["subpopulation"] == level]
                if axis.metrics_long is not None
                else axis.metrics_long
            ),
            bootstrap_summary=(
                axis.bootstrap_summary[
                    axis.bootstrap_summary["subpopulation"] == level
                ]
                if axis.bootstrap_summary is not None
                else None
            ),
            auc_t=auc_t_filt,
            auc_t_bootstrap=(
                axis.auc_t_bootstrap[
                    axis.auc_t_bootstrap["subpopulation"] == level
                ]
                if axis.auc_t_bootstrap is not None
                else None
            ),
        )

    # Outer-align the facets onto the registered report-horizon grid so
    # add_eval_times() horizons appear (as NA until recomputed) in the
    # per-subpopulation view, matching the cohort curve. group_cols carries
    # the subpopulation facet dimension so each (model, subpopulation) series
    # is aligned independently.
    auc_t_aligned, boot_aligned = _inject_report_horizons(
        ctx.state_ref, axis.auc_t, axis.auc_t_bootstrap,
        group_cols=("subpopulation",),
    )
    axis = SubpopulationAxis(
        group_variable=axis.group_variable,
        levels=axis.levels,
        metrics_long=axis.metrics_long,
        bootstrap_summary=axis.bootstrap_summary,
        auc_t=auc_t_aligned,
        auc_t_bootstrap=boot_aligned,
    )

    if return_data:
        return _tidy_model_col(axis.auc_t)

    if show_ci is True:
        boot = axis.auc_t_bootstrap
        if boot is None or boot.empty:
            # stacklevel=6: warn_missing_time_resolved_ci →
            # _plot_metric_time_subpopulation → _dispatch_by_source →
            # _plot_metric_time_impl → SubpopulationView.plot_metric_time →
            # user code.
            warn_missing_time_resolved_ci("AUC", stacklevel=6)

    from mlsurv.viz._metrics_time import plot_grouped_metric_time

    return plot_grouped_metric_time(
        axis,
        time_unit=ctx.time_unit,
        ref_names=_benchmark_reference_names(ctx),
        show_ci=show_ci,
    )


def _plot_metric_time_impl(
    ctx: PlotContext,
    *,
    models=None,
    metric=None,
    return_data: bool = False,
    references=None,
    group_axis: Optional[SubpopulationAxis] = None,
    level: Optional[str] = None,
    raw: bool = False,
    **kwargs,
):
    """Shared implementation for ``plot_metric_time``.

    Parameters
    ----------
    ctx : PlotContext
        Main, validation, or subpopulation context.
    models : None, str, or list[str]
        Filter to specific model names. ``None`` (default) uses all models
        present in the cohort's evaluation result.
    metric : str or list of str, optional
        Metric name(s). ``None`` / ``"auc"`` / ``["auc"]`` draw the
        single-panel AUC(t) curve. A single non-AUC metric (e.g. ``"brier"``)
        renders one panel via the multi-panel builder; a list of 2+ names
        renders a true multi-panel facet. Supported keys: ``"auc"``,
        ``"brier"``. AUC is read from each context's
        ``eval_result.auc_t`` (or ``axis.auc_t`` on subpopulations); Brier is
        computed on-the-fly via :func:`compute_brier_curves` using the main
        learner's training
        set as the IPCW baseline, so validation cohorts and subgroup
        slices produce cohort-specific curves without refitting.
    return_data : bool, default=False
        Return the tidy DataFrame instead of a figure. In multi-panel
        mode the frame is long-format with a ``metric_name`` column.
    references : anything, optional
        Not supported on validation or subpopulation cohorts in v1.
        Passing a non-``None`` value raises ``NotImplementedError``.
    group_axis : SubpopulationAxis, optional
        When set (and ``ctx.source == "subpopulation"``), routes to the
        grouped faceted AUC(t) renderer instead of the single-panel
        validation / main path. Passed in by
        :class:`~mlsurv.learner._views.SubpopulationView`.
    **kwargs
        Forwarded to :func:`mlsurv.viz.metrics.plot_metric_time`.

    Returns
    -------
    plotly.graph_objects.Figure or pandas.DataFrame
    """
    return _dispatch_by_source(
        "_plot_metric_time_impl",
        ctx,
        main=_plot_metric_time_impl_main,
        validation=_plot_metric_time_impl_validation,
        subpopulation=_plot_metric_time_subpopulation,
        models=models,
        metric=metric,
        return_data=return_data,
        references=references,
        group_axis=group_axis,
        level=level,
        raw=raw,
        **kwargs,
    )


def _reject_level_for_non_subpop(level) -> None:
    if level is not None:
        raise NotImplementedError(
            "level= is only supported on subpopulation plot contexts "
            "(SubpopulationView.plot_metric_time)."
        )


def _plot_metric_time_impl_main(
    ctx: PlotContext,
    *,
    models=None,
    metric=None,
    return_data: bool = False,
    references=None,
    level: Optional[str] = None,
    group_axis: Optional[SubpopulationAxis] = None,
    raw: bool = False,
    **kwargs,
):
    """Main-source branch of :func:`_plot_metric_time_impl`.

    Reads ``ctx.state_ref.benchmark_result`` (when ``references=`` is
    set) or ``ctx.eval_result.auc_t`` (otherwise), filters to requested
    models, and renders single-panel AUC(t) or multi-panel
    ``metric=[...]`` curves via
    :func:`~mlsurv.viz.metrics.plot_metric_time`.

    Brier(t) is calibrated by default for models with an attached calibrator
    (``train(calibrate=...)``), matching the deployed model; ``raw=True`` scores
    it raw. AUC(t) is rank-based and unaffected.
    """
    del group_axis
    _reject_level_for_non_subpop(level)

    from mlsurv.viz.metrics import (
        plot_metric_time as _plot_metric_time_viz,
    )

    multi_metrics = _resolve_time_metric_routing(metric)
    state = ctx.state_ref

    # Resolve evaluation source (with optional reference inclusion)
    ref_names = resolve_references_param(
        state.benchmark_result, references,
    )
    if ref_names is not None:
        br = state.benchmark_result
        keep = set(br.model_names) | set(ref_names)
        build_reference_palette(br, ref_names, kwargs)
        eval_result = filter_eval_result(br.evaluation, keep)
    else:
        eval_result = resolve_eval_result(state)

    if eval_result.auc_t is None or eval_result.auc_t.empty:
        # AUC(t) supplies the shared eval-time grid (x-axis) for every
        # time-dependent metric, including Brier(t), so it is required even
        # when metric='brier'.
        raise ValueError(
            "No time-dependent metric data. Run evaluate() with eval_times= "
            "set (this populates the AUC(t) grid that all time metrics share)."
        )

    auc_t = eval_result.auc_t

    # Filter to requested models
    model_names = _resolve_models_arg(ctx, models)
    if model_names is not None:
        if ref_names:
            model_names = [*model_names, *ref_names]
        auc_t = auc_t[auc_t["model_type"].isin(model_names)]
        if auc_t.empty:
            raise ValueError(
                f"No AUC(t) data for models: {model_names}. "
                f"Available: {eval_result.auc_t['model_type'].unique().tolist()}"
            )

    # An explicit show_ci=True (vs the silent "auto" default) means the
    # caller is demanding the ribbon; warn when its bootstrap data is absent.
    show_ci = kwargs.get("show_ci", "auto")

    # Multi-panel: gather data for each requested metric
    if multi_metrics is not None:
        combined_df, combined_std = _vi_main.build_multi_metric_time_data(
            state, multi_metrics, eval_result, model_names, raw=raw,
        )
        combined_df, combined_std = _inject_report_horizons(
            state, combined_df, combined_std, group_cols=("metric_name",),
        )
        if return_data:
            return _tidy_model_col(combined_df)
        if show_ci is True:
            _warn_missing_metric_ribbons(multi_metrics, combined_std, stacklevel=6)
        _inject_time_unit(kwargs, ctx)
        return _plot_metric_time_viz(
            combined_df, std_df=combined_std,
            metric=multi_metrics, model_col="model_type", **kwargs,
        )

    std_df = _vi_main.compute_auc_std_df(eval_result, model_names)
    auc_t, std_df = _inject_report_horizons(state, auc_t, std_df)

    if return_data:
        return _tidy_model_col(auc_t)

    # std_df can be a zero-row frame (bootstrap AUC exists but the model
    # filter removed every matching row), not just None — treat both as
    # "no ribbon" so an explicit show_ci=True still warns.
    if show_ci is True and (std_df is None or std_df.empty):
        warn_missing_time_resolved_ci("AUC", stacklevel=6)
    _inject_time_unit(kwargs, ctx)
    kwargs.setdefault("metric", "auc")
    return _plot_metric_time_viz(
        auc_t, std_df=std_df, model_col="model_type", **kwargs,
    )


def _plot_metric_time_impl_validation(
    ctx: PlotContext,
    *,
    models=None,
    metric=None,
    return_data: bool = False,
    references=None,
    level: Optional[str] = None,
    group_axis: Optional[SubpopulationAxis] = None,
    raw: bool = False,
    **kwargs,
):
    """Validation-source branch of :func:`_plot_metric_time_impl`.

    Reads ``ctx.eval_result.auc_t`` for the cohort and renders single-
    or multi-panel curves. Multi-panel mode (``metric=[...]``) uses
    the validation cohort as the test set and the learner's training
    set as the IPCW baseline (no refit on validation data). Rejects
    ``references=``.

    Brier(t) is calibrated by default on the validation cohort for models with
    an attached calibrator (``train(calibrate=...)``), matching the cohort's
    validated metrics (validate() applies the same calibrator); ``raw=True``
    scores it raw. AUC(t) is rank-based and unaffected.
    """
    del group_axis
    _reject_level_for_non_subpop(level)

    if references is not None:
        raise NotImplementedError(
            "references= is not supported on ValidationView.plot_metric_time in v1"
        )

    multi_metrics = _resolve_time_metric_routing(metric)
    eval_result = ctx.eval_result
    if eval_result is None:
        raise ValueError(
            f"Validation cohort {ctx.name!r} has no evaluation result."
        )
    if eval_result.auc_t is None or eval_result.auc_t.empty:
        raise ValueError(
            f"Cohort {ctx.name!r} has no AUC(t) data. "
            f"Run validate() with eval_times= set."
        )

    from mlsurv.viz.metrics import plot_metric_time as _plot_metric_time_viz

    model_names = _resolve_models_arg(ctx, models)

    # Multi-panel: gather data for each requested metric using the
    # validation cohort as the test set and the learner's training set
    # as the IPCW baseline (matches PlotContext.from_validation's
    # semantics — no refit on validation data).
    if multi_metrics is not None:
        state = ctx.state_ref
        if state is None:
            raise ValueError(
                "Validation multi-metric plot_metric_time requires "
                "ctx.state_ref to access training-set X/y for the "
                "Brier IPCW baseline."
            )
        fitted_models = ctx.fitted_models or {}
        if model_names is not None:
            fitted_models = {
                k: v for k, v in fitted_models.items() if k in model_names
            }
        combined_df, combined_std, dropped = _vi_main.build_multi_metric_time_data_generic(
            fitted_models=fitted_models,
            X_train=state.X_train, y_train=state.y_train,
            X_test=ctx.X_data, y_test=ctx.y_data,
            eval_result=eval_result,
            metrics=multi_metrics,
            model_names=model_names,
            calibrators=_calibrators_for(ctx, fitted_models.keys(), raw),
        )
        if dropped:
            # stacklevel=6: here → _plot_metric_time_impl_validation
            # → _dispatch_by_source → _plot_metric_time_impl
            # → ValidationView.plot_metric_time → user code.
            _vi_main.warn_dropped_metrics(dropped, cohort_label=ctx.name, stacklevel=6)
        combined_df, combined_std = _inject_report_horizons(
            state, combined_df, combined_std, group_cols=("metric_name",),
        )
        if return_data:
            return _tidy_model_col(combined_df)
        if kwargs.get("show_ci", "auto") is True:
            _warn_missing_metric_ribbons(multi_metrics, combined_std, stacklevel=6)
        _inject_time_unit(kwargs, ctx)
        return _plot_metric_time_viz(
            combined_df, std_df=combined_std,
            metric=multi_metrics, model_col="model_type", **kwargs,
        )

    auc_t = eval_result.auc_t
    if model_names is not None:
        auc_t = auc_t[auc_t["model_type"].isin(model_names)]
        if auc_t.empty:
            raise ValueError(
                f"No AUC(t) rows for models {model_names} on cohort {ctx.name!r}"
            )

    # compute_auc_std_df takes an EvaluationResult directly (not state),
    # so it works for validation cohorts unchanged. DO NOT refactor it to
    # take learner state without updating this call site.
    std_df = _vi_main.compute_auc_std_df(eval_result, model_names)
    auc_t, std_df = _inject_report_horizons(ctx.state_ref, auc_t, std_df)

    if return_data:
        return _tidy_model_col(auc_t)

    # See main-path note: a zero-row std_df (model filter emptied it) is also
    # "no ribbon", so an explicit show_ci=True still warns.
    if kwargs.get("show_ci", "auto") is True and (std_df is None or std_df.empty):
        warn_missing_time_resolved_ci("AUC", stacklevel=6)
    _inject_time_unit(kwargs, ctx)

    # Title defaulting is the caller's responsibility (ValidationView
    # or a report section). Each caller owns its own title convention.
    return _plot_metric_time_viz(
        auc_t,
        std_df=std_df,
        metric="auc",
        model_col="model_type",
        **kwargs,
    )


# ======================================================================
# ValidationView ports: plot_metric / plot_survival /
# plot_calibration / plot_roc
# ======================================================================
#
# All five impls support ``ctx.source`` values: ``"main"``,
# ``"validation"``, and ``"subpopulation"``. Each impl branches on
# ``ctx.source`` to route to the correct code path. The main branch
# reads data from ``ctx.state_ref`` (the :class:`LearnerState`),
# matching the learner's direct call conventions. The validation
# branch uses ``ctx.eval_result`` / ``ctx.X_data`` / ``ctx.y_data``
# scoped to the cohort. The subpopulation branch delegates to
# dedicated ``_plot_X_subpopulation`` helpers.


# ----------------------------------------------------------------------
# plot_metric
# ----------------------------------------------------------------------


