"""Per-family plot dispatch for calibration (split from ``_plot_impl.py``)."""
from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from mlsurv._warnings import SuggestionWarning
from mlsurv.viz._context import PlotContext, SubpopulationAxis
from mlsurv.viz._data import collect_surv_function_models
from mlsurv.viz._resolve import calibration_to_tidy
from mlsurv.viz._plot_impl_shared import (
    _available_model_names,
    _calibrator_for,
    _calibrators_for,
    _dispatch_by_source,
    _finalize_subpop_facet_layout,
    _format_subpop_facet_axes,
    _iter_subpopulation_levels,
    _make_subpop_facet_grid,
    _require_group_axis,
    _resolve_models_arg,
    _subpop_grid_dims,
)


def _default_eval_times(state, y_data, y_train):
    """Default calibration display grid via the canonical resolver.

    Delegates to :func:`mlsurv.utils.resolve_display_eval_times`, which applies
    the same resolution the compute surfaces use (explicit / inherited
    ``user_eval_times`` -> auto) and unions any ``add_eval_times`` report
    horizons. So calibration lands on the same horizons as AUC(t)/Brier(t)/ROC
    by construction, even on the auto path. Read-only: never mutates state.

    Returns a non-empty ``list[float]``; raises ``ValueError`` if no grid can be
    derived from the data (propagated from the resolver).
    """
    from mlsurv.utils import resolve_display_eval_times

    return [float(t) for t in resolve_display_eval_times(state, y_data, y_train)]


def _comparison_horizon(user_times, times_list):
    """Pick one horizon for a single-horizon comparison (multi-model / subpop).

    User-supplied times -> the first (the caller warns about any extras).
    Otherwise the central horizon of the materialized default grid, so the
    comparison sits at a representative time rather than the earliest one.
    """
    if user_times is not None:
        return float(user_times[0])
    ordered = sorted(float(t) for t in times_list)
    return ordered[len(ordered) // 2]


def _build_calibration_model_dict(
    ctx: PlotContext,
    models,
) -> dict:
    """Build the ``{name: model}`` dict for calibration, filtering to
    models with ``predict_survival_function``.

    Used by the subpopulation calibration branch. The validation branch
    has its own inline filter that additionally intersects with the
    validated concordance rows.

    Raises ``ValueError`` when no models survive the filter.
    """
    fitted_models = ctx.fitted_models or {}
    requested = _resolve_models_arg(ctx, models)

    model_dict = {}
    dropped_for_survfunc: List[str] = []
    for name, model in fitted_models.items():
        if not hasattr(model, "predict_survival_function"):
            dropped_for_survfunc.append(name)
            continue
        if requested is not None and name not in requested:
            continue
        model_dict[name] = model

    if requested is not None:
        explicit_dropped = [n for n in dropped_for_survfunc if n in requested]
        if explicit_dropped:
            warnings.warn(
                f"Skipping requested model(s) {explicit_dropped}: "
                f"no predict_survival_function() method. Calibration plots "
                f"require survival probability predictions.",
                SuggestionWarning,
                # stacklevel=6: _build_calibration_model_dict
                # -> _plot_calibration_subpopulation
                # -> _dispatch_by_source -> _plot_calibration_impl
                # -> SubpopulationView.plot_calibration -> user code.
                stacklevel=6,
            )

    if not model_dict:
        available = sorted((ctx.fitted_models or {}).keys())
        raise ValueError(
            f"No models support predict_survival_function. "
            f"Available models: {available}."
        )
    return model_dict


def _plot_calibration_subpopulation(
    ctx: PlotContext,
    *,
    group_axis: Optional[SubpopulationAxis],
    models=None,
    eval_times=None,
    n_bins: int = 10,
    level: Optional[str] = None,
    return_data: bool = False,
    raw: bool = False,
    bin_strategy: str = "width",
    **kwargs,
):
    """Subpopulation branch of :func:`_plot_calibration_impl`.

    Computes calibration data per subpopulation level by filtering
    ``ctx.X_data`` / ``ctx.y_data`` and calling
    :func:`~mlsurv.viz.calibration._compute_multi_model_calibration_data`
    on each slice. Produces a faceted ``make_subplots`` grid (one column
    per level, wrapping at 3 columns).

    Subpopulation calibration compares at a single horizon. ``eval_times``
    accepts a scalar or a list; when a list is given only the first horizon
    is used (with a SuggestionWarning), and when omitted the central horizon of
    the canonical eval grid (inherited ``user_eval_times`` -> auto, unioned with
    ``add_eval_times``) is used.

    Calibrated by default: a model with an attached calibrator
    (``train(calibrate=...)``) has its predicted ``S(t|x)`` warped to the
    calibrated scale within each level, matching the deployed model. ``raw=True``
    plots the uncalibrated curves.
    """
    axis = _require_group_axis(ctx, group_axis, "_plot_calibration_impl")

    X_data = ctx.X_data
    y_data = ctx.y_data
    if X_data is None or y_data is None:
        raise ValueError(
            "Subpopulation calibration requires X_test / y_test in the "
            "plot context. ctx.X_data or ctx.y_data is None."
        )

    model_dict = _build_calibration_model_dict(ctx, models)
    calibrators = _calibrators_for(ctx, model_dict.keys(), raw)

    # Subpopulation facets compare at one horizon; collapse the unified
    # eval_times (scalar | list) to a single time. When the user gave nothing,
    # default to the central horizon of the canonical eval grid (which honors
    # an inherited grid and add_eval_times). stacklevel=5: here ->
    # _dispatch_by_source -> _plot_calibration_impl ->
    # SubpopulationView.plot_calibration -> user code.
    user_times = _normalize_eval_times(eval_times)
    times_list = (
        user_times if user_times is not None
        else _default_eval_times(ctx.state_ref, y_data, ctx.y_train)
    )
    eval_time = _comparison_horizon(user_times, times_list)
    if user_times is not None and len(user_times) > 1:
        warnings.warn(
            f"Subpopulation calibration compares at a single horizon; using "
            f"t={eval_time:.1f} (the first of the {len(user_times)} times given).",
            SuggestionWarning, stacklevel=5,
        )

    levels_and_masks = _iter_subpopulation_levels(
        ctx, axis.group_variable, level=level,
    )

    from mlsurv.viz.calibration import (
        _compute_multi_model_calibration_data,
    )

    # Compute per-level calibration data; skip levels that fail.
    per_level_cal: List[Tuple[str, dict]] = []
    for lvl, mask in levels_and_masks:
        X_level = X_data.iloc[mask]
        y_level = y_data[mask]
        try:
            cal_data = _compute_multi_model_calibration_data(
                model_dict, X_level, y_level, eval_time, n_bins,
                calibrators=calibrators,
                bin_strategy=bin_strategy,
            )
        except (ValueError, RuntimeError, TypeError, IndexError) as exc:
            warnings.warn(
                f"Calibration computation failed for subpopulation level "
                f"{lvl!r}: {exc}. Skipping this level.",
                SuggestionWarning,
                # stacklevel=5: here -> _plot_calibration_subpopulation
                # -> _dispatch_by_source -> _plot_calibration_impl
                # -> SubpopulationView.plot_calibration -> user code.
                stacklevel=5,
            )
            continue
        if cal_data:
            per_level_cal.append((lvl, cal_data))
        else:
            warnings.warn(
                f"Calibration for subpopulation level {lvl!r}: all models "
                f"produced empty calibration bins (likely too few samples "
                f"at this level). Skipping this level.",
                SuggestionWarning,
                # stacklevel=5: see comment on the failed-computation
                # warning above.
                stacklevel=5,
            )

    if not per_level_cal:
        raise ValueError(
            f"Calibration could not be computed for any subpopulation "
            f"level of {axis.group_variable!r}."
        )

    # return_data: concat per-level tidy frames with a "subpopulation" col.
    if return_data:
        dfs = []
        for lvl, cal_data in per_level_cal:
            df = calibration_to_tidy(cal_data)
            df.insert(0, "subpopulation", lvl)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    # Single level: delegate to the standard single-panel renderer.
    if len(per_level_cal) == 1:
        from mlsurv.viz.calibration import (
            plot_calibration as _plot_calibration_viz,
        )
        lvl, cal_data = per_level_cal[0]
        title = kwargs.pop(
            "title",
            f"Calibration: {axis.group_variable} = {lvl} "
            f"(t = {eval_time:.1f})",
        )
        return _plot_calibration_viz(cal_data, title=title, **kwargs)

    # Multi-level faceted grid.
    return _render_calibration_facet_grid(
        per_level_cal,
        eval_time=eval_time,
        group_variable=axis.group_variable,
        **kwargs,
    )


def _render_calibration_facet_grid(
    per_level_cal: List[Tuple[str, dict]],
    *,
    eval_time: float,
    group_variable: str,
    **kwargs,
):
    """Render a multi-level faceted calibration grid (one panel per level).

    Independently testable: given ``per_level_cal`` (a list of
    ``(level_name, cal_data)`` where each ``cal_data`` maps model name to a
    dict with ``predicted``/``observed`` arrays), builds a 3-column-wrapping
    ``make_subplots`` grid with a dashed "Perfect" diagonal per panel and
    one marker+line trace per model (shared legend, drawn only on the first
    panel).

    ``title`` (default ``"Subpopulation Calibration: {group_variable}
    (t = ...)"``) and ``width`` / ``height`` are consumed from ``**kwargs``.
    """
    import plotly.graph_objects as go
    from mlsurv.viz._palettes import get_model_color, get_model_display_name
    from mlsurv.viz._theme import MANUSCRIPT_DEFAULTS

    d = MANUSCRIPT_DEFAULTS
    n_levels = len(per_level_cal)
    n_cols, _ = _subpop_grid_dims(n_levels)
    level_names = [lvl for lvl, _ in per_level_cal]
    fig = _make_subpop_facet_grid(level_names)

    for idx, (lvl, cal_data) in enumerate(per_level_cal):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        is_first = idx == 0

        # Diagonal reference
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Perfect",
                showlegend=is_first,
                legendgroup="__perfect__",
            ),
            row=row, col=col,
        )

        for model_name, item in cal_data.items():
            predicted = item["predicted"]
            observed = item["observed"]
            color = get_model_color(model_name)
            display_name = get_model_display_name(model_name)

            fig.add_trace(
                go.Scatter(
                    x=predicted, y=observed,
                    mode="markers+lines",
                    marker=dict(
                        size=d["marker_size_small"], color=color,
                    ),
                    line=dict(
                        color=color, width=d["line_width_secondary"],
                    ),
                    name=display_name,
                    legendgroup=display_name,
                    showlegend=is_first,
                    hovertemplate=(
                        f"<b>{display_name}</b><br>"
                        "Predicted: %{x:.2f}<br>"
                        "Observed: %{y:.2f}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )

    _format_subpop_facet_axes(
        fig, n_levels,
        x_title="Predicted",
        y_title="Observed (KM)",
    )

    title = kwargs.pop(
        "title",
        f"Subpopulation Calibration: {group_variable} (t = {eval_time:.1f})",
    )
    return _finalize_subpop_facet_layout(
        fig,
        title=title,
        legend_anchor="top-right",
        legend_title="Model",
        width=kwargs.pop("width", None),
        height=kwargs.pop("height", None),
    )


def _empty_calibration_figure(eval_time: float, return_data: bool):
    """Return an empty figure (or DataFrame) when no models produce valid
    calibration data, with a user-facing warning."""
    warnings.warn(
        f"No models produced valid calibration data at t={eval_time:.1f}. "
        f"Check preceding warnings for per-model details. Common causes: "
        f"insufficient samples per bin (try reducing n_bins or increasing "
        f"dataset size), models lacking predict_survival_function(), "
        f"or numerical issues during prediction.",
        SuggestionWarning,
        # stacklevel=6: here -> _plot_calibration_impl_{main,validation}
        # -> _dispatch_by_source -> _plot_calibration_impl
        # -> plot_calibration wrapper -> user
        stacklevel=6,
    )
    if return_data:
        return pd.DataFrame(
            columns=["model", "eval_time", "predicted", "observed",
                     "ci_lower", "ci_upper"],
        )
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_annotation(
        text="No calibration data available",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14),
    )
    fig.update_layout(title="Calibration Plot (No Data)")
    return fig


def _normalize_eval_times(eval_times):
    """Normalize the unified ``eval_times`` (scalar | list | None) to a list.

    ``plot_calibration`` takes a single ``eval_times`` argument that accepts
    either one timepoint or a list. Returns ``None`` when nothing was given,
    a non-empty list of floats otherwise. Strings/bytes are treated as scalars
    (they would be a user error, surfaced later as a float() failure).

    Raises ``ValueError`` on an empty list — otherwise the downstream "pick
    the first horizon" step would hit an ``IndexError`` and the single-model
    path would silently draw zero curves.
    """
    if eval_times is None:
        return None
    if isinstance(eval_times, Iterable) and not isinstance(eval_times, (str, bytes)):
        times = [float(t) for t in eval_times]
        if not times:
            raise ValueError(
                "eval_times is an empty list. Pass one or more timepoints, "
                "or omit eval_times to use default horizons."
            )
        return times
    return [float(eval_times)]


def _plot_calibration_impl(
    ctx: PlotContext,
    *,
    models=None,
    eval_times=None,
    n_bins: int = 10,
    group_axis: Optional[SubpopulationAxis] = None,
    level: Optional[str] = None,
    return_data: bool = False,
    raw: bool = False,
    bin_strategy: str = "width",
    **kwargs,
):
    """Shared implementation for ``plot_calibration``.

    On **main** contexts, delegates to the same calibration helpers
    the old ``SurvivalLearner.plot_calibration`` used, reading data
    from the ``PlotContext`` instead of ``self``.

    On **validation** contexts, scopes models to the intersection of
    (a) the cohort's validated concordance rows, (b) fitted estimators
    present in main state, and (c) models supporting
    ``predict_survival_function``.

    On **subpopulation** contexts, dispatches to
    :func:`_plot_calibration_subpopulation` which computes calibration
    per level and produces a faceted grid.

    Parameters
    ----------
    ctx : PlotContext
        Main, validation, or subpopulation context.
    models : None, str, or list[str]
        Filter to specific model names (further intersected with the
        validated whitelist above).
    eval_times : float or list of float, optional
        Evaluation horizon(s). A scalar or list. The main branch seeds an
        interactive dropdown with these horizons; the static validation and
        subpopulation branches use a single horizon (the first when supplied,
        else the central horizon of the default grid). When omitted, defaults to
        the canonical evaluation grid (inherited ``user_eval_times`` -> auto,
        unioned with ``add_eval_times`` horizons).
    n_bins : int, default=10
    bin_strategy : {"width", "quantile"}, default="width"
        Bin predicted probabilities by equal width across [0, 1] or by
        equal-count quantiles. Forwarded to the calibration compute helpers;
        see :func:`mlsurv.viz.calibration.compute_calibration`.
    return_data : bool, default=False
        Return a tidy DataFrame via
        :func:`mlsurv.viz._resolve.calibration_to_tidy`.
    **kwargs
        Forwarded to :func:`mlsurv.viz.calibration.plot_calibration`.

    Returns
    -------
    plotly.graph_objects.Figure or pandas.DataFrame
    """
    return _dispatch_by_source(
        "_plot_calibration_impl",
        ctx,
        main=_plot_calibration_impl_main,
        validation=_plot_calibration_impl_validation,
        subpopulation=_plot_calibration_subpopulation,
        models=models,
        eval_times=eval_times,
        n_bins=n_bins,
        group_axis=group_axis,
        level=level,
        return_data=return_data,
        raw=raw,
        bin_strategy=bin_strategy,
        **kwargs,
    )


def _plot_calibration_impl_main(
    ctx: PlotContext,
    *,
    models=None,
    eval_times=None,
    n_bins: int = 10,
    return_data: bool = False,
    level: Optional[str] = None,
    group_axis: Optional[SubpopulationAxis] = None,
    raw: bool = False,
    bin_strategy: str = "width",
    **kwargs,
):
    """Main-source branch of :func:`_plot_calibration_impl`.

    Filters models to those with ``predict_survival_function`` and reads
    ``X_test`` / ``y_test`` from the ``PlotContext``. Renders an interactive
    calibration figure with a dropdown to switch evaluation horizon: a
    single-model view (titled with the model name) when one model is
    selected, otherwise a multi-model comparison ("Calibration Comparison").
    ``eval_times`` accepts a scalar or a list and seeds the dropdown; when
    omitted, it defaults to the canonical evaluation grid (inherited
    ``user_eval_times`` -> auto, unioned with ``add_eval_times`` horizons), so
    calibration lands on the same horizons as AUC(t)/Brier(t)/ROC.
    ``return_data`` returns a tidy frame spanning every horizon.

    Calibrated by default: a model with an attached calibrator
    (``train(calibrate=...)``) has its predicted ``S(t|x)`` warped to the
    calibrated scale before binning, so the curve matches the deployed model's
    evaluation metrics. ``raw=True`` plots the uncalibrated curves.
    """
    del group_axis, level  # subpopulation-only; ignored here
    from mlsurv.viz._palettes import get_model_display_name
    from mlsurv.viz.calibration import (
        _compute_multi_model_calibration_data,
        compute_calibration,
        plot_calibration_interactive,
    )

    state = ctx.state_ref
    resolved = _resolve_models_arg(ctx, models)
    if resolved is None:
        resolved = list(state.models.keys())

    model_dict = collect_surv_function_models(state, resolved)

    X_test = ctx.X_data
    y_test = ctx.y_data

    # Resolve the dropdown horizons once. When the user passes nothing, default
    # to the canonical eval grid (inherited user_eval_times -> auto, unioned
    # with add_eval_times), so calibration matches AUC(t)/Brier(t)/ROC. The
    # figure shows these in the dropdown.
    times_list = _normalize_eval_times(eval_times)
    if times_list is None:
        times_list = _default_eval_times(state, y_test, ctx.y_train)

    if len(model_dict) == 1:
        model_name, model = next(iter(model_dict.items()))
        cal_data = compute_calibration(
            model, X_test, y_test,
            eval_times=times_list, n_bins=n_bins,
            y_train=ctx.y_train,
            calibrator=_calibrator_for(ctx, model_name, raw),
            model_name=model_name,
            bin_strategy=bin_strategy,
        )
        if not cal_data:
            return _empty_calibration_figure(times_list[0], return_data)
        if return_data:
            return calibration_to_tidy(cal_data, model_name=model_name)
        kwargs.setdefault("title", f"Calibration: {get_model_display_name(model_name)}")
        return plot_calibration_interactive(cal_data, **kwargs)

    # Multi-model comparison: one dropdown entry per horizon.
    calibrators = _calibrators_for(ctx, model_dict.keys(), raw)
    data_by_time = {}
    for t in times_list:
        cal_at_t = _compute_multi_model_calibration_data(
            model_dict, X_test, y_test, t, n_bins,
            calibrators=calibrators,
            bin_strategy=bin_strategy,
        )
        if cal_at_t:
            data_by_time[t] = cal_at_t
    if not data_by_time:
        return _empty_calibration_figure(times_list[0], return_data)
    if return_data:
        frames = [calibration_to_tidy(cal_at_t) for cal_at_t in data_by_time.values()]
        return pd.concat(frames, ignore_index=True)
    kwargs.setdefault("title", "Calibration Comparison")
    return plot_calibration_interactive(data_by_time, **kwargs)


def _plot_calibration_impl_validation(
    ctx: PlotContext,
    *,
    models=None,
    eval_times=None,
    n_bins: int = 10,
    return_data: bool = False,
    level: Optional[str] = None,
    group_axis: Optional[SubpopulationAxis] = None,
    raw: bool = False,
    bin_strategy: str = "width",
    **kwargs,
):
    """Validation-source branch of :func:`_plot_calibration_impl`.

    Scopes models to the intersection of (a) the cohort's validated
    concordance rows, (b) fitted estimators in main state, and
    (c) models supporting ``predict_survival_function``. Requires
    ``X_val`` / ``y_val`` on the cohort (via ``validate(X_val=...,
    y_val=...)``). ``eval_times`` accepts a scalar or a list (single
    horizon used for the multi-model comparison); when omitted, it defaults to
    the canonical evaluation grid (inherited ``user_eval_times`` -> auto,
    unioned with ``add_eval_times`` horizons).

    Calibrated by default: a model with an attached calibrator
    (``train(calibrate=...)``) has its predicted ``S(t|x)`` warped to the
    calibrated scale on the validation cohort, matching the deployed model's
    validated metrics (validate() applies the same calibrator). ``raw=True``
    plots the uncalibrated curves.
    """
    del group_axis, level  # subpopulation-only; ignored here
    X_val = ctx.X_data
    y_val = ctx.y_data
    if X_val is None or y_val is None:
        raise ValueError(
            f"Validation cohort {ctx.name!r} has no stored X_val/y_val; "
            f"re-run validate() with the raw cohort features and outcomes."
        )

    # Build the validated + survival-function-capable whitelist.
    # `_resolve_models_arg` already raises ValueError if the user
    # requests a model that is not in `_available_model_names(ctx)`
    # (i.e. not in the validated concordance rows), so by the time
    # we enter the loop below, every name in `requested` is
    # guaranteed to be in `validated_names` too.
    validated_names = set(_available_model_names(ctx))
    fitted_models = ctx.fitted_models or {}
    requested = _resolve_models_arg(ctx, models)

    model_dict = {}
    dropped_for_survfunc: List[str] = []
    for name, model in fitted_models.items():
        if name not in validated_names:
            continue
        if not hasattr(model, "predict_survival_function"):
            dropped_for_survfunc.append(name)
            continue
        if requested is not None and name not in requested:
            continue
        model_dict[name] = model

    # If the user explicitly requested models that were dropped for
    # lacking predict_survival_function, tell them — a silent drop
    # would leave them wondering why their chosen model is missing
    # from the figure. Only warn for the explicit-request case; the
    # "all validated models" path (requested is None) is expected to
    # silently skip models like FSSVM-regression that cannot produce
    # survival probabilities.
    if requested is not None:
        explicit_dropped = [n for n in dropped_for_survfunc if n in requested]
        if explicit_dropped:
            # stacklevel=4: here → _dispatch_by_source → _plot_calibration_impl
            # → ValidationView.plot_calibration. (Original was 2; +2 for the
            # extracted handler frame and the dispatcher frame.)
            warnings.warn(
                f"Skipping requested model(s) {explicit_dropped} on "
                f"cohort {ctx.name!r}: no predict_survival_function() "
                f"method. Calibration plots require survival probability "
                f"predictions.",
                SuggestionWarning,
                stacklevel=4,
            )

    if not model_dict:
        raise ValueError(
            f"No validated models support predict_survival_function on "
            f"cohort {ctx.name!r}. Validated: {sorted(validated_names)}."
        )

    from mlsurv.viz._palettes import get_model_display_name
    from mlsurv.viz.calibration import (
        _compute_multi_model_calibration_data,
        compute_calibration,
        plot_calibration as _plot_calibration_viz,
    )

    # Default to the canonical eval grid (inherited user_eval_times -> auto,
    # unioned with add_eval_times) so validation calibration matches the
    # cohort's AUC(t)/Brier(t) horizons.
    user_times = _normalize_eval_times(eval_times)
    times_list = (
        user_times if user_times is not None
        else _default_eval_times(ctx.state_ref, y_val, ctx.y_train)
    )

    if len(model_dict) == 1:
        model_name, model = next(iter(model_dict.items()))
        cal_data = compute_calibration(
            model,
            X_val,
            y_val,
            eval_times=times_list,
            n_bins=n_bins,
            y_train=ctx.y_train,
            calibrator=_calibrator_for(ctx, model_name, raw),
            model_name=model_name,
            bin_strategy=bin_strategy,
        )
        if return_data:
            return calibration_to_tidy(cal_data, model_name=model_name)
        kwargs.setdefault("title", f"Calibration: {get_model_display_name(model_name)}")
        return _plot_calibration_viz(cal_data, **kwargs)

    # Multi-model comparison at a single timepoint.
    eval_time = _comparison_horizon(user_times, times_list)
    # stacklevel=5: here -> _dispatch_by_source -> _plot_calibration_impl
    # -> ValidationView.plot_calibration -> user code.
    if user_times is not None and len(user_times) > 1:
        warnings.warn(
            f"Multi-model calibration compares models at a single horizon; "
            f"using t={eval_time:.1f} (the first of the {len(user_times)} times "
            f"given). Call plot_calibration once per horizon for the rest.",
            SuggestionWarning, stacklevel=5,
        )
    cal_data = _compute_multi_model_calibration_data(
        model_dict, X_val, y_val, eval_time, n_bins,
        calibrators=_calibrators_for(ctx, model_dict.keys(), raw),
        bin_strategy=bin_strategy,
    )
    if not cal_data:
        return _empty_calibration_figure(eval_time, return_data)
    if return_data:
        return calibration_to_tidy(cal_data)
    kwargs.setdefault("title", f"Calibration Comparison (t = {eval_time:.1f})")
    return _plot_calibration_viz(cal_data, **kwargs)


# ----------------------------------------------------------------------
# plot_roc
# ----------------------------------------------------------------------


