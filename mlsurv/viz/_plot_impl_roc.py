"""Per-family plot dispatch for time-dependent ROC (split from ``_plot_impl.py``)."""
from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from mlsurv._shared import tidy_model_col as _tidy_model_col
from mlsurv._warnings import (
    InternalWarning,
    SuggestionWarning,
    warn_missing_roc_ci,
)
from mlsurv.logging_utils import get_logger
from mlsurv.viz._context import PlotContext, SubpopulationAxis
from mlsurv.viz._data import _extract_from_eval_result, get_survival_test_data
from mlsurv.viz._resolve import (
    _swap_benchmark_eval,
    compose_models_with_refs,
    resolve_references_param,
)
from mlsurv.viz._plot_impl_shared import (
    _dispatch_by_source,
    _finalize_subpop_facet_layout,
    _format_subpop_facet_axes,
    _iter_subpopulation_levels,
    _make_subpop_facet_grid,
    _require_group_axis,
    _resolve_models_arg,
    _subpop_grid_dims,
)

logger = get_logger("viz")


def _resolve_roc_eval_times_from_ctx(
    ctx: PlotContext,
    t: Optional[float],
) -> List[float]:
    """Resolve ROC evaluation times for a validation cohort.

    Priority: explicit *t* → ``ctx.eval_result.auc_t`` grid → median
    event time in ``ctx.y_data``.
    """
    if t is not None:
        return [float(t)]

    eval_result = ctx.eval_result
    if eval_result is not None and eval_result.auc_t is not None:
        auc_t = eval_result.auc_t
        if not auc_t.empty and "eval_time" in auc_t.columns:
            times = sorted(
                et
                for et in auc_t["eval_time"].unique()
                if np.isfinite(et)
            )
            if times:
                return times

    y_data = ctx.y_data
    if y_data is None:
        raise ValueError(
            f"Cannot determine default ROC evaluation time for cohort "
            f"{ctx.name!r}: no y_data stored. Pass an explicit t=."
        )
    event_times = y_data["time"][y_data["event"]]
    if len(event_times) == 0:
        raise ValueError(
            f"Cannot determine default ROC evaluation time for cohort "
            f"{ctx.name!r}: no events in the validation outcomes. "
            f"Pass an explicit t=."
        )
    return [float(np.median(event_times))]


def _plot_roc_subpopulation(
    ctx: PlotContext,
    *,
    group_axis: Optional[SubpopulationAxis],
    models=None,
    t: Optional[float] = None,
    show_auc: bool = True,
    show_ci="auto",
    level: Optional[str] = None,
    return_data: bool = False,
    references=None,
    **kwargs,
):
    """Subpopulation branch of :func:`_plot_roc_impl`.

    Predicts risk scores from fitted models on the full ``ctx.X_data``,
    then slices per subpopulation level and calls
    :func:`~mlsurv.viz.metrics.compute_roc_curves` per slice.
    IPCW weights are always from ``ctx.y_train`` (training censoring
    distribution), never per-level.
    """
    if references is not None:
        raise NotImplementedError(
            "references= is not supported on SubpopulationView.plot_roc in v1"
        )
    if show_ci not in (True, "auto"):
        # Default is "auto"; True is accepted (warns below). Reject any other
        # explicit override (False or otherwise). Bootstrap CI bands are
        # main/validation only — the subpop renderer delegates to
        # compute_roc_curves per level without bootstrap support.
        raise NotImplementedError(
            "show_ci= is not supported on SubpopulationView.plot_roc in v1 "
            "(bootstrap CI bands are main/validation only)."
        )
    if show_ci is True:
        # Explicit CI request the subpop surface cannot honor: tell the user
        # rather than silently dropping the ribbon. "auto" (the default) stays
        # silent. stacklevel=5: here -> _dispatch_by_source -> _plot_roc_impl
        # -> SubpopulationView.plot_roc -> user code.
        warnings.warn(
            "Bootstrap ROC CI bands are not available on the subpopulation "
            "surface; showing the point ROC curve only. Pass show_ci='auto' "
            "(the default) to silence this.",
            SuggestionWarning,
            stacklevel=5,
        )
    axis = _require_group_axis(ctx, group_axis, "_plot_roc_impl")

    X_data = ctx.X_data
    y_data = ctx.y_data
    y_train = ctx.y_train
    if X_data is None or y_data is None:
        raise ValueError(
            "Subpopulation ROC requires X_test / y_test in the plot "
            "context. ctx.X_data or ctx.y_data is None."
        )
    if y_train is None:
        raise ValueError(
            "Subpopulation ROC requires y_train for IPCW weighting. "
            "ctx.y_train is None."
        )

    fitted_models = ctx.fitted_models or {}
    requested = _resolve_models_arg(ctx, models)
    model_names = (
        [n for n in fitted_models if n in requested]
        if requested is not None
        else list(fitted_models.keys())
    )

    if not model_names:
        raise ValueError(
            "No fitted models available for subpopulation ROC."
        )

    # Predict risk scores on the full test set once.
    risk_scores_full: dict = {}
    for name in model_names:
        model = fitted_models[name]
        try:
            scores = model.predict(X_data)
            if not np.all(np.isfinite(scores)):
                warnings.warn(
                    f"Model {name!r} produced non-finite risk scores; "
                    f"skipping from subpopulation ROC.",
                    SuggestionWarning,
                    stacklevel=5,
                )
                continue
            risk_scores_full[name] = np.asarray(scores)
        except (ValueError, RuntimeError, TypeError, AttributeError) as exc:
            warnings.warn(
                f"Model {name!r} failed to predict risk scores: {exc}. "
                f"Skipping from subpopulation ROC.",
                SuggestionWarning,
                stacklevel=5,
            )

    if not risk_scores_full:
        raise ValueError(
            "No models produced valid risk scores for subpopulation ROC."
        )

    # Determine evaluation time.
    if t is not None:
        eval_time = float(t)
    else:
        event_times = y_data["time"][y_data["event"]]
        if len(event_times) == 0:
            raise ValueError(
                f"Cannot determine default ROC evaluation time for "
                f"subpopulation {axis.group_variable!r}: no events in "
                f"test outcomes. Pass an explicit t=."
            )
        eval_time = float(np.median(event_times))

    levels_and_masks = _iter_subpopulation_levels(
        ctx, axis.group_variable, level=level,
    )

    from mlsurv.viz.metrics import compute_roc_curves

    # Compute per-level ROC curves.
    per_level_roc: List[Tuple[str, pd.DataFrame]] = []
    for lvl, mask in levels_and_masks:
        y_level = y_data[mask]
        risk_level = {
            name: scores[mask] for name, scores in risk_scores_full.items()
        }
        try:
            roc_df = compute_roc_curves(y_train, y_level, risk_level, eval_time)
        except (ValueError, RuntimeError) as exc:
            warnings.warn(
                f"ROC computation failed for subpopulation level "
                f"{lvl!r}: {exc}. Skipping this level.",
                SuggestionWarning,
                stacklevel=5,
            )
            continue
        if not roc_df.empty:
            per_level_roc.append((lvl, roc_df))
        else:
            # Defensive: compute_roc_curves currently raises rather than
            # returning empty, but guard against future refactors.
            warnings.warn(
                f"ROC computation returned empty data for subpopulation "
                f"level {lvl!r}. Skipping this level.",
                SuggestionWarning,
                stacklevel=5,
            )

    if not per_level_roc:
        raise ValueError(
            f"ROC could not be computed for any subpopulation level "
            f"of {axis.group_variable!r}."
        )

    # return_data: concat per-level tidy frames with a "subpopulation" col.
    if return_data:
        dfs = []
        for lvl, roc_df in per_level_roc:
            df = _tidy_model_col(roc_df.copy())
            df.insert(0, "subpopulation", lvl)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    # Single level: delegate to the standard single-panel renderer.
    if len(per_level_roc) == 1:
        from mlsurv.viz.metrics import plot_roc as _plot_roc_viz
        lvl, roc_df = per_level_roc[0]
        title = kwargs.pop(
            "title",
            f"ROC: {axis.group_variable} = {lvl} "
            f"(t = {eval_time:.1f})",
        )
        return _plot_roc_viz(roc_df, show_auc=show_auc, title=title, **kwargs)

    # Multi-level faceted grid.
    return _render_roc_facet_grid(
        per_level_roc,
        show_auc=show_auc,
        eval_time=eval_time,
        group_variable=axis.group_variable,
        **kwargs,
    )


def _render_roc_facet_grid(
    per_level_roc: List[Tuple[str, pd.DataFrame]],
    *,
    show_auc: bool,
    eval_time: float,
    group_variable: str,
    **kwargs,
):
    """Render a multi-level faceted ROC grid (one panel per subpopulation).

    Independently testable: given ``per_level_roc`` (a list of
    ``(level_name, roc_df)`` where each ``roc_df`` has ``model``/``fpr``/
    ``tpr``/``auc`` columns), builds a 3-column-wrapping ``make_subplots``
    grid with a dashed diagonal reference per panel and one ROC line per
    model (shared legend, drawn only on the first panel).

    ``title`` (default ``"Subpopulation ROC: {group_variable} (t = ...)"``)
    and ``width`` / ``height`` are consumed from ``**kwargs``.
    """
    import plotly.graph_objects as go
    from mlsurv.viz._palettes import get_model_color, get_model_display_name

    n_levels = len(per_level_roc)
    n_cols, _ = _subpop_grid_dims(n_levels)
    level_names = [lvl for lvl, _ in per_level_roc]
    fig = _make_subpop_facet_grid(level_names)

    for idx, (lvl, roc_df) in enumerate(per_level_roc):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        is_first = idx == 0

        # Diagonal reference
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="grey", width=1, dash="dash"),
            xref=f"x{idx + 1}" if idx > 0 else "x",
            yref=f"y{idx + 1}" if idx > 0 else "y",
        )

        for model_name, grp in roc_df.groupby("model", sort=False):
            color = get_model_color(model_name)
            display_name = get_model_display_name(model_name)
            auc_val = grp["auc"].iloc[0]

            # Shared legend uses generic model name (no AUC) because
            # each panel has different AUC values — showing only the
            # first panel's AUC would be misleading.
            hover_parts = [f"<b>{display_name}</b>"]
            if show_auc:
                hover_parts.append(f"AUC = {auc_val:.3f}")
            hover_parts.append(
                "FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
            )
            hovertemplate = "<br>".join(hover_parts)

            fig.add_trace(
                go.Scatter(
                    x=grp["fpr"],
                    y=grp["tpr"],
                    mode="lines",
                    name=display_name,
                    legendgroup=display_name,
                    showlegend=is_first,
                    line=dict(color=color),
                    hovertemplate=hovertemplate,
                ),
                row=row, col=col,
            )

    _format_subpop_facet_axes(
        fig, n_levels,
        x_title="False Positive Rate",
        y_title="True Positive Rate",
        dtick=0.2,
        constrain="domain",
    )

    title = kwargs.pop(
        "title",
        f"Subpopulation ROC: {group_variable} (t = {eval_time:.1f})",
    )
    return _finalize_subpop_facet_layout(
        fig,
        title=title,
        legend_anchor="bottom-right",
        width=kwargs.pop("width", None),
        height=kwargs.pop("height", None),
    )


def _resolve_roc_eval_times_main(state, t, resolved):
    """Determine ROC eval timepoints from ``LearnerState``.

    Priority: explicit *t* → evaluation auc_t times → median test event time.
    """
    if t is not None:
        return [float(t)]

    for name in resolved:
        mr = state.models.get(name)
        if mr is None or mr.evaluation is None:
            continue
        auc_t_df = getattr(mr.evaluation, "auc_t", None)
        if (
            auc_t_df is not None
            and not auc_t_df.empty
            and "eval_time" in auc_t_df.columns
        ):
            times = sorted(
                et for et in auc_t_df["eval_time"].unique()
                if np.isfinite(et)
            )
            if times:
                return times

    event_times = state.y_test["time"][state.y_test["event"]]
    if len(event_times) == 0:
        raise ValueError(
            "Cannot determine default evaluation time: the test "
            "set has no events. Pass an explicit 't' parameter."
        )
    return [float(np.median(event_times))]


def _compute_roc_ci_data_main(ctx, state, resolved, eval_times,
                              bootstrap_roc_curves_fn, show_ci="auto"):
    """Compute bootstrap ROC CI bands from ``PlotContext`` / ``LearnerState``.

    Returns the CI data dict, or ``None`` when bootstrap predictions are
    unavailable or computation fails. When ``show_ci is True`` (an explicit
    request) and no model has bootstrap predictions, emits a
    ``SuggestionWarning`` naming the ``bootstrap(...)`` flag to re-run.
    """
    pred_dfs = []
    for name in resolved:
        mr = state.models.get(name)
        if mr is None or mr.bootstrap is None:
            continue
        preds = mr.bootstrap.predictions
        if preds is not None and not preds.empty:
            pred_dfs.append(preds)

    if not pred_dfs:
        if show_ci is True:
            # stacklevel=7: warn_missing_roc_ci -> _compute_roc_ci_data_main
            # -> _plot_roc_impl_main -> _dispatch_by_source -> _plot_roc_impl
            # -> SurvivalLearner.plot_roc -> user code. One frame deeper than
            # the InternalWarning below (that warn fires inside this function;
            # this one fires inside the helper called from this function).
            warn_missing_roc_ci(stacklevel=7)
        return None

    try:
        all_predictions = pd.concat(pred_dfs, ignore_index=True)
        return bootstrap_roc_curves_fn(
            ctx.y_train,
            ctx.y_data,
            all_predictions,
            ctx.X_data.index,
            eval_times,
        )
    except ValueError as exc:
        # Only ValueError is caught — pd.concat shape/dtype mismatches and
        # input-validation raises inside bootstrap_roc_curves (missing
        # columns, invalid ci/n_grid).  RuntimeError/KeyError/TypeError
        # indicate bugs and must surface as tracebacks, not silent
        # ribbon-less plots.
        logger.exception(
            "Bootstrap CI computation failed in _compute_roc_ci_data_main"
        )
        warnings.warn(
            f"Bootstrap CI bands could not be computed and will be "
            f"omitted from the ROC plot: {exc}",
            InternalWarning,
            # stacklevel=6: _compute_roc_ci_data_main -> _plot_roc_impl_main
            # -> _dispatch_by_source -> _plot_roc_impl
            # -> SurvivalLearner.plot_roc -> user code.
            # Locked in by tests/test_plot_impl_stacklevels.py.
            stacklevel=6,
        )
        return None


def _plot_roc_impl(
    ctx: PlotContext,
    *,
    models=None,
    t: Optional[float] = None,
    show_auc: bool = True,
    show_ci="auto",
    group_axis: Optional[SubpopulationAxis] = None,
    level: Optional[str] = None,
    return_data: bool = False,
    references=None,
    **kwargs,
):
    """Shared implementation for ``plot_roc``.

    IPCW weights are always derived from ``ctx.y_train`` -- the *main*
    learner's training outcome. This is by design because the censoring
    distribution used for weighting must be the training distribution,
    not the validation/subpopulation cohort's outcomes. DO NOT "fix"
    this to ``ctx.y_data``.

    Parameters
    ----------
    ctx : PlotContext
        Main, validation, or subpopulation context.
    models, t, show_auc, show_ci, return_data : see
        :meth:`SurvivalLearner.plot_roc`.
    group_axis : SubpopulationAxis, optional
        When set (and ``ctx.source == "subpopulation"``), routes to the
        faceted ROC renderer.
    level : str, optional
        When set (subpopulation only), produce a single panel for this
        subpopulation level.

    Returns
    -------
    plotly.graph_objects.Figure or pandas.DataFrame
    """
    return _dispatch_by_source(
        "_plot_roc_impl",
        ctx,
        main=_plot_roc_impl_main,
        validation=_plot_roc_impl_validation,
        subpopulation=_plot_roc_subpopulation,
        models=models,
        t=t,
        show_auc=show_auc,
        show_ci=show_ci,
        group_axis=group_axis,
        level=level,
        return_data=return_data,
        references=references,
        **kwargs,
    )


def _plot_roc_impl_main(
    ctx: PlotContext,
    *,
    models=None,
    t: Optional[float] = None,
    show_auc: bool = True,
    show_ci="auto",
    return_data: bool = False,
    references=None,
    group_axis: Optional[SubpopulationAxis] = None,
    level: Optional[str] = None,
    **kwargs,
):
    """Main-source branch of :func:`_plot_roc_impl`.

    Reads per-model risk scores from
    :func:`~mlsurv.viz._data.get_survival_test_data`
    and computes ROC curves at each evaluation timepoint via
    :func:`~mlsurv.viz.metrics.compute_roc_curves`. When ``show_ci`` is
    ``"auto"`` (default) or ``True`` and bootstrap predictions are
    available, overlays bootstrap CI bands; an explicit ``True`` with no
    predictions warns. Honors ``references=`` by recursing through
    ``_swap_benchmark_eval``.
    """
    del group_axis, level
    from mlsurv.viz.metrics import (
        assemble_roc_curves_with_references,
        bootstrap_roc_curves,
        plot_roc as _plot_roc_viz,
        plot_roc_interactive as _plot_roc_interactive_viz,
    )

    state = ctx.state_ref

    # -- Reference inclusion: swap in benchmark evaluation data --
    ref_names = resolve_references_param(
        state.benchmark_result, references,
    )
    if ref_names is not None:
        with _swap_benchmark_eval(
            state, state.benchmark_result, ref_names, kwargs,
        ):
            fresh_ctx = PlotContext.from_state(state)
            return _plot_roc_impl(
                fresh_ctx,
                models=compose_models_with_refs(models, ref_names),
                t=t, show_auc=show_auc, show_ci=show_ci,
                return_data=return_data, references=None, **kwargs,
            )

    resolved = _resolve_models_arg(ctx, models)
    if resolved is None:
        resolved = list(state.models.keys())

    # get_survival_test_data already calls validate_risk_scores
    # internally, filtering out models with non-finite scores.
    test_data = get_survival_test_data(state, resolved)
    risk_scores = {
        name: tdf["risk_score"].values
        for name, tdf in test_data.items()
    }
    if not risk_scores:
        raise ValueError(
            f"No test predictions found for models: {resolved}. "
            f"Ensure the models have been trained and evaluated."
        )

    # --- Resolve evaluation timepoints ---
    # Use the per-model iteration (matches old main path behaviour)
    eval_times = _resolve_roc_eval_times_main(state, t, resolved)

    # --- Compute ROC curves at each timepoint ---
    # Partition risk_scores into models (full-length y_data) and references
    # (per-reference NaN-masked y_data), then assemble per-horizon ROC via
    # the shared helper. benchmark() evaluates NaN-bearing references on
    # their valid subset, so their entries in risk_scores have length =
    # non-NaN count — mismatched with the full y_data; the helper masks each
    # reference's outcomes accordingly.
    br = state.benchmark_result
    res = assemble_roc_curves_with_references(
        ctx.y_train, ctx.y_data, risk_scores, eval_times,
        reference_names=getattr(br, "reference_names", []) or [],
        reference_arrays=(
            (getattr(br, "reference_arrays", None) or {})
            if br is not None else {}
        ),
        catch=(ValueError, RuntimeError),
        on_error=lambda exc, kind, name, et: logger.debug(
            "ROC computation failed at t=%s for %s%s: %s",
            et, kind, f" {name!r}" if name else "", exc,
        ),
    )
    roc_data = res.roc_data
    all_roc_dfs = res.all_roc_dfs

    if res.missing_ref_arrays or res.mislength_refs:
        details = []
        if res.missing_ref_arrays:
            details.append(
                f"missing entries for {sorted(res.missing_ref_arrays)}"
            )
        if res.mislength_refs:
            details.append(
                f"length mismatches for {sorted(res.mislength_refs)} "
                f"(expected {len(ctx.y_data)})"
            )
        warnings.warn(
            f"BenchmarkResult.reference_arrays has {'; '.join(details)} "
            f"— ROC for these references will use unmasked y_data and "
            f"will likely fail length-guards downstream.",
            InternalWarning, stacklevel=5,
        )

    if res.dropped_model_horizons:
        warnings.warn(
            f"ROC computation failed for models at "
            f"{len(res.dropped_model_horizons)}"
            f" horizon(s): {res.dropped_model_horizons}. These horizons are "
            f"omitted from the plot.",
            SuggestionWarning, stacklevel=5,
        )
    if res.dropped_by_ref:
        summary = ", ".join(
            f"{r} ({len(horizons)}/{len(eval_times)} horizons)"
            for r, horizons in sorted(res.dropped_by_ref.items())
        )
        warnings.warn(
            f"ROC curves could not be computed for reference(s) at some "
            f"horizons: {summary}. Insufficient cases/controls in the "
            f"valid subset.",
            SuggestionWarning, stacklevel=5,
        )

    if return_data:
        return _tidy_model_col(
            pd.concat(all_roc_dfs, ignore_index=True),
        )

    # --- Compute bootstrap CI bands ---
    ci_data = None
    if show_ci:
        ci_data = _compute_roc_ci_data_main(
            ctx, state, resolved, eval_times, bootstrap_roc_curves,
            show_ci=show_ci,
        )

    # --- Choose plot function ---
    if len(eval_times) == 1 and ci_data is None:
        return _plot_roc_viz(
            roc_data[eval_times[0]], show_auc=show_auc, **kwargs,
        )
    return _plot_roc_interactive_viz(
        roc_data, ci_data=ci_data, show_auc=show_auc, **kwargs,
    )


def _plot_roc_impl_validation(
    ctx: PlotContext,
    *,
    models=None,
    t: Optional[float] = None,
    show_auc: bool = True,
    show_ci="auto",
    return_data: bool = False,
    references=None,
    group_axis: Optional[SubpopulationAxis] = None,
    level: Optional[str] = None,
    **kwargs,
):
    """Validation-source branch of :func:`_plot_roc_impl`.

    Pulls per-model risk scores from
    ``ctx.eval_result.test_set`` (no refit). Computes ROC curves at
    each timepoint and, when ``show_ci`` is ``"auto"`` (default) or
    ``True`` and the cohort has bootstrap predictions plus a stored
    ``X_val``, overlays CI bands; an explicit ``True`` that cannot be
    honored warns. ``references=`` is silently ignored (matches
    pre-refactor behavior; benchmark references are main-state-only).
    """
    del group_axis, level, references
    eval_result = ctx.eval_result
    if eval_result is None or eval_result.test_set is None:
        raise ValueError(
            f"Validation cohort {ctx.name!r} has no test_set data."
        )

    model_names = _resolve_models_arg(ctx, models)

    test_data = _extract_from_eval_result(eval_result, model_names)
    if not test_data:
        requested = (
            f"requested models {model_names}"
            if model_names
            else "any model"
        )
        raise ValueError(
            f"No risk-score test data available for {requested} on "
            f"cohort {ctx.name!r}."
        )
    risk_scores = {
        name: tdf["risk_score"].values for name, tdf in test_data.items()
    }

    eval_times = _resolve_roc_eval_times_from_ctx(ctx, t)

    from mlsurv.viz.metrics import (
        bootstrap_roc_curves,
        compute_roc_curves,
        plot_roc as _plot_roc_viz,
        plot_roc_interactive as _plot_roc_interactive_viz,
    )

    roc_data = {}
    all_roc_dfs = []
    for et in eval_times:
        roc_df = compute_roc_curves(
            ctx.y_train, ctx.y_data, risk_scores, et,
        )
        roc_data[et] = roc_df
        all_roc_dfs.append(roc_df)

    if return_data:
        return _tidy_model_col(
            pd.concat(all_roc_dfs, ignore_index=True),
        )

    # Bootstrap CI bands: silently no-op when the cohort has no bootstrap
    # predictions. Same pattern as _compute_roc_ci_data_main (fall back to
    # None + InternalWarning), but with a broader except clause since the
    # validation path lacks the upstream input guards the main path has.
    #
    # ctx.X_data can legitimately be None on a validation cohort (it is
    # only populated when validate() was called with the raw X_val).
    # The main-path counterpart does not need this guard because
    # learner.X_test is always set after setup(); on the validation
    # branch we have to check explicitly before dereferencing .index.
    # When X_data is missing but bootstrap predictions exist, warn the
    # user — this case is fixable by re-running validate() with the raw
    # X_val. When there are no bootstrap predictions at all, the fix is to
    # re-run bootstrap(include_predictions=True); that warning fires only on
    # an explicit show_ci=True (silent under the "auto" default).
    ci_data = None
    preds = (
        getattr(ctx.bootstrap_result, "predictions", None)
        if ctx.bootstrap_result is not None
        else None
    )
    has_preds = preds is not None and not preds.empty
    if show_ci and has_preds and ctx.X_data is not None:
        try:
            ci_data = bootstrap_roc_curves(
                ctx.y_train,
                ctx.y_data,
                preds,
                ctx.X_data.index,
                eval_times,
            )
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            # stacklevel=4: here -> _dispatch_by_source -> _plot_roc_impl
            # -> ValidationView.plot_roc. (Original was 2; +2 for the
            # extracted handler frame and the dispatcher frame.)
            warnings.warn(
                f"Bootstrap CI bands could not be computed and "
                f"will be omitted from the ROC plot for cohort "
                f"{ctx.name!r}: {exc}",
                InternalWarning,
                stacklevel=4,
            )
    elif show_ci and has_preds and ctx.X_data is None:
        # stacklevel=4: see comment above.
        warnings.warn(
            f"ROC CI bands omitted for cohort {ctx.name!r}: "
            f"bootstrap predictions exist but X_val was not stored "
            f"on this cohort. Re-run validate(X_val=..., y_val=...) "
            f"to enable CI bands.",
            SuggestionWarning,
            stacklevel=4,
        )
    elif show_ci is True:
        # Explicit CI request but no bootstrap predictions at all.
        # stacklevel=6: warn_missing_roc_ci -> _plot_roc_impl_validation ->
        # _dispatch_by_source -> _plot_roc_impl -> ValidationView.plot_roc ->
        # user code. (Lands on user code; verified by the stacklevel test.)
        warn_missing_roc_ci(stacklevel=6)

    if len(eval_times) == 1 and ci_data is None:
        return _plot_roc_viz(
            roc_data[eval_times[0]], show_auc=show_auc, **kwargs,
        )
    return _plot_roc_interactive_viz(
        roc_data, ci_data=ci_data, show_auc=show_auc, **kwargs,
    )


# =====================================================================
# plot_importance
# =====================================================================


