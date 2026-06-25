"""Shared helpers for the per-family ``_plot_*_impl`` modules.

Promoted from ``_plot_impl.py`` during the per-family split. These helpers are independent
of any single plot family and are used by both the per-family modules and
``_plot_impl_secondary.py``.
"""
from __future__ import annotations

import warnings
from math import ceil
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from mlsurv._warnings import SuggestionWarning
from mlsurv.logging_utils import get_logger
from mlsurv.viz._context import PlotContext, SubpopulationAxis

logger = get_logger("viz")

def _dispatch_by_source(
    impl_name: str,
    ctx: PlotContext,
    *,
    main: Optional[Callable] = None,
    validation: Optional[Callable] = None,
    subpopulation: Optional[Callable] = None,
    **kwargs,
):
    """Route a ``_plot_X_impl`` call to a per-source branch handler.

    Routes by ``ctx.source`` ∈ ``{'main', 'validation', 'subpopulation'}``.
    Each handler is an optional callable taking ``(ctx, **kwargs)``. An
    unset (``None``) handler raises ``NotImplementedError`` with a uniform
    message — matching the wording the per-family impls used pre-refactor.

    ``impl_name`` is purely cosmetic (embedded in the unsupported-source
    error); it is not used for routing.

    Note: every handler call goes through this function, so warnings
    emitted inside a handler must add **+2** to the stacklevel they would
    have used pre-refactor — one for the ``_plot_X_impl_<branch>`` frame,
    one for this dispatcher frame.
    """
    handler = {
        "main": main,
        "validation": validation,
        "subpopulation": subpopulation,
    }.get(ctx.source)
    if handler is None:
        raise NotImplementedError(
            f"{impl_name} does not support source={ctx.source!r}."
        )
    return handler(ctx, **kwargs)


def _available_model_names(ctx: PlotContext) -> List[str]:
    """Model names available on this context.

    For ``source='main'``, returns the union of ``fitted_models`` keys
    (authoritative ``_state.models``) and ``concordance['model_type']``
    values — the union matters for tuned models stored in ``_state.models``
    without an attached evaluation, e.g. after ``tune(..., evaluate=False)``.

    For ``source='validation'`` and ``source='subpopulation'``,
    ``fitted_models`` is copied from the main learner state and therefore
    includes models never evaluated against this cohort/group. Availability
    is cohort-scoped, so only ``concordance`` is consulted.
    """
    eval_result = ctx.eval_result
    names = set()
    if ctx.source == "main" and ctx.fitted_models:
        names.update(ctx.fitted_models.keys())
    if eval_result is not None and eval_result.concordance is not None:
        names.update(eval_result.concordance["model_type"].unique())
    if not names and ctx.fitted_models:
        names.update(ctx.fitted_models.keys())
    return list(names)


def _models_type_error(obj) -> TypeError:
    """Build the guiding ``TypeError`` for a bad ``models=`` value.

    A ``ModelResult`` is the common mistake (e.g. passing
    ``learner.best_model``), so point the user at the identifier-based
    contract: a name, list of names, or the ``'best'`` sentinel.
    """
    from mlsurv.results.containers import ModelResult
    if isinstance(obj, ModelResult):
        return TypeError(
            "plot_*(models=...) takes model names, not a ModelResult. "
            f"Pass models={obj.name!r} or models='best'."
        )
    return TypeError(
        "models must be None, str, or list[str] (a model name, list of "
        f"names, or 'best'); got {type(obj).__name__}."
    )


def _resolve_best_name(ctx: PlotContext) -> str:
    """Resolve the ``'best'`` sentinel to the top model's name.

    Selects the best non-synthetic model by CV score from the main
    learner state (``ctx.state_ref``), matching ``learner.best_model``.
    The choice is cohort-independent, so validation/subpopulation plots
    resolve ``'best'`` against the same main-state ranking.
    """
    from mlsurv._shared import best_model_key
    state = ctx.state_ref
    models = getattr(state, "models", None) if state is not None else None
    key = best_model_key(models) if models else None
    if key is None:
        raise ValueError(
            "No trained models to resolve models='best'. "
            "Run train() or run() first."
        )
    return key


def _resolve_models_arg(ctx: PlotContext, models) -> Optional[List[str]]:
    """Resolve the ``models=`` argument to a concrete list of names.

    Accepts ``None`` (all), ``str`` (single), ``list[str]``, or the
    ``'best'`` sentinel (standalone or as a list element), which resolves
    to the top model by CV score (``learner.best_model.name``). A
    ``ModelResult`` is rejected with a guiding ``TypeError`` — plot
    methods select by name, not by object.
    """
    if models is None:
        # On the main learner state, ``models=None`` means "all trained
        # models" — surface a clear error when there are none, rather than
        # letting a downstream "no evaluation results" mask the root cause.
        # View contexts are cohort-scoped (a model can be trained but absent
        # from this cohort's available set), so only guard ``source='main'``.
        if ctx.source == "main" and not _available_model_names(ctx):
            raise ValueError("No trained models. Run train() or run() first.")
        return None
    if isinstance(models, str):
        models = [models]
    if not isinstance(models, (list, tuple)):
        raise _models_type_error(models)
    # Resolve the 'best' sentinel per element and reject stray objects,
    # de-duplicating while preserving order (so ['best', 'coxph'] where
    # 'best' is 'coxph' does not draw the same curve twice). ``ModelResult``
    # is a plain dataclass, so ``m == "best"`` is False against it (its
    # generated __eq__ returns NotImplemented vs str) — it correctly falls
    # to the isinstance reject rather than matching the sentinel.
    resolved: List[str] = []
    from_best = set()
    for m in models:
        if m == "best":
            m = _resolve_best_name(ctx)
            from_best.add(m)
        elif not isinstance(m, str):
            raise _models_type_error(m)
        if m not in resolved:
            resolved.append(m)
    models = resolved
    available = set(_available_model_names(ctx))
    missing = [m for m in models if m not in available]
    if missing:
        # If 'best' resolved to a model absent from this (cohort-scoped)
        # context, name the sentinel rather than blaming a model name the
        # user never typed.
        best_missing = [m for m in missing if m in from_best]
        if best_missing:
            raise ValueError(
                f"models='best' resolved to {best_missing[0]!r} (top model "
                f"by CV score), but it is not available in the {ctx.source} "
                f"context {ctx.name!r}. Available here: {sorted(available)}. "
                f"Evaluate that model on this cohort, or pass an explicit "
                f"model name."
            )
        raise ValueError(
            f"Unknown models {missing} for {ctx.source} context "
            f"{ctx.name!r}. Available: {sorted(available)}."
        )
    return list(models)


def _calibrator_for(ctx: PlotContext, model_name: str, raw: bool):
    """Return the attached ``S(t|x)`` calibrator for *model_name*, or ``None``.

    Reads ``ctx.state_ref.models[model_name].calibrator`` (the deployed
    calibrator from ``train(calibrate=...)``). Returns ``None`` when *raw* is
    ``True``, when there is no learner state, or when the model has no
    calibrator — so the caller renders the raw S(t|x). The ``viz`` layer reaches
    the calibrator through ``LearnerState`` rather than calling the learner,
    keeping the one-way ``learner -> viz`` import rule intact.
    """
    if raw:
        return None
    state = ctx.state_ref
    models = getattr(state, "models", None) if state is not None else None
    result = models.get(model_name) if models else None
    return getattr(result, "calibrator", None) if result is not None else None


def _calibrators_for(ctx: PlotContext, model_names, raw: bool) -> dict:
    """Map ``model_name -> calibrator`` for the named models that have one.

    Thin multi-model wrapper over :func:`_calibrator_for`; models without an
    attached calibrator (and every model when *raw* is ``True``) are simply
    absent from the returned dict, so the multi-model data-prep renders them
    raw.
    """
    out = {}
    for name in model_names:
        cal = _calibrator_for(ctx, name, raw)
        if cal is not None:
            out[name] = cal
    return out


def _inject_time_unit(kwargs: dict, ctx: PlotContext) -> None:
    if "time_unit" not in kwargs and ctx.time_unit:
        kwargs["time_unit"] = ctx.time_unit


def _benchmark_reference_names(ctx: PlotContext) -> set:
    """Extract benchmark reference model names from ``ctx.state_ref``.

    Grouped subpopulation renderers use these to apply the reference
    palette / dashed-line styling. Returns an empty set when no benchmark
    has been run or no state is available.
    """
    state = ctx.state_ref
    if state is None:
        return set()
    bench = getattr(state, "benchmark_result", None)
    if bench is None:
        return set()
    return set(getattr(bench, "reference_names", None) or [])


def _require_group_axis(
    ctx: PlotContext,
    group_axis: Optional[SubpopulationAxis],
    impl_name: str,
) -> SubpopulationAxis:
    """Enforce the subpopulation-context invariant.

    ``SubpopulationView`` always passes a ``SubpopulationAxis`` into
    the impls. A missing axis on a subpopulation context means the
    caller bypassed the view — surface that as a ``ValueError`` rather
    than silently falling through to the validation branch.
    """
    if group_axis is None:
        raise ValueError(
            f"{impl_name}: ctx.source='subpopulation' requires a "
            f"group_axis. Call via SubpopulationView, not by hand."
        )
    return group_axis


def _iter_subpopulation_levels(
    ctx: PlotContext,
    group_variable: str,
    *,
    level: Optional[str] = None,
) -> List[Tuple[str, np.ndarray]]:
    """Yield ``(level_name, boolean_mask)`` pairs for subpopulation filtering.

    Shared by the calibration and ROC subpopulation branches. Reads
    ``ctx.state_ref.subpopulations_test[group_variable]`` to derive masks,
    handles NaN labels and zero-sample levels, and validates the ``level``
    parameter.

    Returns
    -------
    list of (str, ndarray)
        Each entry is ``(level_name, mask)`` where *mask* is a boolean
        array of length ``len(ctx.y_data)``.

    Raises
    ------
    ValueError
        If ``ctx.state_ref`` is ``None``, the group variable has no test
        labels, the requested ``level`` does not exist, or all levels
        are empty after filtering.
    """
    state = ctx.state_ref
    if state is None:
        raise ValueError(
            "Subpopulation plot requires access to learner state "
            "(ctx.state_ref is None)."
        )

    subpops_test = getattr(state, "subpopulations_test", None) or {}
    labels = subpops_test.get(group_variable)
    if labels is None:
        raise ValueError(
            f"No subpopulation test labels for {group_variable!r}. "
            f"Call learner.setup(subpopulations={{...}}) and train() first."
        )

    labels = np.asarray(labels)

    # Drop NaN labels (same treatment as _build_subpopulation_survival_df).
    nan_mask = pd.isna(labels)
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        # stacklevel=6: this function -> _plot_X_subpopulation
        # -> _dispatch_by_source -> _plot_X_impl -> SubpopulationView.plot_X
        # -> user code. Locked in by tests/test_plot_impl_stacklevels.py.
        warnings.warn(
            f"Dropped {n_nan}/{len(labels)} test samples with NaN "
            f"subpopulation labels for variable {group_variable!r}.",
            SuggestionWarning,
            stacklevel=6,
        )

    all_levels = sorted(set(labels[~nan_mask]))

    if level is not None:
        if level not in all_levels:
            raise ValueError(
                f"Unknown subpopulation level {level!r} for variable "
                f"{group_variable!r}. Available: {all_levels}"
            )
        all_levels = [level]

    result: List[Tuple[str, np.ndarray]] = []
    for lvl in all_levels:
        mask = (labels == lvl) & ~nan_mask
        n_samples = int(mask.sum())
        if n_samples == 0:
            # Defensive: with simple string labels this is unreachable
            # (a level in all_levels implies at least one non-NaN sample),
            # but guards against edge cases with unusual label types.
            warnings.warn(
                f"Subpopulation level {lvl!r} for variable "
                f"{group_variable!r} has 0 test samples; skipping.",
                SuggestionWarning,
                stacklevel=6,
            )
            continue
        result.append((str(lvl), mask))

    if not result:
        raise ValueError(
            f"All subpopulation levels for variable {group_variable!r} "
            f"have 0 test samples. Cannot produce a plot."
        )
    return result


def _subpop_auc_long_from_axis(axis: SubpopulationAxis, model_names):
    """Convert ``axis.auc_t`` to the long-format chunk consumed by
    :func:`plot_grouped_multi_metric_time`.

    Returns ``(df_chunk, std_chunk_or_None)`` with columns
    ``model_type, subpopulation, eval_time, value, metric_name`` (and
    ``value_std`` on the std chunk).
    """
    auc_t = axis.auc_t
    if model_names is not None:
        auc_t = auc_t[auc_t["model_type"].isin(model_names)]
    chunk = auc_t[["model_type", "subpopulation", "eval_time"]].copy()
    chunk["value"] = auc_t["auc"].values
    chunk["metric_name"] = "auc"

    std_chunk = None
    boot = axis.auc_t_bootstrap
    if boot is not None and not boot.empty:
        if model_names is not None:
            boot = boot[boot["model_type"].isin(model_names)]
        if not boot.empty:
            std_chunk = (
                boot
                .groupby(["subpopulation", "model_type", "eval_time"])["auc"]
                .std()
                .reset_index()
                .rename(columns={"auc": "value_std"})
            )
            std_chunk["metric_name"] = "auc"
    return chunk, std_chunk


# =====================================================================
# Subpopulation faceted-grid scaffold (shared by the ROC and calibration
# subpopulation renderers). These are PURE,
# data-parameterized helpers — they take titles / dtick / anchor keys as
# data, never an ``is_roc``-style behavior flag. plotly and theme imports
# stay function-local to avoid importing plotly at module load.
# =====================================================================


def _subpop_grid_dims(n_levels: int) -> Tuple[int, int]:
    """Return ``(n_cols, n_rows)`` for a subpopulation facet grid.

    Wraps at three columns: ``n_cols = min(n_levels, 3)`` and
    ``n_rows = ceil(n_levels / n_cols)``.
    """
    n_cols = min(n_levels, 3)
    n_rows = ceil(n_levels / n_cols)
    return n_cols, n_rows


def _make_subpop_facet_grid(level_names: List[str]):
    """Build the empty ``make_subplots`` scaffold for a subpopulation grid.

    One panel per level (titled by ``level_names``), wrapping at three
    columns with shared y-axes and the standard spacing used by both the
    ROC and calibration subpopulation renderers.
    """
    from plotly.subplots import make_subplots

    _, n_rows = _subpop_grid_dims(len(level_names))
    n_cols = min(len(level_names), 3)
    return make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=level_names,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.12 if n_rows > 1 else 0.08,
    )


def _format_subpop_facet_axes(
    fig,
    n_levels: int,
    *,
    x_title: str,
    y_title: str,
    axis_range: Tuple[float, float] = (0.0, 1.0),
    dtick: Optional[float] = None,
    constrain: Optional[str] = None,
) -> None:
    """Apply per-panel axis formatting to a subpopulation facet grid.

    Sets ``range`` on every panel, writes *x_title* only on the bottom row
    and *y_title* only on the first column (interior panels get blank
    titles). ``dtick`` and ``constrain``, when provided, are applied to
    **both** the x and y axes — faithfully reproducing the ROC renderer's
    behavior. Leaving them ``None`` (calibration) emits neither.
    """
    n_cols, n_rows = _subpop_grid_dims(n_levels)
    extra: dict = {}
    if dtick is not None:
        extra["dtick"] = dtick
    if constrain is not None:
        extra["constrain"] = constrain
    for idx in range(n_levels):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        fig.update_xaxes(
            range=list(axis_range), row=row, col=col,
            title_text=x_title if row == n_rows else "",
            **extra,
        )
        fig.update_yaxes(
            range=list(axis_range), row=row, col=col,
            title_text=y_title if col == 1 else "",
            **extra,
        )


# Legend anchor keys → fixed (x, y, xanchor, yanchor) corner positions.
# A data lookup, not a behavior flag.
_SUBPOP_LEGEND_ANCHORS = {
    "bottom-right": dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom"),
    "top-right": dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
}


def _finalize_subpop_facet_layout(
    fig,
    *,
    title: str,
    legend_anchor: str,
    legend_title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
):
    """Apply title, legend placement, and size overrides to a facet grid.

    ``legend_anchor`` selects a corner position from
    :data:`_SUBPOP_LEGEND_ANCHORS`. A ``legend_title`` is added only when
    provided. ``width`` / ``height`` are forwarded to
    :func:`~mlsurv.viz._theme.apply_size_overrides`.
    """
    from mlsurv.viz._theme import apply_size_overrides

    legend = dict(_SUBPOP_LEGEND_ANCHORS[legend_anchor])
    legend["bgcolor"] = "rgba(255,255,255,0.8)"
    if legend_title is not None:
        legend["title"] = dict(text=legend_title)
    fig.update_layout(title_text=title, legend=legend)
    apply_size_overrides(fig, width, height)
    return fig


