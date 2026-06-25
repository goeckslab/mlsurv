"""Per-family plot dispatch for Kaplan-Meier / survival (split from ``_plot_impl.py``)."""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from mlsurv._warnings import SuggestionWarning
from mlsurv.viz._context import PlotContext, SubpopulationAxis
from mlsurv.viz._data import _extract_from_eval_result, get_survival_test_data
from mlsurv.viz._resolve import (
    collect_reference_survival_data,
    resolve_references_param,
    survival_test_data_to_tidy,
)
from mlsurv.viz._plot_impl_shared import (
    _dispatch_by_source,
    _inject_time_unit,
    _require_group_axis,
    _resolve_models_arg,
)


def _build_subpopulation_survival_df(
    ctx: PlotContext, group_variable: str
) -> pd.DataFrame:
    """Build a time/event/Subpopulation DataFrame for KM stratification.

    Reads ``ctx.state_ref.y_test`` + ``ctx.state_ref.subpopulations_test``
    and drops rows with NaN subpopulation labels (e.g. held-out samples
    that lacked the stratifying variable). Emits a ``SuggestionWarning``
    if any rows are dropped, mirroring the earlier
    ``plot_subpopulation_survival`` wrapper.
    """
    state = ctx.state_ref
    if state is None:
        raise ValueError(
            "SubpopulationView.plot_survival requires access to learner "
            "state for y_test + subpopulations_test. ctx.state_ref is None."
        )

    y_test = getattr(state, "y_test", None)
    subpops_test = getattr(state, "subpopulations_test", None) or {}
    labels = subpops_test.get(group_variable)
    if y_test is None or labels is None:
        raise ValueError(
            f"No subpopulation test data for {group_variable!r}. "
            f"Call learner.setup(subpopulations={{...}}) and train() first."
        )

    if isinstance(y_test, np.ndarray) and y_test.dtype.names:
        df = pd.DataFrame({
            "time": y_test["time"],
            "event": y_test["event"].astype(int),
            "Subpopulation": labels,
        })
    else:
        df = pd.DataFrame({
            "time": y_test["time"],
            "event": y_test["event"],
            "Subpopulation": labels,
        })

    n_before = len(df)
    df = df.dropna(subset=["Subpopulation"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        # stacklevel=6 walks up six frames so the warning is attributed
        # to user code, not framework internals. Chain from the warn
        # call upward:
        #   1. _build_subpopulation_survival_df (this function)
        #   2. _plot_survival_subpopulation
        #   3. _dispatch_by_source
        #   4. _plot_survival_impl
        #   5. SubpopulationView.plot_survival
        #   6. user code
        # Locked in by tests/test_learner_views.py so a refactor that
        # changes the call chain also has to update this count.
        warnings.warn(
            f"Dropped {n_dropped}/{n_before} test samples with NaN "
            f"subpopulation labels for variable {group_variable!r}.",
            SuggestionWarning,
            stacklevel=6,
        )
    return df


def _plot_survival_subpopulation(
    ctx: PlotContext,
    *,
    group_axis: Optional[SubpopulationAxis],
    models=None,
    n_groups: Optional[int] = None,
    show_risk_table: Optional[bool] = None,
    time_limit: Optional[float] = None,
    level: Optional[str] = None,
    return_data: bool = False,
    references=None,
    **kwargs,
):
    """Subpopulation branch of :func:`_plot_survival_impl`.

    Renders per-level KM curves by stratifying ``ctx.state_ref.y_test``
    on ``ctx.state_ref.subpopulations_test[group_variable]``. Mirrors the
    behaviour of the deleted ``plot_subpopulation_survival`` wrapper.
    """
    axis = _require_group_axis(ctx, group_axis, "_plot_survival_impl")

    if references is not None:
        raise NotImplementedError(
            "references= is not supported on SubpopulationView.plot_survival in v1"
        )
    if models is not None:
        raise NotImplementedError(
            "models= filtering is not applicable to "
            "SubpopulationView.plot_survival — KM curves are model-free "
            "stratifications of the test outcomes by subpopulation level."
        )
    if n_groups is not None:
        raise NotImplementedError(
            "n_groups= is not supported on SubpopulationView.plot_survival "
            "— the number of KM curves is fixed by the subpopulation levels."
        )

    df = _build_subpopulation_survival_df(ctx, axis.group_variable)
    if df.empty:
        raise ValueError(
            f"All subpopulation labels are NaN for variable "
            f"{axis.group_variable!r}; cannot render KM curves."
        )

    if level is not None:
        # Validate against the just-built df rather than calling
        # _iter_subpopulation_levels, which would re-emit the
        # NaN-dropped-samples SuggestionWarning that
        # _build_subpopulation_survival_df already emitted. The error
        # message format mirrors _iter_subpopulation_levels exactly so
        # callers see the same "Unknown subpopulation level …" message
        # as plot_calibration / plot_roc.
        available = sorted(df["Subpopulation"].unique())
        if level not in available:
            raise ValueError(
                f"Unknown subpopulation level {level!r} for variable "
                f"{axis.group_variable!r}. Available: {available}"
            )
        df = df[df["Subpopulation"] == level]

    if return_data:
        return df

    _inject_time_unit(kwargs, ctx)
    if "title" not in kwargs:
        if level is not None:
            kwargs["title"] = (
                f"Kaplan-Meier: {axis.group_variable} = {level}"
            )
        else:
            kwargs["title"] = f"Kaplan-Meier by {axis.group_variable}"

    # Per-level color map so curves stay consistent with the report
    # section and the earlier plot_subpopulation_survival wrapper.
    from mlsurv.viz._palettes import get_subpopulation_color
    unique_levels = sorted(df["Subpopulation"].unique())
    colors = {
        g: get_subpopulation_color(i) for i, g in enumerate(unique_levels)
    }

    from mlsurv.viz.survival import plot_km

    # show_risk_table defaults to True for parity with the legacy
    # plot_subpopulation_survival wrapper; callers can override.
    if show_risk_table is None:
        show_risk_table = True

    # When filtered to a single level, fall through to plot_km's
    # ungrouped path: the by-group renderer calls get_risk_colors(1)
    # which divides by zero, and the level name is already conveyed
    # by the title override above.
    if level is not None:
        return plot_km(
            df,
            show_ci=True,
            show_risk_table=show_risk_table,
            time_limit=time_limit,
            **kwargs,
        )

    return plot_km(
        df,
        group_col="Subpopulation",
        show_ci=True,
        show_risk_table=show_risk_table,
        time_limit=time_limit,
        colors=colors,
        **kwargs,
    )


def _reject_level_for_non_subpop(level, impl_label: str) -> None:
    if level is not None:
        raise NotImplementedError(
            f"level= is only supported on subpopulation plot contexts "
            f"({impl_label})."
        )


def _plot_survival_impl_main(
    ctx: PlotContext,
    *,
    models=None,
    n_groups: Optional[int] = None,
    show_risk_table: Optional[bool] = None,
    time_limit: Optional[float] = None,
    return_data: bool = False,
    references=None,
    level: Optional[str] = None,
    group_axis: Optional[SubpopulationAxis] = None,
    **kwargs,
):
    """Main-source branch of :func:`_plot_survival_impl`.

    Reads per-model test data from ``ctx.state_ref`` via
    :func:`~mlsurv.viz._data.get_survival_test_data`
    and renders risk-stratified KM curves. Honors ``references=`` by
    merging benchmark reference data into the test set.
    """
    del group_axis  # subpopulation-only; ignored on this branch
    _reject_level_for_non_subpop(level, "SubpopulationView.plot_survival")

    from mlsurv.viz.survival import plot_km

    state = ctx.state_ref

    # Reference inclusion: merge reference test data
    ref_names = resolve_references_param(
        state.benchmark_result, references,
    )
    ref_data = collect_reference_survival_data(state, ref_names)

    resolved = _resolve_models_arg(ctx, models)
    if resolved is None:
        resolved = list(state.models.keys())

    test_data = get_survival_test_data(state, resolved)

    # Merge reference data
    test_data.update(ref_data)

    if not test_data:
        raise ValueError(
            f"No valid test predictions for models: {resolved}. "
            f"All models were excluded due to non-finite risk scores."
        )

    if return_data:
        return survival_test_data_to_tidy(test_data)

    _inject_time_unit(kwargs, ctx)

    if len(test_data) == 1:
        _, df = next(iter(test_data.items()))
        return plot_km(
            df, risk_col="risk_score", n_groups=n_groups,
            show_risk_table=show_risk_table, time_limit=time_limit,
            **kwargs,
        )

    return plot_km(
        test_data, n_groups=n_groups, time_limit=time_limit,
        show_risk_table=show_risk_table, **kwargs,
    )


def _plot_survival_impl_validation(
    ctx: PlotContext,
    *,
    models=None,
    n_groups: Optional[int] = None,
    show_risk_table: Optional[bool] = None,
    time_limit: Optional[float] = None,
    return_data: bool = False,
    references=None,
    level: Optional[str] = None,
    group_axis: Optional[SubpopulationAxis] = None,
    **kwargs,
):
    """Validation-source branch of :func:`_plot_survival_impl`.

    Pulls per-model risk scores from ``ctx.eval_result.test_set`` via
    :func:`~mlsurv.viz._data._extract_from_eval_result`
    (no refit on the validation cohort). ``references=`` is rejected
    here; benchmark references are main-state-only.
    """
    del group_axis  # subpopulation-only; ignored on this branch
    _reject_level_for_non_subpop(level, "SubpopulationView.plot_survival")

    if references is not None:
        raise NotImplementedError(
            "references= is not supported on ValidationView.plot_survival in v1"
        )

    eval_result = ctx.eval_result
    if eval_result is None or eval_result.test_set is None:
        raise ValueError(
            f"Validation cohort {ctx.name!r} has no test_set data. "
            f"Re-run validate() with a cohort that produces risk-score "
            f"predictions."
        )

    model_names = _resolve_models_arg(ctx, models)

    # Reuse the pure helper that takes an EvaluationResult directly.
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

    if return_data:
        return survival_test_data_to_tidy(test_data)

    _inject_time_unit(kwargs, ctx)

    from mlsurv.viz.survival import plot_km

    if len(test_data) == 1:
        _, df = next(iter(test_data.items()))
        return plot_km(
            df,
            risk_col="risk_score",
            n_groups=n_groups,
            show_risk_table=show_risk_table,
            time_limit=time_limit,
            **kwargs,
        )

    return plot_km(
        test_data,
        n_groups=n_groups,
        time_limit=time_limit,
        show_risk_table=show_risk_table,
        **kwargs,
    )


def _plot_survival_impl(
    ctx: PlotContext,
    *,
    models=None,
    n_groups: Optional[int] = None,
    show_risk_table: Optional[bool] = None,
    time_limit: Optional[float] = None,
    return_data: bool = False,
    references=None,
    group_axis: Optional[SubpopulationAxis] = None,
    level: Optional[str] = None,
    **kwargs,
):
    """Shared implementation for ``plot_survival``.

    On main contexts, reads per-model test data from
    ``ctx.state_ref`` and renders risk-stratified KM curves.  On
    validation contexts, pulls per-model risk scores from
    ``ctx.eval_result.test_set`` via the pure helper
    :func:`_extract_from_eval_result` (no refit).  On subpopulation
    contexts, reads ``ctx.state_ref.y_test`` and
    ``ctx.state_ref.subpopulations_test[group_variable]`` and
    stratifies the KM curves by subpopulation level (model-free).
    """
    return _dispatch_by_source(
        "_plot_survival_impl",
        ctx,
        main=_plot_survival_impl_main,
        validation=_plot_survival_impl_validation,
        subpopulation=_plot_survival_subpopulation,
        models=models,
        n_groups=n_groups,
        show_risk_table=show_risk_table,
        time_limit=time_limit,
        return_data=return_data,
        references=references,
        group_axis=group_axis,
        level=level,
        **kwargs,
    )


# ----------------------------------------------------------------------
# plot_calibration
# ----------------------------------------------------------------------


