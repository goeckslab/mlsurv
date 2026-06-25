"""Shared plot implementations — single dispatch point for all plot calls.

This module hosts the ``_plot_X_impl`` functions that every plot call in
mlsurv routes through, regardless of context:

- **Main** — ``SurvivalLearner.plot_*`` methods build a
  ``PlotContext.from_main(self)`` and delegate here.
- **Validation** — ``ValidationView.plot_*`` methods build a
  ``PlotContext.from_validation(learner, val_result)`` and delegate here.
- **Subpopulation** — ``SubpopulationView.plot_*`` methods build a
  ``PlotContext.from_subpopulation(learner, var)`` and delegate here.

Each ``_plot_X_impl`` function takes a :class:`~mlsurv.viz._context.PlotContext`
and branches on ``ctx.source`` to route to the correct code path.

This file is a re-export facade after the per-family split.
"""
from __future__ import annotations

from mlsurv.viz._plot_impl_shared import (
    _available_model_names,
    _benchmark_reference_names,
    _inject_time_unit,
    _iter_subpopulation_levels,
    _require_group_axis,
    _resolve_models_arg,
    _subpop_auc_long_from_axis,
)
from mlsurv.viz._plot_impl_metric_time import (
    _plot_metric_time_impl,
    _plot_metric_time_subpopulation,
    _plot_metric_time_subpopulation_multi,
)
from mlsurv.viz._plot_impl_metric import (
    _plot_metric_impl,
    _plot_metric_subpopulation,
)
from mlsurv.viz._plot_impl_survival import (
    _build_subpopulation_survival_df,
    _plot_survival_impl,
    _plot_survival_subpopulation,
)
from mlsurv.viz._plot_impl_calibration import (
    _build_calibration_model_dict,
    _empty_calibration_figure,
    _plot_calibration_impl,
    _plot_calibration_subpopulation,
)
from mlsurv.viz._plot_impl_roc import (
    _compute_roc_ci_data_main,
    _plot_roc_impl,
    _plot_roc_subpopulation,
    _resolve_roc_eval_times_from_ctx,
    _resolve_roc_eval_times_main,
)
from mlsurv.viz._plot_impl_importance import _plot_importance_impl

__all__ = [
    "_available_model_names",
    "_benchmark_reference_names",
    "_build_calibration_model_dict",
    "_build_subpopulation_survival_df",
    "_compute_roc_ci_data_main",
    "_empty_calibration_figure",
    "_inject_time_unit",
    "_iter_subpopulation_levels",
    "_plot_calibration_impl",
    "_plot_calibration_subpopulation",
    "_plot_importance_impl",
    "_plot_metric_impl",
    "_plot_metric_subpopulation",
    "_plot_metric_time_impl",
    "_plot_metric_time_subpopulation",
    "_plot_metric_time_subpopulation_multi",
    "_plot_roc_impl",
    "_plot_roc_subpopulation",
    "_plot_survival_impl",
    "_plot_survival_subpopulation",
    "_require_group_axis",
    "_resolve_models_arg",
    "_resolve_roc_eval_times_from_ctx",
    "_resolve_roc_eval_times_main",
    "_subpop_auc_long_from_axis",
]
