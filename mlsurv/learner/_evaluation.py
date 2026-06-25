"""EvaluationMixin — thin composition of core / bootstrap / interactions mixins.

The implementation lives in three sibling modules:

- ``_evaluation_core.py`` → :class:`CoreEvaluationMixin` (``evaluate``,
  ``validate``, dispatch, summary printing, paired-comparison cache).
- ``_evaluation_bootstrap.py`` → :class:`BootstrapMixin` (``bootstrap``).
- ``_evaluation_interactions.py`` → :class:`InteractionsMixin`
  (``interactions``, ``importance_test``, ``ph_test``,
  ``linearity_test``, ``rmst_test``, ``logrank_test``).

This file keeps the historic import path
(``from mlsurv.learner._evaluation import EvaluationMixin``) and re-exports
the module-level helpers that ``_benchmarking.py``, ``_prediction.py``, and
several test modules import from this path.
"""

from __future__ import annotations

from ._eval_context import EvalContext
from ._evaluation_bootstrap import BootstrapMixin
from ._evaluation_core import CoreEvaluationMixin
from ._evaluation_interactions import InteractionsMixin
from ._evaluation_helpers import (
    _IPCW_GHAT_SOURCES,
    _IPCW_TAU_AUTO,
    _SUBPOP_LABELS_UNSET,
    _align_columns,
    _compute_and_store_subpopulation_metrics,
    _emit_heavy_censoring_warning,
    _emit_ipcw_truncation_warning,
    _format_auto_eval_times,
    _is_nan,
    _print_evaluation_output,
    _print_subpopulation_summary,
    _resolve_ipcw_g_source,
    _resolve_ipcw_tau,
    _resolve_subpopulation_labels,
    _resolve_validate_subpopulation_override,
    _update_kang_cache,
)


class EvaluationMixin(CoreEvaluationMixin, BootstrapMixin, InteractionsMixin):
    """Mixin for model evaluation, bootstrapping, and external validation."""


__all__ = [
    "EvaluationMixin",
    "CoreEvaluationMixin",
    "BootstrapMixin",
    "InteractionsMixin",
    "EvalContext",
    # Helpers re-exported for backward-compatible imports.
    "_IPCW_GHAT_SOURCES",
    "_IPCW_TAU_AUTO",
    "_SUBPOP_LABELS_UNSET",
    "_align_columns",
    "_compute_and_store_subpopulation_metrics",
    "_emit_heavy_censoring_warning",
    "_emit_ipcw_truncation_warning",
    "_format_auto_eval_times",
    "_is_nan",
    "_print_evaluation_output",
    "_print_subpopulation_summary",
    "_resolve_ipcw_g_source",
    "_resolve_ipcw_tau",
    "_resolve_subpopulation_labels",
    "_resolve_validate_subpopulation_override",
    "_update_kang_cache",
]
