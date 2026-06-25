"""PlotContext — shared data carrier for main / validation / subpopulation plots.

Every ``plot_*`` method on :class:`SurvivalLearner` and the corresponding
proxies (:class:`ValidationView`, future ``SubpopulationView``) ultimately
call the same shared ``_plot_X_impl`` functions in
:mod:`mlsurv.viz._plot_impl`. Those impls need different data depending on
the source cohort (main test split, an external validation cohort, or a
filtered subpopulation). Rather than branching inside every impl on
``learner``/``val_result``/``sp_result`` directly, we build a frozen
:class:`PlotContext` up front and let the impl branch on ``ctx.source``
plus the already-collected frames.

Like the rest of :mod:`mlsurv.viz`, this module holds references to learner
state but does not import from :mod:`mlsurv.learner` at runtime (only for type
hints via ``TYPE_CHECKING``). The whole ``viz`` package keeps the dependency
direction ``learner -> viz`` intact; ``tests/test_import_discipline_1094.py``
enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import pandas as pd

from mlsurv.viz._data_metrics import _has_any_bootstrap

if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner
    from mlsurv.learner._state import LearnerState
    from mlsurv.results.containers import (
        BootstrapResult,
        EvaluationResult,
        SubpopulationResult,
        ValidationResult,
    )


PlotSource = Literal["main", "validation", "subpopulation"]


@dataclass(frozen=True)
class PlotContext:
    """Immutable bag of data required to render a single ``plot_*`` call.

    Fields are intentionally broad: not every impl reads every field, but
    every impl reads *some* of them, and the constructors below build a
    consistent snapshot regardless of source. Impls should treat the context
    as read-only — it is frozen so accidental mutation raises ``FrozenInstanceError``.

    Attributes
    ----------
    name : str
        Short label for the context. ``"main"`` for the learner's main test
        split, the cohort name (e.g. ``"Holdout"``) for validation, the
        subpopulation variable name (e.g. ``"menostat"``) for subpopulations.
    source : {"main", "validation", "subpopulation"}
        Discriminator used by impls to pick the right code path.
    eval_result : EvaluationResult, optional
        Combined per-model evaluation covering this context. ``None`` for
        :meth:`from_subpopulation` (a :class:`SubpopulationResult` stores
        aggregate metric frames, not a full ``EvaluationResult``).
    bootstrap_result : BootstrapResult, optional
        Combined bootstrap output. ``None`` if bootstrap wasn't run or for
        contexts that don't carry a top-level bootstrap aggregate.
    test_data : dict[str, DataFrame], optional
        Per-model ``time/event/risk_score`` frames used by KM and ROC plots.
        Usually ``None`` at construction time — impls build it lazily via
        the existing ``_get_survival_test_data`` helper so we don't pay the
        cost for plot types that don't need it.
    X_data : DataFrame, optional
        Feature matrix for the cohort (``X_test`` on main, ``X_val`` on
        validation, ``X_test`` on subpopulation — subpopulation filtering
        happens inside the impl using ``state_ref.subpopulations_test``).
    y_data : structured array, optional
        Survival outcome matching ``X_data``.
    y_train : structured array, optional
        **Always** the training-set outcome, regardless of cohort. Needed
        by :func:`compute_roc_curves` for IPCW weighting — the weighting
        distribution is the training distribution by definition, even when
        scoring on a validation cohort.
    fitted_models : dict[str, Any], optional
        Trained estimator objects keyed by model name. Pulled from
        ``learner._state.models[name].model`` so every context reuses the
        same fitted pipelines (no refit on validation data).
    eval_times : list[float], optional
        Explicit evaluation times if the user set ``eval_times=`` on
        ``evaluate()`` / ``run()``. ``None`` means auto-pick.
    time_unit : str, optional
        User-facing time unit label ("months", "years", ...).
    scoring : str, optional
        Default metric name (``learner.scoring``). Passed through so
        subpopulation/validation plots inherit the same default as main.
    state_ref : LearnerState, optional
        Escape hatch: a direct reference to ``learner._state`` so future
        impls can reach side-channel state (``benchmark_result``,
        ``subpopulations_test``) without widening this dataclass. Treated
        as read-only by every impl.
    """

    name: str
    source: PlotSource
    eval_result: Optional["EvaluationResult"] = None
    bootstrap_result: Optional["BootstrapResult"] = None
    test_data: Optional[Dict[str, pd.DataFrame]] = None
    X_data: Optional[pd.DataFrame] = None
    y_data: Any = None
    y_train: Any = None
    fitted_models: Optional[Dict[str, Any]] = None
    eval_times: Optional[List[float]] = None
    time_unit: Optional[str] = None
    scoring: Optional[str] = None
    state_ref: Optional["LearnerState"] = None

    def __post_init__(self) -> None:
        if self.source not in ("main", "validation", "subpopulation"):
            raise ValueError(
                f"PlotContext.source must be 'main', 'validation', or "
                f"'subpopulation'; got {self.source!r}"
            )
        # Cross-field invariant: validation contexts must carry an
        # EvaluationResult (no impl can do anything useful without one).
        if self.source == "validation" and self.eval_result is None:
            raise ValueError(
                "PlotContext(source='validation') requires eval_result "
                "to be populated. Use PlotContext.from_validation() to "
                "construct a valid instance from a ValidationResult."
            )

    # --------------------------------------------------------------
    # Constructors
    # --------------------------------------------------------------

    @classmethod
    def from_main(cls, learner: "SurvivalLearner") -> "PlotContext":
        """Build a context for the main learner state.

        ``learner.evaluation_result`` and ``learner.bootstrap_result``
        return ``None`` directly when no models have evaluation / bootstrap
        data — they do not raise. Any exception from these properties
        indicates broken learner state and is allowed to propagate
        rather than being silently converted to ``None``.
        """
        state = learner._state
        eval_result = learner.evaluation_result
        bootstrap_result = learner.bootstrap_result

        fitted_models = {
            name: mr.model
            for name, mr in state.models.items()
            if getattr(mr, "model", None) is not None
        }

        return cls(
            name="main",
            source="main",
            eval_result=eval_result,
            bootstrap_result=bootstrap_result,
            test_data=None,
            X_data=state.X_test,
            y_data=state.y_test,
            y_train=state.y_train,
            fitted_models=fitted_models,
            eval_times=state.user_eval_times,
            time_unit=state.time_unit,
            scoring=state.scoring,
            state_ref=state,
        )

    @classmethod
    def from_state(cls, state: "LearnerState") -> "PlotContext":
        """Build a main context from raw :class:`LearnerState`.

        Used by ``_plot_X_impl`` when re-entering after state mutation
        (e.g., :class:`_swap_benchmark_eval` injects reference models).
        Assembles eval/bootstrap from ``state.models`` directly, matching
        the logic in ``SurvivalLearner.evaluation_result`` /
        ``.bootstrap_result``.
        """
        from mlsurv.results.containers import BootstrapResult, EvaluationResult

        evaluated = {
            n: m for n, m in state.models.items()
            if m.evaluation is not None
        }
        eval_result = (
            EvaluationResult.combine(evaluated) if evaluated else None
        )

        bootstrap_result = None
        if _has_any_bootstrap(state.models):
            bootstrap_result = BootstrapResult.combine(state.models)

        fitted_models = {
            name: mr.model
            for name, mr in state.models.items()
            if getattr(mr, "model", None) is not None
        }

        return cls(
            name="main",
            source="main",
            eval_result=eval_result,
            bootstrap_result=bootstrap_result,
            test_data=None,
            X_data=state.X_test,
            y_data=state.y_test,
            y_train=state.y_train,
            fitted_models=fitted_models,
            eval_times=state.user_eval_times,
            time_unit=state.time_unit,
            scoring=state.scoring,
            state_ref=state,
        )

    @classmethod
    def from_validation(
        cls,
        learner: "SurvivalLearner",
        val_result: "ValidationResult",
    ) -> "PlotContext":
        """Build a context for a single validation cohort.

        The fitted models and ``y_train`` are pulled from the *main*
        learner state — validation never refits, and the IPCW baseline
        distribution is fixed across cohorts.
        """
        state = learner._state
        fitted_models = {
            name: mr.model
            for name, mr in state.models.items()
            if getattr(mr, "model", None) is not None
        }

        return cls(
            name=val_result.name,
            source="validation",
            eval_result=val_result.evaluation,
            bootstrap_result=val_result.bootstrap,
            test_data=None,
            X_data=val_result.X_val,
            y_data=val_result.y_val,
            y_train=state.y_train,
            fitted_models=fitted_models,
            eval_times=state.user_eval_times,
            time_unit=state.time_unit,
            scoring=state.scoring,
            state_ref=state,
        )

    @classmethod
    def from_subpopulation(
        cls,
        learner: "SurvivalLearner",
        subpop_var: str,
    ) -> "PlotContext":
        """Build a context for a single subpopulation variable.

        ``eval_result`` / ``bootstrap_result`` are left ``None``: a
        :class:`SubpopulationResult` stores pre-aggregated metric frames
        (accessed via :class:`SubpopulationAxis`), not a full
        ``EvaluationResult``. Impls that need per-level data
        (calibration, ROC, survival) filter ``state_ref.X_test`` /
        ``state_ref.y_test`` by ``state_ref.subpopulations_test[subpop_var]``.
        """
        state = learner._state
        fitted_models = {
            name: mr.model
            for name, mr in state.models.items()
            if getattr(mr, "model", None) is not None
        }

        return cls(
            name=subpop_var,
            source="subpopulation",
            eval_result=None,
            bootstrap_result=None,
            test_data=None,
            X_data=state.X_test,
            y_data=state.y_test,
            y_train=state.y_train,
            fitted_models=fitted_models,
            eval_times=state.user_eval_times,
            time_unit=state.time_unit,
            scoring=state.scoring,
            state_ref=state,
        )


@dataclass(frozen=True)
class SubpopulationAxis:
    """Structured view over a :class:`SubpopulationResult` for grouped plotting.

    Thin wrapper that ``SubpopulationView`` passes into the
    shared ``_plot_metric_impl`` / ``_plot_metric_time_impl`` to switch them
    into grouped-forest / faceted-AUC(t) mode. Holds only the frames the
    grouped renderers actually consume.

    Attributes
    ----------
    group_variable : str
        Name of the subpopulation grouping variable (e.g. ``"menostat"``).
    levels : list of str
        Sorted subpopulation level names (e.g. ``["post", "pre"]``).
    metrics_long : DataFrame
        Per-``(model, subpopulation)`` point-estimate metrics. Matches the
        shape of ``SubpopulationResult.metrics``.
    bootstrap_summary : DataFrame, optional
        Bootstrap CI bands per ``(model, subpopulation)``. Matches
        ``SubpopulationResult.bootstrap_summary``. ``None`` if bootstrap
        wasn't run.
    auc_t : DataFrame, optional
        Per-``(model, subpopulation, eval_time)`` AUC values. Matches
        ``SubpopulationResult.auc_t``.
    auc_t_bootstrap : DataFrame, optional
        Per-iteration bootstrap AUC(t) used to draw ribbons. Matches
        ``SubpopulationResult.auc_t_bootstrap``.
    """

    group_variable: str
    levels: List[str]
    metrics_long: pd.DataFrame
    bootstrap_summary: Optional[pd.DataFrame] = None
    auc_t: Optional[pd.DataFrame] = None
    auc_t_bootstrap: Optional[pd.DataFrame] = None

    @classmethod
    def from_result(cls, sp_result: "SubpopulationResult") -> "SubpopulationAxis":
        """Construct from a stored :class:`SubpopulationResult`."""
        metrics = sp_result.metrics
        if (
            metrics is None
            or metrics.empty
            or "subpopulation" not in metrics.columns
        ):
            levels: List[str] = []
        else:
            levels = sorted(str(s) for s in metrics["subpopulation"].unique())

        return cls(
            group_variable=sp_result.group_variable,
            levels=levels,
            metrics_long=metrics,
            bootstrap_summary=sp_result.bootstrap_summary,
            auc_t=sp_result.auc_t,
            auc_t_bootstrap=sp_result.auc_t_bootstrap,
        )
