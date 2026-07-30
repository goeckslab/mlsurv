"""LearnerState - Centralized state container for SurvivalLearner."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

# Imported into module globals so typing.get_type_hints(LearnerState) can
# resolve the np.ndarray subpopulation annotations at runtime (all annotations
# here are deferred to strings by `from __future__ import annotations` above).
import numpy as np

if TYPE_CHECKING:
    from mlsurv.guidance import RecommendedModels
else:
    # Runtime alias so typing.get_type_hints(LearnerState) can resolve the
    # quoted "RecommendedModels" annotation without importing guidance (which
    # would pull constants+models into this foundational module's import
    # graph). External tooling sees Any; IDEs use the TYPE_CHECKING import.
    RecommendedModels = Any


@dataclass
class LearnerState:
    """All mutable state for a SurvivalLearner instance.

    Groups are organized by lifecycle phase. Every field has a default so
    ``LearnerState()`` is always valid (representing "just constructed,
    nothing configured yet").

    Notes
    -----
    This dataclass is **not frozen** — state is mutated throughout the
    learner lifecycle (setup, train, tune, evaluate, etc.).
    """

    # -- Input data (set in __init__) ------------------------------------
    X: Any = None
    y: Any = None
    presplit: bool = False

    # -- Data splits (set in setup or __init__ for pre-split) ------------
    X_train: Any = None
    X_test: Any = None
    y_train: Any = None
    y_test: Any = None

    # -- Setup config (set in setup) -------------------------------------
    is_setup: bool = False
    test_fraction: float = 0.3
    random_state: int = 42
    scoring: Any = None
    cv_ci: float = 0.95
    cv_folds: int = 5
    cv_repeats: int = 1
    cv_strategy: Any = "stratifiedkfold"
    n_jobs: int = 1
    device: Optional[str] = None
    experiment_name: Optional[str] = None
    time_col: Optional[str] = None
    event_col: Optional[str] = None
    check_leakage: bool = True
    ph_test_enabled: bool = True
    linearity_test_enabled: bool = True

    # -- Preprocessing (set in setup) ------------------------------------
    imputer: Any = None
    scaler: Any = None
    feature_selector: Any = None
    model_preprocessing: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # -- CV infrastructure (set in setup) --------------------------------
    cv_splitter: Any = None
    cv_splitter_single: Any = None
    stratify_array: Any = None
    groups: Any = None
    # True when the train/test holdout was split group-aware (keeps whole
    # groups together) because `groups` was supplied to setup().
    group_aware_holdout: bool = False
    # What the train/test split actually stratified on, as a display label:
    # "event", a column name, "none", "custom stratify array", or a composite
    # such as "event x stage" or "event x subpopulation(site)". Recorded by the
    # split methods AFTER any binning, augmentation, or rare-level fallback, so
    # it says what happened rather than what setup() was asked for. Reports read
    # it; the array itself is stratify_array above.
    stratify_spec: Any = "event"
    # The same, for the CV folds, which apply their own rare-level threshold and
    # see only the training rows. Equal to stratify_spec when both stages landed
    # in the same place (including when both backed out). None when the folds are
    # not stratified at all (cv_strategy is not stratifiedkfold), which reports
    # treat as "no CV row". "event" when nothing resolved, because _get_cv_folds
    # then falls back to the event indicator.
    cv_stratify_spec: Any = None
    # Resolved [(label, values), ...] behind a list-form stratify=, sliced to
    # the training rows. Kept so CV can rebuild the composite keys at its own
    # rare-level threshold. None for the scalar and array forms, for pre-split
    # data, and for a list under cv_strategy='timeseries' (never stratified).
    stratify_components: Any = None
    stratify_subpopulations: bool = True

    # -- Descriptive (set in setup) --------------------------------------
    univariate_cox: Any = None

    # -- Models & results (set during train/tune/evaluate) ---------------
    # Per-model derived state — the S(t|x) calibrator, importance_test()
    # tables, and rmst/logrank risk-group tables — rides on
    # each ModelResult in `models` (result.calibrator, .pimp_importance_test,
    # .bootstrap_importance_test, .rmst_test, .logrank_test), not in side dicts
    # keyed by name. It therefore dies with the model on overwrite for free.
    # The importance/test dict names are preserved as read-only properties below.
    models: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    benchmark_result: Any = None
    user_eval_times: Any = None  # Explicit eval_times from evaluate()/run(); None = auto
    # Additive "report horizons" registered via add_eval_times(). UNION-ed
    # (not replaced) with the effective base grid for FUTURE computations, then
    # clipped to the censoring-stable window. Kept separate from
    # user_eval_times: that is the *replacement* grid (clearable via
    # eval_times='auto'); this is *additive*. None = no extra horizons.
    user_extra_eval_times: Any = None  # np.ndarray | None

    # -- Paired comparisons (auto-computed or pre-computed) ----------------
    kang_comparison_result: Any = None

    # -- Statistical tests (computed via learner API) -----------------------
    # ph_test/linearity_test operate on training data (not a model's
    # predictions), so they stay singular. rmst_test/logrank_test build risk
    # groups from a specific model's risk scores, so they are per-model and
    # ride on the ModelResult; see the convenience properties
    # `rmst_test_results` / `logrank_test_results` below.
    ph_test_result: Any = None
    linearity_test_result: Any = None

    # -- Setup diagnostics (computed during setup) -------------------------
    model_recommendations: Optional["RecommendedModels"] = None

    # -- Subpopulation analysis (set in setup) ----------------------------------
    subpopulations: Optional[Dict[str, "np.ndarray"]] = None
    subpopulations_train: Optional[Dict[str, "np.ndarray"]] = None
    subpopulations_test: Optional[Dict[str, "np.ndarray"]] = None
    subpopulation_results: Dict[str, Any] = field(default_factory=dict)

    # -- Display labels (set in setup) ------------------------------------
    outcome_name: str = "event"
    time_unit: Optional[str] = None
    feature_groups: Any = None    # Dict[str,str], Sequence[str], or pd.Series
    group_colors: Any = None      # Optional[Dict[str, str]]
    categorical_groups: Any = None  # Optional[Dict[str, List[str]]]
    categorical_features: Any = None  # Optional[List[str]] — discrete single columns

    # -- Runtime flags ---------------------------------------------------
    # Dedup for the TimeSeriesSplit CV-CI MethodologicalWarning. The
    # warning describes the current split's CV fold structure, so it is
    # cleared in ``_reset_state_if_needed`` on ``setup()`` re-init.
    timeseries_ci_warned: bool = False
    epv_warned_at_setup: bool = False
    subpop_test_warned: bool = False

    # -- add_eval_times() render-warning dedup --------------------
    # ``True`` means the one-time SuggestionWarning that names registered
    # report horizons still lacking computed values/CIs (the "re-run
    # evaluate()/bootstrap()" hint) has already fired this session. Reset to
    # False whenever add_eval_times() changes the registered set (a new
    # horizon may now be missing) and in ``_reset_state_if_needed`` on
    # ``setup()`` re-init.
    extra_eval_times_render_warned: bool = False

    # -- Reference feature-overlap warning dedup -------------------
    # Set of ``frozenset(feature_overlap)`` signatures already warned this
    # session. Shared across ``validate()`` and ``benchmark()`` because the
    # leakage concern (a reference name colliding with a feature column) is
    # identical regardless of which site checks it. Cleared in
    # ``_reset_state_if_needed`` on ``setup()`` re-init.
    reference_overlap_warned: Set[frozenset] = field(default_factory=set)

    # -- FitFailureWarning cross-call dedup -----------------------
    # Set of ``(category_name, message)`` keys for warning copies already
    # surfaced this session. Named for its main job, deduplicating
    # FitFailureWarning, but the CV path stores a key for every warning it
    # re-emits, so other categories land here too. Shared across Optuna trials
    # (``BaseTuner._emit_trial_warnings``, bound in
    # ``_create_tuner_for_model``), CV training, the final full-train fit, and
    # calibration OOF refits so the user sees one line per distinct
    # numerical-instability message per learner, not one per call site.
    # Keys are always the *unsuffixed* ``(category_name, message)``: the
    # ``" (xN)"`` and ``" (N occurrences across K <scope>)"`` counts are
    # appended at emit time and must never enter a key, or the phases stop
    # matching each other. Cleared in ``_reset_state_if_needed`` on
    # ``setup()`` re-init.
    fit_failure_warned: Set[tuple] = field(default_factory=set)

    # -- IPCW truncation warning dedup ----------------------------
    # Keyed by cohort name ("main" for evaluate/bootstrap; the validation
    # cohort name for validate). The key matches the ``cohort_name`` arg
    # to ``_emit_ipcw_truncation_warning``. ``True`` means the
    # once-per-cohort MethodologicalWarning has already fired this session.
    # Cleared in ``_reset_state_if_needed`` on ``setup()`` re-init — the
    # truncation cap (``t_max``) derives from the old training split, so
    # every cohort's warning goes stale.
    ipcw_truncation_warned: Dict[str, bool] = field(default_factory=dict)

    # -- Heavy-censoring warning dedup ----------------------------
    # Keyed by cohort name ("main" for evaluate). ``True`` means the
    # once-per-cohort heavy-censoring MethodologicalWarning (Harrell's C is
    # upward-biased; prefer Uno's IPCW C) has already fired this session.
    # Cleared in ``_reset_state_if_needed`` on ``setup()`` re-init — the
    # censoring fraction derives from the old test split.
    heavy_censoring_warned: Dict[str, bool] = field(default_factory=dict)

    # -- IPCW support-violation flag ------------------------
    # Populated by evaluate()/bootstrap() when test follow-up or G_hat
    # violates Uno's concordance assumptions. Read by report sections.
    ipcw_support_flag: Optional[dict] = None

    # -- Artifact tracking --------------------------------------------------
    artifact_path: Optional[str] = None

    # -- Per-model derived-state views ------------------------------
    # The recalibration fit and the importance/risk-group test tables ride on
    # each ModelResult; these read-only properties reassemble the historical
    # sparse `{model_name: ...}` dicts on demand so every read site (reports,
    # exports, predict, CLI) keeps working unchanged. MappingProxyType makes
    # accidental writes raise — state changes only by (re)setting the field on
    # the ModelResult, never by mutating these views.

    # ``getattr(m, ...)`` mirrors the existing ``calibration_flagged`` guard in
    # ``_core.py``: report/test code occasionally stubs ``models`` with light
    # namespaces that omit these fields, and a real ``ModelResult`` always has
    # them, so the default-None lookup is harmless and keeps the views robust.

    @property
    def pimp_importance_test_results(self) -> MappingProxyType:
        """Sparse ``{model_name: pimp_dataframe}`` view."""
        return MappingProxyType({
            n: getattr(m, 'pimp_importance_test', None)
            for n, m in self.models.items()
            if getattr(m, 'pimp_importance_test', None) is not None
        })

    @property
    def bootstrap_importance_test_results(self) -> MappingProxyType:
        """Sparse ``{model_name: bootstrap_dataframe}`` view."""
        return MappingProxyType({
            n: getattr(m, 'bootstrap_importance_test', None)
            for n, m in self.models.items()
            if getattr(m, 'bootstrap_importance_test', None) is not None
        })

    @property
    def rmst_test_results(self) -> MappingProxyType:
        """Sparse ``{model_name: rmst_dataframe}`` view."""
        return MappingProxyType({
            n: getattr(m, 'rmst_test', None) for n, m in self.models.items()
            if getattr(m, 'rmst_test', None) is not None
        })

    @property
    def logrank_test_results(self) -> MappingProxyType:
        """Sparse ``{model_name: logrank_dataframe}`` view."""
        return MappingProxyType({
            n: getattr(m, 'logrank_test', None) for n, m in self.models.items()
            if getattr(m, 'logrank_test', None) is not None
        })

    def __setstate__(self, state):
        """Restore state, filling missing fields with defaults for forward compat."""
        self.__dict__.update(state)
        for f in dataclasses.fields(self):
            if f.name not in self.__dict__:
                if f.default is not dataclasses.MISSING:
                    self.__dict__[f.name] = f.default
                elif f.default_factory is not dataclasses.MISSING:
                    self.__dict__[f.name] = f.default_factory()
