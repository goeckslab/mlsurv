"""BaseTuner and tightly-coupled helper functions for Optuna tuning."""

import os
import threading
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna

from mlsurv._warnings import MethodologicalWarning, SuggestionWarning
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from .search_spaces import DEFAULT_SPACES, get_search_space
from .storage import DEFAULT_STORAGE_BASE
from ..evaluation import score_survival_model
from ..preprocessing.utils import get_default_imputer, get_default_scaler
from ..constants import DEFAULT_SCORING_METRIC
from ..logging_utils import get_logger
from ..utils import SurvivalArray

logger = get_logger("tuning")

# Optuna user_attr key holding the dataset fingerprint that a study was tuned
# on. Stamped when a study is first created; checked on resume to catch a study
# being reused on different data, split, or random_state.
_FINGERPRINT_KEY = "mlsurv_data_fingerprint"


def _default_experiment_name():
    """Generate a timestamp-based default experiment name (`exp_YYYYMMDD_HHMMSS`)."""
    return f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ============================================================================
# Parallelism Configuration
# ============================================================================

# Models whose fit() is not thread-safe for parallel Optuna trials.
# lifelines-based models use autograd (global tracing state) + scipy.optimize
# (BLAS/LAPACK); concurrent threads with near-singular Hessians segfault.
# CoxNet's Cython/C++ solver releases the GIL; under high thread contention
# with concurrent storage writes, Optuna's trial state management can hit an
# internal assertion ("Should not reach") in _run_trial().
THREAD_UNSAFE_TUNING_MODELS = frozenset({'weibull_aft', 'coxnet'})


def _compute_parallelism_split(n_jobs, n_trials, n_samples, model_name):
    """
    Compute how to split cores between trial-level and model-level parallelism.

    Returns (trial_n_jobs, model_n_jobs) where trial_n_jobs * model_n_jobs <= n_jobs.

    The heuristic balances:
    - Trial parallelism: Run multiple Optuna trials concurrently
    - Model parallelism: Let RSF use multiple cores per trial

    Parameters
    ----------
    n_jobs : int
        Total number of cores available. -1 means all cores.
    n_trials : int
        Number of Optuna trials to run.
    n_samples : int
        Number of samples in the dataset.
    model_name : str
        Model identifier ('rsf', 'gbsa', etc.).

    Returns
    -------
    tuple
        (trial_n_jobs, model_n_jobs) - how to split parallelism.

    Notes
    -----
    Heuristic thresholds based on benchmark analysis (16 cores, 10 trials):
    - n=500:  balanced (4,4) wins at 37s vs trial_only 49s vs model_only 40s
    - n=1000: balanced (4,4) wins at 72s vs trial_only 120s vs model_only 97s
    - n=2000: model_only (1,16) wins at 279s vs trial_only 560s vs balanced 456s

    Strategy:
    - LARGE_DATASET_THRESHOLD = 2000: Use model_only for n >= 2000
    - For n < 2000: Use balanced approach (n_jobs/4 trials, 4 cores each)
    - MANY_TRIALS_THRESHOLD = 20: With many trials, trial parallelism helps

    Examples
    --------
    >>> _compute_parallelism_split(1, 10, 500, 'rsf')
    (1, 1)  # n_jobs=1 means 1 core total

    >>> _compute_parallelism_split(16, 10, 2000, 'rsf')
    (1, 16)  # Large data: model_only

    >>> _compute_parallelism_split(16, 10, 1000, 'rsf')
    (4, 4)  # Medium data: balanced

    >>> _compute_parallelism_split(8, 10, 500, 'coxnet')
    (1, 1)  # CoxNet is thread-unsafe
    """
    # Resolve -1 to actual CPU count
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    # Thread-unsafe models must run trials sequentially
    if model_name in THREAD_UNSAFE_TUNING_MODELS:
        return (1, 1)

    # Only RSF and XGBoost support model-level parallelism
    if model_name not in ['rsf', 'xgbcox', 'xgbaft']:
        return (n_jobs, 1)

    # n_jobs=1 means user wants minimal resource usage
    if n_jobs == 1:
        return (1, 1)

    # Heuristic thresholds (from benchmark analysis - Jan 2026, 16-core)
    # LARGE_DATASET: n>=2000 -> model_only wins (RSF tree building dominates)
    # MANY_TRIALS: n_trials>=20 -> balanced approach (trial parallelism helps)
    # Run tests/benchmarks/bench_thresholds.py to re-validate on your hardware.
    LARGE_DATASET_THRESHOLD = 2000
    MANY_TRIALS_THRESHOLD = 20

    if n_samples >= LARGE_DATASET_THRESHOLD:
        # Large datasets: model_only is fastest (RSF tree building dominates)
        # Sequential trials with maximum model parallelism
        return (1, n_jobs)

    if n_trials >= MANY_TRIALS_THRESHOLD:
        # Many trials: trial parallelism becomes more valuable
        # Use balanced approach
        trial_n_jobs = max(2, n_jobs // 4)
        model_n_jobs = max(1, n_jobs // trial_n_jobs)
        return (trial_n_jobs, model_n_jobs)

    # Default for small/medium datasets with moderate trials: balanced approach
    # ~4 trials running with ~4 cores each performed best in benchmarks
    trial_n_jobs = max(2, min(n_jobs // 4, n_trials))
    model_n_jobs = max(1, n_jobs // trial_n_jobs)

    return (trial_n_jobs, model_n_jobs)


# ============================================================================
# Search Space Parsing Helpers
# ============================================================================

def _is_class_name(key):
    """Check if key looks like a class name (starts with uppercase)."""
    return isinstance(key, str) and len(key) > 0 and key[0].isupper()


def _parse_component_search_space(space):
    """
    Parse search space into class-specific and broadcast params.

    Class-specific params are identified by keys that start with uppercase
    (e.g., 'VIFSelector', 'TopKCoxSelector'). Broadcast params are regular
    parameter names that apply to all steps with that parameter.

    Parameters
    ----------
    space : dict
        Search space for a component, e.g.:
        {
            'threshold': (0.5, 0.95),           # Broadcast to all
            'VIFSelector': {'threshold': (5, 15)},  # Class-specific
            'TopKCoxSelector': {'k': (5, 20)},
        }

    Returns
    -------
    class_spaces : dict
        {'VIFSelector': {'threshold': (5, 15)}, ...}
    broadcast_params : dict
        {'threshold': (0.5, 0.95), ...}

    Examples
    --------
    >>> space = {
    ...     'k': (5, 20),
    ...     'VIFSelector': {'threshold': (5, 15)},
    ... }
    >>> class_spaces, broadcast = _parse_component_search_space(space)
    >>> class_spaces
    {'VIFSelector': {'threshold': (5, 15)}}
    >>> broadcast
    {'k': (5, 20)}
    """
    class_spaces = {}
    broadcast_params = {}

    for key, value in space.items():
        if _is_class_name(key):
            if isinstance(value, dict):
                class_spaces[key] = value
            else:
                warnings.warn(
                    f"Search space key '{key}' looks like a class name but its "
                    f"value is {type(value).__name__}, not dict. Treating as a "
                    f"broadcast parameter. Use a dict for class-specific tuning: "
                    f"{{'{key}': {{'param_name': (min, max)}}}}",
                    SuggestionWarning,
                    stacklevel=2,
                )
                broadcast_params[key] = value
        else:
            broadcast_params[key] = value

    return class_spaces, broadcast_params


def _get_step_class_mapping(component):
    """
    Map class names to step names for a component.

    For single transformers, returns the class name mapped to [None].
    For Pipelines, returns class names mapped to their step names.

    Parameters
    ----------
    component : transformer or Pipeline
        The component to analyze.

    Returns
    -------
    dict
        {'VIFSelector': ['sel_0'], 'TopKCoxSelector': ['sel_1']}
        For single transformer: {'ClassName': [None]}

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from mlsurv.preprocessing import VIFSelector, TopKCoxSelector
    >>> pipe = Pipeline([('sel_0', VIFSelector()), ('sel_1', TopKCoxSelector())])
    >>> _get_step_class_mapping(pipe)
    {'VIFSelector': ['sel_0'], 'TopKCoxSelector': ['sel_1']}

    >>> selector = TopKCoxSelector()
    >>> _get_step_class_mapping(selector)
    {'TopKCoxSelector': [None]}
    """
    if component is None:
        return {}

    if hasattr(component, 'steps'):
        # It's a Pipeline
        mapping = {}
        for step_name, step in component.steps:
            class_name = type(step).__name__
            mapping.setdefault(class_name, []).append(step_name)
        return mapping
    else:
        # Single transformer
        return {type(component).__name__: [None]}


class BaseTuner:
    """Base class for model-specific Optuna tuners."""

    DEFAULT_SPACE = {}
    _SUPPRESSED_WARNING_PATTERNS = ()  # Subclasses override; matched via substring

    def __init__(
        self,
        X,
        y: SurvivalArray,
        model_name=None,
        cv_folds=5,
        random_state=None,
        search_spaces=None,
        cv_splitter=None,
        scoring=DEFAULT_SCORING_METRIC,
        imputer=None,
        scaler=None,
        selector=None,
        search_space=None,  # Backward compat alias for search_spaces
        stratify_array=None,
        groups=None,
    ):
        """
        Initialize tuner.

        Parameters
        ----------
        X : DataFrame
            Feature matrix.
        y : SurvivalArray
            Survival outcome.
        model_name : str, optional
            Model identifier (used for auto-generating storage path).
        cv_folds : int, default=5
            Number of CV folds (ignored if cv_splitter provided).
        random_state : int, optional
            Random seed.
        search_spaces : dict, optional
            Search spaces for model and/or pipeline components. Format:
            {
                'model': {'n_estimators': (100, 500), ...},
                'feature_selector': {'k': (5, 20)},
                'imputer': {'n_neighbors': (3, 10)},
                'scaler': {}
            }
            For backward compatibility, also accepts flat dict (treated as model space).
        cv_splitter : sklearn CV splitter, optional
            Pre-configured CV splitter from SurvivalLearner.setup().
        scoring : str, default=DEFAULT_SCORING_METRIC
            Scoring metric for optimization: 'harrell', 'ipcw', 'brier',
            'integrated_brier', or 'ctd'.
        imputer : sklearn Transformer or None, optional
            Imputer instance from SurvivalLearner. None = skip imputation.
        scaler : sklearn Transformer or None, optional
            Scaler instance from SurvivalLearner. None = skip scaling.
        selector : sklearn Transformer or None, optional
            Feature selector instance from SurvivalLearner. None = no selection.
        search_space : dict, optional
            Deprecated alias for search_spaces. Use search_spaces instead.
        stratify_array : array-like or None, default=None
            Labels for stratified CV splits. If None, falls back to
            y['event'] (event indicator).
        groups : array-like or None, default=None
            Group labels for GroupKFold CV splits. Required when
            cv_splitter is a GroupKFold instance.
        """
        # Backward compatibility: search_space -> search_spaces
        if search_space is not None and search_spaces is None:
            search_spaces = search_space

        self.X = X
        self.y = y
        self.model_name = model_name
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.scoring = scoring
        self._stratify_labels = stratify_array if stratify_array is not None else self.y['event']
        self._groups = groups

        # Store component instances (None = skip that step)
        self._imputer = imputer
        self._scaler = scaler
        self._selector = selector
        self._trial_warnings = {}
        self._trial_warnings_lock = threading.Lock()
        # Optional cross-phase FitFailureWarning dedup set, bound to the
        # learner's ``LearnerState.fit_failure_warned`` by
        # ``_create_tuner_for_model``. ``None`` when the tuner is constructed
        # standalone (direct instantiation, unit tests), in which case
        # ``_emit_trial_warnings`` behaves exactly as it did before.
        self._seen_warning_keys = None
        # Per-trial failure messages (keyed by message string -> count), used to
        # surface the dominant cause when 100% of trials fail. Optuna
        # does not store trial exceptions on the FrozenTrial, so we capture them
        # ourselves via _objective_with_failure_capture.
        self._trial_failures = {}
        self._trial_failures_lock = threading.Lock()
        self._precomputed_folds = None
        self._dynamic_component_defs = None

        # Parse search_spaces
        self._search_spaces = self._parse_search_spaces(search_spaces)

        # Merge model search space with defaults (EPV-aware when possible)
        if self.model_name and self.model_name in DEFAULT_SPACES:
            n_samples = self.X.shape[0]
            n_features = self.X.shape[1]
            n_events = int(self.y['event'].sum())
            base_space = get_search_space(
                self.model_name,
                n_samples=n_samples, n_events=n_events, n_features=n_features,
            )
        else:
            base_space = self.DEFAULT_SPACE.copy()
        self.search_space = {**base_space, **self._search_spaces.get('model', {})}
        self._categorical_choices = {}

        # No-silent-data-loss guard: a user-provided model space may contain keys
        # this tuner never applies (e.g. CoxPH consumes no model params). Warn
        # naming them instead of dropping them silently. Runs in the parent
        # process (not a joblib worker), so the warning reaches the user.
        unconsumed = set(self._search_spaces.get('model', {})) - self._consumed_model_keys()
        if unconsumed:
            warnings.warn(
                f"{type(self).__name__} cannot tune model parameter(s) "
                f"{sorted(unconsumed)}; they will be ignored. Remove them from "
                f"search_spaces['model'], or move tunable preprocessing params "
                f"under a component space (e.g. "
                f"search_spaces={{'feature_selector': {{'k': (2, 4)}}}}).",
                SuggestionWarning,
                stacklevel=2,
            )

        # The selector's k is not tuned by default: a bare 'k' in a model space
        # belongs to the model, and TopKCoxSelector(k=N) / k='auto' stay fixed
        # while model params tune. Users opt into searching the feature count in
        # one of two ways: TopKCoxSelector(k='tune') (the default range is
        # injected below), or an explicit search_spaces={'feature_selector':
        # {'k': (...)}}.
        self._maybe_inject_tune_k_space()

        # Use provided CV splitter or create default
        if cv_splitter is not None:
            self._cv = cv_splitter
        else:
            self._cv = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=random_state,
            )

    def _consumed_model_keys(self) -> set:
        """Model search-space keys this tuner actually applies to the model.

        Standard tuners pass their whole resolved model space to the model
        constructor, so by default every key in ``search_space`` is consumed.
        Tuners that apply only a subset (or none) override this so the
        no-silent-data-loss guard can warn about the rest.
        """
        return set(self.search_space)

    @property
    def has_searchable_params(self) -> bool:
        """Whether this tuner has any parameters to search over.

        Counts only model keys the tuner will actually apply (see
        ``_consumed_model_keys``), so an unconsumed model space does not
        falsely report that tuning will happen.
        """
        return bool(self._consumed_model_keys()) or any(
            self._search_spaces.get(c) for c in ('feature_selector', 'imputer', 'scaler')
        )

    def _default_tune_k_range(self):
        """Clamped default (lo, hi) range for a ``k='tune'`` feature selector.

        Reads the reference range from ``PREPROCESSING_DEFAULTS`` (``(5, None)``,
        where ``None`` means ``n_features``) and clamps it to the actual feature
        count so ``suggest_int`` always gets ``1 <= lo <= hi``.
        """
        from .search_spaces import get_preprocessing_defaults

        lo, hi = get_preprocessing_defaults('feature_selector')['TopKCoxSelector']['k']
        n_features = self.X.shape[1]
        hi = n_features if hi is None else min(hi, n_features)
        lo = max(1, min(lo, hi))
        return (lo, hi)

    def _maybe_inject_tune_k_space(self):
        """Wire up ``TopKCoxSelector(k='tune')`` and nudge ``k='auto'`` users.

        When the configured selector opts into searching its feature count via
        ``k='tune'`` and the user gave no explicit ``k`` space, inject the
        clamped default range so Optuna searches it. When a topk selector is left
        at the default ``k='auto'`` with no ``k`` space, emit a one-time
        ``SuggestionWarning`` pointing at ``k='tune'`` (this also flags the
        silent "auto kept all features" case on high-event cohorts).
        """
        from ..preprocessing.selection import TopKCoxSelector

        selector = self._selector
        if selector is None:
            return

        # Collect the k values of any TopKCoxSelector step(s). Handles a bare
        # selector and a multi-step selector Pipeline (steps named sel_0, ...).
        if hasattr(selector, 'steps'):
            topk_ks = [step.k for _, step in selector.steps
                       if isinstance(step, TopKCoxSelector)]
            is_pipeline = True
        elif isinstance(selector, TopKCoxSelector):
            topk_ks = [selector.k]
            is_pipeline = False
        else:
            return
        if not topk_ks:
            return

        # Is a k space already specified (broadcast 'k' or class-specific)?
        space = self._search_spaces.get('feature_selector', {})
        class_space = space.get('TopKCoxSelector')
        user_set_k = 'k' in space or (
            isinstance(class_space, dict) and 'k' in class_space
        )

        if any(k == 'tune' for k in topk_ks):
            if not user_set_k:
                fs = self._search_spaces.setdefault('feature_selector', {})
                if is_pipeline:
                    # Class-specific so it maps to the TopK step(s) by class name.
                    # Rebuild the inner dict rather than mutating in place: under
                    # a broadcast component space, _parse_search_spaces only
                    # shallow-copies, so a user-provided 'TopKCoxSelector' dict
                    # can be shared across every model's tuner.
                    fs['TopKCoxSelector'] = {
                        **fs.get('TopKCoxSelector', {}),
                        'k': self._default_tune_k_range(),
                    }
                else:
                    fs['k'] = self._default_tune_k_range()
        elif not user_set_k and any(k == 'auto' for k in topk_ks):
            warnings.warn(
                "Tuning with a TopKCoxSelector left at k='auto' (Harrell m/15) "
                "does not search the number of features. Set k='tune' on the "
                "selector to let Optuna search it, or pass "
                "search_spaces={'feature_selector': {'k': (lo, hi)}}.",
                SuggestionWarning,
                stacklevel=2,
            )

    def _cv_split(self):
        """Generate CV splits, passing groups if cv is GroupKFold."""
        if isinstance(self._cv, GroupKFold):
            if self._groups is None:
                raise ValueError(
                    "groups must be provided when using GroupKFold. "
                    "Pass groups= to setup()."
                )
            return self._cv.split(self.X, self._stratify_labels, self._groups)
        return self._cv.split(self.X, self._stratify_labels)

    def _accumulate_trial_warnings(self, caught):
        """Deduplicate warnings from a trial and merge into instance dict.

        Thread-safe for ``trial_n_jobs > 1`` (dict access is locked).
        Note: ``warnings.catch_warnings`` itself uses process-global state
        and is not thread-safe per CPython docs, so warning capture is
        best-effort under parallel trials.

        Warnings are keyed by ``(category, message_text)`` — two warnings
        with the same category and string representation are counted as
        one occurrence.

        Parameters
        ----------
        caught : list of warnings.WarningMessage
            Warnings captured via ``warnings.catch_warnings(record=True)``.
        """
        for w in caught:
            msg = str(w.message)
            # Skip warnings matching suppressed patterns
            if any(pat in msg for pat in self._SUPPRESSED_WARNING_PATTERNS):
                continue
            key = (w.category, msg)
            with self._trial_warnings_lock:
                self._trial_warnings[key] = self._trial_warnings.get(key, 0) + 1

    def _emit_trial_warnings(self):
        """Re-emit each unique warning once, annotated with count, then clear.

        ``FitFailureWarning``s are additionally deduplicated against
        ``self._seen_warning_keys`` (the learner's
        ``LearnerState.fit_failure_warned``), so the same numerical-instability
        message does not surface a second time from the post-tune CV re-run or
        the final full-train fit.

        The key added to that set is the *unsuffixed* ``(category.__name__,
        msg)``. The ``" (xN)"`` suffix is presentation only: the CV path builds
        its keys from raw messages, so a suffixed key would never match and the
        dedup would silently do nothing. Same contract as
        ``_warnings._reemit_deduped_warnings``.

        Only ``FitFailureWarning`` takes part here. The set itself is not
        scoped to that category: the CV path stores a key for every warning it
        re-emits, so widening this check would suppress unrelated warnings such
        as CoxPH's ridge-penalty ``MethodologicalWarning``.
        """
        from .._warnings import FitFailureWarning

        with self._trial_warnings_lock:
            snapshot = dict(self._trial_warnings)
            self._trial_warnings.clear()
        seen = self._seen_warning_keys
        for (category, msg), count in snapshot.items():
            if seen is not None and issubclass(category, FitFailureWarning):
                key = (category.__name__, msg)
                if key in seen:
                    continue
                seen.add(key)
            suffix = f" (x{count})" if count > 1 else ""
            warnings.warn(f"{msg}{suffix}", category=category, stacklevel=2)

    def _record_trial_failure(self, message):
        """Record a trial failure message for later diagnostics.

        Thread-safe for ``trial_n_jobs > 1`` (dict access is locked). Messages
        are counted by string so the dominant failure can be surfaced if every
        trial fails (see ``_format_all_trials_failed_message``).
        """
        with self._trial_failures_lock:
            self._trial_failures[message] = self._trial_failures.get(message, 0) + 1

    def _objective_with_failure_capture(self, trial):
        """Run ``self.objective`` and capture its failure message before re-raising.

        Optuna's ``study.optimize(..., catch=(Exception,))`` swallows trial
        exceptions and does not store the message on the trial, so when every
        trial fails the only surviving signal is the opaque ``study.best_params``
        "Record does not exist." error. Two failure shapes reach here: a model
        construction error (the typical cause, e.g. a bad search-space key
        hitting ``RSFModel(**model_params)``) raises a plain exception that
        Optuna marks FAIL, while a fold-level failure inside ``_evaluate_model``
        is converted to ``optuna.TrialPruned`` (trial state PRUNED). We record
        the message in both cases and re-raise unchanged so Optuna's trial-state
        handling is identical to calling ``self.objective`` directly.
        ``MemoryError``/``RecursionError`` propagate without recording,
        matching the catch-and-prune contract used in ``_evaluate_model``
        (see CONTRIBUTING.md "Optuna catch-and-prune").  ``ImportError`` is
        recorded before it is re-raised: ``study.optimize`` runs with
        ``catch=(Exception,)``, so the re-raise never escapes, and the record
        is the only trace the trial leaves behind.
        """
        try:
            return self.objective(trial)
        except optuna.TrialPruned as e:
            self._record_trial_failure(str(e) or "Trial pruned: no valid CV scores")
            raise
        except (MemoryError, RecursionError):
            raise
        except ImportError as e:
            # Recorded, unlike the two above. A missing or unloadable optional
            # dependency fails every trial identically, and `catch=(Exception,)`
            # on study.optimize swallows the re-raise, so without this the
            # study ends with no recorded failures and the caller gets a bare
            # "Record does not exist." instead of the reason. Most often this
            # is xgboost failing to load for want of an OpenMP runtime.
            self._record_trial_failure(f"{type(e).__name__}: {e}")
            raise
        except Exception as e:
            self._record_trial_failure(f"{type(e).__name__}: {e}")
            raise

    def _format_all_trials_failed_message(self, study):
        """Build the error message for a study where no trial completed."""
        total = len(study.trials)
        display_name = self.model_name or "model"
        with self._trial_failures_lock:
            failures = dict(self._trial_failures)
        header = (
            f"All {total} tuning trials for model '{display_name}' failed; none "
            f"completed, so no best parameters could be selected."
        )
        guidance = (
            "This usually means the search space contains a parameter the model "
            "does not accept (a typo, a preprocessing parameter placed under "
            "'model', or a space copied from a different model). Check your "
            "search_spaces= against the model's parameters."
        )
        if not failures:
            return f"{header} {guidance}"
        dominant_msg, count = max(failures.items(), key=lambda kv: kv[1])
        return (
            f"{header} Most common failure ({count}/{total} trials):\n"
            f"  {dominant_msg}\n"
            f"{guidance}"
        )

    def _parse_search_spaces(self, search_spaces):
        """Parse search_spaces into component-keyed dict.

        A flat dict with no component keys is treated as the model search space.
        """
        if search_spaces is None:
            return {'model': {}}

        # Check if it's a flat dict (no component keys)
        # If any key is a known component name, treat as structured
        component_keys = {'model', 'feature_selector', 'imputer', 'scaler'}
        if not any(k in component_keys for k in search_spaces.keys()):
            # Flat dict - all keys belong to the model search space.
            # Bare names are NOT special-cased: the selector's k is tuned only
            # via the explicit {'feature_selector': {'k': ...}} form, so a model
            # may legitimately declare its own 'k' hyperparameter.
            return {'model': dict(search_spaces)}

        # Structured dict
        result = {k: dict(v) if v else {} for k, v in search_spaces.items()}
        result.setdefault('model', {})
        return result

    def _suggest_param(self, trial, name):
        """Suggest parameter value from ``self.search_space[name]``."""
        return self._suggest_value(trial, name, self.search_space[name])

    def _validate_fold_scores(self, scores):
        """Hook for subclass-specific post-loop score validation.

        Called after all CV folds complete but before aggregation.
        Subclasses can override to inspect/filter scores or raise
        ``optuna.TrialPruned`` based on score patterns.

        The base implementation prunes trials where **all** folds returned
        NaN (e.g., numerical instability in Breslow estimation).

        Parameters
        ----------
        scores : list of float
            Per-fold scores from the CV loop.

        Returns
        -------
        list of float
            Validated scores (default: pass-through).

        Notes
        -----
        If this method returns an empty list, the caller
        (``_evaluate_model``) will raise ``optuna.TrialPruned``.

        See Also
        --------
        CoxNetTuner._validate_fold_scores : Prunes trials where all folds
            returned c-index ~0.5 (all-zero CoxNet coefficients).
        """
        if scores and all(np.isnan(s) for s in scores):
            raise optuna.TrialPruned(
                "All CV folds returned NaN scores (possible numerical "
                "instability in survival function estimation)"
            )
        return scores

    def _score_model(self, estimator, X, y: SurvivalArray, y_train: SurvivalArray):
        """
        Score fitted model using configured metric.

        Parameters
        ----------
        estimator : fitted model or Pipeline
            Fitted model with predict() method.
        X : DataFrame
            Validation features.
        y : SurvivalArray
            Validation outcomes.
        y_train : SurvivalArray
            Training outcomes (for IPCW).

        Returns
        -------
        float
            Score value.
        """
        return score_survival_model(
            estimator, X, y,
            scoring=self.scoring,
            y_train=y_train,
        )

    def _suggest_value(self, trial, name, spec):
        """
        Suggest a single parameter value based on spec format.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object.
        name : str
            Parameter name for Optuna (used in study).
        spec : tuple or list
            Parameter specification:
            - (min, max): int range if both int, else float range
            - (min, max, 'log'): float range with log scale
            - [a, b, c]: categorical choices
            - (min, None): int range where None = n_features

        Returns
        -------
        value
            Suggested parameter value.
        """
        if isinstance(spec, list):
            return self._suggest_categorical(trial, name, spec)

        if not isinstance(spec, tuple):
            raise ValueError(f"Invalid search space format for '{name}': {spec}")

        if len(spec) == 3 and spec[2] == 'log':
            return trial.suggest_float(name, spec[0], spec[1], log=True)
        if isinstance(spec[0], float) or isinstance(spec[1], float):
            return trial.suggest_float(name, spec[0], spec[1])

        # Handle k with None as upper bound
        upper = spec[1] if spec[1] is not None else len(self.X.columns)
        return trial.suggest_int(name, spec[0], upper)

    @staticmethod
    def _is_optuna_primitive(value):
        """Return True if value is valid for Optuna categorical persistence."""
        return value is None or isinstance(value, (bool, int, float, str))

    def _suggest_categorical(self, trial, name, choices):
        """
        Suggest categorical parameter with support for non-primitive choices.

        Optuna persistence supports only primitive categorical values. For
        structured values (e.g. list layers like [16, 16]), we optimize over
        integer indices and decode back to the original object.
        """
        if all(self._is_optuna_primitive(choice) for choice in choices):
            return trial.suggest_categorical(name, tuple(choices))

        self._categorical_choices[name] = list(choices)
        idx = trial.suggest_int(name, 0, len(choices) - 1)
        return self._categorical_choices[name][idx]

    def _decode_best_params(self, best_params):
        """Decode index-encoded categorical params back to original values."""
        decoded = {}
        for name, value in best_params.items():
            if name in self._categorical_choices and isinstance(value, (int, np.integer)):
                choices = self._categorical_choices[name]
                idx = int(value)
                if 0 <= idx < len(choices):
                    decoded[name] = choices[idx]
                    continue
                logger.warning(
                    "Categorical index %d out of range for '%s' "
                    "(choices has %d entries). Returning raw index. "
                    "This may happen when resuming a study after "
                    "changing the search space.",
                    idx, name, len(choices),
                )
            decoded[name] = value
        return decoded

    def _get_component(self, component_name):
        """
        Get component by name.

        Parameters
        ----------
        component_name : str
            Component name: 'imputer', 'scaler', 'feature_selector'.

        Returns
        -------
        transformer or None
            The component instance.
        """
        mapping = {
            'imputer': self._imputer,
            'scaler': self._scaler,
            'feature_selector': self._selector,
        }
        return mapping.get(component_name)

    def _suggest_component_param(self, trial, component_name, step_name, param_name, spec):
        """Suggest a single param value with proper naming for flat or nested keys."""
        if step_name is None:
            optuna_name = f'{component_name}__{param_name}'
            key = param_name
        else:
            optuna_name = f'{component_name}__{step_name}__{param_name}'
            key = f'{step_name}__{param_name}'
        return key, self._suggest_value(trial, optuna_name, spec)

    def _apply_broadcast_params(self, trial, component_name, component, broadcast_params, skip_classes):
        """Apply broadcast params to single transformers or pipeline steps."""
        params = {}
        is_pipeline = component is not None and hasattr(component, 'steps')

        for param_name, spec in broadcast_params.items():
            if not is_pipeline:
                key, val = self._suggest_component_param(trial, component_name, None, param_name, spec)
                params[key] = val
            else:
                for step_name, step in component.steps:
                    if type(step).__name__ in skip_classes:
                        continue
                    if hasattr(step, 'get_params') and param_name in step.get_params():
                        key, val = self._suggest_component_param(trial, component_name, step_name, param_name, spec)
                        params[key] = val
        return params

    def _apply_class_specific_params(self, trial, component_name, class_spaces, class_mapping):
        """Apply class-specific params (take precedence over broadcast)."""
        params = {}
        for class_name, class_params in class_spaces.items():
            for step_name in class_mapping.get(class_name, []):
                for param_name, spec in class_params.items():
                    key, val = self._suggest_component_param(trial, component_name, step_name, param_name, spec)
                    params[key] = val
        return params

    def _suggest_component_params(self, trial, component_name):
        """
        Suggest params for a component, handling class-specific spaces.

        Supports three search space formats:
        1. Broadcast params: {'k': (5, 20)} applies to all steps with 'k'
        2. Class-specific: {'VIFSelector': {'threshold': ...}} applies to VIF only
        3. Mixed: Both broadcast and class-specific in same space

        Class-specific params take precedence over broadcast params.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object.
        component_name : str
            Component name: 'model', 'feature_selector', 'imputer', 'scaler'.

        Returns
        -------
        dict
            Suggested parameter values. For single transformers, returns flat
            params like {'k': 5}. For Pipelines, returns nested params like
            {'sel_0__threshold': 10, 'sel_1__k': 15}.

        Examples
        --------
        # Single component
        >>> params = tuner._suggest_component_params(trial, 'feature_selector')
        >>> params  # {'k': 10}

        # Multi-step Pipeline with class-specific tuning
        >>> params = tuner._suggest_component_params(trial, 'feature_selector')
        >>> params  # {'sel_0__threshold': 8.5, 'sel_1__k': 12}
        """
        search_space = self._search_spaces.get(component_name, {})
        if not search_space:
            return {}

        component = self._get_component(component_name)
        class_spaces, broadcast_params = _parse_component_search_space(search_space)
        class_mapping = _get_step_class_mapping(component)

        # Broadcast first, then class-specific (which override broadcast keys)
        params = self._apply_broadcast_params(
            trial, component_name, component, broadcast_params, set(class_spaces.keys())
        )
        params.update(self._apply_class_specific_params(
            trial, component_name, class_spaces, class_mapping
        ))
        return params

    def _precompute_static_prefix(self):
        """Pre-compute the static (non-tuned) prefix of the pipeline for each
        CV fold, so that tuning trials only fit the dynamic suffix.

        Walks the component definitions in order (imputer → scaler → selector)
        and identifies the longest prefix whose search spaces are empty (i.e.,
        parameters are fixed across trials).  For each CV fold, it clones those
        components, runs ``fit_transform`` on the training split and
        ``transform`` on the validation split, then stores the results.

        Sets ``self._precomputed_folds`` to a list of per-fold tuples and
        ``self._dynamic_component_defs`` to the remaining component
        definitions that still need per-trial fitting.  If there is no static
        prefix (the first component is tuned), both are left as ``None`` and
        ``_evaluate_model`` falls back to the original full-pipeline path.
        """
        from sklearn.base import clone

        has_components = (
            self._imputer is not None
            or self._scaler is not None
            or self._selector is not None
        )

        if has_components:
            component_defs = [
                ('imputer', self._imputer, 'imputer'),
                ('scaler', self._scaler, 'scaler'),
                ('selector', self._selector, 'feature_selector'),
            ]
        else:
            # Backward-compat path: default imputer + scaler, always static
            component_defs = [
                ('imputer', get_default_imputer(), 'imputer'),
                ('scaler', get_default_scaler(), 'scaler'),
            ]

        # Find the split point: first component with a non-empty search space
        static_defs = []
        dynamic_defs = []
        found_dynamic = False
        for step_name, component, space_key in component_defs:
            if component is None:
                continue
            if found_dynamic or self._search_spaces.get(space_key):
                found_dynamic = True
                dynamic_defs.append((step_name, component, space_key))
            else:
                static_defs.append((step_name, component))

        if not static_defs:
            # Nothing to pre-compute (first component is tuned)
            self._precomputed_folds = None
            self._dynamic_component_defs = None
            return

        # Pre-compute the static prefix for each CV fold
        precomputed = []
        for train_idx, val_idx in self._cv_split():
            X_train = self.X.iloc[train_idx]
            X_val = self.X.iloc[val_idx]
            y_train = self.y[train_idx]
            y_val = self.y[val_idx]

            X_tr = X_train
            X_v = X_val
            for _step_name, component in static_defs:
                fitted = clone(component)
                if hasattr(fitted, 'set_output'):
                    fitted.set_output(transform='pandas')
                X_tr = fitted.fit_transform(X_tr, y_train)
                X_v = fitted.transform(X_v)

            precomputed.append((X_tr, X_v, y_train, y_val))

        self._precomputed_folds = precomputed
        self._dynamic_component_defs = dynamic_defs

    def _build_tuning_pipeline(self, trial, model):
        """Build pipeline with tuned component params.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object.
        model : BaseSurvivalModel
            Model instance to include in pipeline.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Pipeline with configured components.
        """
        from sklearn.base import clone

        steps = []
        component_defs = [
            ('imputer', self._imputer, 'imputer'),
            ('scaler', self._scaler, 'scaler'),
            ('selector', self._selector, 'feature_selector'),
        ]
        for step_name, component, space_key in component_defs:
            if component is None:
                continue
            cloned = clone(component)
            params = self._suggest_component_params(trial, space_key)
            if params:
                cloned.set_params(**params)
            steps.append((step_name, cloned))

        steps.append(('model', model))
        pipeline = Pipeline(steps)
        pipeline.set_output(transform="pandas")
        return pipeline

    def _build_dynamic_pipeline(self, trial, model):
        """Build a pipeline containing only the dynamic (tuned) suffix
        components plus the model.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object.
        model : BaseSurvivalModel
            Model instance to include in pipeline.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Pipeline with only the dynamic components and model.
        """
        from sklearn.base import clone

        steps = []
        for step_name, component, space_key in self._dynamic_component_defs:
            cloned = clone(component)
            params = self._suggest_component_params(trial, space_key)
            if params:
                cloned.set_params(**params)
            steps.append((step_name, cloned))

        steps.append(('model', model))
        pipeline = Pipeline(steps)
        pipeline.set_output(transform="pandas")
        return pipeline

    def _evaluate_model(self, trial, model):
        """
        Evaluate a model using cross-validation.

        Builds a tuning pipeline, runs cross-validation, and returns the
        mean score. Subclass-specific post-loop validation is handled via
        the ``_validate_fold_scores`` hook.

        When ``_precomputed_folds`` is populated (by ``_precompute_static_prefix``),
        uses pre-transformed fold data and only fits the dynamic suffix pipeline.
        Otherwise falls back to the full pipeline path.

        Parameters
        ----------
        trial : optuna.Trial
            The current Optuna trial.
        model : BaseSurvivalModel
            The model instance to evaluate.

        Returns
        -------
        float
            Mean cross-validation score.

        Raises
        ------
        optuna.TrialPruned
            If any CV fold fails to fit/score, if ``_validate_fold_scores``
            raises it, or if no valid scores remain after validation.
        MemoryError, RecursionError
            Re-raised directly if encountered during fold execution.
        """
        if self._precomputed_folds is not None:
            return self._evaluate_model_precomputed(trial, model)

        if self._imputer is not None or self._scaler is not None or self._selector is not None:
            # Use new component-based pipeline
            pipeline = self._build_tuning_pipeline(trial, model)
        else:
            # Backward compatibility: use factory defaults (no auto selector)
            pipeline = Pipeline([
                ('imputer', get_default_imputer()),
                ('scaler', get_default_scaler()),
                ('model', model),
            ])
            pipeline.set_output(transform="pandas")

        scores = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from .._warnings import IterativeWarning
            warnings.filterwarnings("once", category=IterativeWarning)
            # lifelines pandas-4.0 keyword-only deprecations are not
            # user-actionable from mlsurv.  ``simplefilter('always')``
            # above wipes the import-time filter; re-add it for this worker.
            warnings.filterwarnings(
                "ignore",
                message=r"Starting with pandas version 4\.0",
                category=Warning,
                module=r"lifelines\..*",
            )
            try:
                for train_idx, val_idx in self._cv_split():
                    X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                    y_train, y_val = self.y[train_idx], self.y[val_idx]

                    try:
                        pipeline.fit(X_train, y_train)
                        scores.append(self._score_model(pipeline, X_val, y_val, y_train))
                    except optuna.TrialPruned:
                        raise
                    except (MemoryError, RecursionError, ImportError):
                        raise
                    except Exception as e:
                        logger.debug("Fit/score failed in _evaluate_model, pruning trial: %s: %s", type(e).__name__, e)
                        raise optuna.TrialPruned(f"Fold fit/score failed: {type(e).__name__}: {e}")
            finally:
                # Preserve warnings from successful folds even if a later fold raised
                self._accumulate_trial_warnings(caught)

        scores = self._validate_fold_scores(scores)
        if not scores:
            raise optuna.TrialPruned()
        return np.nanmean(scores)

    def _evaluate_model_precomputed(self, trial, model):
        """Evaluate using pre-computed fold data from the static prefix.

        Same contract as ``_evaluate_model`` but skips the static prefix
        fit/transform — uses cached transformed arrays instead.
        """
        scores = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from .._warnings import IterativeWarning
            warnings.filterwarnings("once", category=IterativeWarning)
            # lifelines pandas-4.0 keyword-only deprecations are not
            # user-actionable from mlsurv.  ``simplefilter('always')``
            # above wipes the import-time filter; re-add it for this worker.
            warnings.filterwarnings(
                "ignore",
                message=r"Starting with pandas version 4\.0",
                category=Warning,
                module=r"lifelines\..*",
            )
            try:
                for X_train_pre, X_val_pre, y_train, y_val in self._precomputed_folds:
                    try:
                        pipeline = self._build_dynamic_pipeline(trial, model)
                        pipeline.fit(X_train_pre, y_train)
                        scores.append(self._score_model(
                            pipeline, X_val_pre, y_val, y_train,
                        ))
                    except optuna.TrialPruned:
                        raise
                    except (MemoryError, RecursionError, ImportError):
                        raise
                    except Exception as e:
                        logger.debug(
                            "Fit/score failed in _evaluate_model_precomputed"
                            ", pruning trial: %s: %s",
                            type(e).__name__, e,
                        )
                        raise optuna.TrialPruned(
                            f"Fold fit/score failed: {type(e).__name__}: {e}"
                        )
            finally:
                self._accumulate_trial_warnings(caught)

        scores = self._validate_fold_scores(scores)
        if not scores:
            raise optuna.TrialPruned()
        return np.nanmean(scores)

    def _get_warm_start_params(self):
        """Return warm-start trial parameters, or None to skip.

        Subclasses can override to enqueue a dataset-aware initial trial
        based on auto-defaults. The returned dict must use the same
        parameter names as the search space (e.g. ``n_layers``, ``n_nodes``).

        Returns
        -------
        dict or None
            Trial parameters to enqueue, or None to skip warm-start.
        """
        return None

    def _clamp_warm_start(self, params):
        """Validate and clamp warm-start params to the resolved search space.

        - Drops keys not in ``self.search_space`` (logs warning).
        - Clamps numeric values to their search space bounds.
        - Returns None if no valid params remain.
        """
        unknown = set(params) - set(self.search_space)
        if unknown:
            logger.warning(
                "Warm-start params %s not in search space; dropping them.",
                unknown,
            )
            params = {k: v for k, v in params.items() if k not in unknown}
        if not params:
            return None

        clamped = {}
        for name, value in params.items():
            spec = self.search_space[name]
            if isinstance(spec, tuple) and len(spec) >= 2:
                lo, hi = spec[0], spec[1]
                if hi is None:
                    hi = len(self.X.columns)
                if isinstance(value, (int, float)):
                    clamped[name] = type(value)(max(lo, min(hi, value)))
                else:
                    clamped[name] = value
            elif isinstance(spec, list):
                # Categorical spec: snap to nearest valid choice
                if value in spec:
                    clamped[name] = value
                else:
                    numeric = [c for c in spec if isinstance(c, (int, float))]
                    if numeric and isinstance(value, (int, float)):
                        clamped[name] = min(numeric, key=lambda c: abs(c - value))
                    else:
                        clamped[name] = spec[0]
                    logger.info(
                        "Warm-start param '%s' value %r not in categorical "
                        "choices %r; snapped to %r.",
                        name, value, spec, clamped[name],
                    )
            else:
                clamped[name] = value
        return clamped

    def objective(self, trial):
        """Define in subclass."""
        raise NotImplementedError

    def _resolve_study_storage(self, storage, experiment_name, study_name, resume):
        """Resolve storage URL, study name, and handle existing study cleanup.

        Returns
        -------
        tuple
            (storage, study_name) with resolved values.
        """
        # Auto-generate storage path if not provided and model_name is known
        if storage is None and self.model_name is not None:
            full_dir = DEFAULT_STORAGE_BASE / experiment_name
            full_dir.mkdir(parents=True, exist_ok=True)
            storage_path = full_dir / f'{self.model_name}_study.db'
            storage = f'sqlite:///{storage_path}'

        # Auto-generate study name with experiment namespace if not provided
        if study_name is None and self.model_name is not None:
            study_name = f'{experiment_name}/{self.model_name}_tuning'

        # If not resuming, delete existing study to start fresh (use resume=True to keep)
        if not resume and storage is not None and study_name is not None:
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
            except KeyError:
                logger.debug("Study %r not found in storage (expected for new studies)", study_name)

        return storage, study_name

    def _resolve_n_trials_and_callbacks(self, n_trials):
        """Resolve n_trials value and create callbacks for 'auto' mode.

        Returns
        -------
        tuple
            (actual_n_trials, callbacks) with resolved values.
        """
        callbacks = []
        actual_n_trials = n_trials

        if n_trials == 'auto':
            try:
                from optuna.terminator import Terminator, TerminatorCallback
                terminator = Terminator()
                callbacks.append(TerminatorCallback(terminator))
                actual_n_trials = 100  # Upper bound
            except ImportError:
                actual_n_trials = 50
                logger.info(
                    "optuna.terminator not available; falling back to "
                    "%d trials (upgrade Optuna for convergence-based stopping)",
                    actual_n_trials,
                )

        return actual_n_trials, callbacks

    def _compute_data_fingerprint(self):
        """Return a deterministic hash of this tuner's training data and seed.

        Used to detect resuming an Optuna study that was tuned on different
        data, a different train/test split, or a different ``random_state``.
        The one hash per ``tune()`` call is negligible next to tuning.

        Returns ``None`` (after a soft warning) if the hash cannot be computed,
        so this advisory check never becomes a hard tuning blocker.
        """
        import joblib  # hard dependency (loky/parallel backend)
        try:
            return joblib.hash((
                tuple(self.X.columns),   # column names + order
                self.X.to_numpy(),       # train-split feature values
                self.y,                  # structured event/time array
                self.random_state,
            ))
        except (AttributeError, TypeError, ValueError) as exc:
            warnings.warn(
                f"Could not compute a dataset fingerprint ({exc}); cannot "
                f"verify that a resumed study was tuned on the current data. "
                f"Tuning proceeds.",
                SuggestionWarning, stacklevel=3,
            )
            return None

    def _verify_study_fingerprint(self, study, resume):
        """Stamp a fresh study's dataset fingerprint, or verify a resumed one.

        A fresh study (no prior trials) gets the current fingerprint stamped
        into its ``user_attrs``. A resumed study with prior trials is checked:

        - fingerprint absent (external or pre-fingerprint study) -> soft
          ``SuggestionWarning`` ("cannot verify"), proceed.
        - fingerprint present and mismatched -> loud ``MethodologicalWarning``,
          proceed.

        All paths proceed; the final model is re-CV'd on the current data
        regardless, so test metrics stay valid.
        """
        current = self._compute_data_fingerprint()
        if current is None:
            # Fingerprint unavailable (already warned). Don't stamp a None (it
            # would read back as "absent") and don't compare against it.
            return
        stored = study.user_attrs.get(_FINGERPRINT_KEY)
        has_trials = len(study.trials) > 0

        if stored is None:
            if resume and has_trials:
                warnings.warn(
                    f"Resuming Optuna study {study.study_name!r}, which has no "
                    f"mlsurv dataset fingerprint (external or older study). "
                    f"Cannot verify it was tuned on the current "
                    f"data/split/random_state; its best_params are adopted "
                    f"as-is.",
                    SuggestionWarning, stacklevel=3,
                )
            else:
                # Fresh study (resume=False deletes any prior; or empty new one).
                study.set_user_attr(_FINGERPRINT_KEY, current)
            return

        if stored != current:
            warnings.warn(
                f"Resuming Optuna study {study.study_name!r} that was tuned on "
                f"different data, a different train/test split, or a different "
                f"random_state than the current learner. Its best_params were "
                f"selected by scores computed on data this learner never saw. "
                f"Pass resume=False or a fresh experiment_name to retune. (The "
                f"final model is still re-CV'd on the current data.)",
                MethodologicalWarning, stacklevel=3,
            )
        # Matching fingerprint -> silent normal resume; no re-stamp needed.

    def _handle_zero_trials(self, study, study_name, resume):
        """Handle n_trials=0: return empty params or load existing study.

        Returns best_params dict. Raises ValueError if resume has no trials.
        """
        if not resume:
            return {}
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            raise ValueError(
                f"n_trials=0 but no completed trials found in study "
                f"'{study_name}'. Run tune() with n_trials > 0 first."
            )
        display_name = self.model_name or "model"
        print(f"  Loading existing study for {display_name} "
              f"({len(completed)} completed trials)...")
        return self._decode_best_params(study.best_params)

    def _enqueue_warm_start(self, study, resume):
        """Enqueue a dataset-aware warm-start trial if not resuming."""
        if resume:
            return
        try:
            warm_params = self._get_warm_start_params()
            if warm_params is not None:
                warm_params = self._clamp_warm_start(warm_params)
                if warm_params is not None:
                    study.enqueue_trial(warm_params)
        except (MemoryError, RecursionError, ImportError):
            raise
        except Exception:
            logger.warning(
                "Warm-start param computation failed; proceeding "
                "with random initialization.",
                exc_info=True,
            )

    def tune(self, n_trials='auto', timeout=None, storage=None, study_name=None,
             experiment_name=None, resume=False, n_jobs=1, optuna_verbose=False,
             show_progress_bar=True):
        """
        Run Optuna optimization.

        Parameters
        ----------
        n_trials : int or 'auto'
            Maximum number of Optuna trials.
            - 'auto': Uses Terminator for automatic convergence detection.
            - 0: Skip tuning and return empty params. With ``resume=True``,
              loads from an existing study (requires at least one completed
              trial).
            - int: Fixed number of trials.
        timeout : float, optional
            Maximum optimization time in seconds.
        storage : str, optional
            Optuna storage URL. If not provided, auto-creates SQLite storage
            at ./optuna_studies/{experiment_name}/{model_name}_study.db.
            When storage is auto-generated (default) and ``random_state`` is
            set, the TPESampler is seeded for reproducible trial suggestions.
            When storage is explicitly provided, the sampler is left unseeded
            for distributed tuning safety.
        study_name : str, optional
            Name for the Optuna study. Defaults to '{experiment_name}/{model_name}_tuning'.
        experiment_name : str or Path, optional
            Experiment name for organizing studies. Useful for separating
            experiments on different datasets. If not provided, auto-generates
            a timestamped name (e.g., 'exp_20260122_181830').
        resume : bool, default=False
            If True, continue from existing study trials. If False (default),
            silently deletes any existing study before starting fresh.
        n_jobs : int, default=1
            Number of parallel jobs for Optuna trial optimization.
            - 1: Sequential execution (default)
            - -1: Use all available CPU cores
            - n > 1: Use n CPU cores
            Note: Parallel trials require thread-safe objective functions.
            SQLite storage is thread-safe for parallel trials.
            Full reproducibility requires ``n_jobs=1``. With ``n_jobs > 1``,
            minor timing-based variation is possible even when seeded.
        optuna_verbose : bool, default=False
            If True, show trial-by-trial Optuna output (INFO level).
            If False, suppress Optuna logging to WARNING level only.
        show_progress_bar : bool, default=True
            Show the Optuna trial progress bar during optimization.

        Returns
        -------
        tuple
            (best_params, study, experiment_name_used)

        Raises
        ------
        ValueError
            If no trial completed and every trial that ran failed,
            indicating a misconfigured search space.
        MemoryError
            Re-raised if the computation exhausts available memory; resource
            errors are never swallowed by the broad failure handler.
        RecursionError
            Re-raised if the computation exceeds the recursion limit; resource
            errors are never swallowed by the broad failure handler.
        """
        # Resolve experiment_name
        if experiment_name is None:
            experiment_name = _default_experiment_name()
        experiment_name = Path(experiment_name)

        # Capture before _resolve_study_storage overwrites None → sqlite URL
        storage_is_local = storage is None

        storage, study_name = self._resolve_study_storage(
            storage, experiment_name, study_name, resume
        )

        # Set Optuna logging verbosity before create_study() to suppress
        # the "A new study created" INFO log when optuna_verbose=False.
        optuna.logging.set_verbosity(
            optuna.logging.INFO if optuna_verbose else optuna.logging.WARNING
        )

        # Seed sampler for reproducibility in local (auto-SQLite) mode only.
        # For distributed tuning (user-provided storage), leave unseeded to
        # avoid duplicate trial suggestions across workers.
        sampler = None
        if storage_is_local and self.random_state is not None:
            sampler = optuna.samplers.TPESampler(seed=self.random_state)

        study = optuna.create_study(
            direction='maximize',
            storage=storage,
            sampler=sampler,
            study_name=study_name,
            load_if_exists=resume,
        )

        # Stamp the dataset fingerprint on a fresh study, or verify it on a
        # resumed one, so reusing a study on different data/split/seed is not
        # silent. Covers both the normal path and the n_trials=0 path below.
        self._verify_study_fingerprint(study, resume)

        # Early return for n_trials=0: skip tuning or load existing study
        if n_trials == 0:
            best_params = self._handle_zero_trials(study, study_name, resume)
            return best_params, study, experiment_name

        # If resuming an existing study with prior trials, replace the seeded
        # sampler with an unseeded default to avoid replaying the startup
        # random sequence and proposing duplicate trial parameters.
        if resume and sampler is not None and len(study.trials) > 0:
            study.sampler = optuna.samplers.TPESampler()

        # Warm-start: enqueue a dataset-aware initial trial so the first
        # trial evaluates a competitive configuration instead of random.
        self._enqueue_warm_start(study, resume)

        actual_n_trials, callbacks = self._resolve_n_trials_and_callbacks(n_trials)

        # Compute parallelism split between trial-level and model-level
        trial_n_jobs, model_n_jobs = _compute_parallelism_split(
            n_jobs=n_jobs,
            n_trials=actual_n_trials,
            n_samples=len(self.X),
            model_name=self.model_name,
        )

        # Store for use in the objective function (model-level parallelism is
        # read back via getattr in the per-model tuners).
        self._model_n_jobs = model_n_jobs

        display_name = self.model_name or "model"
        if show_progress_bar:
            print(f"  Tuning {display_name} ({actual_n_trials} trials)...")
        with self._trial_warnings_lock:
            self._trial_warnings.clear()
        with self._trial_failures_lock:
            self._trial_failures.clear()
        # Multi-threaded Optuna trials (trial_n_jobs > 1) race on the
        # process-global warnings module state used by catch_warnings inside
        # _evaluate_model / _evaluate_model_precomputed.  When threads enter
        # and exit catch_warnings concurrently, the saved/restored values can
        # end up pointing at a leaked trial log's ``list.append`` instead of
        # the caller's (e.g. user-installed catch_warnings(record=True)).
        # Snapshot before optimize() and restore in the finally block so any
        # outer warning capture (including downstream _warn_poor_calibration
        # in evaluate()) reaches the caller, not a dead trial log.
        # Per-module ``__warningregistry__`` is intentionally NOT snapshotted:
        # under default warning actions, registry entries left by trial
        # threads could still suppress duplicate emissions in the parent.
        # The poor-calibration warning fires under an "always"
        # filter installed by ``catch_warnings(record=True)`` (and by the
        # regression test), which bypasses the registry.  File a follow-up
        # if a default-filter caller is ever affected.
        _saved_showwarnmsg_impl = warnings._showwarnmsg_impl
        _saved_showwarning = warnings.showwarning
        _saved_filters = warnings.filters[:]
        try:
            self._precompute_static_prefix()
            study.optimize(
                self._objective_with_failure_capture,
                n_trials=actual_n_trials,
                timeout=timeout,
                callbacks=callbacks,
                show_progress_bar=show_progress_bar,
                n_jobs=trial_n_jobs,
                catch=(Exception,),
            )
        finally:
            self._precomputed_folds = None
            self._dynamic_component_defs = None
            warnings._showwarnmsg_impl = _saved_showwarnmsg_impl
            warnings.showwarning = _saved_showwarning
            warnings.filters[:] = _saved_filters
            try:
                self._emit_trial_warnings()
            except (MemoryError, RecursionError):
                raise
            except Exception:
                logger.debug(
                    "_emit_trial_warnings raised; suppressed to "
                    "preserve original exception"
                )

        # If no trial completed *and* every trial that ran raised, study.best_params
        # would raise a bare "Record does not exist." Surface the dominant trial
        # failure instead. We require recorded failures so this only fires
        # for genuine all-trials-failed runs: a resume that carried prior completed
        # trials (>=1 COMPLETE) skips it, and a run where no trial actually executed
        # (e.g. timeout before the first trial finished) has no recorded failures,
        # so it falls through to study.best_params and behaves exactly as before
        # rather than mislabeling a timeout as a bad search space.
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            with self._trial_failures_lock:
                had_failures = bool(self._trial_failures)
            if had_failures:
                raise ValueError(self._format_all_trials_failed_message(study))

        return self._decode_best_params(study.best_params), study, experiment_name
