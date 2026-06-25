"""SetupMixin - Data splitting, CV configuration, and preprocessing setup."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING, Union

import numpy as np
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    GroupKFold,
    GroupShuffleSplit,
    RepeatedStratifiedKFold,
    RepeatedKFold,
    TimeSeriesSplit,
)

from .._iter_warn import IterativeWarningCoalescer
from .._shared import try_verbose_print
from .._warnings import (
    InternalWarning,
    MethodologicalWarning,
    SmallSubpopulationWarning,
    SuggestionWarning,
)
from ..preprocessing.utils import (
    get_default_imputer,
    get_default_scaler,
)
from ..preprocessing.pipelines import (
    get_imputer,
    get_scaler,
    get_selector,
    warn_preprocessing_order,
)
from ..constants import (
    DEFAULT_SCORING_METRIC, MIN_SAMPLE_SIZE_WARNING, MIN_EVENT_RATE, MAX_EVENT_RATE,
    MISSINGNESS_LEAKAGE_P_THRESHOLD, MISSINGNESS_LEAKAGE_V_THRESHOLD,
)
from ..evaluation import compute_univariate_cox
from ..logging_utils import get_logger
from ..tuning import auto_cv_folds
from ..utils import validate_survival_data, check_truncation_needed
from ._format import (
    _print_param_box,
    _format_pipeline_str,
    _format_per_model_preprocessing,
    _format_data_str,
    _format_cv_str,
    _format_subpopulation_summary,
)
from ._evaluation_helpers import resolve_state_eval_times

logger = get_logger("learner")

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from mlsurv.learner._core import SurvivalLearner

# Separate sentinel for setup() defaults (cannot import _NOT_SET from _core
# due to circular dependency: _core imports _setup at module level)
_NOT_SET_SENTINEL = object()


@dataclass(frozen=True)
class SetupConfig:
    """Bundled config for a single setup() invocation (internal use)."""
    test_fraction: float = 0.3
    stratify: Union[str, Any, None] = "event"
    groups: Union[str, Any, None] = None
    cv_folds: Union[int, str] = 5
    cv_repeats: int = 1
    cv_strategy: Union[str, Any] = "stratifiedkfold"
    cv_ci: float = 0.95
    imputer: Any = _NOT_SET_SENTINEL
    scaler: Any = _NOT_SET_SENTINEL
    feature_selector: Any = _NOT_SET_SENTINEL
    scoring: Union[str, Any] = DEFAULT_SCORING_METRIC
    random_state: int = 42
    experiment_name: Optional[str] = None
    n_jobs: int = 1
    device: Optional[str] = None
    verbose: bool = True
    outcome_name: str = "event"
    time_unit: Optional[str] = None
    feature_groups: Any = None
    group_colors: Any = None
    categorical_groups: Any = None
    categorical_features: Any = None
    preprocessing: Any = None  # per-model preprocessing overrides
    check_leakage: bool = True  # missingness-outcome leakage detection
    ph_test: bool = True  # PH assumption test during setup
    linearity_test: bool = True  # Log-hazard linearity test during setup
    # Subpopulation analysis
    subpopulations: Any = None
    subpopulations_train: Any = None
    subpopulations_test: Any = None
    stratify_subpopulations: bool = True


def _stratify_display_label(stratify) -> str:
    """Convert a user-supplied stratify value to a display label for reports.

    Returns the column name (string), the ``.name`` of a Series/Index when
    available, ``"none"`` for ``None``, or ``"custom stratify array"`` for an
    unnamed array. The label is interpolated into markdown so it intentionally
    avoids characters that markdown/HTML would escape (e.g. angle brackets).
    """
    if isinstance(stratify, str):
        return stratify
    if stratify is None:
        return "none"
    name = getattr(stratify, "name", None)
    if isinstance(name, str) and name:
        return name
    return "custom stratify array"


def _validate_scoring(scoring):
    """Validate scoring metric against known scorable metrics."""
    from ..constants import SCORABLE_METRICS
    if isinstance(scoring, str) and scoring not in SCORABLE_METRICS:
        raise ValueError(
            f"Unknown scoring metric: '{scoring}'. "
            f"Available: {SCORABLE_METRICS}"
        )


def _validate_cv_ci(cv_ci):
    """Validate that cv_ci is between 0 and 1 (exclusive)."""
    if not (0 < cv_ci < 1):
        raise ValueError(
            f"cv_ci must be between 0 and 1 (exclusive), got {cv_ci}"
        )


def _validate_outcome_name(outcome_name):
    """Validate that outcome_name is a non-empty string."""
    if not isinstance(outcome_name, str) or not outcome_name.strip():
        raise ValueError(
            f"outcome_name must be a non-empty string, got {outcome_name!r}"
        )


def _validate_time_unit(time_unit):
    """Validate that time_unit is a non-empty string or None."""
    if time_unit is not None:
        if not isinstance(time_unit, str) or not time_unit.strip():
            raise ValueError(
                f"time_unit must be a non-empty string or None, got {time_unit!r}"
            )


def _validate_feature_groups(feature_groups, group_colors):
    """Validate feature_groups and group_colors types and consistency."""
    if feature_groups is not None:
        import pandas as pd
        if not isinstance(feature_groups, (dict, list, tuple, pd.Series)):
            raise TypeError(
                f"feature_groups must be a dict, list, tuple, Series, or None, "
                f"got {type(feature_groups).__name__}"
            )
    if group_colors is not None:
        if not isinstance(group_colors, dict):
            raise TypeError(
                f"group_colors must be a dict or None, "
                f"got {type(group_colors).__name__}"
            )
        if feature_groups is None:
            warnings.warn(
                "group_colors has no effect without feature_groups. "
                "Set feature_groups to enable grouped importance plots.",
                SuggestionWarning,
                stacklevel=4,
            )


def _validate_categorical_groups_setup(categorical_groups, X):
    """Validate categorical_groups at setup time."""
    if categorical_groups is None:
        return
    if not isinstance(categorical_groups, dict):
        raise TypeError(
            f"categorical_groups must be a dict mapping group names to "
            f"column lists, got {type(categorical_groups).__name__}"
        )
    if X is None:
        return
    columns = set(X.columns)
    seen = set()
    for group_name, cols in categorical_groups.items():
        if not isinstance(group_name, str):
            raise TypeError(
                f"categorical_groups keys must be strings, got "
                f"{type(group_name).__name__} ({group_name!r})"
            )
        if not isinstance(cols, (list, tuple)):
            raise TypeError(
                f"categorical_groups['{group_name}'] must be a list of "
                f"column names, got {type(cols).__name__}"
            )
        for c in cols:
            if c not in columns:
                raise ValueError(
                    f"categorical_groups['{group_name}']: column '{c}' "
                    f"not found in X."
                )
            if c in seen:
                raise ValueError(
                    f"Column '{c}' appears in multiple categorical_groups "
                    f"entries."
                )
            seen.add(c)
    # Check group names don't collide with ungrouped columns
    ungrouped = columns - seen
    collisions = set(categorical_groups.keys()) & ungrouped
    if collisions:
        raise ValueError(
            f"Group name(s) {sorted(collisions)} collide with ungrouped "
            f"column names in X. Rename the groups or the columns."
        )


def _validate_categorical_features_setup(categorical_features, categorical_groups, X):
    """Validate categorical_features at setup time.

    ``None`` means "use the cardinality heuristic" for interaction PDP
    rendering. A list/tuple names single columns to treat as categorical
    (disabling the heuristic; an empty list forces all single columns
    continuous). Each name must be a column in ``X`` and must not also be a
    ``categorical_groups`` member column or group name.
    """
    if categorical_features is None:
        return
    if not isinstance(categorical_features, (list, tuple)):
        raise TypeError(
            f"categorical_features must be a list of column names or None, "
            f"got {type(categorical_features).__name__}"
        )
    if not all(isinstance(c, str) for c in categorical_features):
        raise TypeError(
            "categorical_features must contain only column-name strings."
        )
    if X is None:
        return
    columns = set(X.columns)
    group_names = set()
    grouped_cols = set()
    if isinstance(categorical_groups, dict):
        group_names = set(categorical_groups.keys())
        for cols in categorical_groups.values():
            if isinstance(cols, (list, tuple)):
                grouped_cols.update(cols)
    for c in categorical_features:
        if c not in columns:
            raise ValueError(
                f"categorical_features: column '{c}' not found in X."
            )
        if c in grouped_cols or c in group_names:
            raise ValueError(
                f"categorical_features: '{c}' is already declared in "
                f"categorical_groups; a column cannot be both."
            )


_VALID_PREPROCESSING_KEYS = frozenset({'imputer', 'scaler', 'feature_selector'})


def _validate_preprocessing(preprocessing, context='setup'):
    """Validate per-model preprocessing dict structure.

    Parameters
    ----------
    preprocessing : dict or None
        ``{model_name: {component: value}}``
    context : str
        ``'setup'`` or ``'run'`` for error messages.
    """
    if preprocessing is None:
        return
    if not isinstance(preprocessing, dict):
        raise TypeError(
            f"preprocessing must be a dict, got {type(preprocessing).__name__}"
        )
    for model_name, config in preprocessing.items():
        if not isinstance(model_name, str):
            raise TypeError(
                f"preprocessing keys must be strings (model names), "
                f"got {type(model_name).__name__}: {model_name!r}"
            )
        if not isinstance(config, dict):
            raise TypeError(
                f"preprocessing[{model_name!r}] must be a dict of "
                f"component overrides, got {type(config).__name__}"
            )
        unknown = set(config.keys()) - _VALID_PREPROCESSING_KEYS
        if unknown:
            raise ValueError(
                f"preprocessing[{model_name!r}] contains unknown keys: "
                f"{sorted(unknown)}. Valid keys: {sorted(_VALID_PREPROCESSING_KEYS)}"
            )


def _validate_cv_folds_vs_events(cv_folds, n_events_train):
    """Validate cv_folds vs training events and resolve 'auto'.

    Returns
    -------
    int
        Resolved cv_folds value.
    """
    if isinstance(cv_folds, int) and cv_folds > n_events_train:
        raise ValueError(
            f"cv_folds={cv_folds} exceeds training events "
            f"({n_events_train}). Each fold needs at least 1 event "
            f"for stratified CV. Reduce cv_folds or use "
            f"cv_folds='auto'."
        )
    if cv_folds == 'auto':
        return auto_cv_folds(n_events_train)
    return cv_folds


def _resolve_cv_repeats(cv_strategy, cv_repeats):
    """Resolve cv_repeats, forcing to 1 for GroupKFold/TimeSeriesSplit.

    Returns
    -------
    int
        Resolved cv_repeats value.
    """
    _strategy_names = {"groupkfold": "GroupKFold", "timeseries": "TimeSeriesSplit"}
    if cv_strategy in _strategy_names and cv_repeats > 1:
        # stacklevel=4: user -> setup -> _configure_cv -> _resolve_cv_repeats
        warnings.warn(
            f"cv_repeats > 1 is not supported with "
            f"{_strategy_names[cv_strategy]}. "
            f"Falling back to cv_repeats=1.",
            SuggestionWarning,
            stacklevel=4,
        )
        return 1
    return cv_repeats


def _check_missingness_leakage(X, y):
    """Check whether any feature's missingness correlates with the event.

    Returns a list of ``(column, p_value, v_stat, pct_missing)`` tuples for
    features exceeding the thresholds in ``constants.py``.
    """
    from scipy.stats import chi2_contingency

    events = y["event"].astype(bool)
    n_events = events.sum()
    n_censored = len(events) - n_events
    if n_events == 0 or n_censored == 0:
        return []

    n_rows = len(X)
    flagged = []
    for col in X.columns:
        is_na = X[col].isna()
        n_missing = int(is_na.sum())
        if n_missing == 0 or n_missing == n_rows:
            continue

        # 2x2 contingency: rows = (present, missing), cols = (censored, event)
        table = np.array([
            [int(((~is_na) & (~events)).sum()), int(((~is_na) & events).sum())],
            [int((is_na & (~events)).sum()),     int((is_na & events).sum())],
        ])

        try:
            chi2, p, _, _ = chi2_contingency(table)
        except ValueError:
            logger.debug(
                "chi2_contingency raised ValueError for column '%s'; "
                "skipping missingness leakage check.", col,
            )
            continue

        # Cramer's V for 2x2: min(r,c)-1 = 1, so V = sqrt(chi2 / n)
        v = float(np.sqrt(chi2 / table.sum()))
        if p < MISSINGNESS_LEAKAGE_P_THRESHOLD and v > MISSINGNESS_LEAKAGE_V_THRESHOLD:
            pct = 100.0 * n_missing / n_rows
            flagged.append((col, p, v, pct))

    return flagged


def _warn_missingness_leakage(flagged):
    """Emit a single MethodologicalWarning listing all flagged features."""
    n = len(flagged)
    plural = n > 1
    header = (
        f"{n} feature{'s' if plural else ''} ha{'ve' if plural else 's'} "
        f"missingness strongly associated with the event indicator. "
        f"This may indicate data leakage — features recorded only for "
        f"patients who experienced the event. Consider dropping "
        f"{'them' if plural else 'it'} or investigating the data "
        f"collection process."
    )
    detail_lines = [
        f"  - '{col}' (p={p:.1e}, Cramer's V={v:.2f}, {pct:.1f}% missing)"
        for col, p, v, pct in flagged
    ]
    warnings.warn(
        header + "\n" + "\n".join(detail_lines),
        MethodologicalWarning,
        stacklevel=4,
    )


_OTHER_BUCKET = "__other__"


def _collapse_rare_levels(events, sg_labels, threshold):
    """Build event x subpopulation stratify keys, merging rare levels.

    Parameters
    ----------
    events : array-like of {0, 1}
        Event indicator per sample.
    sg_labels : array-like
        Subpopulation level per sample.
    threshold : int
        Minimum required samples per (event x level) cell.

    Returns
    -------
    combined : np.ndarray of str or None
        Stratify keys ``"<event>_<level>"`` with rare levels replaced by
        ``__other__``. ``None`` when even the merged bucket fails.
    merged_levels : list
        Levels that were merged into ``__other__`` (empty when no merge
        was needed).
    fully_failed : bool
        ``True`` when the merged ``__other__`` bucket itself is below
        ``threshold`` and the caller should fall back to event-only.
    """
    events = np.asarray(events)
    sg_labels = np.asarray(sg_labels)
    if len(events) != len(sg_labels):
        raise ValueError(
            f"events ({len(events)}) and sg_labels ({len(sg_labels)}) "
            f"must have the same length"
        )
    if len(events) == 0:
        return None, [], True
    events_int = events.astype(int)

    cell_counts = Counter(zip(events_int, sg_labels))
    levels = list(dict.fromkeys(sg_labels.tolist()))

    merged_levels = [
        level for level in levels
        if min(cell_counts.get((e, level), 0) for e in (0, 1)) < threshold
    ]

    if not merged_levels:
        combined = np.array([f"{e}_{s}" for e, s in zip(events_int, sg_labels)])
        return combined, [], False

    merged_set = set(merged_levels)
    relabeled = np.array(
        [_OTHER_BUCKET if s in merged_set else s for s in sg_labels]
    )
    collapsed_counts = Counter(zip(events_int, relabeled))
    other_min = min(collapsed_counts.get((e, _OTHER_BUCKET), 0) for e in (0, 1))
    if other_min < threshold:
        return None, merged_levels, True

    combined = np.array([f"{e}_{s}" for e, s in zip(events_int, relabeled)])
    return combined, merged_levels, False


class SetupMixin:
    """Mixin for setup(), data splitting, and CV configuration."""

    def setup(
        self: SurvivalLearner,
        test_fraction: float = 0.3,
        stratify: str | np.ndarray | pd.Series | None = "event",
        groups: str | np.ndarray | pd.Series | None = None,
        cv_folds: int | str = 5,
        cv_repeats: int = 1,
        cv_strategy: str | Any = "stratifiedkfold",
        cv_ci: float = 0.95,
        imputer: str | Any | list | None = _NOT_SET_SENTINEL,
        scaler: str | Any | list | None = _NOT_SET_SENTINEL,
        feature_selector: str | Any | list | None = _NOT_SET_SENTINEL,
        preprocessing: dict | None = None,
        scoring: str | Callable = DEFAULT_SCORING_METRIC,
        random_state: int = 42,
        experiment_name: str | None = None,
        n_jobs: int = 1,
        device: str | None = None,
        verbose: bool = True,
        outcome_name: str = "event",
        time_unit: str | None = None,
        feature_groups: dict | Sequence | pd.Series | None = None,
        group_colors: dict | None = None,
        categorical_groups: dict | None = None,
        categorical_features: list | None = None,
        subpopulations: dict | None = None,
        subpopulations_train: dict | None = None,
        subpopulations_test: dict | None = None,
        stratify_subpopulations: bool = True,
        check_leakage: bool = True,
        ph_test: bool = True,
        linearity_test: bool = True,
        eval_times: np.ndarray | list | str | None = None,
    ) -> SurvivalLearner:
        """
        Configure preprocessing, splits, and random state.

        Parameters
        ----------
        test_fraction : float, default=0.3
            Fraction for test set.
        stratify : str or array, default='event'
            Stratification column or array.
        groups : str or array, optional
            Group labels for GroupKFold.
        cv_folds : int or str, default=5
            Number of CV folds (or 'auto').
        cv_repeats : int, default=1
            Number of times to repeat CV with different random splits.
            Values > 1 provide more stable CV estimates but increase
            computation time. Only applies to evaluation (run(),
            train()); tuning always uses single-run CV for efficiency.
        cv_strategy : str or CV object, default='stratifiedkfold'
            Cross-validation strategy. Options: 'stratifiedkfold' (maintains
            event ratio), 'kfold' (standard), 'groupkfold' (requires groups),
            'timeseries' (temporal ordering). Can also pass a custom CV object.
        cv_ci : float, default=0.95
            Confidence level for t-distribution-based CV confidence intervals.
            Must be between 0 and 1 (exclusive). The CI is computed from fold
            scores and accessible via ``summary()`` and ``get_metric()``.
        imputer : str, Transformer, list, or None, default='median'
            Imputer for missing values. Accepts:
            - String shortcuts: 'median', 'mean', 'knn', 'iterative'
            - Transformer: Custom sklearn transformer
            - List: Wrapped in Pipeline (e.g., [step1, step2])
            - None: Skip imputation
        scaler : str, Transformer, list, or None, default='standard'
            Scaler for feature normalization. Accepts:
            - String shortcuts: 'standard', 'robust', 'minmax', 'pca'
            - Transformer: Custom sklearn transformer
            - List: Wrapped in Pipeline (e.g., [StandardScaler(), PCA()])
            - None: Skip scaling
        feature_selector : str, Transformer, list, or None, default=None
            Feature selection method. Accepts:
            - String shortcuts: 'topk', 'vif', 'correlation'
            - Transformer: Custom sklearn transformer
            - List: Wrapped in Pipeline (e.g., ['vif', TopKCoxSelector(k='auto')])
            - None: No feature selection
        preprocessing : dict, optional
            Per-model preprocessing overrides. Dict of
            ``{model_name: {component: value}}`` where component is one of
            ``'imputer'``, ``'scaler'``, ``'feature_selector'``. Each value
            follows the same conventions as the global parameters above.
            Models not listed use the global defaults. Can be further
            overridden per-call in ``run()``/``train()``.

            Example::

                preprocessing={
                    'coxph': {'feature_selector': 'topk'},
                    'rsf': {'scaler': None},
                }
        scoring : str or callable, default=DEFAULT_SCORING_METRIC
            Scoring metric for CV and tuning.
        random_state : int, default=42
            Random seed.
        experiment_name : str, optional
            Experiment name for organizing Optuna studies. Studies will be saved to
            ./optuna_studies/{experiment_name}/. If not provided, a timestamped
            name is auto-generated when tuning runs (e.g., 'exp_20260122_181830').
            Use this to organize experiments for different datasets
            (e.g., 'gbsg2_analysis', 'flchain_cohort').
        n_jobs : int, default=1
            Number of parallel jobs for bootstrap, CV, and multi-model evaluation.
            - 1: Sequential execution (default, backward compatible)
            - -1: Use all available CPU cores
            - n > 1: Use n CPU cores

            **Oversubscription guarantee**: Total CPU core usage never exceeds
            n_jobs, even for models with their own parallelism (e.g., RSF,
            XGBoost). When outer parallelism is active (CV folds, multi-model
            evaluation, bootstrap iterations), internal model n_jobs is
            automatically set to 1 via ``adjust_model_njobs()``.
        device : str or None, default=None
            Device for deep learning models (DeepSurv, DeepHit):
            'auto' (GPU if available, else CPU), 'cpu', 'cuda', 'cuda:0',
            etc. If None (the default), the device parameter is not
            propagated to models, so each model uses its constructor
            default ('auto'). Non-DL models ignore this setting.
        verbose : bool, default=True
            If True, print the setup summary box after setup completes.
            Set to False to suppress output (e.g., in automated pipelines).
        outcome_name : str, default='event'
            Human-readable name for the outcome (e.g., 'death', 'recurrence',
            'progression'). Used in report text, table headers, and axis labels.
            Internal DataFrame column keys (``y['event']``, ``y['time']``)
            are not affected.
        time_unit : str, optional
            Unit of the survival time variable (e.g., 'months', 'years',
            'days'). When set, axis labels show 'Time (months)' instead of
            'Time'. If None (default), no unit suffix is added.
        feature_groups : dict, sequence, or Series, optional
            Feature-to-group mapping for grouped importance plots. Accepts:
            - Dict[str, str]: ``{'age': 'Clinical', 'wbc': 'Blood'}``
            - Sequence[str]: group labels aligned to feature order
            - pd.Series: index = feature names, values = group labels
            When set, importance plots show color-coded bars with separator
            lines between groups. If None (default), no grouping is applied.
        group_colors : dict, optional
            Mapping of group names to hex color strings, e.g.
            ``{'Clinical': '#1f77b4', 'Blood': '#ff7f0e'}``. Only used when
            ``feature_groups`` is set. If None, colors are auto-assigned.
        categorical_groups : dict of {str: list of str}, optional
            Mapping from an original categorical feature name to its encoded
            (e.g. one-hot) column names. Lets permutation importance permute
            all encoded columns of a categorical feature together. If None
            (default), each column is treated independently.
        categorical_features : list of str, optional
            Single (already-encoded) columns to treat as categorical/discrete
            in 2D interaction partial-dependence plots, so they render as a
            line-per-level overlay or heatmap instead of an interpolated
            contour. If None (default), a column with at most 5 unique values
            is treated as categorical automatically. Passing a list disables
            that heuristic and marks exactly the listed columns categorical
            (an empty list forces all single columns to be treated as
            continuous). Use ``categorical_groups`` for multi-column one-hot
            features instead; a column cannot appear in both.
        subpopulations : dict of {name: array}, optional
            Subpopulation labels for subpopulation analysis, aligned with X. Each array
            defines a categorical grouping variable (e.g., cancer type).
            Auto-split into train/test along with X. For pre-split data,
            use ``subpopulations_train``/``subpopulations_test`` instead.
        subpopulations_train : dict of {name: array}, optional
            Subpopulation labels aligned with X_train (pre-split data only).
            Used for CV stratification by subpopulation.
        subpopulations_test : dict of {name: array}, optional
            Subpopulation labels aligned with X_test (pre-split data only).
            Required for per-subpopulation evaluation.
        stratify_subpopulations : bool, default=True
            When True and subpopulations are specified, stratify the train/test
            split and CV folds by (event x subpopulation) combinations.
        check_leakage : bool, default=True
            When True, check whether any feature's missingness pattern is
            strongly associated with the event indicator (chi-squared
            p < 0.001 and Cramer's V > 0.3). Emits a single
            ``MethodologicalWarning`` listing all flagged features.
            Set to False to skip this check.
        ph_test : bool, default=True
            When True, run the Grambsch-Therneau proportional hazards
            test on training data after preprocessing is configured.
            Emits a ``MethodologicalWarning`` if the global test is
            significant, listing violating features. Set to False to
            skip (e.g., for large datasets where the Cox fit is slow).
        linearity_test : bool, default=True
            When True, run Harrell's single-fit Wald test (rms::anova.cph
            style) for log-hazard linearity on continuous training
            covariates. Emits a ``MethodologicalWarning`` if the global
            test is significant, listing non-linear features. Set to
            False to skip (e.g., for large datasets where the stacked
            RCS Cox fit is slow).
        eval_times : array-like or None, default=None
            Optional timepoint grid for time-dependent metrics (AUC(t),
            Brier(t), IBS, calibration). When provided, it seeds the canonical
            grid that :meth:`evaluate` and :meth:`bootstrap` inherit, so the
            point estimates and bootstrap
            CI bands share one ``eval_time`` set. Each call can still override
            it, or pass ``eval_times='auto'`` to fall back to auto-computation.
            When None (default), the grid is auto-computed from the data.

        Returns
        -------
        self

        Raises
        ------
        MemoryError
            Re-raised if the computation exhausts available memory; resource
            errors are never swallowed by the broad failure handler.
        RecursionError
            Re-raised if the computation exceeds the recursion limit; resource
            errors are never swallowed by the broad failure handler.
        """
        # Fail fast on a malformed eval_times grid (scalar / multi-dim /
        # non-positive / non-finite) before any split or CV state is built, so
        # an invalid grid never leaves the learner half-configured. persist=False
        # means nothing is stored here; the real seeding happens after
        # _finalize_setup so a failed split leaves no stale grid behind.
        resolve_state_eval_times(self._state, eval_times, persist=False)
        config = SetupConfig(
            test_fraction=test_fraction,
            stratify=stratify,
            groups=groups,
            cv_folds=cv_folds,
            cv_repeats=cv_repeats,
            cv_strategy=cv_strategy,
            cv_ci=cv_ci,
            imputer=imputer,
            scaler=scaler,
            feature_selector=feature_selector,
            preprocessing=preprocessing,
            scoring=scoring,
            random_state=random_state,
            experiment_name=experiment_name,
            n_jobs=n_jobs,
            device=device,
            verbose=verbose,
            outcome_name=outcome_name,
            time_unit=time_unit,
            feature_groups=feature_groups,
            group_colors=group_colors,
            categorical_groups=categorical_groups,
            categorical_features=categorical_features,
            subpopulations=subpopulations,
            subpopulations_train=subpopulations_train,
            subpopulations_test=subpopulations_test,
            stratify_subpopulations=stratify_subpopulations,
            check_leakage=check_leakage,
            ph_test=ph_test,
            linearity_test=linearity_test,
        )
        self._validate_setup_inputs(config)
        # Resolve (and thereby validate) preprocessing BEFORE clearing any
        # results — a typo'd preset on re-setup must not destroy trained
        # models and downstream results.
        self._resolve_preprocessing(config)
        self._reset_state_if_needed()
        # Past this point setup() mutates the split/CV config in place
        # (_configure_cv) and can raise (bad stratify/groups column,
        # cv_folds > events, unknown cv_strategy). If it does, the learner
        # is half-configured. Clear is_setup now so a post-reset failure
        # leaves a learner that refuses to run (via _ensure_setup) instead
        # of one that reports ready and trains on mixed old/new state.
        # _finalize_setup sets it back to True only on success.
        self._state.is_setup = False
        self._configure_cv(config)
        self._finalize_setup(config)
        # Seed the canonical eval_times grid if provided. _reset_state_if_needed
        # cleared user_eval_times above; an explicit grid here becomes the
        # default that evaluate/bootstrap/calibration inherit (each can still
        # override per call, or pass eval_times='auto' to fall back to auto).
        # Seeded after _finalize_setup so a failed split leaves no stale grid.
        resolve_state_eval_times(self._state, eval_times)
        try:
            self._run_setup_diagnostics(config)
        except (MemoryError, RecursionError):
            raise
        except (ValueError, KeyError, TypeError, RuntimeError,
                ImportError, AttributeError, np.linalg.LinAlgError) as exc:
            logger.warning(
                "Setup diagnostics harness raised an unexpected %s (%s); "
                "setup is complete but one or more diagnostics may be "
                "unavailable. Each individual diagnostic (PH, linearity, "
                "model recommendations) has its own inner handler, so "
                "reaching this outer catch indicates a failure at the "
                "boundary (e.g., an import error).",
                type(exc).__name__, exc,
            )
        return self

    def _reset_state_if_needed(self: SurvivalLearner):
        """Warn on re-setup and clear previous results."""
        if self._state.is_setup:
            warnings.warn(
                "setup() has already been called. Re-running will overwrite "
                "the existing train/test split and clear all trained models, "
                "evaluation, bootstrap, validation, importance-test, "
                "benchmark, and statistical-test results, and any attached "
                "S(t|x) calibrators; "
                "explicit eval_times are reset to auto.",
                SuggestionWarning,
                stacklevel=3,
            )
            # Calibrators, importance-test, and risk-group-test results ride
            # on each ModelResult, so clearing `models` drops them all
            # in one step — no separate per-field clearing needed.
            self._state.models.clear()
            self._state.validation_results.clear()
            self._state.subpopulation_results.clear()
            # model_preprocessing is NOT cleared here: _resolve_preprocessing
            # runs before this method and unconditionally rebinds it
            # ({} when no per-model overrides), so clearing would wipe the
            # freshly resolved dict.
            self._state.subpopulations = None
            self._state.subpopulations_train = None
            self._state.subpopulations_test = None
            self._state.ph_test_result = None
            self._state.linearity_test_result = None
            self._state.model_recommendations = None
            self._state.user_eval_times = None
            # Registered report horizons and their render-warning dedup derive
            # from the old training split's censoring window, so they go stale
            # on re-setup.
            self._state.user_extra_eval_times = None
            self._state.extra_eval_times_render_warned = False
            # Cross-model results computed from the old split. The per-model
            # derived state (recalibration, importance-test, risk-group-test)
            # rode on the ModelResults cleared above; these aggregates
            # are not on any ModelResult, so clear them explicitly here.
            self._state.benchmark_result = None
            self._state.kang_comparison_result = None
            self._state.ipcw_support_flag = None
            self._state.reference_overlap_warned.clear()
            self._state.fit_failure_warned.clear()
            # These two dedup the IPCW-truncation and TimeSeriesSplit-CI
            # warnings. The truncation cap (t_max) derives from the old
            # training split and the CI warning describes the old CV fold
            # structure, so every cohort's warning goes stale on re-setup.
            self._state.ipcw_truncation_warned.clear()
            self._state.heavy_censoring_warned.clear()
            self._state.timeseries_ci_warned = False

    def _validate_setup_inputs(self: SurvivalLearner, config: SetupConfig):
        """Validate setup() inputs without mutating any state.

        Pure validation: state writes happen later (``_finalize_setup``),
        so a failed re-setup leaves the learner untouched.
        """
        _validate_scoring(config.scoring)
        _validate_cv_ci(config.cv_ci)
        _validate_outcome_name(config.outcome_name)
        _validate_time_unit(config.time_unit)
        _validate_feature_groups(config.feature_groups, config.group_colors)
        _validate_categorical_groups_setup(config.categorical_groups, self._state.X)
        _validate_categorical_features_setup(
            config.categorical_features, config.categorical_groups, self._state.X,
        )
        _validate_preprocessing(config.preprocessing, context='setup')

        if not self._state.presplit:
            tf = config.test_fraction
            if (isinstance(tf, (bool, np.bool_))
                    or not isinstance(tf, (int, float, np.integer, np.floating))
                    or not (0 < tf < 1)):
                raise ValueError(
                    f"test_fraction must be a float in (0, 1), got {tf!r}"
                )

    def _resolve_preprocessing(self: SurvivalLearner, config: SetupConfig):
        """Resolve imputer/scaler/selector and per-model overrides; validate data.

        Runs before ``_reset_state_if_needed`` and must stay atomic: resolve
        everything into locals and assign to ``self._state`` only after all
        resolution and validation succeeds, so a bad preset (or bad data)
        raises without this method touching ``self._state``.
        """
        imputer = (get_default_imputer() if config.imputer is _NOT_SET_SENTINEL
                   else get_imputer(config.imputer, random_state=config.random_state))
        scaler = (get_default_scaler() if config.scaler is _NOT_SET_SENTINEL
                  else get_scaler(config.scaler, random_state=config.random_state))

        if config.feature_selector is _NOT_SET_SENTINEL:
            feature_selector = None
        else:
            feature_selector = get_selector(
                config.feature_selector, random_state=config.random_state
            )

        # Resolve per-model preprocessing overrides
        model_preprocessing = _resolve_per_model_preprocessing(
            config.preprocessing, random_state=config.random_state
        )

        validate_survival_data(
            self.X,
            self.y,
            warn_missing=imputer is None,
        )

        if config.check_leakage:
            flagged = _check_missingness_leakage(self.X, self.y)
            if flagged:
                _warn_missingness_leakage(flagged)

        if not self._state.presplit and not self.X.index.is_unique:
            warnings.warn(
                "DataFrame has non-unique index. This may cause incorrect "
                "CV stratification/grouping. Consider resetting index: "
                "X.reset_index(drop=True)",
                MethodologicalWarning,
                stacklevel=3,
            )

        self._state.imputer = imputer
        self._state.scaler = scaler
        self._state.feature_selector = feature_selector
        self._state.model_preprocessing = model_preprocessing

        _log_preprocessing(imputer, scaler, feature_selector, model_preprocessing)

    def _configure_cv(self: SurvivalLearner, config: SetupConfig):
        """Split data, validate folds, and create CV splitter objects."""
        # Phase 1: Validate subpopulations early (before split) so we can use
        # them to augment stratify_array in train_test_split
        self._validate_subpopulations_early(config)

        # Handle pre-split vs full data
        if self._state.presplit:
            self._setup_presplit_data(
                config.test_fraction, config.stratify, config.groups,
            )
        else:
            self._setup_split_data(
                config.test_fraction, config.stratify, config.groups,
                config.cv_strategy, config.random_state, config,
            )

        # Phase 2: Slice subpopulations into train/test using split indices
        self._split_subpopulations(config)

        # Augment stratify_array with subpopulation for CV if requested.
        # Use the train-sliced stratify_array (stored on state after split),
        # NOT the full-length array returned by _setup_split_data.
        if (config.stratify_subpopulations
                and self._state.subpopulations_train
                and config.cv_strategy == 'stratifiedkfold'):
            train_stratify = self._state.stratify_array
            augmented = self._augment_stratify_with_subpopulations(
                train_stratify, config.cv_folds,
            )
            self._state.stratify_array = augmented
            subpop_keys = list(self._state.subpopulations_train.keys())
            if len(subpop_keys) > 1:
                warnings.warn(
                    f"stratify_subpopulations=True with {len(subpop_keys)} "
                    f"subpopulation variables {subpop_keys!r} — only the "
                    f"first key {subpop_keys[0]!r} is used to augment CV "
                    f"stratification. The remaining keys participate in "
                    f"subpopulation analysis but not in fold composition.",
                    SuggestionWarning,
                    stacklevel=3,
                )

        logger.debug(
            "Data split completed: %d train / %d test samples",
            len(self.y_train), len(self.y_test),
        )

        # Check Events Per Variable (EPV) for overfitting risk
        n_events_train = int(self.y_train['event'].sum())
        n_features = self.X_train.shape[1]
        epv = n_events_train / n_features if n_features > 0 else float('inf')
        if epv < 10:
            warnings.warn(
                f"Low events per variable (EPV={epv:.1f}). With "
                f"{n_events_train} events and {n_features} features, models "
                f"may overfit during hyperparameter tuning. Consider: "
                f"(1) using fixed hyperparameters instead of tuning, "
                f"(2) aggressive feature selection, or (3) stronger "
                f"regularization. Rule of thumb: EPV >= 10.",
                MethodologicalWarning,
                stacklevel=3,
            )
            self._state.epv_warned_at_setup = True
        else:
            self._state.epv_warned_at_setup = False

        # Check for small subpopulations (few events -> unreliable metrics).
        # Sets _state.subpop_test_warned so that downstream evaluate() can
        # skip emitting the same aggregate warning a second time.
        self._state.subpop_test_warned = False
        if self._state.subpopulations_test:
            import pandas as pd
            from ..evaluation._subpopulation import MIN_EVENTS_WARNING

            events = np.asarray(self.y_test['event'])
            for group_var, labels in self._state.subpopulations_test.items():
                unique_labels = sorted(set(labels[~pd.isna(labels)]))
                if not unique_labels:
                    continue
                with IterativeWarningCoalescer(
                    mode="notices",
                    summary_label=f"'{group_var}' subpopulation",
                    notice_phrase=f"have insufficient events (< {MIN_EVENTS_WARNING})",
                    n_total=len(unique_labels),
                    max_emit=0,
                    category=SmallSubpopulationWarning,
                    stacklevel=3,
                    tail_msg="Metric estimates may be unreliable.",
                ) as cz:
                    for label in unique_labels:
                        mask = labels == label
                        n_events = int(events[mask].sum())
                        if n_events < MIN_EVENTS_WARNING:
                            msg = (
                                f"Subpopulation '{label}' (group "
                                f"'{group_var}') has only {n_events} events "
                                f"(< {MIN_EVENTS_WARNING}). Metric estimates "
                                f"may be unreliable."
                            )
                            cz.record_notice(
                                msg,
                                group_key=str(label),
                                display=f"{label} ({n_events})",
                            )
                            self._state.subpop_test_warned = True

        # Validate and resolve CV folds
        cv_folds = _validate_cv_folds_vs_events(config.cv_folds, n_events_train)
        self.cv_folds = cv_folds

        # Effective repeats: GroupKFold/TimeSeriesSplit force cv_repeats=1
        cv_repeats = _resolve_cv_repeats(config.cv_strategy, config.cv_repeats)
        self.cv_repeats = cv_repeats

        # Create CV splitters: one for evaluation (with repeats), one for tuning (single-run)
        # Use train-sliced stratify_array from state (may have been augmented
        # with subpopulation labels above)
        cv_stratify = self._state.stratify_array
        self._state.cv_splitter = self._get_cv_splitter(
            config.cv_strategy, cv_folds, self._state.groups, cv_stratify,
            config.random_state, cv_repeats=cv_repeats,
        )
        self._state.cv_splitter_single = self._get_cv_splitter(
            config.cv_strategy, cv_folds, self._state.groups, cv_stratify,
            config.random_state, cv_repeats=1,
        )
        logger.debug(
            "CV strategy selected: %s, folds=%d, repeats=%d",
            config.cv_strategy, cv_folds, cv_repeats,
        )
        self.cv_strategy = config.cv_strategy

        # Check for conditions where repeated CV might be beneficial
        self._check_cv_warnings(config.cv_repeats)

        # Warn if preprocessing order seems wrong
        warn_preprocessing_order(
            self._state.imputer, self._state.scaler,
            self._state.feature_selector,
        )

    def _finalize_setup(self: SurvivalLearner, config: SetupConfig):
        """Store remaining config, compute descriptive stats, and print summary."""
        self.test_fraction = config.test_fraction
        self.random_state = config.random_state
        self.scoring = config.scoring
        self.cv_ci = config.cv_ci
        self._state.experiment_name = config.experiment_name
        self.n_jobs = config.n_jobs
        self._state.device = config.device
        self._state.outcome_name = config.outcome_name
        self._state.time_unit = config.time_unit
        self._state.feature_groups = config.feature_groups
        self._state.group_colors = config.group_colors
        self._state.categorical_groups = config.categorical_groups
        self._state.categorical_features = config.categorical_features
        self._state.stratify_subpopulations = config.stratify_subpopulations
        self._state.stratify_spec = _stratify_display_label(config.stratify)
        self._state.check_leakage = config.check_leakage
        self._state.ph_test_enabled = config.ph_test
        self._state.linearity_test_enabled = config.linearity_test

        # Descriptive only (reports) — not used for model building or feature selection.
        # Direction labels in reports depend on this, so log details on failure.
        try:
            self.univariate_cox = compute_univariate_cox(self.X, self.y)
        except Exception as exc:
            if isinstance(exc, (MemoryError, RecursionError)):
                raise
            logger.warning(
                "Univariate Cox computation failed (%s: %s); "
                "feature direction info will be unavailable in reports",
                type(exc).__name__, exc,
            )
            self.univariate_cox = None

        self._state.is_setup = True

        if config.verbose:
            try_verbose_print(
                self._print_setup_summary,
                context="Setup summary",
                detail="All configuration is active.",
            )

    def _run_setup_diagnostics(self: SurvivalLearner, config: SetupConfig):
        """Run PH test, linearity test, and model recommendations after setup.

        Called at end of ``setup()`` after ``_finalize_setup()``.  Implemented
        as a registry of ``_DiagnosticSpec`` entries iterated in order; each
        spec runs in its own try/except so a failure in one does not mislead
        the user about the state of the others.  See ``_SETUP_DIAGNOSTICS``
        for the registered diagnostics and their gates.
        """
        # ``None`` signals "test was not run / failed" so the recommendation
        # table can render "PH test unavailable" / "linearity test
        # unavailable"; an empty list signals "ran, no violations".
        ctx: dict = {
            "X_imp": None,
            "ph_violations": None,
            "linearity_violations": None,
            "ph_df": None,
            "lin_df": None,
        }
        for spec in _SETUP_DIAGNOSTICS:
            if not spec.gate(config):
                continue
            try:
                updates = spec.run(ctx, self, config)
            except (MemoryError, RecursionError):
                raise
            except _DIAGNOSTIC_EXCS as exc:
                logger.warning(
                    "%s diagnostic failed (%s: %s); "
                    "result unavailable.",
                    spec.name, type(exc).__name__, exc,
                )
                if spec.warn_on_escape:
                    warnings.warn(
                        f"{spec.name} raised an unexpected "
                        f"{type(exc).__name__} ({exc}) that escaped the "
                        f"inner handler; result unavailable. "
                        f"This is likely a bug — please report.",
                        InternalWarning,
                        stacklevel=3,
                    )
                continue
            for attr, value in updates.items():
                setattr(self._state, attr, value)

    def _setup_presplit_data(self: SurvivalLearner, test_fraction, stratify, groups):
        """Handle setup for pre-split data (Pattern 2).

        Returns
        -------
        array-like
            Stratification array for CV.
        """
        if test_fraction != 0.3:  # Non-default value
            warnings.warn(
                f"test_fraction={test_fraction} ignored when using pre-split data.",
                SuggestionWarning,
                stacklevel=4,
            )
        if not (isinstance(stratify, str) and stratify == "event"):
            warnings.warn(
                f"stratify={stratify!r} ignored when using pre-split data.",
                SuggestionWarning,
                stacklevel=4,
            )

        if list(self.X_test.columns) != list(self.X_train.columns):
            warnings.warn(
                "X_test columns were reordered to match X_train column order.",
                InternalWarning,
                stacklevel=4,
            )
            self.X_test = self.X_test[self.X_train.columns]

        stratify_array = self.y_train['event']
        self._state.stratify_array = stratify_array

        self._state.groups = self.X_train[groups] if isinstance(groups, str) else groups
        return stratify_array

    @staticmethod
    def _resolve_stratify_array(stratify, X, y):
        """Resolve stratify parameter to an array or None.

        Parameters
        ----------
        stratify : str, array-like, or None
        X : DataFrame
        y : SurvivalArray

        Returns
        -------
        array-like or None
        """
        if isinstance(stratify, str):
            return y['event'] if stratify == "event" else X[stratify]
        return stratify  # array-like or None

    @staticmethod
    def _resolve_groups(groups, X):
        """Resolve groups parameter to an array or None.

        Parameters
        ----------
        groups : str, array-like, or None
        X : DataFrame

        Returns
        -------
        ndarray or None
        """
        if isinstance(groups, str):
            return X[groups].values
        if groups is not None:
            return np.asarray(groups)
        return None

    def _setup_split_data(self: SurvivalLearner, test_fraction, stratify, groups,
                          cv_strategy, random_state, config=None):
        """Handle setup for full data with train/test split (Pattern 1).

        Returns
        -------
        array-like or None
            Stratification array for CV.
        """
        stratify_array = self._resolve_stratify_array(stratify, self.X, self.y)
        self._state.stratify_array = stratify_array

        groups_full = self._resolve_groups(groups, self.X)

        # Augment stratify_array with subpopulation labels for train_test_split
        # when stratify_subpopulations=True and subpopulations are validated
        split_stratify = stratify_array
        if (config is not None
                and config.stratify_subpopulations
                and self._state.subpopulations is not None
                and split_stratify is not None
                and cv_strategy != 'timeseries'):
            split_stratify = self._augment_stratify_for_split(
                split_stratify, self._state.subpopulations,
            )

        # Train-test split. A group-aware holdout keeps whole groups together
        # when groups are supplied, so no subject straddles the train/test
        # boundary (timeseries is excluded - it must stay time-ordered).
        train_pos = test_pos = None
        if groups_full is not None and cv_strategy != 'timeseries':
            train_pos, test_pos = self._group_aware_holdout(
                test_fraction, groups_full, random_state,
            )

        # Assign definitively (not just on the True branch) so a re-setup()
        # without groups clears a stale flag from a previous group-aware split.
        self._state.group_aware_holdout = train_pos is not None

        if train_pos is not None:
            self.X_train = self.X.iloc[train_pos]
            self.X_test = self.X.iloc[test_pos]
            self.y_train = self.y[train_pos]
            self.y_test = self.y[test_pos]
        # Time-series CV requires ordered data - no shuffle
        elif cv_strategy == 'timeseries':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_fraction, shuffle=False,
            )
            # The split and CV folds are only temporal if the rows arrive
            # pre-sorted by entry/diagnosis time. mlsurv has no order column to
            # check, so it cannot tell correct use from misuse - it can only
            # remind the user of the precondition. That makes this a
            # SuggestionWarning (actionable: "sort your data first"), not a
            # MethodologicalWarning (which is reserved for a problem mlsurv has
            # actually detected, like low EPV or group leakage). The split runs
            # once per setup(), so the warning fires once per setup() without a
            # dedup flag. (SuggestionWarning's default "once" filter does not
            # collapse repeat setup() calls in practice: "once" dedups via the
            # caller's per-module warning registry, which is cleared whenever the
            # filter version changes, and setup() mutates warning filters in its
            # call chain - so each setup() re-emits.) stacklevel=4 points at the
            # user's setup() call: user -> setup -> _configure_cv ->
            # _setup_split_data -> warn.
            warnings.warn(
                "Temporal split (cv_strategy='timeseries') assumes rows are "
                "pre-sorted chronologically by entry/diagnosis time. mlsurv "
                "does not verify this. If the rows are not in chronological "
                "order, the train/test split and CV folds are not temporal "
                "and the resulting out-of-sample estimate is invalid.",
                SuggestionWarning,
                stacklevel=4,
            )
        elif split_stratify is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_fraction,
                stratify=split_stratify, random_state=random_state,
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_fraction, random_state=random_state,
            )

        # Slice groups and stratify_array to the training rows, preserving
        # X_train row order. The group-aware path already holds clean positional
        # indices (train_pos); use them directly. The train_test_split paths
        # shuffle by label, so recover positions from X_train.index instead
        # (this label round-trip would crash/mis-slice on a non-unique index,
        # which is already flagged with a MethodologicalWarning at setup).
        if train_pos is not None:
            train_positions = train_pos
        else:
            train_positions = self.X.index.get_indexer(self.X_train.index)

        self._state.groups = groups_full[train_positions] if groups_full is not None else None

        if stratify_array is not None:
            self._state.stratify_array = np.asarray(stratify_array)[train_positions]

        return stratify_array

    def _group_aware_holdout(self: SurvivalLearner, test_fraction, groups_full,
                             random_state):
        """Compute a group-pure train/test holdout split.

        Keeps every group (e.g. all rows of one subject) on the same side of
        the train/test boundary, preventing subject-level leakage.

        Parameters
        ----------
        test_fraction : float
            Requested fraction of rows for the test set.
        groups_full : ndarray
            Group label per row of ``self.X``.
        random_state : int
            Random seed.

        Returns
        -------
        (ndarray, ndarray) or (None, None)
            Positional indices into ``self.X`` / ``self.y`` for train and test,
            or ``(None, None)`` to signal the caller should fall back to a plain
            row-level split (only when fewer than two distinct groups exist).
        """
        n_groups = len(np.unique(groups_full))
        if n_groups < 2:
            warnings.warn(
                "groups has only one distinct group, so a leakage-free "
                "train/test holdout is impossible; falling back to a row-level "
                "split. Rows from this group will appear in both train and test.",
                MethodologicalWarning,
                stacklevel=5,
            )
            return None, None

        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_fraction, random_state=random_state,
        )
        # GroupShuffleSplit raises its own group-count-based ValueError (e.g. a
        # large test_fraction leaves no groups for train) before our own check
        # below; translate it into an actionable message that names the cause.
        try:
            train_pos, test_pos = next(
                splitter.split(self.X, self.y, groups=groups_full)
            )
        except ValueError as exc:
            raise ValueError(
                f"test_fraction={test_fraction} with {n_groups} groups cannot "
                f"produce a non-empty group-pure train/test split. Reduce group "
                f"granularity or adjust test_fraction."
            ) from exc

        if len(train_pos) == 0 or len(test_pos) == 0:
            raise ValueError(
                f"test_fraction={test_fraction} with {n_groups} groups produced "
                f"an empty train or test set after keeping groups intact. Reduce "
                f"group granularity or adjust test_fraction."
            )

        realized = len(test_pos) / len(groups_full)
        if abs(realized - test_fraction) > 0.05:
            warnings.warn(
                f"Group-aware holdout kept whole groups intact, so the realized "
                f"test fraction is {realized:.2f} (requested {test_fraction:.2f}). "
                f"Whole groups move as a unit, so the exact fraction cannot always "
                f"be hit.",
                SuggestionWarning,
                stacklevel=5,
            )
        return train_pos, test_pos

    def _get_cv_splitter(self: SurvivalLearner, cv_strategy, cv_folds, groups,
                         stratify_array, random_state, cv_repeats=1):
        """Resolve CV strategy to sklearn splitter.

        Parameters
        ----------
        cv_strategy : str or CV object
            CV strategy name or custom CV object.
        cv_folds : int
            Number of CV folds.
        groups : array-like or None
            Group labels for GroupKFold.
        stratify_array : array-like or None
            Stratification labels.
        random_state : int
            Random seed.
        cv_repeats : int, default=1
            Number of times to repeat CV. If > 1, uses RepeatedStratifiedKFold
            or RepeatedKFold as appropriate.

        Returns
        -------
        CV splitter object
        """
        # Custom CV object passthrough (check for get_n_splits to avoid matching strings)
        if hasattr(cv_strategy, 'split') and hasattr(cv_strategy, 'get_n_splits'):
            return cv_strategy

        builders = {
            "stratifiedkfold": lambda: (
                RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats,
                                        random_state=random_state)
                if cv_repeats > 1 else
                StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                random_state=random_state)
            ),
            "kfold": lambda: (
                RepeatedKFold(n_splits=cv_folds, n_repeats=cv_repeats,
                              random_state=random_state)
                if cv_repeats > 1 else
                KFold(n_splits=cv_folds, shuffle=True,
                      random_state=random_state)
            ),
            "groupkfold": lambda: self._build_group_kfold(
                cv_folds, groups),
            "timeseries": lambda: self._build_timeseries_split(
                cv_folds),
        }

        builder = builders.get(cv_strategy)
        if builder is None:
            raise ValueError(f"Unknown cv_strategy: {cv_strategy}")
        return builder()

    @staticmethod
    def _build_group_kfold(cv_folds, groups):
        """Build GroupKFold splitter with validation."""
        if groups is None:
            raise ValueError("groups required for groupkfold")
        return GroupKFold(n_splits=cv_folds)

    @staticmethod
    def _build_timeseries_split(cv_folds):
        """Build TimeSeriesSplit splitter."""
        return TimeSeriesSplit(n_splits=cv_folds)

    def _validate_subpopulations_early(self: SurvivalLearner, config: SetupConfig):
        """Phase 1: Validate subpopulation labels before train/test split.

        For non-presplit data, validates ``config.subpopulations`` and stores on
        ``_state.subpopulations``. Does NOT access X_train/X_test (not yet split).
        For presplit data, defers entirely to ``_split_subpopulations``.
        """
        if config.subpopulations is None and config.subpopulations_train is None and config.subpopulations_test is None:
            return

        if self._state.presplit:
            # Pre-split validation deferred to _split_subpopulations (needs X_train/X_test)
            return

        # Non-presplit: validate full subpopulations dict
        if config.subpopulations_train is not None or config.subpopulations_test is not None:
            warnings.warn(
                "subpopulations_train/subpopulations_test are ignored for non-presplit "
                "data. Use subpopulations={name: labels} aligned with X instead.",
                SuggestionWarning,
                stacklevel=4,
            )
        if config.subpopulations is not None:
            validated = _validate_subpopulation_dict(
                config.subpopulations, len(self._state.X), "subpopulations"
            )
            self._state.subpopulations = validated

    def _split_subpopulations(self: SurvivalLearner, config: SetupConfig):
        """Phase 2: Slice subpopulations into train/test using split indices.

        For presplit data, validates and stores explicit train/test params.
        For non-presplit data, slices ``_state.subpopulations`` using the
        train/test indices created by ``_setup_split_data``.
        """
        if config.subpopulations is None and config.subpopulations_train is None and config.subpopulations_test is None:
            return

        if self._state.presplit:
            self._split_subpopulations_presplit(config)
        else:
            self._split_subpopulations_full()

    def _split_subpopulations_presplit(self: SurvivalLearner, config: SetupConfig):
        """Handle subpopulation splitting for pre-split data."""
        if config.subpopulations is not None:
            raise ValueError(
                "With pre-split data, use subpopulations_train and/or "
                "subpopulations_test instead of subpopulations."
            )
        if config.subpopulations_test is not None:
            self._state.subpopulations_test = _validate_subpopulation_dict(
                config.subpopulations_test, len(self.y_test), "subpopulations_test"
            )
        if config.subpopulations_train is not None:
            self._state.subpopulations_train = _validate_subpopulation_dict(
                config.subpopulations_train, len(self.y_train), "subpopulations_train"
            )

    def _split_subpopulations_full(self: SurvivalLearner):
        """Slice validated subpopulations into train/test using split indices."""
        validated = self._state.subpopulations
        if validated is not None:
            train_positions = self._state.X.index.get_indexer(self.X_train.index)
            test_positions = self._state.X.index.get_indexer(self.X_test.index)

            self._state.subpopulations_train = {
                k: np.asarray(v)[train_positions] for k, v in validated.items()
            }
            self._state.subpopulations_test = {
                k: np.asarray(v)[test_positions] for k, v in validated.items()
            }

    @staticmethod
    def _augment_stratify_for_split(stratify_array, subpopulations):
        """Combine event stratification with subpopulation labels for train_test_split.

        Collapses rare levels (any (event x level) cell with < 2 samples) into
        an `__other__` bucket; falls back to event-only only if the collapsed
        bucket itself fails the threshold.
        """
        first_key = next(iter(subpopulations))
        sg_labels = np.asarray(subpopulations[first_key])
        events = np.asarray(stratify_array)

        combined, merged_levels, fully_failed = _collapse_rare_levels(
            events, sg_labels, threshold=2
        )
        if fully_failed:
            detail = (
                f"merging rare levels {sorted(merged_levels, key=str)} into '__other__'"
                if merged_levels
                else "no levels could be merged"
            )
            warnings.warn(
                f"Smallest (event x subpopulation) cell remained < 2 samples "
                f"after {detail}. Falling back to event-only stratification "
                f"for train/test split.",
                MethodologicalWarning,
                stacklevel=5,
            )
            return stratify_array
        if merged_levels:
            warnings.warn(
                f"Subpopulation levels {sorted(merged_levels, key=str)} had < 2 samples "
                f"in some (event x level) cell; merged into '__other__' for "
                f"train/test split stratification. Other levels stratify "
                f"normally.",
                SuggestionWarning,
                stacklevel=5,
            )
        return combined

    def _augment_stratify_with_subpopulations(
        self: SurvivalLearner,
        stratify_array,
        cv_folds,
    ):
        """Combine event stratification with subpopulation labels for CV.

        Collapses rare levels (any (event x level) cell with < cv_folds
        samples) into an `__other__` bucket; falls back to event-only only if
        the collapsed bucket itself fails the threshold.
        """
        if stratify_array is None:
            return None

        first_key = next(iter(self._state.subpopulations_train))
        sg_labels = np.asarray(self._state.subpopulations_train[first_key])
        events = np.asarray(stratify_array)

        resolved_folds = cv_folds if isinstance(cv_folds, int) else 5
        combined, merged_levels, fully_failed = _collapse_rare_levels(
            events, sg_labels, threshold=resolved_folds
        )
        if fully_failed:
            detail = (
                f"merging rare levels {sorted(merged_levels, key=str)} into '__other__'"
                if merged_levels
                else "no levels could be merged"
            )
            warnings.warn(
                f"Smallest (event x subpopulation) cell remained < "
                f"cv_folds={resolved_folds} after {detail}. Falling back to "
                f"event-only stratification for CV.",
                MethodologicalWarning,
                stacklevel=4,
            )
            return stratify_array
        if merged_levels:
            warnings.warn(
                f"Subpopulation levels {sorted(merged_levels, key=str)} had < "
                f"cv_folds={resolved_folds} samples in some (event x level) "
                f"cell; merged into '__other__' for CV stratification. Other "
                f"levels stratify normally.",
                SuggestionWarning,
                stacklevel=4,
            )
        return combined

    def _check_cv_warnings(self: SurvivalLearner, cv_repeats):
        """Check for conditions where repeated CV might be beneficial.

        Warns when:
        - Small sample size (n_train < 200)
        - Event imbalance (event rate < 15% or > 85%)
        - Few events per fold (< 10)

        Parameters
        ----------
        cv_repeats : int
            Current cv_repeats setting.
        """
        if cv_repeats > 1:
            return

        n_train = len(self.y_train)
        n_events = int(self.y_train['event'].sum())
        event_rate = n_events / n_train
        events_per_fold = n_events / self.cv_folds

        reasons = []
        if n_train < MIN_SAMPLE_SIZE_WARNING:
            reasons.append(f"small sample size (n={n_train})")
        if event_rate < MIN_EVENT_RATE or event_rate > MAX_EVENT_RATE:
            reasons.append(f"event imbalance ({event_rate:.1%} event rate)")
        if events_per_fold < 10:
            reasons.append(f"few events per fold ({events_per_fold:.1f} events/fold)")

        if reasons:
            warnings.warn(
                f"Consider using cv_repeats > 1 for more stable CV estimates "
                f"due to {', '.join(reasons)}. Example: setup(cv_repeats=3)",
                SuggestionWarning,
                stacklevel=4,
            )

    def _print_setup_summary(self: SurvivalLearner):
        """Print a boxed summary of the setup configuration."""
        pipeline_str = _format_pipeline_str(
            self._state.imputer, self._state.scaler, self._state.feature_selector,
        )
        data_str = _format_data_str(self.y_train, self.y_test, self.X_train)
        cv_str = _format_cv_str(
            self.cv_strategy, self.cv_folds, self.cv_repeats,
            self.scoring, self.cv_ci,
        )

        # State
        state_parts = [f"seed={self.random_state}"]
        if self._state.experiment_name is not None:
            state_parts.append(f"experiment={self._state.experiment_name}")
        state_str = ", ".join(state_parts)

        rows = [
            ("Pipeline:", pipeline_str),
            ("Data:", data_str),
            ("CV:", cv_str),
            ("State:", state_str),
        ]

        # Group-aware holdout note (only when groups kept the split subject-pure)
        if self._state.group_aware_holdout:
            rows.append(
                ("Holdout:", "group-aware (no group split across train/test)")
            )

        # Per-model preprocessing overrides
        if self._state.model_preprocessing:
            per_model_str = _format_per_model_preprocessing(
                self._state.model_preprocessing
            )
            rows.append(("Per-model:", per_model_str))

        # Subpopulation summary
        rows.extend(_format_subpopulation_summary(self._state.subpopulations_test))

        truncation_note = check_truncation_needed(self.y_test, self.y_train)
        if truncation_note:
            rows.append(("IPCW note:", truncation_note))
        _print_param_box("Setup", rows)


# ---------------------------------------------------------------------------
# Shared diagnostic runner
# ---------------------------------------------------------------------------

_DIAGNOSTIC_EXCS = (
    ValueError, KeyError, RuntimeError, TypeError,
    ImportError, AttributeError, np.linalg.LinAlgError,
)


def _run_assumption_tests(
    X,
    y,
    *,
    run_ph: bool = True,
    run_linearity: bool = True,
    stacklevel: int = 4,
):
    """Run PH and/or linearity tests on ``(X, y)`` with graceful fallback.

    Returns a 4-tuple ``(ph_violations, linearity_violations, ph_df,
    linearity_df)``. Each violations entry is a list of feature names
    when the test ran, ``[]`` when the test ran but found nothing, or
    ``None`` when the test was skipped or failed (so callers can render
    "test unavailable"). Each ``df`` is the raw test result, or ``None``
    when unavailable.

    Failure modes are caught (the same ``_DIAGNOSTIC_EXCS`` tuple as
    ``_run_setup_diagnostics``) and surfaced via ``logger.warning`` plus
    an ``InternalWarning``; ``MemoryError`` and ``RecursionError``
    propagate. Caller is responsible for any user-facing
    ``MethodologicalWarning`` (e.g., ``_warn_ph_violations``) — this
    helper deliberately stays out of state mutation and methodological
    messaging so it can be reused outside ``setup()``.

    ``stacklevel`` defaults to 4, which is correct when this helper is
    called directly from a public API (caller stack:
    ``warn → helper → public_fn → user``). Callers nested one frame
    deeper (e.g., ``setup() → _run_setup_diagnostics() → helper``)
    should pass ``stacklevel=5``.
    """
    ph_violations: list[str] | None = None
    linearity_violations: list[str] | None = None
    ph_df = None
    lin_df = None

    if run_ph:
        try:
            from ..evaluation.metrics import ph_test as _ph_test

            time_transform = 'rank'
            df = _ph_test(X, y, time_transform=time_transform)
            if df.empty:
                # ph_test() already emits a single summary warning for the
                # all-NaN / no-rows / empty-summary cases; avoid adding a
                # second user-visible line here.
                logger.debug(
                    "PH test returned empty results (convergence failure "
                    "or insufficient data); skipping PH diagnostic"
                )
            else:
                # Stamp the result-affecting params so the stored DataFrame
                # is self-describing.
                df.attrs['time_transform'] = time_transform
                df.attrs['source'] = 'setup'
                ph_df = df
                ph_violations = _get_ph_violations(df)
        except (MemoryError, RecursionError):
            raise
        except _DIAGNOSTIC_EXCS as exc:
            logger.warning(
                "PH diagnostic failed (%s: %s); result unavailable.",
                type(exc).__name__, exc,
            )
            warnings.warn(
                f"PH diagnostic raised an unexpected "
                f"{type(exc).__name__} ({exc}) that escaped the inner "
                f"handler; PH test result unavailable. "
                f"This is likely a bug — please report.",
                InternalWarning,
                stacklevel=stacklevel,
            )

    if run_linearity:
        try:
            from ..evaluation.metrics import linearity_test as _linearity_test

            n_knots = 4
            df = _linearity_test(X, y, n_knots=n_knots)
            if not df.empty:
                # Stamp the result-affecting params so the stored DataFrame
                # is self-describing.
                df.attrs['n_knots'] = n_knots
                df.attrs['source'] = 'setup'
                lin_df = df
                linearity_violations = _get_linearity_violations(df)
        except (MemoryError, RecursionError):
            raise
        except _DIAGNOSTIC_EXCS as exc:
            logger.warning(
                "Linearity diagnostic failed (%s: %s); result unavailable.",
                type(exc).__name__, exc,
            )
            warnings.warn(
                f"Linearity diagnostic raised an unexpected "
                f"{type(exc).__name__} ({exc}) that escaped the inner "
                f"handler; linearity test result unavailable. "
                f"This is likely a bug — please report.",
                InternalWarning,
                stacklevel=stacklevel,
            )

    return ph_violations, linearity_violations, ph_df, lin_df


# ---------------------------------------------------------------------------
# Setup-diagnostic registry
# ---------------------------------------------------------------------------
#
# ``_run_setup_diagnostics`` iterates ``_SETUP_DIAGNOSTICS`` in order. Each
# spec runs in its own try/except (via the orchestrator); cross-spec
# dataflow is threaded through the shared ``ctx`` dict (e.g. assumption
# tests populate ``ctx["ph_violations"]`` so the recommendations runner can
# read it). State writes happen in the orchestrator via
# ``setattr(self._state, attr, value)`` so each runner just returns the
# updates it wants applied.

@dataclass(frozen=True)
class _DiagnosticSpec:
    """One entry in the setup-diagnostic registry."""
    name: str
    gate: Callable[[SetupConfig], bool]
    run: Callable[[dict, "SurvivalLearner", SetupConfig], dict]
    warn_on_escape: bool = True


def _diag_run_imputation(ctx: dict, learner: SurvivalLearner, config: SetupConfig) -> dict:
    """Impute X_train into ``ctx["X_imp"]``.

    Imputation failure is caught here (not re-raised) so downstream specs
    can short-circuit on ``ctx["X_imp"] is None``. The detailed log message
    naming both PH and linearity state fields matches pre-refactor
    behavior, where imputation failure was a quiet (logger-only) skip.
    """
    try:
        ctx["X_imp"] = learner._get_imputed_X(learner.X_train, raise_on_error=True)
    except _DIAGNOSTIC_EXCS as exc:
        logger.warning(
            "Imputation failed before assumption tests (%s: %s); "
            "skipping PH and linearity diagnostics. "
            "The PH and linearity test results on "
            "learner.model_recommendations are unavailable.",
            type(exc).__name__, exc,
        )
    return {}


def _diag_run_assumption_tests(ctx: dict, learner: SurvivalLearner, config: SetupConfig) -> dict:
    """Run PH + linearity tests on the imputed X (when available)."""
    if ctx["X_imp"] is None:
        return {}
    ph_violations, linearity_violations, ph_df, lin_df = _run_assumption_tests(
        ctx["X_imp"],
        learner.y_train,
        run_ph=config.ph_test,
        run_linearity=config.linearity_test,
        # +1 frame for the extra wrapper: setup → _run_setup_diagnostics →
        # _diag_run_assumption_tests → _run_assumption_tests → warn.
        stacklevel=6,
    )
    ctx["ph_violations"] = ph_violations
    ctx["linearity_violations"] = linearity_violations
    ctx["ph_df"] = ph_df
    ctx["lin_df"] = lin_df

    updates: dict = {}
    if ph_df is not None:
        updates["ph_test_result"] = ph_df
        if ph_violations:
            _warn_ph_violations(ph_violations)
    elif config.ph_test:
        # The Model Recommendations table already surfaces the "PH test
        # unavailable" fallback; this log line was redundant noise on top
        # of the metrics-level summary warning.
        logger.debug(
            "learner.model_recommendations.ph_test_result is unavailable; "
            "PH-assuming model recommendations will fall back to "
            "the 'PH test unavailable' clause."
        )
    if lin_df is not None:
        updates["linearity_test_result"] = lin_df
        if linearity_violations:
            _warn_linearity_violations(linearity_violations)
    elif config.linearity_test:
        # Same rationale as the PH branch — the recommendations table
        # surfaces the fallback clause.
        logger.debug(
            "learner.model_recommendations.linearity_test_result is "
            "unavailable; linearity-assuming model recommendations will "
            "fall back to the 'linearity test unavailable' clause."
        )
    return updates


def _diag_run_recommendations(ctx: dict, learner: SurvivalLearner, config: SetupConfig) -> dict:
    """Compute model recommendations from prior diagnostic outputs."""
    from ..guidance import recommend_models

    lin_df = ctx["lin_df"]
    n_lin_tested = None
    if lin_df is not None:
        n_lin_tested = int((lin_df["feature"] != "GLOBAL").sum())
    rec = recommend_models(
        learner.X_train, learner.y_train,
        ph_violations=ctx["ph_violations"],
        linearity_violations=ctx["linearity_violations"],
        n_linearity_tested=n_lin_tested,
        run_assumption_tests=False,
        verbose=False,
    )
    # Attach the full setup-time PH/linearity test DataFrames so the bundle
    # is self-contained (feature-level evidence alongside the model-level
    # tiers). Both are ``None`` when the corresponding test did not run.
    # This is the setup snapshot — ``ph_test()``/``linearity_test()`` later
    # refresh ``_state.ph_test_result`` only, not this bundle.
    rec.ph_test_result = ctx["ph_df"]
    rec.linearity_test_result = ctx["lin_df"]
    if config.verbose:
        try_verbose_print(
            print, rec,
            context="Model recommendations",
            detail="Result available via learner.model_recommendations.",
        )
    return {"model_recommendations": rec}


_SETUP_DIAGNOSTICS: list[_DiagnosticSpec] = [
    _DiagnosticSpec(
        name="imputation",
        gate=lambda cfg: cfg.ph_test or cfg.linearity_test,
        run=_diag_run_imputation,
        warn_on_escape=False,
    ),
    _DiagnosticSpec(
        name="assumption_tests",
        gate=lambda cfg: cfg.ph_test or cfg.linearity_test,
        run=_diag_run_assumption_tests,
        warn_on_escape=False,
    ),
    _DiagnosticSpec(
        name="model_recommendations",
        gate=lambda cfg: True,
        run=_diag_run_recommendations,
        warn_on_escape=True,
    ),
]


# ---------------------------------------------------------------------------
# PH diagnostic helpers
# ---------------------------------------------------------------------------

def _get_ph_violations(df) -> list[str]:
    """Extract feature names with significant PH violations.

    Returns an empty list when the global test is not significant
    or the DataFrame is empty.
    """
    if df.empty:
        return []
    global_row = df[df['feature'] == 'GLOBAL']
    if global_row.empty or not bool(global_row.iloc[0]['significant']):
        return []
    violations = df[(df['significant']) & (df['feature'] != 'GLOBAL')]
    return violations['feature'].tolist()


def _warn_ph_violations(ph_violations: list[str]) -> None:
    """Emit a MethodologicalWarning listing PH-violating features."""
    n = len(ph_violations)
    display = ", ".join(ph_violations[:5])
    suffix = f" (and {n - 5} more)" if n > 5 else ""
    from ..constants import MODEL_DISPLAY_NAMES, PH_ASSUMPTION_MODELS
    affected = ", ".join(
        sorted(MODEL_DISPLAY_NAMES[m] for m in PH_ASSUMPTION_MODELS)
    )
    warnings.warn(
        f"Proportional hazards assumption violated (Grambsch-Therneau "
        f"global p < 0.05). {n} feature(s) with significant violations: "
        f"{display}{suffix}. PH-assuming models ({affected}) may produce "
        f"biased estimates. Inspect learner.ph_test_result or consider "
        f"flexible non-PH alternatives (RSF, DeepHit, XGBoost-AFT).",
        MethodologicalWarning,
        # setup → _run_setup_diagnostics → _diag_run_assumption_tests →
        # _warn_ph_violations → warn.
        stacklevel=5,
    )


# ---------------------------------------------------------------------------
# Linearity diagnostic helpers
# ---------------------------------------------------------------------------

def _get_linearity_violations(df) -> list[str]:
    """Return BH-significant per-feature linearity violations.

    Unlike :func:`_get_ph_violations`, this does **not** gate on the
    GLOBAL joint-Wald row. The linearity test applies Benjamini-Hochberg
    FDR correction across the per-feature p-values (see
    :func:`mlsurv.evaluation.metrics.linearity_test`), which already
    controls the family-wise false-discovery rate at 0.05. Gating on the
    GLOBAL row would silently suppress single-feature violations on wide
    datasets, where the joint Wald (df = ``(n_knots - 2) * n_tested``)
    dilutes any individual signal — discovered during E2E testing on the
    full GBSG2 feature set.
    """
    if df.empty:
        return []
    violations = df[(df['significant']) & (df['feature'] != 'GLOBAL')]
    return violations['feature'].tolist()


def _warn_linearity_violations(linearity_violations: list[str]) -> None:
    """Emit a MethodologicalWarning listing linearity-violating features.

    The test follows Harrell 2015 §5.5 / R's ``rms::anova.cph``.
    """
    n = len(linearity_violations)
    display = ", ".join(linearity_violations[:5])
    suffix = f" (and {n - 5} more)" if n > 5 else ""
    from ..constants import LINEARITY_ASSUMPTION_MODELS, MODEL_DISPLAY_NAMES
    affected = ", ".join(
        sorted(MODEL_DISPLAY_NAMES[m] for m in LINEARITY_ASSUMPTION_MODELS)
    )
    warnings.warn(
        f"Log-hazard linearity assumption violated (Harrell/rms Wald, "
        f"BH-FDR-adjusted per-feature p < 0.05; Harrell 2015 §5.5). "
        f"{n} continuous feature(s) with significant non-linearity: "
        f"{display}{suffix}. Linearity-assuming models ({affected}) may "
        f"produce biased estimates for these features. Consider non-linear "
        f"transformations (log, Box-Cox), or switch to a flexible model "
        f"(RSF, GBSA, DeepSurv). Inspect learner.linearity_test_result "
        f"and the Linearity Assumption Assessment section of the report "
        f"for per-feature details.",
        MethodologicalWarning,
        # setup → _run_setup_diagnostics → _diag_run_assumption_tests →
        # _warn_linearity_violations → warn.
        stacklevel=5,
    )


def _validate_subpopulation_dict(subpopulations, n_samples, param_name):
    """Validate a subpopulations dict and return normalized {name: ndarray}."""
    import pandas as pd

    if not isinstance(subpopulations, dict):
        raise ValueError(
            f"{param_name} must be a dict of {{name: array}}, "
            f"got {type(subpopulations).__name__}"
        )
    result = {}
    for name, labels in subpopulations.items():
        arr = np.asarray(labels, dtype=object).ravel()
        if len(arr) != n_samples:
            raise ValueError(
                f"{param_name}['{name}'] has length {len(arr)}, "
                f"expected {n_samples}"
            )
        nan_mask = pd.isna(arr)
        n_nan = int(nan_mask.sum())
        if n_nan > 0:
            warnings.warn(
                f"Subpopulation '{name}' has {n_nan} missing (NaN/None) labels. "
                f"These samples will be excluded from subpopulation analysis.",
                SuggestionWarning,
                stacklevel=5,
            )
        n_unique = len(set(arr[~nan_mask]))
        if n_unique > 20:
            warnings.warn(
                f"Subpopulation '{name}' has {n_unique} unique values. "
                f"Subpopulation analysis is intended for categorical variables "
                f"with few levels (e.g., cancer type).",
                SuggestionWarning,
                stacklevel=5,
            )
        result[name] = arr
    return result


def _resolve_per_model_preprocessing(preprocessing, random_state=None):
    """Resolve raw per-model preprocessing dict to transformer objects.

    Parameters
    ----------
    preprocessing : dict or None
        Raw ``{model_name: {component: value}}`` from user.
    random_state : int or None, default=None
        Seed threaded into every component resolver so the seeded string
        presets use the configured seed (matching the top-level components):
        the ``iterative`` imputer's draws, the ``pca`` scaler's randomized SVD,
        and the ``coxlasso``/``coxnet`` selectors' alpha-selection CV.

    Returns
    -------
    dict
        ``{model_name: {component: resolved_transformer_or_None}}``
    """
    if not preprocessing:
        return {}
    resolved = {}
    _resolvers = {
        'imputer': lambda v: get_imputer(v, random_state=random_state),
        'scaler': lambda v: get_scaler(v, random_state=random_state),
        'feature_selector': lambda v: get_selector(v, random_state=random_state),
    }
    for model_name, config in preprocessing.items():
        resolved_config = {}
        for key, resolver in _resolvers.items():
            if key in config:
                try:
                    resolved_config[key] = resolver(config[key])
                except (ValueError, TypeError) as e:
                    raise type(e)(
                        f"preprocessing[{model_name!r}][{key!r}]: {e}"
                    ) from e
        if resolved_config:
            resolved[model_name] = resolved_config
    return resolved


def _log_preprocessing(imputer, scaler, selector, model_preprocessing=None):
    """Log resolved preprocessing component types at DEBUG level."""
    def _name(obj):
        return type(obj).__name__ if obj is not None else None
    logger.debug(
        "Preprocessing resolved: imputer=%s, scaler=%s, selector=%s",
        _name(imputer), _name(scaler), _name(selector),
    )
    if model_preprocessing:
        for model_name, overrides in model_preprocessing.items():
            parts = [f"{k}={_name(v)}" for k, v in overrides.items()]
            logger.debug("  %s: %s", model_name, ", ".join(parts))

