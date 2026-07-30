"""Core SurvivalLearner class definition with shared helpers."""

from __future__ import annotations

import warnings
from types import MappingProxyType
from typing import Optional

import numpy as np
import pandas as pd

from ..models import (
    get_deep_learning_models,
    get_models_with_internal_parallelism,
    get_models_with_random_state,
    get_sequential_cv_models,
)
from ..constants import CORRELATION_THRESHOLD, DEFAULT_SCORING_METRIC
from ..logging_utils import get_logger
from .._warnings import InternalWarning, SuggestionWarning
from ..utils import validate_survival_data, adjust_model_njobs
from ..results.containers import ModelResult

logger = get_logger("learner")

from ._state import LearnerState
from ._setup import SetupMixin
from ._training import TrainingMixin
from ._tuning import TuningMixin
from ._evaluation import EvaluationMixin
from ._eval_times import EvalTimesMixin
from ._benchmarking import BenchmarkingMixin
from ._sweep import SweepMixin
from ._visualization import VisualizationMixin
from ._config import ConfigMixin
from ._persistence import PersistenceMixin
from ._studies import StudyMixin
from ._export import ExportMixin
from ._reports import ReportMixin
from ._explain import ExplainMixin
from ._prediction import PredictionMixin
from ._patient_report import PatientReportMixin
from ._recalibration import RecalibrationMixin


def _state_property(state_attr, doc=None):
    """Create a read/write property that proxies to ``self._state.<state_attr>``.

    Pass *doc* to attach a docstring so the property is self-documenting under
    ``help()``, IDE hovers, and ``pydoc`` (a bare property would show nothing).
    """
    def fget(self):
        return getattr(self._state, state_attr)
    def fset(self, value):
        setattr(self._state, state_attr, value)
    return property(fget, fset, doc=doc)

# Sentinel to distinguish "not passed" from "passed as None"
_NOT_SET = object()

# Models that support random_state parameter (for reproducibility)
# Built dynamically from model class metadata
MODELS_WITH_RANDOM_STATE = get_models_with_random_state()

# Models where CV parallelization overhead exceeds benefit
# CoxPH/CoxNet: Too fast, ~1.5-2s joblib overhead dominates
# DeepSurv/DeepHit: Uses GPU parallelization instead
# Built dynamically from model class metadata
SEQUENTIAL_CV_MODELS = get_sequential_cv_models()

# Models with internal parallelism (n_jobs), e.g. RSF, XGBCox
# These need n_jobs propagated from setup() for fit/predict parallelism
# Built dynamically from model class metadata
MODELS_WITH_INTERNAL_PARALLELISM = get_models_with_internal_parallelism()

# Deep learning models (DeepSurv, DeepHit) that accept a device parameter
# These need device propagated from setup() for GPU/CPU placement
# Built dynamically from model class metadata (complexity == 'deep_learning')
DEEP_LEARNING_MODELS = get_deep_learning_models()

# Backward compatibility alias for _adjust_model_njobs
_adjust_model_njobs = adjust_model_njobs


class SurvivalLearner(
    SetupMixin,
    TrainingMixin,
    TuningMixin,
    EvaluationMixin,
    EvalTimesMixin,
    PredictionMixin,
    BenchmarkingMixin,
    SweepMixin,
    VisualizationMixin,
    ConfigMixin,
    PersistenceMixin,
    StudyMixin,
    ExportMixin,
    ReportMixin,
    ExplainMixin,
    PatientReportMixin,
    RecalibrationMixin,
):
    """
    Main API for survival analysis with mlsurv.

    Supports both quick single-call usage and granular, step-by-step workflows.

    Parameters
    ----------
    X : DataFrame
        Feature matrix.
    y : SurvivalArray
        Survival outcome with 'event' and 'time' fields.

    Notes
    -----
    Assumptions. mlsurv fits standard right-censored survival models. The data
    must meet the conditions below. The package does not check them, so they
    are the user's responsibility:

    - One row per subject. All splitting (the train/test holdout and the CV
      folds) is row-level. With repeated measures or clustered data, pass a
      collapsed one-row-per-subject table, or the same subject can land in both
      train and test.
    - Covariates measured at baseline (time zero). Time-varying covariates are
      not supported. A feature that carries information from after baseline (a
      post-baseline measurement, or time accrued before treatment starts)
      causes immortal-time or look-ahead bias and inflates apparent
      performance.
    - Censoring independent of the event given the covariates. Informative
      censoring (for example, sicker patients dropping out) biases every
      metric, and no internal check can detect it.
    - A single event type. Treating a competing event (for example, death from
      another cause) as ordinary censoring overestimates the event's
      cumulative incidence. mlsurv warns when it sees more than two event
      codes, but cannot model competing risks.

    Examples
    --------
    Quick usage:
    >>> learner = SurvivalLearner(X, y)
    >>> results = learner.run(tune=True)

    Granular usage:
    >>> learner = SurvivalLearner(X, y)
    >>> learner.setup(test_fraction=0.3)
    >>> learner.train()                    # CV on all models
    >>> learner.tune('rsf')                # Tune specific model
    >>> learner.evaluate()
    >>> learner.generate_report()
    """

    # -- Public properties (permanent) — proxy to self._state -----------
    X = _state_property('X')
    y = _state_property('y')
    X_train = _state_property('X_train')
    X_test = _state_property('X_test')
    y_train = _state_property('y_train')
    y_test = _state_property('y_test')
    univariate_cox = _state_property(
        'univariate_cox',
        doc=(
            "Per-feature univariate Cox statistics (DataFrame), or ``None``.\n\n"
            "Model-agnostic: one row per feature describing that feature's "
            "marginal (univariate) association with the outcome, independent of "
            "any trained model. Computed automatically during ``setup()`` on the "
            "full cohort ``(X, y)`` — it is descriptive only and is never used "
            "for model fitting or feature selection.\n\n"
            "Columns: ``feature``, ``p_value`` (Wald), ``hazard_ratio``, "
            "``hr_lower_95``, ``hr_upper_95``, ``coefficient`` (log hazard "
            "ratio); sorted by ascending ``p_value``. Features that fail to fit "
            "get ``p_value=inf`` and NaN ``hazard_ratio``/``coefficient``.\n\n"
            "``None`` before ``setup()`` runs or if the computation failed (a "
            "warning is logged in that case). Read/write — assign a DataFrame "
            "from ``mlsurv.evaluation.compute_univariate_cox`` to override (e.g. "
            "to recompute on ``X_train``/``y_train`` only)."
        ),
    )
    scoring = _state_property('scoring')
    random_state = _state_property('random_state')
    test_fraction = _state_property('test_fraction')
    cv_folds = _state_property('cv_folds')
    cv_repeats = _state_property('cv_repeats')
    cv_ci = _state_property('cv_ci')
    cv_strategy = _state_property('cv_strategy')
    n_jobs = _state_property('n_jobs')

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[np.ndarray] = None,
        *,  # Force keyword-only after this
        X_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_train: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize SurvivalLearner with data.

        Supports two initialization patterns:

        1. Full data (split during setup):
           ``SurvivalLearner(X, y)``

        2. Pre-split data:
           ``SurvivalLearner(X_train=..., X_test=..., y_train=..., y_test=...)``

        Parameters
        ----------
        X : pd.DataFrame, optional
            Full feature matrix (for pattern 1).
        y : SurvivalArray, optional
            Full survival target with 'event' and 'time' fields (for pattern 1).
        X_train : pd.DataFrame, optional
            Training feature matrix (for pattern 2).
        X_test : pd.DataFrame, optional
            Test feature matrix (for pattern 2).
        y_train : SurvivalArray, optional
            Training survival target (for pattern 2).
        y_test : SurvivalArray, optional
            Test survival target (for pattern 2).

        Raises
        ------
        ValueError
            If neither pattern is satisfied, or if both are mixed.
        """
        # Initialize centralized state container
        self._state = LearnerState()

        # Determine initialization mode
        if X is not None and y is not None:
            # Pattern 1: Full data (existing behavior)
            if any(v is not None for v in [X_train, X_test, y_train, y_test]):
                raise ValueError(
                    "Cannot mix positional (X, y) with keyword (X_train, X_test, ...) parameters."
                )
            self._state.X = X
            self._state.y = y
            self._state.presplit = False

        elif all(v is not None for v in [X_train, X_test, y_train, y_test]):
            # Pattern 2: Pre-split data
            self._validate_presplit_data(X_train, X_test, y_train, y_test)
            self._state.X = X_train  # For univariate Cox and compatibility
            self._state.y = y_train
            self._state.X_train = X_train
            self._state.X_test = X_test
            self._state.y_train = y_train
            self._state.y_test = y_test
            self._state.presplit = True

        else:
            raise ValueError(
                "Provide either (X, y) for full data, or all of "
                "(X_train, X_test, y_train, y_test) for pre-split data."
            )

    def __repr__(self) -> str:
        if not self._state.is_setup:
            n = len(self._state.X) if self._state.X is not None else 0
            p = self._state.X.shape[1] if self._state.X is not None else 0
            return f"SurvivalLearner(n={n}, features={p}, state=initialized)"

        n_train = len(self._state.X_train) if self._state.X_train is not None else 0
        n_test = len(self._state.X_test) if self._state.X_test is not None else 0
        p = self._state.X_train.shape[1] if self._state.X_train is not None else 0
        n_models = len(self._state.models)

        if n_models == 0:
            return f"SurvivalLearner(n_train={n_train}, n_test={n_test}, features={p}, state=ready)"

        best = self.best_model
        if best is not None:
            cv_str = f", best='{best.name}' cv={best.cv_metric_mean:.3f}"
        else:
            cv_str = ""
        return f"SurvivalLearner(n_train={n_train}, n_test={n_test}, features={p}, models={n_models}{cv_str})"

    def _validate_presplit_data(self, X_train, X_test, y_train, y_test):
        """
        Validate pre-split data compatibility.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        X_test : pd.DataFrame
            Test feature matrix.
        y_train : SurvivalArray
            Training survival target.
        y_test : SurvivalArray
            Test survival target.

        Raises
        ------
        ValueError
            If data validation fails or features don't match.
        """
        # Validate each dataset (skip size/event checks on test set — it's
        # legitimately smaller than the training set after splitting)
        validate_survival_data(X_train, y_train)
        validate_survival_data(X_test, y_test, min_samples=0, min_events=0)

        # Check feature compatibility
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)

        if test_cols != train_cols:
            missing = train_cols - test_cols
            extra = test_cols - train_cols
            raise ValueError(
                f"Feature mismatch. Missing in test: {missing}, Extra in test: {extra}"
            )

    @property
    def models(self):
        """
        Access all trained and tuned models.

        Returns
        -------
        dict
            Dictionary of {model_key: ModelResult}. Keys are plain algorithm
            names (e.g. ``'rsf'``); tuning overwrites in place, so each
            algorithm has a single entry that is untuned or tuned.
        """
        return self._state.models

    @property
    def is_setup(self):
        """Whether :meth:`setup` has been run (the ``check_is_fitted`` analog).

        Returns
        -------
        bool
            ``True`` once :meth:`setup` has split the data and configured the
            pipeline; ``False`` on a freshly constructed learner.
        """
        return self._state.is_setup

    @property
    def evaluation_result(self):
        """Combined evaluation from all models that have been evaluated.

        Assembles a multi-model :class:`EvaluationResult` from per-model
        ``ModelResult.evaluation`` fields.  Computed on each access from the
        current per-model evaluations, so it always reflects the live state.

        Returns
        -------
        EvaluationResult or None
        """
        from ..results.containers import EvaluationResult

        evaluated = {
            n: m for n, m in self._state.models.items()
            if m.evaluation is not None
        }
        if not evaluated:
            return None
        return EvaluationResult.combine(evaluated)

    @property
    def bootstrap_result(self):
        """Combined bootstrap from all models that have been bootstrapped.

        Assembles a multi-model :class:`BootstrapResult` from per-model
        data.  Computed on each access from the current per-model bootstrap
        data, so it always reflects the live state.

        Returns
        -------
        BootstrapResult or None
        """
        from mlsurv.viz._data_metrics import _has_any_bootstrap

        if not _has_any_bootstrap(self._state.models):
            return None
        from ..results.containers import BootstrapResult
        return BootstrapResult.combine(self._state.models)

    @property
    def validation_results(self):
        """External-validation results, keyed by cohort name.

        Bulk read-only view of every cohort registered by :meth:`validate`.
        For a single cohort use the :meth:`validation` view
        (``learner.validation(cohort=...)``); this is the missing bulk/
        existence accessor.

        Returns
        -------
        mappingproxy
            Read-only map of cohort name to its ``ValidationResult``; empty
            before any :meth:`validate` call. Key assignment raises
            ``TypeError``.
        """
        return MappingProxyType(self._state.validation_results)

    @property
    def subpopulation_results(self):
        """Subpopulation-analysis results, keyed by grouping variable.

        Bulk read-only view of every group configured for subpopulation
        analysis. For a single group use the :meth:`subpopulation` view
        (``learner.subpopulation(group=...)``); this is the missing bulk/
        existence accessor.

        Returns
        -------
        mappingproxy
            Read-only map of grouping-variable name to its
            ``SubpopulationResult``; empty before any subpopulation analysis.
            Key assignment raises ``TypeError``.
        """
        return MappingProxyType(self._state.subpopulation_results)

    @property
    def best_model(self):
        """Best non-synthetic model by CV metric mean.

        Returns
        -------
        ModelResult or None
        """
        # ``best_model_key`` filters synthetic models and picks the top
        # key by ``cv_metric_mean`` (higher-is-better space). It is the
        # shared selector behind the ``models='best'`` plot sentinel, so
        # both stay in sync. See ``mlsurv._shared.cv_sorted_model_keys``
        # for why ``max`` is correct for every metric.
        from .._shared import best_model_key
        key = best_model_key(self._state.models)
        return self._state.models[key] if key is not None else None

    @property
    def coefficients(self):
        """Coefficients across all trained models, with gated inference.

        Assembled live from ``_state.models`` on every access (like
        ``best_model`` — not a stored state field), so it reflects the
        current train/tune state.

        Long format with columns ``model_type``, ``feature``,
        ``coefficient``, ``hazard_ratio``, ``hr_lower_95``,
        ``hr_upper_95``, ``p_value`` (Wald). One block of rows per
        trained model that exposes coefficients (CoxPH, CoxNet, FSSVM,
        Weibull-AFT); models without coefficients (tree/neural) are
        omitted. Rows are in pipeline feature order per model (not
        p-sorted, unlike ``univariate_cox``); selector-dropped features
        appear with NaN in every value column.

        ``hazard_ratio``/CI/``p_value`` are populated only for models
        whose ``hazard_ratio_valid`` flag is True (currently CoxPH);
        they stay NaN for CoxNet (penalized — shrinkage toward 1, no
        valid SE), FSSVM (ranking — coefficients are not log-hazards),
        and AFT (scale coefficients, not log-hazards). For CoxPH on a
        rank-deficient design (singular Hessian), ``coefficient`` and
        ``hazard_ratio`` are populated while CI/p-value are NaN.

        Inference caveats: p-values and CIs condition on any data-driven
        feature selection and tuning, so they are optimistic relative to
        a pre-specified model (``univariate_cox`` avoids this). The
        multivariable fits use the training split, while
        ``univariate_cox`` uses the full cohort — the p-value columns of
        the two tables are not strictly comparable. When CoxPH's
        automatic ridge fallback engaged at its first rung
        (``alpha_ == 0.01``), CIs/p-values are conditional on that
        regularization strength. If the fallback had to escalate further,
        those columns are NaN rather than conditional: see
        ``CoxPHModel.hazard_ratio_valid``.

        Returns
        -------
        pd.DataFrame
            0-row DataFrame (with full column schema) when no trained
            model exposes coefficients. A model is also omitted when its
            coefficients could not be produced or placed at all, which
            warns with ``MetricFailureWarning``. A model whose labels
            could not be tied to the pipeline's features still contributes
            rows, keyed by its own coefficient index or by position, and
            warns the same way. So a ``MetricFailureWarning`` means one
            model's rows are missing or are not keyed by your feature
            names; silence means every row is.

        See Also
        --------
        ModelResult.coefficients : the per-model accessor.
        univariate_cox : per-feature marginal statistics on the full
            cohort.
        """
        from ..evaluation._coefficients import (
            COEFFICIENT_COLUMNS,
            compute_model_coefficients,
        )
        frames = []
        for name, model_result in self._state.models.items():
            df = compute_model_coefficients(
                model_result.model, warn=False, model_name=name,
            )
            if not df.empty:
                df.insert(0, 'model_type', name)
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=['model_type', *COEFFICIENT_COLUMNS])
        return pd.concat(frames, ignore_index=True)

    def feature_summary(self, model=None, *,
                        correlation_threshold=CORRELATION_THRESHOLD):
        """Combined per-feature summary across all analyses.

        One row per (model, feature), joining everything mlsurv knows
        about each feature: univariate Cox statistics, multivariable
        coefficients with gated inference, permutation importance, mean
        absolute SHAP, importance significance, and the
        correlated-feature flag. This is the same table the HTML
        report's feature-analysis section renders.

        The ``uni_*`` (univariate, unadjusted) and ``mv_*``
        (multivariable, adjusted for the other features in the model)
        columns sit side by side, giving the classic unadjusted-vs-
        adjusted comparison per feature.

        Parameters
        ----------
        model : str, optional
            Trained model storage key (e.g. ``'coxph'``). None returns
            the long form with one block of rows per trained model; a
            name returns only that model's block. Unknown names raise
            ``ValueError``.
        correlation_threshold : float, default=CORRELATION_THRESHOLD
            Absolute pairwise correlation at or above which a feature
            is flagged ``correlated`` (computed on the full cohort,
            train + test).

        Returns
        -------
        pd.DataFrame
            Columns ``model_type``, ``feature``, ``uni_p_value``,
            ``uni_hazard_ratio``, ``uni_hr_lower_95``,
            ``uni_hr_upper_95``, ``mv_coefficient``,
            ``mv_hazard_ratio``, ``mv_hr_lower_95``, ``mv_hr_upper_95``,
            ``mv_p_value``, ``perm_mean``, ``perm_std``,
            ``mean_abs_shap``, ``p_adjusted``, ``sig_stars``,
            ``correlated``. Each model block is sorted by
            ``uni_p_value`` ascending (NaN last), then ``perm_mean``
            descending, then feature name. Columns for analyses that
            have not run are NaN
            (no warning — nothing was lost): ``perm_*`` need
            ``evaluate()`` with ``n_importance_repeats > 0``,
            ``mean_abs_shap`` needs SHAP computation,
            ``p_adjusted``/``sig_stars`` need ``importance_test()`` or
            ``bootstrap()`` with importance. ``df.attrs`` carries
            ``significance_source`` (``{model: 'pimp'|'bootstrap'}``),
            ``significance_method`` (p-adjustment per model), and
            ``correlation_threshold``.

            With no trained models (but ``setup()`` done), returns a
            univariate-only block with ``model_type`` NaN and a
            ``SuggestionWarning``.

        Raises
        ------
        ValueError
            If *model* is not a trained model, or if there is nothing
            to summarize (no models and no univariate statistics).

        Notes
        -----
        - **Gating.** ``mv_hazard_ratio``/CI/``mv_p_value`` are
          populated only for models whose ``hazard_ratio_valid`` flag
          is True (currently CoxPH). They stay NaN for CoxNet
          (penalized), FSSVM (ranking), and AFT (scale coefficients) —
          see :attr:`coefficients`.
        - **Cohort asymmetry.** Univariate Cox is fit on the full
          cohort; multivariable models on the training split;
          permutation importance and SHAP on the test split. The
          ``uni_p_value`` and ``mv_p_value`` columns are an
          unadjusted-vs-adjusted comparison, not a same-cohort one.
        - **Post-selection optimism.** ``mv_p_value`` conditions on
          data-driven feature selection and tuning; ``uni_p_value``
          does not.
        - **Significance semantics differ by source.** PIMP tests
          observed importance against permuted-response null models;
          bootstrap tests importance > 0 across resamples. Check
          ``attrs['significance_source']``.

        See Also
        --------
        coefficients : the multivariable block on its own.
        univariate_cox : the univariate block on its own.
        pimp_importance_test_results : per-model PIMP significance tables.
        """
        from ..evaluation._feature_summary import compute_feature_summary

        if model is not None:
            self._lookup_model_by_name(model)  # ValueError when unknown
            model_names = [model]
        else:
            model_names = list(self._state.models)

        if not model_names:
            univar = self._state.univariate_cox
            if univar is None or len(univar) == 0:
                raise ValueError(
                    "Nothing to summarize: no trained models and no "
                    "univariate Cox statistics. Call setup() and "
                    "train()/run() first."
                )
            warnings.warn(
                "No trained models — feature_summary() returns univariate "
                "Cox columns only. Call train()/run() to add per-model "
                "columns.",
                SuggestionWarning,
                stacklevel=2,
            )

        coefficients = self.coefficients
        if model is not None and not coefficients.empty:
            coefficients = coefficients[coefficients['model_type'] == model]

        eval_result = self.evaluation_result
        importance = (
            eval_result.importance if eval_result is not None else None
        )
        shap_values = (
            eval_result.shap_values if eval_result is not None else None
        )

        return compute_feature_summary(
            model_names,
            coefficients=coefficients,
            univariate_cox=self._state.univariate_cox,
            importance=importance,
            shap_values=shap_values,
            pimp_results=self._state.pimp_importance_test_results,
            bootstrap_result=self.bootstrap_result,
            correlated_features=self._correlated_feature_set(
                correlation_threshold
            ),
            correlation_threshold=correlation_threshold,
        )

    def _correlated_feature_set(self, threshold):
        """Features involved in any high-correlation pair, or None.

        Computed on the full cohort (train + test), matching the
        report's correlation analysis. None (not an empty set) when the
        correlation cannot be computed — the summary's ``correlated``
        column is then all NA (distinguishable from False for
        programmatic users; the report simply shows no markers).
        """
        from ..evaluation.importance import detect_correlated_features

        frames = [
            f for f in (self._state.X_train, self._state.X_test)
            if f is not None and len(f) > 0
        ]
        if not frames:
            return None
        try:
            X_full = pd.concat(frames, ignore_index=True)
            pairs = detect_correlated_features(X_full, threshold=threshold)
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            # Log visibly too — InternalWarning is filtered by default.
            logger.warning(
                "feature_summary: could not compute feature "
                "correlations: %s", exc,
            )
            warnings.warn(
                f"Could not compute feature correlations: {exc}. The "
                f"'correlated' column will be NA.",
                InternalWarning,
                stacklevel=3,
            )
            return None
        return {f for f1, f2, _ in pairs for f in (f1, f2)}

    @property
    def calibration_flagged_models(self):
        """Trained models flagged by the calibration-drift check, minus calibrated ones.

        Read-only, ``evaluate()``-derived. Each ``ModelResult`` carries a
        ``calibration_flagged`` boolean stamped when ``evaluate()`` stores its
        concordance; this property aggregates those flags in ``_state.models``
        insertion order, filtering live against each result's ``calibrator``
        so a model drops off the moment a calibrator is attached (no
        re-evaluation needed). Returns ``[]`` when ``evaluate()`` has not run or
        nothing is flagged. Calibrate a flagged model with
        ``train(model, calibrate='breslow_slope')``.

        Returns
        -------
        list of str
        """
        return [
            name for name, m in self._state.models.items()
            if getattr(m, 'calibration_flagged', False)
            and getattr(m, 'calibrator', None) is None
        ]

    @property
    def model_recommendations(self):
        """Setup-time model recommendations bundle (read-only).

        Computed on every ``setup()`` (the diagnostic always runs), so this
        is populated after a successful ``setup()`` and ``None`` before it.
        The bundle carries:

        - **model-level verdict** — the recommendation tiers
          (``recommended`` / ``cautionary`` / ``not_recommended``) plus
          per-model ``details`` reason strings. PH/linearity are already
          folded into these tiers.
        - **dataset stats** — ``epv``, ``n_samples``, ``n_events``,
          ``n_features``.
        - **feature-level evidence** — the PH/linearity test DataFrames
          (``ph_test_result`` / ``linearity_test_result``; ``None`` when a
          test was skipped) and the violation tuples (``ph_violations`` /
          ``linearity_violations``).

        The DataFrames are the setup-time snapshot — a later
        ``ph_test()`` / ``linearity_test()`` call returns a fresh DataFrame
        but does not rewrite this bundle's snapshot.

        Returns
        -------
        RecommendedModels or None
        """
        return self._state.model_recommendations

    @property
    def benchmark_result(self):
        """Aggregate model comparison from the last ``benchmark()`` call.

        Returns
        -------
        BenchmarkResult or None
            ``None`` before ``benchmark()`` has run.
        """
        return self._state.benchmark_result

    @property
    def logrank_test_results(self):
        """Per-model risk-group log-rank tables from ``logrank_test()``.

        Accumulates one entry per model across ``logrank_test()`` calls;
        re-running on a model replaces only that model's entry. There is
        intentionally no singular accessor — each table is built from one
        model's risk scores, so a singular accessor would silently reflect
        only the latest call. Name the model you want:
        ``learner.logrank_test_results[name]``. Entries are dropped
        when the model is re-trained or re-tuned.

        Each DataFrame carries ``model_type`` as its first column and records
        its provenance in ``.attrs``: ``model_name``, ``n_groups``,
        ``p_adjust``, ``break_type``, ``breaks``, ``correction``.

        Read-only convenience view assembled from ``_state.models``; each table
        rides on its ``ModelResult`` (``result.logrank_test``).

        Returns
        -------
        mappingproxy
            Read-only map of model name to its log-rank DataFrame; empty before
            any ``logrank_test()`` call. Key assignment raises ``TypeError``;
            the returned tables are live objects — refresh via
            ``logrank_test()``, don't mutate them.
        """
        return self._state.logrank_test_results

    @property
    def rmst_test_results(self):
        """Per-model risk-group RMST tables from ``rmst_test()``.

        Accumulates one entry per model across ``rmst_test()`` calls;
        re-running on a model replaces only that model's entry. There is
        intentionally no singular accessor — each table is built from one
        model's risk scores, so a singular accessor would silently reflect
        only the latest call. Name the model you want:
        ``learner.rmst_test_results[name]``. Entries are dropped
        when the model is re-trained or re-tuned.

        Each DataFrame carries ``model_type`` as its first column and records
        its provenance in ``.attrs``: ``model_name``, ``n_groups``, ``tau``,
        ``p_adjust``, ``break_type``, ``breaks``, ``correction``.

        Read-only convenience view assembled from ``_state.models``; each table
        rides on its ``ModelResult`` (``result.rmst_test``).

        Returns
        -------
        mappingproxy
            Read-only map of model name to its RMST DataFrame; empty before any
            ``rmst_test()`` call. Key assignment raises ``TypeError``; the
            returned tables are live objects — refresh via ``rmst_test()``,
            don't mutate them.
        """
        return self._state.rmst_test_results

    @property
    def ph_test_result(self):
        """Proportional hazards test from setup or the last ``ph_test()`` call.

        Returns
        -------
        pandas.DataFrame or None
            Computed during ``setup()`` when ``ph_test=True`` (the default);
            ``None`` if the test was skipped, could not be computed during
            ``setup()`` (the recommendations table then shows the "test
            unavailable" fallback), or ``setup()`` has not run. Refreshed
            whenever ``ph_test()`` is called (e.g. with a non-default
            ``time_transform``). ``df.attrs`` records how the result was
            computed: ``time_transform`` and ``source`` (``'setup'`` or
            ``'method'``).
        """
        return self._state.ph_test_result

    @property
    def linearity_test_result(self):
        """Linearity test from setup or the last ``linearity_test()`` call.

        Returns
        -------
        pandas.DataFrame or None
            Computed during ``setup()`` when ``linearity_test=True`` (the
            default); ``None`` if the test was skipped, could not be
            computed during ``setup()`` (the recommendations table then
            shows the "test unavailable" fallback), or ``setup()`` has not
            run. Refreshed whenever ``linearity_test()`` is called (e.g.
            with a non-default ``n_knots``). ``df.attrs`` records how the
            result was computed: ``n_knots`` and ``source`` (``'setup'``
            or ``'method'``).
        """
        return self._state.linearity_test_result

    @property
    def kang_comparison_result(self):
        """Auto-computed paired Kang model-vs-model comparison.

        Populated by ``evaluate()`` when two or more models are present and
        ``bootstrap=False`` (the default); a bootstrap run uses the paired
        bootstrap comparison instead and leaves this ``None``.

        Returns
        -------
        pandas.DataFrame or None
            ``None`` before a multi-model ``evaluate()`` has run.
        """
        return self._state.kang_comparison_result

    @property
    def pimp_importance_test_results(self):
        """Per-model PIMP significance tables from ``importance_test()``.

        Accumulates one entry per model across
        ``importance_test(method='pimp')`` calls; re-running on a model
        replaces only that model's entry. There is intentionally no singular
        accessor — PIMP is single-model-per-call, so name the model you want:
        ``learner.pimp_importance_test_results[name]``. PIMP requires a model
        trained through the learner; a raw pipeline/estimator is rejected
        (use ``compute_pimp()`` for a one-off analysis). Entries are dropped
        when the model is re-trained or re-tuned.

        Each DataFrame records its provenance in ``.attrs``:
        ``n_permutations``, ``n_importance_repeats``, ``random_state``,
        ``p_adjust``, and ``model_name``.

        Read-only convenience view assembled from ``_state.models``; each table
        rides on its ``ModelResult`` (``result.pimp_importance_test``).

        Returns
        -------
        mappingproxy
            Read-only map of model name to its PIMP DataFrame (columns: feature,
            observed_importance, null_mean, null_std, p_value, p_adjusted);
            empty before any PIMP ``importance_test()`` call. Key assignment
            raises ``TypeError``; the returned tables are live objects —
            refresh via ``importance_test()``, don't mutate them.
        """
        return self._state.pimp_importance_test_results

    @property
    def bootstrap_importance_test_results(self):
        """Per-model bootstrap significance tables from ``importance_test()``.

        Accumulates one entry per model across
        ``importance_test(method='bootstrap')`` calls; re-running on a model
        replaces only that model's entry. Entries are dropped when the model
        is re-trained or re-tuned.

        Each DataFrame records ``p_adjust`` and ``model_name`` in ``.attrs``.

        Read-only convenience view assembled from ``_state.models``; each table
        rides on its ``ModelResult`` (``result.bootstrap_importance_test``).

        Returns
        -------
        mappingproxy
            Read-only map of model name to its bootstrap DataFrame (columns:
            feature, mean, std, ci_lower, ci_upper, p_value, p_adjusted); empty
            before any bootstrap ``importance_test()`` call. Key assignment
            raises ``TypeError``; the returned tables are live objects —
            refresh via ``importance_test()``, don't mutate them.
        """
        return self._state.bootstrap_importance_test_results

    def summary(self, metrics=None):
        """CV metrics comparison table across all trained models.

        Excludes synthetic (benchmark-injected) models.

        Parameters
        ----------
        metrics : list of str, optional
            Metrics to include (default: all available).

        Returns
        -------
        pd.DataFrame
            Columns: [model, metric, phase, point, mean, std, ci_lower, ci_upper]
        """
        all_rows = []
        for model_name, model_result in self._state.models.items():
            if model_result.is_synthetic:
                continue
            ms = model_result.summary(metrics)
            if not ms.empty:
                ms.insert(0, 'model', model_name)
                all_rows.append(ms)
        if not all_rows:
            return pd.DataFrame(
                columns=['model', 'metric', 'phase', 'point',
                         'mean', 'std', 'ci_lower', 'ci_upper'],
            )
        return pd.concat(all_rows, ignore_index=True)

    def _ensure_setup(self):
        """Ensure setup() has been called. Warns and auto-runs with defaults if not."""
        if not self._state.is_setup:
            warnings.warn(
                "setup() was not called explicitly. Running auto-setup with "
                f"default parameters (test_fraction=0.3, scoring='{DEFAULT_SCORING_METRIC}', "
                "random_state=42). Call setup() explicitly to customize.",
                SuggestionWarning,
            )
            self.setup(verbose=False)

    def _get_imputed_X(self, X, *, raise_on_error: bool = False):
        """Return *X* with NaN imputed using the configured imputer.

        Clones and fits ``self._state.imputer`` on *X*.  If no imputer is
        configured or *X* has no NaN, returns *X* unchanged.  Falls back to
        median imputation when no imputer is available.

        Parameters
        ----------
        X : DataFrame or None
            Feature matrix, possibly containing NaN values.
        raise_on_error : bool, default=False
            If True, re-raise imputation exceptions after logging so the
            caller can handle them (e.g. ``ph_test()``).  If False
            (default), return *X* unchanged on failure.
        """
        if X is None or not X.isna().any().any():
            return X

        try:
            imputer = self._state.imputer
            if imputer is not None:
                from sklearn.base import clone
                fitted = clone(imputer).fit(X)
            else:
                from sklearn.impute import SimpleImputer
                logger.warning(
                    "No imputer configured; falling back to median for diagnostics"
                )
                fitted = SimpleImputer(strategy="median").fit(X)

            return pd.DataFrame(
                fitted.transform(X), columns=X.columns, index=X.index,
            )
        except (MemoryError, RecursionError):
            raise
        except (ValueError, TypeError, RuntimeError, ImportError,
                AttributeError, KeyError, np.linalg.LinAlgError) as exc:
            if raise_on_error:
                logger.warning("Imputation failed (%s); re-raising", exc)
                raise
            logger.warning("Imputation failed (%s); proceeding with raw data", exc)
            return X

    def _warn_if_overwriting(self, attr_name: str, result_type: str) -> None:
        """Warn if an existing result will be overwritten."""
        existing = getattr(self._state, attr_name)
        if existing is not None:
            warnings.warn(
                f"Existing {result_type} will be overwritten.",
                SuggestionWarning,
                stacklevel=3,
            )

    def _warn_if_overwriting_model_data(
        self, attr_name: str, result_type: str, model_names=None
    ) -> None:
        """Warn if a targeted model already has *attr_name* data to be overwritten.

        With ``model_names=None`` the whole models namespace is checked. With a
        list of names, only those models are checked, so a call that writes a
        disjoint subset does not warn about models it leaves untouched.
        """
        if model_names is None:
            candidates = self._state.models.values()
        else:
            candidates = [
                self._state.models[name]
                for name in model_names
                if name in self._state.models
            ]
        if any(getattr(m, attr_name, None) is not None for m in candidates):
            warnings.warn(
                f"Existing {result_type} will be overwritten.",
                SuggestionWarning,
                stacklevel=3,
            )

    def _resolve_model_input(self, model):
        """
        Resolve various model input types to a standardized format.

        Handles: dict, list/tuple, None, str, ModelResult, or raw model.

        Returns
        -------
        tuple : (resolved_model, cv_model_order, original_model_result)
            - resolved_model: dict of {name: model} for multi-model, or single model/pipeline
            - cv_model_order: list of model names in CV-sorted order (for display), or None
            - original_model_result: reference to ModelResult (for storing evaluation back), or None
        """
        # Handle dict of models directly (extract .model from ModelResult values)
        if isinstance(model, dict):
            resolved = {
                name: (v.model if hasattr(v, 'model') else v)
                for name, v in model.items()
            }
            cv_model_order = self._cv_sorted_keys(list(resolved.keys()))
            return resolved, cv_model_order, None

        # Handle list/tuple of model names / ModelResults -> multi-model dict
        # (consistent with run()/tune() accepting ["coxph", "coxnet"]).
        if isinstance(model, (list, tuple)):
            if not model:
                raise ValueError(
                    "Empty model list. Pass model names, e.g. "
                    "evaluate(['coxph', 'coxnet']), or omit the argument to "
                    "evaluate all trained models."
                )
            resolved = {}
            for element in model:
                mr = self._resolve_model_list_element(element)
                resolved[mr.name] = mr.model
            cv_model_order = self._cv_sorted_keys(list(resolved.keys()))
            return resolved, cv_model_order, None

        # Handle None -> get all trained models
        if model is None:
            return self._resolve_all_models()

        # Handle string model name -> lookup from _state.models
        if isinstance(model, str):
            model = self._lookup_model_by_name(model)

        # Handle ModelResult -> extract .model and save reference
        original_model_result = None
        if hasattr(model, 'model'):
            original_model_result = model
            model = model.model

        return model, None, original_model_result

    def _cv_sorted_keys(self, model_keys):
        """Return ``model_keys`` reordered best-first by CV metric.

        Thin wrapper around :func:`mlsurv._shared.cv_sorted_model_keys`.
        """
        from .._shared import cv_sorted_model_keys
        return cv_sorted_model_keys(self._state.models or {}, model_keys=model_keys)

    def _resolve_all_models(self):
        """Get all trained models sorted by CV metric (best first)."""
        if not self._state.models:
            raise ValueError(
                "No trained models. Call train() or run() first. "
                "Example: learner.train('coxph') or learner.run() for all models."
            )
        cv_model_order = self._cv_sorted_keys(list(self._state.models.keys()))
        models = {name: self._state.models[name].model for name in cv_model_order}
        return models, cv_model_order, None

    def _display_name_map(self):
        """Storage-key → display-name map.

        Pass to ``resolve_display_name`` / ``print_multi_model_*``.
        """
        from ..models import get_model_display_names
        return get_model_display_names()

    def _lookup_model_by_name(self, model_name):
        """Lookup model from _state.models by its storage key."""
        if model_name in self._state.models:
            return self._state.models[model_name]
        from ..models import list_models
        trained = list(self._state.models)
        available = list_models(include_unavailable=False)
        raise ValueError(
            f"Model '{model_name}' not found. "
            f"Trained: {trained}. "
            f"All available: {available}"
        )

    def _warn_if_stale_model_result(self, passed, registered, name):
        """Warn when *passed* is not the registered ModelResult for *name*.

        Call only from directly inside :meth:`_resolve_models_multi`
        (itself called directly by the public multi-model methods
        ``predict``/``importance``/``rmst_test``/``logrank_test``):
        ``stacklevel=4`` walks helper -> ``_resolve_models_multi`` ->
        public method -> user code.
        """
        if passed is registered:
            return
        warnings.warn(
            f"The ModelResult passed for '{name}' is not the model "
            f"currently registered under that name (it was likely "
            f"replaced by tune() or a re-train); computing on the "
            f"registered model. Re-fetch the current result via "
            f"learner.models['{name}'].",
            SuggestionWarning, stacklevel=4,
        )

    def _resolve_model_list_element(self, element):
        """Resolve one element of a list/tuple model selection to a ``ModelResult``.

        Shared by :meth:`_resolve_model_input`,
        :meth:`_resolve_models_multi`, and
        ``_evaluation_helpers._resolve_eval_model_names`` so all resolvers
        validate elements identically. A list/tuple is the multi-model
        selection form, so each element must be a model name (``str``) or a
        ``ModelResult`` (anything exposing ``.model`` and ``.name``); a raw
        model/pipeline is rejected with a clear ``ValueError`` rather than
        crashing later with ``AttributeError``. This is intentionally
        duck-typed and raises ``ValueError`` (not ``TypeError`` like the
        stricter :meth:`_resolve_model_results`), a contract the tests
        pin via the ``"raw models/pipelines"`` message.
        """
        if isinstance(element, str):
            return self._lookup_model_by_name(element)
        if hasattr(element, 'model') and hasattr(element, 'name'):
            return element
        raise ValueError(
            "List/tuple model selection must contain model names (str) or "
            "ModelResult objects, not raw models/pipelines "
            f"(got {type(element).__name__}). To evaluate a single raw model, "
            "pass it directly: evaluate(model)."
        )

    # -- Unified model resolution helpers ------------------------------------

    def _resolve_single_model(self, model_name):
        """Resolve an optional model name to a single ModelResult.

        - ``None`` + exactly one model trained -> return it.
        - ``None`` + multiple models -> raise ``ValueError``.
        - ``str`` -> delegate to :meth:`_lookup_model_by_name`.

        Parameters
        ----------
        model_name : str or None
        """
        models = self._state.models
        if not models:
            raise ValueError(
                "No trained models. Call train() or run() first."
            )
        if model_name is None:
            if len(models) == 1:
                return next(iter(models.values()))
            raise ValueError(
                f"Multiple models trained ({list(models)}). "
                "Pass model='...' to select one."
            )
        return self._lookup_model_by_name(model_name)

    def _resolve_model_names(self, models, *, filter_fn=None):
        """Resolve a flexible *models* argument to a list of name strings.

        - ``None`` -> all model names from ``_state.models`` (optionally
          filtered by *filter_fn*).
        - ``str`` -> ``[str]``.
        - ``ModelResult`` -> ``[result.name]``.
        - ``list`` -> ``list(models)`` (verbatim; element types are not
          validated -- prefer :meth:`_resolve_model_results` if you
          need uniform output types).

        Parameters
        ----------
        models : str, list[str], ModelResult, or None
        filter_fn : callable, optional
            ``(key: str, value: ModelResult) -> bool``.  Applied only when
            *models* is ``None``.
        """
        if models is None:
            candidates = self._state.models
            if filter_fn is not None:
                return [k for k, v in candidates.items() if filter_fn(k, v)]
            return list(candidates.keys())
        if isinstance(models, str):
            return [models]
        # Stricter than ``hasattr(models, 'name')`` -- a duck-typed object
        # (Series, Path, Enum, ...) would otherwise leak a stray ``.name``
        # value into the lookup, producing a confusing "model not found"
        # error downstream rather than a clean type mismatch.
        if isinstance(models, ModelResult):
            return [models.name]
        return list(models)

    def _resolve_model_results(self, models, *, filter_fn=None):
        """Resolve a flexible *models* argument to a list of ``ModelResult``.

        Returns stored ``ModelResult`` objects so downstream helpers
        never have to re-resolve by string name -- passing the object
        through avoids the fragile name-based re-lookup that previously
        broke when a storage key and algorithm identity diverged.

        Accepted inputs:
            * ``None`` -- all trained models, optionally filtered by
              *filter_fn*.
            * ``str`` -- one model by storage key via
              :meth:`_lookup_model_by_name`.
            * ``ModelResult`` -- looked up by ``.name``.
            * ``list`` -- dispatched element-wise; each element must be
              ``str`` or ``ModelResult``. (Note: this is stricter than
              :meth:`_resolve_model_names`, which returns lists
              verbatim; we want a uniform ``list[ModelResult]`` out.)
            * ``[]`` -- returns ``[]`` (caller intent).

        Raises:
            * ``ValueError`` -- *models* is ``None`` and no models have
              been trained. Matches :meth:`_resolve_all_models` to fail
              loudly rather than silently no-op. An empty *filter_fn*
              result on a populated learner returns ``[]``; that's
              caller intent, not an error.
            * ``TypeError`` -- *models* (or a list element) is neither
              a ``str`` nor a ``ModelResult``. Stricter than the
              sibling's duck-typed ``hasattr(.., 'name')`` check so a
              user accidentally passing a ``Series``/``Path``/``Enum``
              gets a clean type error rather than a confusing
              "model not found" downstream.
        """
        if models is None:
            if not self._state.models:
                raise ValueError(
                    "No trained models found. Train at least one model "
                    "before calling this helper, or pass an explicit "
                    "model name / ModelResult."
                )
            names = self._resolve_model_names(None, filter_fn=filter_fn)
            return [self._lookup_model_by_name(n) for n in names]
        if isinstance(models, str):
            return [self._lookup_model_by_name(models)]
        if isinstance(models, ModelResult):
            return [self._lookup_model_by_name(models.name)]
        if not isinstance(models, (list, tuple)):
            raise TypeError(
                f"Unsupported models argument: {type(models).__name__}; "
                "expected None, str, ModelResult, or list/tuple of those."
            )
        # list/tuple — dispatch element-wise.
        out = []
        for m in models:
            if isinstance(m, str):
                out.append(self._lookup_model_by_name(m))
            elif isinstance(m, ModelResult):
                out.append(self._lookup_model_by_name(m.name))
            else:
                raise TypeError(
                    "Unsupported element in models list: "
                    f"{type(m).__name__}; expected str or ModelResult."
                )
        return out

    def _resolve_models_multi(self, models):
        """Resolve a flexible *models* selection for multi-model methods.

        Used by methods whose result shape depends on the selection arity
        (``predict``, ``importance``, ``rmst_test``, ``logrank_test``).
        Single-form inputs (``str``, ``ModelResult``, or ``None`` with
        exactly one trained model) keep the calling method's single-model
        contract; collection inputs (``list``/``tuple``/``dict``) and
        ``None`` with several trained models select a model set.

        All returned objects are the registered ``ModelResult``s from
        ``_state.models`` (resolved once at this public boundary), so
        downstream workers can attach per-model results directly without
        name-based re-lookup. A passed ``ModelResult`` (direct, list
        element, or dict value) that is not the registered object under
        its name (the dict key, for dict values) emits a
        ``SuggestionWarning`` and the registered model is used in its
        place.

        Parameters
        ----------
        models : str, ModelResult, list, tuple, dict, or None

        Returns
        -------
        tuple of (list[ModelResult], bool)
            The resolved models in CV-sorted order (best first) and a
            ``single_form`` flag — ``True`` when the caller should return
            a single-model result rather than a per-model collection.

        Raises
        ------
        ValueError
            For an empty list/dict, an unknown model name, ``None`` with
            no trained models, or a raw model/pipeline inside a
            list/tuple selection (``_resolve_model_list_element``'s
            contract).
        TypeError
            For a raw model/pipeline (or any other unsupported type)
            passed directly as ``models`` — these methods need the
            registered ``ModelResult`` for per-model storage and
            recalibrator lookup.
        """
        if models is None:
            if not self._state.models:
                raise ValueError(
                    "No trained models. Call train() or run() first. "
                    "Example: learner.train('coxph') or learner.run() "
                    "for all models."
                )
            names = self._cv_sorted_keys(list(self._state.models.keys()))
            resolved = [self._state.models[n] for n in names]
            return resolved, len(resolved) == 1
        if isinstance(models, str):
            return [self._lookup_model_by_name(models)], True
        if isinstance(models, ModelResult):
            registered = self._lookup_model_by_name(models.name)
            self._warn_if_stale_model_result(models, registered, models.name)
            return [registered], True
        if isinstance(models, dict):
            if not models:
                raise ValueError(
                    "Empty model dict. Pass a dict of trained models, e.g. "
                    "learner.models, or omit the argument to use all "
                    "trained models."
                )
            by_name = {}
            for key, value in models.items():
                registered = self._lookup_model_by_name(key)
                # Duck-typed like _resolve_model_list_element so the dict
                # and list forms warn on the same objects.
                if hasattr(value, 'model') and hasattr(value, 'name'):
                    self._warn_if_stale_model_result(value, registered, key)
                by_name[key] = registered
            names = self._cv_sorted_keys(list(by_name.keys()))
            return [by_name[n] for n in names], False
        if isinstance(models, (list, tuple)):
            if not models:
                raise ValueError(
                    "Empty model list. Pass model names, e.g. "
                    "['coxph', 'coxnet'], or omit the argument to use all "
                    "trained models."
                )
            by_name = {}
            checked = set()
            for element in models:
                mr = self._resolve_model_list_element(element)
                registered = self._lookup_model_by_name(mr.name)
                # Check every element, not just the first per name —
                # otherwise [registered, stale] would silently drop the
                # stale duplicate. The id() set keeps the warning at one
                # per distinct passed object.
                if id(mr) not in checked:
                    checked.add(id(mr))
                    self._warn_if_stale_model_result(mr, registered, mr.name)
                by_name[mr.name] = registered
            names = self._cv_sorted_keys(list(by_name.keys()))
            return [by_name[n] for n in names], False
        raise TypeError(
            f"Unsupported models argument: {type(models).__name__}; "
            "expected a model name (str), ModelResult, list/dict of those, "
            "or None for all trained models. Raw models/pipelines are not "
            "supported here."
        )
