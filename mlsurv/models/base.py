"""Base class for all mlsurv survival models."""

import itertools
import warnings
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sksurv.base import SurvivalAnalysisMixin
from sksurv.metrics import concordance_index_censored
import numpy as np
import pandas as pd

from ..utils import SurvivalArray
from .._warnings import SuggestionWarning


class BaseSurvivalModel(BaseEstimator, SurvivalAnalysisMixin, ABC):
    """
    Abstract base class for all mlsurv survival models.

    All models must implement fit() and predict(). Other methods have
    default implementations that can be overridden.

    Inherits from sklearn's BaseEstimator for get_params(), set_params(),
    clone() support and scikit-survival's SurvivalAnalysisMixin.

    Return Type Conventions
    -----------------------
    predict_survival_function() and predict_cumulative_hazard_function() have
    polymorphic return types depending on the model backend:

    - **sksurv models** (CoxPH, CoxNet, RSF, GBSA): return ``list of StepFunction``
      or ``ndarray`` when ``return_array=True``.
    - **Deep learning models** (DeepSurv, DeepHit): return ``pd.DataFrame`` with
      time index and sample columns, or ``ndarray`` when ``return_array=True``.
    - **XGBCox**: returns ``list of StepFunction`` (via Breslow estimator)
      or ``ndarray`` when ``return_array=True``.
    - **XGBAFTModel**: returns ``pd.DataFrame`` with time index and sample
      columns, or ``ndarray`` when ``return_array=True`` or ``times`` is provided.

    The evaluation module's ``_surv_funcs_to_probs()`` normalizes all formats
    to a common representation for metric computation.

    Class Attributes
    ----------------
    name : str
        Short identifier for the model type (e.g., 'coxph', 'rsf').
        Must be set by each subclass. Used for result labeling in
        DataFrames (model_type column).
    display_name : str
        Human-readable name for reports/visualizations (e.g., 'CoxPH', 'RSF').
        If None, falls back to name.upper().
    color : str
        Hex color code for visualizations (e.g., '#1f77b4').
        Default is gray ('#7f7f7f').
    supports_random_state : bool
        Whether the model accepts a random_state parameter for reproducibility.
        Used by SurvivalLearner to auto-inject random seeds.
    prefer_sequential_cv : bool
        Whether CV should run sequentially (disable joblib parallelization).
        True for fast models where joblib overhead dominates, or GPU models.
    has_internal_parallelism : bool
        Whether the model has its own n_jobs for internal parallelism (e.g., RSF, XGBCox).
        Controls bootstrap backend selection (threading for heavy models, loky for lightweight)
        and importance parallelism strategy in evaluate().
    complexity : str
        Data-size complexity tier used by ``recommend_models()`` to determine
        sample-size thresholds.  One of ``'linear'``, ``'tree_ensemble'``, or
        ``'deep_learning'``.  Default is ``'linear'``.
    assumes_ph : bool
        Whether the model fits the Cox partial likelihood (or otherwise
        produces a single time-invariant risk score whose hazard ratios
        between subjects are constant by construction). Used by the
        limitations/guidance sections to flag affected models when the
        PH assumption is violated. Default ``False``.
    assumes_linearity : bool
        Whether the model assumes log-hazard linearity in continuous
        covariates (Harrell 2015 §4.6, §5.5). Used by the limitations
        section to flag affected models when linearity is violated.
        Default ``False``.

    Attributes (set after fit)
    --------------------------
    is_fitted_ : bool
        Whether the model has been fitted.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : array-like
        Feature names seen during fit (if X is DataFrame).

    Attributes (set after tuning)
    -----------------------------
    best_params_ : dict, optional
        Best hyperparameters found by Optuna tuning.
    tuning_study_ : str, optional
        Storage URL for the Optuna study (for dashboard access).
    """

    # Model identifier - MUST be overridden by each subclass
    name: str = None

    # Model metadata (defaults for backward compatibility)
    display_name: str = None      # Falls back to name.upper() if None
    color: str = "#7f7f7f"        # Default gray
    supports_random_state: bool = False
    prefer_sequential_cv: bool = False
    has_internal_parallelism: bool = False
    complexity: str = "linear"  # "linear", "tree_ensemble", or "deep_learning"
    assumes_ph: bool = False        # Cox proportional hazards
    assumes_linearity: bool = False # Log-hazard linear in continuous covariates
    rank_based: bool = False        # True for models whose raw decision values
                                    # are ordinal only (not log-hazards). Such
                                    # models need post-hoc recalibration before
                                    # survival probabilities are interpretable.
    score_based: bool = False       # True when the model reduces each subject to
                                    # a single linear-predictor risk score, so a
                                    # Breslow baseline can be refit on those
                                    # scores. Required for the breslow_slope
                                    # calibrator (train(calibrate=...)).
    hazard_ratio_valid: bool = False  # True when exp(coef) is a multivariable
                                      # hazard ratio with valid Wald inference
                                      # (unpenalized PH partial likelihood).
                                      # Only CoxPH qualifies; penalized (CoxNet)
                                      # and ranking (FSSVM) coefficients are
                                      # not HRs, and AFT coefficients are not
                                      # log-hazards.

    # =========================================================================
    # REQUIRED METHODS (must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def fit(self, X, y: SurvivalArray):
        """
        Fit the survival model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features. Can be numpy array or pandas DataFrame.
        y : SurvivalArray of shape (n_samples,)
            Survival outcome with dtype=[('event', bool), ('time', float)].
            - event: True if the event occurred, False if censored
            - time: Observed time (to event or censoring)

        Returns
        -------
        self : object
            Fitted estimator (for method chaining).
        """
        pass

    def predict(self, X):
        """
        Predict risk scores for samples.

        Higher scores indicate higher risk of event (shorter survival).
        This is the standard convention in scikit-survival.

        Default implementation delegates to ``self.model_.predict(X)``.
        Override for models with custom prediction logic (FSSVM, WeibullAFT,
        DeepSurv, DeepHit, etc.).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        risk_scores : ndarray of shape (n_samples,)
            Predicted risk scores. NOT survival probabilities.
        """
        self._check_is_fitted()
        return self.model_.predict(X)

    # =========================================================================
    # DEFAULT IMPLEMENTATIONS (can be overridden)
    # =========================================================================

    def score(self, X, y: SurvivalArray):
        """
        Return the concordance index (C-index) on the given test data.

        The C-index measures the model's ability to correctly rank
        pairs of samples by their risk. Range: 0.5 (random) to 1.0 (perfect).

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : SurvivalArray
            Survival outcomes.

        Returns
        -------
        float
            Concordance index.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the model has not been fitted.
        """
        self._check_is_fitted()
        predictions = self.predict(X)
        result = concordance_index_censored(
            y['event'], y['time'], predictions
        )
        return result[0]

    def predict_survival_function(self, X, return_array=False):
        """
        Predict survival function for each sample.

        Default implementation delegates to ``self.model_.predict_survival_function()``.
        Override for models with custom logic (FSSVM, WeibullAFT, DeepSurv, etc.).

        Parameters
        ----------
        X : array-like
            Feature matrix.
        return_array : bool, default=False
            If True, return ndarray (samples x times) instead of StepFunction objects.

        Returns
        -------
        list of StepFunction, pd.DataFrame, or ndarray
            Survival functions for each sample. Return type depends on model:

            - **sksurv models**: ``list of StepFunction``, or ``ndarray``
              (samples x times) when ``return_array=True``.
            - **Deep learning models**: ``pd.DataFrame`` (times x samples),
              or ``ndarray`` (samples x times) when ``return_array=True``.

        Raises
        ------
        NotImplementedError
            If the model does not support survival function prediction.
        """
        self._check_is_fitted()
        if not hasattr(self, 'model_') or self.model_ is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support predict_survival_function(). "
                "Use predict() for risk scores instead."
            )
        result = self.model_.predict_survival_function(X, return_array=return_array)
        if return_array:
            self._warn_if_large_array(result)
        return result

    def predict_cumulative_hazard_function(self, X, return_array=False):
        """
        Predict cumulative hazard function for each sample.

        Default implementation delegates to ``self.model_.predict_cumulative_hazard_function()``.
        Override for models with custom logic (FSSVM, WeibullAFT, DeepSurv, etc.).

        Parameters
        ----------
        X : array-like
            Feature matrix.
        return_array : bool, default=False
            If True, return ndarray (samples x times) instead of StepFunction objects.

        Returns
        -------
        list of StepFunction, pd.DataFrame, or ndarray
            Cumulative hazard functions for each sample. Return type depends on model:

            - **sksurv models**: ``list of StepFunction``, or ``ndarray``
              (samples x times) when ``return_array=True``.
            - **Deep learning models**: ``pd.DataFrame`` (times x samples),
              or ``ndarray`` (samples x times) when ``return_array=True``.

        Raises
        ------
        NotImplementedError
            If the model does not support cumulative hazard prediction.
        """
        self._check_is_fitted()
        if not hasattr(self, 'model_') or self.model_ is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support predict_cumulative_hazard_function()."
            )
        result = self.model_.predict_cumulative_hazard_function(X, return_array=return_array)
        if return_array:
            self._warn_if_large_array(result)
        return result

    # =========================================================================
    # OPTIONAL METHODS (for specific model types)
    # =========================================================================

    def get_coefficients(self):
        """
        Return model coefficients (for linear models: CoxPH, CoxNet).

        Returns
        -------
        pd.Series or None
            Coefficients indexed by feature name, or None if not available.
        """
        return None

    def get_feature_importance(self, method='permutation', X=None, y=None, n_repeats=100,
                                random_state=None):
        """
        Compute feature importance.

        Parameters
        ----------
        method : str, default='permutation'
            - 'permutation': Permutation importance (works for all models)
            - 'coefficient': Absolute coefficient values (linear models only)
            - 'shap': SHAP values (requires shap package)
        X : array-like, optional
            Feature matrix (required for permutation/shap).
        y : SurvivalArray, optional
            Survival outcomes (required for permutation).
        n_repeats : int, default=100
            Number of permutation repeats.
        random_state : int or None, default=None
            Random seed for reproducibility. If None and the model supports
            random_state, uses the model's random_state.

        Returns
        -------
        pd.DataFrame
            Feature importance with columns ['feature', 'importance', 'std'].

        Raises
        ------
        ValueError
            If ``method='coefficient'`` is used on a model without
            coefficients, if ``X``/``y`` are missing for permutation
            importance, or if ``method`` is unrecognized.
        """
        if method == 'coefficient':
            coef = self.get_coefficients()
            if coef is None:
                raise ValueError(f"{self.__class__.__name__} has no coefficients")
            return pd.DataFrame({
                'feature': coef.index,
                'importance': coef.abs().values,
                'std': np.zeros(len(coef))
            }).sort_values('importance', ascending=False)

        elif method == 'permutation':
            if X is None or y is None:
                raise ValueError("X and y required for permutation importance")
            from sklearn.inspection import permutation_importance
            # Use provided random_state, fall back to model's random_state, then None
            seed = random_state if random_state is not None else self._get_random_state()
            result = permutation_importance(
                self, X, y, n_repeats=n_repeats, random_state=seed
            )
            feature_names = X.columns if hasattr(X, 'columns') else range(X.shape[1])
            return pd.DataFrame({
                'feature': feature_names,
                'importance': result.importances_mean,
                'std': result.importances_std
            }).sort_values('importance', ascending=False)

        else:
            raise ValueError(f"Unknown method: {method}")

    def get_search_space(self) -> dict:
        """
        Return an Optuna search space to make this model tunable.

        Override in a custom subclass to enable hyperparameter tuning via
        ``learner.train(model, tune=True)`` / ``learner.tune(model)``. The
        default returns an empty dict, meaning the model is not tunable
        unless a space is supplied per call via ``search_spaces=``.

        Each entry maps a constructor parameter name (one accepted by the
        model's ``set_params``) to a distribution:

        - ``(min, max)``        : integer range if both bounds are ints,
          otherwise a float range
        - ``(min, max, "log")`` : log-scale float range
        - ``[a, b, c]``         : categorical choices

        This is the same format used by ``DEFAULT_SPACES`` and the
        ``search_spaces=`` keyword. When both a declared space and a per-call
        ``search_spaces={'model': {...}}`` are present, the per-call dict wins.

        Returns
        -------
        dict
            Mapping of parameter name to distribution. An empty dict (the
            default) means no declared search space.
        """
        return {}

    def _impurity_importance(self):
        """Return impurity-based feature importance as a DataFrame.

        Shared helper for tree-based models (RSF, GBSA) that have
        ``self.model_.feature_importances_``.

        Returns
        -------
        pd.DataFrame
            Columns: ``['feature', 'importance', 'std']``, sorted descending.
        """
        self._check_is_fitted()
        importances = self.model_.feature_importances_
        features = (
            self.feature_names_in_ if hasattr(self, 'feature_names_in_')
            else range(len(importances))
        )
        return pd.DataFrame({
            'feature': features,
            'importance': importances,
            'std': np.zeros(len(importances))
        }).sort_values('importance', ascending=False)

    @staticmethod
    def _resolve_device(device):
        """Resolve device string to torch.device.

        Parameters
        ----------
        device : str or None
            Device specification: 'auto', 'cpu', 'cuda', 'cuda:0', etc.
            If None or 'auto', selects CUDA when available, else CPU.

        Returns
        -------
        torch.device

        Raises
        ------
        ValueError
            If the device string is invalid or requests unavailable hardware.
        """
        import torch
        if device is None or device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            resolved = torch.device(device)
        except RuntimeError as exc:
            raise ValueError(
                f"Invalid device: {device!r}. "
                f"Use 'auto', 'cpu', 'cuda', or 'cuda:N'."
            ) from exc
        if resolved.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError(
                f"Device {device!r} requested but CUDA is not available. "
                f"Use device='cpu' or install CUDA-enabled PyTorch."
            )
        return resolved

    @staticmethod
    def _set_torch_seed(random_state, device=None):
        """Set all random seeds for PyTorch reproducibility.

        Shared by DeepSurv and DeepHit.

        Deliberately does NOT call ``np.random.seed`` — that resets the
        process-global NumPy RNG and would silently disturb the caller's
        randomness on every fit (including each CV fold and bootstrap
        resample). It is not needed here: the early-stopping split passes
        ``random_state`` explicitly, the stratified batch sampler uses a
        local ``np.random.RandomState``, and torch weight init is seeded by
        ``torch.manual_seed``.

        Parameters
        ----------
        random_state : int or None
            Seed value.  No-op if None.
        device : torch.device or None
            Target device. CUDA seeds are set when device is CUDA, or
            when device is None and CUDA is available (backward compat).
        """
        if random_state is None:
            return
        import random
        import torch
        random.seed(random_state)
        torch.manual_seed(random_state)
        use_cuda = (
            (device is not None and device.type == 'cuda')
            or (device is None and torch.cuda.is_available())
        )
        if use_cuda:
            torch.cuda.manual_seed_all(random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def _validate_early_stopping_viability(
        y, val_fraction, *, param_name='val_fraction', min_train_events=0
    ):
        """Check if dataset supports early stopping with a validation split.

        Validates ``val_fraction`` and computes split sizes. Does NOT emit
        warnings — callers control their own warning text and stacklevel.

        Parameters
        ----------
        y : SurvivalArray
            Survival outcomes.
        val_fraction : float
            Fraction of data to hold out for validation.
        param_name : str, default='val_fraction'
            Name used in the ValueError message for invalid fractions.
        min_train_events : int, default=0
            Minimum number of events expected in the training split.
            DeepSurv sets this to account for StratifiedSurvivalSampler.

        Returns
        -------
        viable : bool
            True if the dataset is large enough for the requested split.
        info : dict
            Keys: n, n_val, n_train, n_events, n_censored, est_train_events.
        """
        if not (0.0 < val_fraction < 1.0):
            raise ValueError(
                f"{param_name} must be in (0.0, 1.0), got {val_fraction}"
            )
        n = len(y)
        n_val = max(1, int(n * val_fraction))
        n_train = n - n_val
        n_events = int(y['event'].sum())
        n_censored = n - n_events
        est_train_events = n_events - max(1, int(n_events * val_fraction))
        info = dict(
            n=n, n_val=n_val, n_train=n_train,
            n_events=n_events, n_censored=n_censored,
            est_train_events=est_train_events,
        )
        if n_val < 2 or n_train < 2 or min(n_events, n_censored) < 2:
            return False, info
        if min_train_events > 0 and est_train_events < min_train_events:
            return False, info
        return True, info

    @staticmethod
    def _take_rows(X, idx):
        """Select rows by integer index from a DataFrame or ndarray."""
        return X.iloc[idx] if hasattr(X, 'iloc') else X[idx]

    def _stratified_es_split(self, y, validation_fraction):
        """Carve a stratified train/validation split for early stopping.

        Validates ``validation_fraction`` and the dataset size via
        ``_validate_early_stopping_viability``, then splits indices
        stratified on event status. Returns ``(fit_idx, val_idx)``, or
        ``(None, None)`` with a ``SuggestionWarning`` when the data is too
        small or stratification is not possible. Requires ``self.random_state``.

        Parameters
        ----------
        y : SurvivalArray
            Survival outcomes.
        validation_fraction : float
            Fraction held out for validation.

        Returns
        -------
        fit_idx, val_idx : ndarray or None
            Training and validation row indices, or ``(None, None)``.
        """
        from sklearn.model_selection import train_test_split

        viable, info = self._validate_early_stopping_viability(
            y, validation_fraction, param_name='validation_fraction',
        )
        if not viable:
            warnings.warn(
                f"Dataset too small (n={info['n']}, events={info['n_events']}) "
                f"for early stopping with validation_fraction="
                f"{validation_fraction}. Disabling early stopping.",
                SuggestionWarning,
                stacklevel=3,
            )
            return None, None
        try:
            return train_test_split(
                np.arange(len(y)),
                test_size=validation_fraction,
                stratify=y['event'],
                random_state=self.random_state,
            )
        except ValueError as exc:
            if "least populated class" not in str(exc):
                raise
            warnings.warn(
                f"Stratified validation split failed (n={info['n']}, "
                f"events={info['n_events']}). Disabling early stopping.",
                SuggestionWarning,
                stacklevel=3,
            )
            return None, None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _validate_data(self, X, y=None):
        """
        Validate and record input data properties.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : SurvivalArray, optional
            Survival outcomes.
        """
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        self.n_features_in_ = X.shape[1]

    def _check_is_fitted(self):
        """
        Raise error if model is not fitted.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the model has not been fitted.
        """
        if not getattr(self, 'is_fitted_', False):
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    def freeze_for_resampling(self):
        """Return an unfitted clone with all tuned hyperparameters pinned.

        Used by the bootstrap loop to ensure every resample refits with the
        *same* hyperparameters as the point-estimate model. Default behavior
        is ``sklearn.base.clone(self)`` — which preserves every ``__init__``
        param via ``get_params()``. Override in wrappers that perform
        internal selection during ``fit()`` (e.g. CoxNet's alpha path /
        CV-based selection) so the resample cannot re-select.
        """
        return clone(self)

    def _get_random_state(self):
        """
        Get the model's random_state if it supports one.

        Returns
        -------
        int or None
            The random_state value, or None if the model doesn't support it.
        """
        if self.supports_random_state:
            return getattr(self, 'random_state', None)
        return None

    @staticmethod
    def _warn_if_large_array(result, threshold_mb=50):
        """Warn if a return_array=True result is large.

        Parameters
        ----------
        result : ndarray
            The array to check.
        threshold_mb : int, default=50
            Size threshold in MB above which to warn.
        """
        if isinstance(result, np.ndarray):
            estimated_mb = result.nbytes / (1024 ** 2)
            if estimated_mb > threshold_mb:
                warnings.warn(
                    f"return_array=True created {estimated_mb:.1f}MB array "
                    f"({result.shape[0]} samples x {result.shape[1]} times). "
                    "Consider using return_array=False for memory efficiency.",
                    SuggestionWarning,
                )

    def __repr__(self):
        """Return string representation."""
        params = self.get_params(deep=False)
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"


# Moving a module to an absent device does not raise one consistent type: a
# CPU-only torch build asserts ("Torch not compiled with CUDA enabled"), while
# a CUDA build raises RuntimeError for a bad ordinal, OOM, or driver mismatch.
_DEVICE_MOVE_ERRORS = (RuntimeError, AssertionError)


def _has_non_cpu_params(net):
    """Return True if any parameter or buffer of *net* lives off the CPU.

    Named (rather than inlined) so tests can simulate a GPU-resident net on a
    CPU-only machine.
    """
    if net is None or not (hasattr(net, 'parameters') and hasattr(net, 'buffers')):
        # Not a torch Module, so there is nothing to move. Anything else the
        # iteration raises is a real problem and propagates.
        return False
    return any(
        t.device.type != 'cpu'
        for t in itertools.chain(net.parameters(), net.buffers())
    )


class PycoxSerializationMixin:
    """Mixin for pycox-based models (DeepSurv, DeepHit) to handle pickling.

    Pycox stores training callbacks, metrics, and optimizer references that
    contain lambdas and DataLoader references. These are transient training
    artifacts not needed for prediction; this mixin strips them to allow
    joblib pickling.

    Torch tensors remember the device they were saved from, and a plain
    ``joblib.load`` offers no way to remap them, so a GPU-trained model would
    otherwise be unloadable on a CPU-only host. To keep saved models portable,
    ``__getstate__`` writes CPU tensors and ``__setstate__`` puts the model back
    on whichever device is actually available.

    Subclasses must:
    - Store the fitted pycox model as ``self.model_``.
    - Define ``self.weight_decay`` and ``self.gradient_clip`` in ``__init__``
      (required by ``_validate_pycox_params``).
    """

    _PYCOX_TRAINING_ATTRS = (
        'callbacks', 'train_metrics', 'val_metrics', 'log',
        'batch_metrics', 'batch_loss',
    )

    def _validate_pycox_params(self):
        """Validate weight_decay and gradient_clip common to pycox models."""
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError(
                f"gradient_clip must be > 0 or None, got {self.gradient_clip}"
            )

    def __getstate__(self):
        import copy

        state = self.__dict__.copy()
        model = state.get('model_')
        model_copy = None
        if model is not None:
            try:
                model_copy = copy.copy(model)
            except Exception as exc:
                raise TypeError(
                    f"Failed to serialize {type(self).__name__}: "
                    f"copy.copy() on pycox model failed "
                    f"({type(exc).__name__}: {exc}). "
                    f"This may indicate an incompatible pycox/torchtuples "
                    f"version."
                ) from exc
            for attr in self._PYCOX_TRAINING_ATTRS:
                if hasattr(model_copy, attr):
                    setattr(model_copy, attr, None)
            if hasattr(model_copy, '_optimizer'):
                model_copy._optimizer = None
            state['model_'] = model_copy

        # Write CPU tensors so the pickle loads on any host. Skipped entirely
        # for a CPU-trained net (the common case), which keeps that path
        # byte-for-byte what it was.
        net = state.get('net_')
        if net is None and model_copy is not None:
            net = getattr(model_copy, 'net', None)
        if _has_non_cpu_params(net):
            import torch

            # deepcopy first: copy.copy(model) above is shallow, so
            # model_copy.net is still the live net_, and Module.to() moves
            # parameters in place. The copy is allocated on the training device
            # before being moved, so peak GPU memory for the net doubles here;
            # on a memory-tight GPU that turns save_learner() into an OOM. Fine
            # for the MLPs these wrappers build, worth knowing for bigger nets.
            net_cpu = copy.deepcopy(net).to('cpu')
            if 'net_' in state:
                state['net_'] = net_cpu
            if model_copy is not None:
                # Rewiring .net is what actually keeps GPU tensors out of the
                # pickle, so it must not depend on set_device() being present.
                # It also preserves the live invariant that model_.net is net_.
                model_copy.net = net_cpu
                if hasattr(model_copy, 'set_device'):
                    model_copy.set_device(torch.device('cpu'))
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._restore_device()

    def _restore_device(self):
        """Put a just-unpickled model back on an available device.

        ``__getstate__`` stores CPU tensors, so a model trained on a GPU has to
        be moved back after loading. ``self.device`` records what the user asked
        for at fit time; when that device isn't reachable here (no CUDA, or a
        lower GPU index than the training host had), settle for the best device
        this host does have and say so.

        ``self.device`` is deliberately left alone. It is a constructor
        parameter, so it carries the user's request rather than describing where
        the weights currently sit — that is ``model_.device``. Rewriting it
        would drop the request from ``get_params()``/``clone()`` for good, and
        would disagree with the learner's own ``_state.device``, which this
        wrapper cannot see.
        """
        model = getattr(self, 'model_', None)
        net = getattr(self, 'net_', None)
        if model is None and net is None:
            # Nothing fitted yet, so there is no state to place. self.device is
            # left as the user set it; fit() will resolve it and raise a clear
            # ValueError if it names a device this host doesn't have.
            return

        import torch

        requested = getattr(self, 'device', None)
        reason = None
        try:
            target = self._resolve_device(requested)
        except ValueError as exc:
            # No CUDA here, or an unparseable device string.
            target = torch.device('cpu')
            reason = str(exc)

        # Devices to try, best first. A move fails well beyond "no GPU": a
        # 'cuda:3' model on a one-GPU host, out-of-memory, a driver mismatch,
        # or CUDA re-initialization inside a forked worker. Another GPU here
        # may still work, so try what 'auto' picks before giving up on
        # acceleration entirely. The last candidate has to succeed — failing
        # to reach it is a real bug rather than an absent device, so it
        # propagates instead of warning.
        candidates = [target]
        if target.type != 'cpu':
            for fallback in (self._resolve_device('auto'), torch.device('cpu')):
                # Skipping a repeat keeps the warning honest: retrying the very
                # device that just failed can succeed after a transient error,
                # and then the message would read "could not be restored ...
                # loaded onto cuda instead" of cuda.
                if fallback not in candidates:
                    candidates.append(fallback)

        for index, candidate in enumerate(candidates):
            try:
                self._move_to_device(model, net, candidate)
            except _DEVICE_MOVE_ERRORS as exc:
                if index == len(candidates) - 1:
                    raise
                detail = f"{type(exc).__name__}: {exc}"
                # Lead with the device the user actually lost; a fallback that
                # failed on the way down is context, not the headline.
                reason = detail if reason is None else f"{reason}; then {detail}"
            else:
                target = candidate
                break

        if reason is not None:
            warnings.warn(
                f"{type(self).__name__}: model was trained with "
                f"device={requested!r}, which could not be restored here "
                f"({reason}). It has been loaded onto {target} instead, so "
                f"predictions will run there. device={requested!r} is kept for "
                f"re-fitting.",
                SuggestionWarning,
                stacklevel=2,
            )

    @staticmethod
    def _move_to_device(model, net, target):
        """Move the pycox model and its net to *target*.

        The two moves are independent on purpose: a pycox model without
        ``set_device`` must not leave the net stranded on the wrong device.
        ``Module.to()`` is a no-op when the module is already there, so moving
        the same net twice is harmless.
        """
        if net is not None:
            net.to(target)
        if model is not None:
            if hasattr(model, 'set_device'):
                model.set_device(target)
            elif getattr(model, 'net', None) is not None:
                model.net.to(target)


class ImpurityImportanceMixin:
    """Mixin for tree-based models that support impurity-based feature importance.

    Overrides ``get_feature_importance()`` to add the ``'impurity'`` method
    and change the default from ``'permutation'`` to ``'impurity'``.
    """

    def get_feature_importance(self, method='impurity', X=None, y=None,
                               n_repeats=100, random_state=None):
        if method == 'impurity':
            try:
                return self._impurity_importance()
            except NotImplementedError:
                raise NotImplementedError(
                    f"Impurity-based feature importance is not supported by "
                    f"{self.__class__.__name__}. Use "
                    f"get_feature_importance(method='permutation', "
                    f"X=X_test, y=y_test) instead."
                )
        return super().get_feature_importance(
            method=method, X=X, y=y, n_repeats=n_repeats,
            random_state=random_state,
        )
