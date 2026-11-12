"""XGBoost Cox model wrapper."""

import numpy as np
from sksurv.linear_model.coxph import BreslowEstimator

from mlsurv._xgboost import import_xgboost, warn_if_torch_loaded_on_macos
from mlsurv.models.base import BaseSurvivalModel, SurvivalArray
from mlsurv.models._overflow import breslow_call_with_clamp


class XGBCoxModel(BaseSurvivalModel):
    """
    XGBoost Cox model wrapper.

    Wraps XGBoost's XGBRegressor with survival:cox objective for survival analysis.
    Uses gradient boosting with Cox proportional hazards loss function.

    Parameters
    ----------
    n_estimators : int, default=300
        Number of boosting rounds (trees). With early stopping on by default,
        the actual round count is typically trimmed well below this cap.
    max_depth : int, default=3
        Maximum tree depth. Shallow trees suit low-EPV survival data and match
        scikit-survival's gradient-boosting default.
    learning_rate : float, default=0.10
        Step size shrinkage (eta).
    subsample : float, default=1.0
        Fraction of samples for each tree (0, 1].
    colsample_bytree : float, default=1.0
        Fraction of features for each tree (0, 1].
    gamma : float, default=0
        Minimum loss reduction for split.
    min_child_weight : int, default=1
        Minimum sum of instance weight in child.
    max_delta_step : float, default=0.7
        Maximum delta step allowed for each leaf output. Limits leaf value
        magnitudes, reducing overflow risk in the survival:cox objective.
        XGBoost uses 0.7 for Poisson regression but defaults to 0 for Cox;
        we adopt the same guard here.
    early_stopping_rounds : int or None, default=10
        Stop boosting when the held-out validation ``cox-nloglik`` has not
        improved for this many rounds. ``None`` disables early stopping and
        always trains ``n_estimators`` rounds. When set, a stratified
        validation split is carved from the training data; if the data is
        too small to hold one out, early stopping is disabled with a
        ``SuggestionWarning``.
    validation_fraction : float, default=0.1
        Fraction of the training data held out (stratified on event status)
        to monitor early stopping. Ignored when ``early_stopping_rounds`` is
        ``None``.
    n_jobs : int, default=1
        Number of parallel threads for tree building.
        - 1: Sequential execution (default, fully reproducible)
        - -1: Use all available CPU cores
        - n > 1: Use n CPU cores

        .. warning::
           When n_jobs != 1, results may not be perfectly reproducible even
           with the same random_state due to floating-point operation ordering
           in parallel execution. For fully reproducible results, use n_jobs=1.

    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    model_ : xgboost.XGBRegressor
        The fitted XGBoost model.
    is_fitted_ : bool
        Whether the model has been fitted.
    breslow_ : BreslowEstimator
        Breslow estimator for survival function prediction.

    Notes
    -----
    - Uses survival:cox objective for Cox proportional hazards
    - Censored observations have negative labels in XGBoost format
    - predict() returns log-hazard (raw margin) scores; higher = higher risk
    - predict_survival_function() uses Breslow baseline hazard estimation

    Examples
    --------
    >>> from mlsurv.models import XGBCoxModel
    >>> model = XGBCoxModel(n_estimators=200, max_depth=5)
    >>> model.fit(X_train, y_train)
    >>> risk_scores = model.predict(X_test)

    References
    ----------
    - Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
    - XGBoost survival analysis: https://xgboost.readthedocs.io/en/stable/python/survival-examples/
    """

    name = "xgbcox"
    display_name = "XGBoost-Cox"  # Using survival:cox objective
    color = "#882255"
    supports_random_state = True
    prefer_sequential_cv = False
    has_internal_parallelism = True
    complexity = "tree_ensemble"
    assumes_ph = True  # survival:cox objective fits Cox partial likelihood
    score_based = True  # single linear predictor → Breslow baseline

    def __init__(
        self,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.10,
        subsample=1.0,
        colsample_bytree=1.0,
        gamma=0,
        min_child_weight=1,
        max_delta_step=0.7,
        early_stopping_rounds=10,
        validation_fraction=0.1,
        n_jobs=1,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model_ = None
        self.breslow_ = None

    def fit(self, X, y: SurvivalArray):
        """
        Fit the XGBoost-Cox model.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Training features.
        y : SurvivalArray
            Survival outcomes with dtype=[('event', bool), ('time', float)].

        Returns
        -------
        self : XGBCoxModel
            Fitted model.

        Raises
        ------
        ImportError
            If XGBoost is not installed, or is installed but fails to load
            (a missing OpenMP runtime, most often on macOS).

        Warns
        -----
        SuggestionWarning
            On macOS, when PyTorch is already loaded, since the fit can hang.
        """
        xgb = import_xgboost("XGBCoxModel")
        warn_if_torch_loaded_on_macos()

        self._validate_data(X, y)

        # Convert to XGBoost survival format
        # Censored: negative time, Uncensored: positive time
        y_xgb = np.where(y['event'], y['time'], -y['time'])

        # Resolve early stopping: carve a stratified validation split when
        # requested and the dataset is large enough to hold one out. Train on
        # the fit split, but estimate the Breslow baseline on the full cohort.
        #
        # Unlike XGBAFT (which retrains a fresh full-data booster for the
        # early-stopped round count), the tree model here stays fit on the
        # fit split and predicts via iteration_range up to best_iteration.
        # This follows the issue's "respect best_iteration at predict time"
        # guidance and keeps XGBRegressor's eval_set API simple; the ~10%
        # validation rows still inform the Breslow baseline below.
        es_rounds, eval_metric = None, None
        eval_set = None
        X_fit, y_fit_xgb = X, y_xgb
        if self.early_stopping_rounds is not None:
            fit_idx, val_idx = self._stratified_es_split(
                y, self.validation_fraction,
            )
            if fit_idx is not None:
                es_rounds = self.early_stopping_rounds
                eval_metric = 'cox-nloglik'
                eval_set = [(self._take_rows(X, val_idx), y_xgb[val_idx])]
                X_fit = self._take_rows(X, fit_idx)
                y_fit_xgb = y_xgb[fit_idx]

        self.model_ = xgb.XGBRegressor(
            objective='survival:cox',
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            tree_method='hist',  # Faster histogram-based algorithm
            early_stopping_rounds=es_rounds,
            eval_metric=eval_metric,
        )

        if eval_set is not None:
            self.model_.fit(X_fit, y_fit_xgb, eval_set=eval_set, verbose=False)
        else:
            self.model_.fit(X_fit, y_fit_xgb)

        # Fit Breslow estimator for survival function prediction on the full
        # training cohort. Risk scores flow through _margin(), which respects
        # the early-stopped best iteration; BreslowEstimator.fit() internally
        # exponentiates, so we pass the raw log-hazard margin (not exp).
        risk_scores = self._margin(X)
        self.breslow_ = breslow_call_with_clamp(
            risk_scores, BreslowEstimator().fit,
            y['event'], y['time'], model_name="XGBCox",
        )

        self.is_fitted_ = True
        return self

    def _margin(self, X):
        """Raw log-hazard margin, respecting the early-stopped best iteration.

        When early stopping was active, ``best_iteration`` is set and
        predictions use only the rounds up to it; otherwise all rounds are
        used.
        """
        best_iter = getattr(self.model_, 'best_iteration', None)
        if best_iter is not None:
            return self.model_.predict(
                X, output_margin=True, iteration_range=(0, best_iter + 1),
            )
        return self.model_.predict(X, output_margin=True)

    def predict(self, X):
        """
        Predict risk scores (log-hazard).

        Returns the raw margin (log partial hazard) instead of
        ``exp(margin)`` to avoid overflow on large datasets. C-index
        is invariant to monotonic transforms, so rankings are identical.
        Brier/IBS metrics use ``predict_survival_function()``, which
        correctly exponentiates via the Breslow estimator.

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        risk_scores : ndarray
            Log-hazard risk scores (higher = higher risk).
        """
        self._check_is_fitted()
        return self._margin(X)

    def _evaluate_breslow_funcs(self, breslow_method, X, times, return_array):
        """Evaluate a Breslow method and optionally convert to array.

        Shared logic for predict_survival_function and
        predict_cumulative_hazard_function.
        """
        risk_scores = self.predict(X)
        funcs = breslow_call_with_clamp(
            risk_scores, breslow_method, model_name="XGBCox",
        )

        if times is not None:
            return np.array([fn(np.asarray(times)) for fn in funcs])

        if return_array:
            t = funcs[0].x
            arr = np.array([fn(t) for fn in funcs])
            self._warn_if_large_array(arr)
            return arr

        return funcs

    def predict_survival_function(self, X, times=None, return_array=False):
        """
        Predict survival function for each sample.

        Uses Breslow baseline hazard estimation to map risk scores
        to survival probabilities.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        times : array-like or None, default=None
            If provided, evaluate StepFunctions at these times and return
            an ndarray of shape (n_samples, n_times). This avoids downstream
            interpolation errors when computing Brier/IBS metrics.
        return_array : bool, default=False
            If True, return array instead of StepFunction objects.

        Returns
        -------
        list of StepFunction or ndarray
            Survival functions for each sample. When ``times`` is given,
            always returns ndarray (n_samples, n_times).
        """
        self._check_is_fitted()
        return self._evaluate_breslow_funcs(
            self.breslow_.get_survival_function, X, times, return_array,
        )

    def predict_cumulative_hazard_function(self, X, times=None, return_array=False):
        """
        Predict cumulative hazard function for each sample.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        times : array-like or None, default=None
            If provided, evaluate StepFunctions at these times and return
            an ndarray of shape (n_samples, n_times).
        return_array : bool, default=False
            If True, return array instead of StepFunction objects.

        Returns
        -------
        list of StepFunction or ndarray
            Cumulative hazard functions for each sample. When ``times``
            is given, always returns ndarray (n_samples, n_times).
        """
        self._check_is_fitted()
        return self._evaluate_breslow_funcs(
            self.breslow_.get_cumulative_hazard_function, X, times, return_array,
        )
