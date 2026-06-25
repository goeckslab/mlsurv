"""Weibull AFT model wrapper."""

import numpy as np
import pandas as pd

from mlsurv.models.base import BaseSurvivalModel, SurvivalArray
from mlsurv._warnings import FitFailureWarning


class WeibullAFTModel(BaseSurvivalModel):
    """
    Weibull Accelerated Failure Time (AFT) model wrapper.

    Wraps lifelines' WeibullAFTFitter for parametric survival analysis.
    Models survival time directly using Weibull distribution with shape
    and scale parameters.

    Parameters
    ----------
    alpha : float, default=0.0
        Regularization parameter (L2 penalty). 0 = no regularization.
    penalizer : float or None, default=None
        Alias for alpha (for lifelines compatibility). If both specified,
        penalizer takes precedence.
    fit_intercept : bool, default=True
        Whether to fit an intercept term.

    Attributes
    ----------
    model_ : lifelines.WeibullAFTFitter
        The fitted lifelines model.
    is_fitted_ : bool
        Whether the model has been fitted.

    Notes
    -----
    - Parametric model assuming Weibull distribution
    - predict() returns log survival times (higher = longer survival)
    - For risk scores (higher = higher risk), use negative predictions
    - predict_survival_function() returns Weibull survival curves
    - predict_survival_time() returns median/mean survival times
    - Faster than Cox models but less flexible (assumes Weibull distribution)
    - **Structurally PH**: Weibull is the unique distribution under which the
      AFT and PH parameterizations coincide (Kalbfleisch & Prentice 2002,
      ch. 2). The standard Weibull AFT fit (single shared shape parameter,
      no time-varying covariates) can always be re-expressed as a PH model
      with constant hazard ratio, so ``assumes_ph = True`` and
      Grambsch-Therneau PH violations apply to this model the same way they
      apply to CoxPH.

    Examples
    --------
    >>> from mlsurv.models import WeibullAFTModel
    >>> model = WeibullAFTModel(alpha=0.01)
    >>> model.fit(X_train, y_train)
    >>> log_times = model.predict(X_test)
    >>> median_times = model.predict_survival_time(X_test)

    References
    ----------
    - Wei, L.J. (1992). The accelerated failure time model: a useful
      alternative to the Cox regression model in survival analysis.
    - lifelines: https://lifelines.readthedocs.io/
    """

    name = "weibull_aft"
    display_name = "Weibull-AFT"
    color = "#000000"
    supports_random_state = False
    prefer_sequential_cv = True  # lifelines uses autograd (not thread-safe)
    # Weibull is the only distribution that is simultaneously PH and AFT
    # (Kalbfleisch & Prentice 2002, ch. 2); AFT implies log-linearity in covariates.
    assumes_ph = True
    assumes_linearity = True

    def __init__(self, alpha=0.0, penalizer=None, fit_intercept=True):
        # Store params exactly as passed (sklearn clone requires identity preservation)
        self.alpha = alpha
        self.penalizer = penalizer
        self.fit_intercept = fit_intercept
        self.model_ = None

    @property
    def _effective_penalizer(self):
        """Resolve the penalizer value: penalizer wins if set, else alpha."""
        return self.penalizer if self.penalizer is not None else self.alpha

    def fit(self, X, y: SurvivalArray):
        """
        Fit the Weibull AFT model.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Training features.
        y : SurvivalArray
            Survival outcomes with dtype=[('event', bool), ('time', float)].

        Returns
        -------
        self : WeibullAFTModel
            Fitted model.

        Raises
        ------
        ImportError
            If the optional ``lifelines`` dependency is not installed.
        ValueError
            If ``X`` contains a column whose name collides with the
            reserved survival-target columns, or if the fitted model
            produces NaN predictions.
        """
        try:
            from lifelines import WeibullAFTFitter
        except ImportError:
            raise ImportError(
                "lifelines is required for WeibullAFTModel. "
                "Install with: pip install lifelines>=0.27.0"
            )

        self._validate_data(X, y)

        # Convert to DataFrame for lifelines
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Check for column name collisions with survival target columns
        reserved = {'duration', 'event'}
        collisions = reserved & set(X.columns)
        if collisions:
            raise ValueError(
                f"Feature matrix X contains reserved column name(s) {collisions} "
                f"which would be overwritten by survival targets. "
                f"Rename these columns before fitting."
            )

        # Create DataFrame with duration and event columns
        df = X.copy()
        df['duration'] = y['time']
        df['event'] = y['event'].astype(int)

        # Fit model with convergence handling
        effective_pen = self._effective_penalizer
        self.model_ = WeibullAFTFitter(
            penalizer=effective_pen,
            fit_intercept=self.fit_intercept
        )

        try:
            self.model_.fit(df, duration_col='duration', event_col='event')

            # Validate convergence: check that predictions are not NaN
            test_pred = self.model_.predict_expectation(X.iloc[:1])
            if np.isnan(test_pred.values).any():
                raise ValueError("Model produced NaN predictions after fitting")

        except Exception as e:
            if isinstance(e, (MemoryError, RecursionError)):
                raise
            # If fitting fails with no regularization, try with small penalizer
            if effective_pen == 0.0:
                import warnings
                warnings.warn(
                    f"WeibullAFT failed to converge with penalizer=0.0: {e}. "
                    "Retrying with penalizer=0.01",
                    FitFailureWarning,
                )
                self.model_ = WeibullAFTFitter(
                    penalizer=0.01,
                    fit_intercept=self.fit_intercept
                )
                self.model_.fit(df, duration_col='duration', event_col='event')
            else:
                raise

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict risk scores.

        Returns negative of expected survival time as risk score.
        Higher values = higher risk (shorter survival).

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        risk_scores : ndarray
            Predicted risk scores (higher = higher risk).
        """
        self._check_is_fitted()

        # Convert to DataFrame for lifelines
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Predict expected survival time
        expected_times = self.model_.predict_expectation(X)

        # Return negative (to match risk score convention: higher = higher risk)
        return -expected_times.values

    def get_coefficients(self):
        """
        Return AFT scale coefficients.

        These are coefficients on the log-scale parameter (``lambda_``) of
        the Weibull AFT model, NOT log-hazards: a positive coefficient means
        longer survival (lower risk). ``exp(coef)`` is a time-ratio, not a
        hazard ratio, so ``hazard_ratio_valid`` stays False and downstream
        accessors leave HR/CI/p-value blank for this model.

        Returns
        -------
        pd.Series
            Coefficients indexed by feature name (intercept excluded).
        """
        self._check_is_fitted()
        # lifelines params_ is a MultiIndex Series: ('lambda_', covariate)
        # plus ('rho_', 'Intercept') for the shape parameter.
        coef = self.model_.params_['lambda_'].drop('Intercept', errors='ignore')
        return pd.Series(coef.values, index=coef.index, name='coefficient')

    def predict_survival_time(self, X, quantile=0.5):
        """
        Predict survival time at specified quantile.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        quantile : float, default=0.5
            Quantile to predict (0.5 = median survival time).

        Returns
        -------
        times : ndarray
            Predicted survival times at the specified quantile.
        """
        self._check_is_fitted()

        # Convert to DataFrame for lifelines
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Predict median survival time
        times = self.model_.predict_percentile(X, p=quantile)
        return times.values

    def predict_survival_function(self, X, times=None, return_array=False):
        """
        Predict survival function for each sample.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        times : array-like or None, default=None
            Time points at which to evaluate survival function.
            If None, uses a default grid.
        return_array : bool, default=False
            If True, return ndarray of shape (n_samples, n_times).

        Returns
        -------
        surv_df : pd.DataFrame or ndarray
            Survival probabilities. DataFrame has time as index and samples as
            columns; ndarray has shape (n_samples, n_times).

        Raises
        ------
        ValueError
            If ``times`` is not given and the fitted model's timeline is
            empty.
        """
        self._check_is_fitted()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if times is None:
            if len(self.model_.timeline) == 0:
                raise ValueError("Model timeline is empty. Fit the model first.")
            times = np.linspace(0, self.model_.timeline.max(), 100)

        surv_funcs = self.model_.predict_survival_function(X, times=times)

        if return_array:
            result = surv_funcs.T.values
            self._warn_if_large_array(result)
            return result

        return surv_funcs

    def predict_cumulative_hazard_function(self, X, times=None, return_array=False):
        """
        Predict cumulative hazard function for each sample.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        times : array-like or None, default=None
            Time points at which to evaluate cumulative hazard.
            If None, uses a default grid.
        return_array : bool, default=False
            If True, return ndarray of shape (n_samples, n_times).

        Returns
        -------
        cum_hazard : pd.DataFrame or ndarray
            Cumulative hazard at specified times.

        Raises
        ------
        ValueError
            If ``times`` is not given and the fitted model's timeline is
            empty.
        """
        self._check_is_fitted()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if times is None:
            if len(self.model_.timeline) == 0:
                raise ValueError("Model timeline is empty. Fit the model first.")
            times = np.linspace(0, self.model_.timeline.max(), 100)

        chf_funcs = self.model_.predict_cumulative_hazard(X, times=times)

        if return_array:
            result = chf_funcs.T.values
            self._warn_if_large_array(result)
            return result

        return chf_funcs
