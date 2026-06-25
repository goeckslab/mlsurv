"""Cox Proportional Hazards model wrapper."""

import warnings

import numpy as np
import pandas as pd
from scipy.linalg import inv as _inv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.linear_model.coxph import CoxPHOptimizer

from mlsurv.models.base import BaseSurvivalModel, SurvivalArray
from mlsurv.models._overflow import (
    _CONVERGENCE_RE,
    _ILL_CONDITIONED_RE,
    _OVERFLOW_RE,
    predict_with_clamp,
    reemit_non_overflow,
)
from mlsurv._warnings import (
    FitFailureWarning,
    MethodologicalWarning,
    SuggestionWarning,
)

_FALLBACK_ALPHA = 0.01


class CoxPHModel(BaseSurvivalModel):
    """
    Cox Proportional Hazards model wrapper.

    Wraps scikit-survival's CoxPHSurvivalAnalysis with consistent interface.
    This is the baseline model for survival analysis with no hyperparameters
    to tune (only regularization alpha).

    When ``alpha=0`` (unregularized), the optimizer can diverge on small or
    collinear folds during cross-validation, producing either RuntimeWarnings
    (overflow in exp, divide by zero) or a ``ValueError`` from sksurv when the
    search direction contains NaN.  By default, if numerical instability is
    detected the model automatically re-fits with a small regularization
    penalty (``alpha=0.01``) to produce valid survival probabilities.  Set
    ``force=True`` to keep the original fit (C-index remains valid but
    survival function predictions may be unreliable).

    Parameters
    ----------
    alpha : float, default=0
        Regularization parameter (L2 penalty). ``0`` gives a pure
        unregularized Cox model; larger values increase regularization.
    force : bool, default=False
        If True, keep the original fit when numerical overflow warnings are
        detected (skip the fallback re-fit).  Ignored when the optimizer
        fails entirely (``ValueError``) — there is no fit to keep.  Useful
        when only the C-index (ranking metric) is needed.

    Attributes
    ----------
    model_ : CoxPHSurvivalAnalysis
        The fitted scikit-survival model.
    is_fitted_ : bool
        Whether the model has been fitted.
    alpha_ : float
        The effective regularization used for fitting. Equals ``alpha``
        unless a fallback was triggered, in which case it is ``0.01``.
    overflow_detected_ : bool
        True if numerical instability was detected during fitting
        (overflow warnings or optimizer ValueError).
    feature_names_in_ : array-like
        Feature names from training data.
    n_features_in_ : int
        Number of features from training data.

    Examples
    --------
    >>> from mlsurv.models import CoxPHModel
    >>> model = CoxPHModel(alpha=0.01)
    >>> model.fit(X_train, y_train)
    >>> risk_scores = model.predict(X_test)
    >>> c_index = model.score(X_test, y_test)

    References
    ----------
    - Cox, D.R. (1972). Regression models and life-tables.
    - scikit-survival: https://scikit-survival.readthedocs.io/
    """

    name = "coxph"
    display_name = "CoxPH"
    color = "#0072B2"
    supports_random_state = False
    prefer_sequential_cv = True
    assumes_ph = True
    assumes_linearity = True
    score_based = True  # single linear predictor → Breslow baseline
    @property
    def hazard_ratio_valid(self) -> bool:
        """Whether exp(coef) is a valid multivariable hazard ratio.

        True only for the unpenalized fit (``alpha == 0``): the PH partial
        likelihood gives unbiased coefficients and a covariance matrix that
        supports Wald CIs / p-values. A user-specified ``alpha > 0`` (ridge
        penalty) biases coefficients toward zero and makes inference
        conditional on the penalty strength, so this returns False and
        downstream evaluation/viz suppress the hazard-ratio output.

        Gated on the constructor ``alpha`` (user intent), not the effective
        ``alpha_``: the automatic ridge fallback (``alpha=0 -> alpha_=0.01``
        on numerically unstable folds) leaves this True, preserving existing
        behavior for that path (its conditional CIs are noted in
        ``_attach_covariance``).
        """
        return self.alpha == 0

    def __init__(self, alpha=0, force=False):
        self.alpha = alpha
        self.force = force
        self.model_ = None

    def fit(self, X, y: SurvivalArray):
        """
        Fit the Cox PH model.

        If numerical instability is detected (overflow warnings or optimizer
        ValueError) and ``force`` is False, the model is automatically
        re-fitted with ``alpha=0.01``.  When the optimizer raises a
        ``ValueError`` (fit failed entirely), fallback always applies
        regardless of ``force`` — there is no model to keep.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Training features.
        y : SurvivalArray
            Survival outcomes with 'event' and 'time' fields.

        Returns
        -------
        self : CoxPHModel
            Fitted model.

        Raises
        ------
        RuntimeError
            If the regularized fallback fit also fails, indicating the data
            is too degenerate for Cox PH.
        """
        self._validate_data(X, y)
        self.alpha_ = self.alpha
        self.model_ = CoxPHSurvivalAnalysis(alpha=self.alpha)
        self.overflow_detected_ = False

        if self.alpha > 0:
            # A user-specified ridge penalty (e.g. from tuning an explicit
            # alpha space) regularizes the fit. Penalized coefficients are
            # biased toward zero and inference is conditional on the penalty,
            # so surface the caveat (hazard_ratio_valid is also False here).
            warnings.warn(
                f"CoxPH(alpha={self.alpha}) applies an L2 ridge penalty. "
                f"Penalized coefficients are biased toward zero, and the "
                f"hazard ratios, Wald confidence intervals, and p-values "
                f"become conditional on the penalty strength — they are not "
                f"unbiased multivariable estimates. For a sparse/regularized "
                f"Cox model with proper inference, consider CoxNet.",
                MethodologicalWarning,
                stacklevel=2,
            )

        fit_failed, has_warnings, convergence_failed = self._fit_and_check(X, y)

        if not fit_failed and not has_warnings:
            if convergence_failed:
                # Convergence failure is distinct from rank-deficient instability:
                # remediation is more iterations or regularization, not an alpha
                # fallback that silently shrinks coefficients.
                warnings.warn(
                    f"CoxPH(alpha={self.alpha}) optimizer did not converge "
                    f"within sksurv's default n_iter. Coefficients may be "
                    f"unreliable. Consider increasing n_iter (via sksurv's "
                    f"CoxPHSurvivalAnalysis) or using a non-zero alpha for "
                    f"L2 regularization.",
                    FitFailureWarning,
                    stacklevel=2,
                )
            self.is_fitted_ = True
            self._attach_covariance(X, y)
            return self

        # Numerical instability detected.
        self.overflow_detected_ = True

        # force=True keeps the original fit when only warnings occurred
        # (not when the optimizer failed entirely -- there is no model to keep).
        if not fit_failed and self.force:
            warnings.warn(
                f"CoxPH(alpha={self.alpha}) produced numerical instability. "
                f"force=True: keeping original fit. C-index is valid but "
                f"survival function predictions may be unreliable.",
                FitFailureWarning,
                stacklevel=2,
            )
            self.is_fitted_ = True
            self._attach_covariance(X, y)
            return self

        # Fallback: re-fit with regularization.
        _cause_hint = (
            " This typically indicates collinear features, a near-singular "
            "design matrix, or a high feature-to-sample ratio. Consider "
            "using CoxNet with L1 regularization for automatic feature selection."
        )
        if fit_failed:
            msg = (
                f"CoxPH(alpha={self.alpha}) optimizer failed "
                f"(NaN/infinite search direction). Re-fitting with "
                f"alpha={_FALLBACK_ALPHA}"
            )
            if self.force:
                msg += (
                    ". force=True is ignored when the fit fails "
                    "— there is no model to keep."
                )
            else:
                msg += " for valid survival probabilities."
        else:
            msg = (
                f"CoxPH(alpha={self.alpha}) produced numerical "
                f"instability. Re-fitting with alpha={_FALLBACK_ALPHA} "
                f"for valid survival probabilities. Set force=True to "
                f"keep the original fit (C-index still valid, but "
                f"survival function predictions will be unreliable)."
            )
        warnings.warn(msg + _cause_hint, FitFailureWarning, stacklevel=2)

        self.alpha_ = _FALLBACK_ALPHA
        self.model_ = CoxPHSurvivalAnalysis(alpha=_FALLBACK_ALPHA)
        try:
            fb_failed, fb_warnings, _fb_convergence = self._fit_and_check(X, y)
        except Exception as exc:
            raise RuntimeError(
                f"CoxPH fallback fit (alpha={_FALLBACK_ALPHA}) also "
                f"failed: {exc}. Data may be too degenerate for Cox PH."
            ) from exc
        if fb_failed:
            raise RuntimeError(
                f"CoxPH fallback fit (alpha={_FALLBACK_ALPHA}) also "
                f"failed (optimizer ValueError). Data may be too "
                f"degenerate for Cox PH."
            )
        if fb_warnings:
            warnings.warn(
                f"CoxPH fallback fit (alpha={_FALLBACK_ALPHA}) also "
                f"produced numerical instability warnings. Survival "
                f"function predictions may still be unreliable.",
                FitFailureWarning,
                stacklevel=2,
            )

        self.is_fitted_ = True
        self._attach_covariance(X, y)
        return self

    def _fit_and_check(self, X, y):
        """Fit model and check for numerical instability.

        Returns
        -------
        fit_failed : bool
            True if the optimizer raised ValueError (model not fitted).
        has_warnings : bool
            True if numerical-instability ``RuntimeWarning``s were emitted —
            either overflow messages or scipy ``LinAlgWarning``-style
            "ill-conditioned matrix" messages from inside sksurv's solve().
        convergence_failed : bool
            True if sksurv emitted ``ConvergenceWarning`` (IRLS hit
            ``n_iter`` without converging). Tracked separately from
            ``has_warnings`` so it does NOT trigger the alpha-fallback
            re-fit — the right remediation is more iterations or
            regularization, not silently shrinking coefficients.
        """
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.model_.fit(X, y)
        except ValueError as exc:
            reemit_non_overflow(caught)
            msg = str(exc).lower()
            if "nan" in msg or "infinite" in msg or "singular" in msg:
                return True, False, False
            raise

        has_warnings = False
        convergence_failed = False
        for w in caught:
            msg = str(w.message)
            if _CONVERGENCE_RE.search(msg):
                convergence_failed = True
                continue
            if issubclass(w.category, RuntimeWarning) and (
                _OVERFLOW_RE.search(msg) or _ILL_CONDITIONED_RE.search(msg)
            ):
                has_warnings = True
                continue
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        return False, has_warnings, convergence_failed

    def _attach_covariance(self, X, y):
        """Compute and store the covariance matrix on the fitted sksurv model.

        The covariance is the inverse of the (possibly penalized) observed
        information matrix at the fitted coefficients.  When ``alpha_ > 0``
        (user-specified or from the automatic fallback), the L2 penalty is
        included in the Hessian, so the resulting CIs are conditional on the
        regularization strength.

        sksurv's ``CoxPHOptimizer.hessian`` is normalised per observation
        (divided by *n*), so we multiply by *n* before inverting to recover
        the full information matrix.

        On failure (e.g. singular Hessian), the attribute is simply not set
        and ``compute_hazard_ratios`` falls back to point estimates.
        """
        if not hasattr(self.model_, "coef_"):
            warnings.warn(
                "Could not compute covariance matrix for CoxPH: model has "
                "no fitted coefficients (coef_). Confidence intervals will "
                "not be available in forest plots.",
                SuggestionWarning,
                stacklevel=2,
            )
            return

        try:
            n_samples = X.shape[0]
            event, time = y["event"], y["time"]
            alphas = np.full(X.shape[1], float(self.alpha_))
            optimizer = CoxPHOptimizer(
                np.asarray(X, dtype=np.float64), event, time,
                alphas, self.model_.ties,
            )
            # Suppress ill-conditioned LinAlgWarnings (RuntimeWarning subclass)
            # from optimizer.update / optimizer.hessian / scipy.linalg.inv on
            # rank-deficient designs; the wrapping fit() path already
            # surfaces one FitFailureWarning, and the except branch below
            # surfaces a SuggestionWarning if covariance computation outright
            # fails (preserving CI-availability feedback).
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                optimizer.update(self.model_.coef_)
                # sksurv normalises the Hessian by 1/n_samples; undo that
                self.model_.covariance_matrix_ = _inv(
                    optimizer.hessian * n_samples
                )
            reemit_non_overflow(caught)
        except (np.linalg.LinAlgError, ValueError) as exc:
            warnings.warn(
                f"Could not compute covariance matrix for CoxPH: {exc}. "
                f"Confidence intervals will not be available in forest plots.",
                SuggestionWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Prediction overrides — clamp linear predictors to prevent overflow
    # ------------------------------------------------------------------

    def _predict_with_clamp(self, func_name, X, return_array):
        """Call ``func_name`` on the sksurv model with clamped linear predictors.

        Delegates to :func:`mlsurv.models._overflow.predict_with_clamp`.
        See that function's docstring for details on the monkey-patching
        approach and thread-safety caveats.
        """
        self._check_is_fitted()
        result = predict_with_clamp(
            self.model_, func_name, X, return_array,
            model_display_name="CoxPH",
        )
        if return_array:
            self._warn_if_large_array(result)
        return result

    def predict_survival_function(self, X, return_array=False):
        """Predict survival function with overflow-safe linear predictors.

        Clamps the linear predictor to ``[-20, 20]`` before exponentiation
        to prevent float64 overflow in the Breslow estimator.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        return_array : bool, default=False
            If True, return ndarray (samples x times) instead of StepFunction
            objects.

        Returns
        -------
        list of StepFunction or ndarray
            See :meth:`BaseSurvivalModel.predict_survival_function`.
        """
        return self._predict_with_clamp(
            "predict_survival_function", X, return_array
        )

    def predict_cumulative_hazard_function(self, X, return_array=False):
        """Predict cumulative hazard function with overflow-safe linear predictors.

        Clamps the linear predictor to ``[-20, 20]`` before exponentiation
        to prevent float64 overflow in the Breslow estimator.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        return_array : bool, default=False
            If True, return ndarray (samples x times) instead of StepFunction
            objects.

        Returns
        -------
        list of StepFunction or ndarray
            See :meth:`BaseSurvivalModel.predict_cumulative_hazard_function`.
        """
        return self._predict_with_clamp(
            "predict_cumulative_hazard_function", X, return_array
        )

    def get_coefficients(self):
        """
        Return model coefficients.

        Returns
        -------
        pd.Series
            Coefficients indexed by feature name.
        """
        self._check_is_fitted()
        return pd.Series(
            self.model_.coef_,
            index=self.feature_names_in_,
            name='coefficient'
        )
