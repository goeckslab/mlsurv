"""Cox Proportional Hazards model wrapper."""

import warnings

import numpy as np
import pandas as pd
from scipy.linalg import inv as _inv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.linear_model.coxph import CoxPHOptimizer

from mlsurv.models.base import BaseSurvivalModel, SurvivalArray
from mlsurv.models._overflow import (
    _CALLER_ERROR_RE,
    _CONVERGENCE_RE,
    _ILL_CONDITIONED_RE,
    _OVERFLOW_RE,
    _SOLVER_FAILURE_RE,
    predict_with_clamp,
    reemit_non_overflow,
)
from mlsurv._warnings import (
    FitFailureWarning,
    MethodologicalWarning,
    SuggestionWarning,
)

_FALLBACK_ALPHAS = (0.01, 0.1, 1.0)
# First rung, kept as a module-level alias: existing imports and tests refer to
# the single fallback penalty by this name.
_FALLBACK_ALPHA = _FALLBACK_ALPHAS[0]


def _fallback_ladder(alpha):
    """Escalating fallback penalties, never weaker than the user's ``alpha``.

    Each rung is ``max(alpha, step)``, so ``CoxPHModel(alpha=0.5)`` escalates
    to 1.0 and is never silently downgraded to 0.01. Rungs that collapse onto
    the user's own alpha are dropped: re-running the identical deterministic
    fit would fail identically, so it would only cost time and muddy the
    warning text.

    The guard is ``alpha > 0``, never ``if alpha``. ``0`` and ``0.0`` are the
    documented default (unregularized), not "unset", and both are falsy.

    The rungs match the penalizer ladder ``_fit_cox_with_fallback`` uses in
    ``evaluation/metrics.py``; keep the two in step.

    Returns an empty list when ``alpha`` already meets or exceeds the top rung.
    There is no stronger penalty to escalate to, so the caller keeps a fit that
    only warned and raises only when the optimizer failed outright.
    """
    alpha = float(alpha)
    ladder = []
    for step in _FALLBACK_ALPHAS:
        eff = max(alpha, float(step))
        if alpha > 0 and eff <= alpha:
            continue
        ladder.append(eff)
    return ladder


class CoxPHModel(BaseSurvivalModel):
    """
    Cox Proportional Hazards model wrapper.

    Wraps scikit-survival's CoxPHSurvivalAnalysis with consistent interface.
    This is the baseline model for survival analysis with no hyperparameters
    to tune (only regularization alpha).

    When ``alpha=0`` (unregularized), the optimizer can diverge on small or
    collinear folds during cross-validation, producing either RuntimeWarnings
    (overflow in exp, divide by zero) or a ``ValueError``: from sksurv when
    the search direction contains NaN, or from scipy's solver when LAPACK
    reports an illegal value or a singular matrix.  By default, if numerical
    instability is detected the model automatically re-fits with an escalating
    L2 penalty (``0.01``, then ``0.1``, then ``1.0``), keeping the first
    penalty that fits, so a near-singular design yields a usable model instead
    of a dropped one.  Each rung is ``max(alpha, rung)``, so a user-supplied
    ``alpha`` is never weakened (``alpha=0.5`` escalates to ``1.0``, never down
    to ``0.01``).  When ``alpha`` already meets or exceeds the top rung there
    is nothing to escalate to: a fit that only warned is kept, and a fit that
    failed outright raises ``RuntimeError``.  Set ``force=True`` to keep the
    original fit (C-index remains valid but survival function predictions may
    be unreliable).

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
        unless a fallback was triggered, in which case it is the first rung
        of ``_FALLBACK_ALPHAS`` (``0.01``, ``0.1``, ``1.0``) above ``alpha``
        that produced a usable fit.
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

        The automatic fallback is allowed one exception. Escalating to the
        first rung (``0.01``) leaves this True: the shrinkage at that penalty
        is negligible on a standardized design, and its conditional CIs are
        noted in ``_attach_covariance``. Escalating further (``0.1``, ``1.0``)
        shrinks coefficients enough to matter, so this turns False and the
        hazard-ratio table and forest plot are suppressed rather than
        presenting penalized estimates as unbiased ones.
        """
        return self.alpha == 0 and getattr(self, "alpha_", 0) <= _FALLBACK_ALPHAS[0]

    def __init__(self, alpha=0, force=False):
        self.alpha = alpha
        self.force = force
        self.model_ = None

    def fit(self, X, y: SurvivalArray):
        """
        Fit the Cox PH model.

        If numerical instability is detected (overflow warnings or optimizer
        ValueError) and ``force`` is False, the model is automatically
        re-fitted with an escalating L2 penalty (``0.01``, ``0.1``, ``1.0``),
        keeping the first penalty that fits.  Each rung is floored at the
        user's ``alpha``, so a supplied penalty is never weakened.  When the
        optimizer raises a ``ValueError`` (fit failed entirely), fallback
        always applies regardless of ``force`` (there is no model to keep).

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
            If every rung of the escalating fallback ladder also fails,
            indicating the data is too degenerate for Cox PH; or if the
            optimizer failed outright and ``alpha`` already meets or exceeds
            the top rung, leaving no stronger penalty to escalate to and no
            fitted model to keep. Not raised when the fit succeeded and only
            warned, or when ``force=True`` kept the original fit.
        """
        self._validate_data(X, y)
        self.alpha_ = self.alpha
        self.model_ = CoxPHSurvivalAnalysis(alpha=self.alpha)
        self.overflow_detected_ = False
        self._last_solver_error_ = None
        # Set only when the fallback escalates far enough to invalidate
        # hazard-ratio inference; read by the report's model-details section.
        self.hr_suppressed_reason_ = None

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

        # Fallback: re-fit with escalating regularization, taking the first
        # penalty that fits.
        _cause_hint = (
            " This typically indicates collinear features, a near-singular "
            "design matrix, or a high feature-to-sample ratio. Consider "
            "using CoxNet with L1 regularization for automatic feature selection."
        )

        ladder = _fallback_ladder(self.alpha)
        no_escalation = (
            f"CoxPH(alpha={self.alpha}) hit numerical instability, but alpha "
            f"is already at or above the top of the automatic fallback ladder "
            f"({_FALLBACK_ALPHAS[-1]:g}), so there is no stronger penalty to "
            f"escalate to. Weakening your alpha is never done automatically."
        )
        if not ladder and not fit_failed:
            # The original fit produced a usable model and only warned, so keep
            # it rather than throwing it away, the same trade-off force=True
            # makes. Raising here would fail the CV fold and could drop CoxPH
            # from the results, which is the outcome the fallback exists to
            # prevent.
            warnings.warn(
                no_escalation + " Keeping the original fit: the C-index is "
                "valid, but survival function predictions may be unreliable."
                + _cause_hint,
                FitFailureWarning,
                stacklevel=2,
            )
            self.is_fitted_ = True
            self._attach_covariance(X, y)
            return self
        if not ladder:
            raise RuntimeError(
                no_escalation + " The fit produced no model to keep."
                + _cause_hint
            )

        winner, winner_warned, winner_convergence = None, False, False
        # (alpha, message) for the most recent rung that reported a solver
        # failure, so the ladder-exhausted error can quote a real cause. Seeded
        # from the initial fit, whose text _fit_and_check already stashed.
        last_failure = (
            (self.alpha, self._last_solver_error_)
            if fit_failed and self._last_solver_error_ is not None else None
        )
        for eff_alpha in ladder:
            # alpha_ and model_ are rebound together on every rung so
            # _attach_covariance's penalty vector always matches the alpha
            # that actually produced coef_.
            self.alpha_ = eff_alpha
            self.model_ = CoxPHSurvivalAnalysis(alpha=eff_alpha)
            self._last_solver_error_ = None
            # No except clause: _fit_and_check turns solver failures into
            # fb_failed and re-raises everything else on purpose, so anything
            # propagating from here is a caller error or an unrecognized
            # failure that the user needs to see, not a reason to try a
            # stronger penalty.
            fb_failed, fb_warnings, fb_convergence = self._fit_and_check(X, y)
            if fb_failed:
                if self._last_solver_error_ is not None:
                    last_failure = (eff_alpha, self._last_solver_error_)
                continue
            winner, winner_warned = eff_alpha, fb_warnings
            winner_convergence = fb_convergence
            break

        if winner is None:
            tried = ", ".join(f"alpha={a:g}" for a in ladder)
            detail = (
                f" Last solver error (at alpha={last_failure[0]:g}): "
                f"{last_failure[1]}." if last_failure else ""
            )
            raise RuntimeError(
                f"CoxPH fallback fits also failed at every automatically "
                f"escalated penalty ({tried}). Data may be too degenerate for "
                f"Cox PH.{detail} Pass an explicit alpha= to override the "
                f"ladder, or force=True to keep an unpenalized fit when only "
                f"warnings fired.{_cause_hint}"
            )

        # One wrapper warning, naming the penalty actually used. It is emitted
        # after the ladder rather than before it because the winning alpha is
        # not known until the ladder resolves.
        skipped = ladder[: ladder.index(winner)]
        escalated = (
            f" (escalated past {', '.join(f'{a:g}' for a in skipped)})"
            if skipped else ""
        )
        if fit_failed:
            msg = (
                f"CoxPH(alpha={self.alpha}) optimizer failed (NaN/infinite "
                f"search direction or LAPACK solve failure). Re-fitting with "
                f"alpha={winner:g}{escalated}"
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
                f"instability. Re-fitting with alpha={winner:g}{escalated} "
                f"for valid survival probabilities. Set force=True to "
                f"keep the original fit (C-index still valid, but "
                f"survival function predictions will be unreliable)."
            )
        warnings.warn(msg + _cause_hint, FitFailureWarning, stacklevel=2)

        if self.alpha == 0 and winner > _FALLBACK_ALPHAS[0]:
            # Past the first rung the penalty is strong enough to bias the
            # coefficients, so say what that costs. hazard_ratio_valid also
            # turns False here, which suppresses the hazard-ratio table and
            # forest plot; this warning explains why they went missing.
            # The reason is also stored on the model, because a warning at fit
            # time does not reach someone reading the report later.
            self.hr_suppressed_reason_ = (
                f"Suppressed. The fit needed alpha={winner:g} to stay stable, "
                f"and at that penalty the coefficients are shrunk enough that "
                f"hazard ratios, confidence intervals, and p-values would be "
                f"conditional on the penalty rather than unbiased "
                f"multivariable estimates. The C-index and survival "
                f"probabilities are unaffected."
            )
            warnings.warn(
                f"CoxPH escalated to alpha={winner:g} to obtain a stable fit. "
                f"At this penalty the coefficients are meaningfully shrunk "
                f"toward zero, so hazard ratios, Wald confidence intervals, "
                f"and p-values would be conditional on the penalty rather than "
                f"unbiased multivariable estimates. They are suppressed for "
                f"this model. The C-index and survival probabilities remain "
                f"usable. For a regularized Cox model with proper inference, "
                f"consider CoxNet.",
                MethodologicalWarning,
                stacklevel=2,
            )

        if winner_warned:
            warnings.warn(
                f"CoxPH fallback fit (alpha={winner:g}) also "
                f"produced numerical instability warnings. Survival "
                f"function predictions may still be unreliable.",
                FitFailureWarning,
                stacklevel=2,
            )

        if winner_convergence:
            # Same condition the non-fallback path warns about above. Without
            # this the ladder would be quieter about a non-converged fit than
            # a plain one, and its coefficients would be published unflagged.
            warnings.warn(
                f"CoxPH fallback fit (alpha={winner:g}) did not converge "
                f"within sksurv's default n_iter. Coefficients may be "
                f"unreliable. Consider increasing n_iter (via sksurv's "
                f"CoxPHSurvivalAnalysis) or passing a larger alpha.",
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
            True if the optimizer raised a numerical ``ValueError`` and the
            model was not fitted: a NaN/infinite search direction, a LAPACK
            illegal-value report, or a singular matrix. Caller/parameter
            ``ValueError``s propagate instead.
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
            # A numerical failure of the optimizer, or of the scipy solve()
            # inside it, routes into the escalating alpha fallback. Bad input
            # and bad arguments are checked first and propagate unchanged: a
            # NaN cell or a negative alpha is not fixed by regularization, and
            # sklearn's own message says far more about it than we could. See
            # the message census in _overflow.py.
            message = str(exc)
            if _CALLER_ERROR_RE.search(message):
                raise
            if _SOLVER_FAILURE_RE.search(message):
                # Kept so the ladder can quote the real cause if every rung
                # fails; the boolean return alone throws the text away.
                self._last_solver_error_ = message
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
        (user-specified or from the automatic fallback ladder), the L2 penalty is
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
