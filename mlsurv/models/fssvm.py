"""Fast Survival SVM model wrapper."""

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sksurv.linear_model.coxph import BreslowEstimator
from sksurv.svm import FastSurvivalSVM

from mlsurv.models.base import BaseSurvivalModel, SurvivalArray
from mlsurv.models._overflow import breslow_call_with_clamp
from mlsurv._warnings import RegressionModeWarning, SuggestionWarning


# Module-level dedup for the regression-mode caveat, keyed by rank_ratio.
# The caveat is invariant for a fitted model (rank_ratio is fixed at fit time),
# so every repeat carries no new information. fit() is called many times in a
# session (CV folds, bootstrap refits, repeated train() calls), so a per-call
# or per-instance guard would still emit once per fit. A module-level set
# collapses all of those to a single emission per session because it survives
# the ``warnings.catch_warnings`` registry resets that defeat the "once"
# warning filter across train/evaluate/bootstrap phases. Same pattern as
# ``_overflow.py::_nonfinite_warned`` and ``metrics.py::_calibration_warned``.
# Tests clear it between cases (see test_fssvm_validation.py).
_regression_mode_warned: set = set()


def _warn_regression_mode_once(rank_ratio, stacklevel=3):
    """Emit the FSSVM regression-mode caveat at most once per session.

    Keyed by ``rank_ratio`` so distinct settings each surface once.
    """
    if rank_ratio in _regression_mode_warned:
        return
    _regression_mode_warned.add(rank_ratio)
    warnings.warn(
        f"FSSVM with rank_ratio={rank_ratio} uses regression mode. "
        "Breslow-based survival curves may not be valid. "
        "Brier scores should be interpreted with caution.",
        RegressionModeWarning,
        stacklevel=stacklevel,
    )


class FSSVMModel(BaseSurvivalModel):
    """
    Fast Survival SVM model wrapper.

    Wraps scikit-survival's FastSurvivalSVM with consistent interface.
    Uses a ranking or regression SVM formulation for survival prediction.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization parameter (weight of the squared-hinge loss term).
        Larger alpha means *less* regularization; smaller alpha means more.
    rank_ratio : float, default=1.0
        Trade-off between ranking and regression loss.
        - 1.0: Pure ranking (default, recommended)
        - 0.0: Pure regression
        - Between: Hybrid loss
    max_iter : int, default=100
        Maximum number of iterations for the Newton optimizer.
    tol : float or None, default=None
        Convergence tolerance. If None, uses scipy's Newton-CG default.
    optimizer : str, default='avltree'
        Optimization algorithm: 'avltree', 'direct-count', 'rbtree', 'PRSVM', 'simple'.
        Note: 'PRSVM' and 'simple' only work with rank_ratio=1.0 (pure ranking).
    fit_intercept : bool, default=False
        Whether to calculate an intercept for the model. Only available when
        optimizer is 'avltree', 'direct-count', or 'rbtree'.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    model_ : FastSurvivalSVM
        The fitted scikit-survival model.
    is_fitted_ : bool
        Whether the model has been fitted.
    converged_ : bool or None
        Whether the optimizer converged. None before fitting.
    actual_iterations_ : int or None
        Number of iterations the optimizer performed. None before fitting.

    Notes
    -----
    **Critical**: When rank_ratio < 1.0, FastSurvivalSVM.predict() returns
    survival times (higher = longer survival), not risk scores. This wrapper
    automatically negates the output in regression mode to maintain the
    mlsurv convention that higher scores = higher risk.

    Raw SVM decision values are ordinal (well-discriminated for the C-index)
    but not log-hazards, so feeding them directly to a Breslow estimator yields
    well-discriminated but miscalibrated survival probabilities. Calibration is
    handled by the learner: ``train()`` attaches a ``breslow_slope`` calibrator
    to fssvm by default (a van Houwelingen refit fit on held-out OOF of the
    training cohort; van Houwelingen 2000). Pass ``train(calibrate=False)`` for
    an uncalibrated fssvm.

    References for recalibration: van Houwelingen HC. (2000) Validation,
    calibration, revision and combination of prognostic survival models.
    *Statistics in Medicine* 19(24):3401-3415.

    Examples
    --------
    >>> from mlsurv.models import FSSVMModel
    >>> model = FSSVMModel(alpha=1.0, rank_ratio=1.0)
    >>> model.fit(X_train, y_train)
    >>> risk_scores = model.predict(X_test)

    References
    ----------
    - Van Belle, V., Pelckmans, K., Van Huffel, S., & Suykens, J.A. (2011).
      Support vector methods for survival analysis: a comparison between
      ranking and regression approaches.
    """

    name = "fssvm"
    display_name = "FSSVM"
    color = "#CC79A7"
    supports_random_state = True
    prefer_sequential_cv = False
    complexity = "linear"
    # FSSVM optimizes a concordance-ranking objective (Pölsterl et al. 2015),
    # not a hazard model — there is no PH parameterization to violate and no
    # log-hazard linearity claim. Treat as assumption-free for tier purposes.
    assumes_ph = False
    assumes_linearity = False
    # Raw SVM decision values are ordinal (well-discriminated for C-index) but
    # not log-hazards. Survival probabilities require post-hoc recalibration.
    rank_based = True
    # The SVM decision value is a single linear predictor, so a Breslow baseline
    # can be refit on it — eligible for the breslow_slope calibrator.
    score_based = True

    def __init__(
        self,
        alpha=1.0,
        rank_ratio=1.0,
        max_iter=100,
        tol=None,
        optimizer='avltree',
        fit_intercept=False,
        random_state=None,
    ):
        self.alpha = alpha
        self.rank_ratio = rank_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.optimizer = optimizer
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.model_ = None
        self.converged_ = None
        self.actual_iterations_ = None

    def fit(self, X, y: SurvivalArray):
        """
        Fit the Fast Survival SVM.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Training features.
        y : SurvivalArray
            Survival outcomes.

        Returns
        -------
        self : FSSVMModel
            Fitted model.

        Raises
        ------
        ValueError
            If invalid parameter combinations are used.
        """
        self._validate_data(X, y)

        # PRSVM and simple cause segfaults on Windows (scikit-survival 0.26.0
        # C extension bug). Check first — no point validating parameters for
        # an optimizer that will segfault regardless.
        if sys.platform == "win32" and self.optimizer in ('PRSVM', 'simple'):
            raise ValueError(
                f"optimizer='{self.optimizer}' is not supported on Windows due to a "
                "segfault in scikit-survival's C extension. "
                "Use 'avltree', 'rbtree', or 'direct-count' instead."
            )

        # fit_intercept=True is not meaningful in pure ranking mode
        # (rank_ratio=1.0); sksurv rejects this combination with ValueError
        if self.fit_intercept and self.rank_ratio == 1.0:
            raise ValueError(
                "fit_intercept=True is not meaningful when rank_ratio=1.0 "
                "(pure ranking mode). Set fit_intercept=False or use "
                "rank_ratio < 1.0."
            )

        # Validate parameter combinations to avoid crashes
        # PRSVM and simple only work with pure ranking (rank_ratio=1.0)
        if self.optimizer in ('PRSVM', 'simple') and self.rank_ratio != 1.0:
            raise ValueError(
                f"optimizer='{self.optimizer}' only supports rank_ratio=1.0 (pure ranking). "
                f"Got rank_ratio={self.rank_ratio}. Use optimizer='avltree' or 'rbtree' instead."
            )

        self.model_ = self._fit_svm_only(X, y, set_state=True)
        self.is_fitted_ = True

        if not self.converged_:
            warnings.warn(
                f"FSSVM did not converge in {self.actual_iterations_} iterations "
                f"(max_iter={self.max_iter}). Consider increasing max_iter. "
                f"Optimizer: {self.model_.optimizer_result_.message}",
                SuggestionWarning,
                stacklevel=2,
            )

        # Regression mode (rank_ratio < 1.0) produces approximate Breslow
        # survival curves. The caveat is a fixed property of the fitted model,
        # so surface it once here at fit time rather than on every later
        # predict_survival_function() call (which floods eval/bootstrap loops).
        if self.rank_ratio < 1.0:
            _warn_regression_mode_once(self.rank_ratio)

        # Fit Breslow estimator on full training data — the model's raw
        # survival baseline (the learner's breslow_slope calibrator refits on
        # held-out OOF for the deployed S(t|x)).
        # Use wrapper's predict() which ensures higher = higher risk in all modes.
        risk_scores = self.predict(X)
        self.breslow_ = breslow_call_with_clamp(
            risk_scores, BreslowEstimator().fit,
            y['event'], y['time'], model_name="FSSVM",
        )

        return self

    def _fit_svm_only(self, X, y, set_state=True):
        """Build and fit a FastSurvivalSVM without touching prediction state.

        With ``set_state=True`` (used for the final refit), records convergence
        on ``self``. With ``set_state=False`` (used inside CV folds), returns
        the fitted estimator without mutating ``self``.
        """
        svm = FastSurvivalSVM(
            alpha=self.alpha,
            rank_ratio=self.rank_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
            optimizer=self.optimizer,
            fit_intercept=self.fit_intercept,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            svm.fit(X, y)
        if set_state:
            self.actual_iterations_ = svm.n_iter_
            self.converged_ = svm.optimizer_result_.success
        return svm

    def _predict_risk_with(self, svm, X):
        """Predict risk with the same negation rule as ``predict()``.

        Used inside CV folds where ``self.model_`` is not yet set.
        """
        preds = svm.predict(X)
        if self.rank_ratio < 1.0:
            return -preds
        return preds

    def predict(self, X):
        """
        Predict risk scores.

        Returns risk scores where higher = higher risk (shorter survival).
        Automatically negated when rank_ratio < 1.0 to maintain consistent
        semantics (in regression mode, raw output is survival time).

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        risk_scores : ndarray
            Predicted risk scores.
        """
        self._check_is_fitted()
        predictions = self.model_.predict(X)

        # Negate if in regression mode (rank_ratio < 1)
        # In regression mode, predict() returns survival times (higher = better)
        # We need higher = higher risk, so negate
        if self.rank_ratio < 1.0:
            return -predictions
        return predictions

    def predict_survival_function(self, X, times=None, return_array=False):
        """
        Predict survival function for each sample using Breslow estimation.

        Uses the Breslow estimator fitted during training to convert risk scores
        into survival probability curves. This assumes proportional hazards.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        times : array-like or None, default=None
            If provided, evaluate StepFunctions at these times and return
            an ndarray of shape (n_samples, n_times).
        return_array : bool, default=False
            If True, return ndarray (samples x times) instead of StepFunction
            objects.  Ignored when *times* is given (always returns ndarray).

        Returns
        -------
        surv_funcs : list of StepFunction or ndarray
            Survival functions for each sample. When ``times`` is given or
            ``return_array=True``, returns ndarray (n_samples, n_times).

        Warnings
        --------
        When ``rank_ratio < 1.0`` (regression mode), the Breslow approximation
        may not be valid as the model does not strictly follow proportional
        hazards assumptions, and Brier scores from these curves should be
        interpreted with caution. This caveat is emitted at most once per
        session (deduped by ``_warn_regression_mode_once``): normally at fit
        time, or here on the first survival-curve call for a model restored
        from disk (which skips ``fit``). Either way eval/bootstrap loops that
        call this repeatedly aren't flooded.
        """
        self._check_is_fitted()
        # Covers models reloaded via load_learner() — they restore breslow_ and
        # rank_ratio without re-running fit(), so the fit-time emission never
        # happened this session. The module-level dedup keeps it to one line.
        if self.rank_ratio < 1.0:
            _warn_regression_mode_once(self.rank_ratio)
        risk_scores = self.predict(X)
        surv_funcs = breslow_call_with_clamp(
            risk_scores, self.breslow_.get_survival_function,
            model_name="FSSVM",
        )
        return self._format_step_funcs(surv_funcs, times, return_array)

    def _format_step_funcs(self, surv_funcs, times, return_array):
        """Apply ``times``/``return_array`` post-processing to step functions."""
        if times is not None:
            times = np.asarray(times)
            result = np.array([fn(times) for fn in surv_funcs])
            self._warn_if_large_array(result)
            return result
        if return_array:
            result = np.vstack([fn.y for fn in surv_funcs])
            self._warn_if_large_array(result)
            return result
        return surv_funcs

    def predict_cumulative_hazard_function(self, X, return_array=False):
        """
        Predict cumulative hazard function for each sample using Breslow estimation.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        return_array : bool, default=False
            If True, return ndarray (samples x times) instead of StepFunction
            objects.

        Returns
        -------
        cum_hazard : list of StepFunction or ndarray
            Cumulative hazard functions for each sample.
        """
        self._check_is_fitted()
        risk_scores = self.predict(X)
        funcs = breslow_call_with_clamp(
            risk_scores, self.breslow_.get_cumulative_hazard_function,
            model_name="FSSVM",
        )
        if return_array:
            result = np.vstack([fn.y for fn in funcs])
            self._warn_if_large_array(result)
            return result
        return funcs

    def get_coefficients(self):
        """
        Return model coefficients.

        Returns
        -------
        pd.Series or None
            Coefficients indexed by feature name, or None if not available.
        """
        self._check_is_fitted()
        if hasattr(self.model_, 'coef_'):
            return pd.Series(
                self.model_.coef_,
                index=(
                    self.feature_names_in_ if hasattr(self, 'feature_names_in_')
                    else range(len(self.model_.coef_))
                ),
                name='coefficient'
            )
        return None
