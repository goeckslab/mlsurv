"""CoxNet (Elastic Net penalized Cox) model wrapper."""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sksurv.linear_model import CoxnetSurvivalAnalysis

from mlsurv.models.base import BaseSurvivalModel, SurvivalArray
from mlsurv.models._overflow import predict_with_clamp
from mlsurv._warnings import FitFailureWarning, SuggestionWarning


class CoxNetModel(BaseSurvivalModel):
    """
    CoxNet (Elastic Net penalized Cox) model wrapper.

    Wraps scikit-survival's CoxnetSurvivalAnalysis with consistent interface.
    Uses both L1 (Lasso) and L2 (Ridge) penalties for feature selection
    and regularization.

    Parameters
    ----------
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter. 1.0 = Lasso, 0.0 = Ridge.
    alpha_min_ratio : float, default=0.01
        Minimum alpha ratio for regularization path.
    n_alphas : int, default=100
        Number of alphas in the regularization path.
    alpha : {'auto', 'min'}, float, or int, default='auto'
        Alpha (regularization strength) selection strategy:
        - 'auto': Use K-fold CV to select best alpha from regularization path.
          This is the recommended default since scikit-survival lacks a
          CoxnetSurvivalAnalysisCV class (unlike sklearn's LassoCV/ElasticNetCV).
        - 'min': Use smallest alpha (least regularization). This was the
          original scikit-survival default but risks overfitting.
        - float (e.g., 0.05): Use this specific alpha value. Useful when you
          know the optimal alpha from prior analysis or want reproducibility.
        - int (e.g., 42): Use alpha at this index in the regularization path.
          Index 0 = strongest regularization. Values beyond the actual path
          length are clamped to the last (weakest) alpha. Used by Optuna
          tuning to avoid nested CV overhead.
    cv_folds : int, default=5
        Number of CV folds for alpha selection when alpha='auto'.
        Ignored when alpha is 'min', a specific float, or an int index.
    fit_baseline_model : bool, default=True
        If True, fit baseline hazard for predict_survival_function().
        Set to False during tuning (faster) when only C-index is needed.
    max_iter : int, default=100000
        Maximum number of iterations.
    tol : float, default=1e-7
        Convergence tolerance.
    random_state : int or None, default=None
        Seed for the K-fold split used in alpha='auto' selection. Only the
        fold assignment is randomized (shuffle=True); with a fixed seed the
        selected alpha is reproducible. ``None`` gives genuinely random folds
        (the sklearn default). SurvivalLearner injects the configured
        ``setup(random_state=...)`` here automatically.

    Attributes
    ----------
    model_ : CoxnetSurvivalAnalysis
        The fitted scikit-survival model.
    selected_alpha_ : float
        The alpha value used for predictions (set after fit).
    is_fitted_ : bool
        Whether the model has been fitted.

    Notes
    -----
    **Why internal CV for alpha?**

    scikit-survival's CoxnetSurvivalAnalysis fits a regularization path but
    provides no built-in cross-validation for alpha selection.
    Without CV, predictions use the smallest alpha by default, which provides
    minimal regularization and risks overfitting.

    This implementation adds sklearn-style CV (like LassoCV/ElasticNetCV) to
    automatically select the best alpha. This is the standard approach in
    penalized regression.

    **Interaction with Optuna tuning:**

    When using SurvivalLearner.tune(), Optuna jointly optimizes l1_ratio
    (L1/L2 mixing) and alpha_index (position in the regularization path,
    0 to n_alphas-1). Each trial fits the regularization path once and
    picks the alpha at the suggested index — no internal CV overhead.
    Optuna's TPE sampler replaces exhaustive CV for alpha selection.

    Examples
    --------
    >>> from mlsurv.models import CoxNetModel
    >>> # Default: CV-based alpha selection
    >>> model = CoxNetModel(l1_ratio=0.9)
    >>> model.fit(X_train, y_train)
    >>> print(f"Selected alpha: {model.selected_alpha_}")

    >>> # Use minimum alpha (least regularization, old default)
    >>> model = CoxNetModel(l1_ratio=0.9, alpha='min')

    >>> # Use specific alpha value
    >>> model = CoxNetModel(l1_ratio=0.9, alpha=0.05)

    References
    ----------
    - Simon, N., Friedman, J., Hastie, T., & Tibshirani, R. (2011).
      Regularization Paths for Cox's Proportional Hazards Model via
      Coordinate Descent.
    """

    name = "coxnet"
    display_name = "CoxNet"
    color = "#E69F00"
    supports_random_state = True
    prefer_sequential_cv = True
    assumes_ph = True
    assumes_linearity = True
    score_based = True  # single linear predictor → Breslow baseline

    def __init__(
        self,
        l1_ratio=0.5,
        alpha_min_ratio=0.01,
        n_alphas=100,
        alpha='auto',
        cv_folds=5,
        fit_baseline_model=True,
        max_iter=100000,
        tol=1e-7,
        random_state=None
    ):
        self.l1_ratio = l1_ratio
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alphas = n_alphas
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.fit_baseline_model = fit_baseline_model
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.model_ = None
        self.selected_alpha_ = None

    def fit(self, X, y: SurvivalArray):
        """
        Fit the CoxNet model.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Training features.
        y : SurvivalArray
            Survival outcomes.

        Returns
        -------
        self : CoxNetModel
            Fitted model.

        Raises
        ------
        ValueError
            If ``alpha`` (as a path index) is negative, if the
            regularization path produces no alphas for the given
            ``l1_ratio``, or if ``alpha`` is not ``'auto'``, ``'min'``,
            a float, or an int.

        Notes
        -----
        When alpha='auto', the method:
        1. Fits the regularization path to get candidate alphas
        2. Uses K-fold CV to select the best alpha
        3. Refits with the selected alpha and baseline model (if requested)
        """
        self._validate_data(X, y)

        # Handle alpha as path index (integer) — used during Optuna tuning.
        # np.integer covers np.int64 etc. returned by Optuna / numpy.
        if (isinstance(self.alpha, (int, np.integer))
                and not isinstance(self.alpha, bool)):
            if self.alpha < 0:
                raise ValueError(
                    f"alpha as path index must be non-negative, got {self.alpha}. "
                    f"Valid range: 0 to {self.n_alphas - 1}."
                )
            alphas = self._fit_regularization_path(X, y)
            if len(alphas) == 0:
                raise ValueError(
                    f"Regularization path produced no alphas for "
                    f"l1_ratio={self.l1_ratio}. Check that the data is valid."
                )
            idx = min(self.alpha, len(alphas) - 1)
            if idx != self.alpha:
                warnings.warn(
                    f"alpha_index={self.alpha} exceeds regularization path "
                    f"length ({len(alphas)} alphas). Clamping to index {idx}.",
                    SuggestionWarning,
                    stacklevel=2,
                )
            self.selected_alpha_ = alphas[idx]
            return self._fit_final_model(X, y)

        # Handle specific alpha value (no path fitting needed)
        if isinstance(self.alpha, (float, np.floating)) and not isinstance(self.alpha, bool):
            self.selected_alpha_ = self.alpha
            return self._fit_final_model(X, y)

        # Fit regularization path first (no baseline model for speed)
        alphas = self._fit_regularization_path(X, y)

        # Select alpha based on strategy
        if self.alpha == 'auto':
            n_samples = X.shape[0]
            # Validate cv_folds for small datasets
            if self.cv_folds < 2:
                warnings.warn(
                    f"cv_folds={self.cv_folds} is invalid (must be >= 2). "
                    f"Falling back to alpha='min'.",
                    FitFailureWarning,
                )
                self.selected_alpha_ = alphas[-1]
            elif n_samples < self.cv_folds:
                warnings.warn(
                    f"Dataset has only {n_samples} samples, which is less than "
                    f"cv_folds={self.cv_folds}. Falling back to alpha='min'.",
                    FitFailureWarning,
                )
                self.selected_alpha_ = alphas[-1]
            else:
                self.selected_alpha_ = self._select_alpha_cv(X, y, alphas)
        elif self.alpha == 'min':
            self.selected_alpha_ = alphas[-1]  # Smallest alpha (least regularization)
        else:
            raise ValueError(
                f"alpha must be 'auto', 'min', a float, or an int (path index), "
                f"got {self.alpha!r}"
            )

        return self._fit_final_model(X, y)

    def _fit_regularization_path(self, X, y):
        """Fit the regularization path and return the array of alphas.

        Fits with ``fit_baseline_model=False`` for speed since only the
        alpha sequence is needed.

        Returns
        -------
        ndarray
            Alpha values from the fitted regularization path.
        """
        path_model = CoxnetSurvivalAnalysis(
            l1_ratio=self.l1_ratio,
            alpha_min_ratio=self.alpha_min_ratio,
            n_alphas=self.n_alphas,
            fit_baseline_model=False,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        path_model.fit(X, y)
        return path_model.alphas_

    def _fit_final_model(self, X, y):
        """Fit the final model with ``selected_alpha_`` and mark as fitted.

        Must be called after ``self.selected_alpha_`` has been set.

        Returns
        -------
        self : CoxNetModel
            Fitted model.
        """
        self.model_ = CoxnetSurvivalAnalysis(
            l1_ratio=self.l1_ratio,
            alphas=[self.selected_alpha_],
            fit_baseline_model=self.fit_baseline_model,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.model_.fit(X, y)
        self.is_fitted_ = True
        return self

    def _select_alpha_cv(self, X, y, alphas):
        """Select best alpha using K-fold cross-validation.

        Delegates the fold-scoring loop to the shared
        :func:`mlsurv._shared.select_alpha_by_cv_concordance` (mirrored by the
        feature-selection path in preprocessing/selection.py). This model does
        not surface failed-fold counts, so the returned count is discarded.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : SurvivalArray
            Survival outcomes.
        alphas : array-like
            Candidate alpha values from regularization path.

        Returns
        -------
        float
            The alpha with highest mean C-index across folds.
        """
        # Deferred import: mlsurv._shared pulls in mlsurv.constants, which
        # imports mlsurv.models — a module-level import here would be circular.
        from mlsurv._shared import select_alpha_by_cv_concordance

        alpha, _ = select_alpha_by_cv_concordance(
            X, y, alphas,
            l1_ratio=self.l1_ratio,
            cv_folds=self.cv_folds,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
            no_valid_scores_stacklevel=4,
        )
        return alpha

    def _predict_with_clamp(self, func_name, X, return_array):
        """Call ``func_name`` on the sksurv model with clamped linear predictors.

        Requires ``fit_baseline_model=True`` during ``fit()``.

        Delegates to :func:`mlsurv.models._overflow.predict_with_clamp`.
        See that function's docstring for details on the monkey-patching
        approach and thread-safety caveats.
        """
        self._check_is_fitted()
        if not self.fit_baseline_model:
            raise RuntimeError(
                f"{func_name}() requires fit_baseline_model=True. "
                f"Refit the model with fit_baseline_model=True."
            )
        result = predict_with_clamp(
            self.model_, func_name, X, return_array,
            model_display_name="CoxNet",
        )
        if return_array:
            self._warn_if_large_array(result)
        return result

    def predict_survival_function(self, X, return_array=False):
        """Predict survival function with overflow-safe linear predictors.

        Clamps the linear predictor to ``[-20, 20]`` before exponentiation
        to prevent float64 overflow in the Breslow estimator.

        Requires ``fit_baseline_model=True`` during ``fit()``.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        return_array : bool, default=False
            If True, return array instead of StepFunction objects.

        Returns
        -------
        list of StepFunction or ndarray
            Survival functions for each sample.

        Raises
        ------
        RuntimeError
            If fit_baseline_model was False during fit().
        """
        return self._predict_with_clamp(
            "predict_survival_function", X, return_array
        )

    def predict_cumulative_hazard_function(self, X, return_array=False):
        """Predict cumulative hazard function with overflow-safe linear predictors.

        Clamps the linear predictor to ``[-20, 20]`` before exponentiation
        to prevent float64 overflow in the Breslow estimator.

        Requires ``fit_baseline_model=True`` during ``fit()``.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        return_array : bool, default=False
            If True, return array instead of StepFunction objects.

        Returns
        -------
        list of StepFunction or ndarray
            Cumulative hazard functions for each sample.

        Raises
        ------
        RuntimeError
            If fit_baseline_model was False during fit().
        """
        return self._predict_with_clamp(
            "predict_cumulative_hazard_function", X, return_array
        )

    def get_coefficients(self):
        """
        Return model coefficients for the selected alpha.

        Returns
        -------
        pd.Series
            Coefficients indexed by feature name.
            Features dropped by L1 penalty have coefficient = 0.

        Notes
        -----
        Uses the same alpha as predict() (see selected_alpha_ attribute).
        Since the model is refit with a single alpha, coef_ is 1D.
        """
        self._check_is_fitted()
        coef = self.model_.coef_
        # After refitting with single alpha, coef_ should be 1D
        # but handle 2D case for safety (use last/only column)
        if coef.ndim == 2:
            coef = coef[:, -1]
        return pd.Series(
            coef,
            index=self.feature_names_in_,
            name='coefficient'
        )

    def freeze_for_resampling(self):
        """Return a clone with ``alpha`` pinned to ``selected_alpha_``.

        Unlike the base default — which accepts unfitted models — this
        override requires the model to be fitted, because the pinned value
        comes from the fitted ``selected_alpha_``. Calling on an unfitted
        CoxNet raises ``NotFittedError``.

        The default ``clone(self)`` only carries ``__init__`` params, so a
        coxnet fitted with ``alpha='auto'`` or ``alpha=<int path index>``
        would re-run K-fold CV or recompute the regularization path on
        every bootstrap resample. Resampling the training data
        shifts the path's alpha sequence, so the chosen alpha drifts
        materially across resamples even when the path index is fixed.
        Pinning the resolved float here forces ``fit()`` down the
        ``_fit_final_model`` path with a fixed alpha, isolating bootstrap
        variability to coefficient refit.
        """
        self._check_is_fitted()
        frozen = clone(self)
        frozen.alpha = float(self.selected_alpha_)
        return frozen

    @property
    def alphas_(self):
        """Return the alphas used during fitting.

        Note: After fitting, the model is refit with only selected_alpha_,
        so this returns a single-element array.
        """
        self._check_is_fitted()
        return self.model_.alphas_

    @property
    def alpha_(self):
        """Return the alpha used for predictions.

        This is an alias for selected_alpha_ for backwards compatibility.
        """
        self._check_is_fitted()
        return self.selected_alpha_
