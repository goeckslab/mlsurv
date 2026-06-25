"""Gradient Boosting Survival Analysis model wrapper."""

import warnings

from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from mlsurv.models.base import (
    BaseSurvivalModel, ImpurityImportanceMixin, SurvivalArray,
)
from mlsurv._warnings import SuggestionWarning


class GBSAModel(ImpurityImportanceMixin, BaseSurvivalModel):
    """
    Gradient Boosting Survival Analysis model wrapper.

    Wraps scikit-survival's GradientBoostingSurvivalAnalysis with consistent
    interface. Fits an ensemble of regression trees in a stage-wise fashion.

    Parameters
    ----------
    loss : str, default='coxph'
        Loss function:
        - 'coxph': Cox partial likelihood (default)
        - 'squared': Squared error
        - 'ipcwls': IPCW least squares
    learning_rate : float, default=0.1
        Shrinkage coefficient for tree contributions.
    n_estimators : int, default=100
        Number of regression trees.
    max_depth : int, default=3
        Maximum tree depth (shallow for weak learners).
    min_samples_split : int, default=2
        Minimum samples to split a node.
    min_samples_leaf : int, default=1
        Minimum samples at leaf node.
    max_features : int, float, str, or None, default=None
        Features to consider for splits.
    subsample : float, default=1.0
        Fraction of samples for stochastic gradient boosting.
    dropout_rate : float, default=0.0
        DART regularization dropout rate.
    n_iter_no_change : int or None, default=10
        Stop training if validation score does not improve for this many
        consecutive iterations. Set to ``None`` to disable early stopping.
    validation_fraction : float, default=0.1
        Fraction of training data held out for early stopping monitoring.
        Only used when ``n_iter_no_change`` is not ``None``.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    model_ : GradientBoostingSurvivalAnalysis
        The fitted scikit-survival model.
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    >>> from mlsurv.models import GBSAModel
    >>> model = GBSAModel(n_estimators=200, learning_rate=0.05)
    >>> model.fit(X_train, y_train)
    >>> risk_scores = model.predict(X_test)

    References
    ----------
    - Ridgeway, G. (1999). The state of boosting.
    - Hothorn, T., Bühlmann, P., Dudoit, S., Molinaro, A., & Van Der Laan, M.J.
      (2006). Survival ensembles.
    """

    name = "gbsa"
    display_name = "GBSA"
    color = "#009E73"
    supports_random_state = True
    prefer_sequential_cv = False
    complexity = "tree_ensemble"
    # Default loss='coxph' fits the Cox partial likelihood and produces a
    # single time-invariant risk score, so hazard ratios between subjects
    # are constant by construction. If a user sets loss='ipcwls' (or any
    # future non-Cox loss) via model_params, this class-level flag will
    # no longer reflect the resolved assumption — membership should then
    # be gated on the instance's `loss` attribute rather than the class
    # flag. The deferred per-instance resolution is not yet implemented.
    assumes_ph = True
    score_based = True  # single linear predictor → Breslow baseline

    def __init__(
        self,
        loss='coxph',
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        subsample=1.0,
        dropout_rate=0.0,
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=None
    ):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.subsample = subsample
        self.dropout_rate = dropout_rate
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y: SurvivalArray):
        """
        Fit the Gradient Boosting Survival model.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Training features.
        y : SurvivalArray
            Survival outcomes.

        Returns
        -------
        self : GBSAModel
            Fitted model.
        """
        self._validate_data(X, y)
        params = dict(
            loss=self.loss,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            subsample=self.subsample,
            dropout_rate=self.dropout_rate,
            random_state=self.random_state,
        )
        if self.n_iter_no_change is not None:
            viable, info = self._validate_early_stopping_viability(
                y, self.validation_fraction,
                param_name='validation_fraction',
            )
            if viable:
                params['n_iter_no_change'] = self.n_iter_no_change
                params['validation_fraction'] = self.validation_fraction
            else:
                warnings.warn(
                    f"Dataset too small (n={info['n']}, "
                    f"events={info['n_events']}) for early stopping with "
                    f"validation_fraction={self.validation_fraction}. "
                    f"Disabling early stopping.",
                    SuggestionWarning,
                    stacklevel=2,
                )
        self.model_ = GradientBoostingSurvivalAnalysis(**params)
        self.model_.fit(X, y)
        self.is_fitted_ = True
        return self

    @property
    def feature_importances_(self):
        """
        Return impurity-based feature importance.

        Returns
        -------
        ndarray
            Feature importances.
        """
        self._check_is_fitted()
        return self.model_.feature_importances_

