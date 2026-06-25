"""Random Survival Forest model wrapper."""

from sksurv.ensemble import RandomSurvivalForest

from mlsurv.models.base import (
    BaseSurvivalModel, ImpurityImportanceMixin, SurvivalArray,
)


class RSFModel(ImpurityImportanceMixin, BaseSurvivalModel):
    """
    Random Survival Forest model wrapper.

    Wraps scikit-survival's RandomSurvivalForest with consistent interface.
    Ensemble of survival trees using the log-rank splitting criterion.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum tree depth. None = unlimited.
    min_samples_split : int, default=6
        Minimum samples to split a node.
    min_samples_leaf : int, default=3
        Minimum samples at leaf node.
    max_features : int, float, str, or None, default='sqrt'
        Features to consider for splits:
        - int: exact number
        - float: fraction
        - 'sqrt': sqrt(n_features)
        - 'log2': log2(n_features)
        - None: all features
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
        Must be True when ``oob_score=True``.
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization
        concordance. Requires ``bootstrap=True``. When enabled, the fitted
        ``model_.oob_score_`` attribute holds the OOB concordance, and the
        Trained Model Details report section will display it.
    n_jobs : int, default=1
        Number of parallel jobs for tree building.
        - 1: Sequential execution (default)
        - -1: Use all available CPU cores
        - n > 1: Use n CPU cores
        Note: The default is 1 to allow SurvivalLearner to control
        parallelization at the pipeline level.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    model_ : RandomSurvivalForest
        The fitted scikit-survival model.
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    >>> from mlsurv.models import RSFModel
    >>> model = RSFModel(n_estimators=500, max_depth=10)
    >>> model.fit(X_train, y_train)
    >>> risk_scores = model.predict(X_test)

    References
    ----------
    - Ishwaran, H., Kogalur, U.B., Blackstone, E.H., & Lauer, M.S. (2008).
      Random survival forests.
    """

    name = "rsf"
    display_name = "RSF"
    color = "#D55E00"
    supports_random_state = True
    prefer_sequential_cv = False
    has_internal_parallelism = True
    complexity = "tree_ensemble"

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features='sqrt',
        bootstrap=True,
        oob_score=False,
        n_jobs=1,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y: SurvivalArray):
        """
        Fit the Random Survival Forest.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Training features.
        y : SurvivalArray
            Survival outcomes.

        Returns
        -------
        self : RSFModel
            Fitted model.

        Raises
        ------
        ValueError
            If `oob_score` is requested without `bootstrap` enabled.
        """
        self._validate_data(X, y)
        if self.oob_score and not self.bootstrap:
            raise ValueError(
                "RSFModel: oob_score=True requires bootstrap=True "
                "(sksurv does not compute OOB scores without bootstrapping)."
            )
        self.model_ = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
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
            Feature importances (sum to 1.0).

        Raises
        ------
        NotImplementedError
            scikit-survival >= 0.26 deprecated impurity-based importance.
            Use get_feature_importance(method='permutation') instead.
        """
        self._check_is_fitted()
        try:
            return self.model_.feature_importances_
        except NotImplementedError:
            raise NotImplementedError(
                "Impurity-based feature importance is no longer supported in "
                "scikit-survival >= 0.26. Use get_feature_importance(method='permutation', "
                "X=X_test, y=y_test) instead."
            )

