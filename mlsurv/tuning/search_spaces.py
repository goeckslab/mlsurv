"""Default search spaces and auto-configuration for hyperparameter tuning.

References
----------
Random Forest:
    Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
    https://doi.org/10.1023/A:1010933404324

    Probst, P., Wright, M. N., & Boulesteix, A.-L. (2019). Hyperparameters
    and tuning strategies for random forest. WIREs Data Mining and Knowledge
    Discovery, 9(3), e1301. https://doi.org/10.1002/widm.1301

    Boehmke, B., & Greenwell, B. M. (2020). Hands-On Machine Learning with R.
    Chapter 11: Random Forests. https://bradleyboehmke.github.io/HOML/

Gradient Boosting:
    Friedman, J. H. (2001). Greedy function approximation: A gradient boosting
    machine. Annals of Statistics, 29(5), 1189-1232.
    https://doi.org/10.1214/aos/1013203451

    Natekin, A., & Knoll, A. (2013). Gradient boosting machines, a tutorial.
    Frontiers in Neurorobotics, 7, 21. https://doi.org/10.3389/fnbot.2013.00021

Random Survival Forest:
    Dauda, K. A., et al. (2022). Optimal tuning of random survival forest
    hyperparameter with an application to liver disease. Malaysian Journal of
    Medical Sciences, 29(6), 67-76. https://doi.org/10.21315/mjms2022.29.6.7

DeepSurv:
    Katzman, J. L., et al. (2018). DeepSurv: Personalized treatment recommender
    system using a Cox proportional hazards deep neural network. BMC Medical
    Research Methodology, 18(1), 24. https://doi.org/10.1186/s12874-018-0482-1

EPV and Sample Size:
    Riley, R. D., et al. (2019). Minimum sample size for developing a
    multivariable prediction model: Part II. Statistics in Medicine, 38(7),
    1276-1296. https://doi.org/10.1002/sim.7992

    Peduzzi, P., et al. (1996). A simulation study of the number of events
    per variable in logistic regression analysis. Journal of Clinical
    Epidemiology, 49(12), 1373-1379.
    https://doi.org/10.1016/S0895-4356(96)00236-3

    van Smeden, M., et al. (2016). No rationale for 1 variable per 10 events
    criterion for binary logistic regression analysis. BMC Medical Research
    Methodology, 16, 163. https://doi.org/10.1186/s12874-016-0267-3

    van Smeden, M., et al. (2019). Sample size for binary logistic prediction
    models: Beyond events per variable criteria. Statistical Methods in
    Medical Research, 28(8), 2455-2474.
    https://doi.org/10.1177/0962280218784726

Penalized Models with Few Events:
    Ambler, G., Seaman, S., & Omar, R. Z. (2012). An evaluation of penalised
    survival methods for developing prognostic models with rare events.
    Statistics in Medicine, 31(11-12), 1150-1161.
    https://doi.org/10.1002/sim.4371

    Pavlou, M., et al. (2016). Review and evaluation of penalised regression
    methods for risk prediction in low-dimensional data with few events.
    Statistics in Medicine, 35(7), 1159-1177. https://doi.org/10.1002/sim.6782

    Zou, H., & Hastie, T. (2005). Regularization and variable selection via
    the elastic net. Journal of the Royal Statistical Society: Series B,
    67(2), 301-320. https://doi.org/10.1111/j.1467-9868.2005.00503.x

    Pölsterl, S., Navab, N., & Katouzian, A. (2015). Fast training of support
    vector machines for survival analysis. ECML PKDD 2015, LNCS 9285, 243-259.
    https://doi.org/10.1007/978-3-319-23525-7_15

Hyperparameter Optimization:
    Bischl, B., et al. (2023). Hyperparameter optimization: Foundations,
    algorithms, best practices, and open challenges. WIREs Data Mining and
    Knowledge Discovery, 13(2), e1484. https://doi.org/10.1002/widm.1484
"""

import logging
import warnings

import numpy as np

from mlsurv._warnings import MethodologicalWarning, SuggestionWarning

logger = logging.getLogger(__name__)


# Default parameter ranges for preprocessing components
# Used for documentation and as reference for users building custom search spaces
PREPROCESSING_DEFAULTS = {
    'imputer': {
        'KNNImputer': {'n_neighbors': (3, 15)},
    },
    'scaler': {
        'PCA': {'n_components': (0.8, 0.99)},
    },
    'feature_selector': {
        # default k='auto' (Harrell m/15 rule); range below is for manual tuning reference
        'TopKCoxSelector': {'k': (5, None)},  # None = n_features
        'VIFSelector': {'threshold': (5.0, 15.0)},
        'CorrelationSelector': {'threshold': (0.7, 0.95)},
    },
}


def get_preprocessing_defaults(component):
    """
    Get default parameter search space for a preprocessing component.

    Parameters
    ----------
    component : str
        Component name: 'imputer', 'scaler', or 'feature_selector'.

    Returns
    -------
    dict
        Default search space for the component, keyed by class name.

    Examples
    --------
    >>> get_preprocessing_defaults('feature_selector')
    {'TopKCoxSelector': {'k': (5, None)}, 'VIFSelector': {'threshold': (5.0, 15.0)}, ...}
    """
    return PREPROCESSING_DEFAULTS.get(component, {}).copy()


# Default search spaces per model
DEFAULT_SPACES = {
    'coxph': {},  # No hyperparameters to tune

    'coxnet': {
        # CoxNet: Jointly optimize l1_ratio and alpha_index
        #
        # - l1_ratio: L1/L2 mixing parameter (Ridge↔Lasso)
        # - alpha_index: Position in the regularization path (0=strongest,
        #   higher=weaker; clamped to actual path length). Each trial fits
        #   the path once and picks the alpha at this index — no internal
        #   CV overhead.
        #
        # Optuna's TPE sampler replaces exhaustive CV for alpha selection,
        # eliminating nested CV (outer tuning CV × inner alpha CV).
        'l1_ratio': (0.0, 1.0),
        'alpha_index': (0, 99),  # Coupled to CoxNetModel.n_alphas default (100)
    },

    'rsf': {
        'n_estimators': (100, 300),
        'max_depth': (5, 15),
        'min_samples_split': (2, 50),
        'min_samples_leaf': (1, 30),
        'max_features': ['sqrt', 'log2', 0.5, 0.8, 1.0],
        'bootstrap': [True, False],
    },

    'gbsa': {
        'loss': ['coxph'],
        'learning_rate': (0.01, 0.3, 'log'),
        'n_estimators': (100, 300),
        'max_depth': (2, 8),
        'min_samples_split': (2, 50),
        'min_samples_leaf': (1, 30),
        'max_features': ['sqrt', 'log2', 0.5, 0.8, None],
        'subsample': (0.5, 1.0),
        'dropout_rate': (0.0, 0.3),
    },

    'fssvm': {
        'alpha': (1e-5, 100.0, 'log'),
        'max_iter': (100, 5000),
        'tol': (1e-5, 1e-2, 'log'),
        'rank_ratio': (0.0, 1.0),
        # optimizer not tuned: all optimizers solve the same convex problem
        # and produce identical solutions; avltree is hardcoded (O(n log n)).
        # fit_intercept not tuned: True causes Breslow survival curve collapse
        # (prediction scale mismatch → IBS ≈ 0.78); hardcoded to False.
    },

    'deepsurv': {
        # Based on Katzman et al. (2018) Table 3 optimal hyperparameters
        # Paper found 1 layer optimal for 3/4 datasets
        'n_layers': (1, 2),
        'n_nodes': (8, 48),  # Paper: 8 (GBSG) to 44-48 (SUPPORT/WHAS)
        'dropout': (0.1, 0.7),  # Paper: 0.15-0.66; 0.0 rarely optimal
        'activation': ['selu', 'relu'],  # SELU first - optimal for 3/4 datasets
        'weight_decay': (1e-5, 0.01, 'log'),  # L2 regularization
        # Avoid small batch sizes (<32) which cause training instability
        'batch_size': [64, 128, 256],  # Added 64 for smaller datasets
        # Paper: 0.01 (METABRIC) to 0.154 (GBSG)
        'lr': (1e-2, 2e-1, 'log'),
    },

    'xgbcox': {
        'n_estimators': (50, 500),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3, 'log'),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'gamma': (0, 5),
        'min_child_weight': (1, 10),
        'max_delta_step': (0.0, 2.0),
    },

    'xgbaft': {
        'n_estimators': (50, 500),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3, 'log'),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'gamma': (0, 5),
        'min_child_weight': (1, 10),
        # Calibration parameters (aft_loss_distribution, aft_loss_distribution_scale)
        # are intentionally absent — sigma is MLE-fit on training residuals
        # in XGBAFTModel.fit() and the family is fixed at the model default.
    },

    'deephit': {
        'n_layers': (1, 3),
        'n_nodes': (16, 256),
        'dropout': (0.0, 0.5),
        'activation': ['selu', 'relu'],
        'alpha': (0.0, 1.0),  # Likelihood-vs-ranking weight: 1.0=NLL only, 0.0=rank only
        'sigma': (0.01, 1.0, 'log'),  # Smoothing parameter
        'learning_rate': (0.001, 0.1, 'log'),
        'weight_decay': (1e-5, 0.01, 'log'),
        'batch_size': [64, 128, 256, 512],
        'num_durations': (50, 300),
    },

    'weibull_aft': {
        'alpha': (0.0, 1.0),  # L2 regularization (lifelines constraint: [0, 1], inclusive)
    },
}


def auto_n_trials(model_name, n_features=None):
    """
    Determine optimal Optuna trials based on search space complexity.

    Parameters
    ----------
    model_name : str
        Model identifier.
    n_features : int, optional
        Number of features in dataset.

    Returns
    -------
    int
        Recommended number of trials.
    """
    base_trials = {
        "coxph": 20,      # Preprocessing-only tuning
        "coxnet": 30,     # 2 params (l1_ratio, alpha_index)
        "rsf": 50,        # 6 params
        "gbsa": 55,       # 9 params
        "fssvm": 35,      # 4 params (alpha, max_iter, tol, rank_ratio)
        "deepsurv": 60,   # 6 params (neural arch)
        "xgbcox": 55,    # 8 params (similar to GBSA)
        "xgbaft": 60,     # 9 params (tree + AFT distribution)
        "deephit": 75,    # 10 params (neural arch + activation/regularization)
        "weibull_aft": 25,  # 1 param (simple)
    }

    if model_name not in base_trials:
        return 50  # Default for unknown models

    trials = base_trials[model_name]

    # Scale up for high-dimensional data
    if n_features is not None and n_features > 100:
        trials = int(trials * 1.5)

    return trials


def auto_n_bootstrap(n_samples):
    """
    Bootstrap iterations based on sample size.

    Returns 100-500 iterations scaling with sqrt(n).

    Parameters
    ----------
    n_samples : int
        Number of samples.

    Returns
    -------
    int
        Recommended number of bootstrap iterations.
    """
    return min(500, max(100, int(10 * np.sqrt(n_samples))))


def auto_cv_folds(n_events):
    """
    Auto-adjust CV folds based on event count.

    Parameters
    ----------
    n_events : int
        Number of events in dataset.

    Returns
    -------
    int
        Recommended number of CV folds.
    """
    if n_events >= 100:
        return 10
    elif n_events >= 50:
        return 5
    elif n_events >= 20:
        return 3
    else:
        warnings.warn(
            f"Only {n_events} events detected. Using 3 CV folds, "
            "but results may be unstable. Consider collecting more data.",
            MethodologicalWarning,
        )
        return 3



# Absolute-event floors gating the EPV tiers (see _epv_tier). A high EPV reached
# with a tiny feature count AND few absolute events is still unstable, so a
# permissive tier requires both a sufficient ratio AND enough absolute events.
# The literature deliberately avoids a single magic number (Riley et al. 2019;
# van Smeden et al. 2016, 2019); these are pragmatic conventions aligned with the
# rare-event instability shown by Ambler et al. (2012) and Pavlou et al. (2016),
# and sit below the event counts used across the test fixtures so ample-data
# behavior is unchanged.
_TIER_MIN_EVENTS_HIGH = 100   # required to skip narrowing entirely (defaults)
_TIER_MIN_EVENTS_MID = 50     # required to qualify for the moderate tier


# EPV-tier constraint tables for models with capacity- or shrinkage-sensitive
# parameters. Each entry maps a parameter name to (low_range, mid_range), applied
# by tier (see _epv_tier): the 'low' tier uses low_range, 'mid' uses mid_range,
# and 'high' keeps the DEFAULT_SPACES value unchanged.
#
# Two opposite mechanisms share this table:
#   - Capacity-capping models (trees, neural nets): low EPV narrows depth / leaf
#     size / node count and raises dropout floors to limit model capacity.
#   - Shrinkage-forcing models (penalized coxnet, fssvm): low EPV steers AWAY
#     from the weakly-penalized (~unpenalized maximum-likelihood) end, because
#     the cure for few events is more shrinkage, not less capacity.
#
# References:
#   RSF depth/leaf: Probst et al. (2019)
#   GBSA depth/estimators: Friedman (2001)
#   XGBCox/XGBAFT: shallow trees (depth 3-8) at low EPV, same rationale as GBSA
#   DeepSurv: Katzman et al. (2018), auto_deepsurv_defaults() EPV tiers
#   DeepHit: adapted from auto_deephit_defaults()
#   CoxNet l1_ratio (favor L2/grouping at low EPV): Zou & Hastie (2005)
#   CoxNet alpha_index / fssvm alpha (more shrinkage at low EPV):
#     Ambler et al. (2012), Pavlou et al. (2016); fssvm grid Pölsterl et al. (2015)
#   (auto_*_defaults live in mlsurv/models/_auto_defaults.py)
_EPV_CONSTRAINTS = {
    'rsf': {
        'max_depth':        ((3, 8),  (5, 12)),
        'min_samples_leaf': ((5, 30), (3, 30)),
    },
    'gbsa': {
        'n_estimators':     ((100, 200), (100, 300)),
        'max_depth':        ((2, 4),     (2, 5)),
        'min_samples_leaf': ((5, 30),    (3, 30)),
    },
    'xgbcox': {
        'max_depth':        ((3, 6),  (3, 8)),
        'min_child_weight': ((3, 10), (2, 10)),
    },
    'xgbaft': {
        'max_depth':        ((3, 6),  (3, 8)),
        'min_child_weight': ((3, 10), (2, 10)),
    },
    # Neural networks: reduce capacity and force stronger regularization
    # at low EPV. Katzman et al. (2018) found 1 layer optimal for 3/4
    # datasets; auto_deepsurv_defaults() uses dropout=0.6 at low EPV.
    'deepsurv': {
        'n_layers':  ((1, 1),  (1, 2)),
        'n_nodes':   ((8, 24), (8, 36)),
        'dropout':   ((0.3, 0.7), (0.2, 0.7)),
    },
    'deephit': {
        'n_layers':  ((1, 1),   (1, 2)),
        'n_nodes':   ((16, 48), (16, 128)),
        'dropout':   ((0.2, 0.5), (0.1, 0.5)),
    },
    # Penalized linear models: force more shrinkage at low EPV (opposite
    # direction from the capacity-capping models above).
    'coxnet': {
        # l1_ratio: 0=ridge, 1=lasso. Cap the upper end at low EPV to favor the
        # L2 component (grouping effect / stability in p>n; Zou & Hastie 2005).
        'l1_ratio':    ((0.0, 0.5), (0.0, 0.7)),
        # alpha_index: 0=strongest penalty, higher=weaker. Cap the max index at
        # low EPV to stay on the penalized half of the path (Pavlou et al. 2016;
        # Ambler et al. 2012), away from the ~unpenalized-ML end.
        'alpha_index': ((0, 49),    (0, 74)),
    },
    'fssvm': {
        # alpha weights the loss term in sksurv's FastSurvivalSVM, so a LARGER
        # alpha means LESS regularization. Cap the upper end at low EPV to keep
        # the model in the more-regularized region (Ambler et al. 2012; Pavlou
        # et al. 2016; grid from Pölsterl et al. 2015). 'log' preserves log-scale
        # sampling. rank_ratio is intentionally not constrained — it trades
        # ranking vs regression loss, not regularization.
        'alpha': ((1e-5, 1.0, 'log'), (1e-5, 10.0, 'log')),
    },
}


def _epv_tier(epv, n_events):
    """Pick the constraint tier from BOTH the EPV ratio and absolute events.

    The events-per-variable ratio alone is an insufficient sample-size
    criterion: a high EPV reached with a tiny feature count and few absolute
    events is still unstable (Riley et al. 2019; van Smeden et al. 2019). A
    permissive tier therefore requires both a sufficient ratio AND enough
    absolute events; otherwise the tier is demoted so constraints get more
    aggressive.

    Parameters
    ----------
    epv : float
        Events per variable (n_events / n_features).
    n_events : int
        Absolute number of events (uncensored observations).

    Returns
    -------
    str
        One of ``'low'`` (high overfitting risk, aggressive constraints),
        ``'mid'`` (moderate risk), or ``'high'`` (low risk, defaults kept).
    """
    if epv >= 20 and n_events >= _TIER_MIN_EVENTS_HIGH:
        return 'high'
    if epv >= 10 and n_events >= _TIER_MIN_EVENTS_MID:
        return 'mid'
    return 'low'


def _constrain_space_for_epv(model_name, space, epv, n_events, n_samples):
    """Narrow search space based on events-per-variable and absolute events.

    Applies to capacity-capping models (RSF, GBSA, XGBCox, DeepSurv, DeepHit)
    and shrinkage-forcing penalized models (CoxNet, FSSVM). Only modifies
    parameters already present in ``space`` so that custom user spaces are
    respected — if a user omits ``bootstrap``, this function will not add it.

    The tier is chosen by :func:`_epv_tier` from both the EPV ratio and the
    absolute event count (building on Peduzzi et al. 1996's EPV rule; Riley et
    al. 2019 and van Smeden et al. 2019 for why the ratio alone is insufficient):
        - ``'low'``:  high risk of overfitting, aggressive constraints
        - ``'mid'``:  moderate risk, moderate constraints
        - ``'high'``: low risk, defaults preserved

    Parameters
    ----------
    model_name : str
        Model identifier.
    space : dict
        Base search space. Copied before modification for supported
        models; returned as-is for models without constraints.
    epv : float
        Events per variable (n_events / n_features).
    n_events : int
        Absolute number of events (uncensored observations).
    n_samples : int
        Total number of training samples.

    Returns
    -------
    dict
        Narrowed copy of the search space. Models without entries in
        ``_EPV_CONSTRAINTS`` are returned unchanged.
    """
    constraints = _EPV_CONSTRAINTS.get(model_name)
    if constraints is None:
        return space

    space = space.copy()
    tier = _epv_tier(epv, n_events)

    # Apply table-driven constraints (shared pattern across all models)
    for param, (low_range, mid_range) in constraints.items():
        if param not in space:
            continue
        if tier == 'low':
            space[param] = low_range
        elif tier == 'mid':
            space[param] = mid_range

    # Special cases that depend on additional conditions beyond the tier

    # Bootstrap is foundational to RF variance reduction (Breiman 2001).
    # Disabling it on small data destroys the ensemble benefit, so only
    # allow it to be toggled when both data and events are ample.
    if model_name == 'rsf' and 'bootstrap' in space:
        if tier == 'high' and n_samples > 2000:
            space['bootstrap'] = [True, False]
        else:
            space['bootstrap'] = [True]

    # Nonzero dropout floor at low EPV forces regularization,
    # reducing overfitting risk on small datasets.
    if model_name == 'gbsa' and 'dropout_rate' in space and tier == 'low':
        space['dropout_rate'] = (0.05, 0.3)

    return space


def get_search_space(model_name, custom_space=None, *,
                     n_samples=None, n_events=None, n_features=None):
    """
    Get search space for a model, optionally merged with custom overrides.

    When dataset characteristics (``n_samples``, ``n_events``,
    ``n_features``) are provided, the search space is automatically
    narrowed to prevent overfitting-prone configurations on small
    datasets. Capacity-capping models (RSF, GBSA, XGBCox, DeepSurv,
    DeepHit) have their capacity reduced, while penalized models
    (CoxNet, FSSVM) are steered toward stronger shrinkage. The tier is
    chosen from both the events-per-variable (EPV) ratio and the absolute
    event count, since the ratio alone is an insufficient sample-size
    criterion (Peduzzi et al. 1996; Riley et al. 2019; van Smeden et al.
    2019).

    Parameters
    ----------
    model_name : str
        Model identifier.
    custom_space : dict, optional
        Custom parameter ranges to merge with (and override) defaults.
        Applied **after** EPV constraints, so user overrides always win.
    n_samples : int, optional
        Total number of training samples.
    n_events : int, optional
        Number of events (uncensored observations).
    n_features : int, optional
        Number of features in the dataset.

    Returns
    -------
    dict
        Search space dictionary.

    Raises
    ------
    ValueError
        If `model_name` has no default search space, or if any provided
        dataset dimension is non-positive.
    """
    if model_name not in DEFAULT_SPACES:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(DEFAULT_SPACES.keys())}"
        )

    base_space = DEFAULT_SPACES[model_name].copy()

    # Apply EPV-aware constraints when dataset info is available
    dataset_params = (n_samples, n_events, n_features)
    n_provided = sum(p is not None for p in dataset_params)
    if 0 < n_provided < 3:
        warnings.warn(
            f"EPV constraints require all of n_samples, n_events, n_features "
            f"but only {n_provided}/3 were provided. "
            f"Returning unconstrained search space.",
            SuggestionWarning,
        )
    elif n_provided == 3:
        if n_features <= 0 or n_events <= 0 or n_samples <= 0:
            raise ValueError(
                f"Dataset dimensions must be positive: got n_samples={n_samples}, "
                f"n_events={n_events}, n_features={n_features}"
            )
        epv = n_events / n_features
        base_space = _constrain_space_for_epv(
            model_name, base_space, epv, n_events, n_samples,
        )
        if model_name in _EPV_CONSTRAINTS:
            logger.info(
                "Search space constrained for EPV=%.1f (n=%d, events=%d)",
                epv, n_samples, n_events,
            )

    # User overrides always win
    if custom_space is not None:
        base_space.update(custom_space)

    return base_space
