"""Data-driven constructor defaults for models trained without tuning.

Each ``auto_<model>_defaults(n_samples, n_features, n_events)`` function
computes sensible hyperparameter defaults from the data shape. They are
looked up via ``_AUTO_DEFAULTS_REGISTRY`` in ``mlsurv/models/__init__.py``
and applied by ``_apply_dynamic_defaults`` when the caller provides data
characteristics.

Invariant: every value produced here must stay within the corresponding
``DEFAULT_SPACES[model][param]`` range in ``mlsurv/tuning/search_spaces.py``
(see ``CONTRIBUTING.md``).

References
----------
Random Forest:
    Probst, P., Wright, M. N., & Boulesteix, A.-L. (2019). Hyperparameters
    and tuning strategies for random forest. WIREs Data Mining and Knowledge
    Discovery, 9(3), e1301. https://doi.org/10.1002/widm.1301

    Boehmke, B., & Greenwell, B. M. (2020). Hands-On Machine Learning with R.
    Chapter 11: Random Forests. https://bradleyboehmke.github.io/HOML/

Gradient Boosting:
    Friedman, J. H. (2001). Greedy function approximation: A gradient boosting
    machine. Annals of Statistics, 29(5), 1189-1232.

    Natekin, A., & Knoll, A. (2013). Gradient boosting machines, a tutorial.
    Frontiers in Neurorobotics, 7, 21. https://doi.org/10.3389/fnbot.2013.00021

Random Survival Forest:
    Dauda, K. A., et al. (2022). Optimal tuning of random survival forest
    hyperparameter with an application to liver disease. Malaysian Journal of
    Medical Sciences, 29(6), 67-76. https://doi.org/10.21315/mjms2022.29.6.7

DeepSurv:
    Katzman, J. L., et al. (2018). DeepSurv: Personalized treatment recommender
    system using a Cox proportional hazards deep neural network. BMC Medical
    Research Methodology, 18(1), 24.
"""

import warnings

import numpy as np

from mlsurv._warnings import MethodologicalWarning


def _threshold_lookup(value, thresholds, default):
    """Return the mapped value for the first threshold that value is below, else default.

    Parameters
    ----------
    value : numeric
        The value to compare against thresholds.
    thresholds : list of (threshold, mapped_value)
        Sorted ascending list of (cutoff, result) pairs.
    default : any
        Value returned if no threshold matches.
    """
    for cutoff, result in thresholds:
        if value < cutoff:
            return result
    return default


def auto_deepsurv_defaults(n_samples, n_features, n_events):
    """
    Compute dynamic DeepSurv defaults based on dataset characteristics.

    Used when training DeepSurv WITHOUT tuning to provide sensible
    data-adaptive parameters out of the box.

    Based on Katzman et al. (2018) optimal hyperparameters (Table 3):
    - GBSG (686 samples, 7 features): [8], dropout=0.66, lr=0.154
    - WHAS (1638 samples, 5 features): [48,48], dropout=0.15, lr=0.067
    - METABRIC (1980 samples, 9 features): [41], dropout=0.16, lr=0.01
    - SUPPORT (8873 samples, 14 features): [44], dropout=0.26, lr=0.047

    Parameters
    ----------
    n_samples : int
        Number of training samples.
    n_features : int
        Number of features after preprocessing.
    n_events : int
        Number of events (uncensored observations).

    Returns
    -------
    dict
        Dictionary of DeepSurv hyperparameters.

    Examples
    --------
    >>> defaults = auto_deepsurv_defaults(686, 7, 299)
    >>> defaults['n_nodes']  # Small dataset -> few nodes
    8
    >>> defaults['dropout']  # Low EPV -> high dropout
    0.6
    """
    # Compute events per variable (EPV)
    epv = n_events / max(n_features, 1)

    # Warn if EPV is very low
    if epv < 10:
        warnings.warn(
            f"Events per variable (EPV) = {epv:.1f} is below recommended "
            f"minimum of 10. Consider reducing features or collecting more data. "
            f"Using high regularization defaults.",
            MethodologicalWarning,
        )

    # Architecture: Paper found 1 layer optimal for 3/4 datasets
    n_layers = 1

    # Nodes: Scale sub-linearly with sqrt(n_samples), cap based on features
    # GBSG(686) -> 8, SUPPORT(8873) -> ~28
    n_nodes = min(48, max(8, int(np.sqrt(n_samples) * 0.3)))
    n_nodes = max(8, min(n_nodes, n_features * 6))

    # Dropout: Strong inverse relationship with EPV (sample/event density)
    # Paper: GBSG (low EPV) had 0.66, METABRIC (higher EPV) had 0.16
    dropout = _threshold_lookup(epv, [(10, 0.6), (20, 0.4), (50, 0.25)], 0.15)

    # Learning rate: Higher for smaller datasets
    # Paper: GBSG=0.154, WHAS=0.067, METABRIC=0.01, SUPPORT=0.047
    lr = _threshold_lookup(n_samples, [(1000, 0.1), (3000, 0.05), (10000, 0.02)], 0.01)

    # Batch size: Scale with data, match search space minimum (64)
    # Round to nearest power of 2 for GPU efficiency
    raw_batch = min(256, max(64, n_samples // 8))
    batch_size = min(256, max(64, 2 ** round(np.log2(raw_batch))))

    # Epochs: More passes for small data
    epochs = _threshold_lookup(n_samples, [(500, 300), (2000, 200), (10000, 100)], 50)

    return {
        'layers': [n_nodes] * n_layers,
        'dropout': dropout,
        'learning_rate': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'activation': 'selu',      # SELU optimal for 3/4 datasets in paper
        'weight_decay': 0.01,
    }


def auto_rsf_defaults(n_samples, n_features, n_events):
    """
    Compute conservative RSF defaults based on dataset characteristics.

    Used when training RSF WITHOUT tuning to provide sensible data-adaptive
    parameters out of the box.

    Based on recommendations from:
    - Probst et al. (2019): n_estimators (>100 for stability), mtry tuning
    - Boehmke & Greenwell (2020): p×10 trees rule, 1-10 min_samples_leaf range
    - Dauda et al. (2022): mtry and nodesize as key RSF hyperparameters

    Parameters
    ----------
    n_samples : int
        Number of training samples.
    n_features : int
        Number of features after preprocessing.
    n_events : int
        Number of events (uncensored observations).

    Returns
    -------
    dict
        Dictionary of RSF hyperparameters.

    Examples
    --------
    >>> defaults = auto_rsf_defaults(686, 7, 299)
    >>> defaults['n_estimators']  # 7*10 = 70, but min is 100
    100
    >>> defaults['max_depth']  # EPV=42.7 -> unlimited
    None
    """
    epv = n_events / max(n_features, 1)

    # n_estimators: p×10 rule (Boehmke), min 100 (Probst), capped at 300
    n_estimators = min(n_features * 10, 300)
    n_estimators = max(n_estimators, 100)

    # max_depth: limit for low EPV to prevent overfitting
    # Boehmke: "larger n or p training data sets are more tolerable to deeper trees"
    if epv < 10:
        max_depth = 5
    elif epv < 20:
        max_depth = 10
    else:
        max_depth = None

    # min_samples_leaf: 1% of samples, min 3 (Boehmke: 1-10 range)
    min_samples_leaf = max(3, int(n_samples * 0.01))
    min_samples_split = min_samples_leaf * 2

    return {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
    }


def auto_gbsa_defaults(n_samples, n_features, n_events):
    """
    Compute conservative GBSA defaults based on dataset characteristics.

    Used when training GBSA WITHOUT tuning to provide sensible data-adaptive
    parameters out of the box.

    Based on recommendations from:
    - Friedman (2001): shrinkage=0.1 preferred, 100-500 trees, 2-8 terminal nodes
      "Smaller values of v give rise to larger optimal M-values"
    - Boehmke & Greenwell (2020): learning_rate 0.05-0.2, max_depth 3-8,
      "larger n or p training data sets are more tolerable to deeper trees"
    - Natekin & Knoll (2013): v=0.1 standard, small LR yields "dramatic
      improvements in generalization"

    Parameters
    ----------
    n_samples : int
        Number of training samples.
    n_features : int
        Number of features after preprocessing.
    n_events : int
        Number of events (uncensored observations).

    Returns
    -------
    dict
        Dictionary of GBSA hyperparameters.

    Examples
    --------
    >>> defaults = auto_gbsa_defaults(686, 7, 299)
    >>> defaults['learning_rate']  # n<1000 -> 0.1
    0.1
    >>> defaults['n_estimators']  # 100/0.1 = 1000, capped at 300
    300
    """
    epv = n_events / max(n_features, 1)

    # learning_rate: smaller for larger datasets (Friedman: 0.1 baseline)
    if n_samples < 1000:
        learning_rate = 0.1
    elif n_samples < 5000:
        learning_rate = 0.05
    else:
        learning_rate = 0.01

    # n_estimators: inversely related to learning rate (Friedman: 100-500)
    n_estimators = min(int(100 / learning_rate), 300)

    # max_depth: shallow trees for boosting (Friedman: 2-8 terminals)
    # Lower EPV -> shallower to prevent overfitting
    if epv < 10:
        max_depth = 2
    elif epv < 20:
        max_depth = 3
    else:
        max_depth = 4

    # min_samples_leaf: 0.5% of samples, min 1 (Boehmke: 5-15 typical)
    min_samples_leaf = max(1, int(n_samples * 0.005))
    min_samples_split = max(2, min_samples_leaf * 2)

    return {
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_leaf': min_samples_leaf,
        'min_samples_split': min_samples_split,
    }


def auto_deephit_defaults(n_samples, n_features, n_events):
    """
    Compute DeepHit defaults based on dataset characteristics.

    Similar to DeepSurv but with discrete-time specific parameters.

    Parameters
    ----------
    n_samples : int
        Number of training samples.
    n_features : int
        Number of features after preprocessing.
    n_events : int
        Number of events (uncensored observations).

    Returns
    -------
    dict
        Dictionary of DeepHit hyperparameters.
    """
    epv = n_events / max(n_features, 1)

    if n_samples < 500:
        layers = [32]
        dropout = 0.3
        learning_rate = 0.01
        batch_size = 64
        num_durations = 50
    elif n_samples < 2000:
        layers = [64, 64]
        dropout = 0.2
        learning_rate = 0.005
        batch_size = 128
        num_durations = 100
    else:
        layers = [128, 128]
        dropout = 0.1
        learning_rate = 0.005
        batch_size = 256
        num_durations = 200

    # EPV-based dropout floor: force stronger regularization on small data
    if epv < 10:
        dropout = max(dropout, 0.3)

    return {
        'layers': layers,
        'dropout': dropout,
        'activation': 'selu',      # SELU optimal for 3/4 datasets (Katzman 2018)
        'learning_rate': learning_rate,
        'weight_decay': 0.01,
        'batch_size': batch_size,
        'num_durations': num_durations,
        'alpha': 0.0,  # Likelihood-vs-ranking weight; 0.0 maximizes C^td
        'sigma': 0.1,  # Default smoothing
    }


def auto_weibull_aft_defaults(n_samples, n_features, n_events):
    """
    Compute Weibull AFT defaults based on dataset characteristics.

    Parameters
    ----------
    n_samples : int
        Number of training samples.
    n_features : int
        Number of features after preprocessing.
    n_events : int
        Number of events (uncensored observations).

    Returns
    -------
    dict
        Dictionary of Weibull AFT hyperparameters.
    """
    epv = n_events / max(n_features, 1)

    # alpha: regularization based on EPV (lifelines constraint: 0-1)
    if epv < 10:
        alpha = 0.1  # More regularization for low EPV
    elif epv < 20:
        alpha = 0.01
    else:
        alpha = 0.0  # No regularization for high EPV

    return {
        'alpha': alpha,
    }
