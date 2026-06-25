"""Preprocessing pipelines for mlsurv."""

import warnings
from sklearn.pipeline import Pipeline

from mlsurv._warnings import SuggestionWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA


# ============================================================================
# Component Presets
# ============================================================================

# Imputer presets - lazy factory so the user's random_state can be threaded in
def _get_imputer_presets(random_state=None):
    """Get imputer presets (factory functions returning fresh instances).

    ``random_state`` is threaded only into the ``iterative`` imputer, whose
    chained-equations draws depend on it. ``median``/``mean``/``knn`` are
    deterministic and take no seed.
    """
    return {
        'median': lambda: SimpleImputer(strategy='median'),
        'mean': lambda: SimpleImputer(strategy='mean'),
        'knn': lambda: KNNImputer(n_neighbors=5),
        'iterative': lambda: IterativeImputer(max_iter=10, random_state=random_state),
    }


# Scaler presets - lazy factory so the user's random_state can be threaded in
def _get_scaler_presets(random_state=None):
    """Get scaler presets (factory functions returning fresh instances).

    ``random_state`` is threaded only into the ``pca`` preset's PCA step, whose
    randomized SVD depends on it. ``standard``/``robust``/``minmax`` (and the
    ``pca`` preset's StandardScaler) are deterministic and take no seed.
    """
    return {
        'standard': lambda: StandardScaler(),
        'robust': lambda: RobustScaler(),
        'minmax': lambda: MinMaxScaler(),
        'pca': lambda: Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95, random_state=random_state))
        ]),
    }

# Selector presets - lazy import to avoid circular imports
def _get_selector_presets(random_state=None):
    """Get selector presets (lazy import to avoid circular imports).

    ``random_state`` is threaded only into the penalized-Cox selectors
    (``coxlasso``/``coxnet``), which run internal K-fold CV for alpha
    selection. The other selectors are deterministic and take no seed.
    """
    from .selection import (
        TopKCoxSelector, VIFSelector, CorrelationSelector,
        CoxLassoSelector, CoxNetSelector,
    )
    return {
        'topk': lambda: TopKCoxSelector(k='auto'),
        'vif': lambda: VIFSelector(threshold=10.0),
        'correlation': lambda: CorrelationSelector(threshold=0.8),
        'coxlasso': lambda: CoxLassoSelector(random_state=random_state),
        'coxnet': lambda: CoxNetSelector(random_state=random_state),
    }


# ============================================================================
# Component Resolvers
# ============================================================================

def _resolve_component(value, presets, prefix, component_name):
    """
    Resolve transformer from string, list, or transformer.

    Parameters
    ----------
    value : str, list, transformer, or None
        Component specification:
        - None: Return None (skip component)
        - str: Look up in presets
        - list: Wrap in Pipeline
        - transformer: Return as-is
    presets : dict
        Preset name to factory function mapping.
    prefix : str
        Prefix for Pipeline step names (e.g., 'imp', 'scl', 'sel').
    component_name : str
        Human-readable name for error messages.

    Returns
    -------
    transformer or None
        Resolved transformer instance.

    Raises
    ------
    ValueError
        If string value is not a recognized preset.
    """
    if value is None:
        return None

    if isinstance(value, str):
        if value not in presets:
            raise ValueError(
                f"Unknown {component_name} preset: '{value}'. "
                f"Options: {list(presets.keys())}"
            )
        return presets[value]()  # Call factory

    if isinstance(value, list):
        # Wrap list in Pipeline
        steps = []
        for i, item in enumerate(value):
            # Recursively resolve if string
            if isinstance(item, str):
                if item not in presets:
                    raise ValueError(
                        f"Unknown {component_name} preset: '{item}'. "
                        f"Options: {list(presets.keys())}"
                    )
                item = presets[item]()
            steps.append((f'{prefix}_{i}', item))
        return Pipeline(steps)

    # Assume it's already a transformer
    return value


def get_imputer(value, random_state=None):
    """
    Resolve imputer from string, list, or transformer.

    Parameters
    ----------
    value : str, list, transformer, or None
        Imputer specification:
        - 'median': SimpleImputer(strategy='median')
        - 'mean': SimpleImputer(strategy='mean')
        - 'knn': KNNImputer(n_neighbors=5)
        - 'iterative': IterativeImputer(max_iter=10) — chained-equations imputation
        - list: Wrapped in Pipeline
        - transformer: Used as-is
        - None: No imputation
    random_state : int or None, default=None
        Seed for the ``iterative`` string preset's chained-equations draws
        (including those nested in a list). Ignored for the deterministic
        presets and for a user-supplied transformer instance, which is
        returned as-is with its own configuration intact.

    Returns
    -------
    transformer or None
        Resolved imputer instance.

    Examples
    --------
    >>> get_imputer('median')
    SimpleImputer(strategy='median')
    >>> get_imputer([SimpleImputer(), SomePostProcessor()])
    Pipeline(steps=[('imp_0', SimpleImputer()), ('imp_1', SomePostProcessor())])
    """
    presets = _get_imputer_presets(random_state)
    return _resolve_component(value, presets, 'imp', 'imputer')


def get_scaler(value, random_state=None):
    """
    Resolve scaler from string, list, or transformer.

    Parameters
    ----------
    value : str, list, transformer, or None
        Scaler specification:
        - 'standard': StandardScaler()
        - 'robust': RobustScaler()
        - 'minmax': MinMaxScaler()
        - 'pca': StandardScaler + PCA(n_components=0.95)
        - list: Wrapped in Pipeline
        - transformer: Used as-is
        - None: No scaling
    random_state : int or None, default=None
        Seed for the ``pca`` string preset's randomized SVD (including those
        nested in a list). Ignored for the deterministic presets and for a
        user-supplied transformer instance, which is returned as-is with its
        own configuration intact.

    Returns
    -------
    transformer or None
        Resolved scaler instance.

    Examples
    --------
    >>> get_scaler('robust')
    RobustScaler()
    >>> get_scaler('pca')
    Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA(n_components=0.95))])
    """
    presets = _get_scaler_presets(random_state)
    return _resolve_component(value, presets, 'scl', 'scaler')


def get_selector(value, random_state=None):
    """
    Resolve feature selector from string, list, or transformer.

    Parameters
    ----------
    value : str, list, transformer, or None
        Selector specification:
        - 'topk': TopKCoxSelector(k='auto')
        - 'vif': VIFSelector(threshold=10.0)
        - 'correlation': CorrelationSelector(threshold=0.8)
        - 'coxlasso': CoxLassoSelector()
        - 'coxnet': CoxNetSelector()
        - list: Wrapped in Pipeline
        - transformer: Used as-is
        - None: No feature selection
    random_state : int or None, default=None
        Seed for the internal alpha-selection CV in the ``coxlasso`` and
        ``coxnet`` string presets (including those nested in a list). Ignored
        for the deterministic selectors and for a user-supplied transformer
        instance, which is returned as-is with its own configuration intact.

    Returns
    -------
    transformer or None
        Resolved selector instance.

    Examples
    --------
    >>> get_selector('vif')
    VIFSelector()
    >>> get_selector(['vif', TopKCoxSelector(k='auto')])
    Pipeline(steps=[('sel_0', VIFSelector()), ('sel_1', TopKCoxSelector())])
    """
    presets = _get_selector_presets(random_state)
    return _resolve_component(value, presets, 'sel', 'selector')


def warn_preprocessing_order(imputer, scaler, selector):
    """
    Warn if preprocessing order seems wrong.

    Standard order is: imputer → scaler → selector.
    Warns if components appear to be in unexpected positions.

    Parameters
    ----------
    imputer : transformer or None
        Imputer component.
    scaler : transformer or None
        Scaler component.
    selector : transformer or None
        Selector component.
    """
    # Check if scaler contains imputation
    if scaler is not None and imputer is None:
        scaler_type = type(scaler).__name__
        if 'Imputer' in scaler_type:
            warnings.warn(
                f"Scaler '{scaler_type}' appears to be an imputer. "
                "Consider using the imputer parameter instead.",
                SuggestionWarning,
            )

    # Check if imputer contains scaling
    if imputer is not None:
        imputer_type = type(imputer).__name__
        if 'Scaler' in imputer_type or 'Standard' in imputer_type:
            warnings.warn(
                f"Imputer '{imputer_type}' appears to be a scaler. "
                "Consider using the scaler parameter instead.",
                SuggestionWarning,
            )
