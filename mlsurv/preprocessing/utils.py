"""Shared preprocessing utilities."""

from collections import defaultdict

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Single source of truth for defaults
DEFAULT_IMPUTER_STRATEGY = 'median'


def get_default_imputer():
    """Return default imputer: SimpleImputer(strategy='median').

    Returns
    -------
    sklearn.impute.SimpleImputer
        Fresh imputer instance with median strategy.
    """
    return SimpleImputer(strategy=DEFAULT_IMPUTER_STRATEGY)


def get_default_scaler():
    """Return default scaler: StandardScaler().

    Returns
    -------
    sklearn.preprocessing.StandardScaler
        Fresh scaler instance.
    """
    return StandardScaler()


def extract_component_params(params, prefix):
    """Extract prefixed parameters for a component.

    Parameters
    ----------
    params : dict
        Dictionary of parameters with prefixed names.
    prefix : str
        Prefix to extract (e.g., 'model', 'imputer', 'scaler').

    Returns
    -------
    dict
        Parameters with prefix removed.

    Examples
    --------
    >>> extract_component_params({'model__lr': 0.1, 'model__dropout': 0.5}, 'model')
    {'lr': 0.1, 'dropout': 0.5}
    >>> extract_component_params({'model__model__n_estimators': 100}, 'model')
    {'model__n_estimators': 100}
    >>> extract_component_params({'imputer__strategy': 'mean'}, 'model')
    {}
    """
    prefix_len = len(f'{prefix}__')
    return {k[prefix_len:]: v
            for k, v in params.items()
            if k.startswith(f'{prefix}__')}


def get_or_create_selector(selector, k=None):
    """Get existing selector or create TopKCoxSelector if k specified.

    Parameters
    ----------
    selector : sklearn Transformer or None
        Existing selector instance, or None.
    k : int, optional
        Number of top features to select. If provided and selector is not None,
        updates the selector's k parameter. If selector is None and k is provided,
        creates a new TopKCoxSelector.

    Returns
    -------
    sklearn Transformer or None
        Configured selector, or None if both selector and k are None.

    Examples
    --------
    >>> get_or_create_selector(None, k=10)  # Creates TopKCoxSelector(k=10)
    >>> get_or_create_selector(existing_selector, k=15)  # Clones and sets k=15
    >>> get_or_create_selector(None, k=None)  # Returns None
    """
    if selector is not None:
        if k is not None:
            from sklearn.base import clone
            selector = clone(selector)
            selector.set_params(k=k)
        return selector
    elif k is not None:
        from .selection import TopKCoxSelector
        return TopKCoxSelector(k=k)
    return None


# ---------------------------------------------------------------------------
# Categorical group detection
# ---------------------------------------------------------------------------


def _find_binary_columns(X):
    """Return list of column names whose non-NaN values are all in {0, 1}."""
    binary = []
    for col in X.columns:
        vals = X[col].dropna()
        if len(vals) == 0:
            continue
        unique = vals.unique()
        if np.isin(unique, [0, 1]).all():
            binary.append(col)
    return binary


def _is_mutually_exclusive(X, cols):
    """Check that at most one column equals 1 per row (ignoring NaN)."""
    row_sums = X[cols].fillna(0).sum(axis=1)
    return bool((row_sums <= 1).all())


def _build_candidate_groups(binary_cols, sep):
    """Build prefix and suffix candidate groups from binary column names.

    Returns (prefix_groups, suffix_groups) where each is a dict mapping
    the candidate group name to a list of column names.
    """
    prefix_groups = defaultdict(list)
    suffix_groups = defaultdict(list)

    for col in binary_cols:
        if sep not in col:
            continue
        # Prefix: everything before the last separator
        prefix = col.rsplit(sep, 1)[0]
        prefix_groups[prefix].append(col)
        # Suffix: everything after the first separator
        suffix = col.split(sep, 1)[1]
        suffix_groups[suffix].append(col)

    # Require at least 2 members
    prefix_groups = {k: v for k, v in prefix_groups.items() if len(v) >= 2}
    suffix_groups = {k: v for k, v in suffix_groups.items() if len(v) >= 2}
    return prefix_groups, suffix_groups


def detect_categorical_groups(X, sep="_"):
    """Auto-detect one-hot encoded categorical groups from a DataFrame.

    Identifies groups of binary (0/1) columns that share a common prefix
    or suffix (split on *sep*) and are mutually exclusive (at most one
    column is 1 per row), indicating they were produced by one-hot
    encoding.

    Parameters
    ----------
    X : DataFrame
        Encoded feature matrix to analyze.
    sep : str, default="_"
        Separator between original feature name and category value
        (e.g., ``"chapter_Circulatory"`` uses ``"_"``).

    Returns
    -------
    dict of {str: list of str}
        Mapping from detected group name (the shared prefix or suffix)
        to the list of encoded column names belonging to that group.
        Empty dict if no one-hot groups are detected.

    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.get_dummies(
    ...     pd.DataFrame({"chapter": ["A", "B", "C"], "age": [30, 40, 50]}),
    ...     columns=["chapter"],
    ... )
    >>> detect_categorical_groups(X)
    {'chapter': ['chapter_A', 'chapter_B', 'chapter_C']}
    """
    binary_cols = _find_binary_columns(X)
    if len(binary_cols) < 2:
        return {}

    non_binary = set(X.columns) - set(binary_cols)
    prefix_groups, suffix_groups = _build_candidate_groups(binary_cols, sep)

    # Exclude groups whose name collides with a non-binary column
    # (e.g., column "chapter" still present alongside "chapter_A")
    prefix_groups = {k: v for k, v in prefix_groups.items() if k not in non_binary}
    suffix_groups = {k: v for k, v in suffix_groups.items() if k not in non_binary}

    # Validate mutual exclusivity and collect candidates with scores.
    # Score = (group_size, is_prefix) — larger groups win; prefix breaks ties.
    candidates = []
    for name, cols in prefix_groups.items():
        if _is_mutually_exclusive(X, cols):
            candidates.append((len(cols), True, name, cols))
    for name, cols in suffix_groups.items():
        if _is_mutually_exclusive(X, cols):
            candidates.append((len(cols), False, name, cols))

    # Greedy assignment: largest groups first, prefix preferred
    candidates.sort(key=lambda x: (-x[0], not x[1]))

    assigned = set()
    result = {}
    for _size, _is_prefix, name, cols in candidates:
        available = [c for c in cols if c not in assigned]
        if len(available) < 2:
            continue
        if not _is_mutually_exclusive(X, available):
            continue
        if name in result:
            continue
        result[name] = sorted(available)
        assigned.update(available)

    return result
