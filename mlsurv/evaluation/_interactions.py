"""Pairwise feature interaction detection (H-statistic and SHAP)."""

import contextlib
import itertools
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.pipeline import Pipeline

from .._warnings import MethodologicalWarning, SuggestionWarning
from .._xgboost import import_xgboost
from ..logging_utils import get_logger

logger = get_logger("interactions")

# A single scalar column with at most this many unique values is treated as
# categorical (discrete) for PDP rendering, unless an explicit
# ``categorical_features`` list is supplied. See ``_describe_unit_grid``.
_CAT_MAX_CARDINALITY = 5

# ``scale='auto'`` switches a numeric PDP axis to a log spacing only when the
# feature's right-skew (Fisher's moment coefficient) exceeds this threshold.
# 1.0 is a conventional "moderate-to-high positive skew" cutoff. See
# ``_resolve_axis_scale``.
_AUTO_SKEW_THRESHOLD = 1.0


def _fmt_level(value: float) -> str:
    """Format one categorical-scalar grid level as a tick label.

    Integer-valued floats render without a decimal point (``0.0`` -> ``"0"``),
    so a 0/1 column reads as ``"0"``/``"1"``; other values use compact ``g``
    formatting.
    """
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


# ── Feature unit abstraction ────────────────────────────────────────────


class FeatureUnit(NamedTuple):
    """A single numeric column or a grouped categorical feature."""

    name: str              # Display name ("age" or "stage")
    col_indices: List[int]  # [3] for numeric, [5, 6, 7] for group
    is_group: bool         # True for categorical group


def _build_group_unit(
    feat: str, expected: List[str], col_to_idx: Dict[str, int],
) -> FeatureUnit:
    """Build a categorical-group FeatureUnit, warning on missing members."""
    idxs = [col_to_idx[c] for c in expected if c in col_to_idx]
    if not idxs:
        raise ValueError(
            f"Categorical group '{feat}': none of its member columns "
            f"{expected} found in data columns."
        )
    missing = [c for c in expected if c not in col_to_idx]
    if missing:
        warnings.warn(
            f"Categorical group '{feat}': columns {missing} not found "
            f"in data. Only {len(idxs)}/{len(expected)} members "
            f"will be used for interaction computation.",
            SuggestionWarning,
            stacklevel=3,
        )
    return FeatureUnit(name=feat, col_indices=idxs, is_group=True)


def _build_scalar_unit(feat: str, col_to_idx: Dict[str, int]) -> FeatureUnit:
    """Build a single-column FeatureUnit, raising if the column is missing."""
    if feat not in col_to_idx:
        raise ValueError(
            f"Feature '{feat}' not found in columns or categorical_groups."
        )
    return FeatureUnit(name=feat, col_indices=[col_to_idx[feat]], is_group=False)


def _build_interaction_units(
    columns: List[str],
    features: List[str],
    categorical_groups: Optional[Dict[str, List[str]]],
) -> List[FeatureUnit]:
    """Convert selected feature names to FeatureUnit objects."""
    col_to_idx = {c: i for i, c in enumerate(columns)}
    groups = categorical_groups or {}
    return [
        _build_group_unit(feat, groups[feat], col_to_idx) if feat in groups
        else _build_scalar_unit(feat, col_to_idx)
        for feat in features
    ]


def _build_categorical_grid(
    X_values: np.ndarray, col_indices: List[int],
) -> np.ndarray:
    """Build grid of observed one-hot level patterns for a categorical group.

    Returns
    -------
    np.ndarray, shape (n_levels, n_dummies)
        Each row is one observed one-hot pattern.
    """
    patterns = X_values[:, col_indices]
    return np.unique(patterns, axis=0)


def _categorical_grid_labels(
    grid: np.ndarray, col_names: List[str],
) -> List[str]:
    """Generate human-readable labels for categorical grid rows.

    For each one-hot pattern, returns the name of the active dummy
    (the one with value 1.0). If no dummy is active (all zeros, e.g.,
    reference category with drop_first), returns "ref".
    If multiple are active, joins them with "+".
    """
    labels = []
    for row in grid:
        active = [col_names[i] for i, v in enumerate(row) if np.isclose(v, 1.0)]
        if not active:
            labels.append("ref")
        else:
            labels.append("+".join(active))
    return labels


def _build_unit_grid(
    X_values: np.ndarray,
    unit: FeatureUnit,
    grid_resolution: int,
    scale: str = 'linear',
) -> np.ndarray:
    """Build grid for a feature unit (single column or group).

    *scale* (``'linear'``, ``'log'``, or ``'log1p'``) only affects numeric
    units; group (categorical) units always use observed one-hot patterns.

    Returns
    -------
    For numeric: 1D array of shape (n_grid,).
    For group: 2D array of shape (n_levels, n_dummies).
    """
    if unit.is_group:
        return _build_categorical_grid(X_values, unit.col_indices)
    return _build_grid(X_values[:, unit.col_indices[0]], grid_resolution, scale)


# ── Unit-aware PDP functions ────────────────────────────────────────────


def _set_unit_value(X_mod, unit, value):
    """Set a feature unit's columns to a grid value (scalar or pattern)."""
    if unit.is_group:
        X_mod[:, unit.col_indices] = value
    else:
        X_mod[:, unit.col_indices[0]] = value


def _pdp_1d_unit(predict_fn, X_values, unit, grid):
    """Compute 1D partial dependence for a feature unit.

    Parameters
    ----------
    predict_fn : callable
    X_values : np.ndarray, shape (n_samples, n_features)
    unit : FeatureUnit
    grid : np.ndarray
        1D for numeric, 2D (n_levels, n_dummies) for group.

    Returns
    -------
    np.ndarray, shape (n_grid,)
    """
    n_grid = len(grid)
    pdp = np.empty(n_grid)
    for i in range(n_grid):
        X_mod = X_values.copy()
        _set_unit_value(X_mod, unit, grid[i])
        pdp[i] = predict_fn(X_mod).mean()
    return pdp


def _pdp_2d_unit(predict_fn, X_values, unit_1, unit_2, grid_1, grid_2):
    """Compute 2D partial dependence for two feature units.

    Returns
    -------
    np.ndarray, shape (len(grid_1), len(grid_2))
    """
    pdp = np.empty((len(grid_1), len(grid_2)))
    for i in range(len(grid_1)):
        X_mod = X_values.copy()
        _set_unit_value(X_mod, unit_1, grid_1[i])
        for j in range(len(grid_2)):
            _set_unit_value(X_mod, unit_2, grid_2[j])
            pdp[i, j] = predict_fn(X_mod).mean()
    return pdp


def _h_statistic_pair_units(predict_fn, X_values, unit_1, unit_2, grid_1, grid_2):
    """Compute Friedman's H-squared for a pair of feature units."""
    pdp_1 = _pdp_1d_unit(predict_fn, X_values, unit_1, grid_1)
    pdp_2 = _pdp_1d_unit(predict_fn, X_values, unit_2, grid_2)
    pdp_12 = _pdp_2d_unit(predict_fn, X_values, unit_1, unit_2, grid_1, grid_2)

    pdp_1_c = pdp_1 - pdp_1.mean()
    pdp_2_c = pdp_2 - pdp_2.mean()
    pdp_12_c = pdp_12 - pdp_12.mean()

    residual = pdp_12_c - pdp_1_c[:, np.newaxis] - pdp_2_c[np.newaxis, :]

    numerator = np.sum(residual ** 2)
    denominator = np.sum(pdp_12_c ** 2)

    if denominator < 1e-12:
        logger.debug(
            "H-statistic denominator near zero for units (%s, %s); "
            "returning 0.0 (constant 2D PDP).",
            unit_1.name, unit_2.name,
        )
        return 0.0

    h_val = float(np.clip(numerator / denominator, 0.0, 1.0))
    if not np.isfinite(h_val):
        logger.warning(
            "H-statistic is non-finite (%s) for units (%s, %s); "
            "model predictions may contain NaN/Inf.",
            h_val, unit_1.name, unit_2.name,
        )
        warnings.warn(
            f"H-statistic is non-finite for units ({unit_1.name}, {unit_2.name}); "
            "model predictions may contain NaN/Inf. "
            "Setting h_statistic=NaN for this pair.",
            SuggestionWarning,
            stacklevel=5,
        )
        return np.nan
    return h_val


# ── SHAP group aggregation ─────────────────────────────────────────────


def _aggregate_shap_interaction_groups(
    raw_matrix: np.ndarray,
    all_feature_names: List[str],
    units: List[FeatureUnit],
) -> Tuple[np.ndarray, List[str]]:
    """Aggregate SHAP interaction matrix by categorical groups.

    For each pair of units (i, j), the aggregated interaction value per
    sample is the sum of ``raw[s, a, b]`` for all ``a`` in unit_i's
    column indices and ``b`` in unit_j's column indices.

    Parameters
    ----------
    raw_matrix : np.ndarray, shape (n_samples, n_all_features, n_all_features)
    all_feature_names : list of str
        Column names corresponding to axes 1 and 2 of raw_matrix.
    units : list of FeatureUnit
        Feature units with col_indices referencing positions in
        all_feature_names.

    Returns
    -------
    aggregated : np.ndarray, shape (n_samples, n_units, n_units)
    unit_names : list of str
    """
    n_all = len(all_feature_names)
    n_samples = raw_matrix.shape[0]
    n_units = len(units)
    aggregated = np.zeros((n_samples, n_units, n_units))

    # Validate all col_indices are within bounds
    for unit in units:
        oob = [ci for ci in unit.col_indices if ci >= n_all]
        if oob:
            raise IndexError(
                f"FeatureUnit '{unit.name}' has col_indices {oob} out of "
                f"bounds for {n_all} features."
            )

    for ui, unit_i in enumerate(units):
        for uj, unit_j in enumerate(units):
            for a in unit_i.col_indices:
                for b in unit_j.col_indices:
                    aggregated[:, ui, uj] += raw_matrix[:, a, b]

    return aggregated, [u.name for u in units]


def _is_linear_model(model) -> bool:
    """Check if a model/pipeline has a linear predict() (H-stat always zero).

    Linear models (CoxPH, CoxNet, Weibull AFT, FSSVM) produce additive
    partial dependence by construction, so pairwise H-statistics are
    identically zero. Skipping them avoids wasted computation.
    """
    # Pipeline: check last step
    if hasattr(model, 'steps'):
        inner = model.steps[-1][1]
    else:
        inner = model
    return getattr(inner, 'complexity', None) == 'linear'


def _select_interaction_features(
    importance_df: pd.DataFrame,
    top_k: int,
    all_columns: pd.Index,
    categorical_groups: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """Select top-k features for interaction testing.

    Uses permutation importance ranking when available, falling back
    to the first top_k columns (group-aware when *categorical_groups*
    is provided).

    Parameters
    ----------
    importance_df : pd.DataFrame
        Importance DataFrame with 'feature' and 'perm_mean' columns.
        When categorical_groups are active, feature names here already
        include group names (from grouped permutation importance).
    top_k : int
        Number of top features to select.
    all_columns : pd.Index
        All feature columns (fallback when importance is empty).
    categorical_groups : dict or None
        Mapping from group name to list of dummy column names.

    Returns
    -------
    list of str
        Feature names to test (may include group names).
    """
    if importance_df is not None and not importance_df.empty and 'perm_mean' in importance_df.columns:
        # Rounding (1e-12) collapses BLAS/platform float jitter into a single
        # tied bucket; break ties by feature name so top-k selection is
        # deterministic across platforms. NaN
        # importances sort last via -inf, matching the prior na_position='last'.
        sort_key = importance_df['perm_mean'].round(12).fillna(-np.inf)
        ranked = (
            importance_df
            .assign(_sort_key=sort_key)
            .drop_duplicates(subset='feature')
            .sort_values(['_sort_key', 'feature'], ascending=[False, True])
            .head(top_k)['feature']
            .tolist()
        )
        if len(ranked) >= 2:
            return ranked

    # Fallback: first top_k feature units by position order
    groups = categorical_groups or {}
    if groups:
        from .importance import _build_permutation_units
        units = _build_permutation_units(list(all_columns), groups)
        unit_names = [name for name, _ in units]
        fallback = unit_names[:top_k]
    else:
        fallback = list(all_columns[:top_k])

    warnings.warn(
        "Permutation importance not available or insufficient; selecting "
        f"the first {top_k} features by column order for interaction "
        "testing. Run evaluate() with n_importance_repeats > 0 for "
        "importance-based selection.",
        SuggestionWarning,
        stacklevel=3,
    )
    return fallback


def _build_grid(
    values: np.ndarray, grid_resolution: int, scale: str = 'linear',
) -> np.ndarray:
    """Build grid from observed values, capped at *grid_resolution* points.

    With ``scale='linear'`` (the default) and more unique values than
    *grid_resolution*, evenly-spaced percentiles produce a representative grid.

    With ``scale='log'`` the grid is spaced evenly on a base-10 log axis
    (``10**linspace(log10(lo), log10(hi), grid_resolution)``), so points are
    dense where right-skewed data is dense and align with a Plotly
    ``type='log'`` (log10) axis.  ``lo`` is the smallest positive value (zeros
    are floored up to it, since ``0`` has no position on a log axis) and ``hi``
    is the max; NaNs are ignored.  The caller must ensure at least one positive
    value when requesting a log scale (enforced by :func:`_resolve_axis_scale`).

    With ``scale='log1p'`` the grid is spaced evenly in ``log1p`` space
    (``expm1(linspace(log1p(lo), log1p(hi), grid_resolution))``), which behaves
    like a log axis for large values but maps ``0`` to ``0`` cleanly, so it
    suits right-skewed features that include zeros.  ``lo`` is the smallest
    non-negative value (typically ``0``); the caller must ensure no negative
    values (enforced by :func:`_resolve_axis_scale`).

    Unlike the linear branch, the log and log1p branches always return exactly
    *grid_resolution* points and do not short-circuit on few unique values.
    """
    if scale == 'log':
        finite = values[np.isfinite(values)]
        positive = finite[finite > 0]
        lo, hi = float(positive.min()), float(finite.max())
        return 10.0 ** np.linspace(np.log10(lo), np.log10(hi), grid_resolution)
    if scale == 'log1p':
        finite = values[np.isfinite(values)]
        nonneg = finite[finite >= 0]
        lo, hi = float(nonneg.min()), float(finite.max())
        return np.expm1(np.linspace(np.log1p(lo), np.log1p(hi), grid_resolution))
    unique = np.unique(values)
    if len(unique) <= grid_resolution:
        return unique
    percentiles = np.linspace(0, 100, grid_resolution)
    return np.unique(np.percentile(values, percentiles))


def _validate_group_members(
    X: pd.DataFrame, group_name: str, members: List[str],
) -> None:
    """Verify a categorical group's member columns exist and are numeric."""
    missing = [c for c in members if c not in X.columns]
    if missing:
        raise ValueError(
            f"Categorical group '{group_name}' has columns not in data: {missing}"
        )
    non_num = [c for c in members if not pd.api.types.is_numeric_dtype(X[c])]
    if non_num:
        raise TypeError(
            f"Categorical group '{group_name}' has non-numeric columns: "
            f"{non_num}. One-hot dummies must be numeric."
        )


def _validate_scalar_feature(X: pd.DataFrame, name: str) -> None:
    """Verify a single feature column exists and is numeric."""
    if name not in X.columns:
        raise ValueError(f"Features not found in data: [{name!r}]")
    if not pd.api.types.is_numeric_dtype(X[name]):
        raise TypeError(
            f"Interaction testing requires numeric features, but "
            f"'{name}' is non-numeric. Encode categorical features first."
        )


def _validate_features(
    X: pd.DataFrame,
    features: List[str],
    categorical_groups: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Validate features exist and are numeric (group-aware)."""
    groups = categorical_groups or {}
    for f in features:
        if f in groups:
            _validate_group_members(X, f, groups[f])
        else:
            _validate_scalar_feature(X, f)


def _subsample_and_prepare(
    model, X: pd.DataFrame, max_samples: int, random_state: Optional[int],
) -> Tuple[Callable, np.ndarray, List[str]]:
    """Subsample data and build predict function for PDP computation.

    Returns (predict_fn, X_values, columns).
    """
    rng = np.random.RandomState(random_state)
    n = min(max_samples, len(X))
    if n < len(X):
        idx = rng.choice(len(X), size=n, replace=False)
        X_sub = X.iloc[idx]
    else:
        X_sub = X

    columns = X.columns.tolist()

    def predict_fn(arr):
        return model.predict(pd.DataFrame(arr, columns=columns))

    X_values = X_sub.values.astype(np.float64)
    return predict_fn, X_values, columns


def _build_h_pair_grids(
    X_values: np.ndarray, u1: FeatureUnit, u2: FeatureUnit,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-unit H-stat grids (no grid_resolution cap on numerics).

    Numeric features use ALL unique values (preserves the original behavior);
    categorical groups use observed one-hot patterns.
    """
    def _grid(unit: FeatureUnit) -> np.ndarray:
        if unit.is_group:
            return _build_categorical_grid(X_values, unit.col_indices)
        return np.unique(X_values[:, unit.col_indices[0]])

    return _grid(u1), _grid(u2)


def _compute_pair_h_safe(
    predict_fn, X_values, u1: FeatureUnit, u2: FeatureUnit,
    grid_1: np.ndarray, grid_2: np.ndarray,
    f1: str, f2: str,
) -> Tuple[float, bool]:
    """Compute H-stat for one pair, converting backend errors to NaN.

    Returns ``(h_stat, failed)``. ``MemoryError``/``RecursionError``/
    ``ImportError`` re-raise. Other exceptions (model.predict can raise
    any backend type — same rationale as the Optuna catch-and-prune
    pattern in ``tuning/base.py``) emit ``SuggestionWarning`` and yield
    ``(NaN, True)``.
    """
    try:
        h_stat = _h_statistic_pair_units(
            predict_fn, X_values, u1, u2, grid_1, grid_2,
        )
        return h_stat, False
    except (MemoryError, RecursionError, ImportError):
        raise
    except Exception as exc:
        logger.debug(
            "Interaction pair (%s, %s) failed: %s", f1, f2, exc,
            exc_info=True,
        )
        warnings.warn(
            f"Interaction pair ({f1}, {f2}) failed: {exc!r}. "
            "Setting h_statistic=NaN for this pair.",
            SuggestionWarning,
            stacklevel=4,
        )
        return np.nan, True


def _describe_unit_grid(
    unit: FeatureUnit, grid: np.ndarray, columns: List[str],
    categorical_features: Optional[set] = None,
    cat_max_cardinality: int = _CAT_MAX_CARDINALITY,
) -> Tuple[list, str, Optional[List[str]]]:
    """Return (grid_as_list, grid_type, labels) for one feature unit.

    A one-hot group is always categorical. A single scalar column is
    categorical when either:

    - ``categorical_features`` is None (heuristic mode) and the column's grid
      has at most ``cat_max_cardinality`` points, or
    - ``categorical_features`` is a set (explicit mode, heuristic disabled)
      and the column name is in it.

    The heuristic uses ``len(grid)``. On a linear grid that equals the number
    of unique observed values (``_build_grid`` returns ``np.unique``); a
    log-spaced grid always has ``grid_resolution`` points, so a log-scaled
    column is never auto-detected as categorical (log axes are continuous
    anyway).

    Categorical scalars carry value-based labels (``"0"``/``"1"``) so the viz
    layer renders them discretely instead of interpolating a contour across
    impossible intermediate values.
    """
    if unit.is_group:
        group_cols = [columns[i] for i in unit.col_indices]
        return grid.tolist(), 'categorical', _categorical_grid_labels(grid, group_cols)
    if categorical_features is None:
        is_cat = len(grid) <= cat_max_cardinality
    else:
        is_cat = unit.name in categorical_features
    if is_cat:
        if not np.all(np.isfinite(grid)):
            warnings.warn(
                f"Feature '{unit.name}' is treated as categorical but its grid "
                f"contains non-finite values; these render as 'nan'/'inf' axis "
                f"labels. Check for missing values in this column.",
                SuggestionWarning,
                stacklevel=2,
            )
        return grid.tolist(), 'categorical', [_fmt_level(v) for v in grid]
    return grid.tolist(), 'numeric', None


def _build_pdp_surface_record(
    predict_fn, X_values, columns: List[str],
    u1: FeatureUnit, u2: FeatureUnit,
    f1: str, f2: str, grid_resolution: int,
    categorical_features: Optional[set] = None,
    scale_1: str = 'linear', scale_2: str = 'linear',
) -> Dict[str, Any]:
    """Compute and package one PDP surface (capped grids + metadata).

    *scale_1*/*scale_2* set the axis spacing for *u1*/*u2* respectively; each is
    resolved per axis via :func:`_resolve_axis_scale` (``'auto'`` picks from the
    data), and the effective spacing is stored as ``grid_*_scale`` so the
    renderer can lay out the axis to match.
    """
    eff_scale_1 = _resolve_axis_scale(scale_1, u1, X_values, f1)
    eff_scale_2 = _resolve_axis_scale(scale_2, u2, X_values, f2)
    surf_grid_1 = _build_unit_grid(X_values, u1, grid_resolution, eff_scale_1)
    surf_grid_2 = _build_unit_grid(X_values, u2, grid_resolution, eff_scale_2)
    surface = _pdp_2d_unit(predict_fn, X_values, u1, u2, surf_grid_1, surf_grid_2)

    if not np.all(np.isfinite(surface)):
        n_bad = int(np.sum(~np.isfinite(surface)))
        warnings.warn(
            f"PDP surface for ({f1}, {f2}) contains {n_bad} non-finite "
            "values. Model predictions may contain NaN/Inf.",
            SuggestionWarning,
            stacklevel=4,
        )

    record = {
        'feature_1': f1,
        'feature_2': f2,
        'pdp_surface': surface.tolist(),
    }
    scales = {'1': eff_scale_1, '2': eff_scale_2}
    for label, unit, grid in [('1', u1, surf_grid_1), ('2', u2, surf_grid_2)]:
        grid_list, grid_type, labels = _describe_unit_grid(
            unit, grid, columns, categorical_features,
        )
        record[f'grid_{label}'] = grid_list
        record[f'grid_{label}_type'] = grid_type
        record[f'grid_{label}_labels'] = labels
        record[f'grid_{label}_scale'] = scales[label]
    return record


def _h_stat_iterate_pairs(
    pairs: List[Tuple[str, str]],
    unit_map: Dict[str, FeatureUnit],
    predict_fn, X_values, columns: List[str],
    compute_surfaces: bool, grid_resolution: int,
    show_progress: bool, display_name: str,
    categorical_features: Optional[set] = None,
    scale: str = 'linear',
) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], int]:
    """Compute H-stat (and optional PDP surfaces) for every pair.

    *scale* sets the PDP surface axis spacing (applied to both axes, each
    resolved per axis); it does not affect the H-statistic itself.

    Returns ``(results, surfaces_or_None, n_failed)``.
    """
    results: List[Dict[str, Any]] = []
    surfaces: Optional[List[Dict[str, Any]]] = [] if compute_surfaces else None
    n_failed = 0

    for pair_idx, (f1, f2) in enumerate(pairs):
        if show_progress:
            print(
                f"  Interactions {display_name}: pair {pair_idx + 1}/{len(pairs)} "
                f"({f1}, {f2})...",
                end="", flush=True,
            )

        u1, u2 = unit_map[f1], unit_map[f2]
        grid_1, grid_2 = _build_h_pair_grids(X_values, u1, u2)
        h_stat, failed = _compute_pair_h_safe(
            predict_fn, X_values, u1, u2, grid_1, grid_2, f1, f2,
        )
        n_failed += int(failed)

        if show_progress:
            print(f" H={h_stat:.4f}")

        results.append({'feature_1': f1, 'feature_2': f2, 'h_statistic': h_stat})

        if compute_surfaces and np.isfinite(h_stat):
            surfaces.append(_build_pdp_surface_record(
                predict_fn, X_values, columns, u1, u2, f1, f2, grid_resolution,
                categorical_features, scale_1=scale, scale_2=scale,
            ))

    return results, surfaces, n_failed


def _compute_h_statistic(
    model,
    X: pd.DataFrame,
    features: List[str],
    max_samples: int = 100,
    random_state: Optional[int] = None,
    show_progress: bool = False,
    model_name: Optional[str] = None,
    compute_surfaces: bool = False,
    grid_resolution: int = 25,
    categorical_groups: Optional[Dict[str, List[str]]] = None,
    categorical_features: Optional[set] = None,
    scale: str = 'linear',
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
    """Compute Friedman's H-statistic for all pairs of given features.

    Parameters
    ----------
    model : estimator
        Fitted model/pipeline with ``predict(X)`` returning risk scores.
    X : pd.DataFrame
        Test data for PDP computation.
    features : list of str
        Feature names to compute pairwise interactions for.
        Must have at least 2 elements.  May include group names from
        *categorical_groups*.
    max_samples : int, default=100
        Maximum samples for the PDP grid (subsampled from X).
    random_state : int, optional
        Seed for reproducible subsampling.
    show_progress : bool, default=False
        Whether to print progress messages.
    model_name : str, optional
        Display name for progress messages.
    compute_surfaces : bool, default=False
        If True, also compute and return PDP surfaces for each pair.
    grid_resolution : int, default=25
        Maximum grid points per feature for PDP surfaces.
        Only used when ``compute_surfaces=True``.
    categorical_groups : dict or None
        Mapping from group name to list of dummy column names.
        When provided, group names in *features* are treated as
        multi-column categorical units with grouped PDP.
    categorical_features : set or None
        Single columns to treat as categorical (discrete) in PDP surfaces.
        ``None`` (default) uses a cardinality heuristic (a scalar with at
        most ``_CAT_MAX_CARDINALITY`` unique values is categorical); a set
        disables the heuristic and marks exactly those columns categorical.
        Only used when ``compute_surfaces=True``.
    scale : str, default='linear'
        PDP surface axis spacing: ``'linear'``, ``'log'``, ``'log1p'``, or
        ``'auto'`` (pick per axis from the data). Applied to both axes, each
        resolved independently via :func:`_resolve_axis_scale`. Does not affect
        the H-statistic. Only used when ``compute_surfaces=True``.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, list of dict)
        When ``compute_surfaces=False``: DataFrame with columns
        ``[feature_1, feature_2, h_statistic]``.
        When ``compute_surfaces=True``: tuple of (DataFrame, surfaces)
        where each surface dict has keys ``feature_1``, ``feature_2``,
        ``grid_1``, ``grid_2``, ``pdp_surface``.
        Pairs that fail (model prediction error) have ``h_statistic=NaN``
        and emit a ``SuggestionWarning``; other pairs are unaffected.

    Raises
    ------
    ValueError
        If fewer than 2 features are provided.
    """
    if len(features) < 2:
        raise ValueError(
            f"Need at least 2 features for interaction testing, got {len(features)}."
        )

    _validate_features(X, features, categorical_groups)

    predict_fn, X_values, columns = _subsample_and_prepare(
        model, X, max_samples, random_state,
    )

    # Build feature units (handles both scalar columns and grouped categoricals)
    units = _build_interaction_units(columns, features, categorical_groups)
    unit_map = {u.name: u for u in units}

    pairs = list(itertools.combinations(sorted(u.name for u in units), 2))
    display_name = model_name or "model"

    results, surfaces, n_failed = _h_stat_iterate_pairs(
        pairs, unit_map, predict_fn, X_values, columns,
        compute_surfaces, grid_resolution, show_progress, display_name,
        categorical_features, scale,
    )

    if n_failed:
        warnings.warn(
            f"{n_failed} of {len(pairs)} interaction pair(s) failed and were "
            "set to NaN. Check warnings above for details.",
            SuggestionWarning,
            stacklevel=2,
        )

    df = pd.DataFrame(results)
    df['method'] = 'hstat'
    # Sort by interaction strength descending. Round the sort key to the test
    # tolerance (1e-12) so near-zero numerical noise (~1e-32) collapses into a
    # single tied bucket; break ties by feature names so ordering is
    # deterministic across platforms/BLAS builds. NaN (failed)
    # pairs sort last via -inf, preserving the prior na_position='last'.
    sort_key = df['h_statistic'].round(12).fillna(-np.inf)
    df = (
        df.assign(_sort_key=sort_key)
          .sort_values(['_sort_key', 'feature_1', 'feature_2'],
                       ascending=[False, True, True])
          .drop(columns='_sort_key')
          .reset_index(drop=True)
    )

    if compute_surfaces:
        # Re-sort surfaces to match DataFrame order
        pair_order = {
            (row['feature_1'], row['feature_2']): i
            for i, row in df.iterrows()
        }
        surfaces.sort(key=lambda s: pair_order[(s['feature_1'], s['feature_2'])])
        return df, surfaces
    return df


def _auto_axis_scale(finite: np.ndarray) -> str:
    """Pick ``'linear'``/``'log'``/``'log1p'`` for a numeric axis from its data.

    *finite* is the feature's finite (non-NaN) values.  Right-skewed features
    (Fisher skew > ``_AUTO_SKEW_THRESHOLD``) get a log spacing: ``'log'`` when
    strictly positive, ``'log1p'`` when non-negative with zeros.  Low-skew,
    near-categorical (``<= _CAT_MAX_CARDINALITY`` unique values), all-NaN, or
    features with any negative value stay ``'linear'``.
    """
    if finite.size == 0:
        return 'linear'
    # Near-categorical axes (few distinct values, incl. constant columns) read
    # better on a linear/discrete axis; this also keeps skew well-defined below.
    if np.unique(finite).size <= _CAT_MAX_CARDINALITY:
        return 'linear'
    if finite.min() < 0:
        return 'linear'
    if skew(finite) <= _AUTO_SKEW_THRESHOLD:
        return 'linear'
    return 'log' if finite.min() > 0 else 'log1p'


def _resolve_axis_scale(
    scale: str, unit: FeatureUnit, X_values: np.ndarray, feature_name: str,
) -> str:
    """Resolve the effective grid scale for one PDP axis.

    Accepts ``'linear'``, ``'log'``, ``'log1p'``, or ``'auto'`` and returns a
    concrete ``'linear'``/``'log'``/``'log1p'``.

    Explicit ``'log'``/``'log1p'`` fall back to ``'linear'`` (with a
    :class:`SuggestionWarning`) when the axis cannot represent the request:
    categorical groups for either; any negative value for either; and, for
    ``'log'`` only, no positive value.  ``'auto'`` chooses the spacing from the
    data and never warns (see :func:`_auto_axis_scale`); for a categorical group
    it is simply ``'linear'``.
    """
    if scale not in ('linear', 'log', 'log1p', 'auto'):
        raise ValueError(
            f"scale must be 'linear', 'log', 'log1p', or 'auto', got {scale!r}."
        )
    if scale == 'linear':
        return 'linear'

    # Groups (one-hot categoricals) have no continuous axis to rescale.
    if unit.is_group:
        if scale != 'auto':
            warnings.warn(
                f"{scale} scale requested for categorical feature "
                f"'{feature_name}'; falling back to linear.",
                SuggestionWarning,
                stacklevel=3,
            )
        return 'linear'

    # Work on the finite values only (mirrors _build_grid), which also avoids a
    # stray "All-NaN slice" RuntimeWarning on an all-NaN column.
    finite = X_values[:, unit.col_indices[0]]
    finite = finite[np.isfinite(finite)]

    if scale == 'auto':
        return _auto_axis_scale(finite)

    # Explicit 'log'/'log1p' on an all-NaN column: no value to place on a log
    # axis, and _build_grid would crash on the empty min(). Fall back like 'log'
    # already does for the no-positive-value case below.
    if not finite.size:
        warnings.warn(
            f"{scale} scale requested for '{feature_name}', which has no finite "
            "values; a log axis needs at least one. Falling back to linear.",
            SuggestionWarning,
            stacklevel=3,
        )
        return 'linear'

    # Explicit 'log'/'log1p': a negative value has no position on either axis.
    if finite.min() < 0:
        warnings.warn(
            f"{scale} scale requested for '{feature_name}', which has negative "
            "values; a log axis cannot represent them. Falling back to linear.",
            SuggestionWarning,
            stacklevel=3,
        )
        return 'linear'
    if scale == 'log' and not np.any(finite > 0):
        warnings.warn(
            f"log scale requested for '{feature_name}', which has no positive "
            "values; a log axis needs at least one. Falling back to linear.",
            SuggestionWarning,
            stacklevel=3,
        )
        return 'linear'
    return scale


def _resolve_pair_scales(
    model,
    X: pd.DataFrame,
    feature_1: str,
    feature_2: str,
    scale_1: str,
    scale_2: str,
    max_samples: int = 100,
    random_state: Optional[int] = None,
    categorical_groups: Optional[Dict[str, List[str]]] = None,
) -> Tuple[str, str]:
    """Resolve the effective per-axis spacing for a pair without the surface.

    Mirrors the resolution :func:`_compute_pdp_surface` does (same subsample,
    units, and :func:`_resolve_axis_scale` calls) but stops once the concrete
    ``'linear'``/``'log'``/``'log1p'`` spacing is known. The plot path uses it
    to decide whether a cached surface matches the requested scale before
    falling back to a recompute. Does not compute any model predictions.
    """
    _validate_features(X, [feature_1, feature_2], categorical_groups)
    _, X_values, columns = _subsample_and_prepare(
        model, X, max_samples, random_state,
    )
    units = _build_interaction_units(
        columns, [feature_1, feature_2], categorical_groups,
    )
    eff_scale_1 = _resolve_axis_scale(scale_1, units[0], X_values, feature_1)
    eff_scale_2 = _resolve_axis_scale(scale_2, units[1], X_values, feature_2)
    return eff_scale_1, eff_scale_2


def _compute_pdp_surface(
    model,
    X: pd.DataFrame,
    feature_1: str,
    feature_2: str,
    max_samples: int = 100,
    grid_resolution: int = 25,
    random_state: Optional[int] = None,
    categorical_groups: Optional[Dict[str, List[str]]] = None,
    categorical_features: Optional[set] = None,
    scale_1: str = 'linear',
    scale_2: str = 'linear',
) -> Dict[str, Any]:
    """Compute 2D partial dependence surface for a single feature pair.

    Standalone function for on-the-fly PDP computation (e.g., for pairs
    not cached during ``interactions()``).

    Parameters
    ----------
    model : estimator
        Fitted model/pipeline with ``predict(X)`` returning risk scores.
    X : pd.DataFrame
        Data for PDP computation (typically test set).
    feature_1, feature_2 : str
        Feature names (may be group names from *categorical_groups*).
    max_samples : int, default=100
        Maximum samples for PDP grid.
    grid_resolution : int, default=25
        Maximum grid points per feature.
    random_state : int, optional
        Seed for reproducible subsampling.
    categorical_groups : dict or None
        Mapping from group name to list of dummy column names.
    categorical_features : set or None
        Single columns to treat as categorical (discrete). ``None`` uses the
        cardinality heuristic; a set disables it and marks exactly those
        columns categorical. See ``_describe_unit_grid``.
    scale_1, scale_2 : str, default='linear'
        Grid spacing for *feature_1* and *feature_2* respectively: ``'linear'``,
        ``'log'``, ``'log1p'``, or ``'auto'`` (pick from the data). A ``'log'``
        request falls back to linear (with a warning) for categorical,
        negative-valued, or all-non-positive features; ``'log1p'`` falls back
        for categorical or negative-valued features. See
        :func:`_resolve_axis_scale`.

    Returns
    -------
    dict
        Keys: ``feature_1``, ``feature_2``, ``grid_1``, ``grid_2``,
        ``pdp_surface`` (all as Python lists for serialization safety), plus
        ``grid_*_type`` and ``grid_*_scale`` (always present; the latter is the
        effective ``'linear'``/``'log'``/``'log1p'`` spacing actually used) and
        ``grid_*_labels`` (populated only for categorical axes, else ``None``).
    """
    _validate_features(X, [feature_1, feature_2], categorical_groups)

    predict_fn, X_values, columns = _subsample_and_prepare(
        model, X, max_samples, random_state,
    )

    units = _build_interaction_units(
        columns, [feature_1, feature_2], categorical_groups,
    )
    u1, u2 = units[0], units[1]
    eff_scale_1 = _resolve_axis_scale(scale_1, u1, X_values, feature_1)
    eff_scale_2 = _resolve_axis_scale(scale_2, u2, X_values, feature_2)
    grid_1 = _build_unit_grid(X_values, u1, grid_resolution, eff_scale_1)
    grid_2 = _build_unit_grid(X_values, u2, grid_resolution, eff_scale_2)

    surface = _pdp_2d_unit(predict_fn, X_values, u1, u2, grid_1, grid_2)

    if not np.all(np.isfinite(surface)):
        n_bad = int(np.sum(~np.isfinite(surface)))
        warnings.warn(
            f"PDP surface for ({feature_1}, {feature_2}) contains {n_bad} "
            "non-finite values. Model predictions may contain NaN/Inf.",
            SuggestionWarning,
            stacklevel=2,
        )

    result = {
        'feature_1': feature_1,
        'feature_2': feature_2,
        'pdp_surface': surface.tolist(),
    }

    scales = {'1': eff_scale_1, '2': eff_scale_2}
    for label, unit, grid in [('1', u1, grid_1), ('2', u2, grid_2)]:
        grid_list, grid_type, labels = _describe_unit_grid(
            unit, grid, columns, categorical_features,
        )
        result[f'grid_{label}'] = grid_list
        result[f'grid_{label}_type'] = grid_type
        result[f'grid_{label}_labels'] = labels
        result[f'grid_{label}_scale'] = scales[label]

    return result


def _is_xgb_booster(obj) -> bool:
    """Return True if *obj* is an ``xgboost.Booster``; warn if xgb is missing."""
    try:
        xgb = import_xgboost("XGBoost interaction detection")
    except ImportError as exc:
        logger.warning(
            "Found booster_ attribute but xgboost import failed (%s); "
            "cannot verify XGBoost Booster type.", exc
        )
        return False
    return isinstance(obj, xgb.Booster)


def _classify_sksurv_ensemble(inner):
    """Identify a sksurv ensemble; returns ``(estimator, model_type)`` or ``(None, None)``."""
    from sksurv.ensemble import (
        GradientBoostingSurvivalAnalysis,
        RandomSurvivalForest,
    )
    if isinstance(inner, GradientBoostingSurvivalAnalysis):
        return inner, 'gbsa'
    if isinstance(inner, RandomSurvivalForest):
        return inner, 'rsf'
    return None, None


def _extract_tree_estimator(pipeline):
    """Extract a tree estimator suitable for SHAP interaction values.

    Handles XGBCoxModel (XGBRegressor in ``model_``), XGBAFTModel
    (raw Booster in ``booster_``), GBSA
    (GradientBoostingSurvivalAnalysis in ``model_``), and RSF
    (RandomSurvivalForest in ``model_``).

    Returns ``(estimator, model_type)``; ``(None, None)`` if unsupported.
    """
    wrapper = pipeline.steps[-1][1] if hasattr(pipeline, 'steps') else pipeline
    inner = getattr(wrapper, 'model_', None)

    if inner is not None and hasattr(inner, 'get_booster'):
        return inner, 'xgboost'

    booster = getattr(wrapper, 'booster_', None)
    if booster is not None and _is_xgb_booster(booster):
        return booster, 'xgboost'

    if inner is not None and hasattr(inner, 'estimators_'):
        return _classify_sksurv_ensemble(inner)

    return None, None


@contextlib.contextmanager
def _patch_sksurv_for_shap(estimator, model_type):
    """Temporarily patch an sksurv estimator for ``shap.TreeExplainer``.

    SHAP's ``TreeEnsemble`` uses ``safe_isinstance`` checks that don't
    recognise sksurv types.  This swaps ``__class__`` to the
    corresponding sklearn regressor and sets model-specific attributes
    so SHAP can read the tree structure.  Originals are restored on
    exit.

    For GBSA: swaps to ``GradientBoostingRegressor`` and replaces
    ``init_`` (string ``"zero"`` in sksurv) with a
    ``DummyRegressor(constant=0)``.

    For RSF: swaps to ``RandomForestRegressor`` and sets ``criterion``
    to ``"squared_error"`` (RSF uses log-rank splitting, which SHAP
    doesn't recognise).

    Parameters
    ----------
    estimator : sksurv estimator
        The fitted sksurv estimator to patch.
    model_type : {'gbsa', 'rsf'}
        Which model type determines the patches applied.

    Yields
    ------
    estimator
        The same object, temporarily disguised as a sklearn regressor.
    """
    original_class = estimator.__class__

    if model_type == 'gbsa':
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import GradientBoostingRegressor

        original_init = estimator.init_
        dummy = DummyRegressor(strategy="constant", constant=0.0)
        # SHAP reads init_.constant_[0] — set the fitted attribute directly.
        dummy.constant_ = np.array([0.0])

        estimator.__class__ = GradientBoostingRegressor
        estimator.init_ = dummy
    elif model_type == 'rsf':
        from sklearn.ensemble import RandomForestRegressor

        original_criterion = getattr(estimator, 'criterion', None)
        had_criterion = hasattr(estimator, 'criterion')

        estimator.__class__ = RandomForestRegressor
        estimator.criterion = "squared_error"
    else:
        raise ValueError(
            f"Unsupported model_type {model_type!r} for SHAP patching. "
            f"Expected 'gbsa' or 'rsf'."
        )

    try:
        yield estimator
    finally:
        estimator.__class__ = original_class
        if model_type == 'gbsa':
            estimator.init_ = original_init
        elif model_type == 'rsf':
            if had_criterion:
                estimator.criterion = original_criterion
            else:
                del estimator.criterion


def _decompose_pipeline(pipeline, X):
    """Split pipeline into preprocessing and model, transform X.

    Reuses the pattern from ``_explain.py`` for pipeline decomposition
    with robust feature name recovery.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn pipeline.
    X : pd.DataFrame
        Input data to transform through preprocessing steps.

    Returns
    -------
    tuple of (model_wrapper, X_transformed)
        model_wrapper : object
            The last step of the pipeline (survival model wrapper).
        X_transformed : pd.DataFrame
            Input data after all preprocessing/selection steps.
    """
    if not hasattr(pipeline, 'steps') or len(pipeline.steps) < 2:
        # No preprocessing — return as-is
        wrapper = pipeline.steps[-1][1] if hasattr(pipeline, 'steps') else pipeline
        return wrapper, X

    preprocess = Pipeline(pipeline.steps[:-1])
    wrapper = pipeline.steps[-1][1]

    X_t = preprocess.transform(X)

    # Ensure DataFrame with correct feature names
    if not isinstance(X_t, pd.DataFrame):
        n_cols_out = X_t.shape[1]
        cols = X.columns

        if hasattr(preprocess, "get_feature_names_out"):
            try:
                cols = preprocess.get_feature_names_out()
            except (ValueError, AttributeError):
                logger.debug(
                    "get_feature_names_out() failed; using original column names."
                )

        # Truncate or use available names
        cols = list(cols)[:n_cols_out]
        if len(cols) < n_cols_out:
            warnings.warn(
                f"Pipeline preprocessing changed feature count from "
                f"{len(X.columns)} to {n_cols_out} and feature names could "
                f"not be recovered. Using synthetic names (f0, f1, ...).",
                SuggestionWarning,
                stacklevel=2,
            )
            cols = [f"f{i}" for i in range(n_cols_out)]

        X_t = pd.DataFrame(X_t, columns=cols, index=X.index[:len(X_t)])

    return wrapper, X_t


@dataclass
class _ShapSetup:
    """Inputs for SHAP interaction computation, post-validation."""
    tree_estimator: Any
    model_type: str
    X_sub: pd.DataFrame
    all_cols: List[str]
    groups: Dict[str, List[str]]
    n_features: int


def _validate_shap_features(
    available: set, features: List[str], groups: Dict[str, List[str]],
) -> None:
    """Verify each feature (or group's members) exists in the transformed data."""
    for f in features:
        if f in groups:
            missing = [c for c in groups[f] if c not in available]
            if missing:
                raise ValueError(
                    f"Categorical group '{f}' has columns not in "
                    f"transformed data: {missing}. "
                    f"Available: {sorted(available)}"
                )
        elif f not in available:
            raise ValueError(
                f"Features not found in transformed data: [{f!r}]. "
                f"Available: {sorted(available)}"
            )


def _prepare_shap_inputs(
    pipeline, X: pd.DataFrame, features: List[str],
    max_samples: int, random_state: Optional[int],
    categorical_groups: Optional[Dict[str, List[str]]],
) -> _ShapSetup:
    """Validate inputs and prepare the data + estimator needed for SHAP.

    Performs the SHAP-specific setup that does not run the explainer:
    import check, pipeline decomposition, tree-estimator extraction,
    transformed-data feature validation, row subsampling.
    """
    try:
        import shap  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'shap' package is required for SHAP interaction values. "
            "Install with: pip install shap  or  pip install mlsurv[deep]"
        )

    if len(features) < 2:
        raise ValueError(
            f"Need at least 2 features for interaction testing, got {len(features)}."
        )

    groups = categorical_groups or {}

    _wrapper, X_t = _decompose_pipeline(pipeline, X)
    tree_estimator, model_type = _extract_tree_estimator(pipeline)
    if tree_estimator is None:
        raise TypeError(
            "SHAP interaction values require a tree-based model (xgbcox, "
            "xgbaft, gbsa, or rsf). The current model is not a supported tree "
            "type. Use method='hstat' for model-agnostic interaction "
            "detection."
        )

    _validate_shap_features(set(X_t.columns), features, groups)

    rng = np.random.RandomState(random_state)
    n = min(max_samples, len(X_t))
    if n < len(X_t):
        idx = rng.choice(len(X_t), size=n, replace=False)
        X_sub = X_t.iloc[idx]
    else:
        X_sub = X_t

    return _ShapSetup(
        tree_estimator=tree_estimator,
        model_type=model_type,
        X_sub=X_sub,
        all_cols=list(X_t.columns),
        groups=groups,
        n_features=len(features),
    )


def _run_shap_explainer(
    setup: _ShapSetup, show_progress: bool, model_name: Optional[str],
) -> np.ndarray:
    """Run ``shap.TreeExplainer`` and return a 3D interaction array.

    Owns the sksurv-vs-XGBoost context-manager dispatch, the 4D→3D
    output-axis reduction (RSF flattens per-tree (n_times, 2) leaf
    values into a single output axis), the ndim guard, and the
    non-finite warning.

    Interaction values are **path-dependent** (true-to-the-data): the
    explainer is built without a background, since TreeSHAP only supports
    interactions for the path-dependent perturbation. This is a
    different reference than the interventional cohort beeswarm in
    ``evaluation/_shap.py`` (see the call site for the full rationale).
    """
    import shap

    display_name = model_name or "model"
    if show_progress:
        print(
            f"  Interactions {display_name}: computing SHAP interaction values "
            f"({len(setup.X_sub)} samples, {setup.n_features} features)...",
            end="", flush=True,
        )

    if setup.model_type in ('gbsa', 'rsf'):
        ctx = _patch_sksurv_for_shap(setup.tree_estimator, setup.model_type)
    else:
        ctx = contextlib.nullcontext(setup.tree_estimator)

    with ctx as est:
        # Interaction values stay PATH-DEPENDENT on purpose. The cohort
        # beeswarm in ``evaluation/_shap.py``
        # (``_run_explainer``) is a marginal explainer that masks features
        # against a summarized ``X_train`` background, but TreeSHAP only
        # supports interaction values for the path-dependent perturbation.
        # Asking for interventional interactions makes shap print
        # "FEATURE_DEPENDENCE::independent does not support interactions!" and
        # return numbers it declares unsupported (verified on shap 0.51.0), so
        # we do NOT pass ``data=``/``feature_perturbation`` here. The trade-off
        # is that this heatmap uses a true-to-the-data reference while the
        # cohort beeswarm uses a true-to-the-model one; they can diverge when
        # features are correlated.
        explainer = shap.TreeExplainer(est)
        raw_full = explainer.shap_interaction_values(setup.X_sub.values)
    raw_full = np.asarray(raw_full, dtype=np.float64)

    if raw_full.ndim == 4:
        raw_full = raw_full.mean(axis=3)
    if raw_full.ndim != 3:
        raise ValueError(
            f"Expected 3D SHAP interaction array "
            f"(n_samples, n_features, n_features) after aggregation, "
            f"got shape {raw_full.shape} (ndim={raw_full.ndim}). "
            f"This may indicate an unsupported model type or SHAP version."
        )

    n_nonfinite = int(np.sum(~np.isfinite(raw_full)))
    if n_nonfinite:
        warnings.warn(
            f"SHAP interaction values contain {n_nonfinite}/{raw_full.size} "
            f"non-finite entries for {display_name}. Affected interaction "
            f"pairs may show NaN.",
            MethodologicalWarning,
            stacklevel=3,
        )

    if show_progress:
        print(" done.")
    return raw_full


def _summarize_shap_interactions(
    raw_full: np.ndarray, all_cols: List[str], features: List[str],
    groups: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Reduce raw per-sample matrix to a sorted summary DataFrame.

    Aggregates by group (when any feature names a group) or slices by
    feature index (otherwise), then computes mean |interaction| for
    each off-diagonal pair.
    """
    if any(f in groups for f in features):
        units = _build_interaction_units(all_cols, features, groups)
        matrix, feature_names = _aggregate_shap_interaction_groups(
            raw_full, all_cols, units,
        )
    else:
        feat_indices = [all_cols.index(f) for f in features]
        matrix = raw_full[:, feat_indices, :][:, :, feat_indices]
        feature_names = list(features)

    n = len(feature_names)
    rows = []
    for i, j in itertools.combinations(range(n), 2):
        f1, f2 = sorted([feature_names[i], feature_names[j]])
        rows.append({
            'feature_1': f1,
            'feature_2': f2,
            'shap_interaction': float(np.mean(np.abs(matrix[:, i, j]))),
        })

    df = pd.DataFrame(rows)
    df['method'] = 'shap'
    # Sort by interaction strength descending, with the same tolerance-bucketed
    # key + feature-name tie-break as the H-statistic path so near-zero pairs
    # order deterministically across platforms.
    sort_key = df['shap_interaction'].round(12).fillna(-np.inf)
    df = (
        df.assign(_sort_key=sort_key)
          .sort_values(['_sort_key', 'feature_1', 'feature_2'],
                       ascending=[False, True, True])
          .drop(columns='_sort_key')
          .reset_index(drop=True)
    )
    return df, matrix, feature_names


def _compute_shap_interactions(
    pipeline,
    X: pd.DataFrame,
    features: List[str],
    max_samples: int = 100,
    random_state: Optional[int] = None,
    show_progress: bool = False,
    model_name: Optional[str] = None,
    categorical_groups: Optional[Dict[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Compute SHAP interaction values for tree-based models.

    Uses ``shap.TreeExplainer.shap_interaction_values()`` to compute
    per-sample pairwise interaction matrices, then aggregates to
    mean absolute interaction per pair.

    These interaction values are **path-dependent** (true-to-the-data) and so
    use a different reference than the interventional cohort beeswarm SHAP
    (see ``evaluation/_shap.py``); the two can diverge when
    features are correlated.

    Supported model types: XGBoost (xgbcox, xgbaft), GBSA, and RSF.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn pipeline with a tree-based model as the last step.
    X : pd.DataFrame
        Test data (raw, before pipeline preprocessing).
    features : list of str
        Feature names to compute interactions for (pre-preprocessing names
        from X_test; must also exist in the transformed data after
        pipeline preprocessing).  May include group names from
        *categorical_groups*.  At least 2 required.
    max_samples : int, default=100
        Maximum samples for SHAP computation (subsampled from X).
    random_state : int, optional
        Seed for reproducible subsampling.
    show_progress : bool, default=False
        Whether to print progress messages.
    model_name : str, optional
        Display name for progress messages.
    categorical_groups : dict or None
        Mapping from group name to list of dummy column names.
        When provided, SHAP interaction values are aggregated across
        group member columns.

    Returns
    -------
    tuple of (summary_df, raw_array, feature_names)
        summary_df : pd.DataFrame
            Columns: ``[feature_1, feature_2, shap_interaction, method]``.
            Pairs sorted by ``shap_interaction`` descending.
        raw_array : np.ndarray
            Shape ``(n_samples, n_features, n_features)`` — the full
            per-sample interaction matrix for the selected features
            (aggregated by group when categorical_groups is provided).
        feature_names : list of str
            Feature names corresponding to the matrix axes.

    Raises
    ------
    ImportError
        If the ``shap`` package is not installed.
    TypeError
        If the pipeline does not contain a supported tree-based model.
    ValueError
        If fewer than 2 features are provided.
    """
    setup = _prepare_shap_inputs(
        pipeline, X, features, max_samples, random_state, categorical_groups,
    )
    raw_full = _run_shap_explainer(setup, show_progress, model_name)
    return _summarize_shap_interactions(
        raw_full, setup.all_cols, features, setup.groups,
    )
