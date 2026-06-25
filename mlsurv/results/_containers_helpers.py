"""Shared helpers for result containers: constants, formatting, DataFrame utilities, and lazy imports.

Extracted from ``containers.py`` to reduce module complexity.
All names are re-exported by the facade (``containers.py``) so existing
``from mlsurv.results.containers import X`` paths work unchanged.
"""

import json
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# JSON-safe serialization (shared by every container's to_dict())
# ---------------------------------------------------------------------------

def _nan_safe(val):
    """Convert NaN/inf to None so a scalar survives ``json.dumps``."""
    if val is None:
        return None
    try:
        if np.isnan(val) or np.isinf(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _df_to_records(df: Optional[pd.DataFrame]) -> list:
    """Convert a DataFrame to a list of JSON-safe record dicts.

    ``None`` and empty DataFrames yield ``[]``. NaN/inf cells become JSON
    ``null``. We round-trip through ``to_json`` (rather than
    ``df.to_dict(orient='records')``) because ``to_json`` natively maps
    NaN/inf to ``null`` and emits plain Python types, so the result is always
    ``json.dumps``-safe — ``to_dict`` would leave raw ``np.nan``/``np.inf``
    floats that break ``json.dumps``.
    """
    if not _is_nonempty_df(df):
        return []
    # ``orient='records'`` drops the index, so a *named* index (e.g. the
    # ``time`` axis of a survival-function table, or the ``patient`` index of
    # the prediction tables) would be lost. Promote a named index into a
    # column first. We key off the name, not the index type: an unnamed index
    # carries no information worth a column — that covers the default
    # RangeIndex of the model_type-keyed frames and the stepped/plain index
    # left behind after row-filtering (e.g. BootstrapResult's subpopulation
    # filter), both of which ``to_json`` simply omits. If the index name
    # already exists as a column, keep the column and drop the index rather
    # than letting ``reset_index`` raise on the collision.
    idx_name = None if isinstance(df.index, pd.MultiIndex) else df.index.name
    if idx_name is not None and idx_name not in df.columns:
        df = df.reset_index()
    return json.loads(df.to_json(orient="records"))


class DictExportMixin:
    """Generic ``to_dict(full=...)`` layered on an existing ``to_dataframes()``.

    Subclasses override :meth:`_summary_dict` to supply headline scalars. The
    ``full=True`` table dump runs one shared implementation for every
    container: each DataFrame from ``to_dataframes()`` becomes a list of
    JSON-safe records under a ``tables`` key (the table names and contents
    differ per container, but the mechanism is the same).
    """

    def _summary_dict(self) -> dict:
        """Headline scalars for the lightweight summary. Override per class."""
        return {}

    def to_dataframes(self) -> "Dict[str, pd.DataFrame]":
        """Return ``{name: DataFrame}`` for the full dump. Subclasses provide this.

        Declared here so the mixin's dependency is visible and a subclass that
        forgets it fails with a clear message naming the class, rather than a
        bare ``AttributeError`` from deep inside ``to_dict(full=True)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define to_dataframes() to use "
            "to_dict(full=True)."
        )

    def to_dict(self, full: bool = False) -> dict:
        """Return a JSON-serializable dict view of this result.

        Parameters
        ----------
        full : bool, default=False
            When ``False``, return only the headline summary. When ``True``,
            also include a ``tables`` key mapping each ``to_dataframes()`` name
            to a list of JSON-safe record dicts.

        Returns
        -------
        dict
            Always ``json.dumps``-safe (NaN/inf converted to ``null``).
        """
        # Copy so adding 'tables' can never mutate a dict a subclass might
        # cache or share from _summary_dict().
        out = dict(self._summary_dict())
        if full:
            out["tables"] = {
                name: _df_to_records(df)
                for name, df in self.to_dataframes().items()
            }
        return out


# ---------------------------------------------------------------------------
# Column descriptions for data dictionary generation
# ---------------------------------------------------------------------------

_COLUMN_DESCRIPTIONS: Dict[str, str] = {
    # Common identifiers
    'model_type': 'Model name/key (e.g., rsf, coxph, deepsurv)',
    'model': 'Model name/key',
    'model_a': 'First model in paired comparison',
    'model_b': 'Second model in paired comparison',
    'feature': 'Feature/variable name',
    'fold': 'Cross-validation fold index (0-based)',
    'iteration': 'Bootstrap iteration index (0-based)',
    'index': 'Sample index from original dataset',
    'sample_idx': 'Sample index for SHAP values',
    'cohort': 'Validation cohort name',
    # Metrics
    'metric': 'Metric name (harrell, ipcw, integrated_brier, etc.)',
    'metric_name': 'Metric name in bootstrap iterations',
    'phase': 'Evaluation phase (cv, test)',
    'c_index_train': "Harrell's C-index on training set",
    'c_index_test': "Harrell's C-index on test set",
    'integrated_brier_test': 'Integrated Brier Score on test set (lower is better)',
    'c_index_td_test': "Antolini's time-dependent C-index on test set",
    'brier_score_test': 'Brier Score at median time on test set',
    'cal_slope_test': 'Calibration slope on test set (ideal = 1.0)',
    'cal_intercept_test': 'Calibration intercept on test set (ideal = 0.0)',
    'gnd_test': "Greenwood-Nam-D'Agostino calibration test p-value",
    # Statistical values
    'point': 'Full test-set point estimate',
    'mean': 'Mean value (across CV folds or bootstrap iterations)',
    'std': 'Standard deviation',
    'ci_lower': 'Lower bound of confidence interval',
    'ci_upper': 'Upper bound of confidence interval',
    'p_value': 'p-value (unadjusted)',
    'p_adjusted': 'p-value after multiplicity correction',
    'value': 'Metric value for this iteration',
    # Predictions / survival
    'risk_score': 'Model-predicted risk score (higher = more risk)',
    'time': 'Observed survival/follow-up time',
    'event': 'Event indicator (1 = event occurred, 0 = censored)',
    'eval_time': 'Evaluation timepoint for time-dependent metrics',
    'auc': 'Area under the ROC curve at eval_time',
    # Importance
    'coefficient': 'Model coefficient (log hazard ratio); NaN for tree-based models (RSF, GBSA, XGBoost-*, DeepSurv, DeepHit, FSSVM) which lack linear coefficients',
    'perm_mean': 'Mean permutation importance',
    'perm_std': 'Std of permutation importance across repeats',
    'importance': 'Permutation importance value',
    'shap_value': 'SHAP value for feature x sample',
    'observed_importance': 'Observed importance on test set (PIMP)',
    'null_mean': 'Mean importance under null hypothesis (PIMP)',
    'null_std': 'Std of null importance distribution (PIMP)',
    # Comparisons
    'method': 'Statistical method used (paired_bootstrap, kang)',
    'delta': 'Difference between models (A minus B)',
    'z_statistic': 'z-statistic for the comparison',
    # Benchmark
    'reference': 'Reference feature/score name',
    'hazard_ratio': 'Hazard ratio (exp of coefficient)',
    'hr_lower_95': 'Lower 95% CI for hazard ratio',
    'hr_upper_95': 'Upper 95% CI for hazard ratio',
    'spearman_rho': 'Spearman rank correlation coefficient (rho)',
    'spearman_p': 'Spearman correlation p-value',
    'pearson_r': 'Pearson linear correlation coefficient (r)',
    'pearson_p': 'Pearson correlation p-value',
    # Cohort info
    'n_samples': 'Number of samples in cohort',
    'n_events': 'Number of events in cohort',
    'event_rate': 'Proportion of samples with events',
    # Subpopulation analysis
    'subpopulation': 'Subpopulation label (e.g., cancer type)',
    'group_variable': 'Grouping variable name used for subpopulation analysis',
    # CV metrics
    'harrell': "Harrell's C-index (per-fold score)",
    'ipcw': 'IPCW C-index (per-fold score)',
    'integrated_brier': 'Integrated Brier Score (per-fold score)',
    'ctd': "Antolini's C^td (per-fold score)",
}


def _build_data_dictionary(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a data dictionary DataFrame from a collection of DataFrames.

    Parameters
    ----------
    dataframes : dict
        Mapping of table names to DataFrames.

    Returns
    -------
    pd.DataFrame
        Columns: table, column, dtype, description, n_non_null, example.
    """
    rows = []
    for table_name, df in dataframes.items():
        for col in df.columns:
            series = df[col]
            non_null = series.dropna()
            example = str(non_null.iloc[0])[:50] if len(non_null) > 0 else ''
            rows.append({
                'table': table_name,
                'column': col,
                'dtype': str(series.dtype),
                'description': _COLUMN_DESCRIPTIONS.get(col, ''),
                'n_non_null': len(non_null),
                'example': example,
            })
    return pd.DataFrame(rows)


def _is_nonempty_df(obj) -> bool:
    """Return True if obj is a non-None, non-empty DataFrame."""
    return obj is not None and isinstance(obj, pd.DataFrame) and not obj.empty


def _collect_export_fields(obj, fields) -> "Dict[str, pd.DataFrame]":
    """Build ``{key: DataFrame}`` from a declarative field list.

    Parameters
    ----------
    obj : dataclass instance
        Container with DataFrame attributes.
    fields : list of (attr_name, export_key) tuples
        Each pair maps a dataclass attribute to its export key name.
        Attributes that are ``None`` or empty DataFrames are skipped.
    """
    result = {}
    for attr, key in fields:
        val = getattr(obj, attr, None)
        if _is_nonempty_df(val):
            result[key] = val
    return result


def _surv_probs_to_export_df(cache) -> Optional[pd.DataFrame]:
    """Convert ``_surv_probs_cache`` to a single wide DataFrame for export.

    Returns
    -------
    DataFrame or None
        Columns: ``[model_type, sample_idx, t_<time>, ...]``.
        One row per (model, sample).  Returns ``None`` when *cache* is empty.
    """
    if not cache:
        return None
    frames = []
    for model_name, (probs, eval_times) in cache.items():
        if probs is None:
            continue
        arr = np.asarray(probs)
        times = np.asarray(eval_times)
        n_samples = arr.shape[0]
        df = pd.DataFrame(arr, columns=[f"t_{t}" for t in times])
        df.insert(0, 'sample_idx', range(n_samples))
        df.insert(0, 'model_type', model_name)
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _fmt_float(value: Optional[float]) -> str:
    """Format float values safely, handling None/NaN.

    Parameters
    ----------
    value : float or None
        Value to format.

    Returns
    -------
    str
        Formatted string ("N/A" for None/NaN, otherwise 3 decimal places).
    """
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.3f}"


def _count_bootstrap_iterations(iterations_df: Optional[pd.DataFrame]) -> int:
    """Return number of unique bootstrap iterations, or 0 if unavailable."""
    if iterations_df is None:
        return 0
    if len(iterations_df) > 0 and 'iteration' in iterations_df.columns:
        return len(iterations_df['iteration'].unique())
    return 0


# Lazy import to avoid eager loading of printing module at container import time
def _get_printing_functions():
    """Lazy import of printing functions to defer loading cost."""
    from mlsurv.results.printing import (
        print_cv_metrics_table,
        print_multi_model_cv_table,
        print_bootstrap_table,
        print_multi_model_bootstrap_table,
        print_evaluation_table,
        print_multi_model_evaluation_table,
        print_paired_comparison,
    )
    return {
        'cv': print_cv_metrics_table,
        'multi_cv': print_multi_model_cv_table,
        'bootstrap': print_bootstrap_table,
        'multi_bootstrap': print_multi_model_bootstrap_table,
        'evaluation': print_evaluation_table,
        'multi_evaluation': print_multi_model_evaluation_table,
        'paired_comparison': print_paired_comparison,
    }


def _resolve_display(key: str) -> str:
    """Resolve a model key to its human-readable display name.

    Thin wrapper around :func:`mlsurv._shared.display_name`.
    """
    from mlsurv._shared import display_name
    return display_name(key)


def _get_model_names(df: Optional[pd.DataFrame], col: str = 'model_type') -> list:
    """Extract unique values from a DataFrame column (default: 'model_type')."""
    if df is not None and col in df.columns:
        return list(df[col].unique())
    return []


def _get_point_or_mean(row: pd.Series) -> Optional[float]:
    """Return the point estimate if available, otherwise the bootstrap mean."""
    if 'point' in row.index and pd.notna(row['point']):
        return float(row['point'])
    if 'mean' not in row.index:
        return None
    val = row['mean']
    if pd.notna(val):
        return float(val)
    return None


def _filter_model_type(df: Optional[pd.DataFrame], model_name: str, col: str = 'model_type') -> Optional[pd.DataFrame]:
    """Filter a DataFrame to rows matching *model_name* in *col*.

    Returns *df* unchanged when the column is absent, and ``None`` when *df*
    is ``None``.
    """
    if df is None:
        return None
    if col not in df.columns:
        from mlsurv.logging_utils import get_logger
        get_logger("results._containers_helpers").debug(
            "_filter_model_type: column %r absent, returning unfiltered", col,
        )
        return df
    return df[df[col] == model_name].copy()


# ---------------------------------------------------------------------------
# Aggregation helpers (shared by evaluate.py and EvaluationResult.combine)
# ---------------------------------------------------------------------------

def _merge_optional_dicts(objects, attr):
    """Merge an optional dict attribute across objects.

    Parameters
    ----------
    objects : iterable
        Objects with the named attribute.
    attr : str
        Attribute name on each object.

    Returns
    -------
    dict or None
    """
    merged = {}
    for obj in objects:
        val = getattr(obj, attr, None)
        if val is not None:
            merged.update(val)
    return merged or None


def _resolve_interaction_method(objects):
    """Resolve the aggregate interaction method from individual results.

    Parameters
    ----------
    objects : iterable
        Objects with an ``interaction_method`` attribute.

    Returns
    -------
    str or None
    """
    methods = {obj.interaction_method for obj in objects if obj.interaction_method is not None}
    if not methods:
        return None
    if len(methods) == 1 and 'both' not in methods:
        return methods.pop()
    return 'both'
