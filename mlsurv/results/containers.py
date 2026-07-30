"""Result containers for mlsurv evaluation and comparison.

This module is the public facade.  Implementation lives in submodules:

- ``_containers_helpers``: column descriptions, formatting, DataFrame utilities, lazy imports
- ``_containers_extraction``: model extraction, importance summary, comparison
- ``_containers_printing``: print-method delegates

All names are re-exported here so existing ``from mlsurv.results.containers
import X`` statements continue to work unchanged.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Union, TYPE_CHECKING
import warnings
import numpy as np
import pandas as pd

from mlsurv._shared import percentile_ci
from mlsurv._warnings import InternalWarning
from mlsurv.constants import DEFAULT_SCORING_METRIC
from mlsurv.logging_utils import get_logger

logger = get_logger("results.containers")

if TYPE_CHECKING:
    import optuna
    import plotly.graph_objects as go
    from sklearn.isotonic import IsotonicRegression
    from mlsurv.utils import SurvivalArray

# -- Re-exports: helpers ----------------------------------------------------
from mlsurv.results._containers_helpers import (  # noqa: F401
    _COLUMN_DESCRIPTIONS,
    DictExportMixin,
    _df_to_records,
    _nan_safe,
    _build_data_dictionary,
    _is_nonempty_df,
    _collect_export_fields,
    _surv_probs_to_export_df,
    _fmt_float,
    _get_printing_functions,
    _resolve_display,
    _get_model_names,
    _get_point_or_mean,
    _filter_model_type,
    _merge_optional_dicts,
    _resolve_interaction_method,
)

# -- Re-exports: extraction -------------------------------------------------
from mlsurv.results._containers_extraction import (  # noqa: F401
    extract_model_evaluation,
    _extract_model_bootstrap,
)


@dataclass
class EvaluationResult(DictExportMixin):
    """
    Complete evaluation results from one or more models.

    All DataFrames include `model_type` column for consistent filtering
    and multi-model comparison, even when evaluating a single model.

    Attributes
    ----------
    concordance : DataFrame
        Columns: [model_type, c_index_train, c_index_test,
        integrated_brier_test, cal_slope_test, cal_intercept_test, gnd_test]
        Note: brier_score_test may also be present (internal, not in METRIC_REGISTRY).
        Calibration columns are only populated for models that support
        predict_survival_function().
        Four apparent (resubstitution) train-set metrics are also present when
        training data is available: c_index_train (Harrell), c_index_ipcw_train
        (Uno), c_index_td_train (Antolini), and integrated_brier_train (IBS).
        These are point values computed on the training set itself -- an
        optimism/overfitting diagnostic. They flow into export_results and the
        opt-in plot_metric(source="train"), but are deliberately kept out of the
        metric registry, so they never appear in printed output or the HTML
        report.
    test_set : DataFrame
        Columns: [model_type, index, risk_score, time, event]
    auc_t : DataFrame
        Columns: [model_type, eval_time, auc]
    importance : DataFrame
        Columns: [model_type, feature, coefficient, perm_mean, perm_std]
        When ``categorical_groups`` is used, group names replace individual
        dummy columns.
    importance_iterations : DataFrame
        Columns: [model_type, feature, iteration, importance]
        The raw per-repeat permutation values behind the ``importance``
        summary.  This is the supported access path for the raw values
        (per-model: ``result.evaluation.importance_iterations``; aggregated:
        ``learner.evaluation_result.importance_iterations``).
    importance_per_column : DataFrame, optional
        Per-column (ungrouped) importance for individual dummy columns.
        Columns: [model_type, feature, coefficient, perm_mean, perm_std]
        Only populated when ``categorical_groups`` is used; ``None`` otherwise.
    shap_values : DataFrame, optional
        Columns: [model_type, feature, sample_idx, shap_value, feature_value]
        When ``categorical_groups`` is used, group names replace individual
        dummy columns.  The raw per-patient values behind
        ``importance(method='shap')`` — populated by default by that call or by
        ``evaluate(include_shap=True)``, and read by ``plot_shap`` / beeswarm.
        This is the supported access path for the raw values (per-model:
        ``result.evaluation.shap_values``; aggregated:
        ``learner.evaluation_result.shap_values``).
    shap_values_per_column : DataFrame, optional
        Per-column (ungrouped) SHAP values for individual dummy columns.
        Columns: [model_type, feature, sample_idx, shap_value, feature_value]
        Only populated when ``categorical_groups`` is used; ``None`` otherwise.
    interactions : DataFrame, optional
        Pairwise feature interaction strengths.
        For H-statistic: columns [model_type, feature_1, feature_2, h_statistic, method].
        For SHAP: columns [model_type, feature_1, feature_2, shap_interaction, method].
        Only populated when ``include_interactions=True`` in evaluate()
        or ``interactions()`` is called.
    interaction_surfaces : list of dict, optional
        Cached 2D partial dependence surfaces for top interaction pairs.
        Each dict has keys: ``model_type``, ``feature_1``, ``feature_2``,
        ``grid_1``, ``grid_2``, ``pdp_surface`` (all as Python lists).
        Populated alongside ``interactions`` during ``evaluate()`` or
        ``interactions()``.
    interaction_values_raw : np.ndarray, optional
        Per-sample SHAP interaction matrix, shape (n_samples, n_features, n_features).
        Only populated when ``method='shap'`` is used. On multi-model results
        this is ``None``; use ``extract_model_evaluation(result, model_name)``
        or access per-model via ``learner.models['name'].evaluation
        .interaction_values_raw``.
    interaction_feature_names : list of str, optional
        Feature names corresponding to the axes of ``interaction_values_raw``.
    interaction_method : str, optional
        Method used for interaction computation: ``'hstat'`` or ``'shap'``.
    bootstrap : BootstrapResult, optional
        Bootstrap results with confidence intervals (if bootstrap=True in evaluate()).
    truncation_t_max : float, optional
        Training-range cap (``y_train['time'].max()``) applied when computing
        IPCW-weighted metrics (Uno C, AUC(t), IBS). ``None`` when no truncation
        was applied. See ``docs/evaluation.md#training-range-cap``.
    truncation_n_beyond : int
        Number of test observations beyond ``truncation_t_max`` that were
        administratively censored for IPCW metrics. ``0`` when no truncation.
    """
    concordance: pd.DataFrame
    test_set: pd.DataFrame
    auc_t: pd.DataFrame
    importance: pd.DataFrame
    importance_iterations: pd.DataFrame
    importance_per_column: Optional[pd.DataFrame] = None
    shap_values: Optional[pd.DataFrame] = None
    shap_values_per_column: Optional[pd.DataFrame] = None
    interactions: Optional[pd.DataFrame] = None
    interaction_surfaces: Optional[List[Dict[str, Any]]] = None
    interaction_values_raw: Optional[np.ndarray] = None
    interaction_feature_names: Optional[List[str]] = None
    interaction_method: Optional[str] = None
    bootstrap: Optional['BootstrapResult'] = None
    model_order: Optional[List[str]] = None  # CV-sorted model order for display
    # IPCW truncation diagnostic. Populated by evaluate()/
    # bootstrap()/validate() when y_test/y_val extends past y_train's max.
    # Read by report sections to render a footnote on the metrics table.
    truncation_t_max: Optional[float] = None
    truncation_n_beyond: int = 0
    _surv_probs_cache: Optional[Dict[str, tuple]] = field(default=None, repr=False)
    # Dense, train-anchored IBS survival-probability cache
    # ``{model: (surv_probs, ibs_times)}``. Separate from the display-grid
    # ``_surv_probs_cache``; reused by subpopulation IBS so it integrates over
    # the same dense grid as the headline metric.
    _ibs_surv_probs_cache: Optional[Dict[str, tuple]] = field(default=None, repr=False)
    _interaction_raw_cache: Optional[Dict[str, tuple]] = field(default=None, repr=False)

    def __repr__(self) -> str:
        """String representation showing key metrics."""
        models = _get_model_names(self.concordance)
        n_models = len(models)

        if n_models == 0:
            return "EvaluationResult(no models)"
        elif n_models == 1:
            c_test = self.c_index_test
            c_train = self.c_index_train
            return (
                f"EvaluationResult(model='{models[0]}', "
                f"c_index_test={_fmt_float(c_test)}, "
                f"c_index_train={_fmt_float(c_train)})"
            )
        else:
            return f"EvaluationResult({n_models} models: {', '.join(models)})"

    def __setstate__(self, state):
        import dataclasses as _dc
        self.__dict__.update(state)
        # Fill defaults for fields added in later versions
        for f in _dc.fields(self):
            if f.name not in self.__dict__:
                if f.default is not _dc.MISSING:
                    self.__dict__[f.name] = f.default
                elif f.default_factory is not _dc.MISSING:
                    self.__dict__[f.name] = f.default_factory()

    @classmethod
    def combine(cls, models: Dict[str, 'ModelResult']) -> 'EvaluationResult':
        """Combine per-model evaluations into a multi-model view.

        Parameters
        ----------
        models : dict of {str: ModelResult}
            Models that have been evaluated (must have ``.evaluation`` set).

        Returns
        -------
        EvaluationResult
            Combined multi-model evaluation result.

        Raises
        ------
        ValueError
            If no models have evaluation results.
        """
        evals = [
            mr.evaluation for mr in models.values()
            if mr.evaluation is not None
        ]
        if not evals:
            raise ValueError("No models have evaluation results to combine.")

        # Sentinel for "no default given". A `default=pd.DataFrame()` here
        # would be built once per combine() call and shared by every required
        # field that ends up empty — mutating one would corrupt the others.
        _missing = object()

        def _concat(attr, default=_missing):
            parts = [
                getattr(r, attr) for r in evals
                if getattr(r, attr, None) is not None
                and not getattr(r, attr).empty
            ]
            if parts:
                return pd.concat(parts, ignore_index=True)
            return pd.DataFrame() if default is _missing else default

        merged_surfaces = [s for r in evals for s in (r.interaction_surfaces or [])]

        # Truncation diagnostic is cohort-scoped — every per-model eval that
        # shares y_test should carry the same value. Pick the first non-None
        # so the multi-model view also renders the footnote.
        trunc_t_max = next(
            (r.truncation_t_max for r in evals if r.truncation_t_max is not None),
            None,
        )
        trunc_n_beyond = next(
            (r.truncation_n_beyond for r in evals if r.truncation_n_beyond),
            0,
        )

        return cls(
            concordance=_concat('concordance'),
            test_set=_concat('test_set'),
            auc_t=_concat('auc_t'),
            importance=_concat('importance'),
            importance_iterations=_concat('importance_iterations'),
            importance_per_column=_concat('importance_per_column', default=None),
            shap_values=_concat('shap_values', default=None),
            shap_values_per_column=_concat('shap_values_per_column', default=None),
            interactions=_concat('interactions', default=None),
            interaction_surfaces=merged_surfaces or None,
            interaction_method=_resolve_interaction_method(evals),
            bootstrap=BootstrapResult.combine(models),
            model_order=list(models.keys()),
            _surv_probs_cache=_merge_optional_dicts(evals, '_surv_probs_cache'),
            _ibs_surv_probs_cache=_merge_optional_dicts(evals, '_ibs_surv_probs_cache'),
            _interaction_raw_cache=_merge_optional_dicts(evals, '_interaction_raw_cache'),
            truncation_t_max=trunc_t_max,
            truncation_n_beyond=trunc_n_beyond,
        )

    def _get_concordance_metric(self, column: str, prop_name: str) -> Optional[float]:
        """Return a scalar from the concordance DataFrame, warning on multi-model."""
        if self.concordance is None or column not in self.concordance.columns:
            return None
        if 'model_type' in self.concordance.columns and len(self.concordance['model_type'].unique()) > 1:
            warnings.warn(
                f"Multiple models in result. {prop_name} returns first model's value. "
                "Use extract_model_evaluation() for specific model.",
                InternalWarning,
            )
        value = self.concordance[column].iloc[0]
        if value is None or pd.isna(value):
            return None
        return float(value)

    @property
    def c_index_test(self) -> Optional[float]:
        """Primary test set C-index (Harrell's)."""
        return self._get_concordance_metric('c_index_test', 'c_index_test')

    @property
    def c_index_train(self) -> Optional[float]:
        """Training set C-index (Harrell's)."""
        return self._get_concordance_metric('c_index_train', 'c_index_train')

    @property
    def survival_probabilities(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Per-model survival probability matrices.

        Returns
        -------
        dict of {str: DataFrame} or None
            Each key is a model name; each DataFrame has rows = samples
            (integer index 0..n-1) and columns = evaluation time points
            (float). Values are survival probabilities in [0, 1].
            Returns ``None`` only when no cache exists at all (e.g. survival
            functions were never computed). Returns an empty ``{}`` when a
            cache exists but holds no usable probabilities (e.g. the model
            does not support ``predict_survival_function``) — this keeps
            "never computed" distinguishable from "computed, none usable".
        """
        if self._surv_probs_cache is None:
            return None
        result = {}
        for model_name, (probs, eval_times) in self._surv_probs_cache.items():
            if probs is None:
                continue
            probs_arr = np.asarray(probs)
            times_arr = np.asarray(eval_times)
            result[model_name] = pd.DataFrame(
                probs_arr, columns=times_arr,
            )
        return result

    _EXPORT_FIELDS = [
        ("concordance", "metrics"),
        ("test_set", "predictions"),
        ("auc_t", "auc_over_time"),
        ("importance", "importance"),
        ("importance_iterations", "importance_iterations"),
        ("importance_per_column", "importance_per_column"),
        ("shap_values", "shap_values"),
        ("shap_values_per_column", "shap_values_per_column"),
        ("interactions", "interactions"),
    ]

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Return all stored DataFrames as a flat dictionary.

        Keys use broom-inspired snake_case names suitable as Excel sheet
        names or file names.  ``None`` and empty DataFrames are excluded.

        Returns
        -------
        dict
            Mapping of descriptive key to DataFrame.
        """
        result = _collect_export_fields(self, self._EXPORT_FIELDS)
        sp_df = _surv_probs_to_export_df(self._surv_probs_cache)
        if sp_df is not None:
            result['survival_probabilities'] = sp_df
        if self.bootstrap is not None:
            for key, df in self.bootstrap.to_dataframes().items():
                result[f'bootstrap_{key}'] = df
        return result

    def _get_metric_with_bootstrap_fallback(
        self, concordance_col: str, bootstrap_metric: str, prop_name: str
    ) -> Optional[float]:
        """Get metric from concordance, falling back to bootstrap point estimate or mean."""
        value = self._get_concordance_metric(concordance_col, prop_name)
        if value is not None:
            return value
        if self.bootstrap is not None and self.bootstrap.summary is not None:
            mask = self.bootstrap.summary['metric'] == bootstrap_metric
            if mask.any():
                return _get_point_or_mean(self.bootstrap.summary.loc[mask].iloc[0])
        return None

    @property
    def ibs_test(self) -> Optional[float]:
        """Integrated Brier Score on test set."""
        return self._get_metric_with_bootstrap_fallback('integrated_brier_test', 'integrated_brier', 'ibs_test')

    @property
    def ctd_test(self) -> Optional[float]:
        """Antolini's time-dependent C-index on test set."""
        return self._get_metric_with_bootstrap_fallback('c_index_td_test', 'ctd', 'ctd_test')

    @property
    def cal_slope_test(self) -> Optional[float]:
        """Calibration slope on test set (ideal = 1.0)."""
        return self._get_metric_with_bootstrap_fallback('cal_slope_test', 'cal_slope', 'cal_slope_test')

    @property
    def cal_intercept_test(self) -> Optional[float]:
        """Calibration intercept on test set (ideal = 0.0)."""
        return self._get_metric_with_bootstrap_fallback('cal_intercept_test', 'cal_intercept', 'cal_intercept_test')

    @property
    def gnd_test(self) -> Optional[float]:
        """GND calibration test p-value on test set."""
        return self._get_metric_with_bootstrap_fallback('gnd_test', 'gnd', 'gnd_test')

    def _summary_dict(self) -> dict:
        """Headline scalars: model identity plus single-model test metrics.

        Scalar metrics are emitted only for a single model; multi-model
        per-model values live in the full ``tables['metrics']`` (concordance)
        dump and reading the scalar properties here would warn.
        """
        models = _get_model_names(self.concordance)
        out: Dict[str, Any] = {'n_models': len(models), 'models': models}
        if len(models) == 1:
            out['c_index_test'] = _nan_safe(self.c_index_test)
            out['c_index_train'] = _nan_safe(self.c_index_train)
            out['ibs_test'] = _nan_safe(self.ibs_test)
            out['ctd_test'] = _nan_safe(self.ctd_test)
        return out

    def print_metrics(self, ci_level: Optional[float] = None, precision: int = 3) -> None:
        """Print metrics table to console.

        Parameters
        ----------
        ci_level : float, optional
            Confidence level used to label bootstrap intervals. Defaults to
            the level the intervals were computed at
            (``self.bootstrap.ci_level``). Ignored when no bootstrap is
            attached.
        precision : int, default=3
            Decimal precision for formatting.
        """
        from mlsurv.results._containers_printing import print_eval_metrics
        if ci_level is None:
            ci_level = self.bootstrap.ci_level if self.bootstrap is not None else 0.95
        print_eval_metrics(self, ci_level, precision)


@dataclass
class BootstrapResult(DictExportMixin):
    """
    Bootstrap evaluation results with configurable outputs.

    All DataFrames include `model_type` column for consistent filtering
    and multi-model comparison, even when bootstrapping a single model.

    Attributes
    ----------
    summary : DataFrame
        Columns: [model_type, metric, point, mean, std, ci_lower, ci_upper]
    iterations : DataFrame
        Columns: [model_type, iteration, metric_name, value]
    predictions : DataFrame, optional
        Columns: [model_type, iteration, index, risk_score]
    auc_t : DataFrame, optional
        Columns: [model_type, iteration, eval_time, auc]
    brier_t : DataFrame, optional
        Per-time Brier(t) bootstrap iterations for time-resolved CI ribbons,
        the Brier counterpart of ``auc_t``. Columns: [model_type, iteration,
        eval_time, brier]. Collected alongside ``auc_t`` when the learner
        requests time-resolved data; reflects the deployed model's S(t|x)
        (calibrated when a calibrator is attached). ``None`` when not collected.
    subpopulation_auc_t : DataFrame, optional
        Per-subpopulation AUC(t) bootstrap iterations for CI ribbons.
        Columns: [subpopulation, model_type, iteration, eval_time, auc]
    importance : DataFrame, optional
        Columns: [model_type, iteration, feature, importance]
    lower_is_better : set, optional
        Set of metric names where lower values indicate better performance.
        Default includes 'integrated_brier'.
    ci_level : float, default=0.95
        Confidence level used to compute the summary's `ci_lower` and
        `ci_upper` bounds (the `ci` argument given to bootstrap()).
    run_config : dict, optional
        Resolved replay parameters captured when ``bootstrap()`` ran
        (``n_bootstrap`` as a concrete int, ``random_state``, ``ci``,
        ``metrics``, ``stratify``, ``eval_times``, ``ipcw_tau``,
        ``ipcw_g_hat_source``). Lets the calibrated bootstrap replay reproduce
        the identical resamples so recalibrated bootstrap CIs are paired with
        the raw ones. ``None`` for results from older sessions.
    """
    summary: pd.DataFrame
    iterations: pd.DataFrame
    predictions: Optional[pd.DataFrame] = None
    auc_t: Optional[pd.DataFrame] = None
    brier_t: Optional[pd.DataFrame] = None
    subpopulation_auc_t: Optional[pd.DataFrame] = None
    importance: Optional[pd.DataFrame] = None
    lower_is_better: Optional[Set[str]] = None
    ci_level: float = 0.95
    run_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Stamp the CI level on the summary DataFrame so display helpers
        # that only receive the DataFrame (e.g. format_bootstrap_summary)
        # can label the CI column correctly.
        if self.summary is not None:  # combine() may pass summary=None
            self.summary.attrs['ci_level'] = self.ci_level

    def __setstate__(self, state):
        import dataclasses as _dc
        self.__dict__.update(state)
        # Fill defaults for fields added in later versions
        for f in _dc.fields(self):
            if f.name not in self.__dict__:
                if f.default is not _dc.MISSING:
                    self.__dict__[f.name] = f.default
                elif f.default_factory is not _dc.MISSING:
                    self.__dict__[f.name] = f.default_factory()
        # Re-stamp attrs (old pickles lack them; __post_init__ doesn't run here)
        if self.summary is not None:
            self.summary.attrs['ci_level'] = self.ci_level

    @classmethod
    def combine(cls, models: Dict[str, 'ModelResult']) -> 'Optional[BootstrapResult]':
        """Combine per-model bootstrap results into a multi-model view.

        Parameters
        ----------
        models : dict of {str: ModelResult}
            Models that may have ``.bootstrap`` or ``.evaluation.bootstrap``.

        Returns
        -------
        BootstrapResult or None
            Combined multi-model bootstrap, or None if no models have bootstrap.
        """
        boots = []
        for mr in models.values():
            if mr.bootstrap is not None:
                boots.append(mr.bootstrap)
            elif mr.evaluation is not None and mr.evaluation.bootstrap is not None:
                boots.append(mr.evaluation.bootstrap)
        if not boots:
            return None

        def _concat_opt(attr):
            parts = [getattr(b, attr) for b in boots if getattr(b, attr, None) is not None]
            return pd.concat(parts, ignore_index=True) if parts else None

        # Merge lower_is_better as union across all bootstraps. Stay None when
        # no bootstrap sets it, but preserve an explicit empty set() (meaning
        # "no lower-is-better metrics") rather than collapsing it back to None.
        merged_lib = None
        for b in boots:
            if b.lower_is_better is not None:
                merged_lib = (merged_lib or set()) | b.lower_is_better

        summary = _concat_opt('summary')
        iterations = _concat_opt('iterations')
        # If both required fields are None, treat as no bootstrap data
        if summary is None and iterations is None:
            return None
        # Per-model bootstraps normally share one bootstrap() call's ci
        # level; warn when they differ, because the combined view can only
        # carry (and label with) a single level — the first model's.
        levels = {b.ci_level for b in boots}
        if len(levels) > 1:
            from mlsurv._warnings import MethodologicalWarning
            warnings.warn(
                f"Combining bootstrap results computed at different CI levels "
                f"{sorted(levels)}; the combined summary is labeled at "
                f"{boots[0].ci_level:g} but each model's ci_lower/ci_upper "
                f"stay at its own level. Re-run bootstrap() with one ci for "
                f"comparable intervals.",
                MethodologicalWarning,
                stacklevel=2,
            )
        return cls(
            summary=summary,
            iterations=iterations,
            predictions=_concat_opt('predictions'),
            auc_t=_concat_opt('auc_t'),
            brier_t=_concat_opt('brier_t'),
            subpopulation_auc_t=_concat_opt('subpopulation_auc_t'),
            importance=_concat_opt('importance'),
            lower_is_better=merged_lib,
            ci_level=boots[0].ci_level,
        )

    def __repr__(self) -> str:
        """String representation showing bootstrap summary."""
        models = _get_model_names(self.summary)
        n_models = len(models)
        n_bootstrap = (
            len(self.iterations['iteration'].unique())
            if self.iterations is not None and not self.iterations.empty and 'iteration' in self.iterations.columns
            else 0
        )
        metrics = (
            self.summary['metric'].unique()
            if self.summary is not None and 'metric' in self.summary.columns
            else []
        )

        if n_models == 0:
            return "BootstrapResult(no models)"
        elif n_models == 1:
            return f"BootstrapResult(model='{models[0]}', n_bootstrap={n_bootstrap}, metrics={list(metrics)})"
        else:
            return f"BootstrapResult({n_models} models, n_bootstrap={n_bootstrap}, metrics={list(metrics)})"

    def get_distribution(self, metric: str, model_type: str = None) -> np.ndarray:
        """
        Get raw bootstrap distribution for a specific metric.

        Parameters
        ----------
        metric : str
            Metric name (e.g., 'harrell').
        model_type : str, optional
            Filter by model type. If None and only one model_type exists,
            uses that model. If None and multiple model_types exist, raises error.

        Returns
        -------
        np.ndarray
            Bootstrap values for the metric.

        Raises
        ------
        ValueError
            If `model_type` is not specified but the result contains
            multiple model types.
        """
        df = self.iterations
        if model_type is not None:
            df = df[df['model_type'] == model_type]
        else:
            unique_models = df['model_type'].unique()
            if len(unique_models) > 1:
                raise ValueError(
                    f"Multiple model_types found: {list(unique_models)}. "
                    "Specify model_type parameter."
                )
        mask = df['metric_name'] == metric
        return df.loc[mask, 'value'].values

    def compute_ci(
        self,
        metric: str,
        ci: float = 0.95,
        method: str = 'percentile',
        model_type: str = None
    ) -> tuple:
        """
        Compute confidence interval for a metric.

        Parameters
        ----------
        metric : str
            Metric name (e.g., 'harrell', 'ipcw', 'ctd').
        ci : float
            Confidence level (default 0.95).
        method : str
            CI method: 'percentile', 'basic'.
        model_type : str, optional
            Filter by model type (required if multiple models in result).

        Returns
        -------
        tuple
            (ci_lower, ci_upper)

        Raises
        ------
        ValueError
            If `method` is not a recognized CI method, or if `model_type`
            is required but not specified (propagated from get_distribution).
        """
        dist = self.get_distribution(metric, model_type=model_type)
        if method == 'percentile':
            return percentile_ci(dist, ci_level=ci)
        elif method == 'basic':
            alpha = 1 - ci
            theta_hat = np.mean(dist)
            return (
                2 * theta_hat - np.percentile(dist, 100 * (1 - alpha / 2)),
                2 * theta_hat - np.percentile(dist, 100 * alpha / 2)
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def print_summary(self, ci_level=None, precision=3, point_estimates=None):
        """Print bootstrap metrics table to console (delegate).

        Parameters
        ----------
        ci_level : float, optional
            Confidence level used to label the intervals. Defaults to the
            level the intervals were computed at (``self.ci_level``).
            Passing a different level relabels the precomputed bounds; it
            does not recompute them.
        precision : int, default=3
            Decimal precision for formatting.
        point_estimates : dict, optional
            External point estimates that take priority over the 'point'
            column in the summary DataFrame.
        """
        from mlsurv.results._containers_printing import print_bootstrap_summary
        if ci_level is None:
            ci_level = self.ci_level
        print_bootstrap_summary(self, ci_level, precision, point_estimates)

    def print_comparison(self, metric="harrell", ci=0.95, p_adjust="holm", precision=3):
        """Print and return paired model comparison table (delegate)."""
        from mlsurv.results._containers_printing import print_bootstrap_comparison
        return print_bootstrap_comparison(self, metric, ci, p_adjust, precision)

    def compare(
        self,
        metric: str = "harrell",
        models=None,
        ci: float = 0.95,
        p_adjust: str = "holm",
        eval_time: float = None,
        n_bootstrap: int = None,
        random_state: int = None,
    ) -> pd.DataFrame:
        """Compare models using paired bootstrap distributions.

        Parameters
        ----------
        metric : str, default="harrell"
            Metric to compare. Use "mean_auc" for average tdAUC or
            "auc_t" with ``eval_time`` for a specific timepoint.
        models : sequence of str, optional
            Model names to compare. If None, uses all models.
        ci : float, default=0.95
            Confidence level for the difference CI.
        p_adjust : str, default="holm"
            Multiplicity correction: "holm", "bh", "bonferroni", "none".
        eval_time : float, optional
            Required when ``metric="auc_t"``.
        n_bootstrap : int, optional
            Number of bootstrap iterations for permutation p-values.
        random_state : int, optional
            Random seed for reproducibility of permutation p-values.

        Returns
        -------
        pd.DataFrame
            Columns: model_a, model_b, metric, method, delta, ci_lower,
            ci_upper, p_value, p_adjusted.
        """
        from mlsurv.evaluation._paired_comparison import compare_bootstrap
        return compare_bootstrap(
            self, metric=metric, models=models, ci=ci,
            p_adjust=p_adjust, eval_time=eval_time,
            n_bootstrap=n_bootstrap, random_state=random_state,
        )

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Return all stored DataFrames as a flat dictionary.

        Per-subpopulation metrics (``*__subpopulation_*``) are excluded from
        the summary and iterations tables — they are exported via
        dedicated subpopulation sheets instead.

        Returns
        -------
        dict
            Mapping of descriptive key to DataFrame.
        """
        result: Dict[str, pd.DataFrame] = {}
        if _is_nonempty_df(self.summary):
            mask = ~self.summary['metric'].str.contains('__subpopulation_', na=False)
            filtered = self.summary[mask]
            if not filtered.empty:
                result['summary'] = filtered
        if _is_nonempty_df(self.iterations):
            mask = ~self.iterations['metric_name'].str.contains('__subpopulation_', na=False)
            filtered = self.iterations[mask]
            if not filtered.empty:
                result['iterations'] = filtered
        if _is_nonempty_df(self.predictions):
            result['predictions'] = self.predictions
        if _is_nonempty_df(self.auc_t):
            result['auc_over_time'] = self.auc_t
        if _is_nonempty_df(self.subpopulation_auc_t):
            result['subpopulation_auc_over_time'] = self.subpopulation_auc_t
        if _is_nonempty_df(self.importance):
            result['importance'] = self.importance
        return result

    def _summary_dict(self) -> dict:
        """Headline scalars mirroring ``__repr__``: models, iterations, metrics."""
        from mlsurv.results._containers_helpers import _count_bootstrap_iterations
        models = _get_model_names(self.summary)
        return {
            'n_models': len(models),
            'models': models,
            'n_bootstrap': _count_bootstrap_iterations(self.iterations),
            'ci_level': self.ci_level,
            'metrics': _get_model_names(self.summary, col='metric'),
        }

    def importance_summary(self, model_type=None, ci=0.95, p_adjust="bh"):
        """Summarize permutation importance with empirical p-values (delegate)."""
        from mlsurv.results._containers_extraction import importance_summary
        return importance_summary(self, model_type, ci, p_adjust)


@dataclass
class ModelResult(DictExportMixin):
    """
    Results for a single trained model.

    Attributes
    ----------
    name : str
        Storage / display name and algorithm identity (e.g., 'rsf'). Used as
        the key in ``learner._state.models`` and shown in user-facing output.
    base_name : str
        Read-only algorithm identity for registry / factory / metadata
        lookups. Equals ``name`` (storage keys are plain algorithm names).
    model : Pipeline
        Fitted sklearn Pipeline containing preprocessing steps and the
        survival model.
    coefficients : pd.DataFrame
        Read-only computed property: model coefficients with gated
        multivariable Cox inference (HR/CI/p-value only for models whose
        ``hazard_ratio_valid`` flag is True). See the property docstring.
    cv_metric_scores : np.ndarray
        Cross-validation scores per fold for the primary metric.
    cv_metric_mean : float
        Mean of cv_metric_scores.
    cv_metric_std : float
        Standard deviation of cv_metric_scores.
    cv_metrics : dict, optional
        Multi-metric CV results. Dict of {metric_name: {'scores': np.ndarray,
        'mean': float, 'std': float, 'ci_lower': float, 'ci_upper': float}}.
    evaluation : EvaluationResult, optional
        Results from learner.evaluate().
    bootstrap : BootstrapResult, optional
        Results from learner.bootstrap().
    optuna_study : Study, optional
        Optuna study object (if tuned).
    best_params_ : dict, optional
        Best hyperparameters found during tuning.
    scoring_metric : str, optional
        Metric used for hyperparameter optimization.
    cv_ci_lower : float
        Lower bound of t-distribution CI for the primary CV metric.
    cv_ci_upper : float
        Upper bound of t-distribution CI for the primary CV metric.
    cv_ci_level : float
        Confidence level used for CV CIs (e.g. 0.95).
    is_tuned : bool
        Whether this model was created via tune() (True) or train() (False).
    calibration_flagged : bool
        True if this model's evaluation tripped the calibration-drift
        thresholds. Set by ``evaluate()`` when the per-model concordance is
        stored; read via the ``learner.calibration_flagged_models`` property.
    calibrator : SurvivalCalibrator, optional
        Internal S(t|x) calibrator attached by ``train(calibrate=...)``
        (``None`` = uncalibrated). Fit on held-out OOF of the training cohort
        and applied wherever the deployed model produces S(t|x). Rides on the
        result so it dies with the model on overwrite.
    pimp_importance_test : pd.DataFrame, optional
        Per-model PIMP significance table from
        ``importance_test(method='pimp')`` (``None`` before it runs). Read via
        ``learner.pimp_importance_test_results[name]``.
    bootstrap_importance_test : pd.DataFrame, optional
        Per-model bootstrap significance table from
        ``importance_test(method='bootstrap')``. Read via
        ``learner.bootstrap_importance_test_results[name]``.
    rmst_test : pd.DataFrame, optional
        Per-model risk-group RMST table from ``rmst_test()``. Read via
        ``learner.rmst_test_results[name]``.
    logrank_test : pd.DataFrame, optional
        Per-model risk-group log-rank table from ``logrank_test()``.
        Read via ``learner.logrank_test_results[name]``.
    """
    name: str
    model: Any  # BaseSurvivalModel (forward reference)
    cv_metric_scores: np.ndarray
    cv_metric_mean: float
    cv_metric_std: float
    cv_metrics: Optional[Dict[str, Dict[str, Any]]] = None
    evaluation: Optional[EvaluationResult] = None
    bootstrap: Optional[BootstrapResult] = None
    optuna_study: Optional['optuna.Study'] = None
    best_params_: Optional[dict] = None
    scoring_metric: Optional[str] = None
    cv_ci_lower: float = np.nan
    cv_ci_upper: float = np.nan
    cv_ci_level: float = 0.95
    is_tuned: bool = False
    is_synthetic: bool = False
    calibration_flagged: bool = False
    # Internal S(t|x) calibrator attached by train(calibrate=...). A
    # SurvivalCalibrator (isotonic / lowdof / breslow_slope) fit on held-out
    # OOF of the training cohort. None = uncalibrated. Applied wherever the
    # deployed model produces S(t|x). Pickles cleanly with the ModelResult.
    # (Typed Any to avoid a results->evaluation import cycle.)
    calibrator: Optional[Any] = None
    # Pre-calibration test calibration slope, stored by evaluate() so the report
    # can show the before/after when a calibrator is attached. None otherwise.
    precalibration_slope: Optional[float] = None
    pimp_importance_test: Optional[pd.DataFrame] = None
    bootstrap_importance_test: Optional[pd.DataFrame] = None
    rmst_test: Optional[pd.DataFrame] = None
    logrank_test: Optional[pd.DataFrame] = None

    @property
    def base_name(self) -> str:
        """Algorithm identity for registry / factory / metadata lookups.

        Equals ``name``. Storage keys are plain algorithm names, so the
        algorithm identity is just the name itself.
        """
        return self.name

    @property
    def coefficients(self) -> pd.DataFrame:
        """Model coefficients with gated multivariable Cox inference.

        Computed live from the fitted pipeline (``self.model``), so it
        always reflects the current train/tune state.

        Columns: ``feature``, ``coefficient``, ``hazard_ratio``,
        ``hr_lower_95``, ``hr_upper_95``, ``p_value`` (Wald), in pipeline
        feature order (not p-sorted, unlike ``univariate_cox``).
        Selector-dropped features appear with NaN in every value column.

        ``hazard_ratio``, the CI bounds, and ``p_value`` are populated
        only when the model's ``hazard_ratio_valid`` flag is True
        (currently CoxPH): penalized CoxNet and ranking FSSVM
        coefficients are not log-hazards with valid inference, and AFT
        coefficients are not log-hazards at all, so those columns stay
        NaN for them. For CoxPH on a rank-deficient design (singular
        Hessian), ``coefficient`` and ``hazard_ratio`` are still
        populated while CI/p-value are NaN. Models with no coefficients
        (tree/neural) yield a 0-row DataFrame, and so does a model whose
        coefficients could not be placed on their features, which warns
        with ``MetricFailureWarning``.

        Inference caveats: p-values and CIs condition on any data-driven
        feature selection (``topk``/``coxlasso``/``coxnet`` selectors) and
        tuning that produced this model, so they are optimistic relative
        to a pre-specified model (``univariate_cox`` avoids this). The
        multivariable fit uses the training split, while
        ``univariate_cox`` uses the full cohort — the two tables are not
        strictly comparable. When CoxPH's automatic ridge fallback
        engaged at its first rung (``alpha_ == 0.01``), CIs/p-values are
        conditional on that regularization strength (same convention as
        the forest plots). If the fallback had to escalate further, the
        hazard-ratio, CI, and p-value columns are NaN rather than
        conditional: see ``CoxPHModel.hazard_ratio_valid``.

        See Also
        --------
        SurvivalLearner.coefficients : long format across all models.
        SurvivalLearner.univariate_cox : per-feature marginal statistics.
        EvaluationResult.importance : permutation importance alongside
            coefficients.
        """
        from mlsurv.evaluation._coefficients import compute_model_coefficients
        return compute_model_coefficients(self.model, model_name=self.name)

    def __getstate__(self):
        state = self.__dict__.copy()
        study = state.get('optuna_study')
        if study is not None and hasattr(study, 'trials'):
            try:
                meta = {
                    'study_name': study.study_name,
                    'best_value': study.best_value,
                    'best_params': study.best_params,
                    'n_trials': len(study.trials),
                }
            except ValueError:
                # No completed trials — best_value/best_params raise
                meta = {
                    'study_name': getattr(study, 'study_name', None),
                    'n_trials': len(study.trials),
                }
            state['_optuna_study_meta'] = meta

            # Pre-extract trial data for visualization after deserialization.
            # Stored as simple pickle-safe types (list-of-dicts, dict).
            history_rows = []
            for trial in study.trials:
                if trial.state.name != 'COMPLETE':
                    continue
                row = {
                    'trial': trial.number,
                    'value': trial.value,
                    'state': 'COMPLETE',
                }
                row.update(trial.params)
                history_rows.append(row)
            state['_optuna_tuning_history'] = history_rows

            try:
                from optuna.importance import get_param_importances
                state['_optuna_param_importances'] = dict(
                    get_param_importances(study)
                )
            except (ImportError, RuntimeError, ValueError):
                state['_optuna_param_importances'] = None

            state['optuna_study'] = None
        return state

    def __setstate__(self, state):
        import dataclasses as _dc
        self.__dict__.update(state)
        # Fill defaults for fields added in later versions
        for f in _dc.fields(self):
            if f.name not in self.__dict__:
                if f.default is not _dc.MISSING:
                    self.__dict__[f.name] = f.default
                elif f.default_factory is not _dc.MISSING:
                    self.__dict__[f.name] = f.default_factory()

    @classmethod
    def from_artifact(cls, metadata: dict, model) -> 'ModelResult':
        """Construct a ModelResult from saved artifact metadata + loaded model.

        Parameters
        ----------
        metadata : dict
            Artifact metadata (from ``artifact.json``).
        model : object
            Deserialized fitted model.

        Returns
        -------
        ModelResult
            A result populated with the artifact's stored CV metrics.

        Notes
        -----
        Handles incomplete metadata gracefully for backwards compatibility.
        """
        results = metadata.get('results', {})
        if not isinstance(results, dict):
            results = {}
        config = metadata.get('config', {})
        if not isinstance(config, dict):
            config = {}
        cv_scores_raw = results.get('cv_metric_scores')
        cv_scores = (
            np.array(cv_scores_raw, dtype=float)
            if cv_scores_raw is not None
            else np.array([])
        )
        cv_mean = results.get('cv_metric_mean')
        cv_std = results.get('cv_metric_std')

        return cls(
            name=metadata.get('model_name') or 'unknown',
            model=model,
            cv_metric_scores=cv_scores,
            cv_metric_mean=float(cv_mean) if cv_mean is not None else float('nan'),
            cv_metric_std=float(cv_std) if cv_std is not None else float('nan'),
            is_tuned=results.get('is_tuned', False),
            best_params_=results.get('best_params'),
            scoring_metric=results.get('scoring_metric') or config.get('scoring'),
        )

    def __repr__(self) -> str:
        """String representation showing model name and CV performance."""
        tuned_str = " (tuned)" if self.is_tuned else ""
        cv_str = f"cv_mean={_fmt_float(self.cv_metric_mean)}±{_fmt_float(self.cv_metric_std)}"

        if self.evaluation:
            c_test = self.evaluation.c_index_test
            if c_test is not None and not pd.isna(c_test):
                return f"ModelResult(name='{self.name}'{tuned_str}, {cv_str}, c_index_test={_fmt_float(c_test)})"

        return f"ModelResult(name='{self.name}'{tuned_str}, {cv_str})"

    def get_metric(
        self,
        metric: str = DEFAULT_SCORING_METRIC,
        phase: str = 'cv',
        stat: str = 'mean'
    ) -> Union[float, np.ndarray, None]:
        """Unified metric access across all evaluation phases.

        Parameters
        ----------
        metric : str, default=DEFAULT_SCORING_METRIC
            Metric name: 'harrell', 'ipcw', 'integrated_brier', or 'ctd'.
        phase : str, default='cv'
            Evaluation phase: 'cv', 'test', or 'bootstrap'.
        stat : str, default='mean'
            Statistic to return:
            - 'mean': Mean value (all phases)
            - 'std': Standard deviation (cv, bootstrap)
            - 'scores': Raw fold/iteration values (cv, bootstrap)
            - 'point': Point estimate from full test set (bootstrap only)
            - 'ci_lower': Lower confidence bound (cv, bootstrap)
            - 'ci_upper': Upper confidence bound (cv, bootstrap)

        Returns
        -------
        float, np.ndarray, or None
            Requested metric value, or None if not available.
        """
        if phase == 'cv':
            return self._get_cv_metric(metric, stat)
        elif phase == 'test':
            return self._get_test_metric(metric, stat)
        elif phase == 'bootstrap':
            return self._get_bootstrap_metric(metric, stat)
        else:
            warnings.warn(f"Unknown phase '{phase}'. Use 'cv', 'test', or 'bootstrap'.", InternalWarning)
            return None

    def _get_cv_metric(self, metric: str, stat: str) -> Union[float, np.ndarray, None]:
        """Get CV metric value."""
        _VALID_CV_STATS = ('mean', 'std', 'scores', 'ci_lower', 'ci_upper')

        # Build a unified data dict from cv_metrics or harrell fallback
        if self.cv_metrics and metric in self.cv_metrics:
            data = self.cv_metrics[metric]
        elif metric == 'harrell':
            data = {
                'mean': self.cv_metric_mean,
                'std': self.cv_metric_std,
                'scores': self.cv_metric_scores,
                'ci_lower': self.cv_ci_lower,
                'ci_upper': self.cv_ci_upper,
            }
        else:
            warnings.warn(f"Metric '{metric}' not found in CV results.", InternalWarning)
            return None

        if stat == 'scores':
            return data.get('scores')
        if stat in ('mean', 'std', 'ci_lower', 'ci_upper'):
            return float(data.get(stat, np.nan))

        warnings.warn(f"Unknown stat '{stat}' for CV phase. Use {_VALID_CV_STATS}.", InternalWarning)
        return None

    def _get_test_metric(self, metric: str, stat: str) -> Union[float, None]:
        """Get test set metric value.

        Looks up the metric in the concordance DataFrame, falling back
        to bootstrap point estimate (or mean) when the column is absent.
        """
        if stat != 'mean':
            warnings.warn(f"Test phase only supports 'mean' stat, got '{stat}'.", InternalWarning)
            return None

        if not self.evaluation:
            warnings.warn("No evaluation results available. Run evaluate() first.", InternalWarning)
            return None

        from mlsurv.constants import get_concordance_column, METRIC_REGISTRY
        col = get_concordance_column(metric)

        if col is None:
            available = ', '.join(METRIC_REGISTRY.keys())
            warnings.warn(
                f"Metric '{metric}' is not a recognized metric name. "
                f"Available metrics: {available}.",
                InternalWarning,
            )
            return None

        # Look up in concordance DataFrame
        if self.evaluation.concordance is not None and col in self.evaluation.concordance.columns:
            value = self.evaluation.concordance[col].iloc[0]
            if pd.notna(value):
                return float(value)

        # Fall back to bootstrap summary
        boot = self.resolved_bootstrap
        if boot and boot.summary is not None:
            mask = boot.summary['metric'] == metric
            if mask.any():
                return _get_point_or_mean(boot.summary.loc[mask].iloc[0])

        warnings.warn(f"Metric '{metric}' not found in test results.", InternalWarning)
        return None

    @property
    def resolved_bootstrap(self) -> Optional['BootstrapResult']:
        """Resolve bootstrap result, preferring .bootstrap over .evaluation.bootstrap."""
        if self.bootstrap:
            return self.bootstrap
        if self.evaluation and self.evaluation.bootstrap:
            return self.evaluation.bootstrap
        return None

    def _get_bootstrap_metric(self, metric: str, stat: str) -> Union[float, np.ndarray, None]:
        """Get bootstrap metric value."""
        _VALID_BOOT_STATS = ('point', 'mean', 'std', 'ci_lower', 'ci_upper', 'scores')

        boot = self.resolved_bootstrap
        if not boot:
            warnings.warn("No bootstrap results available. Run bootstrap() first.", InternalWarning)
            return None

        summary = boot.summary
        if summary is None or summary.empty:
            warnings.warn("Bootstrap summary is empty.", InternalWarning)
            return None

        mask = summary['metric'] == metric
        if not mask.any():
            warnings.warn(f"Metric '{metric}' not found in bootstrap results.", InternalWarning)
            return None

        row = summary.loc[mask].iloc[0]

        if stat == 'point':
            return _get_point_or_mean(row)
        if stat == 'scores':
            return boot.get_distribution(metric, model_type=self.name)
        if stat in ('mean', 'std', 'ci_lower', 'ci_upper'):
            return float(row[stat])

        warnings.warn(
            f"Unknown stat '{stat}' for bootstrap. Use {_VALID_BOOT_STATS}.",
            InternalWarning,
        )
        return None

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Return all stored DataFrames as a flat dictionary."""
        result: Dict[str, pd.DataFrame] = {}
        result['cv_scores'] = self.cv_results_df()
        result['summary'] = self.summary()
        if self.evaluation is not None:
            for key, df in self.evaluation.to_dataframes().items():
                result[f'eval_{key}'] = df
        elif self.bootstrap is not None:
            for key, df in self.bootstrap.to_dataframes().items():
                result[f'bootstrap_{key}'] = df
        return result

    def cv_results_df(self) -> pd.DataFrame:
        """Get DataFrame with per-fold CV results for all metrics."""
        if not self.cv_metrics:
            # Fall back to primary metric only
            n_folds = len(self.cv_metric_scores)
            return pd.DataFrame({
                'fold': range(n_folds),
                'harrell': self.cv_metric_scores
            })

        # Build DataFrame from cv_metrics
        n_folds = len(next(iter(self.cv_metrics.values()))['scores'])
        data = {'fold': list(range(n_folds))}

        for metric_name, metric_data in self.cv_metrics.items():
            scores = metric_data['scores']
            if len(scores) != n_folds:
                action = "truncating" if len(scores) > n_folds else "padding"
                msg = (
                    f"CV metric '{metric_name}' has {len(scores)} scores "
                    f"but expected {n_folds}; {action} with NaN."
                )
                warnings.warn(msg, InternalWarning)
                logger.warning(msg)
                padded = np.full(n_folds, np.nan)
                n_copy = min(len(scores), n_folds)
                padded[:n_copy] = scores[:n_copy]
                scores = padded
            data[metric_name] = scores

        return pd.DataFrame(data)

    def summary(self, metrics: List[str] = None) -> pd.DataFrame:
        """Get all metrics in a single DataFrame across all phases (delegate)."""
        from mlsurv.results._containers_extraction import model_summary
        return model_summary(self, metrics)

    # Backward-compatible alias; the canonical helper lives at module level.
    _nan_safe = staticmethod(_nan_safe)

    def _summary_dict(self) -> dict:
        """Headline scalars: CV metrics, test C-index, and best params.

        Returned verbatim by ``to_dict(full=False)`` — the historical
        ``to_dict()`` payload, unchanged for backward compatibility.
        """
        _s = _nan_safe
        result = {
            'name': self.name,
            'cv_metric_mean': _s(self.cv_metric_mean),
            'cv_metric_std': _s(self.cv_metric_std),
            'cv_ci_lower': _s(self.cv_ci_lower),
            'cv_ci_upper': _s(self.cv_ci_upper),
            'cv_ci_level': self.cv_ci_level,
            'c_index_train': _s(self.evaluation.c_index_train) if self.evaluation else None,
            'c_index_test': _s(self.evaluation.c_index_test) if self.evaluation else None,
        }
        if self.cv_metrics:
            result['cv_metrics'] = {
                metric: {
                    'mean': _s(data.get('mean')),
                    'std': _s(data.get('std')),
                    'ci_lower': _s(data.get('ci_lower')),
                    'ci_upper': _s(data.get('ci_upper')),
                }
                for metric, data in self.cv_metrics.items()
            }
        if self.best_params_ is not None:
            result['best_params'] = self.best_params_
        elif self.optuna_study:
            try:
                result['best_params'] = self.optuna_study.best_params
            except ValueError:
                logger.debug(
                    "Model '%s': optuna_study has no completed trials; "
                    "omitting best_params from to_dict().",
                    self.name,
                )
        return result


@dataclass
class ValidationResult(DictExportMixin):
    """
    Results from external validation on a single cohort.

    Wraps EvaluationResult with cohort metadata for multi-cohort support.

    Attributes
    ----------
    name : str
        Cohort name (e.g., "TCGA", "External", "MSK").
    evaluation : EvaluationResult
        Full evaluation results including metrics, predictions, bootstrap CIs.
    n_samples : int
        Number of samples in the cohort.
    n_events : int
        Number of events in the cohort.
    X_val : DataFrame, optional
        Feature matrix for the validation cohort.
    y_val : structured array, optional
        Survival outcome for the validation cohort.
    """
    name: str
    evaluation: EvaluationResult
    n_samples: int
    n_events: int
    X_val: Optional[pd.DataFrame] = None
    y_val: "Optional[SurvivalArray]" = None
    subpopulation_results: Optional[Dict[str, Any]] = None
    # Reference biomarkers (LORIS, TMB, PD-L1, ...) scored on
    # this cohort alongside models. Reference rows already appear in
    # ``evaluation.concordance`` (and AUC(t)/Brier where applicable) via
    # ``_aggregate_evaluation_results``; these fields preserve the raw arrays
    # for downstream plots/exports.
    reference_arrays: Optional[Dict[str, "np.ndarray"]] = field(
        default=None, repr=False,
    )
    reference_names: tuple = ()

    def __repr__(self) -> str:
        """String representation showing cohort name and key metrics."""
        c_idx = self.evaluation.c_index_test if self.evaluation else None
        return (
            f"ValidationResult(name='{self.name}', n_samples={self.n_samples}, "
            f"n_events={self.n_events}, c_index={_fmt_float(c_idx)})"
        )

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Return all stored DataFrames as a flat dictionary."""
        result: Dict[str, pd.DataFrame] = {}
        if self.evaluation is not None:
            for key, df in self.evaluation.to_dataframes().items():
                result[key] = df
        result['cohort_info'] = pd.DataFrame([{
            'cohort': self.name,
            'n_samples': self.n_samples,
            'n_events': self.n_events,
            'event_rate': self.event_rate,
        }])
        if self.reference_arrays:
            result['reference_scores'] = pd.DataFrame(self.reference_arrays)
        return result

    def _summary_dict(self) -> dict:
        """Headline scalars: cohort identity, size, and test C-index."""
        return {
            'name': self.name,
            'n_samples': self.n_samples,
            'n_events': self.n_events,
            'event_rate': _nan_safe(self.event_rate),
            'c_index_test': (
                _nan_safe(self.evaluation.c_index_test)
                if self.evaluation is not None else None
            ),
        }

    @property
    def concordance(self) -> Optional[pd.DataFrame]:
        """Concordance DataFrame (convenience accessor)."""
        return self.evaluation.concordance

    @property
    def test_set(self) -> Optional[pd.DataFrame]:
        """Test set predictions (convenience accessor)."""
        return self.evaluation.test_set

    @property
    def bootstrap(self) -> Optional['BootstrapResult']:
        """Bootstrap results (convenience accessor)."""
        return self.evaluation.bootstrap

    @property
    def event_rate(self) -> float:
        """Event rate in the cohort."""
        if self.n_samples > 0:
            return self.n_events / self.n_samples
        return 0.0

# =====================================================================
# Benchmarking
# =====================================================================


@dataclass
class BenchmarkResult(DictExportMixin):
    """Container for benchmarking results.

    Produced by ``SurvivalLearner.benchmark()``.

    Attributes
    ----------
    evaluation : EvaluationResult
        Merged evaluation of both models and references.
    univariate_cox : DataFrame
        Univariate Cox regression for each reference.
    incremental : DataFrame
        Incremental metrics per model x reference pair.
    correlation : DataFrame
        Correlation between model risk scores and each reference.
    subpopulation_incremental : DataFrame or None
        Per-subpopulation incremental metrics.
    reference_names : list[str]
    model_names : list[str]
    sample_sizes : dict[str, int] or None
        Number of observations used for evaluation per entity.
        Models always use the held-out test split; references use the
        selected dataset, with non-finite values excluded.
    dataset : str
        Which data split was used for reference evaluation:
        ``'test'``, ``'train'``, or ``'full'``.  Models always use
        the test split.
    """

    evaluation: 'EvaluationResult'
    univariate_cox: pd.DataFrame
    incremental: pd.DataFrame
    correlation: pd.DataFrame
    reference_names: List[str]
    model_names: List[str]
    subpopulation_incremental: Optional[pd.DataFrame] = None
    sample_sizes: Optional[Dict[str, int]] = None
    dataset: str = "test"
    # Raw test-aligned risk scores retained so figures (e.g.
    # plot_benchmark_scatter) are reproducible from a saved BenchmarkResult.
    # Both dicts are keyed by name and align row-wise; excluded from the
    # repr to keep it readable.
    model_risk_scores: Optional[Dict[str, np.ndarray]] = field(
        default=None, repr=False,
    )
    reference_arrays: Optional[Dict[str, np.ndarray]] = field(
        default=None, repr=False,
    )

    _EXPORT_FIELDS = [
        ("univariate_cox", "univariate_cox"),
        ("incremental", "incremental"),
        ("correlation", "correlation"),
        ("subpopulation_incremental", "subpopulation_incremental"),
    ]

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Return all stored DataFrames as a flat dictionary."""
        result = _collect_export_fields(self, self._EXPORT_FIELDS)
        # Add formatted significance column to p-value-bearing DataFrames
        from mlsurv.evaluation._incremental import format_p_sig
        for key in ("incremental", "subpopulation_incremental",
                     "univariate_cox", "correlation"):
            df = result.get(key)
            if df is None or df.empty:
                continue
            if "p_value" in df.columns:
                df = df.copy()
                df["significance"] = df["p_value"].apply(format_p_sig)
                result[key] = df
            # Correlation has separate p columns
            for pcol in ("spearman_p", "pearson_p"):
                if pcol in df.columns:
                    df = df if df is not result.get(key) else df.copy()
                    df[f"{pcol}_sig"] = df[pcol].apply(format_p_sig)
                    result[key] = df
        if self.evaluation is not None:
            for key, df in self.evaluation.to_dataframes().items():
                result[f'eval_{key}'] = df
        return result

    def _summary_dict(self) -> dict:
        """Headline scalars: model/reference names, dataset, sample sizes."""
        return {
            'model_names': list(self.model_names),
            'reference_names': list(self.reference_names),
            'dataset': self.dataset,
            'sample_sizes': self.sample_sizes,
        }

    def print_summary(self, ci_level: Optional[float] = None) -> None:
        """Print human-readable benchmark summary to stdout (delegate).

        Parameters
        ----------
        ci_level : float, optional
            Confidence level used to label bootstrap intervals. Defaults to
            the level the intervals were computed at (resolved by
            ``EvaluationResult.print_metrics`` from the attached bootstrap).
        """
        from mlsurv.results._containers_printing import print_bench_summary
        print_bench_summary(self, ci_level=ci_level)

    def __repr__(self):
        parts = [
            f"models={self.model_names}",
            f"references={self.reference_names}",
        ]
        if self.dataset != "test":
            parts.append(f"dataset={self.dataset!r}")
        return f"BenchmarkResult({', '.join(parts)})"


# =====================================================================
# Subpopulation analysis
# =====================================================================


@dataclass
class SubpopulationResult:
    """Per-variable subpopulation analysis result.

    Produced by ``compute_subpopulation_metrics()`` in
    ``evaluation/_subpopulation.py``.

    Attributes
    ----------
    group_variable : str
        Name of the grouping variable (e.g., ``'cancer_type'``).
    summary : DataFrame
        Cohort summary per subpopulation level.
    metrics : DataFrame
        Point-estimate metrics per subpopulation x model.
    bootstrap_summary : DataFrame or None
        Bootstrap CIs per subpopulation x model.
    auc_t : DataFrame or None
        Per-subpopulation AUC(t) values.
    auc_t_bootstrap : DataFrame or None
        Per-iteration bootstrap AUC(t) for CI ribbons.
    warnings : list of str
        Diagnostic messages (e.g., small-subpopulation warnings).
    """
    group_variable: str
    summary: pd.DataFrame
    metrics: pd.DataFrame
    bootstrap_summary: Optional[pd.DataFrame] = None
    auc_t: Optional[pd.DataFrame] = None
    auc_t_bootstrap: Optional[pd.DataFrame] = None
    warnings: List[str] = field(default_factory=list)

    def __repr__(self):
        n_groups = len(self.summary) if self.summary is not None else 0
        return (
            f"SubpopulationResult(group_variable='{self.group_variable}', "
            f"n_subpopulations={n_groups})"
        )


@dataclass
class SweepResult(DictExportMixin):
    """Result from :meth:`SurvivalLearner.parameter_sweep`.

    Attributes
    ----------
    data : DataFrame
        Long-format table with columns ``[param_value, model, metric,
        mean, std, ci_lower, ci_upper]``.  One row per
        (param_value, model, metric) combination.
    param_name : str
        Name of the swept parameter (e.g., ``'k'``).
    target : str
        Pipeline component targeted (``'model'``, ``'imputer'``,
        ``'scaler'``, or ``'feature_selector'``).
    """

    data: pd.DataFrame
    param_name: str
    target: str
    scoring: str = "harrell"

    def plot(self, metric: Optional[str] = None, phase: str = "cv",
             ribbon: Optional[str] = None,
             width: str = "double", height: str = "medium") -> "go.Figure":
        """Line plot with ribbon, one line per model.

        Parameters
        ----------
        metric : str, optional
            Which metric to plot.  Defaults to the learner's scoring
            metric (e.g., ``'harrell'``, ``'integrated_brier'``).
        phase : str, default ``'cv'``
            ``'cv'`` for cross-validation means or ``'test'`` for
            held-out test-set point estimates.
        ribbon : str or None
            ``'std'`` for mean ± std, ``'ci'`` for 95% CI, or ``None``
            for the default (``'std'`` for CV, ``'ci'`` for test).
        width : str or int
            Figure width (``'single'``, ``'double'``, or pixel int).
        height : str or int
            Figure height (``'small'``, ``'medium'``, ``'large'``, or pixel int).

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from mlsurv.viz.sweep import plot_sweep

        if metric is None:
            metric = self.scoring
        return plot_sweep(self.data, self.param_name, metric=metric,
                          phase=phase, ribbon=ribbon,
                          width=width, height=height)

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Return all stored DataFrames as a flat dictionary.

        Returns
        -------
        dict
            ``{'sweep': self.data}`` if data is non-empty, else ``{}``.
        """
        if _is_nonempty_df(self.data):
            return {"sweep": self.data}
        return {}

    def _summary_dict(self) -> dict:
        """Headline scalars mirroring ``__repr__``: param, target, models."""
        cols = self.data.columns if _is_nonempty_df(self.data) else []
        has_pv = "param_value" in cols
        has_model = "model" in cols
        return {
            'param_name': self.param_name,
            'target': self.target,
            'scoring': self.scoring,
            'n_values': int(self.data["param_value"].nunique()) if has_pv else 0,
            'models': sorted(self.data["model"].unique().tolist()) if has_model else [],
        }

    def __repr__(self):
        n_values = self.data["param_value"].nunique() if len(self.data) else 0
        models = sorted(self.data["model"].unique()) if len(self.data) else []
        return (
            f"SweepResult(param='{self.param_name}', target='{self.target}', "
            f"n_values={n_values}, models={models})"
        )
