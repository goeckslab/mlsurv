"""Extraction and computation functions for result containers.

Extracted from ``containers.py`` to reduce module complexity.
Public extraction functions (``extract_model_evaluation``,
``_extract_model_bootstrap``) are re-exported by the facade.
Computation functions (``importance_summary``, ``model_summary``)
are accessed via class method delegates.
"""

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

from mlsurv._shared import nan_summary, percentile_ci
from mlsurv.constants import ALL_METRICS
from mlsurv.results._containers_helpers import _filter_model_type

if TYPE_CHECKING:
    from mlsurv.results.containers import (
        BootstrapResult,
        EvaluationResult,
        ModelResult,
    )


# ---------------------------------------------------------------------------
# Model extraction
# ---------------------------------------------------------------------------


def extract_model_evaluation(multi_eval: 'EvaluationResult', model_name: str) -> 'EvaluationResult':
    """Extract single-model EvaluationResult from multi-model result.

    Parameters
    ----------
    multi_eval : EvaluationResult
        Multi-model evaluation result with model_type column in DataFrames.
    model_name : str
        Name of the model to extract.

    Returns
    -------
    EvaluationResult
        Single-model evaluation result filtered to the specified model.
    """
    from mlsurv.results.containers import EvaluationResult

    def f(df):
        return _filter_model_type(df, model_name)

    # Filter interaction_surfaces (list-of-dicts, not DataFrame)
    filtered_surfaces = None
    if multi_eval.interaction_surfaces:
        filtered_surfaces = [
            s for s in multi_eval.interaction_surfaces
            if s.get('model_type') == model_name
        ] or None

    return EvaluationResult(
        concordance=f(multi_eval.concordance),
        test_set=f(multi_eval.test_set),
        auc_t=f(multi_eval.auc_t),
        importance=f(multi_eval.importance),
        importance_iterations=f(multi_eval.importance_iterations),
        importance_per_column=f(multi_eval.importance_per_column),
        shap_values=f(multi_eval.shap_values),
        shap_values_per_column=f(multi_eval.shap_values_per_column),
        interactions=f(multi_eval.interactions),
        interaction_surfaces=filtered_surfaces,
        interaction_values_raw=(
            multi_eval._interaction_raw_cache[model_name][0]
            if multi_eval._interaction_raw_cache and model_name in multi_eval._interaction_raw_cache
            else multi_eval.interaction_values_raw
        ),
        interaction_feature_names=(
            multi_eval._interaction_raw_cache[model_name][1]
            if multi_eval._interaction_raw_cache and model_name in multi_eval._interaction_raw_cache
            else multi_eval.interaction_feature_names
        ),
        interaction_method=multi_eval.interaction_method,
        bootstrap=_extract_model_bootstrap(multi_eval.bootstrap, model_name) if multi_eval.bootstrap else None,
        _surv_probs_cache=(
            {model_name: multi_eval._surv_probs_cache[model_name]}
            if multi_eval._surv_probs_cache and model_name in multi_eval._surv_probs_cache
            else None
        ),
        _ibs_surv_probs_cache=(
            {model_name: multi_eval._ibs_surv_probs_cache[model_name]}
            if multi_eval._ibs_surv_probs_cache and model_name in multi_eval._ibs_surv_probs_cache
            else None
        ),
        _interaction_raw_cache=(
            {model_name: multi_eval._interaction_raw_cache[model_name]}
            if multi_eval._interaction_raw_cache and model_name in multi_eval._interaction_raw_cache
            else None
        ),
        # Truncation diagnostic is cohort-scoped (shared across all models
        # that share y_test) — forward it so the report footnote renders
        # for multi-model evaluates too.
        truncation_t_max=multi_eval.truncation_t_max,
        truncation_n_beyond=multi_eval.truncation_n_beyond,
    )


def _extract_model_bootstrap(multi_boot: 'BootstrapResult', model_name: str) -> 'BootstrapResult':
    """Extract single-model BootstrapResult from multi-model result.

    Parameters
    ----------
    multi_boot : BootstrapResult
        Multi-model bootstrap result with model_type column in DataFrames.
    model_name : str
        Name of the model to extract.

    Returns
    -------
    BootstrapResult
        Single-model bootstrap result filtered to the specified model.
    """
    from mlsurv.results.containers import BootstrapResult

    def f(df):
        return _filter_model_type(df, model_name)

    return BootstrapResult(
        summary=f(multi_boot.summary),
        iterations=f(multi_boot.iterations),
        predictions=f(multi_boot.predictions),
        auc_t=f(multi_boot.auc_t),
        brier_t=f(multi_boot.brier_t),
        subpopulation_auc_t=f(multi_boot.subpopulation_auc_t),
        importance=f(multi_boot.importance),
        lower_is_better=multi_boot.lower_is_better,
        ci_level=multi_boot.ci_level,
        run_config=multi_boot.run_config,
    )


# ---------------------------------------------------------------------------
# BootstrapResult computation
# ---------------------------------------------------------------------------


def importance_summary(
    boot_result: 'BootstrapResult',
    model_type: Optional[str] = None,
    ci: float = 0.95,
    p_adjust: str = "bh",
) -> pd.DataFrame:
    """Summarize permutation importance with empirical p-values.

    Computes per-feature statistics from the bootstrap importance
    distributions, including an empirical one-tailed p-value testing
    whether each feature has positive importance.

    Parameters
    ----------
    boot_result : BootstrapResult
        Bootstrap result containing importance data.
    model_type : str, optional
        Filter by model type. If None and only one model exists,
        uses that model. If None and multiple models exist, raises.
    ci : float, default=0.95
        Confidence level for percentile intervals.
    p_adjust : str, default="bh"
        Multiplicity correction: "bh" (Benjamini-Hochberg FDR),
        "holm" (FWER), "bonferroni", or "none".

    Returns
    -------
    pd.DataFrame
        Columns: feature, mean, std, ci_lower, ci_upper, p_value,
        p_adjusted. Sorted by p_value ascending.

    Raises
    ------
    ValueError
        If importance data is not available or model_type is ambiguous.

    See Also
    --------
    SurvivalLearner.importance_test : Unified interface supporting both
        PIMP (Altmann et al. 2010) and bootstrap methods. PIMP provides a
        stronger null hypothesis (feature-response independence) and is
        recommended for definitive feature selection.
    """
    if boot_result.importance is None or boot_result.importance.empty:
        raise ValueError(
            "No importance data available. Re-run bootstrap() with "
            "importance=True."
        )
    if not 0 < ci < 1:
        raise ValueError(
            f"ci must be between 0 and 1 (exclusive), got {ci}. "
            f"Use ci=0.95 for a 95% confidence interval."
        )

    df = boot_result.importance
    if model_type is not None:
        df = df[df["model_type"] == model_type]
        if df.empty:
            available = list(boot_result.importance["model_type"].unique())
            raise ValueError(
                f"model_type={model_type!r} not found in importance "
                f"data. Available: {available}"
            )
    else:
        unique_models = df["model_type"].unique()
        if len(unique_models) > 1:
            raise ValueError(
                f"Multiple model_types found: {list(unique_models)}. "
                "Specify model_type parameter."
            )

    rows = []
    for feature, grp in df.groupby("feature"):
        vals = grp["importance"].values.astype(np.float64)
        finite = vals[np.isfinite(vals)]
        n = len(finite)
        if n == 0:
            rows.append({
                "feature": feature,
                **nan_summary(p_value=np.nan),
            })
            continue
        lo, hi = percentile_ci(finite, ci_level=ci)
        rows.append({
            "feature": feature,
            "mean": float(np.mean(finite)),
            # ddof=1 for sample std; single-iteration => 0.0
            "std": float(np.std(finite, ddof=1)) if n > 1 else 0.0,
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "p_value": float(np.mean(finite <= 0)),
        })

    result = pd.DataFrame(rows).sort_values("p_value", ignore_index=True)

    from mlsurv.evaluation._paired_comparison import _adjust_p_values
    result["p_adjusted"] = _adjust_p_values(
        result["p_value"].values, method=p_adjust,
    )
    return result


# ---------------------------------------------------------------------------
# ModelResult computation
# ---------------------------------------------------------------------------


def model_summary(
    model_result: 'ModelResult',
    metrics: List[str] = None,
) -> pd.DataFrame:
    """Get all metrics in a single DataFrame across all phases.

    Parameters
    ----------
    model_result : ModelResult
        Model result to summarize.
    metrics : list of str, optional
        Metrics to include. If None, includes all scorable metrics
        (see ``ALL_METRICS``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [metric, phase, point, mean, std, ci_lower, ci_upper]
        CV rows have mean/std from cross-validation folds.
        Test rows have the full-test-set point estimate; when bootstrap is
        available, mean/std/CI are merged from bootstrap (otherwise NaN).
    """
    if metrics is None:
        metrics = list(ALL_METRICS)

    rows = []
    boot = model_result.resolved_bootstrap
    boot_summary = boot.summary if boot else None

    for metric in metrics:
        # CV phase
        cv_mean = model_result.get_metric(metric, 'cv', 'mean')
        if cv_mean is not None:
            cv_std = model_result.get_metric(metric, 'cv', 'std')
            ci_lo = model_result.get_metric(metric, 'cv', 'ci_lower')
            ci_hi = model_result.get_metric(metric, 'cv', 'ci_upper')
            rows.append({
                'metric': metric,
                'phase': 'cv',
                'point': np.nan,
                'mean': cv_mean,
                'std': cv_std,
                'ci_lower': ci_lo,
                'ci_upper': ci_hi,
            })

        # Test phase: point estimate from full test set, CI from bootstrap
        test_point = model_result.get_metric(metric, 'test', 'mean')
        if test_point is not None:
            test_row = {
                'metric': metric,
                'phase': 'test',
                'point': test_point,
                **nan_summary(),
            }
            # Merge bootstrap stats into the test row
            if boot_summary is not None:
                mask = boot_summary['metric'] == metric
                if mask.any():
                    brow = boot_summary.loc[mask].iloc[0]
                    test_row['mean'] = float(brow['mean'])
                    test_row['std'] = float(brow['std'])
                    test_row['ci_lower'] = float(brow['ci_lower'])
                    test_row['ci_upper'] = float(brow['ci_upper'])
            rows.append(test_row)

    if not rows:
        return pd.DataFrame(columns=['metric', 'phase', 'point', 'mean', 'std', 'ci_lower', 'ci_upper'])

    return pd.DataFrame(rows)
