"""Evaluation module for mlsurv."""

from ._ctd import concordance_td
from .metrics import (
    c_index, auc_t, ph_test, ph_test_with_schoenfeld, linearity_test,
    rmst, rmst_test,
    rmst_test_pairwise, RMSTResult, PairwiseRMSTResult,
    score_survival_model, surv_funcs_to_probs, predict_survival_functions,
)
from ._diagnostics import (
    compute_schoenfeld_data,
    compute_deviance_data,
    compute_martingale_data,
)
from ._prediction_helpers import predict_from_pipeline
from .evaluate import evaluate, DEFAULT_N_IMPORTANCE_REPEATS
from .bootstrap import bootstrap
from .importance import (
    compute_pimp,
    compute_univariate_cox,
)
from ._coefficients import (
    COEFFICIENT_COLUMNS,
    compute_model_coefficients,
)
from ._feature_summary import (
    FEATURE_SUMMARY_COLUMNS,
    compute_feature_summary,
    resolve_importance_pvalues,
)
from ._incremental import (
    bootstrap_incremental,
    compute_incremental_metrics,
    format_p_sig,
)
from ._paired_comparison import compare_bootstrap, compare_kang
from ._explain import compute_patient_explanation
from ._univariate_cox import compute_reference_univariate_cox
from ..constants import ALL_METRICS, LOWER_IS_BETTER

__all__ = [
    # Metrics
    "c_index",
    "concordance_td",
    "auc_t",
    "score_survival_model",
    "ph_test",
    "ph_test_with_schoenfeld",
    "linearity_test",
    # Diagnostic residual computations
    "compute_schoenfeld_data",
    "compute_deviance_data",
    "compute_martingale_data",
    "rmst",
    "rmst_test",
    "rmst_test_pairwise",
    "RMSTResult",
    "PairwiseRMSTResult",
    # Survival predictions
    "surv_funcs_to_probs",
    "predict_survival_functions",
    "predict_from_pipeline",
    # Evaluation
    "evaluate",
    # Bootstrap
    "bootstrap",
    # Importance
    "compute_pimp",
    "compute_univariate_cox",
    # Model coefficients
    "COEFFICIENT_COLUMNS",
    "compute_model_coefficients",
    # Feature summary
    "FEATURE_SUMMARY_COLUMNS",
    "compute_feature_summary",
    "resolve_importance_pvalues",
    # Incremental metrics
    "bootstrap_incremental",
    "compute_incremental_metrics",
    "format_p_sig",
    # Patient explanation
    "compute_patient_explanation",
    # Paired model comparison
    "compare_bootstrap",
    "compare_kang",
    # Reference univariate Cox
    "compute_reference_univariate_cox",
    # Constants
    "ALL_METRICS",
    "LOWER_IS_BETTER",
    "DEFAULT_N_IMPORTANCE_REPEATS",
]
