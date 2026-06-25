"""Result containers for mlsurv."""

from mlsurv.results.containers import (
    EvaluationResult,
    BootstrapResult,
    ModelResult,
    ValidationResult,
    BenchmarkResult,
    SweepResult,
    extract_model_evaluation,
)

from mlsurv.results._explanation import PatientExplanation
from mlsurv.results._prediction import PredictionResult

from mlsurv.results.tables import (
    format_metrics_table,
    format_importance_table,
    format_bootstrap_summary,
    format_hazard_ratios,
    to_latex,
    to_markdown,
    to_html,
)

from mlsurv.results.printing import (
    print_cv_metrics_table,
    print_multi_model_cv_table,
    print_bootstrap_table,
    print_evaluation_table,
    print_ph_test,
    print_rmst_test,
    print_rmst_pairwise,
    print_calibration_test,
)

__all__ = [
    # Containers
    "EvaluationResult",
    "BootstrapResult",
    "ModelResult",
    "ValidationResult",
    "BenchmarkResult",
    "SweepResult",
    "extract_model_evaluation",
    "PatientExplanation",
    "PredictionResult",
    # Tables
    "format_metrics_table",
    "format_importance_table",
    "format_bootstrap_summary",
    "format_hazard_ratios",
    "to_latex",
    "to_markdown",
    "to_html",
    # Printing
    "print_cv_metrics_table",
    "print_multi_model_cv_table",
    "print_bootstrap_table",
    "print_evaluation_table",
    "print_ph_test",
    "print_rmst_test",
    "print_rmst_pairwise",
    "print_calibration_test",
]
