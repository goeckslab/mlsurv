"""Centralized constants for mlsurv package."""

from dataclasses import dataclass
from typing import Optional, Tuple

from mlsurv._calibration_thresholds import P_THRESHOLD_SIGNIFICANT


@dataclass(frozen=True)
class MetricInfo:
    """Metadata for a single evaluation metric.

    Attributes
    ----------
    key : str
        Short metric identifier (e.g., 'harrell').
    display_name : str
        Full display name for reports (e.g., "Harrell's C-index").
    short_name : str
        Shorter display name for compact tables (e.g., "Harrell's C").
    table_name : str
        Compact name for report tables (e.g., "Harrell C").
    lower_is_better : bool
        True if lower values indicate better performance.
    concordance_column : str
        Column name in concordance DataFrames (e.g., 'c_index_test').
    ideal_value : float or None
        Target value for a perfectly calibrated model (e.g., 1.0 for slope).
        None for metrics without a fixed ideal.
    diagnostic_only : bool
        If True, metric cannot be used for scoring/optimization — reported
        alongside scorable metrics but excluded from ALL_METRICS.
    """
    key: str
    display_name: str
    short_name: str
    table_name: str
    lower_is_better: bool
    concordance_column: str
    ideal_value: Optional[float] = None
    diagnostic_only: bool = False


METRIC_REGISTRY = {
    'harrell': MetricInfo(
        'harrell', "Harrell's C-index", "Harrell's C", "Harrell C",
        False, 'c_index_test',
    ),
    'ipcw': MetricInfo(
        'ipcw', "Uno's C-index (IPCW)", "Uno's C (IPCW)", "Uno C",
        False, 'c_index_ipcw_test',
    ),
    'integrated_brier': MetricInfo(
        'integrated_brier', "Integrated Brier Score", "IBS", "IBS",
        True, 'integrated_brier_test',
    ),
    'ctd': MetricInfo(
        'ctd', "Antolini's C-index (td)", "C^td", "C td",
        False, 'c_index_td_test',
    ),
    'cal_slope': MetricInfo(
        'cal_slope', "Calibration Slope", "Cal. Slope", "Cal Slope",
        False, 'cal_slope_test',
        ideal_value=1.0, diagnostic_only=True,
    ),
    'cal_intercept': MetricInfo(
        'cal_intercept', "Calibration Intercept", "Cal. Intercept", "Cal Int",
        True, 'cal_intercept_test',
        ideal_value=0.0, diagnostic_only=True,
    ),
    'gnd': MetricInfo(
        'gnd', "GND Calibration p", "GND p", "GND p",
        False, 'gnd_test',
        diagnostic_only=True,
    ),
}

# Derived views (backward compatibility)
LOWER_IS_BETTER = {k for k, v in METRIC_REGISTRY.items() if v.lower_is_better}
ALL_METRICS = [k for k, v in METRIC_REGISTRY.items() if not v.diagnostic_only]
SCORABLE_METRICS = ALL_METRICS  # Alias for semantic clarity at call sites
DIAGNOSTIC_METRICS = [k for k, v in METRIC_REGISTRY.items() if v.diagnostic_only]

# Metrics that structurally require predict_survival_function.  Entities
# that declare ``supports_survival_function = False`` (e.g. ReferenceAdapter)
# must have these filtered out before dispatch — attempting and suppressing
# the resulting NotImplementedError produces unactionable noise.
# Includes metrics callers may pass explicitly (e.g. ``'brier'``) even though
# they are absent from ``ALL_METRICS`` — this is the capability truth set,
# not the default bootstrap list.
SURV_FUNC_METRICS = frozenset({
    'brier', 'integrated_brier', 'ctd',
    'cal_slope', 'cal_intercept', 'gnd',
})

# Metrics that a monotone S(t|x) recalibration actually changes: the
# probability-magnitude metrics. This is ``SURV_FUNC_METRICS`` minus the
# rank-based ``ctd`` (time-dependent concordance), which — like Harrell's and
# Uno's C — is invariant under a monotone remap of the survival probabilities.
# Used to scope recalibrated bootstrap CIs and the ``use_recalibrated`` plot
# flag: rank/discrimination metrics are a silent no-op.
RECAL_AFFECTED_METRICS = SURV_FUNC_METRICS - {'ctd'}

# Default scoring metric used across all subsystems (setup, tuning, viz, CLI,
# results, reports).
DEFAULT_SCORING_METRIC = 'harrell'

# Display names for time-dependent metric keys not in METRIC_REGISTRY.
# Used by get_metric_display_name() fallback.
_DISPLAY_OVERRIDES = {
    "auc": "AUC",
    "brier": "Brier Score",
    "cindex": "C-index",
    "uno": "Uno's C-index",
    "ibs": "IBS",
}

# Artifact persistence
ARTIFACT_METADATA_FILE = 'artifact.json'
ARTIFACT_MODEL_FILE = 'model.joblib'
ARTIFACT_RECALIBRATOR_FILE = 'recalibrator.joblib'
ARTIFACT_VERSION = 1

# Learner persistence (save_learner / load_learner)
LEARNER_FORMAT_VERSION = 1

# Events Per Variable (EPV) thresholds for tuning warnings
EPV_THRESHOLD_WARNING = 2
EPV_THRESHOLD_INFO = 5

# Sample size and event rate thresholds for setup warnings
MIN_SAMPLE_SIZE_WARNING = 200
MIN_EVENT_RATE = 0.15
MAX_EVENT_RATE = 0.85

# IPCW failure threshold for CV aggregation
IPCW_FAILURE_THRESHOLD = 0.2

# Statistical significance thresholds shared by evaluation (feature_summary
# stars) and reports (p-value formatting). ``P_THRESHOLD_SIGNIFICANT`` comes
# from ``mlsurv._calibration_thresholds`` (single source of truth);
# lives here rather than in ``reports/_constants.py`` so the evaluation
# layer never imports from ``reports``.
P_THRESHOLD_EXCELLENT = 0.001
P_THRESHOLD_GOOD = 0.01

# P-value → star marker mapping (ordered strictest-first)
P_STARS: Tuple[Tuple[float, str], ...] = (
    (P_THRESHOLD_EXCELLENT, "***"),
    (P_THRESHOLD_GOOD, "**"),
    (P_THRESHOLD_SIGNIFICANT, "*"),
)

# Correlated-feature threshold shared by evaluation (feature_summary
# ``correlated`` flag) and the report's correlation analysis.
CORRELATION_THRESHOLD = 0.8

# Missingness-outcome leakage detection thresholds
# Feature flagged when BOTH: p < P_THRESHOLD AND V > V_THRESHOLD
MISSINGNESS_LEAKAGE_P_THRESHOLD = 0.001
MISSINGNESS_LEAKAGE_V_THRESHOLD = 0.3

# Model recommendation thresholds (recommend_models)
# Evidence: van der Ploeg 2014, Harrell 2018, Ogundimu 2016, Riley 2019
#
# Thresholds keyed by complexity class: (N_recommended, EPV_recommended,
#                                         N_cautionary,  EPV_cautionary)
RECOMMEND_THRESHOLDS = {
    "linear":        (None, 20,  None, 5),
    "tree_ensemble": (500,  50,  200,  20),
    "deep_learning": (1000, 100, 500,  50),
}
FEATURE_REDUCTION_EPV = 10   # EPV < 10 -> suggest feature reduction
TOO_SMALL_EPV = 2            # EPV < 2  -> flag dataset as likely too small

# Model-indexed metadata derived from the model registry.
# Each model class in mlsurv/models/ declares its own PH/linearity
# assumptions and display name as class attributes; these module-level
# names just collect them, so adding a new model in one place suffices.
# GBSA caveat (default loss='coxph'): if a user sets loss='ipcwls' via
# model_params, GBSA.assumes_ph no longer reflects the resolved
# assumption — see the comment in mlsurv/models/gbsa.py.
from mlsurv.models import (
    get_ph_assumption_models,
    get_linearity_assumption_models,
    get_model_display_names,
)

# Models that assume proportional hazards.
PH_ASSUMPTION_MODELS = get_ph_assumption_models()

# Models that assume log-hazard linearity in continuous covariates
# (Harrell 2015 §4.6, §5.5).
LINEARITY_ASSUMPTION_MODELS = get_linearity_assumption_models()

# Pretty display names for model keys, used by limitations/guidance
# sections when listing affected/eligible models in prose.
MODEL_DISPLAY_NAMES = get_model_display_names()


def get_metric_display_name(metric: str) -> str:
    """
    Get human-readable display name for a metric.

    Parameters
    ----------
    metric : str
        Short metric name (e.g., 'harrell', 'ipcw').

    Returns
    -------
    str
        Full display name, or title-cased metric name if not found.
    """
    info = METRIC_REGISTRY.get(metric)
    if info:
        return info.display_name
    return _DISPLAY_OVERRIDES.get(metric, metric.title())


def get_metric_info(metric: str) -> Optional[MetricInfo]:
    """Get MetricInfo for a metric key.

    Parameters
    ----------
    metric : str
        Short metric name (e.g., 'harrell').

    Returns
    -------
    MetricInfo or None
        Metric metadata, or None if not found.
    """
    return METRIC_REGISTRY.get(metric)


def get_metric_short_name(metric: str, with_asterisk: bool = True) -> str:
    """Get short display name, optionally with * for lower-is-better.

    Parameters
    ----------
    metric : str
        Short metric name.
    with_asterisk : bool
        If True, append ' *' for lower-is-better metrics.

    Returns
    -------
    str
        Short display name.
    """
    info = METRIC_REGISTRY.get(metric)
    if not info:
        return metric.title()
    name = info.short_name
    if with_asterisk:
        name = _annotate_metric_name(name, info)
    return name


def get_metric_table_name(metric: str, with_asterisk: bool = True) -> str:
    """Get compact table name, optionally with annotation.

    Parameters
    ----------
    metric : str
        Short metric name.
    with_asterisk : bool
        If True, annotate with direction/ideal-value suffix.

    Returns
    -------
    str
        Compact table name.
    """
    info = METRIC_REGISTRY.get(metric)
    if not info:
        return metric.title()
    name = info.table_name
    if with_asterisk:
        name = _annotate_metric_name(name, info)
    return name


def _format_ideal(value: float) -> str:
    """Format an ideal value as int when possible (e.g. 1.0 -> '1', 0.5 -> '0.5')."""
    return str(int(value)) if value == int(value) else str(value)


def _annotate_metric_name(name: str, info: MetricInfo) -> str:
    """Append direction or ideal-value annotation to a metric name.

    - Ideal-value metrics: ' (=1)' or ' (=0)'
    - Lower-is-better (no ideal): ' *'
    """
    if info.ideal_value is not None:
        return f"{name} (={_format_ideal(info.ideal_value)})"
    if info.lower_is_better:
        return f"{name} *"
    return name


def get_concordance_column(metric: str) -> Optional[str]:
    """Get concordance DataFrame column name for a metric.

    Parameters
    ----------
    metric : str
        Short metric name.

    Returns
    -------
    str or None
        Concordance column name, or None if not found.
    """
    info = METRIC_REGISTRY.get(metric)
    return info.concordance_column if info else None
