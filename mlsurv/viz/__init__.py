"""Visualization module for mlsurv.

Public API uses ``plot_*`` and ``compute_*`` naming convention.
"""

# ── Metrics ─────────────────────────────────────────────────────────────────
from mlsurv.viz.metrics import (
    plot_metric,
    plot_metric_time,
    compute_brier_curves,
    compute_roc_curves,
    plot_roc,
    plot_roc_interactive,
    plot_metric_heatmap,
    plot_faceted_metric_heatmap,
)

# ── Survival ────────────────────────────────────────────────────────────────
from mlsurv.viz.survival import plot_km

# ── Features ────────────────────────────────────────────────────────────────
from mlsurv.viz.features import (
    plot_importance,
    plot_importance_heatmap,
    plot_coefficients,
    plot_coefficient_comparison,
    plot_hazard_ratios,
    plot_hazard_ratios_comparison,
    compute_hazard_ratios,
    plot_feature_correlation,
)

# ── Missing Data ────────────────────────────────────────────────────────────
from mlsurv.viz.missing import plot_missing

# ── SHAP ────────────────────────────────────────────────────────────────────
from mlsurv.viz.shap import (
    plot_shap_importance,
    plot_shap_beeswarm,
)

# ── Explain ────────────────────────────────────────────────────────────────
from mlsurv.viz.explain import plot_waterfall

# ── Interactions ───────────────────────────────────────────────────────────
from mlsurv.viz.interactions import plot_interaction_heatmap

# ── Calibration ─────────────────────────────────────────────────────────────
from mlsurv.viz.calibration import (
    plot_calibration,
    plot_calibration_interactive,
    plot_calibration_overlay,
    compute_calibration,
)

# ── Distributions ──────────────────────────────────────────────────────────
from mlsurv.viz.distributions import plot_feature_distributions

# ── Diagnostics ─────────────────────────────────────────────────────────────
from mlsurv.viz.diagnostics import (
    plot_schoenfeld,
    plot_schoenfeld_interactive,
    plot_deviance,
    plot_martingale_interactive,
    compute_schoenfeld_data,
    compute_deviance_data,
    compute_martingale_data,
)

# ── Sweep ──────────────────────────────────────────────────────────────────
from mlsurv.viz.sweep import plot_sweep

# ── Export / Flowchart (unchanged) ──────────────────────────────────────────
from mlsurv.viz.export import save_html, save_html_report
from mlsurv.viz.flowchart import pipeline_flowchart

# ── Infrastructure (re-exported for downstream use) ─────────────────────────
from mlsurv.viz._palettes import (
    AUC_COLORSCALE,
    CINDEX_COLORSCALE,
    DISEASE_COLORS,
    FeatureGroupsType,
    MANUSCRIPT_DEFAULTS,
    MODEL_COLORS,
    NAMED_PALETTES,
    QUALITATIVE_PALETTE,
    RISK_QUARTILE_COLORS,
    build_benchmark_palette,
    get_disease_color,
    get_manuscript_layout,
    get_model_color,
    get_reference_color,
    reset_palette,
    resolve_feature_group_colors,
    set_palette,
)
from mlsurv.viz._theme import (
    adaptive_size,
    create_figure,
    add_ci_ribbon,
    add_reference_hline,
    add_reference_vline,
)

__all__ = [
    # Metrics
    "plot_metric",
    "plot_metric_time",
    "compute_brier_curves",
    "compute_roc_curves",
    "plot_roc",
    "plot_roc_interactive",
    "plot_metric_heatmap",
    "plot_faceted_metric_heatmap",
    # Survival
    "plot_km",
    # Features
    "plot_importance",
    "plot_importance_heatmap",
    "plot_coefficients",
    "plot_coefficient_comparison",
    "plot_hazard_ratios",
    "plot_hazard_ratios_comparison",
    "compute_hazard_ratios",
    "plot_feature_correlation",
    # Missing Data
    "plot_missing",
    # SHAP
    "plot_shap_importance",
    "plot_shap_beeswarm",
    # Explain
    "plot_waterfall",
    # Interactions
    "plot_interaction_heatmap",
    # Calibration
    "plot_calibration",
    "plot_calibration_interactive",
    "plot_calibration_overlay",
    "compute_calibration",
    # Distributions
    "plot_feature_distributions",
    # Diagnostics
    "plot_schoenfeld",
    "plot_schoenfeld_interactive",
    "plot_deviance",
    "plot_martingale_interactive",
    "compute_schoenfeld_data",
    "compute_deviance_data",
    "compute_martingale_data",
    # Sweep
    "plot_sweep",
    # Export / Flowchart
    "save_html",
    "save_html_report",
    "pipeline_flowchart",
    # Infrastructure
    "AUC_COLORSCALE",
    "CINDEX_COLORSCALE",
    "DISEASE_COLORS",
    "FeatureGroupsType",
    "MANUSCRIPT_DEFAULTS",
    "MODEL_COLORS",
    "NAMED_PALETTES",
    "QUALITATIVE_PALETTE",
    "RISK_QUARTILE_COLORS",
    "build_benchmark_palette",
    "get_disease_color",
    "get_manuscript_layout",
    "get_model_color",
    "get_reference_color",
    "reset_palette",
    "resolve_feature_group_colors",
    "set_palette",
    "adaptive_size",
    "create_figure",
    "add_ci_ribbon",
    "add_reference_hline",
    "add_reference_vline",
]
