"""Plot subcommand for the mlsurv CLI."""

from __future__ import annotations

import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ._artifacts import _load_learner_if_requested


# ---------------------------------------------------------------------------
# Plot registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PlotSpec:
    """Metadata for a single plot type."""

    method: str
    prerequisite: str  # "setup" | "train" | "evaluate" | "tune" | "benchmark"
    description: str
    valid_flags: frozenset
    build_kwargs: Callable  # (args) -> dict
    model_param: str | None = 'models'  # 'models', 'model', or None


# --- per-type kwarg builders -----------------------------------------------

def _build_survival_kwargs(args) -> dict:
    kw: dict = {}
    if args.n_groups is not None:
        kw['n_groups'] = args.n_groups
    if args.show_risk_table:
        kw['show_risk_table'] = True
    if args.time_limit is not None:
        kw['time_limit'] = args.time_limit
    if args.references:
        kw['references'] = 'all'
    return kw


def _build_metric_kwargs(args) -> dict:
    kw: dict = {}
    if args.style != 'auto':
        kw['style'] = args.style
    if args.metric is not None:
        kw['metric'] = args.metric
    if args.source != 'auto':
        kw['source'] = args.source
    if args.uncertainty != 'auto':
        # plot_metric accepts "sd" (str) or a float CI level (e.g. 0.95)
        val = args.uncertainty
        try:
            val = float(val)
        except (ValueError, TypeError):
            pass
        kw['uncertainty'] = val
    if args.orientation != 'horizontal':
        kw['orientation'] = args.orientation
    if args.references:
        kw['references'] = 'all'
    return kw


def _build_importance_kwargs(args) -> dict:
    kw: dict = {}
    if args.n_features != 20:
        kw['n'] = args.n_features
    if args.style != 'auto':
        kw['style'] = args.style
    return kw


def _build_roc_kwargs(args) -> dict:
    kw: dict = {}
    if args.eval_time is not None:
        kw['t'] = args.eval_time
    if args.no_auc:
        kw['show_auc'] = False
    if args.no_ci:
        kw['show_ci'] = False
    if args.references:
        kw['references'] = 'all'
    return kw


def _build_calibration_kwargs(args) -> dict:
    kw: dict = {}
    if args.n_bins != 10:
        kw['n_bins'] = args.n_bins
    if args.eval_time is not None:
        # plot_calibration takes a single unified ``eval_times`` (scalar or
        # list); the CLI ``--eval-time`` flag is a scalar, which it accepts.
        kw['eval_times'] = args.eval_time
    return kw


def _build_missing_kwargs(args) -> dict:
    return {}


# --- Phase 2 kwarg builders --------------------------------------------------

def _build_diagnostics_kwargs(args) -> dict:
    kw: dict = {}
    if args.kind is not None:
        kw['kind'] = args.kind
    if args.features is not None:
        kw['features'] = args.features
    return kw


def _build_coefficients_kwargs(args) -> dict:
    kw: dict = {}
    if args.n_features != 20:
        kw['n_features'] = args.n_features
    return kw


def _build_hazard_ratios_kwargs(args) -> dict:
    kw: dict = {}
    if args.kind is not None:
        kw['kind'] = args.kind
    if args.n_features != 20:
        kw['n_features'] = args.n_features
    return kw


def _build_feature_correlation_kwargs(args) -> dict:
    kw: dict = {}
    if args.threshold != 0.8:
        kw['threshold'] = args.threshold
    return kw


def _build_feature_distributions_kwargs(args) -> dict:
    kw: dict = {}
    if args.features is not None:
        kw['features'] = args.features
    return kw


def _build_metric_time_kwargs(args) -> dict:
    kw: dict = {}
    if args.metrics is not None:
        # The shared --metrics CLI flag maps to plot_metric_time's unified
        # metric= parameter (str | list). The heatmap builder keeps metrics=.
        kw['metric'] = args.metrics
    if args.references:
        kw['references'] = 'all'
    return kw


# --- Phase 3 kwarg builders --------------------------------------------------

def _build_shap_kwargs(args) -> dict:
    kw: dict = {}
    if args.kind is not None:
        kw['kind'] = args.kind
    kw['n'] = args.n_features  # CLI default 20 is authoritative
    return kw


def _build_interactions_kwargs(args) -> dict:
    kw: dict = {}
    if args.n_features != 20:
        kw['n'] = args.n_features
    return kw


def _build_interaction_pdp_kwargs(args) -> dict:
    features = args.features
    if features is None or len(features) != 2:
        raise ValueError(
            "interaction-pdp requires exactly 2 features via --features "
            f"(got {len(features) if features else 0})"
        )
    kw: dict = {'feature_1': features[0], 'feature_2': features[1]}
    if args.max_samples != 100:
        kw['max_samples'] = args.max_samples
    if args.grid_resolution != 25:
        kw['grid_resolution'] = args.grid_resolution
    return kw


def _build_calibration_overlay_kwargs(args) -> dict:
    kw: dict = {}
    if args.eval_times is not None:
        kw['eval_times'] = args.eval_times
    return kw


def _build_optuna_kwargs(args) -> dict:
    return {}


def _build_metric_heatmap_kwargs(args) -> dict:
    kw: dict = {}
    if args.metrics is not None:
        kw['metrics'] = args.metrics
    return kw


def _build_hazard_ratios_comparison_kwargs(args) -> dict:
    kw: dict = {}
    if args.n_features != 20:
        kw['n_features'] = args.n_features
    return kw


def _build_coefficient_comparison_kwargs(args) -> dict:
    kw: dict = {}
    if args.n_features != 20:
        kw['n_features'] = args.n_features
    return kw


def _build_benchmark_incremental_kwargs(args) -> dict:
    return {}


def _build_benchmark_scatter_kwargs(args) -> dict:
    return {}


def _build_validation_heatmap_kwargs(args) -> dict:
    kw: dict = {}
    if args.metrics is not None:
        kw['metrics'] = args.metrics
    return kw


# --- shared flag sets -------------------------------------------------------

_SURVIVAL_FLAGS = frozenset({
    'n_groups', 'show_risk_table', 'time_limit', 'references',
})
_METRIC_FLAGS = frozenset({
    'style', 'metric', 'source', 'uncertainty', 'orientation', 'references',
})
_IMPORTANCE_FLAGS = frozenset({'style', 'n_features'})
_ROC_FLAGS = frozenset({'eval_time', 'no_auc', 'no_ci', 'references'})
_CALIBRATION_FLAGS = frozenset({'n_bins', 'eval_time'})
_MISSING_FLAGS = frozenset()
_DIAGNOSTICS_FLAGS = frozenset({'kind', 'features'})
_COEFFICIENTS_FLAGS = frozenset({'n_features'})
_HAZARD_RATIOS_FLAGS = frozenset({'kind', 'n_features'})
_FEATURE_CORRELATION_FLAGS = frozenset({'threshold'})
_FEATURE_DISTRIBUTIONS_FLAGS = frozenset({'features'})
_METRIC_TIME_FLAGS = frozenset({'metrics', 'references'})
_SHAP_FLAGS = frozenset({'kind', 'n_features'})
_INTERACTIONS_FLAGS = frozenset({'n_features'})
_INTERACTION_PDP_FLAGS = frozenset({'features', 'max_samples', 'grid_resolution'})
_CALIBRATION_OVERLAY_FLAGS = frozenset({'eval_times'})
_OPTUNA_FLAGS = frozenset()
_METRIC_HEATMAP_FLAGS = frozenset({'metrics'})
_HAZARD_RATIOS_COMPARISON_FLAGS = frozenset({'n_features'})
_COEFFICIENT_COMPARISON_FLAGS = frozenset({'n_features'})
_BENCHMARK_INCREMENTAL_FLAGS = frozenset()
_BENCHMARK_SCATTER_FLAGS = frozenset()
_VALIDATION_HEATMAP_FLAGS = frozenset({'metrics'})

# All type-specific flag attribute names (for validation).
_ALL_TYPE_FLAGS = (
    _SURVIVAL_FLAGS | _METRIC_FLAGS | _IMPORTANCE_FLAGS
    | _ROC_FLAGS | _CALIBRATION_FLAGS | _MISSING_FLAGS
    | _DIAGNOSTICS_FLAGS | _COEFFICIENTS_FLAGS | _HAZARD_RATIOS_FLAGS
    | _FEATURE_CORRELATION_FLAGS | _FEATURE_DISTRIBUTIONS_FLAGS
    | _METRIC_TIME_FLAGS
    | _SHAP_FLAGS | _INTERACTIONS_FLAGS | _INTERACTION_PDP_FLAGS
    | _CALIBRATION_OVERLAY_FLAGS | _OPTUNA_FLAGS | _METRIC_HEATMAP_FLAGS
    | _HAZARD_RATIOS_COMPARISON_FLAGS | _COEFFICIENT_COMPARISON_FLAGS
    | _BENCHMARK_INCREMENTAL_FLAGS
    | _BENCHMARK_SCATTER_FLAGS | _VALIDATION_HEATMAP_FLAGS
)


_PLOT_REGISTRY: dict[str, _PlotSpec] = {
    'survival': _PlotSpec(
        method='plot_survival',
        prerequisite='train',
        description='Kaplan-Meier curves stratified by risk score',
        valid_flags=_SURVIVAL_FLAGS,
        build_kwargs=_build_survival_kwargs,
    ),
    'metric': _PlotSpec(
        method='plot_metric',
        prerequisite='train',
        description='Metric comparison (forest, bar, raincloud, boxplot)',
        valid_flags=_METRIC_FLAGS,
        build_kwargs=_build_metric_kwargs,
    ),
    'importance': _PlotSpec(
        method='plot_importance',
        prerequisite='evaluate',
        description='Feature importance (bar, forest, heatmap)',
        valid_flags=_IMPORTANCE_FLAGS,
        build_kwargs=_build_importance_kwargs,
    ),
    'roc': _PlotSpec(
        method='plot_roc',
        prerequisite='evaluate',
        description='Time-dependent ROC curve',
        valid_flags=_ROC_FLAGS,
        build_kwargs=_build_roc_kwargs,
    ),
    'calibration': _PlotSpec(
        method='plot_calibration',
        prerequisite='train',
        description='Calibration curve (predicted vs. observed)',
        valid_flags=_CALIBRATION_FLAGS,
        build_kwargs=_build_calibration_kwargs,
    ),
    'missing': _PlotSpec(
        method='plot_missing',
        prerequisite='setup',
        description='Missing data heatmap',
        valid_flags=_MISSING_FLAGS,
        build_kwargs=_build_missing_kwargs,
    ),
    # --- Phase 2 types ---
    'diagnostics': _PlotSpec(
        method='plot_diagnostics',
        prerequisite='setup',
        description='Diagnostic residual plots (Schoenfeld, deviance, martingale)',
        valid_flags=_DIAGNOSTICS_FLAGS,
        build_kwargs=_build_diagnostics_kwargs,
    ),
    'coefficients': _PlotSpec(
        method='plot_coefficients',
        prerequisite='train',
        description='Feature coefficients from Cox models',
        valid_flags=_COEFFICIENTS_FLAGS,
        build_kwargs=_build_coefficients_kwargs,
    ),
    'hazard-ratios': _PlotSpec(
        method='plot_hazard_ratios',
        prerequisite='train',
        description='Hazard ratios (univariate or multivariate)',
        valid_flags=_HAZARD_RATIOS_FLAGS,
        build_kwargs=_build_hazard_ratios_kwargs,
    ),
    'feature-correlation': _PlotSpec(
        method='plot_feature_correlation',
        prerequisite='setup',
        description='Feature correlation matrix',
        valid_flags=_FEATURE_CORRELATION_FLAGS,
        build_kwargs=_build_feature_correlation_kwargs,
    ),
    'feature-distributions': _PlotSpec(
        method='plot_feature_distributions',
        prerequisite='setup',
        description='Feature distributions for train/test sets',
        valid_flags=_FEATURE_DISTRIBUTIONS_FLAGS,
        build_kwargs=_build_feature_distributions_kwargs,
    ),
    'metric-time': _PlotSpec(
        method='plot_metric_time',
        prerequisite='evaluate',
        description='Time-dependent metric curves (AUC over time)',
        valid_flags=_METRIC_TIME_FLAGS,
        build_kwargs=_build_metric_time_kwargs,
    ),
    # --- Phase 3 types ---
    'shap': _PlotSpec(
        method='plot_shap',
        prerequisite='evaluate',
        description='SHAP feature importance or beeswarm',
        valid_flags=_SHAP_FLAGS,
        build_kwargs=_build_shap_kwargs,
    ),
    'interactions': _PlotSpec(
        method='plot_interactions',
        prerequisite='evaluate',
        description='Feature interaction heatmap (nonlinear models only)',
        valid_flags=_INTERACTIONS_FLAGS,
        build_kwargs=_build_interactions_kwargs,
        model_param='model',
    ),
    'interaction-pdp': _PlotSpec(
        method='plot_interaction_pdp',
        prerequisite='evaluate',
        description='2D partial dependence for a feature pair (nonlinear models only)',
        valid_flags=_INTERACTION_PDP_FLAGS,
        build_kwargs=_build_interaction_pdp_kwargs,
        model_param='model',
    ),
    'calibration-overlay': _PlotSpec(
        method='plot_calibration_overlay',
        prerequisite='evaluate',
        description='Predicted vs. observed survival curves overlay',
        valid_flags=_CALIBRATION_OVERLAY_FLAGS,
        build_kwargs=_build_calibration_overlay_kwargs,
    ),
    'optuna': _PlotSpec(
        method='plot_optuna',
        prerequisite='tune',
        description='Optuna hyperparameter optimization results',
        valid_flags=_OPTUNA_FLAGS,
        build_kwargs=_build_optuna_kwargs,
    ),
    'metric-heatmap': _PlotSpec(
        method='plot_metric_heatmap',
        prerequisite='evaluate',
        description='Multi-metric heatmap across models',
        valid_flags=_METRIC_HEATMAP_FLAGS,
        build_kwargs=_build_metric_heatmap_kwargs,
    ),
    'hazard-ratios-comparison': _PlotSpec(
        method='plot_hazard_ratios_comparison',
        prerequisite='train',
        description='Univariate vs. multivariate hazard ratios comparison',
        valid_flags=_HAZARD_RATIOS_COMPARISON_FLAGS,
        build_kwargs=_build_hazard_ratios_comparison_kwargs,
        model_param=None,
    ),
    'coefficient-comparison': _PlotSpec(
        method='plot_coefficient_comparison',
        prerequisite='train',
        description='Coefficient scatter across Cox models',
        valid_flags=_COEFFICIENT_COMPARISON_FLAGS,
        build_kwargs=_build_coefficient_comparison_kwargs,
    ),
    'benchmark-incremental': _PlotSpec(
        method='plot_benchmark_incremental',
        prerequisite='benchmark',
        description='Incremental metrics forest plot (delta-C, NRI, IDI)',
        valid_flags=_BENCHMARK_INCREMENTAL_FLAGS,
        build_kwargs=_build_benchmark_incremental_kwargs,
        model_param=None,
    ),
    'benchmark-scatter': _PlotSpec(
        method='plot_benchmark_scatter',
        prerequisite='benchmark',
        description='Scatter of model risk scores vs. reference scores',
        valid_flags=_BENCHMARK_SCATTER_FLAGS,
        build_kwargs=_build_benchmark_scatter_kwargs,
        model_param=None,
    ),
    'validation-heatmap': _PlotSpec(
        method='plot_validation_heatmap',
        prerequisite='validate',
        description='Cross-cohort validation metric heatmap (needs 2+ cohorts)',
        valid_flags=_VALIDATION_HEATMAP_FLAGS,
        build_kwargs=_build_validation_heatmap_kwargs,
        model_param='models',
    ),
}

# Defaults for type-specific flags (used to detect explicitly-set values).
_FLAG_DEFAULTS: dict[str, object] = {
    'style': 'auto',
    'metric': None,
    'source': 'auto',
    'uncertainty': 'auto',
    'orientation': 'horizontal',
    'n_groups': None,
    'show_risk_table': False,
    'time_limit': None,
    'n_features': 20,
    'n_bins': 10,
    'eval_time': None,
    'no_auc': False,
    'no_ci': False,
    'references': False,
    'kind': None,
    'features': None,
    'threshold': 0.8,
    'metrics': None,
    'max_samples': 100,
    'grid_resolution': 25,
    'eval_times': None,
}


# ---------------------------------------------------------------------------
# Prerequisite checking
# ---------------------------------------------------------------------------

_PREREQUISITE_GUIDANCE = {
    'setup': None,  # always satisfied after load_learner
    'train': (
        "at least one trained model",
        "Save the learner after 'mlsurv train' or 'mlsurv run'.",
    ),
    'evaluate': (
        "evaluation results",
        "Save the learner after 'mlsurv evaluate'.",
    ),
    'tune': (
        "at least one tuned model (with Optuna study)",
        "Save the learner after 'mlsurv tune'.",
    ),
    'benchmark': (
        "benchmark results",
        "Save the learner after 'mlsurv benchmark'.",
    ),
    'validate': (
        "validation results from at least 2 cohorts",
        "Save the learner after running 'mlsurv validate' on 2+ external cohorts.",
    ),
}


def _check_prerequisite(learner, spec: _PlotSpec, plot_type: str) -> str | None:
    """Return an error message if the learner lacks the required state."""
    level = spec.prerequisite
    if level == 'setup':
        return None
    if level == 'train':
        if len(learner.models) > 0:
            return None
        what, fix = _PREREQUISITE_GUIDANCE['train']
        return (
            f"Plot type '{plot_type}' requires {what}. {fix}"
        )
    if level == 'evaluate':
        if any(
            mr.evaluation is not None
            for mr in learner.models.values()
        ):
            return None
        what, fix = _PREREQUISITE_GUIDANCE['evaluate']
        return (
            f"Plot type '{plot_type}' requires {what}. {fix}"
        )
    if level == 'tune':
        if any(
            mr.optuna_study is not None
            or bool(getattr(mr, '_optuna_tuning_history', None))
            for mr in learner.models.values()
        ):
            return None
        what, fix = _PREREQUISITE_GUIDANCE['tune']
        return (
            f"Plot type '{plot_type}' requires {what}. {fix}"
        )
    if level == 'benchmark':
        if learner.benchmark_result is not None:
            return None
        what, fix = _PREREQUISITE_GUIDANCE['benchmark']
        return (
            f"Plot type '{plot_type}' requires {what}. {fix}"
        )
    if level == 'validate':
        if len(learner.validation_results) >= 2:
            return None
        what, fix = _PREREQUISITE_GUIDANCE['validate']
        return (
            f"Plot type '{plot_type}' requires {what}. {fix}"
        )
    return f"Unknown prerequisite level: {level}"


# ---------------------------------------------------------------------------
# Flag validation
# ---------------------------------------------------------------------------

def _validate_type_flags(args, spec: _PlotSpec, plot_type: str) -> str | None:
    """Return an error message if a type-specific flag was set for the wrong type."""
    for flag in _ALL_TYPE_FLAGS:
        if flag in spec.valid_flags:
            continue
        val = getattr(args, flag, None)
        default = _FLAG_DEFAULTS.get(flag)
        if val != default:
            cli_flag = '--' + flag.replace('_', '-')
            return f"{cli_flag} is not valid for plot type '{plot_type}'"
    return None


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_figure(fig, output_path: str, fmt: str) -> None:
    """Write a plotly figure to the requested format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == 'html':
        fig.write_html(str(path))
    elif fmt == 'json':
        path.write_text(fig.to_json(), encoding='utf-8')
    else:
        # png, svg, pdf, etc. — handled by kaleido, bounded so a stalled
        # browser raises FigureExportTimeout instead of hanging.
        from mlsurv.reports._figure_export import write_image_safe
        write_image_safe(fig, path)


def _maybe_open_browser(
    output_path: str,
    fmt: str,
    no_open: bool,
    quiet: bool,
) -> None:
    """Open the output file in a browser for HTML format."""
    if fmt != 'html' or no_open or quiet:
        return
    uri = Path(output_path).resolve().as_uri()
    webbrowser.open(uri)


# ---------------------------------------------------------------------------
# Inline plot generation (for --plot on evaluate/bootstrap)
# ---------------------------------------------------------------------------

def _generate_inline_plots(
    learner,
    plot_types: list[str],
    output_dir: Path | None,
    quiet: bool,
) -> int:
    """Generate plots inline after evaluate/bootstrap.

    Validates each type, checks prerequisites, calls the plot method with
    defaults, and saves to ``<output_dir>/<type>.html``.

    Returns the number of failures (0 = all succeeded).
    """
    if not plot_types:
        return 0

    out = Path(output_dir) if output_dir is not None else Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    n_failed = 0

    for plot_type in plot_types:
        if plot_type not in _PLOT_REGISTRY:
            available = ', '.join(sorted(_PLOT_REGISTRY))
            print(
                f"Warning: Unknown plot type '{plot_type}' (skipped). "
                f"Available: {available}",
                file=sys.stderr,
            )
            n_failed += 1
            continue

        spec = _PLOT_REGISTRY[plot_type]

        err = _check_prerequisite(learner, spec, plot_type)
        if err:
            print(f"Warning: Skipping plot '{plot_type}': {err}",
                  file=sys.stderr)
            n_failed += 1
            continue

        try:
            fig = getattr(learner, spec.method)()
        except (MemoryError, RecursionError):
            raise
        except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
            print(f"Warning: Plot '{plot_type}' failed: {exc}",
                  file=sys.stderr)
            n_failed += 1
            continue

        dest = out / f"{plot_type}.html"
        try:
            _write_figure(fig, str(dest), 'html')
        except OSError as exc:
            print(f"Warning: Could not save plot '{plot_type}': {exc}",
                  file=sys.stderr)
            n_failed += 1
            continue

        if not quiet:
            print(f"  Plot saved: {dest}")

    return n_failed


# ---------------------------------------------------------------------------
# --list helper
# ---------------------------------------------------------------------------

def _print_plot_list() -> None:
    """Print available plot types with prerequisites and descriptions."""
    col_w = max(len(n) for n in _PLOT_REGISTRY) + 2
    print("Available plot types:\n")
    print(f"  {'Type':<{col_w}} {'Requires':<12} Description")
    print(f"  {'-' * (col_w - 2)}   {'-' * 10}   {'-' * 30}")
    for name, spec in _PLOT_REGISTRY.items():
        print(f"  {name:<{col_w}} {spec.prerequisite:<12} {spec.description}")
    print(f"\n  ({len(_PLOT_REGISTRY)} types available)")


# ---------------------------------------------------------------------------
# --return-data helper
# ---------------------------------------------------------------------------

def _handle_return_data(result, args, quiet: bool) -> int:
    """Write return_data result (DataFrame or dict) to CSV/JSON/stdout."""
    import json

    import pandas as pd

    output = getattr(args, 'output', None)

    if isinstance(result, pd.DataFrame):
        if output:
            try:
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                result.to_csv(output, index=False)
            except OSError as exc:
                print(
                    f"Error: Failed to write data to {output}: {exc}",
                    file=sys.stderr,
                )
                return 1
            if not quiet:
                print(f"Data saved to: {output}")
        else:
            print(result.to_csv(index=False), end='')
    elif isinstance(result, dict):
        payload = {
            k: v.to_dict(orient='records') if isinstance(v, pd.DataFrame)
            else v
            for k, v in result.items()
        }
        text = json.dumps(payload, indent=2, default=str)
        if output:
            try:
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                Path(output).write_text(text, encoding='utf-8')
            except OSError as exc:
                print(
                    f"Error: Failed to write data to {output}: {exc}",
                    file=sys.stderr,
                )
                return 1
            if not quiet:
                print(f"Data saved to: {output}")
        else:
            print(text)
    else:
        print(
            f"Error: Unexpected return_data type: {type(result).__name__}",
            file=sys.stderr,
        )
        return 1

    return 0


# ---------------------------------------------------------------------------
# Main command handler
# ---------------------------------------------------------------------------

def cmd_plot(args) -> int:
    """Generate a plot from a saved learner session."""
    quiet = getattr(args, 'quiet', False)

    # --list or no type: show available types and exit
    if getattr(args, 'list', False) or args.type is None:
        _print_plot_list()
        return 0

    plot_type = args.type
    if plot_type not in _PLOT_REGISTRY:
        available = ', '.join(sorted(_PLOT_REGISTRY))
        print(
            f"Error: Unknown plot type '{plot_type}'. "
            f"Available: {available}",
            file=sys.stderr,
        )
        return 1

    spec = _PLOT_REGISTRY[plot_type]

    # Load learner (required)
    learner = _load_learner_if_requested(args)
    if learner is None:
        print("Error: --load-learner is required for plot", file=sys.stderr)
        return 1

    # Check prerequisite
    err = _check_prerequisite(learner, spec, plot_type)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    # Validate type-specific flags
    err = _validate_type_flags(args, spec, plot_type)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    # Validate --model names against loaded learner
    models_arg = getattr(args, 'models', None)
    model_param = spec.model_param

    if models_arg and model_param is None:
        print(
            f"Error: --model is not valid for plot type '{plot_type}'",
            file=sys.stderr,
        )
        return 1

    if models_arg:
        available_models = set(learner.models.keys())
        for m in models_arg:
            if m not in available_models:
                print(
                    f"Error: Model '{m}' not found in loaded learner. "
                    f"Available: {', '.join(sorted(available_models)) or '(none)'}",
                    file=sys.stderr,
                )
                return 1

    if model_param == 'model' and models_arg and len(models_arg) > 1:
        print(
            f"Error: Plot type '{plot_type}' accepts at most one model "
            f"(got {len(models_arg)})",
            file=sys.stderr,
        )
        return 1

    # Build kwargs
    try:
        kwargs = spec.build_kwargs(args)
    except (ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Inject model parameter
    if models_arg:
        if model_param == 'model':
            kwargs['model'] = models_arg[0]
        else:
            kwargs['models'] = models_arg

    # Inject return_data
    return_data = getattr(args, 'return_data', False)
    if return_data:
        kwargs['return_data'] = True

    # Call the plot method
    try:
        result = getattr(learner, spec.method)(**kwargs)
    except (MemoryError, RecursionError):
        raise
    except (ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (RuntimeError, AttributeError) as exc:
        print(f"Error: Plot generation failed: {exc}", file=sys.stderr)
        return 1

    # Output
    if return_data:
        return _handle_return_data(result, args, quiet)

    fig = result
    fmt = getattr(args, 'format', 'html')
    output = getattr(args, 'output', None)
    no_open = getattr(args, 'no_open', False)

    if output:
        from mlsurv.reports._figure_export import FigureExportTimeout
        try:
            _write_figure(fig, output, fmt)
        except (OSError, FigureExportTimeout) as exc:
            print(
                f"Error: Failed to write plot to {output}: {exc}",
                file=sys.stderr,
            )
            return 1
        if not quiet:
            print(f"Plot saved to: {output}")
        _maybe_open_browser(output, fmt, no_open, quiet)
    else:
        # No output file — show in browser directly
        fig.show()

    return 0
