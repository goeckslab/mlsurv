"""Argument parser construction for the mlsurv CLI."""

from __future__ import annotations

import argparse

from mlsurv import __version__
from mlsurv.constants import DEFAULT_SCORING_METRIC, SCORABLE_METRICS
from mlsurv.models import list_models
from mlsurv.tuning import TUNER_REGISTRY


def _get_all_models():
    """Get all available model names."""
    return list_models(include_unavailable=False)


def _get_tunable_models():
    """Get models that have tuners available and dependencies installed."""
    available = set(_get_all_models())
    return [m for m in TUNER_REGISTRY.keys() if m in available]


def _parse_n_bootstrap(value: str) -> int | str:
    """Parse bootstrap iteration argument supporting positive ints or 'auto'."""
    if isinstance(value, str) and value.lower() == 'auto':
        return 'auto'
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer or 'auto'") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _parse_n_trials(value: str) -> int | str:
    """Parse Optuna trials argument supporting positive ints or 'auto'."""
    if isinstance(value, str) and value.lower() == 'auto':
        return 'auto'
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer or 'auto'") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _parse_sweep_value(value: str):
    """Parse a single sweep value: try int, then float, then keep as string."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def create_parser() -> argparse.ArgumentParser:
    """Create the top-level parser and wire subcommands."""
    parser = argparse.ArgumentParser(
        prog='mlsurv',
        description='Low-code survival machine learning analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline in one command
  mlsurv run data.csv --output report.html

  # Train one model with default hyperparameters
  mlsurv train data.csv --model rsf

  # Tune a specific model
  mlsurv tune data.csv --model rsf --n-trials 50

  # Validate on an external cohort
  mlsurv validate train.csv --validation-data external.csv --model coxph

  # Predict risk scores from a saved artifact
  mlsurv predict patients.csv --load-artifact artifact_dir

  # Predict survival/risk at specific horizons from a saved artifact
  mlsurv predict patients.csv --load-artifact artifact_dir --times 12 24

Exit Codes:
  0    Success
  1    General error (unexpected failure)
  2    Usage error (invalid arguments)
  3    Data error (file not found, parse failure, validation)
  4    Computation error (model training or evaluation failure)
  130  Interrupted (Ctrl+C)

Shell Completion (optional):
  pip install mlsurv[completion]
  eval "$(register-python-argcomplete mlsurv)"  # Bash/Zsh
"""
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    parser.add_argument(
        '-q', '--quiet', action='store_true', default=False,
        help='Suppress all non-essential output (for scripting/automation)',
    )

    sub = parser.add_subparsers(dest='command', help='Available commands')
    _add_run_parser(sub)
    _add_train_parser(sub)
    _add_tune_parser(sub)
    _add_evaluate_parser(sub)
    _add_calibrate_parser(sub)
    _add_bootstrap_parser(sub)
    _add_validate_parser(sub)
    _add_predict_parser(sub)
    _add_setup_parser(sub)
    _add_models_parser(sub)
    _add_export_study_parser(sub)
    _add_list_studies_parser(sub)
    _add_benchmark_parser(sub)
    _add_sweep_parser(sub)
    _add_artifact_inspect_parser(sub)
    _add_artifact_upgrade_parser(sub)
    _add_explain_parser(sub)
    _add_patient_report_parser(sub)
    _add_ph_test_parser(sub)
    _add_logrank_test_parser(sub)
    _add_rmst_test_parser(sub)
    _add_importance_test_parser(sub)
    _add_linearity_test_parser(sub)
    _add_interactions_parser(sub)
    _add_export_results_parser(sub)
    _add_plot_parser(sub)
    return parser


# ---------------------------------------------------------------------------
# Shared argument groups
# ---------------------------------------------------------------------------


def _add_format_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``--format`` flag for machine-readable output."""
    parser.add_argument(
        '--format', dest='output_format', type=str,
        choices=['table', 'json', 'csv'], default='table',
        help='Output format: table (default), json, or csv',
    )


def _add_data_args(
    parser: argparse.ArgumentParser,
    *,
    optional_data: bool = False,
) -> None:
    """Add common data-related arguments to a parser.

    When *optional_data* is ``True`` the ``data`` positional becomes
    optional (``nargs='?'``).  Commands that accept ``--load-learner``
    use this so the data file can be omitted when a saved learner
    already contains training data.
    """
    if optional_data:
        parser.add_argument(
            'data', type=str, nargs='?', default=None,
            help='Path to CSV data file (not needed with --load-learner)',
        )
    else:
        parser.add_argument(
            'data', type=str, help='Path to CSV data file',
        )
    parser.add_argument(
        '--time-col', type=str, default='time',
        help='Column name for survival time (default: time)',
    )
    parser.add_argument(
        '--event-col', type=str, default='event',
        help='Column name for event indicator (default: event)',
    )
    parser.add_argument(
        '--event-values', nargs='+', default=None,
        help='Values in the event column that indicate an event occurred '
             '(e.g., --event-values deceased progressed). '
             'When omitted, auto-detection is used.',
    )
    parser.add_argument(
        '--start-col', type=str, default=None,
        help='Column name for start date (use with --end-col instead of --time-col)',
    )
    parser.add_argument(
        '--end-col', type=str, default=None,
        help='Column name for end date (use with --start-col)',
    )
    parser.add_argument(
        '--time-unit', type=str, default='days',
        choices=['days', 'months', 'years'],
        help='Time unit for date-based survival time (default: days)',
    )
    parser.add_argument(
        '--test-fraction', type=float, default=0.3,
        help='Fraction of data for testing (default: 0.3)',
    )
    parser.add_argument(
        '--random-state', type=int, default=42,
        help='Random seed (default: 42)',
    )
    parser.add_argument(
        '--group-col', type=str,
        help='Group column for GroupKFold cross-validation',
    )
    parser.add_argument(
        '--cv-strategy', type=str,
        choices=['stratifiedkfold', 'kfold', 'groupkfold', 'timeseries'],
        default='stratifiedkfold',
        help='Cross-validation strategy (default: stratifiedkfold)',
    )
    parser.add_argument(
        '--cv-repeats', type=int, default=1,
        help='Number of repeated CV runs (default: 1)',
    )
    parser.add_argument(
        '--cv-ci', type=float, default=0.95,
        help='Confidence level for CV intervals (default: 0.95)',
    )
    parser.add_argument(
        '--imputer', type=str,
        choices=['median', 'mean', 'knn', 'iterative', 'none'], default='median',
        help='Imputer preset (default: median)',
    )
    parser.add_argument(
        '--scaler', type=str,
        choices=['standard', 'robust', 'minmax', 'pca', 'none'],
        default='standard',
        help='Scaler preset (default: standard)',
    )
    parser.add_argument(
        '--feature-selector', type=str,
        choices=['topk', 'vif', 'correlation', 'none'], default='none',
        help='Feature selector preset (default: none)',
    )
    parser.add_argument(
        '--scoring',
        choices=SCORABLE_METRICS,
        default=DEFAULT_SCORING_METRIC,
        help=f'Primary scoring metric (default: {DEFAULT_SCORING_METRIC})',
    )
    parser.add_argument(
        '--n-jobs', type=int, default=1,
        help='Number of parallel jobs (default: 1; use -1 for all cores)',
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help="Device for deep learning models: 'cpu', 'cuda', 'cuda:0', or 'auto'. "
             "If not specified, models default to 'auto' (GPU if available).",
    )
    parser.add_argument(
        '--verbose', action='store_true', help='Print detailed output',
    )


def _add_report_args(parser: argparse.ArgumentParser) -> None:
    """Add report-customization flags shared across report-producing commands."""
    g = parser.add_argument_group('report customization')
    g.add_argument(
        '--sections', nargs='+',
        help='Report sections to include (e.g., summary evaluation feature_analysis)',
    )
    g.add_argument(
        '--outcome-name', type=str,
        help='Custom label for the outcome variable (e.g., "death")',
    )
    g.add_argument(
        '--manuscript', action='store_true',
        help='Generate manuscript-ready companion markdown',
    )
    g.add_argument(
        '--feature-groups-file', type=str,
        help='Path to JSON file mapping features to groups for importance plots',
    )


def _add_artifact_args(
    parser: argparse.ArgumentParser,
    *,
    save: bool = False,
    load: bool = False,
) -> None:
    """Add --save-artifact and/or --load-artifact flags to a subparser.

    Model-producing steps (train, tune, run) get ``save`` only.
    Model-consuming steps (evaluate, bootstrap) get both ``save`` and ``load``.
    Validate gets ``load`` only (no model produced). Setup gets ``save`` only
    (config-only artifact, no model file).
    """
    if save:
        parser.add_argument(
            '--save-artifact', type=str, metavar='DIR',
            help='Save model artifact (metadata + model) to a directory',
        )
    if load:
        parser.add_argument(
            '--load-artifact', type=str, metavar='DIR',
            help='Load a previously saved model artifact (skips training). '
                 'CV metrics from the original training run are displayed '
                 'if available. '
                 'Only load artifacts you trust — uses pickle internally.',
        )


def _add_learner_args(
    parser: argparse.ArgumentParser,
    *,
    save: bool = False,
    load: bool = False,
) -> None:
    """Add --save-learner and/or --load-learner flags to a subparser.

    All learner-creating commands get ``save``.  Commands that can
    operate on a restored session (evaluate, bootstrap, validate) also
    get ``load``.
    """
    if save:
        parser.add_argument(
            '--save-learner', type=str, metavar='FILE',
            help='Save entire learner state (data, models, results) to a '
                 'file. Recommended extension: .learner',
        )
    if load:
        parser.add_argument(
            '--load-learner', type=str, metavar='FILE',
            help='Load a previously saved learner (skips data loading, '
                 'setup, and training). '
                 'Only load files you trust — uses pickle internally.',
        )


# ---------------------------------------------------------------------------
# Per-subcommand builders
# ---------------------------------------------------------------------------

def _add_run_parser(sub) -> None:
    p = sub.add_parser(
        'run',
        help='Run full pipeline: compare + tune (default; --no-tune to skip) + '
             'evaluate + bootstrap',
    )
    _add_data_args(p)
    all_models = _get_all_models()
    p.add_argument(
        '--models', nargs='+', default=['coxph', 'rsf'],
        help=f"Models to run (default: coxph rsf). Available: {', '.join(all_models)}",
    )
    p.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds (default: 5)')
    p.add_argument(
        '--n-bootstrap', type=_parse_n_bootstrap, default='auto',
        help='Number of bootstrap iterations or auto (default: auto)',
    )
    p.add_argument('--output', type=str, help='Optional output HTML report path')
    p.add_argument(
        '--compare', action='store_true', default=False,
        help='Print paired model comparison table (requires bootstrap)',
    )
    p.add_argument(
        '--compare-output', type=str, default=None, metavar='FILE',
        help='Save comparison results to CSV (requires bootstrap)',
    )
    p.add_argument('--no-tune', dest='tune', action='store_false',
                   help='Skip hyperparameter tuning and use default parameters '
                        '(equivalent to the API run(tune=False))')
    p.set_defaults(tune=True)
    p.add_argument('--n-trials', type=_parse_n_trials, default='auto',
                   help="Optuna trials per tuned model, or 'auto' (default: auto). "
                        "Only used when tuning is enabled.")
    p.add_argument('--resume', action='store_true',
                   help='Resume existing Optuna studies if available')
    p.add_argument('--experiment-name', type=str, default=None,
                   help='Experiment name for Optuna studies (required for --resume)')
    _add_format_arg(p)
    _add_report_args(p)
    _add_artifact_args(p, save=True)
    _add_learner_args(p, save=True)


def _add_train_parser(sub) -> None:
    p = sub.add_parser('train', help='Train a single model with default hyperparameters')
    _add_data_args(p)
    all_models = _get_all_models()
    p.add_argument(
        '--model', type=str, required=True, choices=all_models,
        help=f"Model to train. Available: {', '.join(all_models)}",
    )
    p.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds (default: 5)')
    p.add_argument(
        '--model-params', type=str, metavar='FILE',
        help='Path to a JSON file of constructor kwargs, nested as '
             '{model_name: {param: value}} (applied with default, untuned '
             'training)',
    )
    p.add_argument('--output', type=str, help='Output CSV file for CV metrics')
    p.add_argument('--report', type=str, help='Optional output HTML report path')
    _add_format_arg(p)
    _add_report_args(p)
    _add_artifact_args(p, save=True)
    _add_learner_args(p, save=True)


def _add_tune_parser(sub) -> None:
    p = sub.add_parser('tune', help='Tune hyperparameters for a model')
    _add_data_args(p)
    tunable_models = _get_tunable_models()
    p.add_argument(
        '--model', type=str, required=True, choices=tunable_models,
        help=f"Model to tune. Available: {', '.join(tunable_models)}",
    )
    p.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials (default: 50)')
    p.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds (default: 5)')
    p.add_argument('--output', type=str, help='Output JSON file for best parameters')
    p.add_argument('--report', type=str, help='Optional output HTML report path')
    p.add_argument('--timeout', type=float, help='Maximum tuning time in seconds')
    p.add_argument('--resume', action='store_true', help='Resume an existing Optuna study if available')
    p.add_argument('--experiment-name', type=str, default=None,
                   help='Experiment name for Optuna studies (required for --resume)')
    p.add_argument('--optuna-verbose', action='store_true', help='Show trial-by-trial Optuna logs')
    p.add_argument('--search-spaces', type=str, help='Path to JSON file containing search spaces')
    _add_format_arg(p)
    _add_report_args(p)
    _add_artifact_args(p, save=True)
    _add_learner_args(p, save=True)


def _add_evaluate_parser(sub) -> None:
    p = sub.add_parser('evaluate', help='Evaluate a trained model')
    _add_data_args(p, optional_data=True)
    all_models = _get_all_models()
    p.add_argument(
        '--model', type=str, required=True, choices=all_models,
        help=f"Model to evaluate. Available: {', '.join(all_models)}",
    )
    p.add_argument('--output', type=str, help='Output CSV file for evaluation results')
    p.add_argument('--report', type=str, help='Optional output HTML report path')
    p.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds (default: 5)')
    p.add_argument(
        '--n-importance-repeats', type=int, default=10,
        help='Number of permutation importance repeats (default: 10)',
    )
    p.add_argument(
        '--eval-times', nargs='+', type=float,
        help='Evaluation times for time-dependent metrics (space-separated)',
    )
    p.add_argument('--shap', action='store_true', help='Compute SHAP values during evaluation')
    p.add_argument(
        '--shap-max-background', type=int, default=50,
        help='Maximum background samples for SHAP masker (default: 50)',
    )
    p.add_argument(
        '--shap-max-explain', type=int, default=50,
        help='Maximum test samples to explain with SHAP (default: 50)',
    )
    p.add_argument('--bootstrap', action='store_true', help='Compute bootstrap confidence intervals')
    p.add_argument(
        '--n-bootstrap', type=_parse_n_bootstrap, default='auto',
        help='Number of bootstrap iterations or auto (default: auto)',
    )
    p.add_argument(
        '--interactions', action='store_true',
        help='Compute pairwise feature interactions',
    )
    p.add_argument(
        '--interaction-top-k', type=int, default=5,
        help='Number of top features for interaction testing (default: 5)',
    )
    p.add_argument(
        '--subgroups', nargs='+', metavar='COLUMN',
        help='Column(s) for subpopulation analysis',
    )
    # Statistical test flags (run after evaluation)
    p.add_argument('--ph-test', action='store_true',
                   help='Run proportional hazards test (Schoenfeld residuals)')
    p.add_argument('--linearity-test', action='store_true',
                   help='Run log-hazard linearity test via restricted cubic splines')
    p.add_argument('--logrank-test', action='store_true',
                   help='Run log-rank test across risk groups')
    p.add_argument('--rmst-test', action='store_true',
                   help='Run RMST comparison across risk groups')
    p.add_argument('--importance-test', action='store_true',
                   help='Compute feature importance p-values (PIMP)')
    p.add_argument(
        '--plot', nargs='+', type=str, default=None, metavar='TYPE',
        help='Generate plots after evaluation (e.g. --plot metric importance roc). '
             'Use "mlsurv plot --list" to see available types.',
    )
    _add_format_arg(p)
    _add_report_args(p)
    _add_artifact_args(p, save=True, load=True)
    _add_learner_args(p, save=True, load=True)


def _add_calibrate_parser(sub) -> None:
    p = sub.add_parser(
        'calibrate',
        help='Attach S(t|x) calibrators to trained models '
             '(modern train(calibrate=...) path)',
    )
    _add_data_args(p, optional_data=True)
    all_models = _get_all_models()
    p.add_argument(
        '--models', nargs='+', default=None,
        help=f"Models to calibrate (default: all trained). "
             f"Available: {', '.join(all_models)}",
    )
    p.add_argument(
        '--method', type=str, required=True,
        choices=['isotonic', 'lowdof', 'breslow_slope'],
        help="Calibration method (required). 'breslow_slope' is only valid "
             "for score-based models.",
    )
    p.add_argument(
        '--output', type=str,
        help='Output CSV file summarizing calibrated models',
    )
    p.add_argument('--report', type=str, help='Optional output HTML report path')
    _add_format_arg(p)
    _add_report_args(p)
    _add_artifact_args(p, save=True, load=True)
    _add_learner_args(p, save=True, load=True)


def _add_bootstrap_parser(sub) -> None:
    p = sub.add_parser(
        'bootstrap', help='Train or load a model, then compute bootstrap confidence intervals',
    )
    _add_data_args(p, optional_data=True)
    all_models = _get_all_models()
    p.add_argument(
        '--model', type=str, required=True, choices=all_models,
        help=f"Model to bootstrap. Available: {', '.join(all_models)}",
    )
    p.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds (default: 5)')
    p.add_argument(
        '--n-bootstrap', type=_parse_n_bootstrap, default='auto',
        help='Number of bootstrap iterations or auto (default: auto)',
    )
    p.add_argument(
        '--eval-times', nargs='+', type=float,
        help='Evaluation times for time-dependent metrics (space-separated)',
    )
    p.add_argument('--output', type=str, help='Output CSV file for bootstrap summary')
    p.add_argument('--report', type=str, help='Optional output HTML report path')
    p.add_argument(
        '--plot', nargs='+', type=str, default=None, metavar='TYPE',
        help='Generate plots after bootstrap (e.g. --plot metric importance). '
             'Use "mlsurv plot --list" to see available types.',
    )
    _add_format_arg(p)
    _add_report_args(p)
    _add_artifact_args(p, save=True, load=True)
    _add_learner_args(p, save=True, load=True)


def _add_validate_parser(sub) -> None:
    p = sub.add_parser(
        'validate', help='Train or load a model, then validate on an external cohort',
    )
    _add_data_args(p, optional_data=True)
    p.add_argument(
        '--validation-data', type=str, required=True,
        help='Path to external validation CSV file',
    )
    p.add_argument(
        '--val-time-col', type=str, default='time',
        help='Validation time column (default: time)',
    )
    p.add_argument(
        '--val-start-col', type=str, default=None,
        help='Validation start date column (use with --val-end-col instead of --val-time-col)',
    )
    p.add_argument(
        '--val-end-col', type=str, default=None,
        help='Validation end date column (use with --val-start-col)',
    )
    p.add_argument(
        '--val-time-unit', type=str, default=None,
        choices=['days', 'months', 'years'],
        help='Time unit for validation date-based survival time. '
             'Defaults to --time-unit when omitted.',
    )
    p.add_argument(
        '--val-event-col', type=str, default='event',
        help='Validation event column (default: event)',
    )
    p.add_argument(
        '--val-event-values', nargs='+', default=None,
        help='Values in the validation event column that indicate an event '
             'occurred. Defaults to --event-values when omitted.',
    )
    all_models = _get_all_models()
    p.add_argument(
        '--model', type=str, required=True, choices=all_models,
        help=f"Model to validate. Available: {', '.join(all_models)}",
    )
    p.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds (default: 5)')
    p.add_argument(
        '--cohort-name', type=str, default='External',
        help='External cohort display name (default: External)',
    )
    p.add_argument(
        '--bootstrap', dest='bootstrap', action='store_true',
        help='Enable bootstrap confidence intervals (default: enabled)',
    )
    p.add_argument(
        '--no-bootstrap', dest='bootstrap', action='store_false',
        help='Disable bootstrap confidence intervals',
    )
    p.set_defaults(bootstrap=True)
    p.add_argument(
        '--n-bootstrap', type=_parse_n_bootstrap, default='auto',
        help='Number of bootstrap iterations or auto (default: auto)',
    )
    p.add_argument(
        '--references-file', type=str, default=None,
        help='Path to a CSV file containing reference biomarker score columns '
             '(one row per validation patient, same order as --validation-data). '
             'Reference rows are appended to the concordance output alongside models.',
    )
    p.add_argument(
        '--reference-cols', nargs='+', default=None,
        help='Columns to use from the references file (default: all columns)',
    )
    p.add_argument(
        '--higher-is-worse', dest='higher_is_worse', action='store_true',
        help='Reference values where higher = worse outcome (higher risk). Default.',
    )
    p.add_argument(
        '--no-higher-is-worse', dest='higher_is_worse', action='store_false',
        help='Reference values where higher = better outcome (lower risk), '
             'e.g., LORIS / TMB / PD-L1. Scores are sign-flipped internally.',
    )
    p.set_defaults(higher_is_worse=True)
    p.add_argument('--output', type=str, help='Output CSV file for validation concordance')
    p.add_argument('--report', type=str, help='Optional output HTML report path')
    _add_format_arg(p)
    _add_report_args(p)
    _add_artifact_args(p, load=True)
    _add_learner_args(p, save=True, load=True)


def _add_predict_parser(sub) -> None:
    p = sub.add_parser(
        'predict',
        help='Predict risk scores and optional survival/risk at specific time points',
    )
    p.add_argument('data', type=str, help='Path to CSV data file for new patients')
    p.add_argument(
        '--load-artifact', type=str, metavar='DIR',
        help='Load a previously saved model artifact (no CI support). '
             'Only load artifacts you trust — uses pickle internally.',
    )
    _add_learner_args(p, load=True)
    p.add_argument(
        '--model', type=str, default=None,
        help='Model name for prediction (omit to predict with all trained '
             'models: one trained model gives the standard table, several '
             'give a combined table with one block of rows per model and '
             'model_name as the first column; ignored with --load-artifact)',
    )
    p.add_argument(
        '--times', nargs='+', type=float,
        help='Optional time points at which to generate survival/risk predictions',
    )
    p.add_argument('--output', type=str, help='Optional output CSV file for predictions')
    p.add_argument(
        '--strict', action='store_true', default=False,
        help='Exit with an error when any selected model fails to predict, '
             'or when --times is requested but survival columns cannot be '
             'produced; with several models, any one model failing fails the '
             'command (default: graceful degradation with rc=0; models '
             'without survival columns keep risk_score only, models that '
             'fail entirely are omitted with a warning)',
    )
    p.add_argument(
        '--ci', action='store_true', default=False,
        help='Compute bootstrap confidence intervals (requires --load-learner)',
    )
    p.add_argument(
        '--ci-band', action='store_true', default=False,
        help='Also retain bootstrap survival CI bands for plotting '
             '(requires --ci)',
    )
    p.add_argument(
        '--n-bootstrap', type=int, default=200,
        help='Number of bootstrap iterations for CIs (default: 200)',
    )
    p.add_argument(
        '--ci-level', type=float, default=0.95,
        help='Confidence level for bootstrap CIs (default: 0.95)',
    )
    p.add_argument(
        '--random-state', type=int, default=42,
        help='Random seed for bootstrap reproducibility (default: 42)',
    )
    p.add_argument(
        '--n-jobs', type=int, default=1,
        help='Number of parallel jobs for bootstrap (default: 1)',
    )
    p.add_argument('--verbose', action='store_true', help='Print detailed output')
    _add_format_arg(p)


def _add_setup_parser(sub) -> None:
    p = sub.add_parser(
        'setup', help='Inspect dataset and split/CV setup without training models',
    )
    _add_data_args(p)
    p.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds (default: 5)')
    _add_format_arg(p)
    _add_artifact_args(p, save=True)
    _add_learner_args(p, save=True)


def _add_models_parser(sub) -> None:
    p = sub.add_parser('models', help='List available model names')
    p.add_argument('--verbose', action='store_true', help='Also print whether each model is tunable')
    _add_format_arg(p)


def _add_export_study_parser(sub) -> None:
    p = sub.add_parser('export-study', help='Export Optuna tuning history to CSV or JSON')
    p.add_argument('--experiment', type=str, default=None, help='Experiment name')
    p.add_argument('--model', type=str, default=None, help='Model name (e.g., rsf)')
    p.add_argument(
        '--study-name', type=str, default=None,
        help='Full Optuna study name (overrides --experiment/--model)',
    )
    p.add_argument('--storage', type=str, default=None, help='Optuna storage URL')
    p.add_argument(
        '--format', choices=['csv', 'json'], default=None,
        help='Output format (default: inferred from --output extension, or json)',
    )
    p.add_argument('--output', type=str, required=True, help='Output file path')


def _add_list_studies_parser(sub) -> None:
    p = sub.add_parser(
        'list-studies',
        help='List Optuna tuning studies and where their .db files live',
    )
    p.add_argument(
        '--experiment', type=str, default=None,
        help='Filter to one experiment (default: list all experiments)',
    )
    p.add_argument('--storage', type=str, default=None, help='Optuna storage URL')
    _add_format_arg(p)


def _add_benchmark_parser(sub) -> None:
    p = sub.add_parser(
        'benchmark',
        help='Compare trained models against reference scores from an external file',
    )
    _add_data_args(p)
    all_models = _get_all_models()
    p.add_argument(
        '--models', nargs='+', default=['coxph', 'rsf'],
        help=f"Models to compare (default: coxph rsf). Available: {', '.join(all_models)}",
    )
    p.add_argument(
        '--reference-file', type=str, required=True,
        help='Path to a CSV file containing reference score columns '
             '(one row per patient, same order as data file)',
    )
    p.add_argument(
        '--reference-cols', nargs='+', default=None,
        help='Columns to use from the reference file (default: all columns)',
    )
    p.add_argument(
        '--dataset', type=str, default='test',
        choices=['test', 'train', 'full'],
        help='Data split to evaluate the references on (default: test). '
             'Models always use the held-out test set.',
    )
    p.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds (default: 5)')
    p.add_argument(
        '--n-bootstrap', type=_parse_n_bootstrap, default=200,
        help='Number of bootstrap iterations (default: 200)',
    )
    p.add_argument('--no-bootstrap', dest='bootstrap', action='store_false',
                   help='Disable bootstrap CIs')
    p.set_defaults(bootstrap=True)
    p.add_argument(
        '--subgroups', nargs='+', metavar='COLUMN',
        help='Column(s) for subpopulation analysis',
    )
    p.add_argument('--output', type=str, help='Output CSV file for benchmark concordance')
    p.add_argument('--report', type=str, help='Optional output HTML report path')
    _add_format_arg(p)
    _add_report_args(p)
    _add_artifact_args(p, save=True)
    _add_learner_args(p, save=True)


def _add_sweep_parser(sub) -> None:
    p = sub.add_parser(
        'sweep', help='Sweep a parameter across values and measure CV performance',
    )
    _add_data_args(p)
    all_models = _get_all_models()
    p.add_argument(
        '--models', nargs='+', required=True,
        help=f"Models to sweep. Available: {', '.join(all_models)}",
    )
    p.add_argument(
        '--param', type=str, required=True,
        help='Parameter name to sweep (e.g., n_estimators)',
    )
    p.add_argument(
        '--values', nargs='+', type=_parse_sweep_value, required=True,
        help='Values to try (space-separated, e.g., 50 100 200 500)',
    )
    p.add_argument(
        '--target', type=str, default='model',
        choices=['model', 'imputer', 'scaler', 'feature_selector'],
        help='Pipeline component the parameter belongs to (default: model)',
    )
    p.add_argument(
        '--include-test', action='store_true',
        help='Also evaluate on held-out test set',
    )
    p.add_argument(
        '--n-bootstrap', type=int, default=0,
        help='Bootstrap iterations for test-set CIs (default: 0, requires --include-test)',
    )
    p.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds (default: 5)')
    p.add_argument('--output', type=str, help='Output CSV file for sweep results')
    p.add_argument('--report', type=str, help='Optional output HTML report path')
    _add_format_arg(p)
    _add_report_args(p)


def _add_artifact_inspect_parser(sub) -> None:
    p = sub.add_parser(
        'artifact-inspect', help='Inspect artifact metadata without loading the model',
    )
    p.add_argument('dir', type=str, help='Path to artifact directory')
    p.add_argument(
        '--json', dest='as_json', action='store_true',
        help='Output raw JSON instead of human-readable summary',
    )


def _add_artifact_upgrade_parser(sub) -> None:
    p = sub.add_parser(
        'artifact-upgrade', help='Upgrade artifact metadata to the current schema',
    )
    p.add_argument('dir', type=str, help='Path to artifact directory')
    p.add_argument(
        '--output', type=str, metavar='DIR',
        help='Write upgraded artifact to a new directory (source untouched)',
    )
    p.add_argument('--dry-run', action='store_true', help='Show what would change without writing')


def _add_explain_parser(sub) -> None:
    p = sub.add_parser(
        'explain',
        help='Generate SHAP waterfall explanation for an individual patient',
    )
    p.add_argument(
        '--load-learner', type=str, required=True, metavar='FILE',
        help='Load a previously saved learner (must contain at least one '
             'trained model). '
             'Only load files you trust — uses pickle internally.',
    )
    patient = p.add_mutually_exclusive_group(required=True)
    patient.add_argument(
        '--patient', type=int, metavar='N',
        help='Positional index into the test set (0-based)',
    )
    patient.add_argument(
        '--patient-data', type=str, metavar='PATH',
        help='Path to a 1-row CSV file with patient features',
    )
    p.add_argument(
        '--model', type=str, default=None,
        help='Model name (auto-resolved if only one trained)',
    )
    p.add_argument(
        '--n-features', type=int, default=15,
        help='Maximum features in waterfall plot (default: 15)',
    )
    p.add_argument(
        '--max-background', type=int, default=50,
        help='Maximum background samples for SHAP explainer (default: 50)',
    )
    p.add_argument(
        '--plot', type=str, default=None, metavar='PATH',
        help='Save waterfall plot to HTML file',
    )
    p.add_argument(
        '--output', type=str, default=None, metavar='PATH',
        help='Save SHAP values table to CSV file',
    )
    _add_format_arg(p)


def _add_patient_report_parser(sub) -> None:
    p = sub.add_parser(
        'patient-report',
        help='Generate a standalone interactive single-patient HTML report',
    )
    p.add_argument(
        '--load-learner', type=str, required=True, metavar='FILE',
        help='Load a previously saved learner (must contain at least one '
             'trained model). '
             'Only load files you trust — uses pickle internally.',
    )
    patient = p.add_mutually_exclusive_group(required=True)
    patient.add_argument(
        '--patient', type=str, metavar='INDEX_OR_SPEC',
        help="Row index into the test set (0-based) or a risk-percentile "
             "spec: 'low_risk'/'p10', 'high_risk'/'p90', 'median'/'p50', "
             "or 'pNN' (NN in 0..100)",
    )
    patient.add_argument(
        '--patient-data', type=str, metavar='PATH',
        help='Path to a CSV file with patient features (the report focuses '
             'on the first row)',
    )
    p.add_argument(
        '--model', type=str, default=None,
        help='Model name (auto-resolved if only one trained)',
    )
    p.add_argument(
        '--output', type=str, required=True, metavar='PATH',
        help='Write the HTML report to this file (.html)',
    )
    p.add_argument(
        '--times', nargs='+', type=float,
        help='Milestone times for P(survive > t) annotations (default: '
             'derived from the time unit)',
    )
    p.add_argument(
        '--time-unit', type=str, default=None,
        help="Time unit of the outcome ('days', 'weeks', 'months', 'years'; "
             'plural/singular/abbreviations accepted). Falls back to the '
             "learner's setup() value",
    )
    p.add_argument(
        '--n-neighbors', type=int, default=10,
        help='K for the nearest-neighbor training-cohort anchor (default: 10)',
    )
    p.add_argument(
        '--n-bootstrap', type=int, default=200,
        help='Bootstrap iterations for confidence intervals (default: 200)',
    )
    p.add_argument(
        '--no-ci', action='store_false', dest='ci', default=True,
        help='Skip bootstrap confidence intervals (faster)',
    )
    p.add_argument(
        '--no-ci-band', action='store_false', dest='ci_band', default=True,
        help='Skip the shaded CI band around the survival curve',
    )
    p.add_argument(
        '--no-recalibrated', action='store_false', dest='use_recalibrated',
        default=True,
        help='Show raw model S(t|x) instead of the recalibrated curve when a '
             'calibrator is attached',
    )
    p.add_argument(
        '--offline', action='store_true',
        help='Embed plotly.js into the HTML (larger file, works without '
             'internet)',
    )
    p.add_argument(
        '--random-state', type=int, default=None,
        help='Random seed for bootstrap reproducibility',
    )


# ---------------------------------------------------------------------------
# Standalone statistical test commands
# ---------------------------------------------------------------------------

_P_ADJUST_CHOICES = ['holm', 'bonferroni', 'bh', 'none']


def _add_ph_test_parser(sub) -> None:
    p = sub.add_parser(
        'ph-test', help='Proportional hazards test (Schoenfeld residuals)',
    )
    p.add_argument(
        '--load-learner', type=str, required=True, metavar='FILE',
        help='Load a saved learner file (required). '
             'Only load files you trust — uses pickle internally.',
    )
    p.add_argument(
        '--time-transform', type=str, default='rank',
        choices=['rank', 'km', 'identity', 'log'],
        help='Time transform for the test (default: rank)',
    )
    p.add_argument('--output', type=str, help='Output CSV file for test results')
    p.add_argument('--verbose', action='store_true', help='Print detailed output')
    _add_format_arg(p)


def _add_logrank_test_parser(sub) -> None:
    p = sub.add_parser(
        'logrank-test', help='Log-rank test across risk groups',
    )
    p.add_argument(
        '--load-learner', type=str, required=True, metavar='FILE',
        help='Load a saved learner file (required). '
             'Only load files you trust — uses pickle internally.',
    )
    all_models = _get_all_models()
    p.add_argument(
        '--model', type=str, choices=all_models, default=None,
        help='Model to use for risk groups (default: all trained models)',
    )
    p.add_argument(
        '--n-groups', type=int, default=2,
        help='Number of risk groups (default: 2)',
    )
    p.add_argument(
        '--p-adjust', type=str, default='holm', choices=_P_ADJUST_CHOICES,
        help='P-value adjustment method for 3+ groups (default: holm)',
    )
    p.add_argument('--output', type=str, help='Output CSV file for test results')
    p.add_argument('--verbose', action='store_true', help='Print detailed output')
    _add_format_arg(p)


def _add_rmst_test_parser(sub) -> None:
    p = sub.add_parser(
        'rmst-test', help='RMST comparison across risk groups',
    )
    p.add_argument(
        '--load-learner', type=str, required=True, metavar='FILE',
        help='Load a saved learner file (required). '
             'Only load files you trust — uses pickle internally.',
    )
    all_models = _get_all_models()
    p.add_argument(
        '--model', type=str, choices=all_models, default=None,
        help='Model to use for risk groups (default: all trained models)',
    )
    p.add_argument(
        '--n-groups', type=int, default=2,
        help='Number of risk groups (default: 2)',
    )
    p.add_argument(
        '--tau', type=float, default=None,
        help='RMST time horizon (default: auto-computed)',
    )
    p.add_argument(
        '--p-adjust', type=str, default='holm', choices=_P_ADJUST_CHOICES,
        help='P-value adjustment method for 3+ groups (default: holm)',
    )
    p.add_argument('--output', type=str, help='Output CSV file for test results')
    p.add_argument('--verbose', action='store_true', help='Print detailed output')
    _add_format_arg(p)


def _add_importance_test_parser(sub) -> None:
    p = sub.add_parser(
        'importance-test',
        help='Feature importance p-values (PIMP or bootstrap)',
    )
    p.add_argument(
        '--load-learner', type=str, required=True, metavar='FILE',
        help='Load a saved learner file (required). '
             'Only load files you trust — uses pickle internally.',
    )
    all_models = _get_all_models()
    p.add_argument(
        '--model', type=str, choices=all_models, default=None,
        help='Model to test (default: best model)',
    )
    p.add_argument(
        '--method', type=str, default='pimp', choices=['pimp', 'bootstrap'],
        help='P-value method (default: pimp)',
    )
    p.add_argument(
        '--n-permutations', type=int, default=100,
        help='Number of response permutations for PIMP (default: 100)',
    )
    p.add_argument(
        '--n-importance-repeats', type=int, default=10,
        help='Permutation importance repeats per model (default: 10)',
    )
    p.add_argument(
        '--p-adjust', type=str, default='bh', choices=_P_ADJUST_CHOICES,
        help='P-value adjustment method (default: bh)',
    )
    p.add_argument(
        '--random-state', type=int, default=None,
        help='Random seed (default: use learner seed)',
    )
    p.add_argument(
        '--n-jobs', type=int, default=None,
        help='Parallel jobs for PIMP (default: use learner setting)',
    )
    p.add_argument('--output', type=str, help='Output CSV file for test results')
    p.add_argument('--verbose', action='store_true', help='Print detailed output')
    _add_format_arg(p)


def _add_linearity_test_parser(sub) -> None:
    p = sub.add_parser(
        'linearity-test',
        help='Test log-hazard linearity of continuous features',
    )
    p.add_argument(
        '--load-learner', type=str, required=True, metavar='FILE',
        help='Load a saved learner file (required). '
             'Only load files you trust — uses pickle internally.',
    )
    p.add_argument(
        '--n-knots', type=int, default=4,
        help='Number of spline knots for the linearity test (default: 4)',
    )
    p.add_argument('--output', type=str, help='Output CSV file for test results')
    p.add_argument('--verbose', action='store_true', help='Print detailed output')
    _add_format_arg(p)


def _add_interactions_parser(sub) -> None:
    p = sub.add_parser(
        'interactions',
        help='Compute pairwise feature interactions (requires evaluate)',
    )
    p.add_argument(
        '--load-learner', type=str, required=True, metavar='FILE',
        help='Load a saved learner file (required). '
             'Only load files you trust — uses pickle internally.',
    )
    all_models = _get_all_models()
    p.add_argument(
        '--model', type=str, choices=all_models, default=None,
        help='Model to analyse (default: the single trained model)',
    )
    p.add_argument(
        '--method', type=str, default='hstat', choices=['hstat', 'shap'],
        help="Interaction method: 'hstat' (model-agnostic) or 'shap' "
             "(tree models only) (default: hstat)",
    )
    p.add_argument(
        '--top-k', type=int, default=5,
        help='Number of top features (by importance) to test (default: 5)',
    )
    p.add_argument(
        '--max-samples', type=int, default=100,
        help='Maximum samples subsampled from the test set (default: 100)',
    )
    p.add_argument(
        '--grid-resolution', type=int, default=25,
        help='Grid points per feature for PDP surfaces (default: 25)',
    )
    p.add_argument(
        '--random-state', type=int, default=None,
        help='Seed for reproducible subsampling (default: use learner seed)',
    )
    p.add_argument('--output', type=str, help='Output CSV file for interactions')
    _add_format_arg(p)


def _add_export_results_parser(sub) -> None:
    p = sub.add_parser(
        'export-results',
        help='Export all result tables to xlsx or csv',
    )
    p.add_argument(
        '--load-learner', type=str, required=True, metavar='FILE',
        help='Load a saved learner file (required). '
             'Only load files you trust — uses pickle internally.',
    )
    p.add_argument(
        'output', type=str,
        help='Output path: an .xlsx file, or a directory for csv',
    )
    p.add_argument(
        '--format', type=str, choices=['xlsx', 'csv'], default=None,
        help='Output format (default: inferred from the output extension)',
    )
    p.add_argument(
        '--audit', action='store_true', default=False,
        help='Include the full sheet-name truncation mapping (xlsx only)',
    )


# ---------------------------------------------------------------------------
# Plot command
# ---------------------------------------------------------------------------

def _add_plot_parser(sub) -> None:
    p = sub.add_parser(
        'plot',
        help='Generate visualizations from a saved learner session',
        description=(
            'Generate visualizations from a saved learner session. '
            'Use --list to see available plot types.'
        ),
    )
    p.add_argument(
        'type', nargs='?', default=None, metavar='TYPE',
        help='Plot type (e.g. survival, metric, importance). '
             'Omit to list available types.',
    )
    p.add_argument(
        '--load-learner', type=str, default=None, metavar='FILE',
        help='Load a previously saved learner (required for plotting). '
             'Only load files you trust — uses pickle internally.',
    )
    p.add_argument(
        '--model', dest='models', nargs='+', type=str, default=None,
        metavar='NAME',
        help='Model(s) to include in the plot (default: all trained)',
    )
    p.add_argument(
        '-o', '--output', type=str, default=None, metavar='PATH',
        help='Output file path (e.g. plot.html, plot.png)',
    )
    p.add_argument(
        '-f', '--format', type=str, default='html',
        choices=['html', 'png', 'json'],
        help='Output format (default: html)',
    )
    p.add_argument(
        '--no-open', action='store_true', default=False,
        help='Do not open HTML output in browser',
    )
    p.add_argument(
        '--return-data', action='store_true', default=False,
        help='Output data (CSV/JSON) instead of a figure',
    )
    p.add_argument(
        '--list', action='store_true', default=False,
        help='List available plot types',
    )
    p.add_argument('--verbose', action='store_true', help='Print detailed output')

    # --- Type-specific flags (validated per-type in cmd_plot) ---

    # Shared across metric, importance
    p.add_argument(
        '--style', type=str, default='auto',
        choices=['auto', 'forest', 'bar', 'raincloud', 'boxplot', 'heatmap'],
        help='Plot style (for metric, importance)',
    )

    # metric-specific
    p.add_argument(
        '--metric', type=str, default=None,
        help='Metric name, e.g. harrell, uno (for metric type)',
    )
    p.add_argument(
        '--source', type=str, default='auto',
        choices=['auto', 'bootstrap', 'test', 'cv'],
        help='Data source (for metric type)',
    )
    p.add_argument(
        '--uncertainty', type=str, default='auto',
        help='Uncertainty display: auto, sd, or CI level like 0.95 (for metric type)',
    )
    p.add_argument(
        '--orientation', type=str, default='horizontal',
        choices=['horizontal', 'vertical'],
        help='Plot orientation (for metric type)',
    )

    # survival-specific
    p.add_argument(
        '--n-groups', type=int, default=None,
        help='Number of risk groups (for survival type)',
    )
    p.add_argument(
        '--show-risk-table', action='store_true', default=False,
        help='Show number-at-risk table (for survival type)',
    )
    p.add_argument(
        '--time-limit', type=float, default=None,
        help='Maximum time on x-axis (for survival type)',
    )

    # importance-specific
    p.add_argument(
        '--n-features', type=int, default=20,
        help='Number of top features to display (default: 20)',
    )

    # calibration-specific
    p.add_argument(
        '--n-bins', type=int, default=10,
        help='Number of calibration bins (default: 10)',
    )

    # roc / calibration shared
    p.add_argument(
        '--eval-time', type=float, default=None,
        help='Evaluation timepoint (for roc, calibration types)',
    )

    # roc-specific
    p.add_argument(
        '--no-auc', action='store_true', default=False,
        help='Suppress AUC annotation (for roc type)',
    )
    p.add_argument(
        '--no-ci', action='store_true', default=False,
        help='Suppress CI ribbons (for roc type)',
    )

    # shared: include benchmark references
    p.add_argument(
        '--references', action='store_true', default=False,
        help='Include benchmark references (for survival, metric, roc, metric-time)',
    )

    # --- Phase 2 type-specific flags ---

    # diagnostics / hazard-ratios / shap
    p.add_argument(
        '--kind', type=str, default=None,
        help='Sub-type: schoenfeld|deviance|martingale (diagnostics), '
             'univariate|multivariate (hazard-ratios), '
             'or importance|beeswarm (shap)',
    )

    # diagnostics / feature-distributions
    p.add_argument(
        '--features', nargs='+', type=str, default=None, metavar='FEAT',
        help='Feature names to include (for diagnostics, feature-distributions)',
    )

    # metric-time
    p.add_argument(
        '--metrics', nargs='+', type=str, default=None, metavar='METRIC',
        help='Metric keys for multi-panel display (for metric-time)',
    )

    # feature-correlation
    p.add_argument(
        '--threshold', type=float, default=0.8,
        help='Correlation threshold (default: 0.8, for feature-correlation)',
    )

    # --- Phase 3 type-specific flags ---

    # interaction-pdp
    p.add_argument(
        '--max-samples', type=int, default=100,
        help='Maximum samples for PDP grid (default: 100, for interaction-pdp)',
    )
    p.add_argument(
        '--grid-resolution', type=int, default=25,
        help='Grid points per feature (default: 25, for interaction-pdp)',
    )

    # calibration-overlay
    p.add_argument(
        '--eval-times', nargs='+', type=float, default=None, metavar='T',
        help='Evaluation timepoints (for calibration-overlay)',
    )
