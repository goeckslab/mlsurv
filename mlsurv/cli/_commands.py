"""Command handlers for the mlsurv CLI."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ._artifacts import (
    _load_learner_if_requested,
    _save_artifact_if_requested,
    _save_learner_if_requested,
    _train_or_load_model,
)
from ._data import (
    _build_subpopulations,
    _cli_load,
    _setup_learner,
    load_reference_csv,
)
from ._exit_codes import (
    CLIDataError,
    CLIUsageError,
    EXIT_COMPUTATION,
    EXIT_DATA,
    EXIT_ERROR,
    EXIT_USAGE,
)
from ._output import emit, format_dataframe, format_dict, format_list
from ._helpers import (
    _collect_report_kwargs,
    _invoke_with_supported_kwargs,
    _load_json_file,
    _print_loaded_data,
    _write_csv,
    _write_json,
    _write_report,
)
from ._parsers import _get_all_models, _get_tunable_models


def _prediction_time_label(value: float) -> str:
    """Render a prediction time label by stripping trailing zeros via ``:g``."""
    return f"{float(value):g}"


def _prediction_time_unit(metadata: dict) -> str | None:
    """Return the stored training time unit when present in artifact metadata."""
    contract = metadata.get('data_contract')
    if isinstance(contract, dict):
        time_unit = contract.get('time_unit')
        if isinstance(time_unit, str) and time_unit.strip():
            return time_unit.strip()

    config = metadata.get('config')
    if isinstance(config, dict):
        time_unit = config.get('time_unit')
        if isinstance(time_unit, str) and time_unit.strip():
            return time_unit.strip()

    return None


def _prediction_feature_names(metadata: dict) -> tuple[list[str] | None, str | None]:
    """Return artifact feature names or a user-facing error message."""
    from ._artifacts import _extract_feature_names

    feature_names, error = _extract_feature_names(metadata)
    if error is not None:
        return None, (
            f"{error} Required for prediction. "
            "Use an artifact created with feature metadata."
        )

    if len(feature_names) == 0:
        return None, (
            "Artifact data_contract.feature_names is empty. Prediction "
            "requires at least one feature column."
        )

    return feature_names, None


def _handle_compare(learner, args):
    """Handle --compare / --compare-output flags for paired model comparison."""
    boot_result = learner.bootstrap_result
    if boot_result is None:
        for flag in ('compare_output', 'compare'):
            if getattr(args, flag, False):
                label = '--compare-output' if flag == 'compare_output' else '--compare'
                extra = " CSV not written." if flag == 'compare_output' else (
                    " Use --n-bootstrap N to enable paired comparison."
                )
                print(
                    f"Warning: {label} skipped: no bootstrap data.{extra}",
                    file=sys.stderr,
                )
        return

    try:
        comp_df = boot_result.compare(metric=learner.scoring)
    except ValueError as exc:
        print(f"Warning: Paired comparison skipped: {exc}", file=sys.stderr)
        return

    quiet = getattr(args, 'quiet', False)
    fmt = getattr(args, 'output_format', 'table')
    if args.compare and args.verbose and fmt == 'table':
        from mlsurv.results.printing import print_paired_comparison
        print()
        print_paired_comparison(comp_df)
    if args.compare_output:
        _write_csv(comp_df, args.compare_output, "Paired comparison", quiet=quiet)


def cmd_run(args) -> int:
    """Run the full survival analysis pipeline."""
    from mlsurv import SurvivalLearner

    X, y = _cli_load(args)
    _print_loaded_data(args, X, y)

    learner = SurvivalLearner(X, y)
    _setup_learner(learner, args)

    if args.resume and args.experiment_name is None:
        print(
            "Warning: --resume without --experiment-name will generate a new "
            "experiment name and cannot find prior studies. "
            "Use --experiment-name to resume a specific experiment.",
            file=sys.stderr,
        )

    quiet = getattr(args, 'quiet', False)

    models = learner.run(
        models=args.models,
        tune=args.tune,
        n_trials=args.n_trials,
        verbose=args.verbose,
        n_bootstrap=args.n_bootstrap,
        resume=args.resume,
        experiment_name=args.experiment_name,
        show_eval_progress=not quiet,
        show_progress_bar=not quiet,
    )

    fmt = getattr(args, 'output_format', 'table')

    # Build CV metrics summary from returned models
    metrics_df = pd.DataFrame([
        {'model': n, 'cv_metric_mean': r.cv_metric_mean,
         'cv_metric_std': r.cv_metric_std}
        for n, r in models.items()
    ]).sort_values('cv_metric_mean', ascending=False)

    if fmt != 'table':
        emit(format_dataframe(metrics_df, fmt))
    elif not quiet:
        print("\nRun Results:")
        print("-" * 50)
        print(metrics_df.to_string(index=False))

        best = learner.best_model
        if best is not None:
            print(f"\nBest model: {best.name}")
            print(
                f"CV C-Index: {best.cv_metric_mean:.3f} "
                f"± {best.cv_metric_std:.3f}"
            )
            if best.evaluation is not None:
                c_index_test = best.evaluation.concordance['c_index_test'].iloc[0]
                print(f"Test C-Index: {c_index_test:.3f}")

    if args.compare or args.compare_output:
        _handle_compare(learner, args)

    if args.output:
        eval_result = learner.evaluation_result
        if eval_result is None:
            print(
                "Warning: No evaluation result available for report. "
                "Run with --evaluate to include model performance metrics.",
                file=sys.stderr,
            )
        report_kw = _collect_report_kwargs(args)
        _write_report(
            learner,
            args.output,
            eval_result=eval_result,
            quiet=quiet,
            **report_kw,
        )

    best = learner.best_model
    _save_artifact_if_requested(learner, best, args, step='run')
    _save_learner_if_requested(learner, args)

    return 0


def cmd_train(args) -> int:
    """Train a single model and report CV metrics."""
    from mlsurv import SurvivalLearner

    X, y = _cli_load(args)
    _print_loaded_data(args, X, y)

    learner = SurvivalLearner(X, y)
    _setup_learner(learner, args)

    quiet = getattr(args, 'quiet', False)
    model_params = _load_json_file(
        getattr(args, 'model_params', None), label="Model params file",
    )

    # Only forward model_params when the user supplied a file, so the default
    # train() call stays a plain single-model train.
    train_kwargs = {} if model_params is None else {'model_params': model_params}
    result = learner.train(args.model, **train_kwargs)
    fmt = getattr(args, 'output_format', 'table')

    train_df = pd.DataFrame([{
        'model': args.model,
        'cv_metric_mean': result.cv_metric_mean,
        'cv_metric_std': result.cv_metric_std,
    }])

    if fmt != 'table':
        emit(format_dataframe(train_df, fmt))
    elif not quiet:
        print(f"\nTraining Results for {args.model}:")
        print("-" * 50)
        print(f"CV C-Index: {result.cv_metric_mean:.3f} ± {result.cv_metric_std:.3f}")

    if args.output:
        _write_csv(train_df, args.output, "CV metrics", quiet=quiet)

    report_kw = _collect_report_kwargs(args)
    _write_report(learner, args.report, quiet=quiet, **report_kw)
    _save_artifact_if_requested(learner, result, args, step='train')
    _save_learner_if_requested(learner, args)

    return 0


def cmd_tune(args) -> int:
    """Run hyperparameter tuning."""
    from mlsurv import SurvivalLearner

    X, y = _cli_load(args)

    _print_loaded_data(args, X, y)
    if args.verbose:
        fmt = getattr(args, 'output_format', 'table')
        out = sys.stderr if fmt != 'table' else sys.stdout
        print(f"Tuning model: {args.model}", file=out)

    learner = SurvivalLearner(X, y)
    _setup_learner(learner, args)

    quiet = getattr(args, 'quiet', False)
    search_spaces = _load_json_file(args.search_spaces, label="Search spaces file")

    if args.resume and args.experiment_name is None:
        print(
            "Warning: --resume without --experiment-name will generate a new "
            "experiment name and cannot find prior studies. "
            "Use --experiment-name to resume a specific experiment.",
            file=sys.stderr,
        )

    result = _invoke_with_supported_kwargs(
        learner.tune,
        args.model,
        n_trials=args.n_trials,
        timeout=args.timeout,
        resume=args.resume,
        scoring=args.scoring,
        search_spaces=search_spaces,
        optuna_verbose=args.optuna_verbose,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        experiment_name=args.experiment_name,
        show_eval_progress=not quiet,
        show_progress_bar=not quiet,
    )

    fmt = getattr(args, 'output_format', 'table')
    best_params = result.best_params_ or {}
    tune_payload = {
        'model': args.model,
        'cv_metric_mean': result.cv_metric_mean,
        'cv_metric_std': result.cv_metric_std,
        'best_params': best_params,
    }

    if fmt != 'table':
        text = format_dict(tune_payload, fmt)
        if text is not None:
            emit(text)
    elif not quiet:
        from mlsurv._shared import display_name
        from mlsurv.results.printing import print_best_params

        display = display_name(result.name)
        print(f"\nTuning Results for {display}:")
        print("-" * 50)
        print(f"Best CV C-Index: {result.cv_metric_mean:.3f}")
        print_best_params(best_params, model_name=display)

    if args.output:
        _write_json(tune_payload, args.output, "Results", quiet=quiet)

    report_kw = _collect_report_kwargs(args)
    _write_report(learner, args.report, quiet=quiet, **report_kw)
    _save_artifact_if_requested(learner, result, args, step='tune')
    _save_learner_if_requested(learner, args)

    return 0


def cmd_evaluate(args) -> int:
    """Run model evaluation."""
    from mlsurv import SurvivalLearner

    learner = _load_learner_if_requested(args)
    if learner is not None:
        model_result = learner._lookup_model_by_name(args.model)
    else:
        X, y = _cli_load(args)
        _print_loaded_data(args, X, y)

        subpopulations, err = _build_subpopulations(args, X)
        if err:
            return err

        learner = SurvivalLearner(X, y)
        _setup_learner(learner, args, subpopulations=subpopulations)
        model_result = _train_or_load_model(learner, args, X=X)

    if args.verbose:
        print(f"Evaluating model: {args.model}")

    quiet = getattr(args, 'quiet', False)

    eval_kwargs = dict(
        n_importance_repeats=args.n_importance_repeats,
        eval_times=args.eval_times,
        include_shap=args.shap,
        shap_max_background=args.shap_max_background,
        shap_max_explain=args.shap_max_explain,
        show_progress=not quiet,
    )
    if args.bootstrap:
        eval_kwargs['bootstrap'] = True
        eval_kwargs['n_bootstrap'] = args.n_bootstrap
    if args.interactions:
        eval_kwargs['include_interactions'] = True
        eval_kwargs['interaction_top_k'] = args.interaction_top_k
    eval_result = learner.evaluate(model_result, **eval_kwargs)
    fmt = getattr(args, 'output_format', 'table')

    if fmt != 'table':
        emit(format_dataframe(eval_result.concordance, fmt))
    elif not quiet:
        print(f"\nEvaluation Results for {args.model}:")
        print("-" * 50)
        print(eval_result.concordance.to_string(index=False))

    if args.output:
        _write_csv(eval_result.concordance, args.output, "Results", quiet=quiet)

    report_kw = _collect_report_kwargs(args)
    _write_report(
        learner,
        args.report,
        eval_result=eval_result,
        quiet=quiet,
        **report_kw,
    )

    # --- Statistical tests (in-session flags) ---
    _run_stat_tests_if_requested(learner, args)

    # --- Inline plot generation ---
    plot_types = getattr(args, 'plot', None)
    if plot_types:
        from pathlib import Path

        from ._plot import _generate_inline_plots

        report_path = getattr(args, 'report', None)
        plot_dir = Path(report_path).parent if report_path else None
        if not quiet:
            print("\nGenerating plots...")
        _generate_inline_plots(learner, plot_types, plot_dir, quiet)

    _save_artifact_if_requested(learner, model_result, args, step='evaluate')
    _save_learner_if_requested(learner, args)

    return 0


def cmd_calibrate(args) -> int:
    """Attach S(t|x) calibrators to one or more trained models.

    Wraps the same internal ``_attach_calibrators()`` path that
    ``train(calibrate=...)`` uses, so it supports all three methods
    (``isotonic``, ``lowdof``, ``breslow_slope``).
    """
    from mlsurv import SurvivalLearner

    learner = _load_learner_if_requested(args)
    if learner is not None:
        if args.models:
            # Resolve user-supplied names through ``_lookup_model_by_name`` so
            # they validate against the actual storage keys.
            try:
                resolved_names = [
                    learner._lookup_model_by_name(name).name
                    for name in args.models
                ]
            except ValueError as exc:
                raise CLIUsageError(str(exc)) from exc
        else:
            resolved_names = list(learner.models)
    else:
        X, y = _cli_load(args)
        _print_loaded_data(args, X, y)
        learner = SurvivalLearner(X, y)
        _setup_learner(learner, args)
        if args.models is None:
            raise CLIUsageError(
                "--models is required when no --load-learner is given "
                "(no trained models to calibrate yet)."
            )
        for name in args.models:
            learner.train(name)
        # Normalize to the stored model keys so the learner.models[name]
        # lookups below can't KeyError on a case/alias mismatch.
        resolved_names = [
            learner._lookup_model_by_name(name).name for name in args.models
        ]

    if not learner.models:
        raise CLIUsageError(
            "No trained models on the loaded learner; nothing to calibrate."
        )

    from mlsurv.evaluation._calibrators import resolve_calibration_spec

    quiet = getattr(args, 'quiet', False)
    # Per-model spec so only the requested models are touched; others stay as-is.
    spec = {name: args.method for name in resolved_names}
    # Validate the method/model pairing up front, in a narrow catch, so that a
    # genuine fitting error raised inside _attach_calibrators below propagates
    # as itself instead of being mislabeled a usage error. resolve_calibration_spec
    # rejects breslow_slope on a non-score model; argparse already blocks unknown
    # methods. _attach_calibrators re-runs this resolution internally (cheap).
    score_based = {
        name: learner._model_is_score_based(learner.models[name])
        for name in resolved_names
    }
    try:
        resolve_calibration_spec(spec, resolved_names, score_based)
    except (ValueError, TypeError) as exc:
        raise CLIUsageError(str(exc)) from exc

    # Hold each model's prior calibrator object so the summary counts only the
    # calibrators this run actually (re)fit. Keeping the reference alive matters:
    # comparing ids would be unsafe because a freed old calibrator's address can
    # be reused by the new one, so we keep the object and compare with `is`.
    before = {name: learner.models[name].calibrator for name in resolved_names}
    learner._attach_calibrators(learner.models, spec)

    fitted = {
        name: cal
        for name in resolved_names
        if (cal := learner.models[name].calibrator) is not None
        and cal is not before[name]
    }
    skipped = [n for n in resolved_names if n not in fitted]
    if skipped:
        # _attach_calibrators() warns per skipped model (insufficient OOF, etc.).
        # Echo a consolidated list so the CLI summary makes the partial state
        # explicit.
        print(
            f"Warning: {len(skipped)}/{len(resolved_names)} model(s) skipped "
            f"during calibration: {', '.join(skipped)}.",
            file=sys.stderr,
        )

    # Nothing got calibrated though it was requested: report failure via the
    # exit code, not just a stderr line a pipeline won't inspect.
    if not fitted:
        if not quiet:
            print(
                "\nNo models were calibrated (insufficient data or no eligible "
                "models).",
                file=sys.stderr,
            )
        return EXIT_COMPUTATION

    summary_df = pd.DataFrame([
        {
            'model': name,
            'method': calibrator.method,
            'n_horizons': len(calibrator.horizons),
            'horizons': ','.join(
                f"{float(h):g}" for h in sorted(calibrator.horizons)
            ),
        }
        for name, calibrator in fitted.items()
    ])

    fmt = getattr(args, 'output_format', 'table')
    if fmt != 'table':
        emit(format_dataframe(summary_df, fmt))
    elif not quiet:
        # fitted is non-empty here (total failure returned above), so summary_df
        # always has at least one row.
        print("\nCalibration Results:")
        print("-" * 50)
        print(summary_df.to_string(index=False))

    if args.output:
        _write_csv(summary_df, args.output, "Calibration summary", quiet=quiet)

    report_kw = _collect_report_kwargs(args)
    _write_report(learner, args.report, quiet=quiet, **report_kw)
    _save_artifact_if_requested(learner, learner.best_model, args, step='calibrate')
    _save_learner_if_requested(learner, args)

    return 0


def cmd_bootstrap(args) -> int:
    """Train or load a model, then run bootstrap confidence intervals."""
    from mlsurv import SurvivalLearner

    learner = _load_learner_if_requested(args)
    if learner is not None:
        model_result = learner._lookup_model_by_name(args.model)
    else:
        X, y = _cli_load(args)
        _print_loaded_data(args, X, y)
        learner = SurvivalLearner(X, y)
        _setup_learner(learner, args)
        model_result = _train_or_load_model(learner, args, X=X)

    quiet = getattr(args, 'quiet', False)

    boot_result = learner.bootstrap(
        model_result,
        n_bootstrap=args.n_bootstrap,
        eval_times=args.eval_times,
        verbose=args.verbose,
        show_progress=not quiet,
    )
    fmt = getattr(args, 'output_format', 'table')

    if fmt != 'table':
        emit(format_dataframe(boot_result.summary, fmt))
    # No table-mode print — bootstrap summary goes to report/file only.

    if args.output:
        _write_csv(
            boot_result.summary, args.output, "Bootstrap summary", quiet=quiet,
        )

    report_kw = _collect_report_kwargs(args)
    _write_report(
        learner,
        args.report,
        bootstrap_results={args.model: boot_result},
        quiet=quiet,
        **report_kw,
    )

    # --- Inline plot generation ---
    plot_types = getattr(args, 'plot', None)
    if plot_types:
        from pathlib import Path

        from ._plot import _generate_inline_plots

        report_path = getattr(args, 'report', None)
        plot_dir = Path(report_path).parent if report_path else None
        if not quiet:
            print("\nGenerating plots...")
        _generate_inline_plots(learner, plot_types, plot_dir, quiet)

    _save_artifact_if_requested(learner, model_result, args, step='bootstrap')
    _save_learner_if_requested(learner, args)

    return 0


def cmd_validate(args) -> int:
    """Train or load a model, then validate on an external cohort."""
    from mlsurv import SurvivalLearner

    learner = _load_learner_if_requested(args)
    if learner is not None:
        model_result = learner._lookup_model_by_name(args.model)
    else:
        X, y = _cli_load(args)
        _print_loaded_data(args, X, y)
        learner = SurvivalLearner(X, y)
        _setup_learner(learner, args)
        model_result = _train_or_load_model(learner, args, X=X)

    from mlsurv.utils import load_survival_data

    from ._data import _cli_csv_errors, _validate_csv_path

    try:
        _validate_csv_path(args.validation_data)
    except CLIDataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_DATA

    val_time_unit = getattr(args, 'val_time_unit', None)
    val_event_values = getattr(args, 'val_event_values', None)
    try:
        with _cli_csv_errors(args.validation_data):
            X_val, y_val = load_survival_data(
                args.validation_data,
                time_col=args.val_time_col,
                event_col=args.val_event_col,
                start_col=getattr(args, 'val_start_col', None),
                end_col=getattr(args, 'val_end_col', None),
                time_unit=val_time_unit if val_time_unit is not None else getattr(args, 'time_unit', 'days'),
                event_values=val_event_values if val_event_values is not None else getattr(args, 'event_values', None),
            )
    except CLIDataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_DATA

    quiet = getattr(args, 'quiet', False)

    # Optional reference biomarker scores aligned with X_val.
    references = None
    references_file = getattr(args, 'references_file', None)
    if references_file:
        try:
            references = load_reference_csv(
                references_file,
                reference_cols=getattr(args, 'reference_cols', None),
                expected_n_rows=len(X_val),
            )
        except CLIDataError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return EXIT_DATA

    val_result = learner.validate(
        X_val,
        y_val,
        models=model_result,
        name=args.cohort_name,
        bootstrap=args.bootstrap,
        n_bootstrap=args.n_bootstrap,
        verbose=args.verbose,
        show_progress=not quiet,
        references=references,
        higher_is_worse=getattr(args, 'higher_is_worse', True),
    )

    fmt = getattr(args, 'output_format', 'table')
    if fmt != 'table':
        emit(format_dataframe(val_result.concordance, fmt))
    elif not quiet:
        print(f"\nValidation Results ({args.cohort_name}):")
        print("-" * 50)
        print(val_result.concordance.to_string(index=False))

    if args.output:
        _write_csv(
            val_result.concordance, args.output, "Validation results",
            quiet=quiet,
        )

    if args.report:
        bootstrap_results = None
        if getattr(val_result, 'bootstrap', None) is not None:
            bootstrap_results = {args.model: val_result.bootstrap}
        report_kw = _collect_report_kwargs(args)
        _write_report(
            learner,
            args.report,
            eval_result=getattr(val_result, 'evaluation', None),
            bootstrap_results=bootstrap_results,
            quiet=quiet,
            **report_kw,
        )

    _save_learner_if_requested(learner, args)

    return 0


def cmd_predict(args) -> int:
    """Generate risk scores and optional survival predictions."""
    if getattr(args, 'load_learner', None):
        return _predict_from_learner(args)
    return _predict_from_artifact(args)


def _build_predict_output_df(result, index) -> pd.DataFrame:
    """Build prediction output DataFrame from a PredictionResult.

    Includes CI columns (``*_lower``/``*_upper``) when bootstrap data is
    present in *result*.
    """
    output_df = pd.DataFrame(
        {
            'sample_idx': np.asarray(index),
            'model_name': result.model_name,
            'risk_score': result.risk_scores,
        }
    )

    # Median survival + CI
    if result._median_survival is not None:
        output_df['median_survival'] = result._median_survival
        med_ci = result.median_survival_ci
        if med_ci is not None:
            output_df['median_survival_lower'] = med_ci['ci_lower'].values
            output_df['median_survival_upper'] = med_ci['ci_upper'].values

    # Default RMST + CI
    if result._rmst_values is not None and result._rmst_default_tau is not None:
        tau_label = _prediction_time_label(result._rmst_default_tau)
        output_df[f'rmst_{tau_label}'] = result._rmst_values
        rmst_default_ci = result.rmst_default_ci
        if rmst_default_ci is not None:
            output_df[f'rmst_{tau_label}_lower'] = rmst_default_ci['ci_lower'].values
            output_df[f'rmst_{tau_label}_upper'] = rmst_default_ci['ci_upper'].values

    # Survival / risk at user times + CI
    if result._surv_at_user_times is not None and result._user_times is not None:
        surv_ci = result.survival_at_times_ci
        for j, t in enumerate(result._user_times):
            label = _prediction_time_label(t)
            surv_values = result._surv_at_user_times[:, j]
            output_df[f'survival_{label}'] = surv_values
            if surv_ci is not None:
                tcol = f"t_{label}"
                output_df[f'survival_{label}_lower'] = surv_ci[f'{tcol}_ci_lower'].values
                output_df[f'survival_{label}_upper'] = surv_ci[f'{tcol}_ci_upper'].values
            output_df[f'risk_{label}'] = 1.0 - surv_values
            if surv_ci is not None:
                tcol = f"t_{label}"
                # risk = 1 - survival, so bounds invert
                output_df[f'risk_{label}_lower'] = 1.0 - surv_ci[f'{tcol}_ci_upper'].values
                output_df[f'risk_{label}_upper'] = 1.0 - surv_ci[f'{tcol}_ci_lower'].values

    # RMST at user times + CI
    if result._rmst_at_user_times is not None and result._user_times is not None:
        rmst_times_ci = result.rmst_at_times_ci
        for j, t in enumerate(result._user_times):
            label = _prediction_time_label(t)
            output_df[f'rmst_{label}'] = result._rmst_at_user_times[:, j]
            if rmst_times_ci is not None:
                tcol = f"tau_{label}"
                if f'{tcol}_ci_lower' in rmst_times_ci.columns:
                    output_df[f'rmst_{label}_lower'] = rmst_times_ci[f'{tcol}_ci_lower'].values
                    output_df[f'rmst_{label}_upper'] = rmst_times_ci[f'{tcol}_ci_upper'].values

    return output_df


def _build_multi_predict_output_df(results, index) -> pd.DataFrame:
    """Stack per-model prediction frames into one combined table.

    *results* is a ``Dict[str, PredictionResult]`` as returned by
    ``learner.predict()`` for a multi-model selection (CV-sorted, best
    first). Each model contributes one block of rows; ``model_name`` is
    moved to the front so blocks are identifiable, matching the
    model-column-first stacking convention of ``rmst_test``/``logrank_test``
    (whose tables lead with ``model_type``). Columns a model cannot produce
    (e.g. survival columns after a survival-function failure) are NaN-filled
    by the concat.
    """
    frames = []
    for res in results.values():
        try:
            df = _build_predict_output_df(res, index)
        except KeyError as exc:
            # Re-raise with the model named so the CI-mismatch error
            # message localizes the failure in a combined run.
            raise KeyError(f"model '{res.model_name}': {exc.args[0]}") from exc
        frames.append(
            df[['model_name'] + [c for c in df.columns if c != 'model_name']]
        )
    return pd.concat(frames, ignore_index=True)


def _check_strict_and_output(args, result, X_pred_index, selected_models=None) -> int:
    """Handle --strict check, build output DataFrame, print, and write CSV.

    Shared by both the artifact and learner predict paths. *result* is a
    single ``PredictionResult`` or, for a multi-model learner selection, a
    ``Dict[str, PredictionResult]``. *selected_models* names the models the
    user selected (multi-model path only; used for membership checks only,
    order is not significant) so models that ``predict()`` skipped entirely
    are still caught here.
    Returns the CLI exit code (0 on success, EXIT_COMPUTATION on strict failure
    or internal CI column mismatch).
    """
    has_times = bool(args.times)
    is_multi = isinstance(result, dict)

    # Models predict() skipped entirely contribute no rows at all, so they
    # are reported regardless of --times.
    if is_multi:
        expected = selected_models if selected_models is not None else list(result)
        skipped = [name for name in expected if name not in result]
        if skipped:
            names = ", ".join(skipped)
            if args.strict:
                print(
                    f"Error: Prediction failed for {names}; no predictions "
                    "can be produced for them. Use without --strict to keep "
                    "the remaining models' output (the failed models are "
                    "omitted).",
                    file=sys.stderr,
                )
                return EXIT_COMPUTATION
            print(
                f"Warning: Prediction failed for {names}; they are omitted "
                "from the output.",
                file=sys.stderr,
            )

    if has_times:
        if is_multi:
            degraded = [
                name for name, res in result.items() if res._surv_probs is None
            ]
        else:
            degraded = [result.model_name] if result._surv_probs is None else []
        if degraded:
            names = ", ".join(degraded)
            if args.strict:
                print(
                    f"Error: Survival function prediction failed for "
                    f"{names}; --times columns cannot be produced. Use "
                    "without --strict for graceful degradation (rc=0 with "
                    "risk_score only for the affected models).",
                    file=sys.stderr,
                )
                return EXIT_COMPUTATION
            print(
                f"Warning: Survival function prediction failed for {names}; "
                "only risk_score is included for the affected models. Their "
                "--times columns are omitted.",
                file=sys.stderr,
            )

    try:
        if is_multi:
            output_df = _build_multi_predict_output_df(result, X_pred_index)
        else:
            output_df = _build_predict_output_df(result, X_pred_index)
    except KeyError as exc:
        print(
            f"Error: Internal CI column mismatch: {exc}. "
            "Try running without --ci to confirm base predictions work.",
            file=sys.stderr,
        )
        return EXIT_COMPUTATION

    quiet = getattr(args, 'quiet', False)
    fmt = getattr(args, 'output_format', 'table')

    if fmt != 'table':
        emit(format_dataframe(output_df, fmt))
    elif not quiet:
        print("\nPredictions:")
        print("-" * 50)
        print(output_df.to_string(index=False))

    if args.output:
        _write_csv(output_df, args.output, "Predictions", quiet=quiet)

    return 0


def _predict_from_artifact(args) -> int:
    """Predict using a model-only artifact (no CI support)."""
    import warnings

    from mlsurv.evaluation._prediction_helpers import predict_from_pipeline
    from mlsurv.learner._persistence import PersistenceMixin

    from .._warnings import SuggestionWarning

    if getattr(args, 'model', None):
        warnings.warn(
            "--model is ignored with --load-artifact (the artifact contains "
            "exactly one model)",
            SuggestionWarning,
            stacklevel=1,
        )

    try:
        metadata, model, _recalibrator = PersistenceMixin.load_artifact(args.load_artifact)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_DATA
    except (ValueError, RuntimeError) as exc:
        print(f"Error loading artifact: {exc}", file=sys.stderr)
        return EXIT_DATA

    if model is None:
        print(
            f"Artifact at '{args.load_artifact}' does not contain a model. "
            "Only artifacts from train/tune/run/evaluate/bootstrap are usable "
            "for prediction.",
            file=sys.stderr,
        )
        return EXIT_DATA

    feature_names, feature_error = _prediction_feature_names(metadata)
    if feature_error is not None:
        print(f"Error: {feature_error}", file=sys.stderr)
        return EXIT_DATA

    try:
        X_pred = _cli_load(args, feature_names=feature_names)
    except CLIDataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_DATA

    model_name = (
        metadata.get('model_name')
        or getattr(model, 'name', None)
        or type(model).__name__
    )

    times = np.asarray(args.times, dtype=float) if args.times else None

    if times is not None:
        time_unit = _prediction_time_unit(metadata)
        if time_unit is None:
            print(
                "Note: Prediction times are interpreted in the same unit as "
                "the training data.",
                file=sys.stderr,
            )
        else:
            print(
                "Note: Prediction times are interpreted in the training "
                f"data's time unit ({time_unit}).",
                file=sys.stderr,
            )

    try:
        result = predict_from_pipeline(
            model, X_pred, model_name=model_name, times=times,
        )
    except (MemoryError, RecursionError):
        raise
    except ImportError as exc:
        print(
            f"Error: Missing dependency required for {model_name}: {exc}",
            file=sys.stderr,
        )
        return EXIT_ERROR
    except (ValueError, RuntimeError, TypeError, NotImplementedError,
            np.linalg.LinAlgError) as exc:
        print(f"Error: Could not compute predictions: {exc}", file=sys.stderr)
        return EXIT_COMPUTATION

    return _check_strict_and_output(args, result, X_pred.index)


def _predict_from_learner(args) -> int:
    """Predict using a saved learner (CI support available)."""
    try:
        learner = _load_learner_if_requested(args)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_DATA
    except (ValueError, RuntimeError) as exc:
        print(f"Error loading learner: {exc}", file=sys.stderr)
        return EXIT_DATA

    if learner is None:
        print("Error: --load-learner is required for this path", file=sys.stderr)
        return EXIT_USAGE

    feature_names = list(learner._state.X_train.columns)

    try:
        X_pred = _cli_load(args, feature_names=feature_names)
    except CLIDataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_DATA

    times = np.asarray(args.times, dtype=float) if args.times else None

    if times is not None:
        print(
            "Note: Prediction times are interpreted in the same unit as "
            "the training data.",
            file=sys.stderr,
        )

    # Only forward the bootstrap-CI knobs when --ci is set. They are no-ops
    # otherwise, and the CLI's random_state default (42) differs from predict()'s
    # API default (None), so passing them unconditionally would trip predict()'s
    # "no effect when ci=False" warning on every plain prediction.
    predict_kwargs = dict(
        models=args.model,
        times=times,
        ci=args.ci,
        verbose=args.verbose,
    )
    if args.ci:
        predict_kwargs.update(
            ci_band=args.ci_band,
            n_bootstrap=args.n_bootstrap,
            ci_level=args.ci_level,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
    else:
        # We skip the CI knobs above rather than let predict() warn, so detect
        # an explicitly-set non-default value here and say so — dropping it
        # silently would recreate the very gap the skip-when-ci-off path closes.
        # Defaults mirror the predict subcommand in cli/_parsers.py.
        ci_overrides = [
            flag for flag, value, default in (
                ("--ci-band", args.ci_band, False),
                ("--n-bootstrap", args.n_bootstrap, 200),
                ("--ci-level", args.ci_level, 0.95),
                ("--random-state", args.random_state, 42),
                ("--n-jobs", args.n_jobs, 1),
            )
            if value != default
        ]
        if ci_overrides:
            print(
                f"Warning: {', '.join(ci_overrides)} require --ci and were "
                "ignored. Pass --ci to compute bootstrap confidence intervals.",
                file=sys.stderr,
            )

    try:
        result = learner.predict(X_pred, **predict_kwargs)
    except (MemoryError, RecursionError):
        raise
    except ImportError as exc:
        print(f"Error: Missing dependency: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (ValueError, RuntimeError, TypeError, NotImplementedError,
            np.linalg.LinAlgError) as exc:
        print(f"Error: Could not compute predictions: {exc}", file=sys.stderr)
        return EXIT_COMPUTATION

    return _check_strict_and_output(
        args, result, X_pred.index,
        selected_models=list(learner.models) if args.model is None else None,
    )


def cmd_benchmark(args) -> int:
    """Compare trained models against reference scores from an external file."""
    from mlsurv import SurvivalLearner
    from mlsurv.learner._cv_training import (
        _handle_model_error,
        _raise_if_all_failed,
        _warn_dropped_models,
    )

    X, y = _cli_load(args)

    # Load reference scores from external file (validate before printing)
    references = load_reference_csv(
        args.reference_file,
        reference_cols=args.reference_cols,
        expected_n_rows=len(X),
    )

    _print_loaded_data(args, X, y)

    subpopulations, err = _build_subpopulations(args, X)
    if err:
        return err

    learner = SurvivalLearner(X, y)
    _setup_learner(learner, args, subpopulations=subpopulations)

    # Pre-train with per-model error handling (mirrors _run_model_comparison)
    trained = {}
    errors = {}
    for model_name in args.models:
        try:
            trained[model_name] = learner.train(model_name, verbose=args.verbose)
        except (MemoryError, RecursionError):
            raise
        except Exception as e:  # broad: training can raise unpredictable errors
            errors[model_name] = _handle_model_error(
                e, model_name, "train", warn_user=False,
            )

    _raise_if_all_failed(trained, args.models, "train", errors=errors)
    _warn_dropped_models(errors, len(args.models), "train")

    # Run benchmark
    result = learner.benchmark(
        references=references,
        dataset=args.dataset,
        bootstrap=args.bootstrap,
        n_bootstrap=args.n_bootstrap if args.bootstrap else 0,
        verbose=args.verbose,
    )

    quiet = getattr(args, 'quiet', False)
    fmt = getattr(args, 'output_format', 'table')

    if fmt != 'table':
        emit(format_dataframe(result.evaluation.concordance, fmt))
    elif not quiet:
        print("\nBenchmark Results:")
        print("-" * 50)
        print(result.evaluation.concordance.to_string(index=False))

    if args.output:
        _write_csv(
            result.evaluation.concordance, args.output, "Benchmark results",
            quiet=quiet,
        )

    if args.report:
        report_kw = _collect_report_kwargs(args)
        _write_report(learner, args.report, quiet=quiet, **report_kw)

    _save_artifact_if_requested(learner, None, args, step='benchmark')
    _save_learner_if_requested(learner, args)

    return 0


def cmd_sweep(args) -> int:
    """Sweep a parameter across values and measure CV performance."""
    from mlsurv import SurvivalLearner
    from mlsurv.learner._cv_training import (
        _handle_model_error,
        _raise_if_all_failed,
        _warn_dropped_models,
    )

    X, y = _cli_load(args)
    _print_loaded_data(args, X, y)

    learner = SurvivalLearner(X, y)
    _setup_learner(learner, args)

    # Pre-train with per-model error handling (mirrors _run_model_comparison)
    trained = {}
    errors = {}
    for model_name in args.models:
        try:
            trained[model_name] = learner.train(model_name, verbose=args.verbose)
        except (MemoryError, RecursionError):
            raise
        except Exception as e:  # broad: training can raise unpredictable errors
            errors[model_name] = _handle_model_error(
                e, model_name, "train", warn_user=False,
            )

    _raise_if_all_failed(trained, args.models, "train", errors=errors)
    _warn_dropped_models(errors, len(args.models), "train")

    result = learner.parameter_sweep(
        param=args.param,
        values=args.values,
        target=args.target,
        models=list(trained),
        include_test=args.include_test,
        n_bootstrap=args.n_bootstrap,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )

    quiet = getattr(args, 'quiet', False)

    if result.data.empty:
        print("\nNo sweep results (all parameter values failed).", file=sys.stderr)
        return EXIT_COMPUTATION

    fmt = getattr(args, 'output_format', 'table')
    if fmt != 'table':
        emit(format_dataframe(result.data, fmt))
    elif not quiet:
        print(f"\nSweep Results ({args.param}):")
        print("-" * 50)
        print(result.data.to_string(index=False))

    if args.output:
        _write_csv(result.data, args.output, "Sweep results", quiet=quiet)

    report_kw = _collect_report_kwargs(args)
    _write_report(learner, args.report, quiet=quiet, **report_kw)

    return 0


def cmd_setup(args) -> int:
    """Inspect dataset characteristics and setup-derived split sizes."""
    from mlsurv import SurvivalLearner

    X, y = _cli_load(args)
    learner = SurvivalLearner(X, y)
    _setup_learner(learner, args)

    quiet = getattr(args, 'quiet', False)
    fmt = getattr(args, 'output_format', 'table')

    n_samples = len(X)
    n_features = len(X.columns)
    n_events = int(y['event'].sum())
    event_rate = float(y['event'].mean())
    n_train = len(learner.X_train)
    n_test = len(learner.X_test)

    missing = X.isna().sum()
    missing = missing[missing > 0]
    missing_str = (
        'none' if len(missing) == 0
        else '; '.join(f"{col}={int(c)}" for col, c in missing.sort_values(ascending=False).items())
    )

    diagnostics = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_events': n_events,
        'event_rate': round(event_rate, 3),
        'n_train': n_train,
        'n_test': n_test,
        'missing_values': missing_str,
    }

    if fmt != 'table':
        text = format_dict(diagnostics, fmt)
        if text is not None:
            emit(text)
    elif not quiet:
        print("\nSetup Diagnostics:")
        print("-" * 50)
        print(f"n_samples: {n_samples}")
        print(f"n_features: {n_features}")
        print(f"n_events: {n_events}")
        print(f"event_rate: {event_rate:.3f}")
        print(f"n_train: {n_train}")
        print(f"n_test: {n_test}")

        if len(missing) == 0:
            print("missing_values: none")
        else:
            print("missing_values:")
            for col, count in missing.sort_values(ascending=False).items():
                print(f"  {col}: {int(count)}")

    _save_artifact_if_requested(learner, None, args, step='setup')
    _save_learner_if_requested(learner, args)

    return 0


def cmd_models(args) -> int:
    """List available model names.

    Output is always printed regardless of --quiet since the model list
    IS the result of this command, not decoration.
    """
    all_models = _get_all_models()
    tunable = set(_get_tunable_models())
    fmt = getattr(args, 'output_format', 'table')

    if fmt != 'table' and getattr(args, 'verbose', False):
        if fmt == 'json':
            import json as _json
            items = [{'name': n, 'tunable': n in tunable} for n in all_models]
            emit(_json.dumps(items, indent=2))
        else:  # csv
            lines = ['name,tunable']
            lines.extend(f'{n},{n in tunable}' for n in all_models)
            emit('\n'.join(lines))
    elif fmt != 'table':
        emit(format_list(list(all_models), fmt))
    else:
        for name in all_models:
            if getattr(args, 'verbose', False):
                suffix = " (tunable)" if name in tunable else ""
                print(f"{name}{suffix}")
            else:
                print(name)
    return 0


def cmd_explain(args) -> int:
    """Generate SHAP explanation for an individual patient."""
    learner = _load_learner_if_requested(args)
    if learner is None:
        print("Error: --load-learner is required for explain", file=sys.stderr)
        return EXIT_USAGE

    # Load patient CSV if provided
    X_patient = None
    if args.patient_data is not None:
        from ._data import _read_csv, _validate_csv_path

        try:
            data_path = _validate_csv_path(args.patient_data)
            patient_df = _read_csv(data_path)
        except CLIDataError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return EXIT_DATA
        if len(patient_df) != 1:
            print(
                f"Error: --patient-data must contain exactly 1 row, "
                f"got {len(patient_df)}.",
                file=sys.stderr,
            )
            return EXIT_DATA
        X_patient = patient_df

    try:
        result = learner.explain(
            patient=args.patient,
            X=X_patient,
            model=args.model,
            n_features=args.n_features,
            plot=args.plot is not None,
            max_background=args.max_background,
        )
    except (MemoryError, RecursionError):
        raise
    except ImportError as exc:
        print(
            f"Error: Missing dependency required for explain: {exc}",
            file=sys.stderr,
        )
        return EXIT_ERROR
    except (ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_COMPUTATION
    except (RuntimeError, np.linalg.LinAlgError) as exc:
        print(f"Error: SHAP computation failed: {exc}", file=sys.stderr)
        return EXIT_COMPUTATION

    # Print to stdout
    fmt = getattr(args, 'output_format', 'table')
    if fmt != 'table':
        emit(format_dataframe(result.shap_values, fmt))
    else:
        print("\nSHAP Explanation:")
        print("-" * 50)
        print(result.shap_values.to_string(index=False))

    if args.output:
        _write_csv(result.shap_values, args.output, "SHAP values")

    if args.plot:
        if result.figure is not None:
            try:
                result.figure.write_html(args.plot)
            except OSError as exc:
                print(
                    f"Warning: Failed to write plot to {args.plot}: {exc}",
                    file=sys.stderr,
                )
            else:
                print(f"\nWaterfall plot saved to: {args.plot}")
        else:
            print(
                "Warning: --plot was requested but no waterfall figure was "
                "generated.",
                file=sys.stderr,
            )

    return 0


def cmd_patient_report(args) -> int:
    """Generate a standalone single-patient HTML report."""
    learner = _load_learner_if_requested(args)
    if learner is None:
        print(
            "Error: --load-learner is required for patient-report",
            file=sys.stderr,
        )
        return EXIT_USAGE

    # Resolve the patient cohort X and the row selector. argparse already
    # enforces that exactly one of --patient / --patient-data is given.
    if args.patient_data is not None:
        from ._data import _read_csv, _validate_csv_path

        try:
            data_path = _validate_csv_path(args.patient_data)
            X = _read_csv(data_path)
        except CLIDataError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return EXIT_DATA
        if len(X) == 0:
            print("Error: --patient-data file has no rows.", file=sys.stderr)
            return EXIT_DATA
        patient_arg = 0
    else:
        X = learner._state.X_test
        if X is None:
            print(
                "Error: no test set is available on the learner; pass "
                "--patient-data with a patient CSV instead.",
                file=sys.stderr,
            )
            return EXIT_DATA
        # --patient is a string; an all-digit value is a positional index,
        # anything else is a risk-percentile spec passed through verbatim.
        spec = args.patient
        patient_arg = int(spec) if spec.lstrip('-').isdigit() else spec

    try:
        learner.patient_report(
            X,
            patient=patient_arg,
            model=args.model,
            output_path=args.output,
            times=args.times,
            time_unit=args.time_unit,
            n_neighbors=args.n_neighbors,
            ci=args.ci,
            ci_band=args.ci_band,
            n_bootstrap=args.n_bootstrap,
            offline=args.offline,
            random_state=args.random_state,
            use_recalibrated=args.use_recalibrated,
        )
    except (MemoryError, RecursionError):
        raise
    except ImportError as exc:
        print(
            f"Error: Missing dependency required for patient-report: {exc}",
            file=sys.stderr,
        )
        return EXIT_ERROR
    except (ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_COMPUTATION
    except (RuntimeError, np.linalg.LinAlgError) as exc:
        print(
            f"Error: patient-report computation failed: {exc}",
            file=sys.stderr,
        )
        return EXIT_COMPUTATION
    except OSError as exc:
        print(
            f"Error: Failed to write report to {args.output}: {exc}",
            file=sys.stderr,
        )
        return EXIT_ERROR

    print(f"\nPatient report saved to: {args.output}")
    return 0


def cmd_export_study(args) -> int:
    """Export Optuna study history to file."""
    from mlsurv.tuning import export_study

    kwargs = {}
    if args.study_name and args.storage:
        kwargs['study_name'] = args.study_name
        kwargs['storage'] = args.storage
    elif args.experiment and args.model:
        kwargs['experiment_name'] = args.experiment
        kwargs['model_name'] = args.model
    else:
        print(
            "Error: provide either --experiment + --model or --study-name + --storage",
            file=sys.stderr,
        )
        return EXIT_USAGE

    result = export_study(**kwargs, path=args.output, format=args.format)
    quiet = getattr(args, 'quiet', False)
    if not quiet:
        n_trials = result['metadata']['n_trials']
        best = result['metadata']['best_value']
        best_str = f"{best:.4f}" if best is not None else "N/A"
        print(f"Exported {n_trials} trials (best={best_str}) to: {args.output}")
    return 0


def _list_studies_records(studies, storage_mode):
    """Flatten the ``{experiment: [...]}`` study map into flat records.

    In storage mode each item is a study-info dict (with ``n_trials`` /
    ``best_value``); in file mode each item is a ``*_study.db`` filename, so a
    resolved file ``path`` and a ``sqlite:///`` dashboard URL are attached.
    """
    from mlsurv.tuning.studies import get_study_path

    records = []
    for exp, items in studies.items():
        for item in items:
            if storage_mode:
                records.append({
                    'experiment': exp,
                    'study': item['name'],
                    'full_name': item['full_name'],
                    'n_trials': item['n_trials'],
                    'best_value': item['best_value'],
                })
            else:
                path = (get_study_path(exp, base_dir=None) / item).as_posix()
                records.append({
                    'experiment': exp,
                    'study': item,
                    'path': path,
                    'dashboard': f"sqlite:///{path}",
                })
    return records


def _print_studies_table(studies, storage, storage_mode, quiet):
    """Print the grouped, human-readable study listing."""
    # Use the studies-module binding so the header path matches what
    # list_studies()/get_study_path() actually scanned (they read the same
    # binding); importing it via mlsurv.tuning would be a separate name.
    from mlsurv.tuning.studies import DEFAULT_STORAGE_BASE, get_study_path

    if not quiet:
        if storage_mode:
            print(f"Storage: {storage}")
            print(f"Dashboard: optuna-dashboard {storage}")
        else:
            print(f"Storage: {DEFAULT_STORAGE_BASE.as_posix()}")

    for exp, items in studies.items():
        print(f"\nExperiment: {exp}")
        if storage_mode:
            for info in items:
                best = info['best_value']
                best_str = f"{best:.4f}" if best is not None else "N/A"
                print(f"  {info['name']}  ({info['n_trials']} trials, best={best_str})")
        else:
            exp_dir = get_study_path(exp, base_dir=None)
            print(f"  Path: {exp_dir.as_posix()}")
            print("  Studies:")
            for filename in items:
                study_path = (exp_dir / filename).as_posix()
                print(f"    {filename}    optuna-dashboard sqlite:///{study_path}")


def cmd_list_studies(args) -> int:
    """List Optuna tuning studies and where their .db files live.

    The study listing IS the result of this command, so it is always printed
    regardless of ``--quiet``; only the decorative storage header is suppressed.
    """
    import json

    from mlsurv.tuning import list_studies
    from mlsurv.tuning.studies import DEFAULT_STORAGE_BASE

    storage = args.storage
    storage_mode = storage is not None
    all_studies = list_studies(storage=storage)

    experiment = getattr(args, 'experiment', None)
    if experiment is not None:
        studies = {experiment: all_studies.get(experiment, [])}
    else:
        studies = all_studies

    # Drop experiments with no studies (e.g. a filtered name that has none).
    studies = {exp: items for exp, items in studies.items() if items}

    if not studies:
        if experiment is not None:
            print(f"No studies found for experiment '{experiment}'.")
        elif storage_mode:
            print(f"No studies found in storage: {storage}")
        else:
            print(
                f"No studies found under {DEFAULT_STORAGE_BASE.as_posix()} "
                f"(run tune(...) first)."
            )
        return 0

    fmt = getattr(args, 'output_format', 'table')
    if fmt == 'json':
        records = _list_studies_records(studies, storage_mode)
        emit(json.dumps(records, indent=2, default=str))
    elif fmt == 'csv':
        records = _list_studies_records(studies, storage_mode)
        emit(format_dataframe(pd.DataFrame(records), 'csv'))
    else:
        quiet = getattr(args, 'quiet', False)
        _print_studies_table(studies, storage, storage_mode, quiet)
    return 0


# ---------------------------------------------------------------------------
# Statistical test helpers and standalone commands
# ---------------------------------------------------------------------------

def _print_stat_test(
    df, title: str, output_path: str | None, *, fmt: str = 'table',
) -> None:
    """Print a statistical test result and optionally write CSV."""
    if fmt != 'table':
        if not df.empty:
            emit(format_dataframe(df, fmt))
        if output_path and not df.empty:
            _write_csv(df, output_path, title)
        return

    print(f"\n{title}")
    print("=" * 50)
    if df.empty:
        print("  (no results)")
        if output_path:
            print(f"  Note: CSV not written (no results): {output_path}",
                  file=sys.stderr)
    else:
        print(df.to_string(index=False))
        if output_path:
            _write_csv(df, output_path, title)


def _run_stat_tests_if_requested(learner, args) -> None:
    """Run any statistical tests requested via evaluate flags.

    Each test is wrapped individually so that a failure in one test
    does not prevent the others from running or block artifact/learner
    saving in ``cmd_evaluate``.
    """
    tests = [
        ('ph_test', lambda: learner.ph_test(verbose=False),
         "Proportional Hazards Test"),
        ('linearity_test', lambda: learner.linearity_test(verbose=False),
         "Log-Hazard Linearity Test"),
        ('logrank_test', lambda: learner.logrank_test(models=args.model, verbose=False),
         "Log-Rank Test"),
        ('rmst_test', lambda: learner.rmst_test(models=args.model, verbose=False),
         "RMST Comparison"),
        ('importance_test', lambda: learner.importance_test(models=args.model, verbose=False),
         "Importance P-Values"),
    ]
    fmt = getattr(args, 'output_format', 'table')
    for flag, run_test, title in tests:
        if not getattr(args, flag, False):
            continue
        try:
            df = run_test()
            _print_stat_test(df, title, None, fmt=fmt)
        except (MemoryError, RecursionError):
            raise
        except (ValueError, TypeError, RuntimeError, ImportError,
                np.linalg.LinAlgError) as exc:
            print(f"\nWarning: {title} failed: {exc}", file=sys.stderr)


def _load_learner_for_stat_test(args):
    """Load a saved learner for standalone statistical test commands."""
    learner = _load_learner_if_requested(args)
    if learner is None:
        raise CLIUsageError(
            "--load-learner is required for standalone stat-test commands"
        )
    return learner


def _cli_stat_error(exc: ValueError) -> str:
    """Translate common learner ValueError messages to CLI-friendly guidance."""
    msg = str(exc)
    if "evaluation result" in msg.lower():
        return (
            "The loaded learner has no evaluation data. "
            "Save the learner after running 'mlsurv evaluate' and retry."
        )
    if "bootstrap" in msg.lower() and "importance" in msg.lower():
        return (
            "The loaded learner has no bootstrap importance data. "
            "Save the learner after running 'mlsurv bootstrap' with "
            "include_importance=True, or use --method pimp instead."
        )
    return msg


def cmd_ph_test(args) -> int:
    """Run proportional hazards test on a saved learner."""
    learner = _load_learner_for_stat_test(args)
    fmt = getattr(args, 'output_format', 'table')
    df = learner.ph_test(
        time_transform=args.time_transform,
        verbose=args.verbose,
    )
    if not args.verbose:
        _print_stat_test(df, "Proportional Hazards Test", getattr(args, 'output', None), fmt=fmt)
    elif getattr(args, 'output', None) and not df.empty:
        _write_csv(df, args.output, "Proportional Hazards Test")
    return 0


def cmd_logrank_test(args) -> int:
    """Run log-rank test across risk groups on a saved learner."""
    learner = _load_learner_for_stat_test(args)
    fmt = getattr(args, 'output_format', 'table')
    try:
        df = learner.logrank_test(
            models=args.model,
            n_groups=args.n_groups,
            p_adjust=args.p_adjust,
            verbose=args.verbose,
        )
    except ValueError as exc:
        print(f"Error: {_cli_stat_error(exc)}", file=sys.stderr)
        return EXIT_COMPUTATION
    if df.empty:
        # Multi-model runs degrade per-model failures to warnings and an
        # empty table; an all-empty result is a failure at the CLI level.
        print(
            "Error: Log-rank test produced no results for any trained "
            "model (see warnings above).",
            file=sys.stderr,
        )
        return EXIT_COMPUTATION
    if not args.verbose:
        _print_stat_test(df, "Log-Rank Test", getattr(args, 'output', None), fmt=fmt)
    elif getattr(args, 'output', None) and not df.empty:
        _write_csv(df, args.output, "Log-Rank Test")
    return 0


def cmd_rmst_test(args) -> int:
    """Run RMST comparison across risk groups on a saved learner."""
    learner = _load_learner_for_stat_test(args)
    fmt = getattr(args, 'output_format', 'table')
    try:
        df = learner.rmst_test(
            models=args.model,
            n_groups=args.n_groups,
            tau=args.tau,
            p_adjust=args.p_adjust,
            verbose=args.verbose,
        )
    except ValueError as exc:
        print(f"Error: {_cli_stat_error(exc)}", file=sys.stderr)
        return EXIT_COMPUTATION
    if df.empty:
        # Multi-model runs degrade per-model failures to warnings and an
        # empty table; an all-empty result is a failure at the CLI level.
        print(
            "Error: RMST comparison produced no results for any trained "
            "model (see warnings above).",
            file=sys.stderr,
        )
        return EXIT_COMPUTATION
    if not args.verbose:
        _print_stat_test(df, "RMST Comparison", getattr(args, 'output', None), fmt=fmt)
    elif getattr(args, 'output', None) and not df.empty:
        _write_csv(df, args.output, "RMST Comparison")
    return 0


def cmd_importance_test(args) -> int:
    """Compute feature importance p-values on a saved learner."""
    learner = _load_learner_for_stat_test(args)
    fmt = getattr(args, 'output_format', 'table')
    try:
        df = learner.importance_test(
            models=args.model,
            method=args.method,
            n_permutations=args.n_permutations,
            n_importance_repeats=args.n_importance_repeats,
            p_adjust=args.p_adjust,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
        )
    except ValueError as exc:
        print(f"Error: {_cli_stat_error(exc)}", file=sys.stderr)
        return EXIT_COMPUTATION
    if not args.verbose:
        _print_stat_test(df, "Importance P-Values", getattr(args, 'output', None), fmt=fmt)
    elif getattr(args, 'output', None) and not df.empty:
        _write_csv(df, args.output, "Importance P-Values")
    return 0


def cmd_linearity_test(args) -> int:
    """Test log-hazard linearity of continuous features on a saved learner."""
    learner = _load_learner_for_stat_test(args)
    fmt = getattr(args, 'output_format', 'table')
    try:
        df = learner.linearity_test(
            n_knots=args.n_knots,
            verbose=args.verbose,
        )
    except ValueError as exc:
        print(f"Error: {_cli_stat_error(exc)}", file=sys.stderr)
        return EXIT_COMPUTATION
    if not args.verbose:
        _print_stat_test(df, "Log-Hazard Linearity Test", getattr(args, 'output', None), fmt=fmt)
    elif getattr(args, 'output', None) and not df.empty:
        _write_csv(df, args.output, "Log-Hazard Linearity Test")
    return 0


def cmd_interactions(args) -> int:
    """Compute pairwise feature interactions on a saved learner."""
    learner = _load_learner_for_stat_test(args)
    quiet = getattr(args, 'quiet', False)
    fmt = getattr(args, 'output_format', 'table')
    try:
        df = learner.interactions(
            model=args.model,
            method=args.method,
            top_k=args.top_k,
            max_samples=args.max_samples,
            grid_resolution=args.grid_resolution,
            random_state=args.random_state,
            show_progress=not quiet,
        )
    except (MemoryError, RecursionError):
        raise
    # method='shap' on a non-tree model raises TypeError, and a missing shap
    # install raises ImportError; surface both as clean user errors like the
    # other stat-test commands (see _run_stat_tests_if_requested).
    except (ValueError, TypeError, RuntimeError, ImportError,
            np.linalg.LinAlgError) as exc:
        print(f"Error: {_cli_stat_error(exc)}", file=sys.stderr)
        return EXIT_COMPUTATION
    if df.empty:
        print(
            "Error: interactions produced no results (see warnings above).",
            file=sys.stderr,
        )
        return EXIT_COMPUTATION
    _print_stat_test(df, "Feature Interactions", getattr(args, 'output', None), fmt=fmt)
    return 0


def _infer_export_format(output: str) -> str:
    """Infer 'xlsx'/'csv' from an output path, warning on ambiguous extensions.

    An ``.xlsx`` path means xlsx; ``.csv`` or an extensionless/directory path
    means csv (one file per table). Any other extension (e.g. a ``.xls`` typo)
    would otherwise be written as a csv *directory* named like a file, so warn
    rather than silently picking csv.
    """
    from pathlib import Path
    suffix = Path(output).suffix.lower()
    if suffix == '.xlsx':
        return 'xlsx'
    if suffix not in ('', '.csv'):
        print(
            f"Warning: unrecognized output extension '{suffix}'; exporting as "
            "csv (a directory of CSV files). Pass --format xlsx or "
            "--format csv to be explicit.",
            file=sys.stderr,
        )
    return 'csv'


def cmd_export_results(args) -> int:
    """Export all available result tables to xlsx or csv from a saved learner."""
    learner = _load_learner_for_stat_test(args)
    quiet = getattr(args, 'quiet', False)
    fmt = args.format if args.format is not None else _infer_export_format(args.output)
    try:
        written = learner.export_results(args.output, format=fmt, audit=args.audit)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_COMPUTATION
    except (PermissionError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if not quiet:
        print(f"Results exported to: {written}")
    return 0
