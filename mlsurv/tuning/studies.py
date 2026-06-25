"""Study listing, export, and record extraction for Optuna studies."""

import json
from pathlib import Path

import optuna
import pandas as pd

from .storage import DEFAULT_STORAGE_BASE


def _parse_study_summary(s):
    """Parse a single Optuna study summary into (experiment_name, study_info)."""
    if '/' in s.study_name:
        exp_name, study_suffix = s.study_name.rsplit('/', 1)
    else:
        exp_name, study_suffix = '_default', s.study_name

    study_info = {
        'name': study_suffix,
        'full_name': s.study_name,
        'n_trials': s.n_trials,
        'best_value': s.best_trial.value if s.best_trial else None,
        'datetime_start': s.datetime_start,
    }
    return exp_name, study_info


def _list_studies_from_storage(storage):
    """Query studies from an Optuna storage backend, grouped by experiment name."""
    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage)
    except (MemoryError, RecursionError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to list studies from storage: {e}") from e

    studies = {}
    for s in summaries:
        exp_name, study_info = _parse_study_summary(s)
        studies.setdefault(exp_name, []).append(study_info)
    return studies


def _list_studies_from_files(base_dir):
    """Scan local SQLite study files, grouped by experiment directory."""
    base_dir = Path(base_dir if base_dir is not None else DEFAULT_STORAGE_BASE)
    if not base_dir.exists():
        return {}

    studies = {}
    for subdir in base_dir.iterdir():
        if not subdir.is_dir():
            continue
        db_files = list(subdir.glob('*_study.db'))
        if db_files:
            studies[subdir.name] = [f.name for f in db_files]
    return studies


def list_studies(base_dir=None, storage=None):
    """
    List all available Optuna studies.

    When `storage` is provided, queries the database directly and groups
    studies by experiment_name (extracted from study name prefix).
    Otherwise, scans local SQLite files in `base_dir`.

    Parameters
    ----------
    base_dir : str or Path, optional
        Base directory to search for SQLite files. Defaults to ./optuna_studies/
        Ignored if `storage` is provided.
    storage : str, optional
        Optuna storage URL (e.g., 'postgresql://user:pass@host:port/db').
        If provided, queries studies directly from the database.

    Returns
    -------
    dict
        If storage is provided: dict mapping experiment_name to list of study info:
            {
                'gbsg2_analysis': [
                    {'name': 'rsf_tuning', 'n_trials': 50, 'best_value': 0.85, ...}
                ],
                'flchain_cohort': [
                    {'name': 'rsf_tuning', 'n_trials': 30, 'best_value': 0.72, ...}
                ]
            }
        If scanning files: dict mapping experiment_name to list of model study files:
            {'gbsg2_analysis': ['rsf_study.db', 'gbsa_study.db']}

    Examples
    --------
    >>> # List studies from PostgreSQL
    >>> studies = list_studies(storage='postgresql://localhost/optuna')
    >>> for exp_name, exp_studies in studies.items():
    ...     print(f"{exp_name}:")
    ...     for s in exp_studies:
    ...         print(f"  {s['name']}: {s['n_trials']} trials, best={s['best_value']:.4f}")

    >>> # List local SQLite studies
    >>> studies = list_studies()
    >>> for exp_name, files in studies.items():
    ...     print(f"{exp_name}: {files}")
    """
    if storage is not None:
        return _list_studies_from_storage(storage)
    return _list_studies_from_files(base_dir)


def get_study_path(experiment_name, model_name=None, base_dir=None):
    """
    Get the full path to an experiment directory or specific study file.

    Parameters
    ----------
    experiment_name : str
        Experiment name (directory name under optuna_studies/).
    model_name : str, optional
        If provided, returns path to specific model's study file.
    base_dir : str or Path, optional
        Base directory. Defaults to ./optuna_studies/

    Returns
    -------
    Path
        Full path to experiment directory or study file.
    """
    if base_dir is None:
        base_dir = DEFAULT_STORAGE_BASE

    full_path = Path(base_dir) / experiment_name

    if model_name is not None:
        full_path = full_path / f'{model_name}_study.db'

    return full_path


def _extract_trial_record(trial, all_param_names):
    """Extract a single trial into a flat record dict.

    Parameters
    ----------
    trial : optuna.trial.FrozenTrial
        The trial to extract.
    all_param_names : list of str
        Sorted union of all parameter names across all trials.

    Returns
    -------
    dict
        Flat record with trial metadata and param_* columns.
    """
    has_timestamps = trial.datetime_start is not None and trial.datetime_complete is not None
    record = {
        'trial_number': trial.number,
        'value': trial.value,
        'state': trial.state.name,
        'duration_seconds': (
            (trial.datetime_complete - trial.datetime_start).total_seconds()
            if has_timestamps else None
        ),
        'datetime_start': str(trial.datetime_start) if trial.datetime_start else None,
        'datetime_complete': str(trial.datetime_complete) if trial.datetime_complete else None,
    }
    for param_name in all_param_names:
        record[f'param_{param_name}'] = trial.params.get(param_name)
    return record


def _extract_study_metadata(study):
    """Extract study-level metadata from an Optuna study.

    Parameters
    ----------
    study : optuna.Study
        The study to extract metadata from.

    Returns
    -------
    dict
        Study metadata including best trial info and summary stats.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = study.best_trial if completed else None
    starts = [t.datetime_start for t in study.trials if t.datetime_start is not None]

    return {
        'study_name': study.study_name,
        'direction': study.direction.name,
        'sampler': type(study.sampler).__name__,
        'n_trials': len(study.trials),
        'n_completed': len(completed),
        'best_trial_number': best.number if best else None,
        'best_value': best.value if best else None,
        'best_params': best.params if best else None,
        'datetime_start': str(min(starts)) if starts else None,
    }


def _study_to_records(study):
    """Extract structured trial data and metadata from an Optuna study.

    Parameters
    ----------
    study : optuna.Study
        The Optuna study to extract data from.

    Returns
    -------
    dict
        Dictionary with 'trials' (list of dicts) and 'metadata' (dict).
    """
    all_param_names = sorted({
        name for trial in study.trials for name in trial.params
    })

    trial_records = [
        _extract_trial_record(trial, all_param_names)
        for trial in study.trials
    ]

    return {'trials': trial_records, 'metadata': _extract_study_metadata(study)}


def _resolve_study(
    study, study_name, storage, experiment_name, model_name, base_dir,
):
    """Resolve an Optuna study from the various argument combinations.

    Parameters
    ----------
    study : optuna.Study or None
        Pre-loaded study object.
    study_name : str or None
        Optuna study name (requires storage).
    storage : str or None
        Optuna storage URL (requires study_name).
    experiment_name : str or None
        Experiment name (requires model_name).
    model_name : str or None
        Model name (requires experiment_name).
    base_dir : str, Path, or None
        Base directory for study files.

    Returns
    -------
    optuna.Study

    Raises
    ------
    ValueError
        If insufficient arguments to locate a study.
    FileNotFoundError
        If the study database file does not exist on disk.
    """
    if study is not None:
        return study

    if study_name is not None and storage is not None:
        return optuna.load_study(study_name=study_name, storage=storage)

    if experiment_name is not None and model_name is not None:
        db_path = get_study_path(experiment_name, model_name, base_dir)
        if not db_path.exists():
            raise FileNotFoundError(
                f"Study database not found: {db_path}. "
                f"Run tune('{model_name}', experiment_name='{experiment_name}') first."
            )
        storage_url = f'sqlite:///{db_path}'
        resolved_name = f'{experiment_name}/{model_name}_tuning'
        try:
            return optuna.load_study(study_name=resolved_name, storage=storage_url)
        except KeyError:
            raise ValueError(
                f"Study '{resolved_name}' not found in {db_path}. "
                f"The database exists but may use a different study name."
            ) from None

    incomplete_pairs = [
        (study_name, "study_name", "storage"),
        (storage, "storage", "study_name"),
        (experiment_name, "experiment_name", "model_name"),
        (model_name, "model_name", "experiment_name"),
    ]
    for value, name, partner in incomplete_pairs:
        if value is not None:
            raise ValueError(f"'{name}' requires '{partner}' to be provided.")

    raise ValueError(
        "Provide 'study', 'study_name'+'storage', "
        "or 'experiment_name'+'model_name'."
    )


def _write_export(result, path, output_format):
    """Write export result to a file.

    Parameters
    ----------
    result : dict
        Export data with 'trials' and 'metadata' keys.
    path : Path
        Output file path.
    output_format : str
        Output format: 'csv' or 'json'.

    Raises
    ------
    OSError
        If the output file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == 'csv':
            pd.DataFrame(result['trials']).to_csv(path, index=False)
        else:
            with open(path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
    except OSError as e:
        raise OSError(
            f"Failed to write export file to '{path}': {e}. "
            f"Check that the directory exists and is writable."
        ) from e


def export_study(
    study=None, *, study_name=None, storage=None,
    experiment_name=None, model_name=None,
    path=None, format=None, base_dir=None,
):
    """Export Optuna study history to CSV or JSON for reproducibility.

    Parameters
    ----------
    study : optuna.Study, optional
        An already-loaded study. If None, loads from storage.
    study_name : str, optional
        Optuna study name. Used with ``storage`` as an alternative to
        experiment_name/model_name for locating a study.
    storage : str, optional
        Optuna storage URL. Used with ``study_name`` as an alternative to
        experiment_name/model_name for locating a study.
    experiment_name : str, optional
        Experiment name (used with model_name to locate study on disk).
    model_name : str, optional
        Model name (used with experiment_name to locate study on disk).
    path : str or Path, optional
        Output file path. If None, no file is written.
    format : {'csv', 'json'}, optional
        Output format. If None, inferred from path extension; defaults to 'json'.
    base_dir : str or Path, optional
        Base directory for study files. Defaults to ./optuna_studies/

    Returns
    -------
    dict
        Dictionary with 'trials' (list of dicts) and 'metadata' (dict).
        Returned regardless of whether a file is written.

    Raises
    ------
    ValueError
        If insufficient arguments to locate a study, or if the study name
        is not found in the storage database.
    FileNotFoundError
        If the study database file does not exist on disk (when using
        experiment_name/model_name).
    OSError
        If the output file cannot be written (permission denied, disk full, etc.).
    """
    study = _resolve_study(
        study, study_name, storage, experiment_name, model_name, base_dir,
    )

    result = _study_to_records(study)

    if path is not None:
        if format is not None:
            output_format = format
        elif Path(path).suffix.lower() == '.csv':
            output_format = 'csv'
        else:
            output_format = 'json'
        _write_export(result, Path(path), output_format)

    return result
