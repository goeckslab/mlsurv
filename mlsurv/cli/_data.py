"""Data loading, validation, and learner setup for the mlsurv CLI."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

from ._exit_codes import CLIDataError


# ---------------------------------------------------------------------------
# Shared CSV error handling
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _cli_csv_errors(filepath):
    """Convert CSV-related exceptions to :class:`CLIDataError`.

    Catches :class:`~pandas.errors.EmptyDataError`,
    :class:`~pandas.errors.ParserError`, :class:`ValueError`,
    :class:`UnicodeDecodeError`, :class:`FileNotFoundError`, and
    :class:`PermissionError`.

    ``EmptyDataError`` and ``ParserError`` are :class:`ValueError`
    subclasses, so they must be caught first.
    """
    import pandas as pd

    try:
        yield
    except pd.errors.EmptyDataError:
        raise CLIDataError(
            f"File is empty: {filepath}\n"
            f"Provide a CSV file with at least a header row and one data row."
        ) from None
    except pd.errors.ParserError as exc:
        raise CLIDataError(
            f"Could not parse CSV file: {filepath}\n"
            f"Details: {exc}\n"
            f"Check that the file is a valid CSV with consistent columns."
        ) from exc
    except ValueError as exc:
        raise CLIDataError(str(exc)) from exc
    except UnicodeDecodeError as exc:
        raise CLIDataError(
            f"Cannot decode file: {filepath}\n"
            f"Details: {exc}\n"
            f"The file may use a non-UTF-8 encoding. Try:\n"
            f"  pd.read_csv('{filepath}', encoding='latin-1')"
        ) from exc
    except FileNotFoundError:
        raise CLIDataError(
            f"Data file not found: {filepath}\n"
            f"Check the path or use an absolute path."
        ) from None
    except PermissionError:
        raise CLIDataError(
            f"Permission denied reading file: {filepath}\n"
            f"Check file permissions or run with appropriate access."
        ) from None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _build_load_kwargs(args, *, feature_names=None) -> dict:
    """Build keyword arguments for :func:`load_survival_data` from CLI args."""
    if feature_names is not None:
        return {'feature_names': feature_names}
    return dict(
        time_col=getattr(args, 'time_col', None),
        event_col=getattr(args, 'event_col', None),
        start_col=getattr(args, 'start_col', None),
        end_col=getattr(args, 'end_col', None),
        time_unit=getattr(args, 'time_unit', 'days'),
        event_values=getattr(args, 'event_values', None),
    )


def _cli_load(args, *, feature_names=None):
    """CLI wrapper around :func:`load_survival_data` with friendly errors.

    Supports ``-`` as a data path to read CSV from stdin.
    """
    from mlsurv.utils import load_survival_data

    if args.data == '-':
        return _cli_load_stdin(args, feature_names=feature_names)

    _validate_csv_path(args.data)
    kwargs = _build_load_kwargs(args, feature_names=feature_names)

    with _cli_csv_errors(args.data):
        return load_survival_data(args.data, **kwargs)


def _cli_load_stdin(args, *, feature_names=None):
    """Read CSV data from stdin and load it via a temporary file."""
    import os
    import tempfile

    from mlsurv.utils import load_survival_data

    if sys.stdin.isatty():
        raise CLIDataError(
            "'-' was specified but no data is being piped to stdin. "
            "Usage: cat data.csv | mlsurv run -"
        )

    raw = sys.stdin.read()
    if not raw.strip():
        raise CLIDataError("No data received on stdin.")

    fd, tmp_path = tempfile.mkstemp(suffix='.csv')
    try:
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(raw)
        except OSError:
            # os.fdopen may not have taken ownership of fd
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        kwargs = _build_load_kwargs(args, feature_names=feature_names)
        with _cli_csv_errors(tmp_path):
            return load_survival_data(tmp_path, **kwargs)
    finally:
        os.unlink(tmp_path)


def _validate_csv_path(data: str) -> Path | None:
    """Validate that *data* is an existing CSV file path.

    Returns *None* for the stdin marker ``'-'``.
    """
    if data == '-':
        return None
    data_path = Path(data)
    if not data_path.exists():
        raise CLIDataError(
            f"Data file not found: {data_path}\n"
            f"Check the path or use an absolute path."
        )
    if not data_path.is_file():
        raise CLIDataError(
            f"Path is not a file: {data_path}\n"
            f"Provide a path to a CSV file, not a directory."
        )
    return data_path


def _read_csv(data_path: Path):
    """Read a CSV file with user-friendly error messages."""
    import pandas as pd

    with _cli_csv_errors(data_path):
        return pd.read_csv(data_path)


def load_reference_csv(
    reference_file: str,
    reference_cols: list | None = None,
    expected_n_rows: int | None = None,
) -> dict:
    """Load reference scores from an external CSV file.

    Parameters
    ----------
    reference_file : str
        Path to a CSV containing reference score columns.
    reference_cols : list of str, optional
        Columns to use.  If *None*, all columns are used.
    expected_n_rows : int, optional
        Expected row count (for alignment validation with the data file).

    Returns
    -------
    dict of {column_name: np.ndarray}

    Raises
    ------
    CLIDataError
        If any requested ``reference_cols`` are absent from the file, if the
        file has no usable columns, if any selected column is non-numeric, or
        if the row count does not match ``expected_n_rows``.
    """
    import numpy as np

    ref_path = _validate_csv_path(reference_file)
    ref_df = _read_csv(ref_path)

    if reference_cols is not None:
        missing = [c for c in reference_cols if c not in ref_df.columns]
        if missing:
            raise CLIDataError(
                f"columns not found in reference file {ref_path}: "
                f"{missing}\n"
                f"Available columns: {ref_df.columns.tolist()}"
            )
        ref_df = ref_df[reference_cols]

    if ref_df.empty or ref_df.shape[1] == 0:
        raise CLIDataError(
            f"reference file has no usable columns: {ref_path}"
        )

    import pandas as pd

    non_numeric = [
        c for c in ref_df.columns
        if not pd.api.types.is_numeric_dtype(ref_df[c])
    ]
    if non_numeric:
        samples = {c: ref_df[c].head(3).tolist() for c in non_numeric}
        raise CLIDataError(
            f"non-numeric columns in reference file {ref_path}: "
            f"{non_numeric}\n"
            f"Sample values: {samples}\n"
            f"Reference columns must contain numeric scores."
        )

    if expected_n_rows is not None and len(ref_df) != expected_n_rows:
        raise CLIDataError(
            f"reference file has {len(ref_df)} rows, but data file "
            f"has {expected_n_rows} rows. Row counts must match "
            f"(one reference score per patient, same order)."
        )

    return {col: ref_df[col].to_numpy(dtype=np.float64) for col in ref_df.columns}


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

_ARG_VALIDATIONS = [
    ('cv_folds', lambda v: v >= 2, "--cv-folds must be >= 2"),
    ('n_trials', lambda v: v == 'auto' or v >= 1, "--n-trials must be a positive integer or 'auto'"),
    ('cv_repeats', lambda v: v >= 1, "--cv-repeats must be >= 1"),
    ('cv_ci', lambda v: 0 < v < 1, "--cv-ci must be between 0 and 1"),
    ('timeout', lambda v: v > 0, "--timeout must be > 0"),
    ('shap_max_background', lambda v: v >= 1, "--shap-max-background must be >= 1"),
    ('shap_max_explain', lambda v: v >= 1, "--shap-max-explain must be >= 1"),
    ('n_features', lambda v: v >= 1, "--n-features must be >= 1"),
    ('max_background', lambda v: v >= 1, "--max-background must be >= 1"),
    ('test_fraction', lambda v: 0 < v < 1, "--test-fraction must be between 0 and 1 (exclusive)"),
    ('n_jobs', lambda v: v != 0, "--n-jobs must be -1 or >= 1"),
    ('interaction_top_k', lambda v: v >= 2, "--interaction-top-k must be >= 2"),
    ('n_groups', lambda v: v >= 2, "--n-groups must be >= 2"),
    ('n_permutations', lambda v: v >= 1, "--n-permutations must be >= 1"),
    ('n_bins', lambda v: v >= 2, "--n-bins must be >= 2"),
    ('eval_time', lambda v: v > 0, "--eval-time must be > 0"),
    ('time_limit', lambda v: v > 0, "--time-limit must be > 0"),
    ('threshold', lambda v: 0 < v <= 1, "--threshold must be between 0 and 1"),
    ('max_samples', lambda v: v >= 1, "--max-samples must be >= 1"),
    ('grid_resolution', lambda v: v >= 2, "--grid-resolution must be >= 2"),
]


def _validate_args(parser, args) -> None:
    """Validate numeric arguments."""
    if getattr(args, 'quiet', False) and getattr(args, 'verbose', False):
        parser.error("--quiet and --verbose are mutually exclusive")

    for attr, check, msg in _ARG_VALIDATIONS:
        val = getattr(args, attr, None)
        if val is not None and not check(val):
            parser.error(msg)

    _validate_cross_arg_constraints(parser, args)


# (predicate, message). `predicate(args)` returns truthy when the rule is
# violated. `message` is a str, or a callable taking `args` for dynamic
# interpolation. `n_bootstrap` allows 0 in `sweep`; other commands reject
# 0 upstream in `_parse_n_bootstrap`, so < 0 is sufficient here.
_CROSS_ARG_CONSTRAINTS = [
    (lambda a: isinstance(getattr(a, 'bootstrap', None), int)
               and not isinstance(a.bootstrap, bool)
               and a.bootstrap < 1,
     "--bootstrap must be >= 1"),
    (lambda a: getattr(a, 'cv_strategy', None) == 'groupkfold'
               and not getattr(a, 'group_col', None),
     "--group-col is required when --cv-strategy=groupkfold"),
    (lambda a: getattr(a, 'n_bootstrap', None) not in (None, 'auto')
               and int(a.n_bootstrap) < 0,
     "--n-bootstrap must be >= 0"),
    (lambda a: getattr(a, 'feature_groups_file', None) is not None
               and not Path(a.feature_groups_file).exists(),
     lambda a: f"Feature groups file not found: {a.feature_groups_file}"),
    (lambda a: getattr(a, 'load_learner', None) is not None
               and getattr(a, 'load_artifact', None) is not None,
     "--load-learner and --load-artifact are mutually exclusive"),
    (lambda a: getattr(a, 'load_learner', None) is not None
               and getattr(a, 'data', None) is not None
               and a.command != 'predict',
     "--load-learner and a data file are mutually exclusive. "
     "The learner already contains training data."),
    (lambda a: getattr(a, 'subgroups', None) is not None
               and getattr(a, 'load_learner', None) is not None,
     "--subgroups requires a data file (cannot use with --load-learner)"),
    (lambda a: getattr(a, 'data', None) is None
               and getattr(a, 'load_learner', None) is None
               and a.command in ('evaluate', 'bootstrap', 'validate'),
     "provide either a data file or --load-learner"),
    (lambda a: a.command == 'predict'
               and getattr(a, 'load_artifact', None) is None
               and getattr(a, 'load_learner', None) is None,
     "predict requires --load-artifact or --load-learner"),
    (lambda a: getattr(a, 'ci', False)
               and getattr(a, 'load_learner', None) is None,
     "--ci requires --load-learner (bootstrap CIs need training "
     "data which is not available in model-only artifacts)"),
    (lambda a: getattr(a, 'ci_level', None) is not None
               and not (0 < a.ci_level < 1),
     "--ci-level must be between 0 and 1 (exclusive)"),
]


def _validate_cross_arg_constraints(parser, args) -> None:
    """Validate constraints that span multiple arguments."""
    for predicate, message in _CROSS_ARG_CONSTRAINTS:
        if predicate(args):
            parser.error(message(args) if callable(message) else message)

    for attr_prefix, flag_prefix in (('', ''), ('val_', 'val-')):
        _validate_date_arg_pair(
            parser,
            start_col=getattr(args, f'{attr_prefix}start_col', None),
            end_col=getattr(args, f'{attr_prefix}end_col', None),
            time_col=getattr(args, f'{attr_prefix}time_col', 'time'),
            start_flag=f"--{flag_prefix}start-col",
            end_flag=f"--{flag_prefix}end-col",
            time_flag=f"--{flag_prefix}time-col",
        )

    if getattr(args, 'load_learner', None) is not None:
        _warn_ignored_setup_args(args)


_SETUP_ARG_DEFAULTS = {
    'cv_folds': 5,
    'cv_repeats': 1,
    'cv_ci': 0.95,
    'cv_strategy': 'stratifiedkfold',
    'test_fraction': 0.3,
    'random_state': 42,
    'scaler': 'standard',
    'imputer': 'median',
    'feature_selector': 'none',
    'n_jobs': 1,
}


def _warn_ignored_setup_args(args) -> None:
    """Warn when setup-related args differ from defaults with --load-learner.

    The loaded learner already has its own settings.  Per the project's
    "No Silent Data Loss" rule, user-provided parameters that will be
    discarded must emit a warning.
    """
    overridden = [
        f"--{k.replace('_', '-')}"
        for k, default in _SETUP_ARG_DEFAULTS.items()
        if getattr(args, k, default) != default
    ]
    if overridden:
        print(
            f"Warning: {', '.join(overridden)} ignored when using "
            f"--load-learner (the loaded learner already has its own "
            f"setup configuration).",
            file=sys.stderr,
        )


def _validate_date_arg_pair(
    parser,
    *,
    start_col,
    end_col,
    time_col,
    start_flag: str,
    end_flag: str,
    time_flag: str,
) -> None:
    """Validate paired start/end date args and time-column exclusivity."""
    if (start_col is None) != (end_col is None):
        parser.error(f"{start_flag} and {end_flag} must be used together")
    if start_col is not None and end_col is not None and time_col != 'time':
        parser.error(
            f"{start_flag}/{end_flag} and {time_flag} are mutually exclusive"
        )


# ---------------------------------------------------------------------------
# Subpopulation helpers
# ---------------------------------------------------------------------------


def _build_subpopulations(args, X):
    """Build subpopulations dict from ``--subgroups`` columns.

    Returns ``(dict | None, error_code)``.  When *error_code* is non-zero the
    caller should ``return error_code`` immediately.
    """
    if args.subgroups is None:
        return None, 0
    missing = [c for c in args.subgroups if c not in X.columns]
    if missing:
        print(
            f"Error: subgroup columns not found in data: {missing}",
            file=sys.stderr,
        )
        from ._exit_codes import EXIT_DATA
        return None, EXIT_DATA
    return {col: X[col].to_numpy() for col in args.subgroups}, 0


# ---------------------------------------------------------------------------
# Learner setup
# ---------------------------------------------------------------------------

_SETUP_PASSTHROUGH = ['cv_folds', 'cv_repeats', 'cv_ci', 'cv_strategy', 'scoring', 'n_jobs', 'device']
_SETUP_NONE_MEANS_NONE = ['imputer', 'scaler', 'feature_selector']


def _setup_learner(learner, args, *, subpopulations=None) -> None:
    """Configure learner setup from shared CLI arguments."""
    kw = dict(test_fraction=args.test_fraction, random_state=args.random_state)
    for attr in _SETUP_PASSTHROUGH:
        val = getattr(args, attr, None)
        if val is not None:
            kw[attr] = val
    for attr in _SETUP_NONE_MEANS_NONE:
        val = getattr(args, attr, None)
        if val is not None:
            kw[attr] = None if val == 'none' else val
    if getattr(args, 'group_col', None):
        kw['groups'] = args.group_col
    if subpopulations is not None:
        kw['subpopulations'] = subpopulations
    learner.setup(**kw)
    state = getattr(learner, '_state', None)
    if state is not None:
        state.time_col = getattr(args, 'time_col', None)
        state.event_col = getattr(args, 'event_col', None)
