"""Utility functions for mlsurv."""

import contextlib
import warnings
from functools import lru_cache
from typing import NewType, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from ._warnings import InternalWarning, MethodologicalWarning, SuggestionWarning
from .logging_utils import get_logger

_COMPETING_RISKS_MSG = (
    "mlsurv currently treats all non-zero values as events (single-risk "
    "assumption). If these represent distinct event types (e.g., competing "
    "risks), results may be biased. Consider filtering to a single event "
    "type or using a competing risks framework."
)
_AMBIGUOUS_BINARY_MSG = (
    "Binary numeric event columns with two non-zero codes are ambiguous. "
    "Recode to 0/1 (0=censored, 1=event), or use load_survival_data() / "
    "_parse_event_values() with event_values=[<event_code>] to disambiguate."
)

# NumPy 2.x: np.trapz renamed to np.trapezoid
try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz

logger = get_logger("utils")

SurvivalArray = NewType('SurvivalArray', np.ndarray)
"""Structured ``np.ndarray`` with dtype ``[('event', bool), ('time', float)]``."""


def _validate_numeric_event_codes(
    values: np.ndarray, source: str, *, stacklevel: int = 3
) -> None:
    """Reject ambiguous binary numeric encodings and warn on competing risks.

    ``stacklevel`` is forwarded to the competing-risks warning so it points at
    user code; bump it by one for every extra wrapper between this function and
    the caller.
    """
    unique_values = np.unique(values)
    if len(unique_values) > 2:
        warnings.warn(
            f"{source} contains {len(unique_values)} unique values "
            f"{sorted(unique_values.tolist())}. {_COMPETING_RISKS_MSG}",
            MethodologicalWarning,
            stacklevel=stacklevel,
        )
        return

    if len(unique_values) == 2 and 0 not in unique_values:
        raise ValueError(
            f"{source} contains ambiguous binary numeric values "
            f"{sorted(unique_values.tolist())}. {_AMBIGUOUS_BINARY_MSG}"
        )


def _validate_event_values(events: np.ndarray, *, codes_stacklevel: int = 4) -> None:
    """Validate an event array's encoding without coercing it to bool.

    Raises on NaN-float events and ambiguous binary numeric codes; warns on
    competing risks (>2 codes). Accepts bool, 0/1-int, and 0.0/1.0-float
    encodings (the package coerces with ``.astype(bool)`` everywhere), so it
    checks encoding validity only — it never requires bool dtype.
    """
    events = np.asarray(events)
    # NaN must be caught before any bool conversion (NaN coerces to True).
    if np.issubdtype(events.dtype, np.floating) and np.any(np.isnan(events)):
        raise ValueError("Event array contains NaN values")
    if not np.issubdtype(events.dtype, np.bool_):
        _validate_numeric_event_codes(
            events, "Event column", stacklevel=codes_stacklevel
        )


def _reject_nonfinite_times(times: np.ndarray) -> None:
    """Raise on NaN or infinite survival times."""
    if np.any(np.isnan(times)):
        raise ValueError("Survival times contain NaN values.")
    if np.any(np.isinf(times)):
        raise ValueError("Survival times contain infinite values.")


def _validate_survival_times(times: np.ndarray) -> None:
    """Validate survival times without mutating: raise on NaN, inf, negative, or zero.

    Keeps negative and zero as separate branches: negative times indicate data
    errors, while zero times are clinically valid but incompatible with
    log-based models. Negative takes priority (and mentions any zeros).
    """
    _reject_nonfinite_times(times)

    neg_mask = times < 0
    zero_mask = times == 0
    n_zero = int(zero_mask.sum())

    if np.any(neg_mask):
        n_neg = int(neg_mask.sum())
        bad_idx = np.where(neg_mask)[0][:5].tolist()
        also_zero = (
            f" Additionally, {n_zero} observation(s) have time=0."
            if n_zero else ""
        )
        raise ValueError(
            f"{n_neg} negative survival time(s) detected "
            f"(indices: {bad_idx}).\n"
            f"Negative times indicate data errors "
            f"(e.g., end date before start date).{also_zero}\n"
            f"Filter with: mask = times > 0; "
            f"y = to_structured_array(times[mask], events[mask])"
        )

    if np.any(zero_mask):
        bad_idx = np.where(zero_mask)[0][:5].tolist()
        raise ValueError(
            f"{n_zero} observation(s) have time=0 "
            f"(indices: {bad_idx}).\n"
            f"Zero survival times are clinically valid (same-day "
            f"events) but incompatible with log-based survival "
            f"models.\n"
            f"Options:\n"
            f"  1. Filter:  mask = times > 0; "
            f"y = to_structured_array(times[mask], events[mask])\n"
            f"  2. Epsilon: times[times == 0] = 1e-10\n"
            f"  3. Auto-drop: to_structured_array(times, events, "
            f"drop_nonpositive=True)"
        )


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager for tqdm progress bar with joblib parallel execution.

    Note: This patches ``joblib.parallel.BatchCompletionCallBack`` at module
    level, so it is **not** thread-safe.  Concurrent usage from multiple
    threads would overwrite each other's callback.  In the current codebase
    this is safe because ``tqdm_joblib`` is only used from the main thread
    and inner parallel workers have ``show_progress=False``.
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def to_structured_array(
    times: Union[list, np.ndarray, pd.Series],
    events: Optional[Union[list, np.ndarray, pd.Series]] = None,
    *,
    drop_nonpositive: bool = False,
) -> SurvivalArray:
    """
    Create a SurvivalArray for survival outcomes.

    Supports two calling conventions:

    1. **Separate columns** — pass time and event arrays:
       ``to_structured_array(times, events)``.
    2. **Normalize a structured array** — pass a single 2-field structured
       numpy array: ``to_structured_array(y_struct)``.  This auto-detects the
       event field by dtype (the boolean field is ``event``, the other is
       ``time``), so it is robust to the inconsistent field names and field
       order used by scikit-survival datasets (e.g. ``('cens', 'time')``,
       ``('e.tdm', 't.tdm')``, ``('Status', 'Survival_in_days')``).

    Both conventions share the same validation (length, NaN/inf, nonpositive
    times, ``drop_nonpositive``) and return mlsurv's canonical SurvivalArray.

    Parameters
    ----------
    times : array-like or numpy.ndarray
        Survival times (must be positive).  When `events` is omitted, this is
        instead a 2-field structured numpy array to normalize (see convention 2).
    events : array-like, optional
        Event indicators (True/1 = event occurred, False/0 = censored).  Omit
        to use the structured-array convention.
    drop_nonpositive : bool, default False
        If True, drop observations with non-positive times and emit a
        :class:`MethodologicalWarning`.  If False (default), raise
        a :class:`ValueError` with a descriptive message and suggestions.

    Returns
    -------
    SurvivalArray
        SurvivalArray with dtype [('event', bool), ('time', float)].

    Raises
    ------
    TypeError
        When using the structured-array convention, if `times` is not a
        structured numpy array, does not have exactly 2 fields, or does not
        have exactly one boolean field (zero or more than one).
    ValueError
        If the event array contains NaN, `times` and `events` differ in
        length, the times contain NaN or infinite values, or (when
        `drop_nonpositive` is False) any time is non-positive; also if
        dropping non-positive times would leave no observations.

    Examples
    --------
    Separate columns:

    >>> times = [1.5, 2.0, 3.5]
    >>> events = [True, False, True]
    >>> y = to_structured_array(times, events)
    >>> y.dtype
    dtype([('event', '?'), ('time', '<f8')])

    A scikit-survival structured array (any field names, any order):

    >>> import numpy as np
    >>> y_sksurv = np.array(
    ...     [(True, 12.0), (False, 5.3)],
    ...     dtype=[('cens', bool), ('time', '<f8')],
    ... )
    >>> y = to_structured_array(y_sksurv)
    >>> y.dtype
    dtype([('event', '?'), ('time', '<f8')])
    """
    if events is None:
        # Structured-array convention: normalize a 2-field survival array
        # (one boolean field = event, the other = time) to canonical fields,
        # then fall through to the shared column-path validation below.
        if not isinstance(times, np.ndarray) or times.dtype.names is None:
            raise TypeError(
                "Expected a structured numpy array (e.g. from scikit-survival), "
                f"got {type(times).__name__}"
                + (" with no named fields." if isinstance(times, np.ndarray) else ".")
            )

        names = times.dtype.names
        if len(names) != 2:
            raise TypeError(
                f"Expected exactly 2 fields (event, time), got {len(names)}: {names}."
            )

        # Auto-detect: the boolean field is 'event', the other is 'time'
        bool_fields = [n for n in names if np.issubdtype(times.dtype[n], np.bool_)]
        if len(bool_fields) != 1:
            raise TypeError(
                f"Cannot auto-detect event field: expected exactly one boolean "
                f"field, got {len(bool_fields)}. Field dtypes: "
                + ", ".join(f"{n}={times.dtype[n]}" for n in names)
                + ". Convert the event field to bool before calling "
                "to_structured_array()."
            )

        event_field = bool_fields[0]
        time_field = [n for n in names if n != event_field][0]
        events = times[event_field]
        times = times[time_field]

    times = np.asarray(times, dtype=float)

    # Validate events before bool conversion (NaN coerces to True incorrectly).
    # codes_stacklevel=4: warn -> _validate_numeric_event_codes ->
    # _validate_event_values -> to_structured_array -> user.
    events = np.asarray(events)
    _validate_event_values(events, codes_stacklevel=4)
    events = events.astype(bool)

    if len(times) != len(events):
        raise ValueError(
            f"times and events must have the same length. "
            f"Got {len(times)} and {len(events)}."
        )

    # Validate times
    if drop_nonpositive:
        # NaN/inf can't be dropped meaningfully, so still raise on them; then
        # drop negative and zero together (with a warning).
        _reject_nonfinite_times(times)
        nonpos_mask = (times < 0) | (times == 0)
        if np.any(nonpos_mask):
            n_neg = int((times < 0).sum())
            n_zero = int((times == 0).sum())
            parts = []
            if n_neg:
                parts.append(f"{n_neg} negative")
            if n_zero:
                parts.append(f"{n_zero} zero")
            warnings.warn(
                f"Dropped {int(nonpos_mask.sum())} observation(s) with "
                f"non-positive survival times ({', '.join(parts)}). "
                f"Survival models require strictly positive times.",
                MethodologicalWarning,
                stacklevel=2,
            )
            keep = ~nonpos_mask
            times = times[keep]
            events = events[keep]
            if len(times) == 0:
                raise ValueError(
                    f"All {int(nonpos_mask.sum())} observation(s) have "
                    f"non-positive survival times. No data remains after "
                    f"filtering."
                )
    else:
        _validate_survival_times(times)

    y = np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)]
    )
    return SurvivalArray(y)


def _parse_event_values(
    series: pd.Series,
    event_values: Optional[list] = None,
) -> np.ndarray:
    """Parse an event column into a boolean array.

    String values are stripped of leading/trailing whitespace before
    comparison.  Known true labels: ``1, true, yes, y, deceased, dead,
    death, event``.  Known false labels: ``0, false, no, n, living,
    alive, censored, ""`` (empty string).  Unknown strings are treated
    as censored and emit a warning with guidance on using
    ``event_values``.

    Parameters
    ----------
    series : pd.Series
        Raw event column from a DataFrame.
    event_values : list, optional
        Values that indicate an event occurred. When provided,
        ``series.isin(event_values)`` is used. When ``None``,
        string columns are mapped via known true/false labels
        and numeric columns are cast to bool.

    Returns
    -------
    np.ndarray
        Boolean array (True = event, False = censored).

    Raises
    ------
    ValueError
        If *series* contains missing (NaN/None) values.
    """
    if series.isna().any():
        n_missing = int(series.isna().sum())
        raise ValueError(
            f"Event column contains {n_missing} missing (NaN/None) value(s). "
            f"Impute or drop these rows before computing survival outcomes."
        )

    if event_values is not None:
        if not event_values:
            raise ValueError("event_values must not be empty when provided.")
        # CLI delivers strings via argparse; coerce to column dtype for matching
        coerced = event_values
        if not pd.api.types.is_string_dtype(series):
            try:
                coerced = [series.dtype.type(v) for v in event_values]
            except (ValueError, TypeError):
                pass
        result = series.isin(coerced)
        if not result.any():
            warnings.warn(
                f"No rows matched event_values {event_values}. "
                f"Column dtype is {series.dtype} with unique values "
                f"{sorted(series.unique().tolist())}. "
                f"Check for type mismatches or typos.",
                SuggestionWarning,
                stacklevel=2,
            )
        return result.values

    if pd.api.types.is_string_dtype(series):
        known_true = {'1', 'true', 'yes', 'y', 'deceased', 'dead', 'death', 'event'}
        known_false = {'0', 'false', 'no', 'n', 'living', 'alive', 'censored', ''}
        lowered = series.astype(str).str.strip().str.lower()
        unknown = ~lowered.isin(known_true | known_false)
        if unknown.any():
            unknown_vals = series[unknown].unique().tolist()
            warnings.warn(
                f"Unknown event values treated as censored: {unknown_vals}. "
                f"To treat these as events, pass "
                f"event_values={unknown_vals} explicitly.",
                MethodologicalWarning,
                stacklevel=2,
            )
        return lowered.isin(known_true).values

    _validate_numeric_event_codes(series.values, "Event column")
    return series.astype(bool).values


def load_survival_data(
    filepath: str,
    time_col: Optional[str] = None,
    event_col: Optional[str] = None,
    drop_cols: Optional[list] = None,
    event_values: Optional[list] = None,
    *,
    start_col: Optional[str] = None,
    end_col: Optional[str] = None,
    time_unit: str = 'days',
    drop_nonpositive: bool = False,
    feature_names: Optional[list] = None,
) -> Union[Tuple[pd.DataFrame, SurvivalArray], pd.DataFrame]:
    """
    Load survival data from a CSV file.

    Operates in one of three modes depending on the parameters provided:

    **Standard mode** (``time_col`` + ``event_col``): Extract survival
    outcomes from pre-computed time and event columns.

    **Date-based mode** (``start_col`` + ``end_col`` + ``event_col``):
    Compute survival times from paired date columns.

    **Prediction mode** (``feature_names``): Load feature columns only,
    without survival outcomes.  Returns a :class:`~pandas.DataFrame`
    instead of a ``(X, y)`` tuple.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    time_col : str, optional
        Column name for survival times (standard mode).
    event_col : str, optional
        Column name for event indicator.  Accepts 0/1, True/False, or
        string labels (e.g., ``'deceased'``/``'alive'``).  For non-standard
        numeric encodings (e.g., 1/2), pass *event_values*.
    drop_cols : list, optional
        Additional columns to drop from features.
    event_values : list, optional
        Values in *event_col* that should be treated as events.
        Use this for non-``0/1`` encodings such as ``1/2``.
    start_col : str, optional
        Column name for start dates (date-based mode).
    end_col : str, optional
        Column name for end dates (date-based mode).
    time_unit : str, default ``'days'``
        Time unit for date-based mode: ``'days'``, ``'months'``, or
        ``'years'``.
    drop_nonpositive : bool, default False
        If True, drop rows with non-positive survival times and emit a
        :class:`MethodologicalWarning`.  If False, raise :class:`ValueError`.
    feature_names : list of str, optional
        When provided, load only these feature columns and return a
        :class:`~pandas.DataFrame` (prediction mode).

    Returns
    -------
    (X, y) : tuple of (pd.DataFrame, SurvivalArray)
        Feature matrix and survival outcomes (standard and date-based modes).
    X : pd.DataFrame
        Feature matrix only (prediction mode, when *feature_names* is set).

    Raises
    ------
    ValueError
        If `start_col` and `end_col` are not supplied together, `event_col`
        is missing in date-based mode, the required columns for the selected
        mode are not provided, or a specified column is absent from the data.

    Notes
    -----
    The loaded data is assumed to have one row per subject, with all covariates
    measured at baseline (time zero) and censoring independent of the event
    given the covariates. These assumptions are not checked. See the
    ``SurvivalLearner`` class docstring for what they mean and why they matter.

    Examples
    --------
    Standard mode:

    >>> X, y = load_survival_data('data.csv', 'OS_MONTHS', 'OS_STATUS')

    Date-based mode:

    >>> X, y = load_survival_data('data.csv', event_col='status',
    ...     start_col='diagnosis_date', end_col='last_contact')

    Prediction mode:

    >>> X = load_survival_data('data.csv', feature_names=['age', 'stage'])
    """
    df = pd.read_csv(filepath)

    # --- Prediction mode: return features only ---
    if feature_names is not None:
        return _load_prediction_features(df, feature_names)

    # --- Validate date-column pairing ---
    if (start_col is None) != (end_col is None):
        raise ValueError(
            "Both start_col and end_col must be provided together."
        )

    # --- Date-based mode ---
    if start_col is not None and end_col is not None:
        if event_col is None:
            raise ValueError(
                "event_col is required when using date-based mode "
                "(start_col/end_col)."
            )
        X, y = compute_survival_time(
            df, start_col, end_col, event_col,
            time_unit=time_unit, event_values=event_values,
        )
        if drop_cols:
            X = X.drop(columns=drop_cols, errors='ignore')
        return X, y

    # --- Standard mode ---
    if time_col is None or event_col is None:
        raise ValueError(
            "Provide time_col and event_col for standard mode, "
            "start_col and end_col for date-based mode, "
            "or feature_names for prediction mode."
        )

    if time_col not in df.columns:
        available = df.columns.tolist()
        suffix = f" (and {len(available) - 10} more)" if len(available) > 10 else ""
        raise ValueError(
            f"Time column '{time_col}' not found in data. "
            f"Available columns: {available[:10]}{suffix}"
        )
    if event_col not in df.columns:
        available = df.columns.tolist()
        suffix = f" (and {len(available) - 10} more)" if len(available) > 10 else ""
        raise ValueError(
            f"Event column '{event_col}' not found in data. "
            f"Available columns: {available[:10]}{suffix}"
        )

    # Extract outcome columns
    times = df[time_col].values
    events = _parse_event_values(df[event_col], event_values=event_values)

    # Pre-filter non-positive times to keep X aligned with y
    if drop_nonpositive:
        # NaN must be caught before the mask (NaN > 0 is False, so NaN
        # would be silently dropped and mislabeled as "non-positive").
        if np.any(np.isnan(times)):
            raise ValueError("Survival times contain NaN values.")
        mask = times > 0
        n_dropped = int((~mask).sum())
        if n_dropped:
            warnings.warn(
                f"Dropped {n_dropped} observation(s) with non-positive "
                f"survival times.",
                MethodologicalWarning,
                stacklevel=2,
            )
            times = times[mask]
            events = events[mask]
            df = df[mask]
            if len(times) == 0:
                raise ValueError(
                    f"All {n_dropped} observation(s) had non-positive "
                    f"survival times. No data remains after filtering."
                )

    y = to_structured_array(times, events)

    # Create feature matrix
    drop_columns = [time_col, event_col]
    if drop_cols:
        drop_columns.extend(drop_cols)

    X = df.drop(columns=drop_columns, errors='ignore')

    return X, y


def _load_prediction_features(
    df: pd.DataFrame,
    feature_names: list,
) -> pd.DataFrame:
    """Subset and reorder DataFrame columns to *feature_names*.

    Raises :class:`ValueError` on empty list or missing columns.
    Emits :class:`SuggestionWarning` when extra columns are present.
    """
    if len(feature_names) == 0:
        raise ValueError(
            "feature_names must contain at least one feature column."
        )

    missing = [n for n in feature_names if n not in df.columns]
    if missing:
        raise ValueError(
            f"Data is missing required features: {missing}. "
            f"Ensure your data includes all features the model "
            f"was trained on."
        )

    extra = sorted(set(df.columns) - set(feature_names))
    if extra:
        warnings.warn(
            f"Ignoring {len(extra)} extra column(s) not used by the model: "
            f"{extra}",
            SuggestionWarning,
            stacklevel=2,
        )

    return df.loc[:, feature_names].copy()


_TIME_UNIT_DIVISORS = {
    'days': None,       # Use .dt.days (integer days)
    'months': 30.4375 * 86_400,
    'years': 365.25 * 86_400,
}


def _parse_and_validate_dates(df, start_col, end_col):
    """Parse date columns and raise if any values are NaT."""
    try:
        start = pd.to_datetime(df[start_col])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Could not parse start dates in column '{start_col}': {exc}"
        ) from exc
    try:
        end = pd.to_datetime(df[end_col])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Could not parse end dates in column '{end_col}': {exc}"
        ) from exc

    nat_counts = {}
    if start.isna().any():
        nat_counts[start_col] = int(start.isna().sum())
    if end.isna().any():
        nat_counts[end_col] = int(end.isna().sum())
    if nat_counts:
        raise ValueError(
            "Missing dates (NaT) found: "
            + ", ".join(f"'{k}': {v}" for k, v in nat_counts.items())
        )
    return start, end


def _compute_time_delta(start, end, time_unit):
    """Compute survival times from start/end datetime Series."""
    delta = end - start
    if time_unit == 'days':
        return delta.dt.days.astype(float)
    divisor = _TIME_UNIT_DIVISORS[time_unit]
    return delta.dt.total_seconds() / divisor


def compute_survival_time(
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    event_col: str,
    time_unit: str = 'days',
    event_values: Optional[list] = None,
    drop_date_cols: bool = True,
) -> Tuple[pd.DataFrame, SurvivalArray]:
    """Compute survival time from date columns.

    Converts start/end date columns into a numeric survival time and
    builds the ``(X, y)`` pair ready for :class:`SurvivalLearner`.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing date and event columns.
    start_col : str
        Column name for the start date (e.g. diagnosis date).
    end_col : str
        Column name for the end date (e.g. last follow-up date).
    event_col : str
        Column name for the event indicator.
    time_unit : str, default ``'days'``
        Unit for computed survival time. One of ``'days'``,
        ``'months'``, or ``'years'``.
    event_values : list, optional
        Values in *event_col* that indicate an event occurred
        (e.g. ``['deceased']``). When ``None``, uses automatic
        detection (see :func:`_parse_event_values`).
    drop_date_cols : bool, default ``True``
        If True, drop *start_col* and *end_col* from the returned
        feature matrix.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (event column always dropped; date columns
        dropped when *drop_date_cols* is True).
    y : SurvivalArray
        Survival outcomes with ``'event'`` and ``'time'`` fields.

    Raises
    ------
    ValueError
        If columns are missing, dates cannot be parsed, any dates
        are missing (NaT), survival times are negative, or
        *time_unit* is invalid.

    Examples
    --------
    >>> X, y = compute_survival_time(
    ...     df, start_col='diagnosis_date', end_col='last_followup',
    ...     event_col='vital_status', time_unit='months',
    ...     event_values=['deceased'],
    ... )
    """
    # --- Validate columns exist ---
    for role, col in [('Start', start_col), ('End', end_col),
                      ('Event', event_col)]:
        if col not in df.columns:
            raise ValueError(f"{role} column '{col}' not found in data.")

    if time_unit not in _TIME_UNIT_DIVISORS:
        raise ValueError(
            f"Unknown time_unit '{time_unit}'. "
            f"Must be one of {list(_TIME_UNIT_DIVISORS.keys())}."
        )

    # --- Parse dates and compute time ---
    start, end = _parse_and_validate_dates(df, start_col, end_col)
    times = _compute_time_delta(start, end, time_unit)

    # --- Validate times ---
    negative_mask = times < 0
    if negative_mask.any():
        n_neg = int(negative_mask.sum())
        bad_idx = list(negative_mask[negative_mask].index[:5])
        raise ValueError(
            f"{n_neg} negative survival time(s) (end < start). "
            f"First row indices: {bad_idx}"
        )

    zero_mask = times == 0
    if zero_mask.any():
        n_zero = int(zero_mask.sum())
        warnings.warn(
            f"{n_zero} observation(s) have zero follow-up time "
            f"(start == end). Times are set to a tiny epsilon (1e-10) "
            f"to satisfy the positive-time requirement.",
            MethodologicalWarning,
            stacklevel=2,
        )

    # --- Parse events ---
    events = _parse_event_values(df[event_col], event_values=event_values)

    # --- Build y (to_structured_array validates positive times) ---
    # Zero times are valid clinically but to_structured_array requires > 0.
    # Add a small epsilon to zero times to pass validation.
    times_arr = times.values
    zero_time = times_arr == 0
    if zero_time.any():
        times_arr = times_arr.copy()
        times_arr[zero_time] = 1e-10

    y = to_structured_array(times_arr, events)

    # --- Build X ---
    cols_to_drop = [event_col]
    if drop_date_cols:
        cols_to_drop.extend([start_col, end_col])
    X = df.drop(columns=cols_to_drop)

    return X, y


def auto_eval_times(
    y: SurvivalArray,
    n_times: int = 10,
    percentiles: Optional[list] = None,
    y_train: Optional[SurvivalArray] = None,
    *,
    _quiet: bool = False,
) -> np.ndarray:
    """
    Automatically compute evaluation timepoints from survival data.

    When ``y_train`` is provided, percentiles are derived from training
    event times instead of ``y``.  This ensures that auto-computed
    eval_times fall within the training observed time range, avoiding
    the need for downstream clipping via :func:`clip_eval_times`.

    Parameters
    ----------
    y : SurvivalArray
        Survival outcomes.  Used as the time source when
        *y_train* is ``None``.  When *y_train* is provided, *y* is
        still inspected to check coverage of its time range.
    n_times : int, default=10
        Number of evaluation timepoints.
    percentiles : list, optional
        Specific percentiles to use (overrides n_times).
        Default: evenly spaced from 10th to 90th percentile.
    y_train : SurvivalArray, optional
        Training survival outcomes.  When supplied, percentiles
        are computed from its event times rather than from *y*.

    Returns
    -------
    np.ndarray
        Evaluation timepoints.

    Raises
    ------
    ValueError
        If the time source has no observations, or has fewer than 2 unique
        event times (insufficient for IPCW censoring-distribution estimation).

    See Also
    --------
    clip_eval_times : Clip eval_times to training range.
    truncate_y_to_train_range : Censor y_test observations beyond training max.
    """
    y_source = y_train if y_train is not None else y
    times = y_source['time']

    # Validate input array
    if len(times) == 0:
        source_name = "y_train" if y_train is not None else "y"
        raise ValueError(
            f"Cannot compute evaluation times: {source_name} has no observations"
        )

    event_times = times[y_source['event']]

    if len(event_times) == 0:
        # All censored - use overall time distribution
        event_times = times

    unique_times = np.unique(event_times)
    if len(unique_times) < 2:
        source_name = "y_train" if y_train is not None else "y"
        raise ValueError(
            f"Only {len(unique_times)} unique event time(s) in {source_name}. "
            f"IPCW metrics require >= 2 distinct event times to estimate the "
            f"censoring distribution — review your training data."
        )

    if percentiles is None:
        percentiles = np.linspace(10, 90, n_times)

    eval_times = np.percentile(event_times, percentiles)
    eval_times = np.unique(eval_times)

    # ``_quiet`` lets callers that emit their own once-per-cohort warning
    # (see ``_emit_ipcw_truncation_warning``) suppress this deep-pipeline
    # duplicate. Direct user calls keep the divergence warning by default.
    if y_train is not None and not _quiet:
        y_test_max = y['time'].max()
        y_train_max = y_train['time'].max()
        if y_test_max > 1.5 * y_train_max:
            warnings.warn(
                f"eval_times derived from y_train cover times up to "
                f"{eval_times.max():.1f}, but y_test has observations up to "
                f"{y_test_max:.1f}. IPCW metrics will only reflect model "
                f"performance within the training time range.",
                MethodologicalWarning,
                stacklevel=2,
            )

    return eval_times


def union_extra_eval_times(
    eval_times: np.ndarray,
    extra_eval_times: Optional[np.ndarray],
    y_train: Optional[SurvivalArray],
) -> np.ndarray:
    """Union additive report horizons into an already-resolved eval_times grid.

    Implements the additive part of ``add_eval_times``: the effective grid for
    a computation is ``union(base_grid, extra)`` restricted to the
    censoring-stable window. Out-of-window extras are dropped **silently** here
    — they were already warned about (``MethodologicalWarning``) when the user
    registered them via ``add_eval_times``, and they surface as honest NA at
    render time. Clipping silently (rather than calling :func:`clip_eval_times`)
    avoids a redundant second warning on every evaluate()/bootstrap().

    Parameters
    ----------
    eval_times : np.ndarray
        The already-resolved (auto or explicit, clipped) base grid.
    extra_eval_times : np.ndarray or None
        Registered report horizons. ``None``/empty is a no-op.
    y_train : SurvivalArray or None
        Training outcomes, used to bound the censoring-stable window. When
        ``None`` (or degenerate), extras are unioned without a window filter.

    Returns
    -------
    np.ndarray
        Sorted, de-duplicated union of the base grid and in-window extras.
    """
    if eval_times is None or extra_eval_times is None:
        return eval_times
    extra = np.asarray(extra_eval_times, dtype=float)
    if extra.size == 0:
        return eval_times
    if y_train is not None:
        train_times = np.asarray(y_train['time'], dtype=float)
        t_min, t_max = train_times.min(), train_times.max()
        if not np.isclose(t_min, t_max):
            extra = extra[(extra >= t_min) & (extra < t_max)]
    if extra.size == 0:
        return eval_times
    return np.unique(np.concatenate([np.asarray(eval_times, dtype=float), extra]))


def auto_clip_union_eval_times(
    y_test: SurvivalArray,
    y_train: Optional[SurvivalArray],
    extra_eval_times: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Materialize the auto display grid: ``auto_eval_times`` -> clip -> union.

    The single shared implementation of the three-step auto grid used by
    ``evaluate()`` (its verbose-box pre-resolve) and the viz calibration
    surfaces, so both land on the same horizons. Runs :func:`auto_eval_times`
    on ``y_test`` (training-anchored when ``y_train`` is given), clips to the
    training censoring window, then unions any registered ``add_eval_times``
    horizons (``extra_eval_times``).

    Returns the concrete 1-D grid, or ``None`` when ``auto_eval_times`` cannot
    derive one from the data (the caller decides how to surface that).
    """
    try:
        grid = auto_eval_times(y_test, y_train=y_train)
        if y_train is not None:
            grid = clip_eval_times(grid, y_train)
        grid = union_extra_eval_times(grid, extra_eval_times, y_train)
        return grid
    except (ValueError, TypeError) as exc:
        logger.warning("auto_eval_times failed while materializing grid: %s.", exc)
        return None


def resolve_display_eval_times(
    state,
    y_test: SurvivalArray,
    y_train: Optional[SurvivalArray],
) -> np.ndarray:
    """Resolve the calibration/display eval_times grid (read-only).

    The viz-facing single entry point for the *default* calibration grid. It
    applies the same resolution the compute surfaces use — an explicit grid
    inherited from a prior ``evaluate()``/``run()`` (``state.user_eval_times``)
    takes priority; otherwise an auto grid from the data — and in **both** cases
    unions any ``add_eval_times`` report horizons (``state.user_extra_eval_times``).
    This mirrors what ``evaluate()`` computes (which always unions the extras,
    explicit grid or not), so calibration lands on the same horizons as
    AUC(t)/Brier(t)/ROC by construction.

    ``state`` is duck-typed (only ``user_eval_times`` / ``user_extra_eval_times``
    are read) and never mutated, so plotting leaves learner state untouched and
    this stays in the low-level ``utils`` layer (no ``learner`` import).

    Returns
    -------
    np.ndarray
        The resolved 1-D grid.

    Raises
    ------
    ValueError
        When neither an inherited grid nor an auto grid can be derived from the
        available survival data.
    """
    inherited = getattr(state, "user_eval_times", None) if state is not None else None
    extras = getattr(state, "user_extra_eval_times", None) if state is not None else None
    if inherited is not None:
        inherited = np.asarray(inherited, dtype=float)
        if inherited.size:
            # Union the registered report horizons just as evaluate() does for an
            # explicit grid (evaluate clips too; calibration skips out-of-window
            # horizons downstream, so we union silently and avoid clip's warning
            # + percentile-fallback on every plot call).
            return union_extra_eval_times(inherited, extras, y_train)
    grid = auto_clip_union_eval_times(y_test, y_train, extras)
    if grid is None or len(grid) == 0:
        raise ValueError(
            "plot_calibration could not derive default eval timepoint(s) from "
            "the available survival data. Pass eval_times=[<float>, ...] to "
            "override."
        )
    return grid


def clip_eval_times(
    eval_times: np.ndarray,
    y_train: Optional[SurvivalArray],
) -> np.ndarray:
    """Clip eval_times to the training data's observed time range.

    IPCW-based metrics (Brier, IBS, AUC-t) require eval_times within the
    training censoring distribution. Times outside [min, max) of y_train
    are removed with a warning. Raises ``ValueError`` when all training
    times are identical (degenerate case), since no valid clipping range
    exists.

    .. note::

        For valid IPCW metrics, **both** constraints must hold:

        1. ``eval_times`` lie within the training time range (this function).
        2. All ``y_test`` observation times are covered by the training
           censoring distribution (:func:`truncate_y_to_train_range`).

        When ``auto_eval_times`` is called with ``y_train``, the returned
        times typically satisfy constraint 1, making this function a
        no-op.  A warning may still be raised in the rare case where training
        event times cluster at the maximum observed time.

    Parameters
    ----------
    eval_times : np.ndarray
        Candidate evaluation timepoints.
    y_train : SurvivalArray or None
        Training survival outcomes.  When ``None``, *eval_times*
        is returned unchanged.

    Returns
    -------
    np.ndarray
        Evaluation timepoints within the training range.

    Raises
    ------
    ValueError
        When all training times are identical or near-identical (within
        ``np.isclose`` tolerance), making IPCW metric computation
        impossible.

    See Also
    --------
    auto_eval_times : Compute eval_times from training data directly.
    truncate_y_to_train_range : Censor y_test observations beyond training max.
    """
    if y_train is None:
        return eval_times
    eval_times = np.asarray(eval_times)
    train_times = y_train['time']
    t_min, t_max = train_times.min(), train_times.max()
    # Degenerate case: all training times identical or near-identical.
    if np.isclose(t_min, t_max):
        raise ValueError(
            f"All training times are identical ({t_max:.1f}). "
            f"IPCW-based metrics require distinct training times to estimate "
            f"the censoring distribution — review your training data."
        )
    # Use strict less-than for upper bound: sksurv requires times < max observed
    mask = (eval_times >= t_min) & (eval_times < t_max)
    if mask.all():
        return np.unique(eval_times)
    n_dropped = (~mask).sum()
    warnings.warn(
        f"Dropped {n_dropped} eval time(s) outside training range "
        f"[{t_min:.1f}, {t_max:.1f}). This typically happens when the "
        f"validation cohort has longer follow-up than training data.",
        SuggestionWarning,
        stacklevel=2,
    )
    clipped = np.unique(eval_times[mask])
    if len(clipped) < 2:
        # IBS requires >= 2 time points; fall back to training event times
        event_times = train_times[y_train['event']]
        if len(event_times) < 2:
            event_times = train_times
        clipped = np.percentile(event_times, [25, 50, 75])
        clipped = np.unique(clipped[clipped < t_max])
        if len(clipped) < 2:
            # Last resort: two equally-spaced interior points
            clipped = np.linspace(t_min, t_max, 4)[1:3]
            warnings.warn(
                f"Could not find >= 2 valid eval_times from training data. "
                f"Using 2 equally-spaced interior points: {clipped}. "
                f"Results at these times may not be representative.",
                MethodologicalWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"Fewer than 2 eval_times survived clipping. "
                f"Replaced with training percentiles: {clipped}.",
                MethodologicalWarning,
                stacklevel=2,
            )
    return clipped


def truncate_y_to_train_range(
    y_test: SurvivalArray,
    y_train: Optional[SurvivalArray],
    *,
    warn: bool = False,
) -> SurvivalArray:
    """Administratively censor y_test observations beyond training time range.

    IPCW-based metrics require the censoring distribution (from y_train)
    to cover all observation times in y_test. When y_test has longer
    follow-up, observations beyond y_train's max time are censored at
    that boundary. This preserves all samples for the metric computation.

    .. note::

        This function handles one of two IPCW constraints.  The other —
        ensuring ``eval_times`` fall within the training range — is handled
        by :func:`clip_eval_times` (or avoided entirely by passing
        ``y_train`` to :func:`auto_eval_times`).

    Parameters
    ----------
    y_test : SurvivalArray
        Test survival outcomes.
    y_train : SurvivalArray or None
        Training survival outcomes.  When ``None``, *y_test*
        is returned unchanged.
    warn : bool, default False
        Emit a warning when observations are censored.  Off by
        default because callers typically surface truncation via the
        boxed parameter summary (see :func:`check_truncation_needed`).

    Returns
    -------
    SurvivalArray
        Test survival array with beyond-max observations censored.

    See Also
    --------
    clip_eval_times : Remove eval_times outside training range.
    auto_eval_times : Derive eval_times from training data directly.
    """
    if y_train is None:
        return y_test
    t_max = y_train['time'].max()
    beyond = y_test['time'] > t_max
    if not beyond.any():
        return y_test
    y_truncated = y_test.copy()
    y_truncated['time'] = np.where(beyond, t_max, y_test['time'])
    y_truncated['event'] = np.where(beyond, False, y_test['event'])
    n_truncated = int(beyond.sum())
    if warn:
        warnings.warn(
            f"Censored {n_truncated} y_test observation(s) beyond training max "
            f"({t_max:.1f}). Observations at those times are administratively "
            f"censored at t_max to satisfy IPCW requirements.",
            MethodologicalWarning,
            stacklevel=2,
        )
    else:
        logger.debug(
            "Truncated %d y_test obs beyond training max (%.1f) for IPCW",
            n_truncated, t_max,
        )
    return SurvivalArray(y_truncated)


def clip_eval_times_to_data(
    eval_times: np.ndarray,
    y_data: SurvivalArray,
    y_train: Optional[SurvivalArray] = None,
) -> tuple:
    """Clip eval_times to the observed follow-up range of y_data.

    sksurv requires eval_times to fall strictly within the test data's
    follow-up range ``(t_min, t_max)``.  When *y_data* is a bootstrap
    resample or subpopulation with shorter follow-up than the global eval grid,
    some eval_times may land outside this range.  This function filters
    them, returning NaN-safe outputs for the caller.

    If *y_train* is provided, *y_data* is first truncated to the training
    range via :func:`truncate_y_to_train_range`.

    Parameters
    ----------
    eval_times : ndarray
        Candidate evaluation time points.
    y_data : SurvivalArray
        Test (or resampled) survival outcomes whose follow-up range
        constrains ``eval_times``.
    y_train : SurvivalArray or None, optional
        Training outcomes.  When provided, *y_data* is first truncated
        to the training time range.

    Returns
    -------
    valid_times : ndarray
        Eval times that fall within ``(t_min, t_max)`` of the
        (possibly truncated) data.
    valid_mask : ndarray of bool
        Boolean mask into the original *eval_times* array.
    y_truncated : SurvivalArray
        *y_data* after truncation (unchanged if *y_train* is ``None``).

    See Also
    --------
    clip_eval_times : Clip eval_times to the *training* range.
    truncate_y_to_train_range : Censor y_test observations beyond training max.
    """
    eval_times = np.asarray(eval_times, dtype=float)
    if y_train is not None:
        y_data = truncate_y_to_train_range(y_data, y_train, warn=False)
    times = y_data['time']
    t_min, t_max = float(times.min()), float(times.max())
    valid_mask = (eval_times > t_min) & (eval_times < t_max)
    return eval_times[valid_mask], valid_mask, y_data


def check_truncation_needed(
    y_test: SurvivalArray,
    y_train: Optional[SurvivalArray],
) -> Optional[str]:
    """Return a note string if IPCW truncation would apply, else None.

    Pure query function with no side effects — use in boxed parameter
    summaries to inform the user that truncation would apply to IPCW
    metrics.

    Parameters
    ----------
    y_test : SurvivalArray
        Test survival outcomes.
    y_train : SurvivalArray or None
        Training survival outcomes.  When ``None``, returns
        ``None`` immediately.

    Returns
    -------
    str or None
        Human-readable note if truncation is needed, ``None`` otherwise.

    See Also
    --------
    truncate_y_to_train_range : Perform the actual censoring.
    """
    if y_train is None:
        return None
    t_max = y_train['time'].max()
    n_beyond = int((y_test['time'] > t_max).sum())
    if n_beyond == 0:
        return None
    return (
        f"{n_beyond} y_test obs beyond t_max={t_max:.1f}; "
        "capped for Uno C / AUC(t) / IBS"
    )


def validate_survival_data(
    X: pd.DataFrame,
    y: SurvivalArray,
    *,
    warn_missing: bool = True,
    min_events: int = 5,
    min_samples: int = 10,
) -> None:
    """
    Validate survival data inputs.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : SurvivalArray
        Survival outcomes with 'event' and 'time' fields.
    warn_missing : bool, default=True
        If True, emit a warning when X contains missing values.
    min_events : int, default=5
        Minimum number of uncensored events required. Set to 0 to skip.
    min_samples : int, default=10
        Minimum number of samples required. Set to 0 to skip.

    Raises
    ------
    ValueError
        If X is not a DataFrame, y is not a valid SurvivalArray,
        X and y have different lengths, data is empty,
        sample count is below min_samples, X contains non-numeric columns,
        event count is below min_events, the survival times contain NaN,
        infinite, negative, or zero values, or the event column has an
        invalid encoding (NaN-float or ambiguous binary numeric codes).
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")

    if not isinstance(y, np.ndarray):
        raise ValueError(
            "y must be a SurvivalArray (structured ndarray with 'event' and 'time' fields)."
        )

    if y.dtype.names != ('event', 'time'):
        raise ValueError(
            "y must be a SurvivalArray with fields ('event', 'time'). "
            f"Got {y.dtype.names}."
        )

    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of samples. "
            f"Got {len(X)} and {len(y)}."
        )

    if len(X) == 0:
        raise ValueError("X and y must not be empty.")

    # Validate the event encoding and time values. A hand-built y never goes
    # through to_structured_array, so these checks must run here too (same
    # messages, single-sourced) — otherwise a NaN/inf/zero/negative time or an
    # invalid event encoding passes and crashes later inside the model fit.
    _validate_event_values(y['event'])
    _validate_survival_times(np.asarray(y['time'], dtype=float))

    # Check minimum sample size
    if min_samples > 0 and len(X) < min_samples:
        raise ValueError(
            f"Dataset has only {len(X)} samples — need at least {min_samples} "
            f"for meaningful survival analysis."
        )

    # Check for non-numeric columns
    non_numeric = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    if non_numeric:
        shown = non_numeric[:5]
        extra = f" (and {len(non_numeric) - 5} more)" if len(non_numeric) > 5 else ""
        raise ValueError(
            f"X contains non-numeric columns: {shown}{extra}. "
            "Encode categorical columns before passing to mlsurv "
            "(e.g., pd.get_dummies() or sklearn LabelEncoder)."
        )

    # Check for missing values in X
    n_missing = X.isna().sum().sum()
    if warn_missing and n_missing > 0:
        n_cols_with_missing = (X.isna().sum() > 0).sum()
        warnings.warn(
            f"X contains {n_missing} missing values across {n_cols_with_missing} columns. "
            "Consider using imputation in setup().",
            SuggestionWarning,
        )

    # Check for sufficient events
    n_events = int(y['event'].sum())
    if min_events > 0 and n_events < min_events:
        raise ValueError(
            f"Too few events ({n_events}) for survival analysis — need at least "
            f"{min_events} uncensored observations. Check that your event column "
            f"is coded correctly (1/True = event, 0/False = censored). "
            f"Pass min_events=0 to skip this check if your data is intentionally censored-heavy."
        )


def drop_nan_rows(
    df: pd.DataFrame,
    y: np.ndarray,
    caller: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Drop rows where any feature column in *df* contains NaN.

    Performs complete-case analysis: both *df* and *y* are filtered in
    lockstep so they remain aligned.  Always emits a warning when rows
    are dropped so users are aware of data loss at any log level.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix (will be copied if rows are dropped).
    y : np.ndarray
        Structured survival array aligned with *df*.
    caller : str
        Name of the calling function (used in log messages).

    Returns
    -------
    tuple of (pd.DataFrame, np.ndarray)
        The filtered (df, y) pair.  Unchanged if no NaN rows exist.
    """
    nan_mask = df.isna().any(axis=1)
    n_dropped = int(nan_mask.sum())
    if n_dropped == 0:
        return df, y

    n_total = len(df)
    pct = n_dropped / n_total * 100
    logger.warning(
        "%s: dropping %d/%d rows (%.0f%%) with NaN",
        caller, n_dropped, n_total, pct,
    )
    df = df[~nan_mask].reset_index(drop=True)
    y = y[~nan_mask.values]
    return df, y


def check_is_fitted(model, attributes: list) -> bool:
    """
    Check if a model is fitted by verifying required attributes exist.

    Parameters
    ----------
    model : object
        Model to check.
    attributes : list
        Attribute names that should exist after fitting.

    Returns
    -------
    bool
        True if model appears to be fitted.
    """
    return all(hasattr(model, attr) for attr in attributes)


def _set_njobs_recursive(estimator):
    """Set n_jobs=1 on estimator and its underlying model_ if present."""
    estimator.n_jobs = 1
    if hasattr(estimator, 'model_') and hasattr(estimator.model_, 'n_jobs'):
        estimator.model_.n_jobs = 1


def adjust_model_njobs(model, outer_n_jobs):
    """
    Set model n_jobs=1 when outer parallel loop is active.

    Prevents CPU oversubscription when using parallel CV/bootstrap
    with models that have their own parallelism (e.g., RSF).

    Parameters
    ----------
    model : estimator or Pipeline
        sklearn-compatible model or Pipeline that may contain estimators
        with n_jobs attribute.
    outer_n_jobs : int
        Number of jobs in the outer parallel loop.

    Returns
    -------
    estimator or Pipeline
        Model/Pipeline with n_jobs=1 on relevant estimators if outer
        parallelism is active. For fitted models, modifies n_jobs in-place
        (safe for prediction-only use cases). For unfitted models in
        non-Pipeline cases, returns a clone.
    """
    if outer_n_jobs == 1:
        return model

    # Handle sklearn Pipeline: adjust the 'model' step
    inner_model = unwrap_pipeline(model)
    if inner_model is not model:
        if hasattr(inner_model, 'n_jobs'):
            _set_njobs_recursive(inner_model)
        return model

    # Handle direct model (not in Pipeline)
    if not hasattr(model, 'n_jobs'):
        return model

    is_fitted = getattr(model, 'is_fitted_', False) or hasattr(model, 'model_')
    if is_fitted:
        # For fitted models (e.g., in bootstrap), modify n_jobs in-place
        _set_njobs_recursive(model)
    else:
        # For unfitted models (e.g., in CV training), clone to avoid modifying original
        model = clone(model)
        model.n_jobs = 1
    return model


def unwrap_pipeline(model):
    """Extract the inner model from a Pipeline's ``'model'`` step.

    If *model* is an sklearn ``Pipeline`` with a step named ``'model'``,
    return that step's estimator.  Otherwise return *model* unchanged.

    Parameters
    ----------
    model : estimator or Pipeline
        sklearn-compatible model or Pipeline with a ``'model'`` step.

    Returns
    -------
    estimator
        The inner model from ``named_steps['model']``, or the original model.
    """
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        return model.named_steps['model']
    return model


def supports_survival_function(model) -> bool:
    """Return True if *model* can compute a survival curve.

    An estimator can opt out by setting ``supports_survival_function = False``
    (as ``ReferenceAdapter`` does for raw biomarker scores).  Otherwise the
    check falls back to ``hasattr(..., 'predict_survival_function')`` on the
    unwrapped inner model.  Used by the benchmark dispatch and the
    point-estimate concordance path to skip IBS / C^td / calibration metrics
    for entities that structurally cannot compute them.
    """
    inner = unwrap_pipeline(model)
    if getattr(inner, 'supports_survival_function', True) is False:
        return False
    return hasattr(inner, 'predict_survival_function')


def propagate_feature_names(inner_model, pipeline):
    """Copy ``feature_names_in_`` from a pipeline step to the inner model.

    Iterates *pipeline*'s steps to find the first one that carries
    ``feature_names_in_`` and copies it onto *inner_model*.  No-op if
    *inner_model* already has the attribute or *pipeline* has no steps.

    Parameters
    ----------
    inner_model : estimator
        The unwrapped inner model (e.g. from :func:`unwrap_pipeline`).
    pipeline : Pipeline or estimator
        The original Pipeline.  If not a Pipeline, this is a no-op.

    Returns
    -------
    None
        Modifies *inner_model* in place; no value is returned.
    """
    if hasattr(inner_model, 'feature_names_in_'):
        return
    if not hasattr(pipeline, 'steps'):
        return
    for _, step in pipeline.steps:
        if hasattr(step, 'feature_names_in_'):
            inner_model.feature_names_in_ = step.feature_names_in_
            break


def _unwrap_inner_model(model):
    """Return the inner estimator that owns ``n_jobs``, or ``None``.

    Handles Pipeline (unwraps the 'model' step) and direct estimators.

    Parameters
    ----------
    model : estimator or Pipeline
        Top-level model or Pipeline containing a 'model' step.

    Returns
    -------
    estimator or None
        The estimator with an ``n_jobs`` attribute, or ``None`` if none found.
    """
    inner = unwrap_pipeline(model)
    if hasattr(inner, 'n_jobs'):
        return inner
    return None


def save_model_njobs(model):
    """Save n_jobs values from a model for later restoration.

    Parameters
    ----------
    model : estimator or Pipeline
        Model whose n_jobs values should be saved.

    Returns
    -------
    dict
        Saved values (keys: 'n_jobs', optionally 'model_n_jobs').
        ``model_n_jobs`` captures the fitted sub-estimator ``model_``
        (an mlsurv convention where each ``BaseSurvivalModel`` wrapper
        stores the underlying library estimator as ``self.model_``,
        e.g. ``RSFModel.model_`` is a ``RandomSurvivalForest``).
        Empty dict if the model has no n_jobs attribute.
    """
    inner = _unwrap_inner_model(model)
    if inner is None:
        return {}

    saved = {'n_jobs': inner.n_jobs}
    if hasattr(inner, 'model_') and hasattr(inner.model_, 'n_jobs'):
        saved['model_n_jobs'] = inner.model_.n_jobs
    return saved


def restore_model_njobs(model, saved):
    """Restore n_jobs values previously saved by ``save_model_njobs``.

    Parameters
    ----------
    model : estimator or Pipeline
        Model whose n_jobs values should be restored.
    saved : dict
        Dict returned by ``save_model_njobs``. No-op if empty.

    Returns
    -------
    None
        Modifies *model* in place; no value is returned.
    """
    if not saved:
        return

    inner = _unwrap_inner_model(model)
    if inner is None:
        warnings.warn(
            "restore_model_njobs: saved n_jobs is non-empty but no inner "
            "estimator with n_jobs was found. Model structure may have "
            "changed between save and restore.",
            InternalWarning,
            stacklevel=2,
        )
        return

    inner.n_jobs = saved['n_jobs']
    if 'model_n_jobs' in saved and hasattr(inner, 'model_') and hasattr(inner.model_, 'n_jobs'):
        inner.model_.n_jobs = saved['model_n_jobs']


# Threshold for parallelization benefit (features below this are faster sequential)
# Empirical basis: univariate Cox fitting overhead < joblib startup cost for < 15
# features (benchmarked Jan 2026, 16-core).
# Run tests/benchmarks/bench_thresholds.py to re-validate on your hardware.
MIN_FEATURES_FOR_PARALLEL = 15


def resolve_parallel_n_jobs(n_jobs, n_items, threshold, force_parallel, verbose):
    """Resolve effective n_jobs, auto-disabling parallelization below threshold.

    Parameters
    ----------
    n_jobs : int
        Requested parallelism (1=sequential, -1=all cores, n>1=n cores).
    n_items : int
        Number of items to process (features, iterations, etc.).
    threshold : int
        Minimum items for parallelization benefit.
    force_parallel : bool
        If True, skip the auto-disable threshold check.
    verbose : bool
        Print note when parallelization is auto-disabled.

    Returns
    -------
    int
        Effective n_jobs value to use.
    """
    if n_jobs == 1 or force_parallel or n_items >= threshold:
        return n_jobs
    if verbose:
        print(f"Note: Parallelization disabled ({n_items} items < "
              f"{threshold} threshold). Use force_parallel=True to override.")
    return 1


def _joblib_probe_identity(value):
    """Return *value* unchanged for backend availability probes."""
    return value


@lru_cache(maxsize=None)
def _joblib_backend_usable(backend: str) -> bool:
    """Return True when *backend* can start worker infrastructure here."""
    if backend != "loky":
        return True

    try:
        joblib.Parallel(n_jobs=2, backend=backend)(
            joblib.delayed(_joblib_probe_identity)(0) for _ in range(1)
        )
    except (PermissionError, OSError, RuntimeError) as exc:
        # loky failures surface as RuntimeError subclasses:
        # BrokenProcessPool, TerminatedWorkerError, ShutdownExecutorError,
        # LokyRecursionError.  PermissionError/OSError cover sandbox and
        # filesystem restrictions.
        logger.debug(
            "joblib backend %r probe failed: %s: %s",
            backend, type(exc).__name__, exc,
        )
        warnings.warn(
            f"joblib '{backend}' backend unavailable in this environment "
            f"({type(exc).__name__}); falling back to 'threading'. "
            f"CPU-bound parallel operations may run slower.",
            SuggestionWarning,
            stacklevel=2,
        )
        return False

    return True


def resolve_joblib_backend(
    backend: str = "loky",
    *,
    fallback_backend: str | None = "threading",
) -> str:
    """Resolve a usable joblib backend for the current environment."""
    if fallback_backend is None or backend == fallback_backend:
        return backend
    if _joblib_backend_usable(backend):
        return backend
    return fallback_backend


@contextlib.contextmanager
def joblib_parallel_backend(
    n_jobs: int,
    *,
    backend: str = "loky",
    fallback_backend: str | None = "threading",
):
    """Context manager that applies a usable joblib backend."""
    resolved_backend = resolve_joblib_backend(
        backend, fallback_backend=fallback_backend,
    )
    with joblib.parallel_backend(resolved_backend, n_jobs=n_jobs):
        yield resolved_backend
