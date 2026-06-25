"""Diagnostic residual computations (the numbers behind the viz plots).

These live in ``evaluation/`` alongside their siblings ``ph_test`` /
``linearity_test`` in :mod:`mlsurv.evaluation.metrics` (the "evaluation
computes, viz plots" layering convention). The matching plot functions stay
in :mod:`mlsurv.viz.diagnostics`, which re-exports the names below for
backward compatibility.

- ``compute_schoenfeld_data``: scaled Schoenfeld residuals as a tidy DataFrame
- ``compute_deviance_data``: deviance residuals as a tidy DataFrame
- ``compute_martingale_data``: null-model martingale residuals (long format)

The two Cox-based functions fit through
:func:`mlsurv.evaluation.metrics._fit_cox_with_fallback` (zero-variance
pre-drop + escalating-penalizer ladder), the same robust fit ``ph_test``
uses, so the residual plots stay consistent with the PH-test p-values.
``compute_martingale_data`` uses a null-model Nelson-Aalen fit (no Cox).
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from mlsurv.logging_utils import get_logger
from mlsurv.utils import drop_nan_rows

logger = get_logger("evaluation.diagnostics")


def _schoenfeld_tidy_from_cph(
    cph,
    df: pd.DataFrame,
    scaled_resids: Optional[pd.DataFrame] = None,
    features: Optional[List[str]] = None,
    n_features: int = 6,
) -> pd.DataFrame:
    """Build the tidy ``[feature, time, residual]`` frame from a fitted Cox model.

    Shared by :func:`compute_schoenfeld_data` and
    :func:`mlsurv.evaluation.metrics.ph_test_with_schoenfeld` so both paths
    produce identical residuals from a single fit.

    Parameters
    ----------
    cph : lifelines.CoxPHFitter
        A fitted model.
    df : DataFrame
        The design the model was fit on (still carrying ``T``/``E``).
    scaled_resids : DataFrame, optional
        Pre-computed scaled Schoenfeld residuals. When ``None``, computed
        here from *cph* and *df*.
    features : list of str, optional
        Specific features. If ``None``, selects the top *n_features* by
        absolute coefficient magnitude.
    n_features : int, default=6
        Number of features (ignored when *features* is provided).

    Returns
    -------
    pd.DataFrame
        Columns ``feature``, ``time``, ``residual``. One row per event-time
        per feature.
    """
    if scaled_resids is None:
        scaled_resids = cph.compute_residuals(df, kind='scaled_schoenfeld')

    if features is None:
        coefs = cph.params_.abs().sort_values(ascending=False)
        features = coefs.head(n_features).index.tolist()

    event_times = scaled_resids.index.values
    features = [f for f in features if f in scaled_resids.columns]

    rows = []
    for feat in features:
        resid_values = scaled_resids[feat].values
        for t, r in zip(event_times, resid_values):
            rows.append({"feature": feat, "time": t, "residual": r})

    return pd.DataFrame(rows, columns=["feature", "time", "residual"])


def compute_schoenfeld_data(
    X_train: pd.DataFrame,
    y_train,
    features: Optional[List[str]] = None,
    n_features: int = 6,
) -> pd.DataFrame:
    """Extract scaled Schoenfeld residuals as a tidy DataFrame.

    Re-fits a lifelines ``CoxPHFitter`` internally through
    :func:`mlsurv.evaluation.metrics._fit_cox_with_fallback` (zero-variance
    pre-drop + escalating-penalizer ladder) so the residuals match the
    ``ph_test`` fit on the same data.

    Parameters
    ----------
    X_train : DataFrame
        Training features.
    y_train : structured array
        Training outcomes with 'time' and 'event' fields.
    features : list of str, optional
        Specific features. If ``None``, selects the top *n_features*
        by absolute coefficient magnitude.
    n_features : int
        Number of features (ignored if *features* is provided).

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``time``, ``residual``.
        One row per event-time per feature.
    """
    from mlsurv.evaluation.metrics import _fit_cox_with_fallback

    df = X_train.copy()
    df, y_train = drop_nan_rows(df, y_train, "compute_schoenfeld_data")

    if len(df) == 0:
        logger.warning("compute_schoenfeld_data: no rows remain after NaN removal")
        return pd.DataFrame(columns=["feature", "time", "residual"])

    df['T'] = y_train['time']
    df['E'] = y_train['event'].astype(int)

    cph, _info = _fit_cox_with_fallback(
        df,
        duration_col='T',
        event_col='E',
        context='compute_schoenfeld_data',
        penalizers=(0.01, 0.1, 1.0),
    )

    return _schoenfeld_tidy_from_cph(
        cph, df, features=features, n_features=n_features,
    )


def compute_deviance_data(
    X_train: pd.DataFrame,
    y_train,
) -> pd.DataFrame:
    """Extract deviance residuals as a tidy DataFrame.

    Re-fits a lifelines ``CoxPHFitter`` internally through
    :func:`mlsurv.evaluation.metrics._fit_cox_with_fallback`.

    Parameters
    ----------
    X_train : DataFrame
        Training features.
    y_train : structured array
        Training outcomes with 'time' and 'event' fields.

    Returns
    -------
    pd.DataFrame
        Columns: ``linear_predictor``, ``deviance``, ``event``.
    """
    from mlsurv.evaluation.metrics import _fit_cox_with_fallback

    df = X_train.copy()
    df, y_train = drop_nan_rows(df, y_train, "compute_deviance_data")

    if len(df) == 0:
        logger.warning("compute_deviance_data: no rows remain after NaN removal")
        return pd.DataFrame(columns=["linear_predictor", "deviance", "event"])

    df['T'] = y_train['time']
    df['E'] = y_train['event'].astype(int)

    cph, _info = _fit_cox_with_fallback(
        df,
        duration_col='T',
        event_col='E',
        context='compute_deviance_data',
        penalizers=(0.01, 0.1, 1.0),
    )

    residuals = cph.compute_residuals(df, kind='deviance')
    deviance = residuals['deviance'].values
    linear_predictor = cph.predict_partial_hazard(df).values.flatten()
    event_status = y_train['event'].astype(bool)

    return pd.DataFrame({
        "linear_predictor": linear_predictor,
        "deviance": deviance,
        "event": event_status,
    })


# ── Martingale residuals for linearity diagnostics ────────────────────────


def _identify_continuous_for_martingale(
    X: pd.DataFrame, min_unique: int = 6,
) -> List[str]:
    """Numeric columns with at least ``min_unique`` distinct values.

    Harrell's ``rms`` uses 6 as the default cutoff.
    """
    out = []
    for col in X.columns:
        s = X[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if s.nunique() >= min_unique:
            out.append(col)
    return out


def compute_martingale_data(
    X_train: pd.DataFrame,
    y_train,
    features: Optional[List[str]] = None,
    min_unique: int = 6,
) -> pd.DataFrame:
    """Null-model martingale residuals against continuous covariates.

    Computes per-subject martingale residuals from the **null** model (no
    covariates) as ``M_i = δ_i - H_0(T_i)`` where ``H_0`` is the
    Nelson-Aalen cumulative hazard estimator (Nelson-Aalen is the natural
    MLE of the null-model baseline hazard; Aalen 1978 *Ann. Stat.*), then
    tiles the residual vector against every continuous covariate (numeric
    columns with ``nunique() >= min_unique``).  This is the classical
    Therneau-Grambsch-Fleming 1990 exploratory construction for assessing
    the functional form of a covariate in the log-hazard.

    Do **not** reconstruct this by fitting a Cox model *with* the covariate
    and plotting residuals against it — the feature's linear effect is then
    already in the model and the smoother trend is forced flat regardless
    of any nonlinearity.

    Parameters
    ----------
    X_train : DataFrame
        Training features.
    y_train : structured array
        Training outcomes with ``time`` and ``event`` fields.
    features : list of str, optional
        Restrict to these feature names.  If ``None`` (default), use every
        continuous column in ``X_train``.
    min_unique : int, default=6
        Minimum distinct values for a numeric column to count as continuous.

    Returns
    -------
    DataFrame
        Columns: ``feature``, ``x_value``, ``residual``.  One row per
        observation per feature.  Returns an empty DataFrame (same columns)
        when no continuous features are present or the Nelson-Aalen fit
        fails.

    Raises
    ------
    MemoryError
        Re-raised if the computation exhausts available memory; resource
        errors are never swallowed by the broad failure handler.
    RecursionError
        Re-raised if the computation exceeds the recursion limit; resource
        errors are never swallowed by the broad failure handler.

    References
    ----------
    Therneau, Grambsch & Fleming (1990). "Martingale-based residuals for
    survival models." *Biometrika* 77(1):147-160.
    """
    import warnings as _warnings
    from mlsurv._warnings import InternalWarning, SuggestionWarning

    df = X_train.copy()
    df, y_train = drop_nan_rows(df, y_train, "compute_martingale_data")

    empty = pd.DataFrame(columns=["feature", "x_value", "residual"])
    if len(df) == 0:
        _warnings.warn(
            "compute_martingale_data: no rows remain after NaN removal; "
            "returning empty result. The martingale diagnostic plot will "
            "be omitted from the report.",
            SuggestionWarning,
            stacklevel=2,
        )
        return empty

    all_continuous = _identify_continuous_for_martingale(df, min_unique=min_unique)
    if features is not None:
        selected = [f for f in features if f in all_continuous]
    else:
        selected = all_continuous
    if not selected:
        _warnings.warn(
            f"compute_martingale_data: no continuous features to plot "
            f"(need numeric columns with >={min_unique} unique values). "
            f"The martingale diagnostic plot will be omitted.",
            SuggestionWarning,
            stacklevel=2,
        )
        return empty

    # Null-model martingale residuals via Nelson-Aalen. Guarded so the
    # caller (report section) falls back gracefully rather than raising.
    try:
        from lifelines import NelsonAalenFitter
        times = np.asarray(y_train['time'], dtype=float)
        events = np.asarray(y_train['event']).astype(int)

        naf = NelsonAalenFitter()
        naf.fit(times, event_observed=events)

        # cumulative_hazard_at_times returns a pandas Series indexed by
        # the input times; reset to array.
        H0 = naf.cumulative_hazard_at_times(times).to_numpy(dtype=float)
        martingale = events.astype(float) - H0
    except (MemoryError, RecursionError):
        raise
    except (ValueError, RuntimeError, TypeError, KeyError,
            np.linalg.LinAlgError) as exc:
        _warnings.warn(
            f"compute_martingale_data: Nelson-Aalen fit failed "
            f"({type(exc).__name__}: {exc}); returning empty result. "
            f"The martingale diagnostic plot will be omitted.",
            InternalWarning,
            stacklevel=2,
        )
        return empty

    rows = []
    for feat in selected:
        x_values = df[feat].to_numpy(dtype=float)
        for x_val, r in zip(x_values, martingale):
            rows.append({"feature": feat, "x_value": float(x_val),
                         "residual": float(r)})

    return pd.DataFrame(rows, columns=["feature", "x_value", "residual"])
