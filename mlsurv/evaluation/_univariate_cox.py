"""Univariate Cox regression for individual reference scores.

Fits a separate univariate CoxPH model per reference column and returns
hazard ratios, 95% CIs, and p-values.  Uses ``lifelines.CoxPHFitter``
(already a dependency) for clean summary statistics.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..utils import SurvivalArray
from ..logging_utils import get_logger

logger = get_logger("evaluation.univariate_cox")


def compute_reference_univariate_cox(
    reference_df: pd.DataFrame,
    y: SurvivalArray,
    penalizer: float = 0.01,
) -> pd.DataFrame:
    """Fit univariate Cox PH model for each reference column.

    Parameters
    ----------
    reference_df : DataFrame
        Columns are reference names, rows aligned with *y*.
    y : SurvivalArray
        Survival outcomes.
    penalizer : float
        L2 regularization for convergence stability.

    Returns
    -------
    DataFrame
        Columns: reference, coefficient, hazard_ratio, hr_lower_95,
        hr_upper_95, p_value.
    """
    from lifelines import CoxPHFitter

    surv_df = pd.DataFrame({
        "time": y["time"],
        "event": y["event"].astype(float),
    })

    rows = []
    for col in reference_df.columns:
        values = reference_df[col].values
        if np.isnan(values).all():
            logger.warning("Reference '%s' is all NaN — skipping.", col)
            continue

        df = surv_df.copy()
        df["reference"] = values
        df["reference"] = df["reference"].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["reference"])

        if len(df) < 5:
            logger.warning(
                "Reference '%s' has fewer than 5 non-NaN values — skipping.",
                col,
            )
            continue

        try:
            cph = CoxPHFitter(penalizer=penalizer)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", module="lifelines")
                cph.fit(df, duration_col="time", event_col="event")

            summary = cph.summary
            rows.append({
                "reference": col,
                "coefficient": float(summary["coef"].iloc[0]),
                "hazard_ratio": float(summary["exp(coef)"].iloc[0]),
                "hr_lower_95": float(
                    summary["exp(coef) lower 95%"].iloc[0]
                ),
                "hr_upper_95": float(
                    summary["exp(coef) upper 95%"].iloc[0]
                ),
                "p_value": float(summary["p"].iloc[0]),
            })
        except (np.linalg.LinAlgError, ValueError, ArithmeticError) as exc:
            logger.warning(
                "Cox regression failed for reference '%s': %s", col, exc
            )
            rows.append({
                "reference": col,
                "coefficient": np.nan,
                "hazard_ratio": np.nan,
                "hr_lower_95": np.nan,
                "hr_upper_95": np.nan,
                "p_value": np.nan,
            })

    return pd.DataFrame(rows)
