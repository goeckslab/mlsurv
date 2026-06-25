"""Individual patient SHAP explanation computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ._shap import _shap_array


def compute_patient_explanation(
    pipeline: Pipeline,
    X_patient: pd.DataFrame,
    X_background: pd.DataFrame,
    max_background: int = 50,
    random_state: int | None = None,
    categorical_groups: dict | None = None,
) -> pd.DataFrame:
    """Compute SHAP-based feature contributions for a single patient.

    Explains the **full pipeline in raw feature space** via the shared SHAP
    core (:func:`mlsurv.evaluation._shap._shap_array`) — exactly the engine the
    cohort path uses, with the one patient as the explained set.  Because per-row
    SHAP values are independent of the rest of the explained set, this patient's
    contributions equal the row they would get inside a cohort run with the same
    background and ``random_state``, so the waterfall reconciles 1:1 with the
    cohort beeswarm.

    Selector-dropped raw features appear here with a SHAP value of exactly 0
    (the pipeline ignores them, so perturbing them never moves the prediction).
    They are kept in the returned DataFrame so it matches the cohort; the display
    layers (``plot_waterfall``, the patient-report feature table) hide them.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline (preprocessing + survival model).  Passed whole to the
        explainer — raw features in, risk score out.
    X_patient : pd.DataFrame
        Patient data (exactly 1 row, raw features).
    X_background : pd.DataFrame
        Background data for the SHAP masker (typically ``X_train``, raw features).
    max_background : int, default=50
        Maximum background samples for the SHAP explainer.
    random_state : int, optional
        Seed for reproducible background subsampling.
    categorical_groups : dict, optional
        Mapping from group name to dummy column names. When provided,
        SHAP values for dummy columns are summed per group.

    Returns
    -------
    tuple
        ``(contributions, base_value)`` where ``contributions`` is a
        ``DataFrame`` with columns
        ``[feature, feature_value, shap_value, abs_shap_value]`` (one row
        per raw feature, sorted by ``abs_shap_value`` descending) and
        ``base_value`` is the scalar ``E[f(X)]`` over the background.
        By construction, ``base_value + contributions.shap_value.sum() ==
        pipeline.predict(X_patient)``.

    Raises
    ------
    ImportError
        If the optional ``shap`` package is not installed.
    ValueError
        If ``X_patient`` does not contain exactly one row, or if the background
        is empty or its columns do not match ``X_patient``.
    """
    if len(X_patient) != 1:
        raise ValueError(
            f"X_patient must contain exactly 1 row, got {len(X_patient)}."
        )

    try:
        import shap  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'shap' package is required for patient explanations. "
            "Install with: pip install shap  or  pip install mlsurv[deep]"
        )

    # Explain the full pipeline on raw features — same engine, model, and
    # background as the cohort path (the shared core), so the two cannot drift.
    shap_array, base_value = _shap_array(
        pipeline, X_patient, X_background, max_background,
        random_state=random_state,
    )

    # Build result DataFrame (one row per raw feature).  Raw feature names are
    # the input columns, so feature values come straight from X_patient.
    features = X_patient.columns.tolist()
    feature_values = [X_patient[feat].iloc[0] for feat in features]

    sv = shap_array[0]  # first (only) patient row
    df = pd.DataFrame({
        "feature": features,
        "feature_value": feature_values,
        "shap_value": sv,
        "abs_shap_value": np.abs(sv),
    })

    if categorical_groups:
        from .importance import _validate_categorical_groups
        valid_groups = _validate_categorical_groups(
            categorical_groups, df['feature'].tolist(),
        )
        if valid_groups:
            df = _group_patient_shap(df, valid_groups)

    sorted_df = df.sort_values("abs_shap_value", ascending=False).reset_index(drop=True)
    return sorted_df, base_value


def _group_patient_shap(df, categorical_groups):
    """Group patient SHAP contributions by categorical groups.

    Sums ``shap_value`` across dummy columns for each group, recomputes
    ``abs_shap_value`` as ``abs(summed_shap)`` (not sum of absolutes), and
    sets ``feature_value`` to the active category's dummy column name
    (the one with the highest feature_value in the group).

    Parameters
    ----------
    df : pd.DataFrame
        Patient contributions with columns
        ``[feature, feature_value, shap_value, abs_shap_value]``.
    categorical_groups : dict of {str: list of str}
        Validated group mapping.

    Returns
    -------
    pd.DataFrame
        Grouped contributions (unsorted — caller handles sorting).
    """
    col_to_group = {
        col: group
        for group, cols in categorical_groups.items()
        for col in cols
    }

    grouped_mask = df['feature'].isin(col_to_group)
    ungrouped = df[~grouped_mask]
    to_group = df[grouped_mask].copy()

    if to_group.empty:
        return df

    # Before replacing names, find the active category per group
    # (the dummy with the highest feature_value, i.e. the "1" in one-hot)
    active_category = {}
    for group_name, cols in categorical_groups.items():
        group_rows = to_group[to_group['feature'].isin(cols)]
        if not group_rows.empty:
            active_idx = group_rows['feature_value'].idxmax()
            active_category[group_name] = group_rows.loc[active_idx, 'feature']

    to_group['feature'] = to_group['feature'].map(col_to_group)
    aggregated = (
        to_group.groupby('feature', sort=False)['shap_value']
        .sum()
        .reset_index()
    )
    aggregated['feature_value'] = aggregated['feature'].map(active_category)
    aggregated['abs_shap_value'] = aggregated['shap_value'].abs()

    return pd.concat([ungrouped, aggregated], ignore_index=True)
