"""Patient-level explanation result container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class PatientExplanation:
    """Result from ``learner.explain()`` — per-feature risk breakdown.

    Attributes
    ----------
    shap_values : pd.DataFrame
        Long-format DataFrame with columns:
        ``[model_name, feature, feature_value, shap_value, abs_shap_value,
        risk_score]``.  ``model_name`` and ``risk_score`` are constant per row.
    figure : plotly.graph_objects.Figure or None
        Waterfall plot (``None`` if ``plot=False``).
    base_value : float or None
        ``E[f(X)]`` — the model's expected risk over the background data.
        By construction, ``base_value + sum(shap_values) == risk_score``.
        ``None`` when the explainer did not expose an expected value.
    patient_index : int or None
        Positional row index into ``X_test`` for the explained patient.
        ``None`` when ``explain()`` was called with a new-patient ``X=``
        DataFrame (which has no test-set position).
    """

    shap_values: pd.DataFrame
    figure: Optional[object] = None  # go.Figure (optional import)
    base_value: Optional[float] = None
    patient_index: Optional[int] = None

    def __repr__(self) -> str:
        n_features = len(self.shap_values)
        model = self.shap_values["model_name"].iloc[0] if n_features else "?"
        risk = self.shap_values["risk_score"].iloc[0] if n_features else "?"
        has_fig = self.figure is not None
        return (
            f"PatientExplanation(model={model!r}, risk_score={risk}, "
            f"n_features={n_features}, has_figure={has_fig})"
        )
