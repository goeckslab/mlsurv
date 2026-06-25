"""ExplainMixin — individual patient-level risk explanations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner


class ExplainMixin:
    """Mixin providing ``explain()`` for per-patient SHAP explanations."""

    def explain(
        self: SurvivalLearner,
        patient: int | str | None = None,
        X: pd.DataFrame | None = None,
        model: str | None = None,
        n_features: int = 15,
        plot: bool = True,
        max_background: int = 50,
    ):
        """Explain an individual patient's risk prediction via SHAP values.

        Works after ``train()`` alone — no ``evaluate()`` required.

        Parameters
        ----------
        patient : int or str, optional
            Positional index into ``X_test``, or a risk-percentile spec
            evaluated against the resolved model's predictions on
            ``X_test``: ``'low_risk'`` (= ``'p10'``), ``'high_risk'``
            (= ``'p90'``), ``'median'`` (= ``'p50'``), or ``'pNN'`` for
            any NN in 0..100. Mutually exclusive with *X*.
        X : pd.DataFrame, optional
            New patient data (1 row).  Mutually exclusive with *patient*.
        model : str, optional
            Model name (e.g. ``'coxph'``, ``'rsf'``).  Required when more than
            one model is trained; if only one model exists, it is used
            automatically.
        n_features : int, default=15
            Maximum number of features to show in the waterfall plot.
        plot : bool, default=True
            Whether to generate a waterfall plot.
        max_background : int, default=50
            Maximum background samples for the SHAP explainer.

        Returns
        -------
        PatientExplanation
            Contains ``shap_values`` DataFrame and optional ``figure``.

        Raises
        ------
        TypeError
            If ``X`` is provided but is not a pandas DataFrame.
        ValueError
            If both/neither ``patient`` and ``X`` are provided, if the
            requested model is not trained, or if ``patient`` is out of
            bounds.
        """
        self._ensure_setup()

        # -- Validate patient input ------------------------------------------
        if (patient is None) == (X is None):
            raise ValueError(
                "Provide exactly one of 'patient' (index into X_test) "
                "or 'X' (new patient DataFrame), not both."
            )

        # -- Resolve model ---------------------------------------------------
        model_result = self._resolve_single_model(model)
        pipeline = model_result.model  # sklearn Pipeline
        model_name = model_result.name

        # -- Resolve patient data --------------------------------------------
        # Positional row index into X_test for the patient= path; stays None
        # for the X= (new-patient DataFrame) path, which has no test position.
        patient_index = None
        if patient is not None:
            from ._patient_resolve import resolve_patient_index

            X_test = self._state.X_test
            if X_test is None:
                raise ValueError("No test data available. Call setup() first.")
            patient_index = resolve_patient_index(patient, X_test, pipeline)
            X_patient = X_test.iloc[[patient_index]]
        else:
            if not isinstance(X, pd.DataFrame):
                raise TypeError("X must be a pandas DataFrame.")
            if len(X) != 1:
                raise ValueError(
                    f"X must contain exactly 1 row, got {len(X)}."
                )
            X_patient = X

        # -- Compute risk score ----------------------------------------------
        risk_score = float(pipeline.predict(X_patient)[0])

        # -- Compute SHAP contributions --------------------------------------
        from ..evaluation._explain import compute_patient_explanation

        X_train = self._state.X_train
        contributions, base_value = compute_patient_explanation(
            pipeline, X_patient, X_train,
            max_background=max_background,
            random_state=self._state.random_state,
            categorical_groups=self._state.categorical_groups,
        )

        # Add model_name and risk_score columns
        contributions.insert(0, "model_name", model_name)
        contributions["risk_score"] = risk_score

        # -- Build waterfall plot --------------------------------------------
        figure = None
        if plot:
            from ..viz.explain import plot_waterfall

            # Name the patient in the title by their X_test position (same
            # "patient N" convention as the survival-curve plot) when one was
            # resolved; the X= (new-patient) path has no position to show.
            title = f"Risk Explanation — {model_name}"
            if patient_index is not None:
                title = f"{title} (patient {patient_index})"
            figure = plot_waterfall(
                contributions,
                risk_score=risk_score,
                base_value=base_value,
                n=n_features,
                title=title,
                feature_groups=getattr(self._state, "feature_groups", None),
                group_colors=getattr(self._state, "group_colors", None),
            )

        # -- Return result ---------------------------------------------------
        from ..results._explanation import PatientExplanation

        return PatientExplanation(
            shap_values=contributions, figure=figure, base_value=base_value,
            patient_index=patient_index,
        )
