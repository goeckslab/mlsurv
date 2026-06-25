"""Apparent (resubstitution) train-set metrics.

Covers the four ``*_train`` concordance columns added for the opt-in train
source: ``c_index_train`` (Harrell), ``c_index_ipcw_train`` (Uno),
``c_index_td_train`` (Antolini), and ``integrated_brier_train`` (IBS).

- Computation: all four are produced (and finite) by ``evaluate()`` when
  training data is available, and absent when it is not.
- Export: they flow into ``EvaluationResult.to_dataframes()`` (and thus
  ``export_results``) automatically.
- Hidden: they are deliberately kept out of the metric registry, so they never
  appear in the printed evaluation table or the HTML report.
"""

import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest

TRAIN_COLUMNS = [
    "c_index_train",
    "c_index_ipcw_train",
    "c_index_td_train",
    "integrated_brier_train",
]


# ---------------------------------------------------------------------------
# Registry / print exclusion -- fast (no model training)
# ---------------------------------------------------------------------------


class TestTrainMetricsHidden:
    """Train columns must stay out of the registry-driven surfaces."""

    def test_train_columns_not_registered(self):
        """No registered metric points at a ``*_train`` concordance column."""
        from mlsurv.constants import (
            ALL_METRICS, METRIC_REGISTRY, get_concordance_column,
        )

        for key, info in METRIC_REGISTRY.items():
            col = info.concordance_column
            assert col is None or not col.endswith("_train"), (
                f"metric '{key}' is registered against a train column ({col}); "
                "train metrics must not be registered (keeps them out of "
                "printed output and the HTML report)"
            )
        # Every scorable metric maps to a test column, never a train one.
        for key in ALL_METRICS:
            col = get_concordance_column(key)
            assert col is not None and not col.endswith("_train")

    def test_train_columns_absent_from_printed_table(self):
        """print_evaluation_table iterates ALL_METRICS, so it omits train cols."""
        from mlsurv.results.printing import print_evaluation_table

        conc = pd.DataFrame({
            "model_type": ["coxph"],
            "c_index_test": [0.70],
            "c_index_ipcw_test": [0.69],
            "c_index_td_test": [0.71],
            "integrated_brier_test": [0.15],
            # Train columns present in the table but must not be printed.
            "c_index_train": [0.95],
            "c_index_ipcw_train": [0.94],
            "c_index_td_train": [0.96],
            "integrated_brier_train": [0.05],
        })
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_evaluation_table(conc, model_name="coxph")
        out = buf.getvalue()
        for col in TRAIN_COLUMNS:
            assert col not in out, f"{col} leaked into the printed table"
        # The test metrics it does show prove the table actually rendered.
        assert "0.700" in out


# ---------------------------------------------------------------------------
# Computation + export -- integration (trains CoxPH)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTrainMetricsComputed:
    """evaluate() produces the four train metrics from resubstitution."""

    @pytest.fixture(scope="class")
    def evaluated(self):
        """Fit CoxPH and evaluate with training data available."""
        from sklearn.model_selection import train_test_split

        from mlsurv.evaluation import evaluate
        from mlsurv.models.coxph import CoxPHModel

        rng = np.random.RandomState(123)
        n = 120
        X = pd.DataFrame(rng.randn(n, 3), columns=["a", "b", "c"])
        y = np.array(
            [(bool(rng.rand() > 0.3), rng.exponential(5)) for _ in range(n)],
            dtype=[("event", bool), ("time", float)],
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42,
        )
        model = CoxPHModel()
        model.fit(X_train, y_train)
        result = evaluate(
            model, X_test, y_test, X_train=X_train, y_train=y_train,
            show_progress=False, n_importance_repeats=0,
        )
        return result, model, X_test, y_test

    def test_all_four_present_and_finite(self, evaluated):
        """CoxPH supports survival functions, so all four are real values."""
        result = evaluated[0]
        conc = result.concordance
        for col in TRAIN_COLUMNS:
            assert col in conc.columns, f"missing train column: {col}"
            val = conc[col].iloc[0]
            assert np.isfinite(val), f"{col} should be finite for CoxPH"
        # The three C-indices live in [0, 1]; IBS is a small non-negative loss.
        for col in ("c_index_train", "c_index_ipcw_train", "c_index_td_train"):
            assert 0.0 <= conc[col].iloc[0] <= 1.0
        assert conc["integrated_brier_train"].iloc[0] >= 0.0

    def test_train_optimism_direction(self, evaluated):
        """Apparent train Harrell C should not be below the test C.

        Resubstitution is optimistic, so c_index_train >= c_index_test (allow a
        tiny tolerance for noise on a small sample).
        """
        conc = evaluated[0].concordance
        assert (
            conc["c_index_train"].iloc[0]
            >= conc["c_index_test"].iloc[0] - 0.05
        )

    def test_absent_without_training_data(self, evaluated):
        """Without X_train, none of the four train columns are produced."""
        from mlsurv.evaluation import evaluate

        _, model, X_test, y_test = evaluated
        result = evaluate(
            model, X_test, y_test, show_progress=False, n_importance_repeats=0,
        )
        for col in TRAIN_COLUMNS:
            assert col not in result.concordance.columns, (
                f"{col} should be absent when X_train is not passed"
            )

    def test_exported_via_to_dataframes(self, evaluated):
        """The whole concordance table ships, so train cols reach export."""
        result = evaluated[0]
        metrics_df = result.to_dataframes()["metrics"]
        for col in TRAIN_COLUMNS:
            assert col in metrics_df.columns, (
                f"{col} missing from export 'metrics' DataFrame"
            )
