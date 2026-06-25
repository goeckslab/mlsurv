"""Tests for the PH-test / Schoenfeld single-fit consolidation.

The report's PH section used to fit the same lifelines ``CoxPHFitter`` twice
(once via ``compute_schoenfeld_data`` for the scatter, once via ``ph_test``
for the p-value table). ``ph_test_with_schoenfeld`` now derives both from one
robust fit, and the three ``compute_*`` functions moved from ``viz`` to
``evaluation``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from tests._helpers import make_survival_array

pytest.importorskip("lifelines")


@pytest.fixture
def cox_data():
    """Synthetic data suitable for Cox fitting (clean, full rank)."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        'age': rng.normal(60, 10, n),
        'bmi': rng.normal(25, 5, n),
        'crp': rng.exponential(1, n),
        'albumin': rng.normal(4, 0.5, n),
    })
    y = make_survival_array(
        rng.binomial(1, 0.5, n),
        rng.exponential(24, n),
    )
    return X, y


def _sorted(df):
    return df.sort_values(["feature", "time"]).reset_index(drop=True)


class TestRelocationBackwardCompat:
    """The compute functions moved to evaluation/ but stay importable from viz/."""

    def test_same_object_from_both_paths(self):
        from mlsurv.viz.diagnostics import compute_schoenfeld_data as v_sch
        from mlsurv.evaluation._diagnostics import compute_schoenfeld_data as e_sch
        from mlsurv.evaluation import compute_schoenfeld_data as pkg_sch
        assert v_sch is e_sch is pkg_sch

    def test_deviance_and_martingale_reexported(self):
        from mlsurv.viz.diagnostics import (
            compute_deviance_data, compute_martingale_data,
        )
        from mlsurv.evaluation._diagnostics import (
            compute_deviance_data as e_dev,
            compute_martingale_data as e_mart,
        )
        assert compute_deviance_data is e_dev
        assert compute_martingale_data is e_mart


class TestCombinedEntryPoint:
    """``ph_test_with_schoenfeld`` returns the same numbers as the split path."""

    def test_ph_df_matches_ph_test(self, cox_data):
        from mlsurv.evaluation.metrics import ph_test, ph_test_with_schoenfeld
        X, y = cox_data
        ph_only = ph_test(X, y)
        ph_df, _sch = ph_test_with_schoenfeld(X, y, n_features=X.shape[1])
        pd.testing.assert_frame_equal(
            ph_df.reset_index(drop=True), ph_only.reset_index(drop=True),
        )

    def test_schoenfeld_matches_standalone(self, cox_data):
        from mlsurv.evaluation.metrics import ph_test_with_schoenfeld
        from mlsurv.evaluation._diagnostics import compute_schoenfeld_data
        X, y = cox_data
        _ph, sch_combined = ph_test_with_schoenfeld(X, y, n_features=X.shape[1])
        sch_standalone = compute_schoenfeld_data(X, y, n_features=X.shape[1])
        # Byte-identical residuals: both come from the same robust fit.
        assert len(sch_combined) == len(sch_standalone)
        np.testing.assert_array_equal(
            _sorted(sch_combined)["residual"].values,
            _sorted(sch_standalone)["residual"].values,
        )

    def test_n_features_default_is_six(self):
        """n_features=None selects the top 6 features (not all 8)."""
        from mlsurv.evaluation.metrics import ph_test_with_schoenfeld
        rng = np.random.default_rng(7)
        n = 200
        X = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(8)})
        y = make_survival_array(
            rng.binomial(1, 0.5, n), rng.exponential(24, n),
        )
        _ph, sch = ph_test_with_schoenfeld(X, y)  # n_features=None -> 6
        assert sch["feature"].nunique() == 6

    def test_all_nan_returns_two_empty_frames(self, cox_data):
        """All-NaN input -> empty ph_df AND empty schoenfeld_df, correct schemas."""
        from mlsurv.evaluation.metrics import ph_test_with_schoenfeld
        X, y = cox_data
        X = X.copy()
        X[:] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ph_df, sch_df = ph_test_with_schoenfeld(X, y, n_features=X.shape[1])
        assert ph_df.empty
        assert sch_df.empty
        assert list(sch_df.columns) == ["feature", "time", "residual"]


class TestDegradedResidualPath:
    """Residual computation fails but the Cox fit + PH table succeed."""

    def test_ph_df_survives_schoenfeld_empty(self, cox_data, monkeypatch):
        """_safe_scaled_schoenfeld -> None yields a full ph_df, empty schoenfeld."""
        import mlsurv.evaluation.metrics as metrics
        monkeypatch.setattr(metrics, "_safe_scaled_schoenfeld", lambda cph, df: None)

        X, y = cox_data
        ph_df, sch_df = metrics.ph_test_with_schoenfeld(X, y, n_features=X.shape[1])

        # PH table is still valid (joint test recomputes residuals internally).
        assert not ph_df.empty
        assert "GLOBAL" in set(ph_df["feature"])
        # Schoenfeld scatter is gracefully omitted, with the correct schema.
        assert sch_df.empty
        assert list(sch_df.columns) == ["feature", "time", "residual"]

    def test_report_surfaces_empty_schoenfeld(self, cox_data, monkeypatch):
        """Report shows an 'unavailable' note (not a blank plot) + the PH table."""
        import mlsurv.evaluation.metrics as metrics
        monkeypatch.setattr(metrics, "_safe_scaled_schoenfeld", lambda cph, df: None)

        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.data_preprocessing import (
            DataPreprocessingSection,
        )
        X, y = cox_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3, verbose=False)

        section = DataPreprocessingSection(learner)
        md = []
        section._add_diagnostics(md, [], [])
        text = "\n".join(md)

        assert "Schoenfeld residual plot unavailable" in text
        assert "schoenfeld" not in section.data            # plot helper not run
        assert "ph_test" in section.data                   # table still rendered
        assert not section.data["ph_test"].empty


class TestReportStoresBothFromOneFit:
    """The consolidation's payoff: plot + table populated from one fit."""

    def test_both_data_keys_consistent(self, cox_data):
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.data_preprocessing import (
            DataPreprocessingSection,
        )
        X, y = cox_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3, verbose=False)

        section = DataPreprocessingSection(learner)
        section._add_diagnostics([], [], [])

        assert "schoenfeld" in section.data
        assert "ph_test" in section.data
        ph_features = set(section.data["ph_test"]["feature"]) - {"GLOBAL"}
        sch_features = set(section.data["schoenfeld"]["feature"])
        # Same fit -> the plotted features are a subset of the tested features.
        assert sch_features.issubset(ph_features)


class TestSingleFit:
    """The consolidated path fits CoxPHFitter exactly once (was twice)."""

    def test_combined_function_fits_once(self, cox_data, monkeypatch):
        from lifelines import CoxPHFitter
        calls = {"n": 0}
        original_fit = CoxPHFitter.fit

        def counting_fit(self, *args, **kwargs):
            calls["n"] += 1
            return original_fit(self, *args, **kwargs)

        monkeypatch.setattr(CoxPHFitter, "fit", counting_fit)

        from mlsurv.evaluation.metrics import ph_test_with_schoenfeld
        X, y = cox_data
        ph_test_with_schoenfeld(X, y, n_features=X.shape[1])
        assert calls["n"] == 1

    def test_report_section_fits_once(self, cox_data, monkeypatch):
        """``_add_diagnostics`` does one CoxPHFitter.fit (down from two)."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.data_preprocessing import (
            DataPreprocessingSection,
        )

        X, y = cox_data
        learner = SurvivalLearner(X, y)
        # setup() runs its own diagnostics; count only the report section below.
        learner.setup(random_state=42, cv_folds=3, verbose=False)

        from lifelines import CoxPHFitter
        calls = {"n": 0}
        original_fit = CoxPHFitter.fit

        def counting_fit(self, *args, **kwargs):
            calls["n"] += 1
            return original_fit(self, *args, **kwargs)

        monkeypatch.setattr(CoxPHFitter, "fit", counting_fit)

        section = DataPreprocessingSection(learner)
        section._add_diagnostics([], [], [])
        assert calls["n"] == 1


class TestRobustFitUpgrade:
    """The Schoenfeld scatter now inherits the robust fallback fit."""

    def test_zero_variance_column_is_dropped_not_fatal(self, cox_data):
        """A constant column would make the old fixed-penalizer fit fail.

        ``_fit_cox_with_fallback`` pre-drops it, so the residuals compute and
        the constant column simply does not appear.
        """
        from mlsurv.evaluation._diagnostics import compute_schoenfeld_data
        X, y = cox_data
        X = X.copy()
        X["const"] = 1.0  # zero variance

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sch = compute_schoenfeld_data(X, y, n_features=X.shape[1])

        assert not sch.empty
        assert "const" not in set(sch["feature"])

    def test_fallback_drops_the_zero_variance_column(self, cox_data):
        """Make the robustness claim load-bearing: the fallback pre-drops 'const'.

        The old fixed ``CoxPHFitter(penalizer=0.01)`` had no such guard; the
        constant column produced NaN gradients.
        """
        from mlsurv.evaluation.metrics import _fit_cox_with_fallback
        X, y = cox_data
        df = X.copy()
        df["const"] = 1.0
        df["T"] = y["time"]
        df["E"] = y["event"].astype(int)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _cph, info = _fit_cox_with_fallback(
                df, duration_col="T", event_col="E", context="test",
            )
        assert info["dropped_zero_var"] == ["const"]

    def test_plot_and_table_share_one_fit_on_hard_data(self, cox_data):
        """Combined path stays self-consistent when a column is dropped."""
        from mlsurv.evaluation.metrics import ph_test_with_schoenfeld
        from mlsurv.evaluation._diagnostics import compute_schoenfeld_data
        X, y = cox_data
        X = X.copy()
        X["const"] = 1.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ph, sch_combined = ph_test_with_schoenfeld(X, y, n_features=X.shape[1])
            sch_standalone = compute_schoenfeld_data(X, y, n_features=X.shape[1])

        assert len(sch_combined) == len(sch_standalone)
        np.testing.assert_array_equal(
            _sorted(sch_combined)["residual"].values,
            _sorted(sch_standalone)["residual"].values,
        )
