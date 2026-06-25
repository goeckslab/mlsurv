"""Tests for _ROCSubsection report integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from mlsurv.reports.sections.evaluation._roc import _ROCSubsection

from tests._helpers import make_survival_array


def _mock_section(
    has_auc_t=True,
    has_test_set=True,
    has_y_data=True,
    n_models=2,
    n_times=3,
):
    """Build a mock EvaluationSection with configurable data availability."""
    section = MagicMock()
    section._log_recoverable_exception = MagicMock()
    section._format_table = MagicMock(return_value="| col |\n|-----|\n| val |")
    section.data = {}

    rng = np.random.RandomState(42)
    n = 100

    # eval_result
    er = MagicMock()

    if has_auc_t:
        rows = []
        for mt in [f"model_{i}" for i in range(n_models)]:
            for t in np.linspace(2, 8, n_times):
                rows.append({"model_type": mt, "eval_time": t, "auc": rng.uniform(0.5, 0.9)})
        er.auc_t = pd.DataFrame(rows)
    else:
        er.auc_t = None

    if has_test_set:
        ts_rows = []
        for mt in [f"model_{i}" for i in range(n_models)]:
            for i in range(n):
                ts_rows.append({
                    "model_type": mt,
                    "risk_score": rng.randn(),
                    "time": rng.exponential(5),
                    "event": rng.choice([True, False]),
                })
        er.test_set = pd.DataFrame(ts_rows)
    else:
        er.test_set = None

    section.eval_result = er

    # learner
    learner = MagicMock()
    if has_y_data:
        learner.y_train = make_survival_array(
            rng.choice([True, False], n), rng.exponential(5, n),
        )
        learner.y_test = make_survival_array(
            rng.choice([True, False], n), rng.exponential(5, n),
        )
    else:
        learner.y_train = None
        learner.y_test = None
    section.learner = learner

    return section


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestROCSubsectionAvailability:
    """is_available() checks."""

    def test_available_with_all_data(self):
        s = _mock_section()
        sub = _ROCSubsection(s)
        assert sub.is_available() is True

    def test_unavailable_no_auc_t(self):
        s = _mock_section(has_auc_t=False)
        sub = _ROCSubsection(s)
        assert sub.is_available() is False

    def test_unavailable_no_test_set(self):
        s = _mock_section(has_test_set=False)
        sub = _ROCSubsection(s)
        assert sub.is_available() is False

    def test_unavailable_no_y_data(self):
        s = _mock_section(has_y_data=False)
        sub = _ROCSubsection(s)
        assert sub.is_available() is False

    def test_unavailable_no_eval_result(self):
        s = _mock_section()
        s.eval_result = None
        sub = _ROCSubsection(s)
        assert sub.is_available() is False


class TestROCSubsectionAdd:
    """add() produces markdown, figures, and figure_info."""

    def test_produces_figure(self):
        s = _mock_section()
        sub = _ROCSubsection(s)
        md, figs, fi = [], [], []
        sub.add(md, figs, fi)
        assert len(figs) == 1
        assert figs[0][0] == "roc_curves.png"

    def test_produces_figure_info(self):
        s = _mock_section()
        sub = _ROCSubsection(s)
        md, figs, fi = [], [], []
        sub.add(md, figs, fi)
        assert len(fi) == 1
        assert fi[0].figure_id == "roc_curves"
        assert fi[0].data_key == "roc"

    def test_stores_tidy_data(self):
        s = _mock_section()
        sub = _ROCSubsection(s)
        md, figs, fi = [], [], []
        sub.add(md, figs, fi)
        assert "roc" in s.data
        df = s.data["roc"]
        assert isinstance(df, pd.DataFrame)
        assert {"model", "fpr", "tpr", "auc", "eval_time"}.issubset(df.columns)

    def test_markdown_contains_placeholder(self):
        s = _mock_section()
        sub = _ROCSubsection(s)
        md, figs, fi = [], [], []
        sub.add(md, figs, fi)
        joined = "\n".join(md)
        assert "MLSURV_RAW_HTML" in joined
        assert "roc_curves.png" in joined

    def test_graceful_on_compute_failure(self):
        """If compute_roc_curves fails for all times, show fallback."""
        s = _mock_section()
        sub = _ROCSubsection(s)
        md, figs, fi = [], [], []
        with patch(
            "mlsurv.viz._metrics_roc.compute_roc_curves",
            side_effect=ValueError("fail"),
        ):
            sub.add(md, figs, fi)
        joined = "\n".join(md)
        assert "unavailable" in joined.lower()
        assert len(figs) == 0


class TestROCSubsectionBootstrapCI:
    """CI ribbon integration when bootstrap predictions exist."""

    def _mock_section_with_bootstrap(self, n_bootstrap=5):
        """Build a mock section with bootstrap predictions."""
        s = _mock_section()
        rng = np.random.RandomState(42)
        n = 100

        # Build predictions DataFrame matching real structure
        pred_rows = []
        for mt in ["model_0", "model_1"]:
            for i in range(n_bootstrap):
                idx = rng.randint(0, n, size=n)
                for pos in idx:
                    pred_rows.append({
                        "model_type": mt,
                        "iteration": i,
                        "index": pos,
                        "risk_score": rng.randn(),
                    })
        predictions_df = pd.DataFrame(pred_rows)

        # Attach bootstrap_results as dict of mock BootstrapResults
        boot_mock_0 = MagicMock()
        boot_mock_0.predictions = predictions_df[predictions_df["model_type"] == "model_0"]
        boot_mock_1 = MagicMock()
        boot_mock_1.predictions = predictions_df[predictions_df["model_type"] == "model_1"]
        s.bootstrap_results = {"model_0": boot_mock_0, "model_1": boot_mock_1}

        # Learner needs X_test.index
        s.learner.X_test = MagicMock()
        s.learner.X_test.index = pd.RangeIndex(n)

        return s

    def test_ci_description_present(self):
        """When bootstrap predictions exist, CI description is added."""
        s = self._mock_section_with_bootstrap()
        sub = _ROCSubsection(s)
        md, figs, fi = [], [], []
        sub.add(md, figs, fi)
        joined = "\n".join(md)
        assert "confidence interval" in joined.lower()

    def test_no_ci_without_bootstrap(self):
        """Without bootstrap results, no CI description."""
        s = _mock_section()
        s.bootstrap_results = None
        sub = _ROCSubsection(s)
        md, figs, fi = [], [], []
        sub.add(md, figs, fi)
        joined = "\n".join(md)
        assert "confidence interval" not in joined.lower()
        # Figure should still be produced
        assert len(figs) == 1

    def test_no_ci_when_predictions_none(self):
        """bootstrap_results present but predictions=None → fallback note."""
        s = _mock_section()
        boot_mock = MagicMock()
        boot_mock.predictions = None
        s.bootstrap_results = {"model_0": boot_mock}
        sub = _ROCSubsection(s)
        md, figs, fi = [], [], []
        sub.add(md, figs, fi)
        joined = "\n".join(md)
        assert "confidence interval" not in joined.lower()
        assert "could not be computed" in joined.lower()
        assert len(figs) == 1

    def test_ci_graceful_on_failure(self):
        """If bootstrap CI computation fails, fall back with markdown note."""
        s = self._mock_section_with_bootstrap()
        sub = _ROCSubsection(s)
        md, figs, fi = [], [], []
        with patch(
            "mlsurv.viz._metrics_roc.bootstrap_roc_curves",
            side_effect=RuntimeError("bootstrap fail"),
        ):
            sub.add(md, figs, fi)
        # Should still produce a figure without CI
        assert len(figs) == 1
        joined = "\n".join(md)
        assert "confidence interval" not in joined.lower()
        assert "could not be computed" in joined.lower()
