"""Integration tests for the Linearity Assumption Assessment report section.

Covers the end-to-end contract that the code reviewer flagged as untested:

- ``DataPreprocessingSection._add_linearity_assessment`` renders without errors
- The cached ``_state.linearity_test_result`` from setup is **reused** during
  report generation (no redundant RCS refit)
- The ``martingale`` and ``linearity_test`` data keys are populated for export
- The ``linearity_violation`` limitations flag fires whenever a violation is
  detected: named affected models when Cox-family models are trained,
  otherwise an awareness entry that calls out no Cox-family models are in
  the analysis
- A clean-linear dataset produces a rendered section but no violation flag
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _linear_data(n: int = 350, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, 4)),
        columns=[f"x{i}" for i in range(4)],
    )
    lp = 0.5 * X["x0"] - 0.3 * X["x1"] + 0.2 * X["x2"]
    time = rng.exponential(scale=np.exp(-lp))
    event = rng.random(n) < 0.7
    y = np.array(
        [(bool(e), max(t, 0.01)) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


def _quadratic_data(n: int = 400, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, 4)),
        columns=[f"x{i}" for i in range(4)],
    )
    lp = 0.9 * (X["x0"] ** 2 - 1) - 0.3 * X["x1"]
    time = rng.exponential(scale=np.exp(-lp))
    event = rng.random(n) < 0.7
    y = np.array(
        [(bool(e), max(t, 0.01)) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


def _setup_and_train(X, y, model_name: str):
    learner = SurvivalLearner(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(verbose=False)
        learner.train(model_name, verbose=False)
    return learner


# ---------------------------------------------------------------------------
# DataPreprocessingSection integration
# ---------------------------------------------------------------------------


class TestLinearityAssessmentSection:
    def test_generate_includes_linearity_subsection(self):
        """Section renders with the heading, figure, and test table."""
        X, y = _linear_data()
        learner = _setup_and_train(X, y, "coxph")

        from mlsurv.reports.sections.data_preprocessing import (
            DataPreprocessingSection,
        )
        section = DataPreprocessingSection(learner)
        result = section.generate()

        md = result["markdown"]
        assert "Linearity Assumption Assessment" in md
        assert "Martingale residuals" in md
        assert "Log-Hazard Linearity Test" in md
        # Wald χ² header in the table
        assert "Wald" in md

        # Figure registered
        fig_ids = [f.figure_id for f in result["figure_info"]]
        assert "martingale_residuals" in fig_ids

        # Data keys populated for export
        assert "linearity_test" in section.data
        assert "martingale" in section.data

    def test_cached_linearity_result_is_reused(self, monkeypatch):
        """Report generation must NOT re-run linearity_test when cache exists."""
        X, y = _linear_data()
        learner = _setup_and_train(X, y, "coxph")
        assert learner._state.linearity_test_result is not None

        # Spy on linearity_test — the report must not call it because the
        # cached result is still valid.
        call_count = {"n": 0}
        real_fn = None
        from mlsurv.evaluation import metrics as _metrics_mod
        real_fn = _metrics_mod.linearity_test

        def _spy(*args, **kwargs):
            call_count["n"] += 1
            return real_fn(*args, **kwargs)

        monkeypatch.setattr(
            "mlsurv.evaluation.metrics.linearity_test", _spy,
        )

        from mlsurv.reports.sections.data_preprocessing import (
            DataPreprocessingSection,
        )
        section = DataPreprocessingSection(learner)
        section.generate()

        assert call_count["n"] == 0, (
            f"linearity_test was called {call_count['n']} times during report "
            f"generation; the cached _state.linearity_test_result should have "
            f"been reused instead"
        )

    def test_empty_cache_triggers_fresh_call(self, monkeypatch):
        """When no cache exists, the section falls back to a fresh computation."""
        X, y = _linear_data()
        learner = _setup_and_train(X, y, "coxph")
        learner._state.linearity_test_result = None  # simulate no cache

        from mlsurv.reports.sections.data_preprocessing import (
            DataPreprocessingSection,
        )
        section = DataPreprocessingSection(learner)
        result = section.generate()
        assert "Linearity Assumption Assessment" in result["markdown"]
        # A fresh computation should have populated data['linearity_test']
        assert "linearity_test" in section.data
        assert not section.data["linearity_test"].empty

    def test_section_handles_non_cox_model(self):
        """Section still renders for a non-Cox model (it's descriptive, not
        gated on Cox).  The limitations flag *is* gated, but that's tested
        separately below."""
        X, y = _linear_data()
        learner = _setup_and_train(X, y, "rsf")

        from mlsurv.reports.sections.data_preprocessing import (
            DataPreprocessingSection,
        )
        section = DataPreprocessingSection(learner)
        result = section.generate()
        assert "Linearity Assumption Assessment" in result["markdown"]


# ---------------------------------------------------------------------------
# LimitationsSection integration — flag gating
# ---------------------------------------------------------------------------


class TestLinearityLimitationsFlag:
    def test_flag_fires_on_violation_with_cox(self):
        """Quadratic LP + trained CoxPH → limitations flag fires."""
        X, y = _quadratic_data()
        learner = _setup_and_train(X, y, "coxph")

        from mlsurv.reports.sections.limitations import LimitationsSection
        section = LimitationsSection(learner)
        result = section.generate()

        assert "Log-Hazard Linearity Violations" in result["markdown"]
        assert "x0" in result["markdown"]

    def test_flag_fires_on_violation_without_cox(self):
        """Violation present but only RSF trained → flag still fires, names no Cox models."""
        X, y = _quadratic_data()
        learner = _setup_and_train(X, y, "rsf")

        from mlsurv.reports.sections.limitations import LimitationsSection
        section = LimitationsSection(learner)
        result = section.generate()

        from mlsurv.constants import (
            LINEARITY_ASSUMPTION_MODELS,
            MODEL_DISPLAY_NAMES,
        )

        assert "Log-Hazard Linearity Violations" in result["markdown"]
        assert "No affected models" in result["markdown"]
        assert "would matter if you add one" in result["markdown"]
        for name in LINEARITY_ASSUMPTION_MODELS:
            assert MODEL_DISPLAY_NAMES[name] in result["markdown"]
        assert "Affected trained models" not in result["markdown"]

    def test_flag_not_fired_on_linear_data_with_cox(self):
        """Strictly linear data + CoxPH → no violation, no flag."""
        X, y = _linear_data()
        learner = _setup_and_train(X, y, "coxph")

        from mlsurv.reports.sections.limitations import LimitationsSection
        section = LimitationsSection(learner)
        result = section.generate()

        assert "Log-Hazard Linearity Violations" not in result["markdown"]


# ---------------------------------------------------------------------------
# Full report end-to-end smoke test
# ---------------------------------------------------------------------------


class TestFullReportLinearityIntegration:
    def test_full_report_contains_linearity_section_and_flag(self, tmp_path):
        """Setup → train(coxph) → generate_report: both new pieces render."""
        X, y = _quadratic_data()
        learner = _setup_and_train(X, y, "coxph")

        out = tmp_path / "report.html"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner.generate_report(str(out))

        html = out.read_text(encoding="utf-8")
        assert "Linearity Assumption Assessment" in html
        assert "martingale_residuals" in html
        assert "Log-Hazard Linearity Violations" in html
        # The violating feature should be named in the limitations body
        assert "x0" in html
