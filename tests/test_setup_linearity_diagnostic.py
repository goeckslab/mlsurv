"""Tests for linearity diagnostic integration in setup().

Mirrors ``tests/test_setup_ph_diagnostic.py`` for the single-fit Wald
linearity test wired into ``_run_setup_diagnostics``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import MethodologicalWarning

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_linear_hazard_data(n: int = 400, seed: int = 42):
    """Synthetic data where the log-hazard is strictly linear in every feature."""
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


def _make_quadratic_hazard_data(n: int = 400, seed: int = 42):
    """Synthetic data where x0 enters the log-hazard quadratically."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, 4)),
        columns=[f"x{i}" for i in range(4)],
    )
    lp = 0.8 * (X["x0"] ** 2 - 1) - 0.3 * X["x1"]
    time = rng.exponential(scale=np.exp(-lp))
    event = rng.random(n) < 0.7
    y = np.array(
        [(bool(e), max(t, 0.01)) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


def _make_all_binary_data(n: int = 200, seed: int = 42):
    """All-binary X — no continuous features for the linearity test."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {f"b{i}": rng.integers(0, 2, size=n).astype(float) for i in range(4)},
    )
    lp = 0.5 * X["b0"] - 0.3 * X["b1"]
    time = rng.exponential(scale=np.exp(-lp))
    event = rng.random(n) < 0.7
    y = np.array(
        [(bool(e), max(t, 0.01)) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


@pytest.fixture
def linear_data():
    return _make_linear_hazard_data()


@pytest.fixture
def quadratic_data():
    return _make_quadratic_hazard_data()


# ---------------------------------------------------------------------------
# Tests: linearity test during setup
# ---------------------------------------------------------------------------


class TestSetupLinearityDiagnostic:
    """Tests for the ``linearity_test`` parameter in ``setup()``."""

    def test_setup_runs_linearity_test_by_default(self, linear_data):
        X, y = linear_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        assert learner._state.linearity_test_result is not None
        assert isinstance(learner._state.linearity_test_result, pd.DataFrame)
        assert not learner._state.linearity_test_result.empty

    def test_setup_linearity_test_false_skips(self, linear_data):
        X, y = linear_data
        learner = SurvivalLearner(X, y)
        learner.setup(linearity_test=False, verbose=False)
        assert learner._state.linearity_test_result is None

    def test_setup_linearity_does_not_break_ph_test(self, linear_data):
        """linearity_test=False still leaves PH result populated."""
        X, y = linear_data
        learner = SurvivalLearner(X, y)
        learner.setup(linearity_test=False, verbose=False)
        assert learner._state.ph_test_result is not None

    def test_setup_verbose_false_suppresses_table(self, linear_data, capsys):
        """verbose=False suppresses the unified recommendation table."""
        X, y = linear_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        captured = capsys.readouterr()
        assert "Model Recommendations" not in captured.out
        assert "Linearity Diagnostic" not in captured.out

    def test_setup_verbose_true_prints_recommendation_table(self, linear_data, capsys):
        """Linearity surfaces only via the recommendation table now."""
        X, y = linear_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=True)
        captured = capsys.readouterr()
        assert "Model Recommendations" in captured.out
        # Standalone linearity box is gone — it lived under "Linearity Diagnostic".
        assert "Linearity Diagnostic" not in captured.out

    def test_re_setup_clears_linearity_result(self, linear_data):
        X, y = linear_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        assert learner._state.linearity_test_result is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # re-setup SuggestionWarning
            learner.setup(linearity_test=False, verbose=False)
        assert learner._state.linearity_test_result is None

    def test_setup_linearity_result_has_correct_columns(self, linear_data):
        X, y = linear_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        df = learner._state.linearity_test_result
        expected = {
            "feature", "test_statistic", "p_value", "p_adjusted", "significant",
        }
        assert set(df.columns) == expected

    def test_setup_linearity_result_has_global_row(self, linear_data):
        X, y = linear_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        df = learner._state.linearity_test_result
        assert "GLOBAL" in df["feature"].values

    def test_setup_linearity_all_binary_skipped(self):
        """All-binary X has no continuous features → empty result, no warning."""
        X, y = _make_all_binary_data()
        learner = SurvivalLearner(X, y)
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            learner.setup(verbose=False)
        # Setup still completes; linearity result is None (empty DataFrame
        # from linearity_test is treated as 'no result' in setup wiring)
        assert learner._state.linearity_test_result is None
        # No linearity MethodologicalWarning should fire
        lin_warnings = [
            w for w in w_list
            if issubclass(w.category, MethodologicalWarning)
            and "linearity" in str(w.message).lower()
        ]
        assert len(lin_warnings) == 0


# ---------------------------------------------------------------------------
# Tests: linearity violation warning emission
# ---------------------------------------------------------------------------


class TestLinearityViolationSurfacing:
    """Linearity violations surface via BOTH a MethodologicalWarning
    and the recommendation table.

    The boxed ``_print_linearity_summary`` print was removed (the table
    replaces it), but the warning is preserved so methodological issues
    are not silenced.
    """

    def test_setup_linearity_warning_on_violation(self, quadratic_data):
        """Quadratic LP → MethodologicalWarning fires from setup()."""
        X, y = quadratic_data
        learner = SurvivalLearner(X, y)
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            learner.setup(verbose=False)

        lin_warnings = [
            w for w in w_list
            if type(w.message) is MethodologicalWarning
            and "Log-hazard linearity assumption violated" in str(w.message)
        ]
        assert len(lin_warnings) == 1
        msg = str(lin_warnings[0].message)
        assert "x0" in msg
        # The warning points at the no-recompute read path.
        assert "learner.linearity_test_result" in msg

    def test_violation_recorded_on_recommendation(self, quadratic_data):
        """Violation propagates to model_recommendations.linearity_violations."""
        X, y = quadratic_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        rec = learner._state.model_recommendations
        assert "x0" in rec.linearity_violations
        # And to the underlying DataFrame
        result = learner._state.linearity_test_result
        violations = result[
            (result["feature"] != "GLOBAL") & result["significant"]
        ]
        assert "x0" in violations["feature"].tolist()

    def test_violation_in_recommendation_reason(self, quadratic_data):
        """Linearity-assuming models mention the violation in their reason."""
        X, y = quadratic_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        rec = learner._state.model_recommendations
        # coxph assumes linearity; reason should reference linearity violations.
        assert "linearity violation" in rec.details["coxph"]

    def test_setup_linearity_no_violation_when_not_significant(self, linear_data):
        """Strictly linear data → no recorded linearity violations."""
        X, y = linear_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        rec = learner._state.model_recommendations
        assert rec.linearity_violations == ()


# ---------------------------------------------------------------------------
# Tests: wide-dataset GLOBAL-dilution regression
# ---------------------------------------------------------------------------


def _make_wide_quadratic_data(
    n: int = 400, seed: int = 9, n_features: int = 15,
    quadratic_strength: float = 0.15, censor_rate: float = 0.20,
):
    """Wide dataset where x0 enters quadratically but the GLOBAL Wald is
    diluted by the sheer number of features in the joint test.

    With ``n_features = 15`` and ``n_knots = 4``, the joint Wald has
    df = 30 — it needs a total chi-square of ~43.8 to clear 0.05. A
    moderate single-feature quadratic effect produces a per-feature
    Wald around 8–15 (easily BH-FDR significant), but the remaining
    14 noise features contribute near-zero to the joint statistic, so
    the total lands well under 30 and GLOBAL fails significance.

    After the 0.3 train/test split, ``n_train = 280`` passes the hard
    floor (``max(20, 10 * 15) = 150``).

    The parameters ``(n, n_features, quadratic_strength, censor_rate,
    seed) = (400, 15, 0.15, 0.20, 9)`` were found via a parameter sweep
    (see git history for the probe script) to produce a rock-solid
    ``x0_padj ≈ 0.003 AND GLOBAL p ≈ 0.23`` — a 14× / 4.5× margin on
    either side of the 0.05 threshold, which survives numerical jitter
    across platforms. The test asserts the "per-feature sig, GLOBAL
    not sig" invariant explicitly before asserting the warning fires,
    so a fixture drift becomes a clear failure rather than a silent
    pass.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"x{i}" for i in range(n_features)],
    )
    lp = quadratic_strength * (X["x0"] ** 2 - 1)
    time = rng.exponential(scale=np.exp(-lp))
    event = rng.random(n) > censor_rate
    y = np.array(
        [(bool(e), max(t, 0.01)) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


class TestWideDatasetLinearityWarning:
    """Per-feature BH-significance must fire the warning even when GLOBAL is not.

    Regression guard for the GLOBAL-gating bug discovered during
    E2E on the full GBSG2 feature set. See
    :func:`mlsurv.learner._setup._get_linearity_violations` — prior
    behavior gated on GLOBAL, silently missing single-feature
    violations on wide datasets.
    """

    def test_per_feature_warning_fires_when_global_not_significant(self):
        """Regression: per-feature BH-significance fires the warning
        AND populates the recommendation, even when GLOBAL is not sig.
        """
        X, y = _make_wide_quadratic_data()
        learner = SurvivalLearner(X, y)
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            learner.setup(verbose=False)

        result = learner._state.linearity_test_result
        assert result is not None and not result.empty

        global_row = result[result["feature"] == "GLOBAL"].iloc[0]
        per_feature = result[result["feature"] != "GLOBAL"]
        per_feature_violations = per_feature[per_feature["significant"]][
            "feature"
        ].tolist()

        assert "x0" in per_feature_violations, (
            f"fixture should produce x0 as per-feature BH-significant; "
            f"got {per_feature_violations} with "
            f"x0 p_adjusted="
            f"{per_feature[per_feature['feature'] == 'x0']['p_adjusted'].iloc[0]:.4f}"
        )
        assert not bool(global_row["significant"]), (
            f"fixture should produce GLOBAL not significant "
            f"(p={global_row['p_value']:.4f}); this regression test only "
            f"covers the dilution case — retune the fixture if this fires"
        )

        # Warning fires despite GLOBAL not significant.
        lin_warns = [
            w for w in w_list
            if type(w.message) is MethodologicalWarning
            and "Log-hazard linearity assumption violated" in str(w.message)
        ]
        assert len(lin_warns) == 1
        assert "x0" in str(lin_warns[0].message)

        # And the violation reaches the recommendation surface.
        rec = learner._state.model_recommendations
        assert "x0" in rec.linearity_violations
        assert "linearity violation" in rec.details["coxph"]
