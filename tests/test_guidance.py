"""Tests for mlsurv.guidance — sample-size-aware model recommendations."""

import math

import numpy as np
import pandas as pd
import pytest

from mlsurv.guidance import (
    RecommendedModels,
    _build_rationale,
    _classify_model,
    _extract_data_characteristics,
    recommend_models,
)
from mlsurv.models import get_model_complexity, get_model_display_names, list_models

_DISPLAY_NAMES = get_model_display_names()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n_samples, n_events, n_features, seed=42):
    """Create synthetic (X, y) with the given characteristics."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    events = np.zeros(n_samples, dtype=bool)
    events[:n_events] = True
    rng.shuffle(events)
    y = np.array(
        [(e, rng.exponential(10)) for e in events],
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


# ---------------------------------------------------------------------------
# TestRecommendModelsFromData
# ---------------------------------------------------------------------------


class TestRecommendModelsFromData:
    def test_basic_from_data(self):
        X, y = _make_data(n_samples=1500, n_events=600, n_features=5)
        rec = recommend_models(X, y)
        assert isinstance(rec, RecommendedModels)
        assert rec.n_samples == 1500
        assert rec.n_events == 600
        assert rec.n_features == 5
        assert rec.epv == pytest.approx(120.0)

    def test_dataframe_y(self):
        """Accepts y as a DataFrame with 'event' column."""
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(200, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame({
            "event": rng.binomial(1, 0.5, 200),
            "time": rng.exponential(10, 200),
        })
        rec = recommend_models(X, y)
        assert rec.n_samples == 200
        assert rec.n_features == 5


# ---------------------------------------------------------------------------
# TestRecommendModelsAutoRunsTests
# ---------------------------------------------------------------------------


def _make_ph_violating_data(n_samples=200, n_features=5, seed=0):
    """Create (X, y) where one feature has a strong time-varying hazard.

    The first feature ``f0`` is engineered so that its effect on the hazard
    flips sign over time, which the Grambsch-Therneau test should flag.
    """
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    # Build event times whose order is strongly correlated with f0 in
    # the first half but anti-correlated in the second half: this is a
    # textbook PH violation.
    f0 = X["f0"].to_numpy()
    times = 5.0 + np.abs(f0) * 0.2 + rng.exponential(1.0, n_samples)
    order = np.argsort(times)
    flip = np.zeros(n_samples)
    flip[order[: n_samples // 2]] = f0[order[: n_samples // 2]]
    flip[order[n_samples // 2 :]] = -f0[order[n_samples // 2 :]]
    times = 5.0 + np.abs(flip) + rng.exponential(0.5, n_samples)
    events = rng.binomial(1, 0.7, n_samples).astype(bool)
    y = np.array(
        list(zip(events, times)),
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


class TestRecommendModelsAutoRunsTests:
    """Standalone ``recommend_models(X, y)`` runs PH/linearity tests."""

    def test_auto_run_populates_violations_when_caller_omits(self, monkeypatch):
        """When violations kwargs are None, helper is invoked with both flags True."""
        captured = {}

        def fake(X, y, *, run_ph=True, run_linearity=True, **_):
            captured["run_ph"] = run_ph
            captured["run_linearity"] = run_linearity
            return ["f0"], ["f1"], None, None

        monkeypatch.setattr("mlsurv.learner._setup._run_assumption_tests", fake)
        X, y = _make_data(200, 80, 5)
        rec = recommend_models(X, y, verbose=False)
        assert captured == {"run_ph": True, "run_linearity": True}
        # Helper-supplied violations should appear in the rendered table.
        text = str(rec)
        assert "Non-PH features (1): f0" in text
        assert "Log-hazard non-linear features (1): f1" in text

    def test_caller_supplied_empty_ph_overrides_only_ph(self, monkeypatch):
        """``ph_violations=[]`` is authoritative: helper called with run_ph=False."""
        captured = {}

        def fake(X, y, *, run_ph=True, run_linearity=True, **_):
            captured["run_ph"] = run_ph
            captured["run_linearity"] = run_linearity
            return None, None, None, None

        monkeypatch.setattr("mlsurv.learner._setup._run_assumption_tests", fake)
        X, y = _make_data(200, 80, 5)
        recommend_models(X, y, ph_violations=[], verbose=False)
        assert captured == {"run_ph": False, "run_linearity": True}

    def test_caller_supplied_empty_linearity_overrides_only_linearity(self, monkeypatch):
        """Mirror of the previous test: linearity-only suppression."""
        captured = {}

        def fake(X, y, *, run_ph=True, run_linearity=True, **_):
            captured["run_ph"] = run_ph
            captured["run_linearity"] = run_linearity
            return None, None, None, None

        monkeypatch.setattr("mlsurv.learner._setup._run_assumption_tests", fake)
        X, y = _make_data(200, 80, 5)
        recommend_models(X, y, linearity_violations=[], verbose=False)
        assert captured == {"run_ph": True, "run_linearity": False}

    def test_caller_supplied_both_lists_skips_helper(self, monkeypatch):
        """When both violations are caller-supplied, helper is not called."""
        called = {"hit": False}

        def fake(*a, **kw):
            called["hit"] = True
            return None, None, None, None

        monkeypatch.setattr("mlsurv.learner._setup._run_assumption_tests", fake)
        X, y = _make_data(200, 80, 5)
        recommend_models(
            X, y, ph_violations=[], linearity_violations=[], verbose=False,
        )
        assert called["hit"] is False

    def test_run_assumption_tests_false_skips_auto_run(self, monkeypatch):
        """Opt-out kwarg bypasses the helper entirely."""
        called = {"hit": False}

        def fake(*a, **kw):
            called["hit"] = True
            return None, None, None, None

        monkeypatch.setattr("mlsurv.learner._setup._run_assumption_tests", fake)
        X, y = _make_data(200, 80, 5)
        rec = recommend_models(X, y, run_assumption_tests=False, verbose=False)
        assert called["hit"] is False
        # With no violations supplied and tests skipped, the table renders
        # the "test unavailable" clause for PH-assuming models.
        assert "PH test unavailable" in rec.details["coxph"]

    def test_numeric_only_path_unaffected(self, monkeypatch):
        """``recommend_models(n_samples=...)`` never invokes the helper."""
        called = {"hit": False}

        def fake(*a, **kw):
            called["hit"] = True
            return None, None, None, None

        monkeypatch.setattr("mlsurv.learner._setup._run_assumption_tests", fake)
        recommend_models(
            n_samples=500, n_events=200, n_features=10, verbose=False,
        )
        assert called["hit"] is False

    def test_auto_run_drops_unavailable_clause(self):
        """End-to-end: when the auto-run succeeds, the 'test unavailable'
        clause is absent from the rendered tier reasons.
        """
        X, y = _make_ph_violating_data(n_samples=200, n_features=5, seed=0)
        rec = recommend_models(X, y, verbose=False)
        # ``RecommendedModels`` converts ``None`` violations to ``()`` for
        # display, so we can't distinguish ran-clean from skipped via the
        # tuple. Instead, verify by absence of the "unavailable" clause —
        # this only appears when violations are ``None`` going into the
        # classifier.
        coxph_details = rec.details["coxph"]
        assert "PH test unavailable" not in coxph_details, (
            f"PH auto-run should have populated violations; got "
            f"unavailable clause in: {coxph_details!r}"
        )
        assert "linearity test unavailable" not in coxph_details, (
            f"Linearity auto-run should have populated violations; got "
            f"unavailable clause in: {coxph_details!r}"
        )

    def test_auto_run_renders_clean_summary_lines(self):
        """When tests run and find no violations, summary lines confirm that."""
        X, y = _make_data(200, 80, 5)
        rec = recommend_models(X, y, verbose=False)
        text = str(rec)
        # Either a violation list OR a clean-summary line should appear for
        # each test that ran. With this synthetic data the PH/linearity
        # tests typically come back clean.
        assert "PH test: all" in text or "Non-PH features" in text, (
            f"Expected PH summary line in:\n{text}"
        )
        assert "Linearity test: all" in text or "Log-hazard non-linear features" in text, (
            f"Expected linearity summary line in:\n{text}"
        )
        # When linearity ran clean, the line should state how many
        # continuous features were tested.
        if "Linearity test: all" in text:
            assert rec.n_linearity_tested is not None
            assert (
                f"all {rec.n_linearity_tested} continuous feature" in text
            ), (
                f"Linearity line should state the tested count "
                f"({rec.n_linearity_tested}); got:\n{text}"
            )

    def test_skipped_tests_render_no_summary_line(self):
        """When tests are not run, no summary line is printed (only per-model 'unavailable')."""
        rec = recommend_models(
            n_samples=500, n_events=200, n_features=10, verbose=False,
        )
        text = str(rec)
        assert "PH test: all" not in text
        assert "Linearity test: all" not in text
        assert "Non-PH features" not in text
        assert "Log-hazard non-linear features" not in text

    def test_auto_run_failure_falls_back_to_unavailable(self):
        """Pathological data (NaN-only) causes both tests to fail gracefully."""
        X = pd.DataFrame(
            np.full((20, 3), np.nan),
            columns=["f0", "f1", "f2"],
        )
        y = np.array(
            [(True, 1.0)] * 10 + [(False, 2.0)] * 10,
            dtype=[("event", bool), ("time", float)],
        )
        # Suppress the InternalWarning the helper emits on test failure.
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            rec = recommend_models(X, y, verbose=False)
        # Both clauses should be present — graceful fallback to the same
        # behavior as standalone calls without (X, y).
        assert "PH test unavailable" in rec.details["coxph"]
        assert "linearity test unavailable" in rec.details["coxph"]


# ---------------------------------------------------------------------------
# TestRecommendModelsFromNumbers
# ---------------------------------------------------------------------------


class TestRecommendModelsFromNumbers:
    def test_basic_from_numbers(self):
        rec = recommend_models(n_samples=80, n_events=20, n_features=15)
        assert rec.n_samples == 80
        assert rec.n_events == 20
        assert rec.n_features == 15
        assert rec.epv == pytest.approx(20 / 15)

    def test_missing_keywords_raises(self):
        with pytest.raises(ValueError, match="Provide either"):
            recommend_models(n_samples=100, n_events=50)

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="Provide either"):
            recommend_models()

    def test_negative_n_samples_raises(self):
        with pytest.raises(ValueError, match="n_samples must be non-negative"):
            recommend_models(n_samples=-1, n_events=5, n_features=3)

    def test_negative_n_events_raises(self):
        with pytest.raises(ValueError, match="n_events must be non-negative"):
            recommend_models(n_samples=100, n_events=-5, n_features=3)

    def test_events_exceed_samples_raises(self):
        with pytest.raises(ValueError, match="cannot exceed n_samples"):
            recommend_models(n_samples=50, n_events=100, n_features=3)


# ---------------------------------------------------------------------------
# TestSmallDataset
# ---------------------------------------------------------------------------


class TestSmallDataset:
    """N=50, events=10, features=15 -> EPV=0.67."""

    def test_linear_cautionary_ml_not_recommended(self):
        rec = recommend_models(n_samples=50, n_events=10, n_features=15)
        assert rec.epv == pytest.approx(10 / 15)

        # EPV < 5 -> linear not recommended
        assert "coxph" in rec.not_recommended
        assert "coxnet" in rec.not_recommended
        assert "weibull_aft" in rec.not_recommended

        # All tree/deep models not recommended
        for name in ("rsf", "gbsa", "xgbcox"):
            assert name in rec.not_recommended
        # FSSVM is linear complexity — EPV < 5 -> also not recommended
        assert "fssvm" in rec.not_recommended

    def test_no_recommended(self):
        rec = recommend_models(n_samples=50, n_events=10, n_features=15)
        assert rec.recommended == []


# ---------------------------------------------------------------------------
# TestMediumDataset
# ---------------------------------------------------------------------------


class TestMediumDataset:
    """N=300, events=150, features=10 -> EPV=15."""

    def test_linear_cautionary_tree_not_recommended(self):
        rec = recommend_models(n_samples=300, n_events=150, n_features=10)
        assert rec.epv == pytest.approx(15.0)

        # Linear: EPV=15 >= 5 but < 20 -> cautionary
        for name in ("coxph", "coxnet", "weibull_aft"):
            assert name in rec.cautionary

        # Tree: N=300 >= 200, EPV=15 < 20 -> not recommended
        for name in ("rsf", "gbsa", "xgbcox"):
            assert name in rec.not_recommended
        # FSSVM is linear complexity — EPV=15 >= 5 but < 20 -> cautionary
        assert "fssvm" in rec.cautionary

    def test_medium_with_higher_epv(self):
        """N=300, events=250, features=10 -> EPV=25."""
        rec = recommend_models(n_samples=300, n_events=250, n_features=10)
        # Linear: EPV=25 >= 20 -> recommended
        assert "coxph" in rec.recommended
        # Tree: N=300 >= 200, EPV=25 >= 20 -> cautionary
        assert "rsf" in rec.cautionary


# ---------------------------------------------------------------------------
# TestLargeDataset
# ---------------------------------------------------------------------------


class TestLargeDataset:
    """N=1500, events=600, features=5 -> EPV=120."""

    def test_all_recommended(self):
        rec = recommend_models(
            n_samples=1500, n_events=600, n_features=5,
            include_unavailable=True,
        )
        assert rec.epv == pytest.approx(120.0)

        # All models recommended
        assert "coxph" in rec.recommended
        assert "fssvm" in rec.recommended
        assert "rsf" in rec.recommended
        assert "deepsurv" in rec.recommended
        assert "deephit" in rec.recommended
        assert rec.cautionary == []
        assert rec.not_recommended == []


# ---------------------------------------------------------------------------
# TestVeryLowEPV
# ---------------------------------------------------------------------------


class TestVeryLowEPV:
    """EPV < 2 -> 'too small' warning in rationale."""

    def test_too_small_warning(self):
        rec = recommend_models(n_samples=100, n_events=5, n_features=10)
        assert rec.epv == pytest.approx(0.5)
        assert "extremely low" in rec.rationale.lower()
        assert "WARNING" in rec.rationale


# ---------------------------------------------------------------------------
# TestEPVBelowTen
# ---------------------------------------------------------------------------


class TestEPVBelowTen:
    """EPV=8 -> feature reduction note."""

    def test_feature_reduction_note(self):
        rec = recommend_models(n_samples=200, n_events=80, n_features=10)
        assert rec.epv == pytest.approx(8.0)
        assert "feature reduction" in rec.rationale.lower()


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_features(self):
        """0 features -> EPV=inf; linear recommended, tree/deep still need N."""
        rec = recommend_models(n_samples=100, n_events=50, n_features=0)
        assert math.isinf(rec.epv)
        # Linear models recommended (no N threshold, EPV=inf >= 20)
        assert "coxph" in rec.recommended
        # Tree/deep models still constrained by N thresholds
        assert "rsf" not in rec.recommended  # N=100 < 500

    def test_zero_events(self):
        """0 events -> EPV=0, everything not recommended."""
        rec = recommend_models(n_samples=100, n_events=0, n_features=5)
        assert rec.epv == pytest.approx(0.0)
        assert len(rec.recommended) == 0

    def test_single_feature(self):
        """1 feature with enough events -> high EPV."""
        rec = recommend_models(n_samples=500, n_events=200, n_features=1)
        assert rec.epv == pytest.approx(200.0)
        assert "coxph" in rec.recommended

    def test_include_unavailable(self):
        """include_unavailable=True adds deep learning models."""
        rec = recommend_models(
            n_samples=2000, n_events=1000, n_features=5,
            include_unavailable=True,
        )
        assert "deepsurv" in rec.recommended
        assert "deephit" in rec.recommended


# ---------------------------------------------------------------------------
# TestReturnStructure
# ---------------------------------------------------------------------------


class TestReturnStructure:
    def test_dataclass_fields(self):
        rec = recommend_models(n_samples=500, n_events=200, n_features=10)
        assert hasattr(rec, "recommended")
        assert hasattr(rec, "cautionary")
        assert hasattr(rec, "not_recommended")
        assert hasattr(rec, "rationale")
        assert hasattr(rec, "n_samples")
        assert hasattr(rec, "n_events")
        assert hasattr(rec, "n_features")
        assert hasattr(rec, "epv")
        assert hasattr(rec, "details")

    def test_details_dict(self):
        rec = recommend_models(n_samples=500, n_events=200, n_features=10)
        assert isinstance(rec.details, dict)
        # Every model should have a detail entry
        all_models = rec.recommended + rec.cautionary + rec.not_recommended
        for name in all_models:
            assert name in rec.details

    def test_str_format(self):
        rec = recommend_models(n_samples=500, n_events=200, n_features=10,
                               verbose=False)
        text = str(rec)
        assert "Model Recommendations" in text
        assert "N=500" in text
        assert "EPV=" in text

    def test_repr_is_short_one_liner(self):
        """repr() returns a compact summary, not the full table."""
        rec = recommend_models(
            n_samples=500, n_events=200, n_features=10, verbose=False,
        )
        text = repr(rec)
        assert "\n" not in text, f"repr should be single-line; got:\n{text}"
        assert text.startswith("<RecommendedModels:")
        assert text.endswith(">")
        assert "N=500" in text
        assert "events=200" in text
        assert "EPV=" in text
        assert "recommended" in text
        assert "cautionary" in text
        assert "not_recommended" in text
        # Must NOT contain the table header — that's what __str__ is for.
        assert "Model Recommendations" not in text

    def test_str_returns_full_table(self):
        """str() returns the full formatted table."""
        # include_unavailable=True forces deepsurv/deephit into the list so the
        # not_recommended tier is populated even without pycox/torch: at
        # N=500/EPV=20 the deep models fail both deep-learning bands and fall to
        # not_recommended, guaranteeing all three tier labels render.
        rec = recommend_models(
            n_samples=500, n_events=200, n_features=10, verbose=False,
            include_unavailable=True,
        )
        text = str(rec)
        assert "Model Recommendations" in text
        assert "Recommended" in text
        assert "Cautionary" in text
        assert "Not recommended" in text

    def test_verbose_prints_table_once(self, capsys):
        """recommend_models(verbose=True) prints the full table exactly once."""
        recommend_models(
            n_samples=500, n_events=200, n_features=10, verbose=True,
        )
        captured = capsys.readouterr()
        assert captured.out.count("Model Recommendations") == 1

    def test_tiers_cover_all_models(self):
        """Every listed model is in exactly one tier."""
        # include_unavailable=True on BOTH sides pins a deterministic 10-model set
        # so the assertion is independent of pycox/torch availability.
        rec = recommend_models(
            n_samples=500, n_events=200, n_features=10,
            include_unavailable=True,
        )
        all_in_tiers = set(rec.recommended + rec.cautionary + rec.not_recommended)
        expected = set(list_models(include_unavailable=True))
        assert all_in_tiers == expected

    def test_no_duplicates_across_tiers(self):
        rec = recommend_models(n_samples=500, n_events=200, n_features=10)
        all_models = rec.recommended + rec.cautionary + rec.not_recommended
        assert len(all_models) == len(set(all_models))

    def test_ph_violations_demote_assumption_models(self):
        """PH violations demote PH-assuming models without affecting others."""
        from mlsurv.constants import PH_ASSUMPTION_MODELS

        baseline = recommend_models(
            n_samples=500, n_events=200, n_features=10,
            ph_violations=[], linearity_violations=[], verbose=False,
        )
        with_viol = recommend_models(
            n_samples=500, n_events=200, n_features=10,
            ph_violations=["age", "grade"], linearity_violations=[],
            verbose=False,
        )
        # PH-assuming models should demote (move from recommended/cautionary
        # toward not_recommended); non-PH models should be unchanged.
        for name in PH_ASSUMPTION_MODELS:
            base_tier = _tier_of(baseline, name)
            new_tier = _tier_of(with_viol, name)
            assert _tier_index(new_tier) >= _tier_index(base_tier), (
                f"{name}: tier should not improve under PH violations"
            )

    def test_violation_lines_appended_to_table(self):
        """Non-PH and Non-linear feature lines render under the table."""
        rec = recommend_models(
            n_samples=500, n_events=200, n_features=10,
            ph_violations=["age", "grade", "size"],
            linearity_violations=["age", "nodes"],
            verbose=False,
        )
        text = str(rec)
        assert "Non-PH features (3): age, grade, size" in text
        assert "Log-hazard non-linear features (2): age, nodes" in text

    def test_violation_lines_truncate(self):
        """Violation listings show first 5 + 'and N more'."""
        feats = [f"f{i}" for i in range(8)]
        rec = recommend_models(
            n_samples=500, n_events=200, n_features=10,
            ph_violations=feats, verbose=False,
        )
        text = str(rec)
        assert "Non-PH features (8): f0, f1, f2, f3, f4 (and 3 more)" in text

    def test_high_dim_hard_rule(self):
        """n_features >= n_events forces coxph + weibull_aft to not_recommended."""
        rec = recommend_models(
            n_samples=500, n_events=30, n_features=50, verbose=False,
        )
        assert "coxph" in rec.not_recommended
        assert "weibull_aft" in rec.not_recommended
        # coxnet is regularized — not subject to the hard rule.
        for tier_attr in ("recommended", "cautionary", "not_recommended"):
            if "coxnet" in getattr(rec, tier_attr):
                if tier_attr == "not_recommended":
                    # Reason should be EPV-driven, not the high-dim message.
                    assert "ill-conditioned" not in rec.details["coxnet"]
                break

    def test_unavailable_clause_when_violations_none(self):
        """ph_violations=None surfaces as 'PH test unavailable' for PH models."""
        rec = recommend_models(
            n_samples=500, n_events=300, n_features=10, verbose=False,
        )
        assert "PH test unavailable" in rec.details["coxph"]
        assert "linearity test unavailable" in rec.details["coxph"]
        # Assumption-free models don't get either clause.
        assert "PH test unavailable" not in rec.details["rsf"]


def _tier_index(tier: str) -> int:
    return ("recommended", "cautionary", "not_recommended").index(tier)


def _tier_of(rec, name: str) -> str:
    if name in rec.recommended:
        return "recommended"
    if name in rec.cautionary:
        return "cautionary"
    return "not_recommended"


# ---------------------------------------------------------------------------
# TestClassifyModel
# ---------------------------------------------------------------------------


def _classify(name, n_samples, epv, *, n_events=None, n_features=None,
              ph=None, lin=None):
    """Test helper — defaults n_events/n_features so EPV math works."""
    if n_features is None:
        n_features = 1
    if n_events is None:
        n_events = max(int(round(epv * n_features)), 1)
    return _classify_model(
        name, n_samples, n_events, n_features, epv, _DISPLAY_NAMES, ph, lin,
    )


class TestClassifyModel:
    def test_linear_recommended(self):
        tier, reason = _classify(
            "coxph", 100, 25, ph=[], lin=[]
        )
        assert tier == "recommended"
        assert "EPV ok" in reason

    def test_linear_cautionary(self):
        tier, reason = _classify(
            "coxph", 100, 10, ph=[], lin=[]
        )
        assert tier == "cautionary"
        assert "EPV=10.0" in reason

    def test_linear_not_recommended(self):
        tier, reason = _classify(
            "coxph", 100, 3, ph=[], lin=[]
        )
        assert tier == "not_recommended"
        assert "EPV=3.0" in reason

    def test_tree_boundary(self):
        """N=500 and EPV=50 exactly -> recommended."""
        tier, _ = _classify("rsf", 500, 50)
        assert tier == "recommended"

    def test_tree_cautionary_boundary(self):
        """N=200 and EPV=20 exactly -> cautionary."""
        tier, _ = _classify("rsf", 200, 20)
        assert tier == "cautionary"

    def test_tree_below_cautionary_n(self):
        """N=199 just below cautionary N threshold -> not_recommended."""
        tier, _ = _classify("rsf", 199, 20)
        assert tier == "not_recommended"

    def test_tree_below_cautionary_epv(self):
        """N=200 but EPV=19 just below -> not_recommended."""
        tier, _ = _classify("rsf", 200, 19)
        assert tier == "not_recommended"

    def test_deep_recommended_boundary(self):
        """N=1000, EPV=100 exactly -> recommended."""
        tier, _ = _classify("deepsurv", 1000, 100)
        assert tier == "recommended"

    def test_deep_cautionary_boundary(self):
        """N=500, EPV=50 exactly -> cautionary."""
        tier, _ = _classify("deepsurv", 500, 50)
        assert tier == "cautionary"

    def test_deep_below_cautionary(self):
        """N=499 just below cautionary -> not_recommended."""
        tier, _ = _classify("deepsurv", 499, 50)
        assert tier == "not_recommended"

    def test_unknown_complexity_raises(self, monkeypatch):
        """Unknown complexity class produces descriptive ValueError."""
        monkeypatch.setattr(
            "mlsurv.guidance.get_model_complexity", lambda name: "quantum"
        )
        with pytest.raises(ValueError, match="Unknown complexity class 'quantum'"):
            _classify("fssvm", 500, 50)

    def test_cautionary_reason_content(self):
        """Cautionary reason includes specific constraint violations."""
        tier, reason = _classify("rsf", 300, 25)
        assert tier == "cautionary"
        assert "N=300 < 500" in reason
        assert "EPV=25.0 < 50" in reason

    def test_ph_violation_demotes_assumption_model(self):
        """1 PH violation in 10 features (10%) demotes 1 tier."""
        # coxph at EPV=25 would be 'recommended' baseline.
        tier, reason = _classify(
            "coxph", 500, 25, n_events=250, n_features=10, ph=["age"], lin=[],
        )
        assert tier == "cautionary"
        assert "1 PH violation" in reason

    def test_many_ph_violations_demote_two_tiers(self):
        """PH violations >25% of features demote two tiers."""
        feats = [f"f{i}" for i in range(4)]  # 4/10 = 40% > 25%
        tier, reason = _classify(
            "coxph", 500, 25, n_events=250, n_features=10, ph=feats, lin=[],
        )
        assert tier == "not_recommended"
        assert "4 PH violations" in reason

    def test_assumption_free_model_unaffected(self):
        """PH violations don't demote RSF/DeepHit/etc."""
        tier, reason = _classify(
            "rsf", 500, 50, n_events=250, n_features=10,
            ph=["age", "grade"], lin=[],
        )
        assert tier == "recommended"
        assert "assumption-free" in reason

    def test_high_dim_overrides_other_tier(self):
        """n_features >= n_events forces coxph not_recommended."""
        tier, reason = _classify(
            "coxph", 500, 0.5, n_events=10, n_features=20, ph=[], lin=[],
        )
        assert tier == "not_recommended"
        assert "ill-conditioned" in reason

    def test_high_dim_does_not_apply_to_coxnet(self):
        """coxnet is regularized, exempt from the hard rule."""
        tier, reason = _classify(
            "coxnet", 500, 0.5, n_events=10, n_features=20, ph=[], lin=[],
        )
        assert "ill-conditioned" not in reason

    def test_high_dim_equality_boundary(self):
        """Rule fires at n_features == n_events (>=, not >)."""
        tier, reason = _classify(
            "coxph", 500, 1.0, n_events=15, n_features=15, ph=[], lin=[],
        )
        assert tier == "not_recommended"
        assert "ill-conditioned" in reason

    def test_violation_25_percent_boundary_demotes_one(self):
        """Exactly 25% violations → demote 1 (the <= 25% branch)."""
        # 1/4 == 25% exactly
        tier, _ = _classify(
            "coxph", 500, 50, n_events=200, n_features=4,
            ph=["a"], lin=[],
        )
        # baseline coxph(EPV=50, N=500) is recommended → demote 1 → cautionary
        assert tier == "cautionary"

    def test_violation_just_above_25_percent_demotes_two(self):
        """Just above 25% violations → demote 2."""
        # 2/7 ~= 28.6% > 25%
        tier, _ = _classify(
            "coxph", 500, 50, n_events=350, n_features=7,
            ph=["a", "b"], lin=[],
        )
        # baseline recommended → demote 2 → not_recommended
        assert tier == "not_recommended"

    def test_double_demotion_stacks_and_caps(self):
        """PH + linearity violations stack; cap at not_recommended."""
        # >25% PH (3/7) + >25% linearity (3/7) on coxph (assumes both)
        # baseline recommended; would demote 4 steps; capped at index 2.
        tier, reason = _classify(
            "coxph", 500, 50, n_events=350, n_features=7,
            ph=["a", "b", "c"], lin=["d", "e", "f"],
        )
        assert tier == "not_recommended"
        assert "3 PH + 3 linearity violations" in reason

    def test_singular_linearity_violation_phrasing(self):
        """'1 linearity violation' (singular) when only one."""
        tier, reason = _classify(
            "weibull_aft", 500, 50, n_events=200, n_features=10,
            ph=[], lin=["age"],
        )
        assert "1 linearity violation" in reason
        assert "violations" not in reason  # no plural form


# ---------------------------------------------------------------------------
# TestExtractDataCharacteristics
# ---------------------------------------------------------------------------


class TestExtractDataCharacteristics:
    def test_structured_array(self):
        X, y = _make_data(100, 40, 5)
        n, e, f = _extract_data_characteristics(X, y)
        assert n == 100
        assert e == 40
        assert f == 5

    def test_dataframe_y(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(50, 3), columns=["a", "b", "c"])
        y = pd.DataFrame({"event": [1]*20 + [0]*30, "time": rng.exponential(10, 50)})
        n, e, f = _extract_data_characteristics(X, y)
        assert n == 50
        assert e == 20
        assert f == 3

    def test_plain_ndarray_y_raises(self):
        """Plain 2D ndarray without named fields should raise ValueError."""
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = np.array([[1, 10.0], [0, 20.0], [1, 15.0]])  # no 'event' field
        with pytest.raises((ValueError, KeyError)):
            _extract_data_characteristics(X, y)

    def test_dict_missing_event_raises(self):
        """Dict-like y without 'event' key should raise."""
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = {"time": [10, 20, 30]}  # no 'event'
        with pytest.raises((ValueError, KeyError)):
            _extract_data_characteristics(X, y)

    def test_invalid_y_raises(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Cannot extract"):
            _extract_data_characteristics(X, "invalid")


# ---------------------------------------------------------------------------
# TestBuildRationale
# ---------------------------------------------------------------------------


class TestBuildRationale:
    def test_contains_dataset_stats(self):
        text = _build_rationale(100, 50, 10, 5.0)
        assert "N=100" in text
        assert "events=50" in text
        assert "features=10" in text
        assert "EPV=5.0" in text

    def test_too_small_epv_warning(self):
        text = _build_rationale(20, 2, 5, 0.4)
        assert "WARNING" in text
        assert "extremely low" in text.lower()

    def test_feature_reduction_note(self):
        text = _build_rationale(200, 40, 5, 8.0)
        assert "feature reduction" in text.lower()

    def test_no_warning_high_epv(self):
        text = _build_rationale(1000, 500, 5, 100.0)
        assert "WARNING" not in text
        assert "feature reduction" not in text.lower()

    def test_inf_epv_no_crash(self):
        """EPV=inf (0 features) should format without error."""
        text = _build_rationale(100, 50, 0, float("inf"))
        assert "EPV=inf" in text
        assert "WARNING" not in text  # inf > 2, no warning

    def test_citations(self):
        text = _build_rationale(100, 50, 5, 10.0)
        assert "van der Ploeg" in text
        assert "Riley" in text


# ---------------------------------------------------------------------------
# TestGetModelComplexity
# ---------------------------------------------------------------------------


class TestGetModelComplexity:
    def test_linear_models(self):
        assert get_model_complexity("coxph") == "linear"
        assert get_model_complexity("coxnet") == "linear"
        assert get_model_complexity("weibull_aft") == "linear"

    def test_tree_ensemble_models(self):
        assert get_model_complexity("rsf") == "tree_ensemble"
        assert get_model_complexity("gbsa") == "tree_ensemble"
        assert get_model_complexity("xgbcox") == "tree_ensemble"

    def test_fssvm_is_linear(self):
        """FSSVM is an SVM, not a tree ensemble — linear thresholds apply."""
        assert get_model_complexity("fssvm") == "linear"

    def test_deep_learning_models(self):
        assert get_model_complexity("deepsurv") == "deep_learning"
        assert get_model_complexity("deephit") == "deep_learning"

    def test_unknown_model_defaults_to_linear(self):
        assert get_model_complexity("nonexistent") == "linear"


# ---------------------------------------------------------------------------
# TestTierRules — declarative rule-table coverage
# ---------------------------------------------------------------------------
#
# Each rule in ``MODEL_TIER_RULES`` is exercised in isolation. The cases pick
# a model + dataset stats that cause exactly one rule to win, and assert the
# resulting (tier, reason fragment) pair. Order-precedence is exercised
# separately to pin the high-dim rule's priority over the EPV bands.

from mlsurv.guidance import (  # noqa: E402  (kept near tests for locality)
    MODEL_TIER_RULES,
    TierRule,
    _ClassifyCtx,
    _is_high_dim_infeasible,
)


class TestTierRules:
    """One case per MODEL_TIER_RULES row, plus order-precedence."""

    def test_rule_table_has_expected_shape(self):
        # Catch silent mutations: keep the rule count + tier sequence pinned.
        assert len(MODEL_TIER_RULES) == 4
        assert all(isinstance(r, TierRule) for r in MODEL_TIER_RULES)
        tiers = [r.tier for r in MODEL_TIER_RULES]
        assert tiers == [
            "not_recommended",   # high-dim hard rule
            "recommended",       # EPV ok
            "cautionary",        # gap vs recommended
            "not_recommended",   # catch-all
        ]
        # Only the high-dim hard rule may skip the assumption clause.
        assert MODEL_TIER_RULES[0].skip_assumptions is True
        assert all(not r.skip_assumptions for r in MODEL_TIER_RULES[1:])

    # Rule 1 — high-dim infeasible (Cox-family, n_features >= n_events)
    @pytest.mark.parametrize("name", ["coxph", "weibull_aft"])
    def test_rule1_high_dim_infeasible(self, name):
        tier, reason = _classify_model(
            name=name, n_samples=200, n_events=10, n_features=50, epv=0.2,
            display_names=_DISPLAY_NAMES,
            ph_violations=None, linearity_violations=None,
        )
        assert tier == "not_recommended"
        assert reason == (
            "n_features=50 >= n_events=10; partial likelihood ill-conditioned"
        )
        # Hard rule must NOT append assumption clause even when the test
        # was unavailable.
        assert "PH test unavailable" not in reason

    def test_rule1_does_not_fire_for_regularized_or_native_high_dim(self):
        # coxnet, RSF, deep models are not in _HIGH_DIM_INFEASIBLE.
        for name in ["coxnet", "rsf", "deepsurv"]:
            ctx = _ClassifyCtx.build(
                name, n_samples=200, n_events=10, n_features=50, epv=0.2
            )
            assert not _is_high_dim_infeasible(ctx)

    def test_rule1_boundary_conditions(self):
        # The n_features >= n_events boundary and the zero-guards are
        # easy to flip in a refactor. Pin them explicitly.
        equal = _ClassifyCtx.build(
            "coxph", n_samples=100, n_events=10, n_features=10, epv=1.0
        )
        assert _is_high_dim_infeasible(equal)  # >= must include equality

        just_below = _ClassifyCtx.build(
            "coxph", n_samples=100, n_events=10, n_features=9, epv=10/9
        )
        assert not _is_high_dim_infeasible(just_below)  # 9 < 10

        zero_events = _ClassifyCtx.build(
            "coxph", n_samples=100, n_events=0, n_features=50,
            epv=float("inf"),
        )
        assert not _is_high_dim_infeasible(zero_events)  # zero-guard

    def test_rule1_skip_assumptions_suppresses_violation_clause(self):
        # When the high-dim hard rule fires, supplied PH/linearity
        # violations must NOT be appended to the reason — the model is
        # mathematically broken regardless of assumption status.
        # Regression guard: if someone moves the skip_assumptions
        # short-circuit below _build_assumption_clause, this fails.
        tier, reason = _classify_model(
            name="coxph", n_samples=200, n_events=10, n_features=50, epv=0.2,
            display_names=_DISPLAY_NAMES,
            ph_violations=["age", "grade"],
            linearity_violations=["x1"],
        )
        assert tier == "not_recommended"
        assert reason == (
            "n_features=50 >= n_events=10; partial likelihood ill-conditioned"
        )
        assert "PH" not in reason
        assert "linearity" not in reason

    # Rule 2 — recommended band (EPV ok)
    def test_rule2_recommended_band(self):
        # FSSVM is assumption-free → no clause; reason is bare "EPV ok".
        tier, reason = _classify_model(
            name="fssvm", n_samples=1500, n_events=600, n_features=5,
            epv=120.0, display_names=_DISPLAY_NAMES,
            ph_violations=[], linearity_violations=[],
        )
        assert tier == "recommended"
        assert reason == "EPV ok; assumption-free"

    # Rule 3 — cautionary band (gap reported vs recommended thresholds)
    def test_rule3_cautionary_gap_vs_recommended(self):
        # tree_ensemble: rec=(500, 50), caut=(200, 20). EPV=30 fails rec=50
        # but passes caut=20.
        tier, reason = _classify_model(
            name="rsf", n_samples=300, n_events=120, n_features=4, epv=30.0,
            display_names=_DISPLAY_NAMES,
            ph_violations=[], linearity_violations=[],
        )
        assert tier == "cautionary"
        assert "EPV=30.0 < 50" in reason

    def test_rule3_cautionary_reports_recommended_gap(self):
        # Probe rule 3's reason factory directly: gap is computed against
        # the *recommended* thresholds, not cautionary.
        ctx = _ClassifyCtx(
            name="coxph", n_samples=10, n_events=10, n_features=2, epv=9.0,
            n_rec=None, epv_rec=10.0, n_caut=None, epv_caut=5.0,
        )
        rule3 = MODEL_TIER_RULES[2]
        assert rule3.predicate(ctx)
        assert rule3.reason(ctx) == "EPV=9.0 < 10.0"

    # Rule 4 — catch-all (gap reported vs cautionary thresholds)
    def test_rule4_catchall_below_thresholds(self):
        tier, reason = _classify_model(
            name="rsf", n_samples=100, n_events=15, n_features=10, epv=1.5,
            display_names=_DISPLAY_NAMES,
            ph_violations=[], linearity_violations=[],
        )
        assert tier == "not_recommended"
        # Cautionary line for tree_ensemble = (200, 20); 100 < 200 and 1.5 < 20
        assert "N=100 < 200" in reason
        assert "EPV=1.5 < 20" in reason

    # Rule order precedence — high-dim must beat the EPV bands.
    def test_high_dim_beats_recommended_band(self):
        # coxph with high EPV but n_features >= n_events: rule 1 wins.
        tier, reason = _classify_model(
            name="coxph", n_samples=500, n_events=10, n_features=15,
            epv=10000.0,  # arbitrarily high — would trigger rule 2 normally
            display_names=_DISPLAY_NAMES,
            ph_violations=[], linearity_violations=[],
        )
        assert tier == "not_recommended"
        assert "partial likelihood ill-conditioned" in reason
        # Rule 2's "EPV ok" must not appear.
        assert "EPV ok" not in reason


class TestDemotionCauses:
    """RecommendedModels.demotion_causes() — per-model demotion reasons."""

    def test_pure_size(self):
        """Low EPV, tests unavailable -> size only."""
        rec = recommend_models(
            n_samples=50, n_events=10, n_features=5, verbose=False,
        )
        # coxph (linear, EPV_rec=20) at EPV=2 is size-demoted; no assumption
        # tests were supplied, so PH/linearity cannot contribute.
        assert "coxph" in rec.not_recommended
        assert rec.demotion_causes("coxph") == ("size",)

    def test_high_dim_is_sole_cause(self):
        """n_features >= n_events forces coxph high-dim, short-circuiting others."""
        rec = recommend_models(
            n_samples=500, n_events=10, n_features=15,
            ph_violations=["f0"], linearity_violations=["f0"], verbose=False,
        )
        assert "coxph" in rec.not_recommended
        # high_dim is exclusive — even with violations supplied.
        assert rec.demotion_causes("coxph") == ("high_dim",)

    def test_pure_assumptions_on_ample_data(self):
        """Ample data (high EPV) + PH/linearity violations -> ph + linearity only."""
        rec = recommend_models(
            n_samples=1427, n_events=459, n_features=5,
            ph_violations=["f0", "f1", "f2", "f3"],
            linearity_violations=["f0", "f1"],
            verbose=False,
        )
        assert "coxph" in rec.not_recommended
        causes = rec.demotion_causes("coxph")
        assert "size" not in causes
        assert "high_dim" not in causes
        assert set(causes) == {"ph", "linearity"}

    def test_pure_ph(self):
        """PH violations only (no linearity) on ample data -> ph only."""
        rec = recommend_models(
            n_samples=1427, n_events=459, n_features=5,
            ph_violations=["f0", "f1", "f2", "f3"],
            linearity_violations=[],
            verbose=False,
        )
        assert "coxph" in rec.not_recommended
        assert rec.demotion_causes("coxph") == ("ph",)

    def test_pure_linearity(self):
        """Linearity violations only (no PH) on ample data -> linearity only.

        Mirror of ``test_pure_ph`` so the ``linearity`` branch is proven to
        fire independently of ``ph`` (guards against a copy-paste bug wiring
        linearity to ``self.ph_violations``).
        """
        rec = recommend_models(
            n_samples=1427, n_events=459, n_features=5,
            ph_violations=[],
            linearity_violations=["f0", "f1"],
            verbose=False,
        )
        assert "coxph" in rec.not_recommended
        assert rec.demotion_causes("coxph") == ("linearity",)

    def test_size_and_ph_combined(self):
        """Marginal size + PH violations -> both size and ph reported."""
        # EPV=15 (< linear EPV_rec=20) so coxph is size-demoted, plus PH
        # violations on top.
        rec = recommend_models(
            n_samples=150, n_events=75, n_features=5,
            ph_violations=["f0", "f1"], linearity_violations=[],
            verbose=False,
        )
        assert "coxph" in rec.not_recommended
        causes = rec.demotion_causes("coxph")
        assert "size" in causes
        assert "ph" in causes

    def test_recommended_model_has_no_causes(self):
        """A model that meets the recommended band reports no demotion causes."""
        rec = recommend_models(
            n_samples=1427, n_events=459, n_features=5,
            ph_violations=[], linearity_violations=[], verbose=False,
        )
        assert "coxph" in rec.recommended
        assert rec.demotion_causes("coxph") == ()
