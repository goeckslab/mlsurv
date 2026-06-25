"""Tests for IPCW truncation messaging consolidation.

The truncation surface is meant to fire once per cohort at the top of
evaluate/bootstrap/validate, not per metric. These tests cover the helper,
the EvaluationResult diagnostic fields, the auto_eval_times _quiet kwarg,
and the dedup contract end-to-end.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import MethodologicalWarning
from mlsurv.learner._evaluation import _emit_ipcw_truncation_warning
from mlsurv.learner._state import LearnerState
from mlsurv.results.containers import EvaluationResult
from mlsurv.utils import auto_eval_times

from tests._helpers import make_y


# ---------------------------------------------------------------------------
# Helper: _emit_ipcw_truncation_warning
# ---------------------------------------------------------------------------


class TestEmitIpcwTruncationWarning:
    def test_no_warning_when_no_truncation(self):
        state = LearnerState()
        y_train = make_y([1, 2, 3, 4, 5])
        y_test = make_y([1, 2, 3])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t_max, n_beyond = _emit_ipcw_truncation_warning(
                state, "main", y_train, y_test,
            )
        assert t_max is None and n_beyond == 0
        assert not any(
            issubclass(w.category, MethodologicalWarning) for w in caught
        )
        assert "main" not in state.ipcw_truncation_warned

    def test_warns_once_per_cohort(self):
        state = LearnerState()
        y_train = make_y([1, 2, 3])
        y_test = make_y([1, 2, 5, 6])  # 2 obs beyond t_max=3
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t_max1, n1 = _emit_ipcw_truncation_warning(
                state, "main", y_train, y_test,
            )
            t_max2, n2 = _emit_ipcw_truncation_warning(
                state, "main", y_train, y_test,
            )
        assert (t_max1, n1) == (3.0, 2)
        assert (t_max2, n2) == (3.0, 2)
        meth = [w for w in caught if issubclass(w.category, MethodologicalWarning)]
        assert len(meth) == 1
        assert "Uno C, AUC(t), IBS" in str(meth[0].message)
        assert "training-range-cap" in str(meth[0].message)
        assert state.ipcw_truncation_warned["main"] is True

    def test_separate_warnings_per_cohort(self):
        state = LearnerState()
        y_train = make_y([1, 2, 3])
        y_test = make_y([1, 5, 6])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_ipcw_truncation_warning(state, "main", y_train, y_test)
            _emit_ipcw_truncation_warning(state, "TCGA", y_train, y_test)
            _emit_ipcw_truncation_warning(state, "TCGA", y_train, y_test)
        meth = [w for w in caught if issubclass(w.category, MethodologicalWarning)]
        assert len(meth) == 2
        cohort_names = sorted(
            n for n in ("main", "TCGA")
            if any(f"cohort {n!r}" in str(m.message) for m in meth)
        )
        assert cohort_names == ["TCGA", "main"]

    def test_returns_zero_when_y_train_is_none(self):
        state = LearnerState()
        y_test = make_y([1, 2, 3])
        t_max, n = _emit_ipcw_truncation_warning(state, "main", None, y_test)
        assert t_max is None and n == 0

    def test_warning_includes_tau_when_provided(self):
        """When a resolved tau is threaded in, the
        consolidated message must mention it so we can drop the
        separate ``_resolve_ipcw_tau`` warning for trivial overhangs.
        """
        state = LearnerState()
        y_train = make_y([1, 2, 3])
        y_test = make_y([1, 2, 5])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_ipcw_truncation_warning(
                state, "main", y_train, y_test, tau=2.4,
            )
        meth = [w for w in caught if issubclass(w.category, MethodologicalWarning)]
        assert len(meth) == 1
        assert "Using tau=2.4 for IPCW" in str(meth[0].message)

    def test_warning_omits_tau_clause_when_none(self):
        state = LearnerState()
        y_train = make_y([1, 2, 3])
        y_test = make_y([1, 2, 5])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_ipcw_truncation_warning(
                state, "main", y_train, y_test, tau=None,
            )
        meth = [w for w in caught if issubclass(w.category, MethodologicalWarning)]
        assert len(meth) == 1
        assert "Using tau=" not in str(meth[0].message)


# ---------------------------------------------------------------------------
# auto_eval_times: _quiet kwarg
# ---------------------------------------------------------------------------


class TestAutoEvalTimesQuiet:
    def _build(self):
        rng = np.random.default_rng(0)
        y_train = make_y(rng.uniform(1, 10, 50))
        y_test = make_y(rng.uniform(1, 30, 50))  # extends past 1.5 * 10
        return y_train, y_test

    def test_default_emits_divergence_warning(self):
        y_train, y_test = self._build()
        with pytest.warns(MethodologicalWarning, match="eval_times derived"):
            auto_eval_times(y_test, y_train=y_train)

    def test_quiet_suppresses_divergence_warning(self):
        y_train, y_test = self._build()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            auto_eval_times(y_test, y_train=y_train, _quiet=True)
        assert not any(
            issubclass(w.category, MethodologicalWarning)
            and "eval_times derived" in str(w.message)
            for w in caught
        )


# ---------------------------------------------------------------------------
# EvaluationResult: truncation fields default to None/0
# ---------------------------------------------------------------------------


class TestEvaluationResultTruncationFields:
    def test_default_fields(self):
        empty = pd.DataFrame()
        r = EvaluationResult(
            concordance=empty, test_set=empty, auc_t=empty,
            importance=empty, importance_iterations=empty,
        )
        assert r.truncation_t_max is None
        assert r.truncation_n_beyond == 0


# ---------------------------------------------------------------------------
# Integration: dedup across evaluate() calls; cohort-separation via validate()
# ---------------------------------------------------------------------------


def _split_with_long_test(seed=42):
    """Build presplit data where y_test extends past y_train's max."""
    rng = np.random.default_rng(seed)
    n_train, n_test = 80, 30
    n_features = 4

    X = rng.normal(size=(n_train + n_test, n_features))
    feature_cols = [f"f{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_cols)

    train_times = rng.uniform(1, 10, n_train)
    train_events = rng.binomial(1, 0.7, n_train).astype(bool)
    # Force 5 test samples with times >> y_train.max
    test_times = np.concatenate([
        rng.uniform(1, 10, n_test - 5),
        rng.uniform(20, 40, 5),
    ])
    test_events = rng.binomial(1, 0.7, n_test).astype(bool)

    y_train = make_y(train_times, train_events)
    y_test = make_y(test_times, test_events)

    return (
        X_df.iloc[:n_train].reset_index(drop=True),
        X_df.iloc[n_train:].reset_index(drop=True),
        y_train, y_test,
    )


@pytest.mark.integration
class TestIpcwWarningEndToEnd:
    @pytest.fixture
    def trained_learner(self):
        X_train, X_test, y_train, y_test = _split_with_long_test()
        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
        )
        learner.setup(feature_selector=None, random_state=42)
        learner.train("coxph")
        return learner

    def _count_truncation_warnings(self, caught):
        return [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "IPCW-weighted metrics" in str(w.message)
        ]

    def test_dedup_within_and_across_evaluate(self, trained_learner):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trained_learner.evaluate("coxph", verbose=False)
            trained_learner.evaluate("coxph", verbose=False)
        warns = self._count_truncation_warnings(caught)
        assert len(warns) == 1
        assert "cohort 'main'" in str(warns[0].message)

    def test_evaluation_result_truncation_fields_populated(
        self, trained_learner,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = trained_learner.evaluate("coxph", verbose=False)
        assert result.truncation_t_max is not None
        assert result.truncation_n_beyond > 0

    def test_validate_uses_separate_cohort_name(self, trained_learner):
        X_val = trained_learner.X_test.copy()
        y_val = trained_learner.y_test.copy()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trained_learner.evaluate("coxph", verbose=False)
            trained_learner.validate(
                X_val, y_val, models="coxph", name="ExternalA",
                bootstrap=False, verbose=False,
            )
            # Second call on same cohort must dedup.
            trained_learner.validate(
                X_val, y_val, models="coxph", name="ExternalA",
                bootstrap=False, verbose=False,
            )
        warns = self._count_truncation_warnings(caught)
        assert len(warns) == 2
        cohorts = [str(w.message) for w in warns]
        assert any("cohort 'main'" in c for c in cohorts)
        assert any("cohort 'ExternalA'" in c for c in cohorts)

    def test_bootstrap_dedups_with_prior_evaluate(self, trained_learner):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trained_learner.evaluate("coxph", verbose=False)
            result = trained_learner.bootstrap(
                "coxph", n_bootstrap=2, n_importance_repeats=0,
                show_progress=False, verbose=False,
            )
        warns = self._count_truncation_warnings(caught)
        # evaluate() emits once; bootstrap() shares cohort_name="main" and dedups.
        assert len(warns) == 1
        assert "cohort 'main'" in str(warns[0].message)
        # BootstrapResult carries the persisted diagnostic fields.
        assert getattr(result, 'truncation_t_max', None) is not None
        assert getattr(result, 'truncation_n_beyond', 0) > 0

    def test_evaluate_emits_only_consolidated_warning(self, trained_learner):
        """Regression guard: a single evaluate() must not surface the
        legacy per-metric MethodologicalWarning from
        truncate_y_to_train_range. Step 6 of the plan flips warn=False at
        the four pipeline call sites; if any caller regresses to warn=True,
        we'd see the legacy phrase 'Censored {n} y_test observation' in
        the captured warnings.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trained_learner.evaluate("coxph", verbose=False)
        meth = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
        ]
        legacy_phrase = "Censored "
        assert not any(
            legacy_phrase in str(w.message)
            and "administratively censored at t_max" in str(w.message)
            for w in meth
        ), (
            "Legacy per-metric truncation warning leaked through; "
            "the consolidated top-level warning should be the only one."
        )

    def test_multi_model_per_model_truncation_fields(self, trained_learner):
        """Footnote renders for multi-model evaluates: extract_model_evaluation
        and EvaluationResult.combine must forward the truncation fields onto
        per-model evaluations and the recombined view.
        """
        trained_learner.train("coxnet")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained_learner.evaluate(verbose=False)
        for name, mr in trained_learner._state.models.items():
            assert mr.evaluation is not None, name
            assert mr.evaluation.truncation_t_max is not None, (
                f"per-model truncation_t_max not propagated for {name!r}"
            )
            assert mr.evaluation.truncation_n_beyond > 0, name
        combined = trained_learner.evaluation_result
        assert combined.truncation_t_max is not None
        assert combined.truncation_n_beyond > 0

    def test_no_warning_when_no_truncation(self):
        rng = np.random.default_rng(0)
        n_train, n_test = 60, 20
        X = pd.DataFrame(
            rng.normal(size=(n_train + n_test, 4)),
            columns=[f"f{i}" for i in range(4)],
        )
        train_times = rng.uniform(1, 50, n_train)
        test_times = rng.uniform(1, 10, n_test)  # all within train range
        y_train = make_y(train_times, rng.binomial(1, 0.7, n_train).astype(bool))
        y_test = make_y(test_times, rng.binomial(1, 0.7, n_test).astype(bool))
        learner = SurvivalLearner(
            X_train=X.iloc[:n_train].reset_index(drop=True),
            X_test=X.iloc[n_train:].reset_index(drop=True),
            y_train=y_train, y_test=y_test,
        )
        learner.setup(feature_selector=None, random_state=42)
        learner.train("coxph")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = learner.evaluate("coxph", verbose=False)
        warns = self._count_truncation_warnings(caught)
        assert warns == []
        assert result.truncation_t_max is None
        assert result.truncation_n_beyond == 0


# ---------------------------------------------------------------------------
# Trivial-overhang single-warning consolidation
# ---------------------------------------------------------------------------


def _split_with_one_overhang(seed=42, n_test=120):
    """Build presplit data with exactly 1 test obs beyond y_train.max
    (≤1% of test cohort, so the stability warning should be suppressed).
    """
    rng = np.random.default_rng(seed)
    n_train = 80
    n_features = 4
    X = rng.normal(size=(n_train + n_test, n_features))
    cols = [f"f{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=cols)
    train_times = rng.uniform(1.0, 10.0, n_train)
    train_events = rng.binomial(1, 0.7, n_train).astype(bool)
    test_times = np.concatenate([
        rng.uniform(1.0, 9.5, n_test - 1),
        np.array([train_times.max() * 1.5]),
    ])
    test_events = rng.binomial(1, 0.7, n_test).astype(bool)
    y_train = make_y(train_times, train_events)
    y_test = make_y(test_times, test_events)
    return (
        X_df.iloc[:n_train].reset_index(drop=True),
        X_df.iloc[n_train:].reset_index(drop=True),
        y_train, y_test,
    )


@pytest.mark.integration
class TestConsolidatedTruncationWarning:
    """When n_beyond ≤ 1% of test cohort and there's no real instability
    (min_g ≥ 0.05), `evaluate()` should emit ONE consolidated
    MethodologicalWarning that mentions both truncation and the resolved
    tau — not two near-identical warnings.
    """

    def _meth_warnings(self, caught):
        return [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
        ]

    def test_trivial_overhang_emits_single_warning(self):
        X_train, X_test, y_train, y_test = _split_with_one_overhang()
        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
        )
        learner.setup(feature_selector=None, random_state=42)
        learner.train("coxph")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner.evaluate("coxph", verbose=False)
        meth = self._meth_warnings(caught)
        ipcw_msgs = [str(w.message) for w in meth if "IPCW" in str(w.message)]
        assert len(ipcw_msgs) == 1, (
            f"Expected one consolidated IPCW warning for trivial overhang, "
            f"got {len(ipcw_msgs)}:\n" + "\n---\n".join(ipcw_msgs)
        )
        msg = ipcw_msgs[0]
        assert "IPCW-weighted metrics" in msg
        assert "Using tau=" in msg
        assert "may be unstable" not in msg

    def test_bootstrap_trivial_overhang_threads_tau(self):
        """The bootstrap path was reordered alongside evaluate — verify a
        trivial-overhang bootstrap also yields ONE consolidated warning
        with the tau clause. Guards against a regression that drops
        ``tau=resolved_ipcw_tau`` from the bootstrap call site.
        """
        X_train, X_test, y_train, y_test = _split_with_one_overhang()
        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
        )
        learner.setup(feature_selector=None, random_state=42)
        learner.train("coxph")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner.bootstrap(
                "coxph", n_bootstrap=2, n_importance_repeats=0,
                show_progress=False, verbose=False,
            )
        meth = self._meth_warnings(caught)
        ipcw_msgs = [str(w.message) for w in meth if "IPCW" in str(w.message)]
        assert len(ipcw_msgs) == 1, (
            f"Expected one consolidated IPCW warning for trivial overhang "
            f"in bootstrap(); got {len(ipcw_msgs)}:\n"
            + "\n---\n".join(ipcw_msgs)
        )
        msg = ipcw_msgs[0]
        assert "IPCW-weighted metrics" in msg
        assert "Using tau=" in msg

    def test_validate_trivial_overhang_threads_tau(self):
        """Same guarantee for ``validate()`` — the consolidated warning
        fires once per cohort and includes the tau clause.
        """
        X_train, X_test, y_train, y_test = _split_with_one_overhang()
        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
        )
        learner.setup(feature_selector=None, random_state=42)
        learner.train("coxph")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner.validate(
                X_test, y_test, models="coxph", name="ExternalA",
                bootstrap=False, verbose=False,
            )
        meth = self._meth_warnings(caught)
        ipcw_msgs = [
            str(w.message) for w in meth
            if "IPCW-weighted metrics" in str(w.message)
            or "may be unstable" in str(w.message)
        ]
        assert len(ipcw_msgs) == 1, (
            f"Expected one consolidated IPCW warning from validate(), "
            f"got {len(ipcw_msgs)}:\n" + "\n---\n".join(ipcw_msgs)
        )
        msg = ipcw_msgs[0]
        assert "IPCW-weighted metrics" in msg
        assert "Using tau=" in msg
        assert "cohort 'ExternalA'" in msg

    def test_material_overhang_still_emits_stability_warning(self):
        # 5 obs in 30 test → 16.7% > 1% threshold → both warnings fire.
        X_train, X_test, y_train, y_test = _split_with_long_test()
        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
        )
        learner.setup(feature_selector=None, random_state=42)
        learner.train("coxph")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner.evaluate("coxph", verbose=False)
        meth = self._meth_warnings(caught)
        ipcw_truncation = [w for w in meth if "IPCW-weighted metrics" in str(w.message)]
        ipcw_unstable = [w for w in meth if "may be unstable" in str(w.message)]
        assert len(ipcw_truncation) == 1
        assert len(ipcw_unstable) == 1


# ---------------------------------------------------------------------------
# Report footnote presence/absence
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestReportTruncationFootnote:
    def test_footnote_appears_when_truncated(self, tmp_path):
        X_train, X_test, y_train, y_test = _split_with_long_test()
        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
        )
        learner.setup(feature_selector=None, random_state=42)
        learner.train("coxph")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = learner.evaluate("coxph", verbose=False)
            learner.generate_report(str(tmp_path / "report.html"))
        html = (tmp_path / "report.html").read_text(encoding="utf-8")
        assert "y_test</code> truncated" in html
        assert f"t_max={result.truncation_t_max:.1f}" in html
        assert f"({result.truncation_n_beyond} obs censored)" in html

    def test_footnote_absent_when_no_truncation(self, tmp_path):
        rng = np.random.default_rng(0)
        n_train, n_test = 60, 20
        X = pd.DataFrame(
            rng.normal(size=(n_train + n_test, 4)),
            columns=[f"f{i}" for i in range(4)],
        )
        train_times = rng.uniform(1, 50, n_train)
        test_times = rng.uniform(1, 10, n_test)
        y_train = make_y(train_times, rng.binomial(1, 0.7, n_train).astype(bool))
        y_test = make_y(test_times, rng.binomial(1, 0.7, n_test).astype(bool))
        learner = SurvivalLearner(
            X_train=X.iloc[:n_train].reset_index(drop=True),
            X_test=X.iloc[n_train:].reset_index(drop=True),
            y_train=y_train, y_test=y_test,
        )
        learner.setup(feature_selector=None, random_state=42)
        learner.train("coxph")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner.evaluate("coxph", verbose=False)
            learner.generate_report(str(tmp_path / "report.html"))
        html = (tmp_path / "report.html").read_text(encoding="utf-8")
        assert "y_test</code> truncated" not in html
