"""Tests for warning quality improvements."""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import (
    ClampWarning,
    FitFailureWarning,
    InternalWarning,
    IterativeWarning,
    MethodologicalWarning,
    MetricFailureWarning,
    SmallSubpopulationWarning,
    SuggestionWarning,
    configure_default_filters,
    enable_internal_warnings,
)
from tests._helpers import make_feature_dataframe, make_random_survival_array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_survival_array(n, event_rate=0.5, seed=42):
    """File-specific survival data: uniform events, exp(scale=10) times."""
    return make_random_survival_array(
        n, event_rate=event_rate, time_scale=10.0,
        event_dist="uniform", seed=seed,
    )


def _make_dataframe(n, p=5, seed=42):
    """Feature DataFrame with ``feat_`` column prefix."""
    return make_feature_dataframe(n, p, seed=seed, prefix="feat_")


# ---------------------------------------------------------------------------
# Test: Warning class hierarchy
# ---------------------------------------------------------------------------

class TestWarningHierarchy:
    """Verify the tiered warning class hierarchy is correct."""

    @pytest.mark.parametrize("warning_cls", [
        MethodologicalWarning, SuggestionWarning, InternalWarning, IterativeWarning,
    ])
    def test_base_tiers_are_user_warning(self, warning_cls):
        """All base tiers (and IterativeWarning) inherit from UserWarning."""
        assert issubclass(warning_cls, UserWarning)

    @pytest.mark.parametrize("warning_cls", [
        ClampWarning, FitFailureWarning, MetricFailureWarning, SmallSubpopulationWarning,
    ])
    def test_iterative_subclass_dual_hierarchy(self, warning_cls):
        """Each iterative subclass is both Methodological and Iterative.

        Subsumes the old per-class ``*_hierarchy`` tests and the
        ``test_all_iterative_subclasses`` loop — the IterativeWarning check is
        what the dedup filter relies on.
        """
        assert issubclass(warning_cls, MethodologicalWarning)
        assert issubclass(warning_cls, IterativeWarning)


# ---------------------------------------------------------------------------
# Test: Iterative filter deduplication
# ---------------------------------------------------------------------------

class TestIterativeFilterDedup:
    """filterwarnings("once", category=IterativeWarning) deduplicates all subclasses."""

    def test_iterative_filter_catches_all_subclasses(self):
        """A single IterativeWarning filter deduplicates all 4 subclasses."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings("once", category=IterativeWarning)

            # Emit each subclass twice with the same message
            for cls in (ClampWarning, FitFailureWarning, MetricFailureWarning,
                        SmallSubpopulationWarning):
                warnings.warn(f"test {cls.__name__}", cls)
                warnings.warn(f"test {cls.__name__}", cls)

        # Should get exactly 4 (one per class), not 8
        assert len(caught) == 4, (
            f"Expected 4 warnings (one per class), got {len(caught)}: "
            f"{[str(w.message) for w in caught]}"
        )


# ---------------------------------------------------------------------------
# Test: Default filter behavior
# ---------------------------------------------------------------------------

class TestDefaultFilterBehavior:
    """Verify configure_default_filters() sets the correct filter policies."""

    def test_methodological_always_fires(self):
        """MethodologicalWarning fires even after configure_default_filters."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.resetwarnings()
            configure_default_filters()
            warnings.warn("test methodological", MethodologicalWarning)
            warnings.warn("test methodological", MethodologicalWarning)

        meth = [w for w in caught if issubclass(w.category, MethodologicalWarning)]
        assert len(meth) == 2, "MethodologicalWarning should always fire"

    def test_suggestion_fires_once(self):
        """SuggestionWarning fires once per unique message by default."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.resetwarnings()
            configure_default_filters()
            warnings.warn("test suggestion", SuggestionWarning)
            warnings.warn("test suggestion", SuggestionWarning)

        sugg = [w for w in caught if issubclass(w.category, SuggestionWarning)]
        assert len(sugg) == 1, "SuggestionWarning should fire once"

    def test_internal_suppressed_by_default(self):
        """InternalWarning is suppressed by default."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.resetwarnings()
            configure_default_filters()
            warnings.warn("test internal", InternalWarning)

        internal = [w for w in caught if issubclass(w.category, InternalWarning)]
        assert len(internal) == 0, "InternalWarning should be suppressed by default"

    def test_enable_internal_warnings(self):
        """enable_internal_warnings() re-enables InternalWarning display."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.resetwarnings()
            configure_default_filters()
            enable_internal_warnings()
            warnings.warn("test internal enabled", InternalWarning)

        internal = [w for w in caught if issubclass(w.category, InternalWarning)]
        assert len(internal) == 1, "InternalWarning should fire after enable_internal_warnings()"

    def test_iterative_fires_once(self):
        """IterativeWarning (and subclasses) fire once per message after configure_default_filters."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.resetwarnings()
            configure_default_filters()
            warnings.warn("clamped values", ClampWarning)
            warnings.warn("clamped values", ClampWarning)

        clamp = [w for w in caught if issubclass(w.category, ClampWarning)]
        assert len(clamp) == 1, (
            f"ClampWarning should fire once under default filters, got {len(clamp)}"
        )


# ---------------------------------------------------------------------------
# Test: Cross-phase ClampWarning deduplication
# ---------------------------------------------------------------------------

class TestClampWarningCrossPhaseDedupe:
    """ClampWarning from training/eval should suppress identical warnings in report gen."""

    def test_same_message_across_phases_shows_once(self):
        """Same ClampWarning text emitted from multiple call sites shows only once."""
        msg = "CoxPH: 6 linear predictor(s) clamped to [-20, 20]"
        with warnings.catch_warnings(record=True) as caught:
            warnings.resetwarnings()
            configure_default_filters()

            # Simulate training phase
            warnings.warn(msg, ClampWarning)
            # Simulate report-generation phase (same message, different call site)
            warnings.warn(msg, ClampWarning)

        clamp = [w for w in caught if issubclass(w.category, ClampWarning)]
        assert len(clamp) == 1, (
            f"Expected 1 ClampWarning across phases, got {len(clamp)}"
        )

    def test_distinct_messages_both_shown(self):
        """Different ClampWarning messages are both shown (not over-suppressed)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.resetwarnings()
            configure_default_filters()

            warnings.warn("CoxPH: 6 LP(s) clamped", ClampWarning)
            warnings.warn("CoxNet: 3 LP(s) clamped", ClampWarning)

        clamp = [w for w in caught if issubclass(w.category, ClampWarning)]
        assert len(clamp) == 2, (
            f"Expected 2 distinct ClampWarnings, got {len(clamp)}"
        )


# ---------------------------------------------------------------------------
# Test: Category census — no bare UserWarning in mlsurv/
# ---------------------------------------------------------------------------

class TestCategoryCensus:
    """Verify that no warnings.warn() calls in mlsurv/ use bare UserWarning."""

    # Files/patterns that are intentionally kept as UserWarning or use dynamic categories
    _ALLOWED_BARE = set()

    _DYNAMIC_PATTERNS = {
        # tuning/base.py _emit_trial_warnings: re-emits with dynamic category
        'warn_explicit',
        'warn_category',
    }

    def test_no_bare_user_warning(self):
        """All warnings.warn() calls in mlsurv/ use a tiered category, not UserWarning."""
        mlsurv_dir = Path(__file__).parent.parent / "mlsurv"
        violations = []

        for py_file in sorted(mlsurv_dir.rglob("*.py")):
            if py_file.name in self._ALLOWED_BARE:
                continue

            text = py_file.read_text(encoding="utf-8")
            lines = text.split("\n")

            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "warnings.warn(" not in stripped:
                    continue
                if "filterwarnings" in stripped or "warn_explicit" in stripped:
                    continue

                # Gather the full call (may span multiple lines)
                call_text = line
                depth = line.count("(") - line.count(")")
                j = i + 1
                while depth > 0 and j < len(lines):
                    call_text += " " + lines[j]
                    depth += lines[j].count("(") - lines[j].count(")")
                    j += 1

                # Skip dynamic category passthroughs
                if any(pat in call_text for pat in self._DYNAMIC_PATTERNS):
                    continue

                # Check for UserWarning in the warn() args
                warn_idx = call_text.index("warnings.warn(")
                args_part = call_text[warn_idx:]
                if "UserWarning" in args_part:
                    rel_path = py_file.relative_to(mlsurv_dir.parent)
                    violations.append(f"{rel_path}:{i + 1}")

        assert not violations, (
            f"Found {len(violations)} bare UserWarning calls:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


# ---------------------------------------------------------------------------
# Test: ClampWarning backward compatibility
# ---------------------------------------------------------------------------

class TestClampWarningBackwardCompat:
    """ClampWarning can be imported from both _warnings and _overflow."""

    def test_import_from_warnings(self):
        from mlsurv._warnings import ClampWarning as CW1
        assert CW1 is ClampWarning

    def test_import_from_overflow(self):
        from mlsurv.models._overflow import ClampWarning as CW2
        assert CW2 is ClampWarning


# ---------------------------------------------------------------------------
# Test: ClampWarning fires at most once in bootstrap
# ---------------------------------------------------------------------------

class TestClampWarningInBootstrap:
    """ClampWarning should be deduplicated to at most one per bootstrap run."""

    def test_clamp_warning_fires_once_in_bootstrap(self):
        """Bootstrap with a model producing extreme LPs emits at most 1 ClampWarning."""
        from mlsurv.evaluation.bootstrap import bootstrap

        n = 50
        y_train = _make_survival_array(n, seed=1)
        y_test = _make_survival_array(n, seed=2)
        X_test = _make_dataframe(n, seed=2)

        # Mock model that returns extreme risk scores (will trigger clamping)
        model = MagicMock()
        model.name = "mock"
        model.display_name = "Mock"
        model.predict = MagicMock(return_value=np.full(n, 50.0))
        # predict_survival_function not available — only harrell metric
        model.predict_survival_function = None
        del model.predict_survival_function

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bootstrap(
                model, X_test, y_test, y_train=y_train,
                n_bootstrap=10, metrics=["harrell"],
                show_progress=False, random_state=42,
            )

        clamp_warnings = [w for w in caught if issubclass(w.category, ClampWarning)]
        assert len(clamp_warnings) <= 1, (
            f"Expected at most 1 ClampWarning, got {len(clamp_warnings)}"
        )


# ---------------------------------------------------------------------------
# Test: CoxNet alpha_index warning stacklevel
# ---------------------------------------------------------------------------

class TestAlphaIndexStacklevel:
    """CoxNet alpha_index clamping warning should point to the caller."""

    def test_alpha_index_warning_stacklevel(self):
        """Warning for alpha_index exceeding path length points to caller file."""
        from mlsurv.models.coxnet import CoxNetModel

        model = CoxNetModel(alpha=999, l1_ratio=0.9, fit_baseline_model=False)

        n, p = 100, 5
        X = _make_dataframe(n, p, seed=42)
        y = _make_survival_array(n, seed=42)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X, y)

        alpha_warnings = [
            w for w in caught
            if "alpha_index=" in str(w.message) and "exceeds" in str(w.message)
        ]
        assert len(alpha_warnings) >= 1, "Expected alpha_index clamping warning"
        # stacklevel=2 means the warning filename should be this test file,
        # not coxnet.py
        assert "coxnet.py" not in alpha_warnings[0].filename, (
            f"Warning points to coxnet.py instead of caller: {alpha_warnings[0].filename}"
        )


# ---------------------------------------------------------------------------
# Test: Bootstrap failure summary
# ---------------------------------------------------------------------------

class TestBootstrapFailureSummary:
    """Bootstrap should emit aggregated failure summaries, not per-iteration logs."""

    def test_bootstrap_failure_summary(self):
        """Metric failures are aggregated into a summary, not logged per iteration."""
        from mlsurv.evaluation.bootstrap import _emit_failure_summary

        failures = [
            ("harrell", "ValueError", "not enough events"),
            ("harrell", "ValueError", "not enough events"),
            ("harrell", "ValueError", "not enough events"),
            ("brier", "ZeroDivisionError", "division by zero"),
        ]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_failure_summary(failures, n_bootstrap=10, model_name="CoxPH")

        # Should get exactly 2 MetricFailureWarning emissions (one per group).
        from mlsurv._warnings import MetricFailureWarning
        metric_warns = [w for w in caught
                        if isinstance(w.message, MetricFailureWarning)]
        assert len(metric_warns) == 2

        # Both groups warn unconditionally (each failure represents
        # user-visible data dropped from the bootstrap aggregate).
        harrell_warns = [
            w for w in caught if "harrell" in str(w.message) and "3/10" in str(w.message)
        ]
        assert len(harrell_warns) == 1

        brier_warns = [
            w for w in caught if "brier" in str(w.message) and "1/10" in str(w.message)
        ]
        assert len(brier_warns) == 1

    def test_unexpected_error_always_warns(self):
        """Unexpected exception types always emit a warning regardless of rate."""
        from mlsurv.evaluation.bootstrap import _emit_failure_summary

        # Any failure rate / exception type emits a warning.
        failures = [
            ("harrell", "TypeError", "wrong dtype"),
        ]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_failure_summary(failures, n_bootstrap=100, model_name="CoxPH")

        # Should still warn because TypeError is unexpected
        type_warns = [w for w in caught if "TypeError" in str(w.message)]
        assert len(type_warns) == 1

    def test_no_summary_when_no_failures(self):
        """No output when there are no failures."""
        from mlsurv.evaluation.bootstrap import _emit_failure_summary

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_failure_summary([], n_bootstrap=100, model_name="CoxPH")

        assert caught == []

    def test_multi_model_failure_summary_collapses_identical(self):
        """Identical (metric, exc_type, count) across models -> one line, all models listed."""
        from mlsurv.evaluation.bootstrap import _emit_multi_failure_summary

        shared = [("ipcw", "ValueError", "Found array with 0 sample(s)")]
        model_failures = {name: list(shared) for name in
                          ("coxph", "coxnet", "rsf", "xgbcox", "xgbaft", "fssvm")}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_multi_failure_summary(model_failures, n_bootstrap=100)

        # 1/100 (1%) still represents 6 models worth of dropped IPCW
        # metrics, so the coalesced summary must warn through the
        # MetricFailureWarning (MethodologicalWarning) channel exactly once.
        from mlsurv._warnings import MetricFailureWarning, MethodologicalWarning
        matches = [w for w in caught
                   if isinstance(w.message, MetricFailureWarning)
                   and "ipcw failed" in str(w.message)]
        assert len(matches) == 1
        assert isinstance(matches[0].message, MethodologicalWarning)
        msg = str(matches[0].message)
        assert "ipcw failed in 1/100 iterations for 6 models" in msg
        for name in ("coxph", "coxnet", "rsf", "xgbcox", "xgbaft", "fssvm"):
            assert name in msg

    def test_multi_model_failure_summary_mixed_counts(self):
        """Different counts across models -> per-model breakdown line."""
        from mlsurv.evaluation.bootstrap import _emit_multi_failure_summary
        from mlsurv._warnings import MetricFailureWarning

        model_failures = {
            "coxph": [("ipcw", "ValueError", "x")] * 2,
            "rsf":   [("ipcw", "ValueError", "x")] * 5,
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_multi_failure_summary(model_failures, n_bootstrap=100)

        # Mixed-count cross-model groups must also warn (worst rate
        # 5/100 = 5% would have been gated by the old 10% threshold).
        matches = [w for w in caught
                   if isinstance(w.message, MetricFailureWarning)]
        assert len(matches) == 1
        msg = str(matches[0].message)
        assert "coxph=2" in msg and "rsf=5" in msg

    def test_multi_model_failure_summary_empty(self):
        """No models with failures -> no output."""
        from mlsurv.evaluation.bootstrap import _emit_multi_failure_summary
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_multi_failure_summary({}, n_bootstrap=100)
        assert caught == []

    def test_low_rate_ipcw_failure_surfaces_warning(self):
        """1/100 ipcw ValueError (an "expected" exception below the
        old 10% threshold) must still surface a MethodologicalWarning
        because every failed iteration drops user-visible IPCW data."""
        from mlsurv.evaluation.bootstrap import _emit_multi_failure_summary
        from mlsurv._warnings import MetricFailureWarning, MethodologicalWarning

        model_failures = {
            name: [("ipcw", "ValueError", "Found array with 0 sample(s)")]
            for name in ("coxph", "coxnet", "rsf", "xgbcox", "xgbaft", "fssvm")
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_multi_failure_summary(model_failures, n_bootstrap=100)

        matches = [w for w in caught
                   if isinstance(w.message, MetricFailureWarning)
                   and "ipcw failed" in str(w.message)]
        assert len(matches) == 1
        assert isinstance(matches[0].message, MethodologicalWarning)
        assert "1/100 iterations for 6 models" in str(matches[0].message)

    def test_single_model_low_rate_ipcw_warns(self):
        """Same guarantee on the single-model bootstrap path."""
        from mlsurv.evaluation.bootstrap import _emit_failure_summary
        from mlsurv._warnings import MetricFailureWarning

        failures = [("ipcw", "ValueError", "Found array with 0 sample(s)")]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_failure_summary(failures, n_bootstrap=100, model_name="coxph")

        matches = [w for w in caught
                   if isinstance(w.message, MetricFailureWarning)
                   and "ipcw failed" in str(w.message)]
        assert len(matches) == 1


# ---------------------------------------------------------------------------
# Test: bootstrap-failure coalesce on run()/evaluate(dict) path
# ---------------------------------------------------------------------------

class TestRunEvaluateCoalescesBootstrapFailures:
    """``learner.run([...])`` / ``evaluate(dict, bootstrap=True)`` must
    coalesce identical per-model bootstrap failures into a single
    ``_emit_multi_failure_summary`` call instead of N per-model
    ``_emit_failure_summary`` lines.

    Regression symptom: vig 7 emitted N near-identical
    ``mlsurv.bootstrap [ERROR] <model>: ipcw failed in K/N iterations`` lines
    because each model's evaluate→bootstrap fell into ``_bootstrap_single``,
    which independently called ``_emit_failure_summary``.
    """

    @pytest.mark.integration
    def test_run_multi_model_coalesces_identical_failures(
        self, small_data, monkeypatch,
    ):
        """run() with a multi-model list emits ONE coalesced failure summary.

        Forces every bootstrap iteration to return an identical synthetic
        ``ipcw`` failure (rather than waiting for real edge-case resamples
        to hit), then asserts the multi-model emitter fired exactly once and
        the per-model emitter fired zero times.
        """
        from mlsurv import SurvivalLearner
        import importlib
        boot_mod = importlib.import_module("mlsurv.evaluation.bootstrap")

        # Inject an identical failure into every aggregated failure list.
        real_aggregate = boot_mod._aggregate_iteration_results

        def patched_aggregate(results, config):
            (iter_data, pred_data, auc_data, brier_data, imp_data, sg_data,
             _real_failures, _real_notices) = real_aggregate(results, config)
            failures = [("ipcw", "ValueError",
                         "Found array with 0 sample(s)")]
            return (iter_data, pred_data, auc_data, brier_data, imp_data, sg_data,
                    failures, _real_notices)

        monkeypatch.setattr(
            boot_mod, "_aggregate_iteration_results", patched_aggregate,
        )

        multi_calls = []
        single_calls = []
        monkeypatch.setattr(
            boot_mod, "_emit_multi_failure_summary",
            lambda *a, **kw: multi_calls.append((a, kw)),
        )
        monkeypatch.setattr(
            boot_mod, "_emit_failure_summary",
            lambda *a, **kw: single_calls.append((a, kw)),
        )

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, test_fraction=0.3)
        learner.run(
            ["coxph", "coxnet"],
            bootstrap=True,
            n_bootstrap=2,
            n_importance_repeats=0,
            verbose=False,
            show_eval_progress=False,
        )

        assert len(multi_calls) == 1, (
            f"expected exactly one coalesced multi-failure-summary, "
            f"got {len(multi_calls)}"
        )
        assert len(single_calls) == 0, (
            f"expected zero per-model emissions on multi-model dispatch, "
            f"got {len(single_calls)}: {single_calls}"
        )

        # Collector dict has BOTH models recorded under the same (metric,
        # exc_type) key, which is what enables the cross-model collapse.
        model_failures_arg = multi_calls[0][0][0]
        assert set(model_failures_arg.keys()) == {"coxph", "coxnet"}

    @pytest.mark.integration
    def test_run_single_model_keeps_inline_per_model_emission(
        self, small_data, monkeypatch,
    ):
        """Single-model dispatch must keep the inline per-model summary.

        Guards the ``len(model_items) > 1`` half of the collector creation
        in ``_evaluate_multiple``. If a future refactor drops that length
        check, a single-model ``run(['coxph'])`` would silently swallow its
        per-model warning into a collector with one entry. This test pins
        the current contract: with one model and bootstrap failures, the
        per-model ``_emit_failure_summary`` fires once and the multi-model
        coalescer never fires.
        """
        from mlsurv import SurvivalLearner
        import importlib
        boot_mod = importlib.import_module("mlsurv.evaluation.bootstrap")

        real_aggregate = boot_mod._aggregate_iteration_results

        def patched_aggregate(results, config):
            (iter_data, pred_data, auc_data, brier_data, imp_data, sg_data,
             _real_failures, _real_notices) = real_aggregate(results, config)
            failures = [("ipcw", "ValueError",
                         "Found array with 0 sample(s)")]
            return (iter_data, pred_data, auc_data, brier_data, imp_data, sg_data,
                    failures, _real_notices)

        monkeypatch.setattr(
            boot_mod, "_aggregate_iteration_results", patched_aggregate,
        )

        multi_calls = []
        single_calls = []
        monkeypatch.setattr(
            boot_mod, "_emit_multi_failure_summary",
            lambda *a, **kw: multi_calls.append((a, kw)),
        )
        monkeypatch.setattr(
            boot_mod, "_emit_failure_summary",
            lambda *a, **kw: single_calls.append((a, kw)),
        )

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, test_fraction=0.3)
        learner.run(
            ["coxph"],
            bootstrap=True,
            n_bootstrap=2,
            n_importance_repeats=0,
            verbose=False,
            show_eval_progress=False,
        )

        assert len(multi_calls) == 0, (
            f"multi-model emitter must not fire on single-model dispatch, "
            f"got {len(multi_calls)} call(s)"
        )
        assert len(single_calls) == 1, (
            f"single-model emitter should fire exactly once, "
            f"got {len(single_calls)}"
        )

    @pytest.mark.integration
    def test_run_multi_model_formatter_failure_falls_back_to_per_model(
        self, small_data, monkeypatch,
    ):
        """When ``_emit_multi_failure_summary`` raises, fall back to per-model.

        The new finally block in ``_evaluate_multiple`` wraps the coalesced
        emission in try/except so a formatter regression in the helper does
        not drop user-visible records. This test forces the coalescer to
        raise and asserts the fallback path emits per-model summaries +
        warns about the formatter failure.
        """
        from mlsurv import SurvivalLearner
        import importlib
        boot_mod = importlib.import_module("mlsurv.evaluation.bootstrap")

        real_aggregate = boot_mod._aggregate_iteration_results

        def patched_aggregate(results, config):
            (iter_data, pred_data, auc_data, brier_data, imp_data, sg_data,
             _real_failures, _real_notices) = real_aggregate(results, config)
            failures = [("ipcw", "ValueError", "boom")]
            return (iter_data, pred_data, auc_data, brier_data, imp_data, sg_data,
                    failures, _real_notices)

        monkeypatch.setattr(
            boot_mod, "_aggregate_iteration_results", patched_aggregate,
        )

        single_calls = []

        def raising_multi(*a, **kw):
            raise ValueError("synthetic formatter failure")

        monkeypatch.setattr(
            boot_mod, "_emit_multi_failure_summary", raising_multi,
        )
        monkeypatch.setattr(
            boot_mod, "_emit_failure_summary",
            lambda *a, **kw: single_calls.append((a, kw)),
        )

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, test_fraction=0.3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner.run(
                ["coxph", "coxnet"],
                bootstrap=True,
                n_bootstrap=2,
                n_importance_repeats=0,
                verbose=False,
                show_eval_progress=False,
            )

        # Both models' per-model summaries fired via the fallback.
        emitted_names = sorted(c[0][2] for c in single_calls)
        assert emitted_names == ["coxnet", "coxph"], (
            f"fallback should emit per-model summaries for both models, "
            f"got names {emitted_names}"
        )
        # A MethodologicalWarning announced the formatter failure.
        formatter_warns = [
            w for w in caught
            if isinstance(w.message, MethodologicalWarning)
            and "Failed to format coalesced bootstrap-failure summary"
            in str(w.message)
        ]
        assert len(formatter_warns) == 1, (
            f"expected one MethodologicalWarning about formatter failure, "
            f"got {len(formatter_warns)}: "
            f"{[str(w.message) for w in formatter_warns]}"
        )


# ---------------------------------------------------------------------------
# Test: Failures collector pattern
# ---------------------------------------------------------------------------

class TestFailuresCollectorPattern:
    """Verify the failures list is populated by _safe_compute_metric."""

    def test_safe_compute_metric_appends_to_failures(self):
        """_safe_compute_metric appends to failures list on error."""
        from mlsurv.evaluation._bootstrap_iteration import _safe_compute_metric

        y = _make_survival_array(10, seed=1)
        # Risk scores with wrong length to trigger ValueError
        risk_scores = np.array([1.0])

        failures = []
        value = _safe_compute_metric(
            y, risk_scores, "harrell", y_train=None, model=None,
            X_boot=None, eval_times=None, surv_probs=None, i=0,
            model_name="test", failures=failures,
        )
        assert np.isnan(value)
        assert len(failures) == 1
        assert failures[0][0] == "harrell"  # metric name
        assert len(failures[0]) == 3  # (metric, exc_type, exc_msg)

    def test_safe_compute_metric_logs_when_no_failures_list(self):
        """_safe_compute_metric calls log_metric_failure when failures=None."""
        from mlsurv.evaluation._bootstrap_iteration import _safe_compute_metric

        y = _make_survival_array(10, seed=1)
        risk_scores = np.array([1.0])

        with patch("mlsurv.evaluation._bootstrap_iteration.log_metric_failure") as mock_log:
            value = _safe_compute_metric(
                y, risk_scores, "harrell", y_train=None, model=None,
                X_boot=None, eval_times=None, surv_probs=None, i=0,
                model_name="test", failures=None,
            )
        assert np.isnan(value)
        assert mock_log.call_count == 1

    def test_aggregate_collects_failures(self):
        """_aggregate_iteration_results collects failures from all iterations."""
        from mlsurv.evaluation.bootstrap import _aggregate_iteration_results
        from mlsurv.evaluation._config import BootstrapConfig

        config = BootstrapConfig(
            metrics=["harrell"], custom_metrics={}, eval_times=np.array([1.0]),
            ci=0.95, include_predictions=False, include_auc=False,
            include_importance=False, n_importance_repeats=5, stratify=False,
        )
        results = [
            {"iteration_data": [], "predictions_data": None,
             "auc_data": None, "importance_data": None,
             "failures": [("harrell", "ValueError", "err1")]},
            {"iteration_data": [], "predictions_data": None,
             "auc_data": None, "importance_data": None,
             "failures": [("brier", "ZeroDivisionError", "err2")]},
            {"iteration_data": [], "predictions_data": None,
             "auc_data": None, "importance_data": None,
             "failures": []},
        ]
        _, _, _, _, _, _, all_failures, _ = _aggregate_iteration_results(results, config)
        assert len(all_failures) == 2
        assert all_failures[0][0] == "harrell"
        assert all_failures[1][0] == "brier"


# ---------------------------------------------------------------------------
# Test: CoxPH alpha=0 warning content
# ---------------------------------------------------------------------------

class TestCoxPHWarningContent:
    """CoxPH fallback warnings should include likely causes and recommendations."""

    def test_coxph_alpha0_warning_content(self):
        """Fallback warning contains diagnostic hints (collinear, near-singular)."""
        from mlsurv.models.coxph import CoxPHModel

        # Create near-singular data (duplicate columns)
        n, p = 30, 3
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
        X["f3"] = X["f0"]  # exact collinearity
        X["f4"] = X["f1"] + 1e-15 * rng.randn(n)  # near-collinearity
        y = _make_survival_array(n, seed=42)

        model = CoxPHModel(alpha=0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model.fit(X, y)
            except RuntimeError:
                pass  # OK if fallback also fails

        fallback_warnings = [
            w for w in caught
            if "CoxPH" in str(w.message) and (
                "collinear" in str(w.message) or "near-singular" in str(w.message)
            )
        ]
        # If numerical instability was detected, the warning should contain hints
        instability_warnings = [
            w for w in caught
            if "CoxPH" in str(w.message) and "Re-fitting" in str(w.message)
        ]
        if instability_warnings:
            assert len(fallback_warnings) >= 1, (
                "Fallback warning should mention 'collinear' or 'near-singular'. "
                f"Got: {[str(w.message) for w in instability_warnings]}"
            )


# ---------------------------------------------------------------------------
# Test: PIMP multi-model warning content
# ---------------------------------------------------------------------------

class TestPIMPWarningContent:
    """PIMP multi-model warning should include runtime note and alternative."""

    def test_pimp_multimodel_warning_content(self):
        """Multi-model PIMP warning mentions n_permutations and bootstrap alternative."""
        # Read the source code to verify the warning message content,
        # since mocking the full SurvivalLearner stack is complex.
        import inspect
        from mlsurv.learner import _evaluation

        source = inspect.getsource(_evaluation.EvaluationMixin.importance_test)
        assert "def importance_test" in source, (
            "inspect.getsource returned wrong method (stale __pycache__?); "
            f"got: {source[:60]!r}..."
        )

        # The warning for multi-model PIMP should contain these hints
        assert "n_permutations" in source, (
            "PIMP multi-model warning should mention n_permutations"
        )
        assert "method='bootstrap'" in source, (
            "PIMP multi-model warning should suggest bootstrap alternative"
        )


# ---------------------------------------------------------------------------
# Test: clean warning format (no file/line for mlsurv's own warnings)
# ---------------------------------------------------------------------------

class TestWarningFormat:
    """mlsurv's own warnings render as ``ClassName: message`` with no file/line.

    Third-party warnings keep the default stdlib format (file path + line +
    source echo), which still helps diagnose real upstream issues.
    """

    @pytest.fixture(autouse=True)
    def _ensure_formatter_installed(self):
        """Guarantee the clean formatter is active regardless of test order."""
        configure_default_filters()

    @pytest.mark.parametrize("warning_cls", [
        MethodologicalWarning, SuggestionWarning, InternalWarning,
        IterativeWarning, ClampWarning,
    ])
    def test_mlsurv_warning_has_no_file_or_line(self, warning_cls):
        """Own warnings format to exactly ``ClassName: message`` with no scaffolding."""
        out = warnings.formatwarning(
            "the data has too few events", warning_cls,
            "/abs/path/to/mlsurv/evaluation/evaluate.py", 123, "warnings.warn(msg)",
        )
        assert out == f"{warning_cls.__name__}: the data has too few events\n"
        # No file path, line number, or echoed source line.
        assert "evaluate.py" not in out
        assert "123" not in out
        assert "warnings.warn" not in out

    def test_third_party_warning_keeps_default_format(self):
        """Non-mlsurv warnings keep the stdlib format with file path and line."""
        out = warnings.formatwarning(
            "deprecated thing", DeprecationWarning, "/p/upstream.py", 5, "x = 1",
        )
        assert "/p/upstream.py" in out
        assert "5" in out
        assert "DeprecationWarning" in out

    def test_emitted_warning_renders_cleanly(self):
        """End-to-end: an emitted mlsurv warning prints the clean one-liner."""
        import io
        buf = io.StringIO()
        original_showwarning = warnings.showwarning

        def _capture(message, category, filename, lineno, file=None, line=None):
            buf.write(warnings.formatwarning(message, category, filename, lineno, line))

        warnings.showwarning = _capture
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn("recalibrate to fix calibration", MethodologicalWarning)
        finally:
            warnings.showwarning = original_showwarning

        rendered = buf.getvalue()
        assert rendered == "MethodologicalWarning: recalibrate to fix calibration\n"
        assert "test_warning_quality.py" not in rendered

    def test_repeat_configure_is_idempotent(self):
        """Calling configure_default_filters twice keeps the clean formatter and
        leaves the third-party fallback intact (no double-wrapping)."""
        from mlsurv._warnings import _mlsurv_formatwarning

        configure_default_filters()
        configure_default_filters()
        assert warnings.formatwarning is _mlsurv_formatwarning

        own = warnings.formatwarning("msg", MethodologicalWarning, "/p.py", 1, "x")
        assert own == "MethodologicalWarning: msg\n"
        # Fallback to stdlib format for non-mlsurv warnings must still work.
        third = warnings.formatwarning("dep", DeprecationWarning, "/p.py", 2, "x")
        assert "/p.py" in third and "2" in third
