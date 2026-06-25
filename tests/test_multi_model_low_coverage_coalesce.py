"""Cross-model AUC(t) low-coverage coalescing.

The per-model ``_emit_low_coverage_summary`` runs inside ``_bootstrap_single``.
In a multi-model bootstrap where every model hits the same low-coverage
``(metric, subpopulation)`` condition, that produced N near-identical
``MethodologicalWarning``s. This module verifies the new cross-model emitter
``_emit_multi_low_coverage_summary`` (mirror of ``_emit_multi_failure_summary``)
collapses those into one warning per ``(metric, subpopulation)`` group listing
all affected models, and that the deferral plumbing (``_notices_collector`` /
the ``_return_failures`` 3-tuple) works on both bootstrap dispatch paths.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import MethodologicalWarning
from mlsurv.evaluation.bootstrap import _emit_multi_low_coverage_summary
from mlsurv.utils import to_structured_array


class TestEmitMultiLowCoverageSummary:
    """The new cross-model ``_emit_multi_low_coverage_summary`` aggregator."""

    def test_no_notices_no_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_multi_low_coverage_summary({}, n_bootstrap=10)
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert methodo == []

    def test_four_models_same_group_one_coalesced_warning(self):
        # 4 models all hit the same (metric, subpop) with identical coverage
        # and count → exactly one warning naming all four models.
        models = ["coxph", "rsf", "coxnet", "gbsa"]
        model_notices = {
            m: [("subpopulation_auc_t", "short_fu", 1, 4)] * 3
            for m in models
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_multi_low_coverage_summary(model_notices, n_bootstrap=3)
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        msg = str(methodo[0].message)
        # All four model names listed.
        for m in models:
            assert m in msg
        assert "subpopulation 'short_fu'" in msg
        assert "subpopulation_auc_t" in msg
        # Coalesced form reports shared coverage/count and model count.
        assert "1/4" in msg
        assert "3/3 bootstrap iterations" in msg
        assert "4 models" in msg

    def test_one_warning_per_distinct_group(self):
        # 4 models × 2 distinct (metric, subpop) groups → 2 warnings.
        models = ["coxph", "rsf", "coxnet", "gbsa"]
        model_notices = {}
        for m in models:
            model_notices[m] = [
                ("auc_t", None, 1, 4),
                ("subpopulation_auc_t", "short_fu", 1, 4),
            ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_multi_low_coverage_summary(model_notices, n_bootstrap=1)
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 2
        msgs = [str(w.message) for w in methodo]
        assert any("auc_t" in m and "subpopulation" not in m for m in msgs)
        assert any("subpopulation_auc_t" in m and "short_fu" in m for m in msgs)

    def test_mixed_coverage_per_model_breakdown(self):
        # Models differ in coverage/count for the same group → per-model
        # breakdown listing each model's coverage.
        model_notices = {
            "coxph": [("auc_t", None, 1, 4)],            # 1/4 once
            "rsf": [("auc_t", None, 0, 4), ("auc_t", None, 0, 4)],  # 0/4 twice
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_multi_low_coverage_summary(model_notices, n_bootstrap=2)
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        msg = str(methodo[0].message)
        assert "coxph=1/4" in msg
        assert "rsf=0/4" in msg
        assert "2 models" in msg

    def test_subpop_none_omits_subpop_label(self):
        model_notices = {
            "coxph": [("auc_t", None, 0, 4)],
            "rsf": [("auc_t", None, 0, 4)],
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_multi_low_coverage_summary(model_notices, n_bootstrap=1)
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        msg = str(methodo[0].message)
        assert "subpopulation" not in msg
        assert msg.startswith("auc_t: insufficient")

    def test_single_model_uses_per_model_breakdown(self):
        # A lone model still emits (the coalesced "N models" branch needs >1).
        model_notices = {"coxph": [("auc_t", None, 1, 4)]}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_multi_low_coverage_summary(model_notices, n_bootstrap=1)
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        assert "coxph=1/4" in str(methodo[0].message)

    def test_same_coverage_different_count_falls_to_breakdown(self):
        # Identical coverage (1/4) but different iteration counts must NOT
        # coalesce — the count component of the signature must split them.
        model_notices = {
            "coxph": [("auc_t", None, 1, 4), ("auc_t", None, 1, 4)],  # count 2
            "rsf": [("auc_t", None, 1, 4), ("auc_t", None, 1, 4),
                    ("auc_t", None, 1, 4)],                            # count 3
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_multi_low_coverage_summary(model_notices, n_bootstrap=3)
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        msg = str(methodo[0].message)
        # Per-model breakdown branch (not the coalesced "N models" form).
        assert "coxph=1/4 in 2" in msg
        assert "rsf=1/4 in 3" in msg

    def test_coverage_range_formatting(self):
        # Differing n_valid across iterations for one group → en-dash range.
        model_notices = {
            "coxph": [("auc_t", None, 1, 4), ("auc_t", None, 3, 4)],
            "rsf": [("auc_t", None, 1, 4), ("auc_t", None, 3, 4)],
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_multi_low_coverage_summary(model_notices, n_bootstrap=2)
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        msg = str(methodo[0].message)
        # Coalesced (identical signature) with min–max/total coverage.
        assert "1–3/4" in msg
        assert "2 models" in msg

    def test_multi_model_many_subpops_single_line(self):
        # ≥2 subpopulation levels across multiple models collapse into
        # ONE line. Levels with a shared cross-model signature render the
        # compact "N models" fragment; mixed levels render the per-model
        # breakdown fragment — both within the same line.
        model_notices = {
            # Bladder: identical signature across both models → coalesced frag.
            # Breast: differing coverage across models → breakdown frag.
            "coxph": [
                ("subpopulation_auc_t", "Bladder", 3, 7),
                ("subpopulation_auc_t", "Breast", 1, 7),
            ],
            "rsf": [
                ("subpopulation_auc_t", "Bladder", 3, 7),
                ("subpopulation_auc_t", "Breast", 4, 7),
            ],
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_multi_low_coverage_summary(model_notices, n_bootstrap=10)
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        msg = str(methodo[0].message)
        assert msg.startswith("subpopulation_auc_t: insufficient")
        assert "2 subpopulation levels" in msg
        # Shared-signature level → compact "N models" fragment.
        assert "Bladder (3/7, 1/10 iters, 2 models)" in msg
        # Mixed level → per-model breakdown fragment.
        assert "Breast [coxph=1/7 in 1, rsf=4/7 in 1]" in msg

    def test_multi_model_single_subpop_keeps_original_format(self):
        # Boundary: exactly one distinct level across multiple models
        # must keep the original one-line "N models" format and must NOT
        # enter the collapsed multi-level branch.
        model_notices = {
            m: [("subpopulation_auc_t", "Bladder", 3, 7)]
            for m in ("coxph", "rsf", "coxnet")
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_multi_low_coverage_summary(model_notices, n_bootstrap=5)
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        msg = str(methodo[0].message)
        assert "subpopulation levels" not in msg
        assert msg.startswith(
            "subpopulation_auc_t subpopulation 'Bladder': insufficient"
        )
        assert "for 3 models" in msg

    def test_multi_model_collapsed_level_hit_by_single_model(self):
        # A collapsed multi-level line where one level was hit by only ONE
        # model must render that level via the per-model breakdown fragment
        # (the compact "N models" form requires len(models) > 1).
        model_notices = {
            "coxph": [
                ("subpopulation_auc_t", "Bladder", 3, 7),
                ("subpopulation_auc_t", "Breast", 2, 7),
            ],
            # rsf only hits Bladder, not Breast.
            "rsf": [("subpopulation_auc_t", "Bladder", 3, 7)],
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_multi_low_coverage_summary(model_notices, n_bootstrap=5)
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        msg = str(methodo[0].message)
        assert "2 subpopulation levels" in msg
        assert "Bladder (3/7, 1/5 iters, 2 models)" in msg
        # Single-model level → breakdown fragment, not "1 models".
        assert "Breast [coxph=2/7 in 1]" in msg

    def test_multi_model_cohort_and_levels_emit_separately(self):
        # Defensive: if a metric somehow carries both a cohort (None) bucket
        # and ≥2 named levels, the named levels collapse into one line and the
        # cohort bucket is emitted as its own separate line — nothing dropped,
        # and the level count counts only the named levels.
        model_notices = {
            "coxph": [
                ("subpopulation_auc_t", None, 1, 4),
                ("subpopulation_auc_t", "A", 1, 4),
                ("subpopulation_auc_t", "B", 2, 4),
            ],
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_multi_low_coverage_summary(model_notices, n_bootstrap=3)
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 2
        collapsed = [m for m in (str(w.message) for w in methodo)
                     if "subpopulation levels" in m]
        cohort = [m for m in (str(w.message) for w in methodo)
                  if "subpopulation levels" not in m]
        assert len(collapsed) == 1
        assert "2 subpopulation levels" in collapsed[0]
        assert "A " in collapsed[0] and "B " in collapsed[0]
        # Cohort line carries no subpopulation label and is not folded in.
        assert len(cohort) == 1
        assert "subpopulation '" not in cohort[0]
        assert "None" not in collapsed[0]


def _make_xy(n=80, seed=0):
    rng = np.random.default_rng(seed)
    times_short = rng.uniform(0.2, 1.0, n // 2)
    times_long = rng.uniform(5.0, 20.0, n // 2)
    times = np.concatenate([times_short, times_long])
    events = rng.integers(0, 2, n).astype(bool)
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    y = to_structured_array(times, events)
    return X, y, times


class TestBootstrapSingleDeferral:
    """``_bootstrap_single`` deferral plumbing for notices."""

    @pytest.mark.integration
    def test_notices_collector_suppresses_inline_and_populates(self):
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from mlsurv.evaluation.bootstrap import _bootstrap_single

        X, y, times = _make_xy()
        model = CoxPHSurvivalAnalysis().fit(X.values, y)
        short_labels = np.where(times <= 1.0, "short", "long")

        collector = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _bootstrap_single(
                model, X, y, y_train=y, n_bootstrap=3, include_auc=True,
                eval_times=np.array([1.0, 2.0, 6.0, 12.0], dtype=float),
                subpopulation_labels=short_labels, show_progress=False,
                n_jobs=1, model_name="coxph", _notices_collector=collector,
            )
        # Inline low-coverage emission suppressed; notices deferred to dict.
        inline = [w for w in caught
                  if issubclass(w.category, MethodologicalWarning)
                  and "insufficient eval_times coverage" in str(w.message)]
        assert inline == []
        assert "coxph" in collector
        assert len(collector["coxph"]) > 0


class TestMultiModelBootstrapIntegration:
    """Path A: free ``bootstrap()`` over a dict of models coalesces."""

    @pytest.mark.integration
    def test_multi_model_one_summary_per_group(self):
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from mlsurv.evaluation.bootstrap import bootstrap

        X, y, times = _make_xy()
        models = {
            "coxph_a": CoxPHSurvivalAnalysis().fit(X.values, y),
            "coxph_b": CoxPHSurvivalAnalysis(alpha=0.1).fit(X.values, y),
            "coxph_c": CoxPHSurvivalAnalysis(alpha=1.0).fit(X.values, y),
        }
        short_labels = np.where(times <= 1.0, "short", "long")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bootstrap(
                models, X, y, y_train=y, n_bootstrap=3, include_auc=True,
                eval_times=np.array([1.0, 2.0, 6.0, 12.0], dtype=float),
                subpopulation_labels=short_labels, show_progress=False,
                n_jobs=1,
            )

        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)
                   and "insufficient eval_times coverage" in str(w.message)]
        assert len(methodo) >= 1
        # All subpopulation_auc_t levels (short/long) collapse into a
        # single cross-model line; cohort auc_t emits at most one separate line.
        sub_lines = [w for w in methodo
                     if "subpopulation_auc_t" in str(w.message)]
        cohort_lines = [w for w in methodo
                        if "subpopulation_auc_t" not in str(w.message)]
        assert len(sub_lines) <= 1, (
            f"Expected subpopulation levels coalesced into one line; got "
            f"{[str(w.message) for w in sub_lines]}"
        )
        assert len(cohort_lines) <= 1
        # The collapsed line must NAME every level that hit low coverage —
        # guards against coalescing silently dropping a level.
        if sub_lines:
            msg = str(sub_lines[0].message)
            assert "short" in msg and "long" in msg, (
                f"collapsed line dropped a level: {msg}"
            )
            assert "2 subpopulation levels" in msg
        # Coalesced lines should reference multiple models, not a single name.
        assert any("models" in str(w.message) for w in methodo)

    @pytest.mark.integration
    def test_multi_model_no_low_coverage_no_warning_no_crash(self):
        # eval_times entirely within the observed window → notices empty.
        # Exercises the 3-tuple-with-empty-notices unpacking in both
        # _bootstrap_multiple branches and the no-op _emit_multi_* path.
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from mlsurv.evaluation.bootstrap import bootstrap

        X, y, times = _make_xy()
        models = {
            "coxph_a": CoxPHSurvivalAnalysis().fit(X.values, y),
            "coxph_b": CoxPHSurvivalAnalysis(alpha=0.1).fit(X.values, y),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = bootstrap(
                models, X, y, y_train=y, n_bootstrap=3, include_auc=True,
                # All within ~(0.2, 20) observed window.
                eval_times=np.array([6.0, 9.0, 12.0], dtype=float),
                show_progress=False, n_jobs=1,
            )
        assert result is not None
        low_cov = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)
                   and "insufficient eval_times coverage" in str(w.message)]
        assert low_cov == []

    @pytest.mark.parallel
    @pytest.mark.integration
    def test_multi_model_parallel_path_coalesces(self):
        # Force the parallel model-execution branch so the 4-tuple worker
        # return / unpack is exercised for notices.
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from mlsurv.evaluation.bootstrap import bootstrap

        X, y, times = _make_xy()
        models = {
            f"coxph_{i}": CoxPHSurvivalAnalysis(alpha=0.1 * i).fit(X.values, y)
            for i in range(1, 4)
        }
        short_labels = np.where(times <= 1.0, "short", "long")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bootstrap(
                models, X, y, y_train=y, n_bootstrap=5, include_auc=True,
                eval_times=np.array([1.0, 2.0, 6.0, 12.0], dtype=float),
                subpopulation_labels=short_labels, show_progress=False,
                n_jobs=1, n_jobs_models=2, force_parallel=True,
            )
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)
                   and "insufficient eval_times coverage" in str(w.message)]
        # At minimum the parallel path must not crash on the 4-tuple unpack;
        # if low coverage triggered, lines should be cross-model coalesced.
        for w in methodo:
            assert "bootstrap iterations" in str(w.message)


class TestEvaluateMultipleFallback:
    """The finally-block per-model fallback when the coalesced formatter
    raises (mirror of the per-model coalescer fallback)."""

    @pytest.mark.integration
    def test_formatter_failure_falls_back_to_per_model(self, monkeypatch):
        import importlib
        from mlsurv import SurvivalLearner

        # The package re-exports the `bootstrap` function, shadowing the
        # submodule attribute, so fetch the real module via importlib.
        boot_mod = importlib.import_module("mlsurv.evaluation.bootstrap")

        # Force the cross-model emitter to raise so the fallback fires. The
        # finally block does `from .bootstrap import _emit_multi_low_coverage_summary`
        # at call time, so patching the module attribute takes effect.
        def _boom(*args, **kwargs):
            raise ValueError("synthetic formatter failure")

        monkeypatch.setattr(boot_mod, "_emit_multi_low_coverage_summary", _boom)

        X, y, times = _make_xy(n=120, seed=1)
        lrn = SurvivalLearner(X, y)
        lrn.setup(test_fraction=0.3, cv_folds=2, feature_selector=None,
                  random_state=42)

        # The per-model fallback (_emit_low_coverage_summary) warns one line
        # per model; the coalesced (happy) path warns a single combined line.
        # Asserting BOTH model names appear as separate per-model
        # MethodologicalWarnings proves the fallback ran and re-emitted
        # everything (No Silent Data Loss). The fallback fires inside the
        # multi-model bootstrap evaluate that run() drives, so capture spans
        # run() through the explicit evaluate() (matches the original
        # whole-test caplog scope this replaced).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lrn.run(["coxph", "coxnet"])
            lrn.evaluate(
                bootstrap=True, n_bootstrap=3,
                # In-window eval_times (cohort t_max ~20, so not clipped) whose
                # upper values exceed many short-subgroup bootstrap resamples'
                # observed window → low coverage fires deterministically for
                # both models, so the patched formatter raises → fallback runs.
                eval_times=[1.0, 2.0, 6.0, 12.0],
                n_importance_repeats=1, show_progress=False,
            )
        low_cov = [str(w.message) for w in caught
                   if issubclass(w.category, MethodologicalWarning)
                   and "insufficient eval_times coverage" in str(w.message)]
        assert any(m.startswith("coxph:") for m in low_cov), (
            f"expected per-model coxph fallback line; got: {low_cov}"
        )
        assert any(m.startswith("coxnet:") for m in low_cov), (
            f"expected per-model coxnet fallback line; got: {low_cov}"
        )
        # The coalesced emitter is patched to raise everywhere, so its
        # cross-model "... for N models" line must NEVER appear. Asserting its
        # absence proves the captured per-model lines came from the fallback
        # (not a stray coalesced warning from run()), not the happy path.
        assert not any(" models" in m for m in low_cov), (
            f"coalesced low-coverage line leaked despite patched formatter: {low_cov}"
        )


class TestEvaluateMultipleIntegration:
    """Path B: multi-model evaluate(bootstrap=True) coalesces at loop end."""

    @pytest.mark.integration
    def test_validate_multi_model_one_summary_per_group(self):
        from mlsurv import SurvivalLearner

        X, y, times = _make_xy(n=120, seed=1)
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, feature_selector=None,
                      random_state=42)
        learner.run(["coxph", "coxnet"])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner.evaluate(
                bootstrap=True, n_bootstrap=3,
                eval_times=[1.0, 2.0, 6.0, 12.0],
                n_importance_repeats=1, show_progress=False,
            )

        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)
                   and "insufficient eval_times coverage" in str(w.message)]
        if not methodo:
            pytest.skip("low-coverage condition not triggered on this split")
        signatures = set()
        for w in methodo:
            msg = str(w.message)
            sig = (
                "subpopulation_auc_t" if "subpopulation_auc_t" in msg
                else "auc_t",
                "cohort",
            )
            signatures.add(sig)
        assert len(methodo) == len(signatures), (
            f"Expected one coalesced warning per group; got {len(methodo)}"
        )
