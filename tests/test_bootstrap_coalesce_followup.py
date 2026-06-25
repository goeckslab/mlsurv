"""Bootstrap-side AUC(t) silent-NaN coalescing.

The two bootstrap-iteration AUC(t) helpers used to silently emit all-NaN
rows when ``len(valid_times) < 2`` — no warning at any tier. This module
verifies that those cases now feed a per-iteration ``notices`` list which
``_emit_low_coverage_summary`` collapses into a single
``MethodologicalWarning`` per ``(metric, subpopulation)`` group at the
bootstrap top level.

Tests use direct calls to the private helpers + one integration test
through the free ``bootstrap()`` function to stay in the fast tier.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import MethodologicalWarning
from mlsurv.evaluation._bootstrap_iteration import (
    _compute_auc_for_iteration,
    _compute_subpopulation_auc_for_iter,
    _record_low_coverage,
)
from mlsurv.evaluation.bootstrap import _emit_low_coverage_summary
from mlsurv.utils import to_structured_array

from tests._helpers import make_y


# Training cohort with follow-up to ~24 mo.
Y_TRAIN = make_y(
    [0.5, 1.2, 2.4, 3.5, 5.0, 6.1, 7.8, 9.0, 10.5, 12.0,
     14.2, 15.5, 17.0, 18.4, 20.0, 21.6, 23.5, 24.0, 22.0, 16.5],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
)
# Healthy subgroup; t in roughly (1.0, 18.0).
Y_SUB_OK = make_y(
    [1.0, 2.5, 4.0, 5.5, 7.0, 9.5, 11.0, 13.0, 15.0, 18.0],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
)
RISK_OK = np.linspace(0.1, 1.0, 10)
# Short-follow-up subgroup; t_max ~ 1.0.
Y_SUB_SHORT = make_y(
    [0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
    [1, 1, 1, 1, 0, 1],
)
RISK_SHORT = np.linspace(0.2, 0.9, 6)


class TestRecordLowCoverage:
    """The new ``_record_low_coverage`` helper."""

    def test_appends_tuple(self):
        notices = []
        _record_low_coverage(
            notices, "auc_t", 0, "coxph",
            n_valid=1, n_total=4, subpopulation="short_fu",
        )
        assert notices == [("auc_t", "short_fu", 1, 4)]

    def test_none_emits_direct_methodological_warning(self):
        # No collector: must still surface (No Silent Data Loss rule).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _record_low_coverage(
                None, "auc_t", 0, "coxph", n_valid=0, n_total=3,
                subpopulation="short_fu",
            )
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        msg = str(methodo[0].message)
        assert "coxph subpopulation 'short_fu'" in msg
        assert "auc_t eval_times coverage 0/3" in msg


class TestComputeAUCForIterationLowCoverage:
    """Site 1: per-iteration cohort AUC(t)."""

    def test_records_when_below_threshold(self):
        notices = []
        rows = _compute_auc_for_iteration(
            Y_TRAIN, Y_SUB_SHORT, RISK_SHORT, [1, 2, 6, 9, 12, 18],
            model_name="coxph", iteration=0, notices=notices,
        )
        # All eval_times out of range → all NaN.
        assert rows is not None
        assert all(np.isnan(r['auc']) for r in rows)
        # One notice with subpop=None (cohort-level call).
        assert notices == [("auc_t", None, 0, 6)]

    def test_no_notice_when_coverage_ok(self):
        notices = []
        rows = _compute_auc_for_iteration(
            Y_TRAIN, Y_SUB_OK, RISK_OK, [2, 6, 9, 12],
            model_name="coxph", iteration=0, notices=notices,
        )
        # Some finite AUCs and no low-coverage notice.
        assert any(np.isfinite(r['auc']) for r in rows)
        assert notices == []


class TestComputeSubpopulationAUCForIterLowCoverage:
    """Site 2: per-iteration subpopulation AUC(t)."""

    def test_records_with_subpopulation_key(self):
        notices = []
        rows = _compute_subpopulation_auc_for_iter(
            Y_SUB_SHORT, RISK_SHORT, Y_TRAIN, [1, 2, 6, 9, 12, 18],
            group_name="short_fu", model_name="coxph", iteration=0,
            notices=notices,
        )
        assert all(np.isnan(r['auc']) for r in rows)
        assert notices == [("subpopulation_auc_t", "short_fu", 0, 6)]

    def test_no_notice_when_coverage_ok(self):
        notices = []
        _compute_subpopulation_auc_for_iter(
            Y_SUB_OK, RISK_OK, Y_TRAIN, [2, 6, 9, 12],
            group_name="ok_fu", model_name="coxph", iteration=0,
            notices=notices,
        )
        assert notices == []


class TestBootstrapAUCRowCountInvariant:
    """When len(valid_times) < 2, both bootstrap helpers must
    NaN-fill *every* eval_time (not just the out-of-range subset), so the
    valid-but-too-few time is not silently dropped. Output row count stays
    == len(eval_times) regardless of branch."""

    # Y_SUB_SHORT t_max ~ 1.0; only t=0.6 lands inside → 1 valid (< 2).
    EVAL_TIMES = [0.6, 5, 10, 20]

    def test_cohort_helper_fills_all_eval_times(self):
        rows = _compute_auc_for_iteration(
            Y_TRAIN, Y_SUB_SHORT, RISK_SHORT, self.EVAL_TIMES,
            model_name="coxph", iteration=0, notices=[],
        )
        assert len(rows) == len(self.EVAL_TIMES)
        assert all(np.isnan(r['auc']) for r in rows)

    def test_subpopulation_helper_fills_all_eval_times(self):
        rows = _compute_subpopulation_auc_for_iter(
            Y_SUB_SHORT, RISK_SHORT, Y_TRAIN, self.EVAL_TIMES,
            group_name="short_fu", model_name="coxph", iteration=0,
            notices=[],
        )
        assert len(rows) == len(self.EVAL_TIMES)
        assert all(np.isnan(r['auc']) for r in rows)

    # Partial-coverage (>= 2 valid + some out-of-range) exercises the
    # `else` branch (nan_times = eval_times_arr[~valid_mask]); the invariant
    # must still hold = valid AUC rows + out-of-range NaN rows.
    # Y_SUB_OK t in ~(1.0, 18.0); t=22 exceeds t_max so 3 valid + 1 out.
    PARTIAL_EVAL_TIMES = [2, 6, 9, 22]

    def test_cohort_helper_partial_coverage_invariant(self):
        rows = _compute_auc_for_iteration(
            Y_TRAIN, Y_SUB_OK, RISK_OK, self.PARTIAL_EVAL_TIMES,
            model_name="coxph", iteration=0, notices=[],
        )
        assert len(rows) == len(self.PARTIAL_EVAL_TIMES)
        assert any(np.isfinite(r['auc']) for r in rows)
        assert any(np.isnan(r['auc']) for r in rows)

    def test_subpopulation_helper_partial_coverage_invariant(self):
        rows = _compute_subpopulation_auc_for_iter(
            Y_SUB_OK, RISK_OK, Y_TRAIN, self.PARTIAL_EVAL_TIMES,
            group_name="ok_fu", model_name="coxph", iteration=0,
            notices=[],
        )
        assert len(rows) == len(self.PARTIAL_EVAL_TIMES)
        assert any(np.isfinite(r['auc']) for r in rows)
        assert any(np.isnan(r['auc']) for r in rows)


class TestEmitLowCoverageSummary:
    """The new ``_emit_low_coverage_summary`` aggregator."""

    def test_no_notices_no_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_low_coverage_summary([], n_bootstrap=10, model_name="coxph")
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert methodo == []

    def test_metric_collapses_subpop_levels(self):
        # 3 iterations × 2 metrics × 2 subpops = 12 notices. Each metric spans
        # 2 subpopulation levels → collapses to one line per metric,
        # so 2 warnings (not 4) listing both levels with per-level coverage.
        notices = []
        for it in range(3):
            for metric in ("auc_t", "subpopulation_auc_t"):
                for subpop in ("short_A", "short_B"):
                    notices.append((metric, subpop, 1, 4))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_low_coverage_summary(
                notices, n_bootstrap=3, model_name="coxph",
            )
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 2
        msgs = [str(w.message) for w in methodo]
        # Each collapsed line names both levels, the level count, and retains
        # per-level coverage + iteration detail.
        for m in msgs:
            assert "short_A" in m and "short_B" in m
            assert "2 subpopulation levels" in m
            assert "1/4" in m
            assert "3/3 iters" in m
        # One line per metric.
        assert any("auc_t" in m and "subpopulation_auc_t" not in m for m in msgs)
        assert any("subpopulation_auc_t" in m for m in msgs)

    def test_subpop_none_omits_subpop_label(self):
        # Cohort-level notices (subpopulation=None) should produce a clean
        # message without an empty "subpopulation ''" segment.
        notices = [("auc_t", None, 0, 4), ("auc_t", None, 0, 4)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_low_coverage_summary(
                notices, n_bootstrap=2, model_name="rsf",
            )
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        msg = str(methodo[0].message)
        assert "subpopulation" not in msg
        assert msg.startswith("rsf: auc_t insufficient")

    def test_many_subpops_single_line(self):
        # ≥2 subpopulation levels with differing coverage collapse into
        # exactly one warning listing every level with its own coverage.
        notices = [
            ("subpopulation_auc_t", "Bladder", 3, 7),
            ("subpopulation_auc_t", "Bladder", 6, 7),
            ("subpopulation_auc_t", "Breast", 1, 7),
            ("subpopulation_auc_t", "Lung", 5, 7),
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_low_coverage_summary(
                notices, n_bootstrap=100, model_name="coxph",
            )
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        msg = str(methodo[0].message)
        assert "3 subpopulation levels" in msg
        # Every level named, with per-level coverage detail.
        assert "Bladder (3–6/7, 2/100 iters)" in msg
        assert "Breast (1/7, 1/100 iters)" in msg
        assert "Lung (5/7, 1/100 iters)" in msg

    def test_single_subpop_level_keeps_original_format(self):
        # Boundary: exactly one distinct level → original per-group
        # format (not the collapsed multi-level line).
        notices = [
            ("subpopulation_auc_t", "Bladder", 3, 7),
            ("subpopulation_auc_t", "Bladder", 3, 7),
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_low_coverage_summary(
                notices, n_bootstrap=2, model_name="coxph",
            )
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        msg = str(methodo[0].message)
        assert "subpopulation levels" not in msg
        assert msg.startswith(
            "coxph subpopulation 'Bladder': subpopulation_auc_t insufficient"
        )
        assert "(3/7) in 2/2 bootstrap iterations" in msg

    def test_cohort_and_levels_emit_separately(self):
        # Defensive: a metric carrying both a cohort (None) bucket and ≥2 named
        # levels collapses the named levels into one line and emits the cohort
        # bucket as its own separate line — nothing dropped, level count counts
        # only the named levels, and "None" never leaks into the collapsed line.
        notices = [
            ("subpopulation_auc_t", None, 1, 4),
            ("subpopulation_auc_t", "A", 1, 4),
            ("subpopulation_auc_t", "B", 2, 4),
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_low_coverage_summary(
                notices, n_bootstrap=3, model_name="coxph",
            )
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 2
        msgs = [str(w.message) for w in methodo]
        collapsed = [m for m in msgs if "subpopulation levels" in m]
        cohort = [m for m in msgs if "subpopulation levels" not in m]
        assert len(collapsed) == 1
        assert "2 subpopulation levels" in collapsed[0]
        assert "A " in collapsed[0] and "B " in collapsed[0]
        assert "None" not in collapsed[0]
        # Cohort line carries no subpopulation label.
        assert len(cohort) == 1
        assert "subpopulation '" not in cohort[0]


class TestBootstrapIntegration:
    """End-to-end: free ``bootstrap()`` with a short-FU cohort surfaces
    one summary line per group, no per-iteration spam."""

    def _make_xy(self, n=80, seed=0):
        # Hand-built data with two natural sub-cohorts: 'short' (t <= 1.0)
        # and 'long' (t in ~5–20). Training spans both ranges so the model
        # converges; bootstrap on the short subset triggers low coverage.
        rng = np.random.default_rng(seed)
        times_short = rng.uniform(0.2, 1.0, n // 2)
        times_long = rng.uniform(5.0, 20.0, n // 2)
        times = np.concatenate([times_short, times_long])
        events = rng.integers(0, 2, n).astype(bool)
        # Two informative features.
        X = pd.DataFrame({
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        })
        y = to_structured_array(times, events)
        return X, y, times

    @pytest.mark.integration
    def test_short_followup_emits_one_summary_per_group(self):
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from mlsurv.evaluation.bootstrap import bootstrap

        X, y, times = self._make_xy()
        model = CoxPHSurvivalAnalysis().fit(X.values, y)

        # Subpopulation: short follow-up samples only.
        short_labels = np.where(times <= 1.0, "short", "long")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bootstrap(
                model, X, y, y_train=y, n_bootstrap=3,
                include_auc=True,
                eval_times=np.array([1.0, 2.0, 6.0, 12.0], dtype=float),
                subpopulation_labels=short_labels,
                show_progress=False,
                n_jobs=1,
            )

        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)
                   and "insufficient eval_times coverage" in str(w.message)]
        # All subpopulation_auc_t levels (short/long) collapse into a
        # single line; the cohort auc_t path emits at most one separate line.
        assert len(methodo) >= 1
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
        # guards against coalescing silently dropping a level. Both 'short'
        # and 'long' subgroups straddle the eval_times window here.
        if sub_lines:
            msg = str(sub_lines[0].message)
            assert "short" in msg and "long" in msg, (
                f"collapsed line dropped a level: {msg}"
            )
            assert "2 subpopulation levels" in msg
