"""Silent NaN fallback in AUC(t) violates No-Silent-Data-Loss.

Three sites silently return NaN AUC(t) rows when the subgroup follow-up
window doesn't cover the user's eval grid, and one of them additionally
crashes (TypeError → silently caught) when eval_times is a Python list
instead of an ndarray.

Tests use direct calls to the private helpers — no model training — to
stay in the fast tier.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlsurv._iter_warn import IterativeWarningCoalescer
from mlsurv._warnings import MethodologicalWarning
from mlsurv.evaluation._bootstrap_iteration import (
    _compute_auc_for_iteration,
    _compute_subpopulation_auc_for_iter,
)
from mlsurv.evaluation._subpopulation import _safe_auc_t
from tests._helpers import make_y


# A reasonable training cohort: follow-up to ~24 months, mixed events.
_Y_TRAIN_TIMES = [0.5, 1.2, 2.4, 3.5, 5.0, 6.1, 7.8, 9.0, 10.5, 12.0,
                   14.2, 15.5, 17.0, 18.4, 20.0, 21.6, 23.5, 24.0,
                   22.0, 16.5]
_Y_TRAIN_EVENTS = [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
Y_TRAIN = make_y(_Y_TRAIN_TIMES, _Y_TRAIN_EVENTS)

# A reasonable subgroup with follow-up overlapping the training range.
_Y_SUB_OK_TIMES = [1.0, 2.5, 4.0, 5.5, 7.0, 9.5, 11.0, 13.0, 15.0, 18.0]
_Y_SUB_OK_EVENTS = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
Y_SUB_OK = make_y(_Y_SUB_OK_TIMES, _Y_SUB_OK_EVENTS)
RISK_OK = np.linspace(0.1, 1.0, len(_Y_SUB_OK_TIMES))

# A short-follow-up subgroup: max time ~1.0, eval grid extends to 18 mo.
_Y_SUB_SHORT_TIMES = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
_Y_SUB_SHORT_EVENTS = [1, 1, 1, 1, 0, 1]
Y_SUB_SHORT = make_y(_Y_SUB_SHORT_TIMES, _Y_SUB_SHORT_EVENTS)
RISK_SHORT = np.linspace(0.2, 0.9, len(_Y_SUB_SHORT_TIMES))


class TestSafeAUCTListEvalTimes:
    """Part 1: list eval_times must not trigger the silent TypeError.

    Before the fix, `eval_times[~valid_mask]` on a Python list raised
    ``TypeError: only integer scalar arrays can be converted to a scalar
    index``, which `safe_call` swallowed into a `logger.warning`.
    """

    def test_list_eval_times_produces_finite_auc(self):
        eval_times = [1, 2, 6, 9]  # Python list, not ndarray
        rows = _safe_auc_t(
            Y_SUB_OK, RISK_OK, Y_TRAIN, eval_times,
            model_name="coxph", label="ok_group",
        )
        assert len(rows) == len(eval_times)
        aucs = [r['auc'] for r in rows]
        assert any(np.isfinite(a) for a in aucs), (
            "Expected at least one finite AUC for adequately-followed "
            "subgroup; got all-NaN — list-vs-ndarray regression."
        )

    def test_list_eval_times_no_warning_when_coalescer_absent(self):
        """No coalescer + all eval_times strictly inside subgroup range:
        clean run, no warning."""
        # Y_SUB_OK has t in (1.0, 18.0); valid_mask uses strict `>`/`<`.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _safe_auc_t(
                Y_SUB_OK, RISK_OK, Y_TRAIN, [2, 6, 9, 12],
                model_name="coxph", label="ok_group",
            )
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert methodo == []


class TestSafeAUCTShortFollowup:
    """Part 3: subgroup whose t_max admits 0 or 1 eval_times must
    surface a MethodologicalWarning (via the coalescer) instead of
    silently filling NaN."""

    def test_short_followup_records_to_coalescer(self):
        eval_times = [1, 2, 3, 6, 9, 12, 18]  # only t=... fall in (t_min, t_max)
        coalescer = IterativeWarningCoalescer(
            mode="notices",
            summary_label="AUC(t)",
            notice_phrase="returned NaN",
            n_total=1,
            max_emit=0,
            category=MethodologicalWarning,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with coalescer:
                rows = _safe_auc_t(
                    Y_SUB_SHORT, RISK_SHORT, Y_TRAIN, eval_times,
                    model_name="coxph", label="short_group",
                    auc_t_coalescer=coalescer,
                )
        # Every eval_time row should be present, all NaN.
        assert len(rows) == len(eval_times)
        assert all(np.isnan(r['auc']) for r in rows)
        # The coalescer emits its aggregate summary on __exit__.
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        assert "AUC(t)" in str(methodo[0].message)
        assert "returned NaN" in str(methodo[0].message)

    def test_short_followup_without_coalescer_warns_directly(self):
        """No coalescer: direct callers still get a user-visible
        MethodologicalWarning per "No Silent Data Loss" rule."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rows = _safe_auc_t(
                Y_SUB_SHORT, RISK_SHORT, Y_TRAIN, [1, 2, 6, 9, 12, 18],
                model_name="coxph", label="short_group",
            )
        assert len(rows) == 6
        assert all(np.isnan(r['auc']) for r in rows)
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        assert "AUC(t) for coxph/short_group" in str(methodo[0].message)

    def test_partial_coverage_records_notice(self):
        """Success path with some out-of-range times must also surface a
        notice — those NaN rows would otherwise be silent."""
        # Y_SUB_OK runs to t~18; eval grid includes t=22 (above subgroup t_max).
        eval_times = [2, 6, 9, 22]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rows = _safe_auc_t(
                Y_SUB_OK, RISK_OK, Y_TRAIN, eval_times,
                model_name="coxph", label="ok_group",
            )
        assert len(rows) == 4
        finite = [r for r in rows if np.isfinite(r['auc'])]
        nan_rows = [r for r in rows if not np.isfinite(r['auc'])]
        assert len(finite) >= 1 and len(nan_rows) >= 1
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        assert "outside" in str(methodo[0].message)


class TestSafeAUCTBoundaryConditions:
    """Boundary value coverage required by CONTRIBUTING.md."""

    def test_empty_eval_times(self):
        """Empty eval_times yields empty rows, no warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rows = _safe_auc_t(
                Y_SUB_OK, RISK_OK, Y_TRAIN, [],
                model_name="coxph", label="ok_group",
            )
        assert rows == []
        # Empty input → no notice (nothing to report).
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert methodo == []

    def test_all_eval_times_above_tmax(self):
        """All eval_times above subgroup t_max → all-NaN + one notice."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rows = _safe_auc_t(
                Y_SUB_SHORT, RISK_SHORT, Y_TRAIN, [10, 20, 30],
                model_name="coxph", label="short_group",
            )
        assert len(rows) == 3
        assert all(np.isnan(r['auc']) for r in rows)
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        assert "0/3" in str(methodo[0].message)

    def test_exactly_one_valid_time(self):
        """len(valid_times) == 1: below the >= 2 sksurv threshold,
        treated as the "too few" all-NaN path: every eval_time gets
        a NaN row — including the valid-but-too-few t=0.6 — so the output
        shape is invariant (len(rows) == len(eval_times)) regardless of
        which branch ran."""
        # Subgroup t_max ~ 1.0; only t=0.6 lands inside (0.3, 1.0).
        eval_times = [0.6, 5, 10, 20]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rows = _safe_auc_t(
                Y_SUB_SHORT, RISK_SHORT, Y_TRAIN, eval_times,
                model_name="coxph", label="short_group",
            )
        # All 4 eval_times → 4 NaN rows (shape invariant); the
        # valid-but-too-few t=0.6 is no longer dropped.
        assert len(rows) == 4
        assert all(np.isnan(r['auc']) for r in rows)
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        assert "1/4" in str(methodo[0].message)


class TestSafeAUCTExceptionPath:
    """Part 2: an exception inside the compute body must surface via
    the coalescer rather than disappear into a logger.warning."""

    def test_exception_records_to_coalescer(self, monkeypatch):
        # Force cumulative_dynamic_auc to raise ValueError. _safe_auc_t computes
        # via the shared helper in mlsurv.evaluation.metrics, so patch the name
        # there (the helper holds its own module-level import of it).
        import mlsurv.evaluation.metrics as metrics

        def _boom(*args, **kwargs):
            raise ValueError("forced failure")

        monkeypatch.setattr(metrics, "cumulative_dynamic_auc", _boom)

        eval_times = np.array([1.0, 2.0, 6.0, 9.0])
        coalescer = IterativeWarningCoalescer(
            mode="notices",
            summary_label="AUC(t)",
            notice_phrase="returned NaN",
            n_total=1,
            max_emit=0,
            category=MethodologicalWarning,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with coalescer:
                rows = _safe_auc_t(
                    Y_SUB_OK, RISK_OK, Y_TRAIN, eval_times,
                    model_name="coxph", label="ok_group",
                    auc_t_coalescer=coalescer,
                )
        assert len(rows) == len(eval_times)
        assert all(np.isnan(r['auc']) for r in rows)
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1


class TestSafeAUCTDivideInstabilityNaN:
    """`_safe_auc_t` must surface a notice when cumulative_dynamic_auc returns
    NaN at a *valid* time (the IPCW divide-instability edge), not just for
    out-of-range or <2-valid times. Before the shared-core fix, that NaN was
    appended silently — the same blind spot the `_compute_auc_t` sibling closed.

    The instability is hard to trigger deterministically on real data, so we
    patch the helper's cumulative_dynamic_auc to return a NaN among valid times.
    """

    @staticmethod
    def _nan_among_valid(*args, **kwargs):
        # Signature: cumulative_dynamic_auc(y_train, y_test, estimate, times).
        times = args[3]
        vals = np.full(len(times), 0.7)
        vals[-1] = np.nan  # late-time NaN among the valid times
        return vals, 0.7

    def test_valid_time_nan_records_notice(self, monkeypatch):
        import mlsurv.evaluation.metrics as metrics
        monkeypatch.setattr(metrics, "cumulative_dynamic_auc",
                            self._nan_among_valid)

        # All four strictly inside Y_SUB_OK's (1, 18) range → 4 valid times.
        eval_times = np.array([2.0, 6.0, 9.0, 12.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rows = _safe_auc_t(
                Y_SUB_OK, RISK_OK, Y_TRAIN, eval_times,
                model_name="coxph", label="ok_group",
            )
        assert len(rows) == len(eval_times)
        assert sum(np.isnan(r['auc']) for r in rows) == 1
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        # Pre-fix this fired ZERO warnings (the silent-data-loss regression).
        assert len(methodo) == 1
        assert "returned NaN" in str(methodo[0].message)

    def test_valid_time_nan_records_to_coalescer(self, monkeypatch):
        import mlsurv.evaluation.metrics as metrics
        monkeypatch.setattr(metrics, "cumulative_dynamic_auc",
                            self._nan_among_valid)

        eval_times = np.array([2.0, 6.0, 9.0, 12.0])
        coalescer = IterativeWarningCoalescer(
            mode="notices",
            summary_label="AUC(t)",
            notice_phrase="returned NaN",
            n_total=1,
            max_emit=0,
            category=MethodologicalWarning,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with coalescer:
                rows = _safe_auc_t(
                    Y_SUB_OK, RISK_OK, Y_TRAIN, eval_times,
                    model_name="coxph", label="ok_group",
                    auc_t_coalescer=coalescer,
                )
        assert len(rows) == len(eval_times)  # row-count invariant holds
        assert sum(np.isnan(r['auc']) for r in rows) == 1
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1

    def test_mixed_out_of_range_and_divide_nan_counts_both(self, monkeypatch):
        """A call with one out-of-range time AND a divide-NaN at a valid time:
        the single coalesced count must cover BOTH sources (nan_mask, not just
        out-of-range n_out). Guards against reverting to out-of-range counting."""
        import mlsurv.evaluation.metrics as metrics

        def _nan_first_valid(*args, **kwargs):
            # times here are the VALID (in-range) times only; NaN the first.
            times = args[3]
            vals = np.full(len(times), 0.7)
            vals[0] = np.nan
            return vals, 0.7

        monkeypatch.setattr(metrics, "cumulative_dynamic_auc", _nan_first_valid)

        # 22.0 is beyond Y_SUB_OK's t_max (~18) → out of range; the other three
        # are valid, and the patch NaNs one of them.
        eval_times = np.array([2.0, 6.0, 9.0, 22.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rows = _safe_auc_t(
                Y_SUB_OK, RISK_OK, Y_TRAIN, eval_times,
                model_name="coxph", label="ok_group",
            )
        assert len(rows) == len(eval_times)
        # 1 out-of-range (22.0) + 1 divide-NaN among valid = 2 NaN total.
        assert sum(np.isnan(r['auc']) for r in rows) == 2
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        assert "2/4" in str(methodo[0].message)

    def test_divide_runtimewarning_is_filtered(self, monkeypatch):
        """The numpy "invalid value encountered in divide" RuntimeWarning must
        stay suppressed in `_safe_auc_t` (it lacked the filter before the fix);
        only the MethodologicalWarning reaches the user."""
        import mlsurv.evaluation.metrics as metrics

        def _warn_then_nan(*args, **kwargs):
            warnings.warn("invalid value encountered in divide", RuntimeWarning)
            times = args[3]
            vals = np.full(len(times), 0.7)
            vals[-1] = np.nan
            return vals, 0.7

        monkeypatch.setattr(metrics, "cumulative_dynamic_auc", _warn_then_nan)

        eval_times = np.array([2.0, 6.0, 9.0, 12.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _safe_auc_t(
                Y_SUB_OK, RISK_OK, Y_TRAIN, eval_times,
                model_name="coxph", label="ok_group",
            )
        leaked = [w for w in caught
                  if issubclass(w.category, RuntimeWarning)
                  and "invalid value encountered in divide" in str(w.message)]
        assert leaked == []
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1


class TestCoalescingSingleSummary:
    """Test D: many (model × subgroup) pairs all funnel through the
    same coalescer → exactly ONE MethodologicalWarning per
    compute_subpopulation_metrics call."""

    def test_multiple_pairs_emit_one_summary(self):
        eval_times = [1, 2, 3, 6, 9, 12, 18]
        n_pairs = 2 * 3  # 2 models × 3 problem subgroups
        coalescer = IterativeWarningCoalescer(
            mode="notices",
            summary_label="AUC(t)",
            notice_phrase="returned NaN",
            n_total=n_pairs,
            max_emit=0,
            category=MethodologicalWarning,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with coalescer:
                for model_name in ("coxph", "rsf"):
                    for label in ("short_A", "short_B", "short_C"):
                        _safe_auc_t(
                            Y_SUB_SHORT, RISK_SHORT, Y_TRAIN, eval_times,
                            model_name=model_name, label=label,
                            auc_t_coalescer=coalescer,
                        )
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        # max_emit=0 → only the summary line, regardless of n_pairs.
        assert len(methodo) == 1
        msg = str(methodo[0].message)
        # The summary should report the full count.
        assert f"{n_pairs}/{n_pairs}" in msg


class TestComputeSubpopulationMetricsIntegration:
    """Verify the coalescer is actually wired through
    compute_subpopulation_metrics → _compute_model_subpopulation_metrics
    → _safe_auc_t. Catches a forgotten kwarg or hoisted-context regression."""

    @staticmethod
    def _build_eval_result(model_names, y_sub, risk_per_model):
        """Build a minimal EvaluationResult-shaped object: just a test_set
        DataFrame with model_type + risk_score columns is what
        compute_subpopulation_metrics actually reads."""
        import pandas as pd
        from types import SimpleNamespace
        rows = []
        for mname in model_names:
            for risk in risk_per_model[mname]:
                rows.append({'model_type': mname, 'risk_score': risk})
        return SimpleNamespace(test_set=pd.DataFrame(rows))

    def test_two_models_short_subgroup_emits_one_summary(self):
        from mlsurv.evaluation._subpopulation import compute_subpopulation_metrics

        # Two models × one short-followup subgroup → two notices
        # internally → one coalesced summary.
        models = ('coxph', 'rsf')
        risk_per_model = {m: RISK_SHORT for m in models}
        eval_result = self._build_eval_result(models, Y_SUB_SHORT, risk_per_model)

        # Labels: a single "short" group covering all samples.
        labels = np.array(['short'] * len(Y_SUB_SHORT))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = compute_subpopulation_metrics(
                eval_result,
                {'cohort': labels},
                Y_SUB_SHORT,
                min_events=1,
                y_train=Y_TRAIN,
                eval_times=np.array([1, 2, 6, 9, 12, 18], dtype=float),
            )
        # Exactly one MethodologicalWarning from the AUC(t) coalescer
        # with "2/2" (1 subpop × 2 models).
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        auc_methodo = [w for w in methodo
                       if "AUC(t)" in str(w.message)]
        assert len(auc_methodo) == 1, (
            f"Expected one AUC(t) MethodologicalWarning; got "
            f"{[str(w.message) for w in auc_methodo]}"
        )
        assert "2/2" in str(auc_methodo[0].message), str(auc_methodo[0].message)

        # The auc_t DataFrame should still be populated (with NaN).
        sg = results['cohort']
        assert sg.auc_t is not None
        assert (sg.auc_t['model_type'].nunique() == 2)

    def test_no_eval_times_skips_auc_t_coalescer(self):
        """When eval_times is None, the AUC(t) coalescer should not run
        and no spurious 0/0 warning should fire."""
        from mlsurv.evaluation._subpopulation import compute_subpopulation_metrics

        models = ('coxph',)
        risk_per_model = {m: RISK_OK for m in models}
        eval_result = self._build_eval_result(models, Y_SUB_OK, risk_per_model)
        labels = np.array(['ok'] * len(Y_SUB_OK))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_subpopulation_metrics(
                eval_result,
                {'cohort': labels},
                Y_SUB_OK,
                min_events=1,
                y_train=Y_TRAIN,
                eval_times=None,
            )
        # No AUC(t)-flavoured MethodologicalWarning when eval_times is None.
        auc_msgs = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "AUC(t)" in str(w.message)
        ]
        assert auc_msgs == []


class TestComputeAUCTPointEstimateNaN:
    """The point-estimate path (`_compute_auc_t`, used by `evaluate()`) must
    surface a MethodologicalWarning on every NaN return path, matching the
    subpopulation `_safe_auc_t` sibling. Direct helper calls — fast tier."""

    def test_partial_out_of_range_warns_inline(self):
        """Some eval_times beyond the test follow-up → NaN at those times +
        one inline MethodologicalWarning naming the offending times."""
        from mlsurv.evaluation._concordance import _compute_auc_t

        # Y_SUB_OK runs to t~18; t=22 is above its t_max.
        eval_times = np.array([2.0, 6.0, 9.0, 22.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = _compute_auc_t(
                Y_SUB_OK, Y_TRAIN, eval_times, RISK_OK, model_name="coxph",
            )
        assert len(df) == len(eval_times)
        assert bool(df.loc[df['eval_time'] == 22.0, 'auc'].isna().all())
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        msg = str(methodo[0].message)
        assert "AUC(t) for coxph" in msg
        assert "22.0" in msg
        # The numpy "invalid value encountered in divide" RuntimeWarning must
        # stay filtered — only the MethodologicalWarning reaches the user.
        leaked = [w for w in caught
                  if issubclass(w.category, RuntimeWarning)
                  and "invalid value encountered in divide" in str(w.message)]
        assert leaked == []

    def test_too_few_valid_times_warns_inline(self):
        """All eval_times outside the test range (<2 valid) → all-NaN + one
        inline MethodologicalWarning."""
        from mlsurv.evaluation._concordance import _compute_auc_t

        eval_times = np.array([10.0, 20.0, 30.0])  # all above Y_SUB_SHORT t_max
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = _compute_auc_t(
                Y_SUB_SHORT, Y_TRAIN, eval_times, RISK_SHORT,
                model_name="coxph",
            )
        assert len(df) == 3
        assert bool(df['auc'].isna().all())
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        assert "0/3" in str(methodo[0].message)

    def test_collector_records_instead_of_warning(self):
        """With a collector supplied, no warning fires inline; the NaN notice
        is appended as an ("auc_t", model_name, msg) tuple for parent-side
        coalescing."""
        from mlsurv.evaluation._concordance import _compute_auc_t

        collector: list = []
        eval_times = np.array([2.0, 6.0, 9.0, 22.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _compute_auc_t(
                Y_SUB_OK, Y_TRAIN, eval_times, RISK_OK, model_name="coxph",
                auc_nan_collector=collector,
            )
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert methodo == []
        assert len(collector) == 1
        metric_key, model_name, msg = collector[0]
        assert metric_key == "auc_t"
        assert model_name == "coxph"
        assert "AUC(t) for coxph" in msg

    def test_no_warning_when_all_times_valid(self):
        """All eval_times strictly inside the test range and no divide
        instability → no NaN, no warning."""
        from mlsurv.evaluation._concordance import _compute_auc_t

        eval_times = np.array([2.0, 6.0, 9.0, 12.0])  # all inside (1, 18)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = _compute_auc_t(
                Y_SUB_OK, Y_TRAIN, eval_times, RISK_OK, model_name="coxph",
            )
        if bool(df['auc'].isna().any()):
            pytest.skip("dataset produced an unexpected NaN; not this test's case")
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert methodo == []

    def test_valid_time_divide_nan_warns_inline(self, monkeypatch):
        """A NaN at a *valid* time (IPCW divide instability) must surface one
        inline MethodologicalWarning naming that time — the `_compute_auc_t`
        twin of the subpopulation divide-NaN guard, exercising the distinct
        `_concordance` message and `eval_times_arr[nan_mask]` slicing."""
        from mlsurv.evaluation._concordance import _compute_auc_t
        import mlsurv.evaluation.metrics as metrics

        def _nan_among_valid(*args, **kwargs):
            times = args[3]
            vals = np.full(len(times), 0.7)
            vals[-1] = np.nan
            return vals, 0.7

        monkeypatch.setattr(metrics, "cumulative_dynamic_auc", _nan_among_valid)

        eval_times = np.array([2.0, 6.0, 9.0, 12.0])  # all inside (1, 18)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = _compute_auc_t(
                Y_SUB_OK, Y_TRAIN, eval_times, RISK_OK, model_name="coxph",
            )
        assert len(df) == len(eval_times)
        assert bool(df.loc[df['eval_time'] == 12.0, 'auc'].isna().all())
        assert int(df['auc'].isna().sum()) == 1
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        assert "12.0" in str(methodo[0].message)

    def test_exception_path_warns_inline(self, monkeypatch):
        """A recoverable sksurv failure must warn (not vanish into a
        logger.warning)."""
        import mlsurv.evaluation._concordance as conc
        import mlsurv.evaluation.metrics as metrics

        def _boom(*args, **kwargs):
            raise ValueError("forced failure")

        # _compute_auc_t now computes via the shared helper in metrics, so
        # patch cumulative_dynamic_auc there rather than in _concordance.
        monkeypatch.setattr(metrics, "cumulative_dynamic_auc", _boom)

        eval_times = np.array([2.0, 6.0, 9.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = conc._compute_auc_t(
                Y_SUB_OK, Y_TRAIN, eval_times, RISK_OK, model_name="coxph",
            )
        assert bool(df['auc'].isna().all())
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        assert "failed with ValueError" in str(methodo[0].message)

    def test_no_train_stays_silent(self):
        """y_train=None makes AUC(t) undefined by contract (no IPCW); the
        all-NaN return is expected and stays unwarned."""
        from mlsurv.evaluation._concordance import _compute_auc_t

        eval_times = np.array([2.0, 6.0, 9.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = _compute_auc_t(
                Y_SUB_OK, None, eval_times, RISK_OK, model_name="coxph",
            )
        assert bool(df['auc'].isna().all())
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert methodo == []

    def test_default_model_name_omits_for_segment(self):
        """The model_name="" default (the boundary case) produces a message
        starting "AUC(t):" with no " for <model>" segment."""
        from mlsurv.evaluation._concordance import _compute_auc_t

        eval_times = np.array([2.0, 6.0, 9.0, 22.0])  # 22.0 beyond Y_SUB_OK
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _compute_auc_t(Y_SUB_OK, Y_TRAIN, eval_times, RISK_OK)  # default ""
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        msg = str(methodo[0].message)
        assert msg.startswith("AUC(t):")
        assert " for " not in msg


class TestCoalescedAUCNanSummary:
    """The multi-model coalesced summary names every model that returned NaN."""

    def test_format_names_affected_models(self):
        from mlsurv.evaluation.evaluate import _format_coalesced_auc_nan_warning

        collected = [
            ("auc_t", "coxph", "m1"),
            ("auc_t", "coxnet", "m2"),
            ("auc_t", "coxph", "m3"),  # duplicate model — deduped
        ]
        lines = _format_coalesced_auc_nan_warning(collected, 5)
        assert len(lines) == 1
        assert lines[0] == (
            "AUC(t) returned NaN at one or more requested time points for "
            "2/5 models: [coxph, coxnet]."
        )

    def test_format_single_model_count(self):
        from mlsurv.evaluation.evaluate import _format_coalesced_auc_nan_warning

        lines = _format_coalesced_auc_nan_warning([("auc_t", "coxph", "m")], 1)
        assert lines == [
            "AUC(t) returned NaN at one or more requested time points for "
            "1/1 models: [coxph]."
        ]

    def test_format_empty_collector(self):
        from mlsurv.evaluation.evaluate import _format_coalesced_auc_nan_warning

        assert _format_coalesced_auc_nan_warning([], 5) == []

    def test_emit_formatter_failure_falls_back_per_model(self, monkeypatch):
        """If the formatter raises, no record is dropped: one fallback warning
        plus each original per-model message is emitted."""
        import importlib
        ev = importlib.import_module("mlsurv.evaluation.evaluate")

        def _boom(*args, **kwargs):
            raise ValueError("forced format failure")

        monkeypatch.setattr(ev, "_format_coalesced_auc_nan_warning", _boom)
        collected = [("auc_t", "coxph", "msg-A"), ("auc_t", "coxnet", "msg-B")]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ev._emit_coalesced_auc_nan_warning(collected, 5, 2)
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, MethodologicalWarning)]
        # One "Failed to format..." line + one per collected record.
        assert any("Failed to format coalesced AUC(t)-NaN summary" in m
                   for m in msgs)
        assert "msg-A" in msgs and "msg-B" in msgs

    def test_emit_warns_once(self):
        from mlsurv.evaluation.evaluate import _emit_coalesced_auc_nan_warning

        collected = [("auc_t", "coxph", "m1"), ("auc_t", "coxnet", "m2")]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_coalesced_auc_nan_warning(collected, 5, 2)
        methodo = [w for w in caught
                   if issubclass(w.category, MethodologicalWarning)]
        assert len(methodo) == 1
        assert "2/5 models: [coxph, coxnet]" in str(methodo[0].message)

    def test_emit_empty_is_noop(self):
        from mlsurv.evaluation.evaluate import _emit_coalesced_auc_nan_warning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_coalesced_auc_nan_warning([], 5, 2)
        assert caught == []


class TestBootstrapIterationListEvalTimes:
    """Sites 2 and 3: list-typed eval_times must not crash the
    bootstrap-iteration AUC(t) paths."""

    def test_compute_auc_for_iteration_accepts_list(self):
        eval_times = [1, 2, 6, 9]
        rows = _compute_auc_for_iteration(
            Y_TRAIN, Y_SUB_OK, RISK_OK, eval_times,
            model_name="coxph", iteration=0,
        )
        assert rows is not None
        assert len(rows) == len(eval_times)
        # At least one finite AUC for adequately-followed subgroup.
        assert any(np.isfinite(r['auc']) for r in rows)

    def test_compute_subpopulation_auc_for_iter_accepts_list(self):
        eval_times = [1, 2, 6, 9]
        rows = _compute_subpopulation_auc_for_iter(
            Y_SUB_OK, RISK_OK, Y_TRAIN, eval_times,
            group_name="cohort_A", model_name="coxph", iteration=0,
        )
        assert len(rows) == len(eval_times)
        assert any(np.isfinite(r['auc']) for r in rows)

    def test_compute_auc_for_iteration_short_followup_returns_nan(self):
        # Short follow-up → no valid times → NaN rows, no crash.
        rows = _compute_auc_for_iteration(
            Y_TRAIN, Y_SUB_SHORT, RISK_SHORT, [1, 2, 6, 9, 12, 18],
            model_name="coxph", iteration=0,
        )
        assert rows is not None
        assert len(rows) == 6
        assert all(np.isnan(r['auc']) for r in rows)
