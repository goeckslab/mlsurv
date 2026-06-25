"""Coalesce per-(subpop × model × reference × metric) NaN
bootstrap-drop warnings emitted by ``benchmark()``.

Mirrors the calibration coalescing: when a collector is supplied,
``_build_incremental_summary`` accumulates structured records instead of
emitting one ``MethodologicalWarning`` per call; the caller emits one
summary per metric afterward via
``_emit_coalesced_incremental_nan_warning``. Direct callers of
``bootstrap_incremental`` (no collector) still see inline per-metric
warnings — no behavior change.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlsurv._warnings import MethodologicalWarning
from mlsurv.evaluation import _incremental as inc_mod
from mlsurv.evaluation._incremental import (
    _build_incremental_summary,
    _emit_coalesced_incremental_nan_warning,
    _format_coalesced_incremental_nan_warning,
    bootstrap_incremental,
)


# ---- helpers ---------------------------------------------------------------

def _degenerate_y(n=10, all_events=True):
    """All-event (or all-censored) y forces NRI/IDI to NaN on every
    resample (n_events == 0 OR n_nonevents == 0 each draw)."""
    event = np.full(n, all_events, dtype=bool)
    time = np.arange(1, n + 1, dtype=float)
    return np.array(
        list(zip(event, time)),
        dtype=[("event", bool), ("time", float)],
    )


# ---- collector routing -----------------------------------------------------

class TestCollectorRouting:
    """``_build_incremental_summary`` routes NaN-drop records to the
    collector when provided; inline ``warnings.warn`` only fires when
    ``collector is None``."""

    def test_collector_receives_records_no_inline_warning(self):
        point = {"nri": 0.1, "idi": 0.05}
        results = {
            "nri": [np.nan, np.nan, np.nan, 0.1, 0.2],
            "idi": [np.nan, 0.05, 0.06, 0.04, 0.05],
        }
        collector: list = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build_incremental_summary(
                point, results, ci=0.95,
                collector=collector,
                context={"model": "rsf", "reference": "psa"},
            )
        meth = [w for w in caught
                if issubclass(w.category, MethodologicalWarning)]
        assert meth == [], "no inline warning when collector is provided"
        assert len(collector) == 2
        nri_rec = next(r for r in collector if r["metric"] == "nri")
        assert nri_rec["n_dropped"] == 3
        assert nri_rec["n_total"] == 5
        assert nri_rec["model"] == "rsf"
        assert nri_rec["reference"] == "psa"

    def test_no_collector_emits_inline_warning(self):
        """Regression guard: standalone ``bootstrap_incremental`` users
        still see per-call ``MethodologicalWarning`` lines."""
        point = {"nri": 0.1}
        results = {"nri": [np.nan, np.nan, 0.1, 0.2]}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build_incremental_summary(point, results, ci=0.95)
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, MethodologicalWarning)]
        assert len(msgs) == 1
        assert "non-finite bootstrap" in msgs[0]
        assert "nri" in msgs[0]

    def test_no_dropped_records_when_all_finite(self):
        collector: list = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build_incremental_summary(
                {"nri": 0.1}, {"nri": [0.1, 0.2, 0.15]}, ci=0.95,
                collector=collector,
            )
        assert collector == []
        # No spurious inline warning either.
        assert [w for w in caught
                if issubclass(w.category, MethodologicalWarning)] == []


# ---- bootstrap_incremental integration with collector ---------------------

class TestBootstrapIncrementalCollector:
    """End-to-end check that ``bootstrap_incremental`` forwards collector
    + context through to the summary builder."""

    def test_degenerate_data_populates_collector(self):
        y = _degenerate_y(n=8, all_events=True)
        m = np.linspace(0.1, 0.8, 8)
        r = np.linspace(0.2, 0.7, 8)
        collector: list = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bootstrap_incremental(
                y, m, r,
                n_bootstrap=20, ci=0.95, random_state=0,
                collector=collector,
                context={"model": "rsf", "reference": "psa",
                         "subpopulation": "ER+"},
            )
        # All NRI/IDI draws are NaN — records for both metrics expected.
        metrics = {rec["metric"] for rec in collector}
        assert {"nri", "idi"}.issubset(metrics)
        for rec in collector:
            assert rec["model"] == "rsf"
            assert rec["reference"] == "psa"
            assert rec["subpopulation"] == "ER+"
            assert rec["n_dropped"] >= 1
            assert rec["n_total"] == rec["n_dropped"]  # fully degenerate
        # Collector path suppresses inline NaN warnings; the coalesced
        # "iterations failed" MetricFailureWarning may fire
        # (unconditionally on any bootstrap exception), but that's a
        # different signal (bootstrap exceptions, not metric NaNs).
        inline_nri = [w for w in caught
                      if "non-finite bootstrap" in str(w.message)]
        assert inline_nri == []

    def test_no_collector_emits_inline_for_degenerate_data(self):
        y = _degenerate_y(n=8, all_events=True)
        m = np.linspace(0.1, 0.8, 8)
        r = np.linspace(0.2, 0.7, 8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bootstrap_incremental(
                y, m, r,
                n_bootstrap=20, ci=0.95, random_state=0,
            )
        inline_nri = [w for w in caught
                      if "non-finite bootstrap" in str(w.message)]
        # 3 metrics (delta_c_index, nri, idi); delta_c_index may not be
        # all-NaN on this fixture, but nri+idi will be. Expect ≥ 2.
        assert len(inline_nri) >= 2


# ---- formatter -------------------------------------------------------------

class TestFormatCoalescedWarning:
    def test_groups_by_metric_and_picks_worst_combo(self):
        collected = [
            {"metric": "nri", "n_dropped": 50, "n_total": 100,
             "model": "rsf", "reference": "psa",
             "subpopulation": "ER+"},
            {"metric": "nri", "n_dropped": 100, "n_total": 100,
             "model": "xgbaft", "reference": "PD-L1",
             "subpopulation": "Kato_panCancer"},
            {"metric": "idi", "n_dropped": 33, "n_total": 100,
             "model": "rsf", "reference": "psa",
             "subpopulation": None},
        ]
        lines = _format_coalesced_incremental_nan_warning(collected)
        assert len(lines) == 2  # one per metric
        nri_line = next(line for line in lines if line.startswith("nri:"))
        # Lock in the count + ratio format precisely.
        assert "nri: 2 (subpopulation × model × reference) combinations" in nri_line
        assert "worst 100/100" in nri_line
        assert "Kato_panCancer × xgbaft × PD-L1" in nri_line
        idi_line = next(line for line in lines if line.startswith("idi:"))
        assert "idi: 1 (subpopulation × model × reference) combinations" in idi_line
        assert "worst 33/100" in idi_line
        # subpopulation absent → omitted from label (no leading " × ")
        assert "rsf × psa" in idi_line
        assert "None" not in idi_line

    def test_worst_combo_picked_by_ratio_not_absolute(self):
        """A combo with smaller n_total but higher drop ratio wins."""
        collected = [
            {"metric": "nri", "n_dropped": 80, "n_total": 200,
             "model": "big_combo", "reference": "ref",
             "subpopulation": "big"},
            {"metric": "nri", "n_dropped": 10, "n_total": 10,
             "model": "small_combo", "reference": "ref",
             "subpopulation": "small"},
        ]
        lines = _format_coalesced_incremental_nan_warning(collected)
        assert len(lines) == 1
        assert "worst 10/10" in lines[0]
        assert "small × small_combo × ref" in lines[0]

    def test_falsy_but_meaningful_subpopulation_label(self):
        """An integer-coded subpopulation of 0 is real but falsy; record
        its current rendering so a regression that drops it is loud."""
        collected = [
            {"metric": "nri", "n_dropped": 5, "n_total": 10,
             "model": "rsf", "reference": "psa", "subpopulation": 0},
        ]
        line = _format_coalesced_incremental_nan_warning(collected)[0]
        # Document the current (truthy-only) behavior: 0 is omitted.
        # If this changes, update the assertion AND the formatter — both
        # are tied to whether falsy subpop labels should be preserved.
        assert "rsf × psa" in line
        # Adjacent guard: a non-falsy string label is preserved.
        collected2 = [
            {"metric": "nri", "n_dropped": 5, "n_total": 10,
             "model": "rsf", "reference": "psa", "subpopulation": "0"},
        ]
        line2 = _format_coalesced_incremental_nan_warning(collected2)[0]
        assert "0 × rsf × psa" in line2

    def test_mixed_finite_and_degenerate_metrics(self):
        """Only metrics with drops appear; fully-finite metrics produce
        no collector records (and so no summary line)."""
        point = {"nri": 0.1, "idi": 0.05}
        results = {
            "nri": [0.1, 0.2, 0.15, 0.18],            # all finite
            "idi": [np.nan, np.nan, np.nan, np.nan],  # all NaN
        }
        collector: list = []
        _build_incremental_summary(
            point, results, ci=0.95, collector=collector,
            context={"model": "rsf", "reference": "psa"},
        )
        assert len(collector) == 1
        assert collector[0]["metric"] == "idi"
        assert collector[0]["n_dropped"] == 4
        assert collector[0]["n_total"] == 4


# ---- emit ------------------------------------------------------------------

class TestEmitCoalescedWarning:
    def test_emit_one_warning_per_metric(self):
        collected = [
            {"metric": "nri", "n_dropped": 50, "n_total": 100,
             "model": "rsf", "reference": "psa", "subpopulation": None},
            {"metric": "nri", "n_dropped": 80, "n_total": 100,
             "model": "rsf", "reference": "egfr", "subpopulation": None},
            {"metric": "idi", "n_dropped": 33, "n_total": 100,
             "model": "rsf", "reference": "psa", "subpopulation": None},
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_coalesced_incremental_nan_warning(collected, stacklevel=2)
        meth = [w for w in caught
                if issubclass(w.category, MethodologicalWarning)]
        assert len(meth) == 2
        msgs = [str(w.message) for w in meth]
        assert any(m.startswith("nri:") for m in msgs)
        assert any(m.startswith("idi:") for m in msgs)

    def test_empty_collector_is_noop(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_coalesced_incremental_nan_warning([], stacklevel=2)
        assert caught == []

    @pytest.mark.parametrize("exc_cls", [
        KeyError, AttributeError, TypeError, ValueError,
        IndexError, ZeroDivisionError,
    ])
    def test_formatter_failure_falls_back_to_per_record(
        self, monkeypatch, exc_cls,
    ):
        collected = [
            {"metric": "nri", "n_dropped": 5, "n_total": 10,
             "model": "rsf", "reference": "psa", "subpopulation": None},
            {"metric": "idi", "n_dropped": 4, "n_total": 10,
             "model": "rsf", "reference": "psa", "subpopulation": None},
        ]

        def _boom(_collected):
            raise exc_cls("formatter failed")
        monkeypatch.setattr(
            inc_mod, "_format_coalesced_incremental_nan_warning", _boom,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_coalesced_incremental_nan_warning(collected, stacklevel=2)
        meth = [w for w in caught
                if issubclass(w.category, MethodologicalWarning)]
        # 1 fallback notice + 2 per-record warnings
        assert len(meth) == 3
        msgs = [str(w.message) for w in meth]
        assert any("Failed to format coalesced" in m for m in msgs)
        assert any("nri" in m and "non-finite" in m for m in msgs)
        assert any("idi" in m and "non-finite" in m for m in msgs)


# ---- exception masking guard ----------------------------------------------

class TestExceptionMasking:
    """When ``warnings.filterwarnings('error', MethodologicalWarning)`` is
    active and the metric helpers also raise, the coalesced emit inside
    ``benchmark()``'s ``finally`` block must NOT mask the primary
    exception."""

    def test_warning_as_error_inside_finally_does_not_mask_primary(self):
        """Simulate the exact sequence ``benchmark()`` uses: a primary
        exception in flight + a coalesced ``warnings.warn`` that gets
        promoted to ``MethodologicalWarning`` inside the ``finally``."""
        records = [
            {"metric": "nri", "n_dropped": 5, "n_total": 10,
             "model": "rsf", "reference": "psa", "subpopulation": None},
        ]

        class PrimaryError(RuntimeError):
            pass

        def doit():
            primary_exc_in_flight = False
            try:
                raise PrimaryError("primary")
            except BaseException:
                primary_exc_in_flight = True
                raise
            finally:
                try:
                    _emit_coalesced_incremental_nan_warning(
                        records, stacklevel=2,
                    )
                except Warning:
                    if not primary_exc_in_flight:
                        raise

        with warnings.catch_warnings():
            # Promote MethodologicalWarning → error, the strict-CI config.
            warnings.simplefilter("error", MethodologicalWarning)
            with pytest.raises(PrimaryError, match="primary"):
                doit()

    def test_warning_as_error_propagates_without_primary(self):
        """No primary exception: the warning-as-error MUST surface."""
        records = [
            {"metric": "nri", "n_dropped": 5, "n_total": 10,
             "model": "rsf", "reference": "psa", "subpopulation": None},
        ]

        def doit():
            primary_exc_in_flight = False
            try:
                pass  # succeed
            except BaseException:
                primary_exc_in_flight = True
                raise
            finally:
                try:
                    _emit_coalesced_incremental_nan_warning(
                        records, stacklevel=2,
                    )
                except Warning:
                    if not primary_exc_in_flight:
                        raise

        with warnings.catch_warnings():
            warnings.simplefilter("error", MethodologicalWarning)
            with pytest.raises(MethodologicalWarning):
                doit()


# ---- end-to-end via benchmark() -------------------------------------------

@pytest.mark.integration
def test_benchmark_emits_coalesced_warnings_not_per_combo():
    """Reproduces the vignette-7 scenario in miniature: small subpopulations
    + small bootstrap → many degenerate resamples. Without coalescing this
    test would record O(subpop × model × ref × metric) warnings; with it,
    only O(metric)."""
    from sksurv.datasets import load_gbsg2

    from mlsurv import SurvivalLearner

    X_raw, y_raw = load_gbsg2()
    X = X_raw.copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].cat.codes.astype(float)
    y = np.array([(bool(e), float(t)) for e, t in y_raw],
                 dtype=[("event", bool), ("time", float)])

    # Slice down hard so subpopulation slices are tiny → bootstrap
    # resamples often degenerate.
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), size=80, replace=False)
    X = X.iloc[idx].reset_index(drop=True)
    y = y[idx]

    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, random_state=0, n_jobs=1)
    learner.train("coxph")

    # Two references aligned with full X (benchmark slices to test side).
    refs = {
        "tsize": X["tsize"].to_numpy(dtype=float),
        "pnodes": X["pnodes"].to_numpy(dtype=float),
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        learner.benchmark(
            refs, bootstrap=True, n_bootstrap=10, random_state=0,
            verbose=False,
        )

    nan_drop_warns = [
        w for w in caught
        if issubclass(w.category, MethodologicalWarning)
        and "non-finite bootstrap" in str(w.message)
    ]
    coalesced_warns = [
        w for w in caught
        if issubclass(w.category, MethodologicalWarning)
        and "combinations had non-finite bootstrap draws" in str(w.message)
    ]
    # Coalescing: zero of the old per-combo lines, plus at most one
    # summary per metric (≤ 3 metrics: delta_c_index, nri, idi).
    assert nan_drop_warns == [], (
        "saw legacy per-combo warning(s); coalescing not applied: "
        f"{[str(w.message) for w in nan_drop_warns]}"
    )
    # At most 3 metrics (delta_c_index, nri, idi).  Tight upper bound;
    # zero is allowed because the fixture may not always degenerate.
    assert len(coalesced_warns) <= 3
