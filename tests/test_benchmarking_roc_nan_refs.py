"""Regression test for benchmark ROC overlay with NaN-bearing references.

`generate_report()` used to log ``ValueError: risk_scores['<ref>'] has
length M but y_test has N observations`` once per eval horizon whenever a
benchmark included a NaN-bearing reference, because the report-time ROC
loop reused the full-length ``y_test`` for references that had already
been NaN-masked to a shorter array. The fix in
``mlsurv/reports/sections/benchmarking.py:_add_roc_overlay`` applies the
per-reference NaN mask to ``y_test`` before invoking
``compute_roc_curves``.
"""
from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner


@pytest.mark.integration
def test_benchmarking_roc_handles_nan_bearing_references(gbsg2_data):
    """ROC section must not raise length-mismatch on partial-NaN refs."""
    X_raw, y_raw = gbsg2_data
    X = X_raw.copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].cat.codes.astype(float)
    y = np.array(
        [(bool(e), t) for e, t in y_raw],
        dtype=[("event", bool), ("time", float)],
    )

    rng = np.random.default_rng(42)
    ref_full = rng.normal(size=len(X))
    ref_partial = ref_full.copy()
    ref_partial[rng.random(len(X)) < 0.30] = np.nan

    learner = SurvivalLearner(X, y)
    learner.setup(feature_selector="topk", random_state=42, test_fraction=0.3)
    learner.train("coxph")
    learner.benchmark(
        references={"ref_full": ref_full, "ref_partial": ref_partial},
        n_bootstrap=2,
    )

    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    logging.getLogger("mlsurv").addHandler(handler)
    try:
        with tempfile.TemporaryDirectory() as td:
            learner.generate_report(
                output_path=os.path.join(td, "report.html"),
            )
    finally:
        logging.getLogger("mlsurv").removeHandler(handler)

    mismatch_errors = [
        r for r in records
        if "has length" in r.getMessage()
        and "y_test has" in r.getMessage()
    ]
    assert mismatch_errors == [], (
        f"ROC length-mismatch errors were logged for NaN-bearing "
        f"reference: {[r.getMessage() for r in mismatch_errors]}"
    )


def test_add_roc_overlay_masks_nan_reference_y_test(monkeypatch):
    """Unit test: ``_add_roc_overlay`` applies per-reference NaN mask.

    Stubs ``compute_roc_curves`` to capture the ``y_test`` lengths it
    receives and asserts the reference call gets the masked subset while
    the model call gets the full y_test.
    """
    from mlsurv.reports.sections import benchmarking as bm_mod

    captured: list[tuple[str, int]] = []

    def fake_compute_roc_curves(y_train, y_test, risk_scores, t):
        for name in risk_scores:
            captured.append((name, len(y_test)))
        return pd.DataFrame({
            "model": list(risk_scores.keys()),
            "fpr": [0.0] * len(risk_scores),
            "tpr": [0.0] * len(risk_scores),
            "threshold": [0.0] * len(risk_scores),
            "auc": [0.5] * len(risk_scores),
            "eval_time": [t] * len(risk_scores),
        })

    def fake_plot_roc_interactive(*args, **kwargs):
        class _Fig:
            def write_image(self, *a, **k):
                pass
        return _Fig()

    def fake_bootstrap_roc_curves(*args, **kwargs):
        return None

    import mlsurv.viz._metrics_roc as roc_mod
    monkeypatch.setattr(
        roc_mod, "compute_roc_curves", fake_compute_roc_curves,
    )
    monkeypatch.setattr(
        roc_mod, "plot_roc_interactive", fake_plot_roc_interactive,
    )
    monkeypatch.setattr(
        roc_mod, "bootstrap_roc_curves", fake_bootstrap_roc_curves,
    )

    # Build a minimal BenchmarkResult-like object + test_set with one model
    # row per sample and one (masked) reference row per valid sample.
    n = 50
    rng = np.random.default_rng(0)
    times = rng.uniform(0.5, 10, size=n)
    events = rng.binomial(1, 0.6, size=n).astype(bool)
    y_test = np.array(
        list(zip(events, times)),
        dtype=[("event", bool), ("time", float)],
    )
    ref_arr = rng.normal(size=n)
    nan_mask = np.zeros(n, dtype=bool)
    nan_mask[:15] = True
    ref_arr[nan_mask] = np.nan
    valid = np.isfinite(ref_arr)

    test_set = pd.DataFrame({
        "model_type": (
            ["coxph"] * n + ["ref_partial"] * int(valid.sum())
        ),
        "time": np.concatenate([times, times[valid]]),
        "event": np.concatenate([events, events[valid]]),
        "risk_score": np.concatenate(
            [rng.normal(size=n), ref_arr[valid]],
        ),
    })

    class _Eval:
        auc_t = pd.DataFrame({"eval_time": [1.0, 5.0], "auc": [0.6, 0.7]})
        bootstrap = None

        def __init__(self, test_set):
            self.test_set = test_set

    class _Br:
        evaluation = _Eval(test_set)
        reference_names = ["ref_partial"]
        reference_arrays = {"ref_partial": ref_arr}
        model_names = ["coxph"]

    class _Learner:
        pass

    learner = _Learner()
    learner.y_train = y_test
    learner.y_test = y_test
    learner.X_test = pd.DataFrame(index=range(n))

    section = bm_mod.BenchmarkSection.__new__(bm_mod.BenchmarkSection)
    section.learner = learner
    section.data = {}
    section._log_recoverable_exception = lambda *a, **k: None

    md_lines: list[str] = []
    figures: list = []
    figure_info: list = []
    section._add_roc_overlay(md_lines, figures, figure_info, _Br())

    n_valid = int(valid.sum())
    coxph_lens = {ln for name, ln in captured if name == "coxph"}
    ref_lens = {ln for name, ln in captured if name == "ref_partial"}
    assert coxph_lens == {n}, (
        f"model ROC call should see full y_test (n={n}), got {coxph_lens}"
    )
    assert ref_lens == {n_valid}, (
        f"reference ROC call should see masked y_test "
        f"(n_valid={n_valid}), got {ref_lens}"
    )


def _build_section_with_refs(ref_arrays, monkeypatch):
    """Construct a BenchmarkSection wired with stub br + learner.

    ``ref_arrays`` is ``{name: np.ndarray}`` — each array is full-length
    (n=50) with NaNs marking invalid samples. Returns (section, br_stub,
    captured) where ``captured`` collects (model_name, y_test_len) tuples
    seen by the stubbed ``compute_roc_curves``.
    """
    from mlsurv.reports.sections import benchmarking as bm_mod

    captured: list[tuple[str, int]] = []

    def fake_compute_roc_curves(y_train, y_test, risk_scores, t):
        for name in risk_scores:
            captured.append((name, len(y_test)))
        return pd.DataFrame({
            "model": list(risk_scores.keys()),
            "fpr": [0.0] * len(risk_scores),
            "tpr": [0.0] * len(risk_scores),
            "threshold": [0.0] * len(risk_scores),
            "auc": [0.5] * len(risk_scores),
            "eval_time": [t] * len(risk_scores),
        })

    import mlsurv.viz._metrics_roc as roc_mod
    monkeypatch.setattr(roc_mod, "compute_roc_curves", fake_compute_roc_curves)
    monkeypatch.setattr(
        roc_mod, "plot_roc_interactive",
        lambda *a, **k: type("F", (), {"write_image": lambda *a, **k: None})(),
    )
    monkeypatch.setattr(roc_mod, "bootstrap_roc_curves", lambda *a, **k: None)

    n = 50
    rng = np.random.default_rng(1)
    times = rng.uniform(0.5, 10, size=n)
    events = rng.binomial(1, 0.6, size=n).astype(bool)
    y_test = np.array(
        list(zip(events, times)),
        dtype=[("event", bool), ("time", float)],
    )

    test_set_rows = [
        ("coxph", times, events, rng.normal(size=n)),
    ]
    for rname, arr in ref_arrays.items():
        valid = np.isfinite(arr)
        test_set_rows.append(
            (rname, times[valid], events[valid], arr[valid]),
        )
    test_set = pd.DataFrame({
        "model_type": np.concatenate(
            [np.full(len(t), m) for m, t, _, _ in test_set_rows]
        ),
        "time": np.concatenate([t for _, t, _, _ in test_set_rows]),
        "event": np.concatenate([e for _, _, e, _ in test_set_rows]),
        "risk_score": np.concatenate([s for _, _, _, s in test_set_rows]),
    })

    class _Eval:
        auc_t = pd.DataFrame({"eval_time": [1.0, 5.0], "auc": [0.6, 0.7]})
        bootstrap = None

        def __init__(self, test_set):
            self.test_set = test_set

    class _Br:
        evaluation = _Eval(test_set)
        reference_names = list(ref_arrays.keys())
        reference_arrays = dict(ref_arrays)
        model_names = ["coxph"]

    class _Learner:
        pass

    learner = _Learner()
    learner.y_train = y_test
    learner.y_test = y_test
    learner.X_test = pd.DataFrame(index=range(n))

    section = bm_mod.BenchmarkSection.__new__(bm_mod.BenchmarkSection)
    section.learner = learner
    section.data = {}
    section._log_recoverable_exception = lambda *a, **k: None
    return section, _Br(), captured, n


def test_no_nan_reference_uses_full_y_test(monkeypatch):
    """A reference with zero NaNs gets the full y_test (mask is all-True)."""
    rng = np.random.default_rng(2)
    ref_clean = rng.normal(size=50)
    section, br, captured, n = _build_section_with_refs(
        {"ref_clean": ref_clean}, monkeypatch,
    )
    section._add_roc_overlay([], [], [], br)
    ref_lens = {ln for name, ln in captured if name == "ref_clean"}
    assert ref_lens == {n}, (
        f"no-NaN reference should see full y_test (n={n}), got {ref_lens}"
    )


def test_multiple_refs_do_not_cross_contaminate_masks(monkeypatch):
    """Two refs with different NaN counts must each get their own mask."""
    rng = np.random.default_rng(3)
    ref_a = rng.normal(size=50)
    ref_a[:10] = np.nan  # 40 valid
    ref_b = rng.normal(size=50)
    ref_b[:25] = np.nan  # 25 valid
    section, br, captured, n = _build_section_with_refs(
        {"ref_a": ref_a, "ref_b": ref_b}, monkeypatch,
    )
    section._add_roc_overlay([], [], [], br)
    a_lens = {ln for name, ln in captured if name == "ref_a"}
    b_lens = {ln for name, ln in captured if name == "ref_b"}
    assert a_lens == {40}, f"ref_a should see masked n=40, got {a_lens}"
    assert b_lens == {25}, f"ref_b should see masked n=25, got {b_lens}"


def test_all_nan_reference_does_not_crash(monkeypatch):
    """All-NaN reference: y_test_ref is length 0; section must not crash."""
    ref_all_nan = np.full(50, np.nan)
    # `_extract_risk_scores` would drop this all-non-finite reference upstream,
    # so we exercise the masking branch by injecting a stub test_set where
    # this ref has zero rows but still appears in `reference_arrays`.
    section, br, captured, n = _build_section_with_refs(
        {"ref_all_nan": ref_all_nan}, monkeypatch,
    )
    # Should not raise; the ref is filtered by _extract_risk_scores so the
    # per-ref loop body is skipped entirely.
    md_lines: list[str] = []
    section._add_roc_overlay(md_lines, [], [], br)
    # No reference call captured (filtered upstream by _extract_risk_scores).
    assert not any(name == "ref_all_nan" for name, _ in captured)
