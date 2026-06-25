"""Unit tests for the IterativeWarning coalescer."""

from __future__ import annotations

import warnings

import pytest

from mlsurv._iter_warn import (
    IterativeFailure,
    IterativeNotice,
    IterativeWarningCoalescer,
    coalesce_failures,
    coalesce_notices,
)
from mlsurv._warnings import IterativeWarning


def _make_failures(specs):
    """Build IterativeFailure list from (exc_type_name, message, prefix) tuples."""
    return [IterativeFailure(t, m, p) for t, m, p in specs]


def _capture(callable_):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IterativeWarning)
        callable_()
    return [w for w in caught if issubclass(w.category, IterativeWarning)]


def test_zero_failures_emits_nothing():
    captured = _capture(lambda: coalesce_failures(
        [], n_total=10, summary_label="bootstrap",
    ))
    assert captured == []


def test_single_type_emits_three_examples_plus_summary():
    failures = _make_failures([
        ("ValueError", f"bad input #{i}", f"seed={i}") for i in range(200)
    ])
    captured = _capture(lambda: coalesce_failures(
        failures, n_total=200, summary_label="bootstrap",
        tail_msg="CIs may be less reliable.",
    ))
    # Only one distinct exc_type_name → 1 example + 1 summary = 2 warnings
    assert len(captured) == 2
    msgs = [str(w.message) for w in captured]
    assert "bad input #0" in msgs[0]
    assert "seed=0" in msgs[0]
    assert "200/200 (100%)" in msgs[1]
    assert "ValueError" in msgs[1]
    assert "all shared the same error type" in msgs[1]
    assert "CIs may be less reliable." in msgs[1]


def test_three_distinct_types_one_example_each_plus_summary():
    failures = _make_failures(
        [("ValueError", "v err", f"seed={i}") for i in range(50)]
        + [("RuntimeError", "rt err", f"seed={i}") for i in range(50, 80)]
        + [("TypeError", "ty err", f"seed={i}") for i in range(80, 100)]
    )
    captured = _capture(lambda: coalesce_failures(
        failures, n_total=100, summary_label="bootstrap",
    ))
    # 3 distinct types → 3 examples + 1 summary = 4
    assert len(captured) == 4
    msgs = [str(w.message) for w in captured]
    assert "v err" in msgs[0]
    assert "rt err" in msgs[1]
    assert "ty err" in msgs[2]
    summary = msgs[3]
    assert "100/100 (100%)" in summary
    assert "across 3 distinct error types" in summary
    assert "ValueError (n=50)" in summary
    assert "RuntimeError (n=30)" in summary
    assert "TypeError (n=20)" in summary


def test_five_distinct_types_max_emit_caps_examples():
    failures = _make_failures([
        (f"Err{i}", f"msg{i}", f"seed={i}") for i in range(5)
    ])
    captured = _capture(lambda: coalesce_failures(
        failures, n_total=5, summary_label="bootstrap", max_emit=3,
    ))
    # max_emit=3 → 3 examples + 1 summary = 4
    assert len(captured) == 4
    summary = str(captured[3].message)
    assert "5/5 (100%)" in summary
    assert "across 5 distinct error types" in summary
    # All 5 types should be listed in the summary breakdown
    for i in range(5):
        assert f"Err{i} (n=1)" in summary


def test_partial_failure_summary_shows_correct_pct():
    failures = _make_failures([
        ("ValueError", "v err", f"seed={i}") for i in range(25)
    ])
    captured = _capture(lambda: coalesce_failures(
        failures, n_total=200, summary_label="bootstrap",
    ))
    assert len(captured) == 2
    summary = str(captured[1].message)
    assert "25/200 (12%)" in summary


def test_empty_context_prefix_omits_parens():
    failures = _make_failures([("ValueError", "boom", "")])
    captured = _capture(lambda: coalesce_failures(
        failures, n_total=1, summary_label="bootstrap",
    ))
    assert len(captured) == 2
    msg = str(captured[0].message)
    assert msg == "bootstrap iteration failed: boom"


def test_coalescer_context_manager_records_and_summarizes():
    captured_holder = []

    def run():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IterativeWarning)
            with IterativeWarningCoalescer(
                summary_label="subpop", n_total=4
            ) as cz:
                for label in ["a", "b", "c", "d"]:
                    try:
                        raise ValueError(f"bad {label}")
                    except ValueError as exc:
                        cz.record(exc, prefix=f"label={label}")
            captured_holder.extend(caught)

    run()
    iter_warns = [
        w for w in captured_holder if issubclass(w.category, IterativeWarning)
    ]
    # 1 example + 1 summary
    assert len(iter_warns) == 2
    assert "bad a" in str(iter_warns[0].message)
    assert "label=a" in str(iter_warns[0].message)
    assert "4/4 (100%)" in str(iter_warns[1].message)


def test_coalescer_double_summarize_idempotent():
    cz = IterativeWarningCoalescer(summary_label="bootstrap", n_total=2)
    cz.record(ValueError("x"), prefix="seed=0")
    cz.record(ValueError("y"), prefix="seed=1")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IterativeWarning)
        cz.summarize()
        cz.summarize()  # second call is a no-op

    iter_warns = [w for w in caught if issubclass(w.category, IterativeWarning)]
    # Only the first summarize() emitted (1 example + 1 summary = 2)
    assert len(iter_warns) == 2


def test_from_exception_factory():
    try:
        raise ValueError("boom")
    except ValueError as exc:
        f = IterativeFailure.from_exception(exc, prefix="seed=7")
    assert f == IterativeFailure("ValueError", "boom", "seed=7")


def test_invalid_n_total_rejected():
    with pytest.raises(ValueError, match="n_total must be >= 1"):
        coalesce_failures([], n_total=0, summary_label="x")


def test_invalid_max_emit_rejected():
    with pytest.raises(ValueError, match="max_emit must be >= 0"):
        coalesce_failures([], n_total=1, summary_label="x", max_emit=-1)


def test_failures_exceed_n_total_rejected():
    failures = _make_failures([("ValueError", "boom", "")] * 3)
    with pytest.raises(ValueError, match="exceeds n_total"):
        coalesce_failures(failures, n_total=2, summary_label="x")


def test_coalescer_summarizes_on_propagating_exception():
    """A propagating exception must NOT discard already-recorded failures.

    Previously, ``__exit__`` only summarized when ``exc_type is None``; that
    silently dropped recorded failures whenever the loop body crashed.
    """
    captured_holder = []

    def run():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IterativeWarning)
            try:
                with IterativeWarningCoalescer(
                    summary_label="thing", n_total=4
                ) as cz:
                    cz.record(ValueError("recorded before crash"), prefix="i=0")
                    cz.record(ValueError("recorded before crash"), prefix="i=1")
                    raise RuntimeError("loop body crashed")
            except RuntimeError:
                pass
            captured_holder.extend(caught)

    run()
    iter_warns = [
        w for w in captured_holder if issubclass(w.category, IterativeWarning)
    ]
    # 1 example + 1 summary, even though the with-block raised.
    assert len(iter_warns) == 2
    assert "recorded before crash" in str(iter_warns[0].message)
    assert "2/4 (50%)" in str(iter_warns[1].message)


def test_returns_failure_count():
    failures = _make_failures([
        ("ValueError", "boom", f"seed={i}") for i in range(7)
    ])
    n = coalesce_failures(
        failures, n_total=10, summary_label="bootstrap",
    )
    assert n == 7


# ---------------------------------------------------------------------------
# Notice-mode tests
# ---------------------------------------------------------------------------


def _make_notices(specs):
    """Build IterativeNotice list from (message, group_key) tuples."""
    return [IterativeNotice(m, k) for m, k in specs]


def test_notices_zero_emits_nothing():
    captured = _capture(lambda: coalesce_notices(
        [], n_total=10, summary_label="subpop", notice_phrase="are small",
    ))
    assert captured == []


def test_notices_emit_verbatim_plus_summary():
    notices = _make_notices([
        (f"Subpopulation 'X{i}' has only {i} events (< 30).", f"X{i}")
        for i in range(10)
    ])
    captured = _capture(lambda: coalesce_notices(
        notices, n_total=12, summary_label="'stage' subpopulation",
        notice_phrase="have insufficient events (< 30)",
        tail_msg="Metric estimates may be unreliable.",
    ))
    # max_emit=3 (default) → 3 verbatim + 1 summary = 4
    assert len(captured) == 4
    msgs = [str(w.message) for w in captured]
    # First three are the literal notice messages, no preamble.
    assert msgs[0] == "Subpopulation 'X0' has only 0 events (< 30)."
    assert msgs[1] == "Subpopulation 'X1' has only 1 events (< 30)."
    assert msgs[2] == "Subpopulation 'X2' has only 2 events (< 30)."
    summary = msgs[3]
    assert "10/12" in summary
    assert "(83%)" in summary
    assert "'stage' subpopulations have insufficient events (< 30)" in summary
    # All group_keys should be listed in summary.
    for i in range(10):
        assert f"X{i}" in summary
    assert "Metric estimates may be unreliable." in summary


def test_notices_max_emit_caps_examples():
    notices = _make_notices([(f"msg{i}", f"k{i}") for i in range(5)])
    captured = _capture(lambda: coalesce_notices(
        notices, n_total=5, summary_label="thing", notice_phrase="happened",
        max_emit=2,
    ))
    assert len(captured) == 3  # 2 examples + 1 summary
    assert str(captured[0].message) == "msg0"
    assert str(captured[1].message) == "msg1"


def test_notices_summary_omits_empty_keys_section():
    notices = _make_notices([("notice text", ""), ("other notice", "")])
    captured = _capture(lambda: coalesce_notices(
        notices, n_total=4, summary_label="thing", notice_phrase="happened",
    ))
    assert len(captured) == 3  # 2 examples + 1 summary
    summary = str(captured[2].message)
    # All-empty group_keys → no ": <keys>" section, just trailing period.
    assert summary == "2/4 (50%) things happened."


def test_notices_invalid_n_total_rejected():
    with pytest.raises(ValueError, match="n_total must be >= 1"):
        coalesce_notices([], n_total=0, summary_label="x", notice_phrase="y")


def test_notices_invalid_max_emit_rejected():
    with pytest.raises(ValueError, match="max_emit must be >= 0"):
        coalesce_notices(
            [], n_total=1, summary_label="x", notice_phrase="y", max_emit=-1,
        )


def test_notices_exceed_n_total_rejected():
    notices = _make_notices([("m", "k")] * 3)
    with pytest.raises(ValueError, match="exceeds n_total"):
        coalesce_notices(notices, n_total=2, summary_label="x", notice_phrase="y")


def test_coalescer_notice_mode_records_and_summarizes():
    captured_holder = []

    def run():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IterativeWarning)
            with IterativeWarningCoalescer(
                mode="notices",
                summary_label="'sex' subpopulation",
                notice_phrase="have insufficient events (< 30)",
                n_total=4,
            ) as cz:
                for label, n in [("a", 5), ("b", 7), ("c", 12), ("d", 18)]:
                    cz.record_notice(
                        f"Subpopulation '{label}' has only {n} events (< 30).",
                        group_key=label,
                    )
            captured_holder.extend(caught)

    run()
    iter_warns = [
        w for w in captured_holder if issubclass(w.category, IterativeWarning)
    ]
    # 3 verbatim + 1 summary
    assert len(iter_warns) == 4
    assert "Subpopulation 'a'" in str(iter_warns[0].message)
    summary = str(iter_warns[3].message)
    assert "4/4 (100%)" in summary
    assert "'sex' subpopulations have insufficient events (< 30)" in summary


def test_coalescer_notice_mode_requires_phrase():
    with pytest.raises(ValueError, match="notice_phrase is required"):
        IterativeWarningCoalescer(
            mode="notices", summary_label="x", n_total=1,
        )


def test_coalescer_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        IterativeWarningCoalescer(
            mode="bogus", summary_label="x", n_total=1, notice_phrase="y",
        )


def test_record_in_notice_mode_raises():
    cz = IterativeWarningCoalescer(
        mode="notices", summary_label="x", n_total=1, notice_phrase="y",
    )
    with pytest.raises(RuntimeError, match="record\\(\\) requires mode='failures'"):
        cz.record(ValueError("boom"))


def test_record_failure_in_notice_mode_raises():
    cz = IterativeWarningCoalescer(
        mode="notices", summary_label="x", n_total=1, notice_phrase="y",
    )
    with pytest.raises(RuntimeError, match="record_failure\\(\\) requires"):
        cz.record_failure(IterativeFailure("V", "m", ""))


def test_record_notice_display_overrides_group_key_in_summary():
    """display= drives the summary "affected items" list; group_key alone falls back."""
    captured_holder = []

    def run():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IterativeWarning)
            with IterativeWarningCoalescer(
                mode="notices",
                summary_label="thing",
                notice_phrase="failed",
                n_total=2,
                max_emit=0,
            ) as cz:
                cz.record_notice("first", group_key="A", display="A (42)")
                cz.record_notice("second", group_key="B")
            captured_holder.extend(caught)

    run()
    iter_warns = [
        w for w in captured_holder if issubclass(w.category, IterativeWarning)
    ]
    assert len(iter_warns) == 1
    summary = str(iter_warns[0].message)
    assert "A (42)" in summary
    assert ", B." in summary  # fallback path (display empty → group_key used)
    assert summary.index("A (42)") < summary.index(", B.")  # encounter order preserved


def test_record_notice_in_failure_mode_raises():
    cz = IterativeWarningCoalescer(summary_label="x", n_total=1)
    with pytest.raises(RuntimeError, match="record_notice\\(\\) requires"):
        cz.record_notice("hi", group_key="k")


def test_notice_mode_summarizes_on_propagating_exception():
    captured_holder = []

    def run():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IterativeWarning)
            try:
                with IterativeWarningCoalescer(
                    mode="notices",
                    summary_label="thing",
                    notice_phrase="were small",
                    n_total=4,
                ) as cz:
                    cz.record_notice("first notice", group_key="a")
                    cz.record_notice("second notice", group_key="b")
                    raise RuntimeError("loop body crashed")
            except RuntimeError:
                pass
            captured_holder.extend(caught)

    run()
    iter_warns = [
        w for w in captured_holder if issubclass(w.category, IterativeWarning)
    ]
    # 2 verbatim + 1 summary even though loop body raised
    assert len(iter_warns) == 3
    assert "first notice" in str(iter_warns[0].message)
    assert "2/4 (50%)" in str(iter_warns[2].message)


def test_notice_mode_returns_count():
    n = coalesce_notices(
        _make_notices([(f"m{i}", f"k{i}") for i in range(7)]),
        n_total=10, summary_label="x", notice_phrase="happened",
    )
    assert n == 7


# ---------------------------------------------------------------------------
# Stacklevel attribution (the +2 bump in summarize() is load-bearing)
# ---------------------------------------------------------------------------


def _emit_via_context_manager(mode):
    """Helper used so the warning's caller-frame is THIS function's body."""
    if mode == "failures":
        with IterativeWarningCoalescer(
            summary_label="thing", n_total=2, stacklevel=3,
        ) as cz:
            try:
                raise ValueError("boom")
            except ValueError as exc:
                cz.record(exc)
    else:
        with IterativeWarningCoalescer(
            mode="notices", summary_label="thing", n_total=2,
            notice_phrase="happened", stacklevel=3,
        ) as cz:
            cz.record_notice("a notice", group_key="k")


@pytest.mark.parametrize("mode", ["failures", "notices"])
def test_context_manager_warning_attributes_to_caller(mode):
    """The +2 stacklevel bump in summarize() should make warnings point at
    the caller of the with-block, NOT at _iter_warn.py internals."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IterativeWarning)
        _emit_via_context_manager(mode)
    iter_warns = [w for w in caught if issubclass(w.category, IterativeWarning)]
    assert iter_warns, "expected at least one IterativeWarning"
    for w in iter_warns:
        assert not w.filename.endswith("mlsurv\\_iter_warn.py") and \
               not w.filename.endswith("mlsurv/_iter_warn.py"), (
            f"warning attributed to coalescer internals: {w.filename}:{w.lineno}"
        )
        assert w.filename.endswith("test_iter_warn.py")
