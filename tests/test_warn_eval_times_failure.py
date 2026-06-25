"""Tests for ``mlsurv._warnings._warn_eval_times_failure``."""

import logging
import warnings

from mlsurv._warnings import SuggestionWarning, _warn_eval_times_failure


def test_dual_channel_emission(caplog):
    """Helper emits both ``logger.warning`` and ``SuggestionWarning``."""
    exc = ValueError("not enough unique event times")
    caplog.set_level(logging.WARNING, logger="mlsurv._warnings")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", SuggestionWarning)
        _warn_eval_times_failure(
            exc, "in evaluate()", "Time-dependent metrics will be NaN.",
        )

    expected = (
        "auto_eval_times failed in evaluate(): not enough unique event times. "
        "Time-dependent metrics will be NaN."
    )
    assert any(expected in rec.message for rec in caplog.records)
    suggestion = [w for w in recorded if issubclass(w.category, SuggestionWarning)]
    assert len(suggestion) == 1
    assert str(suggestion[0].message) == expected


def test_stacklevel_2_matches_pre_refactor_baseline():
    """With ``stacklevel=2`` from a wrapper, the helper attributes the warning
    to the same frame the original ``warnings.warn(..., stacklevel=2)``
    pattern did — i.e., the wrapper's caller. The helper's internal ``+1``
    compensates for its own frame.
    """
    def helper_call_site(exc):  # simulates an mlsurv API method
        _warn_eval_times_failure(
            exc, "in evaluate()", "tail.",
        )

    def baseline_call_site(exc):  # the pre-refactor pattern
        warnings.warn(
            f"auto_eval_times failed in evaluate(): {exc}. tail.",
            SuggestionWarning,
            stacklevel=2,
        )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", SuggestionWarning)
        helper_call_site(ValueError("x"))
        baseline_call_site(ValueError("x"))

    assert recorded[0].filename == recorded[1].filename
    assert recorded[0].filename.endswith("test_warn_eval_times_failure.py")


def test_stacklevel_3_matches_pre_refactor_baseline():
    """With ``stacklevel=3`` from a doubly-nested call, the helper matches the
    original ``stacklevel=3`` pattern used at the subpopulation-metrics sites.
    """
    def inner(exc):
        _warn_eval_times_failure(
            exc, "for subpopulation metrics",
            "AUC(t) subpopulation metrics will be unavailable.",
            stacklevel=3,
        )

    def outer(exc):
        inner(exc)

    def baseline_inner(exc):
        warnings.warn(
            f"auto_eval_times failed for subpopulation metrics: {exc}. "
            f"AUC(t) subpopulation metrics will be unavailable.",
            SuggestionWarning,
            stacklevel=3,
        )

    def baseline_outer(exc):
        baseline_inner(exc)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", SuggestionWarning)
        outer(ValueError("x"))
        baseline_outer(ValueError("x"))

    assert recorded[0].filename == recorded[1].filename
    assert recorded[0].filename.endswith("test_warn_eval_times_failure.py")


def test_clip_eval_times_verb():
    """``verb="clip_eval_times"`` preserves clip-only message wording."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", SuggestionWarning)
        _warn_eval_times_failure(
            ValueError("oops"), "in evaluate()",
            "Time-dependent metrics will be NaN.",
            verb="clip_eval_times",
        )
    assert str(recorded[0].message).startswith("clip_eval_times failed in evaluate(): oops.")


def test_metrics_resolve_eval_times_dedup_guard():
    """``_resolve_eval_times`` deduplicates identical failures across calls but
    still surfaces a distinct exception type/message. The helper is pure
    (stateless) — the dedup contract lives at the call site in metrics.py
    and must not regress.
    """
    import numpy as np

    from mlsurv.evaluation.metrics import (
        _eval_times_warned,
        _reset_eval_times_dedup,
        _resolve_eval_times,
    )

    # Degenerate y_train: a single unique event time -> auto_eval_times raises
    # ValueError. y is the test array; the failure path is keyed off y_train.
    y = np.array([(True, 5.0), (False, 10.0)],
                 dtype=[("event", bool), ("time", float)])
    y_train = np.array([(True, 1.0), (True, 1.0)],
                       dtype=[("event", bool), ("time", float)])

    _reset_eval_times_dedup()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", SuggestionWarning)
        # Two identical calls: only the first should warn.
        assert _resolve_eval_times(y, y_train, eval_times=None) is None
        assert _resolve_eval_times(y, y_train, eval_times=None) is None

    suggestion = [w for w in recorded if issubclass(w.category, SuggestionWarning)]
    assert len(suggestion) == 1, (
        f"dedup guard regressed: expected 1 SuggestionWarning, got {len(suggestion)}"
    )
    assert "during scoring" in str(suggestion[0].message)
    assert _eval_times_warned, "dedup key was not recorded"
