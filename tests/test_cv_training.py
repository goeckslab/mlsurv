"""Tests for CV-training fold-boundary handling.

Regression coverage: fold-boundary metric failures must emit
a user-visible ``warnings.warn`` (once per session), not just a log record.
"""

from __future__ import annotations

import warnings

import pytest

import mlsurv.logging_utils as logging_utils
from mlsurv._warnings import MethodologicalWarning, MetricFailureWarning
from mlsurv.learner._cv_training import _compute_ibs_score


@pytest.fixture(autouse=True)
def _reset_warn_once_state():
    logging_utils._seen_log_once_keys.clear()
    logging_utils._seen_warn_once_keys.clear()
    yield
    logging_utils._seen_log_once_keys.clear()
    logging_utils._seen_warn_once_keys.clear()


def _fold_boundary_scoring_fn(*_args, **_kwargs):
    raise ValueError(
        "all times must be within follow-up time of test data: [281.0; 2539.0["
    )


class TestFoldBoundaryWarning:
    def test_compute_ibs_score_emits_methodological_warning_once(self):
        """First fold-boundary failure emits a MetricFailureWarning;
        subsequent failures stay silent at the warnings layer."""
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            _compute_ibs_score(
                _fold_boundary_scoring_fn, None, None, None, None, fold_idx=0
            )
            _compute_ibs_score(
                _fold_boundary_scoring_fn, None, None, None, None, fold_idx=1
            )

        metric_warnings = [
            w for w in captured if issubclass(w.category, MetricFailureWarning)
        ]
        assert len(metric_warnings) == 1, (
            f"expected exactly one MetricFailureWarning, got {len(metric_warnings)}: "
            f"{[str(w.message) for w in metric_warnings]}"
        )
        # MetricFailureWarning is a MethodologicalWarning — satisfies the
        # No-Silent-Data-Loss rule.
        assert issubclass(metric_warnings[0].category, MethodologicalWarning)
        msg = str(metric_warnings[0].message).lower()
        # Wording from _fold_boundary_message — references the metric and the
        # partial-skip semantics, not the misleading "computation failed".
        assert "integrated_brier" in msg
        assert "skipped on at least one cv fold" in msg
        assert "computation failed" not in msg

    def test_compute_ibs_score_returns_degenerate_marker(self):
        result = _compute_ibs_score(
            _fold_boundary_scoring_fn, None, None, None, None, fold_idx=0
        )
        assert "ibs_degenerate" in result
