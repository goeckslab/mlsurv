"""Unit tests for resolve_patient_index."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsurv.learner._patient_resolve import resolve_patient_index


class _StubPipeline:
    """Pipeline stub whose .predict returns a preset risk-score array."""

    def __init__(self, risk: np.ndarray):
        self._risk = np.asarray(risk, dtype=float)

    def predict(self, X):
        if len(X) != len(self._risk):
            raise AssertionError(
                f"stub expected {len(self._risk)} rows, got {len(X)}"
            )
        return self._risk


@pytest.fixture
def X10():
    return pd.DataFrame({"a": np.arange(10, dtype=float)})


@pytest.fixture
def asc_risk_pipeline():
    # Patient i has risk i — ascending, so argsort is identity.
    return _StubPipeline(np.arange(10, dtype=float))


def _expected_nearest_rank(pct: int, n: int) -> int:
    return int(pct / 100 * (n - 1))


@pytest.mark.parametrize(
    "spec,pct",
    [
        ("low_risk", 10),
        ("high_risk", 90),
        ("median", 50),
        ("p0", 0),
        ("p10", 10),
        ("p25", 25),
        ("p50", 50),
        ("p90", 90),
        ("p100", 100),
        ("P25", 25),  # case-insensitive
        ("HIGH_RISK", 90),
    ],
)
def test_string_specs_match_manual_nearest_rank(
    spec, pct, X10, asc_risk_pipeline,
):
    expected = _expected_nearest_rank(pct, len(X10))
    assert resolve_patient_index(spec, X10, asc_risk_pipeline) == expected


def test_low_risk_equals_p10(X10, asc_risk_pipeline):
    assert (
        resolve_patient_index("low_risk", X10, asc_risk_pipeline)
        == resolve_patient_index("p10", X10, asc_risk_pipeline)
    )


def test_high_risk_equals_p90(X10, asc_risk_pipeline):
    assert (
        resolve_patient_index("high_risk", X10, asc_risk_pipeline)
        == resolve_patient_index("p90", X10, asc_risk_pipeline)
    )


def test_median_equals_p50(X10, asc_risk_pipeline):
    assert (
        resolve_patient_index("median", X10, asc_risk_pipeline)
        == resolve_patient_index("p50", X10, asc_risk_pipeline)
    )


def test_picks_lowest_risk_row_when_unsorted():
    # Patient at index 3 has the lowest risk.
    X = pd.DataFrame({"a": range(5)})
    pipeline = _StubPipeline(np.array([5.0, 2.0, 4.0, 0.5, 3.0]))
    assert resolve_patient_index("low_risk", X, pipeline) == 3


def test_picks_highest_risk_row_when_unsorted():
    X = pd.DataFrame({"a": range(5)})
    pipeline = _StubPipeline(np.array([5.0, 2.0, 4.0, 0.5, 3.0]))
    # argsort -> [3,1,4,2,0]; p90 nearest rank = order[int(0.9*4)] = order[3] = 2.
    assert resolve_patient_index("high_risk", X, pipeline) == 2


def test_integer_path_unchanged(X10, asc_risk_pipeline):
    assert resolve_patient_index(0, X10, asc_risk_pipeline) == 0
    assert resolve_patient_index(7, X10, asc_risk_pipeline) == 7


def test_integer_out_of_range_raises(X10, asc_risk_pipeline):
    with pytest.raises(ValueError, match="out of range"):
        resolve_patient_index(10, X10, asc_risk_pipeline)
    with pytest.raises(ValueError, match="out of range"):
        resolve_patient_index(-1, X10, asc_risk_pipeline)


def test_numpy_integer_accepted(X10, asc_risk_pipeline):
    assert resolve_patient_index(np.int64(3), X10, asc_risk_pipeline) == 3


def test_bool_rejected(X10, asc_risk_pipeline):
    with pytest.raises(TypeError, match="bool"):
        resolve_patient_index(True, X10, asc_risk_pipeline)
    with pytest.raises(TypeError, match="bool"):
        resolve_patient_index(False, X10, asc_risk_pipeline)


def test_unrecognized_string_raises(X10, asc_risk_pipeline):
    with pytest.raises(ValueError, match="not recognized"):
        resolve_patient_index("nope", X10, asc_risk_pipeline)
    with pytest.raises(ValueError, match="not recognized"):
        resolve_patient_index("p", X10, asc_risk_pipeline)
    with pytest.raises(ValueError, match="not recognized"):
        resolve_patient_index("p150", X10, asc_risk_pipeline)


def test_non_finite_risk_raises(X10):
    pipeline = _StubPipeline(
        np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    )
    with pytest.raises(ValueError, match="non-finite"):
        resolve_patient_index("high_risk", X10, pipeline)


def test_inf_risk_raises(X10):
    pipeline = _StubPipeline(
        np.array([1.0, 2.0, np.inf, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    )
    with pytest.raises(ValueError, match="non-finite"):
        resolve_patient_index("low_risk", X10, pipeline)


def test_empty_X_raises(asc_risk_pipeline):
    with pytest.raises(ValueError, match="empty"):
        resolve_patient_index(0, pd.DataFrame({"a": []}), asc_risk_pipeline)
    with pytest.raises(ValueError, match="empty"):
        resolve_patient_index(
            "high_risk", pd.DataFrame({"a": []}), asc_risk_pipeline,
        )


def test_other_type_raises(X10, asc_risk_pipeline):
    with pytest.raises(TypeError, match="int or str"):
        resolve_patient_index(1.5, X10, asc_risk_pipeline)
    with pytest.raises(TypeError, match="int or str"):
        resolve_patient_index(None, X10, asc_risk_pipeline)


def test_single_row_cohort_all_percentiles_return_zero():
    """With len(X)==1, nearest-rank int(pct/100 * 0) == 0 for every pct."""
    X = pd.DataFrame({"a": [1.0]})
    pipeline = _StubPipeline(np.array([42.0]))
    for spec in ["low_risk", "high_risk", "median", "p0", "p50", "p100"]:
        assert resolve_patient_index(spec, X, pipeline) == 0


def test_ties_in_risk_use_stable_argsort_order():
    """np.argsort is stable — duplicate risks keep original order."""
    X = pd.DataFrame({"a": range(5)})
    # All identical risk: argsort returns [0, 1, 2, 3, 4].
    pipeline = _StubPipeline(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    assert resolve_patient_index("low_risk", X, pipeline) == 0
    assert resolve_patient_index("high_risk", X, pipeline) == 3
    assert resolve_patient_index("median", X, pipeline) == 2


def test_arg_name_in_error_messages(X10, asc_risk_pipeline):
    with pytest.raises(ValueError, match="patient_index"):
        resolve_patient_index(99, X10, asc_risk_pipeline, arg_name="patient_index")
    with pytest.raises(ValueError, match="patient_index"):
        resolve_patient_index(
            "bogus", X10, asc_risk_pipeline, arg_name="patient_index",
        )
