"""Unit tests for resolve_index_from_scores (percentile vs precomputed scores).

Mirrors the resolver contract that ``PredictionResult.plot()`` relies on, the
score-array sibling of ``resolve_patient_index`` (covered in
``test_patient_resolve.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsurv._patient_select import resolve_index_from_scores

# Ascending risk 0..9 — argsort is the identity, so nearest-rank lands on the
# index equal to int(pct/100 * 9).
RISK10 = np.arange(10, dtype=float)


def _expected_nearest_rank(pct: int, n: int) -> int:
    return int(pct / 100 * (n - 1))


@pytest.mark.parametrize(
    "spec,pct",
    [
        ("low_risk", 10),
        ("high_risk", 90),
        ("median", 50),
        ("p0", 0),
        ("p25", 25),
        ("p50", 50),
        ("p90", 90),
        ("p100", 100),
        ("P25", 25),       # case-insensitive
        ("HIGH_RISK", 90),
    ],
)
def test_string_specs_match_nearest_rank(spec, pct):
    assert resolve_index_from_scores(spec, RISK10) == _expected_nearest_rank(pct, 10)


def test_alias_pNN_equivalence():
    assert (resolve_index_from_scores("low_risk", RISK10)
            == resolve_index_from_scores("p10", RISK10))
    assert (resolve_index_from_scores("high_risk", RISK10)
            == resolve_index_from_scores("p90", RISK10))
    assert (resolve_index_from_scores("median", RISK10)
            == resolve_index_from_scores("p50", RISK10))


def test_resolves_against_unsorted_scores():
    # argsort([0.5, 1.5, 0.1]) -> [2, 0, 1]
    risk = np.array([0.5, 1.5, 0.1])
    assert resolve_index_from_scores("p0", risk) == 2     # lowest risk
    assert resolve_index_from_scores("p100", risk) == 1   # highest risk


def test_int_passthrough():
    assert resolve_index_from_scores(0, RISK10) == 0
    assert resolve_index_from_scores(9, RISK10) == 9
    assert resolve_index_from_scores(np.int64(3), RISK10) == 3


def test_int_out_of_range_raises():
    with pytest.raises(ValueError):
        resolve_index_from_scores(10, RISK10)
    with pytest.raises(ValueError):
        resolve_index_from_scores(-1, RISK10)


def test_bool_rejected():
    with pytest.raises(TypeError):
        resolve_index_from_scores(True, RISK10)
    with pytest.raises(TypeError):
        resolve_index_from_scores(False, RISK10)


@pytest.mark.parametrize("bad", ["nope", "p", "p150"])
def test_unrecognized_string_raises(bad):
    with pytest.raises(ValueError):
        resolve_index_from_scores(bad, RISK10)


@pytest.mark.parametrize("bad", [1.5, None, [0]])
def test_non_int_non_str_raises(bad):
    with pytest.raises(TypeError):
        resolve_index_from_scores(bad, RISK10)


def test_empty_scores_raises():
    with pytest.raises(ValueError):
        resolve_index_from_scores(0, np.array([]))


def test_non_finite_scores_raise_for_string_spec():
    risk = np.array([0.1, np.nan, 0.3])
    with pytest.raises(ValueError):
        resolve_index_from_scores("high_risk", risk)


def test_custom_arg_name_in_error():
    with pytest.raises(ValueError, match="patient_index"):
        resolve_index_from_scores(99, RISK10, arg_name="patient_index")


def test_single_patient_boundary():
    # n=1: int(pct/100 * 0) == 0 for every spec -> always index 0.
    one = np.array([0.7])
    assert resolve_index_from_scores(0, one) == 0
    for spec in ("low_risk", "median", "high_risk", "p0", "p100"):
        assert resolve_index_from_scores(spec, one) == 0
