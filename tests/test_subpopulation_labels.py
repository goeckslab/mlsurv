"""Unit tests for the consolidated _resolve_subpopulation_labels helper."""

import warnings

import numpy as np
import pytest

from mlsurv._warnings import SuggestionWarning
from mlsurv.learner._evaluation import (
    _SUBPOP_LABELS_UNSET,
    _resolve_subpopulation_labels,
)


def test_bootstrap_false_returns_none():
    labels = np.array([0, 1, 0, 1])
    assert _resolve_subpopulation_labels({"a": labels}, bootstrap=False) is None


def test_empty_dict_returns_none():
    assert _resolve_subpopulation_labels({}) is None


def test_none_dict_returns_none():
    assert _resolve_subpopulation_labels(None) is None


def test_single_variable_returns_labels():
    labels = np.array([0, 1, 0, 1])
    result = _resolve_subpopulation_labels({"a": labels})
    np.testing.assert_array_equal(result, labels)


def test_multi_variable_warns_and_returns_first():
    a = np.array([0, 1, 0, 1])
    b = np.array([1, 0, 1, 0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _resolve_subpopulation_labels({"a": a, "b": b})
    np.testing.assert_array_equal(result, a)
    suggestion_warnings = [w for w in caught if issubclass(w.category, SuggestionWarning)]
    assert len(suggestion_warnings) == 1
    assert "'a'" in str(suggestion_warnings[0].message)
    assert "b" in str(suggestion_warnings[0].message)


def test_length_mismatch_returns_none_and_warns():
    labels = np.array([0, 1, 0, 1])
    y_test = np.zeros(10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _resolve_subpopulation_labels({"a": labels}, y_test=y_test)
    assert result is None
    suggestion_warnings = [w for w in caught if issubclass(w.category, SuggestionWarning)]
    assert len(suggestion_warnings) == 1
    assert "does not match" in str(suggestion_warnings[0].message)


def test_override_none_skips():
    labels = np.array([0, 1, 0, 1])
    result = _resolve_subpopulation_labels({"a": labels}, override=None)
    assert result is None


def test_override_array_replaces_dict_lookup():
    dict_labels = np.array([0, 1, 0, 1])
    override_labels = np.array([2, 3, 2, 3])
    result = _resolve_subpopulation_labels(
        {"a": dict_labels}, override=override_labels
    )
    np.testing.assert_array_equal(result, override_labels)


def test_override_length_mismatch_still_guarded():
    override_labels = np.array([0, 1, 0, 1])
    y_test = np.zeros(10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _resolve_subpopulation_labels(
            {}, override=override_labels, y_test=y_test,
        )
    assert result is None
    suggestion_warnings = [w for w in caught if issubclass(w.category, SuggestionWarning)]
    assert len(suggestion_warnings) == 1


def test_sentinel_is_distinct_from_none():
    # Passing None as override must not be confused with the UNSET sentinel.
    assert _SUBPOP_LABELS_UNSET is not None
    labels = np.array([0, 1, 0, 1])
    # UNSET → fall through to dict lookup
    result = _resolve_subpopulation_labels({"a": labels}, override=_SUBPOP_LABELS_UNSET)
    np.testing.assert_array_equal(result, labels)


def test_length_matched_labels_pass_through():
    labels = np.array([0, 1, 0, 1])
    y_test = np.zeros(4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _resolve_subpopulation_labels({"a": labels}, y_test=y_test)
    np.testing.assert_array_equal(result, labels)
    assert not [w for w in caught if issubclass(w.category, SuggestionWarning)]


@pytest.mark.parametrize("ytest_kind", ["list", "ndarray"])
def test_y_test_accepts_sequence_like(ytest_kind):
    labels = np.array([0, 1, 0, 1])
    y_test = [0, 0, 0, 0] if ytest_kind == "list" else np.zeros(4)
    result = _resolve_subpopulation_labels({"a": labels}, y_test=y_test)
    np.testing.assert_array_equal(result, labels)
