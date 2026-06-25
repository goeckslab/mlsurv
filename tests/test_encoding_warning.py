"""Tests for the cp1252/ascii encoding warning at import time."""

import sys
import warnings

from mlsurv._warnings import SuggestionWarning
from mlsurv import _check_stdout_encoding


def _run_check_with_encoding(encoding):
    """Run _check_stdout_encoding with a fake stdout encoding."""
    real_stdout = sys.stdout
    try:
        sys.stdout = type("FakeStdout", (), {"encoding": encoding, "write": lambda *a: None})()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SuggestionWarning)
            _check_stdout_encoding()
        return caught
    finally:
        sys.stdout = real_stdout


def test_warns_on_cp1252():
    """SuggestionWarning fires when stdout.encoding is cp1252."""
    caught = _run_check_with_encoding("cp1252")
    encoding_warnings = [
        w for w in caught
        if issubclass(w.category, SuggestionWarning) and "cp1252" in str(w.message)
    ]
    assert len(encoding_warnings) == 1
    assert "PYTHONIOENCODING=utf-8" in str(encoding_warnings[0].message)


def test_warns_on_ascii():
    """SuggestionWarning fires when stdout.encoding is ascii."""
    caught = _run_check_with_encoding("ascii")
    encoding_warnings = [
        w for w in caught
        if issubclass(w.category, SuggestionWarning) and "ascii" in str(w.message)
    ]
    assert len(encoding_warnings) == 1


def test_no_warning_on_utf8():
    """No encoding warning when stdout.encoding is utf-8."""
    caught = _run_check_with_encoding("utf-8")
    encoding_warnings = [
        w for w in caught
        if issubclass(w.category, SuggestionWarning)
        and "encoding" in str(w.message).lower()
    ]
    assert len(encoding_warnings) == 0


def test_warns_on_us_ascii():
    """SuggestionWarning fires for US-ASCII alias."""
    caught = _run_check_with_encoding("US-ASCII")
    encoding_warnings = [
        w for w in caught
        if issubclass(w.category, SuggestionWarning) and "US-ASCII" in str(w.message)
    ]
    assert len(encoding_warnings) == 1


def test_warns_on_iso8859_1():
    """SuggestionWarning fires for iso-8859-1 (latin-1)."""
    caught = _run_check_with_encoding("iso-8859-1")
    encoding_warnings = [
        w for w in caught
        if issubclass(w.category, SuggestionWarning)
    ]
    assert len(encoding_warnings) == 1


def test_no_warning_on_empty_encoding():
    """No encoding warning when stdout.encoding is empty string."""
    caught = _run_check_with_encoding("")
    encoding_warnings = [
        w for w in caught
        if issubclass(w.category, SuggestionWarning)
        and "encoding" in str(w.message).lower()
    ]
    assert len(encoding_warnings) == 0


def test_no_warning_on_none_encoding():
    """No encoding warning when stdout has no encoding attribute."""
    real_stdout = sys.stdout
    try:
        sys.stdout = type("FakeStdout", (), {"write": lambda *a: None})()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SuggestionWarning)
            _check_stdout_encoding()
    finally:
        sys.stdout = real_stdout
    encoding_warnings = [
        w for w in caught
        if issubclass(w.category, SuggestionWarning)
        and "encoding" in str(w.message).lower()
    ]
    assert len(encoding_warnings) == 0
