"""Tests for _SubsectionBase, _BaseSubsection, and _BaseValidationSubsection."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from mlsurv.reports.sections._subsection_base import _SubsectionBase
from mlsurv.reports.sections.evaluation._base import _BaseSubsection
from mlsurv.reports.sections.evaluation._performance import _PerformanceSubsection
from mlsurv.reports.sections.validation._base import _BaseValidationSubsection
from mlsurv.reports.sections.validation._calibration import (
    _CalibrationSubsection as _ValCalibrationSubsection,
)
from mlsurv.reports.sections.validation._discrimination import (
    _DiscriminationSubsection as _ValDiscriminationSubsection,
)


@pytest.fixture()
def mock_section():
    """Mock section with _log_recoverable_exception and _format_table."""
    section = MagicMock()
    section._log_recoverable_exception = MagicMock()
    section._format_table = MagicMock(return_value="| col |\n|-----|\n| val |")
    return section


# ======================================================================
# Shared base (_SubsectionBase)
# ======================================================================


class TestSubsectionBase:
    """_SubsectionBase is the shared root of both subsection ABCs."""

    def test_eval_base_is_subclass(self):
        assert issubclass(_BaseSubsection, _SubsectionBase)

    def test_validation_base_is_subclass(self):
        assert issubclass(_BaseValidationSubsection, _SubsectionBase)

    def test_has_no_abstract_methods(self, mock_section):
        """_SubsectionBase has no abstract methods — subclass ABCs add them."""
        instance = _SubsectionBase(mock_section)
        assert instance._s is mock_section


# ======================================================================
# Evaluation subsections
# ======================================================================


class TestBaseSubsectionABC:
    """_BaseSubsection cannot be instantiated directly."""

    def test_cannot_instantiate(self, mock_section):
        with pytest.raises(TypeError, match="abstract method"):
            _BaseSubsection(mock_section)


class TestPerformanceIsAvailable:
    """_PerformanceSubsection.is_available() always returns True."""

    def test_returns_true(self, mock_section):
        sub = _PerformanceSubsection(mock_section)
        assert sub.is_available() is True


# ======================================================================
# Validation subsections
# ======================================================================


class TestBaseValidationSubsectionABC:
    """_BaseValidationSubsection cannot be instantiated directly."""

    def test_cannot_instantiate(self, mock_section):
        with pytest.raises(TypeError, match="abstract method"):
            _BaseValidationSubsection(mock_section)


# ======================================================================
# Convenience method delegation (shared via _SubsectionBase)
# ======================================================================


class TestLogErrorDelegation:
    """_log_error() delegates to section._log_recoverable_exception()."""

    def test_delegates_with_defaults(self, mock_section):
        sub = _PerformanceSubsection(mock_section)
        exc = ValueError("test")
        sub._log_error(exc, "context")
        mock_section._log_recoverable_exception.assert_called_once_with(
            exc, "context", warn_user=False, user_message=None,
        )

    def test_delegates_with_kwargs(self, mock_section):
        sub = _PerformanceSubsection(mock_section)
        exc = RuntimeError("boom")
        sub._log_error(exc, "ctx", warn_user=True, user_message="msg")
        mock_section._log_recoverable_exception.assert_called_once_with(
            exc, "ctx", warn_user=True, user_message="msg",
        )

    def test_delegates_from_validation_subsection(self, mock_section):
        sub = _ValCalibrationSubsection(mock_section)
        exc = KeyError("missing")
        sub._log_error(exc, "val-context")
        mock_section._log_recoverable_exception.assert_called_once_with(
            exc, "val-context", warn_user=False, user_message=None,
        )


class TestFormatTableDelegation:
    """_format_table() delegates to section._format_table()."""

    def test_delegates(self, mock_section):
        sub = _PerformanceSubsection(mock_section)
        df = pd.DataFrame({"a": [1]})
        result = sub._format_table(df)
        mock_section._format_table.assert_called_once_with(df, index=False)
        assert result == "| col |\n|-----|\n| val |"

    def test_delegates_with_index(self, mock_section):
        sub = _PerformanceSubsection(mock_section)
        df = pd.DataFrame({"a": [1]})
        sub._format_table(df, index=True)
        mock_section._format_table.assert_called_once_with(df, index=True)

    def test_delegates_from_validation_subsection(self, mock_section):
        sub = _ValDiscriminationSubsection(mock_section)
        df = pd.DataFrame({"b": [2]})
        result = sub._format_table(df)
        mock_section._format_table.assert_called_once_with(df, index=False)
        assert result == "| col |\n|-----|\n| val |"
