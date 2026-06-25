"""Shared abstract base class for evaluation and validation subsections."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from mlsurv.reports.sections.base import BaseSection


class _SubsectionBase(ABC):
    """Shared base for all report subsections (evaluation and validation).

    Provides ``__init__``, ``_log_error``, and ``_format_table``.  Subclass
    hierarchies add the abstract ``add()`` contract; the evaluation
    hierarchy (``_BaseSubsection``) also requires ``is_available()``,
    while the validation hierarchy (``_BaseValidationSubsection``) does not.
    """

    def __init__(self, section: BaseSection) -> None:
        self._s = section

    # ------------------------------------------------------------------
    # Convenience shortcuts
    # ------------------------------------------------------------------

    def _log_error(
        self,
        exception: BaseException,
        context: str,
        *,
        warn_user: bool = False,
        user_message: str | None = None,
    ) -> None:
        """Delegate to the parent section's recoverable-exception logger."""
        self._s._log_recoverable_exception(
            exception,
            context,
            warn_user=warn_user,
            user_message=user_message,
        )

    def _format_table(self, df: pd.DataFrame, index: bool = False) -> str:
        """Delegate to the parent section's markdown table formatter."""
        return self._s._format_table(df, index=index)
