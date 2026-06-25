"""Abstract base class for evaluation subsections."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from mlsurv.reports.sections._subsection_base import _SubsectionBase

if TYPE_CHECKING:
    from mlsurv.reports.sections.evaluation._section import EvaluationSection


class _BaseSubsection(_SubsectionBase):
    """Shared interface for evaluation subsections.

    Every evaluation subsection must implement ``is_available()`` and
    ``add()``.  Convenience shortcuts ``_log_error()`` and
    ``_format_table()`` are inherited from ``_SubsectionBase``.
    """

    _s: EvaluationSection

    def __init__(self, section: EvaluationSection) -> None:
        super().__init__(section)

    @abstractmethod
    def is_available(self) -> bool:
        """Return *True* when this subsection has data to render."""
        ...

    @abstractmethod
    def add(self, md_lines: list, figures: list, figure_info: list) -> None:
        """Append markdown, figures and figure-info for this subsection."""
        ...
