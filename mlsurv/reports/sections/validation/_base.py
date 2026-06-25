"""Abstract base class for validation subsections."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from mlsurv.reports.sections._subsection_base import _SubsectionBase

if TYPE_CHECKING:
    from mlsurv.reports.sections.validation._section import ValidationSection


class _BaseValidationSubsection(_SubsectionBase):
    """Shared interface for validation subsections.

    Unlike evaluation subsections which use ``is_available()`` for section-wide
    gating, validation subsections operate *per-cohort*: the orchestrator
    iterates cohorts and calls ``add()`` with cohort-specific context.

    Convenience shortcuts ``_log_error()`` and ``_format_table()`` are
    inherited from ``_SubsectionBase``.
    """

    _s: ValidationSection

    def __init__(self, section: ValidationSection) -> None:
        super().__init__(section)

    @abstractmethod
    def add(
        self,
        md_lines: list,
        figures: list,
        figure_info: list,
        val_result,
        cohort_name: str,
        cohort_key: str,
    ) -> None:
        """Append markdown, figures and figure-info for one cohort."""
        ...
