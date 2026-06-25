"""External validation section package.

Re-exports ``ValidationSection`` so that existing imports such as
``from mlsurv.reports.sections.validation import ValidationSection``
continue to work unchanged.
"""

from mlsurv.reports.sections.validation._section import ValidationSection

__all__ = ['ValidationSection']
