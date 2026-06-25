"""Model evaluation section package.

Re-exports ``EvaluationSection`` so that existing imports such as
``from mlsurv.reports.sections.evaluation import EvaluationSection``
continue to work unchanged.
"""

from mlsurv.reports.sections.evaluation._section import EvaluationSection

__all__ = ['EvaluationSection']
