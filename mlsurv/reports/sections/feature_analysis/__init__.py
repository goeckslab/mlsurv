"""Feature analysis section package.

Re-exports ``FeatureAnalysisSection`` so that existing imports such as
``from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection``
continue to work unchanged.
"""

from mlsurv.reports.sections.feature_analysis._helpers import (
    CHART_EXCEPTIONS,
    _assess_correlation_severity,
    _md_escape,
)
from mlsurv.reports.sections.feature_analysis._section import FeatureAnalysisSection

__all__ = [
    'CHART_EXCEPTIONS',
    'FeatureAnalysisSection',
    '_assess_correlation_severity',
    '_md_escape',
]
