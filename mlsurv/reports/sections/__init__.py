"""Report sections for mlsurv.

13 default sections, aligned with API methods and TRIPOD+AI:

1. Executive Summary [summary]
2. Data & Preprocessing [data_preprocessing]
3. Model Development [model_development] (includes tuning + trained details)
4. Model Evaluation [evaluation]
5. Feature Analysis [feature_analysis]
6. Subpopulation Analysis [subpopulation_analysis]
7. External Validation [validation]
8. Benchmarking [benchmarking]
9. Limitations & Caveats [limitations]
10. Environment & Reproducibility [environment]
11. References [references]
12. TRIPOD+AI Checklist [tripod_checklist]
"""

from mlsurv.reports.sections.base import BaseSection, FigureInfo
from mlsurv.reports.sections.summary import SummarySection
from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection
from mlsurv.reports.sections.model_development import ModelDevelopmentSection
from mlsurv.reports.sections.evaluation import EvaluationSection
from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection
from mlsurv.reports.sections.validation import ValidationSection
from mlsurv.reports.sections.benchmarking import BenchmarkSection
from mlsurv.reports.sections.limitations import LimitationsSection
from mlsurv.reports.sections.environment import EnvironmentSection
from mlsurv.reports.sections.references import ReferencesSection
from mlsurv.reports.sections.tripod_checklist import TripodChecklistSection
from mlsurv.reports.sections.subpopulation_analysis import SubpopulationAnalysisSection

__all__ = [
    # Base classes
    'BaseSection',
    'FigureInfo',
    # Report sections
    'SummarySection',
    'DataPreprocessingSection',
    'ModelDevelopmentSection',
    'EvaluationSection',
    'FeatureAnalysisSection',
    'SubpopulationAnalysisSection',
    'ValidationSection',
    'BenchmarkSection',
    'LimitationsSection',
    'EnvironmentSection',
    'ReferencesSection',
    'TripodChecklistSection',
]
