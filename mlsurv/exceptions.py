"""Public warning categories for mlsurv, modelled on :mod:`sklearn.exceptions`.

mlsurv flags methodological risks and suggestions through a tiered warning
system. Every category subclasses :class:`UserWarning`, so you control them
with the standard library :mod:`warnings` module — filter by category, the same
way you would filter scikit-learn's ``ConvergenceWarning``.

The tiers are:

* :class:`MethodologicalWarning` — a real risk to result validity (e.g. a
  violated proportional-hazards assumption, a tiny or heavily censored cohort).
* :class:`SuggestionWarning` — a helpful, non-critical nudge.
* :class:`InternalWarning` — internal diagnostic detail, suppressed by default.

:class:`IterativeWarning` is a mixin for warnings that repeat across
bootstrap / CV / tuning iterations; the concrete subclasses
(:class:`ClampWarning`, :class:`FitFailureWarning`, :class:`MetricFailureWarning`,
:class:`SmallSubpopulationWarning`) combine a tier with that mixin.

mlsurv keeps its methodological warnings loud by default — it won't hide a real
risk from you. (Suggestions are shown once per issue, and internal diagnostics
are off by default.) To quiet one tier::

    import warnings
    from mlsurv.exceptions import SuggestionWarning

    warnings.filterwarnings("ignore", category=SuggestionWarning)

See ``docs/warnings.md`` for the full filtering guide.
"""

from mlsurv._warnings import (
    ClampWarning,
    ExportSplitWarning,
    FitFailureWarning,
    InternalWarning,
    IterativeWarning,
    MethodologicalWarning,
    MetricFailureWarning,
    SmallSubpopulationWarning,
    SuggestionWarning,
    enable_internal_warnings,
)

__all__ = [
    "MethodologicalWarning",
    "SuggestionWarning",
    "InternalWarning",
    "IterativeWarning",
    "ClampWarning",
    "ExportSplitWarning",
    "FitFailureWarning",
    "MetricFailureWarning",
    "SmallSubpopulationWarning",
    "enable_internal_warnings",
]
