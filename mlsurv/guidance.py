"""Sample-size-aware model recommendation for survival analysis.

Provides ``recommend_models()`` which classifies mlsurv's 10 models into
*recommended*, *cautionary*, and *not_recommended* tiers based on sample size,
event count, and feature count.

Thresholds are evidence-based:

- Riley RD et al. (2019). Stat Med, 38(7):1276-1296.
- van der Ploeg T et al. (2014). BMC Med Res Methodol, 14:137.
- Ogundimu EO et al. (2016). J Clin Epidemiol, 76:175-182.
- Christodoulou E et al. (2019). J Clin Epidemiol, 110:12-22.
- Harrell FE (2018). "How Can ML be Reliable When the Sample is Adequate
  for Only One Feature?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from mlsurv.constants import (
    FEATURE_REDUCTION_EPV,
    LINEARITY_ASSUMPTION_MODELS,
    PH_ASSUMPTION_MODELS,
    RECOMMEND_THRESHOLDS,
    TOO_SMALL_EPV,
)
from mlsurv.models import (
    get_model_complexity,
    get_model_display_names,
    list_models,
)

# Models with no regularization that become ill-conditioned when
# n_features >= n_events (Hu et al. 2023, Annu Rev Stat Appl).
# coxnet is regularized, RSF/GBSA/deep models handle high-dim natively.
_HIGH_DIM_INFEASIBLE = ("coxph", "weibull_aft")

Tier = Literal["recommended", "cautionary", "not_recommended"]
_TIER_ORDER: Tuple[Tier, ...] = ("recommended", "cautionary", "not_recommended")

_ERROR_MSG = (
    "Cannot extract event information from y. "
    "Expected structured array with 'event' field or "
    "DataFrame with 'event' column."
)


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _extract_data_characteristics(X, y) -> tuple[int, int, int]:
    """Extract (n_samples, n_events, n_features) from data arrays.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix (DataFrame or ndarray).
    y : structured array or DataFrame
        Survival outcome with 'event' field/column.

    Returns
    -------
    tuple of (int, int, int)
        ``(n_samples, n_events, n_features)``
    """
    import numpy as np

    n_samples, n_features = X.shape

    # Structured array (dtype with 'event' field)
    if hasattr(y, "dtype") and y.dtype.names and "event" in y.dtype.names:
        return n_samples, int(np.sum(y["event"])), n_features

    # DataFrame with 'event' column
    if hasattr(y, "columns") and "event" in y.columns:
        return n_samples, int(y["event"].sum()), n_features

    # Generic __getitem__ fallback (e.g., dict-like)
    if hasattr(y, "__getitem__"):
        try:
            return n_samples, int(np.sum(y["event"])), n_features
        except (KeyError, IndexError, TypeError):
            raise ValueError(_ERROR_MSG) from None

    raise ValueError(_ERROR_MSG)


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def _meets_thresholds(n_samples: int, epv: float, n_min, epv_min) -> bool:
    """Return True if n_samples and epv meet the given minimums."""
    n_ok = n_min is None or n_samples >= n_min
    return n_ok and epv >= epv_min


def _threshold_gaps(
    n_samples: int, epv: float, n_min, epv_min
) -> list[str]:
    """List human-readable descriptions of missed thresholds."""
    parts: list[str] = []
    if n_min is not None and n_samples < n_min:
        parts.append(f"N={n_samples} < {n_min}")
    if epv < epv_min:
        parts.append(f"EPV={epv:.1f} < {epv_min}")
    return parts


def _violation_demote(n_violations: int, n_features: int) -> int:
    """Tier-demotion steps for a given violation count.

    0 → no change, 1..25% of features → demote 1, >25% → demote 2.
    """
    if n_violations <= 0:
        return 0
    if n_features <= 0 or n_violations / n_features <= 0.25:
        return 1
    return 2


def _build_assumption_clause(
    name: str,
    n_features: int,
    ph_violations: Optional[List[str]],
    linearity_violations: Optional[List[str]],
) -> tuple[str, int]:
    """Build the per-model assumption text and total demotion steps.

    Returns ``(clause, demote)`` where ``clause`` is the trailing reason
    fragment (e.g. ``"3 PH + 2 linearity violations"``, ``"assumption-free"``,
    or ``""`` when no assumption commentary applies) and ``demote`` is the
    number of tier steps to drop.
    """
    assumes_ph = name in PH_ASSUMPTION_MODELS
    assumes_lin = name in LINEARITY_ASSUMPTION_MODELS

    if not assumes_ph and not assumes_lin:
        return "assumption-free", 0

    count_parts: list[str] = []   # e.g., "3 PH", "2 linearity"
    unavail_parts: list[str] = []  # e.g., "PH test unavailable"
    total_violations = 0
    demote = 0

    if assumes_ph:
        if ph_violations is None:
            unavail_parts.append("PH test unavailable")
        elif ph_violations:
            n = len(ph_violations)
            count_parts.append(f"{n} PH")
            total_violations += n
            demote += _violation_demote(n, n_features)

    if assumes_lin:
        if linearity_violations is None:
            unavail_parts.append("linearity test unavailable")
        elif linearity_violations:
            n = len(linearity_violations)
            count_parts.append(f"{n} linearity")
            total_violations += n
            demote += _violation_demote(n, n_features)

    if not count_parts and not unavail_parts:
        # Tests ran, no violations on the assumptions this model makes.
        return "", 0

    rendered: list[str] = []
    if count_parts:
        suffix = " violation" if total_violations == 1 else " violations"
        rendered.append(" + ".join(count_parts) + suffix)
    rendered.extend(unavail_parts)
    return "; ".join(rendered), demote


@dataclass(frozen=True)
class _ClassifyCtx:
    """Per-model context evaluated against ``MODEL_TIER_RULES``.

    ``n_rec``/``epv_rec`` and ``n_caut``/``epv_caut`` are the recommended
    and cautionary thresholds for the model's complexity class (resolved
    from ``RECOMMEND_THRESHOLDS``).
    """

    name: str
    n_samples: int
    n_events: int
    n_features: int
    epv: float
    n_rec: Optional[int]
    epv_rec: float
    n_caut: Optional[int]
    epv_caut: float

    @classmethod
    def build(
        cls, name: str, n_samples: int, n_events: int,
        n_features: int, epv: float,
    ) -> "_ClassifyCtx":
        complexity = get_model_complexity(name)
        if complexity not in RECOMMEND_THRESHOLDS:
            raise ValueError(
                f"Unknown complexity class '{complexity}' for model '{name}'. "
                "RECOMMEND_THRESHOLDS in constants.py must include an entry "
                f"for '{complexity}'. "
                f"Known classes: {sorted(RECOMMEND_THRESHOLDS)}."
            )
        n_rec, epv_rec, n_caut, epv_caut = RECOMMEND_THRESHOLDS[complexity]
        return cls(
            name=name, n_samples=n_samples, n_events=n_events,
            n_features=n_features, epv=epv,
            n_rec=n_rec, epv_rec=epv_rec,
            n_caut=n_caut, epv_caut=epv_caut,
        )


@dataclass(frozen=True)
class TierRule:
    """One rule in ``MODEL_TIER_RULES``: predicate → tier + reason.

    ``skip_assumptions=True`` short-circuits the assumption-violation
    clause and demotion (used for hard rules where the model is
    mathematically unfit, so PH/linearity status is moot).
    """

    predicate: Callable[[_ClassifyCtx], bool]
    tier: Tier
    reason: Callable[[_ClassifyCtx], str]
    skip_assumptions: bool = False

    def __post_init__(self) -> None:
        if self.tier not in _TIER_ORDER:
            raise ValueError(
                f"TierRule.tier must be one of {_TIER_ORDER}, got {self.tier!r}"
            )


def _gap_reason(default: str, n_min, epv_min) -> Callable[[_ClassifyCtx], str]:
    """Reason factory: render gap list against the given thresholds, or default."""
    def _factory(c: _ClassifyCtx) -> str:
        gaps = _threshold_gaps(c.n_samples, c.epv, n_min, epv_min)
        return ", ".join(gaps) if gaps else default
    return _factory


def _is_high_dim_infeasible(c: _ClassifyCtx) -> bool:
    # Hu et al. 2023: unregularized Cox-family models are ill-conditioned
    # when n_features >= n_events.
    return (
        c.name in _HIGH_DIM_INFEASIBLE
        and c.n_features > 0 and c.n_events > 0
        and c.n_features >= c.n_events
    )


# Ordered tier-classification rules. First matching rule wins.
# Adding/changing tier behavior is data-only; ``_classify_model`` does not
# need to change. Stored as a tuple so the table cannot be mutated at
# runtime — guarding against accidental .append() that would corrupt
# classification globally.
MODEL_TIER_RULES: Tuple[TierRule, ...] = (
    # 1. High-dimensionality hard rule (Cox-family without regularization).
    # The model is mathematically broken regardless of PH/linearity status,
    # so ``skip_assumptions`` suppresses the trailing clause.
    TierRule(
        predicate=_is_high_dim_infeasible,
        tier="not_recommended",
        reason=lambda c: (
            f"n_features={c.n_features} >= n_events={c.n_events}; "
            "partial likelihood ill-conditioned"
        ),
        skip_assumptions=True,
    ),
    # 2. Recommended band.
    TierRule(
        predicate=lambda c: _meets_thresholds(
            c.n_samples, c.epv, c.n_rec, c.epv_rec
        ),
        tier="recommended",
        reason=lambda c: "EPV ok",
    ),
    # 3. Cautionary band — gap is reported against the *recommended* line.
    TierRule(
        predicate=lambda c: _meets_thresholds(
            c.n_samples, c.epv, c.n_caut, c.epv_caut
        ),
        tier="cautionary",
        reason=lambda c: _gap_reason(
            "marginal sample size", c.n_rec, c.epv_rec
        )(c),
    ),
    # 4. Catch-all — gap is reported against the *cautionary* line.
    TierRule(
        predicate=lambda c: True,
        tier="not_recommended",
        reason=lambda c: _gap_reason(
            "below thresholds", c.n_caut, c.epv_caut
        )(c),
    ),
)


def _classify_model(
    name: str,
    n_samples: int,
    n_events: int,
    n_features: int,
    epv: float,
    display_names: dict[str, str],
    ph_violations: Optional[List[str]],
    linearity_violations: Optional[List[str]],
) -> tuple[str, str]:
    """Classify a single model into a recommendation tier.

    Walks ``MODEL_TIER_RULES`` to pick the base tier, then applies
    assumption-violation demotion. Returns ``(tier, reason)`` where
    ``reason`` is the short clause shown in the recommendation table.
    """
    ctx = _ClassifyCtx.build(name, n_samples, n_events, n_features, epv)
    for rule in MODEL_TIER_RULES:
        if rule.predicate(ctx):
            matched = rule
            break

    base_tier, base_reason = matched.tier, matched.reason(ctx)
    if matched.skip_assumptions:
        return base_tier, base_reason

    clause, demote = _build_assumption_clause(
        name, n_features, ph_violations, linearity_violations
    )
    # Apply graduated demotion (cap at not_recommended).
    final_idx = min(_TIER_ORDER.index(base_tier) + demote, len(_TIER_ORDER) - 1)
    reason = base_reason if not clause else f"{base_reason}; {clause}"
    return _TIER_ORDER[final_idx], reason


# ---------------------------------------------------------------------------
# Rationale builder
# ---------------------------------------------------------------------------

def _build_rationale(
    n_samples: int, n_events: int, n_features: int, epv: float
) -> str:
    """Build a human-readable rationale summary.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_events : int
        Number of events.
    n_features : int
        Number of features.
    epv : float
        Events per variable.

    Returns
    -------
    str
        Multi-line rationale with citations.
    """
    lines = [
        f"Dataset: N={n_samples}, events={n_events}, features={n_features}, "
        f"EPV={epv:.1f}",
    ]

    if epv < TOO_SMALL_EPV:
        lines.append(
            f"WARNING: EPV={epv:.1f} is extremely low (< {TOO_SMALL_EPV}). "
            "Even simple Cox regression may produce unstable estimates. "
            "Consider collecting more data or reducing features."
        )
    elif epv < FEATURE_REDUCTION_EPV:
        lines.append(
            f"Note: EPV={epv:.1f} is below 10. Feature reduction or "
            "regularization is recommended to reduce overfitting risk "
            "(van der Ploeg et al. 2014; Ogundimu et al. 2016)."
        )

    lines.append(
        "Thresholds based on van der Ploeg et al. (2014), "
        "Riley et al. (2019), Ogundimu et al. (2016), and "
        "Christodoulou et al. (2019)."
    )

    return "\n".join(lines)


def _format_feature_line(label: str, features) -> str:
    """Format a trailing 'Non-PH features (N): a, b, c (and N-5 more)' line."""
    n = len(features)
    head = ", ".join(features[:5])
    suffix = f" (and {n - 5} more)" if n > 5 else ""
    return f"{label} ({n}): {head}{suffix}"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RecommendedModels:
    """Result of ``recommend_models()``.

    Attributes
    ----------
    recommended : list of str
        Models safe to use with this dataset.
    cautionary : list of str
        Models that may work but carry higher overfitting risk.
    not_recommended : list of str
        Models with insufficient data for reliable training.
    rationale : str
        Summary paragraph with dataset stats and citations.
    n_samples : int
        Number of samples.
    n_events : int
        Number of events.
    n_features : int
        Number of features.
    epv : float
        Events per variable.
    details : dict
        Mapping of model name to reason string.
    ph_test_result : DataFrame or None
        Full feature-level proportional-hazards test output from the
        setup-time run (columns: feature, test_statistic, p_value,
        p_adjusted, significant). ``None`` when the test was
        skipped/unavailable, and ``None`` for any standalone
        ``recommend_models()`` call (only ``setup()`` attaches these
        DataFrames). This is the setup-time snapshot — a later
        ``learner.ph_test()`` call does not refresh it.
    linearity_test_result : DataFrame or None
        Full feature-level log-hazard linearity test output from the
        setup-time run, same columns and semantics as ``ph_test_result``.
    """

    recommended: List[str]
    cautionary: List[str]
    not_recommended: List[str]
    rationale: str
    n_samples: int
    n_events: int
    n_features: int
    epv: float
    details: Dict[str, str] = field(default_factory=dict)
    ph_violations: Tuple[str, ...] = ()
    linearity_violations: Tuple[str, ...] = ()
    # ``True`` when the corresponding test was actually run (either by the
    # caller or auto-run from data); ``False`` when violations were not
    # supplied and the test was skipped/failed. Used by the renderer to
    # distinguish "ran clean" from "didn't run".
    ph_test_ran: bool = False
    linearity_test_ran: bool = False
    # Number of continuous features the linearity test actually examined
    # (numeric columns with ``nunique() >= 6``). ``None`` when unknown
    # (e.g., caller passed ``linearity_violations=[...]`` without supplying
    # the count). Used to make the "Linearity test: all N continuous
    # features ..." summary line precise.
    n_linearity_tested: Optional[int] = None
    # Full feature-level PH / linearity test DataFrames from the setup-time
    # run (columns: feature, test_statistic, p_value, p_adjusted,
    # significant). ``None`` when the test was skipped/unavailable, and
    # ``None`` for any standalone ``recommend_models()`` call (only
    # ``setup()`` attaches the DataFrames). Typed ``Any`` to keep
    # guidance.py's import graph pandas-free. These are the setup-time
    # snapshot; ``learner.ph_test()`` / ``linearity_test()`` refresh the live
    # ``_state.*`` fields only, not this bundle. ``compare=False`` keeps the
    # DataFrames out of the auto-generated ``__eq__`` — comparing DataFrames
    # there would raise "truth value is ambiguous".
    ph_test_result: Optional[Any] = field(default=None, compare=False)
    linearity_test_result: Optional[Any] = field(default=None, compare=False)

    def demotion_causes(self, name: str) -> Tuple[str, ...]:
        """Why ``name`` is demoted, as an ordered subset of cause keys.

        Returns any of ``('high_dim', 'size', 'ph', 'linearity')``: the
        reasons that pushed ``name`` below the *recommended* tier. Mirrors
        the classification in :func:`_classify_model` from the structured
        fields on this result, so callers (e.g. the report's Model
        Appropriateness limitation) can explain a demotion without parsing
        the human-readable ``details`` text.

        An empty tuple means ``name`` met the recommended thresholds and
        had no assumption violations (i.e. it was not demoted). A non-empty
        result only means ``name`` is below *recommended*; a single
        assumption violation on a size-OK model returns ``('ph',)`` while
        leaving it in the *cautionary* tier, so a non-empty result does not
        by itself imply ``not_recommended`` membership.

        Notes
        -----
        ``high_dim`` is mutually exclusive with the other causes: the
        high-dimensionality hard rule short-circuits assumption/size
        reasoning (``skip_assumptions``), so a high-dim model reports only
        ``('high_dim',)``. An *unavailable* assumption test never demotes a
        model, so ``ph``/``linearity`` are reported only when the test ran
        and found violations: a model in ``not_recommended`` with tests
        unavailable is therefore size- or high-dim-driven.
        """
        ctx = _ClassifyCtx.build(
            name, self.n_samples, self.n_events, self.n_features, self.epv
        )
        if _is_high_dim_infeasible(ctx):
            return ("high_dim",)

        causes: list[str] = []
        if not _meets_thresholds(
            ctx.n_samples, ctx.epv, ctx.n_rec, ctx.epv_rec
        ):
            causes.append("size")
        if name in PH_ASSUMPTION_MODELS and self.ph_violations:
            causes.append("ph")
        if name in LINEARITY_ASSUMPTION_MODELS and self.linearity_violations:
            causes.append("linearity")
        return tuple(causes)

    def _format_table(self) -> str:
        """Build a formatted recommendation table string."""
        display_names = get_model_display_names()
        tier_labels = {"recommended": "Recommended", "cautionary": "Cautionary",
                       "not_recommended": "Not recommended"}
        epv_str = f"{self.epv:.1f}" if self.epv != float("inf") else "inf"

        lines = [
            "=" * 70,
            f"Model Recommendations  "
            f"(N={self.n_samples}, events={self.n_events}, "
            f"features={self.n_features}, EPV={epv_str})",
            "=" * 70,
            f"{'Model':<16} | {'Tier':<17} | Reason",
            f"{'-' * 16}-+-{'-' * 17}-+-{'-' * 32}",
        ]

        for tier_key, tier_label in tier_labels.items():
            for name in getattr(self, tier_key):
                display = display_names.get(name, name)
                reason = self.details.get(name, "")
                lines.append(f"{display:<16} | {tier_label:<17} | {reason}")

        lines.append("=" * 70)

        epv_line = self.epv_advisory_line()
        if epv_line is not None:
            lines.append(epv_line)

        ph_line = self.ph_summary_line()
        if ph_line is not None:
            lines.append(ph_line)

        linearity_line = self.linearity_summary_line()
        if linearity_line is not None:
            lines.append(linearity_line)

        return "\n".join(lines)

    def epv_advisory_line(self) -> Optional[str]:
        """Return the EPV-threshold advisory line, or None if EPV is healthy."""
        epv_str = f"{self.epv:.1f}" if self.epv != float("inf") else "inf"
        if self.epv < TOO_SMALL_EPV:
            return (
                f"WARNING: EPV={epv_str} is extremely low (< {TOO_SMALL_EPV}). "
                "Consider collecting more data or reducing features."
            )
        if self.epv < FEATURE_REDUCTION_EPV:
            return (
                f"Note: EPV={epv_str} is below {FEATURE_REDUCTION_EPV}. "
                "Feature reduction or regularization is recommended."
            )
        return None

    def ph_summary_line(self) -> Optional[str]:
        """Return the PH test summary line, or None if the test did not run.

        Distinguishes "ran clean" (all features satisfy PH) from
        "violations found" (lists violators).
        """
        if not self.ph_test_ran:
            return None
        if self.ph_violations:
            return _format_feature_line("Non-PH features", self.ph_violations)
        return (
            f"PH test: all {self.n_features} features satisfy "
            "proportional hazards (Grambsch-Therneau, BH-FDR p ≥ 0.05)."
        )

    def linearity_summary_line(self) -> Optional[str]:
        """Return the linearity test summary line, or None if it did not run.

        ``n_linearity_tested`` may be smaller than ``n_features`` because the
        test skips low-cardinality numerics; ``None`` falls back to a
        less-precise phrasing.
        """
        if not self.linearity_test_ran:
            return None
        if self.linearity_violations:
            return _format_feature_line(
                "Log-hazard non-linear features", self.linearity_violations
            )
        count = self.n_linearity_tested
        tested_phrase = (
            f"all {count} continuous feature{'s' if count != 1 else ''}"
            if count is not None
            else "all continuous features"
        )
        return (
            f"Linearity test: {tested_phrase} pass log-hazard linearity "
            "(Harrell/rms Wald, BH-FDR p ≥ 0.05)."
        )

    def __str__(self) -> str:
        return self._format_table()

    def __repr__(self) -> str:
        epv_str = f"{self.epv:.1f}" if self.epv != float("inf") else "inf"
        return (
            f"<RecommendedModels: "
            f"{len(self.recommended)} recommended, "
            f"{len(self.cautionary)} cautionary, "
            f"{len(self.not_recommended)} not_recommended; "
            f"N={self.n_samples}, events={self.n_events}, EPV={epv_str}>"
        )


# ---------------------------------------------------------------------------
# Public API helpers
# ---------------------------------------------------------------------------

def _resolve_inputs(
    X, y,
    n_samples, n_events, n_features,
    ph_violations, linearity_violations, n_linearity_tested,
    run_assumption_tests,
):
    """Resolve dataset stats + auto-run assumption tests if needed.

    Returns the populated tuple
    ``(n_samples, n_events, n_features,
       ph_violations, linearity_violations, n_linearity_tested)``.

    Two call patterns supported (mirrors ``recommend_models``):

    * ``(X, y)`` — extract stats from data; optionally auto-run PH /
      linearity tests when caller did not supply violations.
    * keyword-only ``(n_samples, n_events, n_features)`` — pure numeric.
    """
    if X is not None and y is not None:
        n_samples, n_events, n_features = _extract_data_characteristics(X, y)
        if run_assumption_tests and (
            ph_violations is None or linearity_violations is None
        ):
            ph_violations, linearity_violations, n_linearity_tested = (
                _auto_run_assumption_tests(
                    X, y, ph_violations, linearity_violations,
                    n_linearity_tested,
                )
            )
        return (
            n_samples, n_events, n_features,
            ph_violations, linearity_violations, n_linearity_tested,
        )

    if n_samples is not None and n_events is not None and n_features is not None:
        return (
            n_samples, n_events, n_features,
            ph_violations, linearity_violations, n_linearity_tested,
        )

    raise ValueError(
        "Provide either (X, y) or all of (n_samples, n_events, n_features)."
    )


def _auto_run_assumption_tests(
    X, y, ph_violations, linearity_violations, n_linearity_tested,
):
    """Run PH/linearity tests when not supplied; return populated tuple.

    Caller-supplied lists (including ``[]``) are authoritative — only
    ``None`` triggers the auto-run for that test. The auto-PH /
    auto-linearity wiring (``_run_assumption_tests``) is the constraint
    that must remain intact.
    """
    from .learner._setup import _run_assumption_tests

    auto_ph, auto_lin, _, auto_lin_df = _run_assumption_tests(
        X, y,
        run_ph=(ph_violations is None),
        run_linearity=(linearity_violations is None),
    )
    if ph_violations is None:
        ph_violations = auto_ph
    if linearity_violations is None:
        linearity_violations = auto_lin
        # The linearity test result has one row per tested continuous
        # feature plus a GLOBAL row.
        if n_linearity_tested is None and auto_lin_df is not None:
            n_linearity_tested = int(
                (auto_lin_df["feature"] != "GLOBAL").sum()
            )
    return ph_violations, linearity_violations, n_linearity_tested


def _validate_and_compute_epv(
    n_samples: int, n_events: int, n_features: int
) -> float:
    """Validate dataset stats and compute EPV (events per variable)."""
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    if n_events < 0:
        raise ValueError(f"n_events must be non-negative, got {n_events}")
    if n_events > n_samples:
        raise ValueError(
            f"n_events ({n_events}) cannot exceed n_samples ({n_samples})"
        )
    if n_features <= 0:
        return float("inf")
    return n_events / n_features


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recommend_models(
    X=None,
    y=None,
    *,
    n_samples: Optional[int] = None,
    n_events: Optional[int] = None,
    n_features: Optional[int] = None,
    ph_violations: Optional[List[str]] = None,
    linearity_violations: Optional[List[str]] = None,
    n_linearity_tested: Optional[int] = None,
    run_assumption_tests: bool = True,
    include_unavailable: bool = False,
    verbose: bool = True,
) -> RecommendedModels:
    """Recommend survival models based on dataset characteristics.

    Two call patterns:

    1. ``recommend_models(X, y)`` — extracts stats from data.
    2. ``recommend_models(n_samples=80, n_events=20, n_features=15)`` — pure
       numeric, no data required.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features), optional
        Feature matrix.
    y : structured array or DataFrame, optional
        Survival outcome with 'event' field/column.
    n_samples : int, optional
        Number of training samples (keyword-only).
    n_events : int, optional
        Number of events (keyword-only).
    n_features : int, optional
        Number of features (keyword-only).
    ph_violations : list of str, optional
        Feature names that violate the proportional hazards assumption
        (from ``ph_test()``).  When provided (even as an empty list),
        the test is treated as run; ``None`` is treated as "test not
        available" and surfaces in the reason for PH-assuming models.
        Violations demote PH-assuming models one tier when the
        violation rate is up to 25% of features, two tiers above that.
        A trailing ``Non-PH features`` line lists the violators (first
        5 plus a count suffix).
    linearity_violations : list of str, optional
        Feature names that violate the log-hazard linearity assumption
        (from ``linearity_test()``).  Same semantics as ``ph_violations``
        but applied to models that assume linearity in continuous
        covariates (Cox family, Weibull AFT, FSSVM, XGBoost-Cox).
    n_linearity_tested : int, optional
        Number of continuous covariates actually tested for log-hazard
        linearity (may be smaller than ``n_features`` because low-cardinality
        numerics are skipped). Used to phrase the linearity summary line;
        ``None`` falls back to less-precise phrasing.
    run_assumption_tests : bool, default=True
        When ``(X, y)`` is supplied and ``ph_violations`` /
        ``linearity_violations`` are left as ``None``, run ``ph_test()``
        and ``linearity_test()`` internally so the recommendation table
        reflects assumption violations — matching the
        assumption-aware tier table that ``setup()`` produces.
        Caller-supplied lists (including empty lists) are authoritative
        and skip the auto-run for that test. Set to ``False`` to skip
        the tests entirely (useful for fast paths or when the data
        contains NaNs that have not been pre-imputed). For datasets
        with missing values, pre-impute ``X`` or use ``setup()``, which
        applies the configured imputer before running the tests.
    include_unavailable : bool, default=False
        If True, include models even if optional dependencies (e.g., torch)
        are not installed.
    verbose : bool, default=True
        If True, print a formatted recommendation table.

    Returns
    -------
    RecommendedModels
        Ranked model recommendations with rationale.

    Raises
    ------
    ValueError
        If neither (X, y) nor all keyword arguments are provided.

    Examples
    --------
    >>> from mlsurv import recommend_models
    >>> rec = recommend_models(n_samples=500, n_events=300, n_features=10)
    >>> rec.recommended
    ['coxph', 'coxnet', 'weibull_aft']
    >>> rec.epv
    30.0
    """
    (
        n_samples, n_events, n_features,
        ph_violations, linearity_violations, n_linearity_tested,
    ) = _resolve_inputs(
        X, y, n_samples, n_events, n_features,
        ph_violations, linearity_violations, n_linearity_tested,
        run_assumption_tests,
    )
    epv = _validate_and_compute_epv(n_samples, n_events, n_features)

    display_names = get_model_display_names()
    tiers: dict[str, list[str]] = {
        "recommended": [], "cautionary": [], "not_recommended": [],
    }
    details: dict[str, str] = {}

    for name in list_models(include_unavailable=include_unavailable):
        tier, reason = _classify_model(
            name, n_samples, n_events, n_features, epv, display_names,
            ph_violations, linearity_violations,
        )
        details[name] = reason
        tiers[tier].append(name)

    result = RecommendedModels(
        recommended=tiers["recommended"],
        cautionary=tiers["cautionary"],
        not_recommended=tiers["not_recommended"],
        rationale=_build_rationale(n_samples, n_events, n_features, epv),
        n_samples=n_samples,
        n_events=n_events,
        n_features=n_features,
        epv=epv,
        details=details,
        ph_violations=tuple(ph_violations) if ph_violations else (),
        linearity_violations=(
            tuple(linearity_violations) if linearity_violations else ()
        ),
        ph_test_ran=ph_violations is not None,
        linearity_test_ran=linearity_violations is not None,
        n_linearity_tested=n_linearity_tested,
    )

    if verbose:
        print(result)

    return result
