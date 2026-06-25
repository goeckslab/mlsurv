"""Incremental value metrics comparing model predictions vs reference scores.

Provides delta C-index, category-free NRI, and IDI with bootstrap CIs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

from .._shared import percentile_ci
from .._warnings import MethodologicalWarning
from ..utils import SurvivalArray
from ._paired_comparison import _continuity_corrected_p


def delta_c_index(
    y: SurvivalArray,
    model_scores: np.ndarray,
    reference_scores: np.ndarray,
) -> float:
    """Difference in Harrell's C-index: C(model) - C(reference).

    Parameters
    ----------
    y : SurvivalArray
        Survival outcomes.
    model_scores : array-like
        Risk scores from the model (higher = higher risk).
    reference_scores : array-like
        Risk scores from the reference (higher = higher risk).

    Returns
    -------
    float
        Positive values indicate the model outperforms the reference.
    """
    events = y["event"]
    times = y["time"]
    c_model = concordance_index_censored(events, times, model_scores)[0]
    c_ref = concordance_index_censored(events, times, reference_scores)[0]
    return float(c_model - c_ref)


def category_free_nri(
    y: SurvivalArray,
    model_scores: np.ndarray,
    reference_scores: np.ndarray,
) -> float:
    """Category-free Net Reclassification Improvement.

    Computes the proportion of concordant reclassifications among events
    minus the proportion among non-events (Pencina et al. 2011).

    Parameters
    ----------
    y : SurvivalArray
        Survival outcomes.
    model_scores : array-like
        Model risk scores.
    reference_scores : array-like
        Reference risk scores.

    Returns
    -------
    float
        NRI value.  Positive indicates model improves reclassification
        over the reference.
    """
    from scipy.stats import rankdata

    events = y["event"].astype(bool)
    model_scores = np.asarray(model_scores, dtype=np.float64)
    reference_scores = np.asarray(reference_scores, dtype=np.float64)

    # Rank-normalize to [0, 1] so scores on different scales are comparable
    n = len(events)
    diff = rankdata(model_scores) / n - rankdata(reference_scores) / n

    # Among events: proportion moved up minus proportion moved down
    n_events = events.sum()
    if n_events == 0:
        nri_events = np.nan
    else:
        nri_events = (
            (diff[events] > 0).sum() - (diff[events] < 0).sum()
        ) / n_events

    # Among non-events: proportion moved down minus proportion moved up
    nonevents = ~events
    n_nonevents = nonevents.sum()
    if n_nonevents == 0:
        nri_nonevents = np.nan
    else:
        nri_nonevents = (
            (diff[nonevents] < 0).sum() - (diff[nonevents] > 0).sum()
        ) / n_nonevents

    return float(nri_events + nri_nonevents)


def integrated_discrimination_improvement(
    y: SurvivalArray,
    model_scores: np.ndarray,
    reference_scores: np.ndarray,
) -> float:
    """Integrated Discrimination Improvement (Pencina et al. 2008).

    IDI = (mean model score among events - mean model score among non-events)
        - (mean reference score among events - mean reference score among non-events)

    Scores are rank-normalised to [0, 1] before computing means so that
    values are comparable across different score scales.

    Parameters
    ----------
    y : SurvivalArray
        Survival outcomes.
    model_scores : array-like
        Model risk scores.
    reference_scores : array-like
        Reference risk scores.

    Returns
    -------
    float
        Positive indicates the model has better discrimination.
    """
    from scipy.stats import rankdata

    events = y["event"].astype(bool)
    n = len(events)

    m = rankdata(np.asarray(model_scores)) / n
    r = rankdata(np.asarray(reference_scores)) / n

    if events.sum() == 0 or (~events).sum() == 0:
        return np.nan

    disc_model = m[events].mean() - m[~events].mean()
    disc_ref = r[events].mean() - r[~events].mean()
    return float(disc_model - disc_ref)


def format_p_sig(p: float) -> str:
    """Format p-value with significance stars.

    Stars: ``*`` p < 0.05, ``**`` p < 0.01, ``***`` p < 0.001.
    Returns the value without stars if *p* is outside [0, 1].
    """
    if np.isnan(p):
        return "\u2014"
    if not (0.0 <= p <= 1.0):
        return f"{p:.2e}"
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        stars = ""
    return f"{p:.2e}{stars}"


def _validate_incremental_inputs(y, model_scores, reference_scores):
    """Validate and coerce inputs for incremental metrics."""
    model_scores = np.asarray(model_scores, dtype=np.float64).ravel()
    reference_scores = np.asarray(reference_scores, dtype=np.float64).ravel()
    n = len(y)
    if len(model_scores) != n:
        raise ValueError(
            f"model_scores length ({len(model_scores)}) != y length ({n})"
        )
    if len(reference_scores) != n:
        raise ValueError(
            f"reference_scores length ({len(reference_scores)}) != "
            f"y length ({n})"
        )
    return model_scores, reference_scores


def compute_incremental_metrics(
    y: SurvivalArray,
    model_scores: np.ndarray,
    reference_scores: np.ndarray,
) -> dict:
    """Compute all incremental metrics for one model-reference pair.

    Parameters
    ----------
    y : SurvivalArray
        Survival outcomes aligned with the score arrays.
    model_scores : np.ndarray
        Risk scores from the model under evaluation.
    reference_scores : np.ndarray
        Risk scores from the reference/baseline model.

    Returns
    -------
    dict
        ``{'delta_c_index': float, 'nri': float, 'idi': float}``
    """
    model_scores, reference_scores = _validate_incremental_inputs(
        y, model_scores, reference_scores,
    )
    return {
        "delta_c_index": delta_c_index(y, model_scores, reference_scores),
        "nri": category_free_nri(y, model_scores, reference_scores),
        "idi": integrated_discrimination_improvement(
            y, model_scores, reference_scores
        ),
    }


def _build_incremental_summary(point, results, ci, *, collector=None, context=None):
    """Build summary DataFrame from point estimates and bootstrap results.

    Parameters
    ----------
    point : dict
        {metric_name: float} point estimates.
    results : dict
        {metric_name: list[float]} bootstrap values.
    ci : float
        Confidence level.
    collector : list, optional
        When provided, non-finite-drop records (one per affected metric) are
        appended to this list as dicts instead of being emitted inline as
        :class:`MethodologicalWarning`.  Callers are responsible
        for emitting a coalesced summary afterward.
    context : dict, optional
        Extra fields merged into each collector record (e.g. ``model``,
        ``reference``, ``subpopulation``).  Ignored when ``collector`` is
        ``None``.

    Returns
    -------
    pd.DataFrame
        Columns: metric, point, mean, std, ci_lower, ci_upper, p_value.
        ``p_value`` is a two-sided bootstrap percentile p-value testing
        H0: metric = 0.
    """
    rows = []
    ctx = context or {}
    for metric_name, point_val in point.items():
        arr = np.array(results[metric_name])
        # Filter non-finite values (NaN from degenerate bootstrap samples)
        # matching the pattern in _paired_comparison.py
        finite_mask = np.isfinite(arr)
        n_dropped = int((~finite_mask).sum())
        if n_dropped > 0:
            if collector is None:
                warnings.warn(
                    f"Dropped {n_dropped}/{len(arr)} non-finite bootstrap "
                    f"iterations for {metric_name}.",
                    MethodologicalWarning,
                    stacklevel=3,
                )
            else:
                collector.append({
                    **ctx,
                    "metric": metric_name,
                    "n_dropped": n_dropped,
                    "n_total": int(len(arr)),
                })
        arr = arr[finite_mask]
        has_data = len(arr) > 0
        p_val = _continuity_corrected_p(arr) if has_data else np.nan
        if has_data:
            lo, hi = percentile_ci(arr, ci_level=ci)
        else:
            lo, hi = np.nan, np.nan
        rows.append({
            "metric": metric_name,
            "point": point_val,
            "mean": float(np.mean(arr)) if has_data else np.nan,
            "std": float(np.std(arr)) if has_data else np.nan,
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "p_value": p_val,
        })
    return pd.DataFrame(rows)


def bootstrap_incremental(
    y: SurvivalArray,
    model_scores: np.ndarray,
    reference_scores: np.ndarray,
    n_bootstrap: int = 200,
    ci: float = 0.95,
    random_state: int | None = None,
    *,
    collector: list | None = None,
    context: dict | None = None,
) -> pd.DataFrame:
    """Bootstrap CIs for incremental metrics.

    Both score arrays are fixed (no model re-fitting), so bootstrap simply
    resamples patient indices.

    Parameters
    ----------
    y : SurvivalArray
        Survival outcomes.
    model_scores : array-like
        Model risk scores.
    reference_scores : array-like
        Reference risk scores.
    n_bootstrap : int
        Number of bootstrap iterations.
    ci : float
        Confidence level (e.g. 0.95).
    random_state : int, optional
        Random seed.
    collector : list, optional
        When provided, non-finite-drop records are appended here instead of
        being emitted inline as :class:`MethodologicalWarning`; the
        caller emits a coalesced summary afterward.
    context : dict, optional
        Extra fields merged into each collector record (e.g. ``model``,
        ``reference``, ``subpopulation``). Ignored when ``collector`` is None.

    Returns
    -------
    DataFrame
        Columns: metric, point, mean, std, ci_lower, ci_upper, p_value.
        ``p_value`` is a two-sided bootstrap percentile p-value testing
        H0: metric = 0.
    """
    rng = np.random.RandomState(random_state)
    n = len(y)
    model_scores = np.asarray(model_scores, dtype=np.float64)
    reference_scores = np.asarray(reference_scores, dtype=np.float64)

    # Point estimates on full data
    point = compute_incremental_metrics(y, model_scores, reference_scores)

    # Bootstrap
    results = {k: [] for k in point}
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_boot = y[idx]
        m_boot = model_scores[idx]
        r_boot = reference_scores[idx]
        try:
            vals = compute_incremental_metrics(y_boot, m_boot, r_boot)
            for k in results:
                results[k].append(vals[k])
        except (ValueError, ZeroDivisionError):
            continue

    first_key = next(iter(point))
    n_successful = len(results[first_key])
    n_failed = n_bootstrap - n_successful
    if n_failed > 0.1 * n_bootstrap:
        warnings.warn(
            f"{n_failed}/{n_bootstrap} bootstrap iterations failed "
            f"(only {n_successful} usable). CIs may be unreliable.",
            MethodologicalWarning,
            stacklevel=2,
        )

    return _build_incremental_summary(
        point, results, ci, collector=collector, context=context,
    )


def _format_coalesced_incremental_nan_warning(collected: list) -> list:
    """Return one summary line per metric for coalesced incremental NaN warns.

    ``collected`` is a list of dicts with keys ``metric``, ``n_dropped``,
    ``n_total``, and any context fields (``model``, ``reference``,
    ``subpopulation``).  Groups by ``metric`` and reports::

        nri: 14/18 combinations had non-finite bootstrap draws;
        worst 100/100 in Kato_panCancer × xgbaft × PD-L1.

    The ``subpopulation`` field is omitted from the worst-case label when
    absent (non-subpopulation incremental rows).
    """
    by_metric: dict[str, list[dict]] = {}
    for rec in collected:
        by_metric.setdefault(rec["metric"], []).append(rec)

    lines = []
    for metric, recs in by_metric.items():
        worst = max(recs, key=lambda r: r["n_dropped"] / max(r["n_total"], 1))
        parts = []
        if worst.get("subpopulation"):
            parts.append(str(worst["subpopulation"]))
        if worst.get("model"):
            parts.append(str(worst["model"]))
        if worst.get("reference"):
            parts.append(str(worst["reference"]))
        worst_label = " × ".join(parts) if parts else "(unlabeled)"
        lines.append(
            f"{metric}: {len(recs)} (subpopulation × model × "
            f"reference) combinations had non-finite bootstrap draws; "
            f"worst {worst['n_dropped']}/{worst['n_total']} in {worst_label}."
        )
    return lines


def _emit_coalesced_incremental_nan_warning(
    collected: list, stacklevel: int,
) -> None:
    """Emit one :class:`MethodologicalWarning` per metric.

    No-op when ``collected`` is empty.  On formatter failure, falls back to
    emitting one warning per collected record so no signal is silently
    dropped (mirrors the calibration coalescing).
    """
    if not collected:
        return
    try:
        lines = _format_coalesced_incremental_nan_warning(collected)
    except (TypeError, ValueError, AttributeError, KeyError,
            IndexError, ZeroDivisionError) as exc:
        warnings.warn(
            f"Failed to format coalesced incremental-NaN summary "
            f"({type(exc).__name__}: {exc}); emitting per-record warnings "
            f"instead.",
            MethodologicalWarning, stacklevel=stacklevel,
        )
        for rec in collected:
            warnings.warn(
                f"Dropped {rec['n_dropped']}/{rec['n_total']} non-finite "
                f"bootstrap iterations for {rec['metric']}.",
                MethodologicalWarning, stacklevel=stacklevel,
            )
        return
    for line in lines:
        warnings.warn(line, MethodologicalWarning, stacklevel=stacklevel)
