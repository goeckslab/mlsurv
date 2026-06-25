"""Maxstat optimal cutpoint search for KM survival stratification.

Public symbols:
    MaxstatResult, DegenerateSubpopulationError, _fast_logrank_chi2,
    _find_maxstat_cutpoint, _find_maxstat_cutpoints_recursive,
    _resolve_maxstat, _evaluate_cutpoint_candidates, _VALID_CORRECTIONS
"""

import warnings
from typing import NamedTuple, Tuple

import numpy as np

from mlsurv._warnings import InternalWarning


class DegenerateSubpopulationError(ValueError):
    """Raised when a subpopulation is too small or degenerate for maxstat splitting."""


class MaxstatResult(NamedTuple):
    """Result of a maxstat cutpoint search.

    ``cutpoint`` is always the root-level (full-data) optimal split.
    ``cutpoints`` contains all splits from recursive binary splitting,
    sorted by value.  For single-split results, ``cutpoints == (cutpoint,)``.
    ``split_stats`` contains ``(cutpoint, chi2_statistic, raw_p_value)`` for
    each split, sorted by cutpoint value.
    ``p_value_corrected`` refers to the root split only (not an omnibus test
    across all splits in multi-group mode).
    """

    cutpoint: float
    test_statistic: float
    p_value_raw: float
    p_value_corrected: float
    correction: str
    n_candidates: int
    cutpoints: Tuple[float, ...] = ()
    split_stats: Tuple[Tuple[float, float, float], ...] = ()


_VALID_CORRECTIONS = {"bonferroni", "lausen-schumacher", "permutation"}


def _fast_logrank_chi2(times1, events1, times2, events2) -> float:
    """Compute log-rank chi2 statistic without lifelines overhead.

    Uses the Mantel-Haenszel formulation: at each distinct event time,
    compares observed vs expected events in group 1.
    """
    all_times = np.concatenate([times1, times2])
    all_events = np.concatenate([events1, events2])
    group = np.concatenate([np.ones(len(times1)), np.zeros(len(times2))])

    # Sort by time
    order = np.argsort(all_times, kind="mergesort")
    t_sorted = all_times[order]
    e_sorted = all_events[order]
    g_sorted = group[order]

    # Distinct event times
    event_mask = e_sorted > 0
    event_times = np.unique(t_sorted[event_mask])
    if len(event_times) == 0:
        return 0.0

    numerator = 0.0
    variance = 0.0

    # Process each distinct event time
    for t in event_times:
        at_risk = t_sorted >= t
        n_j = int(at_risk.sum())  # Python int avoids numpy int32 overflow
        if n_j < 2:
            continue
        at_t = (t_sorted == t) & event_mask
        d_j = int(at_t.sum())
        n1_j = int((at_risk & (g_sorted == 1)).sum())
        d1_j = int((at_t & (g_sorted == 1)).sum())

        e1_j = n1_j * d_j / n_j
        numerator += d1_j - e1_j

        n2_j = n_j - n1_j
        variance += n1_j * n2_j * d_j * (n_j - d_j) / (n_j ** 2 * (n_j - 1))

    if variance <= 0:
        return 0.0
    return numerator ** 2 / variance


def _evaluate_cutpoint_candidates(times, events, scores, valid):
    """Evaluate all cutpoint candidates and return best result.

    Returns (best_stat, best_cut, best_p_raw).
    """
    from lifelines.statistics import logrank_test

    best_stat = -np.inf
    best_cut = None
    best_p_raw = 1.0
    n_failures = 0
    for cut in valid:
        mask_lo = scores <= cut
        mask_hi = ~mask_lo
        if events[mask_lo].sum() == 0 or events[mask_hi].sum() == 0:
            n_failures += 1
            continue
        try:
            result = logrank_test(
                times[mask_lo], times[mask_hi],
                events[mask_lo], events[mask_hi],
            )
        except (ValueError, ZeroDivisionError):
            n_failures += 1
            continue
        if result.test_statistic > best_stat:
            best_stat = result.test_statistic
            best_cut = cut
            best_p_raw = result.p_value
    if best_cut is None and n_failures > 0:
        warnings.warn(
            f"All {n_failures} cutpoint candidate(s) failed log-rank evaluation. "
            f"This may indicate degenerate survival data (e.g., no events in one "
            f"subpopulation at every candidate cutpoint).",
            InternalWarning,
            stacklevel=3,
        )
    return best_stat, best_cut, best_p_raw


def _bonferroni_correction(p_raw, n_candidates):
    """Apply Bonferroni correction."""
    return min(1.0, p_raw * n_candidates)


def _lausen_schumacher_correction(best_stat, min_prop, max_prop):
    """Apply Lausen-Schumacher (1992) p-value correction."""
    from scipy.stats import norm

    z = np.sqrt(best_stat)
    if z <= 0:
        return 1.0
    log_term = np.log((1 - min_prop) * max_prop / (min_prop * (1.0 - max_prop)))
    D = norm.pdf(z) * (z - 1.0 / z) * log_term
    raw_two_sided = 2.0 * (1.0 - norm.cdf(z))
    if D < 0:
        return 1.0
    p = 1.0 - (1.0 - raw_two_sided) * np.exp(-D / 2.0)
    return float(np.clip(p, 0.0, 1.0))


def _permutation_correction(times, events, scores, valid,
                             n_permutations, random_state):
    """Compute permutation-based p-value correction."""
    masks_lo = [scores <= cut for cut in valid]

    def _max_chi2(t, e):
        best = -np.inf
        for m_lo in masks_lo:
            m_hi = ~m_lo
            if e[m_lo].sum() == 0 or e[m_hi].sum() == 0:
                continue
            stat = _fast_logrank_chi2(t[m_lo], e[m_lo], t[m_hi], e[m_hi])
            if stat > best:
                best = stat
        return best

    best_stat_perm = _max_chi2(times, events)
    rng = np.random.RandomState(random_state)
    n = len(scores)
    n_exceed = 0
    for _ in range(n_permutations):
        perm_idx = rng.permutation(n)
        if _max_chi2(times[perm_idx], events[perm_idx]) >= best_stat_perm:
            n_exceed += 1
    return (n_exceed + 1) / (n_permutations + 1)


def _validate_maxstat_inputs(times, events, scores):
    """Validate and convert maxstat inputs to float arrays.

    Returns (times, events, scores) as numpy float arrays.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = len(scores)

    if len(times) != n or len(events) != n:
        raise ValueError(
            f"Array length mismatch: times({len(times)}), "
            f"events({len(events)}), scores({n}). All must be the same length."
        )

    for name, arr in [("times", times), ("events", events), ("scores", scores)]:
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        if n_bad > 0:
            raise ValueError(
                f"Input '{name}' contains {n_bad} NaN/Inf values. "
                f"Remove or impute missing values before calling maxstat."
            )

    if n < 10:
        raise DegenerateSubpopulationError(
            f"Too few samples for maxstat ({n}). Need at least 10."
        )
    if events.sum() < 3:
        raise DegenerateSubpopulationError(
            f"Too few events for maxstat ({int(events.sum())}). Need at least 3."
        )

    return times, events, scores


def _find_valid_candidates(scores, min_prop, max_prop):
    """Find valid cutpoint candidates based on proportion constraints."""
    unique_scores = np.unique(scores)
    if len(unique_scores) < 2:
        raise DegenerateSubpopulationError("All risk scores are identical \u2014 cannot find a cutpoint.")

    candidates = (unique_scores[:-1] + unique_scores[1:]) / 2.0

    sorted_scores = np.sort(scores)
    n = len(scores)
    prop_below = np.searchsorted(sorted_scores, candidates, side="right") / n
    valid = candidates[(prop_below >= min_prop) & (prop_below <= max_prop)]

    if len(valid) < 2:
        raise DegenerateSubpopulationError(
            f"Fewer than 2 valid cutpoint candidates after filtering "
            f"(min_prop={min_prop}, max_prop={max_prop}). "
            f"Scores may be too concentrated."
        )
    return valid


def _correct_p_value(correction, best_p_raw, best_stat, n_candidates,
                     min_prop, max_prop, times, events, scores, valid,
                     n_permutations, random_state):
    """Apply the specified p-value correction method."""
    if correction == "bonferroni":
        return _bonferroni_correction(best_p_raw, n_candidates)
    if correction == "lausen-schumacher":
        return _lausen_schumacher_correction(best_stat, min_prop, max_prop)
    if correction == "permutation":
        return _permutation_correction(
            times, events, scores, valid, n_permutations, random_state,
        )
    raise ValueError(
        f"Unknown correction {correction!r}. "
        f"Choose from {sorted(_VALID_CORRECTIONS)}."
    )


def _find_maxstat_cutpoint(
    times,
    events,
    scores,
    correction="bonferroni",
    min_prop=0.10,
    max_prop=0.90,
    n_permutations=1000,
    random_state=None,
):
    """Find the score cutpoint that maximally separates survival curves.

    Scans candidate cutpoints between consecutive unique score values and
    selects the one with the highest log-rank chi2 statistic.

    Parameters
    ----------
    times : array-like
        Survival times.
    events : array-like
        Event indicators (1=event, 0=censored).
    scores : array-like
        Continuous risk scores.
    correction : {"bonferroni", "lausen-schumacher", "permutation"}
        Multiple-testing correction for the p-value.
    min_prop : float
        Minimum proportion of samples in the lower (below-cutpoint) group.
    max_prop : float
        Maximum proportion of samples in the *lower* group
        (equivalently, 1 - max_prop is the minimum for the upper group).
    n_permutations : int
        Number of permutations for ``correction="permutation"``.
    random_state : int or None
        Seed for permutation reproducibility.

    Returns
    -------
    MaxstatResult
    """
    times, events, scores = _validate_maxstat_inputs(times, events, scores)

    valid = _find_valid_candidates(scores, min_prop, max_prop)

    best_stat, best_cut, best_p_raw = _evaluate_cutpoint_candidates(
        times, events, scores, valid,
    )
    if best_cut is None:
        raise DegenerateSubpopulationError(
            "Maxstat search failed: no candidate cutpoint produced a valid "
            "log-rank statistic. This may indicate that survival times are "
            "independent of the risk scores."
        )

    n_candidates = len(valid)
    p_corrected = _correct_p_value(
        correction, best_p_raw, best_stat, n_candidates,
        min_prop, max_prop, times, events, scores, valid,
        n_permutations, random_state,
    )

    return MaxstatResult(
        cutpoint=best_cut,
        test_statistic=best_stat,
        p_value_raw=best_p_raw,
        p_value_corrected=p_corrected,
        correction=correction,
        n_candidates=n_candidates,
        cutpoints=(best_cut,),
        split_stats=((best_cut, best_stat, best_p_raw),),
    )


def _find_maxstat_cutpoints_recursive(
    times, events, scores,
    n_groups=2, correction="bonferroni",
    min_prop=0.10, max_prop=0.90,
    n_permutations=1000, random_state=None,
):
    """Find multiple cutpoints via greedy binary splitting.

    Starts with the full dataset, finds the best cutpoint, then iteratively
    splits the subpopulation with the highest log-rank chi2 (selected via a
    priority queue) until ``n_groups - 1`` cutpoints are collected.

    Parameters
    ----------
    n_groups : int
        Target number of risk groups (2, 3, or 4).
    correction, min_prop, max_prop, n_permutations, random_state :
        Forwarded to :func:`_find_maxstat_cutpoint`.

    Returns
    -------
    MaxstatResult
        The root-level result, with ``cutpoints`` and ``split_stats`` populated.

    Raises
    ------
    DegenerateSubpopulationError
        If the root data is too small or degenerate for any split.
    """
    import heapq

    if n_groups < 2:
        raise ValueError(f"n_groups={n_groups} is invalid. Must be at least 2.")

    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    scores = np.asarray(scores, dtype=float)

    # Derive independent seeds for subpopulation splits to avoid correlated
    # permutation sequences when correction="permutation".
    seed_rng = np.random.RandomState(random_state) if random_state is not None else None

    def _next_seed():
        return seed_rng.randint(0, 2**31) if seed_rng is not None else None

    # First split on full data
    root = _find_maxstat_cutpoint(
        times, events, scores,
        correction=correction, min_prop=min_prop, max_prop=max_prop,
        n_permutations=n_permutations, random_state=random_state,
    )

    if n_groups == 2:
        return root._replace(
            cutpoints=(root.cutpoint,),
            split_stats=((root.cutpoint, root.test_statistic, root.p_value_raw),),
        )

    # Collect cutpoints via priority-queue greedy: split the subpopulation with
    # the highest chi2 statistic at each step.
    collected_cuts = [root.cutpoint]
    collected_stats = [(root.cutpoint, root.test_statistic, root.p_value_raw)]

    # Each heap entry: (-chi2, lo_bound, hi_bound, sub_result) for a
    # splittable subpopulation.
    # lo_bound/hi_bound are score boundaries (-inf / +inf for the full range).
    heap = []

    def _push_subpopulation(lo_bound, hi_bound):
        mask = (scores > lo_bound) & (scores <= hi_bound)
        sub_t, sub_e, sub_s = times[mask], events[mask], scores[mask]
        try:
            sub_result = _find_maxstat_cutpoint(
                sub_t, sub_e, sub_s,
                correction=correction, min_prop=min_prop, max_prop=max_prop,
                n_permutations=n_permutations, random_state=_next_seed(),
            )
        except DegenerateSubpopulationError:
            return  # subpopulation too small or degenerate -- expected
        heapq.heappush(heap, (-sub_result.test_statistic, lo_bound, hi_bound,
                               sub_result))

    # After the root split, push both subpopulations
    _push_subpopulation(-np.inf, root.cutpoint)
    _push_subpopulation(root.cutpoint, np.inf)

    while len(collected_cuts) < n_groups - 1 and heap:
        _neg_stat, lo_bound, hi_bound, sub_result = heapq.heappop(heap)
        collected_cuts.append(sub_result.cutpoint)
        collected_stats.append(
            (sub_result.cutpoint, sub_result.test_statistic, sub_result.p_value_raw)
        )
        # Push the two children of this split
        _push_subpopulation(lo_bound, sub_result.cutpoint)
        _push_subpopulation(sub_result.cutpoint, hi_bound)

    sorted_cuts = tuple(sorted(collected_cuts))
    sorted_stats = tuple(sorted(collected_stats, key=lambda x: x[0]))

    return root._replace(cutpoints=sorted_cuts, split_stats=sorted_stats)


def _resolve_maxstat(df, time_col, event_col, risk_col,
                     n_groups, breaks, break_type, correction,
                     n_permutations=1000, random_state=None):
    """Convert ``break_type="maxstat"`` into ``break_type="score"``.

    For non-maxstat break types, returns arguments unchanged.

    Returns
    -------
    tuple of (n_groups, breaks, break_type, MaxstatResult or None)
    """
    if break_type != "maxstat":
        return n_groups, breaks, break_type, None

    result = _find_maxstat_cutpoints_recursive(
        df[time_col], df[event_col], df[risk_col],
        n_groups=n_groups, correction=correction,
        n_permutations=n_permutations, random_state=random_state,
    )
    actual_groups = len(result.cutpoints) + 1
    if actual_groups < n_groups:
        warnings.warn(
            f"Requested {n_groups} maxstat groups but only {actual_groups} "
            f"could be formed (subpopulations too small for further splitting).",
            InternalWarning,
            stacklevel=2,
        )
    return actual_groups, list(result.cutpoints), "score", result
