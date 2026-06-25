"""Antolini's time-dependent concordance index (C^td).

Pure numpy implementation following Antolini et al. (2005) with the
adj_antolini tie-handling from Kvamme & Borgan (2021) / pycox.

C^td compares predicted survival *curves* at each patient's event time,
unlike Harrell's C which uses a single risk score.  For proportional-hazards
models the two are equivalent; for non-PH models (DeepHit, RSF) under
heavy censoring C^td can be substantially higher.

References
----------
Antolini L, Boracchi P, Biganzoli E (2005).
    A time-dependent discrimination index for survival data.
    Statistics in Medicine 24(24):3927-3944.

Kvamme H, Borgan O (2021).
    Continuous and Discrete-Time Survival Prediction with Neural Networks.
    Lifetime Data Analysis 27:710-736.
"""

import numpy as np


def concordance_td(durations, events, surv_probs, eval_times):
    """Compute Antolini's time-dependent concordance index (adj_antolini).

    Comparable pairs include both strict-ordering pairs (t_i < t_j, d_i=1)
    and tied-time pairs (t_i == t_j, d_i or d_j = 1).  Tied-time scoring
    follows pycox's adj_antolini method (Sec 5.1, Ishwaran et al. 2008).

    Parameters
    ----------
    durations : array-like, shape (n_samples,)
        Observed survival times.
    events : array-like, shape (n_samples,)
        Event indicators (1 = event, 0 = censored).
    surv_probs : ndarray, shape (n_samples, n_times)
        Predicted survival probabilities.  Row i gives S(t | x_i)
        evaluated at each time in ``eval_times``.
    eval_times : array-like, shape (n_times,)
        Sorted time grid at which ``surv_probs`` columns are evaluated.

    Returns
    -------
    float
        Concordance index in [0, 1].  Returns NaN if no comparable
        pairs exist.
    """
    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(events, dtype=np.float64)
    surv_probs = np.asarray(surv_probs, dtype=np.float64)
    eval_times = np.asarray(eval_times, dtype=np.float64)

    n = len(durations)
    if n < 2 or surv_probs.shape[1] == 0:
        return np.nan

    # Map each patient's duration to the largest eval_time <= duration.
    # Matches pycox EvalSurv's steps='post' convention:
    # searchsorted(side='right') - 1, clipped to [0, n_times-1].
    n_times = len(eval_times)
    time_indices = np.clip(
        np.searchsorted(eval_times, durations, side='right') - 1, 0, n_times - 1
    )

    # Vectorized pairwise comparison using numpy broadcasting.
    # Memory: O(n²) float64 — ~67 MB at n=3000, acceptable for typical datasets.

    # For each patient i, survival prob at their own mapped time index
    s_own = surv_probs[np.arange(n), time_indices]  # shape (n,)

    # Matrix of all patients' survival probs at each patient's time index.
    # S[j, i] = surv_probs[j, time_indices[i]]
    S = surv_probs[:, time_indices]  # shape (n, n)

    # For pair (i, j): s_i = s_own[i], s_j = S[j, i] = S.T[i, j]
    s_i = s_own[:, None]  # (n, 1) broadcasts to (n, n)
    s_j = S.T             # (n, n), [i, j] = surv_probs[j, time_indices[i]]

    # Pairwise masks via broadcasting
    t_row = durations[:, None]       # (n, 1)
    t_col = durations[None, :]       # (1, n)
    d_row = events[:, None] > 0      # (n, 1) bool — event for patient i
    d_col = events[None, :] > 0      # (1, n) bool — event for patient j

    # Strict ordering: t_i < t_j and d_i = 1
    strict = (t_row < t_col) & d_row

    # Tied time: t_i == t_j and (d_i or d_j), excluding self-pairs
    tied = (t_row == t_col) & (d_row | d_col)
    np.fill_diagonal(tied, False)

    # Concordance contributions — strict ordering
    conc_strict = np.sum(strict * ((s_i < s_j) + 0.5 * (s_i == s_j)))

    # Concordance contributions — tied time sub-cases
    both_ev = tied & d_row & d_col
    i_only = tied & d_row & ~d_col
    j_only = tied & ~d_row & d_col

    conc_tied = (
        np.sum(both_ev * (1.0 - 0.5 * (s_i != s_j)))
        + np.sum(i_only * ((s_i < s_j) + 0.5 * (s_i == s_j)))
        + np.sum(j_only * ((s_i > s_j) + 0.5 * (s_i == s_j)))
    )

    comparable = float(np.sum(strict) + np.sum(tied))
    if comparable == 0:
        return np.nan

    concordant = float(conc_strict + conc_tied)
    return concordant / comparable
