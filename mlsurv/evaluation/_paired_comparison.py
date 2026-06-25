"""Paired model comparison for C-index and time-dependent AUC.

Provides statistical tests for comparing two survival models:
- Bootstrap paired difference test (any metric with bootstrap distributions)
- Kang et al. (2015) nonparametric z-test (Harrell's C only, no bootstrap needed)
"""

from __future__ import annotations

import itertools
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .._shared import percentile_ci
from .._warnings import InternalWarning, MethodologicalWarning, SuggestionWarning
from ..logging_utils import get_logger
from ..utils import SurvivalArray

logger = get_logger("paired_comparison")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_bootstrap(
    bootstrap_result,
    metric: str = "harrell",
    models: Sequence[str] | None = None,
    ci: float = 0.95,
    n_bootstrap: int | None = None,
    p_adjust: str = "holm",
    eval_time: float | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Compare models using paired bootstrap distributions.

    Extracts per-iteration metric values from a multi-model BootstrapResult
    and performs paired difference tests for all model pairs.

    Parameters
    ----------
    bootstrap_result : BootstrapResult
        Bootstrap result containing iterations for multiple models.
    metric : str, default="harrell"
        Metric to compare. Standard metrics (e.g. "harrell", "ipcw",
        "integrated_brier") are extracted from ``iterations``.
        Use ``"mean_auc"`` for mean tdAUC across timepoints, or
        ``"auc_t"`` with ``eval_time`` for a specific timepoint.
    models : sequence of str, optional
        Model names to compare. If None, uses all models in the result.
    ci : float, default=0.95
        Confidence level for the difference CI.
    n_bootstrap : int, optional
        If greater than the number of available bootstrap iterations,
        resample with replacement to reach this count.
    p_adjust : str, default="holm"
        Multiplicity correction: "holm", "bh", "bonferroni", or "none".
    eval_time : float, optional
        Required when ``metric="auc_t"`` to select a specific timepoint.
    random_state : int, optional
        Seed for resampling (used only when ``n_bootstrap`` triggers
        shortfall resampling).

    Returns
    -------
    pd.DataFrame
        Columns: model_a, model_b, metric, method, delta, ci_lower,
        ci_upper, p_value, p_adjusted.

    Raises
    ------
    ValueError
        If ``ci`` is outside the valid range, if any name in ``models`` is
        not present in the bootstrap result, or if fewer than two models are
        available to compare.
    """
    _validate_ci(ci)

    # Resolve model names
    all_models = _get_model_names(bootstrap_result)
    if models is not None:
        unknown = set(models) - set(all_models)
        if unknown:
            raise ValueError(
                f"Models not found in bootstrap result: {sorted(unknown)}. "
                f"Available: {sorted(all_models)}"
            )
        all_models = list(models)

    if len(all_models) < 2:
        raise ValueError(
            f"Need at least 2 models to compare, got {len(all_models)}: "
            f"{all_models}"
        )

    # Build all pairs and test
    rows = []
    for model_a, model_b in itertools.combinations(all_models, 2):
        dist_a = _extract_distribution(
            bootstrap_result, metric, model_a, eval_time,
        )
        dist_b = _extract_distribution(
            bootstrap_result, metric, model_b, eval_time,
        )
        result = _paired_bootstrap_test(
            dist_a, dist_b,
            ci=ci,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        metric_label = metric if eval_time is None else f"{metric}@{eval_time}"
        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric_label,
            "method": "paired_bootstrap",
            **result,
        })

    df = pd.DataFrame(rows)
    df["p_adjusted"] = _adjust_p_values(df["p_value"].values, method=p_adjust)
    return df


def _validate_kang_inputs(y, risk_scores_a, risk_scores_b, ci):
    """Validate and coerce inputs for the Kang test.

    Returns
    -------
    tuple of (risk_scores_a, risk_scores_b, events, times, n)
    """
    _validate_ci(ci)
    risk_scores_a = np.asarray(risk_scores_a, dtype=np.float64).ravel()
    risk_scores_b = np.asarray(risk_scores_b, dtype=np.float64).ravel()
    events = y["event"].astype(bool)
    times = y["time"].astype(np.float64)
    n = len(events)

    if len(risk_scores_a) != n or len(risk_scores_b) != n:
        raise ValueError(
            f"risk_scores length mismatch: a={len(risk_scores_a)}, "
            f"b={len(risk_scores_b)}, y={n}"
        )

    if not np.all(np.isfinite(risk_scores_a)) or not np.all(np.isfinite(risk_scores_b)):
        raise ValueError(
            "Risk scores contain NaN or Inf values. Remove or impute "
            "non-finite scores before calling compare_kang()."
        )

    if np.isnan(times).any() or np.isinf(times).any():
        raise ValueError(
            "Survival times contain NaN or Inf values. Clean the survival "
            "array before calling compare_kang()."
        )

    max_pairs = int(events.sum()) * n
    estimated_bytes = max_pairs * 32  # 4 arrays × 8 bytes per float64
    if estimated_bytes > 1e9:  # > 1 GB
        warnings.warn(
            f"Kang test pre-allocation requires ~{estimated_bytes / 1e9:.1f} GB "
            f"for {max_pairs:,} potential pairs (n={n}, events={int(events.sum())}). "
            "Consider using compare_bootstrap() for large datasets.",
            SuggestionWarning,
            stacklevel=3,
        )

    return risk_scores_a, risk_scores_b, events, times, n


def compare_kang(
    y: SurvivalArray,
    risk_scores_a: np.ndarray,
    risk_scores_b: np.ndarray,
    ci: float = 0.95,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
) -> pd.DataFrame:
    """Nonparametric z-test for comparing two C-indices (Kang et al. 2015).

    Computes Harrell's C for each set of risk scores and tests whether
    the difference is statistically significant using a variance estimate
    based on Hoeffding's decomposition of the U-statistic.

    Parameters
    ----------
    y : SurvivalArray
        Survival outcomes with 'event' and 'time' fields.
    risk_scores_a : array-like
        Risk scores from model A (higher = higher risk).
    risk_scores_b : array-like
        Risk scores from model B (higher = higher risk).
    ci : float, default=0.95
        Confidence level for the difference CI.
    model_a_name : str, default="model_a"
        Display name for model A in the result DataFrame.
    model_b_name : str, default="model_b"
        Display name for model B in the result DataFrame.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns: model_a, model_b, metric,
        method, delta, ci_lower, ci_upper, p_value, z_statistic.

    Notes
    -----
    - Only supports Harrell's C-index.
    - O(n^2) memory for n observations; warns when n > 5000.
    - Handles degenerate cases (zero variance) by returning NaN.
    - Under non-random (covariate-dependent) censoring, the Kang test may
      exhibit inflated Type I error rates (Han et al., 2017). When censoring
      depends on covariates, consider compare_bootstrap() as an alternative.
    """
    risk_scores_a, risk_scores_b, events, times, n = _validate_kang_inputs(
        y, risk_scores_a, risk_scores_b, ci,
    )

    # Enumerate comparable pairs: event[i]=True and time[i] < time[j]
    # Concordance indicator: 1 if risk_score[i] > risk_score[j], 0.5 if tied
    h_a, h_b, pair_i, pair_j = _concordance_pairs(
        events, times, risk_scores_a, risk_scores_b,
    )
    n_pairs = len(h_a)

    if n_pairs == 0:
        warnings.warn(
            "Kang test found 0 comparable pairs. This typically means "
            "all observations are censored or there is insufficient "
            "variation in event times. Returning NaN results.",
            MethodologicalWarning,
            stacklevel=2,
        )
        return _empty_kang_row(model_a_name, model_b_name)

    c_a = np.mean(h_a)
    c_b = np.mean(h_b)
    delta = float(c_a - c_b)

    # Hoeffding decomposition for Var(delta_C)
    var_delta = _hoeffding_variance(h_a, h_b, pair_i, pair_j, n)

    if var_delta <= 0 or np.isnan(var_delta):
        warnings.warn(
            "Kang test variance estimate is zero or NaN. This typically "
            "occurs when both models produce identical concordance "
            "orderings. Returning NaN for z_statistic, p_value, and CI.",
            MethodologicalWarning,
            stacklevel=2,
        )
        z_stat, p_value, ci_lower, ci_upper = np.nan, np.nan, np.nan, np.nan
    else:
        se = np.sqrt(var_delta)
        z_stat = delta / se
        p_value = 2.0 * norm.sf(np.abs(z_stat))
        z_crit = norm.ppf(1 - (1 - ci) / 2)
        ci_lower = delta - z_crit * se
        ci_upper = delta + z_crit * se

    return pd.DataFrame([{
        "model_a": model_a_name,
        "model_b": model_b_name,
        "metric": "harrell",
        "method": "kang",
        "delta": delta,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "z_statistic": float(z_stat),
    }])



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _continuity_corrected_p(arr: np.ndarray) -> float:
    """Two-sided bootstrap p-value with continuity correction.

    Uses (count + 1) / (n + 1) per Davison & Hinkley (1997) to avoid
    reporting p=0 from finite bootstrap resolution.
    """
    n = len(arr)
    if n < 2:
        return float(np.nan)
    count_ge = int(np.sum(arr >= 0))
    count_le = int(np.sum(arr <= 0))
    return float(min(2.0 * (min(count_ge, count_le) + 1) / (n + 1), 1.0))


def _validate_ci(ci: float) -> None:
    """Raise ValueError if ci is not in (0, 1)."""
    if not 0 < ci < 1:
        raise ValueError(
            f"ci must be between 0 and 1 (exclusive), got {ci}. "
            f"Use ci=0.95 for a 95% confidence interval, not ci=95."
        )


def _paired_bootstrap_test(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    ci: float = 0.95,
    n_bootstrap: int | None = None,
    random_state: int | None = None,
) -> dict:
    """Paired difference test on two bootstrap distributions.

    Parameters
    ----------
    dist_a, dist_b : array-like
        Paired bootstrap metric values (same indices = same bootstrap sample).
    ci : float
        Confidence level.
    n_bootstrap : int, optional
        Target number of iterations. If larger than available, resample.
    random_state : int, optional
        Seed for resampling.

    Returns
    -------
    dict
        Keys: delta, ci_lower, ci_upper, p_value.
    """
    if n_bootstrap is not None and n_bootstrap < 1:
        raise ValueError(
            f"n_bootstrap must be a positive integer, got {n_bootstrap}."
        )
    dist_a = np.asarray(dist_a, dtype=np.float64)
    dist_b = np.asarray(dist_b, dtype=np.float64)
    if dist_a.shape != dist_b.shape:
        raise ValueError(
            f"Distribution shape mismatch: dist_a={dist_a.shape}, "
            f"dist_b={dist_b.shape}. Paired test requires equal-length "
            f"arrays from the same bootstrap iterations."
        )
    deltas = dist_a - dist_b

    # Drop non-finite iterations (NaN or Inf)
    valid = np.isfinite(deltas)
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        warnings.warn(
            f"Dropped {n_dropped}/{len(deltas)} non-finite bootstrap "
            f"iterations. Proceeding with {valid.sum()} valid iterations.",
            MethodologicalWarning,
            stacklevel=2,
        )
    deltas = deltas[valid]
    n_valid = len(deltas)

    if n_valid == 0:
        warnings.warn(
            "All bootstrap iterations were non-finite. "
            "Returning NaN for all comparison statistics.",
            MethodologicalWarning,
            stacklevel=2,
        )
        return {"delta": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                "p_value": np.nan}

    # Shortfall resampling
    if n_bootstrap is not None and n_bootstrap > n_valid:
        rng = np.random.RandomState(random_state)
        warnings.warn(
            f"Requested {n_bootstrap} iterations but only {n_valid} "
            f"available. Resampling with replacement.",
            MethodologicalWarning,
            stacklevel=2,
        )
        idx = rng.randint(0, n_valid, size=n_bootstrap)
        deltas = deltas[idx]

    mean_delta = float(np.mean(deltas))

    # Percentile CI
    ci_lower, ci_upper = percentile_ci(deltas, ci_level=ci)
    ci_lower, ci_upper = float(ci_lower), float(ci_upper)

    p_value = _continuity_corrected_p(deltas)

    return {
        "delta": mean_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
    }


def _extract_distribution(
    bootstrap_result,
    metric: str,
    model: str,
    eval_time: float | None,
) -> np.ndarray:
    """Extract per-iteration metric values for a single model.

    Parameters
    ----------
    bootstrap_result : BootstrapResult
        The bootstrap result object.
    metric : str
        Metric name or "mean_auc" / "auc_t".
    model : str
        Model name to filter by.
    eval_time : float or None
        Required for "auc_t" metric.

    Returns
    -------
    np.ndarray
        One value per bootstrap iteration.
    """
    if metric == "mean_auc":
        return _extract_mean_auc(bootstrap_result, model)
    if metric == "auc_t":
        if eval_time is None:
            raise ValueError(
                "eval_time is required when metric='auc_t'. "
                "Use metric='mean_auc' for the average across timepoints."
            )
        return _extract_auc_at_time(bootstrap_result, model, eval_time)
    dist = bootstrap_result.get_distribution(metric, model_type=model)
    if len(dist) == 0:
        available = sorted(
            bootstrap_result.iterations["metric_name"].unique()
        )
        raise ValueError(
            f"No bootstrap iterations found for metric '{metric}' "
            f"and model '{model}'. Available metrics: {available}"
        )
    return dist


def _get_auc_df(bootstrap_result) -> pd.DataFrame:
    """Return the auc_t DataFrame, raising if unavailable."""
    auc_df = bootstrap_result.auc_t
    if auc_df is None or auc_df.empty:
        raise ValueError(
            "No auc_t data in bootstrap result. "
            "Re-run bootstrap with time-dependent AUC enabled."
        )
    return auc_df


def _extract_mean_auc(bootstrap_result, model: str) -> np.ndarray:
    """Extract mean tdAUC across timepoints per iteration.

    Only timepoints that are valid (non-NaN) in *all* iterations are
    included so that every iteration averages over the same set,
    preventing variance inflation from inconsistent timepoint counts.
    """
    auc_df = _get_auc_df(bootstrap_result)
    model_df = auc_df[auc_df["model_type"] == model]
    if model_df.empty:
        raise ValueError(f"No auc_t data for model '{model}'.")
    # Pivot: rows=iterations, cols=eval_times, values=auc
    pivot = model_df.pivot_table(
        index="iteration", columns="eval_time", values="auc",
    )
    # Keep only timepoints valid in ALL iterations
    valid_cols = pivot.columns[pivot.notna().all()]
    if len(valid_cols) == 0:
        raise ValueError(
            f"No eval_time is valid across all bootstrap iterations for '{model}'."
        )
    if len(valid_cols) < len(pivot.columns):
        n_dropped = len(pivot.columns) - len(valid_cols)
        warnings.warn(
            f"mean_auc for '{model}': dropped {n_dropped}/{len(pivot.columns)} "
            f"timepoints not valid in all bootstrap iterations. "
            f"The mean is computed over {len(valid_cols)} universally-valid "
            f"timepoints.",
            MethodologicalWarning,
            stacklevel=2,
        )
    return pivot[valid_cols].mean(axis=1).values


def _extract_auc_at_time(
    bootstrap_result, model: str, eval_time: float,
) -> np.ndarray:
    """Extract tdAUC at a specific timepoint per iteration."""
    auc_df = _get_auc_df(bootstrap_result)
    mask = (auc_df["model_type"] == model) & np.isclose(auc_df["eval_time"], eval_time)
    filtered = auc_df[mask]
    if filtered.empty:
        available = sorted(
            auc_df.loc[auc_df["model_type"] == model, "eval_time"].unique()
        )
        raise ValueError(
            f"No auc_t data for model '{model}' at eval_time={eval_time}. "
            f"Available timepoints: {available}"
        )
    n_per_iter = filtered.groupby("iteration").size()
    if (n_per_iter > 1).any():
        warnings.warn(
            f"Multiple eval_times matched near {eval_time} for model "
            f"'{model}'. Averaging duplicates within each iteration.",
            InternalWarning,
            stacklevel=2,
        )
        return filtered.groupby("iteration")["auc"].mean().sort_index().values
    return filtered.sort_values("iteration")["auc"].values


def _get_model_names(bootstrap_result) -> list[str]:
    """Get unique model names from a BootstrapResult."""
    df = bootstrap_result.iterations
    if "model_type" not in df.columns:
        raise ValueError("iterations DataFrame missing 'model_type' column.")
    return list(df["model_type"].unique())


def _adjust_p_values(
    p_values: np.ndarray,
    method: str = "holm",
) -> np.ndarray:
    """Apply multiplicity correction to p-values.

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    method : str
        "holm", "bh", "bonferroni", or "none".

    Returns
    -------
    np.ndarray
        Adjusted p-values, clipped to [0, 1].
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)

    if m == 0 or method == "none":
        return p.copy()

    # Handle NaN: preserve them, adjust only non-NaN
    nan_mask = np.isnan(p)
    if nan_mask.all():
        return p.copy()

    if method == "bonferroni":
        m_valid = int((~nan_mask).sum())
        return np.where(nan_mask, np.nan, np.minimum(p * m_valid, 1.0))

    if method == "bh":
        from scipy.stats import false_discovery_control
        valid = p[~nan_mask]
        adj_valid = false_discovery_control(valid, method="bh")
        adjusted = np.full_like(p, np.nan)
        adjusted[~nan_mask] = adj_valid
        return adjusted

    if method == "holm":
        adjusted = np.full_like(p, np.nan)
        valid_idx = np.where(~nan_mask)[0]
        valid_p = p[valid_idx]
        m_valid = len(valid_p)

        # Sort ascending
        order = np.argsort(valid_p)
        sorted_p = valid_p[order]

        # Holm step-down: p_adj[i] = max(p_adj[i-1], p_(i) * (m_valid - i))
        holm = sorted_p * np.arange(m_valid, 0, -1)
        holm = np.minimum(np.maximum.accumulate(holm), 1.0)

        # Unsort
        unsorted = np.empty(m_valid)
        unsorted[order] = holm
        adjusted[valid_idx] = unsorted
        return adjusted

    raise ValueError(
        f"Unknown p_adjust method: '{method}'. "
        f"Use 'holm', 'bh', 'bonferroni', or 'none'."
    )


# ---------------------------------------------------------------------------
# Kang test internals
# ---------------------------------------------------------------------------

def _concordance_indicator(diff: float) -> float:
    """Return 1.0 if diff > 0, 0.0 if diff < 0, 0.5 if tied."""
    if diff > 0:
        return 1.0
    if diff < 0:
        return 0.0
    return 0.5


def _concordance_pairs(
    events: np.ndarray,
    times: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
):
    """Enumerate comparable pairs and compute concordance indicators.

    A pair (i, j) is comparable when event[i]=True and time[i] < time[j].
    Concordance indicator: 1 if score[i] > score[j], 0.5 if tied, 0 otherwise.

    Returns
    -------
    h_a, h_b : np.ndarray
        Concordance indicators for models A and B.
    pair_i, pair_j : np.ndarray
        Observation indices for each pair.
    """
    n = len(events)
    # Pre-allocate conservatively
    max_pairs = int(events.sum()) * n  # upper bound
    h_a = np.empty(max_pairs)
    h_b = np.empty(max_pairs)
    pair_i = np.empty(max_pairs, dtype=np.intp)
    pair_j = np.empty(max_pairs, dtype=np.intp)

    k = 0
    for i in range(n):
        if not events[i]:
            continue
        for j in range(n):
            if times[i] >= times[j]:
                continue
            pair_i[k] = i
            pair_j[k] = j
            h_a[k] = _concordance_indicator(scores_a[i] - scores_a[j])
            h_b[k] = _concordance_indicator(scores_b[i] - scores_b[j])
            k += 1

    return h_a[:k], h_b[:k], pair_i[:k], pair_j[:k]


def _hoeffding_variance(
    h_a: np.ndarray,
    h_b: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    n: int,
) -> float:
    """Hoeffding decomposition variance for delta C.

    Estimates Var(C_a - C_b) using the first-order kernel means
    (per-observation contributions to the U-statistic).

    Parameters
    ----------
    h_a, h_b : np.ndarray
        Concordance indicators for models A and B.
    pair_i, pair_j : np.ndarray
        Observation indices for each comparable pair.
    n : int
        Total number of observations.

    Returns
    -------
    float
        Estimated variance of delta_C.
    """
    d = h_a - h_b

    # Per-observation kernel means: mean of d for all pairs involving obs i
    kernel_sums = np.zeros(n)
    kernel_counts = np.zeros(n)
    np.add.at(kernel_sums, pair_i, d)
    np.add.at(kernel_sums, pair_j, d)
    np.add.at(kernel_counts, pair_i, 1)
    np.add.at(kernel_counts, pair_j, 1)

    active = kernel_counts > 0
    if active.sum() < 2:
        return np.nan

    kernel_means = np.zeros(n)
    kernel_means[active] = kernel_sums[active] / kernel_counts[active]

    # Var(delta_C) ≈ (4 / n) * Var(kernel_means) over all n observations
    var_kernels = np.var(kernel_means, ddof=1)

    return 4.0 * var_kernels / n


def _empty_kang_row(
    model_a_name: str = "model_a", model_b_name: str = "model_b",
) -> pd.DataFrame:
    """Return an empty result row when no comparable pairs exist."""
    return pd.DataFrame([{
        "model_a": model_a_name,
        "model_b": model_b_name,
        "metric": "harrell",
        "method": "kang",
        "delta": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "p_value": np.nan,
        "z_statistic": np.nan,
    }])
