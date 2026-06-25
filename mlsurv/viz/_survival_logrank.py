"""Log-rank test computation and annotation for KM survival plots.

Public symbols:
    _compute_logrank, _format_logrank_annotation,
    _add_logrank_annotation, _collect_logrank
"""

import warnings

import numpy as np

from mlsurv._warnings import InternalWarning
from mlsurv.viz._palettes import MANUSCRIPT_DEFAULTS


def _compute_logrank(df, time_col, event_col, group_col, p_adjust="holm"):
    """Compute omnibus and pairwise log-rank tests across groups.

    Returns
    -------
    dict or None
        Keys: test, omnibus_statistic, omnibus_p, n_groups,
        pairwise (list or None), p_adjust_method.
        Returns ``None`` when fewer than 2 groups or on statistical failure.
    """
    from lifelines.statistics import multivariate_logrank_test

    groups = df[group_col].unique()
    n_groups = len(groups)

    if n_groups < 2:
        return None

    try:
        result = multivariate_logrank_test(
            df[time_col], df[group_col], df[event_col],
        )
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError) as exc:
        warnings.warn(
            f"Log-rank test failed: {exc}. Skipping p-value annotation.",
            InternalWarning,
            stacklevel=5,
        )
        return None

    omnibus_p = float(result.p_value)
    omnibus_stat = float(result.test_statistic)
    if np.isnan(omnibus_p) or np.isnan(omnibus_stat):
        warnings.warn(
            "Log-rank test returned NaN statistic/p-value. "
            "Skipping p-value annotation.",
            InternalWarning,
            stacklevel=5,
        )
        return None

    out = {
        "test": "log-rank",
        "omnibus_statistic": omnibus_stat,
        "omnibus_p": omnibus_p,
        "n_groups": n_groups,
        "pairwise": None,
        "p_adjust_method": p_adjust if n_groups > 2 else None,
    }

    if n_groups > 2:
        out["pairwise"] = _compute_pairwise_logrank(
            df, time_col, event_col, group_col, groups, p_adjust,
        )

    return out


def _compute_pairwise_logrank(df, time_col, event_col, group_col, groups,
                               p_adjust):
    """Compute pairwise log-rank tests with p-value adjustment.

    Returns list of pairwise result dicts, or None if all pairs failed.
    """
    from itertools import combinations

    from lifelines.statistics import logrank_test

    from mlsurv.evaluation._paired_comparison import _adjust_p_values

    pairs = list(combinations(sorted(groups), 2))
    raw_p = []
    pairwise = []
    n_failed = 0
    for g1, g2 in pairs:
        g1_mask = df[group_col] == g1
        g2_mask = df[group_col] == g2
        try:
            lr = logrank_test(
                df.loc[g1_mask, time_col],
                df.loc[g2_mask, time_col],
                df.loc[g1_mask, event_col],
                df.loc[g2_mask, event_col],
            )
        except (ValueError, ZeroDivisionError,
                np.linalg.LinAlgError) as exc:
            warnings.warn(
                f"Pairwise log-rank test ({g1} vs {g2}) failed: {exc}.",
                InternalWarning,
                stacklevel=5,
            )
            pairwise.append({
                "group_1": g1, "group_2": g2,
                "statistic": None, "p_raw": None,
            })
            n_failed += 1
            continue
        raw_p.append(lr.p_value)
        pairwise.append({
            "group_1": g1, "group_2": g2,
            "statistic": float(lr.test_statistic),
            "p_raw": float(lr.p_value),
        })

    if n_failed:
        warnings.warn(
            f"{n_failed} of {len(pairs)} pairwise log-rank tests failed. "
            "Adjusted p-values are computed over successful pairs only.",
            InternalWarning,
            stacklevel=5,
        )

    # Adjust only successful (non-None) p-values
    if raw_p:
        adjusted = _adjust_p_values(np.array(raw_p), method=p_adjust)
        adj_iter = iter(adjusted)
        for pw in pairwise:
            if pw["p_raw"] is not None:
                pw["p_adjusted"] = float(next(adj_iter))
            else:
                pw["p_adjusted"] = None

    return pairwise or None


def _format_logrank_annotation(logrank_result) -> str:
    """Format log-rank p-value for plot annotation."""
    p = logrank_result["omnibus_p"]
    if p == 0.0:
        p_str = "< 1e-300"
    elif p < 0.001:
        p_str = f"{p:.2e}"
    else:
        p_str = f"{p:.3f}"
    return f"Log-rank p = {p_str}"


def _add_logrank_annotation(fig, logrank_result, y=0.05, **subplot_kwargs):
    """Add a log-rank p-value annotation to a figure.

    Parameters
    ----------
    fig : go.Figure
        Target figure.
    logrank_result : dict
        Output from ``_compute_logrank``.
    y : float
        Vertical position (0 = bottom, 1 = top) within the axes.
    **subplot_kwargs
        Extra kwargs forwarded to ``fig.add_annotation`` (e.g. ``row``,
        ``col``, ``xref``, ``yref`` for subplot-aware placement).
    """
    subplot_kwargs.setdefault("xref", "x domain")
    subplot_kwargs.setdefault("yref", "y domain")
    fig.add_annotation(
        text=_format_logrank_annotation(logrank_result),
        x=0.98, y=y,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=MANUSCRIPT_DEFAULTS["font_size_annotation"]),
        **subplot_kwargs,
    )


def _collect_logrank(df, time_col, event_col, group_col, p_adjust,
                     _logrank_out, label="logrank"):
    """Compute log-rank test and append to the collection list if requested.

    Returns the log-rank result dict (or ``None`` if ``_logrank_out is None``
    or if the data has fewer than 2 groups).
    """
    if _logrank_out is None:
        return None
    result = _compute_logrank(df, time_col, event_col, group_col, p_adjust)
    if result is not None:
        _logrank_out.append((label, result))
    return result
