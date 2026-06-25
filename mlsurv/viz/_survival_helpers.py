"""Shared constants, validation, and helper functions for KM survival plots.

Public symbols:
    _MAXSTAT_LABELS, _VALID_BREAK_TYPES, _VALID_P_ADJUST,
    _pct_range_labels, _validate_km_inputs, _resolve_km_defaults,
    _maxstat_subtitle, _create_risk_groups, _resolve_axes_params,
    _resolve_risk_groups
"""

import warnings

import numpy as np
import pandas as pd

from mlsurv._warnings import InternalWarning, SuggestionWarning
from mlsurv.viz._survival_maxstat import _VALID_CORRECTIONS
from mlsurv.viz._validation import validate_columns


_VALID_BREAK_TYPES = {"quantile", "score", "maxstat"}
_VALID_P_ADJUST = {"holm", "bh", "bonferroni", "none"}

_MAXSTAT_LABELS = {
    2: ("Low risk", "High risk"),
    3: ("Low risk", "Medium risk", "High risk"),
    4: ("Low risk", "Med-low risk", "Med-high risk", "High risk"),
}


def _pct_range_labels(pcts):
    """Build percentile range labels from interior cut-points (0-100 scale).

    Examples
    --------
    >>> _pct_range_labels([25, 50, 75])
    ['< 25%', '25\u201350%', '50\u201375%', '\u2265 75%']
    >>> _pct_range_labels([50])
    ['< 50%', '\u2265 50%']
    """
    boundaries = [None] + list(pcts) + [None]
    labels = []
    for lo_p, hi_p in zip(boundaries, boundaries[1:]):
        if lo_p is None:
            labels.append(f"< {hi_p}%")
        elif hi_p is None:
            labels.append(f"\u2265 {lo_p}%")
        else:
            labels.append(f"{lo_p}\u2013{hi_p}%")
    return labels


def _validate_km_inputs(data, time_col, event_col, group_col, risk_col,
                         break_type, correction, n_groups, n_permutations,
                         p_adjust):
    """Validate plot_km inputs (column existence, parameter values).

    Raises TypeError or ValueError on bad inputs.
    """
    _validate_km_data(data, time_col, event_col, group_col, risk_col)
    _validate_km_params(break_type, correction, n_groups, n_permutations,
                        p_adjust)


def _validate_km_data(data, time_col, event_col, group_col, risk_col):
    """Validate data type and column existence."""
    if isinstance(data, pd.DataFrame):
        required = [time_col, event_col]
        if risk_col is not None:
            required.append(risk_col)
        if group_col is not None:
            required.append(group_col)
        validate_columns(data, required, "plot_km")
    elif isinstance(data, dict):
        effective_risk_col = risk_col or "risk_score"
        for name, df in data.items():
            required = [time_col, event_col, effective_risk_col]
            validate_columns(df, required, f"plot_km[{name}]")
    else:
        raise TypeError(
            f"plot_km() expected a pandas DataFrame or dict[str, DataFrame], "
            f"got {type(data).__name__}."
        )


def _validate_km_params(break_type, correction, n_groups, n_permutations,
                        p_adjust):
    """Validate parameter values for plot_km."""
    if break_type not in _VALID_BREAK_TYPES:
        raise ValueError(
            f"Unknown break_type {break_type!r}. "
            f"Choose from {sorted(_VALID_BREAK_TYPES)}."
        )
    if break_type == "maxstat" and correction not in _VALID_CORRECTIONS:
        raise ValueError(
            f"Unknown correction {correction!r}. "
            f"Choose from {sorted(_VALID_CORRECTIONS)}."
        )
    if break_type == "maxstat" and n_groups > 4:
        raise ValueError(
            f"n_groups={n_groups} exceeds maximum of 4 for maxstat. "
            f"Use n_groups=2, 3, or 4."
        )
    if p_adjust not in _VALID_P_ADJUST:
        raise ValueError(
            f"Unknown p_adjust {p_adjust!r}. "
            f"Choose from {sorted(_VALID_P_ADJUST)}."
        )
    if correction == "permutation" and n_permutations < 100:
        raise ValueError(
            f"n_permutations={n_permutations} is too low for reliable "
            f"permutation p-values. Use at least 100."
        )


def _validate_breaks(breaks, break_type):
    """Validate ``break_type`` and explicit ``breaks`` cut-points.

    Shared by ``plot_km`` and the ``rmst_test``/``logrank_test`` path so the
    plot and statistical-test APIs reject the same bad inputs.

    Raises
    ------
    ValueError
        If ``break_type`` is unknown; or ``breaks`` is empty, contains a
        non-finite value, or is not strictly increasing; or (for
        ``break_type="quantile"``) any value falls outside (0, 1).
    """
    if break_type not in _VALID_BREAK_TYPES:
        raise ValueError(
            f"Unknown break_type {break_type!r}. "
            f"Choose from {sorted(_VALID_BREAK_TYPES)}."
        )
    # maxstat computes its own cutpoints, so explicit breaks are unused.
    if breaks is None or break_type == "maxstat":
        return

    values = list(breaks)
    if not values:
        raise ValueError("breaks must contain at least one cut-point.")
    if not all(np.isfinite(b) for b in values):
        raise ValueError(
            f"breaks must all be finite numbers; got {values}."
        )
    if any(b >= nxt for b, nxt in zip(values, values[1:])):
        raise ValueError(
            f"breaks must be strictly increasing; got {values}."
        )
    if break_type == "quantile" and not all(0 < b < 1 for b in values):
        raise ValueError(
            f"Quantile breaks must be fractions in (0, 1); got {values}."
        )


def _warn_breaks_precedence(n_groups, breaks, break_type, stacklevel=3):
    """Warn when the grouping precedence discards a user-provided parameter.

    ``break_type="maxstat"`` finds its own cutpoints, so an explicit
    ``breaks`` is ignored (``n_groups`` still bounds the number of splits).
    Otherwise ``breaks`` fully determines the groups, so an explicit
    ``n_groups`` is ignored. Either discard is flagged with a
    ``SuggestionWarning`` per the "discarding user params" rule.
    """
    if break_type == "maxstat":
        if breaks is not None:
            warnings.warn(
                "breaks is ignored when break_type='maxstat'; the cutpoints "
                "are found automatically.",
                SuggestionWarning, stacklevel=stacklevel,
            )
        return
    if breaks is not None and n_groups is not None:
        warnings.warn(
            "Both n_groups and breaks were provided; n_groups is ignored "
            "because breaks fully determines the risk groups.",
            SuggestionWarning, stacklevel=stacklevel,
        )


def _resolve_km_defaults(data, risk_col, group_col, break_type, show_rmst,
                          subtitle):
    """Resolve RMST activation and auto-subtitle for plot_km.

    Returns (rmst_active, subtitle).
    """
    rmst_active = False
    if show_rmst:
        if risk_col is None and group_col is None:
            warnings.warn(
                "show_rmst=True requires risk_col or group_col; ignored for overall plots.",
                SuggestionWarning, stacklevel=3,
            )
        elif isinstance(data, dict):
            warnings.warn(
                "show_rmst=True is not supported for panel grid (dict) input; ignored.",
                SuggestionWarning, stacklevel=3,
            )
        else:
            rmst_active = True

    _risk_mode = isinstance(data, dict) or risk_col is not None
    if subtitle is None and _risk_mode and break_type != "maxstat" and not rmst_active:
        subtitle = "Risk group stratification by risk scores"

    return rmst_active, subtitle


def _ci_band_subtitle(subtitle, show_ci):
    """Lowest-priority fallback: name the 95% CI band in the subtitle.

    Only fills the subtitle when nothing else (risk-group, maxstat, RMST, or a
    user value) already set it and a CI ribbon is drawn. An explicit ``""``
    (user-suppressed subtitle) and any existing string are left untouched.
    """
    if subtitle is None and show_ci:
        return "95% CI"
    return subtitle


def _maxstat_subtitle(maxstat_info):
    """Build a subtitle string from a MaxstatResult."""
    n_cuts = len(maxstat_info.cutpoints)
    p = maxstat_info.p_value_corrected
    if n_cuts == 1:
        cut_str = f"cutpoint: {maxstat_info.cutpoints[0]:.3g}"
    else:
        cut_str = "cutpoints: " + ", ".join(
            f"{c:.3g}" for c in maxstat_info.cutpoints
        )
    return f"Maxstat {cut_str} (p = {p:.3g}, {maxstat_info.correction})"


def _create_risk_groups(df, risk_col, n_groups, breaks=None, break_type="quantile"):
    """Assign risk group integer labels; return (df, n_groups, group_labels).

    Parameters
    ----------
    breaks : list, optional
        Custom cut-points.
        - ``break_type="quantile"``: values in (0, 1), e.g. [0.25, 0.5, 0.75]
          -> boundaries at those quantiles of *risk_col*.
        - ``break_type="score"``: raw score values, e.g. [3, 20, 35]
          -> boundaries at those score values.
        When ``None``, falls back to *n_groups* uniform quantiles.
    break_type : {"quantile", "score"}
        Interpretation of *breaks*.  (``"maxstat"`` is resolved to
        ``"score"`` by ``_resolve_maxstat`` before reaching this function.)
    """
    scores = df[risk_col]

    if breaks is not None:
        return _create_risk_groups_with_breaks(df, scores, breaks, break_type, n_groups)

    # Default: uniform quantiles -- label by percentile range
    return _create_risk_groups_quantile(df, scores, n_groups)


def _create_risk_groups_with_breaks(df, scores, breaks, break_type, n_groups):
    """Create risk groups from explicit break points."""
    breaks = sorted(breaks)
    if break_type == "quantile":
        # Convert quantile fractions -> score thresholds
        score_cuts = [scores.quantile(q) for q in breaks]
    else:  # "score"
        score_cuts = breaks

    # Build bin edges: -inf ... cut1 ... cut2 ... +inf
    # Left-open, right-closed: (lo, hi]
    edges = [-np.inf] + score_cuts + [np.inf]
    df["risk_group"] = pd.cut(
        scores, bins=edges, labels=False, include_lowest=True,
    ).astype(int)
    n_actual = df["risk_group"].nunique()

    # Warn if some bins are empty (e.g. a cut-point falls outside the score
    # range), so the user knows fewer groups formed than the breaks requested.
    requested = len(score_cuts) + 1
    if n_actual < requested:
        warnings.warn(
            f"{requested} risk groups were requested via breaks but only "
            f"{n_actual} contain observations; the remaining groups are empty.",
            InternalWarning, stacklevel=3,
        )

    # Build human-readable labels
    if break_type == "quantile":
        pcts = [int(round(q * 100)) for q in breaks]
        labels = _pct_range_labels(pcts)
    else:
        labels = _build_score_break_labels(edges)
    return df, n_actual, labels


def _build_score_break_labels(edges):
    """Build human-readable labels from score-break bin edges."""

    def _fmt(v):
        return f"{v:.3g}" if not np.isinf(v) else None

    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if np.isinf(lo) and np.isinf(hi):
            labels.append("All")
        elif np.isinf(lo):
            labels.append(f"< {_fmt(hi)}")
        elif np.isinf(hi):
            labels.append(f"\u2265 {_fmt(lo)}")
        else:
            labels.append(f"{_fmt(lo)}\u2013{_fmt(hi)}")
    return labels


def _create_risk_groups_quantile(df, scores, n_groups):
    """Create risk groups using uniform quantiles."""
    requested_n = n_groups
    try:
        df["risk_group"] = pd.qcut(
            scores, q=n_groups, labels=False, duplicates="drop",
        )
    except ValueError:
        n_groups = min(n_groups, scores.nunique())
        df["risk_group"] = pd.qcut(
            scores, q=n_groups, labels=False, duplicates="drop",
        )
    actual = df["risk_group"].nunique()
    if actual < requested_n:
        warnings.warn(
            f"Requested {requested_n} risk groups but only {actual} could be "
            f"created from the score distribution. Reducing to {actual} groups.",
            InternalWarning, stacklevel=3,
        )
        n_groups = actual
    pcts = [int(round(i * 100 / n_groups)) for i in range(1, n_groups)]
    labels = _pct_range_labels(pcts)
    return df, n_groups, labels


def _resolve_axes_params(df, time_col, time_limit, xlim, ylim, time_unit):
    """Resolve common axis parameters for KM renderers.

    Returns (time_limit, x_lo, x_hi, y_lo, y_hi, time_label).
    """
    if time_limit is None:
        time_limit = df[time_col].max()
    x_lo, x_hi = xlim if xlim is not None else (0, time_limit)
    y_lo, y_hi = ylim if ylim is not None else (0, 1.05)
    time_label = f"Time ({time_unit})" if time_unit else "Time"
    return time_limit, x_lo, x_hi, y_lo, y_hi, time_label


def _resolve_risk_groups(df, time_col, event_col, risk_col, n_groups,
                         breaks, break_type, correction, subtitle,
                         n_permutations, random_state):
    """Resolve maxstat + create risk groups + override labels.

    Returns (df, n_groups, group_labels, maxstat_info, subtitle, risk_colors).
    """
    from mlsurv.viz._palettes import get_risk_colors
    from mlsurv.viz._survival_maxstat import _resolve_maxstat

    n_groups, breaks, break_type, maxstat_info = _resolve_maxstat(
        df, time_col, event_col, risk_col,
        n_groups, breaks, break_type, correction,
        n_permutations=n_permutations, random_state=random_state,
    )
    if maxstat_info is not None and subtitle is None:
        subtitle = _maxstat_subtitle(maxstat_info)

    df, n_groups, group_labels = _create_risk_groups(
        df, risk_col, n_groups, breaks=breaks, break_type=break_type,
    )
    if maxstat_info is not None:
        group_labels = list(_MAXSTAT_LABELS.get(n_groups, group_labels))
    risk_colors = get_risk_colors(n_groups)

    return df, n_groups, group_labels, maxstat_info, subtitle, risk_colors
