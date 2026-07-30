"""Hazard ratio visualizations.

Contains ``compute_hazard_ratios``, ``plot_hazard_ratios``,
``plot_hazard_ratios_comparison``, and supporting helpers.
"""

import math
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mlsurv._shared import covariance_reason
from mlsurv._warnings import (
    ClampWarning, InternalWarning, MethodologicalWarning, SuggestionWarning,
)
from mlsurv.viz._palettes import (
    FeatureGroupsType,
    MANUSCRIPT_DEFAULTS,
    _add_group_legend_traces,
    _add_group_separator_shapes,
    _make_group_strip_trace,
    _resolve_groups,
    _sort_features_by_group,
    get_model_color,
    hex_to_rgba,
)
from mlsurv.viz._features_helpers import _importance_full_title
from mlsurv.viz._plotly_helpers import asymmetric_error_bar
from mlsurv.viz._theme import (
    adaptive_size, apply_size_overrides, create_figure,
    margin_for_labels, truncate_labels,
)
from mlsurv.viz._validation import validate_columns

_EXP_CLIP = 700  # np.exp(709.78) overflows float64; 700 gives safe margin

# Axis range used whenever the data cannot supply one: no rows, or no hazard
# ratio that a log axis can place. Kept in one place so the fallbacks and the
# messages that name it cannot drift apart.
_DEFAULT_HR_RANGE = (0.25, 4.0)

# Largest and smallest powers of ten float64 holds. Beyond these `10 ** exp`
# becomes an exact Python int that float() refuses to convert.
_MIN_DECADE, _MAX_DECADE = -308, 308
# Above this many decades, label powers of ten only; above the second, thin
# those too. Both sit well clear of any ordinary hazard-ratio range.
_DENSE_DECADE_LIMIT = 8
_MAX_DECADE_TICKS = 12

# exp(coef) is a genuine hazard ratio only for models whose
# hazard_ratio_valid flag is True (unpenalized Cox PH).
HR_NOT_VALID_MSG = (
    "{name} coefficients are not hazard ratios: exp(coef) is a valid "
    "multivariable hazard ratio only for unpenalized Cox PH models "
    "(hazard_ratio_valid=True). Penalized (CoxNet) and ranking (FSSVM) "
    "coefficients are not valid hazard ratios, and AFT coefficients are not "
    "log-hazards. Use plot_coefficients() or plot_coefficient_comparison() "
    "to inspect coefficients instead."
)

# Emitted when a non-HR-valid model is dropped from a multi-model HR plot.
DROP_HR_MSG = (
    "Dropping '{name}' from the hazard-ratio plot: its coefficients are not "
    "hazard ratios (exp(coef) is a valid HR only for unpenalized Cox PH). "
    "Use plot_coefficients() to inspect its coefficients."
)


def _safe_exp(x):
    """Exponentiate with clipping to avoid float64 overflow."""
    arr = np.asarray(x, dtype=float)
    clipped = np.clip(arr, -_EXP_CLIP, _EXP_CLIP)
    n_clipped = np.count_nonzero(arr != clipped)
    if n_clipped:
        warnings.warn(
            f"{n_clipped} coefficient(s) clamped to [{-_EXP_CLIP}, {_EXP_CLIP}] "
            f"before exp() to prevent overflow. Displayed hazard ratios for "
            f"these features are capped, not exact. "
            f"Consider stronger regularization.",
            ClampWarning,
            stacklevel=2,
        )
    return np.exp(clipped)


def _plottable_hr_values(series):
    """Values a log axis can place, and a count of the ``±inf`` and ``0`` ones.

    A hazard-ratio axis is logarithmic, so a value must be finite **and
    strictly positive** to sit on it. Both ends of a Cox confidence interval
    can leave that range on the same degenerate fit: for a separated feature
    ``exp(coef + 1.96*se)`` overflows float64 to ``+inf`` and
    ``exp(coef - 1.96*se)`` underflows to exactly ``0.0``.
    ``Series.dropna()`` catches neither — it removes NaN only — so before this
    was guarded the infinity reached the log-decade tick computation, where
    ``math.ceil`` raises ``OverflowError``.

    Returning the count instead of warning here is what lets a multi-model plot
    emit one warning carrying the total, rather than a partial count per model.
    The count is taken against the *original* series, so an entry that is not a
    number at all is reported rather than coerced away in silence. Values that
    were already NaN are not counted: they were always dropped, and counting
    them would fire the warning on ordinary missing data.
    """
    vals = pd.to_numeric(series, errors="coerce")
    plottable = vals[np.isfinite(vals) & (vals > 0)]
    return plottable, int(series.notna().sum() - len(plottable))


def _warn_unplottable_hr(n_dropped):
    """Say what a hazard ratio the axis cannot place costs the reader."""
    if not n_dropped:
        return
    warnings.warn(
        f"{n_dropped} hazard ratio(s) or confidence bound(s) cannot be placed "
        f"on a log axis, because they are not positive finite numbers. Usually "
        f"that means exp(coef) overflowed to infinity or underflowed to zero. "
        f"They are left out of the axis range and are not drawn, so an affected "
        f"feature loses the whisker on that side, or its marker entirely when "
        f"the point estimate is the bad value. Rescale or drop the feature, or "
        f"recompute the univariate table with a larger penalizer.",
        ClampWarning,
        # 3 frames up is the plotting function, the most useful mlsurv frame to
        # name. The user's own call site is further up, through a learner method
        # whose depth varies, so there is no single correct level.
        stacklevel=3,
    )


def _extract_coefficients(model, feature_names):
    """Extract coefficient array and feature names from a Cox model."""
    if hasattr(model, 'get_coefficients'):
        coef = model.get_coefficients()
    elif hasattr(model, 'model_') and hasattr(model.model_, 'coef_'):
        coef = model.model_.coef_
    else:
        coef = None
    if coef is None:
        raise ValueError(f"{model.__class__.__name__} does not have coefficients")

    if feature_names is None:
        if hasattr(coef, "index"):
            feature_names = coef.index.tolist()
        elif hasattr(model, "feature_names_in_"):
            feature_names = model.feature_names_in_
        else:
            feature_names = [f"Feature {i}" for i in range(len(coef))]
    return coef, feature_names


def _compute_error_bounds(df, model, coef_values, ci_level, error_bars):
    """Add ci_lower, ci_upper, ci_available columns to *df* in place."""
    if hasattr(model, "model_") and hasattr(model.model_, "covariance_matrix_"):
        try:
            se = np.sqrt(np.diag(model.model_.covariance_matrix_))
            if error_bars == "se":
                multiplier = 1.0
            else:
                from scipy.stats import norm as _norm
                multiplier = _norm.ppf(1 - (1 - ci_level) / 2)
            df["ci_lower"] = _safe_exp(coef_values - multiplier * se)
            df["ci_upper"] = _safe_exp(coef_values + multiplier * se)
            df["ci_available"] = True
            return
        except (ValueError, IndexError, TypeError, np.linalg.LinAlgError) as e:
            # SuggestionWarning, not InternalWarning: the default filters
            # ignore InternalWarning, so the forest plot would lose every
            # whisker with nothing said about it.
            warnings.warn(
                f"Could not compute error bounds from covariance matrix "
                f"for {model.__class__.__name__}: {e}. Showing point estimates only.",
                SuggestionWarning,
                stacklevel=3,
            )
    else:
        # Only a hazard-ratio-valid model gets this far, and today that means
        # CoxPH, which does normally provide a covariance matrix. So the old
        # "does not provide a covariance matrix" was a claim about the model
        # type when the truth is about this one fit. Say what is observable
        # instead, and add the model's own reason when it recorded one.
        reason = covariance_reason(model)
        message = (
            f"No covariance matrix is available for "
            f"{model.__class__.__name__}, so confidence intervals cannot be "
            f"computed. In the returned frame, ci_lower and ci_upper equal hr."
        )
        if reason:
            message += f" {reason}"
        warnings.warn(message, SuggestionWarning, stacklevel=3)
    df["ci_lower"] = df["hr"]
    df["ci_upper"] = df["hr"]
    df["ci_available"] = False


def compute_hazard_ratios(
    model,
    feature_names: Optional[List[str]] = None,
    ci_level: float = 0.95,
    error_bars: str = "ci",
) -> pd.DataFrame:
    """Extract hazard ratios with error bounds from a fitted Cox model.

    Parameters
    ----------
    model : fitted model with ``hazard_ratio_valid=True``
        A model whose ``exp(coef)`` is a genuine hazard ratio (currently only
        unpenalized Cox PH). Other models raise ``ValueError`` — use
        ``plot_coefficients`` to inspect their coefficients instead.
    feature_names : list, optional
        Feature names. Uses ``model.feature_names_in_`` if available.
    ci_level : float
        Confidence level as a fraction in (0, 1). Default 0.95.
        Only used when ``error_bars="ci"``.
    error_bars : str, default ``"ci"``
        Type of error bars: ``"ci"`` for confidence intervals
        (``exp(coef ± z·SE)``), ``"se"`` for ±SE
        (``exp(coef ± SE)``).

    Returns
    -------
    DataFrame
        Columns: ``feature``, ``hr``, ``ci_lower``, ``ci_upper``.

    Raises
    ------
    ValueError
        If ``error_bars`` is not ``"ci"`` or ``"se"``, if ``ci_level`` is not
        strictly between 0 and 1, or if the model's coefficients are not
        hazard ratios (``hazard_ratio_valid`` is False).
    """
    if error_bars not in ("ci", "se"):
        raise ValueError(
            f"error_bars must be 'ci' or 'se', got {error_bars!r}."
        )
    if not 0 < ci_level < 1:
        raise ValueError(
            f"ci_level must be between 0 and 1 (exclusive), got {ci_level}. "
            f"Common values: 0.90, 0.95, 0.99."
        )

    # Gate before coefficient extraction so the message is about HR validity,
    # not "does not have coefficients".
    from mlsurv.viz._data import is_hazard_ratio_valid
    if not is_hazard_ratio_valid(model):
        raise ValueError(HR_NOT_VALID_MSG.format(name=model.__class__.__name__))

    coef, feature_names = _extract_coefficients(model, feature_names)
    coef_values = coef.values if hasattr(coef, "values") else np.array(coef)

    df = pd.DataFrame({
        "feature": feature_names,
        "hr": _safe_exp(coef_values),
    })

    _compute_error_bounds(df, model, coef_values, ci_level, error_bars)
    return df


def _error_bar_axis_label(has_ci: bool, error_bars: str, ci_level: float) -> str:
    """Build the x-axis label for the HR forest plot."""
    if not has_ci:
        return "Hazard Ratio"
    if error_bars == "se":
        return "Hazard Ratio (±SE)"
    return f"Hazard Ratio ({ci_level * 100:g}% CI)"


def _error_bar_band_label(has_ci, error_bars, ci_level):
    """Band string for the subtitle, e.g. '95% CI' / '±SE'; None when no band."""
    if not has_ci:
        return None
    if error_bars == "se":
        return "±SE"
    return f"{ci_level * 100:g}% CI"


def _finite_or_nan(values):
    """Marker positions with every non-finite entry replaced by NaN.

    Plotly serializes a numpy array through a binary channel that carries an
    infinity straight to the browser, where a marker position of ``inf`` is
    undefined. NaN travels the same channel but has an agreed meaning: draw
    nothing here. A feature whose hazard ratio overflowed keeps its row, and
    so its label, but gets no marker. (Whiskers go through
    :func:`_whisker_delta`, which has to look at the bound rather than the
    drawn value.)
    """
    arr = np.asarray(values, dtype=float)
    return np.where(np.isfinite(arr), arr, np.nan)


def _whisker_delta(bound, hr, side):
    """Whisker length from *hr* out to *bound*, NaN when there is none to draw.

    *side* is ``"upper"`` or ``"lower"``, naming which way the whisker runs.

    Two things have to be checked, and neither alone is enough:

    - The **bound**. A lower bound that underflowed to ``0.0`` still yields a
      perfectly finite delta — ``hr - 0.0`` is just ``hr`` — which would draw a
      full-width whisker down to a value the axis range deliberately excludes.
    - The **sign**. A bound on the wrong side of the point estimate gives a
      negative delta. Taking its magnitude would draw a whisker to a number
      that appears nowhere in the data, turning a visibly broken plot into a
      confident wrong one.

    Returns ``(deltas, n_wrong_side)``. The caller totals the count and warns
    once per figure, the same way :func:`_plottable_hr_values` works: warning
    here would fire up to six times for one plot and point at an internal
    helper rather than the plotting call.
    """
    bound = np.asarray(bound, dtype=float)
    hr = np.asarray(hr, dtype=float)
    usable = (np.isfinite(bound) & (bound > 0)
              & np.isfinite(hr) & (hr > 0))
    with np.errstate(invalid="ignore", over="ignore"):
        # np.where evaluates both branches, so inf - inf and huge - huge run
        # even where the result is discarded. Their warnings would be noise.
        signed = np.where(usable, bound - hr, np.nan)
        if side == "lower":
            signed = -signed
        wrong_side = usable & (signed < 0)

    return np.where(wrong_side, np.nan, signed), int(np.count_nonzero(wrong_side))


def _warn_wrong_side_bounds(n_wrong):
    """Report bounds that sit on the wrong side of their point estimate."""
    if not n_wrong:
        return
    warnings.warn(
        f"{n_wrong} confidence bound(s) lie on the wrong side of their hazard "
        f"ratio, so no whisker is drawn for them. Check that ci_lower_col and "
        f"ci_upper_col are not swapped.",
        InternalWarning,
        # Called straight from the plotting function, one frame shallower than
        # _warn_unplottable_hr, which goes through _hr_data_range. Both should
        # name the plotting function.
        stacklevel=2,
    )


def _hr_error_x(subset, hr_col, ci_lower_col, ci_upper_col, has_ci, color):
    """Build the ``error_x`` dict and the wrong-side count for one HR trace."""
    if not has_ci:
        return dict(visible=False), 0
    whisker_color = hex_to_rgba(color, 0.5)
    hr = subset[hr_col].values
    upper, n_up = _whisker_delta(subset[ci_upper_col].values, hr, "upper")
    lower, n_lo = _whisker_delta(subset[ci_lower_col].values, hr, "lower")
    return (
        asymmetric_error_bar(upper, lower, color=whisker_color),
        n_up + n_lo,
    )


def _build_hr_hover_ci(has_ci, error_bars):
    """Build the hover CI template string for HR plots."""
    if not has_ci:
        return "CI: not available"
    if error_bars == "se":
        return "±SE: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]"
    return f"CI: [%{{customdata[0]:.2f}}, %{{customdata[1]:.2f}}]"


def _hr_data_range_counted(df, hr_col="hr", ci_lower_col="ci_lower",
                           ci_upper_col="ci_upper"):
    """``_hr_data_range`` plus a count of the values it could not place.

    Split out so :func:`_hr_data_range_multi` can total the counts across
    models and warn once with the total, rather than once per model.
    """
    # Every column is counted before the early return below, so a frame with no
    # usable hazard ratio still reports its bad CI bounds rather than hiding them.
    hr_valid, n_dropped = _plottable_hr_values(df[hr_col])
    ci_lo = ci_hi = None
    if ci_lower_col in df.columns:
        ci_lo, n_bad = _plottable_hr_values(df[ci_lower_col])
        n_dropped += n_bad
    if ci_upper_col in df.columns:
        ci_hi, n_bad = _plottable_hr_values(df[ci_upper_col])
        n_dropped += n_bad

    if hr_valid.empty:
        # No rows, or no hazard ratio a log axis can place.
        return (*_DEFAULT_HR_RANGE, n_dropped)

    lo = hr_valid.min()
    hi = hr_valid.max()
    if ci_lo is not None and not ci_lo.empty:
        lo = min(lo, ci_lo.min())
    if ci_hi is not None and not ci_hi.empty:
        hi = max(hi, ci_hi.max())
    # Floor `lo` at a plottable magnitude, then guarantee a visible spread.
    # `lo * 2` overflows to inf at the top of the float range, which would hand
    # the tick computation exactly the infinity this function exists to keep
    # out of it, so only widen while the result stays finite.
    lo = max(lo, 1e-6)
    with np.errstate(over="ignore"):
        # lo is a numpy scalar, so the multiply itself warns before the guard
        # below can look at the result. That warning tells the reader nothing.
        spread = lo * 2
    if np.isfinite(spread):
        hi = max(hi, spread)
    return float(lo), float(hi), n_dropped


def _hr_data_range(df, hr_col="hr", ci_lower_col="ci_lower",
                   ci_upper_col="ci_upper"):
    """Extract min/max hazard ratio from *df*, including CI bounds if present.

    Hazard ratios and bounds a log axis cannot place are left out of the range
    (see :func:`_plottable_hr_values`) and reported via ``ClampWarning``.
    """
    lo, hi, n_dropped = _hr_data_range_counted(
        df, hr_col, ci_lower_col, ci_upper_col
    )
    _warn_unplottable_hr(n_dropped)
    return lo, hi


def _hr_data_range_multi(hr_data, top_features):
    """Extract min/max HR across multiple model DataFrames.

    *hr_data* is ``{model_name: DataFrame}`` with ``feature``, ``hr``,
    ``ci_lower``, ``ci_upper`` columns. Values a log axis cannot place are
    excluded from the range and reported once with the total across every
    model, so the reader gets one accurate count rather than a partial count
    per model.
    """
    lo, hi = float("inf"), 0.0
    n_dropped = 0
    for hr_df in hr_data.values():
        sub = hr_df[hr_df["feature"].isin(top_features)]
        if sub.empty:
            continue
        sub_lo, sub_hi, n_bad = _hr_data_range_counted(sub)
        n_dropped += n_bad
        lo = min(lo, sub_lo)
        hi = max(hi, sub_hi)
    _warn_unplottable_hr(n_dropped)
    if lo == float("inf"):
        warnings.warn(
            f"No matching features found across models; using default "
            f"tick range [{_DEFAULT_HR_RANGE[0]:g}, {_DEFAULT_HR_RANGE[1]:g}].",
            InternalWarning,
            stacklevel=2,
        )
        lo, hi = _DEFAULT_HR_RANGE
    return lo, hi


def _format_hr_tick(val):
    """Format a hazard ratio tick value as a clean string.

    Examples: 0.1 → '0.1', 0.5 → '0.5', 1 → '1', 2 → '2', 10 → '10'.

    Defensive: no in-package caller passes a non-finite tick value, because
    :func:`_compute_hr_ticks` guarantees finite positive ticks. A direct caller
    could, though, and ``int(inf)`` is an ``OverflowError``.
    """
    if not np.isfinite(val):
        if np.isnan(val):
            return ""
        return "∞" if val > 0 else "-∞"
    if abs(val) >= 1e15:
        # Spelling 1e303 out in full is a 304-character label that would swamp
        # the axis. Only reachable on a hazard ratio far outside any real one.
        return f"{val:g}"
    if val == int(val):
        return str(int(val))
    # Use up to 3 decimal places, strip trailing zeros
    return f"{val:.3f}".rstrip("0").rstrip(".")


def _valid_hr_bound(val, default, which):
    """Return *val* if a log axis can use it as a bound, else *default*.

    A hazard-ratio axis is logarithmic, so a bound must be finite and strictly
    positive: ``math.log10(inf)`` is ``inf`` (and ``math.ceil`` of that raises
    ``OverflowError``), ``math.log10(0)`` raises, and NaN poisons the decade
    arithmetic.

    Both library callers pass bounds that are already valid, so a substitution
    here means something upstream changed. Say so, the way the sibling default
    in :func:`_hr_data_range_multi` does, rather than quietly drawing an axis
    that does not match the data.
    """
    if np.isfinite(val) and val > 0:
        return val
    warnings.warn(
        f"Hazard-ratio axis {which} bound is {val!r}, which a log scale cannot "
        f"place; using {default:g} instead. The plotted range may not match "
        f"the data.",
        InternalWarning,
        stacklevel=3,
    )
    return default


def _compute_hr_ticks(hr_min, hr_max):
    """Generate nice 1-2-5 log-scale tick values spanning *hr_min* to *hr_max*.

    Includes 1.0, the null-effect reference line, whenever the bracketed range
    contains it. The returned ticks bracket the data range
    (``ticks[0] <= hr_min`` and ``ticks[-1] >= hr_max``) whenever both bounds
    are positive and inside the decades float64 can label. Returns a sorted
    list of tick values in linear (not log) space.

    A bound a log axis cannot place falls back to ``_DEFAULT_HR_RANGE``, the
    same range the empty-data path uses. The reader then gets an axis that does
    not bracket the data, which is still better than losing the figure.
    """
    hr_min = _valid_hr_bound(hr_min, _DEFAULT_HR_RANGE[0], "lower")
    hr_max = _valid_hr_bound(hr_max, _DEFAULT_HR_RANGE[1], "upper")
    if hr_max < hr_min:
        # Handed to us backwards; the span is what matters, not the order.
        hr_min, hr_max = hr_max, hr_min
    if hr_max == hr_min:
        # A single point has no span to bracket. Widen around it so the
        # "at least 3 ticks" promise below still holds, then re-check: at the
        # top of the float range `* 2` is inf and at the bottom `/ 2` flushes
        # to zero, either of which would walk straight back into the crash
        # this function exists to prevent.
        hr_min = _valid_hr_bound(hr_min / 2, _DEFAULT_HR_RANGE[0], "lower")
        hr_max = _valid_hr_bound(hr_max * 2, _DEFAULT_HR_RANGE[1], "upper")
        if hr_max <= hr_min:
            hr_min, hr_max = _DEFAULT_HR_RANGE

    # Clamp to the exponents float64 can hold. Past this, `10 ** exp` is an
    # exact Python int that float() cannot convert, which raises OverflowError
    # -- the very crash this function is guarding against, reached the other
    # way: through a bound that is finite but enormous rather than infinite.
    low_decade = max(_MIN_DECADE, math.floor(math.log10(hr_min)) - 1)
    high_decade = min(_MAX_DECADE, math.ceil(math.log10(hr_max)) + 1)

    decades = list(range(low_decade, high_decade + 1))
    # A span of hundreds of decades would otherwise label every one of them.
    # Thin the middle rather than hand the reader a solid wall of text.
    mults = (1, 2, 5) if len(decades) <= _DENSE_DECADE_LIMIT else (1,)
    if len(decades) > _MAX_DECADE_TICKS:
        stride = math.ceil(len(decades) / _MAX_DECADE_TICKS)
        decades = sorted(set(decades[::stride]) | {decades[-1]})

    # 10.0 ** exp keeps this in float arithmetic throughout. Filter the
    # products rather than just the exponents: 10.0 ** 308 is finite, but
    # 2 * 10.0 ** 308 is not, and an infinite candidate would be selected as
    # the bracket for any bound above it and returned as a tick.
    products = (float(mult) * 10.0 ** exp for exp in decades for mult in mults)
    candidates = sorted({v for v in products if np.isfinite(v)} | {1.0})

    # Tightest bracket: last candidate <= hr_min, first candidate >= hr_max.
    # Either search comes up empty when the bound sits outside the decades
    # float64 can label, so fall back to that end of the candidate list.
    at_or_below = [i for i, v in enumerate(candidates) if v <= hr_min]
    at_or_above = [i for i, v in enumerate(candidates) if v >= hr_max]
    lo_idx = max(at_or_below) if at_or_below else 0
    hi_idx = min(at_or_above) if at_or_above else len(candidates) - 1
    result = candidates[lo_idx:hi_idx + 1]

    # Ensure at least 3 ticks
    if len(result) < 3:
        result = sorted({hr_min, 1.0, hr_max, *result})

    return result


def select_plotted_hr_features(df, n_features, hr_col="hr"):
    """The rows a hazard-ratio forest plot will actually draw.

    Ranks by distance from HR = 1 and keeps the top *n_features*. An
    unplottable hazard ratio ranks **first**, not last: ``nlargest`` truncates
    before the axis range is computed, so a feature that sinks here would never
    reach ``_hr_data_range`` and would be dropped with no warning at all.
    ``log(0)`` gives ``-inf``, which sorts first on its own; a negative hazard
    ratio gives NaN, which sorts last, so it has to be said explicitly.

    Shared with the report layer so a description of the plot names the
    features the plot actually shows, not the whole table behind it.
    """
    ranked = df.copy()
    hr_vals = pd.to_numeric(ranked[hr_col], errors="coerce")
    with np.errstate(divide="ignore"):
        ranked["abs_log_hr"] = np.abs(np.log(hr_vals.where(hr_vals >= 0)))
    ranked.loc[hr_vals < 0, "abs_log_hr"] = np.inf
    return ranked.nlargest(n_features, "abs_log_hr")


def _add_hr_category_traces(fig, df, hr_col, ci_lower_col, ci_upper_col,
                            feature_col, has_ci, hover_ci, row=None, col=None):
    """Add category-coloured traces (Higher/N.S./Lower risk) to HR forest plot.

    Returns the number of confidence bounds found on the wrong side of their
    point estimate, totalled across the three traces, so the caller can report
    it in one warning instead of up to six.
    """
    category_style = [
        ("Higher risk", "#d62728"),
        ("N.S.", "#999999"),
        ("Lower risk", "#1f77b4"),
    ]
    n_wrong_side = 0
    for label, color in category_style:
        subset = df[df["category"] == label]
        if subset.empty:
            continue
        error_x, n_wrong = _hr_error_x(
            subset, hr_col, ci_lower_col, ci_upper_col, has_ci, color
        )
        n_wrong_side += n_wrong
        trace = go.Scatter(
            # NaN, not inf: an unplottable point estimate keeps its row (so the
            # feature is still labelled) but draws no marker.
            x=_finite_or_nan(subset[hr_col].values), y=subset[feature_col],
            mode="markers",
            marker=dict(size=10, color=color),
            name=label, legendgroup=label,
            error_x=error_x,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "HR: %{x:.2f}<br>"
                f"{hover_ci}<extra></extra>"
            ),
            # Deliberately the raw bounds, infinities included. The plot cannot
            # draw an unbounded interval, but the hover is where the reader
            # learns the real number, so it must not be sanitized away.
            customdata=np.column_stack([subset[ci_lower_col], subset[ci_upper_col]]),
        )
        if row is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)
    return n_wrong_side


def _add_hr_risk_annotations(fig, ann_y, d, xref="x"):
    """Add 'Lower risk / Higher risk' annotations for HR forest plot.

    Uses domain-relative positioning so annotations stay at the 25% and 75%
    marks of the axis regardless of data range or asymmetry.
    """
    domain_ref = f"{xref} domain"
    fig.add_annotation(
        x=0.25, y=ann_y,
        xref=domain_ref, yref="paper", xanchor="center",
        text="\u2190 Lower risk",
        showarrow=False,
        font=dict(size=d["font_size_annotation"], color="gray"),
    )
    fig.add_annotation(
        x=0.75, y=ann_y,
        xref=domain_ref, yref="paper", xanchor="center",
        text="Higher risk \u2192",
        showarrow=False,
        font=dict(size=d["font_size_annotation"], color="gray"),
    )


def _create_hr_figure(title, feature_order, feat_to_grp, grp_to_color,
                      display_labels, full_labels, height, width,
                      subtitle=None):
    """Create HR figure with optional group-color strip column.

    Returns ``(fig, main_row, main_col, has_groups)``.
    """
    has_groups = bool(feat_to_grp and grp_to_color)
    if has_groups:
        from plotly.subplots import make_subplots
        strip_width = 0.03
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[strip_width, 1 - strip_width],
            horizontal_spacing=0.02,
        )
        fig.update_layout(
            template="mlsurv_minimal",
            title=dict(text=_importance_full_title(f"<b>{title}</b>", subtitle)),
            width=width, height=height,
        )
        fig.add_trace(
            _make_group_strip_trace(
                feature_order, feat_to_grp, grp_to_color,
                orientation="vertical",
            ),
            row=1, col=1,
        )
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
        fig.update_yaxes(
            showgrid=False, showticklabels=True, tickmode="array",
            categoryorder="array", categoryarray=full_labels,
            tickvals=full_labels, ticktext=display_labels,
            row=1, col=1,
        )
        fig.add_vline(x=1, line_dash="dash", line_color="gray", row=1, col=2)
        return fig, 1, 2, True

    fig = create_figure(title=_importance_full_title(title, subtitle),
                        width=width, height=height)
    fig.add_vline(x=1, line_dash="dash", line_color="gray")
    return fig, None, None, False


def _configure_hr_axes(fig, has_groups, has_ci, error_bars, ci_level,
                       display_labels, full_labels, grp_to_color,
                       main_row, main_col, hr_tickvals=None):
    """Configure axes and group legend for HR forest plot.

    When *has_groups* is True, updates subplot axes and adds group legend
    traces; otherwise updates the flat layout.

    *hr_tickvals* overrides the default x-axis ticks.  When ``None``, falls
    back to a hardcoded ``[0.25, 0.5, 1, 2, 4]`` sequence.
    """
    if hr_tickvals is None:
        hr_tickvals = [0.25, 0.5, 1, 2, 4]
    xaxis_kw = dict(
        title=dict(
            text=_error_bar_axis_label(has_ci, error_bars, ci_level),
            standoff=12,
        ),
        type="log",
        tickvals=hr_tickvals,
        ticktext=[_format_hr_tick(v) for v in hr_tickvals],
    )
    yaxis_kw = dict(
        title="",
        categoryorder="array", categoryarray=full_labels,
        tickmode="array", tickvals=full_labels, ticktext=display_labels,
    )
    if has_groups:
        fig.update_xaxes(xaxis_kw, row=1, col=2)
        fig.update_yaxes(yaxis_kw, showticklabels=False, row=1, col=2)
        _add_group_legend_traces(fig, grp_to_color, row=main_row, col=main_col)
    else:
        fig.update_layout(xaxis=xaxis_kw, yaxis=yaxis_kw)


def plot_hazard_ratios(
    df: pd.DataFrame,
    hr_col: str = "hr",
    ci_lower_col: str = "ci_lower",
    ci_upper_col: str = "ci_upper",
    feature_col: str = "feature",
    n_features: int = 20,
    title: str = "Multivariate CoxPH — Hazard Ratios",
    ci_level: float = 0.95,
    error_bars: str = "ci",
    width: Optional[int] = None,
    height: Optional[int] = None,
    feature_groups: FeatureGroupsType = None,
    group_colors: Optional[Dict[str, str]] = None,
    sort_by_group: bool = False,
) -> go.Figure:
    """Forest plot showing hazard ratios with error bars.

    Parameters
    ----------
    df : DataFrame
        Must contain ``hr_col``, ``ci_lower_col``, ``ci_upper_col``,
        and ``feature_col``.

        **DataFrame schema** (default column names)::

            feature   hr      ci_lower   ci_upper
            str       float   float      float
    hr_col : str
        Column with hazard ratio values.
    ci_lower_col : str
        Column with lower bound.
    ci_upper_col : str
        Column with upper bound.
    feature_col : str
        Column with feature names.
    n_features : int
        Maximum features to display.
    title : str
        Plot title.
    ci_level : float
        Confidence level as a fraction in (0, 1), used for axis label.
        Only used when ``error_bars="ci"``.
    error_bars : str, default ``"ci"``
        Type of error bars: ``"ci"`` for confidence intervals,
        ``"se"`` for ±SE.
    width : int, optional
        Override default figure width (pixels).
    height : int, optional
        Override default figure height (pixels).
    feature_groups : dict, list, Series, or None
        Feature-to-group mapping. See :func:`resolve_feature_group_colors`.
    group_colors : dict, optional
        Explicit ``{group: color}`` overrides.
    sort_by_group : bool, default=False
        When True and ``feature_groups`` is set, order features by group
        (primary) and by hazard-ratio magnitude within each group (secondary),
        with separators drawn between groups.

    Returns
    -------
    go.Figure

    Raises
    ------
    ValueError
        If ``df`` is missing any of the required columns, if ``error_bars``
        is not ``"ci"`` or ``"se"``, or if ``ci_level`` is not strictly
        between 0 and 1.
    """
    validate_columns(df, [feature_col, hr_col, ci_lower_col, ci_upper_col], "plot_hazard_ratios")
    if error_bars not in ("ci", "se"):
        raise ValueError(
            f"error_bars must be 'ci' or 'se', got {error_bars!r}."
        )
    if not 0 < ci_level < 1:
        raise ValueError(
            f"ci_level must be between 0 and 1 (exclusive), got {ci_level}. "
            f"Common values: 0.90, 0.95, 0.99."
        )
    df = select_plotted_hr_features(df, n_features, hr_col)
    df = df.sort_values(hr_col, ascending=True)

    # Detect whether real error bounds are available.
    if "ci_available" in df.columns:
        has_ci = df["ci_available"].any()
    else:
        # Legacy fallback for DataFrames not produced by compute_hazard_ratios
        ci_valid = df[ci_lower_col].notna() & df[ci_upper_col].notna()
        has_ci = ci_valid.any() and not (
            df.loc[ci_valid, ci_lower_col] == df.loc[ci_valid, hr_col]
        ).all()

    # Resolve feature groups
    all_features = df[feature_col].unique().tolist()
    feat_to_grp, grp_to_color = _resolve_groups(
        feature_groups, all_features, group_colors,
    )

    # Classify significance
    df["category"] = "N.S."
    if has_ci:
        df.loc[df[ci_lower_col] > 1, "category"] = "Higher risk"
        df.loc[df[ci_upper_col] < 1, "category"] = "Lower risk"

    # Name the error-bar band in a subtitle (in addition to the x-axis label),
    # only when error bars are actually drawn (has_ci).
    subtitle = _error_bar_band_label(has_ci, error_bars, ci_level)

    d = MANUSCRIPT_DEFAULTS
    margin_t = d["margin_t"] + 20 + (20 if subtitle else 0)
    margin_b = 70
    h = adaptive_size(len(df), per_item=30, base=120, min_size=d["height_small"])
    plot_h = h - margin_t - margin_b

    ann_y = 1.0 + 10 / plot_h
    legend_y = -(50 / plot_h)

    hover_ci = _build_hr_hover_ci(has_ci, error_bars)

    feature_order = df[feature_col].tolist()  # df already sorted by hr_col asc

    # sort_by_group: reorder features by group, then by HR within group
    if sort_by_group and feat_to_grp and grp_to_color:
        abs_log_hr = df[hr_col].apply(
            lambda x: abs(np.log(x)) if x > 0 else 0,
        )
        metric_vals = dict(zip(df[feature_col], abs_log_hr))
        feature_order = _sort_features_by_group(
            feature_order, feat_to_grp, grp_to_color, metric_values=metric_vals,
            reverse=True,
        )

    display_labels, full_labels = truncate_labels(feature_order)
    margin_l = margin_for_labels(display_labels)

    fig, main_row, main_col, has_groups = _create_hr_figure(
        title, feature_order, feat_to_grp, grp_to_color,
        display_labels, full_labels, h, d["width_single"], subtitle=subtitle,
    )

    # Always use risk-category coloring for markers (Higher/N.S./Lower);
    # group membership is shown via the color strip alongside.
    n_wrong_side = _add_hr_category_traces(
        fig, df, hr_col, ci_lower_col, ci_upper_col, feature_col,
        has_ci, hover_ci, row=main_row, col=main_col,
    )
    _warn_wrong_side_bounds(n_wrong_side)

    # Compute dynamic ticks from the actual data range
    hr_min, hr_max = _hr_data_range(df, hr_col, ci_lower_col, ci_upper_col)
    ticks = _compute_hr_ticks(hr_min, hr_max)

    _configure_hr_axes(
        fig, has_groups, has_ci, error_bars, ci_level,
        display_labels, full_labels, grp_to_color,
        main_row, main_col, hr_tickvals=ticks,
    )

    if has_groups:
        fig.update_layout(
            margin=dict(l=margin_l, t=margin_t, b=margin_b),
            legend=dict(
                x=1.02, xanchor="left", y=0.5, yanchor="middle",
            ),
        )
    else:
        fig.update_layout(
            margin=dict(l=margin_l, t=margin_t, b=margin_b),
            legend=dict(
                orientation="h", yanchor="top", y=legend_y,
                xanchor="center", x=0.5, itemwidth=30,
            ),
        )

    fig.update_traces(
        marker=dict(size=round(d["marker_size_main"] * 1.5)),
        selector=dict(type="scatter"),
    )
    _add_hr_risk_annotations(fig, ann_y, d, xref="x2" if has_groups else "x")

    if sort_by_group and feat_to_grp and grp_to_color:
        _add_group_separator_shapes(fig, feature_order, feat_to_grp, axis="y")

    apply_size_overrides(fig, width, height)
    return fig


def _safe_extract(models, extractor, label, stacklevel=2):
    """Call *extractor(model_name, model)* for each model, skipping failures.

    Returns ``{model_name: result}`` for models that succeeded.
    """
    results = {}
    for model_name, model in models.items():
        try:
            result = extractor(model_name, model)
        except ValueError as exc:
            warnings.warn(
                f"Model '{model_name}' could not be included in {label}: "
                f"{exc}",
                InternalWarning,
                stacklevel=stacklevel,
            )
            continue
        except AttributeError as exc:
            warnings.warn(
                f"Model '{model_name}' raised an unexpected AttributeError "
                f"during {label}: {exc}. This may indicate a bug in the "
                f"model implementation.",
                InternalWarning,
                stacklevel=stacklevel,
            )
            continue
        if result is None:
            warnings.warn(
                f"Model '{model_name}' returned no data and will be "
                f"excluded from {label}.",
                InternalWarning,
                stacklevel=stacklevel,
            )
            continue
        results[model_name] = result
    return results


def _collect_hr_data(models):
    """Compute HRs for each model (or accept pre-computed DataFrames).

    Returns ``(hr_data, all_features_set)``.
    """
    def _extract(_name, model_or_df):
        if isinstance(model_or_df, pd.DataFrame):
            return model_or_df
        return compute_hazard_ratios(model_or_df)

    hr_data = _safe_extract(models, _extract, "HR comparison", stacklevel=3)
    all_features_set: set = set()
    for hr_df in hr_data.values():
        all_features_set.update(hr_df["feature"].tolist())
    return hr_data, all_features_set


def _rank_features_by_abs_log_hr(hr_data, all_features_set):
    """Rank features by average |log(HR)| across models.

    Returns a dict mapping feature name to average absolute log HR.

    A feature no model could place on a log axis ranks at the top, matching
    the single-model plot. Ranking it last instead would cut it in the
    ``top_features`` truncation, and the reader would lose both the feature
    and the warning that explains why. A feature simply absent from every
    model ranks at the bottom, as before: there is nothing to say about it.
    """
    avg_abs = {}
    for feat in all_features_set:
        vals = []
        n_unplottable = 0
        for mdf in hr_data.values():
            if feat not in mdf["feature"].values:
                continue
            hr_val = mdf.loc[mdf["feature"] == feat, "hr"].iloc[0]
            if hr_val > 0 and np.isfinite(hr_val):
                vals.append(np.abs(np.log(hr_val)))
            else:
                n_unplottable += 1
        if vals:
            avg_abs[feat] = np.mean(vals)
        else:
            avg_abs[feat] = np.inf if n_unplottable else 0
    return avg_abs


def _log_ratio_delta(numerator, denominator):
    """log10 of a ratio for an error-bar delta, NaN wherever it is not finite.

    An infinite bound over a finite one is silently ``inf``; ``inf/inf``, a
    zero denominator, and a huge-over-tiny finite pair each raise a bare numpy
    ``RuntimeWarning`` instead. Both outcomes are useless to the reader, so
    suppress the warnings and map every non-finite result to NaN, which Plotly
    understands as "draw no whisker on this side".
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        delta = np.log10(np.asarray(numerator, dtype=float)
                         / np.asarray(denominator, dtype=float))
    return np.where(np.isfinite(delta), delta, np.nan)


def _add_hr_comparison_traces(fig, hr_data, top_features, d, palette,
                              row=None, col=None):
    """Add per-model scatter traces to HR comparison figure."""
    y_positions = {feat: i for i, feat in enumerate(top_features[::-1])}
    n_models = len(hr_data)
    offset_step = 0.25

    for j, (model_name, hr_df) in enumerate(hr_data.items()):
        color = get_model_color(model_name, palette=palette)
        offset = (j - (n_models - 1) / 2) * offset_step
        plot_df = hr_df[hr_df["feature"].isin(top_features)].copy()
        y_vals = [y_positions[feat] + offset for feat in plot_df["feature"]]

        # Check if real CI bounds are available
        if "ci_available" in plot_df.columns:
            has_ci = plot_df["ci_available"].any()
        else:
            ci_valid = (
                plot_df["ci_lower"].notna()
                & plot_df["ci_upper"].notna()
                & (plot_df["ci_lower"] > 0)
                & (plot_df["ci_upper"] > 0)
                & ~(plot_df["ci_lower"] == plot_df["hr"])
            )
            has_ci = ci_valid.any()

        error_x = None
        if has_ci:
            error_x = asymmetric_error_bar(
                _log_ratio_delta(plot_df["ci_upper"].values, plot_df["hr"].values),
                _log_ratio_delta(plot_df["hr"].values, plot_df["ci_lower"].values),
                color=color, thickness=1.5,
            )

        hover_ci = (
            "CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]"
            if has_ci else "CI: not available"
        )

        trace = go.Scatter(
            x=_finite_or_nan(plot_df["hr"].values), y=y_vals, mode="markers",
            marker=dict(size=round(d["marker_size_main"] * 1.5), color=color),
            name=model_name,
            error_x=error_x,
            customdata=np.column_stack([
                plot_df["ci_lower"].values, plot_df["ci_upper"].values,
            ]),
            hovertemplate=(
                f"<b>{model_name}</b><br>"
                "%{text}<br>"
                f"HR: %{{x:.2f}}<br>{hover_ci}<extra></extra>"
            ),
            text=plot_df["feature"],
        )
        if row is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)


def plot_hazard_ratios_comparison(
    models: dict,
    feature_names: Optional[List[str]] = None,
    n_features: int = 15,
    title: str = "Hazard Ratios by Model",
    width: Optional[int] = None,
    height: Optional[int] = None,
    palette=None,
    feature_groups: FeatureGroupsType = None,
    group_colors: Optional[Dict[str, str]] = None,
    sort_by_group: bool = False,
) -> go.Figure:
    """Compare hazard ratios across multiple sources.

    Parameters
    ----------
    models : dict
        ``{name: fitted_model_or_DataFrame}`` dictionary.  Values can be
        fitted Cox models (``compute_hazard_ratios`` is called) **or**
        pre-computed DataFrames with ``feature``, ``hr``, ``ci_lower``,
        ``ci_upper`` columns.
    feature_names : list, optional
        Feature names to display.
    n_features : int
        Maximum features per model.
    title : str
        Plot title.
    width : int, optional
        Figure width in pixels. Overrides the theme default.
    height : int, optional
        Figure height in pixels. Overrides the theme default.
    palette : dict, optional
        ``{model_name: hex_color}`` overrides.
    feature_groups : dict, list, or Series, optional
        Feature-to-group mapping.  When provided (and groups resolve), a
        colored group strip and legend are drawn.  To additionally reorder
        features so groups are contiguous and add separator lines between
        them, set ``sort_by_group=True``.
    group_colors : dict, optional
        ``{group_name: hex_color}`` overrides for group colours.
    sort_by_group : bool, default=False
        When True (and ``feature_groups``/``group_colors`` resolve), reorder
        features so groups are contiguous and draw separator lines between
        groups. The colored group strip and legend appear whenever feature
        groups resolve, regardless of this flag.

    Returns
    -------
    go.Figure

    Raises
    ------
    ValueError
        If none of the supplied models yield hazard-ratio coefficients.

    Warns
    -----
    MethodologicalWarning
        For each non-DataFrame model whose coefficients are not hazard
        ratios; such models are dropped from the plot.
    """
    from mlsurv.viz._data import is_hazard_ratio_valid

    # Drop fitted models whose exp(coef) is not a hazard ratio, with a clear
    # warning. DataFrame values are kept as-is (a precomputed frame can't be
    # inspected). The compute_hazard_ratios gate still backstops any leak.
    filtered = {}
    for name, value in models.items():
        if not isinstance(value, pd.DataFrame) and not is_hazard_ratio_valid(value):
            warnings.warn(
                DROP_HR_MSG.format(name=name),
                MethodologicalWarning,
                stacklevel=2,
            )
            continue
        filtered[name] = value

    hr_data, all_features_set = _collect_hr_data(filtered)
    if not hr_data:
        raise ValueError("No models with coefficients found")

    avg_abs = _rank_features_by_abs_log_hr(hr_data, all_features_set)
    top_features = sorted(avg_abs, key=avg_abs.get, reverse=True)[:n_features]

    # Resolve feature groups for ordering and visual separators
    feat_to_grp, grp_to_color = _resolve_groups(
        feature_groups, top_features, group_colors,
    )

    if sort_by_group and feat_to_grp and grp_to_color:
        # Re-order top features by group, preserving within-group ranking
        top_features = _sort_features_by_group(
            top_features, feat_to_grp, grp_to_color, metric_values=avg_abs,
        )

    d = MANUSCRIPT_DEFAULTS
    n_actual = len(top_features)
    h = adaptive_size(n_actual, per_item=28, base=80, min_size=d["height_medium"])

    display_labels, full_labels = truncate_labels(top_features[::-1])
    margin_l = margin_for_labels(display_labels)

    has_groups = bool(feat_to_grp and grp_to_color)
    if has_groups:
        from plotly.subplots import make_subplots
        strip_width = 0.03
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[strip_width, 1 - strip_width],
            horizontal_spacing=0.02,
        )
        fig.update_layout(
            template="mlsurv_minimal",
            title=dict(text=f"<b>{title}</b>"),
            width=d["width_double"], height=h,
        )
        fig.add_trace(
            _make_group_strip_trace(
                top_features[::-1], feat_to_grp, grp_to_color,
                orientation="vertical",
            ),
            row=1, col=1,
        )
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
        fig.update_yaxes(
            showgrid=False, showticklabels=True, tickmode="array",
            tickvals=list(range(n_actual)), ticktext=display_labels,
            row=1, col=1,
        )
        main_row, main_col = 1, 2
    else:
        fig = create_figure(title=title, width="medium", height=h)
        main_row, main_col = None, None

    if has_groups:
        fig.add_vline(x=1, line_dash="dash", line_color="gray", row=1, col=2)
    else:
        fig.add_vline(x=1, line_dash="dash", line_color="gray")

    _add_hr_comparison_traces(fig, hr_data, top_features, d, palette,
                              row=main_row, col=main_col)

    # Compute dynamic ticks from the actual data range
    hr_min, hr_max = _hr_data_range_multi(hr_data, top_features)
    ticks = _compute_hr_ticks(hr_min, hr_max)
    ticktext = [_format_hr_tick(v) for v in ticks]

    if has_groups:
        # Separators only make sense once features are reordered group-contiguous,
        # so gate them on sort_by_group; the strip/legend stay group-gated.
        if sort_by_group:
            _add_group_separator_shapes(
                fig, top_features[::-1], feat_to_grp, axis="y",
            )
        _add_group_legend_traces(fig, grp_to_color, row=main_row, col=main_col)
        fig.update_xaxes(
            title="Hazard Ratio", type="log",
            tickvals=ticks, ticktext=ticktext,
            row=1, col=2,
        )
        fig.update_yaxes(
            showticklabels=False,
            tickmode="array",
            tickvals=list(range(n_actual)),
            ticktext=display_labels,
            row=1, col=2,
        )
        fig.update_layout(
            legend=dict(
                x=1.02, xanchor="left", y=0.5, yanchor="middle",
            ),
        )
    else:
        fig.update_layout(
            xaxis_title="Hazard Ratio", xaxis_type="log",
            xaxis=dict(tickvals=ticks, ticktext=ticktext),
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(n_actual)),
                ticktext=display_labels,
            ),
            yaxis_title="",
            legend=dict(title=dict(text="Model")),
        )

    fig.update_layout(margin=dict(l=margin_l, t=d["margin_t"] + 20))
    margin_t = d["margin_t"] + 20
    margin_b = fig.layout.margin.b or 50
    plot_h = h - margin_t - margin_b
    ann_y = 1.0 + 10 / plot_h
    _add_hr_risk_annotations(fig, ann_y, d, xref="x2" if has_groups else "x")

    apply_size_overrides(fig, width, height)
    return fig
