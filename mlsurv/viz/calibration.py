"""Calibration visualizations.

Public API:
    plot_calibration(data)        — single-model (list) or multi-model (dict)
    plot_calibration_overlay()    — predicted vs observed survival overlay
    compute_calibration()         — extract calibration bin data from fitted model
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mlsurv._warnings import SuggestionWarning
from mlsurv.utils import SurvivalArray, clip_eval_times
from mlsurv.viz._palettes import (
    MANUSCRIPT_DEFAULTS,
    get_model_color,
    get_model_display_name,
    get_risk_colors,
)
from mlsurv.viz._features_helpers import _importance_full_title
from mlsurv.viz._plotly_helpers import asymmetric_error_bar
from mlsurv.viz._theme import add_ci_ribbon, apply_size_overrides, create_figure
from mlsurv.evaluation.metrics import _surv_funcs_to_probs
from mlsurv.evaluation._calibrators import (
    SurvivalCalibrator,
    apply_calibrator_safe,
)


# Okabe-Ito-derived timepoint colors for single-model calibration (colorblind-friendly)
_TIMEPOINT_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]

_VALID_CI_STYLES = {"ribbon", "bars", "none"}

_VALID_BIN_STRATEGIES = {"width", "quantile"}


def _calibration_band_subtitle(subtitle, has_ci):
    """Lowest-priority fallback: name the KM CI in the subtitle.

    Observed survival is a Kaplan-Meier estimate, so its interval is a 95% KM
    CI (drawn as a ribbon or error bars). Fills only when the caller left the
    subtitle unset and a CI is actually drawn; an explicit ``""`` (suppressed)
    or any user string is left untouched.
    """
    if subtitle is None and has_ci:
        return "95% CI (KM)"
    return subtitle


def _calibration_band_drawn(ci_style, records):
    """True if a CI band is actually drawn for these calibration records.

    Needs both an enabled ``ci_style`` and at least one record carrying CI
    bounds (``ci_lower``/``ci_upper``). The single-model schema always carries
    them; the comparison / multi-model schema makes them optional per model, so
    a ribbon/error bar appears only when at least one record has them.
    """
    return ci_style != "none" and any(
        r.get("ci_lower") is not None and r.get("ci_upper") is not None
        for r in records
    )


def _cal_interactive_title(title, t, subtitle):
    """Interactive calibration title: bold base + timepoint, optional subtitle."""
    text = f"<b>{title}</b> — t = {t:.1f}"
    if subtitle:
        text += (f"<br><span style='font-size:12px;font-weight:normal'>"
                 f"{subtitle}</span>")
    return text


# ── Compute helpers ─────────────────────────────────────────────────────────


def _resolve_default_eval_times(
    y_test: SurvivalArray,
    y_train: Optional[SurvivalArray],
    *,
    percentiles: List[int],
    caller: str,
    override_kwarg: str,
) -> np.ndarray:
    """Resolve data-driven default eval timepoints via ``auto_eval_times``.

    Re-raises ``ValueError`` from ``auto_eval_times`` with a contextual
    message telling the user how to override (``override_kwarg``).
    """
    from mlsurv.utils import auto_eval_times

    try:
        return auto_eval_times(y_test, percentiles=percentiles, y_train=y_train)
    except ValueError as exc:
        raise ValueError(
            f"{caller} could not derive default eval timepoint(s) from the "
            f"available survival data: {exc} Pass {override_kwarg} explicitly "
            f"to override."
        ) from exc


def _km_survival_at(kmf, t):
    """Return (survival, ci_lower, ci_upper) at time *t* from a fitted KMF."""
    sf = kmf.survival_function_
    if len(sf) == 0:
        return np.nan, np.nan, np.nan

    ci = kmf.confidence_interval_survival_function_

    if t > sf.index.max():
        obs = sf.iloc[-1, 0]
        if len(ci) > 0:
            ci_last = ci.iloc[-1]
            return obs, ci_last.iloc[0], ci_last.iloc[1]
        return obs, np.nan, np.nan

    obs = kmf.survival_function_at_times(t).values[0]
    if len(ci) > 0:
        ci_at_t = ci.loc[ci.index <= t].iloc[-1]
        return obs, ci_at_t.iloc[0], ci_at_t.iloc[1]
    return obs, np.nan, np.nan


def _assign_calibration_bins(pred_probs, n_bins, bin_strategy, context=""):
    """Assign each predicted probability to a bin.

    Returns ``(bin_indices, bin_labels)`` where ``bin_labels`` is the ordered
    list of bin labels to iterate, ascending by predicted probability (so the
    resulting calibration points stay x-ordered for the line plot).

    - ``"width"``: ``n_bins`` equal-width bins across ``[0, 1]``.
    - ``"quantile"``: ``n_bins`` equal-count bins by quantiles of the predicted
      probability. ``pd.qcut(..., duplicates="drop")`` yields fewer than
      ``n_bins`` bins when predictions are tied. When fewer than two distinct
      values exist it cannot form quantile edges: it returns all-NaN, so
      ``labels`` comes back empty and we fall back to a single bin holding every
      patient (and emit a ``SuggestionWarning``). The ``ValueError`` guard
      around ``pd.qcut`` is defensive only — current pandas degrades to all-NaN
      rather than raising — but it keeps the fallback robust across pandas
      versions; the empty-``labels`` check is the load-bearing trigger, not the
      ``except``. This echoes the ``qcut``/``duplicates="drop"`` pattern in
      :func:`plot_calibration_overlay` (whose own fallback differs: it reduces
      the group count rather than collapsing to one).

    ``context`` is an optional ``"<model> at t=<horizon>"`` label folded into
    the fallback warning so multi-model callers identify which curve collapsed.
    """
    if bin_strategy == "quantile":
        try:
            bin_indices = pd.qcut(
                pred_probs, q=n_bins, labels=False, duplicates="drop",
            )
        except ValueError:
            bin_indices = None
        labels = []
        if bin_indices is not None:
            bin_indices = np.asarray(bin_indices, dtype=float)
            labels = [int(b) for b in np.unique(bin_indices) if not np.isnan(b)]
        if not labels:
            # Too few distinct predicted values to form quantile edges (qcut
            # maps everything to NaN -> labels is empty). Fall back to a single
            # bin over all patients, but warn: collapsing to one point is the
            # exact failure mode quantile binning is meant to avoid, so the
            # user needs to know it happened (their model's S(t|x) is nearly
            # constant), not silently see a one-point curve.
            where = f" for {context}" if context else ""
            warnings.warn(
                f"bin_strategy='quantile'{where}: predicted probabilities have "
                f"too few distinct values to form quantile bins; falling back "
                f"to a single bin over all patients, so the calibration curve "
                f"collapses to one point. The model's predicted S(t|x) is "
                f"nearly constant across patients at this horizon.",
                SuggestionWarning, stacklevel=2,
            )
            return np.zeros(len(pred_probs), dtype=int), [0]
        return bin_indices, labels

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(pred_probs, bins) - 1, 0, n_bins - 1)
    return bin_indices, list(range(n_bins))


def _compute_bin_calibration(
    pred_probs, y_test: SurvivalArray, t, n_bins, bin_strategy="width",
    context="",
):
    """Compute predicted vs observed survival per bin at time *t*.

    ``bin_strategy`` is ``"width"`` (equal-width bins across [0, 1], the
    default) or ``"quantile"`` (equal-count bins by predicted-probability
    quantiles). See :func:`_assign_calibration_bins`. ``context`` is an optional
    label (e.g. ``"coxph at t=24.0"``) passed through to the quantile fallback
    warning so the user knows which curve collapsed.

    Returns (predicted, observed, ci_lower_list, ci_upper_list).
    """
    from lifelines import KaplanMeierFitter

    bin_indices, bin_labels = _assign_calibration_bins(
        pred_probs, n_bins, bin_strategy, context=context,
    )

    observed, predicted, ci_lower_list, ci_upper_list = [], [], [], []

    for b in bin_labels:
        mask = bin_indices == b
        if mask.sum() < 5:
            continue

        predicted.append(pred_probs[mask].mean())

        kmf = KaplanMeierFitter()
        kmf.fit(y_test["time"][mask], y_test["event"][mask])

        obs, ci_low, ci_high = _km_survival_at(kmf, t)
        if np.isnan(obs):
            continue
        observed.append(obs)
        ci_lower_list.append(ci_low)
        ci_upper_list.append(ci_high)

    return predicted, observed, ci_lower_list, ci_upper_list


def compute_calibration(
    model,
    X_test: pd.DataFrame,
    y_test: SurvivalArray,
    eval_times: Optional[List[float]] = None,
    n_bins: int = 10,
    y_train: Optional[SurvivalArray] = None,
    calibrator: Optional[SurvivalCalibrator] = None,
    model_name: str = "model",
    bin_strategy: str = "width",
) -> List[Dict[str, Any]]:
    """Extract calibration bin data from a fitted model.

    Parameters
    ----------
    model : BaseSurvivalModel
        Fitted model with ``predict_survival_function()`` method.
    X_test : DataFrame
        Test features.
    y_test : SurvivalArray
        Test outcomes with 'time' and 'event' fields.
    eval_times : list, optional
        Timepoints to evaluate. When ``None``, defaults to the 25/50/75
        percentiles of training event times (or test event times if
        ``y_train`` is not provided), via :func:`mlsurv.utils.auto_eval_times`.
        The learner-facing ``plot_calibration`` always passes the resolved
        evaluation grid here, so this low-level default applies only to direct
        callers without a learner ``state``.
    n_bins : int
        Number of bins for grouping predicted probabilities.
    y_train : SurvivalArray, optional
        Training outcomes used to derive the default ``eval_times``. When
        provided, percentiles come from training event times so the
        evaluation points stay within the training time range.
    calibrator : SurvivalCalibrator, optional
        The model's attached ``S(t|x)`` calibrator (from
        ``train(calibrate=...)``). When provided, the predicted probabilities
        are warped to the calibrated scale before binning, so the curve matches
        the deployed model's evaluation metrics. ``None`` renders raw S(t|x).
    model_name : str, default="model"
        Used only to label the ``MethodologicalWarning`` emitted if the
        calibrator fails to apply (then the raw curve is shown).
    bin_strategy : {"width", "quantile"}, default="width"
        How predicted probabilities are grouped into bins. ``"width"`` uses
        ``n_bins`` equal-width bins across [0, 1]. ``"quantile"`` uses
        ``n_bins`` equal-count bins by predicted-probability quantiles, which
        lets a model with a narrow spread of predictions (e.g. GBSA) show a
        full curve instead of collapsing into one bin. Quantile binning gives
        an uneven x-axis, and the realized point count can still be fewer than
        ``n_bins`` -- from tied predictions (``duplicates="drop"``) or from
        bins that fall below the ~5-sample floor. A fully constant predictor
        collapses to a single bin (with a ``SuggestionWarning``).

    Returns
    -------
    list[dict]
        Each dict contains: ``eval_time``, ``time_index``,
        ``predicted``, ``observed``, ``ci_lower``, ``ci_upper``.

    Raises
    ------
    ValueError
        If ``model`` lacks a ``predict_survival_function()`` method or that
        method raises :class:`NotImplementedError`, or if ``bin_strategy`` is
        not one of the supported values.
    """
    if bin_strategy not in _VALID_BIN_STRATEGIES:
        raise ValueError(
            f"bin_strategy must be one of {sorted(_VALID_BIN_STRATEGIES)}, "
            f"got '{bin_strategy}'"
        )

    unsupported_msg = (
        f"{model.__class__.__name__} does not support predict_survival_function(). "
        "Calibration plots require survival probability predictions."
    )
    if not hasattr(model, "predict_survival_function"):
        raise ValueError(unsupported_msg)

    if eval_times is None:
        eval_times = list(_resolve_default_eval_times(
            y_test, y_train, percentiles=[25, 50, 75],
            caller="compute_calibration", override_kwarg="eval_times=[<float>, ...]",
        ))

    max_time = y_test["time"].max()
    try:
        surv_funcs = model.predict_survival_function(X_test)
    except NotImplementedError as e:
        raise ValueError(unsupported_msg) from e

    skipped = [t for t in eval_times if t > max_time]
    if skipped:
        warnings.warn(
            f"Calibration: {len(skipped)} eval time(s) exceed max observed "
            f"time ({max_time:.1f}) and were skipped.",
            SuggestionWarning, stacklevel=2,
        )

    # Build the full 2D raw S(t|x) for the in-range horizons in one shot so an
    # attached calibrator applies across all horizons together: probability
    # warpers map each column, while breslow_slope ignores these probs and
    # recomputes from pipeline.predict(X). apply_calibrator_safe is a no-op
    # passthrough when calibrator is None.
    in_range_times = [t for t in eval_times if t <= max_time]
    if in_range_times:
        probs_2d = _surv_funcs_to_probs(surv_funcs, in_range_times)
        probs_2d = apply_calibrator_safe(
            calibrator, pipeline=model, X=X_test,
            surv_probs=probs_2d, eval_times=in_range_times,
            model_name=model_name,
        )

    calibration_data = []
    col = 0
    for i, t in enumerate(eval_times):
        if t > max_time:
            continue

        pred_probs = probs_2d[:, col]
        col += 1
        predicted, observed, ci_lower, ci_upper = _compute_bin_calibration(
            pred_probs, y_test, t, n_bins, bin_strategy,
            context=f"{model_name} at t={t:.1f}",
        )

        if len(predicted) == 0:
            continue

        calibration_data.append({
            "eval_time": t,
            "time_index": i,
            "predicted": predicted,
            "observed": observed,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })

    return calibration_data


def _compute_multi_model_calibration_data(
    models: dict,
    X_test: pd.DataFrame,
    y_test: SurvivalArray,
    eval_time: float,
    n_bins: int,
    calibrators: Optional[Dict[str, SurvivalCalibrator]] = None,
    bin_strategy: str = "width",
) -> Dict[str, Dict[str, Any]]:
    """Compute per-model calibration bin data for one eval_time.

    ``calibrators`` maps ``model_name -> SurvivalCalibrator`` for models with an
    attached calibrator (``train(calibrate=...)``); those models' predicted
    ``S(t|x)`` is warped to the calibrated scale before binning. Models absent
    from the mapping (or all models when ``calibrators is None``) render raw.

    ``bin_strategy`` is ``"width"`` (default) or ``"quantile"``; see
    :func:`compute_calibration`.
    """
    if bin_strategy not in _VALID_BIN_STRATEGIES:
        raise ValueError(
            f"bin_strategy must be one of {sorted(_VALID_BIN_STRATEGIES)}, "
            f"got '{bin_strategy}'"
        )
    calibrators = calibrators or {}
    calibration_by_model: Dict[str, Dict[str, Any]] = {}

    for model_name, model in models.items():
        if not hasattr(model, "predict_survival_function"):
            warnings.warn(
                f"Skipping {model_name}: no predict_survival_function() method.",
                SuggestionWarning, stacklevel=2,
            )
            continue

        try:
            surv_funcs = model.predict_survival_function(X_test)
            pred_probs = _surv_funcs_to_probs(surv_funcs, [eval_time])
            pred_probs = apply_calibrator_safe(
                calibrators.get(model_name), pipeline=model, X=X_test,
                surv_probs=pred_probs, eval_times=[eval_time],
                model_name=model_name,
            )[:, 0]
            predicted, observed, ci_lower, ci_upper = _compute_bin_calibration(
                pred_probs, y_test, eval_time, n_bins, bin_strategy,
                context=f"{model_name} at t={eval_time:.1f}",
            )

            if len(predicted) == 0:
                warnings.warn(
                    f"Calibration for {model_name} at t={eval_time:.1f}: "
                    f"all bins had fewer than 5 samples. Consider increasing "
                    f"the test set size or reducing n_bins.",
                    SuggestionWarning, stacklevel=2,
                )
                continue

            calibration_by_model[model_name] = {
                "eval_time": eval_time,
                "predicted": predicted,
                "observed": observed,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        except NotImplementedError:
            warnings.warn(
                f"Skipping {model_name}: predict_survival_function() "
                f"not supported. Calibration requires survival probabilities.",
                SuggestionWarning, stacklevel=2,
            )
        except (ValueError, RuntimeError, ImportError) as e:
            warnings.warn(
                f"Could not compute calibration for {model_name}: {e}",
                SuggestionWarning, stacklevel=2,
            )

    return calibration_by_model


# ── Interactive plot (dropdown) ─────────────────────────────────────────────


def plot_calibration_interactive(
    data,
    title: str = "Calibration",
    default_time_index: Optional[int] = None,
    ci_style: str = "ribbon",
    width: Optional[int] = None,
    height: Optional[int] = None,
    palette=None,
    subtitle: Optional[str] = None,
) -> go.Figure:
    """Calibration plot with dropdown to select evaluation timepoint.

    Dispatches based on input type:

    - ``data`` is ``list[dict]`` → single-model calibration
      (output of :func:`compute_calibration`).
    - ``data`` is ``dict[float, dict]`` → multi-model comparison
      keyed by eval_time, each value a ``{model_name: cal_dict}`` mapping
      (output of :func:`_compute_multi_model_calibration_data` per time).

    Parameters
    ----------
    data : list[dict] or dict[float, dict[str, dict]]
        Calibration data grouped by timepoint.
    title : str
        Base plot title (timepoint appended via dropdown).
    default_time_index : int, optional
        Which dropdown entry is selected initially (default: median).
    ci_style : str
        ``"ribbon"``, ``"bars"``, or ``"none"``.
    width : int, optional
        Figure width in pixels. Overrides the theme default.
    height : int, optional
        Figure height in pixels. Overrides the theme default.
    palette : str or dict, optional
        Color palette override for the multi-model comparison (passed to
        :func:`~mlsurv.viz._palettes.get_model_color`). Ignored for the
        single-model view, which colors traces by timepoint.
    subtitle : str, optional
        Subtitle under the title. Defaults to ``"95% CI (KM)"`` when a band is
        drawn (``ci_style != "none"``); pass ``""`` to suppress it.

    Returns
    -------
    go.Figure

    Raises
    ------
    TypeError
        If ``data`` is neither a list nor a dict.
    ValueError
        If ``ci_style`` is not one of the supported styles.
    """
    if ci_style not in _VALID_CI_STYLES:
        raise ValueError(
            f"ci_style must be one of {sorted(_VALID_CI_STYLES)}, got '{ci_style}'"
        )

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # Multi-model is keyed by eval_time -> {model_name: cal_dict}.
        records = [r for by_model in data.values() for r in by_model.values()]
    else:
        raise TypeError(
            f"plot_calibration_interactive() expected list or dict, "
            f"got {type(data).__name__}"
        )
    subtitle = _calibration_band_subtitle(
        subtitle, _calibration_band_drawn(ci_style, records),
    )

    if isinstance(data, list):
        fig = _build_interactive_single(
            data, title, default_time_index, ci_style, subtitle=subtitle,
        )
    else:
        fig = _build_interactive_multi(
            data, title, default_time_index, ci_style, palette=palette,
            subtitle=subtitle,
        )

    apply_size_overrides(fig, width, height)
    return fig


def _build_interactive_single(
    calibration_data: List[Dict[str, Any]],
    title: str,
    default_time_index: Optional[int],
    ci_style: str,
    subtitle: Optional[str] = None,
) -> go.Figure:
    """Single-model calibration with dropdown for timepoint selection."""
    if not calibration_data:
        raise ValueError("No calibration data to plot.")

    d = MANUSCRIPT_DEFAULTS
    fig = create_figure(title=title, width=500, height="medium")

    # Diagonal reference — always visible (trace 0)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", line=dict(dash="dash", color="gray"),
        name="Perfect calibration", showlegend=True,
    ))

    # Ordered unique eval_times (preserving compute_calibration order)
    seen_times: List[float] = list(dict.fromkeys(
        item["eval_time"] for item in calibration_data
    ))

    # Add traces per timepoint, tracking index ranges
    time_trace_ranges: Dict[float, tuple] = {}

    for item in calibration_data:
        t = item["eval_time"]
        i = item["time_index"]
        predicted = item["predicted"]
        observed = item["observed"]
        ci_lower = item["ci_lower"]
        ci_upper = item["ci_upper"]

        start_idx = len(fig.data)
        color = _TIMEPOINT_COLORS[i % len(_TIMEPOINT_COLORS)]
        trace_name = f"t = {t:.1f}"
        group = f"t{t}"
        ci_custom = np.column_stack([ci_lower, ci_upper])
        hover = (
            f"<b>{trace_name}</b><br>"
            "Predicted: %{x:.2f}<br>"
            "Observed: %{y:.2f}<br>"
            "95% CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]"
            "<extra></extra>"
        )

        if ci_style == "ribbon":
            add_ci_ribbon(
                fig,
                np.array(predicted), np.array(ci_upper), np.array(ci_lower),
                color, name=trace_name, legendgroup=group,
            )

        if ci_style == "bars":
            fig.add_trace(go.Scatter(
                x=predicted, y=observed,
                mode="markers",
                marker=dict(size=d["marker_size_main"], color=color),
                error_y=asymmetric_error_bar(
                    [u - o for o, u in zip(observed, ci_upper)],
                    [o - l for o, l in zip(observed, ci_lower)],
                    color=color,
                ),
                name=trace_name, legendgroup=group,
                customdata=ci_custom, hovertemplate=hover,
            ))
        else:
            # "ribbon" and "none" both use markers+lines; only "ribbon"
            # prepends the CI ribbon above.
            fig.add_trace(go.Scatter(
                x=predicted, y=observed,
                mode="markers+lines",
                marker=dict(size=11, color=color),
                line=dict(color=color, width=3),
                name=trace_name, legendgroup=group,
                customdata=ci_custom, hovertemplate=hover,
            ))

        time_trace_ranges[t] = (start_idx, len(fig.data))

    return _finalize_interactive(
        fig, title, seen_times, time_trace_ranges, default_time_index,
        subtitle=subtitle,
    )


def _build_interactive_multi(
    data_by_time: Dict[float, Dict[str, Dict[str, Any]]],
    title: str,
    default_time_index: Optional[int],
    ci_style: str,
    palette=None,
    subtitle: Optional[str] = None,
) -> go.Figure:
    """Multi-model calibration comparison with dropdown for timepoint."""
    if not data_by_time:
        raise ValueError("No calibration data to plot.")

    fig = create_figure(title=title, width=500, height="medium")

    # Diagonal reference (trace 0)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", line=dict(dash="dash", color="gray"),
        name="Perfect", showlegend=True,
    ))

    seen_times = sorted(data_by_time.keys())
    time_trace_ranges: Dict[float, tuple] = {}

    for t in seen_times:
        calibration_by_model = data_by_time[t]
        start_idx = len(fig.data)

        for model_name, item in calibration_by_model.items():
            predicted = item["predicted"]
            observed = item["observed"]
            ci_lower = item.get("ci_lower")
            ci_upper = item.get("ci_upper")
            color = get_model_color(model_name, palette=palette)
            display_name = get_model_display_name(model_name)
            has_ci = ci_lower is not None and ci_upper is not None
            group = f"t{t}_{model_name}"

            if ci_style == "ribbon" and has_ci:
                add_ci_ribbon(
                    fig,
                    np.array(predicted), np.array(ci_upper), np.array(ci_lower),
                    color, name=display_name, legendgroup=group,
                )

            hover_kw: Dict[str, Any] = {}
            if has_ci:
                hover_kw["customdata"] = np.column_stack([ci_lower, ci_upper])
                hover_kw["hovertemplate"] = (
                    f"<b>{display_name}</b><br>"
                    "Predicted: %{x:.2f}<br>"
                    "Observed: %{y:.2f}<br>"
                    "95% CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]"
                    "<extra></extra>"
                )
            else:
                hover_kw["hovertemplate"] = (
                    f"<b>{display_name}</b><br>"
                    "Predicted: %{x:.2f}<br>"
                    "Observed: %{y:.2f}<extra></extra>"
                )

            error_kw: Dict[str, Any] = {}
            if ci_style == "bars" and has_ci:
                error_kw["error_y"] = asymmetric_error_bar(
                    [u - o for o, u in zip(observed, ci_upper)],
                    [o - l for o, l in zip(observed, ci_lower)],
                    color=color,
                )

            fig.add_trace(go.Scatter(
                x=predicted, y=observed,
                mode="markers+lines",
                marker=dict(size=11, color=color),
                line=dict(color=color, width=3),
                name=display_name, legendgroup=group,
                **hover_kw, **error_kw,
            ))

        time_trace_ranges[t] = (start_idx, len(fig.data))

    return _finalize_interactive(
        fig, title, seen_times, time_trace_ranges, default_time_index,
        subtitle=subtitle,
    )


def _finalize_interactive(
    fig: go.Figure,
    title: str,
    seen_times: List[float],
    time_trace_ranges: Dict[float, tuple],
    default_time_index: Optional[int],
    subtitle: Optional[str] = None,
) -> go.Figure:
    """Add dropdown, set initial visibility, apply layout."""
    n_traces = len(fig.data)

    if default_time_index is None:
        default_time_index = len(seen_times) // 2
    default_time_index = max(0, min(default_time_index, len(seen_times) - 1))

    # Build dropdown buttons
    buttons = []
    for t in seen_times:
        vis = [False] * n_traces
        vis[0] = True  # reference line
        start, end = time_trace_ranges[t]
        for j in range(start, end):
            vis[j] = True

        buttons.append(dict(
            label=f"t = {t:.1f}",
            method="update",
            args=[
                {"visible": vis},
                {"title.text": _cal_interactive_title(title, t, subtitle)},
            ],
        ))

    # Set initial visibility to default timepoint
    default_t = seen_times[default_time_index]
    for i in range(1, n_traces):
        fig.data[i].visible = False
    start, end = time_trace_ranges[default_t]
    for j in range(start, end):
        fig.data[j].visible = True

    fig.update_layout(
        title=dict(text=_cal_interactive_title(title, default_t, subtitle)),
        xaxis_title="Predicted Survival Probability",
        yaxis_title="Observed Survival Probability (KM)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1], automargin=True),
        legend=dict(
            x=1.02, y=0.98, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(l=80, r=120, t=120),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=1.02, xanchor="left",
            y=1.06, yanchor="bottom",
            buttons=buttons,
            active=default_time_index,
            showactive=True,
            bgcolor="#f5f7fa",
            bordercolor="#ddd",
            font=dict(size=11),
        )],
    )
    return fig


# ── Plot functions ──────────────────────────────────────────────────────────


def _validate_calibration_input(data):
    """Validate and classify calibration input data.

    Returns ``(data, input_type)`` where *input_type* is ``'list'`` or ``'dict'``.

    Raises
    ------
    ValueError
        If data is empty or missing required keys.
    TypeError
        If data is not a list or dict, or contains non-dict elements.
    """
    _REQUIRED_SINGLE = {"eval_time", "time_index", "predicted", "observed",
                        "ci_lower", "ci_upper"}
    _REQUIRED_MULTI = {"eval_time", "predicted", "observed"}

    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError(
                "plot_calibration() received an empty list. "
                "No calibration data to plot."
            )
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise TypeError(
                    f"plot_calibration() data[{i}] expected a dict, "
                    f"got {type(item).__name__}."
                )
            missing = _REQUIRED_SINGLE - set(item.keys())
            if missing:
                raise ValueError(
                    f"plot_calibration() data[{i}] missing required keys: "
                    f"{sorted(missing)}. Available: {sorted(item.keys())}"
                )
        return data, "list"
    if isinstance(data, dict):
        if len(data) == 0:
            raise ValueError(
                "plot_calibration() received an empty dict. "
                "No models had valid calibration data."
            )
        for model_name, item in data.items():
            if not isinstance(item, dict):
                raise TypeError(
                    f"plot_calibration() data[{model_name!r}] expected a dict, "
                    f"got {type(item).__name__}."
                )
            missing = _REQUIRED_MULTI - set(item.keys())
            if missing:
                raise ValueError(
                    f"plot_calibration() data[{model_name!r}] missing required keys: "
                    f"{sorted(missing)}. Available: {sorted(item.keys())}"
                )
        return data, "dict"
    raise TypeError(
        f"plot_calibration() expected list[dict] or dict[str, dict], "
        f"got {type(data).__name__}."
    )


def plot_calibration(
    data,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    palette=None,
    ci_style: str = "ribbon",
    subtitle: Optional[str] = None,
) -> go.Figure:
    """Calibration plot — single-model or multi-model comparison.

    Dispatches based on input type:

    - ``data`` is ``list[dict]`` → single-model calibration (one trace per timepoint)
    - ``data`` is ``dict[str, dict]`` → multi-model comparison at one timepoint

    Parameters
    ----------
    data : list[dict] or dict[str, dict]
        Calibration data. For single-model, each dict must contain:
        ``eval_time``, ``time_index``, ``predicted``, ``observed``,
        ``ci_lower``, ``ci_upper``. For multi-model, keyed by model name.

        **Single-model schema** (``list[dict]``)::

            {"eval_time": float, "time_index": int,
             "predicted": list[float], "observed": list[float],
             "ci_lower": list[float], "ci_upper": list[float]}

        **Multi-model schema** (``dict[str, dict]``)::

            {model_name: {"eval_time": float,
             "predicted": list[float], "observed": list[float],
             "ci_lower": list[float],
             "ci_upper": list[float]}}
    title : str, optional
        Plot title (auto-generated if None).
    width : int, optional
        Figure width in pixels (overrides default).
    height : int, optional
        Figure height in pixels (overrides default).
    palette : str or dict, optional
        Color palette override. Only applies to multi-model comparison
        mode; single-model mode colors traces by timepoint.
    ci_style : str
        Confidence interval display: ``"ribbon"`` (default) for shaded
        bands, ``"bars"`` for error bars, or ``"none"`` to hide.
    subtitle : str, optional
        Subtitle under the title. Defaults to ``"95% CI (KM)"`` when a band is
        drawn (``ci_style != "none"``); pass ``""`` to suppress it.

    Returns
    -------
    go.Figure

    Raises
    ------
    ValueError
        If ``ci_style`` is not one of the supported styles.
    """
    if ci_style not in _VALID_CI_STYLES:
        raise ValueError(
            f"ci_style must be one of {sorted(_VALID_CI_STYLES)}, got '{ci_style}'"
        )

    data, input_type = _validate_calibration_input(data)
    records = data if input_type == "list" else data.values()
    subtitle = _calibration_band_subtitle(
        subtitle, _calibration_band_drawn(ci_style, records),
    )

    if input_type == "list":
        fig = _plot_calibration_single(
            data, title=title or "Calibration Plot", ci_style=ci_style,
            subtitle=subtitle,
        )
    else:
        fig = _plot_calibration_comparison(
            data, title=title or "Calibration Comparison", palette=palette,
            ci_style=ci_style, subtitle=subtitle,
        )
    apply_size_overrides(fig, width, height)
    return fig


def _plot_calibration_single(
    calibration_data: List[Dict[str, Any]],
    title: str,
    ci_style: str = "ribbon",
    subtitle: Optional[str] = None,
) -> go.Figure:
    """Single-model calibration with one trace per timepoint.

    Colors represent timepoints (not models), so ``palette=`` does not apply.
    """
    d = MANUSCRIPT_DEFAULTS
    fig = create_figure(
        title=_importance_full_title(title, subtitle), width=450, height="medium",
    )

    # 45-degree reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", line=dict(dash="dash", color="gray"),
        name="Perfect calibration", showlegend=True,
    ))

    for item in calibration_data:
        t = item["eval_time"]
        i = item["time_index"]
        predicted = item["predicted"]
        observed = item["observed"]
        ci_lower = item["ci_lower"]
        ci_upper = item["ci_upper"]

        color = _TIMEPOINT_COLORS[i % len(_TIMEPOINT_COLORS)]
        trace_name = f"t={t}"
        ci_custom = np.column_stack([ci_lower, ci_upper])
        hover = (
            f"<b>{trace_name}</b><br>"
            "Predicted: %{x:.2f}<br>"
            "Observed: %{y:.2f}<br>"
            "95% CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]"
            "<extra></extra>"
        )

        if ci_style == "ribbon":
            add_ci_ribbon(
                fig,
                np.array(predicted), np.array(ci_upper), np.array(ci_lower),
                color, name=trace_name, legendgroup=trace_name,
            )
            fig.add_trace(go.Scatter(
                x=predicted, y=observed,
                mode="markers+lines",
                marker=dict(size=d["marker_size_main"], color=color),
                line=dict(color=color, width=d["line_width_secondary"]),
                name=trace_name,
                legendgroup=trace_name,
                customdata=ci_custom,
                hovertemplate=hover,
            ))
        elif ci_style == "bars":
            error_y_upper = [u - o for o, u in zip(observed, ci_upper)]
            error_y_lower = [o - l for o, l in zip(observed, ci_lower)]
            fig.add_trace(go.Scatter(
                x=predicted, y=observed,
                mode="markers",
                marker=dict(size=d["marker_size_main"], color=color),
                error_y=asymmetric_error_bar(error_y_upper, error_y_lower, color=color),
                name=trace_name,
                customdata=ci_custom,
                hovertemplate=hover,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=predicted, y=observed,
                mode="markers+lines",
                marker=dict(size=d["marker_size_main"], color=color),
                line=dict(color=color, width=d["line_width_secondary"]),
                name=trace_name,
                customdata=ci_custom,
                hovertemplate=hover,
            ))

    fig.update_layout(
        xaxis_title="Predicted Survival Probability",
        yaxis_title="Observed Survival Probability (KM)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1], automargin=True),
        legend=dict(
            title=dict(text="Timepoint"),
            x=0.02, y=0.98, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(l=80, r=15),
    )
    return fig


def _plot_calibration_comparison(
    calibration_by_model: Dict[str, Dict[str, Any]],
    title: str,
    palette=None,
    ci_style: str = "ribbon",
    subtitle: Optional[str] = None,
) -> go.Figure:
    """Multi-model calibration comparison at a single timepoint."""
    d = MANUSCRIPT_DEFAULTS
    fig = create_figure(
        title=_importance_full_title(title, subtitle), width=450, height="medium",
    )

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", line=dict(dash="dash", color="gray"),
        name="Perfect", showlegend=True,
    ))

    for model_name, item in calibration_by_model.items():
        predicted = item["predicted"]
        observed = item["observed"]
        ci_lower = item.get("ci_lower")
        ci_upper = item.get("ci_upper")
        color = get_model_color(model_name, palette=palette)
        display_name = get_model_display_name(model_name)
        has_ci = ci_lower is not None and ci_upper is not None

        if ci_style == "ribbon" and has_ci:
            add_ci_ribbon(
                fig,
                np.array(predicted), np.array(ci_upper), np.array(ci_lower),
                color, name=display_name, legendgroup=display_name,
            )

        hover_kw = {}
        if has_ci:
            hover_kw["customdata"] = np.column_stack([ci_lower, ci_upper])
            hover_kw["hovertemplate"] = (
                f"<b>{display_name}</b><br>"
                "Predicted: %{x:.2f}<br>"
                "Observed: %{y:.2f}<br>"
                "95% CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]"
                "<extra></extra>"
            )
        else:
            hover_kw["hovertemplate"] = (
                f"<b>{display_name}</b><br>"
                "Predicted: %{x:.2f}<br>"
                "Observed: %{y:.2f}<extra></extra>"
            )

        error_kw = {}
        if ci_style == "bars" and has_ci:
            error_kw["error_y"] = asymmetric_error_bar(
                [u - o for o, u in zip(observed, ci_upper)],
                [o - l for o, l in zip(observed, ci_lower)],
                color=color,
            )

        fig.add_trace(go.Scatter(
            x=predicted, y=observed,
            mode="markers+lines",
            marker=dict(size=d["marker_size_small"], color=color),
            line=dict(color=color, width=d["line_width_secondary"]),
            name=display_name,
            legendgroup=display_name,
            **hover_kw,
            **error_kw,
        ))

    fig.update_layout(
        xaxis_title="Predicted Survival Probability",
        yaxis_title="Observed Survival Probability (KM)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1], automargin=True),
        legend=dict(
            title=dict(text="Model"),
            x=0.02, y=0.98, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(l=80, r=15),
    )
    return fig


def plot_calibration_overlay(
    model,
    X_test: pd.DataFrame,
    y_test: SurvivalArray,
    eval_times: Optional[np.ndarray] = None,
    n_groups: int = 1,
    show_ci: bool = True,
    title: str = "Predicted vs. Observed Survival",
    width: Optional[int] = None,
    height: Optional[int] = None,
    time_unit: Optional[str] = None,
    calibrator: Optional[SurvivalCalibrator] = None,
    model_name: str = "model",
    y_train: Optional[SurvivalArray] = None,
    subtitle: Optional[str] = None,
) -> go.Figure:
    """Overlay model's average predicted survival curve on KM estimate.

    Parameters
    ----------
    model : BaseSurvivalModel
        Fitted model with ``predict_survival_function()``.
    X_test : DataFrame
        Test features.
    y_test : SurvivalArray
        Test outcomes with 'time' and 'event' fields.
    eval_times : ndarray, optional
        Timepoints for evaluation. Auto-computed if ``None``.
    n_groups : int
        ``1`` for overall comparison, ``>1`` for per-risk-group panels.
    show_ci : bool
        If True, show KM confidence intervals.
    title : str
        Figure title.
    width : int, optional
        Figure width in pixels. Overrides the theme default.
    height : int, optional
        Figure height in pixels. Overrides the theme default.
    time_unit : str, optional
        Unit label appended to the x-axis title (e.g. ``"months"``).
    calibrator : SurvivalCalibrator, optional
        The model's attached ``S(t|x)`` calibrator (from
        ``train(calibrate=...)``). When provided, the predicted survival curve
        is overlaid on the calibrated scale, matching the deployed model.
        ``None`` overlays the raw predicted curve.
    model_name : str, default="model"
        Used only to label the ``MethodologicalWarning`` emitted if the
        calibrator fails to apply (then the raw curve is shown).
    y_train : SurvivalArray, optional
        Training outcomes. When given, the default ``eval_times`` are clipped
        to the training time range so the fitted survival step functions are
        never queried outside their domain (which would raise). Ignored when
        an explicit ``eval_times`` is supplied. ``None`` (default) leaves the
        default horizons uncapped.
    subtitle : str, optional
        Subtitle under the title. Defaults to ``"95% CI (KM)"`` when a KM CI
        ribbon is drawn (``show_ci`` is True); pass ``""`` to suppress it.

    Returns
    -------
    go.Figure

    Raises
    ------
    ValueError
        If ``model`` lacks a ``predict_survival_function()`` method, or if
        the risk scores have fewer than two distinct values when grouping
        into risk strata.
    """
    from lifelines import KaplanMeierFitter

    if not hasattr(model, "predict_survival_function"):
        raise ValueError(
            f"{model.__class__.__name__} does not support "
            "predict_survival_function()."
        )

    surv_funcs = model.predict_survival_function(X_test)

    if eval_times is None:
        times_obs = y_test["time"]
        eval_times = np.linspace(times_obs.min(), times_obs.max() * 0.95, 50)
        if y_train is not None:
            # Keep default horizons within the training range; otherwise the
            # fitted survival step functions are queried past their domain
            # (t > train t_max) and raise.
            eval_times = clip_eval_times(eval_times, y_train)

    surv_probs = _surv_funcs_to_probs(surv_funcs, eval_times)
    # Calibrate the predicted curve so the overlay matches the deployed model
    # (no-op passthrough when calibrator is None). Warpers map surv_probs;
    # breslow_slope ignores them and recomputes from pipeline.predict(X).
    surv_probs = apply_calibrator_safe(
        calibrator, pipeline=model, X=X_test,
        surv_probs=surv_probs, eval_times=eval_times,
        model_name=model_name,
    )

    subtitle = _calibration_band_subtitle(subtitle, show_ci)

    if n_groups <= 1:
        fig = create_figure(
            title=_importance_full_title(title, subtitle),
            width="single", height="medium",
        )

        mean_predicted = surv_probs.mean(axis=0)
        fig.add_trace(go.Scatter(
            x=eval_times, y=mean_predicted,
            mode="lines", name="Predicted (mean)",
            line=dict(color="#d62728", width=2),
            hovertemplate="Time: %{x:.1f}<br>Predicted: %{y:.3f}<extra></extra>",
        ))

        kmf = KaplanMeierFitter()
        kmf.fit(y_test["time"], y_test["event"])
        sf = kmf.survival_function_
        ci = kmf.confidence_interval_survival_function_

        if show_ci:
            add_ci_ribbon(
                fig, sf.index.values,
                ci.iloc[:, 1].values, ci.iloc[:, 0].values,
                "#1f77b4",
            )

        fig.add_trace(go.Scatter(
            x=sf.index.values, y=sf.iloc[:, 0].values,
            mode="lines", name=f"Observed KM (n={len(y_test)})",
            line=dict(color="#1f77b4", width=2, shape="hv"),
            hovertemplate="Time: %{x:.1f}<br>Observed: %{y:.3f}<extra></extra>",
        ))

    else:
        from plotly.subplots import make_subplots

        risk_scores = model.predict(X_test)
        try:
            group_labels = pd.qcut(risk_scores, q=n_groups, labels=False, duplicates="drop")
        except ValueError:
            n_unique = pd.Series(risk_scores).nunique()
            if n_unique < 2:
                raise ValueError(
                    f"Cannot create risk groups: only {n_unique} unique risk "
                    f"score(s). Calibration overlay requires at least 2 "
                    f"distinct scores."
                )
            original_n = n_groups
            n_groups = min(n_groups, n_unique)
            warnings.warn(
                f"Reduced n_groups from {original_n} to {n_groups} due to "
                f"duplicate risk scores ({n_unique} unique values).",
                SuggestionWarning, stacklevel=2,
            )
            group_labels = pd.qcut(risk_scores, q=n_groups, labels=False, duplicates="drop")

        colors = get_risk_colors(n_groups)
        fig = make_subplots(
            rows=1, cols=n_groups,
            subplot_titles=[f"Risk Group Q{g+1}" for g in range(n_groups)],
            shared_yaxes=True,
        )

        for g in range(n_groups):
            mask = group_labels == g
            col = g + 1
            color = colors[g]

            group_pred = surv_probs[mask].mean(axis=0)
            fig.add_trace(go.Scatter(
                x=eval_times, y=group_pred,
                mode="lines", name=f"Predicted Q{g+1}",
                line=dict(color=color, width=2, dash="dash"),
                legendgroup=f"Q{g+1}",
                showlegend=(g == 0),
            ), row=1, col=col)

            y_group_time = y_test["time"][mask]
            y_group_event = y_test["event"][mask]
            kmf = KaplanMeierFitter()
            kmf.fit(y_group_time, y_group_event)
            sf = kmf.survival_function_

            if show_ci:
                ci = kmf.confidence_interval_survival_function_
                add_ci_ribbon(
                    fig, sf.index.values,
                    ci.iloc[:, 1].values, ci.iloc[:, 0].values,
                    color, row=1, col=col, legendgroup=f"Q{g+1}",
                )

            fig.add_trace(go.Scatter(
                x=sf.index.values, y=sf.iloc[:, 0].values,
                mode="lines", name=f"Observed Q{g+1}",
                line=dict(color=color, width=2, shape="hv"),
                legendgroup=f"Q{g+1}",
            ), row=1, col=col)

        d = MANUSCRIPT_DEFAULTS
        fig.update_layout(
            template="mlsurv_minimal",
            width=d["width_double"], height=d["height_medium"],
            title=dict(text=_importance_full_title(f"<b>{title}</b>", subtitle)),
            margin=dict(t=80 + (20 if subtitle else 0)),
        )
        time_label = f"Time ({time_unit})" if time_unit else "Time"
        for c in range(1, n_groups + 1):
            fig.update_xaxes(title_text=time_label, row=1, col=c)
            fig.update_yaxes(range=[0, 1.05], row=1, col=c)

        apply_size_overrides(fig, width, height)
        return fig

    fig.update_layout(
        xaxis_title=f"Time ({time_unit})" if time_unit else "Time",
        yaxis_title="Survival Probability",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(title=dict(text="")),
        hovermode="x unified",
    )
    apply_size_overrides(fig, width, height)
    return fig
