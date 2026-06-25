"""viz data-prep: metrics domain (state -> tidy frames for the metric renderers).

Peeled out of ``viz/_data.py`` — the Metrics section was ~43% of that
file. Behavior-preserving: turns ``LearnerState`` into the tidy frames the
``viz/_plot_impl_metric*`` renderers consume. Imports nothing from ``learner``;
the renderers and the learner facade import this downward as
``mlsurv.viz._data_metrics``. The sibling ``viz/_data.py`` keeps the other six
domains (Features, Survival, Time, Calibration, Tuning, Benchmarking).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..constants import (
    ALL_METRICS,
    LOWER_IS_BETTER,
    METRIC_REGISTRY,
    get_concordance_column,
    get_metric_display_name,
    get_metric_info,
    get_metric_short_name,
)
from .._warnings import SuggestionWarning
from ._resolve import resolve_eval_result, tidy_model_col
from ._metrics_helpers import _filter_iterations_by_metric
from ._palettes import get_model_display_name
from .metrics import _prepare_forest_data_from_bootstrap, plot_metric as _plot_metric


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _has_any_evaluation(models) -> bool:
    """Return True if any model in *models* has evaluation results."""
    if not models:
        return False
    return any(m.evaluation is not None for m in models.values())


def _has_any_bootstrap(models) -> bool:
    """Return True if any model in *models* has bootstrap data.

    Checks both standalone ``.bootstrap`` and bootstrap embedded inside
    ``.evaluation.bootstrap``.
    """
    if not models:
        return False
    return any(
        m.bootstrap is not None
        or (m.evaluation is not None and m.evaluation.bootstrap is not None)
        for m in models.values()
    )


# ------------------------------------------------------------------
# Source labels
# ------------------------------------------------------------------

SOURCE_LABELS = {
    "bootstrap": "Bootstrap",
    "test": "Test Set",
    "train": "Train Set (apparent)",
    "cv": "Cross-Validation",
}


# ------------------------------------------------------------------
# Source / uncertainty resolution
# ------------------------------------------------------------------

def resolve_metric_source(source, state, find_bootstrap_fn):
    """Resolve ``source="auto"`` to a concrete source string."""
    if source != "auto":
        return source
    boot = find_bootstrap_fn()
    if boot is not None and boot.iterations is not None and not boot.iterations.empty:
        return "bootstrap"
    if _has_any_evaluation(state.models):
        return "test"
    return "cv"


def resolve_metric_uncertainty(uncertainty, source):
    """Resolve ``uncertainty="auto"`` to ``(use_sd, ci_level)``."""
    if uncertainty == "auto":
        uncertainty = "sd" if source == "cv" else 0.95
    use_sd = uncertainty == "sd"
    ci_level = 0.95 if use_sd else float(uncertainty)
    return use_sd, ci_level


# ------------------------------------------------------------------
# Bootstrap data builders
# ------------------------------------------------------------------

def _validate_bootstrap_metric_rows(boot, metric, model_names):
    """Filter and validate bootstrap summary rows for a given metric.

    Returns the filtered metric_rows DataFrame.
    """
    summary = boot.summary
    metric_rows = summary[summary["metric"] == metric]
    if metric_rows.empty:
        raise ValueError(
            f"Metric '{metric}' not found in bootstrap summary. "
            f"Available: {summary['metric'].unique().tolist()}"
        )
    if model_names is not None:
        metric_rows = metric_rows[metric_rows["model_type"].isin(model_names)]
        if metric_rows.empty:
            available = summary["model_type"].unique().tolist()
            raise ValueError(
                f"None of the requested models {model_names} found "
                f"in bootstrap summary. Available: {available}"
            )
    return metric_rows


def _build_bootstrap_summary_df(metric_rows, metric, use_sd, ci_level, boot):
    """Build summary DataFrame from bootstrap metric rows (forest/bar style)."""
    if use_sd:
        points = metric_rows["point"].values
        stds = metric_rows["std"].values
        return pd.DataFrame({
            "model": metric_rows["model_type"],
            "point": points,
            "ci_lower": points - stds,
            "ci_upper": points + stds,
        })
    return _prepare_forest_data_from_bootstrap(
        metric_rows.copy(), metric=metric, ci_level=ci_level,
        iterations=boot.iterations,
    )


def _build_bootstrap_iteration_data(boot, metric, model_names):
    """Build iteration-level data and point-estimate extras (raincloud/boxplot style).

    Returns ``(df, extra_kwargs)``.
    """
    df = boot.iterations
    if model_names is not None:
        df = df[df["model_type"].isin(model_names)]
    # Filter to the requested metric so return_data=True matches the figure,
    # which filters via the same helper at render time. Without this,
    # return_data leaked all metrics and per-subpopulation rows.
    metric_col = "metric_name" if "metric_name" in df.columns else "metric"
    df = _filter_iterations_by_metric(df, metric, metric_col)
    extra_kwargs = dict(model_col="model_type")
    summary = boot.summary
    metric_rows = summary[summary["metric"] == metric]
    if not metric_rows.empty and "point" in metric_rows.columns:
        pe_map = {
            k: v
            for k, v in zip(metric_rows["model_type"], metric_rows["point"])
            if pd.notna(v)
        }
        if pe_map:
            extra_kwargs["point_estimates"] = pe_map
    return df, extra_kwargs


def build_metric_data_bootstrap(find_bootstrap_fn, metric, model_names,
                                use_sd, ci_level, style):
    """Build data for bootstrap source.

    Returns ``(df, extra_kwargs, default_style)``.
    """
    boot = find_bootstrap_fn()
    if boot is None or boot.iterations is None or boot.iterations.empty:
        raise ValueError("No bootstrap data. Run bootstrap() first.")

    if style in ("forest", "bar"):
        metric_rows = _validate_bootstrap_metric_rows(boot, metric, model_names)
        df = _build_bootstrap_summary_df(metric_rows, metric, use_sd, ci_level, boot)
        return df, {}, "raincloud"

    df, extra_kwargs = _build_bootstrap_iteration_data(boot, metric, model_names)
    return df, extra_kwargs, "raincloud"


# ------------------------------------------------------------------
# Test-set data builder
# ------------------------------------------------------------------

def build_metric_data_test(resolve_eval_result_fn, metric, model_names):
    """Build data for test-set source.

    Returns ``(df, extra_kwargs, default_style)``.
    """
    eval_result = resolve_eval_result_fn()
    if eval_result.concordance is None:
        raise ValueError("No test evaluation data. Run evaluate() first.")
    conc = eval_result.concordance
    info = get_metric_info(metric)
    if info is None:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Available: {sorted(METRIC_REGISTRY.keys())}"
        )
    metric_col = info.concordance_column
    if metric_col not in conc.columns:
        metric_col = metric
    if metric_col not in conc.columns:
        available = [c for c in conc.columns if c != 'model_type']
        raise ValueError(
            f"Metric '{metric}' not found in test results. "
            f"Available: {available}"
        )
    df = pd.DataFrame({
        "model": conc["model_type"],
        "point": conc[metric_col],
    })
    df["ci_lower"] = df["point"]
    df["ci_upper"] = df["point"]
    if model_names is not None:
        df = df[df["model"].isin(model_names)]
        if df.empty:
            available = conc["model_type"].tolist()
            raise ValueError(
                f"None of the requested models {model_names} found "
                f"in test results. Available: {available}"
            )
    return df, {}, "bar"


# ------------------------------------------------------------------
# Train-set (apparent / resubstitution) data builder
# ------------------------------------------------------------------

def _train_concordance_column(metric):
    """Map a metric key to its apparent train-set concordance column.

    Derived from the registered test column by swapping the ``_test`` suffix
    for ``_train`` (e.g. ``c_index_test`` -> ``c_index_train``). Returns
    ``None`` for metrics with no registered test column (the diagnostic
    calibration metrics have no train variant).
    """
    info = get_metric_info(metric)
    if info is None or not info.concordance_column:
        return None
    return info.concordance_column.replace("_test", "_train")


def build_metric_data_train(resolve_eval_result_fn, metric, model_names):
    """Build data for the apparent (resubstitution) train-set source.

    Point-only, exactly like the test source: ``ci_lower == ci_upper`` so the
    renderer draws no error bars. Reads the ``*_train`` concordance column for
    *metric*. Only the four scalar metrics in ``ALL_METRICS`` have train
    columns; diagnostic metrics raise.

    Returns ``(df, extra_kwargs, default_style)``.
    """
    eval_result = resolve_eval_result_fn()
    if eval_result.concordance is None:
        raise ValueError("No evaluation data. Run evaluate() first.")
    conc = eval_result.concordance
    if get_metric_info(metric) is None:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Available: {sorted(METRIC_REGISTRY.keys())}"
        )
    metric_col = _train_concordance_column(metric)
    if metric_col is None or metric_col not in conc.columns:
        available = [c for c in conc.columns if c.endswith("_train")]
        raise ValueError(
            f"Train-set metric '{metric}' not available. "
            f"source='train' supports {sorted(ALL_METRICS)}; "
            f"train columns present: {available}"
        )
    df = pd.DataFrame({
        "model": conc["model_type"],
        "point": conc[metric_col],
    })
    df["ci_lower"] = df["point"]
    df["ci_upper"] = df["point"]
    if model_names is not None:
        df = df[df["model"].isin(model_names)]
        if df.empty:
            available = conc["model_type"].tolist()
            raise ValueError(
                f"None of the requested models {model_names} found "
                f"in train results. Available: {available}"
            )
    return df, {}, "bar"


# ------------------------------------------------------------------
# CV data builders
# ------------------------------------------------------------------

def _resolve_cv_models_dict(state):
    """Resolve the CV models dict from state.models.

    Returns the models dict, or raises if no CV results are available.
    """
    if state.models and any(
        getattr(r, "cv_metric_scores", None) is not None
        for r in state.models.values()
    ):
        return state.models
    raise ValueError(
        "No CV results available. Run train() or run() first."
    )


def build_metric_data_cv(state, metric, model_names, use_sd, ci_level,
                         iteration=False):
    """Build data for CV source.

    Parameters
    ----------
    state : LearnerState
        Learner state holding fitted models and CV results.
    metric : str
        Metric key to aggregate (e.g. ``"harrell"``).
    model_names : list of str
        Models to include.
    use_sd : bool
        If True, report ±SD; otherwise report a confidence interval.
    ci_level : float
        Confidence level (fraction) for the interval when ``use_sd`` is False.
    iteration : bool
        If True, return fold-level rows (for raincloud/boxplot).
        If False, return summary rows (for forest/bar).

    Returns
    -------
    tuple
        ``(df, extra_kwargs, default_style)``.

    Raises
    ------
    ValueError
        If none of the requested ``model_names`` are present in the CV
        results.
    """
    models_dict = _resolve_cv_models_dict(state)

    if iteration:
        df = build_cv_fold_iterations(models_dict, metric, model_names)
        return df, {"model_col": "model_type"}, "raincloud"

    cv_metrics = build_cv_summary_from_models(
        models_dict, metric, model_names,
    )
    points = cv_metrics["cv_metric_mean"].values
    stds = cv_metrics["cv_metric_std"].values
    if use_sd:
        half_width = stds
    else:
        from scipy.stats import norm
        half_width = norm.ppf(1 - (1 - ci_level) / 2) * stds
    df = pd.DataFrame({
        "model": cv_metrics["model"],
        "point": points,
        "ci_lower": points - half_width,
        "ci_upper": points + half_width,
    })
    if model_names is not None:
        df = df[df["model"].isin(model_names)]
        if df.empty:
            available = cv_metrics["model"].tolist()
            raise ValueError(
                f"None of the requested models {model_names} found "
                f"in CV results. Available: {available}"
            )
    return df, {}, "forest"


def build_cv_fold_iterations(models_dict, metric, model_names):
    """Build fold-level iteration DataFrame from CV results.

    Returns DataFrame with columns ``[model_type, fold, metric_name, value]``.
    """
    rows = []
    for name, result in models_dict.items():
        if model_names is not None and name not in model_names:
            continue
        if getattr(result, "cv_metric_scores", None) is None:
            warnings.warn(
                f"Model '{name}' has no CV data and was excluded "
                f"from the plot.",
                SuggestionWarning,
                stacklevel=4,
            )
            continue
        display = get_model_display_name(name)
        cv_df = result.cv_results_df()
        if metric in cv_df.columns:
            scores = cv_df[metric].values
        elif metric == result.scoring_metric:
            scores = result.cv_metric_scores
        else:
            warnings.warn(
                f"Model '{name}' has no CV fold data for metric "
                f"'{metric}' and was excluded from the plot.",
                SuggestionWarning,
                stacklevel=4,
            )
            continue
        for fold_idx, score in enumerate(scores):
            rows.append({
                'model_type': display,
                'fold': fold_idx,
                'metric_name': metric,
                'value': float(score),
            })
    if not rows:
        raise ValueError(
            f"No CV fold data found for metric '{metric}'. "
            f"Available metrics depend on CV configuration."
        )
    return pd.DataFrame(rows)


def build_cv_summary_from_models(models_dict, metric, model_names):
    """Build CV summary DataFrame from individual ModelResult objects.
    """
    rows = []
    for name, result in models_dict.items():
        if model_names is not None and name not in model_names:
            continue
        # Try cv_metrics dict first (has per-metric mean/std)
        cv_m = getattr(result, "cv_metrics", None) or {}
        if metric in cv_m:
            entry = cv_m[metric]
            mean_val = entry.get("mean", entry.get("cv_metric_mean"))
            std_val = entry.get("std", entry.get("cv_metric_std"))
        elif metric == getattr(result, "scoring_metric", None):
            mean_val = getattr(result, "cv_metric_mean", None)
            std_val = getattr(result, "cv_metric_std", None)
        else:
            warnings.warn(
                f"Model '{name}' has no CV summary data for metric "
                f"'{metric}' and was excluded from the plot.",
                SuggestionWarning,
                stacklevel=4,
            )
            continue
        if mean_val is not None:
            rows.append({
                "model": name,
                "cv_metric_mean": float(mean_val),
                "cv_metric_std": float(std_val) if std_val is not None else float("nan"),
            })
        else:
            warnings.warn(
                f"Model '{name}' has metric '{metric}' in cv_metrics "
                f"but mean value is None; excluded from the plot.",
                SuggestionWarning,
                stacklevel=4,
            )
    if not rows:
        raise ValueError(
            f"No CV summary data found for metric '{metric}'."
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Title builder
# ------------------------------------------------------------------

def _bootstrap_n_iter(find_bootstrap_fn):
    """Return the number of bootstrap iterations, or 0 when unavailable.

    Null-guards a missing/empty bootstrap so subtitle/title builders never crash
    on a no-bootstrap cohort.
    """
    boot = find_bootstrap_fn()
    if boot is not None and boot.iterations is not None and not boot.iterations.empty:
        return int(boot.iterations["iteration"].nunique())
    return 0


def _bootstrap_subtitle(find_bootstrap_fn, use_sd, ci_pct, style):
    """Build subtitle for bootstrap source."""
    n_iter = _bootstrap_n_iter(find_bootstrap_fn)
    if style in ("forest", "bar"):
        if use_sd:
            return f"Test-set estimate ± bootstrap SD (n={n_iter})"
        return f"Test-set estimate with bootstrap {ci_pct}% CI (n={n_iter})"
    return f"Test-set bootstrap (n={n_iter})"


def _cv_subtitle(state, u_symbol, style):
    """Build subtitle for CV source."""
    k = state.cv_folds
    r = state.cv_repeats
    cv_desc = f"{r}× {k}-fold" if r > 1 else f"{k}-fold"
    if style == "forest":
        return f"CV mean {u_symbol} ({cv_desc})"
    return f"Cross-validation ({cv_desc})"


def _metric_subtitle_span(subtitle):
    """Wrap a subtitle in the 12px normal-weight span used by metric titles."""
    return (
        f"<br><span style='font-size:12px;font-weight:normal'>"
        f"{subtitle}</span>"
    )


def _compose_metric_title(header, subtitle):
    """Combine a header line with an optional band/source subtitle.

    A caller-supplied title replaces only the *header*; the band subtitle still
    appears below it, so a custom title never strips the description of what the
    error bars represent. Returns ``None`` when there is no header (unknown
    source with no user title), leaving the renderer's generic fallback.
    """
    if header is None:
        return None
    if subtitle:
        return header + _metric_subtitle_span(subtitle)
    return header


def _resolve_metric_title(user_title, header, subtitle):
    """Compose the final title, letting a user title replace only the header.

    ``user_title is None`` keeps the default header; any other value (including
    ``""``) overrides it, while the band/source ``subtitle`` is always kept.
    """
    if user_title is not None:
        header = user_title
    return _compose_metric_title(header, subtitle)


def _multi_metric_title_parts(state, find_bootstrap_fn, source,
                              effective_style, is_iteration_style,
                              use_sd, ci_level, uncertainty_label):
    """Return ``(header, subtitle)`` for a multi-metric panel title.

    Mirrors single-metric ``_metric_title_parts`` wording per source so the
    multi-metric forest/bar subtitle names the band actually drawn (percentile
    CI vs SD) and the test/train notes flag the point estimate. Reuses
    ``_bootstrap_subtitle`` for exact single-metric parity. ``header`` is
    ``None`` for an unknown source (generic fallback); ``subtitle`` is ``None``
    for the iteration styles, which already show the distribution itself.
    """
    if source == "cv":
        k = state.cv_folds
        r = state.cv_repeats
        cv_desc = f"{r}× {k}-fold" if r > 1 else f"{k}-fold"
        if is_iteration_style:
            # raincloud/boxplot show the per-fold distribution itself.
            return f"CV Fold Distribution ({cv_desc})", None
        return ("Cross-Validation Performance",
                f"CV mean ± {uncertainty_label} ({cv_desc})")
    if source == "bootstrap":
        if is_iteration_style:
            # raincloud/boxplot show the per-iteration distribution itself.
            n_iter = _bootstrap_n_iter(find_bootstrap_fn)
            return f"Bootstrap Distribution (n={n_iter})", None
        ci_pct = f"{ci_level * 100:g}"
        subtitle = _bootstrap_subtitle(
            find_bootstrap_fn, use_sd, ci_pct, effective_style,
        )
        return "Bootstrap Performance", subtitle
    if source == "test":
        return "Test-Set Performance", "Test-set estimate"
    if source == "train":
        return "Train-Set Performance", "Apparent (train-set) estimate"
    return None, None


def _multi_metric_default_title(state, find_bootstrap_fn, source,
                                effective_style, is_iteration_style,
                                use_sd, ci_level, uncertainty_label):
    """Default multi-metric panel title: descriptive header + band subtitle.

    Returns ``None`` for an unknown source, leaving
    ``_dispatch_multi_panel``'s generic fallback.
    """
    header, subtitle = _multi_metric_title_parts(
        state, find_bootstrap_fn, source, effective_style,
        is_iteration_style, use_sd, ci_level, uncertainty_label,
    )
    return _compose_metric_title(header, subtitle)


def _metric_title_parts(state, find_bootstrap_fn, metric, source, use_sd,
                        ci_level, style):
    """Return ``(header, subtitle)`` for a single-metric plot_metric title."""
    display = get_metric_display_name(metric)
    ci_pct = f"{ci_level * 100:g}"
    u_symbol = "± SD" if use_sd else f"± {ci_pct}% CI"

    if source == "bootstrap":
        subtitle = _bootstrap_subtitle(find_bootstrap_fn, use_sd, ci_pct, style)
    elif source == "test":
        subtitle = "Test-set estimate"
    elif source == "train":
        subtitle = "Apparent (train-set) estimate"
    elif source == "cv":
        subtitle = _cv_subtitle(state, u_symbol, style)
    else:
        subtitle = None

    return display, subtitle


def find_bootstrap(state):
    """Find bootstrap result from state models."""
    if not _has_any_bootstrap(state.models):
        return None
    from ..results.containers import BootstrapResult
    return BootstrapResult.combine(state.models)


# ------------------------------------------------------------------
# Multi-metric helpers
# ------------------------------------------------------------------

def resolve_multi_metric_source(source, state, find_bootstrap_fn):
    """Resolve ``source="auto"`` to a concrete source string for multi-metric."""
    if source != "auto":
        return source
    boot = find_bootstrap_fn()
    if boot is not None and boot.summary is not None and not boot.summary.empty:
        return "bootstrap"
    if _has_any_evaluation(state.models):
        return "test"
    return "cv"


def resolve_multi_metric_style(style, source):
    """Map auto style to effective style and validate compatibility.

    Returns ``(effective_style, is_iteration_style)``.
    """
    if style != "auto":
        effective_style = style
    elif source in ("bootstrap", "cv"):
        effective_style = "raincloud"
    else:
        effective_style = "forest"
    is_iteration_style = effective_style in ("raincloud", "boxplot")

    if is_iteration_style and source not in ("bootstrap", "cv"):
        raise ValueError(
            f"style={effective_style!r} requires source='bootstrap' or "
            f"source='cv' (per-iteration data). Got source={source!r}."
        )
    return effective_style, is_iteration_style


def dispatch_multi_metric_data(state, find_bootstrap_fn, metrics, models,
                               is_iteration_style, source, ci_level=0.95):
    """Dispatch to the appropriate multi-metric data builder.

    Returns ``(df, extra_kwargs)``.
    """
    extra_kwargs = {}

    if is_iteration_style and source == "bootstrap":
        df = build_multi_metric_bootstrap_iterations(find_bootstrap_fn, metrics, models)
        pe = build_multi_metric_point_estimates(find_bootstrap_fn, metrics)
        if pe:
            extra_kwargs["point_estimates"] = pe
    elif is_iteration_style and source == "cv":
        df = build_multi_metric_cv_iterations(state, metrics, models)
    elif source == "bootstrap":
        df = build_multi_metric_bootstrap(
            find_bootstrap_fn, metrics, models, ci_level=ci_level,
        )
    elif source == "test":
        df = build_multi_metric_test(state, metrics, models)
    elif source == "train":
        df = build_multi_metric_train(state, metrics, models)
    elif source == "cv":
        df = build_multi_metric_cv_summary(state, metrics, models)
    else:
        raise ValueError(
            f"Unknown source '{source}'. Must be 'auto', 'bootstrap', "
            f"'test', 'train', or 'cv' for multi-metric plots (metric=[...])."
        )
    return df, extra_kwargs


def validate_and_filter_bootstrap_df(source_df, metrics, model_names,
                                     metric_col, source_label):
    """Validate metrics exist in *source_df* and filter by metrics/models."""
    available_metrics = source_df[metric_col].unique().tolist()
    for m in metrics:
        if m not in available_metrics:
            raise ValueError(
                f"Metric '{m}' not found in bootstrap {source_label}. "
                f"Available: {available_metrics}"
            )

    df = source_df[source_df[metric_col].isin(metrics)].copy()
    if model_names is not None:
        df = df[df["model_type"].isin(model_names)]
        if df.empty:
            available = source_df["model_type"].unique().tolist()
            raise ValueError(
                f"None of the requested models {model_names} found "
                f"in bootstrap {source_label}. Available: {available}"
            )
    return df


def build_multi_metric_bootstrap(find_bootstrap_fn, metrics, model_names,
                                 *, ci_level=0.95):
    """Build wide-format DataFrame from bootstrap summary for multi-metric forest.

    Recomputes CI bounds from raw iterations at the requested *ci_level* so
    the forest plot's bars and hover label always agree, regardless of the
    level the bootstrap was originally computed at. Falls back to the stored
    ``boot.summary['ci_lower'/'ci_upper']`` only when *ci_level* is the
    bootstrap-time default (0.95) AND iterations are unavailable.
    """
    import warnings as _warnings
    from mlsurv._warnings import SuggestionWarning

    boot = find_bootstrap_fn()
    if boot is None or boot.summary is None or boot.summary.empty:
        raise ValueError("No bootstrap data. Run bootstrap() first.")

    rows = validate_and_filter_bootstrap_df(
        boot.summary, metrics, model_names,
        metric_col="metric", source_label="summary",
    )

    # Recompute ci_lower/ci_upper from raw iterations at the requested level.
    iterations = boot.iterations
    if iterations is not None and not iterations.empty:
        alpha = 1 - ci_level
        lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)
        iter_metric_col = (
            "metric_name" if "metric_name" in iterations.columns else "metric"
        )
        recomputed_lo = {}
        recomputed_hi = {}
        missing_pairs = []
        for mtype in rows["model_type"].unique():
            for m in metrics:
                vals = iterations.loc[
                    (iterations["model_type"] == mtype)
                    & (iterations[iter_metric_col] == m),
                    "value",
                ].dropna().astype(float).values
                if len(vals) > 0:
                    recomputed_lo[(mtype, m)] = float(np.percentile(vals, lo_pct))
                    recomputed_hi[(mtype, m)] = float(np.percentile(vals, hi_pct))
                else:
                    recomputed_lo[(mtype, m)] = np.nan
                    recomputed_hi[(mtype, m)] = np.nan
                    missing_pairs.append((mtype, m))
        rows = rows.copy()
        rows["ci_lower"] = [
            recomputed_lo[(r.model_type, r.metric)] for r in rows.itertuples()
        ]
        rows["ci_upper"] = [
            recomputed_hi[(r.model_type, r.metric)] for r in rows.itertuples()
        ]
        if missing_pairs:
            pretty = ", ".join(f"{mt}/{m}" for mt, m in missing_pairs)
            _warnings.warn(
                f"Bootstrap iterations missing or all-NaN for "
                f"{len(missing_pairs)} (model, metric) pair(s): {pretty}. "
                f"Forest plot will render these points without error bars.",
                SuggestionWarning,
                stacklevel=2,
            )

    # Pivot to wide format: one column per metric, plus {metric}_std,
    # {metric}_ci_lower, {metric}_ci_upper. CI bounds match the renderer's
    # ci_level so bars and hover label agree.
    point_wide = rows.pivot(
        index="model_type", columns="metric", values="point",
    )
    std_wide = rows.pivot(
        index="model_type", columns="metric", values="std",
    )
    std_wide.columns = [f"{c}_std" for c in std_wide.columns]
    ci_lo_wide = rows.pivot(
        index="model_type", columns="metric", values="ci_lower",
    )
    ci_lo_wide.columns = [f"{c}_ci_lower" for c in ci_lo_wide.columns]
    ci_hi_wide = rows.pivot(
        index="model_type", columns="metric", values="ci_upper",
    )
    ci_hi_wide.columns = [f"{c}_ci_upper" for c in ci_hi_wide.columns]
    df = pd.concat(
        [point_wide, std_wide, ci_lo_wide, ci_hi_wide], axis=1,
    ).reset_index()

    # Reorder columns: model_type, then metric/std/ci pairs in request order
    ordered_cols = ["model_type"]
    for m in metrics:
        ordered_cols.extend([m, f"{m}_std", f"{m}_ci_lower", f"{m}_ci_upper"])
    df = df[ordered_cols]

    return df


def build_multi_metric_bootstrap_iterations(find_bootstrap_fn, metrics,
                                            model_names):
    """Build long-format iteration DataFrame for multi-metric iteration plots."""
    boot = find_bootstrap_fn()
    if boot is None or boot.iterations is None or boot.iterations.empty:
        raise ValueError("No bootstrap iteration data. Run bootstrap() first.")

    return validate_and_filter_bootstrap_df(
        boot.iterations, metrics, model_names,
        metric_col="metric_name", source_label="iterations",
    )


def build_multi_metric_point_estimates(find_bootstrap_fn, metrics):
    """Build nested point_estimates dict for multi-panel plots.

    Returns ``{metric: {model_type: value}}`` extracted from bootstrap summary.
    """
    boot = find_bootstrap_fn()
    if boot is None or boot.summary is None or boot.summary.empty:
        return {}
    summary = boot.summary
    if "point" not in summary.columns:
        return {}
    result = {}
    for metric in metrics:
        rows = summary[summary["metric"] == metric]
        if rows.empty:
            continue
        pe_map = {
            k: v
            for k, v in zip(rows["model_type"], rows["point"])
            if pd.notna(v)
        }
        if pe_map:
            result[metric] = pe_map
    return result


def build_multi_metric_cv_iterations(state, metrics, model_names):
    """Build long-format fold-level DataFrame for multi-metric CV plots."""
    models_dict = _resolve_cv_models_dict(state)

    rows = []
    for metric in metrics:
        fold_df = build_cv_fold_iterations(models_dict, metric, model_names)
        rows.append(fold_df)

    return pd.concat(rows, ignore_index=True)


def build_multi_metric_test(state, metrics, model_names):
    """Build wide-format DataFrame from test evaluation for multi-metric forest."""
    eval_result = resolve_eval_result(state)
    if eval_result.concordance is None:
        raise ValueError("No test evaluation data. Run evaluate() first.")

    conc = eval_result.concordance
    col_map = {}
    for m in metrics:
        info = get_metric_info(m)
        if info is None:
            raise ValueError(
                f"Unknown metric '{m}'. "
                f"Available: {sorted(METRIC_REGISTRY.keys())}"
            )
        col = info.concordance_column
        if col not in conc.columns:
            available = [c for c in conc.columns if c != "model_type"]
            raise ValueError(
                f"Metric '{m}' (column '{col}') not found in test results. "
                f"Available: {available}"
            )
        col_map[m] = col

    df = conc[["model_type"]].copy()
    for m in metrics:
        df[m] = conc[col_map[m]].values

    if model_names is not None:
        df = df[df["model_type"].isin(model_names)].reset_index(drop=True)
        if df.empty:
            available = conc["model_type"].unique().tolist()
            raise ValueError(
                f"None of the requested models {model_names} found "
                f"in test evaluation. Available: {available}"
            )

    return df


def build_multi_metric_train(state, metrics, model_names):
    """Build wide-format DataFrame from apparent train-set metrics.

    Point-only (no spread columns), exactly like :func:`build_multi_metric_test`,
    so the forest renders bare points with no error bars. Reads each metric's
    ``*_train`` concordance column; only the four scalar metrics in
    ``ALL_METRICS`` have train columns.
    """
    eval_result = resolve_eval_result(state)
    if eval_result.concordance is None:
        raise ValueError("No evaluation data. Run evaluate() first.")

    conc = eval_result.concordance
    col_map = {}
    for m in metrics:
        if get_metric_info(m) is None:
            raise ValueError(
                f"Unknown metric '{m}'. "
                f"Available: {sorted(METRIC_REGISTRY.keys())}"
            )
        col = _train_concordance_column(m)
        if col is None or col not in conc.columns:
            available = [c for c in conc.columns if c.endswith("_train")]
            raise ValueError(
                f"Train-set metric '{m}' not available. "
                f"source='train' supports {sorted(ALL_METRICS)}; "
                f"train columns present: {available}"
            )
        col_map[m] = col

    df = conc[["model_type"]].copy()
    for m in metrics:
        df[m] = conc[col_map[m]].values

    if model_names is not None:
        df = df[df["model_type"].isin(model_names)].reset_index(drop=True)
        if df.empty:
            available = conc["model_type"].unique().tolist()
            raise ValueError(
                f"None of the requested models {model_names} found "
                f"in train evaluation. Available: {available}"
            )

    return df


def build_multi_metric_cv_summary(state, metrics, model_names):
    """Build wide-format DataFrame from CV summaries for multi-metric forest/bar.

    Mirrors :func:`build_multi_metric_test` but sources each metric's point and
    spread from cross-validation (``ModelResult.cv_metrics``) rather than the
    test set. Returns one row per model with columns
    ``[model_type, m1, m1_std, m2, m2_std, ...]`` in *metrics* order. The
    forest/bar renderers turn ``{m}_std`` into a band: ``point +/- std`` by
    default (± SD, mirroring the single-metric CV forest) or ``point +/- z*std``
    at the requested ``ci_level`` when the caller passes an explicit float.
    """
    models_dict = _resolve_cv_models_dict(state)

    merged = None
    for m in metrics:
        # Reuse the validated per-metric CV summary builder (handles the
        # cv_metrics lookup, scoring-metric fallback, per-model missing-data
        # warning, and empty -> ValueError guard).
        summary = build_cv_summary_from_models(models_dict, m, model_names)
        summary = summary.rename(columns={
            "model": "model_type",
            "cv_metric_mean": m,
            "cv_metric_std": f"{m}_std",
        })
        merged = summary if merged is None else merged.merge(
            summary, on="model_type", how="outer",
        )

    # Deterministic model order, following the CV results.
    order = [n for n in models_dict if n in set(merged["model_type"])]
    merged["model_type"] = pd.Categorical(
        merged["model_type"], categories=order, ordered=True,
    )
    merged = merged.sort_values("model_type").reset_index(drop=True)
    merged["model_type"] = merged["model_type"].astype(str)

    ordered_cols = ["model_type"]
    for m in metrics:
        ordered_cols.extend([m, f"{m}_std"])
    return merged[ordered_cols]


def build_multi_metric_plot(state, find_bootstrap_fn, *, metrics, models,
                            style, source, uncertainty, return_data,
                            orientation, **kwargs):
    """Dispatch multi-metric forest plot construction."""
    # Validate each metric name
    for m in metrics:
        if get_metric_info(m) is None:
            raise ValueError(
                f"Unknown metric '{m}'. "
                f"Available: {sorted(METRIC_REGISTRY.keys())}"
            )

    # Style guard
    _VALID_MULTI_STYLES = {"auto", "forest", "bar", "raincloud", "boxplot"}
    if style not in _VALID_MULTI_STYLES:
        raise ValueError(
            f"style={style!r} is not supported with multi-metric plots "
            f"(metric=[...]). Must be one of: {sorted(_VALID_MULTI_STYLES)}"
        )

    source = resolve_multi_metric_source(source, state, find_bootstrap_fn)
    effective_style, is_iteration_style = resolve_multi_metric_style(
        style, source,
    )

    # Resolve uncertainty -> (use_sd, ci_level). Resolved BEFORE dispatch so the
    # bootstrap builder can recompute CI bounds at the requested level —
    # guarantees forest bars and hover label agree.
    #
    # CV summary plots default to ± SD, mirroring the single-metric CV forest
    # (resolve_metric_uncertainty); bootstrap defaults to a percentile CI. Both
    # cv and bootstrap carry a per-model std column, so 'sd' is a valid override
    # for either (point ± std). 'test' yields a single point estimate with no
    # spread, so 'sd' stays unsupported there. An explicit float forces a CI
    # band for cv/bootstrap.
    use_sd = (
        (source == "cv" and uncertainty in ("auto", "sd"))
        or (source == "bootstrap" and uncertainty == "sd")
    )
    if uncertainty == "sd" and source == "test" and not is_iteration_style:
        raise ValueError(
            "uncertainty='sd' is not supported with source='test' for "
            "multi-metric plots (metric=[...]) -- the test set yields a single "
            "point estimate with no spread. Pass a float (e.g. 0.95) or "
            "use 'auto'."
        )
    ci_level = 0.95 if uncertainty in ("auto", "sd") else float(uncertainty)
    uncertainty_label = "SD" if use_sd else f"{ci_level * 100:g}% CI"

    df, extra_kwargs = dispatch_multi_metric_data(
        state, find_bootstrap_fn, metrics, models, is_iteration_style, source,
        ci_level=ci_level,
    )

    if return_data:
        return tidy_model_col(df)

    # Build source-aware title for multi-metric panels: descriptive header plus a
    # subtitle naming the band actually drawn (percentile CI vs SD), or a
    # point-estimate note for test/train. A caller-supplied title replaces only
    # the header so the band subtitle is never lost.
    header, subtitle = _multi_metric_title_parts(
        state, find_bootstrap_fn, source, effective_style,
        is_iteration_style, use_sd, ci_level, uncertainty_label,
    )
    title = _resolve_metric_title(kwargs.pop("title", None), header, subtitle)

    # The internal uncertainty_label is authoritative; drop any caller-supplied
    # one from **kwargs so it can't collide with the explicit argument below.
    kwargs.pop("uncertainty_label", None)
    return _plot_metric(
        df, model_col="model_type", metric=list(metrics),
        style=effective_style, title=title,
        ci_level=ci_level, orientation=orientation,
        uncertainty_label=uncertainty_label,
        **extra_kwargs, **kwargs,
    )


# ------------------------------------------------------------------
# Heatmap data builder (for plot_metric_heatmap)
# ------------------------------------------------------------------

def _get_heatmap_metric_value(concordance, col):
    """Extract a single metric value from a concordance DataFrame row."""
    if col not in concordance.columns:
        return None
    val = concordance[col].iloc[0]
    if pd.isna(val):
        return None
    return float(val)


def build_heatmap_dataframe(state, model_names=None):
    """Build wide-format heatmap DataFrame from evaluated models.

    For a model with an attached calibrator (``train(calibrate=...)``) the
    deployed (calibrated) metrics already live in each model's ``evaluation``
    because evaluate() applies the calibrator inline, so the heatmap needs no
    calibrate/raw switch.

    Parameters
    ----------
    state : LearnerState
    model_names : list[str] or None
        If given, only include these models.

    Returns
    -------
    pd.DataFrame or None
        Wide-format DataFrame ``[model_type, <metric_1>, ...]``,
        or ``None`` if fewer than 2 models have evaluation results.
    """
    models = state.models or {}
    if model_names is not None:
        models = {k: v for k, v in models.items() if k in model_names}

    rows = []
    for name, result in models.items():
        if result.evaluation is None:
            continue
        concordance = getattr(result.evaluation, 'concordance', None)
        if concordance is None:
            continue
        row = {'model_type': name}
        for metric_name in ALL_METRICS:
            col = get_concordance_column(metric_name)
            val = _get_heatmap_metric_value(concordance, col)
            if val is not None:
                row[metric_name] = val
        rows.append(row)

    if len(rows) < 2:
        return None
    return pd.DataFrame(rows)


def build_validation_heatmap_data(validation_results, model_names=None):
    """Assemble cross-cohort heatmap inputs from external-validation results.

    Shared by ``learner.plot_validation_heatmap`` (the public method) and the
    report's validation section, so both build the faceted metric heatmap from
    identical data. Each cohort contributes one wide-format frame of raw metric
    values; the tidy frame is the long view used for programmatic access.

    Parameters
    ----------
    validation_results : Mapping[str, ValidationResult]
        Cohort name -> ``ValidationResult``. Each result carries a
        ``concordance`` table (one row per model). Cohorts whose
        ``concordance`` is ``None`` are skipped.
    model_names : list[str] or None
        If given, keep only these models (by ``model_type``) in both the wide
        and tidy frames, so a filtered plot and its ``return_data`` agree.
        ``None`` keeps every model.

    Returns
    -------
    cohort_dfs : dict[str, pandas.DataFrame]
        Per-cohort wide-format frames ``[model_type, <metric_key>...]`` of raw
        float metric values. Empty dict when no cohort has metrics.
    metric_cols : list[str]
        Numeric metric columns present in the cohort frames (``[]`` when none).
    heatmap_recal : set[str]
        Retained for return-shape stability; always an empty set (the
        deployed calibrator is applied inline, so cohort rows carry no
        separate "(recal)" marking).
    tidy_df : pandas.DataFrame or None
        Long/tidy frame (``Cohort``, ``Model``, then per-metric string columns
        with ``*``-suffixed lower-is-better headers) for programmatic access;
        ``None`` when no rows were produced.
    """
    cohort_dfs = {}
    table_data = []

    for name, val_result in validation_results.items():
        if val_result.concordance is None:
            continue

        concordance_df = val_result.concordance
        wide_rows = []
        for _, row in concordance_df.iterrows():
            model_name = row['model_type']
            if model_names is not None and model_name not in model_names:
                continue
            display_name = get_model_display_name(model_name)

            # Wide-format row for heatmap (metric keys as columns)
            wide_row = {'model_type': model_name}
            # Tidy row for programmatic access
            tidy_row = {'Cohort': name, 'Model': display_name}

            for metric_key in ALL_METRICS:
                col = get_concordance_column(metric_key)
                if col in row.index and not pd.isna(row[col]):
                    wide_row[metric_key] = float(row[col])
                    col_header = get_metric_short_name(
                        metric_key, with_asterisk=False,
                    )
                    if metric_key in LOWER_IS_BETTER:
                        col_header += " *"
                    tidy_row[col_header] = f"{row[col]:.3f}"

            wide_rows.append(wide_row)
            table_data.append(tidy_row)

        if wide_rows:
            cohort_dfs[name] = pd.DataFrame(wide_rows)

    tidy_df = pd.DataFrame(table_data) if table_data else None

    if not cohort_dfs:
        return cohort_dfs, [], set(), tidy_df

    # Determine metric columns
    first_df = next(iter(cohort_dfs.values()))
    metric_cols = [
        c for c in first_df.columns
        if c != 'model_type' and pd.api.types.is_numeric_dtype(first_df[c])
    ]

    # The deployed calibrator is applied inline on every cohort, so there is
    # no separate "(recal)" marking; kept in the return shape for stability.
    heatmap_recal: set = set()

    return cohort_dfs, metric_cols, heatmap_recal, tidy_df


# ------------------------------------------------------------------
# Phase 2: CC-reduction helpers (metrics)
# ------------------------------------------------------------------

def dispatch_single_metric_data(
    state, find_bootstrap_fn, resolve_eval_fn,
    source, metric, model_names, use_sd, ci_level, style,
):
    """Build ``(df, extra_kwargs, default_style)`` by source, with guards.

    Parameters
    ----------
    state : LearnerState
        Learner state holding fitted models and CV results.
    find_bootstrap_fn : callable
        Returns the bootstrap result for the requested models.
    resolve_eval_fn : callable
        Returns the test-set evaluation for the requested models.
    source : str
        Data source: ``"bootstrap"``, ``"test"``, ``"train"``, or ``"cv"``.
    metric : str
        Metric key to plot.
    model_names : list of str
        Models to include.
    use_sd : bool
        If True, report ±SD; otherwise report a confidence interval.
    ci_level : float
        Confidence level (fraction) for the interval when ``use_sd`` is False.
    style : str
        Requested plot style (``"auto"``, ``"forest"``, ``"bar"``,
        ``"raincloud"``, ``"boxplot"``).

    Returns
    -------
    df : pd.DataFrame
        Data for plotting; the column set depends on the resolved style.
    extra_kwargs : dict
        Extra keyword arguments for the downstream renderer (e.g.
        ``model_col``, ``point_estimates``); may be empty.
    default_style : str
        The resolved plot style (``"forest"``, ``"bar"``, ``"raincloud"``,
        or ``"boxplot"``) after applying ``style`` and source defaults.

    Raises
    ------
    ValueError
        If *source* is unrecognised, or *style* requires iterations but
        *source* is ``"test"`` or ``"train"`` (both point-only).
    """
    if source == "bootstrap":
        df, extra_kwargs, default_style = build_metric_data_bootstrap(
            find_bootstrap_fn, metric, model_names, use_sd, ci_level, style,
        )
    elif source == "test":
        df, extra_kwargs, default_style = build_metric_data_test(
            resolve_eval_fn, metric, model_names,
        )
    elif source == "train":
        df, extra_kwargs, default_style = build_metric_data_train(
            resolve_eval_fn, metric, model_names,
        )
    elif source == "cv":
        iteration = style in ("raincloud", "boxplot")
        df, extra_kwargs, default_style = build_metric_data_cv(
            state, metric, model_names, use_sd, ci_level,
            iteration=iteration,
        )
    else:
        raise ValueError(
            f"Unknown source '{source}'. "
            "Must be 'auto', 'bootstrap', 'test', 'train', or 'cv'."
        )

    if style in ("raincloud", "boxplot") and source in ("test", "train"):
        raise ValueError(
            f"style='{style}' requires iteration data, but source='{source}' "
            f"provides only point estimates. Use style='bar' or style='forest' "
            f"for {source} data, or source='bootstrap'/'cv' for distribution plots."
        )

    default_style = style if style != "auto" else default_style
    return df, extra_kwargs, default_style
