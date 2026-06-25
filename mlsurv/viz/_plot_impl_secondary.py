"""Secondary plot implementations — main-only dispatch.

Hosts ``_plot_X_impl`` functions for the 15 specialized plot methods
that have not yet been migrated to the full 3-branch
(main/validation/subpopulation) dispatch in ``_plot_impl.py``.

Each function takes a :class:`~mlsurv.viz._context.PlotContext` built by
the thin wrapper in ``learner/_visualization.py``.  Only the ``"main"``
source is supported; validation and subpopulation branches can be added
incrementally as needed.

Migrated here as part of the per-family plot split.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd

from mlsurv._shared import tidy_model_col as _tidy_model_col
from mlsurv._warnings import InternalWarning
from mlsurv.viz._context import PlotContext
from mlsurv.viz._data import (
    _DEFAULT_COX_MODEL,
    _hr_valid_models,
    build_coefficient_title,
    build_combined_optuna_figure,
    build_incremental_figure,
    build_tuning_data,
    collect_all_coef_frames,
    collect_tuned_models,
    compute_overlay_return_data,
    find_cox_models,
    find_first_surv_model,
    get_coef_series,
    normalize_coef_models_param,
    prepare_univariate_hr_df,
)
from mlsurv.viz._data_metrics import build_heatmap_dataframe
from mlsurv.viz._resolve import resolve_eval_result
from mlsurv.viz._plot_impl import (
    _inject_time_unit,
    _resolve_models_arg,
)
from mlsurv.viz._plot_impl_shared import _calibrator_for

if TYPE_CHECKING:
    from mlsurv.results.containers import ModelResult

# Shown when no trained model has valid hazard ratios for an HR plot.
NO_HR_MODEL_MSG = (
    "No model with valid hazard ratios is available. Multivariate hazard "
    "ratios require an unpenalized Cox PH model (e.g. "
    "learner.run(models=['coxph'])). CoxNet/FSSVM coefficients are not "
    "hazard ratios; use plot_coefficients() to inspect them."
)

# ======================================================================
# Tier 1 — Simple state-access (no model resolution)
# ======================================================================


def _plot_missing_impl(
    ctx: PlotContext,
    *,
    return_data: bool = False,
    **kwargs,
):
    """Shared implementation for ``plot_missing``."""
    state = ctx.state_ref
    X = state.X_train
    if X is None:
        raise ValueError("No training data. Run setup() first.")

    if return_data:
        return X.isnull()

    from mlsurv.viz.missing import plot_missing as _plot

    return _plot(X, **kwargs)


def _plot_feature_correlation_impl(
    ctx: PlotContext,
    *,
    threshold: float = 0.8,
    return_data: bool = False,
    **kwargs,
):
    """Shared implementation for ``plot_feature_correlation``."""
    state = ctx.state_ref
    X = state.X_train
    if X is None:
        raise ValueError("No training data. Run setup() first.")

    if return_data:
        from mlsurv.viz._features_helpers import _filter_numeric_for_corr

        # stacklevel=3: _filter_numeric_for_corr -> impl -> wrapper -> user
        numeric = _filter_numeric_for_corr(X, stacklevel=3)
        corr = numeric.corr()
        return corr.reset_index(names="feature")

    from mlsurv.viz.features import plot_feature_correlation as _plot

    return _plot(X, threshold=threshold, **kwargs)


def _plot_benchmark_incremental_impl(
    ctx: PlotContext,
    *,
    return_data: bool = False,
    **kwargs,
):
    """Shared implementation for ``plot_benchmark_incremental``."""
    state = ctx.state_ref
    br = state.benchmark_result
    if br is None:
        raise ValueError("No benchmark results. Call benchmark() first.")
    inc = br.incremental
    if inc.empty:
        raise ValueError("No incremental metrics in benchmark results.")
    if return_data:
        return inc

    return build_incremental_figure(inc, **kwargs)


def _plot_feature_distributions_impl(
    ctx: PlotContext,
    *,
    features=None,
    return_data: bool = False,
    **kwargs,
):
    """Shared implementation for ``plot_feature_distributions``."""
    state = ctx.state_ref
    X_train = state.X_train
    X_test = state.X_test
    if X_train is None or X_test is None:
        raise ValueError("No training/test data. Run setup() first.")

    if features is not None:
        missing = [f for f in features if f not in X_train.columns]
        if missing:
            raise ValueError(
                f"Features not found in data: {missing}. "
                f"Available: {X_train.columns.tolist()}"
            )
        X_train = X_train[features]
        X_test = X_test[features]

    if return_data:
        parts = []
        for split_name, X in [("train", X_train), ("test", X_test)]:
            numeric = X.select_dtypes(include="number")
            melted = numeric.melt(var_name="feature", value_name="value")
            melted["split"] = split_name
            parts.append(melted)
        return pd.concat(parts, ignore_index=True)[
            ["feature", "split", "value"]
        ]

    from mlsurv.viz.distributions import plot_feature_distributions as _plot

    return _plot(
        X_train,
        X_test,
        categorical_groups=state.categorical_groups,
        **kwargs,
    )


# ======================================================================
# Tier 2 — Model-resolving methods
# ======================================================================


def _plot_shap_impl(
    ctx: PlotContext,
    *,
    models=None,
    kind: str = "importance",
    n: int = 20,
    return_data: bool = False,
    **kwargs,
):
    """Shared implementation for ``plot_shap``."""
    resolved = _resolve_models_arg(ctx, models)
    eval_result = resolve_eval_result(ctx.state_ref, resolved)

    if eval_result.shap_values is None:
        raise ValueError(
            "No SHAP values. Run evaluate(include_shap=True) or "
            "importance(method='shap') first."
        )

    if return_data:
        return _tidy_model_col(eval_result.shap_values)

    if kind == "importance":
        from mlsurv.viz.shap import plot_shap_importance as _plot

        return _plot(eval_result.shap_values, n=n, **kwargs)
    elif kind == "beeswarm":
        from mlsurv.viz.shap import plot_shap_beeswarm as _plot

        return _plot(
            eval_result.shap_values, X=ctx.X_data, n=n, **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown kind '{kind}'. Must be 'importance' or 'beeswarm'."
        )


def _plot_interactions_impl(
    ctx: PlotContext,
    *,
    model=None,
    n: int = 10,
    return_data: bool = False,
    **kwargs,
):
    """Shared implementation for ``plot_interactions``."""
    resolved = _resolve_models_arg(
        ctx, [model] if isinstance(model, str) else model,
    )
    eval_result = resolve_eval_result(ctx.state_ref, resolved)

    if eval_result.interactions is None:
        raise ValueError(
            "No interaction data. Run evaluate(include_interactions=True) "
            "or learner.interactions() first."
        )

    if return_data:
        return _tidy_model_col(_pivot_interactions_for_return(eval_result.interactions))

    from mlsurv.viz.interactions import plot_interaction_heatmap

    return plot_interaction_heatmap(eval_result.interactions, n=n, **kwargs)


def _pivot_interactions_for_return(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long-format interactions DataFrame to one row per pair.

    Storage keeps separate rows per method (``h_statistic`` and ``shap_interaction``)
    distinguished by a ``method`` column, because the heatmap plotter splits
    traces on method. For ``return_data=True`` users want a wide table with one
    row per ``(model_type, feature_1, feature_2)`` triple. Method columns whose
    values are entirely NaN are dropped rather than NaN-filled.
    """
    keys = [c for c in ('model_type', 'feature_1', 'feature_2') if c in df.columns]
    if not keys:
        return df
    metric_cols = [c for c in ('h_statistic', 'shap_interaction') if c in df.columns]
    if not metric_cols:
        return df
    present = [c for c in metric_cols if df[c].notna().any()]
    if not present:
        return df
    collapsed = (
        df.groupby(keys, dropna=False, sort=False)[present]
        .first()
        .reset_index()
    )
    return collapsed


def _plot_calibration_overlay_impl(
    ctx: PlotContext,
    *,
    models=None,
    eval_times=None,
    return_data: bool = False,
    raw: bool = False,
    **kwargs,
):
    """Shared implementation for ``plot_calibration_overlay``.

    Calibrated by default: when the selected model has an attached calibrator
    (``train(calibrate=...)``), its predicted ``S(t|x)`` curve is overlaid on
    the calibrated scale, matching the deployed model. ``raw=True`` overlays the
    uncalibrated curve.
    """
    state = ctx.state_ref
    resolved = _resolve_models_arg(ctx, models)
    # find_first_surv_model requires a list, not None
    if resolved is None:
        resolved = list((state.models or {}).keys())
    model_obj, model_name = find_first_surv_model(state, resolved)
    calibrator = _calibrator_for(ctx, model_name, raw)

    if return_data:
        return compute_overlay_return_data(
            model_obj, model_name, state, eval_times,
            calibrator=calibrator,
        )

    from mlsurv.viz._palettes import get_model_display_name
    from mlsurv.viz.calibration import plot_calibration_overlay as _plot

    kwargs.setdefault(
        "title", f"Calibration Overlay: {get_model_display_name(model_name)}"
    )
    _inject_time_unit(kwargs, ctx)
    return _plot(
        model_obj, ctx.X_data, ctx.y_data,
        eval_times=eval_times,
        calibrator=calibrator, model_name=model_name,
        y_train=ctx.y_train, **kwargs,
    )


def _plot_metric_heatmap_impl(
    ctx: PlotContext,
    *,
    models=None,
    metrics=None,
    return_data: bool = False,
    **kwargs,
):
    """Shared implementation for ``plot_metric_heatmap``.

    For a model with an attached calibrator (``train(calibrate=...)``) the
    deployed (calibrated) metrics already live in each model's ``evaluation``
    because evaluate() applies the calibrator inline, so there is no
    plot-level calibrate/raw switch.
    """
    state = ctx.state_ref
    resolved = _resolve_models_arg(ctx, models)
    df = build_heatmap_dataframe(state, resolved)

    if df is None:
        n_eval = sum(
            1
            for r in (state.models or {}).values()
            if r.evaluation is not None
        )
        raise ValueError(
            f"plot_metric_heatmap requires at least 2 evaluated "
            f"models. Found {n_eval}. Run evaluate() on more models."
        )

    if return_data:
        return _tidy_model_col(df)

    from mlsurv.viz.metrics import plot_metric_heatmap as _plot

    return _plot(df, metrics=metrics, **kwargs)


def _plot_validation_heatmap_impl(
    ctx: PlotContext,
    *,
    models=None,
    metrics=None,
    return_data: bool = False,
    **kwargs,
):
    """Shared implementation for ``plot_validation_heatmap``.

    Faceted cross-cohort metric heatmap: one panel per external-validation
    cohort, with metrics z-scored across cohorts. Distinct from
    ``plot_metric_heatmap`` (single panel, test-set, main-state models) — it
    reads every cohort registered by ``validate()``.
    """
    from mlsurv.viz._data_metrics import build_validation_heatmap_data

    state = ctx.state_ref
    validation_results = state.validation_results

    resolved = _resolve_models_arg(ctx, models)
    cohort_dfs, metric_cols, heatmap_recal, tidy_df = (
        build_validation_heatmap_data(validation_results, model_names=resolved)
    )

    if len(cohort_dfs) < 2 or len(metric_cols) < 2:
        raise ValueError(
            f"plot_validation_heatmap needs at least 2 validation cohorts, "
            f"each with at least 2 metrics. Found {len(cohort_dfs)} cohort(s) "
            f"with metrics (of {len(validation_results)} registered) and "
            f"{len(metric_cols)} metric(s). Call validate() on additional "
            f"cohorts."
        )

    if return_data:
        return tidy_df

    # An explicit ``metrics=`` must still satisfy the >= 2-metric minimum the
    # cross-cohort z-score needs; the guard above only checks the auto-detected
    # columns, so a 1-element ``metrics=`` would otherwise slip through.
    plot_metrics = list(metrics) if metrics is not None else metric_cols
    if len(plot_metrics) < 2:
        raise ValueError(
            f"plot_validation_heatmap needs at least 2 metrics to compare "
            f"across cohorts; metrics={metrics!r} selects {len(plot_metrics)}."
        )

    from mlsurv.viz.metrics import plot_faceted_metric_heatmap

    kwargs.setdefault(
        "title", "Validation Performance (z-scored, red = better)"
    )
    return plot_faceted_metric_heatmap(
        cohort_dfs,
        metrics=plot_metrics,
        recalibrated_models=heatmap_recal or None,
        **kwargs,
    )


def _plot_optuna_impl(
    ctx: PlotContext,
    *,
    models=None,
    return_data: bool = False,
    **kwargs,
):
    """Shared implementation for ``plot_optuna``."""
    state = ctx.state_ref
    resolved = _resolve_models_arg(ctx, models)
    tuned = collect_tuned_models(state, resolved)

    if not tuned:
        raise ValueError("No tuned models found. Run tune() first.")

    if return_data:
        history_df, importance_df = build_tuning_data(tuned)
        return {
            "tuning_history": _tidy_model_col(history_df),
            "param_importance": _tidy_model_col(importance_df),
        }

    return build_combined_optuna_figure(tuned, **kwargs)


# ======================================================================
# Tier 3 — Specialized resolution
# ======================================================================


def _plot_coefficients_impl(
    ctx: PlotContext,
    *,
    models=None,
    n_features: int = 20,
    return_data: bool = False,
    **kwargs,
):
    """Shared implementation for ``plot_coefficients``."""
    state = ctx.state_ref
    want_uni, trained_filter = normalize_coef_models_param(models)

    df = collect_all_coef_frames(
        state, want_uni, trained_filter, models is None,
    )

    if return_data:
        return df

    palette = kwargs.pop("palette", None)
    if palette is None and want_uni:
        palette = {"univariate": "#4393c3"}

    if "title" not in kwargs:
        kwargs["title"] = build_coefficient_title(df)

    from mlsurv.viz.features import plot_coefficients as _plot

    return _plot(df, n=n_features, palette=palette, **kwargs)


def _plot_coefficient_comparison_impl(
    ctx: PlotContext,
    *,
    models=None,
    n_features: int = 20,
    return_data: bool = False,
    **kwargs,
):
    """Shared implementation for ``plot_coefficient_comparison``."""
    state = ctx.state_ref
    cox_models = find_cox_models(state, models)
    if len(cox_models) < 2:
        raise ValueError(
            "plot_coefficient_comparison() requires at least 2 Cox-family "
            "models. Found: "
            + (list(cox_models.keys())[0] if cox_models else "none")
            + ". Run learner.run(models=['coxph', 'coxnet', ...])."
        )

    # Build comparison DataFrame: one column per model's coefficients
    names = list(cox_models.keys())
    coef_dfs = {
        name: get_coef_series(model).rename(name)
        for name, model in cox_models.items()
    }

    raw = pd.DataFrame(coef_dfs)
    merged = raw.dropna()
    n_dropped = len(raw) - len(merged)
    if n_dropped > 0:
        warnings.warn(
            f"{n_dropped} features dropped from coefficient comparison "
            f"because they are not present in all models. "
            f"Remaining features: {len(merged)}.",
            InternalWarning,
            # stacklevel=3: impl -> wrapper -> user
            stacklevel=3,
        )
    if merged.empty:
        raise ValueError(
            "No features are shared between the selected Cox models. "
            "Cannot create coefficient comparison."
        )
    merged.index.name = "feature"
    merged = merged.reset_index()

    # Rename columns to match standalone function's defaults
    x_col = f"{names[0]}_coef"
    y_col = f"{names[1]}_coef"
    merged = merged.rename(columns={names[0]: x_col, names[1]: y_col})

    if return_data:
        return merged

    from mlsurv.viz.features import plot_coefficient_comparison as _plot

    return _plot(
        merged, x_col=x_col, y_col=y_col,
        max_annotations=n_features, **kwargs,
    )


def _plot_hazard_ratios_impl(
    ctx: PlotContext,
    *,
    kind: str = "univariate",
    n_features: int = 20,
    return_data: bool = False,
    **kwargs,
):
    """Shared implementation for ``plot_hazard_ratios``."""
    state = ctx.state_ref

    if kind == "univariate":
        df = prepare_univariate_hr_df(state)
        if return_data:
            return df

        title = kwargs.pop(
            "title",
            "Univariate Cox \u2014 Hazard Ratios",
        )
        from mlsurv.viz.features import plot_hazard_ratios as _plot

        return _plot(df, n_features=n_features, title=title, **kwargs)

    elif kind == "multivariate":
        from mlsurv.viz.features import compute_hazard_ratios

        cox_models = _hr_valid_models(find_cox_models(state))
        if not cox_models:
            raise ValueError(NO_HR_MODEL_MSG)
        model = cox_models.get(
            _DEFAULT_COX_MODEL, next(iter(cox_models.values())),
        )
        ci_level = kwargs.pop("ci_level", 0.95)
        error_bars = kwargs.pop("error_bars", "ci")
        hr_df = compute_hazard_ratios(
            model, ci_level=ci_level, error_bars=error_bars,
        )

        if return_data:
            return hr_df

        from mlsurv.viz.features import plot_hazard_ratios as _plot

        return _plot(
            hr_df, n_features=n_features,
            ci_level=ci_level, error_bars=error_bars, **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown kind '{kind}'. Must be 'univariate' or 'multivariate'."
        )


def _plot_hazard_ratios_comparison_impl(
    ctx: PlotContext,
    *,
    n_features: int = 20,
    return_data: bool = False,
    **kwargs,
):
    """Shared implementation for ``plot_hazard_ratios_comparison``."""
    from mlsurv.viz.features import compute_hazard_ratios

    state = ctx.state_ref

    # --- Univariate HRs ---
    uni_df = prepare_univariate_hr_df(state)

    # --- Multivariate HRs ---
    cox_models = _hr_valid_models(find_cox_models(state))
    if not cox_models:
        raise ValueError(NO_HR_MODEL_MSG)
    model = cox_models.get(
        _DEFAULT_COX_MODEL, next(iter(cox_models.values())),
    )
    multi_df = compute_hazard_ratios(model)

    if return_data:
        uni_out = uni_df.copy()
        uni_out["model"] = "Univariate"
        multi_out = multi_df.copy()
        multi_out["model"] = "Multivariate"
        return pd.concat([uni_out, multi_out], ignore_index=True)

    from mlsurv.viz.features import plot_hazard_ratios_comparison as _plot

    title = kwargs.pop(
        "title",
        "Univariate vs Multivariate \u2014 Hazard Ratios",
    )
    palette = kwargs.pop("palette", None)
    if palette is None:
        palette = {
            "univariate": "#4393c3",   # steel blue
            "multivariate": "#d6604d",  # muted red
        }
    return _plot(
        {"Univariate": uni_df, "Multivariate": multi_df},
        n_features=n_features, title=title, palette=palette,
        **kwargs,
    )


def _plot_interaction_pdp_impl(
    ctx: PlotContext,
    *,
    model_result: "ModelResult",
    feature_1: str,
    feature_2: str,
    max_samples: int = 100,
    grid_resolution: int = 25,
    return_data: bool = False,
    xscale: str = "auto",
    yscale: str = "auto",
    **kwargs,
):
    """Shared implementation for ``plot_interaction_pdp``.

    The wrapper pre-resolves the model via
    ``self._resolve_single_model()`` and passes it as *model_result*.

    *xscale*/*yscale* (``'linear'``/``'log'``/``'log1p'``/``'auto'``) control the
    grid spacing of the x-axis (*feature_2*) and y-axis (*feature_1*).  A cached
    surface is reused only when its stored spacing matches the requested scale
    (resolved per axis); otherwise the pair is recomputed on the requested
    scale.
    """
    from mlsurv.evaluation._interactions import (
        _compute_pdp_surface,
        _is_linear_model,
        _resolve_pair_scales,
    )

    mr = model_result
    if mr.evaluation is None:
        raise ValueError(
            f"Model '{mr.name}' has no evaluation result.  "
            "Call evaluate() before plot_interaction_pdp()."
        )

    if _is_linear_model(mr.model):
        raise ValueError(
            f"Model '{mr.name}' is linear \u2014 its partial dependence is "
            "additive by construction (no interactions to visualize). "
            "Use a nonlinear model (RSF, GBSA, XGBoost, DeepSurv, DeepHit)."
        )

    state = ctx.state_ref
    cat_groups = state.categorical_groups or None

    # Reuse a cached surface only when its stored spacing matches the scale the
    # caller asked for (resolved per axis). evaluate() may cache auto-scaled
    # (log/log1p) surfaces, so matching on the requested scale -- not just "is
    # this a linear request" -- is what keeps an explicit xscale='linear' from
    # rendering a cached log surface.
    surface_data = None
    cached = mr.evaluation.interaction_surfaces
    if cached:
        # Map each axis's request by feature name so a swapped cached pair
        # (feature order differs) still compares the right scale to each axis.
        requested = {feature_1: yscale, feature_2: xscale}
        pair = sorted([feature_1, feature_2])
        for s in cached:
            if sorted([s["feature_1"], s["feature_2"]]) != pair:
                continue
            cf1, cf2 = s["feature_1"], s["feature_2"]
            # Speculative resolution: capture (don't emit) any fallback warning.
            # If the pair is recomputed below, _compute_pdp_surface re-emits it;
            # if instead we reuse the cached surface (no recompute fires), we
            # replay it so a log->linear fallback is never silently dropped on
            # the cache-hit path.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                eff1, eff2 = _resolve_pair_scales(
                    mr.model, ctx.X_data, cf1, cf2,
                    requested[cf1], requested[cf2],
                    max_samples=max_samples,
                    random_state=state.random_state,
                    categorical_groups=cat_groups,
                )
            # grid_*_scale is written by every surface builder; the "linear"
            # default only covers legacy surfaces saved before scale tracking.
            if (eff1 == s.get("grid_1_scale", "linear")
                    and eff2 == s.get("grid_2_scale", "linear")):
                surface_data = s
                for w in caught:
                    warnings.warn(w.message, w.category)
            break

    # Cache miss or scale mismatch: recompute on the requested scale.
    if surface_data is None:
        cat_features = (
            set(state.categorical_features)
            if state.categorical_features is not None else None
        )
        # Axis mapping: feature_1 = y-axis (scale_1 = yscale),
        #               feature_2 = x-axis (scale_2 = xscale).
        surface_data = _compute_pdp_surface(
            mr.model, ctx.X_data, feature_1, feature_2,
            max_samples=max_samples,
            grid_resolution=grid_resolution,
            random_state=state.random_state,
            categorical_groups=cat_groups,
            categorical_features=cat_features,
            scale_1=yscale,
            scale_2=xscale,
        )

    if return_data:
        # Build tidy DataFrame: one row per grid cell
        rows = []
        grid_1 = surface_data["grid_1"]
        grid_2 = surface_data["grid_2"]
        pdp = surface_data["pdp_surface"]
        for i, v1 in enumerate(grid_1):
            for j, v2 in enumerate(grid_2):
                rows.append({
                    surface_data["feature_1"]: v1,
                    surface_data["feature_2"]: v2,
                    "prediction": pdp[i][j],
                })
        return pd.DataFrame(rows)

    # Forward categorical + scale metadata from surface dict to viz function
    meta_keys = (
        "grid_1_type", "grid_2_type", "grid_1_labels", "grid_2_labels",
        "grid_1_scale", "grid_2_scale",
    )
    meta_kwargs = {k: surface_data[k] for k in meta_keys if k in surface_data}

    from mlsurv.viz.interactions import plot_interaction_pdp

    return plot_interaction_pdp(
        surface_data["grid_1"],
        surface_data["grid_2"],
        surface_data["pdp_surface"],
        surface_data["feature_1"],
        surface_data["feature_2"],
        **meta_kwargs,
        **kwargs,
    )


def _plot_diagnostics_impl(
    ctx: PlotContext,
    *,
    X_imputed,
    kind: str = "schoenfeld",
    features=None,
    return_data: bool = False,
    **kwargs,
):
    """Shared implementation for ``plot_diagnostics``.

    The wrapper pre-resolves the imputed feature matrix via
    ``self._get_imputed_X()`` and passes it as *X_imputed*.
    """
    y_train = ctx.y_train

    if kind == "schoenfeld":
        from mlsurv.viz.diagnostics import (
            compute_schoenfeld_data,
            plot_schoenfeld,
        )

        if return_data:
            return compute_schoenfeld_data(
                X_imputed, y_train, features=features,
                n_features=kwargs.get("n_features", 6),
            )
        _inject_time_unit(kwargs, ctx)
        return plot_schoenfeld(
            X_imputed, y_train, features=features, **kwargs,
        )
    elif kind == "deviance":
        from mlsurv.viz.diagnostics import (
            compute_deviance_data,
            plot_deviance,
        )

        if return_data:
            return compute_deviance_data(X_imputed, y_train)
        return plot_deviance(X_imputed, y_train, **kwargs)
    elif kind == "martingale":
        from mlsurv.viz.diagnostics import (
            compute_martingale_data,
            plot_martingale_interactive,
        )

        if return_data:
            return compute_martingale_data(
                X_imputed, y_train, features=features,
            )
        return plot_martingale_interactive(
            X_imputed, y_train, features=features, **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown kind '{kind}'. Must be one of: "
            f"'schoenfeld', 'deviance', 'martingale'."
        )
