"""viz data-prep layer: state -> tidy data for the Plotly renderers.

Domain helpers (features, survival, time, calibration, tuning, benchmarking)
that turn ``LearnerState`` into the tidy frames the ``viz/_plot_impl_*``
renderers consume. ``learner``'s ``VisualizationMixin`` and the renderers
import this module downward as ``mlsurv.viz._data``; it imports nothing from
``learner`` (relocated from ``learner/_visualization_impl`` to
dissolve the old ``viz <-> learner`` import cycle).

The Metrics section was peeled out into ``viz/_data_metrics.py``;
import metric data builders from there.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .._warnings import InternalWarning, SuggestionWarning
from ..logging_utils import get_logger
from ..utils import unwrap_pipeline, propagate_feature_names, clip_eval_times
from ._resolve import (
    filter_eval_result,
    validate_risk_scores,
)
from ._palettes import MODEL_DISPLAY_NAMES, get_model_display_name
from ._theme import MANUSCRIPT_DEFAULTS, create_figure
from .metrics import compute_brier_curves

logger = get_logger("learner")


# ===================================================================
# Section: Features (from _visualization_features.py)
# ===================================================================

def unwrap_pipeline_model(pipeline):
    """Unwrap an sklearn Pipeline to the inner model.

    Returns the inner model if it has ``get_coefficients()``,
    otherwise ``None``.
    """
    actual_model = unwrap_pipeline(pipeline)
    if not hasattr(actual_model, 'get_coefficients'):
        return None
    return actual_model


def ensure_feature_names(model, pipeline, X_train):
    """Populate ``feature_names_in_`` from pipeline steps or X_train."""
    propagate_feature_names(model, pipeline)
    if not hasattr(model, 'feature_names_in_') and X_train is not None:
        model.feature_names_in_ = np.array(list(X_train.columns))


def try_get_coefficients(name, model):
    """Try ``model.get_coefficients()``, returning coef or ``None``."""
    try:
        coef = model.get_coefficients()
    except (AttributeError, NotImplementedError) as exc:
        logger.debug(
            "Skipping %s in _find_cox_models: %r", name, exc,
        )
        return None
    except (ValueError, RuntimeError, KeyError, IndexError) as exc:
        warnings.warn(
            f"Unexpected error from {name}.get_coefficients(): "
            f"{exc!r}. Skipping this model.",
            InternalWarning,
            stacklevel=3,
        )
        return None
    return coef


# Cox-family model names for coefficient extraction.
# PH_ASSUMPTION_MODELS (constants.py) is used for membership checks elsewhere.
_COX_NAMES = ['coxph', 'coxnet', 'fssvm']

# Preferred model for multivariate hazard ratio plots when no explicit model
# is specified.  Falls back to first available Cox model if coxph isn't trained.
_DEFAULT_COX_MODEL = 'coxph'


def _resolve_cox_candidates(trained, models):
    """Resolve model filter to a list of (name, model_result) candidates."""
    if models is not None:
        names = [models] if isinstance(models, str) else list(models)
        return [(n, trained[n]) for n in names if n in trained]
    return [(n, trained[n]) for n in _COX_NAMES if n in trained]


def _extract_cox_model(name, model_result, X_train):
    """Extract and validate a single Cox model. Returns model or None."""
    pipeline = model_result.model
    actual_model = unwrap_pipeline_model(pipeline)
    if actual_model is None:
        return None
    ensure_feature_names(actual_model, pipeline, X_train)
    coef = try_get_coefficients(name, actual_model)
    if coef is None:
        return None
    return actual_model


def find_cox_models(state, models=None):
    """Find trained Cox-family models that expose coefficients.

    Parameters
    ----------
    state : LearnerState
        The learner's state object.
    models : None, str, or list[str]
        Model filter.  ``None`` checks all trained models.

    Returns
    -------
    dict[str, object]
        ``{name: unwrapped_model}`` for models with ``get_coefficients()``.

    Raises
    ------
    ValueError
        If no models have been trained.
    """
    trained = state.models
    if not trained:
        raise ValueError("No trained models. Run train() or run() first.")

    candidates = _resolve_cox_candidates(trained, models)
    result = {}
    for name, model_result in candidates:
        actual = _extract_cox_model(name, model_result, state.X_train)
        if actual is not None:
            result[name] = actual

    return result


def is_hazard_ratio_valid(model) -> bool:
    """True if ``exp(coef)`` is a genuine multivariable hazard ratio.

    Reads the model's ``hazard_ratio_valid`` flag (see
    ``mlsurv.models.base``). Only unpenalized Cox PH qualifies; penalized
    (CoxNet) and ranking (FSSVM) coefficients are not valid hazard ratios,
    and AFT coefficients are not log-hazards.
    """
    return bool(getattr(model, "hazard_ratio_valid", False))


def _hr_valid_models(models: dict) -> dict:
    """Filter ``{name: unwrapped_model}`` to those with valid hazard ratios."""
    return {n: m for n, m in models.items() if is_hazard_ratio_valid(m)}


def get_coef_series(model):
    """Extract coefficients from a Cox model as a named Series.

    Parameters
    ----------
    model : object
        Unwrapped Cox model with ``get_coefficients()``.

    Returns
    -------
    pd.Series
        Coefficients indexed by feature name.
    """
    coef = model.get_coefficients()
    if hasattr(coef, 'index'):
        return pd.Series(coef.values, index=coef.index)
    feature_names = (
        list(model.feature_names_in_)
        if hasattr(model, 'feature_names_in_')
        else [f"Feature {i}" for i in range(len(coef))]
    )
    return pd.Series(np.array(coef), index=feature_names)


def normalize_coef_models_param(models):
    """Normalize *models* param for ``plot_coefficients``.

    Returns ``(want_uni, trained_filter)``.
    """
    if models is None:
        return False, None
    if isinstance(models, str):
        models = [models]
    want_uni = 'univariate' in models
    trained_filter = [m for m in models if m != 'univariate'] or None
    return want_uni, trained_filter


def collect_trained_coef_frames(state, trained_filter, models_was_none):
    """Call ``find_cox_models`` and build coef DataFrames.

    Returns a list of DataFrames (may be empty).
    """
    cox_models = find_cox_models(state, trained_filter)
    if models_was_none and not cox_models:
        raise ValueError(
            "No Cox-family models with coefficients found. "
            "Run learner.run(models=['coxph', ...]) first."
        )
    if trained_filter is not None:
        missing = set(trained_filter) - set(cox_models.keys())
        if missing:
            warnings.warn(
                f"Requested models {sorted(missing)} do not have "
                f"extractable coefficients and were excluded.",
                InternalWarning,
                # collect_trained_coef_frames -> collect_all_coef_frames
                # -> impl -> wrapper -> user
                stacklevel=4,
            )
    frames = []
    for name, model in cox_models.items():
        coef = get_coef_series(model)
        frames.append(pd.DataFrame({
            'feature': coef.index,
            'coefficient': coef.values,
            'model': name,
        }))
    return frames


def build_coefficient_title(df):
    """Auto-generate a title from model display names in *df*."""
    displays = [
        get_model_display_name(m) if m.lower() in MODEL_DISPLAY_NAMES else m
        for m in df['model'].unique()
    ]
    joined = ' vs '.join(displays) if len(displays) > 1 else displays[0]
    return f"Feature Coefficients: {joined}"


def prepare_univariate_hr_df(state):
    """Prepare a univariate hazard ratio DataFrame from state.

    Renames columns to match the plot_hazard_ratios expected schema
    and fills missing CI columns.

    Returns the prepared DataFrame.
    """
    uni = state.univariate_cox
    if uni is None:
        raise ValueError("No univariate Cox results. Run setup() first.")

    col_map = {}
    if 'hazard_ratio' in uni.columns:
        col_map['hazard_ratio'] = 'hr'
    if 'hr_lower_95' in uni.columns:
        col_map['hr_lower_95'] = 'ci_lower'
    if 'hr_upper_95' in uni.columns:
        col_map['hr_upper_95'] = 'ci_upper'

    df = uni.rename(columns=col_map).copy()

    if 'hr' not in df.columns:
        raise ValueError(
            "univariate_cox DataFrame missing 'hazard_ratio' column."
        )
    ci_missing = [c for c in ('ci_lower', 'ci_upper') if c not in df.columns]
    if ci_missing:
        # SuggestionWarning for the same reason as the multivariate path in
        # viz/_features_hazard_ratios.py: ci_lower == ci_upper == hr costs the
        # plot every whisker and its CI axis label, and InternalWarning is
        # ignored by the default filters, so it happened silently. The report
        # skips this block outright when the columns are missing, but a direct
        # plot_hazard_ratios(kind='univariate') call has no such guard.
        warnings.warn(
            "Univariate Cox results do not include confidence "
            "intervals. ci_lower and ci_upper are set equal to hr.",
            SuggestionWarning,
            # prepare_univariate_hr_df -> impl -> wrapper -> user
            stacklevel=4,
        )
        for col in ci_missing:
            df[col] = df['hr']
    return df


# ------------------------------------------------------------------
# Phase 2: CC-reduction helpers (features)
# ------------------------------------------------------------------

def resolve_importance_iterations(eval_result):
    """Extract ``(iterations_df, n_repeats)`` from an EvaluationResult."""
    iterations_df = getattr(eval_result, 'importance_iterations', None)
    n_repeats = None
    if (iterations_df is not None
            and not iterations_df.empty
            and 'iteration' in iterations_df.columns):
        n_repeats = int(iterations_df['iteration'].nunique())
    return iterations_df, n_repeats


def append_single_model_title(fig, model_names):
    """Append model display name to the title for single-model importance plots."""
    if (len(model_names) == 1
            and fig.layout.title and fig.layout.title.text):
        display = get_model_display_name(model_names[0])
        parts = fig.layout.title.text.split('<br>', 1)
        parts[0] = f"{parts[0]}: {display}"
        fig.layout.title.text = '<br>'.join(parts)


def collect_all_coef_frames(state, want_uni, trained_filter, models_was_none):
    """Collect coefficient DataFrames from trained Cox models and univariate.

    Returns a concatenated DataFrame.

    Parameters
    ----------
    state : LearnerState
        Learner state holding trained models and univariate Cox results.
    want_uni : bool
        If True, append univariate Cox coefficients.
    trained_filter : list of str or None
        Restrict to these trained model names; ``None`` auto-detects.
    models_was_none : bool
        True when the caller passed ``models=None`` (collect all trained
        Cox-family models).

    Returns
    -------
    pd.DataFrame
        Concatenated coefficient frame with columns ``['feature',
        'coefficient', 'model']`` — one row per feature per model.

    Raises
    ------
    ValueError
        If no coefficient data is found.
    """
    frames = []
    if trained_filter is not None or models_was_none:
        frames = collect_trained_coef_frames(state, trained_filter, models_was_none)
    if want_uni:
        frames.append(build_univariate_cox_frame(state))
    if not frames:
        requested = trained_filter or []
        requested_str = ", ".join(requested) if requested else "(auto-detect)"
        raise ValueError(
            f"No coefficient data found. Requested models: "
            f"{requested_str}. Ensure Cox-family models are "
            f"trained with learner.run()."
        )
    return pd.concat(frames, ignore_index=True)


def build_univariate_cox_frame(state):
    """Build a coefficient DataFrame from univariate Cox results.

    Parameters
    ----------
    state : LearnerState
        Learner state holding ``univariate_cox`` results from setup().

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        If no univariate Cox results are available.
    """
    uni = state.univariate_cox
    if uni is None:
        raise ValueError("No univariate Cox results. Run setup() first.")
    col = 'reference' if 'reference' in uni.columns else 'feature'
    return pd.DataFrame({
        'feature': uni[col].values,
        'coefficient': uni['coefficient'].values,
        'model': 'Univariate',
    })


# ===================================================================
# Section: Survival (from _visualization_survival.py)
# ===================================================================

def _extract_from_eval_result(eval_result, model_names):
    """Extract test data from an evaluation result's test_set DataFrame.

    Returns a dict of ``{model_name: DataFrame}`` or ``None`` if no data found.
    """
    if eval_result is None or eval_result.test_set is None:
        return None
    result = {}
    names = model_names or eval_result.test_set['model_type'].unique()
    for name in names:
        model_df = eval_result.test_set[
            eval_result.test_set['model_type'] == name
        ]
        if len(model_df) > 0:
            result[name] = model_df[['time', 'event', 'risk_score']].copy()
    return result or None


def _predict_from_models(state, model_names):
    """Fallback: predict risk scores from fitted models."""
    models = state.models
    if model_names:
        models = {k: v for k, v in models.items() if k in model_names}
    result = {}
    for name, model_result in models.items():
        model = model_result.model
        risk_scores = model.predict(state.X_test)
        result[name] = pd.DataFrame({
            'time': state.y_test['time'],
            'event': state.y_test['event'],
            'risk_score': risk_scores,
        })
    return result


def get_survival_test_data(state, model_names=None):
    """Return ``{model_name: DataFrame}`` with time/event/risk_score."""
    from ..results.containers import EvaluationResult

    evaluated = {
        n: m for n, m in state.models.items()
        if m.evaluation is not None
    }
    eval_result = EvaluationResult.combine(evaluated) if evaluated else None

    if model_names is not None and eval_result is not None:
        eval_result = filter_eval_result(eval_result, model_names)

    extracted = _extract_from_eval_result(eval_result, model_names)
    if extracted is not None:
        return validate_risk_scores(extracted)

    # Fallback: predict from models
    result = _predict_from_models(state, model_names)
    return validate_risk_scores(result)


# ===================================================================
# Section: Time (from _visualization_time.py)
# ===================================================================

def gather_auc_time_data(eval_result, model_names, boot):
    """Extract AUC time-series chunk and optional bootstrap std chunk.

    Returns ``(chunk, std_chunk_or_None)``.
    """
    auc_t = eval_result.auc_t
    if model_names is not None:
        auc_t = auc_t[auc_t["model_type"].isin(model_names)]
    chunk = auc_t[["model_type", "eval_time"]].copy()
    value_col = "auc" if "auc" in auc_t.columns else "cindex"
    chunk["value"] = auc_t[value_col].values
    chunk["metric_name"] = "auc"

    std_chunk = None
    if boot is not None and boot.auc_t is not None and not boot.auc_t.empty:
        boot_auc = boot.auc_t
        if model_names is not None:
            boot_auc = boot_auc[boot_auc["model_type"].isin(model_names)]
        std_chunk = (
            boot_auc
            .groupby(["model_type", "eval_time"])[value_col]
            .std()
            .reset_index()
            .rename(columns={value_col: "value_std"})
        )
        std_chunk["metric_name"] = "auc"

    return chunk, std_chunk


def gather_brier_time_data_generic(
    *, fitted_models, X_train, y_train, X_test, y_test,
    eval_result=None, eval_times=None, calibrators=None,
):
    """Compute Brier curves from an explicit fitted-models dict + X/y pair.

    Used by the main-learner, validation, and subpopulation plot paths.
    Callers build the ``fitted_models`` dict and pass the relevant test X/y
    (which are not always ``state.X_test`` / ``state.y_test``).

    ``eval_times`` takes precedence over ``eval_result.auc_t["eval_time"]``
    when both are supplied; either alone is sufficient, and ``None`` for
    both lets :func:`compute_brier_curves` auto-pick.

    Returns a chunk DataFrame with columns ``model_type, eval_time, value,
    metric_name`` or ``None`` on failure (no fitted models, or
    ``compute_brier_curves`` raises ``ValueError``).
    """
    if not fitted_models:
        warnings.warn(
            "No fitted models available for Brier score computation.",
            InternalWarning, stacklevel=4,
        )
        return None

    if eval_times is None and eval_result is not None \
            and eval_result.auc_t is not None and not eval_result.auc_t.empty:
        eval_times = np.sort(eval_result.auc_t["eval_time"].unique())

    try:
        brier_df = compute_brier_curves(
            fitted_models,
            X_train, y_train,
            X_test, y_test,
            eval_times=eval_times,
            calibrators=calibrators,
        )
    except ValueError as exc:
        warnings.warn(
            f"Could not compute Brier curves: {exc}",
            InternalWarning, stacklevel=4,
        )
        return None

    chunk = brier_df[["model", "eval_time"]].copy()
    chunk = chunk.rename(columns={"model": "model_type"})
    chunk["value"] = brier_df["brier_score"].values
    chunk["metric_name"] = "brier"
    return chunk


_SUPPORTED_MULTI_METRIC_TIME_METRICS = ("auc", "brier")


def _validate_multi_metric_time_args(metrics):
    if isinstance(metrics, str):
        raise TypeError(
            f"multi-panel metrics must be a list of names, not a string. "
            f"Use metric=['{metrics}'] instead."
        )
    if not metrics:
        raise ValueError("multi-panel metrics must be a non-empty list.")
    bad = [
        m for m in metrics if m not in _SUPPORTED_MULTI_METRIC_TIME_METRICS
    ]
    if bad:
        raise ValueError(
            f"Unknown metric(s) {bad} for multi-panel time plot. "
            f"Supported: {list(_SUPPORTED_MULTI_METRIC_TIME_METRICS)}."
        )


def build_multi_metric_time_data_generic(
    *, fitted_models, X_train, y_train, X_test, y_test,
    eval_result, metrics, model_names=None, calibrators=None,
):
    """Cohort-agnostic multi-metric time-curve assembly.

    Builds the ``(combined_df, combined_std, dropped)`` tuple consumed by
    the multi-panel renderer, parameterized on the training baseline and
    test cohort so it can serve main, validation, and per-subpopulation
    slices.

    AUC times are read from ``eval_result.auc_t`` and passed explicitly to
    the Brier gatherer so both metrics share the same x-axis grid.

    Supported ``metrics``: ``"auc"``, ``"brier"``. Metrics that produce no
    data (e.g. Brier when ``compute_brier_curves`` raises) are reported
    back via ``dropped`` so the caller can emit a user-facing warning at
    the correct stacklevel depth for its call chain.
    """
    _validate_multi_metric_time_args(metrics)

    frames = []
    std_frames = []
    boot = getattr(eval_result, "bootstrap", None)
    eval_times = None
    if eval_result is not None and eval_result.auc_t is not None \
            and not eval_result.auc_t.empty:
        eval_times = np.sort(eval_result.auc_t["eval_time"].unique())

    dropped = []
    for metric in metrics:
        if metric == "auc":
            chunk, std_chunk = gather_auc_time_data(
                eval_result, model_names, boot,
            )
            frames.append(chunk)
            if std_chunk is not None:
                std_frames.append(std_chunk)
        elif metric == "brier":
            chunk = gather_brier_time_data_generic(
                fitted_models=fitted_models,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                eval_times=eval_times,
                calibrators=calibrators,
            )
            if chunk is not None:
                frames.append(chunk)
                # Brier(t) CI ribbon from the bootstrap brier_t distribution,
                # symmetric with AUC(t). brier_t reflects the deployed model's
                # S(t|x) (calibrated when a calibrator is attached), so the
                # ribbon sits on the same scale as the on-the-fly point curve.
                brier_std = compute_brier_std_df(eval_result, model_names)
                if brier_std is not None:
                    std_chunk = brier_std.rename(columns={"brier_std": "value_std"})
                    std_chunk["metric_name"] = "brier"
                    std_frames.append(std_chunk)
            else:
                dropped.append(metric)

    if not frames:
        raise ValueError("No valid metric data could be gathered.")

    combined = pd.concat(frames, ignore_index=True)
    combined_std = pd.concat(std_frames, ignore_index=True) if std_frames else None
    return combined, combined_std, dropped


def warn_dropped_metrics(dropped, cohort_label, stacklevel):
    """Emit a user-facing ``SuggestionWarning`` listing metrics that the
    gather step couldn't produce.

    The stacklevel must be chosen by the caller, since the depth from
    here back to user code differs by call path (main / validation /
    subpopulation).
    """
    if not dropped:
        return
    where = f" on cohort {cohort_label!r}" if cohort_label else ""
    warnings.warn(
        f"plot_metric_time: requested metric(s) {dropped}{where} "
        f"could not be computed and will be omitted from the plot. "
        f"See earlier warnings for the cause.",
        SuggestionWarning, stacklevel=stacklevel,
    )


def build_multi_metric_time_data(state, metrics, eval_result, model_names=None,
                                 raw=False):
    """Gather time-dependent curve data for multiple metrics on main state.

    Thin wrapper around :func:`build_multi_metric_time_data_generic`.
    Returns ``(df, std_df)`` for backward compatibility with the
    pre-``cohort_label`` main-path caller; silent drops are surfaced by
    the caller (:mod:`mlsurv.viz._plot_impl`) so the warning can pick the
    right stacklevel depth. Supported metrics: ``"auc"``, ``"brier"``.

    Brier(t) is calibrated by default for models with an attached calibrator
    (``train(calibrate=...)``); ``raw=True`` scores all models raw. AUC(t) is
    rank-based and unaffected either way.
    """
    selected = {
        name: mr
        for name, mr in state.models.items()
        if model_names is None or name in model_names
    }
    fitted_models = {name: mr.model for name, mr in selected.items()}
    calibrators = None if raw else {
        name: mr.calibrator
        for name, mr in selected.items()
        if getattr(mr, "calibrator", None) is not None
    }
    combined, combined_std, dropped = build_multi_metric_time_data_generic(
        fitted_models=fitted_models,
        X_train=state.X_train, y_train=state.y_train,
        X_test=state.X_test, y_test=state.y_test,
        eval_result=eval_result,
        metrics=metrics,
        model_names=model_names,
        calibrators=calibrators,
    )
    if dropped:
        # stacklevel=5: here → _plot_metric_time_impl main branch →
        # VisualizationMixin.plot_metric_time → user code.
        warn_dropped_metrics(dropped, cohort_label=None, stacklevel=5)
    return combined, combined_std


def compute_auc_std_df(eval_result, model_names):
    """Compute AUC(t) standard-deviation DataFrame from bootstrap iterations.

    Returns ``None`` if no bootstrap data is available.
    """
    boot = getattr(eval_result, "bootstrap", None)
    if boot is None or boot.auc_t is None or boot.auc_t.empty:
        return None
    boot_auc = boot.auc_t
    if model_names is not None:
        boot_auc = boot_auc[boot_auc["model_type"].isin(model_names)]
    return (
        boot_auc
        .groupby(["model_type", "eval_time"])["auc"]
        .std()
        .reset_index()
        .rename(columns={"auc": "auc_std"})
    )


def compute_brier_std_df(eval_result, model_names):
    """Compute Brier(t) standard-deviation DataFrame from bootstrap iterations.

    The Brier counterpart of :func:`compute_auc_std_df`: reads the per-horizon
    ``brier_t`` bootstrap distribution (collected by
    ``bootstrap(include_time_resolved=True)`` or ``evaluate(bootstrap=True)``)
    and returns per-``(model_type, eval_time)`` standard deviations. Returns
    ``None`` when no time-resolved Brier bootstrap data is available.
    """
    boot = getattr(eval_result, "bootstrap", None)
    if boot is None or boot.brier_t is None or boot.brier_t.empty:
        return None
    boot_brier = boot.brier_t
    if model_names is not None:
        boot_brier = boot_brier[boot_brier["model_type"].isin(model_names)]
    if boot_brier.empty:
        return None
    return (
        boot_brier
        .groupby(["model_type", "eval_time"])["brier"]
        .std()
        .reset_index()
        .rename(columns={"brier": "brier_std"})
    )


def align_metric_time_to_grid(
    df, render_grid, *, model_col="model_type", time_col="eval_time",
    group_cols=(),
):
    """Outer-align a time-dependent metric frame onto ``render_grid``.

    Inserts NaN-valued rows for any ``(key, time)`` combination on
    ``render_grid`` missing from ``df``, so registered report horizons that
    have not been computed yet (or fall outside the censoring-stable window)
    render as honest NA instead of being silently absent. Keys are the unique
    combinations of ``model_col`` and any ``group_cols`` (e.g. ``metric_name``,
    ``subpopulation``) already present in ``df`` — no new keys are invented.

    Parameters
    ----------
    df : pandas.DataFrame or None
        Tidy time-dependent metric frame (point or std). ``None``/empty is
        returned unchanged.
    render_grid : array-like
        Target timepoints. The output spans ``sort(unique(render_grid))`` for
        every key combination.
    model_col, time_col : str
        Column names for the model key and the timepoint.
    group_cols : sequence of str
        Extra key columns to hold fixed alongside ``model_col`` (long-format
        frames carrying ``metric_name`` / ``subpopulation``).

    Returns
    -------
    pandas.DataFrame
        ``df`` with the union grid; value columns are NaN at inserted rows.
    """
    if df is None or len(df) == 0:
        return df
    grid = np.unique(np.asarray(render_grid, dtype=float))
    if grid.size == 0:
        return df
    key_cols = [model_col, *group_cols]
    combos = df[key_cols].drop_duplicates()
    skeleton = combos.merge(
        pd.DataFrame({time_col: grid}), how="cross",
    )
    return skeleton.merge(df, on=[*key_cols, time_col], how="left")


# ===================================================================
# Section: Calibration (from _visualization_calibration.py)
# ===================================================================

def find_first_surv_model(state, resolved):
    """Find the first model that supports predict_survival_function.

    Returns ``(model_obj, model_name)`` or raises ValueError.
    Warns about excluded models.
    """
    model_obj = None
    model_name = None
    excluded = []
    for name in resolved:
        mr = state.models[name]
        if hasattr(mr.model, 'predict_survival_function'):
            model_obj = mr.model
            model_name = name
            break
        else:
            excluded.append(name)

    if excluded:
        warnings.warn(
            f"Models excluded (no predict_survival_function): "
            f"{', '.join(excluded)}.",
            InternalWarning,
            # find_first_surv_model -> impl -> wrapper -> user
            stacklevel=4,
        )

    if model_obj is None:
        raise ValueError(
            "No models support predict_survival_function(). "
            "Calibration overlay requires survival probability predictions."
        )
    return model_obj, model_name


def compute_overlay_return_data(model_obj, model_name, state, eval_times,
                                calibrator=None):
    """Compute return_data DataFrame for calibration overlay.

    Returns a tidy DataFrame with predicted vs observed survival. When
    *calibrator* (the model's attached ``S(t|x)`` calibrator from
    ``train(calibrate=...)``) is given, the predicted curve is on the calibrated
    scale, matching the deployed model; ``None`` returns the raw curve.
    """
    from lifelines import KaplanMeierFitter
    from mlsurv.evaluation._calibrators import apply_calibrator_safe

    if eval_times is None:
        times_obs = state.y_test["time"]
        eval_times = np.linspace(
            times_obs.min(), times_obs.max() * 0.95, 50,
        )
        # Keep default horizons within the training range so the fitted
        # survival step functions are never queried past their domain
        # (t > train t_max), which would raise. No-op when y_train is None.
        eval_times = clip_eval_times(eval_times, state.y_train)
    else:
        eval_times = np.asarray(eval_times)

    try:
        surv_funcs = model_obj.predict_survival_function(state.X_test)
    except (NotImplementedError, ValueError, RuntimeError, TypeError) as exc:
        raise ValueError(
            f"predict_survival_function() failed: {exc}. "
            f"This model may not support survival function "
            f"predictions for the given test data."
        ) from exc

    # Convert step functions to probabilities at eval_times
    if hasattr(surv_funcs, 'iloc'):
        surv_probs = np.array([
            np.interp(eval_times, surv_funcs.index, surv_funcs.iloc[:, i])
            for i in range(surv_funcs.shape[1])
        ])
    else:
        surv_probs = np.array([fn(eval_times) for fn in surv_funcs])
    # Calibrate the predicted curve so return_data matches the figure and the
    # deployed model (no-op passthrough when calibrator is None).
    surv_probs = apply_calibrator_safe(
        calibrator, pipeline=model_obj, X=state.X_test,
        surv_probs=surv_probs, eval_times=eval_times,
        model_name=model_name,
    )
    predicted = surv_probs.mean(axis=0)

    kmf = KaplanMeierFitter()
    kmf.fit(state.y_test["time"], state.y_test["event"])
    observed = np.interp(
        eval_times,
        kmf.survival_function_.index.values,
        kmf.survival_function_.iloc[:, 0].values,
    )

    return pd.DataFrame({
        'model': model_name,
        'eval_time': eval_times,
        'predicted': predicted,
        'observed': observed,
    })


def collect_surv_function_models(state, resolved):
    """Build ``{name: model}`` dict for models with predict_survival_function.

    Returns the dict; raises ValueError if empty.
    """
    model_dict = {}
    for name in resolved:
        mr = state.models[name]
        model = mr.model
        if hasattr(model, 'predict_survival_function'):
            model_dict[name] = model
    if not model_dict:
        raise ValueError(
            "No models support predict_survival_function(). "
            "Calibration plots require survival probability predictions."
        )
    return model_dict


# ===================================================================
# Section: Tuning (from _visualization_tuning.py)
# ===================================================================

# ------------------------------------------------------------------
# Model collection
# ------------------------------------------------------------------

def _has_tuning_data(result) -> bool:
    """Check if a ModelResult has tuning data (live study or serialized)."""
    return (
        result.optuna_study is not None
        or bool(getattr(result, '_optuna_tuning_history', None))
    )


def collect_tuned_models(state, model_names=None):
    """Return ``{name: ModelResult}`` for models with tuning data.

    Accepts models with a live ``optuna_study`` or pre-extracted
    ``_optuna_tuning_history`` (from deserialized learners).

    Parameters
    ----------
    state : LearnerState
    model_names : list[str] or None
        If given, only include these models.

    Returns
    -------
    dict[str, ModelResult]
    """
    models = state.models or {}
    if model_names is not None:
        models = {k: v for k, v in models.items() if k in model_names}
    return {
        name: result
        for name, result in models.items()
        if _has_tuning_data(result)
    }


# ------------------------------------------------------------------
# Tidy data builders
# ------------------------------------------------------------------

def build_tuning_data(models_with_studies):
    """Build tidy history and param-importance DataFrames.

    Parameters
    ----------
    models_with_studies : dict[str, ModelResult]
        Models that have ``optuna_study`` attached.

    Returns
    -------
    (history_df, importance_df) : tuple[pd.DataFrame, pd.DataFrame]
    """
    history_rows = []
    importance_rows = []

    for model_name, result in models_with_studies.items():
        study = result.optuna_study

        if study is not None:
            # --- Live study path ---
            for trial in study.trials:
                row = {
                    'model_type': model_name,
                    'trial': trial.number,
                    'value': trial.value,
                    'state': str(trial.state.name),
                }
                for param_name, param_value in trial.params.items():
                    row[param_name] = param_value
                history_rows.append(row)

            try:
                from optuna.importance import get_param_importances
                importances = get_param_importances(study)
                for param_name, importance in importances.items():
                    importance_rows.append({
                        'model_type': model_name,
                        'parameter': param_name,
                        'importance': importance,
                    })
            except ImportError:
                warnings.warn(
                    f"optuna.importance not available — skipping parameter "
                    f"importance for {model_name}.",
                    InternalWarning,
                    stacklevel=4,
                )
            except (RuntimeError, ValueError) as exc:
                warnings.warn(
                    f"Could not compute parameter importance for "
                    f"{model_name}: {exc}",
                    InternalWarning,
                    stacklevel=4,
                )
        else:
            # --- Deserialized fallback: use pre-extracted data ---
            saved_history = getattr(result, '_optuna_tuning_history', None)
            if saved_history is not None and len(saved_history) > 0:
                for row in saved_history:
                    row_copy = dict(row)
                    row_copy['model_type'] = model_name
                    history_rows.append(row_copy)

            saved_imp = getattr(result, '_optuna_param_importances', None)
            if saved_imp is not None and len(saved_imp) > 0:
                for param_name, importance in saved_imp.items():
                    importance_rows.append({
                        'model_type': model_name,
                        'parameter': param_name,
                        'importance': importance,
                    })

    history_df = pd.DataFrame(history_rows) if history_rows else pd.DataFrame()
    importance_df = (
        pd.DataFrame(importance_rows) if importance_rows else pd.DataFrame()
    )
    return history_df, importance_df


# ------------------------------------------------------------------
# Optuna figure builder
# ------------------------------------------------------------------

def _copy_bar_trace(trace):
    """Copy a Plotly bar trace for use in subplots."""
    import plotly.graph_objects as go
    return go.Bar(
        x=getattr(trace, 'x', None),
        y=getattr(trace, 'y', None),
        orientation=getattr(trace, 'orientation', None),
        marker=getattr(trace, 'marker', None),
        name=getattr(trace, 'name', None),
        showlegend=False,
    )


def _copy_scatter_trace(trace):
    """Copy a Plotly scatter trace for use in subplots."""
    import plotly.graph_objects as go
    kwargs = dict(
        x=getattr(trace, 'x', None),
        y=getattr(trace, 'y', None),
        marker=getattr(trace, 'marker', None),
        name=getattr(trace, 'name', None),
        showlegend=False,
    )
    mode = getattr(trace, 'mode', None)
    if mode:
        kwargs['mode'] = mode
        kwargs['line'] = getattr(trace, 'line', None)
    return go.Scatter(**kwargs)


def _add_optuna_traces(fig, param_imp_fig, opt_hist_fig, row=1):
    """Add param importance and history traces to *fig* at the given row."""
    if param_imp_fig is not None:
        for trace in param_imp_fig.data:
            fig.add_trace(_copy_bar_trace(trace), row=row, col=1)

    if opt_hist_fig is not None:
        for trace in opt_hist_fig.data:
            fig.add_trace(_copy_scatter_trace(trace), row=row, col=2)


def _collect_optuna_plots(study):
    """Generate individual Optuna plots, returning ``(imp_fig, hist_fig)``."""
    import optuna.visualization as vis

    param_imp_fig = None
    opt_hist_fig = None

    try:
        param_imp_fig = vis.plot_param_importances(study)
    except (RuntimeError, ValueError) as exc:
        warnings.warn(
            f"Optuna parameter importance plot failed: {exc}",
            InternalWarning,
            # _collect_optuna_plots -> build_combined.. -> impl -> wrapper -> user
            stacklevel=5,
        )

    try:
        opt_hist_fig = vis.plot_optimization_history(study)
    except (RuntimeError, ValueError) as exc:
        warnings.warn(
            f"Optuna optimization history plot failed: {exc}",
            InternalWarning,
            # _collect_optuna_plots -> build_combined.. -> impl -> wrapper -> user
            stacklevel=5,
        )

    return param_imp_fig, opt_hist_fig


def _build_plots_from_saved_data(result):
    """Build param importance + history plots from pre-extracted data."""
    import plotly.graph_objects as go

    param_imp_fig = None
    opt_hist_fig = None

    saved_imp = getattr(result, '_optuna_param_importances', None)
    if saved_imp:
        params = list(saved_imp.keys())
        values = list(saved_imp.values())
        param_imp_fig = go.Figure(data=[
            go.Bar(y=params, x=values, orientation='h')
        ])

    saved_history = getattr(result, '_optuna_tuning_history', None)
    if saved_history:
        trials = [r['trial'] for r in saved_history]
        vals = [r['value'] for r in saved_history]
        # Match Optuna's plot_optimization_history (used when the live study
        # is available): one point per trial plus the best value so far.
        # mlsurv studies always maximize, so "best" is the running maximum.
        best_so_far = np.fmax.accumulate(np.asarray(vals, dtype=float))  # skips NaN
        opt_hist_fig = go.Figure(data=[
            go.Scatter(x=trials, y=vals, mode='markers',
                       name='Objective Value'),
            go.Scatter(x=trials, y=best_so_far.tolist(), mode='lines',
                       name='Best Value'),
        ])

    return param_imp_fig, opt_hist_fig


def build_combined_optuna_figure(models_with_studies, **kwargs):
    """Build a combined Optuna figure with param importance + history.

    For a single model: 1 row, 2 columns.
    For multiple models: N rows, 2 columns (one row per model).

    Parameters
    ----------
    models_with_studies : dict[str, ModelResult]
    **kwargs
        ``width``, ``height``, ``title`` forwarded to ``fig.update_layout``.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If no Optuna plots could be generated (e.g. the studies have too few
        completed trials to visualize).
    """
    from plotly.subplots import make_subplots

    n_models = len(models_with_studies)

    # Build subplot titles: ["Model A – Param Importance", "Model A – History", ...]
    subplot_titles = []
    for name in models_with_studies:
        display = get_model_display_name(name)
        if n_models == 1:
            subplot_titles.extend(["Parameter Importance", "Optimization History"])
        else:
            subplot_titles.extend([
                f"{display} – Param Importance",
                f"{display} – History",
            ])

    # Plotly sizes the gap between rows as a fraction of the total height, so
    # a fixed fraction grows with the number of models. Keep it ~80 px.
    height = kwargs.pop('height', 400 * n_models)
    fig = make_subplots(
        rows=n_models, cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.12,
        vertical_spacing=min(0.15, 80 / height) if n_models > 1 else 0.1,
    )

    for row_idx, (name, result) in enumerate(models_with_studies.items(), 1):
        if result.optuna_study is not None:
            param_imp_fig, opt_hist_fig = _collect_optuna_plots(result.optuna_study)
        else:
            param_imp_fig, opt_hist_fig = _build_plots_from_saved_data(result)
        _add_optuna_traces(fig, param_imp_fig, opt_hist_fig, row=row_idx)

        fig.update_xaxes(title_text="Importance", row=row_idx, col=1)
        fig.update_yaxes(title_text="Hyperparameter", row=row_idx, col=1)
        fig.update_xaxes(title_text="Trial", row=row_idx, col=2)
        fig.update_yaxes(title_text="Objective Value", row=row_idx, col=2)

    if not fig.data:
        raise ValueError(
            "Could not generate any Optuna plots. This typically means "
            "the studies have too few completed trials for visualization."
        )

    default_title = "Hyperparameter Optimization"
    if n_models == 1:
        only_name = next(iter(models_with_studies))
        default_title = (
            f"{get_model_display_name(only_name)} Hyperparameter Optimization"
        )

    fig.update_layout(
        title=kwargs.pop('title', default_title),
        height=height,
        width=kwargs.pop('width', 1000),
        showlegend=False,
    )

    return fig


# ===================================================================
# Section: Benchmarking (from _visualization_benchmarking.py)
# ===================================================================

_DISPLAY_MAP = {
    "delta_c_index": "\u0394 C-index",
    "nri": "NRI",
    "idi": "IDI",
}
_METRIC_COLORS = {
    "delta_c_index": "#0072B2",
    "nri": "#D55E00",
    "idi": "#009E73",
}


def _build_incremental_data_arrays(inc):
    """Build parallel data arrays for the incremental forest plot."""
    y_labels, points, ci_lowers, ci_uppers, p_values, colors = (
        [], [], [], [], [], [],
    )
    for metric_name in ["delta_c_index", "nri", "idi"]:
        sub = inc[inc["metric"] == metric_name]
        if sub.empty:
            continue
        label_prefix = _DISPLAY_MAP.get(metric_name, metric_name)
        color = _METRIC_COLORS.get(metric_name, "#999999")
        for _, row in sub.iterrows():
            y_labels.append(
                f"{label_prefix}: {row['model']} vs {row['reference']}"
            )
            points.append(row["point"])
            ci_lowers.append(row.get("ci_lower", np.nan))
            ci_uppers.append(row.get("ci_upper", np.nan))
            p_values.append(row.get("p_value", np.nan))
            colors.append(color)
    return y_labels, points, ci_lowers, ci_uppers, p_values, colors


def _build_hover_texts(y_labels, points, ci_lowers, ci_uppers, p_values, has_ci):
    """Build hover text strings for each point."""
    from mlsurv.evaluation._incremental import format_p_sig

    texts = []
    for i, label in enumerate(y_labels):
        ht = f"{label}<br>Value: {points[i]:.3f}"
        if has_ci and not np.isnan(ci_lowers[i]):
            ht += f"<br>CI: [{ci_lowers[i]:.3f}, {ci_uppers[i]:.3f}]"
        if not np.isnan(p_values[i]):
            ht += f"<br>p = {format_p_sig(p_values[i])}"
        texts.append(ht)
    return texts


def _add_ci_shapes(fig, y_pos, ci_lowers, ci_uppers, colors):
    """Add CI line shapes to the figure."""
    for i in range(len(y_pos)):
        if not np.isnan(ci_lowers[i]):
            fig.add_shape(
                type="line",
                x0=ci_lowers[i], x1=ci_uppers[i],
                y0=y_pos[i], y1=y_pos[i],
                line=dict(color=colors[i], width=2),
            )


def build_incremental_figure(inc, **kwargs):
    """Build incremental metrics forest plot figure.

    Parameters
    ----------
    inc : pd.DataFrame
        Incremental metrics DataFrame from ``benchmark_result.incremental``.
    **kwargs
        Figure customization (title, width, height, ci_level).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    y_labels, points, ci_lowers, ci_uppers, p_values, colors = (
        _build_incremental_data_arrays(inc)
    )
    n_rows = len(y_labels)
    y_pos = list(range(n_rows))
    has_ci = any(not np.isnan(v) for v in ci_lowers)

    title = kwargs.pop("title", "Incremental Value Over References")
    if has_ci:
        ci_pct = f"{kwargs.pop('ci_level', 0.95) * 100:g}"
        title += (
            f"</b><br><sup style='font-weight:normal'>"
            f"{ci_pct}% bootstrap percentile CI</sup><b>"
        )

    defaults = MANUSCRIPT_DEFAULTS
    fig = create_figure(
        title=title,
        width=kwargs.pop("width", "single"),
        height=kwargs.pop("height", max(250, 30 * n_rows + 100)),
    )
    fig.update_layout(title_pad_b=8)

    hover_texts = _build_hover_texts(y_labels, points, ci_lowers, ci_uppers, p_values, has_ci)

    fig.add_trace(go.Scatter(
        x=points,
        y=y_pos,
        mode="markers",
        marker=dict(size=defaults["marker_size_main"], color=colors),
        showlegend=False,
        hovertemplate="%{text}<extra></extra>",
        text=hover_texts,
    ))

    if has_ci:
        _add_ci_shapes(fig, y_pos, ci_lowers, ci_uppers, colors)

    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        yaxis=dict(
            tickvals=y_pos,
            ticktext=y_labels,
            autorange="reversed",
        ),
        xaxis_title="Metric Value",
        margin=dict(l=defaults["margin_l"] + 180),
    )
    return fig
