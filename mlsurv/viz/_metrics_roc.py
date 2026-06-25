"""ROC/Brier computation and plotting.

Contains ROC/Brier compute helpers (``compute_brier_curves()``,
``compute_roc_curves()``), the multi-reference assembly helper
(``assemble_roc_curves_with_references()`` -> ``RocAssembly``),
bootstrap CI bands (``bootstrap_roc_curves()``), and plotting
(``plot_roc()``, ``plot_roc_interactive()``).
"""

from typing import Callable, Dict, List, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mlsurv._shared import percentile_ci
from mlsurv._warnings import InternalWarning, MethodologicalWarning
from mlsurv.viz._palettes import (
    get_model_color,
    get_model_display_name as _model_display,
)
from mlsurv.viz._theme import apply_size_overrides, create_figure
from mlsurv.viz._validation import validate_columns


# =============================================================================
# Compute helper: compute_brier_curves()
# =============================================================================

def compute_brier_curves(
    models: Dict,
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
    eval_times: Optional[np.ndarray] = None,
    calibrators: Optional[Dict] = None,
) -> pd.DataFrame:
    """Compute Brier score curves for multiple models.

    Parameters
    ----------
    models : dict
        ``{model_name: fitted_model}`` dictionary.
    X_train : DataFrame
        Training features (for IPCW estimation).
    y_train : SurvivalArray
        Training outcomes (for IPCW estimation).
    X_test : DataFrame
        Test features for evaluation.
    y_test : SurvivalArray
        Test outcomes for evaluation.
    eval_times : array-like, optional
        Timepoints for evaluation.
    calibrators : dict, optional
        ``{model_name: SurvivalCalibrator}`` for models with an attached
        calibrator (``train(calibrate=...)``). Those models' predicted
        ``S(t|x)`` is warped to the calibrated scale before scoring, so the
        Brier(t) curve (and IBS) match the deployed model's evaluation metrics.
        Models absent from the mapping (or all models when ``None``) are scored
        raw.

    Returns
    -------
    DataFrame
        Columns: ``model``, ``eval_time``, ``brier_score``, ``ibs``.

    Raises
    ------
    ValueError
        If fewer than two evaluation timepoints fall within the observed
        time range, or if Brier scores could not be computed for any model.
    """
    import warnings
    from sksurv.metrics import brier_score
    from mlsurv.evaluation.metrics import _surv_funcs_to_probs, _integrated_brier_dense
    from mlsurv.evaluation._calibrators import apply_calibrator_safe
    from mlsurv.utils import truncate_y_to_train_range
    from mlsurv.viz.calibration import _resolve_default_eval_times

    calibrators = calibrators or {}

    if eval_times is None:
        # Unify the default grid with the canonical resolver (training-anchored
        # event-time percentiles, clipped to the censoring-stable window) so the
        # Brier(t) panel shares the AUC(t) panel's x-axis instead of defaulting
        # to raw test-cohort percentiles.
        eval_times = _resolve_default_eval_times(
            y_test, y_train, percentiles=[10, 25, 50, 75, 90],
            caller="compute_brier_curves",
            override_kwarg="eval_times=[<float>, ...]",
        )

    eval_times = np.sort(np.asarray(eval_times, dtype=float))
    max_time = y_test['time'].max()
    eval_times = eval_times[eval_times < max_time]

    if len(eval_times) < 2:
        raise ValueError("Not enough evaluation timepoints within observed range")

    y_test_ipcw = truncate_y_to_train_range(y_test, y_train)

    rows = []
    for model_name, model in models.items():
        if not hasattr(model, 'predict_survival_function'):
            warnings.warn(
                f"Skipping {model_name}: does not expose "
                f"predict_survival_function(). Brier score requires "
                f"survival probabilities.",
                InternalWarning, stacklevel=2,
            )
            continue

        try:
            cal = calibrators.get(model_name)
            surv_funcs = model.predict_survival_function(X_test)
            surv_probs = _surv_funcs_to_probs(surv_funcs, eval_times)
            # Calibrate the displayed Brier(t) curve so it matches the deployed
            # model (no-op when cal is None). Warpers map surv_probs;
            # breslow_slope ignores them and recomputes from pipeline.predict(X).
            surv_probs = apply_calibrator_safe(
                cal, pipeline=model, X=X_test,
                surv_probs=surv_probs, eval_times=eval_times,
                model_name=model_name,
            )
            _, bs_scores = brier_score(y_train, y_test_ipcw, surv_probs, eval_times)
            # Brier(t) curve stays on the display grid above; IBS is integrated
            # over the dense, train-anchored event-time grid (decoupled from the
            # display grid) so the reported IBS matches evaluate(). The calibrator
            # is reapplied on the dense grid via the transform hook.
            cal_transform = None
            if cal is not None:
                def cal_transform(probs, times, _cal=cal, _model=model,
                                  _name=model_name):
                    return apply_calibrator_safe(
                        _cal, pipeline=_model, X=X_test,
                        surv_probs=probs, eval_times=times, model_name=_name,
                    )
            ibs = _integrated_brier_dense(
                model, X_test, y_test, y_train,
                surv_probs_transform=cal_transform,
            )

            for t, bs in zip(eval_times, bs_scores):
                rows.append({
                    'model': model_name,
                    'eval_time': t,
                    'brier_score': bs,
                    'ibs': ibs,
                })
        except NotImplementedError:
            warnings.warn(
                f"Skipping {model_name}: predict_survival_function() "
                f"not supported. Brier score requires survival probabilities.",
                InternalWarning, stacklevel=2,
            )
        except (ValueError, RuntimeError, ZeroDivisionError, ImportError, np.linalg.LinAlgError) as e:
            warnings.warn(
                f"Could not compute Brier score for {model_name}: {e}",
                InternalWarning, stacklevel=2,
            )

    if not rows:
        raise ValueError(
            f"Could not compute Brier score for any model. "
            f"All {len(models)} model(s) failed: {sorted(models.keys())}. "
            f"Check warnings above for per-model details."
        )

    return pd.DataFrame(rows)


# =============================================================================
# Time-dependent ROC curve (IPCW-weighted)
# =============================================================================

def compute_roc_curves(
    y_train,
    y_test,
    risk_scores: Dict[str, np.ndarray],
    t: float,
    *,
    tied_tol: float = 1e-8,
) -> pd.DataFrame:
    """Compute IPCW-weighted time-dependent ROC curves at a single time *t*.

    Reimplements the algorithm from
    :func:`sksurv.metrics.cumulative_dynamic_auc` but retains the full
    TPR/FPR curve instead of discarding it after AUC integration.

    Parameters
    ----------
    y_train : SurvivalArray
        Training survival outcomes (for fitting the censoring distribution).
    y_test : SurvivalArray
        Test survival outcomes.
    risk_scores : dict[str, ndarray]
        ``{model_name: risk_score_array}`` — higher scores mean higher risk.
    t : float
        Evaluation time point.
    tied_tol : float, default=1e-8
        Tolerance for declaring tied risk scores.

    Returns
    -------
    pandas.DataFrame
        Columns: ``model``, ``fpr``, ``tpr``, ``threshold``,
        ``auc``, ``eval_time``.

    Raises
    ------
    ValueError
        If IPCW censoring-distribution estimation fails, if there are no
        cases (events before *t*) or no controls (observations after *t*),
        or if a model's risk-score array length does not match the number
        of test observations.
    """
    import warnings

    from sksurv.nonparametric import CensoringDistributionEstimator

    from mlsurv.utils import truncate_y_to_train_range

    y_test_trunc = truncate_y_to_train_range(y_test, y_train)

    try:
        cens = CensoringDistributionEstimator()
        cens.fit(y_train)
        ipcw = cens.predict_ipcw(y_test_trunc)
    except ValueError as e:
        raise ValueError(
            f"IPCW censoring distribution estimation failed: {e}. "
            f"Check that y_train contains both events and censored "
            f"observations."
        ) from e

    if not np.all(np.isfinite(ipcw)):
        n_bad = int((~np.isfinite(ipcw)).sum())
        warnings.warn(
            f"IPCW weights contain {n_bad} non-finite value(s). "
            f"This typically indicates heavy censoring near t={t}.",
            InternalWarning, stacklevel=2,
        )

    test_event = y_test_trunc["event"]
    test_time = y_test_trunc["time"]

    is_case = (test_time <= t) & test_event
    is_control = test_time > t
    n_controls = is_control.sum()
    n_weighted_cases = (is_case * ipcw).sum()

    if n_weighted_cases == 0:
        warnings.warn(
            f"No cases (events before t={t}). Skipping all models.",
            MethodologicalWarning, stacklevel=2,
        )
        raise ValueError(
            f"Could not compute ROC curve for any model. "
            f"No cases (events before t={t})."
        )
    if n_controls == 0:
        warnings.warn(
            f"No controls (observations after t={t}). Skipping all models.",
            MethodologicalWarning, stacklevel=2,
        )
        raise ValueError(
            f"Could not compute ROC curve for any model. "
            f"No controls (observations after t={t})."
        )

    from mlsurv.utils import trapz as _trapz

    rows = []
    for model_name, scores in risk_scores.items():
        if len(scores) != len(test_time):
            raise ValueError(
                f"risk_scores['{model_name}'] has length {len(scores)} "
                f"but y_test has {len(test_time)} observations."
            )
        if not np.all(np.isfinite(scores)):
            n_bad = int((~np.isfinite(scores)).sum())
            warnings.warn(
                f"risk_scores['{model_name}'] contains {n_bad} non-finite "
                f"value(s). Skipping this model.",
                InternalWarning, stacklevel=2,
            )
            continue

        # Sort by risk score descending (same as sksurv)
        order = np.argsort(-scores)
        scores_sorted = scores[order]
        is_case_sorted = is_case[order]
        is_control_sorted = is_control[order]
        ipcw_sorted = ipcw[order]

        # Cumulative sums for TPR/FPR
        cumsum_tp = np.cumsum(is_case_sorted * ipcw_sorted)
        cumsum_fp = np.cumsum(is_control_sorted)

        tpr = cumsum_tp / cumsum_tp[-1]
        fpr = cumsum_fp / n_controls

        # Handle ties: prepend inf sentinel, find tied positions, remove them
        scores_with_sentinel = np.concatenate(([np.inf], scores_sorted))
        is_tied = np.abs(np.diff(scores_with_sentinel)) <= tied_tol
        tied_idx = np.flatnonzero(is_tied) - 1

        tpr_no_ties = np.delete(tpr, tied_idx)
        fpr_no_ties = np.delete(fpr, tied_idx)
        thresholds_no_ties = np.delete(scores_sorted, tied_idx)

        # Prepend origin (0, 0) with inf threshold
        tpr_curve = np.r_[0.0, tpr_no_ties]
        fpr_curve = np.r_[0.0, fpr_no_ties]
        threshold_curve = np.r_[np.inf, thresholds_no_ties]

        auc_val = float(_trapz(tpr_curve, fpr_curve))

        if not np.isfinite(auc_val):
            warnings.warn(
                f"AUC is non-finite for model '{model_name}' at t={t}. "
                f"Skipping. Likely caused by degenerate IPCW weights.",
                InternalWarning, stacklevel=2,
            )
            continue

        rows.append(pd.DataFrame({
            "model": model_name,
            "fpr": fpr_curve,
            "tpr": tpr_curve,
            "threshold": threshold_curve,
            "auc": auc_val,
            "eval_time": t,
        }))

    if not rows:
        raise ValueError(
            f"Could not compute ROC curve for any model. "
            f"All {len(risk_scores)} model(s) failed at t={t}. "
            f"Check warnings above for per-model details."
        )

    return pd.concat(rows, ignore_index=True)


class RocAssembly(NamedTuple):
    """Results and diagnostics from one assembly run.

    ``roc_data`` maps each horizon to its concatenated ROC frame;
    ``all_roc_dfs`` is the ordered list of ``roc_data``'s values, provided
    for direct concat/export (the two are co-derived — kept in sync by the
    producer). ``dropped_by_ref`` / ``dropped_model_horizons`` record
    horizons that failed; ``missing_ref_arrays`` / ``mislength_refs`` flag
    references that could not be NaN-masked.
    """

    roc_data: Dict[float, pd.DataFrame]
    all_roc_dfs: List[pd.DataFrame]
    dropped_by_ref: Dict[str, List[float]]
    dropped_model_horizons: List[float]
    missing_ref_arrays: List[str]
    mislength_refs: List[str]


def assemble_roc_curves_with_references(
    y_train,
    y_data,
    risk_scores: Dict[str, np.ndarray],
    eval_times,
    *,
    reference_names=(),
    reference_arrays: Optional[Dict[str, np.ndarray]] = None,
    catch: tuple = (ValueError, RuntimeError),
    on_error: Optional[Callable] = None,
) -> RocAssembly:
    """Assemble per-horizon time-dependent ROC curves for models + references.

    Single source of truth for the partition + NaN-mask + per-horizon
    ``compute_roc_curves`` loop shared by ``_plot_roc_impl_main`` (viz),
    ``_add_roc_overlay`` (benchmark report), and the reference-free
    evaluation ROC subsection. ``risk_scores`` is split into models (scored
    on the full ``y_data``) and references named in ``reference_names``
    (scored on the finite subset of ``reference_arrays[name]``, since
    ``benchmark()`` evaluates NaN-bearing references on their valid rows
    only). ``catch`` sets which exceptions are caught per horizon; failures
    are recorded in the returned :class:`RocAssembly` and passed to the
    optional ``on_error(exc, kind, name, t)`` callback (``kind`` is
    ``"model"`` or ``"reference"``; ``name`` is ``None`` for models) so each
    caller emits its own warnings.
    """
    ref_names_set = set(reference_names or ())
    ref_arrays = reference_arrays or {}
    model_scores = {
        k: v for k, v in risk_scores.items() if k not in ref_names_set
    }
    ref_scores = {
        k: v for k, v in risk_scores.items() if k in ref_names_set
    }

    missing_ref_arrays = [r for r in ref_scores if ref_arrays.get(r) is None]
    mislength_refs = [
        r for r in ref_scores
        if ref_arrays.get(r) is not None
        and len(ref_arrays[r]) != len(y_data)
    ]

    roc_data: Dict[float, pd.DataFrame] = {}
    all_roc_dfs: List[pd.DataFrame] = []
    dropped_by_ref: Dict[str, List[float]] = {}
    dropped_model_horizons: List[float] = []

    for t in eval_times:
        per_t_dfs = []
        if model_scores:
            try:
                per_t_dfs.append(
                    compute_roc_curves(y_train, y_data, model_scores, t)
                )
            except catch as exc:
                dropped_model_horizons.append(t)
                if on_error is not None:
                    on_error(exc, "model", None, t)
        for rname, rscores in ref_scores.items():
            raw = ref_arrays.get(rname)
            if raw is not None and len(raw) == len(y_data):
                valid = np.isfinite(np.asarray(raw, dtype=float))
                y_ref = y_data[valid]
            else:
                y_ref = y_data
            try:
                per_t_dfs.append(
                    compute_roc_curves(y_train, y_ref, {rname: rscores}, t)
                )
            except catch as exc:
                dropped_by_ref.setdefault(rname, []).append(t)
                if on_error is not None:
                    on_error(exc, "reference", rname, t)
        if per_t_dfs:
            roc_df = pd.concat(per_t_dfs, ignore_index=True)
            roc_data[t] = roc_df
            all_roc_dfs.append(roc_df)

    return RocAssembly(
        roc_data=roc_data,
        all_roc_dfs=all_roc_dfs,
        dropped_by_ref=dropped_by_ref,
        dropped_model_horizons=dropped_model_horizons,
        missing_ref_arrays=missing_ref_arrays,
        mislength_refs=mislength_refs,
    )


# =============================================================================
# Bootstrap CI bands for ROC curves
# =============================================================================

def bootstrap_roc_curves(
    y_train,
    y_test,
    bootstrap_predictions: pd.DataFrame,
    X_test_index: pd.Index,
    eval_times,
    *,
    n_grid: int = 100,
    ci: float = 0.95,
) -> Dict[float, Dict[str, Dict[str, np.ndarray]]]:
    """Compute pointwise bootstrap CI bands for time-dependent ROC curves.

    For each bootstrap iteration the resampled test outcomes and risk
    scores are reconstructed from *bootstrap_predictions*, a full ROC
    curve is computed via :func:`compute_roc_curves`, and the curve is
    interpolated onto a common FPR grid.  Percentile bands are then
    derived across iterations.

    Parameters
    ----------
    y_train : SurvivalArray
        Training survival outcomes (for IPCW censoring distribution).
    y_test : SurvivalArray
        Test survival outcomes.
    bootstrap_predictions : pandas.DataFrame
        Per-iteration predictions from ``BootstrapResult.predictions``.
        Required columns: ``model_type``, ``iteration``, ``index``,
        ``risk_score``.
    X_test_index : pandas.Index
        Pandas index of the original X_test DataFrame, used to map the
        ``index`` column back to integer positions in *y_test*.
    eval_times : array-like of float
        Evaluation time points at which to compute ROC curves.
    n_grid : int, default=100
        Number of equally-spaced FPR grid points for interpolation.
    ci : float, default=0.95
        Confidence level for the bands.

    Returns
    -------
    dict[float, dict[str, dict[str, ndarray]]]
        ``{eval_time: {model_name: {'fpr': fpr_grid,
        'tpr_lower': array, 'tpr_median': array,
        'tpr_upper': array}}}``

        Models or time points with fewer than 2 successful bootstrap
        iterations are omitted from the result.

    Raises
    ------
    ValueError
        If ``ci`` is not strictly between 0 and 1, if ``n_grid`` is less
        than 2, if ``bootstrap_predictions`` is missing required columns,
        or if it contains index labels not present in ``X_test_index``
        (a stale bootstrap result).
    """
    import warnings

    if not 0 < ci < 1:
        raise ValueError(
            f"ci must be between 0 and 1 (exclusive), got {ci}"
        )
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")

    required_cols = {"model_type", "iteration", "index", "risk_score"}
    missing = required_cols - set(bootstrap_predictions.columns)
    if missing:
        raise ValueError(
            f"bootstrap_predictions missing required columns: {sorted(missing)}. "
            f"Got: {sorted(bootstrap_predictions.columns)}"
        )

    # Validate bootstrap index labels against the current X_test index.
    # Stale ``BootstrapResult`` (e.g., after re-``setup()`` or passing a
    # serialized result from a different split) can leave label mismatches
    # that would otherwise trip ``pd.Series.loc`` below with a bare
    # ``KeyError``. Convert to a descriptive ``ValueError`` so upstream
    # callers (``_compute_roc_ci_data_main``) render a ribbon-less plot with a
    # clear warning instead of surfacing an opaque traceback.
    pred_labels = pd.Index(bootstrap_predictions["index"].unique())
    stale_labels = pred_labels.difference(X_test_index)
    if len(stale_labels) > 0:
        sample = list(stale_labels[:5])
        raise ValueError(
            f"bootstrap_predictions contains {len(stale_labels)} index "
            f"label(s) not present in X_test.index. This usually means "
            f"the bootstrap result is stale (e.g., learner was re-setup "
            f"or predictions came from a different split). "
            f"First mismatched labels: {sample}"
        )

    fpr_grid = np.linspace(0, 1, n_grid)

    # Map pandas labels → integer positions (one-time)
    label_to_pos = pd.Series(range(len(X_test_index)), index=X_test_index)

    iterations = sorted(bootstrap_predictions["iteration"].unique())
    model_names = bootstrap_predictions["model_type"].unique().tolist()

    # Collect interpolated TPR curves: {(t, model): list of arrays}
    tpr_arrays: Dict[tuple, list] = {}
    for t in eval_times:
        for m in model_names:
            tpr_arrays[(t, m)] = []

    n_skipped = 0
    n_total = 0
    for i in iterations:
        iter_df = bootstrap_predictions[bootstrap_predictions["iteration"] == i]

        for model_name in model_names:
            model_iter = iter_df[iter_df["model_type"] == model_name]
            if model_iter.empty:
                continue

            positions = label_to_pos.loc[model_iter["index"].values].values
            y_boot = y_test[positions]
            risk_boot = model_iter["risk_score"].values

            for t in eval_times:
                n_total += 1
                try:
                    with warnings.catch_warnings():
                        # Suppress expected degenerate-sample warnings
                        warnings.filterwarnings("ignore", category=MethodologicalWarning)
                        warnings.filterwarnings("ignore", category=InternalWarning)
                        roc_df = compute_roc_curves(
                            y_train, y_boot, {model_name: risk_boot}, t,
                        )
                except (ValueError, RuntimeError):
                    # Degenerate bootstrap sample (no cases/controls)
                    n_skipped += 1
                    continue

                model_roc = roc_df[roc_df["model"] == model_name]
                fpr = model_roc["fpr"].values
                tpr = model_roc["tpr"].values

                tpr_interp = np.interp(fpr_grid, fpr, tpr)
                tpr_arrays[(t, model_name)].append(tpr_interp)

    if n_skipped > 0 and n_total > 0 and n_skipped / n_total > 0.5:
        warnings.warn(
            f"ROC CI bands: {n_skipped}/{n_total} bootstrap-time "
            f"combinations failed (degenerate samples). "
            f"CI bands may be unreliable.",
            MethodologicalWarning,
            stacklevel=2,
        )

    # Assemble percentile bands
    result: Dict[float, Dict[str, Dict[str, np.ndarray]]] = {}
    for t in eval_times:
        time_dict: Dict[str, Dict[str, np.ndarray]] = {}
        for m in model_names:
            curves = tpr_arrays[(t, m)]
            if len(curves) < 2:
                continue
            matrix = np.array(curves)
            lo, hi = percentile_ci(matrix, ci_level=ci)
            time_dict[m] = {
                "fpr": fpr_grid,
                "tpr_lower": lo,
                "tpr_median": np.percentile(matrix, 50, axis=0),
                "tpr_upper": hi,
            }
        if time_dict:
            result[t] = time_dict
    return result


def plot_roc(
    df: pd.DataFrame,
    *,
    model_col: str = "model",
    show_auc: bool = True,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    return_data: bool = False,
) -> Union[go.Figure, dict]:
    """Plot time-dependent ROC curve from :func:`compute_roc_curves` output.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`compute_roc_curves`.  Required columns:
        ``model``, ``fpr``, ``tpr``, ``auc``, ``eval_time``.
    model_col : str, default="model"
        Column identifying models.
    show_auc : bool, default=True
        Annotate each curve with its AUC in the legend.
    title : str, optional
        Figure title.  Defaults to
        ``"Time-Dependent ROC Curve (t = {eval_time})"``.
    width : int, optional
        Override default figure width (pixels).
    height : int, optional
        Override default figure height (pixels).
    return_data : bool, default=False
        If True, return ``{"figure": fig, "data": df}`` instead of just the
        figure.

    Returns
    -------
    go.Figure or dict
    """
    validate_columns(df, {model_col, "fpr", "tpr", "auc", "eval_time"}, "plot_roc")

    eval_time = df["eval_time"].iloc[0]
    if title is None:
        title = f"Time-Dependent ROC Curve (t = {eval_time:.1f})"

    fig = create_figure(title, width=420, height=400)

    # Diagonal reference line
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="grey", width=1, dash="dash"),
    )

    for model_name, grp in df.groupby(model_col, sort=False):
        display_name = _model_display(model_name)
        if show_auc:
            auc_val = grp["auc"].iloc[0]
            display_name = f"{display_name} (AUC = {auc_val:.3f})"

        fig.add_trace(go.Scatter(
            x=grp["fpr"],
            y=grp["tpr"],
            mode="lines",
            name=display_name,
            line=dict(color=get_model_color(model_name)),
        ))

    fig.update_xaxes(title_text="False Positive Rate", range=[0, 1],
                     dtick=0.2, constrain="domain")
    fig.update_yaxes(title_text="True Positive Rate", range=[0, 1],
                     dtick=0.2, scaleanchor="x", scaleratio=1,
                     constrain="domain")
    fig.update_layout(legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom"))

    apply_size_overrides(fig, width, height)
    if return_data:
        return {"figure": fig, "data": df}
    return fig


# =============================================================================
# Interactive ROC with eval-time dropdown
# =============================================================================

_MENU_STYLE = dict(bgcolor="#f5f7fa", bordercolor="#ddd", font=dict(size=11))


def plot_roc_interactive(
    roc_data: Dict[float, pd.DataFrame],
    *,
    title: str = "Time-Dependent ROC Curve",
    default_time_index: Optional[int] = None,
    show_auc: bool = True,
    palette: Optional[Dict[str, str]] = None,
    ci_data: Optional[Dict[float, Dict[str, Dict[str, np.ndarray]]]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> go.Figure:
    """ROC plot with a dropdown to select evaluation timepoint.

    Parameters
    ----------
    roc_data : dict[float, DataFrame]
        Keys are evaluation times; values are DataFrames from
        :func:`compute_roc_curves` (columns: ``model``, ``fpr``, ``tpr``,
        ``auc``, ``eval_time``).
    title : str
        Base plot title (timepoint appended via dropdown).
    default_time_index : int, optional
        Which dropdown entry is selected initially (default: median index).
    show_auc : bool, default=True
        Annotate each curve with its AUC in the legend.
    palette : dict, optional
        ``{model_name: hex_color}`` overrides for per-model curve colours.
    ci_data : dict, optional
        Bootstrap CI bands from :func:`bootstrap_roc_curves`.  Structure:
        ``{eval_time: {model: {'fpr': array, 'tpr_lower': array,
        'tpr_upper': array}}}``.  When provided, a shaded ribbon is
        drawn around each model's ROC curve.
    width : int, optional
        Override default figure width (pixels).
    height : int, optional
        Override default figure height (pixels).

    Returns
    -------
    go.Figure

    Raises
    ------
    ValueError
        If ``roc_data`` is empty.
    """
    from mlsurv.viz._theme import add_ci_ribbon

    if not roc_data:
        raise ValueError("roc_data is empty — nothing to plot.")

    fig = create_figure(title=title, width=650, height=455)

    # Trace 0: diagonal reference (always visible)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Random",
        showlegend=True,
    ))

    seen_times = sorted(roc_data.keys())
    time_trace_ranges: Dict[float, tuple] = {}

    for t in seen_times:
        df_t = roc_data[t]
        start_idx = len(fig.data)

        for model_name, grp in df_t.groupby("model", sort=False):
            color = (palette or {}).get(model_name, get_model_color(model_name))

            # CI ribbon (2 traces) — must precede the line so it renders behind
            if ci_data and t in ci_data and model_name in ci_data[t]:
                bands = ci_data[t][model_name]
                add_ci_ribbon(
                    fig,
                    bands["fpr"],
                    bands["tpr_upper"],
                    bands["tpr_lower"],
                    color,
                    legendgroup=model_name,
                )

            display_name = _model_display(model_name)
            if show_auc:
                auc_val = grp["auc"].iloc[0]
                display_name = f"{display_name} (AUC = {auc_val:.3f})"

            # When CI bands exist, interpolate the ROC curve onto the
            # same FPR grid so the line sits inside its ribbon.
            fpr = grp["fpr"].values
            tpr = grp["tpr"].values
            if ci_data and t in ci_data and model_name in ci_data[t]:
                fpr_grid = ci_data[t][model_name]["fpr"]
                tpr = np.interp(fpr_grid, fpr, tpr)
                fpr = fpr_grid

            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=display_name,
                line=dict(color=color),
                legendgroup=model_name,
            ))

        time_trace_ranges[t] = (start_idx, len(fig.data))

    # --- Dropdown and visibility ---
    n_traces = len(fig.data)

    if default_time_index is None:
        default_time_index = len(seen_times) // 2
    default_time_index = max(0, min(default_time_index, len(seen_times) - 1))

    buttons = []
    for t in seen_times:
        vis = [False] * n_traces
        vis[0] = True  # diagonal reference
        start, end = time_trace_ranges[t]
        for j in range(start, end):
            vis[j] = True

        title_text = f"<b>{title}</b> \u2014 t = {t:.1f}"
        buttons.append(dict(
            label=f"t = {t:.1f}",
            method="update",
            args=[{"visible": vis}, {"title.text": title_text}],
        ))

    active_idx = default_time_index

    # Set initial visibility
    default_t = seen_times[default_time_index]
    for i in range(1, n_traces):
        fig.data[i].visible = False
    start, end = time_trace_ranges[default_t]
    for j in range(start, end):
        fig.data[j].visible = True

    fig.update_xaxes(
        title_text="False Positive Rate", range=[0, 1],
        dtick=0.2, constrain="domain",
    )
    fig.update_yaxes(
        title_text="True Positive Rate", range=[0, 1],
        dtick=0.2, scaleanchor="x", scaleratio=1, constrain="domain",
    )
    fig.update_layout(
        title=dict(text=f"<b>{title}</b> \u2014 t = {default_t:.1f}"),
        legend=dict(
            x=1.02, y=0.88, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(l=80, r=120, t=100),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=1.02, xanchor="left",
            y=1.09, yanchor="top",
            buttons=buttons,
            active=active_idx,
            showactive=True,
            **_MENU_STYLE,
        )],
    )

    apply_size_overrides(fig, width, height)
    return fig
