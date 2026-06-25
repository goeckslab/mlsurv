"""Subpopulation analysis: per-subpopulation metric computation from cached predictions."""

import contextlib
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ._ctd import concordance_td
from .._iter_warn import IterativeWarningCoalescer
from .._shared import safe_call
from .._warnings import MethodologicalWarning, SmallSubpopulationWarning
from ..logging_utils import get_logger
from ..results.containers import SubpopulationResult
from ..utils import SurvivalArray, truncate_y_to_train_range
from .metrics import c_index, _auc_values_and_nan_mask, _integrated_brier_from_probs

logger = get_logger("evaluation.subpopulation")

# Minimum events per subpopulation for reliable C-index estimation
MIN_EVENTS_WARNING = 30
MIN_EVENTS_COMPUTE = 2


def _validate_subpopulations(
    subpopulations: dict,
    n_samples: int,
) -> Dict[str, np.ndarray]:
    """Validate and normalize subpopulation labels.

    Parameters
    ----------
    subpopulations : dict of {name: array-like}
        Subpopulation labels aligned with the data.
    n_samples : int
        Expected length of each array.

    Returns
    -------
    dict of {name: ndarray}
        Validated arrays with NaN entries masked.

    Raises
    ------
    ValueError
        If subpopulations is not a dict or lengths mismatch.
    """
    if not isinstance(subpopulations, dict):
        raise ValueError(
            f"subpopulations must be a dict of {{name: array}}, got {type(subpopulations).__name__}"
        )
    if not subpopulations:
        raise ValueError("subpopulations dict is empty")

    result = {}
    for name, labels in subpopulations.items():
        arr = np.asarray(labels).ravel()
        if len(arr) != n_samples:
            raise ValueError(
                f"Subpopulation '{name}' has length {len(arr)}, "
                f"expected {n_samples}"
            )
        result[name] = arr
    return result


def _compute_subpopulation_summary(
    labels: np.ndarray,
    y: SurvivalArray,
) -> pd.DataFrame:
    """Build per-subpopulation cohort summary.

    Returns
    -------
    DataFrame
        Columns: subpopulation, n_samples, n_events, event_rate.
    """
    events = np.asarray(y['event'])
    unique_labels = sorted(set(labels[~pd.isna(labels)]))
    rows = []
    for label in unique_labels:
        mask = labels == label
        n = int(mask.sum())
        n_ev = int(events[mask].sum())
        rows.append({
            'subpopulation': label,
            'n_samples': n,
            'n_events': n_ev,
            'event_rate': round(n_ev / n, 4) if n > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def _safe_metric(fn, *args, metric_name="", model_name="", label="", **kwargs):
    """Call fn(*args, **kwargs), returning NaN on failure.

    Thin wrapper around :func:`~mlsurv._shared.safe_call` that builds
    a descriptive context string from the metric/model/label identifiers.
    """
    return safe_call(
        fn, *args,
        context=f"{metric_name} for {model_name}/{label}",
        log=logger,
        **kwargs,
    )


def _compute_model_subpopulation_metrics(
    mname, test_set, mask, y_sub, y_train, eval_times,
    surv_probs_by_model, label, ipcw_tau=None, ipcw_y_train=None,
    auc_t_coalescer=None, dense_surv_probs_by_model=None,
):
    """Compute all subpopulation metrics for one model on one subpopulation.

    Returns
    -------
    tuple of (metric_rows: list, auc_t_rows: list)
    """
    metric_rows = []
    auc_t_rows = []

    model_data = test_set[test_set['model_type'] == mname]
    risk_all = np.asarray(model_data['risk_score'].values, dtype=np.float64)

    if len(risk_all) != len(mask):
        # compute_subpopulation_metrics now detects this once per
        # (model, group_var) and skips the helper for affected models
        # The MethodologicalWarning here is a defensive
        # user-visible fallback for direct callers of this helper —
        # required by the "No Silent Data Loss" rule (NaN return must
        # surface a user-facing warning, not just a log line).
        warnings.warn(
            f"Risk score length ({len(risk_all)}) != subpopulation "
            f"labels length ({len(mask)}) for model '{mname}'. "
            f"Skipping; harrell metric returned NaN.",
            MethodologicalWarning,
            stacklevel=2,
        )
        metric_rows.append({
            'subpopulation': label, 'model_type': mname,
            'metric': 'harrell', 'value': np.nan,
        })
        return metric_rows, auc_t_rows

    risk_sub = risk_all[mask]

    # Harrell C-index (always computed)
    harrell_val = _safe_metric(
        c_index, y_sub, risk_sub, method='harrell',
        metric_name='harrell', model_name=mname, label=label,
    )
    metric_rows.append({
        'subpopulation': label, 'model_type': mname,
        'metric': 'harrell', 'value': harrell_val,
    })

    # IPCW C-index (when y_train available). When ipcw_y_train is set,
    # use it as the G_hat fitting cohort instead of y_train.
    y_for_g = ipcw_y_train if ipcw_y_train is not None else y_train
    if y_for_g is not None:
        ipcw_val = _safe_ipcw(
            y_sub, risk_sub, y_for_g,
            model_name=mname, label=label, ipcw_tau=ipcw_tau,
        )
        metric_rows.append({
            'subpopulation': label, 'model_type': mname,
            'metric': 'ipcw', 'value': ipcw_val,
        })

    # IBS (when the dense IBS surv_probs cache is available for this model +
    # y_train). Uses the dense, train-anchored grid — the same one the headline
    # IBS integrates over — sliced to this subpopulation's rows. The grid is
    # shared across subpopulations, so per-subpop IBS stays comparable.
    if (dense_surv_probs_by_model is not None
            and y_train is not None
            and mname in dense_surv_probs_by_model):
        ibs_sp, ibs_sp_times = dense_surv_probs_by_model[mname]
        # ``ibs_sp`` is None when the dense grid was degenerate for this model
        # (cached as (None, times)). Guard before slicing: ``ibs_sp[mask, :]``
        # is a call argument, so it would raise outside _safe_ibs's safe_call.
        if ibs_sp is not None:
            ibs_val = _safe_ibs(
                y_sub, ibs_sp[mask, :], ibs_sp_times, y_train,
                model_name=mname, label=label,
            )
            metric_rows.append({
                'subpopulation': label, 'model_type': mname,
                'metric': 'integrated_brier', 'value': ibs_val,
            })

    # C^td (when surv_probs cached for this model)
    if (surv_probs_by_model is not None
            and mname in surv_probs_by_model):
        sp, sp_times = surv_probs_by_model[mname]
        ctd_val = _safe_ctd(
            y_sub, sp[mask, :], sp_times,
            model_name=mname, label=label,
        )
        metric_rows.append({
            'subpopulation': label, 'model_type': mname,
            'metric': 'ctd', 'value': ctd_val,
        })

    # AUC(t) (when y_train + eval_times available)
    if y_train is not None and eval_times is not None:
        auc_rows = _safe_auc_t(
            y_sub, risk_sub, y_train, eval_times,
            model_name=mname, label=label,
            auc_t_coalescer=auc_t_coalescer,
        )
        auc_t_rows.extend(auc_rows)

    return metric_rows, auc_t_rows


def compute_subpopulation_metrics(
    eval_result,
    subpopulations: Dict[str, np.ndarray],
    y_test: SurvivalArray,
    min_events: int = MIN_EVENTS_WARNING,
    y_train: Optional[SurvivalArray] = None,
    eval_times: Optional[np.ndarray] = None,
    surv_probs_by_model: Optional[Dict[str, tuple]] = None,
    ipcw_tau: Optional[float] = None,
    ipcw_y_train: Optional[SurvivalArray] = None,
    _warn_stacklevel: int = 4,
    _suppress_aggregate_warning: bool = False,
    dense_surv_probs_by_model: Optional[Dict[str, tuple]] = None,
) -> Dict[str, SubpopulationResult]:
    """Compute per-subpopulation metrics from cached risk scores.

    Computes Harrell C-index for all subpopulations. When ``y_train`` is
    provided, also computes IPCW C-index. When both ``y_train`` and
    ``eval_times`` are provided, also computes AUC(t). When
    ``surv_probs_by_model`` is provided, also computes IBS.

    Parameters
    ----------
    eval_result : EvaluationResult
        Must have ``test_set`` DataFrame with ``model_type`` and
        ``risk_score`` columns.
    subpopulations : dict of {name: array}
        Subpopulation labels aligned with y_test.
    y_test : SurvivalArray
        Test outcomes.
    min_events : int
        Warn when a subpopulation has fewer events.
    y_train : SurvivalArray, optional
        Training outcomes (for IPCW + AUC(t) + IBS metrics).
    eval_times : ndarray, optional
        Evaluation time points (for AUC(t)).
    surv_probs_by_model : dict of {model_name: (surv_probs, eval_times)}, optional
        Cached display-grid survival probabilities per model, from
        ``_compute_concordance``. Used for C^td (sliced per subpopulation).
    dense_surv_probs_by_model : dict of {model_name: (surv_probs, ibs_times)}, optional
        Cached dense, train-anchored IBS survival probabilities per model. When
        provided, IBS is computed per subpopulation by slicing this array over
        the dense grid (the same grid the headline IBS integrates over).
    ipcw_tau : float, optional
        Evaluation horizon τ for Uno's IPCW concordance per subpopulation.
        When unset, a conservative default is auto-selected (see
        :func:`mlsurv.evaluation.metrics.compute_safe_ipcw_tau`).
    ipcw_y_train : SurvivalArray, optional
        Cohort whose censoring distribution fits Uno's IPCW ``G_hat``
        estimator; overrides ``y_train`` for that purpose when given.
    _warn_stacklevel : int, default 4
        ``stacklevel`` passed to :func:`warnings.warn` for
        :class:`SmallSubpopulationWarning`. Default 4 is calibrated for the
        evaluate-storage call chain:
        ``user -> evaluate -> _compute_and_store_subpopulation_metrics ->
        compute_subpopulation_metrics -> warn``. Direct callers (e.g.
        ``validate()``) should pass 3. Mirrors the stacklevel pattern in
        :mod:`mlsurv.evaluation.evaluate`.
    _suppress_aggregate_warning : bool, default False
        When True, skip the per-group aggregate ``SmallSubpopulationWarning``
        but still populate ``SubpopulationResult.warnings``. Used by the
        evaluate-storage path (``_compute_and_store_subpopulation_metrics``)
        when ``setup()`` already warned about the same test subpops.
        The "no usable labels (all NaN)" warning is unaffected.

    Returns
    -------
    dict of {group_variable: SubpopulationResult}
    """
    test_set = getattr(eval_result, 'test_set', None)
    if test_set is None or len(test_set) == 0:
        return {}

    # Per-model risk_score length. Used to detect models whose test_set
    # rows differ from the subpopulation labels length, which would
    # otherwise emit one logger.warning per (level, model) inside
    # _compute_model_subpopulation_metrics.
    risk_lens = test_set.groupby('model_type').size().to_dict()

    validated = _validate_subpopulations(subpopulations, len(y_test))
    results = {}

    for group_var, labels in validated.items():
        summary_df = _compute_subpopulation_summary(labels, y_test)
        model_names = test_set['model_type'].unique()
        metric_rows = []
        auc_t_rows = []
        warn_msgs = []

        n_subpops = len(summary_df)
        if n_subpops == 0:
            warnings.warn(
                f"Subpopulation '{group_var}' has no usable labels (all NaN); "
                f"returning empty result.",
                SmallSubpopulationWarning,
                stacklevel=_warn_stacklevel,
            )
            results[group_var] = SubpopulationResult(
                group_variable=group_var,
                summary=summary_df,
                metrics=pd.DataFrame(),
                auc_t=None,
                warnings=warn_msgs,
            )
            continue

        # Detect models whose cached risk_score length doesn't match the
        # subpopulation labels length. The mismatch is a property of
        # (model, group_var), not subpop level, so emit ONE coalesced
        # MethodologicalWarning per group_var listing all affected models,
        # then skip them in the per-level loop.
        mismatched_models = {
            mname: risk_lens[mname]
            for mname in model_names
            if risk_lens.get(mname, 0) != len(labels)
        }
        if mismatched_models:
            with IterativeWarningCoalescer(
                mode="notices",
                summary_label=f"'{group_var}' subpopulation model",
                notice_phrase=(
                    f"have risk_score length != labels length "
                    f"({len(labels)}); skipping across all {n_subpops} levels"
                ),
                n_total=len(model_names),
                max_emit=0,
                category=MethodologicalWarning,
                # +1 vs the direct warnings.warn sites above — the
                # coalescer emits from inside coalesce_notices(), one frame
                # deeper than a direct warn at this function, so the user's
                # _warn_stacklevel (calibrated for the direct sites) needs one
                # more hop to attribute to the caller rather than this module.
                # (The coalescer already self-bumps +2 internally for its
                # summarize()/__exit__ frames — see IterativeWarningCoalescer
                # in mlsurv/_iter_warn.py; that +2 is relative to a direct
                # coalesce_*() call, and this +1 covers the coalesce_notices
                # frame itself vs the plain warnings.warn baseline.)
                stacklevel=_warn_stacklevel + 1,
            ) as miss_cz:
                for mname, n_risk in mismatched_models.items():
                    miss_cz.record_notice(
                        f"Model '{mname}' risk_score length ({n_risk}) != "
                        f"subpopulation labels length ({len(labels)}); skipping.",
                        group_key=f"{mname} (n_risk={n_risk})",
                    )

        if _suppress_aggregate_warning:
            coalescer_ctx = contextlib.nullcontext()
        else:
            coalescer_ctx = IterativeWarningCoalescer(
                mode="notices",
                summary_label=f"'{group_var}' subpopulation",
                notice_phrase=f"have insufficient events (< {min_events})",
                n_total=n_subpops,
                max_emit=0,
                category=SmallSubpopulationWarning,
                stacklevel=_warn_stacklevel + 1,  # see note above
                tail_msg="Metric estimates may be unreliable.",
            )
        # Separate coalescer for AUC(t) NaN-fallback summary.
        # Only created when AUC(t) is actually computable — _safe_auc_t
        # is only invoked when y_train is not None and eval_times is not
        # None (see _compute_model_subpopulation_metrics).
        auc_t_eligible = y_train is not None and eval_times is not None
        if auc_t_eligible:
            auc_t_ctx = IterativeWarningCoalescer(
                mode="notices",
                summary_label=f"'{group_var}' subpopulation AUC(t)",
                notice_phrase=(
                    "returned NaN (subgroup follow-up too short, "
                    "partial eval_times coverage, IPCW weights unreliable "
                    "near the end of follow-up, or sksurv error)"
                ),
                n_total=n_subpops * len(model_names),
                max_emit=0,
                category=MethodologicalWarning,
                stacklevel=_warn_stacklevel + 1,  # see note above
            )
        else:
            auc_t_ctx = contextlib.nullcontext()
        with coalescer_ctx as cz, auc_t_ctx as auc_cz:
            for label in summary_df['subpopulation']:
                mask = labels == label
                n_events = int(np.asarray(y_test['event'])[mask].sum())

                if n_events < min_events:
                    msg = (
                        f"Subpopulation '{label}' has only {n_events} events "
                        f"(< {min_events}). Metric estimates may be unreliable."
                    )
                    warn_msgs.append(msg)
                    if cz is not None:
                        cz.record_notice(msg, group_key=str(label))

                if n_events < MIN_EVENTS_COMPUTE:
                    for mname in model_names:
                        metric_rows.append({
                            'subpopulation': label,
                            'model_type': mname,
                            'metric': 'harrell',
                            'value': np.nan,
                        })
                    continue

                y_sub = y_test[mask]

                for mname in model_names:
                    if mname in mismatched_models:
                        # Length-mismatch warning already emitted at the
                        # group level; populate NaN harrell row to
                        # preserve metrics_df shape and skip metric compute.
                        metric_rows.append({
                            'subpopulation': label,
                            'model_type': mname,
                            'metric': 'harrell',
                            'value': np.nan,
                        })
                        continue
                    m_rows, a_rows = _compute_model_subpopulation_metrics(
                        mname, test_set, mask, y_sub, y_train, eval_times,
                        surv_probs_by_model, label, ipcw_tau=ipcw_tau,
                        ipcw_y_train=ipcw_y_train,
                        auc_t_coalescer=auc_cz,
                        dense_surv_probs_by_model=dense_surv_probs_by_model,
                    )
                    metric_rows.extend(m_rows)
                    auc_t_rows.extend(a_rows)

        metrics_df = pd.DataFrame(metric_rows)
        auc_t_df = pd.DataFrame(auc_t_rows) if auc_t_rows else None

        results[group_var] = SubpopulationResult(
            group_variable=group_var,
            summary=summary_df,
            metrics=metrics_df,
            auc_t=auc_t_df,
            warnings=warn_msgs,
        )

    return results


def _safe_ipcw(y_sub, risk_sub, y_train, model_name="", label="", ipcw_tau=None):
    """Compute IPCW C-index for a subpopulation, returning NaN on failure."""
    def _compute():
        from sksurv.metrics import concordance_index_ipcw
        y_sub_trunc = truncate_y_to_train_range(y_sub, y_train, warn=False)
        return concordance_index_ipcw(y_train, y_sub_trunc, risk_sub, tau=ipcw_tau)[0]

    return safe_call(
        _compute, context=f"IPCW for {model_name}/{label}", log=logger,
    )


def _safe_ibs(y_sub, surv_probs_sub, ibs_times, y_train,
              model_name="", label=""):
    """Compute IBS for a subpopulation over the dense grid, NaN on failure.

    ``surv_probs_sub`` is the dense IBS probability matrix sliced to this
    subpopulation's rows; ``ibs_times`` is the shared dense, train-anchored
    grid. Clipping to the subpopulation's follow-up range (and the ``< 2 points
    -> NaN`` guard) is handled by :func:`_integrated_brier_from_probs`.
    """
    def _compute():
        return _integrated_brier_from_probs(
            surv_probs_sub, ibs_times, y_sub, y_train,
        )

    return safe_call(
        _compute, context=f"IBS for {model_name}/{label}", log=logger,
    )


def _safe_ctd(y_sub, surv_probs_sub, eval_times, model_name="", label=""):
    """Compute Antolini's C^td for a subpopulation, returning NaN on failure."""
    def _compute():
        return float(concordance_td(
            y_sub['time'], y_sub['event'], surv_probs_sub, eval_times,
        ))

    return safe_call(
        _compute, context=f"C^td for {model_name}/{label}", log=logger,
    )


def _safe_auc_t(y_sub, risk_sub, y_train, eval_times, model_name="", label="",
                auc_t_coalescer=None):
    """Compute AUC(t) for a subpopulation, returning list of row dicts.

    Filters eval_times to the subpopulation's truncated time range to avoid
    sksurv ValueError when times fall outside [min, max] of the test set.
    Returns NaN rows for times outside the valid range, and (via the shared
    :func:`~mlsurv.evaluation.metrics._auc_values_and_nan_mask` core) detects
    the IPCW divide-instability NaN that can occur among *valid* times near
    the end of follow-up rather than reporting it silently.

    Every NaN-fallback path (``len(valid_times) < 2``, partial out-of-range
    coverage, divide-instability NaN, and recoverable sksurv exceptions) is
    recorded on ``auc_t_coalescer`` when supplied, so a single
    :class:`MethodologicalWarning` is emitted per
    :func:`compute_subpopulation_metrics` call instead of one per
    ``(model, subgroup)`` pair.
    """
    # Coerce up front so we can zip eval_times with auc_all and build the
    # nan_rows fallback when callers pass a Python list. The shared core
    # coerces internally too, but that conversion does not propagate back here.
    eval_times_arr = np.asarray(eval_times, dtype=float)
    if eval_times_arr.size == 0:
        return []
    nan_rows = [
        {'subpopulation': label, 'model_type': model_name,
         'eval_time': float(t), 'auc': np.nan}
        for t in eval_times_arr
    ]

    def _record(reason: str) -> None:
        """Surface a NaN-fallback notice. When a coalescer is provided,
        record into it; otherwise emit a standalone MethodologicalWarning
        so the "No Silent Data Loss" rule is honored for direct callers."""
        msg = (
            f"AUC(t) for {model_name}/{label}: {reason}. Reporting NaN."
        )
        if auc_t_coalescer is not None:
            auc_t_coalescer.record_notice(
                msg,
                group_key=f"{label}/{model_name}",
                display=f"{label}/{model_name}",
            )
        else:
            warnings.warn(msg, MethodologicalWarning, stacklevel=3)

    try:
        # Shared clip -> compute -> divide-filter -> NaN-detect core
        # (mirrored by _compute_auc_t in _concordance.py). auc_all is aligned
        # to eval_times with NaN at every unevaluable time, so the row list
        # stays len(eval_times) by construction.
        auc_all, nan_mask, valid_times = _auc_values_and_nan_mask(
            y_sub, y_train, eval_times_arr, risk_sub,
        )
        rows = [
            {'subpopulation': label, 'model_type': model_name,
             'eval_time': float(t), 'auc': float(a)}
            for t, a in zip(eval_times_arr, auc_all)
        ]

        if len(valid_times) < 2:
            _record(
                f"only {len(valid_times)}/{len(eval_times_arr)} eval_times "
                f"fall in the subgroup follow-up range. Pass eval_times "
                f"within the subgroup's observed time window"
            )
        elif nan_mask.any():
            # nan_mask catches out-of-range times AND divide-instability NaN
            # among the valid times (IPCW weights unreliable near the end of
            # follow-up) — the latter was silently dropped before.
            _record(
                f"{int(nan_mask.sum())}/{len(eval_times_arr)} eval_times "
                f"returned NaN (outside the subgroup follow-up range, or IPCW "
                f"weights unreliable near the end of follow-up)"
            )
        return rows
    except (MemoryError, RecursionError):
        raise
    except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
        # sksurv's documented failure modes for cumulative_dynamic_auc on
        # degenerate data. TypeError/IndexError stay uncaught — those
        # would indicate a real programmer bug we should not paper over.
        logger.warning(
            "AUC(t) for %s/%s failed: %s", model_name, label, exc,
        )
        _record(f"failed with {type(exc).__name__} ({exc})")
        return nan_rows


def extract_subpopulation_bootstrap_summary(
    bootstrap_summary: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Extract per-subpopulation rows from the bootstrap summary DataFrame.

    Per-subpopulation metrics use namespaced keys like
    ``harrell__subpopulation_NSCLC`` or ``ipcw__subpopulation_NSCLC``. This
    function filters and parses them into a tidy DataFrame.

    Parameters
    ----------
    bootstrap_summary : DataFrame
        Full bootstrap summary with ``metric`` column.

    Returns
    -------
    DataFrame or None
        Columns: subpopulation, model_type, metric, point, mean, std,
        ci_lower, ci_upper.
    """
    _SUBPOPULATION_TAG = '__subpopulation_'
    mask = bootstrap_summary['metric'].str.contains(_SUBPOPULATION_TAG, na=False)
    if not mask.any():
        return None

    sub_df = bootstrap_summary[mask].copy()

    # Parse "harrell__subpopulation_NSCLC" → metric='harrell', subpopulation='NSCLC'
    parts = sub_df['metric'].str.split(_SUBPOPULATION_TAG, n=1)
    sub_df['metric'] = parts.str[0]
    sub_df['subpopulation'] = parts.str[1]

    cols = ['subpopulation', 'model_type', 'metric']
    for c in ('point', 'mean', 'std', 'ci_lower', 'ci_upper'):
        if c in sub_df.columns:
            cols.append(c)

    return sub_df[cols].reset_index(drop=True)


def _fill_bootstrap_point_estimates(boot_sg, metrics):
    """Fill 'point' column in bootstrap summary in-place.

    For each row in *boot_sg*, looks up the matching point estimate from
    *metrics* by ``(subpopulation, model_type, metric)`` and writes it
    into the ``point`` column.
    """
    # boot_sg['subpopulation'] is always str (parsed from metric name
    # suffixes like "harrell__subpopulation_0.0"), while
    # metrics['subpopulation'] preserves the original data type (float,
    # int, etc.).  Cast metrics to str for reliable comparison.
    metrics_sub_str = metrics['subpopulation'].astype(str)
    for idx, row in boot_sg.iterrows():
        match = metrics[
            (metrics_sub_str == row['subpopulation'])
            & (metrics['model_type'] == row['model_type'])
            & (metrics['metric'] == row['metric'])
        ]
        if len(match) > 0:
            boot_sg.loc[idx, 'point'] = match.iloc[0]['value']


def _attach_per_model_bootstrap_cis(sg_results, eval_bootstrap):
    """Attach bootstrap CIs and AUC(t) ribbons to subpopulation results.

    For each group variable in *sg_results*, extracts the per-subpopulation
    rows from the bootstrap summary, fills point estimates from the computed
    metrics, and attaches AUC(t) bootstrap data.  No-op when
    *eval_bootstrap* is ``None``.

    Parameters
    ----------
    sg_results : dict of {str: SubpopulationResult}
        Subpopulation results to mutate in-place.
    eval_bootstrap : BootstrapResult or None
        Bootstrap result containing ``.summary`` and optionally
        ``.subpopulation_auc_t``.
    """
    if eval_bootstrap is None:
        return
    for gvar, sg_result in sg_results.items():
        levels = sg_result.summary['subpopulation'].tolist()
        boot_sg = extract_subpopulation_bootstrap_summary(
            eval_bootstrap.summary,
        )
        if boot_sg is not None:
            _fill_bootstrap_point_estimates(boot_sg, sg_result.metrics)
            sg_result.bootstrap_summary = boot_sg
        _attach_subpopulation_auc_bootstrap(
            sg_result,
            getattr(eval_bootstrap, 'subpopulation_auc_t', None),
            levels,
        )


def _attach_subpopulation_auc_bootstrap(sg_result, boot_auc_t, levels):
    """Filter per-subpopulation AUC(t) bootstrap data to this group's
    levels and attach it to sg_result in place.

    No-op when ``boot_auc_t`` is ``None`` or empty, and when no rows
    match ``levels`` (e.g., this group variable has no bootstrap rows
    because the bootstrap call received a different set of
    ``subpopulation_labels``).

    Extracted from duplicated blocks in the evaluate and benchmark paths
    to prevent drift.

    Parameters
    ----------
    sg_result : SubpopulationResult
        Target result to mutate; ``auc_t_bootstrap`` is set on it.
    boot_auc_t : DataFrame or None
        Per-subpopulation AUC(t) bootstrap iterations, sourced from
        ``<ModelResult>.bootstrap.subpopulation_auc_t``. Must have a
        ``subpopulation`` column.
    levels : iterable
        Subpopulation level names belonging to this group variable.
        Used to filter ``boot_auc_t`` rows.
    """
    if boot_auc_t is None or boot_auc_t.empty:
        return
    levels_set = {str(lv) for lv in levels}
    filtered = boot_auc_t[boot_auc_t['subpopulation'].isin(levels_set)]
    if not filtered.empty:
        # .copy() guards against sibling sg_results mutating a shared
        # parent view on downstream access (multiple group variables
        # can pull filtered slices from the same boot_auc_t frame).
        sg_result.auc_t_bootstrap = filtered.copy()


def _merge_df_by_model(old_df, new_df, new_models):
    """Merge two DataFrames by model_type, preferring *new_df* for *new_models*.

    Returns the merged DataFrame, or ``None`` if the result would be empty.
    """
    if old_df is None and new_df is None:
        return None
    if old_df is None:
        return new_df.copy()
    if new_df is None:
        # New evaluation didn't produce this DataFrame at all — keep old
        # data intact for all models.  This preserves e.g. bootstrap CIs
        # when re-evaluating without bootstrap.
        return old_df.copy()
    old_keep = old_df[~old_df['model_type'].isin(new_models)]
    merged = pd.concat([old_keep, new_df], ignore_index=True)
    return merged if not merged.empty else None


def _merge_subpopulation_result(old_sg, new_sg):
    """Return a new ``SubpopulationResult`` merging *new_sg* into *old_sg*.

    Preserves data for models not present in *new_sg* while replacing data
    for models that were re-evaluated.  Neither input is mutated.
    """
    # Empty metrics → nothing to merge on; treat as empty set of new models.
    new_models = set(new_sg.metrics['model_type'].unique())

    # Merge per-model DataFrames
    old_keep = old_sg.metrics[~old_sg.metrics['model_type'].isin(new_models)]
    merged_metrics = pd.concat([old_keep, new_sg.metrics], ignore_index=True)

    merged_auc_t = _merge_df_by_model(
        old_sg.auc_t, new_sg.auc_t, new_models,
    )
    merged_boot = _merge_df_by_model(
        old_sg.bootstrap_summary, new_sg.bootstrap_summary, new_models,
    )
    merged_auc_boot = _merge_df_by_model(
        old_sg.auc_t_bootstrap, new_sg.auc_t_bootstrap, new_models,
    )

    # Update point estimates in the merged bootstrap summary so they
    # reflect the latest metrics for *all* models (old + new).
    if merged_boot is not None:
        _fill_bootstrap_point_estimates(merged_boot, merged_metrics)

    return SubpopulationResult(
        group_variable=old_sg.group_variable,
        summary=old_sg.summary,
        metrics=merged_metrics,
        bootstrap_summary=merged_boot,
        auc_t=merged_auc_t,
        auc_t_bootstrap=merged_auc_boot,
        warnings=old_sg.warnings,
    )


def _merge_subpopulation_results_dict(existing, new_results):
    """Merge *new_results* into *existing* subpopulation results dict.

    For group variables present in both, merges per-model data via
    :func:`_merge_subpopulation_result`.  Group variables in *existing*
    but not in *new_results* are carried forward unchanged.

    Parameters
    ----------
    existing : dict of {str: SubpopulationResult}
        Prior results to merge into.
    new_results : dict of {str: SubpopulationResult}
        Newly computed results.

    Returns
    -------
    dict of {str: SubpopulationResult}
        Merged results (mutates *new_results* in place and returns it).
    """
    if not existing:
        return new_results
    for gvar, new_sg in new_results.items():
        old_sg = existing.get(gvar)
        if old_sg is not None:
            new_results[gvar] = _merge_subpopulation_result(old_sg, new_sg)
    for gvar in existing:
        if gvar not in new_results:
            new_results[gvar] = existing[gvar]
    return new_results
