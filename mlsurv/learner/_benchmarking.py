"""BenchmarkingMixin - benchmark() method for reference score comparison."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from .._shared import iter_subpopulations, nan_summary, safe_call
from ..evaluation import evaluate
from ..evaluation._incremental import (
    _emit_coalesced_incremental_nan_warning,
    bootstrap_incremental,
    compute_incremental_metrics,
)
from ..evaluation._univariate_cox import compute_reference_univariate_cox
from ..evaluation.evaluate import _aggregate_evaluation_results
from ..results.containers import BenchmarkResult
from ._evaluation import _resolve_subpopulation_labels
from ._evaluation_helpers import resolve_state_eval_times
from ._references import (
    _build_reference_adapters,
    _evaluate_references_on,
    _report_nonfinite_references,
    _resolve_higher_is_worse,
)
from ..logging_utils import get_logger
from .._warnings import MethodologicalWarning, SuggestionWarning, _warn_eval_times_failure

logger = get_logger("learner.benchmarking")

_VALID_DATASETS = ("test", "train", "full")


def _filter_nan_scores(reference_scores, *arrays):
    """Drop positions where reference_scores is non-finite; filter all arrays in lockstep."""
    valid = np.isfinite(reference_scores)
    if valid.all():
        return (reference_scores, *arrays)
    return (reference_scores[valid], *(a[valid] for a in arrays))


if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner


class _BenchmarkData(NamedTuple):
    """Resolved dataset for benchmarking."""

    X: pd.DataFrame
    y: np.ndarray
    subpopulations: dict | None


def _build_incremental_rows(
    y_test, model_risk_scores, adapters,
    model_names, reference_names,
    *, bootstrap, n_bootstrap, bootstrap_ci, random_state,
    nan_collector=None,
) -> list:
    """Build incremental metric rows for every model x reference pair."""
    rows = []
    for mname in model_names:
        m_scores = model_risk_scores[mname]
        for rname in reference_names:
            r_scores = adapters[rname].predict()
            r_filt, y_filt, m_filt = _filter_nan_scores(
                r_scores, y_test, m_scores,
            )
            if len(r_filt) < 2:
                logger.warning(
                    "Skipping incremental metrics for '%s' vs '%s': "
                    "only %d valid observations after filtering (need >= 2).",
                    mname, rname, len(r_filt),
                )
                continue
            try:
                if bootstrap:
                    inc_df = bootstrap_incremental(
                        y_filt, m_filt, r_filt,
                        n_bootstrap=n_bootstrap,
                        ci=bootstrap_ci,
                        random_state=random_state,
                        collector=nan_collector,
                        context={"model": mname, "reference": rname},
                    )
                    for _, row in inc_df.iterrows():
                        rows.append({
                            "model": mname,
                            "reference": rname,
                            **row.to_dict(),
                        })
                else:
                    vals = compute_incremental_metrics(
                        y_filt, m_filt, r_filt,
                    )
                    for metric_name, value in vals.items():
                        rows.append({
                            "model": mname,
                            "reference": rname,
                            "metric": metric_name,
                            "point": value,
                            **nan_summary(p_value=np.nan),
                        })
            except (MemoryError, RecursionError):
                raise
            except Exception as exc:
                logger.warning(
                    "Incremental metrics failed for '%s' vs '%s': %s",
                    mname, rname, exc,
                )
                warnings.warn(
                    f"Incremental metrics skipped for '{mname}' vs "
                    f"'{rname}': {exc}",
                    SuggestionWarning,
                    stacklevel=2,
                )
    return rows


def _build_subpopulation_incremental_rows(
    subpopulations_test, y_test, model_risk_scores, reference_arrays,
    model_names, reference_names,
    *, bootstrap, n_bootstrap, bootstrap_ci, random_state, logger,
    nan_collector=None,
) -> list:
    """Build per-subpopulation incremental metric rows."""
    if not subpopulations_test:
        return []
    rows = []
    for group_var, label, mask, y_sub in iter_subpopulations(
        subpopulations_test, y_test, min_events=2,
    ):
        for mname in model_names:
            m_sub = model_risk_scores[mname][mask]
            for rname in reference_names:
                r_sub = reference_arrays[rname][mask]
                r_filt, y_filt, m_filt = _filter_nan_scores(
                    r_sub, y_sub, m_sub,
                )
                ctx = f"Subpopulation incremental for {label}/{mname}/{rname}"
                if len(r_filt) < 2:
                    logger.info(
                        "Skipping %s: only %d valid observations "
                        "after filtering.", ctx, len(r_filt),
                    )
                    continue
                if bootstrap:
                    sub_ctx = {
                        "subpopulation": label,
                        "group_variable": group_var,
                        "model": mname,
                        "reference": rname,
                    }

                    def _boot(ys=y_filt, ms=m_filt, rs=r_filt, ctx=sub_ctx):
                        inc_df = bootstrap_incremental(
                            ys, ms, rs,
                            n_bootstrap=n_bootstrap,
                            ci=bootstrap_ci,
                            random_state=random_state,
                            collector=nan_collector,
                            context=ctx,
                        )
                        return [row.to_dict() for _, row in inc_df.iterrows()]

                    result = safe_call(_boot, default=None, context=ctx, log=logger)
                    if result is not None:
                        for row in result:
                            rows.append({
                                "subpopulation": label,
                                "group_variable": group_var,
                                "model": mname,
                                "reference": rname,
                                **row,
                            })
                else:
                    def _point(ys=y_filt, ms=m_filt, rs=r_filt):
                        return compute_incremental_metrics(ys, ms, rs)

                    vals = safe_call(_point, default=None, context=ctx, log=logger)
                    if vals is not None:
                        for metric_name, value in vals.items():
                            rows.append({
                                "subpopulation": label,
                                "group_variable": group_var,
                                "model": mname,
                                "reference": rname,
                                "metric": metric_name,
                                "point": value,
                                **nan_summary(p_value=np.nan),
                            })
    return rows


def _compute_correlations(
    model_risk_scores, reference_arrays,
    model_names, reference_names, *, logger,
) -> pd.DataFrame:
    """Compute Spearman and Pearson correlations for model x reference pairs."""
    correlation_rows = []
    for mname in model_names:
        m_scores = model_risk_scores[mname]
        for rname in reference_names:
            r_scores = reference_arrays[rname]
            r_filt, m_filt = _filter_nan_scores(r_scores, m_scores)
            try:
                sp_result = stats.spearmanr(m_filt, r_filt)
                pe_result = stats.pearsonr(m_filt, r_filt)
                correlation_rows.append({
                    "model": mname,
                    "reference": rname,
                    "spearman_rho": float(sp_result.statistic),
                    "spearman_p": float(sp_result.pvalue),
                    "pearson_r": float(pe_result.statistic),
                    "pearson_p": float(pe_result.pvalue),
                })
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Correlation failed for %s vs %s: %s",
                    mname, rname, exc,
                )
                correlation_rows.append({
                    "model": mname,
                    "reference": rname,
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                })
    return pd.DataFrame(correlation_rows)


def _attach_subpopulation_results(
    eval_result, subpopulations_test, y_test, y_train, *, eval_times=None, logger,
):
    """Compute and return subpopulation results dict, or None on failure."""
    try:
        from ..evaluation._subpopulation import (
            _attach_per_model_bootstrap_cis,
            compute_subpopulation_metrics,
        )
        from ..utils import auto_eval_times as _auto_et
        _sg_eval_times = eval_times  # Use inherited/explicit if available
        if _sg_eval_times is None and y_train is not None:
            try:
                _sg_eval_times = _auto_et(y_test, y_train=y_train, _quiet=True)
            except (ValueError, TypeError) as exc:
                _warn_eval_times_failure(
                    exc, "for benchmark subpopulation metrics",
                    "AUC(t) subpopulation metrics will be unavailable.",
                    stacklevel=3,
                )
        _sg_surv_probs = eval_result._surv_probs_cache
        sg_results = compute_subpopulation_metrics(
            eval_result, subpopulations_test, y_test,
            y_train=y_train,
            eval_times=_sg_eval_times,
            surv_probs_by_model=_sg_surv_probs,
            dense_surv_probs_by_model=eval_result._ibs_surv_probs_cache,
            _warn_stacklevel=5,
        )
        # Attach bootstrap CIs if available
        _attach_per_model_bootstrap_cis(sg_results, eval_result.bootstrap)
        return sg_results
    except (MemoryError, RecursionError):
        raise
    # Broad catch is intentional: compute_subpopulation_metrics calls
    # sksurv/numpy/pandas which can raise many exception types.
    # MemoryError/RecursionError are re-raised above.
    except Exception as exc:
        logger.warning(
            "Benchmark subpopulation metric update failed: %s",
            exc, exc_info=True,
        )
        warnings.warn(
            f"Benchmark subpopulation metric update failed "
            f"({type(exc).__name__}: {exc}). Any prior subpopulation "
            f"results are preserved; the failed update was discarded.",
            SuggestionWarning,
            stacklevel=3,  # → benchmark() via _compute_benchmark_subpopulation_results
        )
        return None


def _evaluate_with_nan_references(
    resolved_model, adapters, reference_arrays,
    X_eval, y_eval,
    *, X_train, y_train, eval_times, hiw_map, hiw_default,
    n_importance_repeats, bootstrap, n_bootstrap,
    bootstrap_ci, random_state, sg_labels, _warn_stacklevel=5,
):
    """Evaluate models on the selected dataset + each reference on its valid subset."""
    model_eval = evaluate(
        resolved_model, X_eval, y_eval,
        X_train=X_train,
        y_train=y_train,
        eval_times=eval_times,
        n_importance_repeats=n_importance_repeats,
        bootstrap=bootstrap,
        n_bootstrap=n_bootstrap,
        bootstrap_ci=bootstrap_ci,
        include_predictions=bootstrap,
        random_state=random_state,
        show_progress=False,
        _warn_stacklevel=_warn_stacklevel,
        subpopulation_labels=sg_labels,
    )

    ref_parts = _evaluate_references_on(
        X_eval, y_eval,
        reference_arrays=reference_arrays,
        adapters=adapters,
        hiw_map=hiw_map, hiw_default=hiw_default,
        y_train=y_train, eval_times=eval_times, sg_labels=sg_labels,
        bootstrap=bootstrap, n_bootstrap=n_bootstrap,
        bootstrap_ci=bootstrap_ci, random_state=random_state,
        _warn_stacklevel=_warn_stacklevel,
    )

    return _aggregate_evaluation_results(
        [("_models_", model_eval)] + ref_parts
    )


def _build_sample_sizes(model_names, reference_names, y_model, nonfinite_counts,
                        y_ref=None):
    """Build {name: n_valid} map for models and references.

    Parameters
    ----------
    y_model : structured array
        Outcome array for model evaluation (determines model sample size).
    y_ref : structured array, optional
        Outcome array for reference evaluation.  Defaults to *y_model*.
    """
    if y_ref is None:
        y_ref = y_model
    n_model = len(y_model)
    n_ref = len(y_ref)
    sizes = {mname: n_model for mname in model_names}
    for rname in reference_names:
        n_bad = nonfinite_counts.get(rname, 0)
        sizes[rname] = n_ref - n_bad
    return sizes


def _build_benchmark_adapters(
    reference_arrays, reference_arrays_test, hiw_map, hiw_default,
    split_eval, have_test_refs,
):
    """Build ReferenceAdapter dicts for evaluation and incremental metrics.

    Returns (adapters, adapters_test).
    """
    adapters = _build_reference_adapters(reference_arrays, hiw_map, hiw_default)
    adapters_test = (
        _build_reference_adapters(reference_arrays_test, hiw_map, hiw_default)
        if split_eval and have_test_refs
        else adapters
    )
    return adapters, adapters_test


def _run_benchmark_evaluation(
    resolved_model, adapters, reference_arrays, reference_names,
    X_data, y_data,
    *, X_test, y_test, X_train, y_train,
    eval_times, hiw_map, hiw_default,
    n_importance_repeats, bootstrap, n_bootstrap,
    bootstrap_ci, random_state,
    sg_labels, subpopulations_test,
    split_eval, nonfinite_counts,
):
    """Dispatch evaluation for models and references.

    Returns (eval_result, model_eval_or_None).
    """
    # user → benchmark → _run_benchmark_evaluation → evaluate
    warn_stacklevel = 5

    if split_eval:
        # warn → _resolve → _run_benchmark_evaluation → benchmark → user.
        sg_labels_test = _resolve_subpopulation_labels(
            subpopulations_test, y_test=y_test, _warn_stacklevel=4,
        )
        # Models on held-out test data; references on selected dataset
        model_eval = evaluate(
            resolved_model, X_test, y_test,
            X_train=X_train,
            y_train=y_train,
            eval_times=eval_times,
            n_importance_repeats=n_importance_repeats,
            bootstrap=bootstrap,
            n_bootstrap=n_bootstrap,
            bootstrap_ci=bootstrap_ci,
            include_predictions=bootstrap,
            random_state=random_state,
            show_progress=False,
            _warn_stacklevel=warn_stacklevel,
            subpopulation_labels=sg_labels_test,
        )
        ref_parts = _evaluate_references_on(
            X_data, y_data,
            reference_arrays=reference_arrays,
            adapters=adapters,
            hiw_map=hiw_map, hiw_default=hiw_default,
            y_train=y_train, eval_times=eval_times, sg_labels=sg_labels,
            bootstrap=bootstrap, n_bootstrap=n_bootstrap,
            bootstrap_ci=bootstrap_ci, random_state=random_state,
            _warn_stacklevel=warn_stacklevel,
        )
        eval_result = _aggregate_evaluation_results(
            [("_models_", model_eval)] + ref_parts
        )
        return eval_result, model_eval

    if any(n > 0 for n in nonfinite_counts.values()):
        # dataset == "test" with NaN references
        # _warn_stacklevel=6: user → benchmark → _run → _evaluate_with_nan → evaluate
        eval_result = _evaluate_with_nan_references(
            resolved_model, adapters, reference_arrays,
            X_data, y_data,
            X_train=X_train,
            y_train=y_train,
            eval_times=eval_times,
            hiw_map=hiw_map, hiw_default=hiw_default,
            n_importance_repeats=n_importance_repeats,
            bootstrap=bootstrap, n_bootstrap=n_bootstrap,
            bootstrap_ci=bootstrap_ci, random_state=random_state,
            sg_labels=sg_labels, _warn_stacklevel=6,
        )
        return eval_result, None

    # dataset == "test", no NaN — evaluate models together with default metrics,
    # then references one-at-a-time with metrics filtered per capability.
    model_eval = evaluate(
        resolved_model,
        X_data,
        y_data,
        X_train=X_train,
        y_train=y_train,
        eval_times=eval_times,
        n_importance_repeats=n_importance_repeats,
        bootstrap=bootstrap,
        n_bootstrap=n_bootstrap,
        bootstrap_ci=bootstrap_ci,
        include_predictions=bootstrap,
        random_state=random_state,
        show_progress=False,
        _warn_stacklevel=warn_stacklevel,
        subpopulation_labels=sg_labels,
    )
    ref_parts = _evaluate_references_on(
        X_data, y_data,
        reference_arrays=reference_arrays,
        adapters=adapters,
        hiw_map=hiw_map, hiw_default=hiw_default,
        y_train=y_train, eval_times=eval_times, sg_labels=sg_labels,
        bootstrap=bootstrap, n_bootstrap=n_bootstrap,
        bootstrap_ci=bootstrap_ci, random_state=random_state,
        _warn_stacklevel=warn_stacklevel,
    )
    eval_result = _aggregate_evaluation_results(
        [("_models_", model_eval)] + ref_parts
    )
    return eval_result, None


def _compute_post_eval_metrics(
    resolved_model, model_names, reference_names,
    adapters_test, reference_arrays_test,
    X_test, y_test, X_data, y_data, subpopulations_test, subpopulations_data,
    *, split_eval, have_test_refs,
    bootstrap, n_bootstrap, bootstrap_ci, random_state,
    nan_collector,
):
    """Compute incremental, subpopulation-incremental, and correlation metrics.

    Returns (incremental_df, subpopulation_incremental_df_or_None,
    correlation_df, model_risk_scores). ``model_risk_scores`` is the
    test-aligned raw prediction dict that is row-aligned with
    ``reference_arrays_test`` — callers persist it on the BenchmarkResult
    so scatter plots are reproducible from a saved learner.

    ``nan_collector`` is owned by the caller (``benchmark()``) so that
    accumulated NaN-drop records survive even if this helper raises
    partway through; the caller wraps the call in ``try/finally`` and
    emits one coalesced warning per metric.
    """
    use_test = split_eval and have_test_refs
    X_predict = X_test if use_test else X_data
    model_risk_scores = {
        mname: resolved_model[mname].predict(X_predict)
        for mname in model_names
    }

    y_incr = y_test if use_test else y_data
    incremental = pd.DataFrame(_build_incremental_rows(
        y_incr, model_risk_scores, adapters_test,
        model_names, reference_names,
        bootstrap=bootstrap, n_bootstrap=n_bootstrap,
        bootstrap_ci=bootstrap_ci, random_state=random_state,
        nan_collector=nan_collector,
    ))

    subpopulations_incr = subpopulations_test if use_test else subpopulations_data
    subpopulation_incremental_rows = _build_subpopulation_incremental_rows(
        subpopulations_incr, y_incr, model_risk_scores,
        reference_arrays_test, model_names, reference_names,
        bootstrap=bootstrap, n_bootstrap=n_bootstrap,
        bootstrap_ci=bootstrap_ci, random_state=random_state,
        logger=logger,
        nan_collector=nan_collector,
    )
    subpopulation_incremental = (
        pd.DataFrame(subpopulation_incremental_rows)
        if subpopulation_incremental_rows else None
    )

    correlation = _compute_correlations(
        model_risk_scores, reference_arrays_test,
        model_names, reference_names, logger=logger,
    )

    return (
        incremental, subpopulation_incremental, correlation,
        model_risk_scores,
    )


def _compute_benchmark_subpopulation_results(
    eval_result, model_eval,
    subpopulations_data, subpopulations_test,
    y_data, y_test, y_train,
    *, split_eval, eval_times, logger,
):
    """Compute subpopulation results from benchmark evaluation, or None."""
    # When split-evaluating, use model_eval (test-data aligned)
    # to avoid index mismatch from mixed model/reference data sizes.
    sg_eval = model_eval if model_eval is not None else eval_result
    sg_sub = subpopulations_test if split_eval else subpopulations_data
    sg_y = y_test if split_eval else y_data
    return _attach_subpopulation_results(
        sg_eval, sg_sub, sg_y, y_train,
        eval_times=eval_times, logger=logger,
    )


class BenchmarkingMixin:
    """Mixin adding ``benchmark()`` to SurvivalLearner."""

    def benchmark(
        self: SurvivalLearner,
        references: dict[str, np.ndarray],
        models=None,
        *,
        dataset: str = "test",
        eval_times=None,
        higher_is_worse: bool | dict[str, bool] = True,
        bootstrap: bool = True,
        n_bootstrap: int = 200,
        bootstrap_ci: float = 0.95,
        n_importance_repeats: int = 0,
        random_state: int | None = None,
        verbose: bool = True,
    ) -> BenchmarkResult:
        """Compare trained models against reference scores.

        Parameters
        ----------
        references : dict of {name: array-like}
            Reference values per patient.  Each array must be aligned
            with either ``X`` (full data — the appropriate portion is
            extracted automatically) or the selected dataset split.
        models : str, list, dict, ModelResult, or None, optional
            Model(s) to compare.  Same flexibility as ``evaluate()``:
            ``None`` → all trained models, str, dict, or ModelResult.
        dataset : {'test', 'train', 'full'}, default 'test'
            Which data split to evaluate **references** on:

            - ``'test'`` — held-out test set (default).  Both models
              and references use the same data.
            - ``'train'`` — references evaluated on training set for
              more observations.  Models still use held-out test set.
            - ``'full'`` — references evaluated on entire dataset for
              maximum statistical power.  Models still use held-out
              test set.

            Incremental metrics and correlations always use test data
            for both models and references (they require matched
            observations).
        eval_times : array-like or "auto", optional
            Timepoints for time-dependent metrics (AUC(t), Brier score).
            Pass ``"auto"`` to clear any inherited eval_times and force
            auto-computation from the data.
            When None, inherits from the most recent ``evaluate()`` or
            ``run()`` call if those used user-specified eval_times.
            Falls back to auto-computing from the data.
        higher_is_worse : bool or dict of {name: bool}, default True
            Whether higher reference values indicate worse outcome.
            Pass a dict to set per-reference (e.g.
            ``{'psa': True, 'albumin': False}``).
        bootstrap : bool
            Compute bootstrap CIs for evaluation and incremental metrics.
        n_bootstrap : int
            Number of bootstrap iterations.
        bootstrap_ci : float
            Confidence level for bootstrap intervals.
        n_importance_repeats : int
            Permutation importance repeats (0 = skip).
        random_state : int, optional
            Seed for reproducibility.  Falls back to ``self.random_state``.
        verbose : bool
            Print summary tables.

        Returns
        -------
        BenchmarkResult

        Raises
        ------
        ValueError
            If the ``references`` dict is empty.
        """
        self._ensure_setup()
        random_state = random_state if random_state is not None else self.random_state

        # ---- Resolve dataset ----
        ctx = self._resolve_benchmark_data(dataset)

        # Resolve eval_times through the canonical resolver: "auto" clears >
        # explicit array > inherited > auto. Models are always evaluated on the
        # test cohort here, so an inherited grid is appropriate — no custom-data
        # risk check needed.
        eval_times = resolve_state_eval_times(self._state, eval_times)

        # ---- Resolve model input ----
        resolved_model, _, _ = self._resolve_model_input(models)
        if not isinstance(resolved_model, dict):
            name = getattr(resolved_model, 'name', 'model')
            resolved_model = {name: resolved_model}
        model_names = list(resolved_model.keys())

        # ---- Resolve & validate references ----
        if not references:
            raise ValueError(
                "references dict is empty. Provide at least one reference "
                "as {name: array} to benchmark against."
            )
        reference_arrays, nonfinite_counts = self._resolve_reference_arrays(
            references, ctx.X,
        )
        reference_names = list(reference_arrays.keys())
        self._validate_reference_names(reference_names, model_names)

        # Resolve test-split references for incremental metrics
        split_eval = dataset != "test"
        if split_eval:
            reference_arrays_test, have_test_refs = (
                self._resolve_test_references(references, reference_arrays)
            )
        else:
            reference_arrays_test, have_test_refs = reference_arrays, True

        # ---- Warn if reference names match feature columns ----
        feature_overlap = set(reference_names) & set(ctx.X.columns)
        if feature_overlap:
            # Dedupe by overlap signature, sharing the set with the
            # validate() check — same leakage concern at train- or
            # validate-time.
            sig = frozenset(feature_overlap)
            if sig not in self._state.reference_overlap_warned:
                warnings.warn(
                    f"Reference names match feature columns in X: "
                    f"{sorted(feature_overlap)}. Verify that reference scores "
                    f"were not accidentally included as training features "
                    f"(data leakage).",
                    SuggestionWarning,
                    stacklevel=2,
                )
                self._state.reference_overlap_warned.add(sig)

        # ---- Build adapters ----
        hiw_map, hiw_default = _resolve_higher_is_worse(higher_is_worse)
        adapters, adapters_test = _build_benchmark_adapters(
            reference_arrays, reference_arrays_test,
            hiw_map, hiw_default, split_eval, have_test_refs,
        )

        # ---- Sample sizes + verbose reporting ----
        # Models always use test data; references use the selected dataset.
        y_model = self._state.y_test if split_eval else ctx.y
        sample_sizes = _build_sample_sizes(
            model_names, reference_names,
            y_model=y_model, nonfinite_counts=nonfinite_counts,
            y_ref=ctx.y,
        )
        if verbose:
            _report_nonfinite_references(reference_arrays, nonfinite_counts)

        sg_labels = _resolve_subpopulation_labels(
            ctx.subpopulations, y_test=ctx.y,
        )

        # ---- Evaluation dispatch ----
        eval_result, model_eval = _run_benchmark_evaluation(
            resolved_model, adapters, reference_arrays, reference_names,
            ctx.X, ctx.y,
            X_test=self._state.X_test, y_test=self._state.y_test,
            X_train=self._state.X_train, y_train=self._state.y_train,
            eval_times=eval_times,
            hiw_map=hiw_map, hiw_default=hiw_default,
            n_importance_repeats=n_importance_repeats,
            bootstrap=bootstrap, n_bootstrap=n_bootstrap,
            bootstrap_ci=bootstrap_ci, random_state=random_state,
            sg_labels=sg_labels,
            subpopulations_test=self._state.subpopulations_test,
            split_eval=split_eval, nonfinite_counts=nonfinite_counts,
        )

        # ---- Univariate Cox on references ----
        univariate_cox = compute_reference_univariate_cox(
            pd.DataFrame(reference_arrays), ctx.y,
        )

        # ---- Incremental + subpopulation-incremental + correlation ----
        # Own the collector at this level so partial accumulation survives
        # if the metric helpers raise mid-way.  ``try/finally``
        # guarantees the coalesced summary fires regardless.  stacklevel=3
        # routes the warning to the user's benchmark(...) call site
        # (user → benchmark → _emit_coalesced_incremental_nan_warning),
        # verified empirically.
        incremental_nan_records: list = []
        primary_exc_in_flight = False
        try:
            (
                incremental, subpopulation_incremental, correlation,
                model_risk_scores,
            ) = _compute_post_eval_metrics(
                resolved_model, model_names, reference_names,
                adapters_test, reference_arrays_test,
                self._state.X_test, self._state.y_test,
                ctx.X, ctx.y,
                self._state.subpopulations_test, ctx.subpopulations,
                split_eval=split_eval, have_test_refs=have_test_refs,
                bootstrap=bootstrap, n_bootstrap=n_bootstrap,
                bootstrap_ci=bootstrap_ci, random_state=random_state,
                nan_collector=incremental_nan_records,
            )
        except BaseException:
            primary_exc_in_flight = True
            raise
        finally:
            try:
                _emit_coalesced_incremental_nan_warning(
                    incremental_nan_records, stacklevel=3,
                )
            except Warning:
                # ``warnings.filterwarnings("error", ...)`` promoted our
                # coalesced MethodologicalWarning to an exception (a
                # plausible strict-CI configuration).  If a primary
                # exception from _compute_post_eval_metrics is already
                # propagating, swallow this secondary one so the original
                # traceback survives; otherwise re-raise to surface the
                # warning-as-error to the user.
                if not primary_exc_in_flight:
                    raise

        # ---- Package result ----
        result = BenchmarkResult(
            evaluation=eval_result,
            univariate_cox=univariate_cox,
            incremental=incremental,
            correlation=correlation,
            reference_names=reference_names,
            model_names=model_names,
            subpopulation_incremental=subpopulation_incremental,
            sample_sizes=sample_sizes,
            dataset=dataset,
            model_risk_scores=model_risk_scores,
            reference_arrays=dict(reference_arrays_test),
        )

        self._warn_if_overwriting('benchmark_result', 'benchmark result')
        self._state.benchmark_result = result

        # Populate subpopulation_results from benchmark evaluation
        if ctx.subpopulations:
            sg_results = _compute_benchmark_subpopulation_results(
                eval_result, model_eval,
                ctx.subpopulations, self._state.subpopulations_test,
                ctx.y, self._state.y_test, self._state.y_train,
                split_eval=split_eval, eval_times=eval_times,
                logger=logger,
            )
            if sg_results is not None:
                # Merge into existing state rather than overwriting, so that
                # data from a prior learner.run() (e.g. auc_t_bootstrap CI
                # ribbons) is preserved when the new benchmark results
                # don't populate the same fields.
                from mlsurv.evaluation._subpopulation import (
                    _merge_subpopulation_results_dict,
                )
                self._state.subpopulation_results = (
                    _merge_subpopulation_results_dict(
                        self._state.subpopulation_results, sg_results,
                    )
                )

        if verbose:
            result.print_summary(ci_level=bootstrap_ci)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_benchmark_data(
        self: SurvivalLearner, dataset: str,
    ) -> _BenchmarkData:
        """Resolve X, y, subpopulations for the given dataset split."""
        if dataset not in _VALID_DATASETS:
            raise ValueError(
                f"dataset must be one of {_VALID_DATASETS}, got {dataset!r}"
            )

        if dataset == "test":
            X_data = self._state.X_test
            y_data = self._state.y_test
            subpopulations_data = self._state.subpopulations_test
        elif dataset == "train":
            X_data = self._state.X_train
            y_data = self._state.y_train
            subpopulations_data = self._state.subpopulations_train
        else:  # "full"
            if self._state.presplit:
                raise ValueError(
                    "dataset='full' is not supported with pre-split data "
                    "because no combined dataset exists. Use 'test' or "
                    "'train', or provide unsplit (X, y) data."
                )
            X_data = self._state.X
            y_data = self._state.y
            subpopulations_data = self._state.subpopulations

        if X_data is None or y_data is None:
            raise ValueError(
                f"No data available for dataset={dataset!r}. "
                f"Ensure setup() was called with the appropriate data."
            )

        if dataset != "test":
            label = "training" if dataset == "train" else "training + test"
            warnings.warn(
                f"Benchmarking on dataset={dataset!r}: references are "
                f"evaluated on {label} data for greater statistical power. "
                f"Models use held-out test data. Incremental metrics and "
                f"correlations use test data for both.",
                MethodologicalWarning,
                stacklevel=3,
            )

        return _BenchmarkData(X_data, y_data, subpopulations_data)

    def _resolve_test_references(
        self: SurvivalLearner,
        references: dict,
        reference_arrays: dict,
    ) -> tuple[dict, bool]:
        """Resolve reference arrays against the test split.

        Returns (reference_arrays_test, have_test_refs).  Falls back to
        *reference_arrays* when alignment to test data fails.
        """
        try:
            reference_arrays_test, _ = self._resolve_reference_arrays(
                references, self._state.X_test,
            )
            return reference_arrays_test, True
        except ValueError:
            # References cannot be aligned to test data (e.g. train-
            # length arrays with dataset="train").  Fall back to using
            # the selected dataset for incremental/correlation.
            return reference_arrays, False

    def _resolve_reference_arrays(
        self: SurvivalLearner,
        references: dict[str, np.ndarray],
        X_subset: pd.DataFrame,
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        """Resolve reference arrays, extracting the appropriate portion if needed.

        Accepts arrays aligned with either ``X_subset`` (the selected
        dataset split) or full ``X`` (automatically extracted via index
        alignment).

        Returns
        -------
        resolved : dict of {name: array}
            Reference arrays aligned with X_subset (may contain NaN).
        nonfinite_counts : dict of {name: int}
            Number of non-finite values (NaN + Inf) per reference (0 if clean).
        """
        n_full = len(self._state.X) if self._state.X is not None else 0
        n_subset = len(X_subset)

        resolved = {}
        nonfinite_counts = {}
        for name, arr in references.items():
            arr = np.asarray(arr, dtype=np.float64).ravel()

            if len(arr) == n_subset:
                resolved[name] = arr
            elif len(arr) == n_full and n_full != n_subset:
                # Extract subset portion via positional index alignment
                positions = self._state.X.index.get_indexer(X_subset.index)
                if (positions == -1).any():
                    n_missing = int((positions == -1).sum())
                    raise ValueError(
                        f"Reference '{name}' has length {len(arr)} matching "
                        f"full X, but {n_missing} indices could not be "
                        f"located in X.index. Provide reference array "
                        f"aligned with the selected dataset "
                        f"({n_subset} samples) instead."
                    )
                resolved[name] = arr[positions]
            else:
                raise ValueError(
                    f"Reference '{name}' has length {len(arr)}, expected "
                    f"{n_subset} (selected dataset) or {n_full} (full X)."
                )

            # Count non-finite values and check for all-non-finite
            ref_arr = resolved[name]
            n_inf = int(np.sum(np.isinf(ref_arr)))
            n_nonfinite = int(np.sum(~np.isfinite(ref_arr)))
            nonfinite_counts[name] = n_nonfinite

            if n_nonfinite == len(ref_arr):
                n_nan = n_nonfinite - n_inf
                raise ValueError(
                    f"Reference '{name}' has no finite values "
                    f"({n_nan} NaN, {n_inf} Inf out of {len(resolved[name])}). "
                    f"Cannot compute metrics."
                )

            if n_inf > 0:
                warnings.warn(
                    f"Reference '{name}' has {n_inf} infinite values. "
                    f"These will be excluded from metric computation.",
                    MethodologicalWarning,
                    stacklevel=3,
                )

        return resolved, nonfinite_counts

    @staticmethod
    def _validate_reference_names(reference_names, model_names):
        """Ensure reference names don't collide with model names."""
        overlap = set(reference_names) & set(model_names)
        if overlap:
            raise ValueError(
                f"Reference names overlap with model names: {overlap}. "
                "Use different names to avoid ambiguity."
            )
