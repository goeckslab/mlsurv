"""CoreEvaluationMixin — evaluate() and validate() for the SurvivalLearner."""

from __future__ import annotations

import itertools
import warnings
from typing import Any, TYPE_CHECKING

import numpy as np

from .._shared import try_verbose_print
from .._warnings import InternalWarning, SuggestionWarning, _warn_eval_times_failure
from ..evaluation import evaluate as _evaluate
from ..evaluation.evaluate import (
    _aggregate_evaluation_results,
    _assess_calibration_drift,
)
from ._references import (
    _build_reference_adapters,
    _evaluate_references_on,
    _report_nonfinite_references,
    _resolve_higher_is_worse,
    _resolve_validation_reference_arrays,
)
from ..results.containers import EvaluationResult, ValidationResult
from ..utils import auto_clip_union_eval_times, check_truncation_needed
from ._format import (
    _print_param_box,
    _format_model_display,
    _format_state_overrides,
    _format_eval_times_row,
    _format_importance_row,
    _format_bootstrap_row,
)
from ..logging_utils import get_logger
from ._eval_context import EvalContext
from ._evaluation_helpers import (
    _IPCW_TAU_AUTO,
    _SUBPOP_LABELS_UNSET,
    _align_columns,
    _compute_and_store_subpopulation_metrics,
    _emit_heavy_censoring_warning,
    _emit_ipcw_truncation_warning,
    _print_evaluation_output,
    _resolve_eval_model_names,
    _resolve_ipcw_g_source,
    _resolve_ipcw_tau,
    _resolve_subpopulation_labels,
    _resolve_validate_subpopulation_override,
    _update_kang_cache,
    resolve_state_eval_times,
)

logger = get_logger("learner")

if TYPE_CHECKING:
    import pandas as pd

    from mlsurv.learner._core import SurvivalLearner


_INTERACTION_FIELDS = (
    'interactions',
    'interaction_surfaces',
    'interaction_values_raw',
    'interaction_feature_names',
    'interaction_method',
    '_interaction_raw_cache',
)


def _preserve_interactions_on_overwrite(
    old_eval, new_eval, *, include_interactions_requested: bool,
) -> None:
    """Copy interaction fields from ``old_eval`` onto ``new_eval`` in place
    when the caller did NOT request fresh interactions.

    vignette_07 calls ``learner.evaluate()`` after attaching a calibrator
    to refresh test-set metrics. Without preservation, the fresh evaluation's
    ``interactions=None`` overwrote interactions populated by an earlier
    ``run(..., include_interactions=True)``, breaking the next-cell
    ``plot_interactions()`` call.

    Gating on the caller's *intent* (``include_interactions=False``) — not on
    whether the new result happens to contain interactions — avoids silently
    masking a legitimate empty result from an explicit
    ``evaluate(include_interactions=True)`` call (e.g. <2 features survive
    selection, or a SHAP exception). In that case the user asked for fresh
    interactions and should see what was actually computed.
    """
    if include_interactions_requested:
        return
    if old_eval is None or new_eval is None:
        return
    if getattr(old_eval, 'interactions', None) is None:
        return
    for field_name in _INTERACTION_FIELDS:
        setattr(new_eval, field_name, getattr(old_eval, field_name, None))


def _concordance_flags_drift(model, concordance_df, model_name) -> bool:
    """Whether ``model``'s evaluation tripped the calibration-drift thresholds.

    Thin wrapper over the pure :func:`_assess_calibration_drift` (which reads
    ``cal_slope_test`` / ``gnd_test`` from the one-row ``concordance_df``) that
    tolerates a ``None`` frame. Used to stamp ``ModelResult.calibration_flagged``
    at evaluate()-store time so the ``calibration_flagged_models`` property is a
    cheap read rather than a per-call re-derivation.
    """
    if concordance_df is None:
        return False
    return _assess_calibration_drift(model, concordance_df, model_name) is not None


class CoreEvaluationMixin:
    """Mixin for ``evaluate()`` and ``validate()``."""

    def _store_precalibration_slopes(
        self: SurvivalLearner, X_test, y_test, eval_times,
    ) -> None:
        """Store the raw central-horizon test calibration slope per calibrated model.

        Cheap single-track diagnostic (one predict + one slope fit per model)
        so the report can render the before -> after; the headline metrics
        stay calibrated. Failures leave ``precalibration_slope`` as ``None``.
        """
        from ..evaluation.metrics import (
            calibration_slope_intercept, predict_survival_functions,
            surv_funcs_to_probs,
        )
        if eval_times is None:
            return
        times = np.asarray(eval_times, dtype=float)
        if times.size == 0:
            return
        jc = times.size // 2
        hc = float(times[jc])
        for name in list(self._state.models):
            if not self._has_calibrator(name):
                continue
            result = self._state.models.get(name)
            if result is None:
                continue
            try:
                raw = np.asarray(
                    surv_funcs_to_probs(
                        predict_survival_functions(result.model, X_test, times),
                        times,
                    ),
                    dtype=float,
                )
                slope, _ = calibration_slope_intercept(y_test, raw[:, jc], hc)
                result.precalibration_slope = (
                    float(slope) if np.isfinite(slope) else None
                )
            except (ValueError, RuntimeError, IndexError) as exc:
                logger.debug("precalibration slope for %s failed: %s", name, exc)
                result.precalibration_slope = None

    def evaluate(
        self: SurvivalLearner,
        models: Any = None,
        X: pd.DataFrame | None = None,
        y: np.ndarray | None = None,
        eval_times: np.ndarray | list | str | None = None,
        n_importance_repeats: int | str | dict = "auto",
        importance_max_samples: int | None = None,
        include_shap: bool = False,
        shap_max_background: int = 50,
        shap_max_explain: int = 50,
        custom_metrics: dict | None = None,
        random_state: int | None = None,
        show_progress: bool = True,
        verbose: bool = True,
        n_jobs: int | None = None,
        # Bootstrap parameters
        bootstrap: bool = False,
        n_bootstrap: int | str = "auto",
        bootstrap_ci: float = 0.95,
        bootstrap_metrics: list | None = None,
        bootstrap_stratify: bool = False,
        # Bootstrap predictions (for ROC CI bands)
        include_predictions: bool = False,
        # Feature interactions
        include_interactions: bool = False,
        interaction_top_k: int = 5,
        interaction_max_samples: int = 100,
        interaction_method: str = 'hstat',
        interaction_scale: str = 'auto',
        # IPCW concordance horizon
        ipcw_tau: float | str = _IPCW_TAU_AUTO,
        # IPCW G_hat fitting cohort: 'train' for in-distribution
        # evaluation, 'test' for external validation. validate() overrides
        # the default to 'test' since exchangeability fails by construction.
        ipcw_g_hat_source: str = "train",
        # GND calibration test bin count
        gnd_n_bins: int = 10,
        # Internal parameter - not part of public API
        _store_result=True,
        _subpopulation_labels_override=_SUBPOP_LABELS_UNSET,
        _warn_stacklevel=3,
        _cohort_name="main",
        _surv_probs_transforms=None,
        # Forwarded to the free evaluate() to suppress the calibration-drift
        # summary / unused-parameter warnings on internal bookkeeping passes.
        _skip_drift_warning=False,
        _skip_param_warnings=False,
        # Held-out-test concordance DataFrame. validate() passes this so the
        # verbose metrics table annotates each cell with its change (Δ) from
        # the development metrics; plain evaluate() leaves it None.
        _delta_baseline=None,
    ) -> EvaluationResult:
        """
        Evaluate model(s) on the test set.

        Parameters
        ----------
        models : BaseSurvivalModel, Pipeline, ModelResult, dict, or None
            Model(s) to evaluate. Accepts:
            - Single model or ModelResult: evaluates that model
            - dict of {name: model/ModelResult}: evaluates all models in dict
            - None: evaluates all trained models
        X : DataFrame, optional
            Test features. Uses stored X_test if None.
        y : SurvivalArray, optional
            Test outcomes. Uses stored y_test if None.
        eval_times : array-like or "auto", optional
            Timepoints for AUC(t).  Pass ``"auto"`` to clear any inherited
            eval_times and force auto-computation from the data.
        n_importance_repeats : int, "auto", or dict, default="auto"
            Number of permutation importance repeats.  When ``"auto"``
            (the default), uses 10 repeats for all models.
            Pass a dict of {model_name: int} for per-model repeat counts;
            models not in the dict fall back to ``"auto"``.
        importance_max_samples : int, optional
            Cap on the test rows used for permutation importance. ``None``
            (the default) uses every row. Permutation importance costs
            ``n_features x n_repeats x (predict + C-index over n_test
            rows)``, and the C-index term is superlinear in ``n_test``, so
            on large cohorts importance can out-cost model fitting. A value
            at or above the cohort size is a no-op; the minimum is 2. The
            draw is seeded from
            *random_state*, so with a seed set every model sees the same
            rows and their importances stay comparable; without one, each
            model draws its own subsample and mlsurv warns.
        include_shap : bool, default=False
            If True, compute SHAP values.
        shap_max_background : int, default=50
            Maximum background samples for the SHAP masker/explainer.
        shap_max_explain : int, default=50
            Maximum test samples to explain with SHAP.
        custom_metrics : dict, optional
            Additional metrics to compute.
        random_state : int, optional
            Seed for reproducibility.
        show_progress : bool, default=True
            Show progress messages during permutation importance calculation.
        verbose : bool, default=True
            If True, print metrics table after evaluation.
        n_jobs : int, optional
            Number of parallel jobs. If None, uses the value from setup().
        bootstrap : bool, default=False
            If True, run bootstrap evaluation to compute confidence intervals.
        n_bootstrap : int or "auto", default="auto"
            Number of bootstrap iterations. If "auto", computed as
            min(500, max(100, 10 * sqrt(n_samples))).
        bootstrap_ci : float, default=0.95
            Confidence level for bootstrap intervals (only used if bootstrap=True).
        bootstrap_metrics : list of str, optional
            Metrics to compute during bootstrap. Default: all available metrics.
            Options: 'harrell', 'ipcw', 'integrated_brier', 'ctd'.
        bootstrap_stratify : bool, default=False
            If True, stratify bootstrap samples by event status.
        include_predictions : bool, default=False
            Store per-iteration risk scores for ROC CI bands. Only used
            when bootstrap=True.
        include_interactions : bool, default=False
            If True, compute pairwise feature interactions.
        interaction_top_k : int, default=5
            Number of top features (by permutation importance) to test.
            Produces ``C(top_k, 2)`` pairs.
        interaction_max_samples : int, default=100
            Maximum samples for interaction computation.
        interaction_method : str, default='hstat'
            Interaction method: ``'hstat'`` (Friedman's H-statistic,
            model-agnostic) or ``'shap'`` (SHAP interaction values via
            TreeExplainer, XGBoost models only).
        interaction_scale : str, default='auto'
            Axis spacing for the report's 2D interaction PDP surfaces:
            ``'auto'`` picks ``'linear'``/``'log'``/``'log1p'`` per axis from
            each feature's right-skew; pass ``'linear'`` to keep all axes
            linear, or ``'log'``/``'log1p'`` to force a spacing everywhere.
            Negative-valued and categorical axes always stay linear.
        ipcw_tau : float or "auto", optional
            Evaluation horizon τ for Uno's IPCW concordance, passed through to
            :func:`sksurv.metrics.concordance_index_ipcw`. When unset, a
            conservative default is auto-selected (see
            :func:`mlsurv.evaluation.metrics.compute_safe_ipcw_tau`) to avoid
            unstable estimates when test follow-up extends beyond training
            censoring support.
        ipcw_g_hat_source : {'train', 'test'}, default='train'
            Cohort whose censoring distribution feeds Uno's IPCW G_hat
            estimator. ``'train'`` (the default for in-distribution
            evaluation) is consistent under exchangeability between train
            and test. For external validation use ``validate()``, which
            defaults this to ``'test'`` — Uno's ``1/G(T)^2`` weight is
            consistent for the population whose pairs are evaluated, and
            using ``y_test`` introduces no leakage (G uses censoring times
            only).
        gnd_n_bins : int, default=10
            Number of equal-frequency bins for the Greenwood-Nam-D'Agostino
            (GND) calibration test reported in the concordance table. The test
            requires >=5 samples per bin, so it returns NaN when
            ``len(X_test) < 5 * gnd_n_bins`` (the default 10 bins needs N>=50).
            On a small external cohort, lower this (keep it <= N/5) to allow
            GND to be computed at the cost of coarser calibration granularity;
            very small or tied-prediction cohorts may still return NaN if fewer
            than 2 bins remain usable. Must be an integer >= 2. The default of
            10 follows Demler et al. 2015.

        Returns
        -------
        EvaluationResult
            Complete evaluation results (with bootstrap_results if bootstrap=True).

        Raises
        ------
        TypeError
            If ``eval_times`` is a scalar rather than an array-like of
            timepoints.
        ValueError
            If ``gnd_n_bins`` is not an integer >= 2, or if ``eval_times``
            is not one-dimensional.

        Notes
        -----
        IPCW-weighted metrics (Uno C, AUC(t), IBS) are computed on test
        observations within the training time range; observations beyond
        ``train_max`` are administratively censored. Harrell's C and
        calibration plots use the full test set. See
        ``docs/evaluation.md#training-range-cap``.

        All concordance, Brier/IBS, and AUC(t) metrics assume censoring is
        independent of the event given the covariates. Informative censoring
        biases them and cannot be detected internally. See the
        ``SurvivalLearner`` class docstring for the full list of modeling
        assumptions.

        Examples
        --------
        >>> # Evaluate all trained models
        >>> result = learner.evaluate()
        >>>
        >>> # Evaluate models from run()
        >>> results = learner.run()
        >>> result = learner.evaluate(results)
        >>>
        >>> # Per-model importance repeats
        >>> result = learner.evaluate(n_importance_repeats={'coxph': 50, 'rsf': 10})
        """
        self._ensure_setup()

        # A GND test needs >=2 bins to assess calibration (a single
        # bin is degenerate). Reject sub-2 / non-integer values at the public
        # boundary with a clear ValueError — downstream the value is consumed
        # inside a try/except that would otherwise silently coerce a bad
        # n_bins into a NaN GND statistic.
        if not isinstance(gnd_n_bins, (int, np.integer)) or gnd_n_bins < 2:
            raise ValueError(
                f"gnd_n_bins must be an integer >= 2, got {gnd_n_bins!r}."
            )

        # Resolve eval_times through the canonical resolver: "auto" clears >
        # explicit array > inherited > auto. Persists explicit/cleared grids to
        # user_eval_times so bootstrap()/calibration reuse the same grid.
        eval_times = resolve_state_eval_times(
            self._state, eval_times,
            custom_data=(X is not None or y is not None),
            y_for_risk=y,
            # The inherited-eval_times warning fires inside
            # resolve_state_eval_times, one frame below evaluate -- the same
            # depth as evaluate's other helper-emitted warnings, so it takes
            # _warn_stacklevel verbatim (no -1). Chain (stacklevel counts from
            # the resolver outward): resolver=1 -> evaluate=2 -> user=3 on the
            # direct path; resolver=1 -> evaluate=2 -> validate=3 -> user=4 on
            # the validate path (which passes _warn_stacklevel=4).
            stacklevel=_warn_stacklevel,
        )

        # Only warn when this call will actually overwrite stored evaluation;
        # ``_store_result=False`` paths (e.g. ``validate()`` — which runs the
        # core twice on the recalibrated path) leave per-model
        # ``ModelResult.evaluation`` untouched so the warning is spurious.
        if _store_result:
            target_names = _resolve_eval_model_names(self, models)
            self._warn_if_overwriting_model_data(
                'evaluation', 'evaluation result', target_names,
            )

        # Capture original args before fallback (for override detection)
        orig_random_state = random_state
        orig_n_jobs = n_jobs

        # Use self.random_state as fallback for reproducibility
        random_state = random_state if random_state is not None else self.random_state
        n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        X_test = X if X is not None else self.X_test
        y_test = y if y is not None else self.y_test

        # Bundle the threaded evaluation params into a single
        # EvalContext so helpers and the summary box share one source of truth.
        ctx = EvalContext(
            eval_times=eval_times,
            extra_eval_times=self._state.user_extra_eval_times,
            n_importance_repeats=n_importance_repeats,
            importance_max_samples=importance_max_samples,
            include_shap=include_shap,
            shap_max_background=shap_max_background,
            shap_max_explain=shap_max_explain,
            custom_metrics=custom_metrics,
            bootstrap=bootstrap,
            n_bootstrap=n_bootstrap,
            bootstrap_ci=bootstrap_ci,
            bootstrap_metrics=bootstrap_metrics,
            bootstrap_stratify=bootstrap_stratify,
            include_predictions=include_predictions,
            include_interactions=include_interactions,
            interaction_top_k=interaction_top_k,
            interaction_max_samples=interaction_max_samples,
            interaction_method=interaction_method,
            interaction_scale=interaction_scale,
            ipcw_tau=ipcw_tau,
            ipcw_g_hat_source=ipcw_g_hat_source,
            random_state=random_state,
            n_jobs=n_jobs,
            show_progress=show_progress,
            verbose=verbose,
        )

        # Resolve IPCW tau first so the once-per-cohort truncation
        # warning can include it in a single consolidated message.
        # ``y_for_g`` picks the cohort whose censoring distribution feeds
        # G_hat; the truncation cap itself always uses ``self.y_train``.
        y_for_g = _resolve_ipcw_g_source(
            ctx.ipcw_g_hat_source, self.y_train, y_test,
        )
        resolved_ipcw_tau, ipcw_support_flag = _resolve_ipcw_tau(
            y_for_g, y_test, ctx.ipcw_tau,
            metrics=ctx.bootstrap_metrics if ctx.bootstrap else None,
            context="evaluate",
            _warn_stacklevel=_warn_stacklevel,
        )
        if ipcw_support_flag is not None:
            ipcw_support_flag["g_hat_source"] = ctx.ipcw_g_hat_source

        _trunc_t_max, _trunc_n_beyond = _emit_ipcw_truncation_warning(
            self._state, _cohort_name, self.y_train, y_test,
            tau=resolved_ipcw_tau,
            stacklevel=_warn_stacklevel,
        )

        # Heavy-censoring guard on the headline Harrell c_index_test: warn once
        # per cohort when the eval cohort is heavily censored, pointing to
        # Uno's IPCW C (c_index_ipcw_test).
        _emit_heavy_censoring_warning(
            self._state, _cohort_name, y_test,
            stacklevel=_warn_stacklevel,
        )

        # When user passes custom X, validate columns against training data
        if X is not None:
            X_test = _align_columns(
                X_test, self.X_train.columns.tolist(), context="Test data",
            )

        resolved_model, cv_model_order, original_model_result = self._resolve_model_input(models)

        # Resolve eval_times once so the verbose box can display the auto-picked
        # times and downstream evaluate() reuses the same array.
        # On failure, log and fall back to None — downstream evaluate() also
        # wraps its own auto_eval_times call and emits the user-facing
        # SuggestionWarning, so we don't double-warn here.
        eval_times_was_auto = eval_times is None
        if eval_times_was_auto:
            # Reflect registered report horizons in the displayed grid; the
            # engine unions them again (idempotently) for the computation. The
            # viz calibration surfaces materialize the same grid via the shared
            # helper so they land on the same horizons.
            eval_times = auto_clip_union_eval_times(
                y_test, self.y_train, self._state.user_extra_eval_times,
            )
            if eval_times is None:
                # auto_eval_times failed; downstream evaluate() also wraps its
                # own call and emits the user-facing SuggestionWarning, so we
                # don't double-warn here — just skip the verbose-box display.
                eval_times_was_auto = False

        # Print evaluate summary box (cosmetic — never abort evaluation on failure)
        if verbose:
            try_verbose_print(
                self._print_evaluate_summary,
                ctx,
                resolved_model=resolved_model,
                eval_times=eval_times,
                eval_times_was_auto=eval_times_was_auto,
                orig_random_state=orig_random_state,
                orig_n_jobs=orig_n_jobs,
                y_test=y_test,
                context="Evaluation summary",
                detail="All parameters are active.",
            )

        _sg_labels = _resolve_subpopulation_labels(
            self._state.subpopulations_test,
            bootstrap=ctx.bootstrap,
            y_test=y_test,
            override=_subpopulation_labels_override,
            _warn_stacklevel=_warn_stacklevel,
        )

        # Single-track calibration: when the caller did not pass explicit
        # transforms, build them from attached internal calibrators so evaluate
        # (and any in-call bootstrap, which reuses the same transforms) reports
        # the calibrated, deployed S(t|x). The closure binds the evaluate
        # cohort X so the score->baseline calibrator can recompute from
        # pipeline.predict(X). Probability-warpers ignore X. This applies on
        # every cohort, including external validate() cohorts: the calibrator
        # is a fixed dev-cohort transform the external rows never saw, so it is
        # leakage-free.
        if _surv_probs_transforms is None:
            calibrated_names = [
                n for n in self._state.models if self._has_calibrator(n)
            ]
            if calibrated_names:
                _surv_probs_transforms = {
                    n: (lambda sp, et, _n=n, _X=X_test:
                        self._apply_calibrator(_n, sp, et, X=_X))
                    for n in calibrated_names
                }

        eval_params = dict(
            X_train=self.X_train,
            y_train=self.y_train,
            # IPCW G_hat fitting cohort. When None, downstream falls
            # back to y_train (in-distribution exchangeability assumption).
            ipcw_y_train=(y_for_g if ctx.ipcw_g_hat_source == "test" else None),
            eval_times=eval_times,
            extra_eval_times=ctx.extra_eval_times,
            n_importance_repeats=ctx.n_importance_repeats,
            importance_max_samples=ctx.importance_max_samples,
            include_shap=ctx.include_shap,
            shap_max_background=ctx.shap_max_background,
            shap_max_explain=ctx.shap_max_explain,
            random_state=ctx.random_state,
            show_progress=ctx.show_progress,
            bootstrap=ctx.bootstrap,
            n_bootstrap=ctx.n_bootstrap,
            bootstrap_ci=ctx.bootstrap_ci,
            bootstrap_metrics=ctx.bootstrap_metrics,
            bootstrap_stratify=ctx.bootstrap_stratify,
            include_predictions=ctx.include_predictions,
            n_jobs=ctx.n_jobs,
            # `mlsurv.evaluation.evaluate` warnings fire two frames deeper
            # than this resolver's warnings (extra dispatch helpers). The
            # `+2` offset preserves the original calibration (3 + 2 = 5
            # on the direct chain) and extends correctly to validate
            # (4 + 2 = 6). See `mlsurv.evaluation.evaluate._warn_stacklevel`.
            _warn_stacklevel=_warn_stacklevel + 2,
            subpopulation_labels=_sg_labels,
            categorical_groups=self._state.categorical_groups,
            categorical_features=self._state.categorical_features,
            include_interactions=ctx.include_interactions,
            interaction_top_k=ctx.interaction_top_k,
            interaction_max_samples=ctx.interaction_max_samples,
            interaction_method=ctx.interaction_method,
            interaction_scale=ctx.interaction_scale,
            ipcw_tau=resolved_ipcw_tau,
            gnd_n_bins=gnd_n_bins,
            _surv_probs_transforms=_surv_probs_transforms,
            # Route residual drift of internally-calibrated models to the
            # "try a different train(calibrate=...) method" advice.
            _calibrated_models=frozenset(
                n for n in self._state.models if self._has_calibrator(n)
            ),
            _skip_drift_warning=_skip_drift_warning,
            _skip_param_warnings=_skip_param_warnings,
        )

        result = self._dispatch_evaluate(
            resolved_model, X_test, y_test, eval_params,
            cv_model_order, original_model_result, ctx.custom_metrics,
            _store_result=_store_result,
            include_interactions_requested=ctx.include_interactions,
        )

        # Store the raw (pre-calibration) test slope for calibrated models so the
        # report can show before -> after. Only on the internal-test, main path
        # (skip user-supplied external cohorts).
        if _store_result and X is None and y is None:
            self._store_precalibration_slopes(X_test, y_test, eval_times)

        try:
            object.__setattr__(result, 'ipcw_support_flag', ipcw_support_flag)
        except (AttributeError, TypeError):
            pass
        # Always assign — None clears a stale flag from a prior violating call.
        self._state.ipcw_support_flag = ipcw_support_flag

        # Persist truncation diagnostic so report sections can render a
        # footnote on the metrics table.
        try:
            object.__setattr__(result, 'truncation_t_max', _trunc_t_max)
            object.__setattr__(result, 'truncation_n_beyond', _trunc_n_beyond)
        except (AttributeError, TypeError):
            pass

        if _store_result:
            # Store per-model evaluations on ModelResult (authoritative).
            # Pass include_interactions intent so the helper knows whether
            # the caller asked for fresh interactions or is OK preserving prior.
            self._store_per_model_evaluations(
                result, resolved_model, original_model_result,
                include_interactions_requested=ctx.include_interactions,
            )

            # Auto-compute subpopulation metrics from stored subpopulations
            if self._state.subpopulations_test:
                _compute_and_store_subpopulation_metrics(
                    self._state, result, y_test,
                    ipcw_y_train=(y_for_g if ctx.ipcw_g_hat_source == "test" else None),
                )

        _update_kang_cache(
            result, self._state, ctx.bootstrap, y_test, self._auto_kang_comparison,
        )

        if ctx.verbose:
            _print_evaluation_output(
                result, self._state, ctx.bootstrap_ci, self.scoring,
                delta_baseline=_delta_baseline,
            )
        return result

    def _dispatch_evaluate(self: SurvivalLearner, resolved_model, X_test, y_test,
                           eval_params, cv_model_order, original_model_result,
                           custom_metrics, *, _store_result=True,
                           include_interactions_requested: bool = False):
        """Run evaluate() for single or multi-model and attach results.

        Wraps NotFittedError with a user-friendly message suggesting
        train() or tune() before evaluate().
        """
        try:
            if isinstance(resolved_model, dict):
                result = _evaluate(resolved_model, X_test, y_test, **eval_params)
                if cv_model_order is not None:
                    result.model_order = cv_model_order
                return result

            model_name = original_model_result.name if original_model_result else None
            result = _evaluate(
                resolved_model, X_test, y_test,
                custom_metrics=custom_metrics, model_name=model_name,
                **eval_params,
            )
            # Skip the single-model write-back when the caller explicitly
            # opted out (e.g. validate() reuses evaluate() on an external
            # cohort and stores its own ValidationResult instead — writing
            # the cohort-scope predictions onto the trained-on-test
            # ModelResult.evaluation would silently corrupt downstream
            # plot_roc / get_survival_test_data).
            if _store_result and original_model_result is not None:
                _preserve_interactions_on_overwrite(
                    original_model_result.evaluation, result,
                    include_interactions_requested=include_interactions_requested,
                )
                original_model_result.evaluation = result
            return result
        except Exception as e:
            if isinstance(e, (MemoryError, RecursionError)):
                raise
            from sklearn.exceptions import NotFittedError
            if isinstance(e, NotFittedError):
                model_hint = ""
                if not isinstance(resolved_model, dict) and original_model_result:
                    model_hint = f" ('{original_model_result.name}')"
                raise NotFittedError(
                    f"Model{model_hint} not fitted. Call train() or tune() before evaluate(). "
                    f"Original error: {e}"
                ) from e
            raise

    def _store_per_model_evaluations(
        self: SurvivalLearner, result, resolved_model, original_model_result,
        *, include_interactions_requested: bool = False,
    ):
        """Store evaluation result on each ModelResult (authoritative per-model storage).

        For single-model evaluates, stores the result directly on the
        ModelResult.  For multi-model evaluates, splits the combined result
        into per-model views and stores each on its corresponding ModelResult.
        """
        from ..results.containers import extract_model_evaluation

        if isinstance(resolved_model, dict):
            # Multi-model: split combined result into per-model evaluations.
            # Iterate the input dict directly — its keys are the storage
            # keys passed to evaluate() (and stamped into result.concordance
            # as model_type), so they match _state.models. This avoids
            # round-tripping through result.concordance and silently dropping
            # entries when the two sets diverge.
            for name in resolved_model:
                mr = self._state.models.get(name)
                if mr is not None:
                    new_eval = extract_model_evaluation(result, name)
                    _preserve_interactions_on_overwrite(
                        mr.evaluation, new_eval,
                        include_interactions_requested=include_interactions_requested,
                    )
                    mr.evaluation = new_eval
                    # Stamp the calibration-drift flag from the concordance we
                    # just stored, so calibration_flagged_models is a cheap read.
                    mr.calibration_flagged = _concordance_flags_drift(
                        mr.model, new_eval.concordance, name,
                    )
                    # Hoist embedded bootstrap to top-level for unified access
                    if mr.evaluation.bootstrap is not None and mr.bootstrap is None:
                        mr.bootstrap = mr.evaluation.bootstrap
                else:
                    warnings.warn(
                        f"Model '{name}' from evaluate() input not found in "
                        f"learner.models; evaluation result was computed but "
                        f"not stored on any ModelResult.",
                        SuggestionWarning,
                        stacklevel=4,
                    )
        else:
            # Single-model: stamp the drift flag and hoist bootstrap if present
            if original_model_result is not None:
                eval_r = original_model_result.evaluation
                if eval_r is not None:
                    original_model_result.calibration_flagged = _concordance_flags_drift(
                        original_model_result.model, eval_r.concordance,
                        original_model_result.name,
                    )
                    if eval_r.bootstrap is not None and original_model_result.bootstrap is None:
                        original_model_result.bootstrap = eval_r.bootstrap

    def _print_evaluate_summary(
        self: SurvivalLearner,
        ctx: EvalContext,
        *,
        resolved_model,
        eval_times,
        eval_times_was_auto,
        orig_random_state,
        orig_n_jobs,
        y_test,
    ):
        """Print a boxed summary of evaluation parameters."""
        rows = [
            ("Models:", _format_model_display(resolved_model)),
            ("Eval times:", _format_eval_times_row(eval_times, eval_times_was_auto)),
            ("Importance:", _format_importance_row(ctx)),
            ("Bootstrap:", _format_bootstrap_row(ctx, len(y_test))),
            ("State:", _format_state_overrides(
                ctx.random_state, ctx.n_jobs, orig_random_state, orig_n_jobs,
                self.random_state, self.n_jobs,
            )),
        ]
        if y_test is not self.y_test:
            truncation_note = check_truncation_needed(y_test, self.y_train)
            if truncation_note:
                rows.append(("IPCW note:", truncation_note))
        _print_param_box("Evaluate", rows)

    def _auto_kang_comparison(
        self: SurvivalLearner,
        result,
        y_test,
        model_names,
    ):
        """Auto-compute Kang z-test for all model pairs.

        Returns concatenated DataFrame of pairwise Kang tests, or None.
        """
        import pandas as pd
        from ..evaluation._paired_comparison import compare_kang

        # Extract risk scores per model from test_set
        test_set = getattr(result, 'test_set', None)
        if test_set is None or len(test_set) == 0:
            return None

        risk_scores = {}
        for name in model_names:
            model_data = test_set[test_set['model_type'] == name]
            arr = np.array(model_data['risk_score'].dropna().values, dtype=np.float64)
            if len(arr) > 0 and np.all(np.isfinite(arr)):
                risk_scores[name] = arr

        if len(risk_scores) < 2:
            return None

        # All score arrays must be same length
        lengths = {len(v) for v in risk_scores.values()}
        if len(lengths) != 1:
            logger.debug(
                "Auto-Kang skipped: risk score arrays have different lengths %s "
                "across models %s", lengths, list(risk_scores.keys()),
            )
            return None

        dfs = [
            compare_kang(
                y_test,
                risk_scores[a],
                risk_scores[b],
                model_a_name=a,
                model_b_name=b,
            )
            for a, b in itertools.combinations(risk_scores.keys(), 2)
        ]
        return pd.concat(dfs, ignore_index=True)

    def validate(
        self: SurvivalLearner,
        X_val,
        y_val,
        models=None,
        name="External",
        bootstrap=True,
        n_bootstrap="auto",
        bootstrap_ci=0.95,
        verbose=True,
        subpopulations=None,
        use_recalibrated='auto',
        references=None,
        higher_is_worse=True,
        **kwargs,
    ):
        """
        Validate model on external data.

        Wrapper around evaluate() with defaults suited for external validation:
        bootstrap enabled, no permutation importance. Supports multiple external
        cohorts - each call accumulates results in ``_state.validation_results`` dict.

        Parameters
        ----------
        X_val : DataFrame
            External validation features.
        y_val : SurvivalArray
            External validation outcomes.
        models : str, list, dict, BaseSurvivalModel, or None, default=None
            Model(s) to validate. ``None`` validates every trained model
            (CV-sorted). Accepts the same flexible inputs as ``evaluate()``:
            a name, a list/dict of names, or a model/ModelResult.
        name : str, default="External"
            Cohort name for identification (e.g., "TCGA", "MSK", "DFCI").
            Used as key in ``_state.validation_results`` dict.
        bootstrap : bool, default=True
            If True, compute bootstrap confidence intervals.
        n_bootstrap : int or "auto", default="auto"
            Number of bootstrap iterations.
        bootstrap_ci : float, default=0.95
            Confidence level for bootstrap intervals.
        verbose : bool, default=True
            If True, print metrics table. When a prior :meth:`evaluate` left a
            held-out-test baseline, each metric is annotated with its change
            (``Δ``) from that baseline (the generalization gap); the baseline is
            the most recent ``evaluate()`` result, so models absent from it (and
            reference rows) show no ``Δ``. When no baseline exists, a one-line
            hint to run ``evaluate()`` is printed instead.
        subpopulations : dict of {group_var: labels_array}, optional
            Subgroup labels aligned with ``X_val``/``y_val`` for stratified
            metrics on this cohort, mapping each subgroup-variable name to a
            cohort-aligned label array. When omitted, no subpopulation
            breakdown is computed for the cohort.
        use_recalibrated : {'auto', True, False}, default='auto'
            Whether to apply the model's attached internal calibrator (from
            ``train(calibrate=...)``) to the external-cohort S(t|x). ``'auto'``
            (and ``True``) apply it, so the external metrics reflect the
            **deployed** (calibrated) model — this is a fixed dev-cohort
            transform the external rows never saw, so it is leakage-free, and
            it lets validate answer whether the development-cohort calibration
            transports (e.g. "slope 1.0 internal -> 1.4 external"). ``False``
            forces raw S(t|x) for all models. Models without a calibrator are
            raw either way. (Fitting a *new* calibrator on the external cohort
            is a planned separate feature.)
        references : dict of {name: array-like}, optional
            Reference biomarker scores (LORIS, TMB, PD-L1, etc.) to evaluate
            alongside models on this cohort. Each array must be aligned with
            ``X_val`` — one value per validation patient. Reference rows are
            appended to the cohort's concordance/AUC(t)/Brier tables with the
            same columns as model rows; non-finite values are filtered per
            reference (a :class:`MethodologicalWarning` is emitted reporting
            how many observations were dropped). Unlike :meth:`benchmark`,
            only array-aligned references are accepted — column-name lookups
            against ``X_val`` are not supported because external cohorts may
            not share column semantics with the training data.
        higher_is_worse : bool or dict of {name: bool}, default True
            Whether higher reference values indicate worse outcome. Pass a
            dict to set per-reference (e.g. ``{'LORIS': False, 'TMB': False,
            'risk_score': True}``); references not listed in the dict fall
            back to ``True``. Only consulted when ``references`` is given.
        **kwargs
            Additional arguments passed to evaluate(). Notably ``gnd_n_bins``
            — lower it (<= N/5) to compute the GND calibration test on
            small external cohorts where the default 10 bins would return NaN.

        Returns
        -------
        ValidationResult
            Validation metrics with cohort metadata and optional bootstrap CIs.

        Raises
        ------
        ValueError
            If any reference name collides with a model name being
            evaluated, which would make the merged concordance table
            ambiguous.
        MemoryError
            Re-raised if the computation exhausts available memory; resource
            errors are never swallowed by the broad failure handler.
        RecursionError
            Re-raised if the computation exceeds the recursion limit; resource
            errors are never swallowed by the broad failure handler.

        Notes
        -----
        IPCW-weighted metrics (Uno C, AUC(t), IBS) are computed on
        validation observations within the training time range;
        observations beyond ``train_max`` are administratively censored.
        Harrell's C and calibration plots use the full validation set.
        See ``docs/evaluation.md#training-range-cap``.

        Examples
        --------
        >>> # Validate on multiple external cohorts
        >>> learner.validate(X_tcga, y_tcga, name="TCGA")
        >>> learner.validate(X_msk, y_msk, name="MSK")
        >>> learner.validate(X_dfci, y_dfci, name="DFCI")
        >>> # Access each cohort's results view via learner.validation("TCGA")
        >>> # Report shows all cohorts
        >>> learner.generate_report("report.html")
        """
        # Default: skip importance for external validation (no training data context)
        kwargs.setdefault('n_importance_repeats', 0)
        # External validation should weight Uno IPCW with the validation
        # cohort's own censoring distribution (training G_hat is correct only
        # under exchangeability, which fails by definition for `validate()`).
        kwargs.setdefault('ipcw_g_hat_source', 'test')

        # Auto-align column order to training data (Pipelines are order-sensitive)
        X_val = _align_columns(
            X_val, self.X_train.columns.tolist(), context="Validation data",
        )

        # Resolve which models will be evaluated (used below for
        # recalibration eligibility and reference name-collision check).
        eval_model_names = _resolve_eval_model_names(self, models)

        # Resolve references: score reference biomarkers alongside
        # models on the external cohort. Must happen after _align_columns so
        # len(X_val) matches the array contract.
        reference_arrays = None
        reference_names = []
        hiw_map, hiw_default = {}, True
        if references:
            reference_arrays, nonfinite_counts = (
                _resolve_validation_reference_arrays(references, X_val)
            )
            reference_names = list(reference_arrays.keys())
            # Name collisions with models would make the merged concordance
            # table ambiguous.
            overlap = set(reference_names) & set(eval_model_names)
            if overlap:
                raise ValueError(
                    f"Reference names overlap with model names: {sorted(overlap)}. "
                    "Use different names to avoid ambiguity."
                )
            feature_overlap = set(reference_names) & set(X_val.columns)
            if feature_overlap:
                # Dedupe by overlap signature. Vignette 07's 5-cohort
                # validation loop reuses the same references dict + column
                # schema and fired this warning once per cohort.
                sig = frozenset(feature_overlap)
                if sig not in self._state.reference_overlap_warned:
                    warnings.warn(
                        f"Reference names match feature columns in X_val: "
                        f"{sorted(feature_overlap)}. Verify that reference scores "
                        f"were not accidentally included as validation features "
                        f"(data leakage).",
                        SuggestionWarning,
                        stacklevel=2,
                    )
                    self._state.reference_overlap_warned.add(sig)
            if verbose:
                _report_nonfinite_references(reference_arrays, nonfinite_counts)
            hiw_map, hiw_default = _resolve_higher_is_worse(higher_is_worse)

        # Route cohort-scope subpopulations into the bootstrap config so the
        # resampler sees validation-aligned labels. When none were
        # provided we explicitly opt out of bootstrap subpop CIs — the
        # training-scope labels in state.subpopulations_test are not aligned
        # with X_val/y_val and would crash _compute_subpopulation_iteration_metrics.
        _sg_override = _resolve_validate_subpopulation_override(
            subpopulations, y_val, self._state,
        )

        # Held-out-test baseline for the verbose metrics table: annotate each
        # cell with its change (Δ) from the development metrics. validate()
        # never overwrites per-model ``.evaluation`` (``_store_result=False``),
        # so ``self.evaluation_result`` reflects a prior ``evaluate()`` on the
        # held-out test split. ``None`` when the user skipped ``evaluate()`` —
        # in that case we print a one-line hint and show no deltas.
        delta_baseline = None
        if verbose:
            prior_eval = self.evaluation_result
            if (prior_eval is not None and prior_eval.concordance is not None
                    and not prior_eval.concordance.empty):
                delta_baseline = prior_eval.concordance
            else:
                print("Run evaluate() first to see Δ vs held-out test.")

        # Reject anything outside the documented {'auto', True, False} so an
        # unexpected value can't silently force raw S(t|x) via falsiness.
        if use_recalibrated not in ('auto', True, False):
            raise ValueError(
                f"use_recalibrated must be 'auto', True, or False; "
                f"got {use_recalibrated!r}."
            )
        # Single pass. The attached internal calibrator is the deployed model,
        # so by default it is applied to the external cohort (leakage-free —
        # it is a fixed dev-cohort transform). ``use_recalibrated=False`` forces
        # raw S(t|x) by passing an explicit empty transforms dict, which
        # suppresses evaluate()'s auto-build of calibrator transforms.
        _surv_probs_transforms = None if use_recalibrated else {}
        eval_result = self.evaluate(
            models=models,
            X=X_val,
            y=y_val,
            n_bootstrap=n_bootstrap,
            bootstrap_ci=bootstrap_ci,
            _store_result=False,
            _subpopulation_labels_override=_sg_override,
            # validate() wraps evaluate(), so the chain from the resolver's
            # warn() is `warn → _resolve_… → evaluate → validate → user`
            # (4 arrows) instead of the default 3-arrow chain.
            _warn_stacklevel=4,
            # Scope the IPCW truncation warning to this cohort so each
            # external cohort gets exactly one warning per session.
            _cohort_name=name,
            bootstrap=bootstrap,
            verbose=verbose,
            _surv_probs_transforms=_surv_probs_transforms,
            _delta_baseline=delta_baseline,
            **kwargs,
        )

        # Score reference biomarkers on this cohort and merge into the
        # concordance/AUC/Brier tables alongside model rows.
        if reference_arrays:
            adapters = _build_reference_adapters(
                reference_arrays, hiw_map, hiw_default,
            )
            ref_parts = _evaluate_references_on(
                X_val, y_val,
                reference_arrays=reference_arrays,
                adapters=adapters,
                hiw_map=hiw_map, hiw_default=hiw_default,
                y_train=self._state.y_train,
                eval_times=kwargs.get('eval_times'),
                sg_labels=None,
                bootstrap=bootstrap,
                n_bootstrap=n_bootstrap,
                bootstrap_ci=bootstrap_ci,
                random_state=kwargs.get('random_state'),
                # user → validate → _evaluate_references_on → evaluate
                _warn_stacklevel=4,
            )
            eval_result = _aggregate_evaluation_results(
                [("_models_", eval_result)] + ref_parts
            )

        # Compute cohort metadata
        n_samples = len(X_val)
        n_events = int(y_val['event'].sum())

        # Wrap in ValidationResult with cohort metadata
        val_result = ValidationResult(
            name=name,
            evaluation=eval_result,
            n_samples=n_samples,
            n_events=n_events,
            X_val=X_val,
            y_val=y_val,
            reference_arrays=dict(reference_arrays) if reference_arrays else None,
            reference_names=tuple(reference_names),
        )

        # Compute subpopulation metrics for validation data if provided
        if subpopulations is not None:
            try:
                from ..evaluation._subpopulation import compute_subpopulation_metrics
                from ..utils import auto_eval_times as _auto_et
                _val_et = kwargs.get('eval_times', None)
                if _val_et is None and self._state.y_train is not None:
                    try:
                        _val_et = _auto_et(
                            y_val, y_train=self._state.y_train, _quiet=True,
                        )
                    except (ValueError, TypeError) as exc:
                        _warn_eval_times_failure(
                            exc, "for validation subpopulation metrics",
                            "AUC(t) subpopulation metrics will be unavailable.",
                            stacklevel=3,
                        )
                _val_surv_probs = eval_result._surv_probs_cache
                _val_g_source = kwargs.get('ipcw_g_hat_source', 'test')
                _val_ipcw_y_train = (
                    y_val if _val_g_source == 'test' else None
                )
                val_result.subpopulation_results = compute_subpopulation_metrics(
                    eval_result, subpopulations, y_val,
                    y_train=self._state.y_train,
                    eval_times=_val_et,
                    surv_probs_by_model=_val_surv_probs,
                    dense_surv_probs_by_model=eval_result._ibs_surv_probs_cache,
                    ipcw_y_train=_val_ipcw_y_train,
                    _warn_stacklevel=3,
                )
            except (MemoryError, RecursionError):
                raise
            # Broad catch is intentional: compute_subpopulation_metrics calls
            # sksurv/numpy/pandas which can raise many exception types.
            # MemoryError/RecursionError are re-raised above.
            except Exception as exc:
                logger.error("Validation subpopulation metrics failed: %s", exc, exc_info=True)
                warnings.warn(
                    f"Validation subpopulation metrics failed: {exc}. "
                    f"Check that subpopulation labels align with validation data.",
                    SuggestionWarning,
                    stacklevel=2,
                )

        # Store (accumulate, don't overwrite without warning)
        if name in self._state.validation_results:
            warnings.warn(
                f"Overwriting existing validation result for '{name}'",
                InternalWarning,
                stacklevel=2,
            )
        self._state.validation_results[name] = val_result

        return val_result
