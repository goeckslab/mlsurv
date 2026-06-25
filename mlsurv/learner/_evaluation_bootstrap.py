"""BootstrapMixin — bootstrap CIs for the SurvivalLearner."""

from __future__ import annotations

import warnings
from typing import Any, TYPE_CHECKING

from .._shared import try_verbose_print
from .._warnings import SuggestionWarning
from ..evaluation import bootstrap as _bootstrap
from ..tuning import auto_n_bootstrap
from ..results.containers import BootstrapResult, _get_model_names
from ..utils import check_truncation_needed
from ._format import _print_param_box, _format_model_display, _format_state_overrides
from ..logging_utils import get_logger
from ._eval_context import EvalContext
from ._evaluation_helpers import (
    _IPCW_TAU_AUTO,
    _emit_ipcw_truncation_warning,
    _resolve_eval_model_names,
    _resolve_ipcw_g_source,
    _resolve_ipcw_tau,
    _resolve_subpopulation_labels,
    resolve_state_eval_times,
)

logger = get_logger("learner")

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from mlsurv.learner._core import SurvivalLearner


class BootstrapMixin:
    """Mixin for bootstrap confidence intervals."""

    def bootstrap(
        self: SurvivalLearner,
        models: Any = None,
        X: pd.DataFrame | None = None,
        y: np.ndarray | None = None,
        n_bootstrap: int | str = "auto",
        metrics: list | None = None,
        custom_metrics: dict | None = None,
        lower_is_better: list | set | None = None,
        ci: float = 0.95,
        eval_times: np.ndarray | list | str | None = None,
        include_predictions: bool = False,
        include_time_resolved: bool = False,
        include_importance: bool = False,
        n_importance_repeats: int = 10,
        stratify: bool = False,
        random_state: int | None = None,
        show_progress: bool = True,
        verbose: bool = True,
        n_jobs: int | None = None,
        ipcw_tau: float | str = _IPCW_TAU_AUTO,
        ipcw_g_hat_source: str = "train",
    ) -> BootstrapResult:
        """
        Bootstrap evaluation for confidence intervals.

        Parameters
        ----------
        models : BaseSurvivalModel, Pipeline, ModelResult, dict, or None
            Model(s) to bootstrap. Accepts:
            - Single model or ModelResult: bootstraps that model
            - dict of {name: model/ModelResult}: bootstraps all models in dict
            - None: bootstraps all trained models
        X : DataFrame, optional
            Test features.
        y : SurvivalArray, optional
            Test outcomes.
        n_bootstrap : int or "auto", default="auto"
            Number of bootstrap iterations. If "auto", computed as
            min(500, max(100, 10 * sqrt(n_samples))).
        metrics : list, optional
            Metrics to compute. Default: all metrics (harrell, ipcw, integrated_brier, ctd).
        custom_metrics : dict, optional
            Additional custom metrics. Dict of {name: scorer_fn}.
            Scorer signature: scorer(estimator, X, y) -> float
        lower_is_better : list or set, optional
            Metric names where lower values indicate better performance.
            Built-in 'integrated_brier' is auto-included.
        ci : float, default=0.95
            Confidence level.
        eval_times : array-like or "auto", optional
            Timepoints for AUC(t).  Pass ``"auto"`` to clear any inherited
            eval_times and force auto-computation from the data.
        include_predictions : bool, default=False
            Store predictions per iteration.
        include_time_resolved : bool, default=False
            Also store the per-iteration time-resolved **Brier(t)** distribution
            that ``plot_metric_time``'s Brier panel needs for its
            confidence-interval ribbon. Brier(t) costs an extra
            survival-function pass per iteration, so it is opt-in. AUC(t) is
            cheap (risk scores only) and is always collected, so the AUC ribbon
            appears on any bootstrap regardless of this flag.
        include_importance : bool, default=False
            Store importance per iteration.
        n_importance_repeats : int, default=10
            Permutation repeats per iteration.
        stratify : bool, default=False
            Stratify resampling by event.
        random_state : int, optional
            Seed for reproducibility.
        show_progress : bool, default=True
            Show progress bar during bootstrapping.
        verbose : bool, default=True
            If True, print bootstrap metrics table after completion.
        n_jobs : int, optional
            Number of parallel jobs. If None, uses the value from setup().
        ipcw_tau : float or "auto", optional
            Evaluation horizon τ for Uno's IPCW concordance, passed through to
            :func:`sksurv.metrics.concordance_index_ipcw`. When unset, a
            conservative default is auto-selected (see
            :func:`mlsurv.evaluation.metrics.compute_safe_ipcw_tau`) to avoid
            unstable estimates when test follow-up extends beyond training
            censoring support.
        ipcw_g_hat_source : {'train', 'test'}, default='train'
            Cohort whose censoring distribution feeds Uno's IPCW G_hat
            estimator. See ``evaluate()``. ``validate()``
            overrides the default to ``'test'``.

        Returns
        -------
        BootstrapResult
            Bootstrap results with summary and per-iteration data.
            When multiple models are provided, results are aggregated with
            model_type column identifying each model.

        Notes
        -----
        IPCW-weighted metrics (Uno C, AUC(t), IBS) are computed on test
        observations within the training time range; observations beyond
        ``train_max`` are administratively censored. Harrell's C and
        calibration plots use the full test set. See
        ``docs/evaluation.md#training-range-cap``.

        Examples
        --------
        >>> # Bootstrap all trained models
        >>> result = learner.bootstrap()
        >>>
        >>> # Bootstrap models from comparison
        >>> comparison = learner.run()
        >>> result = learner.bootstrap(comparison)
        >>>
        >>> # Bootstrap single model by name
        >>> result = learner.bootstrap('rsf')
        """
        self._ensure_setup()
        target_names = _resolve_eval_model_names(self, models)
        self._warn_if_overwriting_model_data(
            'bootstrap', 'bootstrap result', target_names,
        )

        # Capture original args before fallback (for override detection)
        orig_random_state = random_state
        orig_n_jobs = n_jobs

        # Use self.random_state as fallback for reproducibility
        if random_state is None:
            random_state = self.random_state

        # Resolve n_jobs: use parameter if provided, otherwise fall back to instance default
        n_jobs = n_jobs if n_jobs is not None else self.n_jobs

        X_test = X if X is not None else self.X_test
        y_test = y if y is not None else self.y_test

        # Resolve eval_times through the canonical resolver so bootstrap honors
        # the grid evaluate()/setup() persisted on user_eval_times. Without
        # this, a plain bootstrap() after evaluate(eval_times=[...]) would let
        # the free bootstrap() recompute its own auto grid, landing the CI band
        # on a different eval_time set than the point curve.
        eval_times = resolve_state_eval_times(
            self._state, eval_times,
            custom_data=(X is not None or y is not None),
            y_for_risk=y,
            # Chain: warn → bootstrap → user.
            stacklevel=3,
        )

        # Bundle threaded params into a single EvalContext.
        # bootstrap()'s public API uses different field names than evaluate()
        # (metrics, ci, stratify, n_importance_repeats default=10), so map
        # them onto the shared dataclass at construction.
        ctx = EvalContext(
            eval_times=eval_times,
            extra_eval_times=self._state.user_extra_eval_times,
            n_importance_repeats=n_importance_repeats,
            custom_metrics=custom_metrics,
            bootstrap=True,
            n_bootstrap=n_bootstrap,
            bootstrap_ci=ci,
            bootstrap_metrics=metrics,
            bootstrap_stratify=stratify,
            bootstrap_lower_is_better=lower_is_better,
            bootstrap_include_time_resolved=include_time_resolved,
            bootstrap_include_importance=include_importance,
            include_predictions=include_predictions,
            ipcw_tau=ipcw_tau,
            ipcw_g_hat_source=ipcw_g_hat_source,
            random_state=random_state,
            n_jobs=n_jobs,
            show_progress=show_progress,
            verbose=verbose,
        )

        # Resolve IPCW tau first so the once-per-cohort truncation
        # warning can include it in a single consolidated message.
        y_for_g = _resolve_ipcw_g_source(
            ctx.ipcw_g_hat_source, self.y_train, y_test,
        )
        resolved_ipcw_tau, ipcw_support_flag = _resolve_ipcw_tau(
            y_for_g, y_test, ctx.ipcw_tau,
            metrics=ctx.bootstrap_metrics,
            context="bootstrap",
            _warn_stacklevel=3,
        )
        if ipcw_support_flag is not None:
            ipcw_support_flag["g_hat_source"] = ctx.ipcw_g_hat_source

        _trunc_t_max, _trunc_n_beyond = _emit_ipcw_truncation_warning(
            self._state, "main", self.y_train, y_test,
            tau=resolved_ipcw_tau, stacklevel=3,
        )

        # Resolve model input to standardized format
        resolved_model, _, original_model_result = self._resolve_model_input(models)

        # Print bootstrap summary box (cosmetic — never abort bootstrap on failure)
        if verbose:
            try_verbose_print(
                self._print_bootstrap_summary,
                ctx,
                resolved_model=resolved_model,
                orig_random_state=orig_random_state,
                orig_n_jobs=orig_n_jobs,
                y_test=y_test,
                context="Bootstrap summary",
                detail="All parameters are active.",
            )

        _sg_labels = _resolve_subpopulation_labels(
            self._state.subpopulations_test,
            bootstrap=True, y_test=y_test,
        )

        # Build common bootstrap parameters
        bootstrap_params = dict(
            y_train=self.y_train,
            # IPCW G_hat fitting cohort. When None, downstream falls
            # back to y_train.
            ipcw_y_train=(y_for_g if ctx.ipcw_g_hat_source == "test" else None),
            n_bootstrap=ctx.n_bootstrap,
            metrics=ctx.bootstrap_metrics,
            custom_metrics=ctx.custom_metrics,
            lower_is_better=ctx.bootstrap_lower_is_better,
            ci=ctx.bootstrap_ci,
            eval_times=ctx.eval_times,
            extra_eval_times=ctx.extra_eval_times,
            include_predictions=ctx.include_predictions,
            # AUC(t) is cheap — risk scores are already computed for the
            # C-index — so collect it on every bootstrap for a free
            # plot_metric_time ribbon. Brier(t) needs an extra
            # survival-function pass per iteration, so it stays opt-in behind
            # include_time_resolved.
            include_auc=True,
            include_brier_t=ctx.bootstrap_include_time_resolved,
            include_importance=ctx.bootstrap_include_importance,
            n_importance_repeats=ctx.n_importance_repeats,
            stratify=ctx.bootstrap_stratify,
            random_state=ctx.random_state,
            show_progress=ctx.show_progress,
            n_jobs=ctx.n_jobs,
            subpopulation_labels=_sg_labels,
            model_name=original_model_result.name if original_model_result else None,
            categorical_groups=self._state.categorical_groups,
            # Apply attached internal calibrators so standalone bootstrap CIs
            # sit on the same calibrated scale as evaluate's metrics. The
            # closure binds the bootstrap cohort X (score->baseline needs it).
            surv_probs_transforms=({
                n: (lambda sp, et, _n=n, _X=X_test:
                    self._apply_calibrator(_n, sp, et, X=_X))
                for n in self._state.models if self._has_calibrator(n)
            } or None),
            ipcw_tau=resolved_ipcw_tau,
            # Chain: warn → _emit_failure_summary → _bootstrap_single →
            # free bootstrap → mixin.bootstrap → user (5 frames). Free
            # bootstrap adds +1 internally for the dict-of-models dispatch
            # path. Mirror of the `_warn_stacklevel` convention
            # on _resolve_subpopulation_labels.
            _warn_stacklevel=5,
        )

        # Call bootstrap (works for both dict and single model)
        result = _bootstrap(resolved_model, X_test, y_test, **bootstrap_params)

        # Persist the resolved replay config so a later pass can reproduce the
        # identical resamples (same seed/config) if needed.
        resolved_nb = (
            auto_n_bootstrap(len(y_test))
            if ctx.n_bootstrap == "auto" else int(ctx.n_bootstrap)
        )
        result.run_config = {
            "n_bootstrap": resolved_nb,
            "random_state": ctx.random_state,
            "ci": ctx.bootstrap_ci,
            "metrics": ctx.bootstrap_metrics,
            "stratify": ctx.bootstrap_stratify,
            "eval_times": ctx.eval_times,
            "ipcw_tau": resolved_ipcw_tau,
            "ipcw_g_hat_source": ctx.ipcw_g_hat_source,
        }

        try:
            object.__setattr__(result, 'ipcw_support_flag', ipcw_support_flag)
        except (AttributeError, TypeError):
            pass
        # Always assign — None clears a stale flag from a prior violating call.
        self._state.ipcw_support_flag = ipcw_support_flag

        # Persist truncation diagnostic on result if it carries the field
        # (BootstrapResult may not — guarded with try/except).
        try:
            object.__setattr__(result, 'truncation_t_max', _trunc_t_max)
            object.__setattr__(result, 'truncation_n_beyond', _trunc_n_beyond)
        except (AttributeError, TypeError):
            pass

        # Store per-model on ModelResult (authoritative)
        self._store_per_model_bootstrap(result, resolved_model, original_model_result)

        if ctx.verbose:
            result.print_summary(ci_level=ctx.bootstrap_ci)
        return result

    def _store_per_model_bootstrap(
        self: SurvivalLearner, result, resolved_model, original_model_result,
    ):
        """Store bootstrap result on each ModelResult (authoritative per-model storage).

        For multi-model bootstraps, splits the combined result into per-model
        BootstrapResults.  For single-model, stores directly.
        """
        from ..results.containers import _extract_model_bootstrap

        if isinstance(resolved_model, dict):
            # Iterate the input dict directly so the write-back targets the
            # storage keys the user actually passed in (matching _state.models)
            # rather than re-resolving via result.summary['model_type'].
            for name in resolved_model:
                mr = self._state.models.get(name)
                if mr is not None:
                    mr.bootstrap = _extract_model_bootstrap(result, name)
                else:
                    warnings.warn(
                        f"Model '{name}' from bootstrap() input not found in "
                        f"learner.models; bootstrap result was computed but "
                        f"not stored on any ModelResult.",
                        SuggestionWarning,
                        stacklevel=4,
                    )
        else:
            # Single model: use caller-provided reference, or fall back to name lookup
            mr = original_model_result
            if mr is None:
                model_names = _get_model_names(result.summary)
                if model_names:
                    mr = self._state.models.get(model_names[0])
            if mr is not None:
                mr.bootstrap = result
            else:
                logger.debug(
                    "Bootstrap result could not be stored on any ModelResult; "
                    "model not found in state.models."
                )

    def _print_bootstrap_summary(
        self: SurvivalLearner,
        ctx: EvalContext,
        *,
        resolved_model,
        orig_random_state,
        orig_n_jobs,
        y_test,
    ):
        """Print a boxed summary of bootstrap parameters."""
        if ctx.n_bootstrap == "auto":
            resolved_n = auto_n_bootstrap(len(y_test))
            iterations_str = f"auto (→{resolved_n})"
        else:
            iterations_str = str(ctx.n_bootstrap)

        metrics_str = (
            ", ".join(ctx.bootstrap_metrics)
            if ctx.bootstrap_metrics is not None
            else "all"
        )

        ci_parts = [f"{ctx.bootstrap_ci * 100:g}%"]
        if ctx.bootstrap_stratify:
            ci_parts.append("stratified")
        ci_str = ", ".join(ci_parts)

        rows = [
            ("Models:", _format_model_display(resolved_model)),
            ("Iterations:", iterations_str),
            ("Metrics:", metrics_str),
            ("CI:", ci_str),
            ("State:", _format_state_overrides(
                ctx.random_state, ctx.n_jobs, orig_random_state, orig_n_jobs,
                self.random_state, self.n_jobs,
            )),
        ]
        if y_test is not self.y_test:
            truncation_note = check_truncation_needed(y_test, self.y_train)
            if truncation_note:
                rows.append(("IPCW note:", truncation_note))
        _print_param_box("Bootstrap", rows)
