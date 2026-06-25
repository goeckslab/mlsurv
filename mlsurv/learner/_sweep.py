"""SweepMixin — parameter_sweep() for SurvivalLearner."""

from __future__ import annotations

import pickle
import warnings
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from tqdm.auto import tqdm

from mlsurv._warnings import SuggestionWarning
from mlsurv.logging_utils import get_logger
from mlsurv.results.containers import SweepResult
from mlsurv.utils import resolve_joblib_backend, tqdm_joblib as _tqdm_joblib

if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner

logger = get_logger("learner.sweep")

_VALID_TARGETS = frozenset({"model", "imputer", "scaler", "feature_selector"})

# Map _compute_concordance column names → CV metric names for consistent
# metric naming in the SweepResult DataFrame.
_TEST_METRIC_MAP = {
    "c_index_test": "harrell",
    "c_index_ipcw_test": "ipcw",
    "integrated_brier_test": "integrated_brier",
    "brier_score_test": "brier_score",
    "c_index_td_test": "ctd",
}

# Exceptions that are always re-raised (never caught as sweep-point failures).
_FATAL_EXCEPTIONS = (MemoryError, RecursionError, KeyboardInterrupt)

# Exceptions caught per sweep point — broad enough to prevent one bad point
# from crashing the entire sweep, especially in parallel mode where an
# uncaught exception destroys ALL results.
_SWEEP_CAUGHT_EXCEPTIONS = (
    ValueError, RuntimeError, TypeError, ArithmeticError,
    ImportError, KeyError, AttributeError, IndexError,
    NotImplementedError, OSError,
)


# -- Module-level helpers for joblib pickling --------------------------------

def _warn_sweep_failure(param, value, model_name, exc_msg):
    """Log and warn about a failed sweep point.

    ``stacklevel=3`` points the warning at the public
    ``parameter_sweep`` caller regardless of whether this is invoked
    from ``_sweep_sequential`` or ``_sweep_parallel``.
    """
    logger.warning(
        "Sweep point %s=%r for model '%s' failed: %s",
        param, value, model_name, exc_msg,
    )
    warnings.warn(
        f"Sweep point {param}={value!r} for model "
        f"'{model_name}' failed: {exc_msg}",
        SuggestionWarning,
        stacklevel=3,
    )


def _extract_sweep_metrics(result, value, model_name):
    """Extract CV metrics from a ModelResult into sweep rows."""
    rows = []
    if result.cv_metrics:
        for metric_name, mdata in result.cv_metrics.items():
            rows.append({
                "param_value": value,
                "model": model_name,
                "phase": "cv",
                "metric": metric_name,
                "mean": mdata["mean"],
                "std": mdata["std"],
                "ci_lower": mdata["ci_lower"],
                "ci_upper": mdata["ci_upper"],
            })
    return rows


def _run_sweep_point(learner, model_name, param, value, target,
                     include_test, eval_times, n_bootstrap):
    """Run one sweep point -- module-level for joblib pickling.

    Returns ``(rows, caught_warnings, warning_info)`` where
    *caught_warnings* is a list of serializable warning tuples and
    *warning_info* is ``None`` on success or
    ``(param, value, model_name, exc_str)`` on failure.
    """
    rows = []
    caught_warnings = []
    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            # lifelines pandas-4.0 keyword-only deprecations are not
            # user-actionable from mlsurv.  ``simplefilter('always')``
            # above wipes the import-time filter; re-add it for this worker.
            warnings.filterwarnings(
                "ignore",
                message=r"Starting with pandas version 4\.0",
                category=Warning,
                module=r"lifelines\..*",
            )
            result = learner._run_single_sweep_point(
                model_name, param, value, target, _cv_n_jobs=1,
            )
            rows = _extract_sweep_metrics(result, value, model_name)
            if include_test:
                learner._collect_test_metrics(
                    rows, result, value, model_name, eval_times, n_bootstrap,
                )
        caught_warnings = [
            (w.category.__name__, str(w.message), w.filename, w.lineno)
            for w in ws
        ]
    except _FATAL_EXCEPTIONS:
        raise
    except _SWEEP_CAUGHT_EXCEPTIONS as exc:
        # Capture any warnings that fired before the exception
        caught_warnings = [
            (w.category.__name__, str(w.message), w.filename, w.lineno)
            for w in ws
        ]
        return rows, caught_warnings, (param, value, model_name, str(exc))
    return rows, caught_warnings, None


# Warning category lookup for re-emitting serialized warnings from workers.
_WARNING_CLASSES = {
    "SuggestionWarning": SuggestionWarning,
}
# Lazy-populate from mlsurv._warnings on first use.
_WARNING_CLASSES_POPULATED = False


def _resolve_warning_class(name):
    """Return the warning class for *name*, falling back to UserWarning."""
    global _WARNING_CLASSES_POPULATED  # noqa: PLW0603
    if not _WARNING_CLASSES_POPULATED:
        try:
            from mlsurv._warnings import (
                InternalWarning,
                MethodologicalWarning,
                SuggestionWarning as _SW,
            )
            _WARNING_CLASSES.update({
                "MethodologicalWarning": MethodologicalWarning,
                "InternalWarning": InternalWarning,
                "SuggestionWarning": _SW,
            })
        except ImportError:
            pass
        _WARNING_CLASSES_POPULATED = True
    return _WARNING_CLASSES.get(name, UserWarning)


class SweepMixin:
    """Provides :meth:`parameter_sweep` for exploratory hyperparameter analysis."""

    def parameter_sweep(
        self: SurvivalLearner,
        param: str,
        values: Sequence,
        *,
        target: str = "model",
        models: Union[str, List[str], None] = None,
        include_test: bool = False,
        n_bootstrap: int = 0,
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> SweepResult:
        """Sweep a single parameter across values and measure CV performance.

        Runs cross-validated training for each (value, model) combination
        and returns a :class:`SweepResult` containing all CV metrics.
        The learner's state is **not** modified.

        Parameters
        ----------
        param : str
            Name of the parameter to sweep (e.g., ``'k'``,
            ``'n_estimators'``).
        values : Sequence
            Values to try for *param*.
        target : str, default ``'model'``
            Which pipeline component *param* belongs to.  One of
            ``'model'``, ``'imputer'``, ``'scaler'``, or
            ``'feature_selector'``.
        models : str, list of str, or None
            Model(s) to sweep.  If ``None``, uses already-trained model
            keys (raises if none exist).
        include_test : bool, default False
            If True, also evaluate each fitted model on the held-out test
            set and include test-set metrics (``phase='test'``) alongside
            the CV metrics (``phase='cv'``).
        n_bootstrap : int, default 0
            Number of bootstrap resamples for test-set confidence
            intervals.  Only used when ``include_test=True``.  When 0,
            test metrics are point estimates (no CI).
        n_jobs : int, default 1
            Number of parallel jobs for the outer sweep loop.  When
            ``n_jobs > 1``, inner CV and model-internal parallelism are
            disabled to avoid CPU oversubscription.  Set to ``-1`` to
            use all CPUs.
        verbose : bool, default True
            Show a ``tqdm`` progress bar.

        Returns
        -------
        SweepResult
            Contains a ``.data`` DataFrame and a ``.plot()`` method.

        Raises
        ------
        ValueError
            If ``target`` is not a recognized pipeline component, if
            ``values`` is empty, or if ``n_jobs`` is 0.

        Examples
        --------
        >>> learner.setup(feature_selector='topk', random_state=42)
        >>> result = learner.parameter_sweep(
        ...     'k', range(5, 30, 5),
        ...     target='feature_selector',
        ...     models=['coxph', 'rsf'],
        ... )
        >>> result.plot(metric='harrell')
        """
        self._ensure_setup()

        # -- Validate inputs --------------------------------------------------
        if target not in _VALID_TARGETS:
            raise ValueError(
                f"target must be one of {sorted(_VALID_TARGETS)}, "
                f"got '{target}'"
            )

        values = list(values)
        if not values:
            raise ValueError("values must be non-empty")

        if n_jobs == 0:
            raise ValueError(
                "n_jobs=0 has no meaning; use 1 for sequential or -1 "
                "for all CPUs."
            )

        model_names = self._resolve_sweep_models(models)

        # -- Validate component exists for preprocessing targets ---------------
        if target != "model":
            for model_name in model_names:
                self._validate_sweep_component(model_name, target)

        # -- Prepare test-set eval times (once, outside loop) ------------------
        eval_times: Optional[np.ndarray] = None
        if include_test:
            from mlsurv.utils import auto_eval_times, clip_eval_times
            eval_times = auto_eval_times(
                self._state.y_test, y_train=self._state.y_train, _quiet=True,
            )
            eval_times = clip_eval_times(eval_times, self._state.y_train)

        # -- Build work items -------------------------------------------------
        work_items = [
            (value, model_name)
            for value in values
            for model_name in model_names
        ]
        total = len(work_items)

        # -- Run sweep --------------------------------------------------------
        if n_jobs == 1:
            rows = self._sweep_sequential(
                work_items, param, target, include_test, eval_times,
                n_bootstrap, verbose, total,
            )
        else:
            rows = self._sweep_parallel(
                work_items, param, target, include_test, eval_times,
                n_bootstrap, n_jobs, verbose, total,
            )

        data = pd.DataFrame(rows)
        if data.empty:
            warnings.warn(
                f"All {total} sweep points for param '{param}' failed. "
                f"Check the per-point warnings above.",
                SuggestionWarning,
                stacklevel=2,
            )
        return SweepResult(
            data=data, param_name=param, target=target,
            scoring=self._state.scoring,
        )

    # -- Private helpers ------------------------------------------------------

    def _sweep_sequential(
        self: SurvivalLearner,
        work_items: list,
        param: str,
        target: str,
        include_test: bool,
        eval_times,
        n_bootstrap: int,
        verbose: bool,
        total: int,
    ) -> list:
        """Run sweep points sequentially with per-point tqdm updates."""
        rows: list[dict] = []

        with tqdm(total=total, desc=f"Sweeping {param}", disable=not verbose) as pbar:
            for value, model_name in work_items:
                pbar.set_postfix_str(f"{model_name} {param}={value}")
                try:
                    result = self._run_single_sweep_point(
                        model_name, param, value, target,
                    )
                    rows.extend(
                        _extract_sweep_metrics(result, value, model_name),
                    )
                    if include_test:
                        self._collect_test_metrics(
                            rows, result, value, model_name,
                            eval_times, n_bootstrap,
                        )
                except _FATAL_EXCEPTIONS:
                    raise
                except _SWEEP_CAUGHT_EXCEPTIONS as exc:
                    _warn_sweep_failure(param, value, model_name, exc)
                finally:
                    pbar.update(1)
        return rows

    def _sweep_parallel(
        self: SurvivalLearner,
        work_items: list,
        param: str,
        target: str,
        include_test: bool,
        eval_times,
        n_bootstrap: int,
        n_jobs: int,
        verbose: bool,
        total: int,
    ) -> list:
        """Run sweep points in parallel with joblib."""
        backend = resolve_joblib_backend("loky")
        jobs = (
            delayed(_run_sweep_point)(
                self, model_name, param, value, target,
                include_test, eval_times, n_bootstrap,
            )
            for value, model_name in work_items
        )
        try:
            with _tqdm_joblib(tqdm(
                total=total, desc=f"Sweeping {param}", disable=not verbose,
            )):
                results = Parallel(n_jobs=n_jobs, backend=backend)(jobs)
        except (pickle.PicklingError, TypeError) as exc:
            raise RuntimeError(
                f"Failed to serialize learner for parallel sweep. "
                f"This can happen with deep learning models or custom "
                f"objects. Use n_jobs=1 for sequential execution. "
                f"Original error: {exc}"
            ) from exc

        # Collect rows and re-emit warnings from workers on the main process
        all_rows: list[dict] = []
        for point_rows, caught_warnings, warning_info in results:
            all_rows.extend(point_rows)
            for cat_name, msg, fname, lineno in caught_warnings:
                cat = _resolve_warning_class(cat_name)
                warnings.warn_explicit(msg, cat, fname, lineno)
            if warning_info is not None:
                _warn_sweep_failure(*warning_info)
        return all_rows

    def _resolve_sweep_models(
        self: SurvivalLearner,
        models: Union[str, List[str], None],
    ) -> List[str]:
        """Normalize *models* argument to a list of model name strings."""
        names = self._resolve_model_names(
            models, filter_fn=lambda k, v: not v.is_synthetic,
        )
        if not names and models is None:
            if not self._state.models:
                raise ValueError(
                    "No models have been trained. Pass 'models' explicitly "
                    "(e.g., models=['coxph', 'rsf'])."
                )
            raise ValueError(
                "All trained models are synthetic (reference models). "
                "Pass 'models' explicitly "
                "(e.g., models=['coxph', 'rsf'])."
            )
        return names

    def _run_single_sweep_point(
        self: SurvivalLearner,
        model_name: str,
        param: str,
        value,
        target: str,
        _cv_n_jobs: Optional[int] = None,
    ):
        """Train one (model, param-value) combination without storing results."""
        if target == "model":
            return self._train_model_internal(
                model_name, _store=False, _cv_n_jobs=_cv_n_jobs,
                **{param: value},
            )

        preprocessing = self._build_sweep_preprocessing(
            model_name, target, param, value,
        )
        return self._train_model_internal(
            model_name, _store=False, _cv_n_jobs=_cv_n_jobs,
            _preprocessing=preprocessing,
        )

    def _validate_sweep_component(
        self: SurvivalLearner,
        model_name: str,
        target: str,
    ) -> None:
        """Raise if the target preprocessing component is not configured."""
        imputer, scaler, selector = self._resolve_model_preprocessing(model_name)
        component_map = {
            "imputer": imputer,
            "scaler": scaler,
            "feature_selector": selector,
        }
        if component_map[target] is None:
            raise ValueError(
                f"No {target} is configured.  Set one via "
                f"setup({target}=...) before sweeping its parameters."
            )

    def _collect_test_metrics(
        self: SurvivalLearner,
        rows: list,
        model_result,
        value,
        model_name: str,
        eval_times,
        n_bootstrap: int,
    ) -> None:
        """Compute test-set metrics for one sweep point and append to *rows*.

        When *n_bootstrap* > 0, bootstrap-resample the test set to
        produce mean / std / CI for each metric.
        """
        from mlsurv.evaluation._concordance import _compute_concordance

        pipeline = model_result.model
        X_test = self._state.X_test
        y_test = self._state.y_test
        y_train = self._state.y_train

        # Point estimate
        conc_df, _, _, _, _ = _compute_concordance(
            pipeline, X_test, y_test,
            X_train=self._state.X_train,
            y_train=y_train,
            eval_times=eval_times,
        )

        if n_bootstrap <= 0:
            # No bootstrap — point estimates only
            for col in conc_df.columns:
                metric_name = _TEST_METRIC_MAP.get(col)
                if metric_name is None:
                    continue
                val = conc_df[col].iloc[0]
                if np.isnan(val):
                    continue
                rows.append({
                    "param_value": value,
                    "model": model_name,
                    "phase": "test",
                    "metric": metric_name,
                    "mean": val,
                    "std": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                })
            return

        # Bootstrap — resample test set, recompute metrics each iteration
        rng = np.random.RandomState(self._state.random_state)
        n_test = len(X_test)

        # Pre-compute risk scores once (resample by index)
        risk_all = pipeline.predict(X_test)

        boot_values: dict[str, list[float]] = {}
        for _ in range(n_bootstrap):
            idx = rng.choice(n_test, size=n_test, replace=True)
            X_boot = X_test.iloc[idx]
            y_boot = y_test[idx]
            risk_boot = risk_all[idx]

            boot_df, _, _, _, _ = _compute_concordance(
                pipeline, X_boot, y_boot,
                y_train=y_train,
                eval_times=eval_times,
                risk_scores=risk_boot,
            )
            for col in boot_df.columns:
                metric_name = _TEST_METRIC_MAP.get(col)
                if metric_name is None:
                    continue
                boot_values.setdefault(metric_name, []).append(
                    boot_df[col].iloc[0],
                )

        for metric_name, vals in boot_values.items():
            arr = np.array(vals, dtype=float)
            finite = arr[np.isfinite(arr)]
            if len(finite) < 2:
                continue
            mean = float(np.mean(finite))
            std = float(np.std(finite, ddof=1))
            ci_lower = float(np.percentile(finite, 2.5))
            ci_upper = float(np.percentile(finite, 97.5))
            rows.append({
                "param_value": value,
                "model": model_name,
                "phase": "test",
                "metric": metric_name,
                "mean": mean,
                "std": std,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            })

    def _build_sweep_preprocessing(
        self: SurvivalLearner,
        model_name: str,
        target: str,
        param: str,
        value,
    ) -> dict:
        """Build a ``_preprocessing`` override dict for one sweep point."""
        imputer, scaler, selector = self._resolve_model_preprocessing(model_name)

        component_map = {
            "imputer": imputer,
            "scaler": scaler,
            "feature_selector": selector,
        }
        current = component_map[target]

        modified = clone(current)
        modified.set_params(**{param: value})
        return {model_name: {target: modified}}
