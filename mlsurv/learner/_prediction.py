"""PredictionMixin — predict() for new-patient survival predictions."""

from __future__ import annotations

import warnings
from typing import Any, TYPE_CHECKING, Dict, Optional, Union

import numpy as np
import pandas as pd

from .._warnings import MethodologicalWarning, SuggestionWarning

if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner
    from mlsurv.results._prediction import PredictionResult


# CI-only predict() params and their documented defaults. These are consumed
# only on the bootstrap path, so when ci=False a non-default value is a no-op
# that likely means the user forgot ci=True — warn rather than drop it silently.
_CI_ONLY_PREDICT_DEFAULTS = {
    "n_bootstrap": 200,
    "ci_level": 0.95,
    "random_state": None,
    "n_jobs": 1,
}


class PredictionMixin:
    """Mixin providing ``predict()`` for per-patient survival predictions."""

    def predict(
        self: SurvivalLearner,
        X: pd.DataFrame,
        models: Optional[Union[str, list, dict, Any]] = None,
        times: Optional[Union[list, np.ndarray]] = None,
        ci: bool = False,
        ci_band: bool = False,
        n_bootstrap: int = 200,
        ci_level: float = 0.95,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        verbose: bool = True,
        use_recalibrated: bool = True,
        band_calibration: str = 'refit',
    ) -> "Union[PredictionResult, Dict[str, PredictionResult]]":
        """Predict survival for new patients.

        Parameters
        ----------
        X : DataFrame
            Feature matrix for new patients (1 or more rows).
            Must contain the same columns as the training data.
        models : str, ModelResult, list, dict, or None
            Model selection. A name (e.g. ``'rsf'``) predicts with that
            model and returns a single :class:`PredictionResult`; so does
            ``None`` when exactly one model is trained. A list/dict — or
            ``None`` with several models trained — predicts with each
            selected model and returns a dict keyed by model name, so the
            models' predictions can be compared side by side. A
            ``ModelResult`` is resolved by name to the currently
            registered model; a stale reference (e.g. held from before
            ``tune()``) triggers a ``SuggestionWarning`` and the
            registered model is used.
        times : list or array-like, optional
            Specific time points at which to evaluate survival probability
            and RMST.  E.g. ``[12, 24, 60]`` for 1-year, 2-year, 5-year.
        ci : bool, default False
            If ``True``, compute bootstrap confidence intervals via model
            retraining. Significantly more expensive than point estimates.
        ci_band : bool, default False
            If ``True`` (and ``ci=True``), also retain the full bootstrap
            survival functions over the reference time grid so
            :attr:`PredictionResult.survival_function_ci` returns lower/upper
            CI bands for plotting. Adds ``n_bootstrap × n_patients × n_times``
            floats of memory per call; opt in when you need the band, leave
            off for batch ``predict()`` over many patients.

            When the selected model carries a calibrator (and
            ``use_recalibrated=True``), the band is built on the calibrated
            ``S(t|x)`` scale so it matches the calibrated point estimate. By
            default (``band_calibration='refit'``) the calibration step is
            replicated on every resample, so the band carries both model-refit
            and calibration-fit variability — see ``band_calibration``.
        n_bootstrap : int, default 200
            Number of bootstrap iterations (used when ``ci=True``).
        ci_level : float, default 0.95
            Confidence level for bootstrap CIs.
        random_state : int, optional
            Seed for bootstrap reproducibility.  If ``None``, uses the
            learner's ``random_state``.
        n_jobs : int, default 1
            Number of parallel jobs for bootstrap.
        verbose : bool, default True
            Print progress messages.
        use_recalibrated : bool, default True
            When ``True`` (default) and the selected model has an attached
            calibrator (from ``train(calibrate=...)``), return the calibrated
            ``S(t|x)`` curve and re-derived median/RMST/user-times outputs.
            Pass ``False`` to inspect the raw model output. The default is
            ``True`` because the calibrated probabilities are the deployed
            model's output, the more clinically meaningful one.
        band_calibration : {'refit', 'fixed'}, default 'refit'
            How a warper calibrator (``isotonic``, ``lowdof``) enters the CI
            band when ``ci_band=True`` and the model is calibrated. Only
            affects the band width, never the point estimate, and is a no-op
            without a warper calibrator.

            - ``'refit'`` (default): refit the warper on each resample's own
              out-of-fold ``S(t|x)`` before applying it — full-pipeline
              replication, so the band carries the calibration step's own
              sampling variability (Riley et al. 2024; Steyerberg ABCD; rms
              ``calibrate``). Costs ~``cv_folds × n_bootstrap`` extra pipeline
              fits, the same order as ``breslow_slope``.
            - ``'fixed'``: reapply the one map fit on the original training
              cohort to every resample (cheap, but omits calibration-fit
              uncertainty; where the map is flat — e.g. isotonic plateaus near
              ``S=0``/``S=1`` — it compresses the band, which can read as
              over-confident).

            ``breslow_slope`` always refits per resample regardless of this
            flag (it has no fixed-map form).

        Returns
        -------
        PredictionResult or dict of {str: PredictionResult}
            For a single-model selection (a name, or ``None`` with exactly
            one trained model): a container with ``risk_scores``,
            ``survival_function``, ``median_survival``, ``rmst``, and
            optional CI attributes. For a multi-model selection: a dict of
            those containers keyed by model name, in CV-sorted order
            (best first).

        Raises
        ------
        ValueError
            If ``ci=True`` and ``n_bootstrap`` is less than 1, or if
            ``ci=True`` and ``ci_level`` is not strictly between 0 and 1.
            Also for an unknown model name, an empty list/dict selection,
            or ``None`` with no trained models.
        TypeError
            If ``models`` is a raw model/pipeline (or other unsupported
            type) — pass a trained model name instead.
        RuntimeError
            If every model in a multi-model selection fails to predict.
            (A failure of just some models skips them with a warning.)

        Examples
        --------
        >>> pred = learner.predict(X_new, models='rsf')
        >>> pred.risk_scores
        array([1.23, 0.87, 2.45])
        >>> pred.median_survival
        array([36.2, 48.1, 18.7])

        >>> pred = learner.predict(X_new, models='rsf', times=[12, 24, 60])
        >>> pred.survival_at_times
              t_12   t_24   t_60
        0    0.85   0.72   0.42
        1    0.91   0.83   0.55

        >>> pred = learner.predict(X_new, models='rsf', ci=True, n_bootstrap=50)
        >>> pred.median_survival_ci
           median  ci_lower  ci_upper
        0   36.2     28.1     45.8

        >>> preds = learner.predict(X_new, models=['coxph', 'rsf'])
        >>> preds['coxph'].risk_scores, preds['rsf'].risk_scores
        (array([1.10, 0.92, 2.31]), array([1.23, 0.87, 2.45]))
        """
        # -- Validation ------------------------------------------------------
        if band_calibration not in ('refit', 'fixed'):
            raise ValueError(
                f"band_calibration must be 'refit' or 'fixed', "
                f"got {band_calibration!r}."
            )
        if ci:
            if n_bootstrap < 1:
                raise ValueError(
                    f"n_bootstrap must be >= 1 when ci=True, got {n_bootstrap}."
                )
            if not (0 < ci_level < 1):
                raise ValueError(
                    f"ci_level must be between 0 and 1 (exclusive), "
                    f"got {ci_level}. "
                    "Use e.g. 0.95 for 95% confidence intervals."
                )
        if ci_band and not ci:
            warnings.warn(
                "ci_band=True has no effect when ci=False; CI bands require "
                "bootstrap (ci=True). Ignoring ci_band. Pass ci=True to compute "
                "survival-function CI bands.",
                SuggestionWarning,
                stacklevel=2,
            )
        if not ci:
            passed = {
                "n_bootstrap": n_bootstrap,
                "ci_level": ci_level,
                "random_state": random_state,
                "n_jobs": n_jobs,
            }
            ignored = [
                name for name, value in passed.items()
                if value != _CI_ONLY_PREDICT_DEFAULTS[name]
            ]
            if ignored:
                names = ", ".join(ignored)
                plural = len(ignored) > 1
                warnings.warn(
                    f"{names} ha{'ve' if plural else 's'} no effect when "
                    f"ci=False and {'were' if plural else 'was'} ignored; these "
                    "parameters configure bootstrap confidence intervals. Pass "
                    "ci=True to compute CIs.",
                    SuggestionWarning,
                    stacklevel=2,
                )
        self._ensure_setup()
        resolved, single_form = self._resolve_models_multi(models)
        X = self._validate_predict_input(X)

        predict_kwargs = dict(
            times=times, ci=ci, ci_band=ci_band, n_bootstrap=n_bootstrap,
            ci_level=ci_level, random_state=random_state, n_jobs=n_jobs,
            verbose=verbose, use_recalibrated=use_recalibrated,
            band_calibration=band_calibration,
        )

        if single_form:
            return self._predict_single(resolved[0], X, **predict_kwargs)

        results: Dict[str, PredictionResult] = {}
        failures = {}
        for mr in resolved:
            try:
                results[mr.name] = self._predict_single(
                    mr, X, **predict_kwargs,
                )
            except (RuntimeError, NotImplementedError,
                    np.linalg.LinAlgError) as exc:
                # Skip this model so the others still predict.
                # ValueError/TypeError are treated as input errors and
                # propagate — their common cause is malformed X/times,
                # which would fail for every model.
                failures[mr.name] = exc
                warnings.warn(
                    f"Prediction failed for model '{mr.name}' "
                    f"({type(exc).__name__}: {exc}); model skipped.",
                    MethodologicalWarning,
                    stacklevel=2,
                )
        if not results:
            details = "; ".join(
                f"{name}: {type(exc).__name__}: {exc}"
                for name, exc in failures.items()
            )
            raise RuntimeError(f"predict() failed for all models ({details}).")
        return results

    def _predict_single(
        self: SurvivalLearner,
        model_result,
        X: pd.DataFrame,
        *,
        times=None,
        ci: bool = False,
        ci_band: bool = False,
        n_bootstrap: int = 200,
        ci_level: float = 0.95,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        verbose: bool = True,
        use_recalibrated: bool = True,
        band_calibration: str = 'refit',
    ) -> "PredictionResult":
        """Predict with one resolved model (the per-model :meth:`predict` worker).

        ``X`` must already be validated via ``_validate_predict_input``;
        the parameter-combination warnings also live in :meth:`predict`
        so they fire once per call, not once per model.
        """
        from mlsurv.evaluation._prediction_helpers import predict_from_pipeline
        from mlsurv.results._prediction import PredictionResult

        pipeline = model_result.model
        n_patients = len(X)

        if verbose and ci:
            print(
                f"Predicting for {n_patients} patient(s) with "
                f"{model_result.name} (bootstrap CIs: {n_bootstrap} iterations)..."
            )

        # -- Core prediction (shared with CLI) -------------------------------
        rmst_tau = float(self._state.y_train['time'].max())
        # When a recalibrator will overwrite S(t|x) below, the raw
        # median-undefined count is internal bookkeeping the user never sees
        # (the post-recalibration count is what they should act on). Suppress
        # the raw warning to avoid duplicate emission per predict() call.
        recal_will_apply = use_recalibrated and self._has_calibrator(
            model_result.name
        )
        result = predict_from_pipeline(
            pipeline, X,
            model_name=model_result.name,
            times=times,
            rmst_default_tau=rmst_tau,
            warn_undefined_median=not recal_will_apply,
        )

        # -- Bootstrap CIs (learner-only, needs training data) ---------------
        if ci and result._surv_probs is None:
            warnings.warn(
                f"Model '{model_result.name}' does not support "
                "predict_survival_function(). ci parameter is ignored; "
                "only risk_scores are available.",
                SuggestionWarning,
                stacklevel=2,
            )
        if ci and result._surv_probs is not None:
            # Calibrate each bootstrap iteration's S(t|x) so the CI band sits
            # on the same scale as the calibrated point estimate (the point
            # estimate is calibrated below via
            # _apply_recalibration_to_prediction). calib_plan is None when the
            # model has no internal calibrator, leaving the band on the raw
            # scale.
            calib_plan = (
                self._build_prediction_calib_plan(
                    model_result, band_calibration,
                )
                if use_recalibrated else None
            )
            bootstrap_results = self._run_prediction_bootstrap(
                X, model_result, result._time_grid, times,
                result._rmst_default_tau,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=verbose,
                compute_surv_band=ci_band,
                calib_plan=calib_plan,
            )
            result = PredictionResult(
                risk_scores=result.risk_scores,
                model_name=result.model_name,
                n_patients=result.n_patients,
                _time_grid=result._time_grid,
                _surv_probs=result._surv_probs,
                _median_survival=result._median_survival,
                _rmst_values=result._rmst_values,
                _rmst_default_tau=result._rmst_default_tau,
                _user_times=result._user_times,
                _surv_at_user_times=result._surv_at_user_times,
                _rmst_at_user_times=result._rmst_at_user_times,
                _bootstrap_medians=bootstrap_results.get('medians'),
                _bootstrap_surv_at_times=bootstrap_results.get('surv_at_times'),
                _bootstrap_rmst=bootstrap_results.get('rmst'),
                _bootstrap_rmst_at_times=bootstrap_results.get('rmst_at_times'),
                _bootstrap_surv_probs=bootstrap_results.get('surv_probs'),
                _ci_level=ci_level,
            )

        # Apply the attached calibrator to the point-estimate S(t|x) and
        # re-derive median/RMST when the selected model has one. The bootstrap
        # band was already calibrated per iteration (see calib_plan above), so
        # the point estimate and CI band live on the same scale.
        if use_recalibrated:
            self._apply_recalibration_to_prediction(result, model_result.name, X=X)

        # Attach training-cohort context so result.summary() / patient_report()
        # can describe this patient relative to the training distribution
        # without re-querying the learner.
        self._attach_cohort_context(result, pipeline, model_result, X)

        return result

    # -- Private helpers -----------------------------------------------------

    def _attach_cohort_context(
        self: SurvivalLearner,
        result,
        pipeline,
        model_result,
        X,
    ):
        """Populate training-cohort context fields on a PredictionResult.

        Sets: ``training_cohort_size``, ``feature_names``, ``feature_bounds``,
        ``training_risk_scores``, ``training_y``, ``test_c_index`` (if
        ``evaluate()`` has run), ``imputation_mask`` (from ``X.isna()``), and
        ``input_X``.  Errors during risk-score collection are silently skipped
        (narrative fields are best-effort).
        """
        X_train = self._state.X_train
        y_train = self._state.y_train

        result.training_cohort_size = len(X_train)
        result.feature_names = list(X_train.columns)
        result.training_X = X_train
        result.training_y = y_train
        result.input_X = X
        result.time_unit = getattr(self._state, "time_unit", None)

        # NaN inputs -> imputation_mask. We don't run the imputer — we just
        # flag which cells the user supplied as NaN (will be filled by whatever
        # imputer the pipeline contains). If there are no NaNs, leave the
        # mask as None so rendering code can skip the banner cleanly.
        na_mask = X.isna()
        if na_mask.to_numpy().any():
            result.imputation_mask = na_mask

        # Per-feature training bounds for OOD checks. Only numeric columns.
        bounds = {}
        for col in X_train.columns:
            s = X_train[col]
            try:
                lo = float(s.min())
                hi = float(s.max())
            except (TypeError, ValueError):
                continue
            if np.isnan(lo) or np.isnan(hi):
                continue
            bounds[col] = (lo, hi)
        if bounds:
            result.feature_bounds = bounds

        # Training risk-score distribution (one extra pipeline.predict on
        # X_train). Powers percentile rank and KNN anchor.
        try:
            train_risk = np.asarray(pipeline.predict(X_train))
            if train_risk.ndim == 1 and train_risk.shape[0] == len(X_train):
                result.training_risk_scores = train_risk
        except (AttributeError, ValueError, RuntimeError):
            pass

        # Test C-index — only available after learner.evaluate() has run.
        evaluation = getattr(model_result, "evaluation", None)
        if evaluation is not None:
            c = getattr(evaluation, "c_index_test", None)
            if c is not None:
                try:
                    result.test_c_index = float(c)
                except (TypeError, ValueError):
                    pass

    def _validate_predict_input(self: SurvivalLearner, X):
        """Validate and align X for prediction."""
        from mlsurv.learner._evaluation import _align_columns

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"X must be a pandas DataFrame, got {type(X).__name__}."
            )
        if len(X) == 0:
            raise ValueError("X must contain at least 1 row.")

        train_cols = list(self._state.X_train.columns)
        return _align_columns(X, train_cols, context="Prediction data",
                              stacklevel=4)

    def _build_prediction_calib_plan(
        self: SurvivalLearner, model_result, band_calibration='refit',
    ):
        """Resolve how each bootstrap iteration should calibrate its S(t|x).

        Takes the resolved ``ModelResult`` (not a name string — the calibrator
        rides on the result, so there's no need to re-resolve through
        ``_state.models``). Returns a picklable ``calib_plan`` dict consumed by
        ``_build_iter_calibrate`` in the bootstrap worker, or ``None`` when the
        model has no internal calibrator (the band stays raw). The calibrator
        surface is the single ``ModelResult.calibrator`` (from
        ``train(calibrate=...)``), so this mirrors the point-estimate path in
        ``_apply_recalibration_to_prediction``.

        Two plan shapes feed ``_build_iter_calibrate``:

        * a ``'fixed'`` plan ``{'method', 'calibrator'}`` carries the
          already-fitted map, reapplied unchanged on every resample;
        * a ``'refit'`` plan ``{'method', 'horizons', 'splitter', 'model_name'}``
          carries the recipe to refit the calibrator on each resample's own OOF
          — full-pipeline replication, so the band gains the calibration step's
          sampling variability.

        ``band_calibration`` (``'refit'`` default, or ``'fixed'``) selects which
        shape the warpers (``isotonic``, ``lowdof``) get. ``breslow_slope``
        always takes the refit shape: it recomputes ``S(t|x)`` from each
        resample's refitted scores rather than mapping stored probabilities, so
        a stored baseline can't be reused, and it has no fixed-map form.
        """
        cal = getattr(model_result, "calibrator", None)
        if cal is None:
            return None

        def _refit_plan(method):
            """Refit recipe for *method*, or None if no usable OOF splitter."""
            base_splitter = (
                self._state.cv_splitter_single
                if self._state.cv_splitter_single is not None
                else self._state.cv_splitter
            )
            splitter = self._calibration_splitter(
                cal, base_splitter, model_result.name,
            )
            if splitter is None:
                return None
            return {
                'method': method,
                'horizons': self._resolve_recalibration_horizons(None),
                'splitter': splitter,
                'model_name': model_result.name,
            }

        if cal.method in ('isotonic', 'lowdof'):
            if band_calibration == 'fixed':
                return {'method': cal.method, 'calibrator': cal}
            # 'refit': replicate the warper fit per resample.
            refit = _refit_plan(cal.method)
            if refit is not None:
                return refit
            # No OOF splitter available: keep the band on the calibrated scale
            # with the fixed map rather than dropping it to raw, but tell the
            # user the per-resample refit they asked for (and its calibration-
            # fit variability) was skipped.
            warnings.warn(
                f"band_calibration='refit' requested for '{model_result.name}' "
                f"but no out-of-fold splitter is available; reapplying the fixed "
                f"calibration map instead, so the CI band omits calibration-fit "
                f"variability.",
                SuggestionWarning, stacklevel=2,
            )
            return {'method': cal.method, 'calibrator': cal}
        if cal.method == 'breslow_slope':
            return _refit_plan('breslow_slope')
        return None

    def _run_prediction_bootstrap(
        self: SurvivalLearner,
        X_new,
        model_result,
        time_grid,
        user_times,
        rmst_default_tau,
        n_bootstrap,
        random_state,
        n_jobs,
        verbose,
        compute_surv_band=False,
        calib_plan=None,
    ):
        """Run bootstrap retraining loop for prediction CIs."""
        from mlsurv.evaluation._prediction_bootstrap import (
            run_prediction_bootstrap,
        )

        if random_state is None:
            random_state = self._state.random_state

        # Pass the unfitted final-step estimator directly — the bootstrap
        # iteration clones it on every iteration, so we don't pre-clone
        # here. Passing the object instead of a registry name is what
        # keeps tuned models correct (see the State Consistency guidance
        # in ``CONTRIBUTING.md``).
        pipeline = model_result.model
        model_proto = pipeline.named_steps['model']
        base_name = model_result.base_name

        # Get unfitted preprocessing components from state
        imputer, scaler, selector = self._resolve_model_preprocessing(
            model_result.name,
        )

        return run_prediction_bootstrap(
            X_train=self._state.X_train,
            y_train=self._state.y_train,
            X_new=X_new,
            model_name=base_name,
            model_proto=model_proto,
            imputer=imputer,
            scaler=scaler,
            selector=selector,
            time_grid=time_grid,
            rmst_default_tau=rmst_default_tau,
            user_times=user_times,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            compute_surv_band=compute_surv_band,
            calib_plan=calib_plan,
        )
