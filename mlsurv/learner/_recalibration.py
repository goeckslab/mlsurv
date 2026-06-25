"""RecalibrationMixin — internal S(t|x) calibration.

The supported entry point is ``train(calibrate=...)``, which fits a
:class:`~mlsurv.evaluation._calibrators.SurvivalCalibrator` on held-out
out-of-fold (OOF) predictions of the training cohort and attaches it to the
model's ``ModelResult`` (``result.calibrator``). The attached calibrator is the
deployed model's behavior, so it is applied wherever the model produces
``S(t|x)`` (evaluate, bootstrap, predict, validate, subpopulation).

This mixin holds the shared helpers behind that path: fitting and attaching
calibrators (``_attach_calibrators``), applying them to a prediction
(``_apply_calibrator``, ``_apply_recalibration_to_prediction``), and resolving
the horizon grid and OOF splitter. External / transport recalibration (fitting
a new calibrator on a new cohort) is planned as a separate future feature.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

import numpy as np

from .._warnings import SuggestionWarning
from ..evaluation._recalibration import _collect_oof_survival
from ..utils import auto_eval_times
from ._evaluation_helpers import resolve_state_eval_times

if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner


class RecalibrationMixin:
    """Mixin providing internal calibration of survival probabilities."""

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_recalibration_horizons(
        self: "SurvivalLearner",
        horizons: Optional[Sequence[float]],
    ) -> np.ndarray:
        """Default horizons from user_eval_times → auto_eval_times(y_train)."""
        # Explicit > inherited (user_eval_times) tiers via the canonical
        # resolver. persist=False: calibration horizons are conceptually
        # separate from the eval_times grid and must not overwrite it.
        resolved = resolve_state_eval_times(self._state, horizons, persist=False)
        if resolved is None:
            # No explicit horizons and no persisted grid: calibration uses a
            # training-only auto grid (no y_test, no clip — calibrators fit on
            # OOF training predictions).
            resolved = auto_eval_times(self._state.y_train)
        horizons = np.asarray(resolved, dtype=float)
        if horizons.ndim != 1 or horizons.size == 0:
            raise ValueError(
                f"horizons must be a 1D non-empty sequence; got shape {horizons.shape}."
            )
        if not np.all(np.isfinite(horizons)) or np.any(horizons <= 0):
            raise ValueError(
                f"horizons must be positive and finite; got {horizons!r}."
            )
        return np.sort(np.unique(horizons))

    # ------------------------------------------------------------------
    # Helpers (used by predict / patient_report / evaluate / bootstrap)
    # ------------------------------------------------------------------

    def _has_calibrator(self: "SurvivalLearner", model_name: str) -> bool:
        """Return True when *model_name* has an internal calibrator attached.

        The calibrator comes from ``train(calibrate=...)`` and rides on
        ``ModelResult.calibrator``. It drives the deployed-model metrics on
        every prediction surface.
        """
        result = self._state.models.get(model_name)
        return result is not None and getattr(result, "calibrator", None) is not None

    def _apply_calibrator(
        self: "SurvivalLearner",
        model_name: str,
        surv_probs: np.ndarray,
        eval_times: Iterable[float],
        *,
        X: Any = None,
    ) -> np.ndarray:
        """Apply the attached internal calibrator to a 2D ``S(t|x)`` array.

        When *model_name* has no calibrator, returns *surv_probs* unchanged.
        For probability-warping calibrators (isotonic, lowdof) the raw
        *surv_probs* are mapped; for the score->baseline calibrator
        (breslow_slope) they are ignored and ``S(t|x)`` is recomputed from
        ``pipeline.predict(X)``. *X* must match the rows of *surv_probs*; when
        omitted it defaults to the test cohort (correct for evaluate/bootstrap
        full-cohort sites). predict()/plots/validate() pass their own *X*.
        """
        result = self._state.models.get(model_name)
        cal = getattr(result, "calibrator", None) if result is not None else None
        if cal is None:
            return np.asarray(surv_probs, dtype=float)
        cohort_X = X if X is not None else self._state.X_test
        # Delegate to the shared guard so plots / reports get identical
        # calibrate-or-raw (+ MethodologicalWarning) behaviour without the viz
        # layer importing the learner. See evaluation/_calibrators.py.
        from ..evaluation._calibrators import apply_calibrator_safe

        return apply_calibrator_safe(
            cal, pipeline=result.model, X=cohort_X,
            surv_probs=surv_probs, eval_times=eval_times,
            model_name=model_name,
        )

    # ------------------------------------------------------------------
    # train(calibrate=...) — fit and attach internal calibrators
    # ------------------------------------------------------------------

    def _model_is_score_based(self: "SurvivalLearner", result: Any) -> bool:
        """Whether *result*'s fitted model reduces to a single risk score."""
        model = result.model
        estimator = model
        if hasattr(model, "named_steps"):
            estimator = model.named_steps.get("model", model)
        return bool(getattr(estimator, "score_based", False))

    def _calibration_splitter(
        self: "SurvivalLearner", calibrator: Any, base_splitter: Any, model_name: str,
    ) -> Any:
        """Pick the OOF splitter for *calibrator*.

        ``None`` ``oof_cv_folds`` inherits the learner's CV splitter (isotonic,
        lowdof). A pinned count (breslow_slope -> 5) builds a fresh
        ``StratifiedKFold`` on the event indicator, auto-degrading the fold
        count to ``min(folds, n_events, n_censored)`` so small cohorts still
        fit; returns ``None`` (skip) when fewer than 2 of either are available.
        """
        if calibrator.oof_cv_folds is None:
            return base_splitter
        from sklearn.model_selection import StratifiedKFold
        events = np.asarray(self._state.y_train["event"]).astype(bool)
        n_events = int(events.sum())
        n_censored = int((~events).sum())
        k = min(int(calibrator.oof_cv_folds), n_events, n_censored)
        if k < 2:
            warnings.warn(
                f"Skipping {calibrator.method} calibration for '{model_name}': "
                f"needs at least 2 events and 2 censored for a stratified OOF "
                f"split (have {n_events} events, {n_censored} censored).",
                SuggestionWarning, stacklevel=2,
            )
            return None
        return StratifiedKFold(
            n_splits=k, shuffle=True, random_state=self.random_state,
        )

    def _attach_calibrators(
        self: "SurvivalLearner", results: Any, calibrate: Any,
    ) -> Any:
        """Fit and attach internal calibrators per ``train(calibrate=...)``.

        *results* is a single ``ModelResult`` or a ``{name: ModelResult}`` dict
        (as ``train()`` returns). Each targeted model gets a calibrator fit on
        held-out OOF of the training cohort and stored on
        ``ModelResult.calibrator``. A no-op when the resolved spec is all-raw.
        """
        from ..evaluation._calibrators import (
            create_calibrator, resolve_calibration_spec,
        )

        if isinstance(results, dict):
            # Skip failed models (None) and any without a fitted pipeline.
            items = [(n, r) for n, r in results.items()
                     if r is not None and getattr(r, "model", None) is not None]
        elif results is not None and getattr(results, "model", None) is not None:
            items = [(results.name, results)]
        else:
            return results
        if not items:
            return results

        names = [name for name, _ in items]
        score_based = {name: self._model_is_score_based(r) for name, r in items}
        spec = resolve_calibration_spec(calibrate, names, score_based)
        if not any(spec.values()):
            return results

        horizons_arr = self._resolve_recalibration_horizons(None)
        base_splitter = (
            self._state.cv_splitter_single
            if self._state.cv_splitter_single is not None
            else self._state.cv_splitter
        )
        for name, result in items:
            method = spec.get(name)
            if method is None:
                continue
            calibrator = create_calibrator(method)
            splitter = self._calibration_splitter(calibrator, base_splitter, name)
            if splitter is None:
                continue
            # breslow_slope pins its own stratified split on the event indicator
            # (reproduces fssvm's internal recalibration); the warpers inherit
            # the learner's stratify/groups.
            pinned = calibrator.oof_cv_folds is not None
            try:
                oof = _collect_oof_survival(
                    pipeline=result.model,
                    X_train=self._state.X_train,
                    y_train=self._state.y_train,
                    splitter=splitter,
                    horizons=horizons_arr,
                    stratify_array=None if pinned else self._state.stratify_array,
                    groups=None if pinned else self._state.groups,
                    seen_messages=self._state.fit_failure_warned,
                    want_scores=calibrator.requires_scores,
                )
            except (ValueError, RuntimeError) as exc:
                warnings.warn(
                    f"Skipping {method} calibration for '{name}': {exc}",
                    SuggestionWarning, stacklevel=2,
                )
                continue
            oof_S, oof_scores = oof if calibrator.requires_scores else (oof, None)
            try:
                calibrator.fit(
                    y_train=self._state.y_train,
                    horizons=horizons_arr,
                    oof_S=oof_S,
                    oof_scores=oof_scores,
                    model_name=name,
                )
            except (ValueError, RuntimeError) as exc:
                warnings.warn(
                    f"Skipping {method} calibration for '{name}': {exc}",
                    SuggestionWarning, stacklevel=2,
                )
                continue
            result.calibrator = calibrator
        return results

    def _apply_recalibration_to_prediction(
        self: "SurvivalLearner",
        result: Any,
        model_name: str,
        X: Any = None,
    ) -> Any:
        """Calibrate ``S(t|x)`` and re-derive median/RMST/user-times outputs.

        Mutates *result* in place. Applies the attached internal calibrator
        (``train(calibrate=...)``); no-op when none is attached. The
        score->baseline calibrator recomputes from ``pipeline.predict(X)``, so
        the patient features *X* are threaded in. Re-derives the point estimate
        only; the bootstrap CI band is calibrated separately, per iteration,
        inside the prediction bootstrap (see ``_build_prediction_calib_plan``
        and ``_build_iter_calibrate``), so the point estimate and band share
        the calibrated scale.
        """
        if not self._has_calibrator(model_name):
            return result
        if getattr(result, "_surv_probs", None) is None:
            return result

        from ..evaluation._prediction_helpers import (
            compute_patient_median_survival,
            compute_patient_rmst,
            interpolate_surv_at_times,
        )

        time_grid = np.asarray(result._time_grid, dtype=float)
        surv_probs = np.asarray(result._surv_probs, dtype=float)
        recal_probs = self._apply_calibrator(
            model_name, surv_probs, time_grid, X=X,
        )
        result._surv_probs = recal_probs
        result._median_survival = compute_patient_median_survival(
            time_grid, recal_probs,
        )
        if result._rmst_default_tau is not None:
            result._rmst_values = compute_patient_rmst(
                time_grid, recal_probs, float(result._rmst_default_tau),
            )
        if result._user_times is not None:
            user_times = np.asarray(result._user_times, dtype=float)
            result._surv_at_user_times = interpolate_surv_at_times(
                time_grid, recal_probs, user_times,
            )
            result._rmst_at_user_times = np.column_stack([
                compute_patient_rmst(time_grid, recal_probs, float(tau))
                for tau in user_times
            ])
        return result
