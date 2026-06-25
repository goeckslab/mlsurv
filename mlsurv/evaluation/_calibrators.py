"""Internal S(t|x) calibrators for ``train(calibrate=...)``.

A calibrator is fit on **held-out out-of-fold (OOF)** predictions of the
training cohort and attached to a ``ModelResult`` (``result.calibrator``). It
is then applied to the downstream survival probabilities of evaluate, bootstrap,
and predict, so the deployed model's ``S(t|x)`` is well calibrated (and the
report's metric tables, which read evaluate, follow). The standalone calibration
plots (``plot_calibration``, ``plot_calibration_overlay``), the metric-time
Brier(t) panel, and the report's calibration figures also calibrate by default,
via :func:`apply_calibrator_safe` (the ``viz`` layer reaches the attached
calibrator through ``LearnerState`` rather than calling the learner).

Two kinds of calibrator share one interface:

* **Probability-warpers** (:class:`IsotonicCalibrator`, :class:`LowDoFCalibrator`)
  map raw ``S(t|x)`` probabilities to calibrated ones. They are monotone, so
  rank metrics (C-index) are preserved, and model-agnostic.
* **Score -> baseline** (:class:`BreslowSlopeCalibrator`) ignores the raw
  probabilities and recomputes ``S(t|x)`` from a Breslow baseline refit on the
  model's single linear-predictor risk score (van Houwelingen 2000). It only
  applies to score-based models.

The single ``apply(*, pipeline, X, surv_probs, eval_times)`` signature subsumes
both: warpers read ``surv_probs``; score->baseline reads ``pipeline.predict(X)``.

The numeric recipes (low-DoF cloglog, Breslow slope refit) are lifted from
``scripts/benchmark_calibrators.py``; the isotonic and IPCW helpers are reused
from the sibling :mod:`mlsurv.evaluation._recalibration` module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .._warnings import MethodologicalWarning, SuggestionWarning
from ._recalibration import (
    _apply_recalibrators_to_array,
    _fit_censoring_distribution,
    _fit_isotonic_recalibrators,
    _ipcw_calibration_weights,
)

# Clip survival probabilities away from {0, 1} before the cloglog transform so
# ``log(-log S)`` stays finite. Matches the benchmark prototype.
_EPS = 1e-6

# Sentinel for an unset ``calibrate=`` argument (apply the default registry).
# Defined here so ``evaluation`` does not import from ``learner``.
_CALIBRATE_UNSET = object()


class SurvivalCalibrator(ABC):
    """Base class for an attached, OOF-fit ``S(t|x)`` calibrator.

    Subclasses are plain-pickle-safe (no closures/lambdas stored on them) so
    they round-trip through ``save_learner`` / ``load_learner`` as part of the
    ``ModelResult``.
    """

    #: Registry / grammar key (e.g. ``"isotonic"``).
    method: str = ""
    #: True when ``apply`` needs ``pipeline.predict(X)`` rather than the raw
    #: probabilities (score -> baseline calibrators).
    requires_scores: bool = False
    #: Number of OOF folds the learner should use when collecting predictions
    #: for this calibrator. ``None`` inherits the learner's CV splitter.
    oof_cv_folds: Optional[int] = None

    def __init__(self) -> None:
        self.horizons: List[float] = []

    @abstractmethod
    def fit(
        self,
        *,
        y_train: Any,
        horizons: np.ndarray,
        oof_S: Optional[np.ndarray] = None,
        oof_scores: Optional[np.ndarray] = None,
        model_name: str = "model",
    ) -> "SurvivalCalibrator":
        """Fit on held-out OOF predictions of the training cohort."""

    @abstractmethod
    def apply(
        self,
        *,
        pipeline: Any,
        X: Any,
        surv_probs: Optional[np.ndarray],
        eval_times: Iterable[float],
    ) -> np.ndarray:
        """Return calibrated ``S(t|x)`` of shape ``(len(X), len(eval_times))``."""


def apply_calibrator_safe(
    calibrator: Optional[SurvivalCalibrator],
    *,
    pipeline: Any,
    X: Any,
    surv_probs: Optional[np.ndarray],
    eval_times: Iterable[float],
    model_name: str,
) -> np.ndarray:
    """Apply *calibrator* to a 2D ``S(t|x)`` array, raw fallback on failure.

    Shared by ``SurvivalLearner._apply_calibrator`` and the viz/report
    data-prep so every surface (predict, evaluate, plots, report) gets
    identical calibrate-or-raw behaviour without the ``viz`` layer importing
    ``mlsurv.learner`` (the one-way ``learner -> viz`` rule). When *calibrator*
    is ``None`` the raw *surv_probs* are returned unchanged. A calibrator whose
    ``apply`` raises is isolated: the failure is downgraded to a
    ``MethodologicalWarning`` and raw ``S(t|x)`` is returned for that model, so
    one model can't take down a multi-model plot / evaluate.

    *pipeline* and *X* are required by score->baseline calibrators
    (:class:`BreslowSlopeCalibrator`), which ignore *surv_probs* and recompute
    from ``pipeline.predict(X)``; the probability-warpers ignore them.
    """
    if calibrator is None:
        return np.asarray(surv_probs, dtype=float)
    import warnings

    try:
        calibrated = calibrator.apply(
            pipeline=pipeline, X=X,
            surv_probs=surv_probs, eval_times=eval_times,
        )
    except (ValueError, RuntimeError, FloatingPointError, TypeError) as exc:
        warnings.warn(
            f"Calibrator for '{model_name}' failed to apply "
            f"({type(exc).__name__}: {exc}); reporting raw S(t|x) for this "
            f"model.",
            MethodologicalWarning, stacklevel=2,
        )
        return np.asarray(surv_probs, dtype=float)
    calibrated = np.asarray(calibrated, dtype=float)
    # Guard a successful-but-garbage return: a calibrator that runs without
    # raising but yields a mis-shaped array, or introduces non-finite values
    # where the raw S(t|x) was finite, must not silently corrupt the curve.
    # (surv_probs is None for breslow_slope, which recomputes from scores —
    # there is no raw array to compare against, so the check is skipped.)
    if surv_probs is not None:
        raw = np.asarray(surv_probs, dtype=float)
        introduced_nonfinite = (
            calibrated.shape == raw.shape
            and bool((~np.isfinite(calibrated) & np.isfinite(raw)).any())
        )
        if calibrated.shape != raw.shape or introduced_nonfinite:
            warnings.warn(
                f"Calibrator for '{model_name}' returned a non-finite or "
                f"mis-shaped S(t|x) ({calibrated.shape} vs raw {raw.shape}); "
                f"reporting raw S(t|x) for this model.",
                MethodologicalWarning, stacklevel=2,
            )
            return raw
    return calibrated


class IsotonicCalibrator(SurvivalCalibrator):
    """Per-horizon isotonic regression on ``S(t|x)`` (IPCW-weighted).

    The per-horizon isotonic ``S(t|x)`` mapping. Rank-unsafe in general
    (it collapses predictions onto plateaus, creating ties), so it is a default
    only for deephit, whose miscalibration is too severe for low-DoF.
    """

    method = "isotonic"
    requires_scores = False

    def __init__(self) -> None:
        super().__init__()
        self.mappings: Dict[float, Any] = {}

    def fit(self, *, y_train, horizons, oof_S=None, oof_scores=None,
            model_name="model"):
        if oof_S is None:
            raise ValueError("IsotonicCalibrator.fit requires oof_S.")
        horizons = np.asarray(horizons, dtype=float)
        per_horizon, dropped = _fit_isotonic_recalibrators(oof_S, horizons, y_train)
        self.mappings = per_horizon
        self.horizons = sorted(per_horizon.keys())
        if not per_horizon:
            warn_no_horizons(model_name, self.method)
        elif dropped:
            warn_dropped_horizons(model_name, self.method, dropped)
        return self

    def apply(self, *, pipeline, X, surv_probs, eval_times):
        if surv_probs is None:
            raise ValueError("IsotonicCalibrator.apply requires surv_probs.")
        if not self.mappings:
            return np.asarray(surv_probs, dtype=float)
        return _apply_recalibrators_to_array(self.mappings, surv_probs, eval_times)


class LowDoFCalibrator(SurvivalCalibrator):
    """Two-parameter cloglog recalibration ``S' = exp(-exp(a + b*log(-log S)))``.

    Fit by IPCW-weighted maximum likelihood on pooled OOF predictions. Monotone
    and model-agnostic; the default for rsf.
    """

    method = "lowdof"
    requires_scores = False

    def __init__(self) -> None:
        super().__init__()
        self.a: float = 0.0
        self.b: float = 1.0

    def fit(self, *, y_train, horizons, oof_S=None, oof_scores=None,
            model_name="model"):
        if oof_S is None:
            raise ValueError("LowDoFCalibrator.fit requires oof_S.")
        horizons = np.asarray(horizons, dtype=float)
        self.a, self.b = _fit_lowdof(oof_S, horizons, y_train, model_name=model_name)
        self.horizons = [float(h) for h in horizons]
        return self

    def apply(self, *, pipeline, X, surv_probs, eval_times):
        if surv_probs is None:
            raise ValueError("LowDoFCalibrator.apply requires surv_probs.")
        return _apply_lowdof(self.a, self.b, np.asarray(surv_probs, float),
                             np.asarray(list(eval_times), float))


class BreslowSlopeCalibrator(SurvivalCalibrator):
    """Breslow baseline refit on OOF risk scores, scaled by a Cox slope.

    This is van Houwelingen's full recalibration: a CoxPH model is fit on the
    held-out risk scores to recover the slope ``beta``, and a Breslow baseline
    is refit on ``beta * scores``. Score-based models only. The slope refit is
    essential — a level-only Breslow (``beta = 1``) reproduces the raw scale.

    OOF is pinned to 5 folds (``oof_cv_folds = 5``): the fssvm equivalence
    holds at 5 folds; 3 folds under-fit the slope.
    """

    method = "breslow_slope"
    requires_scores = True
    oof_cv_folds = 5

    def __init__(self) -> None:
        super().__init__()
        self.beta: float = 1.0
        self.baseline_: Any = None
        self._model_name: str = "model"

    def fit(self, *, y_train, horizons, oof_S=None, oof_scores=None,
            model_name="model"):
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.linear_model.coxph import BreslowEstimator
        from ..models._overflow import breslow_call_with_clamp

        if oof_scores is None:
            raise ValueError("BreslowSlopeCalibrator.fit requires oof_scores.")
        scores = np.asarray(oof_scores, dtype=float)
        finite = np.isfinite(scores)
        if finite.sum() < 2:
            raise ValueError(
                f"breslow_slope calibration of '{model_name}' needs at least 2 "
                f"finite OOF scores; got {int(finite.sum())}.")
        s = scores[finite]
        y = y_train[finite]

        # Recover the calibration slope from a Cox fit on the OOF scores. On
        # failure, fall back to a level-only Breslow (slope fixed at 1).
        try:
            cox = CoxPHSurvivalAnalysis(alpha=1e-6).fit(s.reshape(-1, 1), y)
            self.beta = float(cox.coef_[0])
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            self.beta = 1.0
            import warnings
            warnings.warn(
                f"breslow_slope: Cox slope refit failed for '{model_name}' "
                f"({type(exc).__name__}); using a level-only Breslow baseline "
                f"(slope=1).",
                MethodologicalWarning, stacklevel=2)

        self.baseline_ = breslow_call_with_clamp(
            self.beta * s, BreslowEstimator().fit,
            y["event"], y["time"], model_name=model_name)
        self._model_name = model_name
        self.horizons = [float(h) for h in np.asarray(horizons, dtype=float)]
        return self

    def apply(self, *, pipeline, X, surv_probs, eval_times):
        from ..models._overflow import breslow_call_with_clamp

        if self.baseline_ is None:
            raise ValueError("BreslowSlopeCalibrator.apply before fit.")
        scores = np.asarray(pipeline.predict(X), dtype=float)
        eval_times = np.asarray(list(eval_times), dtype=float)
        # Pass the real model name so the non-finite-output warning dedup key
        # in breslow_call_with_clamp does not collide across models.
        sfns = breslow_call_with_clamp(
            self.beta * scores, self.baseline_.get_survival_function,
            model_name=self._model_name)
        return np.array([[float(fn(t)) for t in eval_times] for fn in sfns])


# ---------------------------------------------------------------------------
# Numeric recipes (ported from scripts/benchmark_calibrators.py)
# ---------------------------------------------------------------------------

def _fit_lowdof(oof_S: np.ndarray, horizons: np.ndarray, y_train: Any,
                model_name: str = "model") -> tuple:
    """Fit ``(a, b)`` for ``S' = exp(-exp(a + b*log(-log S)))`` by IPCW NLL.

    Returns the identity map ``(0.0, 1.0)`` (a transparent no-op, so the model
    stays raw rather than mis-warped) with a ``MethodologicalWarning`` when no
    usable OOF samples remain or the optimizer fails to converge.
    """
    import warnings
    from scipy.optimize import minimize

    t = np.asarray(y_train["time"])
    e = np.asarray(y_train["event"]).astype(bool)
    cde = _fit_censoring_distribution(y_train)
    log_h, fail, wts = [], [], []
    for j, h in enumerate(horizons):
        col = oof_S[:, j]
        S = np.clip(col, _EPS, 1 - _EPS)
        ok = np.isfinite(col)
        w = _ipcw_calibration_weights(t, e, float(h), cde)
        m = ok & (w > 0)
        log_h.append(np.log(-np.log(S[m])))
        fail.append((t[m] <= float(h)).astype(float))   # I[T <= h]
        wts.append(w[m])
    lh = np.concatenate(log_h)
    fl = np.concatenate(fail)
    w = np.concatenate(wts)

    if lh.size == 0:
        warnings.warn(
            f"lowdof calibration of '{model_name}': no usable OOF samples "
            f"(all horizons masked out); leaving the model uncalibrated.",
            SuggestionWarning, stacklevel=2)
        return 0.0, 1.0

    def nll(ab):
        a, b = ab
        eta = a + b * lh
        mu = np.clip(1 - np.exp(-np.exp(np.clip(eta, -30, 30))), _EPS, 1 - _EPS)
        return -np.sum(w * (fl * np.log(mu) + (1 - fl) * np.log(1 - mu)))

    res = minimize(nll, x0=[0.0, 1.0], method="L-BFGS-B",
                   bounds=[(-10, 10), (0.0, 10)])
    if not res.success or not np.isfinite(res.x).all():
        warnings.warn(
            f"lowdof calibration of '{model_name}': optimizer did not converge "
            f"({res.message}); leaving the model uncalibrated (identity map).",
            MethodologicalWarning, stacklevel=2)
        return 0.0, 1.0
    return float(res.x[0]), float(res.x[1])


def _apply_lowdof(a: float, b: float, S2d: np.ndarray,
                  horizons: np.ndarray) -> np.ndarray:
    """Apply the cloglog map to a 2D ``S(t|x)`` array, enforcing monotonicity."""
    S = np.clip(S2d, _EPS, 1 - _EPS)
    eta = a + b * np.log(-np.log(S))
    out = np.exp(-np.exp(np.clip(eta, -30, 30)))
    out = np.clip(out, 0.0, 1.0)
    order = np.argsort(horizons)
    so = out[:, order]
    np.minimum.accumulate(so, axis=1, out=so)
    return so[:, np.argsort(order)]


# ---------------------------------------------------------------------------
# Registry, defaults, and the ``calibrate=`` grammar resolver
# ---------------------------------------------------------------------------

_CALIBRATOR_REGISTRY: Dict[str, type] = {
    "isotonic": IsotonicCalibrator,
    "lowdof": LowDoFCalibrator,
    "breslow_slope": BreslowSlopeCalibrator,
}

# Per-model default calibrators (the rest stay raw). fssvm and deephit gain
# uniformly and substantially from calibration in the benchmark; rsf's lowdof
# gain is only a cross-dataset median (it regresses on several datasets,
# including gbsg2/veterans/whas500), so rsf stays raw by default and users opt
# in explicitly with train(calibrate={'rsf': 'lowdof'}).
_DEFAULT_CALIBRATION: Dict[str, str] = {
    "fssvm": "breslow_slope",
    "deephit": "isotonic",
}


def create_calibrator(method: str) -> SurvivalCalibrator:
    """Instantiate the calibrator class registered under *method*."""
    try:
        return _CALIBRATOR_REGISTRY[method]()
    except KeyError:
        raise ValueError(
            f"Unknown calibration method '{method}'. Available: "
            f"{sorted(_CALIBRATOR_REGISTRY)}.") from None


def resolve_calibration_spec(
    calibrate: Any,
    model_names: Iterable[str],
    score_based: Dict[str, bool],
) -> Dict[str, Optional[str]]:
    """Resolve a ``calibrate=`` argument to ``{model_name: method | None}``.

    Grammar:

    * unset (sentinel) -> the default registry, others ``None``
    * ``False`` / ``None`` -> all ``None`` (raw)
    * a method string -> that method for every targeted model
    * a dict ``{model: method}`` -> per-model; missing keys map to ``None``
      (an explicit dict means explicit control, not the registry)

    Raises ``ValueError`` for an unknown method or an inapplicable
    model<->calibrator pair (e.g. ``breslow_slope`` on a non-score model).
    """
    names = list(model_names)
    if calibrate is _CALIBRATE_UNSET:
        spec = {n: _DEFAULT_CALIBRATION.get(n) for n in names}
    elif calibrate is False or calibrate is None:
        spec = {n: None for n in names}
    elif isinstance(calibrate, str):
        spec = {n: calibrate for n in names}
    elif isinstance(calibrate, dict):
        unknown = set(calibrate) - set(names)
        if unknown:
            import warnings
            warnings.warn(
                f"calibrate=: ignoring keys for models not being trained: "
                f"{sorted(unknown)}.",
                SuggestionWarning, stacklevel=2)
        spec = {n: calibrate.get(n) for n in names}
    else:
        raise TypeError(
            "calibrate= must be unset, False/None, a method string, or a "
            f"{{model: method}} dict; got {type(calibrate).__name__}.")

    for name, method in spec.items():
        if method is None:
            continue
        if method not in _CALIBRATOR_REGISTRY:
            raise ValueError(
                f"Unknown calibration method '{method}' for model '{name}'. "
                f"Available: {sorted(_CALIBRATOR_REGISTRY)}.")
        cls = _CALIBRATOR_REGISTRY[method]
        if cls.requires_scores and not score_based.get(name, False):
            raise ValueError(
                f"'{method}' is only valid for score->baseline models (a single "
                f"linear predictor); '{name}' is not score-based. Use 'isotonic' "
                f"or 'lowdof' instead.")
    return spec


# ---------------------------------------------------------------------------
# Shared warnings (kept module-level so messages are uniform)
# ---------------------------------------------------------------------------

def warn_dropped_horizons(model_name: str, method: str, dropped: list) -> None:
    """Warn that some horizons were skipped for lack of usable OOF samples."""
    import warnings
    horizons = ", ".join(f"{h:g}" for h, _ in dropped)
    warnings.warn(
        f"{method} calibration of '{model_name}': {len(dropped)} horizon(s) "
        f"({horizons}) had too few usable OOF samples and were left "
        f"uncalibrated.",
        MethodologicalWarning, stacklevel=2)


def warn_no_horizons(model_name: str, method: str) -> None:
    """Warn that no horizon could be calibrated (calibrator is a no-op)."""
    import warnings
    warnings.warn(
        f"{method} calibration of '{model_name}': no horizon had enough usable "
        f"OOF samples; the model is left uncalibrated.",
        SuggestionWarning, stacklevel=2)
