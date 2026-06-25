"""Regression test: the PH-test internal Cox fit must not leak raw overflow.

The proportional-hazards test re-fits a lifelines ``CoxPHFitter`` internally
(in ``_fit_cox_with_fallback``). On large-magnitude / collinear designs
lifelines exponentiates the linear predictor and emits a raw
``RuntimeWarning: overflow encountered in exp``. That signal is already
reported through the clean tiered warnings, so the raw copy must be dropped
(routed through ``reemit_non_overflow``, the same guard used at model-fit
sites). See ``mlsurv/models/_overflow.py``.
"""

import warnings

import numpy as np
import pandas as pd


def _well_behaved_survival_data(n=80, seed=0):
    """Small, easily-converging Cox design (so the fit itself succeeds)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
    })
    lp = 0.5 * X["a"]
    time = rng.exponential(np.exp(-lp))
    event = rng.random(n) < 0.7
    y = np.array([(bool(e), max(float(t), 0.01)) for e, t in zip(event, time)],
                 dtype=[("event", bool), ("time", float)])
    return X, y


def test_ph_test_suppresses_overflow_from_internal_cox_fit(monkeypatch):
    """A lifelines overflow ``RuntimeWarning`` raised during the internal Cox
    fit is dropped, not replayed to the user."""
    import lifelines

    from mlsurv.evaluation.metrics import ph_test

    real_cox = lifelines.CoxPHFitter

    class NoisyCox(real_cox):
        # Mimic lifelines' raw overflow warning during fitting, then fit for real.
        def fit(self, *args, **kwargs):
            warnings.warn("overflow encountered in exp", RuntimeWarning)
            return super().fit(*args, **kwargs)

    # _fit_cox_with_fallback does ``from lifelines import CoxPHFitter`` at call
    # time, so patching the module attribute is enough.
    monkeypatch.setattr(lifelines, "CoxPHFitter", NoisyCox)

    X, y = _well_behaved_survival_data()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = ph_test(X, y)

    overflow = [
        w for w in recorded
        if issubclass(w.category, RuntimeWarning) and "overflow" in str(w.message)
    ]
    assert not overflow, (
        "raw overflow RuntimeWarning leaked from the PH-test Cox fit: "
        f"{[str(w.message) for w in overflow]}"
    )
    # The diagnostic itself still ran (the fit succeeded on well-behaved data).
    assert not result.empty


def test_ph_test_still_reemits_non_overflow_warning(monkeypatch):
    """The overflow filter must NOT swallow genuine soft diagnostics. A
    lifelines ``ConvergenceWarning`` (collinearity / separation) carries
    actionable signal and its message does not match the overflow /
    ill-conditioned / sksurv-convergence patterns, so it must still reach the
    user (No-Silent-Data-Loss)."""
    import lifelines
    from lifelines.exceptions import ConvergenceWarning

    from mlsurv.evaluation.metrics import ph_test

    real_cox = lifelines.CoxPHFitter

    class CollinearWarnCox(real_cox):
        def fit(self, *args, **kwargs):
            warnings.warn(
                "Newton-Raphson convergence completed but norm(delta) is still "
                "high. Perhaps there is collinearity or complete separation.",
                ConvergenceWarning,
            )
            return super().fit(*args, **kwargs)

    monkeypatch.setattr(lifelines, "CoxPHFitter", CollinearWarnCox)

    X, y = _well_behaved_survival_data()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        ph_test(X, y)

    survived = [w for w in recorded if issubclass(w.category, ConvergenceWarning)]
    assert survived, "lifelines ConvergenceWarning was wrongly suppressed"


def test_ph_test_suppresses_overflow_from_schoenfeld_residuals(monkeypatch):
    """The real leak site: lifelines raises a numpy overflow ``RuntimeWarning``
    from ``proportional_hazard_test`` while forming Schoenfeld residuals AFTER
    the fit returns, so it falls outside the fit guard. The ``np.errstate``
    guard around the call must silence that numpy floating-point overflow."""
    import lifelines.statistics as lstats

    from mlsurv.evaluation.metrics import ph_test

    real_pht = lstats.proportional_hazard_test

    def noisy_pht(*args, **kwargs):
        # Reproduce the same numpy ufunc overflow lifelines hits in
        # exp(dot(X, params)); the np.errstate(over="ignore") guard must
        # suppress it at the floating-point layer.
        np.array([1e308, 1e308], dtype=float) ** 2
        return real_pht(*args, **kwargs)

    # _ph_test_core does ``from lifelines.statistics import
    # proportional_hazard_test`` at call time, so patching the module attribute
    # rebinds the name it imports.
    monkeypatch.setattr(lstats, "proportional_hazard_test", noisy_pht)

    X, y = _well_behaved_survival_data()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = ph_test(X, y)

    overflow = [
        w for w in recorded
        if issubclass(w.category, RuntimeWarning) and "overflow" in str(w.message)
    ]
    assert not overflow, (
        "numpy overflow RuntimeWarning leaked from the Schoenfeld/PH-test path: "
        f"{[str(w.message) for w in overflow]}"
    )
    assert not result.empty


def test_ph_test_preserves_warning_when_residual_computation_raises(monkeypatch):
    """A genuine diagnostic emitted just before the Schoenfeld computation
    raises must still reach the user. The ``np.errstate`` guard only governs
    numpy floating-point errors and never captures Python warnings, so genuine
    diagnostics surface normally even on the exception path (No-Silent-Data-Loss)."""
    import lifelines.statistics as lstats

    from mlsurv.evaluation.metrics import ph_test

    def warn_then_raise(*args, **kwargs):
        warnings.warn("genuine diagnostic before failure", UserWarning)
        raise ValueError("residual computation failed")

    monkeypatch.setattr(lstats, "proportional_hazard_test", warn_then_raise)

    X, y = _well_behaved_survival_data()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        # ph_test catches the ValueError internally and returns the empty
        # sentinel; the genuine warning must survive the guard.
        ph_test(X, y)

    survived = [w for w in recorded if "genuine diagnostic" in str(w.message)]
    assert survived, "genuine warning was swallowed on the exception path"
