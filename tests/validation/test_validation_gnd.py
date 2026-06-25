"""Reference-implementation validation for Greenwood-Nam-D'Agostino test.

Independent Python implementation of the GND calibration test
(Demler et al. 2015, Cook's lab) compared against the production
``greenwood_nam_dagostino()`` on real clinical datasets.

The reference uses lifelines' KaplanMeierFitter directly (bypassing
production's ``_km_survival_at`` wrapper) and re-implements Greenwood
variance and chi2 accumulation from scratch, without calling any
production helpers (``_assign_bins``, ``_km_survival_at``).

NOTE: Both reference and production use pd.qcut for equal-frequency binning.
This is intentional (same algorithm is required for agreement), but means
binning bugs are NOT detected by this validation.

Reference
---------
Demler OV, Paynter NP, Cook NR. Tests of calibration and goodness-of-fit
in the survival setting. Stat Med. 2015;34(10):1659-1680.
"""

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter, KaplanMeierFitter

from mlsurv.evaluation.metrics import greenwood_nam_dagostino
from mlsurv.utils import SurvivalArray


# ======================================================================
# Reference implementation — independent from production code
# ======================================================================


def _ref_greenwood_var(kmf, eval_time):
    """Greenwood variance of S(eval_time), independent from production code.

    Computes: V = S(t)^2 * sum_{t_j <= t} d_j / (n_j * (n_j - d_j))
    """
    et = kmf.event_table
    et_t = et.loc[et.index <= eval_time]
    d = et_t["observed"].values.astype(float)
    n = et_t["at_risk"].values.astype(float)
    denom = n * (n - d)
    valid = denom > 0
    if not valid.any():
        return None
    S_t = float(kmf.survival_function_at_times(eval_time).iloc[0])
    return S_t**2 * np.sum(d[valid] / denom[valid])


def _ref_gnd_calib(times, events, pred_surv, eval_time, n_bins=10):
    """Reference GND: independent implementation for validation.

    Bins by predicted survival probability (same strategy as production)
    and computes KM estimates and Greenwood variance through an independent
    code path — no production helpers are called.

    Parameters
    ----------
    times, events : array-like
        Observed times and boolean event indicators.
    pred_surv : array-like
        Predicted survival probability S(eval_time) per sample.
    eval_time : float
        Calendar time at which to assess calibration.
    n_bins : int
        Number of equal-frequency bins.

    Returns
    -------
    chi2, p_value, df, n_valid_bins : float, float, int, int
    """
    from scipy.stats import chi2 as chi2_dist

    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    pred_surv = np.asarray(pred_surv, dtype=float)

    bin_labels = pd.qcut(pred_surv, q=n_bins, labels=False, duplicates="drop")
    unique_bins = np.unique(bin_labels)

    chi2_val = 0.0
    valid_bins = 0

    for b in unique_bins:
        mask = bin_labels == b
        n_k = int(mask.sum())
        if n_k < 5:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(times[mask], events[mask])
        O_k = float(kmf.survival_function_at_times(eval_time).iloc[0])
        E_k = float(pred_surv[mask].mean())
        V_k = _ref_greenwood_var(kmf, eval_time)
        if V_k is None or V_k < 1e-12:
            continue

        chi2_val += (O_k - E_k) ** 2 / V_k
        valid_bins += 1

    if valid_bins < 2:
        return (np.nan, np.nan, 0, valid_bins)

    df = valid_bins - 1
    p_value = float(chi2_dist.sf(chi2_val, df=df))
    return (float(chi2_val), p_value, df, valid_bins)


# ======================================================================
# Data helpers
# ======================================================================


def _encode_categoricals(X):
    """One-hot encode categorical/object columns for CoxPHFitter."""
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
    return X


def _prepare_flchain(admin_cens=4000):
    """Load flchain dataset, apply admin censoring, select Cook's covariates."""
    from sksurv.datasets import load_flchain

    X, y = load_flchain()
    times = y["futime"].astype(float).copy()
    events = y["death"].astype(bool).copy()

    # Administrative censoring
    if admin_cens is not None:
        cens_mask = times > admin_cens
        events[cens_mask] = False
        times[cens_mask] = float(admin_cens)

    # Cook's covariates: age, sex, flc.grp, lambda
    X = X[["age", "sex", "flc.grp", "lambda"]].copy()
    X["sex"] = (X["sex"] == "M").astype(float)

    # Drop rows with missing values
    valid = X.notna().all(axis=1)
    X = X.loc[valid].reset_index(drop=True)
    times = times[valid.values]
    events = events[valid.values]

    return X, np.asarray(times), np.asarray(events)


def _fit_coxph_pred_surv(X, times, events, eval_time):
    """Fit CoxPH with lifelines, return predicted S(eval_time) per sample."""
    df = X.copy()
    df["T"] = times
    df["E"] = events

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df, duration_col="T", event_col="E")

    sf = cph.predict_survival_function(X, times=[eval_time])
    return sf.iloc[0].values


def _make_surv_array(times, events):
    """Build a SurvivalArray from raw arrays."""
    return SurvivalArray(
        np.array(
            [(bool(e), float(t)) for e, t in zip(events, times)],
            dtype=[("event", bool), ("time", float)],
        )
    )


def _compare_ref_vs_prod(
    times, events, pred_surv, eval_time, n_bins=10, rtol=1e-6, label=""
):
    """Run reference and production GND, assert agreement.

    Returns
    -------
    chi2, p_value, df : float, float, int
        Reference implementation results (already validated against production).
    """
    ref_chi2, ref_p, ref_df, ref_valid = _ref_gnd_calib(
        times, events, pred_surv, eval_time, n_bins=n_bins
    )

    y_test = _make_surv_array(times, events)
    prod_chi2, prod_p, prod_df = greenwood_nam_dagostino(
        y_test, pred_surv, eval_time, n_bins=n_bins
    )

    print(f"\n{'=' * 60}")
    print(f"GND Validation: {label}")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<20} {'Reference':>15} {'Production':>15}")
    print(f"  {'-' * 50}")
    print(f"  {'chi2':<20} {ref_chi2:>15.6f} {prod_chi2:>15.6f}")
    print(f"  {'p-value':<20} {ref_p:>15.6f} {prod_p:>15.6f}")
    print(f"  {'df':<20} {ref_df:>15d} {prod_df:>15d}")
    print(f"  {'valid bins':<20} {ref_valid:>15d} {'':>15}")
    print(f"{'=' * 60}")

    assert not np.isnan(ref_chi2), f"Reference returned NaN chi2 ({label})"
    assert not np.isnan(prod_chi2), f"Production returned NaN chi2 ({label})"

    assert ref_df == prod_df, f"df mismatch: ref={ref_df}, prod={prod_df}"
    np.testing.assert_allclose(
        prod_chi2, ref_chi2, rtol=rtol,
        err_msg=f"chi2 mismatch ({label})",
    )
    np.testing.assert_allclose(
        prod_p, ref_p, rtol=rtol,
        err_msg=f"p-value mismatch ({label})",
    )

    assert prod_chi2 >= 0, "chi2 must be non-negative"
    assert 0 <= prod_p <= 1, "p-value must be in [0, 1]"
    assert prod_df >= 1, "df must be >= 1"

    return ref_chi2, ref_p, ref_df


def _simulate_exponential_survival(n=2000, seed=42):
    """Simulate exponential survival data with known hazard and ~20% censoring.

    Returns
    -------
    times, events, true_rate, eval_time : arrays and float
    """
    rng = np.random.RandomState(seed)
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    lp = 0.5 * x1 + 0.3 * x2
    true_rate = np.exp(lp)

    event_times = rng.exponential(1.0 / true_rate)
    censor_times = rng.exponential(3.0, size=n)
    times = np.minimum(event_times, censor_times)
    events = event_times <= censor_times
    eval_time = float(np.median(event_times))

    return times, events, true_rate, eval_time


def _extract_times_events(y, time_col, event_col):
    """Extract times and events arrays from a structured array."""
    return y[time_col].astype(float), y[event_col].astype(bool)


# ======================================================================
# Tests
# ======================================================================


@pytest.mark.validation
class TestGNDValidation:
    """Validate production GND against independent reference implementation."""

    def test_flchain_coxph(self):
        """Cook lab reference: flchain dataset, CoxPH, t=4000 days."""
        try:
            from sksurv.datasets import load_flchain  # noqa: F401
        except ImportError:
            pytest.skip("load_flchain not available in this sksurv version")

        X, times, events = _prepare_flchain(admin_cens=4000)
        eval_time = 4000.0

        pred_surv = _fit_coxph_pred_surv(X, times, events, eval_time)
        _compare_ref_vs_prod(
            times, events, pred_surv, eval_time,
            label="flchain CoxPH t=4000",
        )

    def test_gbsg2_coxph(self, gbsg2_data):
        """Cross-check: GBSG2 (686 patients), CoxPH, median event time."""
        X, y = gbsg2_data
        times, events = _extract_times_events(y, "time", "cens")
        X_enc = _encode_categoricals(X)
        eval_time = float(np.median(times[events]))

        pred_surv = _fit_coxph_pred_surv(X_enc, times, events, eval_time)
        _compare_ref_vs_prod(
            times, events, pred_surv, eval_time,
            label="GBSG2 CoxPH median event time",
        )

    def test_veterans_coxph(self, veterans_data):
        """Cross-check: Veterans lung cancer (137 patients), high event rate."""
        X, y = veterans_data
        times, events = _extract_times_events(y, "Survival_in_days", "Status")
        X_enc = _encode_categoricals(X)
        eval_time = float(np.median(times[events]))

        pred_surv = _fit_coxph_pred_surv(X_enc, times, events, eval_time)
        _compare_ref_vs_prod(
            times, events, pred_surv, eval_time,
            n_bins=5,  # smaller dataset -> fewer bins
            label="Veterans CoxPH median event time",
        )

    def test_perfect_calibration_simulated(self):
        """Sanity check: known-truth predictions -> small chi2, large p."""
        times, events, true_rate, eval_time = _simulate_exponential_survival()

        # True survival: S(t|x) = exp(-rate * t) for exponential model
        pred_surv = np.exp(-true_rate * eval_time)

        chi2, p_val, _ = _compare_ref_vs_prod(
            times, events, pred_surv, eval_time,
            label="Simulated perfect calibration",
        )

        assert p_val > 0.01, (
            f"Perfectly calibrated predictions should not be significant: "
            f"chi2={chi2:.4f}, p={p_val:.4f}"
        )

    def test_multiple_eval_times(self, gbsg2_data):
        """Validate agreement at 25th, 50th, and 75th percentile eval times."""
        X, y = gbsg2_data
        times, events = _extract_times_events(y, "time", "cens")
        X_enc = _encode_categoricals(X)
        event_times = times[events]

        # Fit CoxPH once (shared across eval times)
        df = X_enc.copy()
        df["T"] = times
        df["E"] = events
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(df, duration_col="T", event_col="E")

        for pct in [25, 50, 75]:
            eval_time = float(np.percentile(event_times, pct))
            sf = cph.predict_survival_function(X_enc, times=[eval_time])
            pred_surv = sf.iloc[0].values

            _compare_ref_vs_prod(
                times, events, pred_surv, eval_time,
                label=f"GBSG2 CoxPH p{pct} (t={eval_time:.0f})",
            )

    def test_miscalibrated_predictions_detected(self):
        """Deliberately miscalibrated predictions should produce small p-value."""
        times, events, true_rate, eval_time = _simulate_exponential_survival()

        # Deliberately miscalibrated: shift predictions substantially
        pred_surv = np.clip(np.exp(-true_rate * eval_time) + 0.3, 0.01, 0.99)

        y_test = _make_surv_array(times, events)
        chi2, p_value, df = greenwood_nam_dagostino(
            y_test, pred_surv, eval_time
        )

        assert p_value < 0.05, (
            f"Miscalibrated predictions should be detected: "
            f"chi2={chi2:.4f}, p={p_value:.4f}"
        )

    def test_heavy_censoring_agreement(self):
        """Validate agreement under heavy censoring (~70% censored)."""
        rng = np.random.RandomState(42)
        n = 1000

        x1 = rng.randn(n)
        lp = 0.5 * x1
        true_rate = np.exp(lp)
        event_times = rng.exponential(1.0 / true_rate)
        # Heavy censoring: short censoring times
        censor_times = rng.exponential(0.5, size=n)
        times = np.minimum(event_times, censor_times)
        events = event_times <= censor_times

        eval_time = float(np.percentile(event_times, 30))
        pred_surv = np.exp(-true_rate * eval_time)

        _compare_ref_vs_prod(
            times, events, pred_surv, eval_time,
            n_bins=5,  # fewer bins needed with fewer effective events
            label=f"Simulated heavy censoring (~{100 - 100*events.mean():.0f}%)",
        )
