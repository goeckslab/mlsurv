"""Reference-implementation validation for Grambsch-Therneau PH test.

Independent Python implementation of the Grambsch-Therneau test compared
against the production ``ph_test()`` on real clinical datasets and
simulated data with known PH behavior.

The reference uses lifelines' ``proportional_hazard_test()``
independently, then cross-checks against production output.

Reference
---------
Grambsch PM, Therneau TM. Proportional hazards tests and diagnostics
based on weighted residuals. Biometrika. 1994;81(3):515-526.
"""

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from scipy.stats import chi2

from mlsurv.evaluation.metrics import ph_test
from mlsurv.utils import SurvivalArray


# ======================================================================
# Reference implementation — independent from production code
# ======================================================================


def _ref_ph_test(X, times, events, time_transform='rank'):
    """Reference PH test: fit lifelines CoxPH and call proportional_hazard_test.

    Returns per-feature (feature, test_statistic, p_value) + global row.
    Uses Grambsch-Therneau joint covariance for the global test.
    """
    df = X.copy()
    df['T'] = times
    df['E'] = events.astype(int)

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df, duration_col='T', event_col='E')

    result = proportional_hazard_test(cph, df, time_transform=time_transform)
    summary = result.summary

    rows = []
    for feat in summary.index:
        ts = float(summary.loc[feat, 'test_statistic'])
        pv = float(summary.loc[feat, 'p'])
        if np.isnan(ts) or np.isnan(pv):
            continue
        rows.append({'feature': feat, 'test_statistic': ts, 'p_value': pv})

    # Joint global test using coefficient covariance matrix
    try:
        from lifelines.statistics import TimeTransformers
        scaled_resids = cph.compute_residuals(df, 'scaled_schoenfeld')
        n_deaths = int(df['E'].sum())

        time_transformer = TimeTransformers().get(time_transform)
        times_g = time_transformer(
            cph.durations, cph.event_observed, cph.weights
        )[cph.event_observed.values]
        g = times_g - times_g.mean()
        S_gg = float((g ** 2).sum())

        valid_feats = [r['feature'] for r in rows]
        resid_cols = [c for c in scaled_resids.columns if c in valid_feats]
        U = (g.values[:, None] * scaled_resids[resid_cols].values).sum(axis=0)

        Sigma_beta = cph.variance_matrix_.loc[resid_cols, resid_cols].values
        Var_U = S_gg * n_deaths * Sigma_beta
        Var_U_inv = np.linalg.inv(Var_U)
        total_chi2 = float(U @ Var_U_inv @ U)
        joint_df = len(resid_cols)
    except (np.linalg.LinAlgError, ValueError):
        total_chi2 = sum(r['test_statistic'] for r in rows)
        joint_df = len(rows)

    global_p = float(chi2.sf(total_chi2, df=joint_df))
    rows.append({'feature': 'GLOBAL', 'test_statistic': total_chi2, 'p_value': global_p})

    return pd.DataFrame(rows)


# ======================================================================
# Data helpers
# ======================================================================


def _encode_categoricals(X):
    """One-hot encode categorical/object columns for CoxPHFitter."""
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
    return X


def _make_surv_array(times, events):
    """Build a SurvivalArray from raw arrays."""
    return SurvivalArray(
        np.array(
            [(bool(e), float(t)) for e, t in zip(events, times)],
            dtype=[("event", bool), ("time", float)],
        )
    )


def _extract_times_events(y, time_col, event_col):
    """Extract times and events arrays from a structured array."""
    return y[time_col].astype(float), y[event_col].astype(bool)


def _compare_ref_vs_prod(X, times, events, time_transform='rank', rtol=1e-4, label=""):
    """Run reference and production PH test, assert agreement."""
    ref = _ref_ph_test(X, times, events, time_transform=time_transform)

    y = _make_surv_array(times, events)
    prod = ph_test(X, y, time_transform=time_transform)

    print(f"\n{'=' * 60}")
    print(f"PH Test Validation: {label}")
    print(f"{'=' * 60}")
    print(f"  {'Feature':<20} {'Ref chi2':>12} {'Prod chi2':>12} {'Ref p':>10} {'Prod p':>10}")
    print(f"  {'-' * 64}")
    for _, rrow in ref.iterrows():
        prow = prod[prod['feature'] == rrow['feature']]
        if prow.empty:
            print(f"  {rrow['feature']:<20} {rrow['test_statistic']:>12.4f} {'MISSING':>12}")
            continue
        prow = prow.iloc[0]
        print(
            f"  {rrow['feature']:<20} {rrow['test_statistic']:>12.4f} "
            f"{prow['test_statistic']:>12.4f} {rrow['p_value']:>10.6f} "
            f"{prow['p_value']:>10.6f}"
        )
    print(f"{'=' * 60}")

    assert len(ref) == len(prod), f"Row count mismatch: ref={len(ref)}, prod={len(prod)}"

    for _, rrow in ref.iterrows():
        prow = prod[prod['feature'] == rrow['feature']].iloc[0]
        np.testing.assert_allclose(
            prow['test_statistic'], rrow['test_statistic'], rtol=rtol,
            err_msg=f"test_statistic mismatch for {rrow['feature']} ({label})",
        )
        np.testing.assert_allclose(
            prow['p_value'], rrow['p_value'], rtol=rtol,
            err_msg=f"p_value mismatch for {rrow['feature']} ({label})",
        )

    # Sanity checks
    assert (prod['p_value'] >= 0).all()
    assert (prod['p_value'] <= 1).all()
    assert (prod['test_statistic'] >= 0).all()

    return prod


# ======================================================================
# Tests
# ======================================================================


@pytest.mark.validation
class TestPHTestValidation:
    """Validate production ph_test against independent reference implementation."""

    def test_gbsg2_coxph(self, gbsg2_data):
        """GBSG2 dataset (686 patients), reference vs production agreement."""
        X, y = gbsg2_data
        times, events = _extract_times_events(y, "time", "cens")
        X_enc = _encode_categoricals(X)

        _compare_ref_vs_prod(
            X_enc, times, events,
            label="GBSG2 CoxPH",
        )

    def test_veterans_coxph(self, veterans_data):
        """Veterans dataset (137 patients, high event rate)."""
        X, y = veterans_data
        times, events = _extract_times_events(y, "Survival_in_days", "Status")
        X_enc = _encode_categoricals(X)

        _compare_ref_vs_prod(
            X_enc, times, events,
            label="Veterans CoxPH",
        )

    def test_simulated_ph_holds(self):
        """Simulated exponential Cox data where PH holds.

        Expect non-significant global p-value (> 0.05).
        """
        rng = np.random.RandomState(42)
        n = 500
        x1 = rng.randn(n)
        x2 = rng.randn(n)
        lp = 0.5 * x1 + 0.3 * x2
        event_times = rng.exponential(1.0 / np.exp(lp))
        censor_times = rng.exponential(3.0, size=n)
        times = np.minimum(event_times, censor_times)
        events = event_times <= censor_times

        X = pd.DataFrame({'x1': x1, 'x2': x2})

        result = _compare_ref_vs_prod(
            X, times, events,
            label="Simulated PH holds",
        )

        global_row = result[result['feature'] == 'GLOBAL'].iloc[0]
        assert global_row['p_value'] > 0.05, (
            f"PH should hold for exponential Cox data: "
            f"chi2={global_row['test_statistic']:.4f}, p={global_row['p_value']:.4f}"
        )

    def test_simulated_ph_violation(self):
        """Simulated data with time-varying coefficient — PH violation.

        Expect significant global p-value (< 0.05).
        """
        rng = np.random.RandomState(123)
        n = 1000
        x1 = rng.randn(n)
        x2 = rng.randn(n)

        # Time-varying effect: x1's effect grows with time
        # Simulate via Weibull with shape depending on x1
        # shape > 1 for high x1, shape < 1 for low x1 → PH violation
        shape = np.exp(0.5 * x1)  # time-varying hazard
        scale = np.exp(-0.3 * x2)
        event_times = scale * rng.weibull(shape)
        censor_times = rng.exponential(2.0, size=n)
        times = np.minimum(event_times, censor_times)
        events = event_times <= censor_times

        X = pd.DataFrame({'x1': x1, 'x2': x2})
        y = _make_surv_array(times, events)

        result = ph_test(X, y)
        global_row = result[result['feature'] == 'GLOBAL'].iloc[0]

        print(f"\nPH Violation test: global chi2={global_row['test_statistic']:.4f}, "
              f"p={global_row['p_value']:.6f}")

        assert global_row['p_value'] < 0.05, (
            f"PH violation should be detected: "
            f"chi2={global_row['test_statistic']:.4f}, p={global_row['p_value']:.4f}"
        )

    def test_flchain_coxph(self):
        """flchain dataset cross-check (skip if unavailable)."""
        try:
            from sksurv.datasets import load_flchain
        except ImportError:
            pytest.skip("load_flchain not available in this sksurv version")

        X, y = load_flchain()
        times = y["futime"].astype(float)
        events = y["death"].astype(bool)

        X = X[["age", "sex", "flc.grp", "lambda"]].copy()
        X["sex"] = (X["sex"] == "M").astype(float)

        valid = X.notna().all(axis=1)
        X = X.loc[valid].reset_index(drop=True)
        times = np.asarray(times[valid.values])
        events = np.asarray(events[valid.values])

        _compare_ref_vs_prod(
            X, times, events,
            label="flchain CoxPH",
        )
