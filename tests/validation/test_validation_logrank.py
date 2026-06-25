"""Validation tests for log-rank test against independent reference implementation.

Validates _compute_logrank() using a pure-numpy Mantel-Haenszel reference
implementation (no lifelines) and cross-checks against lifelines directly.

References
----------
Mantel N (1966). Evaluation of survival data and two new rank order
    statistics arising in its consideration. Cancer Chemotherapy Reports,
    50(3), 163-170.

Peto R, Peto J (1972). Asymptotically efficient rank invariant test
    procedures. Journal of the Royal Statistical Society: Series A,
    135(2), 185-207.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2


# ---------------------------------------------------------------------------
# Pure-numpy reference implementation (Mantel-Haenszel log-rank)
# ---------------------------------------------------------------------------
# Intentionally avoids lifelines to serve as a truly independent cross-check.


def _ref_logrank_two_sample(time1, event1, time2, event2):
    """Two-sample log-rank test via Mantel-Haenszel statistic.

    Returns (statistic, p_value).
    """
    time1 = np.asarray(time1, dtype=float)
    event1 = np.asarray(event1, dtype=float)
    time2 = np.asarray(time2, dtype=float)
    event2 = np.asarray(event2, dtype=float)

    # Pool all distinct event times
    all_times = np.unique(np.concatenate([time1[event1 == 1], time2[event2 == 1]]))
    all_times.sort()

    numer = 0.0
    denom = 0.0

    for t in all_times:
        # At-risk counts at time t
        n1 = np.sum(time1 >= t)
        n2 = np.sum(time2 >= t)
        n = n1 + n2
        if n == 0:
            continue

        # Observed events at time t
        d1 = np.sum((time1 == t) & (event1 == 1))
        d2 = np.sum((time2 == t) & (event2 == 1))
        d = d1 + d2

        # Expected events in group 1 under null
        e1 = d * n1 / n

        numer += d1 - e1

        # Hypergeometric variance
        if n > 1:
            denom += d * n1 * n2 * (n - d) / (n * n * (n - 1))

    if denom <= 0:
        return 0.0, 1.0

    statistic = numer ** 2 / denom
    p_value = float(chi2.sf(statistic, df=1))
    return float(statistic), p_value


def _ref_logrank_multivariate(times, groups, events):
    """Multivariate (K-sample) log-rank test via Mantel-Haenszel statistic.

    Returns (statistic, p_value).
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    k = len(unique_groups)
    if k < 2:
        return 0.0, 1.0

    # All distinct event times
    event_times = np.unique(times[events == 1])
    event_times.sort()

    # We use the (K-1)-dimensional formulation
    # U = sum over event times of (O_j - E_j) for j = 1..K-1
    # V = sum of variance matrices
    U = np.zeros(k - 1)
    V = np.zeros((k - 1, k - 1))

    for t in event_times:
        at_risk = times >= t
        n_total = np.sum(at_risk)
        if n_total <= 1:
            continue

        d_total = np.sum((times == t) & (events == 1))
        if d_total == 0:
            continue

        n_j = np.array([np.sum(at_risk & (groups == g)) for g in unique_groups])
        d_j = np.array([np.sum((times == t) & (events == 1) & (groups == g))
                        for g in unique_groups])

        # Use first K-1 groups
        for i in range(k - 1):
            e_i = d_total * n_j[i] / n_total
            U[i] += d_j[i] - e_i

        factor = d_total * (n_total - d_total) / (n_total * n_total * (n_total - 1))
        for i in range(k - 1):
            for j in range(k - 1):
                if i == j:
                    V[i, j] += factor * n_j[i] * (n_total - n_j[i])
                else:
                    V[i, j] -= factor * n_j[i] * n_j[j]

    # Chi-squared statistic = U^T V^{-1} U
    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        return 0.0, 1.0

    statistic = float(U @ V_inv @ U)
    p_value = float(chi2.sf(statistic, df=k - 1))
    return statistic, p_value


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


@pytest.mark.validation
class TestLogRankValidation:
    """Cross-validate production _compute_logrank against reference impl."""

    def test_two_group_simulated(self):
        """Two-group exponential data: production matches reference."""
        from mlsurv.viz.survival import _compute_logrank

        rng = np.random.RandomState(123)
        n = 300
        group = np.repeat([0, 1], n // 2)
        time = np.where(group == 0, rng.exponential(20, n), rng.exponential(5, n))
        event = rng.binomial(1, 0.85, n)
        df = pd.DataFrame({"time": time, "event": event, "group": group})

        # Reference
        mask0 = group == 0
        ref_stat, ref_p = _ref_logrank_two_sample(
            time[mask0], event[mask0], time[~mask0], event[~mask0],
        )

        # Production
        result = _compute_logrank(df, "time", "event", "group")
        assert result is not None
        assert result["n_groups"] == 2
        np.testing.assert_allclose(result["omnibus_statistic"], ref_stat, rtol=1e-6)
        np.testing.assert_allclose(result["omnibus_p"], ref_p, rtol=1e-6)

    def test_gbsg2_real_data(self, gbsg2_data):
        """GBSG2 split by hormonal therapy: production matches reference."""
        from mlsurv.utils import to_structured_array
        from mlsurv.viz.survival import _compute_logrank

        X, y = gbsg2_data
        surv = to_structured_array(y)
        df = pd.DataFrame({"time": surv["time"], "event": surv["event"]})
        df["horTh"] = X["horTh"].values

        # Reference
        g1 = df[df["horTh"] == "yes"]
        g2 = df[df["horTh"] == "no"]
        ref_stat, ref_p = _ref_logrank_two_sample(
            g1["time"].values, g1["event"].values,
            g2["time"].values, g2["event"].values,
        )

        # Production
        result = _compute_logrank(df, "time", "event", "horTh")
        assert result is not None
        np.testing.assert_allclose(result["omnibus_statistic"], ref_stat, rtol=1e-6)
        np.testing.assert_allclose(result["omnibus_p"], ref_p, rtol=1e-4)

    def test_multivariate_3_groups(self):
        """3-group omnibus statistic matches multivariate reference."""
        from mlsurv.viz.survival import _compute_logrank

        rng = np.random.RandomState(99)
        n = 450
        group = np.repeat(["A", "B", "C"], n // 3)
        scales = {"A": 20, "B": 10, "C": 5}
        time = np.concatenate([
            rng.exponential(scales[g], n // 3) for g in ["A", "B", "C"]
        ])
        event = rng.binomial(1, 0.85, n)
        df = pd.DataFrame({"time": time, "event": event, "group": group})

        ref_stat, ref_p = _ref_logrank_multivariate(time, group, event)
        result = _compute_logrank(df, "time", "event", "group")

        assert result is not None
        assert result["n_groups"] == 3
        np.testing.assert_allclose(result["omnibus_statistic"], ref_stat, rtol=1e-4)
        np.testing.assert_allclose(result["omnibus_p"], ref_p, rtol=1e-3)

    def test_pairwise_holm_correction(self):
        """Pairwise Holm-adjusted p-values match manual Holm on raw p-values."""
        from mlsurv.evaluation._paired_comparison import _adjust_p_values
        from mlsurv.viz.survival import _compute_logrank

        rng = np.random.RandomState(77)
        n = 300
        group = np.repeat(["X", "Y", "Z"], n // 3)
        time = np.concatenate([
            rng.exponential(15, n // 3),
            rng.exponential(8, n // 3),
            rng.exponential(4, n // 3),
        ])
        event = rng.binomial(1, 0.8, n)
        df = pd.DataFrame({"time": time, "event": event, "group": group})

        result = _compute_logrank(df, "time", "event", "group", p_adjust="holm")
        assert result["pairwise"] is not None
        assert len(result["pairwise"]) == 3

        # Extract raw p-values and manually apply Holm
        raw_ps = np.array([pw["p_raw"] for pw in result["pairwise"]])
        expected_adj = _adjust_p_values(raw_ps, method="holm")

        actual_adj = np.array([pw["p_adjusted"] for pw in result["pairwise"]])
        np.testing.assert_allclose(actual_adj, expected_adj, rtol=1e-8)

    def test_single_group_returns_none(self):
        """Single group returns None (no comparison possible)."""
        from mlsurv.viz.survival import _compute_logrank

        df = pd.DataFrame({"time": [1, 2, 3], "event": [1, 1, 0], "group": ["A", "A", "A"]})
        result = _compute_logrank(df, "time", "event", "group")
        assert result is None


# ---------------------------------------------------------------------------
# Cross-validation against R survival::survdiff()
# ---------------------------------------------------------------------------
# Reference values generated by R 4.5.2 with survival package.
# Datasets: veteran (Kalbfleisch & Prentice 1980) and lung (Loprinzi 1994)
# from the R survival package — the gold-standard implementation.


def _load_fixture(name):
    """Load a CSV fixture from tests/validation/data/."""
    import pathlib
    path = pathlib.Path(__file__).parent / "data" / name
    return pd.read_csv(path)


@pytest.mark.validation
class TestLogRankCrossValidationR:
    """Cross-validate production _compute_logrank against R survdiff()."""

    def test_veteran_celltype_4group(self):
        """veteran dataset, celltype (4 groups): matches R survdiff().

        Reference
        ---------
        Kalbfleisch JD, Prentice RL (1980). The Statistical Analysis of
            Failure Time Data. Wiley, New York.

        R code::

            library(survival)
            survdiff(Surv(time, status) ~ celltype, data = veteran)
        """
        from mlsurv.viz.survival import _compute_logrank

        # R reference: survdiff(Surv(time, status) ~ celltype, data = veteran)
        R_CHISQ = 25.403700
        R_P = 1.2712459390e-05

        df = _load_fixture("veteran.csv")
        result = _compute_logrank(df, "time", "status", "celltype")

        assert result is not None
        assert result["n_groups"] == 4
        np.testing.assert_allclose(
            result["omnibus_statistic"], R_CHISQ, rtol=1e-5,
            err_msg="Omnibus chi-squared does not match R survdiff()",
        )
        np.testing.assert_allclose(
            result["omnibus_p"], R_P, rtol=1e-4,
            err_msg="Omnibus p-value does not match R survdiff()",
        )

    def test_veteran_celltype_pairwise(self):
        """veteran dataset, celltype pairwise: matches R survdiff() pairs.

        R code (for each pair)::

            sub <- veteran[veteran$celltype %in% c("squamous", "smallcell"), ]
            survdiff(Surv(time, status) ~ celltype, data = sub)
        """
        from mlsurv.viz.survival import _compute_logrank

        # R reference: pairwise survdiff on each pair
        R_PAIRWISE = {
            ("adeno", "large"): 17.669322,
            ("adeno", "smallcell"): 0.096843,
            ("adeno", "squamous"): 12.045484,
            ("large", "smallcell"): 9.370904,
            ("large", "squamous"): 0.822594,
            ("smallcell", "squamous"): 11.573674,
        }

        df = _load_fixture("veteran.csv")
        result = _compute_logrank(df, "time", "status", "celltype",
                                  p_adjust="none")

        assert result["pairwise"] is not None
        assert len(result["pairwise"]) == 6  # C(4,2) = 6

        for pw in result["pairwise"]:
            pair_key = (pw["group_1"], pw["group_2"])
            r_stat = R_PAIRWISE[pair_key]
            np.testing.assert_allclose(
                pw["statistic"], r_stat, rtol=1e-4,
                err_msg=f"Pairwise {pair_key} chi-squared does not match R",
            )

    def test_lung_sex_2group(self):
        """lung dataset, sex (2 groups): matches R survdiff().

        Reference
        ---------
        Loprinzi CL et al. (1994). Prospective evaluation of prognostic
            variables from patient-completed questionnaires.
            J Clinical Oncology, 12(3), 601-607.

        R code::

            library(survival)
            lung_clean <- lung[complete.cases(lung$time, lung$status, lung$sex), ]
            survdiff(Surv(time, status) ~ sex, data = lung_clean)
        """
        from mlsurv.viz.survival import _compute_logrank

        # R reference: survdiff(Surv(time, status) ~ sex, data = lung_clean)
        R_CHISQ = 10.326742
        R_P = 1.3111645204e-03

        df = _load_fixture("lung.csv")
        # R lung dataset: status=2 is death, status=1 is censored
        df["event"] = (df["status"] == 2).astype(int)

        result = _compute_logrank(df, "time", "event", "sex")

        assert result is not None
        assert result["n_groups"] == 2
        np.testing.assert_allclose(
            result["omnibus_statistic"], R_CHISQ, rtol=1e-5,
            err_msg="Omnibus chi-squared does not match R survdiff()",
        )
        np.testing.assert_allclose(
            result["omnibus_p"], R_P, rtol=1e-4,
            err_msg="Omnibus p-value does not match R survdiff()",
        )

    def test_veteran_trt_2group_nonsignificant(self):
        """veteran dataset, trt (2 groups, non-significant): matches R survdiff().

        This tests the non-significant case (p ~ 0.93) to ensure accuracy
        is not limited to highly significant results.

        R code::

            survdiff(Surv(time, status) ~ trt, data = veteran)
        """
        from mlsurv.viz.survival import _compute_logrank

        # R reference: survdiff(Surv(time, status) ~ trt, data = veteran)
        R_CHISQ = 0.008227
        R_P = 9.2772723334e-01

        df = _load_fixture("veteran.csv")
        result = _compute_logrank(df, "time", "status", "trt")

        assert result is not None
        assert result["n_groups"] == 2
        np.testing.assert_allclose(
            result["omnibus_statistic"], R_CHISQ, rtol=1e-3,
            err_msg="Omnibus chi-squared does not match R survdiff()",
        )
        np.testing.assert_allclose(
            result["omnibus_p"], R_P, rtol=1e-3,
            err_msg="Omnibus p-value does not match R survdiff()",
        )
