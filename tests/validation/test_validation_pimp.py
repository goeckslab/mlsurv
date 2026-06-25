"""Validation tests for PIMP importance p-values (Altmann et al. 2010).

Validates that PIMP p-values:
1. Match a pure-Python reference implementation (deterministic, same seed)
2. Correctly identify known-significant features on GBSG2
3. Correctly separate signal from noise on synthetic data with known ground truth

Reference
---------
Altmann A, Tolosi L, Sander O, Lengauer T (2010).
    Permutation importance: a corrected feature importance measure.
    Bioinformatics, 26(10), 1340-1347.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from mlsurv.evaluation.importance import compute_pimp
from mlsurv.utils import SurvivalArray


# ---------------------------------------------------------------------------
# Pure-Python reference implementation
# ---------------------------------------------------------------------------

def _ref_pimp(model, X_train, y_train, X_test, y_test, n_perms, n_repeats, seed):
    """Brute-force PIMP: manual loops, no parallelism, no shortcuts."""
    rng = np.random.RandomState(seed)

    # 1. Observed importance
    obs = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, random_state=seed, n_jobs=1,
    )
    observed = obs.importances_mean

    # 2. Null distribution
    perm_seeds = rng.randint(0, 2**31, size=n_perms)
    imp_seeds = rng.randint(0, 2**31, size=n_perms)

    n_features = X_test.shape[1]
    null_matrix = np.zeros((n_perms, n_features))

    for i in range(n_perms):
        perm_rng = np.random.RandomState(int(perm_seeds[i]))
        idx = perm_rng.permutation(len(y_train))
        y_perm = y_train[idx]

        m = clone(model)
        m.fit(X_train, y_perm)
        null_result = permutation_importance(
            m, X_test, y_test,
            n_repeats=n_repeats, random_state=int(imp_seeds[i]), n_jobs=1,
        )
        null_matrix[i] = null_result.importances_mean

    # 3. Empirical p-values
    p_values = np.array([
        (np.sum(null_matrix[:, j] >= observed[j]) + 1) / (n_perms + 1)
        for j in range(n_features)
    ])

    return observed, null_matrix, p_values


# ---------------------------------------------------------------------------
# Reference implementation cross-check
# ---------------------------------------------------------------------------

@pytest.mark.validation
class TestPimpReferenceImplementation:
    """Cross-check production compute_pimp() against reference impl."""

    @pytest.fixture(scope="class")
    def small_coxph_data(self):
        """Small synthetic dataset + fitted CoxPH for fast testing."""
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        rng = np.random.RandomState(42)
        n = 80
        X = pd.DataFrame({
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "x3": rng.normal(0, 1, n),
        })
        times = np.exp(0.5 * X["x1"] + 0.2 * X["x2"] + rng.normal(0, 0.5, n))
        events = rng.binomial(1, 0.7, n).astype(bool)
        y = SurvivalArray(np.array(
            [(e, t) for e, t in zip(events, times)],
            dtype=[("event", bool), ("time", float)],
        ))

        X_train, X_test = X.iloc[:60], X.iloc[60:]
        y_train, y_test = y[:60], y[60:]

        model = CoxPHSurvivalAnalysis(alpha=0.1)
        model.fit(X_train, y_train)
        return model, X_train, y_train, X_test, y_test

    def test_reference_matches_production(self, small_coxph_data):
        """Production compute_pimp() matches reference on same seed."""
        model, X_train, y_train, X_test, y_test = small_coxph_data
        n_perms, n_repeats, seed = 10, 5, 42

        # Reference
        ref_obs, ref_null, ref_p = _ref_pimp(
            model, X_train, y_train, X_test, y_test,
            n_perms, n_repeats, seed,
        )

        # Production
        prod = compute_pimp(
            model, X_train, y_train, X_test, y_test,
            n_permutations=n_perms,
            n_importance_repeats=n_repeats,
            random_state=seed,
            p_adjust="none",
            show_progress=False,
        )

        # Match observed importance
        prod_sorted = prod.sort_values("feature").reset_index(drop=True)
        feat_order = sorted(range(len(X_test.columns)),
                           key=lambda i: X_test.columns[i])
        np.testing.assert_allclose(
            prod_sorted["observed_importance"].values,
            ref_obs[feat_order],
            atol=1e-8,
        )

        # Match p-values
        np.testing.assert_allclose(
            prod_sorted["p_value"].values,
            ref_p[feat_order],
            atol=1e-8,
        )


# ---------------------------------------------------------------------------
# GBSG2 validation
# ---------------------------------------------------------------------------

@pytest.mark.validation
class TestPimpGBSG2:
    """Validate PIMP on GBSG2 dataset."""

    @pytest.fixture(scope="class")
    def gbsg2_pimp(self):
        """Run PIMP on GBSG2 with RSF (cached per class)."""
        import warnings
        import sys
        import io

        warnings.filterwarnings("ignore")

        from sksurv.datasets import load_gbsg2
        from sksurv.ensemble import RandomSurvivalForest
        from sklearn.model_selection import train_test_split
        from mlsurv.utils import to_structured_array

        X_raw, y_raw = load_gbsg2()
        Xt = X_raw.copy()
        Xt["horTh"] = (Xt["horTh"] == "yes").astype(int)
        Xt["menostat"] = (Xt["menostat"] == "Post").astype(int)
        Xt["tgrade"] = Xt["tgrade"].map({"I": 1, "II": 2, "III": 3}).astype(int)
        y = to_structured_array(y_raw["time"], y_raw["cens"])

        X_train, X_test, y_train, y_test = train_test_split(
            Xt, y, test_size=0.25, random_state=20,
        )

        rsf = RandomSurvivalForest(
            n_estimators=100, min_samples_split=10,
            min_samples_leaf=15, n_jobs=-1, random_state=20,
        )
        rsf.fit(X_train, y_train)

        # Suppress progress output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = compute_pimp(
                rsf, X_train, y_train, X_test, y_test,
                n_permutations=50,
                n_importance_repeats=10,
                random_state=42,
                show_progress=True,
            )
        finally:
            sys.stdout = old_stdout

        return result

    def test_pnodes_significant(self, gbsg2_pimp):
        """pnodes should have raw p < 0.05.

        With only 50 permutations, adjusted p-values can be inflated
        by Holm correction across 8 features. Raw p-value is the
        appropriate check at this sample size.
        """
        row = gbsg2_pimp[gbsg2_pimp["feature"] == "pnodes"]
        assert len(row) == 1
        assert row["p_value"].iloc[0] < 0.05, (
            f"pnodes p_value={row['p_value'].iloc[0]:.4f}"
        )

    def test_progrec_ranks_in_top_half(self, gbsg2_pimp):
        """progrec should rank in the top half of features by p-value.

        PIMP tests a stronger null (feature-response independence) than
        bootstrap (importance > 0). With only 50 permutations, progrec
        may not reach significance, but it should still rank well.
        """
        n_features = len(gbsg2_pimp)
        # gbsg2_pimp is sorted by p_value ascending
        progrec_rank = gbsg2_pimp["feature"].tolist().index("progrec")
        assert progrec_rank < n_features // 2, (
            f"progrec rank={progrec_rank + 1}/{n_features} "
            f"(p_value={gbsg2_pimp.iloc[progrec_rank]['p_value']:.4f})"
        )

    def test_top_features_include_known_important(self, gbsg2_pimp):
        """Top 3 features by p-value should include pnodes."""
        top3 = set(gbsg2_pimp.head(3)["feature"])
        assert "pnodes" in top3, f"Top 3 features: {top3}"


# ---------------------------------------------------------------------------
# Synthetic ground truth
# ---------------------------------------------------------------------------

@pytest.mark.validation
class TestPimpSyntheticGroundTruth:
    """Validate PIMP on synthetic data with known signal/noise structure."""

    @pytest.fixture(scope="class")
    def synthetic_pimp(self):
        """Generate exponential Cox data and run PIMP."""
        import warnings
        import sys
        import io

        warnings.filterwarnings("ignore")

        from sksurv.linear_model import CoxPHSurvivalAnalysis

        rng = np.random.RandomState(123)
        n = 300
        n_signal = 3
        n_noise = 5
        coefs = np.array([1.0, 0.5, 0.2])

        # Signal features
        X_signal = rng.normal(0, 1, (n, n_signal))
        # Noise features
        X_noise = rng.normal(0, 1, (n, n_noise))
        X_all = np.hstack([X_signal, X_noise])

        feature_names = (
            [f"signal_{i}" for i in range(n_signal)]
            + [f"noise_{i}" for i in range(n_noise)]
        )
        X = pd.DataFrame(X_all, columns=feature_names)

        # Generate survival times from exponential Cox model
        lp = X_signal @ coefs
        baseline_hazard = 0.1
        times = -np.log(rng.uniform(0, 1, n)) / (baseline_hazard * np.exp(lp))
        times = np.clip(times, 0.01, 100)
        events = rng.binomial(1, 0.7, n).astype(bool)

        y = SurvivalArray(np.array(
            [(e, t) for e, t in zip(events, times)],
            dtype=[("event", bool), ("time", float)],
        ))

        X_train, X_test = X.iloc[:200], X.iloc[200:]
        y_train, y_test = y[:200], y[200:]

        model = CoxPHSurvivalAnalysis(alpha=0.1)
        model.fit(X_train, y_train)

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = compute_pimp(
                model, X_train, y_train, X_test, y_test,
                n_permutations=50,
                n_importance_repeats=10,
                random_state=42,
                show_progress=True,
            )
        finally:
            sys.stdout = old_stdout

        return result, coefs

    def test_signal_features_significant(self, synthetic_pimp):
        """signal_0 and signal_1 (strongest effects) should be significant."""
        result, _ = synthetic_pimp
        for feat in ["signal_0", "signal_1"]:
            row = result[result["feature"] == feat]
            assert row["p_value"].iloc[0] < 0.05, (
                f"{feat} p_value={row['p_value'].iloc[0]:.4f}"
            )

    def test_noise_features_mostly_nonsignificant(self, synthetic_pimp):
        """Majority of noise features should have p > 0.05."""
        result, _ = synthetic_pimp
        noise_rows = result[result["feature"].str.startswith("noise_")]
        n_nonsig = (noise_rows["p_value"] > 0.05).sum()
        assert n_nonsig >= 3, (
            f"Only {n_nonsig}/5 noise features non-significant: "
            f"{noise_rows[['feature', 'p_value']].to_dict('records')}"
        )

    def test_ranking_matches_true_effects(self, synthetic_pimp):
        """PIMP ranking should correlate with true coefficient magnitudes."""
        result, coefs = synthetic_pimp
        signal_rows = result[result["feature"].str.startswith("signal_")]
        signal_rows = signal_rows.sort_values("feature").reset_index(drop=True)

        # True ranking: signal_0 > signal_1 > signal_2
        pimp_importance = signal_rows["observed_importance"].values
        rho, _ = spearmanr(coefs, pimp_importance)
        assert rho > 0.5, f"Spearman rho={rho:.3f} between true coefs and PIMP"
