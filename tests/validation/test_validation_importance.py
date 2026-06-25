"""Validation tests for importance_summary() empirical p-values.

Validates that bootstrap-derived importance p-values:
1. Agree with sklearn's permutation_importance on feature ranking (GBSG2)
2. Correctly identify known-significant features (pnodes, progrec)
3. Match a pure-Python reference implementation of the empirical p-value

Reference
---------
scikit-survival RSF tutorial on GBSG2:
    https://scikit-survival.readthedocs.io/en/stable/user_guide/random-survival-forest.html

Altmann A, Tolosi L, Sander O, Lengauer T (2010).
    Permutation importance: a corrected feature importance measure.
    Bioinformatics, 26(10), 1340-1347.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from mlsurv.results.containers import BootstrapResult
from mlsurv.evaluation._paired_comparison import _adjust_p_values


# ---------------------------------------------------------------------------
# Pure-Python reference for empirical p-value
# ---------------------------------------------------------------------------

def _ref_importance_test(importance_df, model_type=None, ci=0.95):
    """Reference implementation: compute importance summary from scratch.

    Intentionally uses a different code path (iterrows + manual loops)
    to cross-check the production vectorised implementation.
    """
    df = importance_df.copy()
    if model_type is not None:
        df = df[df["model_type"] == model_type]

    alpha = (1 - ci) / 2
    results = []
    for feat in sorted(df["feature"].unique()):
        vals = df.loc[df["feature"] == feat, "importance"].values.astype(float)
        finite = vals[np.isfinite(vals)]
        n = len(finite)
        if n == 0:
            results.append({
                "feature": feat,
                "mean": np.nan, "std": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan,
                "p_value": np.nan,
            })
            continue

        # Manual computation — no numpy shortcuts
        total = 0.0
        count_le_zero = 0
        for v in finite:
            total += v
            if v <= 0:
                count_le_zero += 1
        mean = total / n

        if n > 1:
            ss = sum((v - mean) ** 2 for v in finite)
            std = (ss / (n - 1)) ** 0.5
        else:
            std = 0.0

        ci_lower = np.percentile(finite, 100 * alpha)
        ci_upper = np.percentile(finite, 100 * (1 - alpha))
        p_value = count_le_zero / n

        results.append({
            "feature": feat,
            "mean": mean, "std": std,
            "ci_lower": float(ci_lower), "ci_upper": float(ci_upper),
            "p_value": p_value,
        })

    ref_df = pd.DataFrame(results).sort_values("p_value", ignore_index=True)
    ref_df["p_adjusted"] = _adjust_p_values(
        ref_df["p_value"].values, method="holm",
    )
    return ref_df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_bootstrap_result(importance_df):
    """Build a minimal BootstrapResult with given importance DataFrame."""
    models = importance_df["model_type"].unique()
    summary = pd.DataFrame({
        "model_type": list(models),
        "metric": ["harrell"] * len(models),
        "mean": [0.70] * len(models),
        "std": [0.02] * len(models),
        "ci_lower": [0.66] * len(models),
        "ci_upper": [0.74] * len(models),
    })
    n_iter = importance_df["iteration"].nunique()
    iterations_rows = []
    for m in models:
        for i in range(n_iter):
            iterations_rows.append({
                "model_type": m, "iteration": i,
                "metric_name": "harrell", "value": 0.70 + 0.01 * i,
            })
    iterations = pd.DataFrame(iterations_rows)
    return BootstrapResult(
        summary=summary, iterations=iterations, importance=importance_df,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.validation
class TestImportancePvalueReference:
    """Cross-check production importance_summary() against reference impl."""

    def test_reference_matches_production_deterministic(self):
        """Deterministic data: reference and production give identical results."""
        n = 100
        rng = np.random.RandomState(42)
        importance = pd.DataFrame({
            "model_type": ["rsf"] * (n * 4),
            "iteration": list(range(n)) * 4,
            "feature": (
                ["pnodes"] * n + ["progrec"] * n
                + ["age"] * n + ["tsize"] * n
            ),
            "importance": np.concatenate([
                rng.normal(0.08, 0.02, n),   # clearly positive
                rng.normal(0.04, 0.015, n),  # positive
                rng.normal(0.01, 0.01, n),   # weakly positive
                rng.normal(-0.002, 0.01, n), # near zero / negative
            ]),
        })
        result = _make_bootstrap_result(importance)

        prod = result.importance_summary(p_adjust="holm")
        ref = _ref_importance_test(importance, ci=0.95)

        # Same features in same order
        assert list(prod["feature"]) == list(ref["feature"])

        # All statistics match
        for col in ["mean", "std", "ci_lower", "ci_upper", "p_value", "p_adjusted"]:
            np.testing.assert_allclose(
                prod[col].values, ref[col].values, atol=1e-8,
                err_msg=f"Column {col} mismatch",
            )

    def test_reference_matches_production_with_nans(self):
        """NaN values: both implementations exclude them identically."""
        n = 50
        rng = np.random.RandomState(7)
        vals = rng.normal(0.05, 0.02, n)
        # Inject NaNs at known positions
        vals[10] = np.nan
        vals[25] = np.nan
        vals[40] = np.inf

        importance = pd.DataFrame({
            "model_type": ["rsf"] * n,
            "iteration": list(range(n)),
            "feature": ["age"] * n,
            "importance": vals,
        })
        result = _make_bootstrap_result(importance)

        prod = result.importance_summary(p_adjust="holm")
        ref = _ref_importance_test(importance, ci=0.95)

        assert prod["mean"].iloc[0] == pytest.approx(ref["mean"].iloc[0])
        assert prod["std"].iloc[0] == pytest.approx(ref["std"].iloc[0])
        assert prod["p_value"].iloc[0] == pytest.approx(ref["p_value"].iloc[0])


@pytest.mark.validation
class TestImportancePvalueGBSG2:
    """Validate importance p-values against known GBSG2 feature significance.

    The scikit-survival RSF tutorial and clinical literature consistently
    identify pnodes (positive lymph nodes) and progrec (progesterone
    receptor) as the most important prognostic features in GBSG2.
    """

    @pytest.fixture(scope="class")
    def gbsg2_bootstrap(self):
        """Run bootstrap with importance on GBSG2 (cached per class)."""
        import warnings
        import sys
        import io
        warnings.filterwarnings("ignore")

        from sksurv.datasets import load_gbsg2
        from mlsurv import SurvivalLearner
        from mlsurv.utils import to_structured_array

        X_raw, y_raw = load_gbsg2()
        Xt = X_raw.copy()
        Xt["horTh"] = (Xt["horTh"] == "yes").astype(int)
        Xt["menostat"] = (Xt["menostat"] == "Post").astype(int)
        Xt["tgrade"] = Xt["tgrade"].map({"I": 1, "II": 2, "III": 3}).astype(int)
        y = to_structured_array(y_raw["time"], y_raw["cens"])

        # Suppress mlsurv console output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            learner = SurvivalLearner(Xt, y)
            learner.setup(test_fraction=0.25, random_state=20, cv_folds=5)
            learner.train("rsf")
            boot = learner.bootstrap(
                "rsf", n_bootstrap=50,
                include_importance=True, random_state=42,
            )
        finally:
            sys.stdout = old_stdout

        return boot

    def test_pnodes_and_progrec_significant(self, gbsg2_bootstrap):
        """pnodes and progrec should have adjusted p < 0.05."""
        df = gbsg2_bootstrap.importance_summary(p_adjust="holm")
        sig = set(df.loc[df["p_adjusted"] < 0.05, "feature"])
        assert "pnodes" in sig, f"pnodes not significant: {df.to_dict('records')}"
        assert "progrec" in sig, f"progrec not significant: {df.to_dict('records')}"

    def test_pnodes_positive_importance(self, gbsg2_bootstrap):
        """pnodes should have clearly positive mean importance."""
        df = gbsg2_bootstrap.importance_summary()
        pnodes = df.loc[df["feature"] == "pnodes"]
        assert pnodes["mean"].iloc[0] > 0.03  # well above zero
        assert pnodes["ci_lower"].iloc[0] > 0  # CI doesn't cross zero

    def test_ranking_correlates_with_sklearn(self, gbsg2_bootstrap):
        """Feature ranking should correlate with sklearn permutation_importance.

        We don't require exact rank match (methods differ), but Spearman
        rho > 0.5 indicates the rankings are meaningfully correlated.
        """
        import warnings
        warnings.filterwarnings("ignore")
        from sksurv.datasets import load_gbsg2
        from sksurv.ensemble import RandomSurvivalForest
        from sklearn.model_selection import train_test_split
        from sklearn.inspection import permutation_importance
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
            n_estimators=1000, min_samples_split=10,
            min_samples_leaf=15, n_jobs=-1, random_state=20,
        )
        rsf.fit(X_train, y_train)
        sk_result = permutation_importance(
            rsf, X_test, y_test, n_repeats=15, random_state=20,
        )

        # Build sklearn ranking (by mean importance, descending)
        sk_rank = {
            feat: i for i, feat in enumerate(
                Xt.columns[np.argsort(-sk_result.importances_mean)]
            )
        }

        # Build our ranking (by p_value ascending = most significant first)
        df = gbsg2_bootstrap.importance_summary()
        our_rank = {row["feature"]: i for i, (_, row) in enumerate(df.iterrows())}

        # Compare on shared features
        feats = list(Xt.columns)
        ranks_sk = [sk_rank[f] for f in feats]
        ranks_ours = [our_rank[f] for f in feats]
        rho, _ = spearmanr(ranks_sk, ranks_ours)

        assert rho > 0.5, (
            f"Spearman rho={rho:.3f} too low. "
            f"sklearn ranking: {sk_rank}, ours: {our_rank}"
        )

    def test_nonsignificant_features_have_ci_crossing_zero(self, gbsg2_bootstrap):
        """Features with p_adjusted >= 0.05 should have CIs crossing zero."""
        df = gbsg2_bootstrap.importance_summary(p_adjust="holm")
        nonsig = df[df["p_adjusted"] >= 0.05]
        for _, row in nonsig.iterrows():
            assert row["ci_lower"] <= 0, (
                f"{row['feature']}: ci_lower={row['ci_lower']:.4f} > 0 "
                f"but p_adjusted={row['p_adjusted']:.3f}"
            )
