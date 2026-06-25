"""Unit tests for ``linearity_test`` and helpers.

Covers the standalone Harrell/rms single-fit Wald implementation in
``mlsurv.evaluation.metrics`` — schema, known-linear / known-nonlinear
behavior, closed-form cross-check against a hand-computed Wald chi-square,
FDR correction, empty-data paths, sample-size guards, knot-collapse, and
the ``ConvergenceError`` → ``penalizer=0.01`` fallback.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import MethodologicalWarning, SuggestionWarning
from mlsurv.evaluation.metrics import (
    linearity_test,
    _rcs_basis,
    _identify_continuous_features,
    _build_linearity_design,
    _LINEARITY_COLUMNS,
)


# ---------------------------------------------------------------------------
# Fixtures — small data generators
# ---------------------------------------------------------------------------


def _make_linear(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, 4)),
        columns=[f"x{i}" for i in range(4)],
    )
    lp = 0.5 * X["x0"] - 0.3 * X["x1"] + 0.2 * X["x2"]
    time = rng.exponential(scale=np.exp(-lp))
    event = rng.random(n) < 0.7
    y = np.array(
        [(bool(e), max(t, 0.01)) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


def _make_quadratic(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, 4)),
        columns=[f"x{i}" for i in range(4)],
    )
    lp = 0.8 * (X["x0"] ** 2 - 1) - 0.3 * X["x1"]
    time = rng.exponential(scale=np.exp(-lp))
    event = rng.random(n) < 0.7
    y = np.array(
        [(bool(e), max(t, 0.01)) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


# ---------------------------------------------------------------------------
# _rcs_basis + _identify_continuous_features
# ---------------------------------------------------------------------------


class TestRcsBasis:
    def test_shape_4_knots(self):
        x = np.linspace(-3, 3, 50)
        knots = np.percentile(x, [5, 35, 65, 95])
        basis = _rcs_basis(x, knots)
        assert basis.shape == (50, 2)  # n_knots - 2 nonlinear columns

    def test_shape_5_knots(self):
        x = np.linspace(-3, 3, 100)
        knots = np.percentile(x, [5, 27.5, 50, 72.5, 95])
        basis = _rcs_basis(x, knots)
        assert basis.shape == (100, 3)

    def test_at_first_knot_is_zero(self):
        """Before the first knot, the positive-part cubes are all zero."""
        x = np.array([-10.0, -5.0, -1.0])
        knots = np.array([0.0, 1.0, 2.0, 3.0])
        basis = _rcs_basis(x, knots)
        np.testing.assert_allclose(basis, np.zeros_like(basis))

    def test_requires_min_three_knots(self):
        with pytest.raises(ValueError, match=">= 3 knots"):
            _rcs_basis(np.array([0.0, 1.0]), np.array([0.0, 1.0]))


class TestIdentifyContinuous:
    def test_numeric_with_many_unique(self):
        X = pd.DataFrame({
            "cont": np.linspace(0, 10, 20),
            "binary": [0, 1] * 10,
            "low_ord": [0, 1, 2] * 6 + [0, 1],
        })
        assert _identify_continuous_features(X) == ["cont"]

    def test_skips_text_columns(self):
        X = pd.DataFrame({
            "cont": np.linspace(0, 10, 20),
            "text": ["a"] * 20,
        })
        assert _identify_continuous_features(X) == ["cont"]

    def test_min_unique_override(self):
        X = pd.DataFrame({"vals": [0, 1, 2] * 5})
        assert _identify_continuous_features(X, min_unique=3) == ["vals"]
        assert _identify_continuous_features(X, min_unique=6) == []


# ---------------------------------------------------------------------------
# linearity_test — schema
# ---------------------------------------------------------------------------


class TestLinearitySchema:
    def test_columns(self):
        X, y = _make_linear()
        df = linearity_test(X, y)
        assert list(df.columns) == _LINEARITY_COLUMNS

    def test_has_global_row(self):
        X, y = _make_linear()
        df = linearity_test(X, y)
        assert "GLOBAL" in df["feature"].values

    def test_per_feature_rows_are_continuous_only(self):
        """All per-feature rows must be features identified as continuous."""
        X, y = _make_linear()
        X["binary"] = np.tile([0.0, 1.0], len(X) // 2)
        df = linearity_test(X, y)
        per_feature = df[df["feature"] != "GLOBAL"]["feature"].tolist()
        assert "binary" not in per_feature

    def test_p_adjusted_ge_p_value_per_feature(self):
        """BH FDR correction always yields p_adjusted >= p_value."""
        X, y = _make_linear()
        df = linearity_test(X, y)
        per_feature = df[df["feature"] != "GLOBAL"]
        assert (per_feature["p_adjusted"] >= per_feature["p_value"]).all()

    def test_global_row_p_adjusted_is_nan(self):
        X, y = _make_linear()
        df = linearity_test(X, y)
        global_row = df[df["feature"] == "GLOBAL"].iloc[0]
        assert pd.isna(global_row["p_adjusted"])


# ---------------------------------------------------------------------------
# linearity_test — detection power
# ---------------------------------------------------------------------------


class TestLinearityDetection:
    def test_linear_data_not_significant(self):
        X, y = _make_linear()
        df = linearity_test(X, y)
        global_row = df[df["feature"] == "GLOBAL"].iloc[0]
        assert not global_row["significant"], (
            f"expected linear data not significant, got p={global_row['p_value']}"
        )
        # No per-feature violations either
        per_feature = df[df["feature"] != "GLOBAL"]
        assert not per_feature["significant"].any()

    def test_quadratic_flags_x0(self):
        X, y = _make_quadratic()
        df = linearity_test(X, y)
        global_row = df[df["feature"] == "GLOBAL"].iloc[0]
        assert global_row["significant"], (
            f"expected quadratic GLOBAL significant, got p={global_row['p_value']}"
        )
        violations = df[(df["feature"] != "GLOBAL") & df["significant"]]
        assert "x0" in violations["feature"].tolist()

    def test_quadratic_other_features_not_flagged(self):
        X, y = _make_quadratic()
        df = linearity_test(X, y)
        # x1, x2, x3 enter linearly (or not at all) → should not be flagged
        per_feature = df[df["feature"] != "GLOBAL"]
        non_x0 = per_feature[per_feature["feature"] != "x0"]
        assert not non_x0["significant"].any(), (
            f"unexpected violations on linear features: "
            f"{non_x0[non_x0['significant']]['feature'].tolist()}"
        )


# ---------------------------------------------------------------------------
# linearity_test — closed-form Wald cross-check
# ---------------------------------------------------------------------------


class TestWaldClosedForm:
    """Pin the Wald computation to its published definition.

    Uses ``lifelines.CoxPHFitter`` directly on the same stacked RCS design
    built by ``_build_linearity_design``, extracts the non-linear block's
    coefficient vector and covariance submatrix, computes the Wald
    chi-square by hand (``β̂ᵀ V⁻¹ β̂``), and asserts the production code
    matches within floating-point tolerance.
    """

    def test_hand_computed_wald_matches(self):
        from lifelines import CoxPHFitter
        from scipy.stats import chi2

        X, y = _make_quadratic(n=300, seed=1)
        # Use the same helper the production code uses so we compute on an
        # identical design matrix
        continuous = _identify_continuous_features(X)
        design, nonlinear_cols, _skipped = _build_linearity_design(
            X, continuous, n_knots=4,
        )
        design = design.copy()
        design["T"] = y["time"]
        design["E"] = y["event"].astype(int)

        cph = CoxPHFitter(penalizer=0)
        cph.fit(design, duration_col="T", event_col="E")

        # Hand-compute the x0 Wald chi-square
        nl_cols = nonlinear_cols["x0"]
        beta = cph.params_.loc[nl_cols].to_numpy(dtype=float)
        V = cph.variance_matrix_.loc[nl_cols, nl_cols].to_numpy(dtype=float)
        W_ref = float(beta @ np.linalg.solve(V, beta))
        p_ref = float(chi2.sf(W_ref, df=len(nl_cols)))

        # Production path
        df = linearity_test(X, y, n_knots=4)
        row = df[df["feature"] == "x0"].iloc[0]
        np.testing.assert_allclose(
            row["test_statistic"], W_ref, rtol=1e-6,
            err_msg="production Wald stat differs from hand-computed reference",
        )
        np.testing.assert_allclose(
            row["p_value"], p_ref, rtol=1e-6,
            err_msg="production p-value differs from hand-computed reference",
        )

    def test_global_row_matches_joint_wald(self):
        from lifelines import CoxPHFitter
        from scipy.stats import chi2

        X, y = _make_quadratic(n=300, seed=1)
        continuous = _identify_continuous_features(X)
        design, nonlinear_cols, _skipped = _build_linearity_design(
            X, continuous, n_knots=4,
        )
        design = design.copy()
        design["T"] = y["time"]
        design["E"] = y["event"].astype(int)

        cph = CoxPHFitter(penalizer=0)
        cph.fit(design, duration_col="T", event_col="E")

        all_nl = [c for f in continuous for c in nonlinear_cols[f]]
        beta = cph.params_.loc[all_nl].to_numpy(dtype=float)
        V = cph.variance_matrix_.loc[all_nl, all_nl].to_numpy(dtype=float)
        W_ref = float(beta @ np.linalg.solve(V, beta))
        p_ref = float(chi2.sf(W_ref, df=len(all_nl)))

        df = linearity_test(X, y, n_knots=4)
        global_row = df[df["feature"] == "GLOBAL"].iloc[0]
        np.testing.assert_allclose(
            global_row["test_statistic"], W_ref, rtol=1e-6,
        )
        np.testing.assert_allclose(
            global_row["p_value"], p_ref, rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# linearity_test — edge cases
# ---------------------------------------------------------------------------


class TestLinearityEdgeCases:
    def test_no_continuous_features_returns_empty(self):
        n = 200
        rng = np.random.default_rng(0)
        X = pd.DataFrame({f"b{i}": rng.integers(0, 2, n).astype(float)
                          for i in range(4)})
        lp = 0.5 * X["b0"]
        time = rng.exponential(scale=np.exp(-lp))
        event = rng.random(n) < 0.7
        y = np.array(
            [(bool(e), max(t, 0.01)) for e, t in zip(event, time)],
            dtype=[("event", bool), ("time", float)],
        )
        df = linearity_test(X, y)
        assert df.empty
        assert list(df.columns) == _LINEARITY_COLUMNS

    def test_insufficient_rows_returns_empty(self):
        X, y = _make_linear(n=15)
        df = linearity_test(X, y)
        assert df.empty

    def test_underpowered_warning_fires(self):
        """20 <= n < 100 → SuggestionWarning about underpower."""
        X, y = _make_linear(n=60)
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            df = linearity_test(X, y)
        underpowered = [
            w for w in w_list
            if issubclass(w.category, SuggestionWarning)
            and "underpowered" in str(w.message).lower()
        ]
        assert len(underpowered) >= 1
        # Still returns a populated result (warning is a soft flag)
        assert not df.empty

    def test_knot_collapse_skips_feature(self):
        """Feature with a mass point → knots collapse → skipped (not crashed)."""
        n = 400
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            rng.standard_normal((n, 3)),
            columns=["x0", "x1", "x2"],
        )
        # Massive mass point at zero for x2 (>60% of rows = 0)
        X["x2"] = 0.0
        X.loc[: int(0.35 * n), "x2"] = rng.standard_normal(int(0.35 * n) + 1)
        lp = 0.5 * X["x0"] - 0.3 * X["x1"]
        time = rng.exponential(scale=np.exp(-lp))
        event = rng.random(n) < 0.7
        y = np.array(
            [(bool(e), max(t, 0.01)) for e, t in zip(event, time)],
            dtype=[("event", bool), ("time", float)],
        )
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            df = linearity_test(X, y)

        # User-visible SuggestionWarning should name x2
        skip_warnings = [
            w for w in w_list
            if issubclass(w.category, SuggestionWarning)
            and "x2" in str(w.message)
            and "skipped" in str(w.message)
        ]
        assert len(skip_warnings) >= 1, (
            f"expected SuggestionWarning mentioning x2; got "
            f"{[(w.category.__name__, str(w.message)[:100]) for w in w_list]}"
        )
        # Behavior: x0 and x1 still tested, x2 excluded
        per_feature = df[df["feature"] != "GLOBAL"]["feature"].tolist()
        assert "x0" in per_feature
        assert "x1" in per_feature
        assert "x2" not in per_feature

    def test_drops_zero_variance_column(self):
        """Constant column is dropped before Cox fit; remaining continuous features covered."""
        X, y = _make_linear()
        X["const_feat"] = 0.0
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            df = linearity_test(X, y)

        # Result populated for the original 4 continuous features
        assert not df.empty
        assert "const_feat" not in df["feature"].values

        drop = [
            w for w in w_list
            if issubclass(w.category, SuggestionWarning)
            and "zero-variance" in str(w.message)
            and "const_feat" in str(w.message)
        ]
        assert len(drop) >= 1

    def test_unrecoverable_convergence_returns_empty_with_warning(self, monkeypatch):
        """Every penalty in ladder fails → empty result + MethodologicalWarning."""
        from lifelines import CoxPHFitter
        from lifelines.exceptions import ConvergenceError

        def always_fail(self, *args, **kwargs):
            raise ConvergenceError("simulated total failure")

        monkeypatch.setattr(CoxPHFitter, "fit", always_fail)

        X, y = _make_linear()
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            df = linearity_test(X, y)

        assert df.empty
        ladder = [
            w for w in w_list
            if issubclass(w.category, MethodologicalWarning)
            and "linearity_test" in str(w.message)
            and "ladder" in str(w.message)
        ]
        assert len(ladder) >= 1, (
            f"expected MethodologicalWarning naming ladder exhaustion; got "
            f"{[(w.category.__name__, str(w.message)[:120]) for w in w_list]}"
        )

    def test_convergence_fallback_uses_penalizer(self, monkeypatch):
        """ConvergenceError on first fit → retry with penalizer=0.01 + warning."""
        from lifelines import CoxPHFitter
        from lifelines.exceptions import ConvergenceError

        call_log = []
        original_fit = CoxPHFitter.fit

        def flaky_fit(self, *args, **kwargs):
            call_log.append(self.penalizer)
            if self.penalizer == 0 and len(call_log) == 1:
                raise ConvergenceError("simulated unpenalized fit failure")
            return original_fit(self, *args, **kwargs)

        monkeypatch.setattr(CoxPHFitter, "fit", flaky_fit)

        X, y = _make_linear()
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            df = linearity_test(X, y)

        # Unpenalized attempt first, then fallback
        assert call_log[0] == 0
        assert 0.01 in call_log[1:]
        # Result still populated
        assert not df.empty
        # Approximation warning emitted
        approx = [
            w for w in w_list
            if issubclass(w.category, SuggestionWarning)
            and "approximation" in str(w.message).lower()
        ]
        assert len(approx) >= 1

    def test_convergence_detail_logged_at_debug_not_warning(
        self, monkeypatch, caplog
    ):
        """The per-rung 'did not converge' lifelines detail must be
        logged at DEBUG, not WARNING — the user-facing SuggestionWarning
        already covers the actionable fact, so the logger line would only
        duplicate it in the default stderr stream.
        """
        import logging
        from lifelines import CoxPHFitter
        from lifelines.exceptions import ConvergenceError

        call_log = []
        original_fit = CoxPHFitter.fit

        def flaky_fit(self, *args, **kwargs):
            call_log.append(self.penalizer)
            if self.penalizer == 0 and len(call_log) == 1:
                raise ConvergenceError("simulated unpenalized fit failure")
            return original_fit(self, *args, **kwargs)

        monkeypatch.setattr(CoxPHFitter, "fit", flaky_fit)

        X, y = _make_linear()
        with caplog.at_level(logging.DEBUG, logger="mlsurv.metrics"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = linearity_test(X, y)

        assert not df.empty
        convergence_records = [
            r for r in caplog.records
            if "did not converge at penalizer" in r.getMessage()
        ]
        assert convergence_records, (
            "expected the convergence detail to be logged at DEBUG"
        )
        assert all(r.levelno == logging.DEBUG for r in convergence_records), (
            f"convergence detail should be DEBUG, got levels "
            f"{[r.levelname for r in convergence_records]}"
        )

    def test_wald_pinv_failure_detail_logged_at_debug_not_warning(
        self, monkeypatch, caplog
    ):
        """When both solve and pinv fail in _wald_chi2, the raw-exception
        detail is logged at DEBUG, not WARNING. The caller already surfaces the
        non-finite feature to the user via a MethodologicalWarning, so a WARNING
        line here would be redundant stderr noise.
        """
        import logging
        import numpy as np
        from mlsurv.evaluation import metrics

        def boom(*args, **kwargs):
            raise np.linalg.LinAlgError("simulated singular covariance")

        monkeypatch.setattr(metrics.np.linalg, "solve", boom)
        monkeypatch.setattr(metrics.np.linalg, "pinv", boom)

        beta = np.array([1.0, 2.0])
        V = np.eye(2)
        with caplog.at_level(logging.DEBUG, logger="mlsurv.metrics"):
            result = metrics._wald_chi2(beta, V)

        assert np.isnan(result)
        pinv_records = [
            r for r in caplog.records
            if "pseudoinverse failed" in r.getMessage()
        ]
        assert pinv_records, "expected the pinv-failure detail to be logged at DEBUG"
        assert all(r.levelno == logging.DEBUG for r in pinv_records), (
            f"pinv-failure detail should be DEBUG, got levels "
            f"{[r.levelname for r in pinv_records]}"
        )

    def test_linalg_warning_suppressed_on_fallback(self, monkeypatch):
        """scipy LinAlgWarning from a failed ladder rung must not
        leak — the fallback loop absorbs it and the user-facing fallback
        SuggestionWarning still fires for the successful rung.
        """
        from lifelines import CoxPHFitter
        from lifelines.exceptions import ConvergenceError
        from scipy.linalg import LinAlgWarning

        original_fit = CoxPHFitter.fit
        call_log = []

        def flaky_fit(self, *args, **kwargs):
            call_log.append(self.penalizer)
            if self.penalizer == 0 and len(call_log) == 1:
                warnings.warn("ill-conditioned matrix", LinAlgWarning)
                raise ConvergenceError("simulated unpenalized fit failure")
            return original_fit(self, *args, **kwargs)

        monkeypatch.setattr(CoxPHFitter, "fit", flaky_fit)

        X, y = _make_linear()
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            df = linearity_test(X, y)

        assert not df.empty
        linalg = [w for w in w_list if issubclass(w.category, LinAlgWarning)]
        assert linalg == [], (
            f"LinAlgWarning leaked from fallback ladder: "
            f"{[str(w.message) for w in linalg]}"
        )
        suggestion = [
            w for w in w_list
            if issubclass(w.category, SuggestionWarning)
            and "approximation" in str(w.message).lower()
        ]
        assert len(suggestion) >= 1, (
            "ridge-fallback SuggestionWarning should still fire"
        )

    def test_ladder_exhaustion_still_warns_methodological(self, monkeypatch):
        """When the entire ladder fails (each rung
        emitting LinAlgWarning), the final MethodologicalWarning must
        still reach the user — suppression is per-attempt only.
        """
        from lifelines import CoxPHFitter
        from lifelines.exceptions import ConvergenceError
        from scipy.linalg import LinAlgWarning

        def always_fails(self, *args, **kwargs):
            warnings.warn("ill-conditioned matrix", LinAlgWarning)
            raise ConvergenceError("simulated failure at every rung")

        monkeypatch.setattr(CoxPHFitter, "fit", always_fails)

        X, y = _make_linear()
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            df = linearity_test(X, y)

        assert df.empty
        ladder = [
            w for w in w_list
            if issubclass(w.category, MethodologicalWarning)
            and "ladder" in str(w.message)
        ]
        assert len(ladder) >= 1, (
            "ladder-exhaustion MethodologicalWarning must remain user-visible"
        )

    def test_convergence_warning_suppressed_on_fallback(self, monkeypatch):
        """lifelines ConvergenceWarning emitted alongside a
        ConvergenceError on a discarded ladder rung must also be
        suppressed — this is a sibling of the LinAlgWarning case.
        """
        from lifelines import CoxPHFitter
        from lifelines.exceptions import ConvergenceError, ConvergenceWarning

        original_fit = CoxPHFitter.fit
        call_log = []

        def flaky_fit(self, *args, **kwargs):
            call_log.append(self.penalizer)
            if self.penalizer == 0 and len(call_log) == 1:
                warnings.warn("did not converge", ConvergenceWarning)
                raise ConvergenceError("simulated unpenalized fit failure")
            return original_fit(self, *args, **kwargs)

        monkeypatch.setattr(CoxPHFitter, "fit", flaky_fit)

        X, y = _make_linear()
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            df = linearity_test(X, y)

        assert not df.empty
        leaked = [
            w for w in w_list
            if issubclass(w.category, ConvergenceWarning)
            and "did not converge" in str(w.message)
        ]
        assert leaked == [], (
            f"ConvergenceWarning from discarded rung leaked: "
            f"{[str(w.message) for w in leaked]}"
        )

    def test_kept_fit_warnings_are_replayed(self, monkeypatch):
        """Silent-failure guard: warnings from the *successful*
        (kept) Cox fit must NOT be suppressed — only warnings from
        attempts that the retry loop discards are silenced. Otherwise
        soft diagnostics like 'Variance matrix is singular; using
        pseudo-inverse' on the returned fit would be invisible.
        """
        from lifelines import CoxPHFitter
        from lifelines.exceptions import ConvergenceWarning

        original_fit = CoxPHFitter.fit

        def chatty_fit(self, *args, **kwargs):
            warnings.warn(
                "variance matrix is singular; pseudo-inverse used",
                ConvergenceWarning,
            )
            return original_fit(self, *args, **kwargs)

        monkeypatch.setattr(CoxPHFitter, "fit", chatty_fit)

        X, y = _make_linear()
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            df = linearity_test(X, y)

        assert not df.empty
        replayed = [
            w for w in w_list
            if issubclass(w.category, ConvergenceWarning)
            and "pseudo-inverse" in str(w.message)
        ]
        assert len(replayed) >= 1, (
            "ConvergenceWarning emitted by the kept (successful) Cox fit "
            "must reach the user — record-and-replay should preserve it"
        )

    # ------------------------------------------------------------------
    # linearity_test follow-ups: n_knots parametrization, hard-floor boundary,
    # dropped_features aggregated warning
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("n_knots", [3, 4, 5, 6, 7])
    def test_linearity_test_supports_all_knot_counts(self, n_knots):
        """Every supported n_knots value produces a populated result on
        well-behaved linear data.

        Guards ``_RCS_KNOT_PERCENTILES`` against typos — only ``n_knots=4``
        was previously exercised end-to-end via ``linearity_test``, so a
        typo in the 3/5/6/7 percentile lists would ship undetected.
        """
        X, y = _make_linear(n=400, seed=0)  # 4 continuous features
        df = linearity_test(X, y, n_knots=n_knots)
        assert not df.empty, f"empty result for n_knots={n_knots}"
        assert "GLOBAL" in df["feature"].values

        per_feature = df[df["feature"] != "GLOBAL"]
        # All per-feature test statistics are finite (not NaN/inf)
        assert per_feature["test_statistic"].apply(np.isfinite).all(), (
            f"non-finite Wald stat at n_knots={n_knots}"
        )
        # Linear data should not flag any feature at any knot count
        assert (per_feature["p_adjusted"] > 0.05).all(), (
            f"linear data spuriously flagged features at n_knots={n_knots}: "
            f"{per_feature[per_feature['p_adjusted'] <= 0.05]['feature'].tolist()}"
        )

    def test_hard_floor_just_below_returns_empty(self):
        """``n = floor - 1`` triggers the hard-floor guard and a
        ``MethodologicalWarning``, returns an empty DataFrame.

        ``_make_linear`` produces 4 continuous features, so
        ``hard_floor = max(20, 10 * 4) = 40``. ``n = 39`` is just below.
        """
        X, y = _make_linear(n=39, seed=0)
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            df = linearity_test(X, y)

        assert df.empty, (
            f"expected empty result below hard floor, got {len(df)} rows"
        )
        floor_warns = [
            w for w in w_list
            if issubclass(w.category, MethodologicalWarning)
            and "hard floor" in str(w.message).lower()
        ]
        assert len(floor_warns) == 1, (
            f"expected one MethodologicalWarning about the hard floor; got "
            f"{[(type(w.message).__name__, str(w.message)[:80]) for w in w_list]}"
        )

    def test_hard_floor_exactly_at_returns_populated(self):
        """``n = floor`` passes the hard-floor check and runs the test.

        At the boundary, the guard uses ``<`` (strict), so ``n = 40``
        with 4 continuous features is accepted. The test may still hit
        the soft-floor ``SuggestionWarning`` (n < 100 is underpowered)
        but the hard-floor ``MethodologicalWarning`` must NOT fire.
        """
        X, y = _make_linear(n=40, seed=0)
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            df = linearity_test(X, y)

        hard_floor_warns = [
            w for w in w_list
            if issubclass(w.category, MethodologicalWarning)
            and "hard floor" in str(w.message).lower()
        ]
        assert len(hard_floor_warns) == 0, (
            "hard-floor branch fired at n = floor (should be exclusive)"
        )
        # The test ran to completion (populated) or hit a downstream
        # non-convergence path — but the hard-floor guard did not fire.
        if not df.empty:
            assert "GLOBAL" in df["feature"].values

    def test_dropped_features_warning_fires_on_partial_drop(self, monkeypatch):
        """Aggregated ``MethodologicalWarning`` fires when ``_wald_chi2``
        returns non-finite for some (but not all) features.

        Uses ``monkeypatch`` to replace the module-level ``_wald_chi2``
        with a wrapper that returns ``inf`` on its first call (so the
        first per-feature test drops) and delegates to the original on
        subsequent calls. After the patch:

        - ``dropped_features`` contains the first continuous feature (``x0``)
        - The aggregated warning fires once and names ``x0``
        - The GLOBAL joint Wald excludes ``x0``'s nonlinear block and is
          still finite
        - Three per-feature rows remain (x1, x2, x3) + GLOBAL
        """
        from mlsurv.evaluation import metrics as _metrics_mod

        real_wald = _metrics_mod._wald_chi2
        call_count = {"n": 0}

        def flaky_wald(beta, V):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return float("inf")
            return real_wald(beta, V)

        monkeypatch.setattr(
            "mlsurv.evaluation.metrics._wald_chi2", flaky_wald,
        )

        X, y = _make_linear(n=400, seed=0)  # 4 continuous features
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            df = linearity_test(X, y)

        # Three per-feature rows remain after dropping the first
        per_feature = df[df["feature"] != "GLOBAL"]
        assert len(per_feature) == 3, (
            f"expected 3 per-feature rows after dropping one; got "
            f"{len(per_feature)}: {per_feature['feature'].tolist()}"
        )

        # Aggregated warning fires exactly once
        drop_warns = [
            w for w in w_list
            if type(w.message) is MethodologicalWarning
            and "non-finite Wald" in str(w.message)
        ]
        assert len(drop_warns) == 1, (
            f"expected one aggregated dropped_features warning; got "
            f"{[(type(w.message).__name__, str(w.message)[:80]) for w in w_list]}"
        )
        msg = str(drop_warns[0].message)
        # x0 is the first continuous feature iterated in column order
        assert "x0" in msg
        assert "excluded" in msg.lower()

        # GLOBAL row still finite and present (its Wald only uses the
        # remaining 3 features' nonlinear blocks)
        global_row = df[df["feature"] == "GLOBAL"].iloc[0]
        assert np.isfinite(global_row["test_statistic"])
