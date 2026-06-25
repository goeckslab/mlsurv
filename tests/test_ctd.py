"""Tests for Antolini's time-dependent concordance index (C^td).

Unit tests (pure numpy), integration tests (evaluate/bootstrap/CV wiring),
and optional validation against pycox's concordance_td.
"""

import numpy as np
import pandas as pd
import pytest

from mlsurv.evaluation._ctd import concordance_td

from tests._helpers import make_survival_array


# ---------------------------------------------------------------------------
# Unit tests — known survival curves with hand-computed C^td
# ---------------------------------------------------------------------------


class TestConcordanceTdUnit:
    """Unit tests for the pure numpy concordance_td implementation."""

    def test_perfect_concordance(self):
        """Patient who dies first has lower survival probability at event time."""
        durations = np.array([1.0, 3.0])
        events = np.array([1, 1])
        eval_times = np.array([1.0, 2.0, 3.0])
        # Patient 0 dies at t=1: S(1|x0)=0.3, S(1|x1)=0.9
        # Patient 1 dies at t=3: S(3|x1)=0.1, S(3|x0)=0.5
        # Pair (0,1): t0<t1, e0=1 → S(1|x0)=0.3 < S(1|x1)=0.9 → concordant
        # Pair (1,0): t1>t0, e1=1 → S(3|x1)=0.1 < S(3|x0)=0.5 → concordant
        surv_probs = np.array([
            [0.3, 0.4, 0.5],  # patient 0
            [0.9, 0.5, 0.1],  # patient 1
        ])
        result = concordance_td(durations, events, surv_probs, eval_times)
        assert result == 1.0

    def test_perfect_discordance(self):
        """Patient who dies first has HIGHER survival probability — fully discordant."""
        durations = np.array([1.0, 3.0])
        events = np.array([1, 1])
        eval_times = np.array([1.0, 2.0, 3.0])
        # Pair (0,1): S(1|x0)=0.9 > S(1|x1)=0.3 → discordant
        # Pair (1,0): t1>t0, but we also need t1<t0 which is false → only pair (0,1)
        surv_probs = np.array([
            [0.9, 0.8, 0.7],  # patient 0 (dies first but high S)
            [0.3, 0.2, 0.1],  # patient 1 (dies later but low S)
        ])
        result = concordance_td(durations, events, surv_probs, eval_times)
        assert result == 0.0

    def test_tied_predictions(self):
        """Tied survival probabilities at event time → 0.5 concordance (adj_antolini)."""
        durations = np.array([1.0, 3.0])
        events = np.array([1, 1])
        eval_times = np.array([1.0, 3.0])
        surv_probs = np.array([
            [0.5, 0.2],
            [0.5, 0.3],  # same S(1|.) for both patients
        ])
        result = concordance_td(durations, events, surv_probs, eval_times)
        # Only comparable pair from patient 0: S(1|x0)=0.5 == S(1|x1)=0.5 → tied
        # Patient 1 event at t=3: no j with t_j > 3
        # Total: 0 concordant, 0 discordant, 1 tied → (0 + 0.5*1)/1 = 0.5
        assert result == 0.5

    def test_all_censored(self):
        """No events → no comparable pairs → NaN."""
        durations = np.array([1.0, 2.0, 3.0])
        events = np.array([0, 0, 0])
        eval_times = np.array([1.0, 2.0, 3.0])
        np.random.seed(42)
        surv_probs = np.random.rand(3, 3)
        result = concordance_td(durations, events, surv_probs, eval_times)
        assert np.isnan(result)

    def test_single_patient(self):
        """Single patient → no comparable pairs → NaN."""
        durations = np.array([1.0])
        events = np.array([1])
        eval_times = np.array([1.0])
        surv_probs = np.array([[0.5]])
        result = concordance_td(durations, events, surv_probs, eval_times)
        assert np.isnan(result)

    def test_all_events_same_time(self):
        """All events at same time → tied-time pairs are comparable.

        With adj_antolini, both-event tied pairs score 1.0 if predictions
        are equal, 0.5 if different.  With identical survival curves,
        all 6 directed pairs (3 patients × 2) score 1.0 → concordance=1.0.
        """
        durations = np.array([2.0, 2.0, 2.0])
        events = np.array([1, 1, 1])
        eval_times = np.array([1.0, 2.0, 3.0])
        # Identical predictions → all tied pairs score 1.0
        surv_probs = np.array([
            [0.9, 0.5, 0.2],
            [0.9, 0.5, 0.2],
            [0.9, 0.5, 0.2],
        ])
        result = concordance_td(durations, events, surv_probs, eval_times)
        assert result == 1.0

        # Different predictions → tied pairs score 0.5
        np.random.seed(0)
        surv_probs_diff = np.random.rand(3, 3)
        result_diff = concordance_td(durations, events, surv_probs_diff, eval_times)
        assert 0 <= result_diff <= 1

    def test_mixed_events_and_censored(self):
        """Censored patient with later time creates a valid comparison pair."""
        durations = np.array([1.0, 5.0])
        events = np.array([1, 0])  # patient 0 dies, patient 1 censored
        eval_times = np.array([1.0, 3.0, 5.0])
        # Pair (0,1): t0=1 < t1=5, e0=1 → comparable
        # S(1|x0)=0.2 < S(1|x1)=0.8 → concordant
        surv_probs = np.array([
            [0.2, 0.1, 0.05],  # patient 0 (low survival)
            [0.8, 0.7, 0.6],   # patient 1 (high survival)
        ])
        result = concordance_td(durations, events, surv_probs, eval_times)
        assert result == 1.0

    def test_ph_equivalent(self):
        """Under proportional hazards (constant HR), C^td ≈ Harrell's C.

        If risk is proportional (no time-varying hazard ratio), the
        ranking at every time point is the same, making C^td equivalent
        to Harrell's C-index.
        """
        rng = np.random.RandomState(42)
        n = 200
        eval_times = np.linspace(0.1, 10.0, 50)

        # Generate data from an exponential model (constant hazard)
        baseline_hazard = 0.1
        betas = rng.randn(3) * 0.5
        X = rng.randn(n, 3)
        log_hr = X @ betas
        hazard = baseline_hazard * np.exp(log_hr)
        durations = rng.exponential(1.0 / hazard)
        events = np.ones(n)  # all observed

        # Survival probs under the true model: S(t|x) = exp(-h*t)
        surv_probs = np.exp(-np.outer(hazard, eval_times))

        ctd = concordance_td(durations, events, surv_probs, eval_times)

        # Harrell's C: higher risk → higher hazard
        from sksurv.metrics import concordance_index_censored
        y_struct = make_survival_array(events, durations)
        harrell_c = concordance_index_censored(
            y_struct['event'], y_struct['time'], hazard
        )[0]

        assert abs(ctd - harrell_c) < 0.02, (
            f"Under PH, C^td should ≈ Harrell's C: "
            f"C^td={ctd:.4f}, Harrell={harrell_c:.4f}"
        )

    def test_eval_times_not_aligned(self):
        """Event times not in eval_times grid → rounds down to largest t <= duration."""
        durations = np.array([1.5, 4.5])  # not on the grid
        events = np.array([1, 1])
        eval_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # dur=1.5 → round down to t=1.0: S(1|x0)=0.5 < S(1|x1)=0.9 → concordant
        # dur=4.5 → round down to t=4.0: S(4|x1)=0.3 < S(4|x0)=0.5 → concordant
        surv_probs = np.array([
            [0.5, 0.3, 0.4, 0.5, 0.6],
            [0.9, 0.8, 0.6, 0.3, 0.1],
        ])
        result = concordance_td(durations, events, surv_probs, eval_times)
        assert result == 1.0

    def test_empty_eval_times(self):
        """Empty eval_times → NaN."""
        durations = np.array([1.0, 2.0])
        events = np.array([1, 1])
        eval_times = np.array([])
        surv_probs = np.empty((2, 0))
        result = concordance_td(durations, events, surv_probs, eval_times)
        assert np.isnan(result)

    def test_returns_float(self):
        """Result should always be a plain float (or NaN)."""
        durations = np.array([1.0, 2.0, 3.0])
        events = np.array([1, 0, 1])
        eval_times = np.array([1.0, 2.0, 3.0])
        surv_probs = np.array([
            [0.3, 0.2, 0.1],
            [0.8, 0.7, 0.6],
            [0.5, 0.4, 0.3],
        ])
        result = concordance_td(durations, events, surv_probs, eval_times)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_tied_time_both_events_equal_predictions(self):
        """Two events at same time with identical predictions → concordance = 1.0."""
        durations = np.array([2.0, 2.0])
        events = np.array([1, 1])
        eval_times = np.array([1.0, 2.0, 3.0])
        surv_probs = np.array([
            [0.9, 0.5, 0.2],  # same predictions
            [0.9, 0.5, 0.2],
        ])
        result = concordance_td(durations, events, surv_probs, eval_times)
        # adj_antolini: both events, equal predictions → concordance = 1.0
        assert result == 1.0

    def test_tied_time_both_events_different_predictions(self):
        """Two events at same time with different predictions → concordance = 0.5."""
        durations = np.array([2.0, 2.0])
        events = np.array([1, 1])
        eval_times = np.array([1.0, 2.0, 3.0])
        surv_probs = np.array([
            [0.9, 0.3, 0.1],  # lower survival at t=2
            [0.9, 0.7, 0.5],  # higher survival at t=2
        ])
        result = concordance_td(durations, events, surv_probs, eval_times)
        # adj_antolini: both events, different predictions → 0.5 each direction
        # (i,j): 1 - 0.5*(0.3 != 0.7) = 0.5;  (j,i): 1 - 0.5*(0.7 != 0.3) = 0.5
        # total concordant = 1.0, comparable = 2 → C^td = 0.5
        assert result == 0.5

    def test_tied_time_event_and_censored(self):
        """Event + censored at same time → comparable pair."""
        durations = np.array([2.0, 2.0])
        events = np.array([1, 0])  # patient 0 event, patient 1 censored
        eval_times = np.array([1.0, 2.0, 3.0])
        # Patient 0 (event) has lower survival → concordant
        surv_probs = np.array([
            [0.9, 0.3, 0.1],  # event patient: low survival
            [0.9, 0.7, 0.5],  # censored patient: high survival
        ])
        result = concordance_td(durations, events, surv_probs, eval_times)
        # (0,1): d_0=1, d_1=0, t_0==t_1 → comparable
        # S(2|x0)=0.3 < S(2|x1)=0.7 → concordant = 1.0
        # (1,0): d_1=0, d_0=1, t_1==t_0 → comparable (d_i|d_j = True)
        # S(2|x1)=0.7 > S(2|x0)=0.3 → concordant = 1.0
        # total concordant = 2.0, comparable = 2 → C^td = 1.0
        assert result == 1.0

    def test_matches_pycox_with_tied_times(self):
        """C^td should match pycox's adj_antolini on data with tied event times."""
        pytest.importorskip("pycox")
        from pycox.evaluation.concordance import concordance_td as pycox_ctd

        rng = np.random.RandomState(42)
        n = 100
        # Create data with many tied times (integer durations)
        durations = rng.randint(1, 20, size=n).astype(float)
        events = rng.binomial(1, 0.7, size=n).astype('int32')
        eval_times = np.arange(0, 21, dtype=float)
        surv_probs = np.sort(rng.rand(n, len(eval_times)), axis=1)[:, ::-1]

        # mlsurv
        ctd_ours = concordance_td(durations, events, surv_probs, eval_times)

        # pycox (expects surv[n_times, n] and surv_idx)
        surv_pycox = surv_probs.T.copy(order='C')
        surv_idx = np.searchsorted(eval_times, durations, side='right') - 1
        surv_idx = surv_idx.clip(0, len(eval_times) - 1).astype('int32')
        ctd_pycox = pycox_ctd(durations, events, surv_pycox, surv_idx)

        # Cross-implementation comparison — may need loosening if pycox
        # changes numerical internals (see flaky-test audit 2026-04-14).
        assert abs(ctd_ours - ctd_pycox) < 1e-10, (
            f"C^td mismatch: mlsurv={ctd_ours:.8f} vs pycox={ctd_pycox:.8f}"
        )


# ---------------------------------------------------------------------------
# Integration tests — wiring into evaluate / bootstrap / CV
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestConcordanceTdIntegration:
    """Integration tests verifying C^td flows through the evaluation pipeline."""

    @pytest.fixture
    def toy_data(self):
        """Create small toy survival data for quick integration tests."""
        rng = np.random.RandomState(123)
        n = 100
        X = pd.DataFrame(rng.randn(n, 3), columns=['a', 'b', 'c'])
        y = np.array(
            [(bool(rng.rand() > 0.3), rng.exponential(5)) for _ in range(n)],
            dtype=[('event', bool), ('time', float)],
        )
        return X, y

    def test_evaluate_returns_ctd(self, toy_data):
        """evaluate() should produce c_index_td_test in concordance DataFrame."""
        X, y = toy_data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42,
        )
        # Use CoxPH which supports predict_survival_function
        from mlsurv.models.coxph import CoxPHModel
        model = CoxPHModel()
        model.fit(X_train, y_train)

        from mlsurv.evaluation import evaluate
        result = evaluate(model, X_test, y_test, X_train=X_train, y_train=y_train,
                          show_progress=False, n_importance_repeats=0)
        assert 'c_index_td_test' in result.concordance.columns
        val = result.concordance['c_index_td_test'].iloc[0]
        assert not np.isnan(val), "C^td should not be NaN for CoxPH"
        assert 0.0 < val < 1.0

    def test_evaluate_ctd_property(self, toy_data):
        """EvaluationResult.ctd_test property should return the value."""
        X, y = toy_data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42,
        )
        from mlsurv.models.coxph import CoxPHModel
        model = CoxPHModel()
        model.fit(X_train, y_train)

        from mlsurv.evaluation import evaluate
        result = evaluate(model, X_test, y_test, X_train=X_train, y_train=y_train,
                          show_progress=False, n_importance_repeats=0)
        assert result.ctd_test is not None
        assert 0.0 < result.ctd_test < 1.0

    def test_score_survival_model_ctd(self, toy_data):
        """score_survival_model(scoring='ctd') should work."""
        X, y = toy_data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42,
        )
        from mlsurv.models.coxph import CoxPHModel
        model = CoxPHModel()
        model.fit(X_train, y_train)

        from mlsurv.evaluation import score_survival_model
        score = score_survival_model(model, X_test, y_test, scoring='ctd',
                                     y_train=y_train)
        assert isinstance(score, float)
        assert 0.0 < score < 1.0

    def test_ctd_in_constants(self):
        """C^td should be in ALL_METRICS and SCORABLE_METRICS."""
        from mlsurv.constants import ALL_METRICS, SCORABLE_METRICS, METRIC_REGISTRY
        assert 'ctd' in ALL_METRICS
        assert 'ctd' in SCORABLE_METRICS
        assert 'ctd' in METRIC_REGISTRY
        info = METRIC_REGISTRY['ctd']
        assert info.concordance_column == 'c_index_td_test'
        assert info.lower_is_better is False
        assert info.diagnostic_only is False

    def test_ctd_nan_without_surv_probs(self):
        """Models that don't support survival functions should get NaN for C^td."""
        y = make_survival_array(
            [True, False, True], [1.0, 2.0, 3.0]
        )
        from mlsurv.evaluation._concordance import _add_ctd_metric
        data = {}
        _add_ctd_metric(data, y, None, None)
        assert np.isnan(data['c_index_td_test'][0])


# ---------------------------------------------------------------------------
# Optional validation: compare with pycox's concordance_td
# ---------------------------------------------------------------------------


@pytest.mark.validation
class TestConcordanceTdValidation:
    """Compare our numpy C^td against pycox's implementation."""

    @pytest.fixture
    def _skip_if_no_pycox(self):
        pytest.importorskip("pycox")

    def test_matches_pycox(self, _skip_if_no_pycox):
        """Our C^td should match pycox's concordance_td within tolerance."""
        from pycox.evaluation import EvalSurv

        rng = np.random.RandomState(99)
        n = 150
        n_times = 50

        durations = rng.exponential(5, size=n)
        events = rng.binomial(1, 0.7, size=n).astype(float)
        eval_times = np.linspace(0.5, np.percentile(durations, 90), n_times)

        # Generate monotone-decreasing survival curves
        base = rng.rand(n)
        surv_probs = np.zeros((n, n_times))
        for j in range(n_times):
            surv_probs[:, j] = base ** (eval_times[j] / eval_times[0])

        # Our implementation
        our_ctd = concordance_td(durations, events, surv_probs, eval_times)

        # pycox's implementation (uses transposed surv DataFrame)
        surv_df = pd.DataFrame(surv_probs.T, index=eval_times)
        ev = EvalSurv(surv_df, durations, events, censor_surv='km')
        pycox_ctd = ev.concordance_td()

        assert abs(our_ctd - pycox_ctd) < 0.001, (
            f"Mismatch: ours={our_ctd:.6f}, pycox={pycox_ctd:.6f}"
        )
