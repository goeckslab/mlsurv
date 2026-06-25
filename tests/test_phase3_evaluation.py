"""Test evaluation functions for Phase 3."""

import pytest
import numpy as np
import pandas as pd

from tests._helpers import make_survival_array, make_y


class TestMetrics:
    """Test evaluation metrics."""

    def test_surv_funcs_to_probs_accepts_ndarray(self):
        """Verify ndarray survival probabilities are handled directly."""
        from mlsurv.evaluation.metrics import _surv_funcs_to_probs

        eval_times = np.array([1.0, 2.0, 3.0])
        surv_probs = np.array([
            [0.9, 0.8, 0.7],
            [0.95, 0.85, 0.75],
        ])

        converted = _surv_funcs_to_probs(surv_probs, eval_times)
        np.testing.assert_allclose(converted, surv_probs)

        # Accept transposed shape (n_times, n_samples) and align to (n_samples, n_times)
        converted_t = _surv_funcs_to_probs(surv_probs.T, eval_times)
        np.testing.assert_allclose(converted_t, surv_probs)

    def test_predict_survival_functions_prefers_eval_times(self):
        """Verify explicit eval_times are forwarded when model supports times."""
        from mlsurv.evaluation.metrics import predict_survival_functions

        class DummyModel:
            def __init__(self):
                self.seen_times = None

            def predict_survival_function(self, X, times=None):
                self.seen_times = times
                return np.ones((len(X), len(times)))

        X = pd.DataFrame({'x': [1, 2, 3]})
        eval_times = np.array([1.0, 5.0, 10.0])
        model = DummyModel()

        surv = predict_survival_functions(model, X, eval_times=eval_times)
        assert model.seen_times is not None
        np.testing.assert_allclose(model.seen_times, eval_times)
        assert surv.shape == (3, 3)

    def test_c_index_harrell(self, synthetic_data_clean):
        """Verify Harrell's C-index computation."""
        from mlsurv.evaluation import c_index
        from mlsurv.models import CoxPHModel

        X, y = synthetic_data_clean
        model = CoxPHModel()
        model.fit(X, y)
        risk_scores = model.predict(X)

        c = c_index(y, risk_scores, method='harrell')

        assert 0.0 <= c <= 1.0
        assert c > 0.5  # Better than random

    def test_c_index_ipcw(self, train_test_data):
        """Verify IPCW C-index computation."""
        from mlsurv.evaluation import c_index
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)
        risk_scores = model.predict(X_test)

        c = c_index(y_test, risk_scores, method='ipcw', y_train=y_train)

        assert 0.0 <= c <= 1.0

    def test_c_index_ipcw_requires_y_train(self, synthetic_data_clean):
        """Verify IPCW requires y_train."""
        from mlsurv.evaluation import c_index

        X, y = synthetic_data_clean
        risk_scores = np.random.randn(len(y))

        with pytest.raises(ValueError, match="y_train required"):
            c_index(y, risk_scores, method='ipcw')

    def test_c_index_ipcw_tau_override(self, train_test_data):
        """Verify explicit tau passes through to concordance_index_ipcw."""
        from mlsurv.evaluation import c_index
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)
        risk = model.predict(X_test)

        t_max_train = float(y_train['time'].max())
        tau = t_max_train * 0.5
        c_tau = c_index(y_test, risk, method='ipcw', y_train=y_train, tau=tau)
        c_none = c_index(y_test, risk, method='ipcw', y_train=y_train, tau=None)
        assert 0.0 <= c_tau <= 1.0
        assert 0.0 <= c_none <= 1.0

    def test_compute_safe_ipcw_tau_default(self):
        """Safe tau caps at min(train_t_max_event, p90 test events)."""
        from mlsurv.evaluation.metrics import compute_safe_ipcw_tau

        y_train = make_survival_array(
            [True, True, False, True], [10.0, 20.0, 25.0, 30.0]
        )
        y_test = make_survival_array(
            [True, True, False, True], [5.0, 100.0, 200.0, 50.0]
        )
        tau = compute_safe_ipcw_tau(y_train, y_test)
        # Expected: min(train_t_max_event=30, p90 test events)
        assert tau is not None
        assert tau <= 30.0

    def test_check_ipcw_support_fires_on_violation(self):
        """check_ipcw_support detects follow-up exceeding training support."""
        from mlsurv.evaluation.metrics import check_ipcw_support

        y_train = make_survival_array(
            [True, True, False], [10.0, 20.0, 30.0]
        )
        y_test = make_survival_array(
            [True, True, False], [5.0, 100.0, 200.0]
        )
        flag = check_ipcw_support(y_train, y_test, tau=25.0)
        assert flag is not None
        assert flag['n_beyond'] == 2
        assert flag['t_max_train'] == 30.0

    def test_check_ipcw_support_silent_when_safe(self):
        """No flag when test follow-up is within training support and the
        G_hat cohort has ample events and censored observations."""
        from mlsurv.evaluation.metrics import check_ipcw_support

        # >= 10 events and >= 10 censored so the composition branch stays
        # silent; the point of this test is the n_beyond / g_hat branches.
        # Censored observations sit beyond tau=25 so G_hat stays at ~1.0 over
        # the probed window and the g_hat-floor branch also stays silent.
        events = [True] * 12 + [False] * 12
        times = [float(1 + i) for i in range(12)] + [float(30 + i) for i in range(12)]
        y_train = make_survival_array(events, times)
        y_test = make_survival_array(
            [True, True, False, True], [5.0, 10.0, 15.0, 20.0]
        )
        flag = check_ipcw_support(y_train, y_test, tau=25.0)
        assert flag is None

    def test_check_ipcw_support_fires_on_all_events_cohort(self):
        """All-events G_hat cohort: composition branch fires while the
        G_hat-floor branch stays silent (G_hat sits at ~1.0, not < 0.05)."""
        from mlsurv.evaluation.metrics import check_ipcw_support

        # 12 observations, all events, no censoring -> G_hat never steps down.
        times = [float(t) for t in range(5, 17)]
        y_train = make_survival_array([True] * 12, times)
        # Test entirely within training support so n_beyond == 0.
        y_test = make_survival_array(
            [True, False, True, False], [6.0, 8.0, 10.0, 12.0]
        )
        flag = check_ipcw_support(y_train, y_test, tau=15.0)
        assert flag is not None
        assert flag["n_g_censored"] == 0
        assert flag["n_g_events"] == 12
        assert flag["min_ipcw_support"] == 10
        assert flag["n_beyond"] == 0
        # The old numerical-instability branch did NOT fire: G_hat ~ 1.0.
        assert not (np.isfinite(flag["min_g_hat"]) and flag["min_g_hat"] < 0.05)

    def test_check_ipcw_support_fires_on_thin_censoring(self):
        """Plenty of events but only 6 censored -> composition branch fires."""
        from mlsurv.evaluation.metrics import check_ipcw_support

        events = [True] * 30 + [False] * 6
        times = [float(5 + i) for i in range(36)]
        y_train = make_survival_array(events, times)
        y_test = make_survival_array(
            [True, False, True, False], [6.0, 8.0, 10.0, 12.0]
        )
        flag = check_ipcw_support(y_train, y_test, tau=20.0)
        assert flag is not None
        assert flag["n_g_censored"] == 6
        assert flag["n_g_events"] == 30

    def test_check_ipcw_support_fires_on_thin_events(self):
        """Symmetric end: only 6 events -> composition branch fires too."""
        from mlsurv.evaluation.metrics import check_ipcw_support

        events = [True] * 6 + [False] * 30
        times = [float(5 + i) for i in range(36)]
        y_train = make_survival_array(events, times)
        y_test = make_survival_array(
            [True, False, True, False], [6.0, 8.0, 10.0, 12.0]
        )
        flag = check_ipcw_support(y_train, y_test, tau=20.0)
        assert flag is not None
        assert flag["n_g_events"] == 6
        assert flag["n_g_censored"] == 30

    def test_check_ipcw_support_silent_on_healthy_cohort(self):
        """>= floor events and censored, test in range -> no flag."""
        from mlsurv.evaluation.metrics import check_ipcw_support

        events = [True] * 15 + [False] * 15
        times = [float(2 + i) for i in range(30)]
        y_train = make_survival_array(events, times)
        y_test = make_survival_array(
            [True, False, True, False], [3.0, 5.0, 8.0, 10.0]
        )
        flag = check_ipcw_support(y_train, y_test, tau=20.0)
        assert flag is None

    def test_check_ipcw_support_composition_boundary(self):
        """Boundary: exactly 10 of each is silent; 9 of each fires."""
        from mlsurv.evaluation.metrics import check_ipcw_support

        y_test = make_survival_array(
            [True, False, True, False], [3.0, 5.0, 8.0, 10.0]
        )

        # Exactly 10 events + 10 censored -> at the floor, silent.
        events_ok = [True] * 10 + [False] * 10
        times_ok = [float(2 + i) for i in range(20)]
        y_ok = make_survival_array(events_ok, times_ok)
        assert check_ipcw_support(y_ok, y_test, tau=15.0) is None

        # 9 events + 9 censored -> below the floor, fires.
        events_thin = [True] * 9 + [False] * 9
        times_thin = [float(2 + i) for i in range(18)]
        y_thin = make_survival_array(events_thin, times_thin)
        flag = check_ipcw_support(y_thin, y_test, tau=15.0)
        assert flag is not None
        assert flag["n_g_events"] == 9
        assert flag["n_g_censored"] == 9

    def test_auc_t(self, train_test_data):
        """Verify time-dependent AUC computation."""
        from mlsurv.evaluation import auc_t
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)
        risk_scores = model.predict(X_test)

        eval_times = np.quantile(y_test['time'], [0.25, 0.5, 0.75])
        auc_values, mean_auc = auc_t(y_train, y_test, risk_scores, eval_times)

        assert len(auc_values) == len(eval_times)
        assert all(0.0 <= v <= 1.0 for v in auc_values)


class TestSurvFuncsToProbs:
    """Tests for _surv_funcs_to_probs type-dispatch helpers."""

    # --- DataFrame path (_df_to_probs) ---

    def test_df_exact_times(self):
        """DataFrame with exact matching eval_times."""
        from mlsurv.evaluation.metrics import _surv_funcs_to_probs

        df = pd.DataFrame(
            {0: [1.0, 0.8, 0.6], 1: [0.9, 0.7, 0.5]},
            index=[1.0, 2.0, 3.0],
        )
        result = _surv_funcs_to_probs(df, [1.0, 2.0, 3.0])
        expected = np.array([[1.0, 0.8, 0.6], [0.9, 0.7, 0.5]])
        np.testing.assert_allclose(result, expected)
        assert result.shape == (2, 3)

    def test_df_step_interpolation(self):
        """DataFrame with eval_times between index values uses step interpolation."""
        from mlsurv.evaluation.metrics import _surv_funcs_to_probs

        df = pd.DataFrame(
            {0: [1.0, 0.8, 0.4]},
            index=[1.0, 3.0, 5.0],
        )
        # t=2.0 falls between 1.0 and 3.0 → use last known value at 1.0
        result = _surv_funcs_to_probs(df, [2.0])
        np.testing.assert_allclose(result, np.array([[1.0]]))

    def test_df_before_first_time(self):
        """DataFrame with eval_time before first index uses first value."""
        from mlsurv.evaluation.metrics import _surv_funcs_to_probs

        df = pd.DataFrame({0: [0.9, 0.5]}, index=[2.0, 4.0])
        result = _surv_funcs_to_probs(df, [0.5])
        np.testing.assert_allclose(result, np.array([[0.9]]))

    def test_df_after_last_time(self):
        """DataFrame with eval_time after last index uses last value."""
        from mlsurv.evaluation.metrics import _surv_funcs_to_probs

        df = pd.DataFrame({0: [0.9, 0.5]}, index=[2.0, 4.0])
        result = _surv_funcs_to_probs(df, [10.0])
        np.testing.assert_allclose(result, np.array([[0.5]]))

    def test_df_multi_sample_interpolation(self):
        """DataFrame with multiple samples and mixed interpolation scenarios."""
        from mlsurv.evaluation.metrics import _surv_funcs_to_probs

        df = pd.DataFrame(
            {0: [1.0, 0.8, 0.4], 1: [0.9, 0.7, 0.3]},
            index=[1.0, 3.0, 5.0],
        )
        # t=0.5 (before first), t=3.0 (exact), t=4.0 (between 3 and 5)
        result = _surv_funcs_to_probs(df, [0.5, 3.0, 4.0])
        expected = np.array([
            [1.0, 0.8, 0.8],   # sample 0: first val, exact, step from 3.0
            [0.9, 0.7, 0.7],   # sample 1: first val, exact, step from 3.0
        ])
        np.testing.assert_allclose(result, expected)

    # --- ndarray path (_array_to_probs) ---

    def test_array_1d_single_sample(self):
        """1D array matching n_times → single sample."""
        from mlsurv.evaluation.metrics import _surv_funcs_to_probs

        arr = np.array([0.9, 0.8, 0.7])
        result = _surv_funcs_to_probs(arr, [1.0, 2.0, 3.0])
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result, arr.reshape(1, 3))

    def test_array_1d_single_time(self):
        """1D array with single eval_time → (n_samples, 1)."""
        from mlsurv.evaluation.metrics import _surv_funcs_to_probs

        arr = np.array([0.9, 0.8, 0.7])
        result = _surv_funcs_to_probs(arr, [5.0])
        assert result.shape == (3, 1)

    def test_array_1d_shape_mismatch(self):
        """1D array with incompatible length raises ValueError."""
        from mlsurv.evaluation.metrics import _surv_funcs_to_probs

        arr = np.array([0.9, 0.8, 0.7, 0.6])
        with pytest.raises(ValueError, match="1D survival array"):
            _surv_funcs_to_probs(arr, [1.0, 2.0])

    def test_array_2d_shape_mismatch(self):
        """2D array with incompatible dimensions raises ValueError."""
        from mlsurv.evaluation.metrics import _surv_funcs_to_probs

        arr = np.ones((4, 5))
        with pytest.raises(ValueError, match="incompatible"):
            _surv_funcs_to_probs(arr, [1.0, 2.0, 3.0])

    def test_array_2d_square_assumes_samples_by_times(self):
        """Square 2D array: shape (n, n) is treated as (n_samples, n_times)."""
        from mlsurv.evaluation.metrics import _surv_funcs_to_probs

        # Asymmetric values so transposition would be detectable
        arr = np.array([[0.9, 0.8, 0.7],
                        [0.6, 0.5, 0.4],
                        [0.3, 0.2, 0.1]])
        result = _surv_funcs_to_probs(arr, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, arr)  # returned as-is, not transposed

    def test_array_3d_raises(self):
        """3D array raises ValueError."""
        from mlsurv.evaluation.metrics import _surv_funcs_to_probs

        arr = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="1D or 2D"):
            _surv_funcs_to_probs(arr, [1.0, 2.0, 3.0, 4.0])

    # --- Object array of callables ---

    def test_object_array_of_callables(self):
        """Object ndarray containing callable objects."""
        from mlsurv.evaluation.metrics import _surv_funcs_to_probs

        fn1 = lambda t: 1.0 / (1.0 + t)
        fn2 = lambda t: 0.5 / (1.0 + t)
        arr = np.array([fn1, fn2], dtype=object)

        result = _surv_funcs_to_probs(arr, [0.0, 1.0])
        expected = np.array([[1.0, 0.5], [0.5, 0.25]])
        np.testing.assert_allclose(result, expected)

    # --- list/tuple of numeric values ---

    def test_list_of_lists_numeric(self):
        """Nested list of numbers converts via ndarray path."""
        from mlsurv.evaluation.metrics import _surv_funcs_to_probs

        data = [[0.9, 0.8, 0.7], [0.85, 0.75, 0.65]]
        result = _surv_funcs_to_probs(data, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, np.array(data))

    def test_tuple_of_lists_numeric(self):
        """Tuple of lists of numbers converts via ndarray path."""
        from mlsurv.evaluation.metrics import _surv_funcs_to_probs

        data = ([0.9, 0.8], [0.7, 0.6])
        result = _surv_funcs_to_probs(data, [1.0, 2.0])
        np.testing.assert_allclose(result, np.array(data))

    # --- list of callables (StepFunction-like) ---

    def test_list_of_callables(self):
        """Plain list of callable objects dispatches to the callable path."""
        from mlsurv.evaluation.metrics import _surv_funcs_to_probs

        fn1 = lambda t: np.exp(-0.1 * t)
        fn2 = lambda t: np.exp(-0.2 * t)
        result = _surv_funcs_to_probs([fn1, fn2], [0.0, 1.0])
        expected = np.array([
            [np.exp(0.0), np.exp(-0.1)],
            [np.exp(0.0), np.exp(-0.2)],
        ])
        np.testing.assert_allclose(result, expected)


class TestEvaluate:
    """Test evaluate() function.

    Iteration counts are intentionally low for test speed.
    """

    def test_evaluate_basic(self, train_test_data):
        """Verify basic evaluation returns correct structure."""
        from mlsurv.evaluation import evaluate
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            X_train=X_train, y_train=y_train,
            n_importance_repeats=0
        )

        # Check result attributes
        assert hasattr(result, 'concordance')
        assert hasattr(result, 'test_set')
        assert hasattr(result, 'auc_t')
        assert hasattr(result, 'importance')
        assert hasattr(result, 'importance_iterations')

        # Check model_type column
        assert 'model_type' in result.concordance.columns
        assert 'model_type' in result.test_set.columns

    def test_evaluate_concordance_df(self, train_test_data):
        """Verify concordance DataFrame structure."""
        from mlsurv.evaluation import evaluate
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            X_train=X_train, y_train=y_train,
            n_importance_repeats=0
        )

        assert 'c_index_test' in result.concordance.columns
        assert 0.0 <= result.concordance['c_index_test'].iloc[0] <= 1.0

    def test_evaluate_test_set_df(self, train_test_data):
        """Verify test_set DataFrame structure."""
        from mlsurv.evaluation import evaluate
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            X_train=X_train, y_train=y_train,
            n_importance_repeats=0
        )

        assert 'risk_score' in result.test_set.columns
        assert 'time' in result.test_set.columns
        assert 'event' in result.test_set.columns
        assert len(result.test_set) == len(X_test)

    def test_evaluate_importance_df(self, train_test_data):
        """Verify importance DataFrame structure."""
        from mlsurv.evaluation import evaluate
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            X_train=X_train, y_train=y_train,
            n_importance_repeats=5
        )

        assert 'feature' in result.importance.columns
        assert 'perm_mean' in result.importance.columns
        assert 'perm_std' in result.importance.columns
        assert len(result.importance) == X_train.shape[1]


class TestBootstrap:
    """Test bootstrap() function.

    Iteration counts are intentionally low for test speed.
    """

    def test_bootstrap_basic(self, train_test_data):
        """Verify basic bootstrap returns correct structure."""
        from mlsurv.evaluation import bootstrap
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = bootstrap(
            model, X_test, y_test,
            y_train=y_train,
            n_bootstrap=2
        )

        # Check result attributes
        assert hasattr(result, 'summary')
        assert hasattr(result, 'iterations')

        # Check model_type column
        assert 'model_type' in result.summary.columns
        assert 'model_type' in result.iterations.columns

    def test_bootstrap_summary_structure(self, train_test_data):
        """Verify summary DataFrame structure."""
        from mlsurv.evaluation import bootstrap
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = bootstrap(
            model, X_test, y_test,
            y_train=y_train,
            n_bootstrap=2
        )

        assert 'metric' in result.summary.columns
        assert 'point' in result.summary.columns
        assert 'mean' in result.summary.columns
        assert 'std' in result.summary.columns
        assert 'ci_lower' in result.summary.columns
        assert 'ci_upper' in result.summary.columns

        # Verify column order
        expected_cols = ['model_type', 'metric', 'point', 'mean', 'std', 'ci_lower', 'ci_upper']
        assert list(result.summary.columns) == expected_cols

        # Point estimates should be non-NaN for all default metrics
        for metric_name in ['harrell', 'ipcw', 'integrated_brier']:
            mask = result.summary['metric'] == metric_name
            if mask.any():
                point_val = result.summary.loc[mask, 'point'].iloc[0]
                assert pd.notna(point_val), f"point estimate is NaN for {metric_name}"

    def test_bootstrap_stores_ci_level(self, train_test_data):
        """bootstrap(ci=...) stores the level on the result and summary attrs."""
        from mlsurv.evaluation import bootstrap
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = bootstrap(
            model, X_test, y_test,
            y_train=y_train,
            n_bootstrap=2,
            ci=0.80,
        )

        assert result.ci_level == 0.80
        assert result.summary.attrs['ci_level'] == 0.80

    def test_bootstrap_multiple_metrics(self, train_test_data):
        """Verify bootstrap with multiple metrics."""
        from mlsurv.evaluation import bootstrap
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = bootstrap(
            model, X_test, y_test,
            y_train=y_train,
            metrics=['harrell', 'ipcw'],
            n_bootstrap=2
        )

        metrics = result.summary['metric'].unique()
        assert 'harrell' in metrics
        assert 'ipcw' in metrics

    def test_bootstrap_point_matches_evaluate(self, train_test_data):
        """Verify bootstrap point estimates match evaluate() concordance values."""
        from mlsurv.evaluation import bootstrap, evaluate
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        eval_result = evaluate(model, X_test, y_test, y_train=y_train)
        boot_result = bootstrap(
            model, X_test, y_test,
            y_train=y_train,
            metrics=['harrell'],
            n_bootstrap=2,
        )

        eval_harrell = eval_result.concordance['c_index_test'].iloc[0]
        boot_harrell = boot_result.summary.loc[
            boot_result.summary['metric'] == 'harrell', 'point'
        ].iloc[0]

        assert abs(eval_harrell - boot_harrell) < 1e-8

    def test_bootstrap_get_distribution(self, train_test_data):
        """Verify get_distribution method."""
        from mlsurv.evaluation import bootstrap
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = bootstrap(
            model, X_test, y_test,
            y_train=y_train,
            n_bootstrap=2
        )

        dist = result.get_distribution('harrell')
        assert len(dist) == 2

    def test_bootstrap_include_predictions(self, train_test_data):
        """Verify bootstrap with predictions included."""
        from mlsurv.evaluation import bootstrap
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = bootstrap(
            model, X_test, y_test,
            y_train=y_train,
            n_bootstrap=2,
            include_predictions=True
        )

        assert result.predictions is not None
        assert 'iteration' in result.predictions.columns
        assert 'risk_score' in result.predictions.columns

    def test_bootstrap_zero_iterations(self, train_test_data):
        """n_bootstrap=0 should return NaN summary without crashing."""
        from mlsurv.evaluation import bootstrap
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = bootstrap(
            model, X_test, y_test,
            y_train=y_train,
            n_bootstrap=0,
        )

        # Summary should exist with expected columns, all NaN stats
        assert result.summary is not None
        assert not result.summary.empty
        for col in ('mean', 'std', 'ci_lower', 'ci_upper'):
            assert result.summary[col].isna().all(), f"{col} should be all NaN"

        # Iterations DataFrame should be empty
        assert result.iterations is not None
        assert len(result.iterations) == 0

        # Printing should not crash
        result.print_summary()


class TestGetCoefficientsWithSelector:
    """Test _get_coefficients alignment when TopKCoxSelector is in the pipeline."""

    @staticmethod
    def _make_selector(features):
        """Create a mock selector with the given selected_features_."""
        class _Selector:
            selected_features_ = features
        return _Selector()

    @staticmethod
    def _make_cox_model(feature_names, coefs):
        """Create a mock CoxPH-like model with get_coefficients()."""
        names = np.array(feature_names)
        class _CoxModel:
            feature_names_in_ = names
            def get_coefficients(self):
                return pd.Series(coefs, index=self.feature_names_in_)
        return _CoxModel()

    def test_coefficients_aligned_to_full_features(self):
        """Pipeline + selector + CoxPH-like model returns coefficients for all features."""
        from sklearn.pipeline import Pipeline
        from mlsurv.evaluation.importance import _get_coefficients

        selected = ['feat_0', 'feat_2', 'feat_4']
        selector = self._make_selector(selected)
        model = self._make_cox_model(selected, [0.5, -0.3, 0.2])
        pipeline = Pipeline([('selector', selector), ('model', model)])

        result = _get_coefficients(pipeline, ['feat_0', 'feat_1', 'feat_2', 'feat_3', 'feat_4'])

        assert len(result) == 5
        assert result[0] == 0.5    # feat_0 selected
        assert np.isnan(result[1]) # feat_1 not selected
        assert result[2] == -0.3   # feat_2 selected
        assert np.isnan(result[3]) # feat_3 not selected
        assert result[4] == 0.2    # feat_4 selected

    def test_no_selector_passthrough(self):
        """Without selector, coefficients returned as-is."""
        from sklearn.pipeline import Pipeline
        from mlsurv.evaluation.importance import _get_coefficients

        all_feats = ['feat_0', 'feat_1', 'feat_2']
        model = self._make_cox_model(all_feats, [0.5, -0.3, 0.2])
        pipeline = Pipeline([('model', model)])

        result = _get_coefficients(pipeline, all_feats)

        assert len(result) == 3
        assert result == [0.5, -0.3, 0.2]

    def test_build_importance_summary_no_crash(self):
        """_build_importance_summary succeeds with aligned coefficients."""
        from sklearn.pipeline import Pipeline
        from mlsurv.evaluation.importance import _get_coefficients, _build_importance_summary

        class MockResult:
            importances_mean = np.array([0.1, 0.0, 0.2, 0.0, 0.15])
            importances_std = np.array([0.01, 0.0, 0.02, 0.0, 0.015])

        selected = ['feat_0', 'feat_2', 'feat_4']
        selector = self._make_selector(selected)
        model = self._make_cox_model(selected, [0.5, -0.3, 0.2])
        pipeline = Pipeline([('selector', selector), ('model', model)])
        features = ['feat_0', 'feat_1', 'feat_2', 'feat_3', 'feat_4']

        coefficients = _get_coefficients(pipeline, features)
        summary_df = _build_importance_summary(MockResult(), features, coefficients)

        assert len(summary_df) == 5
        assert set(summary_df.columns) == {'feature', 'coefficient', 'perm_mean', 'perm_std'}

    def test_coef_fallback_path_with_selector(self):
        """coef_ attribute path also aligns when selector is present."""
        from sklearn.pipeline import Pipeline
        from mlsurv.evaluation.importance import _get_coefficients

        class MockCoefModel:
            coef_ = np.array([0.5, -0.3])

        selector = self._make_selector(['feat_0', 'feat_2'])
        pipeline = Pipeline([('selector', selector), ('model', MockCoefModel())])

        result = _get_coefficients(pipeline, ['feat_0', 'feat_1', 'feat_2'])

        assert len(result) == 3
        assert result[0] == 0.5
        assert np.isnan(result[1])
        assert result[2] == -0.3

    def test_integer_index_selector(self):
        """Selector with integer indices (from scaler stripping names) maps correctly."""
        from sklearn.pipeline import Pipeline
        from mlsurv.evaluation.importance import _get_coefficients

        # Selector saw integer columns (after scaler stripped names)
        selector = self._make_selector([1, 3])
        # Inner model has integer feature_names_in_ matching selector output
        model = self._make_cox_model([1, 3], [0.5, -0.3])
        pipeline = Pipeline([('selector', selector), ('model', model)])

        # Full features use string names (original DataFrame columns)
        result = _get_coefficients(pipeline, ['a', 'b', 'c', 'd'])

        assert len(result) == 4
        assert np.isnan(result[0])  # 'a' not selected
        assert result[1] == 0.5     # 'b' = index 1
        assert np.isnan(result[2])  # 'c' not selected
        assert result[3] == -0.3    # 'd' = index 3

    def test_tree_model_nan_unaffected(self):
        """Tree models (no coefficients) still return NaN list regardless of selector."""
        from sklearn.pipeline import Pipeline
        from mlsurv.evaluation.importance import _get_coefficients

        class MockTreeModel:
            pass

        selector = self._make_selector(['feat_0', 'feat_2'])
        pipeline = Pipeline([('selector', selector), ('model', MockTreeModel())])

        result = _get_coefficients(pipeline, ['feat_0', 'feat_1', 'feat_2'])

        assert len(result) == 3
        assert all(np.isnan(v) for v in result)


class TestGetCoefficientsErrorHandling:
    """Test error handling and logging in _get_coefficients and _get_selected_features."""

    def test_get_coefficients_method_raises_returns_nan(self):
        """get_coefficients() raising returns NaN list."""
        from sklearn.pipeline import Pipeline
        from mlsurv.evaluation.importance import _get_coefficients

        class FailingModel:
            def get_coefficients(self):
                raise ValueError("model not fitted")

        pipeline = Pipeline([('model', FailingModel())])
        result = _get_coefficients(pipeline, ['a', 'b', 'c'])

        assert len(result) == 3
        assert all(np.isnan(v) for v in result)

    def test_get_coefficients_method_raises_logs_warning(self, caplog):
        """get_coefficients() failure logs WARNING with model type name."""
        import logging
        from sklearn.pipeline import Pipeline
        from mlsurv.evaluation.importance import _get_coefficients

        class FailingModel:
            def get_coefficients(self):
                raise ValueError("model not fitted")

        pipeline = Pipeline([('model', FailingModel())])
        with caplog.at_level(logging.WARNING, logger="mlsurv.importance"):
            _get_coefficients(pipeline, ['a', 'b'])

        assert any("FailingModel" in r.message for r in caplog.records)
        assert any("model not fitted" in r.message for r in caplog.records)

    def test_get_coefficients_method_catches_broad_exceptions(self, caplog):
        """get_coefficients() path catches TypeError (beyond ValueError/AttributeError)."""
        import logging
        from sklearn.pipeline import Pipeline
        from mlsurv.evaluation.importance import _get_coefficients

        class TypeErrorModel:
            def get_coefficients(self):
                raise TypeError("unexpected type")

        pipeline = Pipeline([('model', TypeErrorModel())])
        with caplog.at_level(logging.WARNING, logger="mlsurv.importance"):
            result = _get_coefficients(pipeline, ['a', 'b'])

        assert len(result) == 2
        assert all(np.isnan(v) for v in result)
        assert any("TypeErrorModel" in r.message for r in caplog.records)

    def test_coef_fallback_exception_returns_nan(self):
        """coef_ fallback path exception returns NaN list."""
        from sklearn.pipeline import Pipeline
        from mlsurv.evaluation.importance import _get_coefficients

        class BadCoefModel:
            coef_ = [0.1, 0.2]  # list has no .ndim → AttributeError

        pipeline = Pipeline([('model', BadCoefModel())])
        result = _get_coefficients(pipeline, ['a', 'b', 'c'])

        assert len(result) == 3
        assert all(np.isnan(v) for v in result)

    def test_coef_fallback_exception_logs_warning(self, caplog):
        """coef_ fallback path exception logs WARNING with model type."""
        import logging
        from sklearn.pipeline import Pipeline
        from mlsurv.evaluation.importance import _get_coefficients

        class BadCoefModel:
            coef_ = [0.1, 0.2]  # list has no .ndim → AttributeError

        pipeline = Pipeline([('model', BadCoefModel())])
        with caplog.at_level(logging.WARNING, logger="mlsurv.importance"):
            _get_coefficients(pipeline, ['a', 'b'])

        assert any("BadCoefModel" in r.message for r in caplog.records)
        assert any("ndim" in r.message for r in caplog.records)

    def test_selected_features_missing_returns_none(self):
        """Selector step without selected_features_ returns None."""
        from sklearn.pipeline import Pipeline
        from mlsurv.evaluation.importance import _get_selected_features

        class BareSelectorStep:
            pass

        pipeline = Pipeline([('selector', BareSelectorStep()), ('model', object())])
        result = _get_selected_features(pipeline, ['a', 'b'])

        assert result is None

    def test_selected_features_missing_logs_debug(self, caplog):
        """Selector step without selected_features_ logs debug message."""
        import logging
        from sklearn.pipeline import Pipeline
        from mlsurv.evaluation.importance import _get_selected_features

        class BareSelectorStep:
            pass

        pipeline = Pipeline([('selector', BareSelectorStep()), ('model', object())])
        with caplog.at_level(logging.DEBUG, logger="mlsurv.importance"):
            _get_selected_features(pipeline, ['a', 'b'])

        assert any("BareSelectorStep" in r.message for r in caplog.records)
        assert any("selected_features_" in r.message for r in caplog.records)


class TestEvalTimesClipping:
    """Test that evaluate() clips eval_times when validation follow-up is longer."""

    def test_evaluate_with_longer_validation_followup(self):
        """Evaluate succeeds when test times exceed training range."""
        import warnings
        from mlsurv.evaluation import evaluate
        from mlsurv.models import CoxPHModel

        np.random.seed(99)
        n_train, n_test, n_feat = 100, 40, 5

        X_train = pd.DataFrame(
            np.random.randn(n_train, n_feat),
            columns=[f'f{i}' for i in range(n_feat)],
        )
        # Training times in [0.1, 5]
        y_train = make_survival_array(
            np.random.binomial(1, 0.7, n_train),
            np.random.uniform(0.1, 5.0, n_train),
        )

        X_test = pd.DataFrame(
            np.random.randn(n_test, n_feat),
            columns=[f'f{i}' for i in range(n_feat)],
        )
        # Test times in [0.1, 20] — extends well beyond training range
        y_test = make_survival_array(
            np.random.binomial(1, 0.7, n_test),
            np.random.uniform(0.1, 20.0, n_test),
        )

        model = CoxPHModel()
        model.fit(X_train, y_train)

        # Should succeed with a clipping warning, not raise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = evaluate(
                model, X_test, y_test,
                X_train=X_train, y_train=y_train,
                n_importance_repeats=0,
            )

        # Brier / IBS should be present and not NaN in concordance DataFrame
        conc = result.concordance
        assert 'brier_score_test' in conc.columns
        assert 'integrated_brier_test' in conc.columns
        assert not conc['brier_score_test'].isna().all()
        assert not conc['integrated_brier_test'].isna().all()

        # AUC-t should also be non-NaN (tests _compute_auc_t truncation path)
        assert 'auc' in result.auc_t.columns
        assert not result.auc_t['auc'].isna().all()


class TestEvaluateMultiModel:
    """Test evaluate() with multiple models (dict input)."""

    @pytest.mark.timeout(120)
    def test_evaluate_multi_model_basic(self, train_test_data):
        """Verify evaluate with dict aggregates results."""
        from mlsurv.evaluation import evaluate
        from mlsurv.models import CoxPHModel, RSFModel

        X_train, X_test, y_train, y_test = train_test_data

        # Train models
        models = {}
        coxph = CoxPHModel()
        coxph.fit(X_train, y_train)
        models['coxph'] = coxph

        rsf = RSFModel(n_estimators=10, random_state=42)
        rsf.fit(X_train, y_train)
        models['rsf'] = rsf

        result = evaluate(
            models, X_test, y_test,
            X_train=X_train, y_train=y_train,
            n_importance_repeats=0
        )

        # Check aggregated results
        assert len(result.concordance) == 2
        model_types = result.concordance['model_type'].unique()
        assert 'coxph' in model_types or 'CoxPHModel' in model_types


@pytest.mark.bootstrap_heavy
class TestBootstrapMultiModel:
    """Test bootstrap() with multiple models."""

    def test_bootstrap_multi_model_basic(self, train_test_data):
        """Verify bootstrap aggregates results for multiple models."""
        from mlsurv.evaluation import bootstrap
        from mlsurv.models import CoxPHModel, RSFModel

        X_train, X_test, y_train, y_test = train_test_data

        # Train models
        models = {}
        coxph = CoxPHModel()
        coxph.fit(X_train, y_train)
        models['coxph'] = coxph

        rsf = RSFModel(n_estimators=10, random_state=42)
        rsf.fit(X_train, y_train)
        models['rsf'] = rsf

        result = bootstrap(
            models, X_test, y_test,
            y_train=y_train,
            n_bootstrap=10,
            show_progress=False
        )

        # Check aggregated results: 2 models × 4 metrics = 8 rows
        # (harrell, ipcw, integrated_brier, ctd)
        assert len(result.summary) == 8
        model_types = result.summary['model_type'].unique()
        assert len(model_types) == 2
        # Verify all metrics present
        metrics = result.summary['metric'].unique()
        assert len(metrics) == 4

    def test_bootstrap_with_dict_interface(self, train_test_data):
        """Verify bootstrap() accepts dict of models (new unified interface)."""
        from mlsurv.evaluation import bootstrap
        from mlsurv.models import CoxPHModel, RSFModel

        X_train, X_test, y_train, y_test = train_test_data

        # Train models
        models = {}
        coxph = CoxPHModel()
        coxph.fit(X_train, y_train)
        models['coxph'] = coxph

        rsf = RSFModel(n_estimators=10, random_state=42)
        rsf.fit(X_train, y_train)
        models['rsf'] = rsf

        # Use bootstrap() with dict (new interface)
        result = bootstrap(
            models, X_test, y_test,
            y_train=y_train,
            n_bootstrap=10,
            show_progress=False
        )

        # Check aggregated results: 2 models × 4 metrics = 8 rows
        assert len(result.summary) == 8
        model_types = result.summary['model_type'].unique()
        assert len(model_types) == 2
        assert 'coxph' in model_types
        assert 'rsf' in model_types


@pytest.mark.bootstrap_heavy
class TestBootstrapResultMethods:
    """Test BootstrapResult convenience methods."""

    def test_compute_ci_percentile(self, train_test_data):
        """Verify percentile CI computation."""
        from mlsurv.evaluation import bootstrap
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = bootstrap(
            model, X_test, y_test,
            y_train=y_train,
            n_bootstrap=20
        )

        ci_lower, ci_upper = result.compute_ci('harrell', ci=0.95, method='percentile')

        assert ci_lower < ci_upper
        assert 0.0 <= ci_lower <= 1.0
        assert 0.0 <= ci_upper <= 1.0

    def test_compute_ci_basic(self, train_test_data):
        """Verify basic CI computation."""
        from mlsurv.evaluation import bootstrap
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = bootstrap(
            model, X_test, y_test,
            y_train=y_train,
            n_bootstrap=20
        )

        ci_lower, ci_upper = result.compute_ci('harrell', ci=0.95, method='basic')

        assert ci_lower < ci_upper

    def test_compute_ci_unknown_method(self, train_test_data):
        """Verify error on unknown CI method."""
        from mlsurv.evaluation import bootstrap
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = bootstrap(
            model, X_test, y_test,
            y_train=y_train,
            n_bootstrap=10
        )

        with pytest.raises(ValueError, match="Unknown method"):
            result.compute_ci('harrell', method='unknown')

    def test_get_distribution_multi_model_error(self, train_test_data):
        """Verify error when model_type not specified with multiple models."""
        from mlsurv.evaluation import bootstrap
        from mlsurv.models import CoxPHModel, RSFModel

        X_train, X_test, y_train, y_test = train_test_data

        models = {}
        coxph = CoxPHModel()
        coxph.fit(X_train, y_train)
        models['coxph'] = coxph

        rsf = RSFModel(n_estimators=10, random_state=42)
        rsf.fit(X_train, y_train)
        models['rsf'] = rsf

        result = bootstrap(
            models, X_test, y_test,
            y_train=y_train,
            n_bootstrap=10,
            show_progress=False
        )

        # Should error without model_type specified
        with pytest.raises(ValueError, match="Multiple model_types"):
            result.get_distribution('harrell')

        # Should work with model_type specified
        dist = result.get_distribution('harrell', model_type='coxph')
        assert len(dist) == 10


class TestModelResultMethods:
    """Test ModelResult dataclass methods."""

    def test_to_dict(self, synthetic_data_clean):
        """Verify to_dict returns correct structure."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup()

        model_result = learner.train('coxph')

        result_dict = model_result.to_dict()

        assert 'name' in result_dict
        assert 'cv_metric_mean' in result_dict
        assert 'cv_metric_std' in result_dict
        assert result_dict['name'] == 'coxph'
        assert isinstance(result_dict['cv_metric_mean'], float)

    def test_properties_before_evaluation(self, synthetic_data_clean):
        """Verify properties return None before evaluate()."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup()

        model_result = learner.train('coxph')

        # Before evaluation, evaluation should be None
        assert model_result.evaluation is None


class TestRunResultAndLearnerProperties:
    """Test run() result and learner properties (dict-of-ModelResult API)."""

    def test_metrics_attribute(self, small_data):
        """Verify learner.summary() returns metrics DataFrame after run()."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        result = learner.run(models=['coxph'], verbose=False)

        # run() returns dict; metrics accessed via learner.summary()
        assert isinstance(result, dict)
        metrics = learner.summary()
        assert isinstance(metrics, pd.DataFrame)
        assert 'model' in metrics.columns
        assert 'mean' in metrics.columns

    def test_best_model_property(self, small_data):
        """Verify learner.best_model returns correct model after run()."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        result = learner.run(models=['coxph'], verbose=False)

        # run() returns dict; best_model accessed via learner property
        assert isinstance(result, dict)
        best = learner.best_model
        assert best is not None
        assert best.name == 'coxph'


class TestBrierScore:
    """Test Brier score computation."""

    def test_brier_score_at_times(self, train_test_data):
        """Verify Brier score at specific times."""
        from mlsurv.evaluation.metrics import _brier_score_at_times
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        # Get survival functions
        surv_funcs = model.predict_survival_function(X_test)

        eval_times = np.quantile(y_test['time'], [0.25, 0.5, 0.75])
        brier_scores = _brier_score_at_times(
            y_train, y_test, surv_funcs, eval_times
        )

        assert len(brier_scores) == len(eval_times)
        # Brier scores should be between 0 and 1
        assert all(0.0 <= b <= 1.0 for b in brier_scores)

    def test_integrated_brier(self, train_test_data):
        """Verify integrated Brier score computation approach."""
        from mlsurv.evaluation.metrics import _brier_score_at_times
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        # Get survival functions
        surv_funcs = model.predict_survival_function(X_test)

        # Use multiple timepoints and average Brier scores (approximates IBS)
        eval_times = np.quantile(y_test['time'], [0.2, 0.4, 0.6, 0.8])
        brier_scores = _brier_score_at_times(
            y_train, y_test, surv_funcs, eval_times
        )

        # Approximate IBS as mean of Brier scores
        approx_ibs = np.mean(brier_scores)

        assert isinstance(approx_ibs, float)
        assert 0.0 <= approx_ibs <= 1.0


def _has_shap():
    """Check if shap is available."""
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.shap
class TestSHAPIntegration:
    """Test SHAP value computation (requires shap package)."""

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_evaluate_with_shap(self, train_test_data):
        """Verify evaluate() computes SHAP values when requested."""
        from mlsurv import SurvivalLearner

        X_train, X_test, y_train, y_test = train_test_data
        X = pd.concat([X_train, X_test])
        y = np.concatenate([y_train, y_test])

        learner = SurvivalLearner(X, y)
        learner.setup()
        model = learner.train('coxph')

        result = learner.evaluate(model, include_shap=True, n_importance_repeats=0)

        assert result.shap_values is not None
        assert 'feature' in result.shap_values.columns
        assert 'shap_value' in result.shap_values.columns

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_shap_values_structure(self, train_test_data):
        """Verify SHAP DataFrame structure."""
        from mlsurv.evaluation import evaluate
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            X_train=X_train, y_train=y_train,
            include_shap=True, n_importance_repeats=0
        )

        assert result.shap_values is not None
        assert 'model_type' in result.shap_values.columns
        assert 'sample_idx' in result.shap_values.columns

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_shap_with_rsf(self, train_test_data):
        """Verify SHAP works with tree-based models (RSF)."""
        from mlsurv.models import RSFModel
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data
        model = RSFModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            X_train=X_train, y_train=y_train,
            include_shap=True, n_importance_repeats=0
        )

        assert result.shap_values is not None


@pytest.mark.shap
class TestSHAPModelCoverage:
    """Test SHAP works for all built-in models."""

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    @pytest.mark.parametrize("model_name", [
        'coxph', 'coxnet', 'rsf', 'gbsa', 'weibull_aft'
    ])
    def test_shap_all_base_models(self, train_test_data, model_name):
        """Verify SHAP works with all base models."""
        from mlsurv.models import (
            CoxPHModel, CoxNetModel, RSFModel, GBSAModel, WeibullAFTModel
        )
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        # Create and fit model
        model_classes = {
            'coxph': CoxPHModel,
            'coxnet': lambda: CoxNetModel(fit_baseline_model=True),
            'rsf': lambda: RSFModel(n_estimators=10, random_state=42),
            'gbsa': lambda: GBSAModel(n_estimators=10, random_state=42),
            'weibull_aft': WeibullAFTModel,
        }

        model_cls = model_classes[model_name]
        if callable(model_cls) and not isinstance(model_cls, type):
            model = model_cls()
        else:
            model = model_cls()

        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            X_train=X_train, y_train=y_train,
            include_shap=True, n_importance_repeats=0,
            show_progress=False
        )

        assert result.shap_values is not None
        assert len(result.shap_values) > 0
        assert 'feature' in result.shap_values.columns
        assert 'shap_value' in result.shap_values.columns

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_shap_fssvm(self, train_test_data):
        """Verify SHAP works with FSSVM."""
        from mlsurv.models import FSSVMModel
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        model = FSSVMModel(random_state=42)
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            X_train=X_train, y_train=y_train,
            include_shap=True, n_importance_repeats=0,
            show_progress=False
        )

        assert result.shap_values is not None

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_shap_xgbcox(self, train_test_data):
        """Verify SHAP works with XGBCox if available."""
        try:
            from mlsurv.models import XGBCoxModel
        except ImportError:
            pytest.skip("xgboost not installed")

        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        model = XGBCoxModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            X_train=X_train, y_train=y_train,
            include_shap=True, n_importance_repeats=0,
            show_progress=False
        )

        assert result.shap_values is not None


@pytest.mark.shap
class TestSHAPCustomModels:
    """Test SHAP validation for custom models."""

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_shap_custom_model_valid(self, train_test_data):
        """Valid custom model with predict() works."""
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        class CustomModel:
            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.random.randn(len(X))

            def score(self, X, y):
                return 0.5

        model = CustomModel().fit(X_train, y_train)
        result = evaluate(
            model, X_test, y_test,
            include_shap=True, n_importance_repeats=0,
            show_progress=False
        )

        assert result.shap_values is not None

    def test_shap_custom_model_no_predict(self, train_test_data):
        """Model without predict() raises TypeError from _compute_shap validation."""
        from mlsurv.evaluation._shap import _compute_shap

        X_train, X_test, y_train, y_test = train_test_data

        class BadModel:
            pass

        with pytest.raises(TypeError, match="must have a callable predict"):
            _compute_shap(BadModel(), X_test)

    def test_shap_custom_model_bad_predict_signature(self, train_test_data):
        """Model with predict() that fails on input raises TypeError."""
        from mlsurv.evaluation._shap import _compute_shap

        X_train, X_test, y_train, y_test = train_test_data

        class BadModel:
            def predict(self, X):
                raise ValueError("Model doesn't accept DataFrames")

        with pytest.raises(TypeError, match="predict.*failed on test input"):
            _compute_shap(BadModel(), X_test)

    def test_shap_custom_model_bad_predict_output(self, train_test_data):
        """Model with predict() that returns wrong format raises TypeError."""
        from mlsurv.evaluation._shap import _compute_shap

        X_train, X_test, y_train, y_test = train_test_data

        class BadModel:
            def predict(self, X):
                # Return wrong length (always 1 element regardless of input)
                return np.array([1.0])

        with pytest.raises(TypeError, match="must return array-like with one value per sample"):
            _compute_shap(BadModel(), X_test)


@pytest.mark.shap
class TestSHAPValueValidation:
    """Test SHAP value properties and correctness."""

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_shap_values_finite(self, train_test_data):
        """Verify SHAP values contain no NaN or Inf."""
        from mlsurv.models import CoxPHModel
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            include_shap=True, n_importance_repeats=0,
            show_progress=False
        )

        shap_values = result.shap_values['shap_value'].values
        assert np.all(np.isfinite(shap_values)), "SHAP values contain NaN or Inf"

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_shap_correct_dimensions(self, train_test_data):
        """Verify SHAP DataFrame has n_samples * n_features rows."""
        from mlsurv.models import CoxPHModel
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            include_shap=True, n_importance_repeats=0,
            show_progress=False
        )

        n_samples = min(len(X_test), 50)  # _compute_shap caps at max_explain=50
        n_features = X_test.shape[1]
        expected_rows = n_samples * n_features

        assert len(result.shap_values) == expected_rows

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_shap_all_features_present(self, train_test_data):
        """Verify all feature names are present in SHAP output."""
        from mlsurv.models import CoxPHModel
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            include_shap=True, n_importance_repeats=0,
            show_progress=False
        )

        shap_features = set(result.shap_values['feature'].unique())
        expected_features = set(X_test.columns)

        assert shap_features == expected_features


@pytest.mark.shap
class TestSHAPEdgeCases:
    """Test SHAP with edge case datasets."""

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_shap_small_dataset(self, small_data):
        """Verify SHAP works with small dataset (50 samples)."""
        from mlsurv.models import CoxPHModel
        from mlsurv.evaluation import evaluate

        X, y = small_data
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            include_shap=True, n_importance_repeats=0,
            show_progress=False
        )

        assert result.shap_values is not None
        assert len(result.shap_values) > 0

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_shap_high_dimensional(self, high_dimensional_data):
        """Verify SHAP works with high-dimensional data (100 samples, 100 features)."""
        from mlsurv.models import RSFModel
        from mlsurv.evaluation import evaluate

        X, y = high_dimensional_data
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Use RSF for high-dimensional data (more robust than Cox)
        model = RSFModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            include_shap=True, n_importance_repeats=0,
            show_progress=False
        )

        assert result.shap_values is not None
        # Verify all 100 features are present
        assert len(result.shap_values['feature'].unique()) == 100

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_shap_single_feature(self, single_feature_data):
        """Verify SHAP works with single feature edge case."""
        from mlsurv.models import CoxPHModel
        from mlsurv.evaluation import evaluate

        X, y = single_feature_data
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            include_shap=True, n_importance_repeats=0,
            show_progress=False
        )

        assert result.shap_values is not None
        # Verify single feature is present
        assert len(result.shap_values['feature'].unique()) == 1
        assert result.shap_values['feature'].iloc[0] == 'single_feat'


@pytest.mark.shap
class TestSHAPErrorHandling:
    """Test SHAP error handling and graceful failures."""

    def test_shap_import_error_message(self, train_test_data, monkeypatch):
        """Verify helpful message when shap not installed."""
        from mlsurv.evaluation import evaluate
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        # Mock shap import failure
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'shap':
                raise ImportError("No module named 'shap'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', mock_import)

        with pytest.raises(ImportError, match="shap.*package is required"):
            evaluate(
                model, X_test, y_test,
                include_shap=True, n_importance_repeats=0,
                show_progress=False
            )

    def test_evaluate_without_shap_flag(self, train_test_data):
        """Verify include_shap=False returns None for shap_values."""
        from mlsurv.evaluation import evaluate
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            include_shap=False, n_importance_repeats=0,
            show_progress=False
        )

        assert result.shap_values is None


@pytest.mark.shap
class TestSHAPOutputFormat:
    """The one model-agnostic explainer yields the same schema for every model."""

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_output_format_identical_across_models(self, train_test_data):
        """CoxPH, RSF, and XGBCox all produce identical column structure."""
        from mlsurv.models import CoxPHModel, RSFModel
        from mlsurv.evaluation._shap import _compute_shap

        X_train, X_test, y_train, y_test = train_test_data
        expected_cols = {'feature', 'sample_idx', 'shap_value', 'feature_value'}

        coxph = CoxPHModel()
        coxph.fit(X_train, y_train)
        result_cox = _compute_shap(coxph, X_test, max_background=20, max_explain=10)
        assert set(result_cox.columns) == expected_cols

        rsf = RSFModel(n_estimators=10, random_state=42)
        rsf.fit(X_train, y_train)
        result_rsf = _compute_shap(rsf, X_test, max_background=20, max_explain=10)
        assert set(result_rsf.columns) == expected_cols

        try:
            from mlsurv.models import XGBCoxModel
            xgb = XGBCoxModel(n_estimators=20, random_state=42)
            xgb.fit(X_train, y_train)
            result_xgb = _compute_shap(xgb, X_test, max_background=20, max_explain=10)
            assert set(result_xgb.columns) == expected_cols
        except ImportError:
            pass  # XGBoost not installed


@pytest.mark.shap
class TestSHAPSampleSizeParams:
    """Test shap_max_background / shap_max_explain parameter behavior."""

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_compute_shap_respects_random_state(self, train_test_data):
        """Same random_state produces identical SHAP results; different seeds differ."""
        from mlsurv.models import CoxPHModel
        from mlsurv.evaluation._shap import _compute_shap

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        # max_explain < len(X_test) triggers subsampling
        r1 = _compute_shap(model, X_test, max_explain=5, random_state=42)
        r2 = _compute_shap(model, X_test, max_explain=5, random_state=42)
        pd.testing.assert_frame_equal(r1, r2)

        r3 = _compute_shap(model, X_test, max_explain=5, random_state=99)
        # Different seed should select different samples → different feature_values
        assert not r1['feature_value'].equals(r3['feature_value'])

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_compute_shap_reproducible_for_nonadditive_model(self, train_test_data):
        """random_state seeds the permutation order too, so even a non-additive
        model (RSF) gives identical SHAP values run to run.

        RSF risk is non-additive, so the permutation order changes the result.
        The explainer is constructed with ``seed=random_state``, which seeds
        ``np.random`` — without it two runs would differ.
        """
        from mlsurv.models import RSFModel
        from mlsurv.evaluation._shap import _compute_shap

        X_train, X_test, y_train, y_test = train_test_data
        model = RSFModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        r1 = _compute_shap(model, X_test, max_background=20, max_explain=8,
                           random_state=7)
        r2 = _compute_shap(model, X_test, max_background=20, max_explain=8,
                           random_state=7)
        pd.testing.assert_frame_equal(r1, r2)

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_background_below_max_background_is_used_whole(self, train_test_data):
        """A background smaller than max_background runs (the else branch).

        Covers the ``len(X_background) <= max_background`` path in
        ``_run_explainer`` (no subsample), including a 1-row background, and
        explaining more rows than the background has. Model-agnostic, so CoxPH
        exercises the same single explainer every model uses.
        """
        from mlsurv.models import CoxPHModel
        from mlsurv.evaluation._shap import _compute_shap

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        # 3-row background, far below max_background, explaining 10 rows.
        r3 = _compute_shap(model, X_test, X_background=X_train.iloc[:3],
                           max_background=50, max_explain=10, random_state=0)
        assert len(r3) > 0 and np.isfinite(r3['shap_value']).all()

        # 1-row background (boundary).
        r1 = _compute_shap(model, X_test, X_background=X_train.iloc[:1],
                           max_background=50, max_explain=10, random_state=0)
        assert len(r1) > 0 and np.isfinite(r1['shap_value']).all()

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_max_explain_controls_output_size(self, train_test_data):
        """Non-default max_explain changes the number of rows in the SHAP DataFrame."""
        from mlsurv.models import CoxPHModel
        from mlsurv.evaluation._shap import _compute_shap

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        n_features = X_test.shape[1]

        r5 = _compute_shap(model, X_test, max_explain=5, random_state=0)
        assert len(r5) == 5 * n_features

        r10 = _compute_shap(model, X_test, max_explain=10, random_state=0)
        assert len(r10) == 10 * n_features

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_evaluate_forwards_shap_params_to_compute_shap(self, train_test_data):
        """evaluate() correctly threads shap_max_background/explain to _compute_shap."""
        from unittest import mock
        from mlsurv.models import CoxPHModel
        from mlsurv.evaluation import evaluate
        from mlsurv.evaluation._shap import _compute_shap

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        with mock.patch(
            'mlsurv.evaluation.evaluate._compute_shap',
            wraps=_compute_shap,
        ) as spy:
            evaluate(
                model, X_test, y_test,
                include_shap=True,
                shap_max_background=25,
                shap_max_explain=15,
                n_importance_repeats=0,
                show_progress=False,
            )
            spy.assert_called_once()
            call_kwargs = spy.call_args
            assert call_kwargs.kwargs['max_background'] == 25
            assert call_kwargs.kwargs['max_explain'] == 15

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_compute_shap_rejects_invalid_params(self):
        """max_background < 1 and max_explain < 1 raise ValueError."""
        from mlsurv.evaluation._shap import _compute_shap

        # Dummy model — validation happens before predict() is called
        class DummyModel:
            def predict(self, X):
                return np.zeros(len(X))

        dummy = DummyModel()
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        with pytest.raises(ValueError, match="max_background must be >= 1"):
            _compute_shap(dummy, X, max_background=0)

        with pytest.raises(ValueError, match="max_explain must be >= 1"):
            _compute_shap(dummy, X, max_explain=0)


class TestComputeMetricDispatch:
    """Test _compute_metric dispatch table error handling."""

    def test_invalid_metric_raises_valueerror(self):
        """Verify _compute_metric raises ValueError for unknown metric names."""
        from mlsurv.evaluation._bootstrap_iteration import _compute_metric

        y = make_survival_array([True, False], [1.0, 2.0])
        risk_scores = np.array([0.5, 0.3])

        with pytest.raises(ValueError, match="Unknown metric"):
            _compute_metric(y, risk_scores, 'nonexistent')


class TestSupportsPredictTimes:
    """Test _supports_predict_times introspection helper."""

    def test_returns_true_when_times_param_present(self):
        from mlsurv.evaluation.metrics import _supports_predict_times
        def fn(X, times=None): pass
        assert _supports_predict_times(fn) is True

    def test_returns_false_when_no_times_param(self):
        from mlsurv.evaluation.metrics import _supports_predict_times
        def fn(X, return_array=False): pass
        assert _supports_predict_times(fn) is False

    def test_returns_false_for_uninspectable(self):
        from mlsurv.evaluation.metrics import _supports_predict_times
        # Built-in callables can't be inspected
        assert _supports_predict_times(len) is False


class TestPredictSurvivalFunctionsDispatcher:
    """Test that predict_survival_functions dispatcher correctly routes times."""

    def test_dispatcher_returns_ndarray_for_times_model(self):
        """Dispatcher returns ndarray when model supports times parameter."""
        from mlsurv.evaluation.metrics import predict_survival_functions

        class TimesModel:
            def predict_survival_function(self, X, times=None):
                if times is not None:
                    return np.ones((len(X), len(times))) * 0.5
                return [lambda t: 0.5]  # StepFunction-like fallback

        model = TimesModel()
        X = pd.DataFrame({'x': [1, 2, 3]})
        eval_times = np.array([1.0, 5.0, 10.0])

        result = predict_survival_functions(model, X, eval_times)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)

    def test_dispatcher_falls_back_without_times_support(self):
        """Dispatcher falls back for models without times parameter."""
        from mlsurv.evaluation.metrics import predict_survival_functions

        class NoTimesModel:
            def predict_survival_function(self, X):
                return [lambda t: 0.9] * len(X)

        model = NoTimesModel()
        X = pd.DataFrame({'x': [1, 2]})
        eval_times = np.array([1.0, 5.0])

        result = predict_survival_functions(model, X, eval_times)
        # Should return list of callables (StepFunction-like)
        assert callable(result[0])

    def test_dispatcher_xgboost_returns_ndarray(self, train_test_data):
        """XGBoost dispatcher path returns ndarray with correct shape."""
        pytest.importorskip("xgboost")
        from mlsurv.models import get_model
        from mlsurv.evaluation.metrics import predict_survival_functions

        X_train, X_test, y_train, y_test = train_test_data

        # Use low learning_rate to prevent overfitting on synthetic data
        model = get_model('xgbcox', n_estimators=50, max_depth=3,
                          learning_rate=0.01, random_state=42)
        model.fit(X_train, y_train)

        eval_times = np.array([0.5, 2.0, 5.0, 8.0])
        result = predict_survival_functions(model, X_test, eval_times)

        assert isinstance(result, np.ndarray)
        assert result.shape == (len(X_test), len(eval_times))
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0) and np.all(result <= 1)


class TestXGBoostIBSFix:
    """Regression tests for XGBoost IBS inflation fix."""

    def test_xgboost_ibs_not_inflated_vs_coxph(self, train_test_data):
        """XGBoost IBS should not be >2x worse than CoxPH on same data."""
        pytest.importorskip("xgboost")
        from mlsurv.models import get_model
        from mlsurv.evaluation.metrics import _integrated_brier, predict_survival_functions
        from mlsurv.utils import auto_eval_times, clip_eval_times, truncate_y_to_train_range

        X_train, X_test, y_train, y_test = train_test_data

        # Fit both models
        cox = get_model('coxph')
        cox.fit(X_train, y_train)

        # Use low learning_rate to prevent overfitting on synthetic data
        # (default lr=0.3 produces extreme risk scores that overflow Breslow)
        xgb = get_model('xgbcox', n_estimators=100, max_depth=3,
                         learning_rate=0.01, random_state=42)
        xgb.fit(X_train, y_train)

        # Compute eval_times and truncate
        eval_times = auto_eval_times(y_test, y_train=y_train)
        eval_times = clip_eval_times(eval_times, y_train)
        y_test_ipcw = truncate_y_to_train_range(y_test, y_train)

        # Use dispatcher for both
        cox_surv = predict_survival_functions(cox, X_test, eval_times)
        xgb_surv = predict_survival_functions(xgb, X_test, eval_times)

        from mlsurv.evaluation.metrics import _surv_funcs_to_probs
        cox_probs = _surv_funcs_to_probs(cox_surv, eval_times)
        xgb_probs = _surv_funcs_to_probs(xgb_surv, eval_times)

        ibs_cox = _integrated_brier(y_train, y_test_ipcw, cox_probs, eval_times)
        ibs_xgb = _integrated_brier(y_train, y_test_ipcw, xgb_probs, eval_times)

        # Both should be finite
        assert np.isfinite(ibs_cox), f"CoxPH IBS is not finite: {ibs_cox}"
        assert np.isfinite(ibs_xgb), f"XGBoost IBS is not finite: {ibs_xgb}"

        # XGBoost should not be dramatically worse than CoxPH
        # Before fix: XGBoost IBS was ~2.5x CoxPH. After fix they should be comparable.
        ratio = ibs_xgb / ibs_cox if ibs_cox > 0 else 1.0
        assert ratio < 2.0, (
            f"XGBoost IBS ({ibs_xgb:.4f}) is still >2x worse than "
            f"CoxPH IBS ({ibs_cox:.4f}), ratio={ratio:.2f}"
        )


class TestScoreBrierReRaise:
    """Test that _score_brier re-raises MemoryError/RecursionError."""

    def test_memory_error_propagates(self):
        """MemoryError should not be swallowed by the except Exception block."""
        from unittest.mock import patch, MagicMock
        from mlsurv.evaluation.metrics import _score_brier

        estimator = MagicMock()
        X = MagicMock()
        y = MagicMock()
        y_train = MagicMock()

        with patch(
            "mlsurv.evaluation.metrics.truncate_y_to_train_range",
            return_value=y,
        ), patch(
            "mlsurv.evaluation.metrics._resolve_eval_times",
            side_effect=MemoryError("out of memory"),
        ):
            with pytest.raises(MemoryError, match="out of memory"):
                _score_brier(estimator, X, y, "brier", y_train, None)

    def test_recursion_error_propagates(self):
        """RecursionError should not be swallowed by the except Exception block."""
        from unittest.mock import patch, MagicMock
        from mlsurv.evaluation.metrics import _score_brier

        estimator = MagicMock()
        X = MagicMock()
        y = MagicMock()
        y_train = MagicMock()

        with patch(
            "mlsurv.evaluation.metrics.truncate_y_to_train_range",
            return_value=y,
        ), patch(
            "mlsurv.evaluation.metrics._resolve_eval_times",
            side_effect=RecursionError("max recursion"),
        ):
            with pytest.raises(RecursionError, match="max recursion"):
                _score_brier(estimator, X, y, "ibs", y_train, None)


class TestScoreBrierNarrowedExceptions:
    """Test that _score_brier routes expected exceptions to quiet path."""

    @pytest.mark.parametrize(
        "exc_class, exc_msg, expect_user_warning",
        [
            (RuntimeError, "prediction failed", False),
            (TypeError, "bad type", False),
            (AttributeError, "no attribute", True),
        ],
        ids=["RuntimeError-quiet", "TypeError-quiet", "AttributeError-warns"],
    )
    def test_exception_routing(self, exc_class, exc_msg, expect_user_warning):
        """Expected exceptions return NaN quietly; unexpected ones emit a user warning."""
        import warnings
        from unittest.mock import patch, MagicMock
        from mlsurv.evaluation.metrics import _score_brier

        estimator, X, y, y_train = MagicMock(), MagicMock(), MagicMock(), MagicMock()

        with patch(
            "mlsurv.evaluation.metrics.truncate_y_to_train_range",
            return_value=y,
        ), patch(
            "mlsurv.evaluation.metrics._resolve_eval_times",
            side_effect=exc_class(exc_msg),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = _score_brier(estimator, X, y, "brier", y_train, None)

        assert np.isnan(result)
        warned = any(exc_msg in str(wi.message) for wi in w)
        assert warned == expect_user_warning


class TestResolveEvalTimesGracefulDegradation:
    """_resolve_eval_times catches degenerate-input failures, emits a
    deduped SuggestionWarning, and returns None so scorers short-circuit
    to NaN."""

    @pytest.fixture(autouse=True)
    def _reset_dedup(self):
        from mlsurv.evaluation import metrics as metrics_mod
        metrics_mod._reset_eval_times_dedup()
        yield
        metrics_mod._reset_eval_times_dedup()

    @pytest.fixture
    def degenerate_y(self):
        y = make_survival_array([True] * 5, [1.0] * 5)
        return y, y.copy()

    @pytest.mark.parametrize(
        "exc_class, exc_msg",
        [
            (ValueError, "Only 1 unique event time(s)"),
            (TypeError, "non-array y_train"),
        ],
    )
    def test_returns_none_and_warns(self, degenerate_y, exc_class, exc_msg):
        import warnings as wmod
        from unittest.mock import patch
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.evaluation import metrics as metrics_mod

        y, y_train = degenerate_y

        with patch(
            "mlsurv.evaluation.metrics.auto_eval_times",
            side_effect=exc_class(exc_msg),
        ):
            with wmod.catch_warnings(record=True) as w:
                wmod.simplefilter("always")
                result = metrics_mod._resolve_eval_times(y, y_train, None)

        assert result is None
        suggestion_warnings = [
            wi for wi in w
            if issubclass(wi.category, SuggestionWarning)
            and "auto_eval_times failed" in str(wi.message)
        ]
        assert len(suggestion_warnings) == 1
        assert exc_msg in str(suggestion_warnings[0].message)

    def test_clip_eval_times_failure_also_handled(self, degenerate_y):
        """Failure originating from clip_eval_times (not auto_eval_times)
        should still produce the same single-warning + None contract."""
        import warnings as wmod
        from unittest.mock import patch
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.evaluation import metrics as metrics_mod

        y, y_train = degenerate_y

        with patch(
            "mlsurv.evaluation.metrics.auto_eval_times",
            return_value=np.array([1.0]),
        ), patch(
            "mlsurv.evaluation.metrics.clip_eval_times",
            side_effect=ValueError("eval_times outside training range"),
        ):
            with wmod.catch_warnings(record=True) as w:
                wmod.simplefilter("always")
                result = metrics_mod._resolve_eval_times(y, y_train, None)

        assert result is None
        suggestion_warnings = [
            wi for wi in w if issubclass(wi.category, SuggestionWarning)
        ]
        assert len(suggestion_warnings) == 1

    def test_dedup_across_repeated_calls(self, degenerate_y):
        """One warning per session even with simplefilter('always') in effect."""
        import warnings as wmod
        from unittest.mock import patch
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.evaluation import metrics as metrics_mod

        y, y_train = degenerate_y

        with patch(
            "mlsurv.evaluation.metrics.auto_eval_times",
            side_effect=ValueError("Only 1 unique event time(s)"),
        ):
            with wmod.catch_warnings(record=True) as w:
                wmod.simplefilter("always")
                for _ in range(10):
                    metrics_mod._resolve_eval_times(y, y_train, None)

        suggestion_warnings = [
            wi for wi in w if issubclass(wi.category, SuggestionWarning)
        ]
        assert len(suggestion_warnings) == 1

    def test_distinct_root_causes_each_warn_once(self, degenerate_y):
        """Different (exc_type, msg) keys each produce one warning so genuinely
        new failures aren't hidden by the dedup."""
        import warnings as wmod
        from unittest.mock import patch
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.evaluation import metrics as metrics_mod

        y, y_train = degenerate_y

        with wmod.catch_warnings(record=True) as w:
            wmod.simplefilter("always")
            with patch(
                "mlsurv.evaluation.metrics.auto_eval_times",
                side_effect=ValueError("Only 1 unique event time(s)"),
            ):
                for _ in range(3):
                    metrics_mod._resolve_eval_times(y, y_train, None)
            with patch(
                "mlsurv.evaluation.metrics.auto_eval_times",
                side_effect=TypeError("non-array y_train"),
            ):
                for _ in range(3):
                    metrics_mod._resolve_eval_times(y, y_train, None)

        suggestion_warnings = [
            wi for wi in w if issubclass(wi.category, SuggestionWarning)
        ]
        assert len(suggestion_warnings) == 2

    def test_cross_scorer_dedup_shared(self, degenerate_y):
        """A single warning is shared across _score_brier and _score_ctd
        when both are called sequentially with the same root cause."""
        import warnings as wmod
        from unittest.mock import patch, MagicMock
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.evaluation.metrics import _score_brier, _score_ctd

        y, y_train = degenerate_y
        estimator, X = MagicMock(), MagicMock()

        with patch(
            "mlsurv.evaluation.metrics.truncate_y_to_train_range",
            return_value=y,
        ), patch(
            "mlsurv.evaluation.metrics.auto_eval_times",
            side_effect=ValueError("Only 1 unique event time(s)"),
        ):
            with wmod.catch_warnings(record=True) as w:
                wmod.simplefilter("always")
                r1 = _score_brier(estimator, X, y, "brier", y_train, None)
                r2 = _score_ctd(estimator, X, y, y_train, None)

        assert np.isnan(r1) and np.isnan(r2)
        suggestion_warnings = [
            wi for wi in w if issubclass(wi.category, SuggestionWarning)
        ]
        assert len(suggestion_warnings) == 1

    @pytest.mark.parametrize("scoring", ["brier", "integrated_brier"])
    def test_score_brier_short_circuits_no_log_error(self, degenerate_y, caplog, scoring):
        """_score_brier returns NaN with no log_metric_failure ERROR when
        _resolve_eval_times returns None, for both 'brier' and 'integrated_brier'."""
        import logging
        from unittest.mock import patch, MagicMock
        from mlsurv.evaluation.metrics import _score_brier

        y, y_train = degenerate_y
        estimator, X = MagicMock(), MagicMock()

        with patch(
            "mlsurv.evaluation.metrics.truncate_y_to_train_range",
            return_value=y,
        ), patch(
            "mlsurv.evaluation.metrics.auto_eval_times",
            side_effect=ValueError("Only 1 unique event time(s)"),
        ):
            with caplog.at_level(logging.ERROR, logger="mlsurv.metrics"):
                result = _score_brier(estimator, X, y, scoring, y_train, None)

        assert np.isnan(result)
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert not any("metric=" in r.getMessage() for r in error_records)

    def test_score_ctd_short_circuits_no_log_error(self, degenerate_y, caplog):
        """_score_ctd returns NaN with no log_metric_failure ERROR when
        _resolve_eval_times returns None."""
        import logging
        from unittest.mock import patch, MagicMock
        from mlsurv.evaluation.metrics import _score_ctd

        y, y_train = degenerate_y
        estimator, X = MagicMock(), MagicMock()

        with patch(
            "mlsurv.evaluation.metrics.auto_eval_times",
            side_effect=ValueError("Only 1 unique event time(s)"),
        ):
            with caplog.at_level(logging.ERROR, logger="mlsurv.metrics"):
                result = _score_ctd(estimator, X, y, y_train, None)

        assert np.isnan(result)
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert not any("metric=ctd" in r.getMessage() for r in error_records)


class TestPHTestNarrowedExceptions:
    """Test that ph_test routes expected exceptions to quiet path."""

    @pytest.fixture
    def ph_data(self):
        """Minimal data for ph_test exception routing tests."""
        X = pd.DataFrame({'x1': [1.0, 2.0, 3.0] * 10})
        y = np.array(
            [(True, float(i)) for i in range(30)],
            dtype=[('event', bool), ('time', float)],
        )
        return X, y

    @pytest.mark.parametrize(
        "exc_class, exc_msg, expect_user_warning",
        [
            (ValueError, "convergence failure", False),
            (KeyError, "missing column", False),
            (RuntimeError, "solver failed", False),
            (np.linalg.LinAlgError, "singular matrix", False),
            (AttributeError, "api changed", True),
        ],
        ids=["ValueError-quiet", "KeyError-quiet", "RuntimeError-quiet",
             "LinAlgError-quiet", "AttributeError-warns"],
    )
    def test_exception_routing(self, ph_data, exc_class, exc_msg, expect_user_warning):
        """Expected exceptions return empty quietly; unexpected ones emit a user warning."""
        import warnings
        from unittest.mock import patch
        from mlsurv.evaluation.metrics import ph_test

        X, y = ph_data

        with patch(
            "lifelines.CoxPHFitter.fit",
            side_effect=exc_class(exc_msg),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = ph_test(X, y)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        warned = any(exc_msg in str(wi.message) for wi in w)
        assert warned == expect_user_warning


class TestPHTestConvergenceFallback:
    """ph_test pre-drops zero-variance cols and escalates ridge penalty."""

    @staticmethod
    def _make_data(seed=0, n=300, n_features=2):
        rng = np.random.default_rng(seed)
        cols = {f"x{i}": rng.standard_normal(n) for i in range(n_features)}
        X = pd.DataFrame(cols)
        lp = 0.4 * X["x0"]
        T = rng.exponential(scale=np.exp(-lp))
        C = rng.exponential(scale=2.0, size=n)
        time = np.minimum(T, C)
        event = (T <= C).astype(bool)
        y = np.array(
            [(bool(e), float(t)) for e, t in zip(event, time)],
            dtype=[("event", bool), ("time", float)],
        )
        return X, y

    def test_drops_zero_variance_column(self):
        """Constant column is dropped before fit; result covers remaining features."""
        import warnings
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.evaluation.metrics import ph_test

        X, y = self._make_data()
        X["const_feat"] = 0.0  # zero-variance

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ph_test(X, y)

        assert not result.empty
        per_feat = result[result["feature"] != "GLOBAL"]["feature"].tolist()
        assert "const_feat" not in per_feat
        assert "x0" in per_feat and "x1" in per_feat

        drop_warns = [
            wi for wi in w
            if issubclass(wi.category, SuggestionWarning)
            and "zero-variance" in str(wi.message)
            and "const_feat" in str(wi.message)
        ]
        assert len(drop_warns) >= 1, (
            f"expected SuggestionWarning naming const_feat; got "
            f"{[(wi.category.__name__, str(wi.message)[:120]) for wi in w]}"
        )

    def test_penalty_escalates_on_convergence_error(self, monkeypatch):
        """First penalty fails → ladder retries with next; SuggestionWarning fires."""
        import warnings
        from lifelines import CoxPHFitter
        from lifelines.exceptions import ConvergenceError
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.evaluation.metrics import ph_test

        original_fit = CoxPHFitter.fit
        call_log = []

        def flaky_fit(self, *args, **kwargs):
            call_log.append(self.penalizer)
            if self.penalizer == 0.01:
                raise ConvergenceError("simulated 0.01 failure")
            return original_fit(self, *args, **kwargs)

        monkeypatch.setattr(CoxPHFitter, "fit", flaky_fit)

        X, y = self._make_data()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ph_test(X, y)

        assert call_log[0] == 0.01  # ladder starts at 0.01 for ph_test
        assert 0.1 in call_log  # next rung tried
        assert not result.empty
        approx = [
            wi for wi in w
            if issubclass(wi.category, SuggestionWarning)
            and "approximation" in str(wi.message).lower()
            and "penalizer=0.1" in str(wi.message)
        ]
        assert len(approx) >= 1

    def test_unrecoverable_returns_empty_with_warning(self, monkeypatch):
        """Every penalty in ladder fails → ph_test returns empty + MethodologicalWarning."""
        import warnings
        from lifelines import CoxPHFitter
        from lifelines.exceptions import ConvergenceError
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import ph_test

        def always_fail(self, *args, **kwargs):
            raise ConvergenceError("simulated total failure")

        monkeypatch.setattr(CoxPHFitter, "fit", always_fail)

        X, y = self._make_data()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ph_test(X, y)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

        ladder_warns = [
            wi for wi in w
            if issubclass(wi.category, MethodologicalWarning)
            and "ph_test" in str(wi.message)
            and "ladder" in str(wi.message)
        ]
        assert len(ladder_warns) >= 1, (
            f"expected MethodologicalWarning naming ladder exhaustion; got "
            f"{[(wi.category.__name__, str(wi.message)[:120]) for wi in w]}"
        )

    def test_all_features_zero_variance_returns_empty(self):
        """Every feature constant → empty design → empty result + warnings."""
        import warnings
        from mlsurv._warnings import MethodologicalWarning, SuggestionWarning
        from mlsurv.evaluation.metrics import ph_test

        n = 200
        X = pd.DataFrame({"const0": np.zeros(n), "const1": np.ones(n)})
        rng = np.random.default_rng(0)
        time = rng.exponential(size=n)
        event = rng.random(n) < 0.6
        y = np.array(
            [(bool(e), float(t)) for e, t in zip(event, time)],
            dtype=[("event", bool), ("time", float)],
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ph_test(X, y)

        assert result.empty
        # SuggestionWarning names both dropped columns
        drop = [
            wi for wi in w
            if issubclass(wi.category, SuggestionWarning)
            and "zero-variance" in str(wi.message)
            and "const0" in str(wi.message) and "const1" in str(wi.message)
        ]
        assert len(drop) >= 1
        # MethodologicalWarning explains no features remain
        empty = [
            wi for wi in w
            if issubclass(wi.category, MethodologicalWarning)
            and "no features remain" in str(wi.message)
        ]
        assert len(empty) >= 1

    def test_no_rows_path_not_double_logged(self, caplog):
        """The empty-return paths emit the message once via warnings.warn,
        not also via logger.warning — otherwise users watching stderr see each
        event twice (the WARNING line duplicates the MethodologicalWarning).
        """
        import logging
        import warnings
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import ph_test

        n = 50
        # One fully-NaN column => drop_nan_rows empties the frame => no-rows path.
        X = pd.DataFrame({"x0": np.full(n, np.nan)})
        rng = np.random.default_rng(0)
        y = np.array(
            [(bool(e), float(t))
             for e, t in zip(rng.random(n) < 0.6, rng.exponential(size=n))],
            dtype=[("event", bool), ("time", float)],
        )

        with caplog.at_level(logging.WARNING, logger="mlsurv.metrics"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = ph_test(X, y)

        assert result.empty
        # The user-facing warning still fires exactly via warnings.warn.
        user = [
            wi for wi in w
            if issubclass(wi.category, MethodologicalWarning)
            and "no rows remain after NaN removal" in str(wi.message)
        ]
        assert len(user) >= 1
        # ...and the same string is NOT also emitted on the logger at WARNING.
        dup = [
            r for r in caplog.records
            if r.name == "mlsurv.metrics"
            and "no rows remain after NaN removal" in r.getMessage()
        ]
        assert not dup, (
            f"ph_test no-rows message should not be logged at WARNING "
            f"(duplicates the MethodologicalWarning); got {[r.getMessage() for r in dup]}"
        )

    def test_empty_summary_path_not_double_logged(self, caplog):
        """The empty-summary path (distinct from no-rows: it lives inside
        the try block, reached when proportional_hazard_test returns an empty
        summary) also emits the message once via warnings.warn, not logger.warning.
        """
        import logging
        import warnings
        from unittest.mock import patch
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import ph_test

        rng = np.random.default_rng(3)
        n = 200
        X = pd.DataFrame({"x0": rng.standard_normal(n), "x1": rng.standard_normal(n)})
        y = np.array(
            [(bool(e), float(t))
             for e, t in zip(rng.random(n) < 0.6, rng.exponential(size=n))],
            dtype=[("event", bool), ("time", float)],
        )

        class _EmptySummary:
            summary = pd.DataFrame()

        with caplog.at_level(logging.WARNING, logger="mlsurv.metrics"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with patch("lifelines.statistics.proportional_hazard_test",
                           return_value=_EmptySummary()):
                    result = ph_test(X, y)

        assert result.empty
        user = [
            wi for wi in w
            if issubclass(wi.category, MethodologicalWarning)
            and "no features in proportional_hazard_test summary" in str(wi.message)
        ]
        assert len(user) >= 1
        dup = [
            r for r in caplog.records
            if r.name == "mlsurv.metrics"
            and "no features in proportional_hazard_test summary" in r.getMessage()
        ]
        assert not dup, (
            f"ph_test empty-summary message should not be logged at WARNING; "
            f"got {[r.getMessage() for r in dup]}"
        )


class TestPHTest:
    """Test ph_test() Grambsch-Therneau proportional hazards test."""

    @pytest.fixture
    def cox_data(self):
        """Synthetic Cox data with known PH behavior."""
        rng = np.random.RandomState(42)
        n = 300
        x1 = rng.randn(n)
        x2 = rng.randn(n)
        lp = 0.5 * x1 + 0.3 * x2
        event_times = rng.exponential(1.0 / np.exp(lp))
        censor_times = rng.exponential(3.0, size=n)
        times = np.minimum(event_times, censor_times)
        events = event_times <= censor_times

        X = pd.DataFrame({'x1': x1, 'x2': x2})
        y = make_survival_array(events, times)
        return X, y

    def test_basic_output_shape(self, cox_data):
        """ph_test returns DataFrame with expected columns and GLOBAL row."""
        from mlsurv.evaluation.metrics import ph_test

        X, y = cox_data
        result = ph_test(X, y)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['feature', 'test_statistic', 'p_value', 'p_adjusted', 'significant']
        # 2 features + GLOBAL row
        assert len(result) == 3
        assert 'GLOBAL' in result['feature'].values

    def test_global_row_exists(self, cox_data):
        """GLOBAL row is last and aggregates per-feature statistics."""
        from mlsurv.evaluation.metrics import ph_test

        X, y = cox_data
        result = ph_test(X, y)
        global_row = result[result['feature'] == 'GLOBAL'].iloc[0]

        assert global_row['test_statistic'] >= 0
        assert 0 <= global_row['p_value'] <= 1

    def test_p_values_in_range(self, cox_data):
        """All p-values should be in [0, 1]."""
        from mlsurv.evaluation.metrics import ph_test

        X, y = cox_data
        result = ph_test(X, y)

        assert (result['p_value'] >= 0).all()
        assert (result['p_value'] <= 1).all()

    def test_significant_column_uses_fdr(self, cox_data):
        """Per-feature significant uses FDR-adjusted p; GLOBAL uses raw p."""
        from mlsurv.evaluation.metrics import ph_test

        X, y = cox_data
        result = ph_test(X, y)

        # Per-feature rows: significant based on p_adjusted
        features = result[result['feature'] != 'GLOBAL']
        expected_feat_sig = features['p_adjusted'] < 0.05
        assert (features['significant'] == expected_feat_sig).all()

        # GLOBAL row: significant based on raw p_value (p_adjusted is NaN)
        global_row = result[result['feature'] == 'GLOBAL'].iloc[0]
        assert global_row['significant'] == (global_row['p_value'] < 0.05)
        assert np.isnan(global_row['p_adjusted'])

    @pytest.mark.parametrize("transform", ['rank', 'km', 'identity', 'log'])
    def test_time_transform_options(self, cox_data, transform):
        """All time_transform options should produce valid output."""
        from mlsurv.evaluation.metrics import ph_test

        X, y = cox_data
        result = ph_test(X, y, time_transform=transform)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert (result['p_value'] >= 0).all()
        assert (result['p_value'] <= 1).all()

    def test_global_chi2_is_joint_test(self, cox_data):
        """Global row should have valid joint test statistic and p-value."""
        from mlsurv.evaluation.metrics import ph_test

        X, y = cox_data
        result = ph_test(X, y)

        global_row = result[result['feature'] == 'GLOBAL'].iloc[0]
        assert global_row['test_statistic'] > 0
        assert 0 <= global_row['p_value'] <= 1

    def test_single_feature(self):
        """Single-feature input should produce 2 rows (feature + GLOBAL)."""
        from mlsurv.evaluation.metrics import ph_test

        rng = np.random.RandomState(99)
        n = 200
        x1 = rng.randn(n)
        lp = 0.5 * x1
        event_times = rng.exponential(1.0 / np.exp(lp))
        censor_times = rng.exponential(3.0, size=n)
        times = np.minimum(event_times, censor_times)
        events = event_times <= censor_times

        X = pd.DataFrame({'x1': x1})
        y = make_survival_array(events, times)
        result = ph_test(X, y)

        assert len(result) == 2
        assert result.iloc[0]['feature'] == 'x1'
        assert result.iloc[1]['feature'] == 'GLOBAL'
        # With single feature, global stat equals the feature's stat
        np.testing.assert_allclose(
            result.iloc[1]['test_statistic'],
            result.iloc[0]['test_statistic'],
            rtol=1e-8,
        )

    def test_error_returns_empty_dataframe(self):
        """Completely degenerate data should return empty DataFrame, not crash."""
        from mlsurv.evaluation.metrics import ph_test

        # Empty DataFrame — cannot fit Cox model
        X = pd.DataFrame({'x1': pd.Series([], dtype=float)})
        y = make_y([])
        result = ph_test(X, y)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ['feature', 'test_statistic', 'p_value', 'p_adjusted', 'significant']

    def test_p_adjusted_in_range(self, cox_data):
        """p_adjusted should be in [0, 1] for per-feature rows, NaN for GLOBAL."""
        from mlsurv.evaluation.metrics import ph_test

        X, y = cox_data
        result = ph_test(X, y)

        features = result[result['feature'] != 'GLOBAL']
        assert (features['p_adjusted'] >= 0).all()
        assert (features['p_adjusted'] <= 1).all()

        global_row = result[result['feature'] == 'GLOBAL'].iloc[0]
        assert np.isnan(global_row['p_adjusted'])

    def test_p_adjusted_geq_raw(self, cox_data):
        """BH-adjusted p-values should be >= raw p-values."""
        from mlsurv.evaluation.metrics import ph_test

        X, y = cox_data
        result = ph_test(X, y)

        features = result[result['feature'] != 'GLOBAL']
        assert (features['p_adjusted'].values >= features['p_value'].values - 1e-15).all()

    def test_single_feature_fdr_noop(self):
        """With 1 feature, BH correction is a no-op: p_adjusted == p_value."""
        from mlsurv.evaluation.metrics import ph_test

        rng = np.random.RandomState(99)
        n = 200
        x1 = rng.randn(n)
        lp = 0.5 * x1
        event_times = rng.exponential(1.0 / np.exp(lp))
        censor_times = rng.exponential(3.0, size=n)
        times = np.minimum(event_times, censor_times)
        events = event_times <= censor_times

        X = pd.DataFrame({'x1': x1})
        y = make_survival_array(events, times)
        result = ph_test(X, y)

        feat_row = result[result['feature'] == 'x1'].iloc[0]
        np.testing.assert_allclose(feat_row['p_adjusted'], feat_row['p_value'], rtol=1e-8)

    def test_nan_feature_skipped(self):
        """If a per-feature result is NaN, it should be excluded from output."""
        from unittest.mock import patch
        from mlsurv.evaluation.metrics import ph_test
        from lifelines.statistics import proportional_hazard_test as real_ph

        rng = np.random.RandomState(77)
        n = 200
        x1 = rng.randn(n)
        x2 = rng.randn(n)
        lp = 0.5 * x1 + 0.3 * x2
        event_times = rng.exponential(1.0 / np.exp(lp))
        censor_times = rng.exponential(3.0, size=n)
        times = np.minimum(event_times, censor_times)
        events = event_times <= censor_times

        X = pd.DataFrame({'x1': x1, 'x2': x2})
        y = make_survival_array(events, times)

        def _inject_nan(cph, df, time_transform='rank'):
            result = real_ph(cph, df, time_transform=time_transform)
            # pandas 3.0 CoW: lifelines' internal arrays may be read-only
            ts = result._test_statistic.copy()
            ts[0] = float('nan')
            result._test_statistic = ts
            pv = result._p_value.copy()
            pv[0] = float('nan')
            result._p_value = pv
            return result

        with patch('lifelines.statistics.proportional_hazard_test',
                   side_effect=_inject_nan):
            result = ph_test(X, y)

        # x1 (NaN) excluded, only x2 + GLOBAL remain
        assert len(result) == 2
        assert result.iloc[0]['feature'] == 'x2'
        assert result.iloc[1]['feature'] == 'GLOBAL'
        assert not result['test_statistic'].isna().any()
        assert not result['p_value'].isna().any()

    def test_excluded_feature_emits_suggestion_warning(self):
        """Partial NaN exclusion (some rows survive) → SuggestionWarning naming
        the dropped feature, not a silent logger.warning."""
        import warnings
        from unittest.mock import patch
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.evaluation.metrics import ph_test
        from lifelines.statistics import proportional_hazard_test as real_ph

        rng = np.random.RandomState(77)
        n = 200
        x1 = rng.randn(n)
        x2 = rng.randn(n)
        lp = 0.5 * x1 + 0.3 * x2
        event_times = rng.exponential(1.0 / np.exp(lp))
        censor_times = rng.exponential(3.0, size=n)
        times = np.minimum(event_times, censor_times)
        events = event_times <= censor_times

        X = pd.DataFrame({'x1': x1, 'x2': x2})
        y = make_survival_array(events, times)

        def _inject_nan(cph, df, time_transform='rank'):
            result = real_ph(cph, df, time_transform=time_transform)
            ts = result._test_statistic.copy()
            ts[0] = float('nan')
            result._test_statistic = ts
            pv = result._p_value.copy()
            pv[0] = float('nan')
            result._p_value = pv
            return result

        with patch('lifelines.statistics.proportional_hazard_test',
                   side_effect=_inject_nan):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = ph_test(X, y)

        # Surviving rows still returned (x2 + GLOBAL)
        assert len(result) == 2
        # SuggestionWarning naming the excluded feature
        excl = [
            wi for wi in w
            if issubclass(wi.category, SuggestionWarning)
            and "excluded" in str(wi.message)
            and "x1" in str(wi.message)
        ]
        assert len(excl) >= 1, (
            f"expected SuggestionWarning naming excluded feature x1; got "
            f"{[(wi.category.__name__, str(wi.message)[:120]) for wi in w]}"
        )

    def test_singular_joint_covariance_emits_methodological_warning(
        self, cox_data, monkeypatch
    ):
        """Singular joint covariance → sum-of-chi2 fallback emits a
        MethodologicalWarning (GLOBAL row is an independence-assuming
        approximation), not a silent logger.warning.

        We force the degraded branch by stubbing the joint-test helper to
        report ``degraded=True`` (a global ``np.linalg.inv`` patch would also
        break the lifelines Cox fit, which inverts its information matrix)."""
        import warnings
        import mlsurv.evaluation.metrics as metrics_mod
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import ph_test

        X, y = cox_data

        def _degraded(cph, df, rows, time_transform, scaled_resids=None):
            # Mirror the real fallback's return: sum-of-per-feature chi2,
            # df = n_features, degraded=True.
            return sum(r['test_statistic'] for r in rows), len(rows), True

        monkeypatch.setattr(metrics_mod, "_compute_joint_global_test", _degraded)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ph_test(X, y)

        # GLOBAL row still produced from the sum-of-chi2 fallback
        assert 'GLOBAL' in result['feature'].values
        approx = [
            wi for wi in w
            if issubclass(wi.category, MethodologicalWarning)
            and "ph_test" in str(wi.message)
            and "independence" in str(wi.message).lower()
        ]
        assert len(approx) >= 1, (
            f"expected MethodologicalWarning about independence approximation; "
            f"got {[(wi.category.__name__, str(wi.message)[:120]) for wi in w]}"
        )

    def test_joint_helper_returns_degraded_on_singular(self, cox_data, monkeypatch):
        """`_compute_joint_global_test` itself returns the 3-tuple with
        degraded=True when the joint covariance inversion fails.

        Exercises the real helper's ``except`` fallback — complements
        test_singular_joint_covariance_emits_methodological_warning (which
        stubs the helper to test only the caller). Guards against a regression
        where a fallback return reverts to a 2-tuple (which would crash the
        caller's unpack and silently return an empty DataFrame). Patching the
        inner ``np.linalg.inv`` is safe here: the Cox fit completes before the
        helper is called."""
        from lifelines import CoxPHFitter
        import mlsurv.evaluation.metrics as metrics_mod

        X, y = cox_data
        df = X.copy()
        df['T'] = y['time']
        df['E'] = y['event'].astype(int)
        cph = CoxPHFitter().fit(df, duration_col='T', event_col='E')
        rows = [
            {'feature': 'x1', 'test_statistic': 1.5},
            {'feature': 'x2', 'test_statistic': 2.5},
        ]

        # Sanity: clean path returns degraded=False.
        stat, joint_df, degraded = metrics_mod._compute_joint_global_test(
            cph, df, rows, 'rank',
        )
        assert degraded is False

        # Force the inversion to fail → fallback branch.
        def _raise_singular(a, *args, **kwargs):
            raise np.linalg.LinAlgError("simulated singular joint covariance")

        monkeypatch.setattr(metrics_mod.np.linalg, "inv", _raise_singular)
        stat, joint_df, degraded = metrics_mod._compute_joint_global_test(
            cph, df, rows, 'rank',
        )
        assert degraded is True
        # Fallback statistic is the sum of per-feature chi2; df = n features.
        assert stat == pytest.approx(1.5 + 2.5)
        assert joint_df == len(rows)

    def test_clean_joint_covariance_no_degraded_warning(self, cox_data):
        """Non-singular joint covariance → no degraded MethodologicalWarning
        (guards against the warning firing on the happy path)."""
        import warnings
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import ph_test

        X, y = cox_data
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ph_test(X, y)

        assert 'GLOBAL' in result['feature'].values
        degraded = [
            wi for wi in w
            if issubclass(wi.category, MethodologicalWarning)
            and "independence" in str(wi.message).lower()
        ]
        assert not degraded, (
            f"degraded warning should not fire on clean data; got "
            f"{[str(wi.message)[:120] for wi in degraded]}"
        )

    def test_nan_data_does_not_crash(self):
        """ph_test drops NaN rows and returns valid results."""
        from mlsurv.evaluation.metrics import ph_test

        rng = np.random.RandomState(42)
        n = 300
        x1 = rng.randn(n)
        x2 = rng.randn(n)
        lp = 0.5 * x1 + 0.3 * x2
        event_times = rng.exponential(1.0 / np.exp(lp))
        censor_times = rng.exponential(3.0, size=n)
        times = np.minimum(event_times, censor_times)
        events = event_times <= censor_times

        X = pd.DataFrame({'x1': x1, 'x2': x2})
        # Inject ~10% NaN
        nan_idx = rng.choice(n, size=30, replace=False)
        X.loc[nan_idx, 'x1'] = np.nan

        y = make_survival_array(events, times)
        result = ph_test(X, y)

        assert isinstance(result, pd.DataFrame)
        # 2 features + GLOBAL
        assert len(result) == 3
        assert 'GLOBAL' in result['feature'].values
        assert not result['p_value'].isna().any()
        assert not result['test_statistic'].isna().any()

    def test_all_nan_returns_empty(self):
        """ph_test returns empty DataFrame when all features are NaN."""
        from mlsurv.evaluation.metrics import ph_test

        X = pd.DataFrame({'x1': [np.nan, np.nan, np.nan], 'x2': [np.nan, np.nan, np.nan]})
        y = make_survival_array([True, False, True], [1.0, 2.0, 3.0])
        result = ph_test(X, y)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_per_feature_nan_warnings_collapsed_into_summary(self, caplog):
        """Per-feature NaN exclusions emit one summary WARNING, not N.

        The pre-fix behavior logged one ``logger.warning`` per excluded
        feature plus a summary "all features produced NaN — returning
        empty" line. That produced 11 stderr lines for FLCHAIN-style
        failures (8 features + 1 summary + 2 downstream). A later fix collapsed the
        per-feature lines to DEBUG, then dropped the summary
        ``logger.warning`` entirely (it duplicated the MethodologicalWarning
        string), so the all-NaN path now emits ZERO logger WARNING lines —
        the summary lives only on the warnings stack. Verify both: no
        WARNING-level logger records, per-feature lines at DEBUG, and the
        MethodologicalWarning required by the No-Silent-Data-Loss rule.
        """
        import logging
        import warnings as _warnings
        from unittest.mock import patch
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation.metrics import ph_test
        from lifelines.statistics import proportional_hazard_test as real_ph

        rng = np.random.RandomState(11)
        n = 200
        features = {f'x{i}': rng.randn(n) for i in range(5)}
        X = pd.DataFrame(features)
        times = rng.exponential(1.0, size=n)
        events = rng.random(n) < 0.6
        y = make_survival_array(events, times)

        def _force_all_nan(cph, df, time_transform='rank'):
            result = real_ph(cph, df, time_transform=time_transform)
            ts = result._test_statistic.copy()
            pv = result._p_value.copy()
            ts[:] = float('nan')
            pv[:] = float('nan')
            result._test_statistic = ts
            result._p_value = pv
            return result

        with caplog.at_level(logging.DEBUG, logger='mlsurv.metrics'), \
                _warnings.catch_warnings(record=True) as w, \
                patch('lifelines.statistics.proportional_hazard_test',
                      side_effect=_force_all_nan):
            _warnings.simplefilter("always")
            result = ph_test(X, y)

        assert result.empty
        warning_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and r.name == 'mlsurv.metrics'
            and 'ph_test' in r.getMessage()
            and 'NaN' in r.getMessage()
        ]
        # zero summary WARNINGs on the logger — neither the (removed)
        # summary line nor the (DEBUG-demoted) per-feature lines reach WARNING.
        assert len(warning_records) == 0, (
            f"expected 0 logger WARNINGs (summary now warnings.warn-only); "
            f"got {len(warning_records)}: {[r.getMessage() for r in warning_records]}"
        )

        # Per-feature names still recoverable at DEBUG.
        debug_records = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG
            and r.name == 'mlsurv.metrics'
            and 'NaN result for feature' in r.getMessage()
        ]
        assert len(debug_records) == 5

        # MethodologicalWarning emitted to the warnings stack so
        # programmatic callers see the empty return is intentional.
        meth = [
            wi for wi in w
            if issubclass(wi.category, MethodologicalWarning)
            and "all 5 features produced NaN" in str(wi.message)
        ]
        assert len(meth) == 1, [
            (wi.category.__name__, str(wi.message)) for wi in w
        ]

    def test_partial_nan_emits_single_summary_warning(self, caplog):
        """Partial NaN exclusion (some features OK) emits one summary."""
        import logging
        from unittest.mock import patch
        from mlsurv.evaluation.metrics import ph_test
        from lifelines.statistics import proportional_hazard_test as real_ph

        rng = np.random.RandomState(77)
        n = 200
        x1 = rng.randn(n)
        x2 = rng.randn(n)
        lp = 0.5 * x1 + 0.3 * x2
        event_times = rng.exponential(1.0 / np.exp(lp))
        censor_times = rng.exponential(3.0, size=n)
        times = np.minimum(event_times, censor_times)
        events = event_times <= censor_times

        X = pd.DataFrame({'x1': x1, 'x2': x2})
        y = make_survival_array(events, times)

        def _inject_nan(cph, df, time_transform='rank'):
            result = real_ph(cph, df, time_transform=time_transform)
            ts = result._test_statistic.copy()
            pv = result._p_value.copy()
            ts[0] = float('nan')
            pv[0] = float('nan')
            result._test_statistic = ts
            result._p_value = pv
            return result

        with caplog.at_level(logging.DEBUG, logger='mlsurv.metrics'), \
                patch('lifelines.statistics.proportional_hazard_test',
                      side_effect=_inject_nan):
            result = ph_test(X, y)

        # Valid features survive (x2 + GLOBAL).
        assert not result.empty
        assert 'x2' in result['feature'].values

        warning_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and r.name == 'mlsurv.metrics'
            and 'features produced NaN' in r.getMessage()
        ]
        assert len(warning_records) == 1, (
            f"expected 1 partial-exclusion WARNING; got "
            f"{[r.getMessage() for r in warning_records]}"
        )
        assert "1/2" in warning_records[0].getMessage()

    def test_setup_does_not_emit_redundant_learner_warnings(self, caplog):
        """setup() with a failing ph_test emits at most one
        WARNING from mlsurv.metrics and zero from mlsurv.learner._setup.

        Pre-fix, an all-NaN ph_test failure produced:
        - 8 per-feature WARNINGs from mlsurv.metrics
        - 1 "all features produced NaN" summary from mlsurv.metrics
        - 1 "PH test returned empty results" WARNING from mlsurv.learner
        - 1 "ph_test_result is unavailable" WARNING from mlsurv.learner
        Total: 11. Post-fix: 1.
        """
        import logging
        from unittest.mock import patch
        from mlsurv import SurvivalLearner
        from lifelines.statistics import proportional_hazard_test as real_ph

        rng = np.random.RandomState(0)
        n = 120
        n_feat = 5
        X = pd.DataFrame({f'f{i}': rng.randn(n) for i in range(n_feat)})
        y = np.array(
            [(bool(rng.binomial(1, 0.5)), float(rng.exponential(2)))
             for _ in range(n)],
            dtype=[('event', bool), ('time', float)],
        )

        def _force_all_nan(cph, df, time_transform='rank'):
            r = real_ph(cph, df, time_transform=time_transform)
            ts = r._test_statistic.copy()
            pv = r._p_value.copy()
            ts[:] = float('nan')
            pv[:] = float('nan')
            r._test_statistic = ts
            r._p_value = pv
            return r

        with caplog.at_level(logging.WARNING), \
                patch('lifelines.statistics.proportional_hazard_test',
                      side_effect=_force_all_nan):
            learner = SurvivalLearner(X, y)
            learner.setup(feature_selector=None, random_state=42)

        # No WARNING-level learner messages about empty/unavailable ph/lin.
        forbidden_substrings = (
            "PH test returned empty results",
            "ph_test_result is unavailable",
            "linearity_test_result is unavailable",
        )
        offenders = [
            r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING
            and r.name.startswith('mlsurv.learner')
            and any(s in r.getMessage() for s in forbidden_substrings)
        ]
        assert offenders == [], (
            f"expected no redundant WARNINGs from mlsurv.learner; got: "
            f"{offenders}"
        )

        # At most one ph_test summary WARNING from mlsurv.metrics.
        metric_warnings = [
            r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING
            and r.name == 'mlsurv.metrics'
            and 'ph_test' in r.getMessage()
        ]
        assert len(metric_warnings) <= 1, (
            f"expected ≤1 ph_test WARNING from mlsurv.metrics; got: "
            f"{metric_warnings}"
        )


class TestRMST:
    """Test rmst() computation."""

    def test_constant_hazard(self):
        """Exponential survival: RMST = (1 - exp(-lambda*tau)) / lambda."""
        from mlsurv.evaluation.metrics import rmst

        rng = np.random.RandomState(42)
        lam = 0.5
        n = 5000
        times = rng.exponential(1.0 / lam, size=n)
        events = np.ones(n, dtype=bool)  # no censoring
        tau = 3.0

        rmst_val, se_val = rmst(times, events, tau=tau)
        expected = (1 - np.exp(-lam * tau)) / lam

        np.testing.assert_allclose(rmst_val, expected, rtol=0.05)
        assert se_val > 0

    def test_no_events_before_tau(self):
        """All censored: KM stays at 1.0, RMST = tau."""
        from mlsurv.evaluation.metrics import rmst

        times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        events = np.zeros(5, dtype=bool)
        tau = 5.0

        rmst_val, se_val = rmst(times, events, tau=tau)
        np.testing.assert_allclose(rmst_val, tau, rtol=1e-6)

    def test_default_tau_is_max_time(self):
        """Default tau should be maximum observed time."""
        from mlsurv.evaluation.metrics import rmst

        rng = np.random.RandomState(42)
        times = rng.exponential(5.0, size=100)
        events = rng.binomial(1, 0.7, size=100).astype(bool)

        rmst_val, se_val = rmst(times, events)
        assert rmst_val > 0
        assert se_val >= 0

    def test_rmst_increases_with_tau(self):
        """RMST should increase as tau increases."""
        from mlsurv.evaluation.metrics import rmst

        rng = np.random.RandomState(42)
        times = rng.exponential(5.0, size=200)
        events = rng.binomial(1, 0.7, size=200).astype(bool)

        r1, _ = rmst(times, events, tau=2.0)
        r2, _ = rmst(times, events, tau=5.0)
        r3, _ = rmst(times, events, tau=10.0)

        assert r1 < r2 < r3

    def test_se_positive(self):
        """SE should be positive for non-trivial data."""
        from mlsurv.evaluation.metrics import rmst

        rng = np.random.RandomState(42)
        times = rng.exponential(5.0, size=200)
        events = rng.binomial(1, 0.7, size=200).astype(bool)

        _, se_val = rmst(times, events, tau=5.0)
        assert se_val > 0

    def test_empty_array_raises(self):
        """Empty input should raise ValueError."""
        from mlsurv.evaluation.metrics import rmst

        with pytest.raises(ValueError, match="at least one observation"):
            rmst(np.array([]), np.array([]), tau=5.0)

    def test_nan_times_raises(self):
        """NaN in times should raise ValueError."""
        from mlsurv.evaluation.metrics import rmst

        with pytest.raises(ValueError, match="NaN"):
            rmst(np.array([1.0, np.nan, 3.0]), np.array([1, 1, 0]), tau=5.0)

    def test_nan_events_raises(self):
        """NaN in events should raise ValueError."""
        from mlsurv.evaluation.metrics import rmst

        with pytest.raises(ValueError, match="NaN"):
            rmst(np.array([1.0, 2.0, 3.0]), np.array([1.0, np.nan, 0.0]), tau=5.0)

    def test_negative_tau_raises(self):
        """Negative tau should raise ValueError."""
        from mlsurv.evaluation.metrics import rmst

        with pytest.raises(ValueError, match="tau must be positive"):
            rmst(np.array([1.0, 2.0]), np.array([1, 0]), tau=-1.0)


class TestRMSTTest:
    """Test rmst_test() group comparison."""

    @pytest.fixture
    def two_group_data(self):
        """Two groups with different hazard rates."""
        rng = np.random.RandomState(42)
        n = 300
        # Group 0: low risk (lambda=0.1)
        times0 = rng.exponential(10.0, size=n)
        events0 = rng.binomial(1, 0.8, size=n).astype(bool)
        # Group 1: high risk (lambda=0.5)
        times1 = rng.exponential(2.0, size=n)
        events1 = rng.binomial(1, 0.8, size=n).astype(bool)

        times = np.concatenate([times0, times1])
        events = np.concatenate([events0, events1])
        groups = np.array([0] * n + [1] * n)
        return times, events, groups

    def test_basic_output(self, two_group_data):
        """rmst_test returns RMSTResult with expected attributes."""
        from mlsurv.evaluation.metrics import rmst_test, RMSTResult

        times, events, groups = two_group_data
        result = rmst_test(times, events, groups, tau=5.0)

        assert isinstance(result, RMSTResult)
        assert len(result.rmst) == 2
        assert len(result.se) == 2
        assert result.tau == 5.0
        assert 0 <= result.p_value <= 1

    def test_significant_difference(self, two_group_data):
        """Groups with very different hazards should give p < 0.05."""
        from mlsurv.evaluation.metrics import rmst_test

        times, events, groups = two_group_data
        result = rmst_test(times, events, groups, tau=5.0)

        assert result.p_value < 0.05
        # Group 0 (low risk) should have higher RMST than group 1
        assert result.rmst['0'] > result.rmst['1']

    def test_no_difference(self):
        """Same distribution -> p > 0.05 (most of the time)."""
        from mlsurv.evaluation.metrics import rmst_test

        rng = np.random.RandomState(42)
        n = 200
        times = rng.exponential(5.0, size=2 * n)
        events = rng.binomial(1, 0.7, size=2 * n).astype(bool)
        groups = np.array([0] * n + [1] * n)

        result = rmst_test(times, events, groups, tau=5.0)
        assert result.p_value > 0.01

    def test_ci_contains_diff(self, two_group_data):
        """95% CI should contain the point estimate."""
        from mlsurv.evaluation.metrics import rmst_test

        times, events, groups = two_group_data
        result = rmst_test(times, events, groups, tau=5.0)

        assert result.ci_lower <= result.diff <= result.ci_upper

    def test_wrong_groups_raises(self):
        """More than 2 groups should raise ValueError."""
        from mlsurv.evaluation.metrics import rmst_test

        times = np.array([1.0, 2.0, 3.0])
        events = np.array([1, 1, 1])
        groups = np.array([0, 1, 2])

        with pytest.raises(ValueError, match="exactly 2 groups"):
            rmst_test(times, events, groups)

    def test_default_tau(self, two_group_data):
        """Default tau = min of max times per group."""
        from mlsurv.evaluation.metrics import rmst_test

        times, events, groups = two_group_data
        result = rmst_test(times, events, groups)

        max_g0 = times[groups == 0].max()
        max_g1 = times[groups == 1].max()
        expected_tau = min(max_g0, max_g1)
        np.testing.assert_allclose(result.tau, expected_tau)

    def test_diff_se_positive(self, two_group_data):
        """Difference SE should be positive."""
        from mlsurv.evaluation.metrics import rmst_test

        times, events, groups = two_group_data
        result = rmst_test(times, events, groups, tau=5.0)

        assert result.diff_se > 0

    def test_zero_diff_se_all_censored(self):
        """Both groups all-censored beyond tau: SE=0, z=0, p=1."""
        from mlsurv.evaluation.metrics import rmst_test

        # All censored, times well beyond tau — KM stays at 1.0, SE=0
        times = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        events = np.zeros(6, dtype=bool)
        groups = np.array([0, 0, 0, 1, 1, 1])

        result = rmst_test(times, events, groups, tau=5.0)
        assert result.z_stat == 0.0
        assert result.p_value == 1.0
        assert result.diff == 0.0

    def test_single_group_raises(self):
        """Single group should raise ValueError."""
        from mlsurv.evaluation.metrics import rmst_test

        times = np.array([1.0, 2.0, 3.0])
        events = np.array([1, 1, 1])
        groups = np.array([0, 0, 0])

        with pytest.raises(ValueError, match="exactly 2 groups"):
            rmst_test(times, events, groups)


class TestRMSTTestPairwise:
    """Test rmst_test_pairwise() for 2+ groups."""

    @pytest.fixture
    def three_group_data(self):
        """Three groups with different hazard rates."""
        rng = np.random.RandomState(42)
        n = 100
        times0 = rng.exponential(10.0, size=n)
        events0 = rng.binomial(1, 0.8, size=n).astype(bool)
        times1 = rng.exponential(5.0, size=n)
        events1 = rng.binomial(1, 0.8, size=n).astype(bool)
        times2 = rng.exponential(2.0, size=n)
        events2 = rng.binomial(1, 0.8, size=n).astype(bool)

        times = np.concatenate([times0, times1, times2])
        events = np.concatenate([events0, events1, events2])
        groups = np.array([0] * n + [1] * n + [2] * n)
        return times, events, groups

    def test_pairwise_3_groups_structure(self, three_group_data):
        """rmst_test_pairwise with 3 groups returns correct structure."""
        from mlsurv.evaluation.metrics import rmst_test_pairwise, PairwiseRMSTResult

        times, events, groups = three_group_data
        result = rmst_test_pairwise(times, events, groups, tau=5.0)

        assert isinstance(result, PairwiseRMSTResult)
        assert len(result.rmst) == 3
        assert len(result.se) == 3
        assert len(result.pairwise) == 3  # C(3,2) = 3 pairs
        assert result.tau == 5.0
        for pair in result.pairwise:
            assert "groups" in pair
            assert "diff" in pair
            assert "raw_p" in pair
            assert "adjusted_p" in pair

    def test_pairwise_2_groups_matches_rmst_test(self):
        """rmst_test_pairwise with 2 groups matches rmst_test values."""
        from mlsurv.evaluation.metrics import rmst_test, rmst_test_pairwise

        rng = np.random.RandomState(42)
        n = 200
        times = np.concatenate([
            rng.exponential(10.0, size=n),
            rng.exponential(2.0, size=n),
        ])
        events = rng.binomial(1, 0.8, size=2 * n).astype(bool)
        groups = np.array([0] * n + [1] * n)

        r2 = rmst_test(times, events, groups, tau=5.0)
        rp = rmst_test_pairwise(times, events, groups, tau=5.0)

        assert len(rp.pairwise) == 1
        np.testing.assert_allclose(rp.pairwise[0]["diff"], r2.diff, rtol=1e-8)
        np.testing.assert_allclose(rp.pairwise[0]["raw_p"], r2.p_value, rtol=1e-8)
        assert rp.rmst == r2.rmst

    def test_pairwise_adjusted_p_values(self, three_group_data):
        """Adjusted p-values should be >= raw p-values (Holm correction)."""
        from mlsurv.evaluation.metrics import rmst_test_pairwise

        times, events, groups = three_group_data
        result = rmst_test_pairwise(times, events, groups, tau=5.0, p_adjust="holm")

        for pair in result.pairwise:
            assert pair["adjusted_p"] >= pair["raw_p"] - 1e-15

    def test_pairwise_single_group_raises(self):
        """rmst_test_pairwise with 1 group raises ValueError."""
        from mlsurv.evaluation.metrics import rmst_test_pairwise

        with pytest.raises(ValueError, match=">= 2 groups"):
            rmst_test_pairwise(
                np.array([1.0, 2.0, 3.0]),
                np.array([1, 1, 0]),
                np.array([0, 0, 0]),
            )

    def test_pairwise_invalid_p_adjust_raises(self):
        """rmst_test_pairwise with invalid p_adjust raises ValueError."""
        from mlsurv.evaluation.metrics import rmst_test_pairwise

        with pytest.raises(ValueError, match="Unknown p_adjust"):
            rmst_test_pairwise(
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([1, 1, 0, 1]),
                np.array([0, 0, 1, 1]),
                p_adjust="invalid",
            )


class TestAucTTruncation:
    """auc_t applies truncation guard so y_test beyond y_train max doesn't error."""

    def test_y_test_beyond_train_max_succeeds(self):
        """auc_t should not raise when y_test times exceed y_train max."""
        from mlsurv.utils import to_structured_array
        from mlsurv.evaluation.metrics import auc_t

        rng = np.random.RandomState(42)
        n = 100
        # y_train: times up to 10
        y_train = to_structured_array(
            times=rng.uniform(0.5, 10.0, n),
            events=rng.binomial(1, 0.6, n).astype(bool),
        )
        # y_test: times up to 20 (beyond y_train max)
        y_test = to_structured_array(
            times=rng.uniform(0.5, 20.0, n),
            events=rng.binomial(1, 0.6, n).astype(bool),
        )
        risk_scores = rng.randn(n)
        eval_times = np.array([2.0, 4.0, 6.0, 8.0])

        # Should not raise — truncation guard handles the mismatch
        auc_values, mean_auc = auc_t(y_train, y_test, risk_scores, eval_times)
        assert len(auc_values) == len(eval_times)
        assert np.isfinite(mean_auc)


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------

class TestPublicSurvivalUtilities:
    """surv_funcs_to_probs and predict_survival_functions are importable."""

    def test_surv_funcs_to_probs_import(self):
        from mlsurv.evaluation import surv_funcs_to_probs
        # Smoke test with a simple 2D array
        probs = surv_funcs_to_probs(
            np.array([[0.9, 0.8], [0.7, 0.6]]),
            np.array([1.0, 2.0]),
        )
        assert probs.shape == (2, 2)
        np.testing.assert_array_almost_equal(probs[0], [0.9, 0.8])

    def test_predict_survival_functions_import(self):
        from mlsurv.evaluation import predict_survival_functions
        assert callable(predict_survival_functions)


@pytest.mark.integration
class TestIncludePredictions:
    """Tests for the include_predictions parameter in evaluate()."""

    def test_default_no_predictions(self, synthetic_data_clean):
        """evaluate(bootstrap=True) should NOT include predictions by default."""
        from mlsurv.evaluation import evaluate
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        model = learner.models["coxph"].model
        result = evaluate(
            model, learner.X_test, learner.y_test,
            y_train=learner.y_train,
            bootstrap=True, n_bootstrap=5,
            show_progress=False,
        )
        assert result.bootstrap is not None
        assert result.bootstrap.predictions is None

    def test_include_predictions_true(self, synthetic_data_clean):
        """evaluate(bootstrap=True, include_predictions=True) stores predictions."""
        from mlsurv.evaluation import evaluate
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        model = learner.models["coxph"].model
        result = evaluate(
            model, learner.X_test, learner.y_test,
            y_train=learner.y_train,
            bootstrap=True, n_bootstrap=5,
            include_predictions=True,
            show_progress=False,
        )
        assert result.bootstrap is not None
        assert result.bootstrap.predictions is not None
        assert len(result.bootstrap.predictions) > 0

    def test_predict_from_pipeline_import(self):
        from mlsurv.evaluation import predict_from_pipeline
        assert callable(predict_from_pipeline)
