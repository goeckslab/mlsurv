"""Tests for feature interaction detection (H-statistic, 2D PDP, and SHAP)."""

import re
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mlsurv.evaluation._interactions import (
    FeatureUnit,
    _aggregate_shap_interaction_groups,
    _auto_axis_scale,
    _build_categorical_grid,
    _build_grid,
    _build_interaction_units,
    _build_unit_grid,
    _categorical_grid_labels,
    _compute_h_statistic,
    _compute_pdp_surface,
    _resolve_axis_scale,
    _compute_shap_interactions,
    _extract_tree_estimator,
    _h_statistic_pair_units,
    _patch_sksurv_for_shap,
    _pdp_1d_unit,
    _pdp_2d_unit,
    _select_interaction_features,
)


# Tree-model fit cost scales with forest size; the multi-XGB fixtures below only need
# trained models (not tuned), so they train with tune=False and a tiny n_estimators.
_SMALL_TREE_PARAMS = {'n_estimators': 20}


# ── Fixtures ──────────────────────────────────────────────────────────────

def _to_array(X):
    """Convert DataFrame to ndarray (shared by test model fixtures)."""
    if isinstance(X, pd.DataFrame):
        return X.values
    return X


class _AdditiveModel:
    """Synthetic model with no interactions: predict = x0 + x1."""

    def predict(self, X):
        X = _to_array(X)
        return X[:, 0] + X[:, 1]


class _InteractionModel:
    """Synthetic model with an interaction: predict = x0 * x1."""

    def predict(self, X):
        X = _to_array(X)
        return X[:, 0] * X[:, 1]


class _ConstantModel:
    """Model that always predicts the same value."""

    def predict(self, X):
        X = _to_array(X)
        return np.ones(X.shape[0]) * 5.0


def _make_data(n=50, seed=42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        'x0': rng.randn(n),
        'x1': rng.randn(n),
        'x2': rng.randn(n),
    })


# ── Unit tests: H-statistic computation ──────────────────────────────────

class TestHStatisticComputation:

    def test_additive_model_low_h_stat(self):
        """Additive model (no interaction) should have H-stat near 0."""
        X = _make_data(n=30)
        result = _compute_h_statistic(
            _AdditiveModel(), X, ['x0', 'x1'],
            max_samples=30, random_state=42,
        )
        assert len(result) == 1
        assert result.iloc[0]['feature_1'] == 'x0'
        assert result.iloc[0]['feature_2'] == 'x1'
        # Additive model: H-stat should be ~0
        assert result.iloc[0]['h_statistic'] < 0.05

    def test_interaction_model_high_h_stat(self):
        """Multiplicative model (strong interaction) should have high H-stat."""
        X = _make_data(n=30)
        result = _compute_h_statistic(
            _InteractionModel(), X, ['x0', 'x1'],
            max_samples=30, random_state=42,
        )
        assert len(result) == 1
        # Multiplicative interaction: H-stat should be substantial
        assert result.iloc[0]['h_statistic'] > 0.3

    def test_constant_model_zero_h_stat(self):
        """Constant-prediction model should return H-stat = 0."""
        X = _make_data(n=20)
        result = _compute_h_statistic(
            _ConstantModel(), X, ['x0', 'x1'],
            max_samples=20, random_state=42,
        )
        assert result.iloc[0]['h_statistic'] == 0.0

    def test_multiple_pairs(self):
        """Three features produce C(3,2) = 3 pairs."""
        X = _make_data(n=20)
        result = _compute_h_statistic(
            _AdditiveModel(), X, ['x0', 'x1', 'x2'],
            max_samples=20, random_state=42,
        )
        assert len(result) == 3
        # Pairs should be alphabetically ordered
        pairs = set(zip(result['feature_1'], result['feature_2']))
        assert ('x0', 'x1') in pairs
        assert ('x0', 'x2') in pairs
        assert ('x1', 'x2') in pairs

    def test_sorted_by_h_statistic(self):
        """Results should be sorted descending by h_statistic.

        Ordering uses a tolerance-bucketed sort key (rounded to 1e-12) so
        near-zero noise pairs are tied and broken by feature name;
        assert the bucketed key is descending rather than the raw
        values, which can differ within a tie by float jitter.
        """
        X = _make_data(n=20)
        result = _compute_h_statistic(
            _InteractionModel(), X, ['x0', 'x1', 'x2'],
            max_samples=20, random_state=42,
        )
        keys = result['h_statistic'].round(12).tolist()
        assert keys == sorted(keys, reverse=True)

    def test_reproducible_with_seed(self):
        """Same random_state should produce identical results."""
        X = _make_data(n=50)
        r1 = _compute_h_statistic(
            _InteractionModel(), X, ['x0', 'x1'],
            max_samples=20, random_state=123,
        )
        r2 = _compute_h_statistic(
            _InteractionModel(), X, ['x0', 'x1'],
            max_samples=20, random_state=123,
        )
        pd.testing.assert_frame_equal(r1, r2)

    def test_subsampling(self):
        """When max_samples < n, should subsample without error."""
        X = _make_data(n=200)
        result = _compute_h_statistic(
            _AdditiveModel(), X, ['x0', 'x1'],
            max_samples=10, random_state=42,
        )
        assert len(result) == 1
        assert 0.0 <= result.iloc[0]['h_statistic'] <= 1.0


class TestInputValidation:

    def test_fewer_than_2_features_raises(self):
        X = _make_data(n=10)
        with pytest.raises(ValueError, match="at least 2 features"):
            _compute_h_statistic(_AdditiveModel(), X, ['x0'])

    def test_missing_feature_raises(self):
        X = _make_data(n=10)
        with pytest.raises(ValueError, match="Features not found"):
            _compute_h_statistic(_AdditiveModel(), X, ['x0', 'nonexistent'])


class TestSelectInteractionFeatures:

    def test_selects_top_k_by_importance(self):
        imp = pd.DataFrame({
            'feature': ['x0', 'x1', 'x2', 'x3'],
            'perm_mean': [0.1, 0.4, 0.3, 0.2],
        })
        result = _select_interaction_features(imp, top_k=3, all_columns=pd.Index([]))
        assert result == ['x1', 'x2', 'x3']

    def test_fallback_to_columns(self):
        # Empty importance — should fall back to first top_k columns
        imp = pd.DataFrame(columns=['feature', 'perm_mean'])
        cols = pd.Index(['a', 'b', 'c', 'd'])
        result = _select_interaction_features(imp, top_k=3, all_columns=cols)
        assert result == ['a', 'b', 'c']

    def test_none_importance_fallback(self):
        cols = pd.Index(['a', 'b', 'c'])
        result = _select_interaction_features(None, top_k=2, all_columns=cols)
        assert result == ['a', 'b']

    def test_tie_break_is_deterministic(self):
        # Two features share an identical perm_mean straddling the top_k=2
        # boundary; selection must be stable and alphabetical on the tie so
        # which features get interaction-tested is platform-independent.
        imp = pd.DataFrame({
            'feature': ['zeta', 'alpha', 'beta'],
            'perm_mean': [0.5, 0.3, 0.3],
        })
        result_a = _select_interaction_features(imp, top_k=2, all_columns=pd.Index([]))
        result_b = _select_interaction_features(
            imp.iloc[[2, 1, 0]].reset_index(drop=True),
            top_k=2, all_columns=pd.Index([]),
        )
        assert result_a == result_b == ['zeta', 'alpha']

    def test_near_tie_within_tolerance_breaks_by_name(self):
        # perm_means differ by less than the 1e-12 rounding tolerance, so they
        # collapse into one bucket and the name tie-break (not the raw value)
        # must decide — this is the cross-platform BLAS-jitter case.
        # 'beta' has the marginally higher raw value but 'alpha' wins by name:
        # the two mechanisms disagree, so this assertion holds ONLY when the
        # rounding is present (without it, 'beta' would edge out 'alpha').
        imp = pd.DataFrame({
            'feature': ['top', 'alpha', 'beta'],
            'perm_mean': [0.9, 0.3, 0.3 + 5e-13],
        })
        result = _select_interaction_features(imp, top_k=2, all_columns=pd.Index([]))
        assert result == ['top', 'alpha']

    def test_nan_perm_mean_sorts_last(self):
        # NaN importances must sort to the bottom (fillna(-inf)) so a real-signal
        # feature is never displaced by a NaN-importance one.
        imp = pd.DataFrame({
            'feature': ['a', 'b', 'c'],
            'perm_mean': [np.nan, 0.4, 0.2],
        })
        result = _select_interaction_features(imp, top_k=2, all_columns=pd.Index([]))
        assert result == ['b', 'c']


# ── Integration: evaluate() with include_interactions ─────────────────────

@pytest.mark.integration
@pytest.mark.timeout(120)
class TestEvaluateIntegration:

    @pytest.fixture(scope="class")
    def learner_with_model(self):
        """Create a quick learner with RSF trained (nonlinear, supports interactions)."""
        from sksurv.datasets import load_gbsg2
        from mlsurv import SurvivalLearner

        X_raw, y_raw = load_gbsg2()
        X = X_raw.copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].cat.codes.astype(float)
        y = np.array(
            [(bool(e), t) for e, t in y_raw],
            dtype=[("event", bool), ("time", float)],
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('rsf', n_estimators=50)
        return learner

    def test_evaluate_with_interactions(self, learner_with_model):
        """evaluate(include_interactions=True) stores results for nonlinear models."""
        result = learner_with_model.evaluate(
            include_interactions=True,
            interaction_top_k=3,
            interaction_max_samples=20,
            verbose=False,
            n_importance_repeats=0,
        )
        assert result.interactions is not None
        assert len(result.interactions) > 0
        assert 'h_statistic' in result.interactions.columns
        assert 'feature_1' in result.interactions.columns
        assert 'feature_2' in result.interactions.columns
        assert 'model_type' in result.interactions.columns
        # PDP surfaces should also be cached
        assert result.interaction_surfaces is not None
        assert len(result.interaction_surfaces) > 0
        assert 'pdp_surface' in result.interaction_surfaces[0]
        assert 'model_type' in result.interaction_surfaces[0]

    def test_evaluate_without_interactions(self, learner_with_model):
        """Default evaluate() does NOT compute interactions."""
        result = learner_with_model.evaluate(verbose=False, n_importance_repeats=0)
        assert result.interactions is None
        assert result.interaction_surfaces is None

    def test_evaluate_skips_linear_model(self):
        """Linear models (CoxPH) are skipped with interactions=None."""
        from sksurv.datasets import load_gbsg2
        from mlsurv import SurvivalLearner

        X_raw, y_raw = load_gbsg2()
        X = X_raw.copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].cat.codes.astype(float)
        y = np.array(
            [(bool(e), t) for e, t in y_raw],
            dtype=[("event", bool), ("time", float)],
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')
        result = learner.evaluate(
            include_interactions=True, verbose=False, n_importance_repeats=0,
        )
        # CoxPH is linear — interactions should be skipped (None)
        assert result.interactions is None

    def test_standalone_interactions(self, learner_with_model):
        """learner.interactions() returns DataFrame for nonlinear models."""
        learner_with_model.evaluate(verbose=False, n_importance_repeats=0)
        result = learner_with_model.interactions(
            top_k=3, max_samples=20, grid_resolution=10,
        )
        assert isinstance(result, pd.DataFrame)
        assert 'model_type' in result.columns
        assert 'h_statistic' in result.columns
        assert len(result) == 3  # C(3,2) = 3 pairs

    def test_standalone_interactions_rejects_linear(self):
        """learner.interactions() raises ValueError for linear models."""
        from sksurv.datasets import load_gbsg2
        from mlsurv import SurvivalLearner

        X_raw, y_raw = load_gbsg2()
        X = X_raw.copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].cat.codes.astype(float)
        y = np.array(
            [(bool(e), t) for e, t in y_raw],
            dtype=[("event", bool), ("time", float)],
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')
        learner.evaluate(verbose=False, n_importance_repeats=0)
        with pytest.raises(ValueError, match="linear"):
            learner.interactions()


# ── Warning tests ─────────────────────────────────────────────────────────

class TestParameterWarnings:

    def test_unused_interaction_params_warns(self):
        """Setting interaction params without include_interactions warns."""
        from mlsurv.evaluation.evaluate import _warn_unused_interaction_params
        from mlsurv._warnings import SuggestionWarning

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_unused_interaction_params(
                include_interactions=False,
                interaction_top_k=10,  # non-default
                interaction_max_samples=100,
                stacklevel=2,
            )
        suggestion_warnings = [x for x in w if issubclass(x.category, SuggestionWarning)]
        assert len(suggestion_warnings) == 1
        assert "interaction_top_k" in str(suggestion_warnings[0].message)

    def test_no_warning_when_enabled(self):
        """No warning when include_interactions=True."""
        from mlsurv.evaluation.evaluate import _warn_unused_interaction_params

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_unused_interaction_params(
                include_interactions=True,
                interaction_top_k=10,
                interaction_max_samples=50,
                stacklevel=2,
            )
        assert len(w) == 0

    def test_no_warning_when_defaults(self):
        """No warning when using default values."""
        from mlsurv.evaluation.evaluate import _warn_unused_interaction_params

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_unused_interaction_params(
                include_interactions=False,
                interaction_top_k=5,
                interaction_max_samples=100,
                stacklevel=2,
            )
        assert len(w) == 0


# ── Edge case tests ──────────────────────────────────────────────────────

class _CrashingModel:
    """Model that raises RuntimeError when both x0 and x1 are constant.

    During 2D PDP grid evaluation the (x0, x1) pair has both columns set to
    single grid values, triggering the crash.  Other pairs compute normally.
    """

    def predict(self, X):
        X = _to_array(X)
        if np.all(X[:, 0] == X[0, 0]) and np.all(X[:, 1] == X[0, 1]):
            raise RuntimeError("Simulated overflow in forward pass")
        return X[:, 0] + X[:, 1]


class TestEdgeCases:

    def test_crashing_pair_degrades_gracefully(self):
        """One pair raising an exception → NaN for that pair, others survive."""
        from mlsurv._warnings import SuggestionWarning

        X = _make_data(n=30)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _compute_h_statistic(
                _CrashingModel(), X, ['x0', 'x1', 'x2'],
                max_samples=30, random_state=42,
            )

        # Should have 3 pairs total: (x0,x1), (x0,x2), (x1,x2)
        assert len(result) == 3

        # The (x0, x1) pair should be NaN (crashed)
        crashed = result[
            (result['feature_1'] == 'x0') & (result['feature_2'] == 'x1')
        ]
        assert len(crashed) == 1
        assert np.isnan(crashed.iloc[0]['h_statistic'])

        # Other pairs should have finite values
        others = result[
            ~((result['feature_1'] == 'x0') & (result['feature_2'] == 'x1'))
        ]
        assert others['h_statistic'].notna().all()

        # SuggestionWarning: per-pair warning + summary warning
        suggestion_warnings = [
            x for x in w if issubclass(x.category, SuggestionWarning)
        ]
        assert len(suggestion_warnings) == 2  # per-pair + summary
        assert "x0" in str(suggestion_warnings[0].message)
        assert "x1" in str(suggestion_warnings[0].message)
        assert "1 of 3" in str(suggestion_warnings[1].message)

    def test_crashing_pair_reraises_memory_error(self):
        """MemoryError should propagate, not be caught."""
        class _OOMModel:
            def predict(self, X):
                raise MemoryError("out of memory")

        X = _make_data(n=10)
        with pytest.raises(MemoryError):
            _compute_h_statistic(_OOMModel(), X, ['x0', 'x1'], max_samples=10)

    def test_crashing_pair_reraises_recursion_error(self):
        """RecursionError should propagate, not be caught."""
        class _RecursionModel:
            def predict(self, X):
                raise RecursionError("maximum recursion depth")

        X = _make_data(n=10)
        with pytest.raises(RecursionError):
            _compute_h_statistic(
                _RecursionModel(), X, ['x0', 'x1'], max_samples=10,
            )

    def test_crashing_pair_reraises_import_error(self):
        """ImportError should propagate, not be caught."""
        class _ImportModel:
            def predict(self, X):
                raise ImportError("No module named 'torch'")

        X = _make_data(n=10)
        with pytest.raises(ImportError):
            _compute_h_statistic(
                _ImportModel(), X, ['x0', 'x1'], max_samples=10,
            )

    def test_all_pairs_fail_returns_all_nan(self):
        """When every pair fails, all h_statistic values should be NaN."""
        class _AlwaysCrashModel:
            def predict(self, X):
                raise RuntimeError("always fails")

        X = _make_data(n=10)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = _compute_h_statistic(
                _AlwaysCrashModel(), X, ['x0', 'x1', 'x2'], max_samples=10,
            )
        assert len(result) == 3
        assert result['h_statistic'].isna().all()

    def test_constant_feature_returns_zero(self):
        """Feature with all identical values → single grid point → H=0."""
        X = _make_data(n=30)
        X['x_const'] = 1.0
        result = _compute_h_statistic(
            _InteractionModel(), X, ['x0', 'x_const'],
            max_samples=30, random_state=42,
        )
        assert len(result) == 1
        assert result.iloc[0]['h_statistic'] == 0.0

    def test_nan_in_features(self):
        """Features containing NaN → completes without raising; H-stat is NaN."""
        X = _make_data(n=30)
        X.iloc[0, 0] = np.nan
        result = _compute_h_statistic(
            _AdditiveModel(), X, ['x0', 'x1'],
            max_samples=30, random_state=42,
        )
        assert len(result) == 1
        assert np.isnan(result.iloc[0]['h_statistic'])

    def test_inf_in_features(self):
        """Features containing Inf → H-stat is NaN (non-finite guard)."""
        X = _make_data(n=30)
        X.iloc[0, 0] = np.inf
        result = _compute_h_statistic(
            _AdditiveModel(), X, ['x0', 'x1'],
            max_samples=30, random_state=42,
        )
        assert len(result) == 1
        assert np.isnan(result.iloc[0]['h_statistic'])


# ── Unit tests: _build_grid ────────────────────────────────────────────


class TestBuildGrid:

    def test_fewer_unique_than_resolution(self):
        """When uniques <= resolution, return all unique values."""
        values = np.array([1.0, 2.0, 3.0, 1.0, 2.0])
        grid = _build_grid(values, grid_resolution=10)
        np.testing.assert_array_equal(grid, [1.0, 2.0, 3.0])

    def test_caps_at_resolution(self):
        """When uniques > resolution, grid length <= resolution."""
        values = np.arange(100, dtype=float)
        grid = _build_grid(values, grid_resolution=10)
        assert len(grid) <= 10

    def test_grid_covers_range(self):
        """Grid should span min to max of input."""
        values = np.arange(100, dtype=float)
        grid = _build_grid(values, grid_resolution=10)
        assert grid[0] == values.min()
        assert grid[-1] == values.max()


# ── Unit tests: _compute_pdp_surface ───────────────────────────────────


class TestBuildGridLogScale:
    """Log-spaced grid construction in _build_grid (issue: log PDP grids)."""

    def test_log_spacing_is_even_on_log10_axis(self):
        values = np.linspace(1.0, 1000.0, 200)
        grid = _build_grid(values, 10, scale='log')
        assert len(grid) == 10
        # Even on the log10 axis Plotly uses (type='log').
        diffs = np.diff(np.log10(grid))
        assert np.allclose(diffs, diffs[0])
        assert grid[0] == pytest.approx(values.min())
        assert grid[-1] == pytest.approx(values.max())

    def test_log_floors_zero_to_smallest_positive(self):
        values = np.concatenate([[0.0], np.linspace(1.0, 100.0, 50)])
        grid = _build_grid(values, 8, scale='log')
        assert np.all(np.isfinite(grid))
        assert np.all(grid > 0)  # 0 has no place on a log axis
        assert grid[0] == pytest.approx(1.0)  # smallest positive value

    def test_log_ignores_nan(self):
        values = np.array([np.nan, 1.0, 10.0, 100.0, 1000.0])
        grid = _build_grid(values, 6, scale='log')
        assert np.all(np.isfinite(grid))
        assert grid[0] == pytest.approx(1.0)
        assert grid[-1] == pytest.approx(1000.0)

    def test_log_single_unique_value(self):
        """Constant positive feature: finite, all-equal grid (no NaN/div0)."""
        values = np.full(50, 7.0)
        grid = _build_grid(values, 10, scale='log')
        assert len(grid) == 10
        assert np.all(np.isfinite(grid))
        assert np.allclose(grid, 7.0)

    def test_log_grid_denser_at_low_end_than_linear(self):
        """Log spacing places more points in the dense low region."""
        values = np.linspace(1.0, 1000.0, 500)
        log_grid = _build_grid(values, 25, scale='log')
        lin_grid = _build_grid(values, 25, scale='linear')
        midpoint = 500.0
        assert np.sum(log_grid < midpoint) > np.sum(lin_grid < midpoint)

    def test_linear_default_matches_explicit_linear(self):
        values = np.linspace(0.0, 100.0, 200)
        assert np.array_equal(
            _build_grid(values, 10), _build_grid(values, 10, scale='linear'),
        )


class TestBuildGridLog1pScale:
    """log1p-spaced grid construction in _build_grid (handles zeros)."""

    def test_log1p_round_trips_through_expm1(self):
        values = np.array([0.0, 1.0, 5.0, 50.0, 200.0])
        grid = _build_grid(values, 10, scale='log1p')
        assert len(grid) == 10
        assert np.all(np.diff(grid) > 0)            # monotone increasing
        # Even spacing in log1p space.
        diffs = np.diff(np.log1p(grid))
        assert np.allclose(diffs, diffs[0])

    def test_log1p_includes_zero_floor(self):
        values = np.array([0.0, 0.0, 3.0, 9.0, 99.0])
        grid = _build_grid(values, 8, scale='log1p')
        assert grid[0] == pytest.approx(0.0)        # 0 maps to 0 (unlike log)
        assert grid[-1] == pytest.approx(99.0)

    def test_log1p_ignores_nan(self):
        values = np.array([np.nan, 0.0, 1.0, 10.0, 100.0])
        grid = _build_grid(values, 6, scale='log1p')
        assert np.all(np.isfinite(grid))
        assert grid[0] == pytest.approx(0.0)
        assert grid[-1] == pytest.approx(100.0)


class TestAutoAxisScale:
    """Skew-driven scale selection: _auto_axis_scale / _resolve_axis_scale 'auto'."""

    def _skewed(self, seed=0, n=500):
        rng = np.random.default_rng(seed)
        return np.exp(rng.normal(2.0, 1.2, n))       # strictly positive, right-skewed

    def test_strictly_positive_right_skew_picks_log(self):
        assert _auto_axis_scale(self._skewed()) == 'log'

    def test_nonnegative_with_zeros_right_skew_picks_log1p(self):
        vals = np.r_[np.zeros(200), self._skewed(n=300)]
        assert _auto_axis_scale(vals) == 'log1p'

    def test_low_skew_stays_linear(self):
        rng = np.random.default_rng(1)
        assert _auto_axis_scale(rng.normal(50.0, 5.0, 500)) == 'linear'

    def test_negative_values_stay_linear(self):
        rng = np.random.default_rng(2)
        assert _auto_axis_scale(rng.normal(0.0, 5.0, 500)) == 'linear'

    def test_near_categorical_stays_linear(self):
        rng = np.random.default_rng(3)
        # <= _CAT_MAX_CARDINALITY (5) unique values, even if heavily skewed.
        vals = np.r_[np.zeros(400), np.full(50, 1.0), np.full(50, 99.0)]
        assert _auto_axis_scale(vals) == 'linear'

    def test_empty_stays_linear(self):
        assert _auto_axis_scale(np.array([])) == 'linear'

    def test_resolve_auto_is_deterministic(self):
        vals = self._skewed()
        X = vals.reshape(-1, 1)
        u = FeatureUnit(name='f', col_indices=[0], is_group=False)
        first = _resolve_axis_scale('auto', u, X, 'f')
        assert first == _resolve_axis_scale('auto', u, X, 'f') == 'log'

    def test_resolve_auto_group_is_linear(self):
        X = np.zeros((10, 3))
        u = FeatureUnit(name='g', col_indices=[0, 1, 2], is_group=True)
        # Auto on a categorical group is linear and emits no warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _resolve_axis_scale('auto', u, X, 'g') == 'linear'

    def test_resolve_log1p_with_zeros_ok(self):
        X = np.r_[np.zeros(50), np.arange(1.0, 51.0)].reshape(-1, 1)
        u = FeatureUnit(name='f', col_indices=[0], is_group=False)
        assert _resolve_axis_scale('log1p', u, X, 'f') == 'log1p'

    def test_resolve_log1p_with_negatives_falls_back(self):
        from mlsurv._warnings import SuggestionWarning
        X = np.array([-1.0, 0.0, 1.0, 2.0]).reshape(-1, 1)
        u = FeatureUnit(name='f', col_indices=[0], is_group=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scale = _resolve_axis_scale('log1p', u, X, 'f')
        assert scale == 'linear'
        msgs = [str(x.message) for x in w
                if issubclass(x.category, SuggestionWarning)]
        assert any('negative' in m for m in msgs)

    def test_resolve_invalid_scale_raises(self):
        X = np.ones((4, 1))
        u = FeatureUnit(name='f', col_indices=[0], is_group=False)
        with pytest.raises(ValueError, match="scale must be"):
            _resolve_axis_scale('sqrt', u, X, 'f')

    def test_resolve_log1p_all_nan_falls_back(self):
        # All-NaN feature: explicit log1p must degrade to linear (not crash on
        # an empty-array min), mirroring the 'log' no-positive-value guard.
        from mlsurv._warnings import SuggestionWarning
        X = np.full((5, 1), np.nan)
        u = FeatureUnit(name='f', col_indices=[0], is_group=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scale = _resolve_axis_scale('log1p', u, X, 'f')
        assert scale == 'linear'
        msgs = [str(x.message) for x in w
                if issubclass(x.category, SuggestionWarning)]
        assert any('no finite' in m for m in msgs)


def _make_positive_data(n=120, seed=0):
    """Non-negative skewed data for log-scale tests."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        'x0': np.concatenate([[0.0], rng.uniform(1.0, 500.0, n - 1)]),
        'x1': np.concatenate([[0.0], rng.uniform(1.0, 500.0, n - 1)]),
        'x2': rng.uniform(1.0, 500.0, n),
    })


class TestPdpSurfaceLogScale:
    """scale_1/scale_2 plumbing through _compute_pdp_surface."""

    def test_grid_2_log_spaced(self):
        X = _make_positive_data()
        result = _compute_pdp_surface(
            _InteractionModel(), X, 'x0', 'x1',
            max_samples=120, grid_resolution=10, random_state=42,
            scale_2='log',
        )
        assert result['grid_2_scale'] == 'log'
        assert result['grid_1_scale'] == 'linear'
        diffs = np.diff(np.log10(np.array(result['grid_2'])))
        assert np.allclose(diffs, diffs[0])

    def test_grid_1_log_spaced(self):
        X = _make_positive_data()
        result = _compute_pdp_surface(
            _InteractionModel(), X, 'x0', 'x1',
            max_samples=120, grid_resolution=10, random_state=42,
            scale_1='log',
        )
        assert result['grid_1_scale'] == 'log'
        diffs = np.diff(np.log10(np.array(result['grid_1'])))
        assert np.allclose(diffs, diffs[0])

    def test_zeros_floored_both_axes_log_spaced(self):
        X = _make_positive_data()  # x0/x1 both contain a 0.0
        result = _compute_pdp_surface(
            _InteractionModel(), X, 'x0', 'x1',
            max_samples=120, grid_resolution=10, random_state=42,
            scale_1='log', scale_2='log',
        )
        for key in ('grid_1', 'grid_2'):
            grid = np.array(result[key])
            assert np.all(np.isfinite(grid))
            assert np.all(grid > 0)  # zeros floored off a log axis
            diffs = np.diff(np.log10(grid))
            assert np.allclose(diffs, diffs[0])

    def test_all_zero_feature_warns_and_falls_back(self):
        X = _make_positive_data()
        X['x0'] = 0.0  # no positive values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _compute_pdp_surface(
                _InteractionModel(), X, 'x0', 'x1',
                max_samples=120, grid_resolution=10, random_state=42,
                scale_1='log',
            )
        from mlsurv._warnings import SuggestionWarning
        msgs = [str(x.message) for x in w
                if issubclass(x.category, SuggestionWarning)]
        assert any('no positive' in m for m in msgs)
        assert result['grid_1_scale'] == 'linear'

    def test_nan_in_feature_produces_finite_log_grid(self):
        X = _make_positive_data()
        X.loc[0, 'x1'] = np.nan  # NaN alongside positive values
        result = _compute_pdp_surface(
            _InteractionModel(), X, 'x0', 'x1',
            max_samples=120, grid_resolution=10, random_state=42,
            scale_2='log',
        )
        grid_2 = np.array(result['grid_2'])
        assert result['grid_2_scale'] == 'log'
        assert np.all(np.isfinite(grid_2))
        assert np.all(grid_2 > 0)

    def test_negative_feature_warns_and_falls_back(self):
        X = _make_data(n=40)  # randn columns contain negatives
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _compute_pdp_surface(
                _InteractionModel(), X, 'x0', 'x1',
                max_samples=40, grid_resolution=10, random_state=42,
                scale_2='log',
            )
        from mlsurv._warnings import SuggestionWarning
        msgs = [str(x.message) for x in w
                if issubclass(x.category, SuggestionWarning)]
        assert any('negative' in m for m in msgs)
        assert result['grid_2_scale'] == 'linear'

    def test_invalid_scale_raises(self):
        X = _make_positive_data()
        with pytest.raises(ValueError, match="scale must be"):
            _compute_pdp_surface(
                _InteractionModel(), X, 'x0', 'x1',
                max_samples=20, grid_resolution=5, random_state=42,
                scale_1='sqrt',
            )

    def test_linear_default_has_no_log_scale(self):
        X = _make_positive_data()
        result = _compute_pdp_surface(
            _InteractionModel(), X, 'x0', 'x1',
            max_samples=20, grid_resolution=5, random_state=42,
        )
        assert result['grid_1_scale'] == 'linear'
        assert result['grid_2_scale'] == 'linear'

    def test_log1p_axis_keeps_zeros_and_spaces_in_log1p(self):
        X = _make_positive_data()  # x0/x1 both contain a 0.0
        result = _compute_pdp_surface(
            _InteractionModel(), X, 'x0', 'x1',
            max_samples=120, grid_resolution=10, random_state=42,
            scale_1='log1p',
        )
        assert result['grid_1_scale'] == 'log1p'
        grid = np.array(result['grid_1'])
        assert grid[0] == pytest.approx(0.0)        # zeros stay (unlike log)
        diffs = np.diff(np.log1p(grid))
        assert np.allclose(diffs, diffs[0])

    def test_log1p_negative_feature_falls_back(self):
        from mlsurv._warnings import SuggestionWarning
        X = _make_data(n=40)  # randn columns contain negatives
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _compute_pdp_surface(
                _InteractionModel(), X, 'x0', 'x1',
                max_samples=40, grid_resolution=10, random_state=42,
                scale_2='log1p',
            )
        assert result['grid_2_scale'] == 'linear'
        msgs = [str(x.message) for x in w
                if issubclass(x.category, SuggestionWarning)]
        assert any('negative' in m for m in msgs)


def _make_skewed_interaction_data(n=200, seed=0):
    """Two right-skewed columns: x0 strictly positive, x1 non-negative w/ zeros."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'x0': np.exp(rng.normal(2.0, 1.2, n)),                       # -> 'log'
        'x1': np.r_[np.zeros(n // 2), np.exp(rng.normal(2.0, 1.2, n - n // 2))],  # -> 'log1p'
    })


class TestHStatAutoScaleSurfaces:
    """scale='auto' threads per-axis log/log1p into _compute_h_statistic surfaces."""

    def test_auto_picks_log_and_log1p_per_axis(self):
        X = _make_skewed_interaction_data()
        _, surfaces = _compute_h_statistic(
            _InteractionModel(), X, ['x0', 'x1'],
            max_samples=200, random_state=42,
            compute_surfaces=True, grid_resolution=12, scale='auto',
        )
        assert len(surfaces) == 1
        s = surfaces[0]
        by_feature = {s['feature_1']: s['grid_1_scale'],
                      s['feature_2']: s['grid_2_scale']}
        assert by_feature['x0'] == 'log'
        assert by_feature['x1'] == 'log1p'

    def test_linear_scale_keeps_all_axes_linear(self):
        X = _make_skewed_interaction_data()
        _, surfaces = _compute_h_statistic(
            _InteractionModel(), X, ['x0', 'x1'],
            max_samples=200, random_state=42,
            compute_surfaces=True, grid_resolution=12, scale='linear',
        )
        s = surfaces[0]
        assert s['grid_1_scale'] == 'linear'
        assert s['grid_2_scale'] == 'linear'

    def test_default_scale_is_linear(self):
        X = _make_skewed_interaction_data()
        _, surfaces = _compute_h_statistic(
            _InteractionModel(), X, ['x0', 'x1'],
            max_samples=200, random_state=42,
            compute_surfaces=True, grid_resolution=12,
        )
        s = surfaces[0]
        assert s['grid_1_scale'] == 'linear'
        assert s['grid_2_scale'] == 'linear'


class TestPdpSurface:

    def test_surface_shape(self):
        """_compute_pdp_surface returns dict with matching grid/surface shapes."""
        X = _make_data(n=30)
        result = _compute_pdp_surface(
            _InteractionModel(), X, 'x0', 'x1',
            max_samples=30, grid_resolution=10, random_state=42,
        )
        assert 'grid_1' in result
        assert 'grid_2' in result
        assert 'pdp_surface' in result
        assert result['feature_1'] == 'x0'
        assert result['feature_2'] == 'x1'
        n1 = len(result['grid_1'])
        n2 = len(result['grid_2'])
        assert len(result['pdp_surface']) == n1
        assert all(len(row) == n2 for row in result['pdp_surface'])

    def test_interaction_model_surface_not_additive(self):
        """For x0*x1 model, surface should not decompose as f(x0)+f(x1)."""
        X = _make_data(n=30)
        result = _compute_pdp_surface(
            _InteractionModel(), X, 'x0', 'x1',
            max_samples=30, grid_resolution=10, random_state=42,
        )
        surface = np.array(result['pdp_surface'])
        # Decompose: residual = surface - row_means - col_means + grand_mean
        row_means = surface.mean(axis=1, keepdims=True)
        col_means = surface.mean(axis=0, keepdims=True)
        grand_mean = surface.mean()
        residual = surface - row_means - col_means + grand_mean
        # Interaction model: residual should be nonzero
        assert np.abs(residual).max() > 0.01

    def test_additive_model_surface_nearly_additive(self):
        """For x0+x1 model, residual should be near zero."""
        X = _make_data(n=30)
        result = _compute_pdp_surface(
            _AdditiveModel(), X, 'x0', 'x1',
            max_samples=30, grid_resolution=10, random_state=42,
        )
        surface = np.array(result['pdp_surface'])
        row_means = surface.mean(axis=1, keepdims=True)
        col_means = surface.mean(axis=0, keepdims=True)
        grand_mean = surface.mean()
        residual = surface - row_means - col_means + grand_mean
        assert np.abs(residual).max() < 0.01

    def test_grid_resolution_caps_grid(self):
        """grid_resolution limits grid points per feature."""
        X = _make_data(n=100)
        result = _compute_pdp_surface(
            _AdditiveModel(), X, 'x0', 'x1',
            max_samples=100, grid_resolution=5, random_state=42,
        )
        assert len(result['grid_1']) <= 5
        assert len(result['grid_2']) <= 5

    def test_missing_feature_raises(self):
        X = _make_data(n=10)
        with pytest.raises(ValueError, match="Features not found"):
            _compute_pdp_surface(_AdditiveModel(), X, 'x0', 'nonexistent')

    def test_non_numeric_feature_raises(self):
        X = _make_data(n=10)
        X['cat'] = 'a'
        with pytest.raises(TypeError, match="non-numeric"):
            _compute_pdp_surface(_AdditiveModel(), X, 'x0', 'cat')

    def test_reproducible_with_seed(self):
        X = _make_data(n=50)
        r1 = _compute_pdp_surface(
            _InteractionModel(), X, 'x0', 'x1',
            max_samples=20, random_state=123,
        )
        r2 = _compute_pdp_surface(
            _InteractionModel(), X, 'x0', 'x1',
            max_samples=20, random_state=123,
        )
        assert r1['pdp_surface'] == r2['pdp_surface']

    def test_returns_python_lists(self):
        """All arrays should be Python lists for serialization safety."""
        X = _make_data(n=20)
        result = _compute_pdp_surface(
            _AdditiveModel(), X, 'x0', 'x1',
            max_samples=20, random_state=42,
        )
        assert isinstance(result['grid_1'], list)
        assert isinstance(result['grid_2'], list)
        assert isinstance(result['pdp_surface'], list)
        assert isinstance(result['pdp_surface'][0], list)


# ── Unit tests: _compute_h_statistic with compute_surfaces ─────────────


class TestComputeHStatisticWithSurfaces:

    def test_compute_surfaces_flag(self):
        """compute_surfaces=True returns (df, surfaces) tuple."""
        X = _make_data(n=30)
        result = _compute_h_statistic(
            _InteractionModel(), X, ['x0', 'x1'],
            max_samples=30, random_state=42,
            compute_surfaces=True,
        )
        assert isinstance(result, tuple)
        df, surfaces = result
        assert isinstance(df, pd.DataFrame)
        assert isinstance(surfaces, list)
        assert len(surfaces) == 1  # 1 pair

    def test_surfaces_match_pairs(self):
        """Each surface dict has matching feature names."""
        X = _make_data(n=20)
        df, surfaces = _compute_h_statistic(
            _InteractionModel(), X, ['x0', 'x1', 'x2'],
            max_samples=20, random_state=42,
            compute_surfaces=True,
        )
        assert len(surfaces) == 3  # C(3,2) = 3
        for s in surfaces:
            assert 'feature_1' in s
            assert 'feature_2' in s
            assert 'grid_1' in s
            assert 'grid_2' in s
            assert 'pdp_surface' in s

    def test_surfaces_sorted_like_dataframe(self):
        """Surfaces should be sorted in same order as DataFrame (by H-stat)."""
        X = _make_data(n=30)
        df, surfaces = _compute_h_statistic(
            _InteractionModel(), X, ['x0', 'x1', 'x2'],
            max_samples=30, random_state=42,
            compute_surfaces=True,
        )
        for i, row in df.iterrows():
            assert surfaces[i]['feature_1'] == row['feature_1']
            assert surfaces[i]['feature_2'] == row['feature_2']

    def test_without_surfaces_returns_dataframe(self):
        """compute_surfaces=False (default) returns just DataFrame."""
        X = _make_data(n=20)
        result = _compute_h_statistic(
            _AdditiveModel(), X, ['x0', 'x1'],
            max_samples=20, random_state=42,
        )
        assert isinstance(result, pd.DataFrame)


# ── H-statistic method column ────────────────────────────────────────────

class TestHStatMethodColumn:

    def test_method_column_present(self):
        """H-stat results should include method='hstat' column."""
        X = _make_data(n=20)
        result = _compute_h_statistic(
            _AdditiveModel(), X, ['x0', 'x1'],
            max_samples=20, random_state=42,
        )
        assert 'method' in result.columns
        assert (result['method'] == 'hstat').all()


# ── Unit tests: plot functions ─────────────────────────────────────────


class TestPlotInteractionPdp:

    def _make_surface(self):
        grid_1 = [0.0, 1.0, 2.0]
        grid_2 = [0.0, 0.5, 1.0]
        pdp_surface = [[1.0, 1.1, 1.2], [1.3, 1.5, 1.7], [1.6, 2.0, 2.4]]
        return grid_1, grid_2, pdp_surface

    def test_returns_figure(self):
        g1, g2, s = self._make_surface()
        fig = _make_pdp_fig(g1, g2, s, 'feat_a', 'feat_b')
        assert isinstance(fig, go.Figure)

    def test_contour_trace(self):
        g1, g2, s = self._make_surface()
        fig = _make_pdp_fig(g1, g2, s, 'feat_a', 'feat_b')
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Contour)

    def test_axis_labels(self):
        g1, g2, s = self._make_surface()
        fig = _make_pdp_fig(g1, g2, s, 'feat_a', 'feat_b')
        # x=grid_2 (feature_2), y=grid_1 (feature_1) due to Contour convention
        assert fig.layout.xaxis.title.text == 'feat_b'
        assert fig.layout.yaxis.title.text == 'feat_a'

    def test_colorbar_present(self):
        g1, g2, s = self._make_surface()
        fig = _make_pdp_fig(g1, g2, s, 'feat_a', 'feat_b')
        assert fig.data[0].colorbar is not None

    def test_xaxis_log_type(self):
        g1, g2, s = self._make_surface()
        fig = _make_pdp_fig(g1, g2, s, 'feat_a', 'feat_b', grid_2_scale='log')
        assert fig.layout.xaxis.type == 'log'
        assert fig.layout.yaxis.type != 'log'

    def test_yaxis_log_type(self):
        g1, g2, s = self._make_surface()
        fig = _make_pdp_fig(g1, g2, s, 'feat_a', 'feat_b', grid_1_scale='log')
        assert fig.layout.yaxis.type == 'log'
        assert fig.layout.xaxis.type != 'log'

    def test_linear_default_not_log(self):
        g1, g2, s = self._make_surface()
        fig = _make_pdp_fig(g1, g2, s, 'feat_a', 'feat_b')
        assert fig.layout.xaxis.type != 'log'
        assert fig.layout.yaxis.type != 'log'

    def test_xaxis_log1p_uses_linear_axis_with_custom_ticks(self):
        # Plotly has no log1p axis: it's a linear axis with real-unit ticks.
        g1, g2, s = self._make_surface()
        fig = _make_pdp_fig(g1, g2, s, 'feat_a', 'feat_b', grid_2_scale='log1p')
        assert fig.layout.xaxis.type == 'linear'
        assert fig.layout.xaxis.tickvals is not None
        assert fig.layout.xaxis.ticktext is not None
        assert 'log1p' in fig.layout.xaxis.title.text

    def test_log1p_contour_keeps_real_units_in_hover(self):
        # log1p coordinates are pre-transformed, so hover reads from customdata.
        g1, g2, s = self._make_surface()
        fig = _make_pdp_fig(g1, g2, s, 'feat_a', 'feat_b', grid_1_scale='log1p')
        assert fig.data[0].customdata is not None
        assert 'customdata' in fig.data[0].hovertemplate

    def test_log_axis_title_annotated(self):
        g1, g2, s = self._make_surface()
        fig = _make_pdp_fig(g1, g2, s, 'feat_a', 'feat_b', grid_2_scale='log')
        assert 'log scale' in fig.layout.xaxis.title.text


class TestPlotInteractionPdpGallery:

    def _make_surfaces(self, n=3):
        surfaces = []
        for i in range(n):
            surfaces.append({
                'feature_1': f'f{i}',
                'feature_2': f'f{i+1}',
                'grid_1': [0.0, 1.0],
                'grid_2': [0.0, 1.0],
                'pdp_surface': [[1.0, 1.1], [1.2, 1.5]],
                'model_type': 'rsf',
            })
        return surfaces

    def test_dropdown_with_multiple_pairs(self):
        from mlsurv.viz.interactions import plot_interaction_pdp_gallery
        surfaces = self._make_surfaces(3)
        fig = plot_interaction_pdp_gallery(surfaces)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3
        # Dropdown should exist
        assert fig.layout.updatemenus is not None
        assert len(fig.layout.updatemenus) == 1
        assert len(fig.layout.updatemenus[0].buttons) == 3

    def test_single_pair_no_dropdown(self):
        from mlsurv.viz.interactions import plot_interaction_pdp_gallery
        surfaces = self._make_surfaces(1)
        fig = plot_interaction_pdp_gallery(surfaces)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        # No dropdown for single pair
        assert fig.layout.updatemenus is None or len(fig.layout.updatemenus) == 0

    def test_empty_surfaces(self):
        from mlsurv.viz.interactions import plot_interaction_pdp_gallery
        fig = plot_interaction_pdp_gallery([])
        assert isinstance(fig, go.Figure)
        # Should have annotation about no data
        assert fig.layout.annotations is not None

    def test_heatmap_and_lines_blocks_bookkeeping(self):
        """Variable-size trace blocks (1 heatmap + N lines) stay correctly tracked."""
        from mlsurv.viz.interactions import plot_interaction_pdp_gallery
        surfaces = [
            {  # categorical × categorical -> 1 heatmap trace
                'feature_1': 'stage', 'feature_2': 'grade',
                'grid_1': [[1, 0], [0, 1]], 'grid_2': [[1, 0], [0, 1]],
                'pdp_surface': [[1.0, 1.1], [1.2, 1.3]],
                'grid_1_type': 'categorical', 'grid_2_type': 'categorical',
                'grid_1_labels': ['I', 'II'], 'grid_2_labels': ['lo', 'hi'],
            },
            {  # continuous × categorical (3 levels) -> 3 line traces
                'feature_1': 'age', 'feature_2': 'stage',
                'grid_1': [0.0, 1.0], 'grid_2': [0.0, 1.0, 2.0],
                'pdp_surface': [[1.0, 1.1, 1.2], [1.3, 1.5, 1.7]],
                'grid_1_type': 'numeric', 'grid_2_type': 'categorical',
                'grid_2_labels': ['I', 'II', 'III'],
            },
        ]
        fig = plot_interaction_pdp_gallery(surfaces)
        # 1 heatmap + 3 lines = 4 traces, 2 buttons.
        assert len(fig.data) == 4
        assert isinstance(fig.data[0], go.Heatmap)
        assert all(isinstance(t, go.Scatter) for t in fig.data[1:])
        buttons = fig.layout.updatemenus[0].buttons
        assert len(buttons) == 2
        # Block 0 = heatmap (1 trace), block 1 = lines (3 traces).
        assert list(buttons[0].args[0]['visible']) == [True, False, False, False]
        assert list(buttons[1].args[0]['visible']) == [False, True, True, True]
        # Heatmap button hides the legend; lines button shows it.
        assert buttons[0].args[1]['showlegend'] is False
        assert buttons[1].args[1]['showlegend'] is True


# ── Tree estimator extraction helper ─────────────────────────────────────

class TestExtractTreeEstimator:

    def test_returns_none_for_non_tree(self):
        """Non-tree model returns (None, None)."""
        est, mtype = _extract_tree_estimator(_AdditiveModel())
        assert est is None
        assert mtype is None

    def test_returns_none_for_additive_pipeline(self):
        """Pipeline with non-tree model returns (None, None)."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', _AdditiveModel()),
        ])
        est, mtype = _extract_tree_estimator(pipe)
        assert est is None
        assert mtype is None


# ── SHAP interaction validation ──────────────────────────────────────────

class TestShapInteractionValidation:

    def test_invalid_method_raises(self, small_data):
        """Invalid method name raises ValueError."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('rsf', n_estimators=10)
        learner.evaluate(verbose=False, n_importance_repeats=0)
        with pytest.raises(ValueError, match="Invalid interaction method"):
            learner.interactions(method='invalid')

    def test_shap_not_installed_raises(self):
        """Missing shap package raises ImportError with install hint."""
        X = _make_data(n=20)
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([('model', _AdditiveModel())])

        with patch.dict('sys.modules', {'shap': None}):
            with pytest.raises(ImportError, match="pip install shap"):
                _compute_shap_interactions(
                    pipe, X, ['x0', 'x1'],
                    max_samples=10, random_state=42,
                )

    @pytest.mark.requires_shap
    def test_non_tree_model_raises_type_error(self):
        """Non-tree pipeline raises TypeError."""
        X = _make_data(n=20)
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([('model', _AdditiveModel())])

        with pytest.raises(TypeError, match="tree-based model"):
            _compute_shap_interactions(
                pipe, X, ['x0', 'x1'],
                max_samples=10, random_state=42,
            )

    @pytest.mark.requires_shap
    def test_fewer_than_2_features_raises(self):
        """Less than 2 features raises ValueError."""
        X = _make_data(n=10)
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([('model', _AdditiveModel())])

        with pytest.raises(ValueError, match="at least 2 features"):
            _compute_shap_interactions(
                pipe, X, ['x0'],
                max_samples=10, random_state=42,
            )


# ── Integration: learner.plot_interaction_pdp ──────────────────────────


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestLearnerPlotInteractionPdp:

    @pytest.fixture(scope="class")
    def learner_with_interactions(self):
        """Learner with RSF trained, evaluated, and interactions computed."""
        from sksurv.datasets import load_gbsg2
        from mlsurv import SurvivalLearner

        X_raw, y_raw = load_gbsg2()
        X = X_raw.copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].cat.codes.astype(float)
        y = np.array(
            [(bool(e), t) for e, t in y_raw],
            dtype=[("event", bool), ("time", float)],
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('rsf', n_estimators=50)
        learner.evaluate(verbose=False, n_importance_repeats=0)
        learner.interactions(top_k=3, max_samples=20, grid_resolution=10)
        return learner

    def test_plot_cached_pair(self, learner_with_interactions):
        """Cached pair should return figure without recomputing."""
        learner = learner_with_interactions
        mr = list(learner.models.values())[0]
        surfaces = mr.evaluation.interaction_surfaces
        assert surfaces is not None and len(surfaces) > 0
        f1 = surfaces[0]['feature_1']
        f2 = surfaces[0]['feature_2']
        fig = learner.plot_interaction_pdp(f1, f2)
        assert isinstance(fig, go.Figure)

    def test_plot_uncached_pair(self, learner_with_interactions):
        """Uncached pair should compute on-the-fly."""
        learner = learner_with_interactions
        # Use features that are NOT in the top-3 interactions
        all_features = learner.X_test.columns.tolist()
        mr = list(learner.models.values())[0]
        cached_features = set()
        for s in mr.evaluation.interaction_surfaces:
            cached_features.add(s['feature_1'])
            cached_features.add(s['feature_2'])
        uncached = [f for f in all_features if f not in cached_features]
        if len(uncached) >= 2:
            fig = learner.plot_interaction_pdp(uncached[0], uncached[1])
            assert isinstance(fig, go.Figure)

    def test_return_data(self, learner_with_interactions):
        """return_data=True returns tidy DataFrame."""
        learner = learner_with_interactions
        mr = list(learner.models.values())[0]
        surfaces = mr.evaluation.interaction_surfaces
        f1 = surfaces[0]['feature_1']
        f2 = surfaces[0]['feature_2']
        df = learner.plot_interaction_pdp(f1, f2, return_data=True)
        assert isinstance(df, pd.DataFrame)
        assert 'prediction' in df.columns
        assert f1 in df.columns or surfaces[0]['feature_1'] in df.columns

    def test_linear_model_raises(self):
        """CoxPH should raise ValueError."""
        from sksurv.datasets import load_gbsg2
        from mlsurv import SurvivalLearner

        X_raw, y_raw = load_gbsg2()
        X = X_raw.copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].cat.codes.astype(float)
        y = np.array(
            [(bool(e), t) for e, t in y_raw],
            dtype=[("event", bool), ("time", float)],
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')
        learner.evaluate(verbose=False, n_importance_repeats=0)
        with pytest.raises(ValueError, match="linear"):
            learner.plot_interaction_pdp('age', 'tsize')

    def test_no_eval_raises(self):
        """Should raise ValueError if evaluate() not called."""
        from sksurv.datasets import load_gbsg2
        from mlsurv import SurvivalLearner

        X_raw, y_raw = load_gbsg2()
        X = X_raw.copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].cat.codes.astype(float)
        y = np.array(
            [(bool(e), t) for e, t in y_raw],
            dtype=[("event", bool), ("time", float)],
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('rsf', n_estimators=10)
        with pytest.raises(ValueError, match="evaluation"):
            learner.plot_interaction_pdp('age', 'tsize')

    def test_surfaces_stored_after_interactions(self, learner_with_interactions):
        """interactions() should populate interaction_surfaces."""
        mr = list(learner_with_interactions.models.values())[0]
        assert mr.evaluation.interaction_surfaces is not None
        assert len(mr.evaluation.interaction_surfaces) > 0
        for s in mr.evaluation.interaction_surfaces:
            assert s['model_type'] == mr.name
            # Verify grid_resolution=10 propagated through the pipeline
            assert len(s['grid_1']) <= 10
            assert len(s['grid_2']) <= 10

    def test_log_xscale_sets_axis_type(self, learner_with_interactions):
        """Explicit xscale='log' renders a log x-axis; yscale='linear' keeps y."""
        learner = learner_with_interactions
        fig = learner.plot_interaction_pdp(
            'pnodes', 'progrec', xscale='log', yscale='linear',
        )
        assert isinstance(fig, go.Figure)
        assert fig.layout.xaxis.type == 'log'
        assert fig.layout.yaxis.type != 'log'

    def test_linear_request_recomputes_from_nonlinear_cache(
        self, learner_with_interactions,
    ):
        """A linear request must not reuse a cached non-linear (log/log1p)
        surface; it recomputes a linear one. A matching log1p request still
        reuses it.

        Cache-leak regression guard: evaluate() can cache auto-scaled surfaces,
        so reuse must hinge on the requested scale, not on "is this a linear
        request". The shared fixture's small subsample resolves every axis to
        linear, so this forces a genuine log1p surface into the cache.
        """
        from mlsurv.evaluation._interactions import _compute_pdp_surface
        learner = learner_with_interactions
        mr = list(learner.models.values())[0]
        state = learner._state
        # A numeric x numeric cached pair, so a log1p x-axis is meaningful.
        pair = next(
            s for s in mr.evaluation.interaction_surfaces
            if s['grid_1_type'] == 'numeric' and s['grid_2_type'] == 'numeric'
        )
        f1, f2 = pair['feature_1'], pair['feature_2']
        cat_features = (
            set(state.categorical_features)
            if state.categorical_features is not None else None
        )
        log_surface = _compute_pdp_surface(
            mr.model, learner.X_test, f1, f2,
            max_samples=20, random_state=state.random_state,
            categorical_groups=state.categorical_groups,
            categorical_features=cat_features,
            scale_1='linear', scale_2='log1p',
        )
        log_surface['model_type'] = mr.name
        assert log_surface['grid_2_scale'] == 'log1p'  # precondition
        # Sentinel grid value: a recompute would never produce it, so its
        # presence in returned data proves the cached surface was reused (not
        # recomputed onto a coincidentally-identical grid).
        SENTINEL = 91237.0
        log_surface['grid_2'] = list(log_surface['grid_2'])
        log_surface['grid_2'][0] = SENTINEL

        original = mr.evaluation.interaction_surfaces
        try:
            mr.evaluation.interaction_surfaces = [log_surface]

            # Linear request → recompute a linear axis (NOT the cached log1p).
            fig = learner.plot_interaction_pdp(
                f1, f2, xscale='linear', yscale='linear',
            )
            assert fig.layout.xaxis.type != 'log'
            # A linear axis title carries no "(log scale)"/"(log1p scale)" suffix.
            assert 'scale' not in (fig.layout.xaxis.title.text or '')
            df = learner.plot_interaction_pdp(
                f1, f2, xscale='linear', yscale='linear', return_data=True,
            )
            assert SENTINEL not in df[f2].to_numpy()  # recomputed, not reused

            # Matching log1p request → reuse the cached surface (sentinel back).
            df2 = learner.plot_interaction_pdp(
                f1, f2, xscale='log1p', yscale='linear', return_data=True,
            )
            assert SENTINEL in df2[f2].to_numpy()
        finally:
            mr.evaluation.interaction_surfaces = original

    def test_default_auto_reuses_cached_surface(self, learner_with_interactions):
        """The default (auto) plot of a cached pair reuses the cached surface
        instead of recomputing (the cache was built with auto)."""
        learner = learner_with_interactions
        mr = list(learner.models.values())[0]
        s0 = mr.evaluation.interaction_surfaces[0]
        f1, f2 = s0['feature_1'], s0['feature_2']
        cached_g1 = np.unique(np.array(s0['grid_1']))
        cached_g2 = np.unique(np.array(s0['grid_2']))
        # Match the cache's subsample (max_samples) so 'auto' resolves identically.
        df = learner.plot_interaction_pdp(f1, f2, max_samples=20, return_data=True)
        got_g1 = np.unique(df[f1].to_numpy())
        got_g2 = np.unique(df[f2].to_numpy())
        # Reuse → identical grids (a recompute would use a different resolution).
        assert np.array_equal(np.sort(got_g1), np.sort(cached_g1))
        assert np.array_equal(np.sort(got_g2), np.sort(cached_g2))

    def test_log1p_request_renders_log1p_axis(self, learner_with_interactions):
        """An explicit log1p request labels the axis as a log1p scale."""
        learner = learner_with_interactions
        # progrec has exact zeros, so log1p is valid (log would floor them).
        fig = learner.plot_interaction_pdp(
            'pnodes', 'progrec', xscale='log1p', yscale='linear',
        )
        assert isinstance(fig, go.Figure)
        # log1p renders on a linear Plotly axis with a "(log1p scale)" title.
        assert fig.layout.xaxis.type != 'log'
        assert 'log1p' in (fig.layout.xaxis.title.text or '')

    def test_reuse_matches_swapped_cached_pair(self, learner_with_interactions):
        """Reuse keys the requested scale by feature name, so a cached surface
        whose feature order is swapped relative to the call still matches."""
        from mlsurv.evaluation._interactions import _compute_pdp_surface
        learner = learner_with_interactions
        mr = list(learner.models.values())[0]
        state = learner._state
        pair = next(
            s for s in mr.evaluation.interaction_surfaces
            if s['grid_1_type'] == 'numeric' and s['grid_2_type'] == 'numeric'
        )
        a, b = pair['feature_1'], pair['feature_2']
        cat_features = (
            set(state.categorical_features)
            if state.categorical_features is not None else None
        )
        # Build the surface with features SWAPPED (feature_1=b, feature_2=a),
        # so b is on grid_1 (log1p) and a is on grid_2 (linear).
        swapped = _compute_pdp_surface(
            mr.model, learner.X_test, b, a,
            max_samples=20, random_state=state.random_state,
            categorical_groups=state.categorical_groups,
            categorical_features=cat_features,
            scale_1='log1p', scale_2='linear',
        )
        swapped['model_type'] = mr.name
        assert swapped['grid_1_scale'] == 'log1p'  # b is on grid_1
        SENTINEL = 81237.0
        swapped['grid_1'] = list(swapped['grid_1'])
        swapped['grid_1'][0] = SENTINEL

        original = mr.evaluation.interaction_surfaces
        try:
            mr.evaluation.interaction_surfaces = [swapped]
            # Query in (a, b) order: b is feature_2 (x), so b's scale = xscale.
            # b matches the cached log1p axis only via name-keyed mapping.
            df = learner.plot_interaction_pdp(
                a, b, xscale='log1p', yscale='linear', return_data=True,
            )
            assert SENTINEL in df[b].to_numpy()  # reused despite swapped order
        finally:
            mr.evaluation.interaction_surfaces = original

    def test_scale_mismatch_log_vs_log1p_recomputes(self, learner_with_interactions):
        """A 'log' request must not reuse a cached 'log1p' surface (the scale
        strings differ); it recomputes a true log axis."""
        from mlsurv.evaluation._interactions import _compute_pdp_surface
        learner = learner_with_interactions
        mr = list(learner.models.values())[0]
        state = learner._state
        pair = next(
            s for s in mr.evaluation.interaction_surfaces
            if s['grid_1_type'] == 'numeric' and s['grid_2_type'] == 'numeric'
        )
        f1, f2 = pair['feature_1'], pair['feature_2']
        cat_features = (
            set(state.categorical_features)
            if state.categorical_features is not None else None
        )
        log1p_surface = _compute_pdp_surface(
            mr.model, learner.X_test, f1, f2,
            max_samples=20, random_state=state.random_state,
            categorical_groups=state.categorical_groups,
            categorical_features=cat_features,
            scale_1='linear', scale_2='log1p',
        )
        log1p_surface['model_type'] = mr.name
        original = mr.evaluation.interaction_surfaces
        try:
            mr.evaluation.interaction_surfaces = [log1p_surface]
            # Request 'log' on the x-axis whose cache is 'log1p' → recompute.
            fig = learner.plot_interaction_pdp(f1, f2, xscale='log', yscale='linear')
            # A true log axis sets type='log'; log1p keeps type != 'log'.
            assert fig.layout.xaxis.type == 'log'
        finally:
            mr.evaluation.interaction_surfaces = original

    def test_log_fallback_warns_on_cache_hit(self):
        """Cache-hit replay: 'log' on a negative-valued feature falls back to
        linear and still warns when the matching cached (linear) surface is
        reused (the fallback is not silently dropped on the cache-hit path)."""
        from mlsurv import SurvivalLearner
        from mlsurv._warnings import SuggestionWarning

        rng = np.random.RandomState(0)
        n = 80
        X = pd.DataFrame({
            'a': rng.normal(0.0, 1.0, n),     # has negatives → 'log' falls back
            'b': rng.uniform(1.0, 10.0, n),
            'c': rng.uniform(1.0, 10.0, n),
        })
        t = rng.uniform(1.0, 100.0, n)
        e = rng.rand(n) > 0.3
        y = np.array(list(zip(e, t)), dtype=[('event', bool), ('time', float)])
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=0, test_fraction=0.3)
        learner.train('rsf', n_estimators=20)
        learner.evaluate(verbose=False, n_importance_repeats=0,
                         include_interactions=True, interaction_top_k=2,
                         interaction_max_samples=30)
        mr = list(learner.models.values())[0]
        pair = next(
            (s for s in (mr.evaluation.interaction_surfaces or [])
             if 'a' in (s['feature_1'], s['feature_2'])),
            None,
        )
        assert pair is not None, "expected 'a' among the cached interaction pairs"
        f1, f2 = pair['feature_1'], pair['feature_2']
        # Request 'log' on the negative-valued 'a' axis; keep the other linear.
        kw = {'xscale': 'linear', 'yscale': 'linear'}
        kw['xscale' if f2 == 'a' else 'yscale'] = 'log'
        with pytest.warns(SuggestionWarning, match='negative'):
            fig = learner.plot_interaction_pdp(f1, f2, **kw)
        assert isinstance(fig, go.Figure)


def _make_pdp_fig(g1, g2, s, f1, f2, **kwargs):
    """Helper to call the viz function."""
    from mlsurv.viz.interactions import plot_interaction_pdp
    return plot_interaction_pdp(g1, g2, s, f1, f2, **kwargs)


# ── SHAP interaction integration (requires XGBoost + shap) ──────────────

@pytest.mark.integration
class TestShapInteractionIntegration:

    @pytest.fixture
    def learner_with_xgbcox(self):
        """Create a quick learner with XGBCox trained."""
        from sksurv.datasets import load_gbsg2
        from mlsurv import SurvivalLearner

        X_raw, y_raw = load_gbsg2()
        X = X_raw.copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].cat.codes.astype(float)
        y = np.array(
            [(bool(e), t) for e, t in y_raw],
            dtype=[("event", bool), ("time", float)],
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('xgbcox')
        learner.evaluate(verbose=False, n_importance_repeats=0)
        return learner

    def test_shap_interactions_returns_dataframe(self, learner_with_xgbcox):
        """interactions(method='shap') returns DataFrame with correct columns."""
        result = learner_with_xgbcox.interactions(
            method='shap', top_k=3, max_samples=20,
        )
        assert isinstance(result, pd.DataFrame)
        assert 'shap_interaction' in result.columns
        assert 'method' in result.columns
        assert 'model_type' in result.columns
        assert 'feature_1' in result.columns
        assert 'feature_2' in result.columns
        assert (result['method'] == 'shap').all()
        assert len(result) == 3  # C(3,2) = 3 pairs

    def test_shap_interactions_stores_raw_matrix(self, learner_with_xgbcox):
        """Raw interaction matrix is stored on evaluation result."""
        learner_with_xgbcox.interactions(
            method='shap', top_k=3, max_samples=20,
        )
        mr = learner_with_xgbcox.models['xgbcox']
        assert mr.evaluation.interaction_values_raw is not None
        assert mr.evaluation.interaction_values_raw.ndim == 3
        assert mr.evaluation.interaction_values_raw.shape[1] == 3
        assert mr.evaluation.interaction_values_raw.shape[2] == 3
        assert mr.evaluation.interaction_method == 'shap'
        assert mr.evaluation.interaction_feature_names is not None
        assert len(mr.evaluation.interaction_feature_names) == 3

    def test_shap_interactions_accepts_rsf(self):
        """RSF model with method='shap' succeeds."""
        from sksurv.datasets import load_gbsg2
        from mlsurv import SurvivalLearner

        X_raw, y_raw = load_gbsg2()
        X = X_raw.copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].cat.codes.astype(float)
        y = np.array(
            [(bool(e), t) for e, t in y_raw],
            dtype=[("event", bool), ("time", float)],
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('rsf', n_estimators=50)
        learner.evaluate(verbose=False, n_importance_repeats=0)
        result = learner.interactions(method='shap', top_k=3, max_samples=20)
        assert result is not None
        assert len(result) > 0
        assert 'shap_interaction' in result.columns

    def test_shap_interactions_rejects_coxph(self):
        """CoxPH model with method='shap' raises TypeError."""
        from sksurv.datasets import load_gbsg2
        from mlsurv import SurvivalLearner

        X_raw, y_raw = load_gbsg2()
        X = X_raw.copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].cat.codes.astype(float)
        y = np.array(
            [(bool(e), t) for e, t in y_raw],
            dtype=[("event", bool), ("time", float)],
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph')
        learner.evaluate(verbose=False, n_importance_repeats=0)
        with pytest.raises(TypeError, match="tree-based model"):
            learner.interactions(method='shap', top_k=3, max_samples=20)

    def test_evaluate_with_interaction_method_shap(self, learner_with_xgbcox):
        """evaluate(include_interactions=True, interaction_method='shap') works."""
        # Re-evaluate with SHAP interactions
        result = learner_with_xgbcox.evaluate(
            include_interactions=True,
            interaction_method='shap',
            interaction_top_k=3,
            interaction_max_samples=20,
            verbose=False,
            n_importance_repeats=0,
        )
        assert result.interactions is not None
        assert 'shap_interaction' in result.interactions.columns
        assert result.interaction_method == 'shap'
        # interaction_method propagates to per-model evaluation
        mr = learner_with_xgbcox.models['xgbcox']
        assert mr.evaluation.interaction_method == 'shap'

    def test_plot_interactions_after_shap(self, learner_with_xgbcox):
        """plot_interactions() works after SHAP interaction computation."""
        learner_with_xgbcox.interactions(
            method='shap', top_k=3, max_samples=20,
        )
        fig = learner_with_xgbcox.plot_interactions()
        assert fig is not None
        # Verify it's a Plotly figure
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)

    def test_plot_interactions_return_data_pivots_both_methods(
        self, learner_with_xgbcox,
    ):
        """Regression: return_data=True collapses H-stat + SHAP rows."""
        # First populate H-statistic rows
        learner_with_xgbcox.interactions(
            method='hstat', top_k=3, max_samples=20,
        )
        # Then add SHAP rows for the same model
        learner_with_xgbcox.interactions(
            method='shap', top_k=3, max_samples=20,
        )
        df = learner_with_xgbcox.plot_interactions(return_data=True)
        pairs = df[['feature_1', 'feature_2']].apply(tuple, axis=1)
        assert pairs.is_unique
        assert 'method' not in df.columns
        assert df['h_statistic'].notna().all()
        assert df['shap_interaction'].notna().all()

    def test_plot_interactions_return_data_single_method_drops_absent(
        self, learner_with_xgbcox,
    ):
        """When only one method exists, the other column is absent, not NaN-filled."""
        learner_with_xgbcox.interactions(
            method='shap', top_k=3, max_samples=20,
        )
        df = learner_with_xgbcox.plot_interactions(return_data=True)
        pairs = df[['feature_1', 'feature_2']].apply(tuple, axis=1)
        assert pairs.is_unique
        assert 'shap_interaction' in df.columns
        assert df['shap_interaction'].notna().all()
        assert 'h_statistic' not in df.columns


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestMultiModelShapRawMatrixPropagation:
    """Multi-model evaluate() preserves per-model SHAP raw matrices."""

    @pytest.fixture
    def learner_with_two_xgb(self):
        """Create a learner with two XGBoost models trained."""
        from sksurv.datasets import load_gbsg2
        from mlsurv import SurvivalLearner

        X_raw, y_raw = load_gbsg2()
        X = X_raw.copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].cat.codes.astype(float)
        y = np.array(
            [(bool(e), t) for e, t in y_raw],
            dtype=[("event", bool), ("time", float)],
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.run(['xgbcox', 'xgbaft'], tune=False,
                    model_params={'xgbcox': _SMALL_TREE_PARAMS,
                                  'xgbaft': _SMALL_TREE_PARAMS})
        return learner

    def test_evaluate_multi_model_shap_raw_preserved(self, learner_with_two_xgb):
        """Per-model interaction_values_raw is not None after multi-model evaluate()."""
        learner_with_two_xgb.evaluate(
            include_interactions=True,
            interaction_method='shap',
            interaction_top_k=3,
            interaction_max_samples=20,
            verbose=False,
            n_importance_repeats=0,
        )
        for name in ['xgbcox', 'xgbaft']:
            mr = learner_with_two_xgb.models[name]
            assert mr.evaluation.interaction_values_raw is not None, (
                f"{name}: interaction_values_raw should not be None"
            )
            assert mr.evaluation.interaction_values_raw.ndim == 3
            assert mr.evaluation.interaction_values_raw.shape[1] == 3
            assert mr.evaluation.interaction_values_raw.shape[2] == 3
            assert mr.evaluation.interaction_feature_names is not None
            assert len(mr.evaluation.interaction_feature_names) == 3
            assert mr.evaluation.interaction_method == 'shap'

    def test_standalone_interactions_multi_model_shap_raw(self, learner_with_two_xgb):
        """Per-model raw matrices via standalone interactions() method."""
        learner_with_two_xgb.evaluate(verbose=False, n_importance_repeats=0)
        for name in ['xgbcox', 'xgbaft']:
            learner_with_two_xgb.interactions(
                model=name, method='shap', top_k=3, max_samples=20,
            )
            mr = learner_with_two_xgb.models[name]
            assert mr.evaluation.interaction_values_raw is not None
            assert mr.evaluation._interaction_raw_cache is not None
            assert name in mr.evaluation._interaction_raw_cache


class TestPlotInteractionHeatmapMissingModelName:
    """Regression: None/NaN/pd.NA model_name must not render as
    literal 'None'/'nan'/'<NA>' in titles, dropdown buttons, or dropdown titles.
    """

    @staticmethod
    def _collect_text(fig):
        """Collect every user-visible text string from the figure for assertion.

        Covers the figure title, each dropdown button label, and each button's
        ``title.text`` update. Also includes layout annotations (empty-data
        fallback paths).
        """
        texts = [fig.layout.title.text or '']
        for ann in fig.layout.annotations or []:
            texts.append(ann.text or '')
        for menu in fig.layout.updatemenus or []:
            for btn in menu.buttons:
                texts.append(str(btn.label))
                if btn.args and len(btn.args) > 1:
                    title_text = btn.args[1].get('title.text')
                    if title_text:
                        texts.append(str(title_text))
        return texts

    _LEAK_PATTERN = re.compile(r'\b(None|nan|<NA>|NaT)\b', re.IGNORECASE)

    def _assert_no_missing_sentinels(self, fig):
        """Assert no missing-value sentinel leaked into any user-visible text."""
        for t in self._collect_text(fig):
            m = self._LEAK_PATTERN.search(t)
            assert m is None, (
                f"literal {m.group()!r} leaked into {t!r}"
            )

    def test_mixed_none_and_named_no_method_col(self):
        """DataFrame with None + named model_type and no method column — the
        exact path that triggers literal 'None' in dropdown labels and titles."""
        from mlsurv.viz.interactions import plot_interaction_heatmap
        df = pd.DataFrame({
            'model_type': [None, None, 'rsf', 'rsf'],
            'feature_1': ['a', 'a', 'a', 'b'],
            'feature_2': ['b', 'c', 'b', 'c'],
            'h_statistic': [0.3, 0.2, 0.4, 0.1],
        })
        fig = plot_interaction_heatmap(df)
        self._assert_no_missing_sentinels(fig)

    def test_mixed_nan_and_named_with_method_col(self):
        """DataFrame with NaN + named model_type leaks 'nan' via _method_label
        (np.nan is truthy, so the short-circuit in _method_label doesn't fire)."""
        from mlsurv.viz.interactions import plot_interaction_heatmap
        df = pd.DataFrame({
            'model_type': [np.nan, np.nan, 'rsf', 'rsf'],
            'feature_1': ['a', 'a', 'a', 'b'],
            'feature_2': ['b', 'c', 'b', 'c'],
            'h_statistic': [0.3, 0.2, 0.4, 0.1],
            'method': ['hstat'] * 4,
        })
        fig = plot_interaction_heatmap(df)
        self._assert_no_missing_sentinels(fig)

    def test_mixed_pd_na_and_named_string_dtype(self):
        """DataFrame with pd.NA (from nullable string dtype) + named model_type.
        pd.NA is not a float, so a naive isinstance(float) check misses it."""
        from mlsurv.viz.interactions import plot_interaction_heatmap
        df = pd.DataFrame({
            'model_type': pd.array([pd.NA, pd.NA, 'rsf', 'rsf'], dtype='string'),
            'feature_1': ['a', 'a', 'a', 'b'],
            'feature_2': ['b', 'c', 'b', 'c'],
            'h_statistic': [0.3, 0.2, 0.4, 0.1],
            'method': ['hstat'] * 4,
        })
        fig = plot_interaction_heatmap(df)
        self._assert_no_missing_sentinels(fig)

    def test_anonymous_with_both_methods(self):
        """Anonymous (no model_type) DataFrame with both shap and hstat methods
        exercises the ``len(methods_in_model) > 1`` branch with display_name=''.
        Covers the dropdown button-label fallback in the multi-trace path."""
        from mlsurv.viz.interactions import plot_interaction_heatmap
        df = pd.DataFrame({
            'feature_1': ['a', 'a', 'b', 'a', 'a', 'b'],
            'feature_2': ['b', 'c', 'c', 'b', 'c', 'c'],
            'h_statistic': [0.3, 0.2, 0.1, np.nan, np.nan, np.nan],
            'shap_interaction': [np.nan, np.nan, np.nan, 0.5, 0.4, 0.3],
            'method': ['hstat', 'hstat', 'hstat', 'shap', 'shap', 'shap'],
        })
        fig = plot_interaction_heatmap(df)
        self._assert_no_missing_sentinels(fig)
        # Multi-trace dropdown should be present with at least 2 buttons (one
        # per method), each labeled as the method (not None/empty).
        menus = list(fig.layout.updatemenus or [])
        assert menus, "expected a dropdown menu for anonymous multi-method df"
        labels = {str(b.label) for b in menus[0].buttons}
        assert 'SHAP' in labels and 'H\u00b2' in labels

    def test_single_model_no_model_type_column(self):
        """Sanity: a DataFrame without model_type produces a clean title."""
        from mlsurv.viz.interactions import plot_interaction_heatmap
        df = pd.DataFrame({
            'feature_1': ['a', 'a', 'b'],
            'feature_2': ['b', 'c', 'c'],
            'h_statistic': [0.3, 0.2, 0.1],
            'method': ['hstat'] * 3,
        })
        fig = plot_interaction_heatmap(df)
        self._assert_no_missing_sentinels(fig)

    def test_method_label_handles_all_missing_sentinels(self):
        """_method_label must short-circuit on None, NaN, pd.NA, and empty string."""
        from mlsurv.viz.interactions import _method_label
        # None is already handled (falsy)
        assert _method_label(None, 'hstat') == 'H\u00b2'
        # NaN is truthy in bool context but should still collapse to the suffix
        assert _method_label(np.nan, 'hstat') == 'H\u00b2'
        assert _method_label(np.nan, 'shap') == 'SHAP'
        # pd.NA is neither None nor a float — the fix must recognize it.
        assert _method_label(pd.NA, 'hstat') == 'H\u00b2'
        # Empty string short-circuits
        assert _method_label('', 'hstat') == 'H\u00b2'
        # Real name passes through
        assert _method_label('coxph', 'hstat') == 'coxph (H\u00b2)'


def _make_grouped_data(n=100):
    """Create a DataFrame with age, bmi, and one-hot stage columns."""
    rng = np.random.RandomState(42)
    age = rng.randn(n)
    bmi = rng.randn(n)
    stage = rng.choice(3, size=n)
    return pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'stage_I': (stage == 0).astype(float),
        'stage_II': (stage == 1).astype(float),
        'stage_III': (stage == 2).astype(float),
    })


_GROUPS = {'stage': ['stage_I', 'stage_II', 'stage_III']}


class _CategoricalInteractionModel:
    """predict = age * stage_II — stage interacts with age."""

    def predict(self, X):
        X = _to_array(X)
        return X[:, 0] * X[:, 3]  # age * stage_II


class _CategoricalAdditiveModel:
    """predict = age + stage_II — no interaction."""

    def predict(self, X):
        X = _to_array(X)
        return X[:, 0] + X[:, 3]


class TestBuildInteractionUnits:

    def test_numeric_only(self):
        cols = ['age', 'bmi', 'x0']
        units = _build_interaction_units(cols, ['age', 'bmi'], None)
        assert len(units) == 2
        assert units[0] == FeatureUnit('age', [0], False)
        assert units[1] == FeatureUnit('bmi', [1], False)

    def test_group_only(self):
        cols = ['age', 'bmi', 'stage_I', 'stage_II', 'stage_III']
        units = _build_interaction_units(cols, ['stage'], _GROUPS)
        assert len(units) == 1
        assert units[0].name == 'stage'
        assert units[0].col_indices == [2, 3, 4]
        assert units[0].is_group is True

    def test_mixed(self):
        cols = ['age', 'bmi', 'stage_I', 'stage_II', 'stage_III']
        units = _build_interaction_units(cols, ['age', 'stage'], _GROUPS)
        assert len(units) == 2
        assert units[0].is_group is False  # age
        assert units[1].is_group is True   # stage

    def test_unknown_feature_raises(self):
        cols = ['age', 'bmi']
        with pytest.raises(ValueError, match="not found"):
            _build_interaction_units(cols, ['age', 'nonexistent'], None)


class TestBuildCategoricalGrid:

    def test_one_hot_grid(self):
        X = _make_grouped_data(n=100)
        X_values = X.values.astype(np.float64)
        grid = _build_categorical_grid(X_values, [2, 3, 4])
        # Should have 3 unique one-hot patterns
        assert grid.shape == (3, 3)
        # Each row should sum to 1 (proper one-hot)
        np.testing.assert_array_equal(grid.sum(axis=1), [1.0, 1.0, 1.0])

    def test_labels(self):
        grid = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        labels = _categorical_grid_labels(grid, ['stage_I', 'stage_II', 'stage_III'])
        assert labels == ['stage_I', 'stage_II', 'stage_III']

    def test_labels_ref_category(self):
        """All-zeros row (drop_first reference) should be labeled 'ref'."""
        grid = np.array([[0., 0.], [1., 0.], [0., 1.]])
        labels = _categorical_grid_labels(grid, ['stage_II', 'stage_III'])
        assert labels[0] == 'ref'


class TestBuildUnitGrid:

    def test_numeric_unit(self):
        X = _make_grouped_data(n=30)
        X_values = X.values.astype(np.float64)
        unit = FeatureUnit('age', [0], False)
        grid = _build_unit_grid(X_values, unit, 25)
        assert grid.ndim == 1

    def test_group_unit(self):
        X = _make_grouped_data(n=30)
        X_values = X.values.astype(np.float64)
        unit = FeatureUnit('stage', [2, 3, 4], True)
        grid = _build_unit_grid(X_values, unit, 25)
        assert grid.ndim == 2
        assert grid.shape[1] == 3


class TestPdp1dUnit:

    def test_categorical_unit(self):
        """Categorical unit PDP should set all dummy columns."""
        X = _make_grouped_data(n=30)
        X_values = X.values.astype(np.float64)
        model = _CategoricalInteractionModel()
        cols = X.columns.tolist()

        def predict_fn(arr):
            return model.predict(pd.DataFrame(arr, columns=cols))

        unit = FeatureUnit('stage', [2, 3, 4], True)
        grid = _build_categorical_grid(X_values, [2, 3, 4])
        pdp = _pdp_1d_unit(predict_fn, X_values, unit, grid)
        assert pdp.shape == (len(grid),)
        assert np.all(np.isfinite(pdp))


class TestPdp2dUnit:

    def test_categorical_x_numeric(self):
        X = _make_grouped_data(n=30)
        X_values = X.values.astype(np.float64)
        model = _CategoricalInteractionModel()
        cols = X.columns.tolist()

        def predict_fn(arr):
            return model.predict(pd.DataFrame(arr, columns=cols))

        u_cat = FeatureUnit('stage', [2, 3, 4], True)
        u_num = FeatureUnit('age', [0], False)
        g_cat = _build_categorical_grid(X_values, [2, 3, 4])
        g_num = np.unique(X_values[:, 0])
        result = _pdp_2d_unit(predict_fn, X_values, u_cat, u_num, g_cat, g_num)
        assert result.shape == (len(g_cat), len(g_num))
        assert np.all(np.isfinite(result))


class TestHStatPairUnits:

    def test_additive_model_low_h(self):
        """Additive model with categorical should have H-stat near 0."""
        X = _make_grouped_data(n=30)
        X_values = X.values.astype(np.float64)
        model = _CategoricalAdditiveModel()
        cols = X.columns.tolist()

        def predict_fn(arr):
            return model.predict(pd.DataFrame(arr, columns=cols))

        u1 = FeatureUnit('age', [0], False)
        u2 = FeatureUnit('stage', [2, 3, 4], True)
        g1 = np.unique(X_values[:, 0])
        g2 = _build_categorical_grid(X_values, [2, 3, 4])
        h = _h_statistic_pair_units(predict_fn, X_values, u1, u2, g1, g2)
        assert h < 0.05

    def test_interaction_model_high_h(self):
        """Model with stage-dependent effect should have high H."""
        X = _make_grouped_data(n=50)
        X_values = X.values.astype(np.float64)
        model = _CategoricalInteractionModel()
        cols = X.columns.tolist()

        def predict_fn(arr):
            return model.predict(pd.DataFrame(arr, columns=cols))

        u1 = FeatureUnit('age', [0], False)
        u2 = FeatureUnit('stage', [2, 3, 4], True)
        g1 = np.unique(X_values[:, 0])
        g2 = _build_categorical_grid(X_values, [2, 3, 4])
        h = _h_statistic_pair_units(predict_fn, X_values, u1, u2, g1, g2)
        assert h > 0.2


class TestSelectInteractionFeaturesGrouped:

    def test_group_names_from_importance(self):
        """When importance has group names, they should be selected."""
        imp = pd.DataFrame({
            'feature': ['age', 'stage', 'bmi'],
            'perm_mean': [0.3, 0.5, 0.1],
        })
        result = _select_interaction_features(
            imp, top_k=2, all_columns=pd.Index([]),
            categorical_groups=_GROUPS,
        )
        assert result == ['stage', 'age']

    def test_fallback_uses_units(self):
        """Fallback with groups should use unit names."""
        cols = pd.Index(['age', 'bmi', 'stage_I', 'stage_II', 'stage_III'])
        result = _select_interaction_features(
            None, top_k=3, all_columns=cols,
            categorical_groups=_GROUPS,
        )
        # Should include 'stage' group name, not individual dummies
        assert 'stage' in result
        assert 'stage_I' not in result


class TestComputeHStatisticGrouped:

    def test_grouped_h_stat(self):
        """_compute_h_statistic with categorical_groups works end-to-end."""
        X = _make_grouped_data(n=30)
        result = _compute_h_statistic(
            _CategoricalInteractionModel(), X, ['age', 'stage'],
            max_samples=30, random_state=42,
            categorical_groups=_GROUPS,
        )
        assert len(result) == 1
        assert result.iloc[0]['feature_1'] == 'age'
        assert result.iloc[0]['feature_2'] == 'stage'
        assert result.iloc[0]['h_statistic'] > 0.1

    def test_backward_compat_no_groups(self):
        """Without categorical_groups, results match the original behavior."""
        X = _make_data(n=30)
        r1 = _compute_h_statistic(
            _AdditiveModel(), X, ['x0', 'x1'],
            max_samples=30, random_state=42,
        )
        r2 = _compute_h_statistic(
            _AdditiveModel(), X, ['x0', 'x1'],
            max_samples=30, random_state=42,
            categorical_groups=None,
        )
        pd.testing.assert_frame_equal(r1, r2)

    def test_surface_metadata(self):
        """Surfaces have grid_*_type and grid_*_labels metadata."""
        X = _make_grouped_data(n=30)
        df, surfaces = _compute_h_statistic(
            _CategoricalInteractionModel(), X, ['age', 'stage'],
            max_samples=30, random_state=42,
            compute_surfaces=True,
            categorical_groups=_GROUPS,
        )
        assert len(surfaces) == 1
        s = surfaces[0]
        # age is numeric, stage is categorical
        assert s['grid_1_type'] == 'numeric'
        assert s['grid_1_labels'] is None
        assert s['grid_2_type'] == 'categorical'
        assert s['grid_2_labels'] is not None
        assert len(s['grid_2_labels']) == len(s['grid_2'])


class TestComputePdpSurfaceGrouped:

    def test_with_groups(self):
        """_compute_pdp_surface with categorical_groups returns correct metadata."""
        X = _make_grouped_data(n=30)
        result = _compute_pdp_surface(
            _CategoricalInteractionModel(), X, 'age', 'stage',
            max_samples=30, random_state=42,
            categorical_groups=_GROUPS,
        )
        assert result['grid_2_type'] == 'categorical'
        assert result['grid_2_labels'] is not None
        assert result['grid_1_type'] == 'numeric'


class TestAggregateSHAPGroups:

    def test_aggregation_shape(self):
        """Aggregated matrix shape matches number of units."""
        n_samples = 10
        n_all = 5  # age, bmi, stage_I, stage_II, stage_III
        raw = np.random.RandomState(42).randn(n_samples, n_all, n_all)
        all_names = ['age', 'bmi', 'stage_I', 'stage_II', 'stage_III']

        units = [
            FeatureUnit('age', [0], False),
            FeatureUnit('stage', [2, 3, 4], True),
        ]
        agg, names = _aggregate_shap_interaction_groups(raw, all_names, units)
        assert agg.shape == (n_samples, 2, 2)
        assert names == ['age', 'stage']

    def test_single_column_passthrough(self):
        """Single-column units should pass through unchanged."""
        n_samples = 5
        n_all = 3
        raw = np.random.RandomState(42).randn(n_samples, n_all, n_all)
        all_names = ['x0', 'x1', 'x2']

        units = [
            FeatureUnit('x0', [0], False),
            FeatureUnit('x1', [1], False),
        ]
        agg, names = _aggregate_shap_interaction_groups(raw, all_names, units)
        assert agg.shape == (n_samples, 2, 2)
        # For single-column units, values should equal the raw slice
        np.testing.assert_array_almost_equal(
            agg[:, 0, 1], raw[:, 0, 1],
        )

    def test_group_sums_submatrix(self):
        """Group aggregation should sum SHAP values across all dummy pairs."""
        n_samples = 3
        raw = np.ones((n_samples, 4, 4))
        all_names = ['age', 'stage_I', 'stage_II', 'stage_III']

        units = [
            FeatureUnit('age', [0], False),
            FeatureUnit('stage', [1, 2, 3], True),
        ]
        agg, _ = _aggregate_shap_interaction_groups(raw, all_names, units)
        # age x stage: sum of raw[s, 0, 1] + raw[s, 0, 2] + raw[s, 0, 3]
        # = 1 + 1 + 1 = 3
        np.testing.assert_array_almost_equal(agg[:, 0, 1], 3.0)
        # stage x age: sum of raw[s, 1, 0] + raw[s, 2, 0] + raw[s, 3, 0]
        np.testing.assert_array_almost_equal(agg[:, 1, 0], 3.0)


class TestValidateFeaturesGrouped:

    def test_group_name_accepted(self):
        """Group names should be accepted when categorical_groups is provided."""
        from mlsurv.evaluation._interactions import _validate_features
        X = _make_grouped_data(n=10)
        # Should not raise
        _validate_features(X, ['age', 'stage'], _GROUPS)

    def test_missing_group_member_raises(self):
        from mlsurv.evaluation._interactions import _validate_features
        X = _make_grouped_data(n=10).drop(columns=['stage_III'])
        with pytest.raises(ValueError, match="not in data"):
            _validate_features(X, ['stage'], _GROUPS)


class TestVizCategoricalDispatch:

    def test_heatmap_for_both_categorical_pair(self):
        """Categorical × categorical pair should produce a Heatmap trace."""
        from mlsurv.viz.interactions import plot_interaction_pdp
        fig = plot_interaction_pdp(
            grid_1=[[1, 0], [0, 1]],
            grid_2=[[1, 0], [0, 1]],
            pdp_surface=[[1.0, 1.1], [1.3, 1.5]],
            feature_1='stage',
            feature_2='grade',
            grid_1_type='categorical',
            grid_2_type='categorical',
            grid_1_labels=['stage_I', 'stage_II'],
            grid_2_labels=['grade_1', 'grade_2'],
        )
        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Heatmap)

    def test_lines_for_mixed_pair_rows_categorical(self):
        """Continuous × categorical (categorical on rows) renders one line per level."""
        from mlsurv.viz.interactions import plot_interaction_pdp
        fig = plot_interaction_pdp(
            grid_1=[0.0, 1.0],
            grid_2=[10.0, 20.0, 30.0],
            pdp_surface=[[1.0, 1.1, 1.2], [1.3, 1.5, 1.7]],
            feature_1='treat',
            feature_2='age',
            grid_1_type='categorical',
            grid_2_type='numeric',
            grid_1_labels=['untreated', 'treated'],
        )
        assert all(isinstance(t, go.Scatter) for t in fig.data)
        assert len(fig.data) == 2  # one line per category level
        # Continuous feature (age) is the x-axis; no interpolation across levels.
        assert list(fig.data[0].x) == [10.0, 20.0, 30.0]
        assert fig.layout.xaxis.title.text == 'age'
        assert fig.layout.yaxis.title.text == 'Predicted Risk'
        assert fig.layout.legend.title.text == 'treat'
        assert {t.name for t in fig.data} == {'untreated', 'treated'}

    def test_lines_for_mixed_pair_cols_categorical(self):
        """Continuous × categorical (categorical on cols) renders one line per level."""
        from mlsurv.viz.interactions import plot_interaction_pdp
        fig = plot_interaction_pdp(
            grid_1=[10.0, 20.0, 30.0],
            grid_2=[0.0, 1.0],
            pdp_surface=[[1.0, 1.1], [1.3, 1.5], [1.6, 2.0]],
            feature_1='age',
            feature_2='treat',
            grid_1_type='numeric',
            grid_2_type='categorical',
            grid_2_labels=['untreated', 'treated'],
        )
        assert all(isinstance(t, go.Scatter) for t in fig.data)
        assert len(fig.data) == 2
        assert list(fig.data[0].x) == [10.0, 20.0, 30.0]  # continuous f1 grid
        assert fig.layout.xaxis.title.text == 'age'
        assert fig.layout.legend.title.text == 'treat'

    def test_lines_log_scale_continuous_axis_rows_categorical(self):
        """Log scale applies to the continuous (x) axis of a mixed pair (cat on rows)."""
        from mlsurv.viz.interactions import plot_interaction_pdp
        fig = plot_interaction_pdp(
            grid_1=[0.0, 1.0],
            grid_2=[10.0, 20.0, 30.0],
            pdp_surface=[[1.0, 1.1, 1.2], [1.3, 1.5, 1.7]],
            feature_1='treat', feature_2='age',
            grid_1_type='categorical', grid_2_type='numeric',
            grid_1_labels=['0', '1'],
            grid_2_scale='log',  # continuous feature_2 is on x
        )
        assert fig.layout.xaxis.type == 'log'

    def test_lines_log_scale_continuous_axis_cols_categorical(self):
        """Log scale applies to the continuous (x) axis of a mixed pair (cat on cols)."""
        from mlsurv.viz.interactions import plot_interaction_pdp
        fig = plot_interaction_pdp(
            grid_1=[10.0, 20.0, 30.0],
            grid_2=[0.0, 1.0],
            pdp_surface=[[1.0, 1.1], [1.3, 1.5], [1.6, 2.0]],
            feature_1='age', feature_2='treat',
            grid_1_type='numeric', grid_2_type='categorical',
            grid_2_labels=['0', '1'],
            grid_1_scale='log',  # continuous feature_1 is on x
        )
        assert fig.layout.xaxis.type == 'log'

    def test_lines_label_length_mismatch_raises(self):
        """Mismatched category labels raise a clear ValueError (no silent mis-render)."""
        from mlsurv.viz.interactions import plot_interaction_pdp
        with pytest.raises(ValueError, match='grid_1_labels length'):
            plot_interaction_pdp(
                grid_1=[0.0, 1.0],
                grid_2=[10.0, 20.0, 30.0],
                pdp_surface=[[1.0, 1.1, 1.2], [1.3, 1.5, 1.7]],
                feature_1='treat', feature_2='age',
                grid_1_type='categorical', grid_2_type='numeric',
                grid_1_labels=['only_one'],  # 1 label for 2 levels
            )

    def test_lines_ragged_surface_raises(self):
        """A non-rectangular pdp_surface raises ValueError instead of mis-rendering."""
        from mlsurv.viz.interactions import plot_interaction_pdp
        with pytest.raises(ValueError, match='shape does not match'):
            plot_interaction_pdp(
                grid_1=[0.0, 1.0],
                grid_2=[10.0, 20.0, 30.0],
                pdp_surface=[[1.0, 1.1, 1.2], [1.3, 1.5]],  # ragged
                feature_1='treat', feature_2='age',
                grid_1_type='categorical', grid_2_type='numeric',
                grid_1_labels=['0', '1'],
            )

    def test_pdp_adaptive_margins_categorical(self):
        """plot_interaction_pdp computes adaptive margins for long categorical labels."""
        from mlsurv.viz.interactions import plot_interaction_pdp
        long_labels = [
            'Complete Response', 'Partial Response',
            'Stable Disease', 'Progressive Disease',
        ]
        fig = plot_interaction_pdp(
            grid_1=list(range(4)),
            grid_2=[0.0, 1.0],
            pdp_surface=[[1.0, 1.1]] * 4,
            feature_1='response',
            feature_2='grade',
            grid_1_type='categorical',
            grid_2_type='categorical',
            grid_1_labels=long_labels,
            grid_2_labels=['low', 'high'],
        )
        assert isinstance(fig.data[0], go.Heatmap)
        # "Progressive Disease" = 19 chars → 19*7+20 = 153, clamped to [80, 250]
        assert fig.layout.margin.l >= 80
        assert fig.layout.margin.b >= 70

    def test_pdp_adaptive_margins_x_axis_categorical(self):
        """Bottom margin adapts for long categorical x-axis labels."""
        from mlsurv.viz.interactions import plot_interaction_pdp
        x_labels = ['Very Long Treatment A', 'Very Long Treatment B']
        fig = plot_interaction_pdp(
            grid_1=[[1, 0], [0, 1]],
            grid_2=[[1, 0], [0, 1]],
            pdp_surface=[[1.0, 1.1], [1.2, 1.3]],
            feature_1='stage',
            feature_2='treatment',
            grid_1_type='categorical',
            grid_2_type='categorical',
            grid_1_labels=['s1', 's2'],
            grid_2_labels=x_labels,
        )
        # "Very Long Treatment A" = 22 chars → 22*5+10 = 120 > min_margin=70
        assert fig.layout.margin.b > 70

    def test_pdp_no_margin_override_for_numeric(self):
        """Numeric-only PDP should not set custom margins."""
        fig = _make_pdp_fig(
            [0.0, 1.0, 2.0], [0.0, 0.5, 1.0],
            [[1.0, 1.1, 1.2], [1.3, 1.5, 1.7], [1.6, 2.0, 2.4]],
            'feat_a', 'feat_b',
        )
        # Default template margins (not overridden)
        assert fig.layout.margin.l is None or fig.layout.margin.l <= 80

    def test_contour_for_numeric_pair(self):
        """Numeric pair should produce a Contour trace (backward compat)."""
        fig = _make_pdp_fig(
            [0.0, 1.0, 2.0], [0.0, 0.5, 1.0],
            [[1.0, 1.1, 1.2], [1.3, 1.5, 1.7], [1.6, 2.0, 2.4]],
            'feat_a', 'feat_b',
        )
        assert isinstance(fig.data[0], go.Contour)

    def test_gallery_mixed_types(self):
        """Gallery with mixed numeric/categorical surfaces should render."""
        from mlsurv.viz.interactions import plot_interaction_pdp_gallery
        surfaces = [
            {
                'feature_1': 'age', 'feature_2': 'bmi',
                'grid_1': [0.0, 1.0], 'grid_2': [0.0, 1.0],
                'pdp_surface': [[1.0, 1.1], [1.2, 1.5]],
                'model_type': 'rsf',
            },
            {
                'feature_1': 'age', 'feature_2': 'stage',
                'grid_1': [0.0, 1.0], 'grid_2': [[1, 0], [0, 1]],
                'pdp_surface': [[1.0, 1.1], [1.2, 1.5]],
                'grid_1_type': 'numeric', 'grid_2_type': 'categorical',
                'grid_1_labels': None,
                'grid_2_labels': ['stage_I', 'stage_II'],
                'model_type': 'rsf',
            },
        ]
        fig = plot_interaction_pdp_gallery(surfaces)
        assert isinstance(fig, go.Figure)
        # First surface (numeric×numeric) is one Contour trace; the second
        # (continuous×categorical) is a block of one line per level (2 levels).
        assert len(fig.data) == 3
        assert isinstance(fig.data[0], go.Contour)
        assert isinstance(fig.data[1], go.Scatter)
        assert isinstance(fig.data[2], go.Scatter)
        # Two dropdown buttons (one per surface), second toggles both lines.
        buttons = fig.layout.updatemenus[0].buttons
        assert len(buttons) == 2
        assert list(buttons[0].args[0]['visible']) == [True, False, False]
        assert list(buttons[1].args[0]['visible']) == [False, True, True]


class TestCategoricalScalarDetection:
    """_describe_unit_grid types low-cardinality single columns as categorical."""

    def _scalar(self):
        return FeatureUnit('treat', [0], False)

    def test_binary_scalar_is_categorical(self):
        from mlsurv.evaluation._interactions import _describe_unit_grid
        gl, gt, lab = _describe_unit_grid(
            self._scalar(), np.array([0.0, 1.0]), ['treat'],
        )
        assert gt == 'categorical'
        assert lab == ['0', '1']  # value labels, not "ref"/dummy names

    def test_boundary_five_is_categorical(self):
        from mlsurv.evaluation._interactions import _describe_unit_grid
        _, gt, _ = _describe_unit_grid(self._scalar(), np.arange(5.0), ['treat'])
        assert gt == 'categorical'

    def test_boundary_six_is_numeric(self):
        from mlsurv.evaluation._interactions import _describe_unit_grid
        _, gt, lab = _describe_unit_grid(self._scalar(), np.arange(6.0), ['treat'])
        assert gt == 'numeric'
        assert lab is None

    def test_explicit_list_forces_categorical(self):
        from mlsurv.evaluation._interactions import _describe_unit_grid
        # Many unique values, but named explicitly -> categorical.
        _, gt, _ = _describe_unit_grid(
            self._scalar(), np.arange(6.0), ['treat'],
            categorical_features={'treat'},
        )
        assert gt == 'categorical'

    def test_explicit_empty_forces_numeric(self):
        from mlsurv.evaluation._interactions import _describe_unit_grid
        # Empty set disables the heuristic -> even a binary stays numeric.
        _, gt, _ = _describe_unit_grid(
            self._scalar(), np.array([0.0, 1.0]), ['treat'],
            categorical_features=set(),
        )
        assert gt == 'numeric'

    def test_explicit_excludes_unlisted(self):
        from mlsurv.evaluation._interactions import _describe_unit_grid
        # In explicit mode, an unlisted binary stays numeric.
        _, gt, _ = _describe_unit_grid(
            self._scalar(), np.array([0.0, 1.0]), ['treat'],
            categorical_features={'other'},
        )
        assert gt == 'numeric'

    def test_group_always_categorical_ignores_explicit(self):
        from mlsurv.evaluation._interactions import _describe_unit_grid
        unit = FeatureUnit('stage', [0, 1], True)
        grid = np.array([[1.0, 0.0], [0.0, 1.0]])
        _, gt, lab = _describe_unit_grid(
            unit, grid, ['stage_I', 'stage_II'], categorical_features=set(),
        )
        assert gt == 'categorical'  # one-hot groups are unconditional
        assert lab is not None

    def test_compute_pdp_surface_honors_explicit_list(self):
        """End-to-end: _compute_pdp_surface marks a listed low-card column categorical."""
        from mlsurv.evaluation._interactions import _compute_pdp_surface

        rng = np.random.RandomState(0)
        X = pd.DataFrame({
            'age': rng.randn(40),
            'grade': rng.choice([0.0, 1.0, 2.0], size=40),  # 3 unique
        })

        class _M:
            def predict(self, arr):
                a = _to_array(arr)
                return a[:, 0] * a[:, 1]

        # Heuristic: grade has 3 unique -> categorical by default.
        s_cat = _compute_pdp_surface(_M(), X, 'age', 'grade', max_samples=40, random_state=0)
        assert s_cat['grid_2_type'] == 'categorical'

        # Explicit empty list -> heuristic off -> grade numeric.
        s_num = _compute_pdp_surface(
            _M(), X, 'age', 'grade', max_samples=40, random_state=0,
            categorical_features=set(),
        )
        assert s_num['grid_2_type'] == 'numeric'

        # Metadata-only invariant: typing a column categorical vs numeric must
        # not change the computed surface or grids (only grid_*_type/labels).
        assert s_cat['pdp_surface'] == s_num['pdp_surface']
        assert s_cat['grid_1'] == s_num['grid_1']
        assert s_cat['grid_2'] == s_num['grid_2']

    def test_compute_h_statistic_surface_honors_explicit_list(self):
        """The H-stat surface path (_build_pdp_surface_record) honors categorical_features."""
        from mlsurv.evaluation._interactions import _compute_h_statistic

        rng = np.random.RandomState(0)
        X = pd.DataFrame({
            'age': rng.randn(40),
            'grade': rng.choice([0.0, 1.0, 2.0], size=40),  # 3 unique
        })

        class _M:
            def predict(self, arr):
                a = _to_array(arr)
                return a[:, 0] * a[:, 1]

        # Explicit set names only 'grade' -> grade categorical, age numeric.
        _, surfaces = _compute_h_statistic(
            _M(), X, ['age', 'grade'], max_samples=40, random_state=0,
            compute_surfaces=True, categorical_features={'grade'},
        )
        s = surfaces[0]
        types = {s['feature_1']: s['grid_1_type'], s['feature_2']: s['grid_2_type']}
        assert types['grade'] == 'categorical'
        assert types['age'] == 'numeric'

    def test_fmt_level_non_integer_branch(self):
        """Non-integer categorical levels use compact %g formatting."""
        _, gt, lab = self._describe(np.array([0.0, 1.5, 3.0]))
        assert gt == 'categorical'
        assert lab == ['0', '1.5', '3']

    def test_non_finite_categorical_grid_warns(self):
        """A categorical-by-heuristic column with NaN levels warns (no silent 'nan' label)."""
        from mlsurv._warnings import SuggestionWarning
        with pytest.warns(SuggestionWarning, match='non-finite'):
            _, gt, lab = self._describe(np.array([0.0, 1.0, np.nan]))
        assert gt == 'categorical'
        assert 'nan' in lab

    def _describe(self, grid):
        from mlsurv.evaluation._interactions import _describe_unit_grid
        return _describe_unit_grid(self._scalar(), grid, ['treat'])


class TestValidateCategoricalFeaturesSetup:
    """Setup-time validation of the categorical_features parameter."""

    def _X(self):
        return pd.DataFrame({
            'age': [1.0, 2.0, 3.0],
            'treat': [0.0, 1.0, 0.0],
            'stage_I': [1.0, 0.0, 1.0],
            'stage_II': [0.0, 1.0, 0.0],
        })

    def test_none_is_ok(self):
        from mlsurv.learner._setup import _validate_categorical_features_setup
        _validate_categorical_features_setup(None, None, self._X())

    def test_empty_list_is_ok(self):
        from mlsurv.learner._setup import _validate_categorical_features_setup
        _validate_categorical_features_setup([], None, self._X())

    def test_valid_list_passes(self):
        from mlsurv.learner._setup import _validate_categorical_features_setup
        _validate_categorical_features_setup(['treat'], None, self._X())

    def test_non_list_raises(self):
        from mlsurv.learner._setup import _validate_categorical_features_setup
        with pytest.raises(TypeError, match='list of column names'):
            _validate_categorical_features_setup('treat', None, self._X())

    def test_non_string_entries_raise(self):
        from mlsurv.learner._setup import _validate_categorical_features_setup
        with pytest.raises(TypeError, match='only column-name strings'):
            _validate_categorical_features_setup([1, 2], None, self._X())

    def test_absent_column_raises(self):
        from mlsurv.learner._setup import _validate_categorical_features_setup
        with pytest.raises(ValueError, match='not found in X'):
            _validate_categorical_features_setup(['nope'], None, self._X())

    def test_overlap_with_group_member_raises(self):
        from mlsurv.learner._setup import _validate_categorical_features_setup
        with pytest.raises(ValueError, match='already declared'):
            _validate_categorical_features_setup(
                ['stage_I'], {'stage': ['stage_I', 'stage_II']}, self._X(),
            )

    def test_overlap_with_group_name_raises(self):
        from mlsurv.learner._setup import _validate_categorical_features_setup
        X = self._X().rename(columns={'treat': 'stage'})
        with pytest.raises(ValueError, match='already declared'):
            _validate_categorical_features_setup(
                ['stage'], {'stage': ['stage_I', 'stage_II']}, X,
            )


# ── Regression tests ──────────────────────────────────────────────────────


def _make_heatmap_pairs(feature_names, models=None):
    """Build a heatmap DataFrame from all pairwise combos of *feature_names*.

    If *models* is provided, rows are duplicated per model with a
    ``model_type`` column.
    """
    rows = []
    model_list = models or [None]
    for model in model_list:
        for i, f1 in enumerate(feature_names):
            for f2 in feature_names[i + 1:]:
                row = {'feature_1': f1, 'feature_2': f2, 'h_statistic': 0.1}
                if model is not None:
                    row['model_type'] = model
                rows.append(row)
    return pd.DataFrame(rows)


def _make_numeric_surface(f1='a', f2='b', **overrides):
    """Build a minimal numeric PDP surface dict."""
    surface = {
        'feature_1': f1, 'feature_2': f2,
        'grid_1': [0.0, 1.0], 'grid_2': [0.0, 1.0],
        'pdp_surface': [[1.0, 1.1], [1.2, 1.5]],
    }
    surface.update(overrides)
    return surface


class TestHeatmapMargins:
    """Heatmap labels should not be cut off with long feature names."""

    def test_single_heatmap_margin_adjusts_for_long_names(self):
        """Single-model heatmap should have adaptive left and bottom margins."""
        from mlsurv.viz.interactions import plot_interaction_heatmap
        names = [f"very_long_feature_name_{i:03d}" for i in range(5)]
        fig = plot_interaction_heatmap(_make_heatmap_pairs(names))
        assert fig.layout.margin.l > 60
        assert fig.layout.margin.b > 50

    def test_single_heatmap_labels_truncated(self):
        """Tick labels should be truncated; tickvals should hold full names."""
        from mlsurv.viz.interactions import plot_interaction_heatmap
        long_name = "a" * 50
        df = pd.DataFrame({
            'feature_1': [long_name],
            'feature_2': ['short'],
            'h_statistic': [0.5],
        })
        fig = plot_interaction_heatmap(df)
        x_ticktext = fig.layout.xaxis.ticktext
        assert x_ticktext is not None
        assert any("..." in str(t) for t in x_ticktext)
        # tickvals must hold full (untruncated) names for data mapping
        assert long_name in list(fig.layout.xaxis.tickvals)

    def test_multi_heatmap_margin_adjusts(self):
        """Multi-model heatmap dropdown should also have adaptive margins."""
        from mlsurv.viz.interactions import plot_interaction_heatmap
        names = [f"long_feature_name_{i:03d}" for i in range(4)]
        fig = plot_interaction_heatmap(
            _make_heatmap_pairs(names, models=['rsf', 'coxph']),
        )
        assert fig.layout.margin.l > 60
        assert fig.layout.margin.b > 50

    def test_multi_heatmap_dropdown_updates_ticktext(self):
        """Dropdown button args should include ticktext and tickvals."""
        from mlsurv.viz.interactions import plot_interaction_heatmap
        names = [f"very_long_feature_name_{i:03d}" for i in range(4)]
        fig = plot_interaction_heatmap(
            _make_heatmap_pairs(names, models=['rsf', 'coxph']),
        )
        layout_args = fig.layout.updatemenus[0].buttons[0].args[1]
        assert "xaxis.ticktext" in layout_args
        assert "yaxis.ticktext" in layout_args
        # tickvals must hold full names for correct data mapping
        assert "xaxis.tickvals" in layout_args
        assert "yaxis.tickvals" in layout_args
        for val in layout_args["xaxis.tickvals"]:
            assert "..." not in str(val), "tickvals must not be truncated"


class TestPdpGalleryAxisTypeSwitching:
    """PDP gallery dropdown must update axis type for categoricals."""

    def test_dropdown_args_contain_axis_type(self):
        """Each dropdown button sets axis types; a mixed surface switches to lines."""
        from mlsurv.viz.interactions import plot_interaction_pdp_gallery
        surfaces = [
            _make_numeric_surface('age', 'bmi', model_type='rsf'),
            {
                'feature_1': 'age', 'feature_2': 'stage',
                'grid_1': [0.0, 1.0], 'grid_2': [[1, 0], [0, 1]],
                'pdp_surface': [[1.0, 1.1], [1.2, 1.5]],
                'grid_1_type': 'numeric', 'grid_2_type': 'categorical',
                'grid_2_labels': ['stage_I', 'stage_II'],
                'model_type': 'rsf',
            },
        ]
        fig = plot_interaction_pdp_gallery(surfaces)
        btn0 = fig.layout.updatemenus[0].buttons[0]
        assert btn0.args[1]['xaxis.type'] == 'linear'
        assert btn0.args[1]['yaxis.type'] == 'linear'
        assert btn0.args[1]['showlegend'] is False
        # Mixed surface (age × stage) renders as lines: continuous feature on
        # x (linear), categorical becomes the legend.
        btn1 = fig.layout.updatemenus[0].buttons[1]
        assert btn1.args[1]['xaxis.type'] == 'linear'
        assert btn1.args[1]['yaxis.type'] == 'linear'
        assert btn1.args[1]['showlegend'] is True
        assert btn1.args[1]['xaxis.title.text'] == 'age'
        assert btn1.args[1]['legend.title.text'] == 'stage'

    def test_initial_axis_type_matches_first_surface(self):
        """Initial layout axis type should match the first surface."""
        from mlsurv.viz.interactions import plot_interaction_pdp_gallery
        surfaces = [
            {
                'feature_1': 'stage', 'feature_2': 'treatment',
                'grid_1': [[1, 0], [0, 1]], 'grid_2': [[1, 0], [0, 1]],
                'pdp_surface': [[1.0, 1.1], [1.2, 1.5]],
                'grid_1_type': 'categorical', 'grid_2_type': 'categorical',
                'grid_1_labels': ['I', 'II'], 'grid_2_labels': ['A', 'B'],
            },
            _make_numeric_surface('age', 'bmi'),
        ]
        fig = plot_interaction_pdp_gallery(surfaces)
        assert fig.layout.xaxis.type == 'category'
        assert fig.layout.yaxis.type == 'category'

    def test_mixed_surface_per_axis_type(self):
        """A continuous × categorical surface renders as a line overlay."""
        from mlsurv.viz.interactions import plot_interaction_pdp_gallery
        surfaces = [
            _make_numeric_surface('age', 'bmi'),
            {
                'feature_1': 'stage', 'feature_2': 'age',
                'grid_1': [[1, 0], [0, 1]], 'grid_2': [0.0, 1.0],
                'pdp_surface': [[1.0, 1.1], [1.2, 1.5]],
                'grid_1_type': 'categorical', 'grid_2_type': 'numeric',
                'grid_1_labels': ['I', 'II'],
            },
        ]
        fig = plot_interaction_pdp_gallery(surfaces)
        btn1 = fig.layout.updatemenus[0].buttons[1]
        # Both axes linear; the categorical (stage, on rows) becomes the legend,
        # the continuous feature (age) goes on the x-axis.
        assert btn1.args[1]['yaxis.type'] == 'linear'
        assert btn1.args[1]['xaxis.type'] == 'linear'
        assert btn1.args[1]['showlegend'] is True
        assert btn1.args[1]['xaxis.title.text'] == 'age'
        assert btn1.args[1]['legend.title.text'] == 'stage'

    def test_mixed_scale_gallery_sets_per_surface_axes(self):
        """log / log1p / linear surfaces each get the right axis layout + label."""
        from mlsurv.viz.interactions import plot_interaction_pdp_gallery
        surfaces = [
            _make_numeric_surface(
                'gx', 'gy', grid_1=[1.0, 10.0, 100.0], grid_2=[1.0, 10.0, 100.0],
                pdp_surface=[[1.0, 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]],
                grid_1_type='numeric', grid_2_type='numeric',
                grid_1_scale='log', grid_2_scale='log1p',
            ),
            _make_numeric_surface('lx', 'ly'),  # linear/linear
        ]
        fig = plot_interaction_pdp_gallery(surfaces)
        btn0 = fig.layout.updatemenus[0].buttons[0].args[1]
        # y = grid_1 (log), x = grid_2 (log1p)
        assert btn0['yaxis.type'] == 'log'
        assert btn0['xaxis.type'] == 'linear'
        assert btn0['xaxis.tickvals'] is not None
        assert 'log1p' in btn0['xaxis.title.text']
        assert 'log scale' in btn0['yaxis.title.text']
        # Switching to the linear surface must clear the log1p manual ticks.
        btn1 = fig.layout.updatemenus[0].buttons[1].args[1]
        assert btn1['xaxis.type'] == 'linear'
        assert btn1['xaxis.tickvals'] is None
        assert btn1['yaxis.tickvals'] is None


class TestPdpGalleryLabelTruncation:
    """PDP gallery dropdown labels should be truncated."""

    def test_dropdown_label_truncated(self):
        """Dropdown button labels should truncate long feature names."""
        from mlsurv.viz.interactions import plot_interaction_pdp_gallery
        surfaces = [
            _make_numeric_surface('very_long_feature_name_one',
                                  'very_long_feature_name_two'),
            _make_numeric_surface('short', 'tiny'),
        ]
        fig = plot_interaction_pdp_gallery(surfaces)
        btn0_label = fig.layout.updatemenus[0].buttons[0].label
        untruncated = "very_long_feature_name_one × very_long_feature_name_two"
        assert len(btn0_label) < len(untruncated)
        btn1_label = fig.layout.updatemenus[0].buttons[1].label
        assert "..." not in btn1_label

    def test_title_uses_full_names(self):
        """Title text in dropdown args should use full (untruncated) names."""
        from mlsurv.viz.interactions import plot_interaction_pdp_gallery
        long_name = "very_long_feature_name_alpha"
        surfaces = [
            _make_numeric_surface(long_name, 'beta'),
            _make_numeric_surface('gamma', 'delta'),
        ]
        fig = plot_interaction_pdp_gallery(surfaces)
        btn0_title = fig.layout.updatemenus[0].buttons[0].args[1]['title.text']
        assert long_name in btn0_title

    def test_axis_titles_use_full_names(self):
        """Axis titles in dropdown button args should use full feature names."""
        from mlsurv.viz.interactions import plot_interaction_pdp_gallery
        long_f1 = "feature_with_a_very_long_name_1"
        long_f2 = "feature_with_a_very_long_name_2"
        surfaces = [
            _make_numeric_surface(long_f1, long_f2),
            _make_numeric_surface('a', 'b'),
        ]
        fig = plot_interaction_pdp_gallery(surfaces)
        btn0_args = fig.layout.updatemenus[0].buttons[0].args[1]
        assert btn0_args['xaxis.title.text'] == long_f2
        assert btn0_args['yaxis.title.text'] == long_f1


# ── Integration: learner API with categorical_groups ──────────────────────


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestGroupedInteractionsIntegration:
    """End-to-end test with learner.setup(categorical_groups=...) + interactions()."""

    @pytest.fixture(scope="class")
    def learner_with_groups(self):
        from sksurv.datasets import load_gbsg2
        from mlsurv import SurvivalLearner

        X_raw, y_raw = load_gbsg2()
        X = X_raw.copy()
        groups = {}
        for col in X.select_dtypes(include=["category", "object"]).columns:
            dummies = pd.get_dummies(X[col], prefix=col, dtype=float)
            groups[col] = dummies.columns.tolist()
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
        y = np.array(
            [(bool(e), t) for e, t in y_raw],
            dtype=[("event", bool), ("time", float)],
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, categorical_groups=groups)
        learner.train('rsf', n_estimators=50)
        learner.evaluate(verbose=False, n_importance_repeats=0)
        return learner, groups

    def test_interactions_with_groups(self, learner_with_groups):
        """learner.interactions() includes group names when top_k is large enough."""
        learner, groups = learner_with_groups
        result = learner.interactions(
            top_k=8, max_samples=20, grid_resolution=10, show_progress=False,
        )
        all_features = set(result['feature_1']) | set(result['feature_2'])
        group_names_in_result = all_features & set(groups.keys())
        assert len(group_names_in_result) > 0, (
            f"Expected group names in result, got features: {sorted(all_features)}"
        )

    def test_surfaces_have_categorical_metadata(self, learner_with_groups):
        """Surfaces include grid_*_type metadata for categorical pairs."""
        learner, groups = learner_with_groups
        learner.interactions(
            top_k=8, max_samples=20, grid_resolution=10, show_progress=False,
        )
        mr = learner.models['rsf']
        surfaces = mr.evaluation.interaction_surfaces
        assert surfaces is not None
        cat_surfaces = [
            s for s in surfaces
            if s.get('grid_1_type') == 'categorical'
            or s.get('grid_2_type') == 'categorical'
        ]
        assert len(cat_surfaces) > 0, "Expected at least one categorical surface"
        for s in cat_surfaces:
            if s.get('grid_1_type') == 'categorical':
                assert s.get('grid_1_labels') is not None
            if s.get('grid_2_type') == 'categorical':
                assert s.get('grid_2_labels') is not None

    def test_plot_interaction_pdp_categorical(self, learner_with_groups):
        """plot_interaction_pdp renders cached categorical surfaces discretely.

        A continuous × group pair is a line-per-level overlay; a group × group
        pair is a heatmap. Neither interpolates a contour across levels.
        """
        from mlsurv.viz.interactions import _classify_pair
        learner, groups = learner_with_groups
        learner.interactions(
            top_k=8, max_samples=20, grid_resolution=10, show_progress=False,
        )
        mr = learner.models['rsf']
        surfaces = mr.evaluation.interaction_surfaces
        # Find a categorical surface
        cat_surface = None
        for s in surfaces:
            if s.get('grid_1_type') == 'categorical' or s.get('grid_2_type') == 'categorical':
                cat_surface = s
                break
        assert cat_surface is not None, "No categorical surface found"
        kind = _classify_pair(
            cat_surface.get('grid_1_type'), cat_surface.get('grid_2_type'),
        )
        fig = learner.plot_interaction_pdp(
            cat_surface['feature_1'], cat_surface['feature_2'],
        )
        assert isinstance(fig, go.Figure)
        # Never a contour across a categorical axis.
        assert not isinstance(fig.data[0], go.Contour)
        if kind == 'heatmap':
            assert isinstance(fig.data[0], go.Heatmap)
        else:  # mixed -> lines
            assert all(isinstance(t, go.Scatter) for t in fig.data)


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestCategoricalScalarIntegration:
    """End-to-end: binary single columns auto-detect as categorical (no groups)."""

    @pytest.fixture(scope="class")
    def learner(self):
        from sksurv.datasets import load_gbsg2
        from mlsurv import SurvivalLearner

        X_raw, y_raw = load_gbsg2()
        X = X_raw.copy()
        # Integer-code categoricals -> single columns (horTh/menostat become 0/1).
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].cat.codes.astype(float)
        y = np.array(
            [(bool(e), t) for e, t in y_raw],
            dtype=[("event", bool), ("time", float)],
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('rsf', n_estimators=50)
        learner.evaluate(verbose=False, n_importance_repeats=0)
        return learner

    def test_binary_surface_typed_categorical(self, learner):
        """A binary single column (horTh/menostat) is typed categorical in surfaces."""
        learner.interactions(
            top_k=8, max_samples=20, grid_resolution=10, show_progress=False,
        )
        surfaces = learner.models['rsf'].evaluation.interaction_surfaces
        typed_binary = False
        for s in surfaces:
            for ax in ('1', '2'):
                if (s[f'feature_{ax}'] in ('horTh', 'menostat')
                        and s[f'grid_{ax}_type'] == 'categorical'):
                    typed_binary = True
                    assert s[f'grid_{ax}_labels'] is not None
        assert typed_binary, "expected a binary feature typed categorical"

    def test_mixed_pair_renders_lines_not_contour(self, learner):
        """A binary × continuous pair renders as a line overlay, not a contour."""
        from mlsurv.viz.interactions import _classify_pair
        learner.interactions(
            top_k=8, max_samples=20, grid_resolution=10, show_progress=False,
        )
        surfaces = learner.models['rsf'].evaluation.interaction_surfaces
        mixed = next(
            (s for s in surfaces
             if _classify_pair(s.get('grid_1_type'), s.get('grid_2_type')) == 'lines'),
            None,
        )
        assert mixed is not None, "expected at least one continuous×categorical pair"
        fig = learner.plot_interaction_pdp(mixed['feature_1'], mixed['feature_2'])
        assert all(isinstance(t, go.Scatter) for t in fig.data)
        assert len(fig.data) >= 2

    def test_override_forces_continuous_contour(self):
        """categorical_features=[] disables the heuristic; binary × cont -> contour."""
        from sksurv.datasets import load_gbsg2
        from mlsurv import SurvivalLearner

        X_raw, y_raw = load_gbsg2()
        X = X_raw.copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].cat.codes.astype(float)
        y = np.array(
            [(bool(e), t) for e, t in y_raw],
            dtype=[("event", bool), ("time", float)],
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, categorical_features=[])
        learner.train('rsf', n_estimators=50)
        learner.evaluate(verbose=False, n_importance_repeats=0)
        learner.interactions(
            top_k=8, max_samples=20, grid_resolution=10, show_progress=False,
        )
        surfaces = learner.models['rsf'].evaluation.interaction_surfaces
        # With the heuristic off, no single-column surface is typed categorical.
        for s in surfaces:
            assert s['grid_1_type'] == 'numeric'
            assert s['grid_2_type'] == 'numeric'


# ── Unit tests: _patch_sksurv_for_shap ────────────────────────────────────


class TestPatchSksurvForShap:
    """Unit tests for the unified SHAP patching context manager."""

    def test_gbsa_patches_class_and_init(self):
        """GBSA path swaps __class__ and init_ during context."""
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import GradientBoostingRegressor

        class FakeGBSA:
            init_ = "zero"

        est = FakeGBSA()
        with _patch_sksurv_for_shap(est, 'gbsa') as patched:
            assert patched is est
            assert est.__class__ is GradientBoostingRegressor
            assert isinstance(est.init_, DummyRegressor)
            assert est.init_.constant_[0] == 0.0

        assert est.__class__ is FakeGBSA
        assert est.init_ == "zero"

    def test_rsf_patches_class_and_criterion(self):
        """RSF path swaps __class__ and sets criterion during context."""
        from sklearn.ensemble import RandomForestRegressor

        class FakeRSF:
            criterion = "log_rank"

        est = FakeRSF()
        with _patch_sksurv_for_shap(est, 'rsf') as patched:
            assert patched is est
            assert est.__class__ is RandomForestRegressor
            assert est.criterion == "squared_error"

        assert est.__class__ is FakeRSF
        assert est.criterion == "log_rank"

    def test_rsf_restores_when_criterion_absent(self):
        """RSF path removes criterion on exit if it didn't exist before."""
        class FakeRSFNoCriterion:
            pass

        est = FakeRSFNoCriterion()
        assert not hasattr(est, 'criterion')

        with _patch_sksurv_for_shap(est, 'rsf'):
            assert est.criterion == "squared_error"

        assert not hasattr(est, 'criterion')

    def test_gbsa_restores_on_error(self):
        """GBSA path restores attributes even if body raises."""
        class FakeGBSA:
            init_ = "zero"

        est = FakeGBSA()
        original_class = est.__class__

        with pytest.raises(RuntimeError, match="boom"):
            with _patch_sksurv_for_shap(est, 'gbsa'):
                raise RuntimeError("boom")

        assert est.__class__ is original_class
        assert est.init_ == "zero"

    def test_rsf_restores_on_error(self):
        """RSF path restores attributes even if body raises."""
        class FakeRSF:
            criterion = "log_rank"

        est = FakeRSF()
        original_class = est.__class__

        with pytest.raises(RuntimeError, match="boom"):
            with _patch_sksurv_for_shap(est, 'rsf'):
                raise RuntimeError("boom")

        assert est.__class__ is original_class
        assert est.criterion == "log_rank"

    def test_rsf_restores_absent_criterion_on_error(self):
        """RSF path removes criterion on error exit if absent before."""
        class FakeRSFNoCriterion:
            pass

        est = FakeRSFNoCriterion()

        with pytest.raises(RuntimeError, match="boom"):
            with _patch_sksurv_for_shap(est, 'rsf'):
                assert est.criterion == "squared_error"
                raise RuntimeError("boom")

        assert not hasattr(est, 'criterion')

    def test_invalid_model_type_raises(self):
        """Unsupported model_type raises ValueError immediately."""
        est = object()
        with pytest.raises(ValueError, match="Unsupported model_type"):
            with _patch_sksurv_for_shap(est, 'xgboost'):
                pass

    def test_none_model_type_raises(self):
        """None model_type raises ValueError immediately."""
        est = object()
        with pytest.raises(ValueError, match="Unsupported model_type"):
            with _patch_sksurv_for_shap(est, None):
                pass


# ── GBSA SHAP interactions ──────────────────────────────────────────────


def _train_tree_shap_learner(model_name):
    """Train a tree-based model on GBSG2 for SHAP interaction tests."""
    from sksurv.datasets import load_gbsg2
    from mlsurv import SurvivalLearner

    X_raw, y_raw = load_gbsg2()
    X = X_raw.copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].cat.codes.astype(float)
    y = np.array(
        [(bool(e), t) for e, t in y_raw],
        dtype=[("event", bool), ("time", float)],
    )
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, test_fraction=0.3)
    learner.train(model_name, n_estimators=50)
    learner.evaluate(verbose=False, n_importance_repeats=0)
    return learner


@pytest.mark.integration
@pytest.mark.requires_shap
@pytest.mark.timeout(120)
class TestShapInteractions:
    """SHAP interaction values for tree-based models (GBSA, RSF).

    The structurally identical shape/contract tests are parametrized over both
    models; model-specific cleanup/extraction/array-shape checks are kept
    separate below.
    """

    @pytest.fixture(scope="class")
    def gbsa_learner(self):
        return _train_tree_shap_learner('gbsa')

    @pytest.fixture(scope="class")
    def rsf_learner(self):
        return _train_tree_shap_learner('rsf')

    @pytest.mark.parametrize("learner_fixture", ["gbsa_learner", "rsf_learner"])
    def test_shap_interactions_produce_results(self, request, learner_fixture):
        """Tree model produces a SHAP interaction DataFrame."""
        learner = request.getfixturevalue(learner_fixture)
        result = learner.interactions(method='shap', top_k=4)
        assert result is not None
        assert len(result) > 0
        assert 'shap_interaction' in result.columns
        assert 'method' in result.columns
        assert (result['method'] == 'shap').all()

    @pytest.mark.parametrize("learner_fixture", ["gbsa_learner", "rsf_learner"])
    def test_shap_interactions_correct_pair_count(self, request, learner_fixture):
        """Number of pairs matches C(top_k, 2)."""
        learner = request.getfixturevalue(learner_fixture)
        top_k = 4
        result = learner.interactions(method='shap', top_k=top_k)
        from math import comb
        assert len(result) == comb(top_k, 2)

    @pytest.mark.parametrize("learner_fixture", ["gbsa_learner", "rsf_learner"])
    def test_shap_interactions_sorted_descending(self, request, learner_fixture):
        """Results are sorted by shap_interaction descending.

        Assert the tolerance-bucketed key (rounded to 1e-12) is descending,
        matching the deterministic name tie-break the SHAP path applies to
        near-zero pairs.
        """
        learner = request.getfixturevalue(learner_fixture)
        result = learner.interactions(method='shap', top_k=4)
        keys = result['shap_interaction'].round(12).tolist()
        assert keys == sorted(keys, reverse=True)

    @pytest.mark.parametrize("learner_fixture", ["gbsa_learner", "rsf_learner"])
    def test_shap_interactions_nonnegative(self, request, learner_fixture):
        """Mean absolute SHAP interaction values are non-negative."""
        learner = request.getfixturevalue(learner_fixture)
        result = learner.interactions(method='shap', top_k=4)
        assert (result['shap_interaction'] >= 0).all()

    @pytest.mark.parametrize("learner_fixture", ["gbsa_learner", "rsf_learner"])
    def test_evaluate_with_shap_interactions(self, request, learner_fixture):
        """evaluate() with method='shap' populates interaction fields."""
        learner = request.getfixturevalue(learner_fixture)
        result = learner.evaluate(
            include_interactions=True,
            interaction_method='shap',
            interaction_top_k=3,
            verbose=False,
            n_importance_repeats=0,
        )
        assert result.interactions is not None
        assert len(result.interactions) > 0
        assert (result.interactions['method'] == 'shap').all()

    @pytest.mark.parametrize("learner_fixture", ["gbsa_learner", "rsf_learner"])
    def test_no_warning_for_tree_shap(self, request, learner_fixture):
        """Tree models should not trigger the 'requires tree-based model' warning."""
        import warnings as _w
        from mlsurv._warnings import SuggestionWarning

        learner = request.getfixturevalue(learner_fixture)
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            learner.interactions(method='shap', top_k=3)

        shap_skip_warnings = [
            w for w in caught
            if issubclass(w.category, SuggestionWarning)
            and 'tree-based model' in str(w.message)
        ]
        assert len(shap_skip_warnings) == 0

    # ── Model-specific checks (distinct assertions, kept separate) ──────────

    def test_patch_cleanup_gbsa(self, gbsa_learner):
        """GBSA estimator is restored after SHAP computation."""
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis

        pipeline = gbsa_learner._state.models['gbsa'].model
        wrapper = pipeline.steps[-1][1]
        inner = wrapper.model_

        # Verify it's GBSA before
        assert isinstance(inner, GradientBoostingSurvivalAnalysis)
        original_init = inner.init_

        # Run SHAP interactions
        gbsa_learner.interactions(method='shap', top_k=3)

        # Verify restoration
        assert isinstance(inner, GradientBoostingSurvivalAnalysis)
        assert inner.init_ is original_init

    def test_extract_tree_estimator_gbsa(self, gbsa_learner):
        """_extract_tree_estimator returns GBSA estimator with 'gbsa' type."""
        from mlsurv.evaluation._interactions import _extract_tree_estimator

        pipeline = gbsa_learner._state.models['gbsa'].model
        est, mtype = _extract_tree_estimator(pipeline)
        assert est is not None
        assert mtype == 'gbsa'

    def test_patch_cleanup_rsf(self, rsf_learner):
        """RSF estimator is restored after SHAP computation."""
        from sksurv.ensemble import RandomSurvivalForest

        pipeline = rsf_learner._state.models['rsf'].model
        wrapper = pipeline.steps[-1][1]
        inner = wrapper.model_

        # Verify it's RSF before
        assert isinstance(inner, RandomSurvivalForest)
        had_criterion = hasattr(inner, 'criterion')

        # Run SHAP interactions
        rsf_learner.interactions(method='shap', top_k=3)

        # Verify restoration
        assert isinstance(inner, RandomSurvivalForest)
        assert hasattr(inner, 'criterion') == had_criterion

    def test_extract_tree_estimator_rsf(self, rsf_learner):
        """_extract_tree_estimator returns RSF estimator with 'rsf' type."""
        from mlsurv.evaluation._interactions import _extract_tree_estimator

        pipeline = rsf_learner._state.models['rsf'].model
        est, mtype = _extract_tree_estimator(pipeline)
        assert est is not None
        assert mtype == 'rsf'

    def test_raw_array_is_3d(self, rsf_learner):
        """Raw interaction array is 3D after 4D→3D aggregation."""
        rsf_learner.interactions(method='shap', top_k=4)
        mr = rsf_learner._state.models['rsf']
        raw = mr.evaluation.interaction_values_raw
        assert raw is not None
        assert raw.ndim == 3
        assert raw.shape[1] == 4  # top_k features
        assert raw.shape[2] == 4
        assert mr.evaluation.interaction_method == 'shap'
        assert mr.evaluation.interaction_feature_names is not None
        assert len(mr.evaluation.interaction_feature_names) == 4
