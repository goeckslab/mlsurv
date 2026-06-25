"""Tests for per-model preprocessing pipeline support."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from mlsurv.learner import SurvivalLearner
from mlsurv.learner._setup import _validate_preprocessing, _resolve_per_model_preprocessing
from mlsurv.preprocessing.selection import TopKCoxSelector
from mlsurv.utils import SurvivalArray, unwrap_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_data():
    """Small clean dataset for fast tests."""
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        'age': np.random.uniform(30, 80, n),
        'stage': np.random.randint(1, 5, n),
        'grade': np.random.uniform(1, 4, n),
        'tumor_size': np.random.uniform(0.5, 10, n),
        'biomarker': np.random.randn(n),
    })
    times = np.random.exponential(2, n).clip(0.1, 10)
    events = np.random.binomial(1, 0.6, n).astype(bool)
    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)],
    ))
    return X, y


@pytest.fixture
def learner(small_data):
    """Configured SurvivalLearner ready for training."""
    X, y = small_data
    sl = SurvivalLearner(X, y)
    sl.setup(random_state=42, verbose=False)
    return sl


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    """Test _validate_preprocessing input validation."""

    def test_rejects_non_dict(self):
        with pytest.raises(TypeError, match="must be a dict"):
            _validate_preprocessing("topk")

    def test_rejects_non_string_model_keys(self):
        with pytest.raises(TypeError, match="must be strings"):
            _validate_preprocessing({123: {'scaler': 'robust'}})

    def test_rejects_non_dict_config_value(self):
        with pytest.raises(TypeError, match="must be a dict of component"):
            _validate_preprocessing({'rsf': 'robust'})

    def test_rejects_unknown_component_keys(self):
        with pytest.raises(ValueError, match="unknown keys"):
            _validate_preprocessing({'rsf': {'model': 'coxph'}})

    def test_accepts_valid_structure(self):
        _validate_preprocessing({
            'coxph': {'feature_selector': 'topk', 'scaler': 'robust'},
            'rsf': {'imputer': None},
        })

    def test_accepts_none(self):
        _validate_preprocessing(None)

    def test_accepts_empty_dict(self):
        _validate_preprocessing({})


# ---------------------------------------------------------------------------
# Resolution tests
# ---------------------------------------------------------------------------

class TestResolution:
    """Test _resolve_per_model_preprocessing resolution."""

    def test_none_returns_empty(self):
        assert _resolve_per_model_preprocessing(None) == {}

    def test_empty_dict_returns_empty(self):
        assert _resolve_per_model_preprocessing({}) == {}

    def test_resolves_string_shortcuts(self):
        result = _resolve_per_model_preprocessing({
            'coxph': {'scaler': 'robust'},
        })
        assert 'coxph' in result
        assert isinstance(result['coxph']['scaler'], RobustScaler)

    def test_resolves_none_value(self):
        result = _resolve_per_model_preprocessing({
            'rsf': {'scaler': None},
        })
        assert result['rsf']['scaler'] is None

    def test_resolves_transformer_instance(self):
        scaler = MinMaxScaler()
        result = _resolve_per_model_preprocessing({
            'rsf': {'scaler': scaler},
        })
        assert isinstance(result['rsf']['scaler'], MinMaxScaler)

    def test_resolves_feature_selector_string(self):
        result = _resolve_per_model_preprocessing({
            'coxph': {'feature_selector': 'topk'},
        })
        assert isinstance(result['coxph']['feature_selector'], TopKCoxSelector)


class TestModelPreprocessingResolution:
    """Test _resolve_model_preprocessing on SurvivalLearner."""

    def test_global_default_when_no_overrides(self, learner):
        """Without per-model config, returns global defaults."""
        imp, scl, sel = learner._resolve_model_preprocessing('rsf')
        assert imp is learner._state.imputer
        assert scl is learner._state.scaler
        assert sel is learner._state.feature_selector

    def test_setup_override_applied(self, small_data):
        X, y = small_data
        sl = SurvivalLearner(X, y)
        sl.setup(
            random_state=42, verbose=False,
            preprocessing={'rsf': {'scaler': 'robust'}},
        )
        imp, scl, sel = sl._resolve_model_preprocessing('rsf')
        assert isinstance(scl, RobustScaler)
        # Other components unchanged
        assert imp is sl._state.imputer

    def test_run_override_beats_setup(self, small_data):
        X, y = small_data
        sl = SurvivalLearner(X, y)
        sl.setup(
            random_state=42, verbose=False,
            preprocessing={'rsf': {'scaler': 'robust'}},
        )
        run_preproc = _resolve_per_model_preprocessing({
            'rsf': {'scaler': 'minmax'},
        })
        imp, scl, sel = sl._resolve_model_preprocessing('rsf', run_preproc)
        assert isinstance(scl, MinMaxScaler)

    def test_partial_override(self, small_data):
        X, y = small_data
        sl = SurvivalLearner(X, y)
        sl.setup(
            random_state=42, verbose=False,
            preprocessing={'coxph': {'scaler': 'robust'}},
        )
        # coxph gets robust scaler, but imputer and selector are global defaults
        imp, scl, sel = sl._resolve_model_preprocessing('coxph')
        assert isinstance(scl, RobustScaler)
        assert imp is sl._state.imputer
        assert sel is sl._state.feature_selector

    def test_none_disables_component(self, small_data):
        X, y = small_data
        sl = SurvivalLearner(X, y)
        sl.setup(
            random_state=42, verbose=False,
            preprocessing={'rsf': {'scaler': None}},
        )
        imp, scl, sel = sl._resolve_model_preprocessing('rsf')
        assert scl is None

    def test_model_not_in_preprocessing(self, small_data):
        """Models not mentioned use global defaults."""
        X, y = small_data
        sl = SurvivalLearner(X, y)
        sl.setup(
            random_state=42, verbose=False,
            preprocessing={'coxph': {'scaler': 'robust'}},
        )
        # rsf not in preprocessing → uses global defaults
        imp, scl, sel = sl._resolve_model_preprocessing('rsf')
        assert scl is sl._state.scaler


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """Test per-model preprocessing in actual training workflows."""

    def test_different_pipelines_per_model(self, small_data):
        """coxph and rsf get different preprocessing."""
        X, y = small_data
        sl = SurvivalLearner(X, y)
        sl.setup(
            random_state=42, verbose=False,
            preprocessing={
                'coxph': {'scaler': 'robust'},
                'rsf': {'scaler': 'minmax'},
            },
        )
        results = sl.train(['coxph', 'rsf'], verbose=False)

        coxph_scaler = results['coxph'].model.named_steps['scaler']
        rsf_scaler = results['rsf'].model.named_steps['scaler']
        assert isinstance(coxph_scaler, RobustScaler)
        assert isinstance(rsf_scaler, MinMaxScaler)

    def test_none_disables_scaler_in_pipeline(self, small_data):
        """Setting scaler=None for a model removes the scaler step."""
        X, y = small_data
        sl = SurvivalLearner(X, y)
        sl.setup(
            random_state=42, verbose=False,
            preprocessing={'rsf': {'scaler': None}},
        )
        results = sl.train(['coxph', 'rsf'], verbose=False)

        assert 'scaler' in results['coxph'].model.named_steps
        assert 'scaler' not in results['rsf'].model.named_steps

    def test_run_override_in_run(self, small_data):
        """run() preprocessing param overrides setup()."""
        X, y = small_data
        sl = SurvivalLearner(X, y)
        sl.setup(
            random_state=42, verbose=False,
            preprocessing={'coxph': {'scaler': 'robust'}},
        )
        results = sl.run(
            ['coxph'], tune=False, evaluate=False, bootstrap=False, verbose=False,
            preprocessing={'coxph': {'scaler': 'minmax'}},
        )
        coxph_scaler = results['coxph'].model.named_steps['scaler']
        assert isinstance(coxph_scaler, MinMaxScaler)

    def test_setup_preprocessing_persists_across_trains(self, small_data):
        """Per-model preprocessing from setup() works across multiple train() calls."""
        X, y = small_data
        sl = SurvivalLearner(X, y)
        sl.setup(
            random_state=42, verbose=False,
            preprocessing={'coxph': {'scaler': 'robust'}},
        )
        result1 = sl.train('coxph', verbose=False)
        result2 = sl.train('coxph', verbose=False)

        assert isinstance(result1.model.named_steps['scaler'], RobustScaler)
        assert isinstance(result2.model.named_steps['scaler'], RobustScaler)

    def test_backward_compat_no_preprocessing(self, small_data):
        """Without preprocessing param, behavior is unchanged."""
        X, y = small_data
        sl = SurvivalLearner(X, y)
        sl.setup(random_state=42, verbose=False)
        results = sl.train(['coxph', 'rsf'], verbose=False)

        # Both should have StandardScaler (the default)
        from sklearn.preprocessing import StandardScaler
        assert isinstance(results['coxph'].model.named_steps['scaler'], StandardScaler)
        assert isinstance(results['rsf'].model.named_steps['scaler'], StandardScaler)

    def test_empty_preprocessing_dict_is_noop(self, small_data):
        X, y = small_data
        sl = SurvivalLearner(X, y)
        sl.setup(random_state=42, verbose=False, preprocessing={})
        results = sl.train('coxph', verbose=False)
        from sklearn.preprocessing import StandardScaler
        assert isinstance(results.model.named_steps['scaler'], StandardScaler)

    def test_unused_preprocessing_key_warns_multi_model(self, small_data):
        """Warn if preprocessing has model names not in the models list (multi-model path)."""
        X, y = small_data
        sl = SurvivalLearner(X, y)
        sl.setup(random_state=42, verbose=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sl.train(
                ['coxph'], verbose=False,
                preprocessing={'gbsa': {'scaler': 'robust'}},
            )
        preprocessing_warnings = [
            x for x in w if "preprocessing" in str(x.message).lower()
            and "unused" in str(x.message).lower()
        ]
        assert len(preprocessing_warnings) == 1

    def test_unused_preprocessing_key_warns_single_model(self, small_data):
        """Warn about unused preprocessing keys even on the single-model train() path."""
        X, y = small_data
        sl = SurvivalLearner(X, y)
        sl.setup(random_state=42, verbose=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sl.train(
                'coxph', verbose=False,
                preprocessing={'rsf': {'scaler': 'robust'}},
            )
        preprocessing_warnings = [
            x for x in w if "preprocessing" in str(x.message).lower()
            and "unused" in str(x.message).lower()
        ]
        assert len(preprocessing_warnings) == 1

    def test_setup_verbose_shows_per_model(self, small_data, capsys):
        """Setup summary shows per-model preprocessing when configured."""
        X, y = small_data
        sl = SurvivalLearner(X, y)
        sl.setup(
            random_state=42, verbose=True,
            preprocessing={'coxph': {'scaler': 'robust'}},
        )
        captured = capsys.readouterr()
        assert "Per-model:" in captured.out
        assert "coxph" in captured.out


class TestSetupReset:
    """Test that re-calling setup() clears per-model preprocessing."""

    def test_re_setup_clears_model_preprocessing(self, small_data):
        X, y = small_data
        sl = SurvivalLearner(X, y)
        sl.setup(
            random_state=42, verbose=False,
            preprocessing={'coxph': {'scaler': 'robust'}},
        )
        assert len(sl._state.model_preprocessing) > 0

        # Re-setup without preprocessing
        sl.setup(random_state=42, verbose=False)
        assert len(sl._state.model_preprocessing) == 0


# ---------------------------------------------------------------------------
# Regression test: coefficient feature names with selector
# ---------------------------------------------------------------------------

class TestCoefficientFeatureNamesWithSelector:
    """Regression test: get_coefficients() returns integer indices."""

    @pytest.mark.parametrize("model_name", ["coxph", "coxnet"])
    def test_coefficients_have_string_feature_names(self, small_data, model_name):
        X, y = small_data
        sl = SurvivalLearner(X, y)
        sl.setup(random_state=42, verbose=False, feature_selector='topk')
        sl.train(model_name, verbose=False)

        pipeline = sl.models[model_name].model
        inner = unwrap_pipeline(pipeline)
        coefs = inner.get_coefficients()

        assert set(coefs.index).issubset(set(X.columns)), (
            f"Expected feature names from X.columns, got: {coefs.index.tolist()}"
        )
