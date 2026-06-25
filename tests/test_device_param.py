"""Tests for the explicit device parameter on deep learning models."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pycox = pytest.importorskip("pycox")

from mlsurv.models.base import BaseSurvivalModel
from mlsurv.models.deepsurv import DeepSurvModel
from mlsurv.models.deephit import DeepHitModel
from mlsurv.utils import to_structured_array


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_data():
    """Small synthetic dataset for quick DL model tests."""
    rng = np.random.RandomState(42)
    n, p = 80, 5
    X = rng.randn(n, p).astype(np.float32)
    time = rng.exponential(10, n).astype(np.float32)
    event = rng.binomial(1, 0.6, n).astype(bool)
    y = to_structured_array(time, event)
    import pandas as pd
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    return X_df, y


# ---------------------------------------------------------------------------
# _resolve_device unit tests
# ---------------------------------------------------------------------------

class TestResolveDevice:
    def test_cpu(self):
        dev = BaseSurvivalModel._resolve_device('cpu')
        assert dev == torch.device('cpu')

    def test_auto_no_gpu(self):
        # On CI (no GPU), 'auto' should resolve to CPU
        dev = BaseSurvivalModel._resolve_device('auto')
        if not torch.cuda.is_available():
            assert dev == torch.device('cpu')

    def test_none_same_as_auto(self):
        dev_none = BaseSurvivalModel._resolve_device(None)
        dev_auto = BaseSurvivalModel._resolve_device('auto')
        assert dev_none == dev_auto

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid device"):
            BaseSurvivalModel._resolve_device('not_a_device_xyz')

    def test_cuda_unavailable_raises(self):
        if not torch.cuda.is_available():
            with pytest.raises(ValueError, match="CUDA is not available"):
                BaseSurvivalModel._resolve_device('cuda')


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

class TestDeviceParamStored:
    def test_deepsurv_default(self):
        m = DeepSurvModel()
        assert m.device == 'auto'

    def test_deepsurv_explicit(self):
        m = DeepSurvModel(device='cpu')
        assert m.device == 'cpu'

    def test_deephit_default(self):
        m = DeepHitModel()
        assert m.device == 'auto'

    def test_deephit_explicit(self):
        m = DeepHitModel(device='cpu')
        assert m.device == 'cpu'


# ---------------------------------------------------------------------------
# Fit + predict on CPU
# ---------------------------------------------------------------------------

class TestFitCPU:
    @pytest.mark.parametrize("model_cls", [DeepSurvModel, DeepHitModel])
    def test_fit_cpu(self, small_data, model_cls):
        X, y = small_data
        m = model_cls(layers=[8], epochs=5, patience=None, device='cpu')
        m.fit(X, y)
        assert m.is_fitted_
        # Check weights are on CPU
        param = next(m.net_.parameters())
        assert param.device == torch.device('cpu')

    @pytest.mark.parametrize("model_cls", [DeepSurvModel, DeepHitModel])
    def test_predict_after_cpu_fit(self, small_data, model_cls):
        X, y = small_data
        m = model_cls(layers=[8], epochs=5, patience=None, device='cpu')
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    @pytest.mark.parametrize("model_cls", [DeepSurvModel, DeepHitModel])
    def test_surv_func_after_cpu_fit(self, small_data, model_cls):
        X, y = small_data
        m = model_cls(layers=[8], epochs=5, patience=None, device='cpu')
        m.fit(X, y)
        surv = m.predict_survival_function(X)
        assert surv.shape[1] == len(X)


# ---------------------------------------------------------------------------
# Learner integration
# ---------------------------------------------------------------------------

class TestLearnerDevice:
    def test_setup_device_stored(self, small_data):
        from mlsurv.learner import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(device='cpu', verbose=False)
        assert learner._state.device == 'cpu'

    def test_setup_device_none_by_default(self, small_data):
        from mlsurv.learner import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        assert learner._state.device is None

    def test_train_propagates_device(self, small_data):
        from mlsurv.learner import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(device='cpu', verbose=False)
        result = learner.train('deepsurv', verbose=False, epochs=3, patience=None)
        # The model inside the pipeline should have device='cpu'
        model = result.model.named_steps['model']
        assert model.device == 'cpu'

    def test_train_kwargs_device_override(self, small_data):
        from mlsurv.learner import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(device='cuda', verbose=False)  # set cuda (won't be used)
        result = learner.train(
            'deepsurv', verbose=False, epochs=3, patience=None, device='cpu',
        )
        model = result.model.named_steps['model']
        assert model.device == 'cpu'

    def test_non_dl_models_ignore_device(self, small_data):
        from mlsurv.learner import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(device='cpu', verbose=False)
        result = learner.train('coxph', verbose=False)
        assert result.cv_metric_mean is not None


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

class TestCLIDeviceFlag:
    def test_device_flag_parsed(self):
        from mlsurv.cli._parsers import create_parser
        parser = create_parser()
        args = parser.parse_args(['train', 'data.csv', '--model', 'deepsurv', '--device', 'cpu'])
        assert args.device == 'cpu'

    def test_device_flag_default_none(self):
        from mlsurv.cli._parsers import create_parser
        parser = create_parser()
        args = parser.parse_args(['train', 'data.csv', '--model', 'deepsurv'])
        assert args.device is None


# ---------------------------------------------------------------------------
# Propagation guard: device=None at setup leaves model default intact
# ---------------------------------------------------------------------------

class TestDeviceNonePreservesModelDefault:
    def test_setup_none_model_keeps_auto(self, small_data):
        from mlsurv.learner import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)  # device=None (default)
        result = learner.train('deepsurv', verbose=False, epochs=3, patience=None)
        model = result.model.named_steps['model']
        assert model.device == 'auto'  # not None — model's own default


# ---------------------------------------------------------------------------
# config property includes device
# ---------------------------------------------------------------------------

class TestGetConfigDevice:
    def test_get_config_has_device(self, small_data):
        from mlsurv.learner import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(device='cpu', verbose=False)
        config = learner.config
        assert 'device' in config
        assert config['device'] == 'cpu'

    def test_get_config_device_none(self, small_data):
        from mlsurv.learner import SurvivalLearner
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        config = learner.config
        assert 'device' in config
        assert config['device'] is None
