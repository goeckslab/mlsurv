"""Tests for save_learner() / load_learner() persistence.

Covers roundtrip serialization of the entire learner state, including
data splits, trained models, evaluation results, bootstrap CIs, and
tuning metadata.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import SuggestionWarning
from mlsurv.learner import SurvivalLearner
from mlsurv.learner._persistence import _gpu_state_hint
from mlsurv.models.base import BaseSurvivalModel, PycoxSerializationMixin, _has_non_cpu_params

try:  # torch/torchtuples are the optional [deep] extra
    import torch  # noqa: F401
    import torchtuples  # noqa: F401
    _HAS_TORCH = True
except Exception:  # noqa: BLE001 - a blocked native dep need not raise ImportError
    _HAS_TORCH = False


class _RaisesOnUnpickle:
    """Reproduces torch's cross-device failure from inside a joblib load.

    Module level so it pickles; the message is torch's verbatim.
    """

    def __init__(self):
        # Pickle skips __setstate__ entirely when the state is falsy, so the
        # instance needs at least one attribute for this to fire on load.
        self._payload = 1

    def __setstate__(self, state):
        raise RuntimeError(
            "Attempting to deserialize object on a CUDA device but "
            "torch.cuda.is_available() is False."
        )


class _ToyPycoxModel(PycoxSerializationMixin, BaseSurvivalModel):
    """Stand-in for DeepSurv/DeepHit in serialization tests.

    Same mixin order and same torch attributes (``net_``, ``model_``,
    ``device``) as the real wrappers, without the cost of training a network.
    Defined at module level so it pickles.
    """

    def fit(self, X, y):  # pragma: no cover - never called
        raise NotImplementedError

    def predict(self, X):  # pragma: no cover - never called
        raise NotImplementedError

# RSF tuning cost scales with forest size, so the tuned roundtrip test passes a tiny
# Optuna space. Partial spaces merge per-param with the defaults. The test still tunes
# (it asserts an Optuna study was created), so the space only shrinks the forest.
_SMALL_TREE_SPACE = {'n_estimators': (10, 20), 'max_depth': (3, 5)}


@pytest.mark.integration
class TestSaveLearner:
    """Tests for save_learner() and load_learner() roundtrip."""

    def test_save_creates_file(self, small_data, tmp_path):
        """save_learner() should create a non-empty file."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        path = tmp_path / "learner.joblib"
        learner.save_learner(path)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_roundtrip_after_train(self, small_data, tmp_path):
        """Trained models should survive save/load roundtrip."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(["coxph"])

        original_preds = learner._state.models["coxph"].model.predict(learner.X_test)
        original_cv_mean = learner._state.models["coxph"].cv_metric_mean

        path = tmp_path / "learner.joblib"
        learner.save_learner(path)

        loaded = SurvivalLearner.load_learner(path)

        # Same model keys
        assert set(loaded._state.models.keys()) == {"coxph"}

        # CV metrics preserved
        assert loaded._state.models["coxph"].cv_metric_mean == original_cv_mean

        # Predictions match
        loaded_preds = loaded._state.models["coxph"].model.predict(loaded.X_test)
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_roundtrip_after_evaluate(self, small_data, tmp_path):
        """evaluation_result should survive roundtrip via computed property."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(["coxph"])
        learner.evaluate()

        # evaluation_result is now a computed property combining per-model data
        original_concordance = learner.evaluation_result.concordance

        path = tmp_path / "learner.joblib"
        learner.save_learner(path)
        loaded = SurvivalLearner.load_learner(path)

        assert loaded.evaluation_result is not None
        pd.testing.assert_frame_equal(
            original_concordance,
            loaded.evaluation_result.concordance,
        )

    def test_roundtrip_after_bootstrap(self, small_data, tmp_path):
        """bootstrap_result should survive roundtrip via computed property."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(["coxph"])
        learner.bootstrap(n_bootstrap=2)

        # bootstrap_result is now a computed property combining per-model data
        original_summary = learner.bootstrap_result.summary

        path = tmp_path / "learner.joblib"
        learner.save_learner(path)
        loaded = SurvivalLearner.load_learner(path)

        assert loaded.bootstrap_result is not None
        pd.testing.assert_frame_equal(
            original_summary,
            loaded.bootstrap_result.summary,
        )

    def test_roundtrip_with_tuned_model(self, small_data, tmp_path):
        """Tuned model roundtrip: optuna_study is None, metadata preserved."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        # RSF uses Optuna (CoxPH uses CoxNet's built-in CV, no Optuna study)
        learner.tune("rsf", n_trials=2, search_space=_SMALL_TREE_SPACE)

        model_result = learner._state.models["rsf"]
        assert model_result.optuna_study is not None, "RSF tune should set optuna_study"
        original_best_params = model_result.best_params_

        path = tmp_path / "learner.joblib"
        learner.save_learner(path)
        loaded = SurvivalLearner.load_learner(path)

        loaded_model = loaded._state.models["rsf"]
        # optuna_study should be None after roundtrip
        assert loaded_model.optuna_study is None
        # Metadata should be preserved
        assert hasattr(loaded_model, "_optuna_study_meta")
        meta = loaded_model._optuna_study_meta
        assert "study_name" in meta
        assert "best_params" in meta
        assert "n_trials" in meta
        assert meta["n_trials"] == 2
        # best_params_ should survive
        assert loaded_model.best_params_ == original_best_params

    def test_roundtrip_presplit(self, train_test_data, tmp_path):
        """Pre-split initialization should roundtrip correctly."""
        X_train, X_test, y_train, y_test = train_test_data
        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
        )
        learner.setup(random_state=42)
        learner.train(["coxph"])

        path = tmp_path / "learner.joblib"
        learner.save_learner(path)
        loaded = SurvivalLearner.load_learner(path)

        assert loaded._state.presplit is True
        pd.testing.assert_frame_equal(loaded.X_train, X_train)
        pd.testing.assert_frame_equal(loaded.X_test, X_test)

    def test_save_before_setup_raises(self, small_data, tmp_path):
        """save_learner() before setup() should raise ValueError."""
        X, y = small_data
        learner = SurvivalLearner(X, y)

        with pytest.raises(ValueError, match="Cannot save.*not been set up"):
            learner.save_learner(tmp_path / "learner.joblib")

    def test_load_file_not_found(self, tmp_path):
        """load_learner() with nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Learner file not found"):
            SurvivalLearner.load_learner(tmp_path / "nonexistent.joblib")

    def test_load_bad_version(self, tmp_path):
        """load_learner() should reject files with unsupported version."""
        import joblib

        path = tmp_path / "bad_version.joblib"
        envelope = {"_learner_version": 999, "state": None}
        joblib.dump(envelope, path)

        with pytest.raises(ValueError, match="Unsupported learner format version"):
            SurvivalLearner.load_learner(path)

    def test_load_missing_version(self, tmp_path):
        """load_learner() should reject files with no version key."""
        import joblib

        path = tmp_path / "no_version.joblib"
        envelope = {"state": None}
        joblib.dump(envelope, path)

        with pytest.raises(ValueError, match="Unsupported learner format version"):
            SurvivalLearner.load_learner(path)

    def test_load_corrupt_file(self, tmp_path):
        """load_learner() should raise RuntimeError for corrupt files."""
        path = tmp_path / "corrupt.joblib"
        path.write_bytes(b"not a valid joblib file")

        with pytest.raises(RuntimeError, match="Failed to load learner"):
            SurvivalLearner.load_learner(path)

    def test_load_non_dict_payload(self, tmp_path):
        """load_learner() should reject non-dict joblib payloads."""
        import joblib

        path = tmp_path / "not_learner.joblib"
        joblib.dump([1, 2, 3], path)

        with pytest.raises(ValueError, match="not a valid learner file"):
            SurvivalLearner.load_learner(path)

    def test_load_missing_state_key(self, tmp_path):
        """load_learner() should reject files without a state key."""
        import joblib

        path = tmp_path / "no_state.joblib"
        joblib.dump({"_learner_version": 1}, path)

        with pytest.raises(ValueError, match="missing the 'state' key"):
            SurvivalLearner.load_learner(path)

    @pytest.mark.parametrize("bad_version", [0, -1, "v1", 2.5])
    def test_load_rejects_invalid_version_values(self, tmp_path, bad_version):
        """load_learner() should reject version 0, negative, non-int."""
        import joblib

        path = tmp_path / "bad.joblib"
        joblib.dump({"_learner_version": bad_version, "state": None}, path)

        with pytest.raises(ValueError, match="Unsupported learner format version"):
            SurvivalLearner.load_learner(path)

    def test_forward_compat_missing_state_fields(self, small_data, tmp_path):
        """Loading a file saved before new LearnerState fields were added."""
        import joblib

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        path = tmp_path / "learner.joblib"
        learner.save_learner(path)

        # Simulate an older file missing a field
        envelope = joblib.load(path)
        del envelope["state"].__dict__["benchmark_result"]
        joblib.dump(envelope, path)

        loaded = SurvivalLearner.load_learner(path)
        # Missing field should get default value (None)
        assert loaded._state.benchmark_result is None

    def test_round_trip_preserves_col_names(self, small_data, tmp_path):
        """save_learner/load_learner should preserve time_col and event_col."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner._state.time_col = 'OS_MONTHS'
        learner._state.event_col = 'OS_STATUS'

        path = tmp_path / "learner.joblib"
        learner.save_learner(path)
        loaded = SurvivalLearner.load_learner(path)

        assert loaded._state.time_col == 'OS_MONTHS'
        assert loaded._state.event_col == 'OS_STATUS'

    def test_load_version_mismatch_warns(self, small_data, tmp_path):
        """load_learner() should warn when mlsurv version differs."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        path = tmp_path / "learner.joblib"
        learner.save_learner(path)

        # Patch the saved version to differ
        import joblib
        envelope = joblib.load(path)
        envelope["_mlsurv_version"] = "0.0.0-fake"
        joblib.dump(envelope, path)

        with pytest.warns(SuggestionWarning, match="saved with mlsurv 0.0.0-fake"):
            SurvivalLearner.load_learner(path)

    def test_compression_reduces_size(self, small_data, tmp_path):
        """compress=3 should produce a smaller file than compress=0."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(["coxph"])

        path_compressed = tmp_path / "compressed.joblib"
        path_uncompressed = tmp_path / "uncompressed.joblib"

        learner.save_learner(path_compressed, compress=3)
        learner.save_learner(path_uncompressed, compress=0)

        assert path_compressed.stat().st_size < path_uncompressed.stat().st_size

    @pytest.mark.integration
    def test_full_analysis_roundtrip(self, small_data, tmp_path):
        """End-to-end: setup -> train -> evaluate -> bootstrap -> save -> load.

        Verifies that a complete analysis session can be restored in a new
        session without re-running anything.
        """
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train(["coxph", "rsf"])
        learner.evaluate()
        learner.bootstrap(n_bootstrap=2)

        # Capture originals
        original_summary = learner.summary()
        original_config = learner.config
        original_preds = {
            name: mr.model.predict(learner.X_test)
            for name, mr in learner._state.models.items()
        }
        original_eval = learner.evaluation_result.concordance.copy()
        original_bootstrap = learner.bootstrap_result.summary.copy()

        # Save and destroy original
        path = tmp_path / "full_analysis.learner"
        learner.save_learner(path)
        del learner

        # Load into fresh variable
        loaded = SurvivalLearner.load_learner(path)

        # 1. summary() returns identical DataFrame
        pd.testing.assert_frame_equal(original_summary, loaded.summary())

        # 2. Each model's predictions match
        for name, orig_preds in original_preds.items():
            loaded_preds = loaded._state.models[name].model.predict(loaded.X_test)
            np.testing.assert_array_almost_equal(orig_preds, loaded_preds)

        # 3. evaluation_result.concordance matches (via computed property)
        pd.testing.assert_frame_equal(
            original_eval,
            loaded.evaluation_result.concordance,
        )

        # 4. bootstrap_result.summary matches (via computed property)
        pd.testing.assert_frame_equal(
            original_bootstrap,
            loaded.bootstrap_result.summary,
        )

        # 5. Config values match
        loaded_config = loaded.config
        assert loaded_config["cv_folds"] == original_config["cv_folds"]
        assert loaded_config["scoring"] == original_config["scoring"]
        assert loaded_config["random_state"] == original_config["random_state"]

        # 6. Plotting methods don't crash
        loaded.plot_survival()
        loaded.plot_metric()

    @pytest.mark.integration
    def test_per_model_derived_state_roundtrips(self, small_data, tmp_path):
        """The calibrator + importance/risk-group test tables ride on the
        ModelResult, so they round-trip via the pickled ``_state.models`` and
        the views reassemble after load."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3, feature_selector=None)
        learner.train(["coxph"], calibrate="isotonic")
        learner.evaluate(verbose=False)
        learner.importance_test("coxph", method="pimp", n_permutations=5,
                                n_importance_repeats=2, show_progress=False,
                                verbose=False)
        learner.logrank_test("coxph", verbose=False)
        learner.rmst_test("coxph", verbose=False)

        orig_mappings = dict(learner._state.models["coxph"].calibrator.mappings)

        path = tmp_path / "derived.learner"
        learner.save_learner(path)
        del learner
        loaded = SurvivalLearner.load_learner(path)

        # The calibrator reassembles from the loaded ModelResult.
        loaded_cal = loaded._state.models["coxph"].calibrator
        assert loaded_cal.method == "isotonic"
        assert set(loaded_cal.mappings) == set(orig_mappings)
        assert "coxph" in loaded.pimp_importance_test_results
        assert "coxph" in loaded.logrank_test_results
        assert "coxph" in loaded.rmst_test_results
        # The risk-group tables keep their model_type column and provenance
        # stamp across the round-trip.
        assert "model_type" in loaded.rmst_test_results["coxph"].columns
        assert loaded.rmst_test_results["coxph"].attrs["model_name"] == "coxph"
        # The calibrator's isotonic maps survived pickling.
        for h, iso in loaded_cal.mappings.items():
            assert hasattr(iso, "predict")


@pytest.mark.requires_pycox
@pytest.mark.integration
class TestDeepSurvPersistence:
    """Test DeepSurv save/load roundtrip."""

    def test_deepsurv_roundtrip(self, small_data, tmp_path):
        """Trained DeepSurv model should survive save_learner/load_learner."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(["deepsurv"])

        original_preds = learner._state.models["deepsurv"].model.predict(
            learner.X_test
        )

        path = tmp_path / "deepsurv.joblib"
        learner.save_learner(path)

        loaded = SurvivalLearner.load_learner(path)
        loaded_preds = loaded._state.models["deepsurv"].model.predict(
            loaded.X_test
        )

        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

        # The loaded model sits on a device this host actually has.
        loaded_ds = loaded._state.models["deepsurv"].model.named_steps["model"]
        assert loaded_ds.model_.device.type == 'cpu'

    def test_deepsurv_callback_pickle_roundtrip(self, synthetic_data_clean, tmp_path):
        """DeepSurv with early-stopping callbacks must survive pickle."""
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(["deepsurv"], deepsurv_params={"epochs": 5, "patience": 3})

        # Verify callbacks exist on the pycox model (the unpicklable state)
        pipeline = learner._state.models["deepsurv"].model
        ds_model = pipeline.named_steps["model"]
        assert ds_model.model_.callbacks is not None

        original_preds = pipeline.predict(learner.X_test)

        path = tmp_path / "deepsurv_callbacks.joblib"
        learner.save_learner(path)

        loaded = SurvivalLearner.load_learner(path)
        loaded_preds = loaded._state.models["deepsurv"].model.predict(
            loaded.X_test
        )
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

        # Callbacks should be stripped after load
        loaded_ds = loaded._state.models["deepsurv"].model.named_steps["model"]
        assert loaded_ds.model_.callbacks is None

        # Original model must still work (getstate must not mutate live model)
        post_save_preds = pipeline.predict(learner.X_test)
        np.testing.assert_array_almost_equal(original_preds, post_save_preds)
        assert ds_model.model_.callbacks is not None


@pytest.mark.requires_pycox
@pytest.mark.integration
class TestDeepHitPersistence:
    """Test DeepHit save/load roundtrip."""

    def test_deephit_callback_pickle_roundtrip(self, synthetic_data_clean, tmp_path):
        """DeepHit with early-stopping callbacks must survive pickle."""
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(["deephit"], deephit_params={"epochs": 5, "patience": 3})

        pipeline = learner._state.models["deephit"].model
        dh_model = pipeline.named_steps["model"]
        assert dh_model.model_.callbacks is not None

        original_preds = pipeline.predict(learner.X_test)

        path = tmp_path / "deephit_callbacks.joblib"
        learner.save_learner(path)

        loaded = SurvivalLearner.load_learner(path)
        loaded_preds = loaded._state.models["deephit"].model.predict(
            loaded.X_test
        )
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

        loaded_dh = loaded._state.models["deephit"].model.named_steps["model"]
        assert loaded_dh.model_.callbacks is None
        assert loaded_dh.model_.device.type == 'cpu'


@pytest.mark.skipif(not _HAS_TORCH, reason="requires torch and torchtuples")
class TestDevicePortableSerialization:
    """Deep-model pickles must load on a host without the training device.

    Torch tensors remember the device they were pickled from, so a GPU-trained
    model has to be written as CPU state and put back on an available device at
    load time. These tests need torch/torchtuples but not pycox, so they stay in
    the fast tier (the ``requires_pycox`` marker is excluded from it).
    """

    @staticmethod
    def _make_wrapper(device='auto'):
        """A minimal wrapper carrying the same torch state as DeepSurv/DeepHit."""
        import torch.nn as nn
        import torchtuples as tt

        net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
        wrapper = _ToyPycoxModel()
        wrapper.device = device
        wrapper.net_ = net
        wrapper.model_ = tt.Model(net, loss=None)
        return wrapper

    def test_cpu_model_state_is_untouched(self):
        """A CPU-trained net must not be copied — that path is unchanged."""
        wrapper = self._make_wrapper()
        state = wrapper.__getstate__()

        assert state['net_'] is wrapper.net_

    def test_cpu_model_roundtrip_keeps_net_identity(self, tmp_path):
        """model_.net and net_ must remain the same object after a load."""
        import joblib

        wrapper = self._make_wrapper()
        path = tmp_path / "toy.joblib"
        joblib.dump(wrapper, path)
        loaded = joblib.load(path)

        assert loaded.model_.net is loaded.net_
        assert loaded.model_.device.type == 'cpu'

    def test_gpu_state_is_written_as_cpu(self, monkeypatch):
        """A GPU-resident net is deep-copied to CPU, leaving the live one alone.

        There is no GPU here, so patch the device predicate to take the branch.
        """
        import torch

        wrapper = self._make_wrapper(device='cuda')
        # A torch.device is constructible without the hardware, so the live
        # model can carry the device a GPU host would have left on it.
        wrapper.model_._device = torch.device('cuda')
        live_net = wrapper.net_
        live_weights = live_net[0].weight.detach().clone()

        monkeypatch.setattr(
            'mlsurv.models.base._has_non_cpu_params', lambda net: True
        )
        state = wrapper.__getstate__()

        # A distinct CPU copy, with the same weights...
        assert state['net_'] is not live_net
        assert torch.equal(state['net_'][0].weight, live_weights)
        # ...wired into the copied pycox model, which is now CPU-bound...
        assert state['model_'].net is state['net_']
        assert state['model_'].device == torch.device('cpu')
        # ...and the live model untouched.
        assert wrapper.net_ is live_net
        assert torch.equal(live_net[0].weight, live_weights)
        assert wrapper.model_.device == torch.device('cuda')
        assert wrapper.device == 'cuda'

    def test_gpu_state_roundtrips_through_a_real_load(self, tmp_path, monkeypatch):
        """The whole contract: a GPU-saved file opens here, on the CPU.

        Saves through the GPU branch of __getstate__ and loads it back, so the
        object graph (net_ replaced, model_.net rewired separately) is exercised
        end to end rather than one half at a time.
        """
        import joblib

        wrapper = self._make_wrapper(device='cuda')
        path = tmp_path / "toy_gpu_roundtrip.joblib"
        monkeypatch.setattr(
            'mlsurv.models.base._has_non_cpu_params', lambda net: True
        )
        joblib.dump(wrapper, path)
        monkeypatch.undo()

        with pytest.warns(SuggestionWarning, match="could not be restored here"):
            loaded = joblib.load(path)

        assert loaded.model_.net is loaded.net_
        assert loaded.model_.device.type == 'cpu'
        assert not _has_non_cpu_params(loaded.net_)

    def test_getstate_rewires_net_without_set_device(self, monkeypatch):
        """A pycox model lacking set_device must not keep the GPU net.

        Rewiring `.net` is what keeps GPU tensors out of the pickle, so it can't
        depend on an optional torchtuples method being present.
        """
        import torch.nn as nn

        class _NoSetDevice:
            def __init__(self, net):
                self.net = net

        wrapper = _ToyPycoxModel()
        wrapper.device = 'cuda'
        wrapper.net_ = nn.Linear(4, 1)
        wrapper.model_ = _NoSetDevice(wrapper.net_)
        live_net = wrapper.net_

        monkeypatch.setattr(
            'mlsurv.models.base._has_non_cpu_params', lambda net: True
        )
        state = wrapper.__getstate__()

        assert state['model_'].net is state['net_']
        assert state['model_'].net is not live_net

    @pytest.mark.parametrize("device", ['cuda', 'cuda:0', 'cuda:3'])
    def test_load_falls_back_to_cpu_when_cuda_missing(self, tmp_path, device):
        """An unreachable CUDA device loads on CPU and says so."""
        import joblib

        wrapper = self._make_wrapper(device=device)
        path = tmp_path / f"toy_{device.replace(':', '_')}.joblib"
        joblib.dump(wrapper, path)

        with pytest.warns(SuggestionWarning, match="could not be restored here"):
            loaded = joblib.load(path)

        assert loaded.model_.device.type == 'cpu'
        assert not _has_non_cpu_params(loaded.net_)
        # device is a constructor param — it records the request, not where the
        # weights ended up. Rewriting it would lose the request for good.
        assert loaded.device == device

    def test_move_failure_falls_back_and_reports_the_cause(self, tmp_path, monkeypatch):
        """A GPU that resolves but cannot be moved to must not fail silently.

        This is the default-config hole: with device='auto' on a host where CUDA
        resolves but the move fails (OOM, driver mismatch, forked worker), the
        model quietly lands on CPU. It has to warn, and name the real cause.
        """
        import joblib
        import torch

        from mlsurv.models.base import PycoxSerializationMixin

        wrapper = self._make_wrapper(device='auto')
        path = tmp_path / "toy_move_fails.joblib"
        joblib.dump(wrapper, path)

        real_move = PycoxSerializationMixin._move_to_device

        def only_cpu_works(model, net, target):
            if target.type != 'cpu':
                raise AssertionError("Torch not compiled with CUDA enabled")
            real_move(model, net, target)

        monkeypatch.setattr(
            PycoxSerializationMixin, '_move_to_device', staticmethod(only_cpu_works)
        )
        monkeypatch.setattr(
            _ToyPycoxModel,
            '_resolve_device',
            staticmethod(lambda d: torch.device('cuda:0' if d == 'auto' else 'cuda:3')),
        )

        with pytest.warns(SuggestionWarning, match="Torch not compiled with CUDA"):
            loaded = joblib.load(path)

        assert loaded.model_.device.type == 'cpu'

    def test_restores_onto_an_available_accelerator(self, tmp_path, monkeypatch):
        """The GPU->GPU path: an available accelerator is used, and we stay quiet.

        No GPU here, so this uses torch's 'meta' device — a genuinely non-CPU
        device that needs no hardware. It exercises the real move (torchtuples
        set_device) rather than a stubbed one; what it cannot cover is moving
        actual tensor data between devices.
        """
        import joblib
        import torch

        wrapper = self._make_wrapper(device='cuda')
        path = tmp_path / "toy_accelerator.joblib"
        joblib.dump(wrapper, path)

        monkeypatch.setattr(
            _ToyPycoxModel, '_resolve_device', staticmethod(lambda d: torch.device('meta'))
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = joblib.load(path)

        assert loaded.model_.device.type == 'meta'
        assert next(loaded.net_.parameters()).device.type == 'meta'
        # The device was reachable, so there is nothing to tell the user.
        assert not [w for w in caught if issubclass(w.category, SuggestionWarning)]

    def test_falls_back_to_another_gpu_before_giving_up(self, tmp_path, monkeypatch):
        """A model trained on 'cuda:3' should still use cuda:0 on a 1-GPU host.

        Dropping straight to CPU here would quietly cost a GPU user all their
        acceleration. Simulated: this host has no CUDA at all.
        """
        import joblib
        import torch

        from mlsurv.models.base import PycoxSerializationMixin

        wrapper = self._make_wrapper(device='cuda:3')
        path = tmp_path / "toy_wrong_ordinal.joblib"
        joblib.dump(wrapper, path)

        attempted = []
        real_move = PycoxSerializationMixin._move_to_device

        def only_cuda3_fails(model, net, target):
            attempted.append(str(target))
            if str(target) == 'cuda:3':
                raise RuntimeError("CUDA error: invalid device ordinal")
            real_move(model, net, torch.device('cpu'))

        monkeypatch.setattr(
            PycoxSerializationMixin, '_move_to_device', staticmethod(only_cuda3_fails)
        )
        monkeypatch.setattr(
            _ToyPycoxModel,
            '_resolve_device',
            staticmethod(lambda d: torch.device('cuda:0' if d == 'auto' else 'cuda:3')),
        )

        with pytest.warns(SuggestionWarning, match="invalid device ordinal"):
            joblib.load(path)

        # It settled on the other GPU rather than falling all the way to CPU.
        assert attempted == ['cuda:3', 'cuda:0']

    def test_no_retry_on_the_device_that_just_failed(self, tmp_path, monkeypatch):
        """'auto' resolving to the failed device must not produce a retry.

        Otherwise a transient failure could be retried on the same device and
        succeed, leaving the warning saying the device 'could not be restored'
        while reporting that same device as the destination.
        """
        import joblib
        import torch

        from mlsurv.models.base import PycoxSerializationMixin

        wrapper = self._make_wrapper(device='cuda')
        path = tmp_path / "toy_same_device.joblib"
        joblib.dump(wrapper, path)

        attempted = []
        real_move = PycoxSerializationMixin._move_to_device

        def cuda_fails(model, net, target):
            attempted.append(str(target))
            if target.type != 'cpu':
                raise RuntimeError("CUDA out of memory")
            real_move(model, net, target)

        monkeypatch.setattr(
            PycoxSerializationMixin, '_move_to_device', staticmethod(cuda_fails)
        )
        monkeypatch.setattr(
            _ToyPycoxModel, '_resolve_device', staticmethod(lambda d: torch.device('cuda'))
        )

        with pytest.warns(SuggestionWarning, match="loaded onto cpu instead"):
            joblib.load(path)

        assert attempted == ['cuda', 'cpu']

    def test_cpu_move_failure_is_not_swallowed(self, tmp_path, monkeypatch):
        """A CPU move that fails is a real bug — it must propagate."""
        import joblib

        from mlsurv.models.base import PycoxSerializationMixin

        wrapper = self._make_wrapper(device='cpu')
        path = tmp_path / "toy_cpu_fails.joblib"
        joblib.dump(wrapper, path)

        def always_fails(model, net, target):
            raise RuntimeError("something genuinely broken")

        monkeypatch.setattr(
            PycoxSerializationMixin, '_move_to_device', staticmethod(always_fails)
        )

        with pytest.raises(RuntimeError, match="genuinely broken"):
            joblib.load(path)

    @pytest.mark.parametrize("device", ['auto', 'cpu', None])
    def test_load_on_cpu_host_is_quiet(self, tmp_path, device, recwarn):
        """'auto'/'cpu'/None already mean this host — no warning to give."""
        import joblib

        wrapper = self._make_wrapper(device=device)
        path = tmp_path / f"toy_{device}.joblib"
        joblib.dump(wrapper, path)
        loaded = joblib.load(path)

        assert loaded.device == device
        assert loaded.model_.device.type == 'cpu'
        assert not [w for w in recwarn if issubclass(w.category, SuggestionWarning)]

    def test_net_without_pycox_model_is_moved(self, tmp_path):
        """A wrapper with net_ but no model_ (interrupted fit) still loads."""
        import joblib
        import torch.nn as nn

        wrapper = _ToyPycoxModel()
        wrapper.device = 'cpu'
        wrapper.net_ = nn.Linear(4, 1)
        path = tmp_path / "toy_net_only.joblib"
        joblib.dump(wrapper, path)
        loaded = joblib.load(path)

        assert not _has_non_cpu_params(loaded.net_)

    def test_unfitted_wrapper_roundtrips(self, tmp_path, recwarn):
        """No torch state yet — nothing to move, and nothing should break."""
        import joblib

        wrapper = _ToyPycoxModel()
        wrapper.device = 'cuda'
        path = tmp_path / "toy_unfitted.joblib"
        joblib.dump(wrapper, path)
        loaded = joblib.load(path)

        assert loaded.device == 'cuda'
        assert not [w for w in recwarn if issubclass(w.category, SuggestionWarning)]


@pytest.mark.skipif(
    not (_HAS_TORCH and torch.cuda.is_available()),
    reason="requires a real CUDA device",
)
@pytest.mark.requires_pycox
@pytest.mark.integration
class TestGpuToCpuRoundTrip:
    """The half that needs hardware: save on a GPU, open as if CPU-only.

    Skipped everywhere without a GPU (including CI), so run it on a GPU node:

        pytest tests/test_save_learner.py -k GpuToCpu -m "requires_pycox"
    """

    def test_gpu_saved_learner_loads_without_cuda(self, small_data, tmp_path):
        from unittest import mock

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, device='cuda')
        learner.train('deepsurv', tune=False, epochs=10)

        pipeline = learner._state.models['deepsurv'].model
        wrapper = pipeline.named_steps['model']
        assert wrapper.model_.device.type == 'cuda', "expected GPU training"
        gpu_preds = pipeline.predict(learner.X_test)

        path = tmp_path / 'gpu.learner'
        learner.save_learner(path)

        # Saving must not disturb the live GPU model.
        assert wrapper.model_.device.type == 'cuda'
        np.testing.assert_array_almost_equal(
            gpu_preds, pipeline.predict(learner.X_test)
        )

        # torch's deserializer checks torch.cuda.is_available() directly, so
        # this is exactly what the file looks like to a CPU-only host. Any CUDA
        # tensor left in the pickle raises "Attempting to deserialize object on
        # a CUDA device" here.
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            with pytest.warns(SuggestionWarning, match="could not be restored here"):
                loaded = SurvivalLearner.load_learner(path)
            cpu_preds = loaded._state.models['deepsurv'].model.predict(loaded.X_test)

        loaded_wrapper = (
            loaded._state.models['deepsurv'].model.named_steps['model']
        )
        assert loaded_wrapper.model_.device.type == 'cpu'
        assert not _has_non_cpu_params(loaded_wrapper.net_)
        # The user's request survives; only the placement changed.
        assert loaded_wrapper.device == 'cuda'
        # Loosened from the usual 6dp: float32 matmul is not bit-identical
        # across devices, so an exact match would be flaky, not correct.
        np.testing.assert_array_almost_equal(gpu_preds, cpu_preds, decimal=4)


@pytest.mark.skipif(not _HAS_TORCH, reason="requires torch")
class TestHasNonCpuParams:
    """The predicate that decides whether CPU normalization is needed.

    torch's 'meta' device gives a genuinely non-CPU module without any
    accelerator, so this can be tested for real rather than monkeypatched.
    """

    def test_cpu_net(self):
        import torch.nn as nn

        assert not _has_non_cpu_params(nn.Linear(4, 1))

    def test_non_cpu_net(self):
        import torch.nn as nn

        assert _has_non_cpu_params(nn.Linear(4, 1).to('meta'))

    def test_detects_buffers_not_just_parameters(self):
        """BatchNorm running stats are buffers — a params-only check misses them."""
        import torch.nn as nn

        net = nn.BatchNorm1d(3).to('meta')
        assert not list(net.parameters()) or _has_non_cpu_params(net)
        assert _has_non_cpu_params(net)

    @pytest.mark.parametrize("value", [None, object(), "not a net"])
    def test_non_modules_have_nothing_to_move(self, value):
        assert not _has_non_cpu_params(value)


class TestGpuStateHint:
    """Old GPU-written files should not be reported as corruption."""

    def test_cuda_error_gets_a_targeted_message(self):
        exc = RuntimeError(
            "Attempting to deserialize object on a CUDA device but "
            "torch.cuda.is_available() is False."
        )
        hint = _gpu_state_hint(exc, "learner.joblib")

        assert "saved on a GPU device" in hint
        assert "re-save it" in hint
        assert "RuntimeError" in hint  # the original error survives

    def test_wrong_gpu_ordinal_also_matches(self):
        exc = RuntimeError(
            "Attempting to deserialize object on CUDA device 3 but "
            "torch.cuda.device_count() is 1."
        )
        assert _gpu_state_hint(exc, "x.joblib") is not None

    @pytest.mark.parametrize("message", [
        "bad magic number",
        # 'deserialize' or 'CUDA' alone must not trigger it: a truncated file
        # and an OOM on a host that does have a GPU are different problems with
        # different fixes, and the generic message is the honest one.
        "Could not deserialize object of type Foo",
        "unpickling error: failed to deserialize the payload",
        "CUDA error: out of memory",
    ])
    def test_unrelated_errors_are_left_alone(self, message):
        assert _gpu_state_hint(ValueError(message), "x.joblib") is None

    def test_load_learner_surfaces_the_hint(self, tmp_path):
        """The hint has to reach the user through load_learner, not just exist."""
        import joblib

        path = tmp_path / "gpu_written.learner"
        joblib.dump({
            '_learner_version': 1,
            '_timestamp': '2026-07-28T00:00:00+00:00',
            '_mlsurv_version': '0.1.0',
            'state': _RaisesOnUnpickle(),
        }, path)

        with pytest.raises(RuntimeError, match="saved on a GPU device"):
            SurvivalLearner.load_learner(path)

    def test_load_artifact_surfaces_the_hint(self, tmp_path):
        """The artifact path holds the same deep models and needs the same hint."""
        import json

        import joblib

        from mlsurv.constants import ARTIFACT_METADATA_FILE, ARTIFACT_MODEL_FILE

        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        (artifact_dir / ARTIFACT_METADATA_FILE).write_text(
            json.dumps({'artifact_version': 1, 'model_name': 'deepsurv'})
        )
        joblib.dump(_RaisesOnUnpickle(), artifact_dir / ARTIFACT_MODEL_FILE)

        with pytest.raises(RuntimeError, match="saved on a GPU device"):
            SurvivalLearner.load_artifact(artifact_dir)

    def test_ordinary_corruption_keeps_the_generic_message(self, tmp_path):
        path = tmp_path / "corrupt.learner"
        path.write_bytes(b"not a valid joblib file")

        with pytest.raises(RuntimeError, match="may be corrupt"):
            SurvivalLearner.load_learner(path)


class TestLoadLearnerConvenience:
    """Tests for the top-level load_learner convenience alias."""

    def test_import_from_mlsurv(self):
        """load_learner should be importable from mlsurv."""
        from mlsurv import load_learner
        assert callable(load_learner)

    def test_convenience_alias_works(self, small_data, tmp_path):
        """mlsurv.load_learner should load a saved learner."""
        from mlsurv import load_learner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        path = tmp_path / "learner.joblib"
        learner.save_learner(path)

        loaded = load_learner(path)
        assert loaded._state.is_setup is True
