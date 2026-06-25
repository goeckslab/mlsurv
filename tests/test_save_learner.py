"""Tests for save_learner() / load_learner() persistence.

Covers roundtrip serialization of the entire learner state, including
data splits, trained models, evaluation results, bootstrap CIs, and
tuning metadata.
"""

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import SuggestionWarning
from mlsurv.learner import SurvivalLearner

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
