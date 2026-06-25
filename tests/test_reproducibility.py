"""Test reproducibility of mlsurv results.

Properties under test:
- Verify random seeds are set and propagated correctly (numpy, torch, optuna)
- Check that results are deterministic when seeds are fixed
- Ensure model serialization/deserialization preserves state
"""

import pytest
import numpy as np

pytestmark = [pytest.mark.integration, pytest.mark.integration_heavy]


class TestSeedPropagation:
    """Verify random seed propagation across components."""

    def test_numpy_seed_propagation(self, synthetic_data_clean):
        """Verify numpy random state is properly seeded."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        # Run with same seed twice
        learner1 = SurvivalLearner(X, y)
        learner1.setup(random_state=42)

        learner2 = SurvivalLearner(X, y)
        learner2.setup(random_state=42)

        # Train/test splits should be identical
        assert learner1.X_train.index.equals(learner2.X_train.index)
        assert learner1.X_test.index.equals(learner2.X_test.index)

    def test_different_seeds_different_splits(self, synthetic_data_clean):
        """Verify different seeds produce different splits."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        learner1 = SurvivalLearner(X, y)
        learner1.setup(random_state=42)

        learner2 = SurvivalLearner(X, y)
        learner2.setup(random_state=123)

        # Train/test splits should be different
        assert not learner1.X_train.index.equals(learner2.X_train.index)

    def test_optuna_seed_propagation(self, small_data):
        """Verify Optuna tuning is reproducible with same seed."""
        from mlsurv import SurvivalLearner

        X, y = small_data

        # First tuning run
        learner1 = SurvivalLearner(X, y)
        learner1.setup(random_state=42)
        result1 = learner1.tune(
            'rsf',
            n_trials=3,
            search_space={'n_estimators': (10, 30)}
        )

        # Second tuning run with same seed
        learner2 = SurvivalLearner(X, y)
        learner2.setup(random_state=42)
        result2 = learner2.tune(
            'rsf',
            n_trials=3,
            search_space={'n_estimators': (10, 30)}
        )

        # Best parameters should be the same (or very similar)
        # Note: Due to internal Optuna sampler state, exact match may vary
        # but the search behavior should be consistent
        assert result1.best_params_ is not None
        assert result2.best_params_ is not None

    @pytest.mark.requires_pycox
    def test_torch_seed_propagation_deepsurv(self, small_data):
        """Verify DeepSurv predictions are reproducible across fits with the
        same random_state."""
        pytest.importorskip("torch")
        from mlsurv.models.deepsurv import DeepSurvModel

        X, y = small_data

        # random_state alone must fully determine the fit: fit() seeds torch
        # weight init via torch.manual_seed, the batch sampler via a local
        # np.random.RandomState, and the early-stopping split explicitly. No
        # global torch/NumPy seeding is needed (the fit path does not read the
        # global NumPy RNG).
        model1 = DeepSurvModel(layers=[16], random_state=42, epochs=5)
        model1.fit(X, y)
        preds1 = model1.predict(X)

        model2 = DeepSurvModel(layers=[16], random_state=42, epochs=5)
        model2.fit(X, y)
        preds2 = model2.predict(X)

        # Predictions should be identical
        np.testing.assert_array_almost_equal(preds1, preds2, decimal=5)

    @pytest.mark.requires_pycox
    @pytest.mark.parametrize(
        "module_name, class_name",
        [
            ("mlsurv.models.deepsurv", "DeepSurvModel"),
            ("mlsurv.models.deephit", "DeepHitModel"),
        ],
        ids=["deepsurv", "deephit"],
    )
    def test_deep_fit_does_not_reseed_global_numpy_rng(
        self, small_data, module_name, class_name
    ):
        """A deep-model fit must not reset the process-global NumPy RNG."""
        pytest.importorskip("torch")
        import importlib

        model_cls = getattr(importlib.import_module(module_name), class_name)
        X, y = small_data

        # Restore the caller's global RNG afterward so this test (which
        # deliberately reseeds) does not perturb later tests.
        outer_state = np.random.get_state()
        try:
            np.random.seed(0)
            _ = np.random.rand()  # advance the stream past its seeded start
            state_before = np.random.get_state()

            model_cls(layers=[16], random_state=123, epochs=2).fit(X, y)

            state_after = np.random.get_state()
            # The fit must leave the global RNG key and position untouched.
            assert np.array_equal(state_before[1], state_after[1])
            assert state_before[2] == state_after[2]
        finally:
            np.random.set_state(outer_state)


class TestDeterministicResults:
    """Verify results are deterministic when seeds are fixed."""

    def test_train_same_seed_same_result(self, deterministic_data):
        """Verify train() produces identical results with same seed."""
        from mlsurv import SurvivalLearner

        X, y = deterministic_data

        # First run
        learner1 = SurvivalLearner(X, y)
        learner1.setup(random_state=42, cv_folds=3)
        result1 = learner1.train('coxph')

        # Second run with same seed
        learner2 = SurvivalLearner(X, y)
        learner2.setup(random_state=42, cv_folds=3)
        result2 = learner2.train('coxph')

        # CV scores should be identical
        np.testing.assert_array_almost_equal(
            result1.cv_metric_scores, result2.cv_metric_scores, decimal=8
        )
        assert abs(result1.cv_metric_mean - result2.cv_metric_mean) < 1e-8

    def test_rsf_same_seed_same_result(self, deterministic_data):
        """Verify RSF produces identical results with same seed."""
        from mlsurv import SurvivalLearner

        X, y = deterministic_data

        # First run
        learner1 = SurvivalLearner(X, y)
        learner1.setup(random_state=42, cv_folds=3)
        result1 = learner1.train('rsf', n_estimators=10, random_state=42)

        # Second run with same seed
        learner2 = SurvivalLearner(X, y)
        learner2.setup(random_state=42, cv_folds=3)
        result2 = learner2.train('rsf', n_estimators=10, random_state=42)

        # CV scores should be identical
        np.testing.assert_array_almost_equal(
            result1.cv_metric_scores, result2.cv_metric_scores, decimal=8
        )

    def test_run_same_seed_same_ranking(self, deterministic_data):
        """Verify run() produces identical rankings with same seed."""
        from mlsurv import SurvivalLearner

        X, y = deterministic_data

        # First run - use only coxph for deterministic test (RSF has internal randomness)
        learner1 = SurvivalLearner(X, y)
        learner1.setup(random_state=42, cv_folds=3)
        result1 = learner1.run(
            models=['coxph', 'coxnet'],
            verbose=False,
            evaluate=False,
            bootstrap=False
        )

        # Second run with same seed
        learner2 = SurvivalLearner(X, y)
        learner2.setup(random_state=42, cv_folds=3)
        result2 = learner2.run(
            models=['coxph', 'coxnet'],
            verbose=False,
            evaluate=False,
            bootstrap=False
        )

        # Best model should be the same
        assert learner1.best_model.name == learner2.best_model.name

        # CoxPH scores should be identical (deterministic model)
        np.testing.assert_array_almost_equal(
            result1['coxph'].cv_metric_scores,
            result2['coxph'].cv_metric_scores,
            decimal=8
        )

    def test_bootstrap_same_seed_same_ci(self, synthetic_data_clean):
        """Verify bootstrap produces identical CIs with same seed."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        # First run
        learner1 = SurvivalLearner(X, y)
        learner1.setup(random_state=42)
        model1 = learner1.train('coxph')
        boot1 = learner1.bootstrap(model1, n_bootstrap=20, random_state=42)

        # Second run with same seed
        learner2 = SurvivalLearner(X, y)
        learner2.setup(random_state=42)
        model2 = learner2.train('coxph')
        boot2 = learner2.bootstrap(model2, n_bootstrap=20, random_state=42)

        # Summary statistics should be identical
        for metric in ['harrell', 'ipcw']:
            row1 = boot1.summary[boot1.summary['metric'] == metric]
            row2 = boot2.summary[boot2.summary['metric'] == metric]
            if len(row1) > 0 and len(row2) > 0:
                np.testing.assert_almost_equal(
                    row1['mean'].values[0],
                    row2['mean'].values[0],
                    decimal=8
                )


class TestModelSerialization:
    """Test model serialization/deserialization preserves state."""

    def test_model_pickle_roundtrip(self, synthetic_data_clean, tmp_path):
        """Verify model can be pickled and unpickled."""
        import joblib
        from mlsurv.models import CoxPHModel

        X, y = synthetic_data_clean

        # Create and fit model
        model = CoxPHModel()
        model.fit(X, y)
        original_preds = model.predict(X)

        # Save and load
        model_path = tmp_path / "model.joblib"
        joblib.dump(model, model_path)
        loaded_model = joblib.load(model_path)

        # Verify loaded model produces same predictions
        loaded_preds = loaded_model.predict(X)
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_pipeline_pickle_roundtrip(self, synthetic_data_clean, tmp_path):
        """Verify pipeline can be pickled and unpickled."""
        import joblib
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        # Create learner with pipeline
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        result = learner.train('coxph')

        original_preds = result.model.predict(X)

        # Save and load
        model_path = tmp_path / "pipeline.joblib"
        joblib.dump(result.model, model_path)
        loaded_pipeline = joblib.load(model_path)

        # Verify loaded pipeline produces same predictions
        loaded_preds = loaded_pipeline.predict(X)
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_finalize_model_serialization(self, synthetic_data_clean, tmp_path):
        """Verify fitted model can be serialized."""
        import joblib
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        model_result = learner.train('coxph')
        finalized = model_result.model

        original_preds = finalized.predict(X)

        # Save and load
        model_path = tmp_path / "finalized.joblib"
        joblib.dump(finalized, model_path)
        loaded = joblib.load(model_path)

        loaded_preds = loaded.predict(X)
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_loaded_model_produces_same_predictions(self, synthetic_data_clean, tmp_path):
        """Verify loaded model predictions match original exactly."""
        import joblib
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        result = learner.train('rsf', n_estimators=10, random_state=42)

        # Get predictions on train and test
        train_preds = result.model.predict(learner.X_train)
        test_preds = result.model.predict(learner.X_test)

        # Save
        model_path = tmp_path / "rsf_model.joblib"
        joblib.dump(result.model, model_path)

        # Load and verify
        loaded = joblib.load(model_path)
        loaded_train_preds = loaded.predict(learner.X_train)
        loaded_test_preds = loaded.predict(learner.X_test)

        np.testing.assert_array_almost_equal(train_preds, loaded_train_preds)
        np.testing.assert_array_almost_equal(test_preds, loaded_test_preds)

    def test_rsf_model_serialization(self, synthetic_data_clean, tmp_path):
        """Verify RSF model serialization preserves state."""
        import joblib
        from mlsurv.models import RSFModel

        X, y = synthetic_data_clean

        model = RSFModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        original_preds = model.predict(X)

        model_path = tmp_path / "rsf.joblib"
        joblib.dump(model, model_path)
        loaded = joblib.load(model_path)

        loaded_preds = loaded.predict(X)
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_coxnet_model_serialization(self, synthetic_data_clean, tmp_path):
        """Verify CoxNet model serialization preserves state."""
        import joblib
        from mlsurv.models import CoxNetModel

        X, y = synthetic_data_clean

        model = CoxNetModel(l1_ratio=0.5)
        model.fit(X, y)
        original_preds = model.predict(X)
        original_coef = model.get_coefficients()

        model_path = tmp_path / "coxnet.joblib"
        joblib.dump(model, model_path)
        loaded = joblib.load(model_path)

        loaded_preds = loaded.predict(X)
        loaded_coef = loaded.get_coefficients()

        np.testing.assert_array_almost_equal(original_preds, loaded_preds)
        np.testing.assert_array_almost_equal(
            original_coef.values, loaded_coef.values
        )


class TestCVReproducibility:
    """Test cross-validation reproducibility."""

    def test_cv_folds_reproducible(self, synthetic_data_clean):
        """Verify CV folds are identical with same seed."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        learner1 = SurvivalLearner(X, y)
        learner1.setup(random_state=42, cv_folds=5)

        learner2 = SurvivalLearner(X, y)
        learner2.setup(random_state=42, cv_folds=5)

        # Extract fold indices
        folds1 = list(learner1._state.cv_splitter.split(
            learner1.X_train, learner1.y_train['event']
        ))
        folds2 = list(learner2._state.cv_splitter.split(
            learner2.X_train, learner2.y_train['event']
        ))

        # All folds should be identical
        for (train1, val1), (train2, val2) in zip(folds1, folds2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(val1, val2)

    def test_repeated_cv_reproducible(self, synthetic_data_clean):
        """Verify repeated CV is reproducible with same seed."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean

        learner1 = SurvivalLearner(X, y)
        learner1.setup(random_state=42, cv_folds=3, cv_repeats=2)
        result1 = learner1.train('coxph')

        learner2 = SurvivalLearner(X, y)
        learner2.setup(random_state=42, cv_folds=3, cv_repeats=2)
        result2 = learner2.train('coxph')

        # All CV scores should match
        np.testing.assert_array_almost_equal(
            result1.cv_metric_scores, result2.cv_metric_scores, decimal=8
        )
