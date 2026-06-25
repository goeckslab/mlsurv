"""Tests for newly added models: XGBoost, DeepHit, Weibull AFT."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from mlsurv._warnings import SuggestionWarning


class TestXGBCoxModel:
    """Tests for XGBoost-Cox model."""

    def test_fit_predict(self, train_test_data):
        """XGBoost model can fit and predict."""
        pytest.importorskip("xgboost")
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('xgbcox')
        model.fit(X_train, y_train)

        # Predict risk scores
        risk_scores = model.predict(X_test)

        assert len(risk_scores) == len(X_test)
        assert not np.any(np.isnan(risk_scores))

    def test_predict_survival_function(self, train_test_data):
        """XGBoost can predict survival functions."""
        pytest.importorskip("xgboost")
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('xgbcox')
        model.fit(X_train, y_train)

        # Predict survival functions
        surv_funcs = model.predict_survival_function(X_test)

        assert len(surv_funcs) == len(X_test)

    def test_score(self, train_test_data):
        """XGBoost score method returns reasonable C-index."""
        pytest.importorskip("xgboost")
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('xgbcox')
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)

        assert 0.4 < score < 1.0

    def test_serialization(self, train_test_data, tmp_path):
        """XGBoost model can be saved and loaded."""
        pytest.importorskip("xgboost")
        import joblib
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('xgbcox', n_estimators=50)
        model.fit(X_train, y_train)

        # Save model
        model_path = tmp_path / "xgbcox.joblib"
        joblib.dump(model, model_path)

        # Load model
        loaded = joblib.load(model_path)

        # Predictions should match
        pred_original = model.predict(X_test)
        pred_loaded = loaded.predict(X_test)

        assert_array_almost_equal(pred_original, pred_loaded)

    def test_predict_survival_function_with_times(self, train_test_data):
        """XGBoost predict_survival_function(times=...) returns aligned ndarray."""
        pytest.importorskip("xgboost")
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        # Use low learning_rate to prevent overfitting on synthetic data
        # (default lr=0.3 produces extreme risk scores that overflow Breslow estimation)
        model = get_model('xgbcox', n_estimators=50, max_depth=3,
                          learning_rate=0.01, random_state=42)
        model.fit(X_train, y_train)

        # Times must be within training data range (times clipped to [0.1, 10])
        times = np.array([0.5, 2.0, 5.0, 8.0])
        result = model.predict_survival_function(X_test, times=times)

        # Should return ndarray (n_samples, n_times)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(X_test), len(times))
        # Probabilities should be finite and in [0, 1]
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_predict_survival_function_times_matches_manual(self, train_test_data):
        """times= ndarray matches manual StepFunction evaluation."""
        pytest.importorskip("xgboost")
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('xgbcox', n_estimators=50, max_depth=3,
                          learning_rate=0.01, random_state=42)
        model.fit(X_train, y_train)

        times = np.array([0.5, 2.0, 5.0])

        # Method 1: use times parameter
        result_times = model.predict_survival_function(X_test, times=times)

        # Method 2: manual StepFunction evaluation
        surv_funcs = model.predict_survival_function(X_test)
        result_manual = np.array([[fn(t) for t in times] for fn in surv_funcs])

        np.testing.assert_allclose(result_times, result_manual)

    def test_predict_cumulative_hazard_function_with_times(self, train_test_data):
        """XGBoost predict_cumulative_hazard_function(times=...) returns aligned ndarray."""
        pytest.importorskip("xgboost")
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('xgbcox', n_estimators=50, max_depth=3,
                          learning_rate=0.01, random_state=42)
        model.fit(X_train, y_train)

        # Times must be within training data range (times clipped to [0.1, 10])
        times = np.array([0.5, 2.0, 5.0, 8.0])
        result = model.predict_cumulative_hazard_function(X_test, times=times)

        assert isinstance(result, np.ndarray)
        assert result.shape == (len(X_test), len(times))
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)

    def test_njobs_parameter(self):
        """XGBoost respects n_jobs parameter."""
        pytest.importorskip("xgboost")
        import warnings
        from mlsurv.models import get_model

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"^XGBoost with n_jobs != 1", category=UserWarning)
            model4 = get_model('xgbcox', n_jobs=4)
            model_all = get_model('xgbcox', n_jobs=-1)

        assert model4.n_jobs == 4
        assert model_all.n_jobs == -1


class TestXGBSurvivalDefaults:
    """Survival-appropriate defaults for XGBCox / XGBAFT (shallow, low LR)."""

    @pytest.mark.parametrize("class_name", ["XGBCoxModel", "XGBAFTModel"])
    def test_constructor_defaults(self, class_name):
        """The model constructor ships shallow-tree, low-LR defaults."""
        pytest.importorskip("xgboost")
        import mlsurv.models

        model = getattr(mlsurv.models, class_name)()
        assert model.max_depth == 3
        assert model.learning_rate == 0.10
        assert model.n_estimators == 300

    @pytest.mark.parametrize("name", ["xgbcox", "xgbaft"])
    def test_no_dynamic_default_shadowing(self, name):
        """The untuned (data-aware) path no longer overrides the new defaults.

        xgbcox/xgbaft were removed from the auto-defaults registry, so passing
        data characteristics must not inject the old EPV-tiered depth / lr=0.3.
        """
        pytest.importorskip("xgboost")
        from mlsurv.models import get_model

        model = get_model(name, n_samples=100, n_features=10, n_events=60)
        assert model.max_depth == 3
        assert model.learning_rate == 0.10
        assert model.n_estimators == 300

    @pytest.mark.parametrize("name", ["xgbcox", "xgbaft"])
    def test_constructor_defaults_within_search_space(self, name):
        """New constructor defaults must stay inside the tuning search space.

        Guards the CONTRIBUTING.md invariant that defaults fall within
        DEFAULT_SPACES; the deleted auto_*_defaults tests used to cover this.
        """
        pytest.importorskip("xgboost")
        from mlsurv.models import get_model
        from mlsurv.tuning.search_spaces import DEFAULT_SPACES

        model = get_model(name)
        space = DEFAULT_SPACES[name]
        for param in ("n_estimators", "max_depth", "learning_rate"):
            low, high = space[param][0], space[param][1]
            value = getattr(model, param)
            assert low <= value <= high, (
                f"{name}.{param}={value} outside DEFAULT_SPACES range ({low}, {high})"
            )


class TestDeepHitModel:
    """Tests for DeepHit model."""

    def test_fit_predict(self, train_test_data):
        """DeepHit model can fit and predict."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('deephit', layers=[16], epochs=10)
        model.fit(X_train, y_train)

        # Predict risk scores
        risk_scores = model.predict(X_test)

        assert len(risk_scores) == len(X_test)
        assert not np.any(np.isnan(risk_scores))

    def test_predict_survival_function(self, train_test_data):
        """DeepHit can predict survival functions."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('deephit', layers=[16], epochs=10)
        model.fit(X_train, y_train)

        # Predict survival functions
        surv_df = model.predict_survival_function(X_test)

        assert surv_df.shape[1] == len(X_test)  # columns = samples

    def test_score(self, train_test_data):
        """DeepHit score method returns reasonable C-index."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('deephit', layers=[16], epochs=10)
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)

        # DeepHit with few epochs may have lower performance
        assert 0.3 < score < 1.0

    def test_serialization(self, train_test_data, tmp_path):
        """DeepHit model predictions are consistent after fitting."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('deephit', layers=[16], epochs=10, patience=None, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions twice - should be identical
        pred1 = model.predict(X_test)
        pred2 = model.predict(X_test)

        assert_array_almost_equal(pred1, pred2, decimal=5)

    def test_early_stopping_uses_validation(self, train_test_data):
        """DeepHit with patience splits data and produces valid predictions."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model(
            'deephit', layers=[16], epochs=200, patience=5,
            val_fraction=0.2, random_state=42,
        )
        model.fit(X_train, y_train)

        assert model.is_fitted_
        risk_scores = model.predict(X_test)
        assert len(risk_scores) == len(X_test)
        assert not np.any(np.isnan(risk_scores))

        # Verify early stopping actually fired (trained < max epochs)
        log = model.model_.log.to_pandas()
        assert len(log) < 200, (
            f"Expected early stopping before epoch 200, "
            f"but trained for {len(log)} epochs"
        )

    def test_no_val_split_when_patience_none(self, train_test_data):
        """patience=None disables early stopping and val split entirely."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model(
            'deephit', layers=[16], epochs=10, patience=None, random_state=42,
        )
        model.fit(X_train, y_train)

        assert model.is_fitted_
        risk_scores = model.predict(X_test)
        assert not np.any(np.isnan(risk_scores))

    def test_small_dataset_disables_early_stopping(self):
        """Tiny dataset emits warning and disables early stopping."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv.models import get_model

        # 8 samples with only 1 event: min(events, censored) < 2
        rng = np.random.RandomState(42)
        X_tiny = rng.randn(8, 3)
        y_tiny = np.array(
            [(True, 1.0), (False, 2.0), (False, 3.0), (False, 4.0),
             (False, 5.0), (False, 6.0), (False, 7.0), (False, 8.0)],
            dtype=[('event', bool), ('time', float)],
        )

        model = get_model(
            'deephit', layers=[8], epochs=5, patience=3,
            val_fraction=0.2, random_state=42,
        )

        with pytest.warns(SuggestionWarning, match="Dataset too small"):
            model.fit(X_tiny, y_tiny)

        assert model.is_fitted_

    def test_val_fraction_invalid_raises(self):
        """val_fraction=0.0 or 1.0 raises ValueError."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv.models import get_model

        rng = np.random.RandomState(0)
        X = rng.randn(50, 3)
        y = np.array(
            [(bool(i % 2), float(i + 1)) for i in range(50)],
            dtype=[('event', bool), ('time', float)],
        )

        for bad_frac in (0.0, 1.0, -0.5, 1.5):
            model = get_model(
                'deephit', layers=[8], epochs=5, patience=3,
                val_fraction=bad_frac,
            )
            with pytest.raises(ValueError, match="val_fraction must be in"):
                model.fit(X, y)

    def test_stratified_split_failure_fallback(self, train_test_data):
        """Stratified split ValueError falls back to no early stopping."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from unittest.mock import patch
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model(
            'deephit', layers=[8], epochs=5, patience=3,
            val_fraction=0.2, random_state=42,
        )

        with patch(
            'sklearn.model_selection.train_test_split',
            side_effect=ValueError(
                "The least populated class in y has only 1 member"
            ),
        ):
            with pytest.warns(SuggestionWarning, match="Stratified val split failed"):
                model.fit(X_train, y_train)

        assert model.is_fitted_
        risk_scores = model.predict(X_test)
        assert not np.any(np.isnan(risk_scores))


    def test_checkpoint_cleanup_after_fit(self, train_test_data):
        """Checkpoint .pt files are cleaned up after fit with early stopping."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        import tempfile
        from pathlib import Path
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        ckpt_dir = Path(tempfile.gettempdir()) / "mlsurv_checkpoints"
        before = set(ckpt_dir.glob("deephit_*.pt")) if ckpt_dir.exists() else set()

        model = get_model(
            'deephit', layers=[16], epochs=10, patience=3,
            val_fraction=0.2, random_state=42,
        )
        model.fit(X_train, y_train)

        after = set(ckpt_dir.glob("deephit_*.pt")) if ckpt_dir.exists() else set()
        new_files = after - before
        assert new_files == set(), f"Orphan checkpoint files: {new_files}"

    def test_checkpoint_cleanup_on_training_failure(self, train_test_data):
        """Checkpoint .pt files are cleaned up even when fit() raises."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        ckpt_dir = Path(tempfile.gettempdir()) / "mlsurv_checkpoints"
        before = set(ckpt_dir.glob("deephit_*.pt")) if ckpt_dir.exists() else set()

        model = get_model(
            'deephit', layers=[16], epochs=10, patience=3,
            val_fraction=0.2, random_state=42,
        )

        with patch(
            'pycox.models.DeepHitSingle.fit',
            side_effect=RuntimeError("simulated crash"),
        ):
            with pytest.raises(RuntimeError, match="simulated crash"):
                model.fit(X_train, y_train)

        after = set(ckpt_dir.glob("deephit_*.pt")) if ckpt_dir.exists() else set()
        new_files = after - before
        assert new_files == set(), f"Orphan checkpoint files: {new_files}"

    def test_deephit_activation_param(self):
        """activation param is in get_params() with default 'selu'."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv.models.deephit import DeepHitModel

        model = DeepHitModel()
        params = model.get_params()
        assert 'activation' in params
        assert params['activation'] == 'selu'

    def test_deephit_alpha_default(self):
        """alpha defaults to 0.0 (pure ranking, C^td-optimal).

        DeepHit is treated as a rank-based model: at the default fit it
        optimizes the ranking loss, then ``train('deephit', calibrate='isotonic')``
        applies post-hoc isotonic for calibrated probabilities.
        """
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv.models.deephit import DeepHitModel

        model = DeepHitModel()
        params = model.get_params()
        assert params['alpha'] == 0.0

    def test_deephit_rank_based(self):
        """DeepHitModel.rank_based is True.

        Locks the rank-based classification both at the class level (used by
        the report and warning logic) and on instances (defensive — class
        attributes are accessible on instances unless shadowed).
        """
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv.models.deephit import DeepHitModel

        assert DeepHitModel.rank_based is True
        assert DeepHitModel().rank_based is True

    def test_deephit_invalid_activation(self, train_test_data):
        """Fitting with an unsupported activation raises ValueError."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv.models.deephit import DeepHitModel

        X_train, X_test, y_train, y_test = train_test_data

        model = DeepHitModel(layers=[8], epochs=1, activation='tanh', patience=None)
        with pytest.raises(ValueError, match="Unknown activation"):
            model.fit(X_train, y_train)

    def test_deephit_weight_decay_param(self):
        """weight_decay param is in get_params() with default 0.01."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv.models.deephit import DeepHitModel

        model = DeepHitModel()
        params = model.get_params()
        assert 'weight_decay' in params
        assert params['weight_decay'] == 0.01

    def test_deephit_gradient_clip_param(self):
        """gradient_clip param is in get_params() with default 1.0."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv.models.deephit import DeepHitModel

        model = DeepHitModel()
        params = model.get_params()
        assert 'gradient_clip' in params
        assert params['gradient_clip'] == 1.0

    def test_deephit_gradient_clip_none(self):
        """gradient_clip=None is accepted and stored."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv.models.deephit import DeepHitModel

        model = DeepHitModel(gradient_clip=None)
        params = model.get_params()
        assert params['gradient_clip'] is None

    def test_deephit_predict_returns_expected_survival_time(self, train_test_data):
        """predict() returns negative expected survival time (not median)."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv.models.deephit import DeepHitModel

        X_train, X_test, y_train, y_test = train_test_data

        model = DeepHitModel(layers=[16], epochs=10, patience=None, random_state=42)
        model.fit(X_train, y_train)

        risk_scores = model.predict(X_test)

        # Risk scores should be negative expected survival times (continuous)
        assert np.all(risk_scores < 0), "Risk scores should be negative (neg expected time)"
        # Expected survival times are continuous -- should have few ties
        n_unique = len(np.unique(risk_scores))
        assert n_unique > len(risk_scores) * 0.5, (
            f"Expected mostly unique risk scores (continuous), "
            f"got {n_unique}/{len(risk_scores)} unique"
        )

    def test_deephit_batchnorm_safe_batch_size(self):
        """batch_size+1 fix prevents BatchNorm crash when n_train % batch_size == 1."""
        pytest.importorskip("torch")
        pytest.importorskip("pycox")

        from mlsurv.models.deephit import DeepHitModel

        rng = np.random.RandomState(42)
        # Create data where n_train % batch_size == 1
        # With batch_size=32 and n_samples=65, n_train=65 % 32 == 1
        n_samples = 65
        X = rng.randn(n_samples, 5).astype(np.float32)
        y = np.array(
            [(bool(i % 2), float(i + 1)) for i in range(n_samples)],
            dtype=[('event', bool), ('time', float)],
        )

        model = DeepHitModel(
            layers=[8], epochs=2, batch_size=32,
            patience=None, random_state=42,
        )
        # Should not raise RuntimeError from BatchNorm
        model.fit(X, y)
        assert model.is_fitted_


class TestWeibullAFTModel:
    """Tests for Weibull AFT model."""

    def test_fit_predict(self, train_test_data):
        """Weibull AFT model can fit and predict."""
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('weibull_aft')
        model.fit(X_train, y_train)

        # Predict risk scores (negative log survival times)
        risk_scores = model.predict(X_test)

        assert len(risk_scores) == len(X_test)
        assert not np.any(np.isnan(risk_scores))

    def test_predict_survival_time(self, train_test_data):
        """Weibull AFT can predict survival times."""
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('weibull_aft')
        model.fit(X_train, y_train)

        # Predict median survival time
        median_times = model.predict_survival_time(X_test, quantile=0.5)

        assert len(median_times) == len(X_test)
        assert np.all(median_times > 0)  # Survival times should be positive

    def test_predict_survival_function(self, train_test_data):
        """Weibull AFT returns a DataFrame with time index and sample columns."""
        import pandas as pd
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('weibull_aft')
        model.fit(X_train, y_train)

        # Predict survival functions
        surv_df = model.predict_survival_function(X_test)

        assert isinstance(surv_df, pd.DataFrame)
        # DataFrame: time as index, samples as columns
        assert surv_df.shape[1] == len(X_test)
        assert surv_df.index.dtype in [np.float64, np.float32, np.int64]
        assert np.all((surv_df.values >= 0) & (surv_df.values <= 1))

    def test_predict_survival_function_with_times(self, train_test_data):
        """Passing explicit times returns a DataFrame indexed by those times."""
        import pandas as pd
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('weibull_aft')
        model.fit(X_train, y_train)

        times = np.array([1.0, 5.0, 10.0, 20.0])
        surv_df = model.predict_survival_function(X_test, times=times)

        assert isinstance(surv_df, pd.DataFrame)
        assert_array_almost_equal(surv_df.index.values, times)
        assert surv_df.shape[1] == len(X_test)

    def test_predict_cumulative_hazard_function(self, train_test_data):
        """Cumulative hazard returns DataFrame with time index."""
        import pandas as pd
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('weibull_aft')
        model.fit(X_train, y_train)

        chf = model.predict_cumulative_hazard_function(X_test)

        assert isinstance(chf, pd.DataFrame)
        assert chf.shape[1] == len(X_test)
        assert (chf.values >= 0).all()

    def test_brier_scoring_compatibility(self, train_test_data):
        """WeibullAFT survival output works with _brier_score_at_times."""
        from mlsurv.models import get_model
        from mlsurv.evaluation.metrics import _brier_score_at_times

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('weibull_aft')
        model.fit(X_train, y_train)

        surv_funcs = model.predict_survival_function(X_test)

        # Use eval_times within the observed time range
        event_times = y_test['time']
        eval_times = np.linspace(
            np.quantile(event_times, 0.10),
            np.quantile(event_times, 0.90),
            5,
        )

        # This raised ValueError before the fix
        scores = _brier_score_at_times(y_train, y_test, surv_funcs, eval_times)

        assert len(scores) == len(eval_times)
        assert all(0 <= s <= 1 for s in scores)

    def test_score(self, train_test_data):
        """Weibull AFT score method returns reasonable C-index."""
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('weibull_aft')
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)

        assert 0.4 < score < 1.0

    def test_convergence_fallback_warning(self, train_test_data):
        """WeibullAFT emits warning and retries with penalizer=0.01 on convergence failure."""
        import warnings
        from unittest.mock import patch
        from lifelines import WeibullAFTFitter
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data
        model = get_model('weibull_aft')

        real_fit = WeibullAFTFitter.fit
        call_count = [0]

        def side_effect(self_fitter, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Simulated convergence failure")
            return real_fit(self_fitter, *args, **kwargs)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch.object(WeibullAFTFitter, 'fit', autospec=True,
                              side_effect=side_effect):
                model.fit(X_train, y_train)

        fallback_warnings = [x for x in w if "penalizer=0.01" in str(x.message)]
        assert len(fallback_warnings) == 1
        assert model.is_fitted_
        assert call_count[0] == 2, "Expected 2 calls to fit: first fails, second succeeds"

    def test_convergence_failure_with_penalizer_propagates(self, train_test_data):
        """WeibullAFT with non-zero penalizer does NOT retry — exception propagates."""
        from unittest.mock import patch
        from lifelines import WeibullAFTFitter
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data
        model = get_model('weibull_aft', alpha=0.1)

        with patch.object(WeibullAFTFitter, 'fit', autospec=True,
                          side_effect=ValueError("convergence failure")):
            with pytest.raises(ValueError, match="convergence failure"):
                model.fit(X_train, y_train)

    def test_serialization(self, train_test_data, tmp_path):
        """Weibull AFT model can be saved and loaded."""
        import joblib
        from mlsurv.models import get_model

        X_train, X_test, y_train, y_test = train_test_data

        model = get_model('weibull_aft', alpha=0.01)
        model.fit(X_train, y_train)

        # Save model
        model_path = tmp_path / "weibull_aft.joblib"
        joblib.dump(model, model_path)

        # Load model
        loaded = joblib.load(model_path)

        # Predictions should match
        pred_original = model.predict(X_test)
        pred_loaded = loaded.predict(X_test)

        assert_array_almost_equal(pred_original, pred_loaded)


    def test_column_name_collision_raises(self, train_test_data):
        """WeibullAFT raises ValueError if X has 'duration' or 'event' columns."""
        from mlsurv.models import get_model

        X_train, _, y_train, _ = train_test_data

        # Add a column named 'duration' to X
        X_collision = X_train.copy()
        X_collision['duration'] = 1.0

        model = get_model('weibull_aft')
        with pytest.raises(ValueError, match="reserved column name"):
            model.fit(X_collision, y_train)

        # Also test 'event' collision
        X_collision2 = X_train.copy()
        X_collision2['event'] = 1

        with pytest.raises(ValueError, match="reserved column name"):
            model.fit(X_collision2, y_train)


class TestWeibullAFTReturnArray:
    """Test WeibullAFT return_array parameter for survival/CHF predictions."""

    def test_predict_survival_return_array(self, train_test_data):
        """return_array=True produces (n_samples, n_times) ndarray."""
        from mlsurv.models import get_model

        X_train, X_test, y_train, _ = train_test_data
        model = get_model('weibull_aft')
        model.fit(X_train, y_train)

        surv = model.predict_survival_function(X_test, return_array=True)
        assert isinstance(surv, np.ndarray)
        assert surv.shape[0] == len(X_test)
        assert surv.ndim == 2
        assert np.all((surv >= 0) & (surv <= 1))

    def test_predict_cumulative_hazard_return_array(self, train_test_data):
        """return_array=True produces (n_samples, n_times) ndarray for CHF."""
        from mlsurv.models import get_model

        X_train, X_test, y_train, _ = train_test_data
        model = get_model('weibull_aft')
        model.fit(X_train, y_train)

        chf = model.predict_cumulative_hazard_function(X_test, return_array=True)
        assert isinstance(chf, np.ndarray)
        assert chf.shape[0] == len(X_test)
        assert chf.ndim == 2
        assert np.all(chf >= 0)

    def test_cumulative_hazard_empty_timeline_raises(self, train_test_data):
        """Empty model timeline raises ValueError for CHF (mirrors survival)."""
        from mlsurv.models import get_model

        X_train, X_test, y_train, _ = train_test_data
        model = get_model('weibull_aft')
        model.fit(X_train, y_train)
        # Simulate empty timeline
        model.model_.timeline = np.array([])

        with pytest.raises(ValueError, match="Model timeline is empty"):
            model.predict_cumulative_hazard_function(X_test)


# TestNewModelsIntegration (list_models / get_model for the newer models) was
# folded into test_phase2_models.py::TestModelRegistry — the registry is
# global and model-agnostic, so those assertions belong with the other registry
# tests rather than duplicated here.
