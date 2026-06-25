"""Tests for the learner config/persistence/studies mixins and helpers.

Covers the config property (ConfigMixin), the _environment_fingerprint()
helper, save_artifact() / load_artifact() roundtrip (PersistenceMixin),
list_studies() and the experiment_path property (StudyMixin), and the
CV-vs-holdout comparison helper (TrainingMixin).
These are core user-facing APIs - incorrect behavior leads to lost
model state or misleading results.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from mlsurv._warnings import InternalWarning
from mlsurv.learner import SurvivalLearner


class TestGetConfig:
    """Tests for the config property return structure and completeness."""

    def test_get_config_returns_expected_keys(self, small_data):
        """config should return all documented configuration keys."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        config = learner.config

        expected_keys = {
            'test_fraction', 'cv_folds', 'cv_repeats', 'cv_ci',
            'cv_strategy', 'stratify', 'groups',
            'random_state', 'n_jobs', 'presplit',
            'scoring', 'imputer', 'scaler',
            'feature_selector', 'n_train', 'n_test', 'n_features',
            'n_events_train', 'tuned_models', 'experiment_name', 'device',
            'time_unit', 'outcome_name',
            'check_leakage', 'ph_test', 'linearity_test',
            'stratify_subpopulations', 'subpopulations_declared',
            'categorical_groups_declared', 'categorical_features_declared',
            'feature_groups_declared',
            'report_eval_times',
        }
        assert set(config.keys()) == expected_keys

    def test_get_config_values_after_setup(self, small_data):
        """Config values should reflect setup parameters."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, random_state=42, scoring='harrell')

        config = learner.config

        assert config['test_fraction'] == 0.3
        assert config['random_state'] == 42
        assert config['scoring'] == 'harrell'
        assert config['n_features'] == 5  # small_data has 5 features
        assert config['n_train'] is not None
        assert config['n_test'] is not None
        assert config['tuned_models'] == []
        assert config['n_jobs'] == 1
        assert config['presplit'] is False
        assert config['check_leakage'] is True
        assert config['ph_test'] is True
        assert config['linearity_test'] is True
        assert config['stratify'] == 'event'
        assert config['groups'] == 'none'
        assert config['subpopulations_declared'] is False
        assert config['categorical_groups_declared'] is False
        assert config['categorical_features_declared'] is False
        assert config['feature_groups_declared'] is False

    def test_get_config_diagnostic_flags_non_default(self, small_data):
        """Non-default diagnostic flags should round-trip through config."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(
            random_state=42,
            check_leakage=False,
            ph_test=False,
            linearity_test=False,
            stratify_subpopulations=False,
        )

        config = learner.config

        assert config['check_leakage'] is False
        assert config['ph_test'] is False
        assert config['linearity_test'] is False
        assert config['stratify_subpopulations'] is False

    def test_get_config_subpopulation_declared_flags(self, small_data):
        """Declared subpopulation/feature group flags should be True bools."""
        import pandas as pd
        X, y = small_data
        subpops = {'cohort': pd.Series(['A'] * len(X), index=X.index)}
        learner = SurvivalLearner(X, y)
        learner.setup(
            random_state=42,
            subpopulations=subpops,
            categorical_groups={'stage': ['stage']},
            feature_groups={c: 'g1' for c in X.columns},
        )

        config = learner.config

        assert config['subpopulations_declared'] is True
        assert config['categorical_groups_declared'] is True
        assert config['feature_groups_declared'] is True

    def test_get_config_state_setstate_forward_compat(self):
        """LearnerState.__setstate__ should backfill missing diagnostic flags."""
        from mlsurv.learner._state import LearnerState

        # Simulate an old pickled state that lacks the new fields. Drop both a
        # scalar-default field (the diagnostic flags) and a default_factory
        # field (``models``) so both branches of the backfill loop are covered.
        state = LearnerState()
        legacy_dict = {
            k: v for k, v in state.__dict__.items()
            if k not in ('check_leakage', 'ph_test_enabled',
                         'linearity_test_enabled', 'models')
        }
        restored = LearnerState.__new__(LearnerState)
        restored.__setstate__(legacy_dict)

        assert restored.check_leakage is True
        assert restored.ph_test_enabled is True
        assert restored.linearity_test_enabled is True
        assert restored.models == {}

    def test_get_config_n_events_train(self, small_data):
        """n_events_train should count events in training set."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        config = learner.config
        assert isinstance(config['n_events_train'], int)
        assert config['n_events_train'] > 0


class TestGetExperimentPath:
    """Tests for the experiment_path property directory resolution."""

    def test_returns_none_before_tuning(self, small_data):
        """Should return None if no tuning has been performed."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        assert learner.experiment_path is None

    def test_returns_path_after_experiment_name_set(self, small_data):
        """Should return a Path when experiment name is set."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()
        learner._state.experiment_name = "test_experiment"

        path = learner.experiment_path
        assert path is not None
        assert isinstance(path, Path)
        assert "test_experiment" in str(path)


class TestListStudies:
    """Tests for list_studies() scoped to the calling learner's experiment."""

    def test_empty_when_no_experiment(self, small_data, monkeypatch):
        """No experiment name set -> empty list, short-circuits before any I/O."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()
        assert learner._state.experiment_name is None

        # The short-circuit must return before delegating to the module-level
        # function, so make that function fail if it is ever reached.
        def _should_not_run(storage=None):
            pytest.fail("list_studies should short-circuit before scanning")

        monkeypatch.setattr(
            "mlsurv.learner._studies.list_studies", _should_not_run
        )

        result = learner.list_studies()
        assert result == []

    def test_scoped_to_own_experiment(self, small_data, monkeypatch):
        """Returns only this learner's experiment, as a flat list."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()
        learner._state.experiment_name = "expA"

        monkeypatch.setattr(
            "mlsurv.learner._studies.list_studies",
            lambda storage=None: {
                "expA": ["rsf_study.db", "gbsa_study.db"],
                "expB": ["coxph_study.db"],
            },
        )

        result = learner.list_studies()
        assert result == ["rsf_study.db", "gbsa_study.db"]
        assert "coxph_study.db" not in result

    def test_empty_when_experiment_absent(self, small_data, monkeypatch):
        """Experiment name not present on disk -> empty list."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()
        learner._state.experiment_name = "missing_exp"

        monkeypatch.setattr(
            "mlsurv.learner._studies.list_studies",
            lambda storage=None: {"expA": ["rsf_study.db"]},
        )

        assert learner.list_studies() == []

    def test_storage_passthrough(self, small_data, monkeypatch):
        """The storage arg is forwarded to the module-level function."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()
        learner._state.experiment_name = "expA"

        captured = {}

        def fake_list_studies(storage=None):
            captured["storage"] = storage
            return {"expA": [{"name": "rsf_tuning", "n_trials": 50}]}

        monkeypatch.setattr(
            "mlsurv.learner._studies.list_studies", fake_list_studies
        )

        # Explicit URL is forwarded.
        result = learner.list_studies(storage="postgresql://localhost/optuna")
        assert captured["storage"] == "postgresql://localhost/optuna"
        assert result == [{"name": "rsf_tuning", "n_trials": 50}]

        # Default forwards storage=None (file-scan mode).
        learner.list_studies()
        assert captured["storage"] is None


class TestArtifactPathTracking:
    """Tests for artifact_path tracking in LearnerState."""

    def test_save_artifact_stores_path_in_state(self, small_data, tmp_path):
        """save_artifact() should set _state.artifact_path."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(['coxph'])

        model_result = learner._state.models['coxph']
        artifact_dir = tmp_path / "my_artifact"

        learner.save_artifact(model_result, artifact_dir, step='train')

        assert learner._state.artifact_path is not None
        assert learner._state.artifact_path == str(artifact_dir.resolve())

    def test_artifact_path_overwrite_warns(self, small_data, tmp_path):
        """Second save to a different path should warn about overwrite."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(['coxph'])

        model_result = learner._state.models['coxph']

        # First save — no warning expected
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner.save_artifact(model_result, tmp_path / "artifact1", step='train')
        overwrite_warnings = [w for w in caught if "artifact_path is being overwritten" in str(w.message)]
        assert not overwrite_warnings, "First save should not warn about overwrite"

        # Second save to different path — should warn about overwrite
        with pytest.warns(InternalWarning, match="artifact_path is being overwritten"):
            learner.save_artifact(model_result, tmp_path / "artifact2", step='train')

        # Final path should point to the second artifact
        assert learner._state.artifact_path == str((tmp_path / "artifact2").resolve())

    def test_artifact_path_same_path_no_warn(self, small_data, tmp_path):
        """Re-saving to the same path should not warn."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(['coxph'])

        model_result = learner._state.models['coxph']
        artifact_dir = tmp_path / "artifact1"

        learner.save_artifact(model_result, artifact_dir, step='train')

        # Same path again — no warning
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner.save_artifact(model_result, artifact_dir, step='train')
        overwrite_warnings = [w for w in caught if "artifact_path is being overwritten" in str(w.message)]
        assert not overwrite_warnings, "Same-path re-save should not warn"


class TestCheckCvVsHoldoutBest:
    """Tests for _check_cv_vs_holdout_best() informational message."""

    def test_no_message_when_verbose_false(self, small_data, capsys):
        """Should not print when verbose=False."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        capsys.readouterr()  # flush setup summary echo

        mock_result = MagicMock()
        mock_result.evaluation = None

        learner._check_cv_vs_holdout_best(mock_result, verbose=False)
        assert capsys.readouterr().out == ""

    def test_no_message_when_no_evaluation(self, small_data, capsys):
        """Should not print when result has no evaluation."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        capsys.readouterr()  # flush setup summary echo

        mock_result = MagicMock()
        mock_result.evaluation = None

        learner._check_cv_vs_holdout_best(mock_result, verbose=True)
        assert capsys.readouterr().out == ""

    def test_no_message_when_no_holdout_scores(self, small_data, capsys):
        """Should not print when models have no holdout C-index."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        capsys.readouterr()  # flush setup summary echo

        # Models with None c_index_test
        model1 = MagicMock()
        model1.evaluation = MagicMock()
        model1.evaluation.c_index_test = None
        models = {'coxph': model1}

        learner._check_cv_vs_holdout_best(models, verbose=True)
        assert capsys.readouterr().out == ""

    def test_no_message_when_holdout_agrees(self, small_data, capsys):
        """Should not print when holdout best matches CV best."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        capsys.readouterr()  # flush setup summary echo

        # Both CV and holdout agree: rsf is best
        model1 = MagicMock()
        model1.evaluation = MagicMock()
        model1.evaluation.c_index_test = 0.72  # Holdout winner
        model1.cv_metric_mean = 0.72

        model2 = MagicMock()
        model2.evaluation = MagicMock()
        model2.evaluation.c_index_test = 0.65
        model2.cv_metric_mean = 0.68

        models = {'rsf': model1, 'coxph': model2}

        learner._check_cv_vs_holdout_best(models, verbose=True)
        assert capsys.readouterr().out == ""

    def test_message_when_holdout_differs(self, small_data, capsys):
        """Should print note when holdout best differs from CV best."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        capsys.readouterr()  # flush setup summary echo

        # CV best is 'coxph' but holdout best is 'rsf'
        model1 = MagicMock()
        model1.evaluation = MagicMock()
        model1.evaluation.c_index_test = 0.65
        model1.cv_metric_mean = 0.72  # CV winner

        model2 = MagicMock()
        model2.evaluation = MagicMock()
        model2.evaluation.c_index_test = 0.70  # Holdout winner
        model2.cv_metric_mean = 0.68

        models = {'coxph': model1, 'rsf': model2}

        learner._check_cv_vs_holdout_best(models, verbose=True)
        output = capsys.readouterr().out
        assert "Holdout test shows" in output
        assert "rsf" in output


class TestEnvironmentFingerprint:
    """Tests for the _environment_fingerprint() helper."""

    def test_returns_expected_keys(self):
        """_environment_fingerprint() should return all documented keys."""
        from mlsurv.learner._persistence import _environment_fingerprint

        env = _environment_fingerprint()

        expected_keys = {
            'mlsurv_version', 'python_version', 'platform',
            'scikit_survival_version', 'scikit_learn_version', 'numpy_version',
        }
        assert set(env.keys()) == expected_keys

    def test_mlsurv_version_matches_package(self):
        """mlsurv_version should match mlsurv.__version__."""
        import mlsurv
        from mlsurv.learner._persistence import _environment_fingerprint

        env = _environment_fingerprint()
        assert env['mlsurv_version'] == mlsurv.__version__

    def test_python_version_is_nonempty(self):
        """python_version should be a non-empty string."""
        from mlsurv.learner._persistence import _environment_fingerprint

        env = _environment_fingerprint()
        assert isinstance(env['python_version'], str)
        assert len(env['python_version']) > 0


class TestArtifactSerialization:
    """Tests for save_artifact() / load_artifact() roundtrip."""

    def test_save_creates_directory_with_both_files(self, small_data, tmp_path):
        """save_artifact() should create artifact.json and model.joblib."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        result = learner.train(['coxph'])
        model_result = learner._state.models['coxph']
        assert result['coxph'] is model_result

        artifact_dir = tmp_path / 'test_artifact'
        learner.save_artifact(
            model_result, artifact_dir,
            step='train', time_col='time', event_col='event',
        )

        assert (artifact_dir / 'artifact.json').exists()
        assert (artifact_dir / 'model.joblib').exists()

    def test_artifact_json_contains_required_keys(self, small_data, tmp_path):
        """artifact.json should contain all top-level keys."""
        import json

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        result = learner.train(['coxph'])
        model_result = learner._state.models['coxph']
        assert result['coxph'] is model_result

        artifact_dir = tmp_path / 'test_artifact'
        learner.save_artifact(
            model_result, artifact_dir,
            step='train', time_col='time', event_col='event',
        )

        with open(artifact_dir / 'artifact.json') as f:
            metadata = json.load(f)

        expected_keys = {
            'artifact_version', 'step', 'model_name', 'timestamp',
            'environment', 'config', 'data_contract', 'results',
        }
        assert expected_keys <= set(metadata.keys())
        assert metadata['artifact_version'] == 1
        assert metadata['step'] == 'train'

    def test_artifact_json_contains_ci_fields(self, small_data, tmp_path):
        """artifact.json results should include CV CI fields."""
        import json

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_ci=0.90)
        learner.train(['coxph'])
        model_result = learner._state.models['coxph']

        artifact_dir = tmp_path / 'test_ci_artifact'
        learner.save_artifact(
            model_result, artifact_dir,
            step='train', time_col='time', event_col='event',
        )

        with open(artifact_dir / 'artifact.json') as f:
            metadata = json.load(f)

        results = metadata['results']
        assert 'cv_ci_lower' in results
        assert 'cv_ci_upper' in results
        assert 'cv_ci_level' in results
        assert results['cv_ci_level'] == pytest.approx(0.90)
        # NaN values should be serialized as null, not as NaN literal
        json_text = (artifact_dir / 'artifact.json').read_text()
        assert 'NaN' not in json_text

    def test_roundtrip_produces_identical_predictions(self, small_data, tmp_path):
        """Loaded model should produce identical predictions."""
        from mlsurv.learner._persistence import PersistenceMixin

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(['coxph'])
        model_result = learner._state.models['coxph']
        original_preds = model_result.model.predict(learner.X_test)

        artifact_dir = tmp_path / 'test_artifact'
        learner.save_artifact(
            model_result, artifact_dir,
            step='train', time_col='time', event_col='event',
        )

        metadata, loaded_model, _recal = PersistenceMixin.load_artifact(artifact_dir)
        loaded_preds = loaded_model.predict(learner.X_test)
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_save_accepts_raw_model(self, small_data, tmp_path):
        """save_artifact() should accept a raw model (not ModelResult).

        When a raw model (no ``.model`` attribute) is saved, ``model_name``
        falls back to ``step`` and ``results`` is empty.
        """
        import json

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(['coxph'])
        raw_model = learner._state.models['coxph'].model

        artifact_dir = tmp_path / 'test_artifact'
        learner.save_artifact(
            raw_model, artifact_dir,
            step='train', time_col='time', event_col='event',
        )

        assert (artifact_dir / 'artifact.json').exists()
        assert (artifact_dir / 'model.joblib').exists()

        with open(artifact_dir / 'artifact.json') as f:
            metadata = json.load(f)
        # Raw model has no .model attr, so model_name falls back to step
        assert metadata['model_name'] == 'train'
        # No ModelResult fields → empty results dict
        assert metadata['results'] == {}

    def test_load_raises_for_missing_directory(self, tmp_path):
        """load_artifact() should raise FileNotFoundError for missing dir."""
        from mlsurv.learner._persistence import PersistenceMixin

        with pytest.raises(FileNotFoundError, match="Artifact directory not found"):
            PersistenceMixin.load_artifact(tmp_path / 'nonexistent')

    def test_load_raises_for_missing_metadata(self, tmp_path):
        """load_artifact() should raise FileNotFoundError for missing artifact.json."""
        from mlsurv.learner._persistence import PersistenceMixin

        empty_dir = tmp_path / 'empty_artifact'
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="artifact.json not found"):
            PersistenceMixin.load_artifact(empty_dir)

    def test_load_raises_for_unsupported_version(self, tmp_path):
        """load_artifact() should raise ValueError for unsupported version."""
        import json
        from mlsurv.learner._persistence import PersistenceMixin

        artifact_dir = tmp_path / 'future_artifact'
        artifact_dir.mkdir()
        metadata = {'artifact_version': 999}
        with open(artifact_dir / 'artifact.json', 'w') as f:
            json.dump(metadata, f)

        with pytest.raises(ValueError, match="Unsupported artifact version"):
            PersistenceMixin.load_artifact(artifact_dir)

    @pytest.mark.parametrize('bad_version', [0, -1, None, 'v1', 2.5])
    def test_load_rejects_invalid_version_values(self, tmp_path, bad_version):
        """load_artifact() should reject version 0, negative, None, non-int."""
        import json
        from mlsurv.learner._persistence import PersistenceMixin

        artifact_dir = tmp_path / 'bad_version_artifact'
        artifact_dir.mkdir()
        metadata = {'artifact_version': bad_version}
        with open(artifact_dir / 'artifact.json', 'w') as f:
            json.dump(metadata, f)

        with pytest.raises(ValueError, match="Unsupported artifact version"):
            PersistenceMixin.load_artifact(artifact_dir)

    def test_load_pre_versioned_artifact_warns(self, tmp_path):
        """Artifact with no artifact_version key loads with warning."""
        import json
        import warnings
        from mlsurv.learner._persistence import PersistenceMixin

        artifact_dir = tmp_path / 'pre_versioned'
        artifact_dir.mkdir()
        metadata = {'step': 'train', 'model_name': 'coxph'}
        with open(artifact_dir / 'artifact.json', 'w') as f:
            json.dump(metadata, f)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded_metadata, model, _recal = PersistenceMixin.load_artifact(artifact_dir)

        assert loaded_metadata['artifact_version'] == 0
        assert model is None
        assert any("pre-versioned" in str(w.message).lower() for w in caught)

    def test_load_version_0_explicit_still_raises(self, tmp_path):
        """Explicit artifact_version=0 should still raise ValueError."""
        import json
        from mlsurv.learner._persistence import PersistenceMixin

        artifact_dir = tmp_path / 'explicit_v0'
        artifact_dir.mkdir()
        metadata = {'artifact_version': 0}
        with open(artifact_dir / 'artifact.json', 'w') as f:
            json.dump(metadata, f)

        with pytest.raises(ValueError, match="Unsupported artifact version"):
            PersistenceMixin.load_artifact(artifact_dir)

    def test_config_in_artifact_matches_config(self, small_data, tmp_path):
        """Config stored in artifact should match the config property."""
        import json

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(['coxph'])
        model_result = learner._state.models['coxph']
        expected_config = learner.config

        artifact_dir = tmp_path / 'test_artifact'
        learner.save_artifact(
            model_result, artifact_dir,
            step='train', time_col='time', event_col='event',
        )

        with open(artifact_dir / 'artifact.json') as f:
            metadata = json.load(f)

        # Compare key config values (JSON round-trip may stringify some)
        assert metadata['config']['random_state'] == expected_config['random_state']
        assert metadata['config']['cv_folds'] == expected_config['cv_folds']
        assert metadata['config']['test_fraction'] == expected_config['test_fraction']

    def test_feature_names_match_training_data(self, small_data, tmp_path):
        """Feature names in artifact should match training data columns."""
        import json

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train(['coxph'])
        model_result = learner._state.models['coxph']

        artifact_dir = tmp_path / 'test_artifact'
        learner.save_artifact(
            model_result, artifact_dir,
            step='train', time_col='time', event_col='event',
        )

        with open(artifact_dir / 'artifact.json') as f:
            metadata = json.load(f)

        assert metadata['data_contract']['feature_names'] == list(X.columns)

    def test_setup_only_artifact_has_no_model(self, small_data, tmp_path):
        """Setup-only artifact should have artifact.json but no model.joblib."""
        from mlsurv.learner._persistence import PersistenceMixin

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        artifact_dir = tmp_path / 'setup_artifact'
        learner.save_artifact(
            None, artifact_dir,
            step='setup', time_col='time', event_col='event',
        )

        assert (artifact_dir / 'artifact.json').exists()
        assert not (artifact_dir / 'model.joblib').exists()

        metadata, model, _recal = PersistenceMixin.load_artifact(artifact_dir)
        assert model is None
        assert metadata['step'] == 'setup'
        assert metadata['model_name'] is None

    def test_save_raises_before_setup(self, small_data, tmp_path):
        """save_artifact() should raise ValueError if learner has no data."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        # Do NOT call setup() — X is set in __init__ so we need to force None
        learner.X = None

        with pytest.raises(ValueError, match="Learner must be set up"):
            learner.save_artifact(
                None, tmp_path / 'no_setup',
                step='test', time_col='time', event_col='event',
            )

    def test_load_raises_for_corrupt_json(self, tmp_path):
        """load_artifact() should raise ValueError for corrupt JSON."""
        from mlsurv.learner._persistence import PersistenceMixin

        artifact_dir = tmp_path / 'corrupt_artifact'
        artifact_dir.mkdir()
        (artifact_dir / 'artifact.json').write_text('{bad json}')

        with pytest.raises(ValueError, match="Corrupt artifact metadata"):
            PersistenceMixin.load_artifact(artifact_dir)

    def test_load_raises_for_corrupt_model(self, tmp_path):
        """load_artifact() should raise RuntimeError for corrupt model.joblib."""
        import json
        from mlsurv.learner._persistence import PersistenceMixin

        artifact_dir = tmp_path / 'corrupt_model'
        artifact_dir.mkdir()
        metadata = {'artifact_version': 1}
        (artifact_dir / 'artifact.json').write_text(json.dumps(metadata))
        (artifact_dir / 'model.joblib').write_bytes(b'not a valid joblib file')

        with pytest.raises(RuntimeError, match="Failed to load model"):
            PersistenceMixin.load_artifact(artifact_dir)


class TestArtifactEncoder:
    """Tests for _ArtifactEncoder numpy/Path serialization."""

    def test_handles_numpy_types(self):
        """_ArtifactEncoder should serialize numpy scalars, arrays, and Paths."""
        import json
        from mlsurv.learner._persistence import _ArtifactEncoder

        data = {
            'int_val': np.int64(42),
            'float_val': np.float64(3.14),
            'bool_val': np.bool_(True),
            'array_val': np.array([1, 2, 3]),
            'path_val': Path('/tmp/test'),
        }
        result = json.loads(json.dumps(data, cls=_ArtifactEncoder))

        assert result['int_val'] == 42
        assert result['float_val'] == pytest.approx(3.14)
        assert result['bool_val'] is True
        assert result['array_val'] == [1, 2, 3]
        assert result['path_val'] == str(Path('/tmp/test'))

    def test_raises_for_unsupported_types(self):
        """_ArtifactEncoder should raise TypeError for unsupported types."""
        import json
        from mlsurv.learner._persistence import _ArtifactEncoder

        with pytest.raises(TypeError):
            json.dumps({'bad': object()}, cls=_ArtifactEncoder)


