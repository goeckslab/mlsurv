"""Tests for export_study() and _study_to_records()."""

import csv
import json
from unittest.mock import MagicMock

import optuna
import pytest

from mlsurv.tuning.optuna_tuner import _study_to_records, export_study


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_study(n_trials=3, direction="maximize", study_name="test_study"):
    """Create an in-memory Optuna study with completed trials."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction, study_name=study_name)

    def objective(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        y = trial.suggest_int("y", 1, 10)
        return x + y

    study.optimize(objective, n_trials=n_trials)
    return study


def _create_mixed_study():
    """Create a study with COMPLETE and PRUNED trials."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="mixed")

    def objective(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        if trial.number == 1:
            raise optuna.TrialPruned("pruned for test")
        return x

    study.optimize(objective, n_trials=3)
    return study


# ---------------------------------------------------------------------------
# _study_to_records tests
# ---------------------------------------------------------------------------

class TestStudyToRecords:
    def test_completed_trials(self):
        study = _create_study(n_trials=3)
        result = _study_to_records(study)

        assert len(result['trials']) == 3
        assert result['metadata']['n_trials'] == 3
        assert result['metadata']['n_completed'] == 3
        assert result['metadata']['best_value'] is not None
        assert result['metadata']['best_params'] is not None
        assert result['metadata']['study_name'] == "test_study"
        assert result['metadata']['direction'] == "MAXIMIZE"

        # Each trial should have param columns
        for trial in result['trials']:
            assert 'param_x' in trial
            assert 'param_y' in trial
            assert trial['state'] == 'COMPLETE'
            assert trial['duration_seconds'] is not None

    def test_mixed_states(self):
        study = _create_mixed_study()
        result = _study_to_records(study)

        states = [t['state'] for t in result['trials']]
        assert 'COMPLETE' in states
        assert 'PRUNED' in states
        # Best trial only from completed
        assert result['metadata']['n_completed'] < result['metadata']['n_trials']

    def test_empty_study(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(study_name="empty")
        result = _study_to_records(study)

        assert result['trials'] == []
        assert result['metadata']['n_trials'] == 0
        assert result['metadata']['n_completed'] == 0
        assert result['metadata']['best_trial_number'] is None
        assert result['metadata']['best_value'] is None
        assert result['metadata']['best_params'] is None

    def test_different_param_sets(self):
        """Trials with different parameters should all appear as columns."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(study_name="varied_params")

        # Manually add trials with different param sets
        study.add_trial(
            optuna.trial.create_trial(
                params={"a": 1.0},
                distributions={"a": optuna.distributions.FloatDistribution(0, 2)},
                values=[0.5],
                state=optuna.trial.TrialState.COMPLETE,
            )
        )
        study.add_trial(
            optuna.trial.create_trial(
                params={"b": 2.0},
                distributions={"b": optuna.distributions.FloatDistribution(0, 3)},
                values=[0.7],
                state=optuna.trial.TrialState.COMPLETE,
            )
        )

        result = _study_to_records(study)
        assert len(result['trials']) == 2
        # Both param_a and param_b should exist
        assert result['trials'][0]['param_a'] == 1.0
        assert result['trials'][0]['param_b'] is None
        assert result['trials'][1]['param_a'] is None
        assert result['trials'][1]['param_b'] == 2.0


# ---------------------------------------------------------------------------
# export_study tests
# ---------------------------------------------------------------------------

class TestExportStudy:
    def test_csv_output(self, tmp_path):
        study = _create_study(n_trials=3)
        out = tmp_path / "trials.csv"
        result = export_study(study=study, path=out)

        assert out.exists()
        assert len(result['trials']) == 3

        # Read back and verify
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert 'trial_number' in rows[0]
        assert 'param_x' in rows[0]

    def test_json_output(self, tmp_path):
        study = _create_study(n_trials=2)
        out = tmp_path / "trials.json"
        export_study(study=study, path=out)

        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert 'metadata' in data
        assert 'trials' in data
        assert len(data['trials']) == 2
        assert data['metadata']['study_name'] == "test_study"

    def test_format_from_extension(self, tmp_path):
        study = _create_study(n_trials=1)

        csv_path = tmp_path / "out.csv"
        export_study(study=study, path=csv_path)
        # CSV files don't start with '{'
        text = csv_path.read_text()
        assert text.startswith('trial_number')

        json_path = tmp_path / "out.json"
        export_study(study=study, path=json_path)
        data = json.loads(json_path.read_text())
        assert 'metadata' in data

    def test_return_only_no_path(self):
        study = _create_study(n_trials=2)
        result = export_study(study=study)

        assert 'trials' in result
        assert 'metadata' in result
        assert len(result['trials']) == 2

    def test_explicit_format_overrides_extension(self, tmp_path):
        study = _create_study(n_trials=1)
        out = tmp_path / "out.csv"
        export_study(study=study, path=out, format='json')
        # Should be JSON despite .csv extension
        data = json.loads(out.read_text())
        assert 'metadata' in data

    def test_missing_args_raises(self):
        with pytest.raises(ValueError, match="Provide"):
            export_study()

    def test_missing_partial_args_experiment_only(self):
        with pytest.raises(ValueError, match="'experiment_name' requires 'model_name'"):
            export_study(experiment_name="exp")

    def test_missing_partial_args_study_name_only(self):
        with pytest.raises(ValueError, match="'study_name' requires 'storage'"):
            export_study(study_name="my/study")

    def test_missing_partial_args_storage_only(self):
        with pytest.raises(ValueError, match="'storage' requires 'study_name'"):
            export_study(storage="sqlite:///db.sqlite3")

    def test_missing_partial_args_model_name_only(self):
        with pytest.raises(ValueError, match="'model_name' requires 'experiment_name'"):
            export_study(model_name="rsf")

    def test_missing_db_file_raises(self, tmp_path):
        """FileNotFoundError when study DB does not exist on disk."""
        with pytest.raises(FileNotFoundError, match="Study database not found"):
            export_study(
                experiment_name="nonexistent", model_name="rsf", base_dir=tmp_path
            )

    def test_wrong_study_name_in_db_raises(self, tmp_path):
        """ValueError when DB exists but study name doesn't match."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        db_path = exp_dir / "rsf_study.db"
        storage = f"sqlite:///{db_path}"
        # Create study with a different name than convention expects
        optuna.create_study(study_name="different_name", storage=storage)

        with pytest.raises(ValueError, match="not found in"):
            export_study(experiment_name="exp", model_name="rsf", base_dir=tmp_path)

    def test_file_write_error_gives_context(self, tmp_path):
        """OSError from file write includes contextual message."""
        study = _create_study(n_trials=1)
        # Use a path that points to a file as a directory (can't mkdir inside a file)
        blocker = tmp_path / "blocker"
        blocker.write_text("I'm a file")
        bad_path = blocker / "subdir" / "out.json"

        with pytest.raises(OSError, match="Failed to write export file"):
            export_study(study=study, path=bad_path)

    def test_load_from_storage(self, tmp_path):
        """Test loading study from SQLite storage."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        db_path = tmp_path / "test_study.db"
        storage = f"sqlite:///{db_path}"
        study = optuna.create_study(
            study_name="exp/rsf_tuning", storage=storage, direction="maximize"
        )
        study.optimize(lambda t: t.suggest_float("lr", 0.01, 0.1), n_trials=2)

        result = export_study(study_name="exp/rsf_tuning", storage=storage)
        assert len(result['trials']) == 2

    def test_load_from_experiment_model(self, tmp_path):
        """Test loading study from experiment_name + model_name convention."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # Create directory structure matching get_study_path convention
        exp_dir = tmp_path / "my_exp"
        exp_dir.mkdir()
        db_path = exp_dir / "rsf_study.db"
        storage = f"sqlite:///{db_path}"
        study = optuna.create_study(
            study_name="my_exp/rsf_tuning", storage=storage, direction="maximize"
        )
        study.optimize(lambda t: t.suggest_float("lr", 0.01, 0.1), n_trials=2)

        result = export_study(
            experiment_name="my_exp", model_name="rsf", base_dir=tmp_path
        )
        assert len(result['trials']) == 2


# ---------------------------------------------------------------------------
# StudyMixin.export_study tests
# ---------------------------------------------------------------------------

class TestConfigMixinExportStudy:
    def _make_learner_mock(self, models=None, experiment_name=None):
        """Create a mock learner with _state.models and _state.experiment_name."""
        from mlsurv.learner._studies import StudyMixin
        from mlsurv.learner._state import LearnerState
        learner = MagicMock(spec=StudyMixin)
        state = MagicMock(spec=LearnerState)
        state.models = models or {}
        state.experiment_name = experiment_name
        learner._state = state
        learner._lookup_model_by_name = MagicMock(
            side_effect=lambda name: learner._state.models.get(
                name, learner._state.models.get(f"{name}_tuned")
            )
        )
        return learner

    def test_in_memory_study(self, tmp_path):
        from mlsurv.learner._studies import StudyMixin

        study = _create_study(n_trials=2)
        model_result = MagicMock()
        model_result.is_tuned = True
        model_result.optuna_study = study
        model_result.name = "rsf"

        learner = self._make_learner_mock(
            models={"rsf_tuned": model_result}, experiment_name="exp"
        )
        out = tmp_path / "out.json"
        result = StudyMixin.export_study(learner, path=out)
        assert len(result['trials']) == 2
        assert out.exists()

    def test_model_name_string(self, tmp_path):
        from mlsurv.learner._studies import StudyMixin

        study = _create_study(n_trials=1)
        model_result = MagicMock()
        model_result.is_tuned = True
        model_result.optuna_study = study
        model_result.name = "gbsa"

        learner = self._make_learner_mock(
            models={"gbsa_tuned": model_result}, experiment_name="exp"
        )
        learner._lookup_model_by_name = MagicMock(return_value=model_result)

        result = StudyMixin.export_study(learner, model="gbsa")
        assert len(result['trials']) == 1

    def test_no_tuned_models_raises(self):
        from mlsurv.learner._studies import StudyMixin

        model_result = MagicMock()
        model_result.is_tuned = False
        learner = self._make_learner_mock(models={"rsf": model_result})

        with pytest.raises(ValueError, match="No tuned models"):
            StudyMixin.export_study(learner)

    def test_named_model_not_tuned_raises(self):
        from mlsurv.learner._studies import StudyMixin

        model_result = MagicMock()
        model_result.is_tuned = False
        learner = self._make_learner_mock(models={"rsf": model_result})
        learner._lookup_model_by_name = MagicMock(return_value=model_result)

        with pytest.raises(ValueError, match="has not been tuned"):
            StudyMixin.export_study(learner, model="rsf")

    def test_fallback_to_storage(self, tmp_path):
        """When model has no in-memory study, falls back to experiment/model storage."""
        from mlsurv.learner._studies import StudyMixin

        # Create a real study on disk
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        exp_dir = tmp_path / "my_exp"
        exp_dir.mkdir()
        db_path = exp_dir / "rsf_study.db"
        storage = f"sqlite:///{db_path}"
        study = optuna.create_study(
            study_name="my_exp/rsf_tuning", storage=storage, direction="maximize"
        )
        study.optimize(lambda t: t.suggest_float("lr", 0.01, 0.1), n_trials=2)

        # Model result with no in-memory study
        model_result = MagicMock()
        model_result.is_tuned = True
        model_result.optuna_study = None
        model_result.name = "rsf"

        learner = self._make_learner_mock(
            models={"rsf_tuned": model_result}, experiment_name="my_exp"
        )

        import mlsurv.tuning.studies as studies_mod
        original = studies_mod.DEFAULT_STORAGE_BASE
        studies_mod.DEFAULT_STORAGE_BASE = tmp_path
        try:
            result = StudyMixin.export_study(learner)
            assert len(result['trials']) == 2
        finally:
            studies_mod.DEFAULT_STORAGE_BASE = original

    def test_no_study_locatable_raises(self):
        """ValueError when model has no in-memory study and no experiment name."""
        from mlsurv.learner._studies import StudyMixin

        model_result = MagicMock()
        model_result.is_tuned = True
        model_result.optuna_study = None
        model_result.name = "rsf"

        # No experiment_name set
        learner = self._make_learner_mock(
            models={"rsf_tuned": model_result}, experiment_name=None
        )

        with pytest.raises(ValueError, match="Cannot locate study"):
            StudyMixin.export_study(learner)

    def test_invalid_model_type_raises(self):
        """TypeError when model is not str, ModelResult, or None."""
        from mlsurv.learner._studies import StudyMixin

        learner = self._make_learner_mock(models={})

        with pytest.raises(TypeError, match="Expected model name"):
            StudyMixin.export_study(learner, model=42)

    def test_selects_last_tuned_model(self):
        """With multiple models, model=None selects the last tuned one."""
        from mlsurv.learner._studies import StudyMixin

        study_first = _create_study(n_trials=1, study_name="first")
        study_last = _create_study(n_trials=2, study_name="last")

        first = MagicMock()
        first.is_tuned = True
        first.optuna_study = study_first
        first.name = "coxph"

        untuned = MagicMock()
        untuned.is_tuned = False

        last = MagicMock()
        last.is_tuned = True
        last.optuna_study = study_last
        last.name = "rsf"

        learner = self._make_learner_mock(
            models={"coxph_tuned": first, "gbsa": untuned, "rsf_tuned": last},
            experiment_name="exp",
        )

        result = StudyMixin.export_study(learner)
        # Should pick rsf (last tuned), which has 2 trials
        assert len(result['trials']) == 2
        assert result['metadata']['study_name'] == "last"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestCLIExportStudy:
    def test_parser_accepts_export_study(self):
        from mlsurv.cli import create_parser
        parser = create_parser()
        args = parser.parse_args([
            'export-study', '--experiment', 'exp', '--model', 'rsf',
            '--output', 'out.json',
        ])
        assert args.command == 'export-study'
        assert args.experiment == 'exp'
        assert args.model == 'rsf'
        assert args.output == 'out.json'

    def test_parser_study_name_storage(self):
        from mlsurv.cli import create_parser
        parser = create_parser()
        args = parser.parse_args([
            'export-study', '--study-name', 'my/study',
            '--storage', 'sqlite:///db.sqlite3',
            '--output', 'out.csv', '--format', 'csv',
        ])
        assert args.study_name == 'my/study'
        assert args.storage == 'sqlite:///db.sqlite3'
        assert args.format == 'csv'

    def test_handler_experiment_model(self, tmp_path, monkeypatch):
        from mlsurv.cli import cmd_export_study
        import mlsurv.tuning.studies as studies_mod

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        db_path = exp_dir / "rsf_study.db"
        storage = f"sqlite:///{db_path}"
        study = optuna.create_study(
            study_name="exp/rsf_tuning", storage=storage, direction="maximize"
        )
        study.optimize(lambda t: t.suggest_float("lr", 0.01, 0.1), n_trials=2)

        out = tmp_path / "result.json"
        args = MagicMock()
        args.experiment = "exp"
        args.model = "rsf"
        args.study_name = None
        args.storage = None
        args.output = str(out)
        args.format = None

        monkeypatch.setattr(studies_mod, "DEFAULT_STORAGE_BASE", tmp_path)
        rc = cmd_export_study(args)

        assert rc == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data['trials']) == 2

    def test_handler_study_name_storage(self, tmp_path):
        from mlsurv.cli import cmd_export_study

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        db_path = tmp_path / "studies.db"
        storage = f"sqlite:///{db_path}"
        study = optuna.create_study(
            study_name="my/study", storage=storage, direction="maximize"
        )
        study.optimize(lambda t: t.suggest_float("lr", 0.01, 0.1), n_trials=3)

        out = tmp_path / "result.csv"
        args = MagicMock()
        args.experiment = None
        args.model = None
        args.study_name = "my/study"
        args.storage = storage
        args.output = str(out)
        args.format = "csv"

        rc = cmd_export_study(args)

        assert rc == 0
        assert out.exists()
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_handler_missing_args(self, capsys):
        from mlsurv.cli import cmd_export_study
        args = MagicMock()
        args.experiment = None
        args.model = None
        args.study_name = None
        args.storage = None
        args.output = "out.json"
        args.format = None

        rc = cmd_export_study(args)
        assert rc == 2  # EXIT_USAGE: missing required arguments


class TestCLIListStudies:
    @staticmethod
    def _make_args(**overrides):
        args = MagicMock()
        args.experiment = None
        args.storage = None
        args.output_format = "table"
        args.quiet = False
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    @staticmethod
    def _seed_file_studies(base_dir):
        """Create two *_study.db files under base_dir/gbsg2_analysis/.

        File mode only globs filenames (it never opens the db), so empty
        touched files are sufficient and avoid SQLite file-handle locks.
        """
        exp_dir = base_dir / "gbsg2_analysis"
        exp_dir.mkdir()
        (exp_dir / "rsf_study.db").touch()
        (exp_dir / "gbsa_study.db").touch()

    def test_parser_accepts_list_studies(self):
        from mlsurv.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(['list-studies', '--experiment', 'exp'])
        assert args.command == 'list-studies'
        assert args.experiment == 'exp'
        assert args.storage is None
        assert args.output_format == 'table'

    def test_parser_storage_and_format(self):
        from mlsurv.cli import create_parser
        parser = create_parser()
        args = parser.parse_args([
            'list-studies', '--storage', 'sqlite:///db.sqlite3',
            '--format', 'json',
        ])
        assert args.storage == 'sqlite:///db.sqlite3'
        assert args.output_format == 'json'

    def test_handler_file_mode_lists_studies(self, tmp_path, monkeypatch, capsys):
        from mlsurv.cli import cmd_list_studies
        import mlsurv.tuning.studies as studies_mod

        self._seed_file_studies(tmp_path)
        monkeypatch.setattr(studies_mod, "DEFAULT_STORAGE_BASE", tmp_path)

        rc = cmd_list_studies(self._make_args())
        assert rc == 0
        out = capsys.readouterr().out
        assert "gbsg2_analysis" in out
        assert "rsf_study.db" in out
        assert "gbsa_study.db" in out
        assert "optuna-dashboard sqlite:///" in out

    def test_handler_experiment_filter_present(self, tmp_path, monkeypatch, capsys):
        from mlsurv.cli import cmd_list_studies
        import mlsurv.tuning.studies as studies_mod

        self._seed_file_studies(tmp_path)
        (tmp_path / "other_exp").mkdir()
        (tmp_path / "other_exp" / "coxph_study.db").touch()
        monkeypatch.setattr(studies_mod, "DEFAULT_STORAGE_BASE", tmp_path)

        rc = cmd_list_studies(self._make_args(experiment="gbsg2_analysis"))
        assert rc == 0
        out = capsys.readouterr().out
        assert "gbsg2_analysis" in out
        assert "other_exp" not in out

    def test_handler_experiment_filter_absent(self, tmp_path, monkeypatch, capsys):
        from mlsurv.cli import cmd_list_studies
        import mlsurv.tuning.studies as studies_mod

        self._seed_file_studies(tmp_path)
        monkeypatch.setattr(studies_mod, "DEFAULT_STORAGE_BASE", tmp_path)

        rc = cmd_list_studies(self._make_args(experiment="missing"))
        assert rc == 0
        out = capsys.readouterr().out
        assert "No studies found for experiment 'missing'." in out

    def test_handler_json_format(self, tmp_path, monkeypatch, capsys):
        from mlsurv.cli import cmd_list_studies
        import mlsurv.tuning.studies as studies_mod

        self._seed_file_studies(tmp_path)
        monkeypatch.setattr(studies_mod, "DEFAULT_STORAGE_BASE", tmp_path)

        rc = cmd_list_studies(self._make_args(output_format="json"))
        assert rc == 0
        records = json.loads(capsys.readouterr().out)
        assert len(records) == 2
        first = records[0]
        assert first["experiment"] == "gbsg2_analysis"
        assert first["study"].endswith("_study.db")
        assert first["dashboard"].startswith("sqlite:///")

    def test_handler_csv_format(self, tmp_path, monkeypatch, capsys):
        from mlsurv.cli import cmd_list_studies
        import mlsurv.tuning.studies as studies_mod

        self._seed_file_studies(tmp_path)
        monkeypatch.setattr(studies_mod, "DEFAULT_STORAGE_BASE", tmp_path)

        rc = cmd_list_studies(self._make_args(output_format="csv"))
        assert rc == 0
        out = capsys.readouterr().out
        assert out.splitlines()[0] == "experiment,study,path,dashboard"
        assert "gbsg2_analysis,rsf_study.db" in out

    def test_handler_empty_dir(self, tmp_path, monkeypatch, capsys):
        from mlsurv.cli import cmd_list_studies
        import mlsurv.tuning.studies as studies_mod

        monkeypatch.setattr(studies_mod, "DEFAULT_STORAGE_BASE", tmp_path)

        rc = cmd_list_studies(self._make_args())
        assert rc == 0
        assert "No studies found" in capsys.readouterr().out

    def test_handler_storage_mode(self, tmp_path, capsys):
        from mlsurv.cli import cmd_list_studies

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage = f"sqlite:///{tmp_path / 'studies.db'}"
        study = optuna.create_study(
            study_name="gbsg2_analysis/rsf_tuning", storage=storage,
            direction="maximize",
        )
        study.optimize(lambda t: t.suggest_float("lr", 0.01, 0.1), n_trials=2)

        rc = cmd_list_studies(self._make_args(storage=storage))
        assert rc == 0
        out = capsys.readouterr().out
        assert "Dashboard: optuna-dashboard" in out
        assert "gbsg2_analysis" in out
        assert "trials" in out

    def test_handler_storage_mode_json(self, tmp_path, capsys):
        from mlsurv.cli import cmd_list_studies

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage = f"sqlite:///{tmp_path / 'studies.db'}"
        study = optuna.create_study(
            study_name="gbsg2_analysis/rsf_tuning", storage=storage,
            direction="maximize",
        )
        study.optimize(lambda t: t.suggest_float("lr", 0.01, 0.1), n_trials=2)

        rc = cmd_list_studies(
            self._make_args(storage=storage, output_format="json")
        )
        assert rc == 0
        records = json.loads(capsys.readouterr().out)
        assert len(records) == 1
        rec = records[0]
        # Storage-mode records carry a different schema than file mode.
        assert rec["experiment"] == "gbsg2_analysis"
        assert rec["study"] == "rsf_tuning"
        assert rec["full_name"] == "gbsg2_analysis/rsf_tuning"
        assert rec["n_trials"] == 2
        assert rec["best_value"] is not None

    def test_handler_storage_mode_csv(self, tmp_path, capsys):
        from mlsurv.cli import cmd_list_studies

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage = f"sqlite:///{tmp_path / 'studies.db'}"
        study = optuna.create_study(
            study_name="gbsg2_analysis/rsf_tuning", storage=storage,
            direction="maximize",
        )
        study.optimize(lambda t: t.suggest_float("lr", 0.01, 0.1), n_trials=2)

        rc = cmd_list_studies(
            self._make_args(storage=storage, output_format="csv")
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert out.splitlines()[0] == (
            "experiment,study,full_name,n_trials,best_value"
        )
        assert "gbsg2_analysis,rsf_tuning,gbsg2_analysis/rsf_tuning,2," in out

    def test_handler_storage_mode_best_value_none(self, tmp_path, capsys):
        from mlsurv.cli import cmd_list_studies

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage = f"sqlite:///{tmp_path / 'studies.db'}"
        # A study with no completed trials -> best_value is None.
        optuna.create_study(
            study_name="gbsg2_analysis/rsf_tuning", storage=storage,
            direction="maximize",
        )

        rc = cmd_list_studies(self._make_args(storage=storage))
        assert rc == 0
        out = capsys.readouterr().out
        assert "best=N/A" in out

    def test_handler_storage_mode_empty(self, tmp_path, capsys):
        from mlsurv.cli import cmd_list_studies

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # Create then delete a study so the storage exists but holds none.
        storage = f"sqlite:///{tmp_path / 'studies.db'}"
        optuna.create_study(study_name="tmp", storage=storage)
        optuna.delete_study(study_name="tmp", storage=storage)

        rc = cmd_list_studies(self._make_args(storage=storage))
        assert rc == 0
        out = capsys.readouterr().out
        assert f"No studies found in storage: {storage}" in out

    def test_handler_quiet_suppresses_header_only(self, tmp_path, monkeypatch, capsys):
        from mlsurv.cli import cmd_list_studies
        import mlsurv.tuning.studies as studies_mod

        self._seed_file_studies(tmp_path)
        monkeypatch.setattr(studies_mod, "DEFAULT_STORAGE_BASE", tmp_path)

        rc = cmd_list_studies(self._make_args(quiet=True))
        assert rc == 0
        out = capsys.readouterr().out
        # The listing itself is the result and must still print.
        assert "rsf_study.db" in out
        # The decorative storage header is suppressed under --quiet.
        assert "Storage:" not in out
