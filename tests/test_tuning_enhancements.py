"""Tests for tuning enhancements: n_trials=0 and reload_study."""

import warnings

import pytest

from mlsurv.learner import SurvivalLearner
from mlsurv.learner._tuning import TuningMixin


# Model with actual searchable params (CoxPH has none → fallback path).
_MODEL = "rsf"


# ============================================================================
# n_trials=0
# ============================================================================

class TestNTrialsZeroValidation:
    """Validation accepts n_trials=0 and rejects negative values."""

    def test_n_trials_zero_scalar_passes(self):
        TuningMixin._validate_tune_inputs(0, None, None, None)

    def test_n_trials_zero_dict_passes(self):
        TuningMixin._validate_tune_inputs({'rsf': 0}, None, None, None)

    def test_n_trials_negative_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            TuningMixin._validate_tune_inputs(-1, None, None, None)

    def test_n_trials_negative_dict_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            TuningMixin._validate_tune_inputs({'rsf': -1}, None, None, None)


@pytest.mark.integration
class TestNTrialsZeroTuner:
    """BaseTuner.tune() with n_trials=0."""

    def test_n_trials_zero_skips_tuning(self, small_data):
        """n_trials=0 without resume trains with defaults."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        result = learner.tune(_MODEL, n_trials=0, resume=False, verbose=False)
        assert result.is_tuned
        assert result.best_params_ == {}
        assert result.optuna_study is None  # no study created

    def test_n_trials_zero_via_run(self, small_data):
        """run() with n_trials=0 trains models with defaults."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        learner.run(
            models=['coxph', 'rsf'], n_trials=0, verbose=False,
            evaluate=False, bootstrap=False,
        )
        # RSF has searchable params → our n_trials=0 early return
        rsf = learner._state.models['rsf']
        assert rsf.is_tuned
        assert rsf.best_params_ == {}
        # CoxPH has no tuner → existing NoTunerError fallback (still trained)
        assert 'coxph' in learner._state.models

    def test_n_trials_zero_dict_mixed(self, small_data):
        """Per-model n_trials dict with 0 uses defaults for that model."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        result = learner.tune(_MODEL, n_trials={_MODEL: 0}, verbose=False)
        assert result.is_tuned
        assert result.best_params_ == {}

    def test_n_trials_zero_no_completed_trials(self, small_data, tmp_path):
        X, y = small_data
        exp_name = f"test_empty_{tmp_path.name}"
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, experiment_name=exp_name)

        # Create an empty study (no trials)
        import optuna
        from mlsurv.tuning import DEFAULT_STORAGE_BASE
        exp_dir = DEFAULT_STORAGE_BASE / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        db_path = exp_dir / f"{_MODEL}_study.db"
        storage = f"sqlite:///{db_path}"
        study_name = f"{exp_name}/{_MODEL}_tuning"
        optuna.create_study(
            direction="maximize",
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
        )

        with pytest.raises(ValueError, match="no completed trials"):
            learner.tune(_MODEL, n_trials=0, resume=True, verbose=False)

    def test_n_trials_zero_loads_existing(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        # First: tune normally
        result1 = learner.tune(
            _MODEL, n_trials=2, verbose=False,
            experiment_name="test_reload_rsf",
        )
        best_params_1 = result1.best_params_

        # Second: reload with n_trials=0 (fresh learner)
        learner2 = SurvivalLearner(X, y)
        learner2.setup(random_state=42)
        result2 = learner2.tune(
            _MODEL, n_trials=0, resume=True, verbose=False,
            experiment_name="test_reload_rsf",
        )
        best_params_2 = result2.best_params_

        assert best_params_1 == best_params_2

    def test_n_trials_zero_no_new_trials(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        # First tune with 2 trials
        result1 = learner.tune(
            _MODEL, n_trials=2, verbose=False,
            experiment_name="test_no_new_rsf",
        )
        n_trials_before = len(result1.optuna_study.trials)

        # Load with n_trials=0
        learner2 = SurvivalLearner(X, y)
        learner2.setup(random_state=42)
        result2 = learner2.tune(
            _MODEL, n_trials=0, resume=True, verbose=False,
            experiment_name="test_no_new_rsf",
        )
        n_trials_after = len(result2.optuna_study.trials)

        assert n_trials_after == n_trials_before


# ============================================================================
# all trials fail (invalid model search space)
# ============================================================================

@pytest.mark.integration
class TestAllTrialsFailed:
    """When 100% of trials fail, surface the dominant cause, not the opaque
    Optuna ``Record does not exist.`` from ``study.best_params``."""

    def test_invalid_model_param_surfaces_real_cause(self, small_data):
        """A typo'd param fails every trial -> clear ValueError naming it."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        with pytest.raises(ValueError) as exc:
            learner.tune(
                _MODEL, n_trials=2, verbose=False,
                search_spaces={'model': {'typo_param': [1]}},
            )

        msg = str(exc.value)
        # The underlying TypeError names the offending kwarg...
        assert "typo_param" in msg
        # ...and the model is identified.
        assert f"'{_MODEL}'" in msg
        # The opaque Optuna message must NOT leak through.
        assert "Record does not exist" not in msg

    def test_failure_message_structure(self, small_data):
        """The surfaced error includes the dominant-failure line and guidance."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        with pytest.raises(ValueError) as exc:
            learner.tune(
                _MODEL, n_trials=2, verbose=False,
                search_spaces={'model': {'typo_param': [1]}},
            )

        msg = str(exc.value)
        assert "Most common failure" in msg
        assert "none completed" in msg
        assert "search_spaces=" in msg

    def test_valid_space_does_not_trigger(self, small_data):
        """A valid search space still tunes normally (no false trigger)."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        result = learner.tune(
            _MODEL, n_trials=2, verbose=False,
            search_spaces={'model': {'n_estimators': [10, 20]}},
        )
        assert result.is_tuned
        # Decoded best params are component-prefixed (e.g. 'model__n_estimators').
        assert any('n_estimators' in k for k in result.best_params_)

    def test_resume_with_prior_success_skips_guard(self, small_data):
        """A resume carrying >=1 completed trial must not trigger the guard,
        even when every newly-added trial fails."""
        X, y = small_data
        exp_name = "test_resume_guard_rsf"

        # First: a valid run leaves completed trials in the study.
        learner1 = SurvivalLearner(X, y)
        learner1.setup(random_state=42)
        result1 = learner1.tune(
            _MODEL, n_trials=2, verbose=False, experiment_name=exp_name,
        )
        assert result1.is_tuned

        # Resume and add trials that all fail (invalid param). Because prior
        # COMPLETE trials exist, tune() must fall through to best_params (no
        # spurious "all trials failed" ValueError).
        learner2 = SurvivalLearner(X, y)
        learner2.setup(random_state=42)
        result2 = learner2.tune(
            _MODEL, n_trials=2, resume=True, verbose=False,
            experiment_name=exp_name,
            search_spaces={'model': {'typo_param': [1]}},
        )
        assert result2.is_tuned


# ============================================================================
# reload_study()
# ============================================================================

@pytest.mark.integration
class TestReloadStudy:
    """reload_study() restores Optuna studies after load_learner()."""

    @pytest.fixture(autouse=True)
    def _isolate_storage(self, monkeypatch, tmp_path):
        """Redirect Optuna storage to tmp_path to avoid CWD-relative path issues."""
        import mlsurv.tuning.storage as storage_mod
        import mlsurv.tuning.base as base_mod
        import mlsurv.tuning.studies as studies_mod
        import mlsurv.tuning as tuning_pkg
        import mlsurv.learner._format as format_mod
        monkeypatch.setattr(storage_mod, "DEFAULT_STORAGE_BASE", tmp_path)
        monkeypatch.setattr(base_mod, "DEFAULT_STORAGE_BASE", tmp_path)
        monkeypatch.setattr(studies_mod, "DEFAULT_STORAGE_BASE", tmp_path)
        monkeypatch.setattr(tuning_pkg, "DEFAULT_STORAGE_BASE", tmp_path)
        monkeypatch.setattr(format_mod, "DEFAULT_STORAGE_BASE", tmp_path)

    def test_reload_study_after_load_learner(self, small_data, tmp_path):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        result1 = learner.tune(_MODEL, n_trials=2, verbose=False)

        # Verify study exists before save
        assert result1.optuna_study is not None
        n_trials = len(result1.optuna_study.trials)

        # Save and reload
        path = tmp_path / "learner.joblib"
        learner.save_learner(path)
        loaded = SurvivalLearner.load_learner(path)

        # Study is None after load
        loaded_result = loaded._state.models[_MODEL]
        assert loaded_result.optuna_study is None

        # Restore via reload_study()
        study = loaded.reload_study(_MODEL)
        assert study is not None
        assert loaded_result.optuna_study is study
        assert len(study.trials) == n_trials

    def test_reload_study_already_loaded(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.tune(_MODEL, n_trials=2, verbose=False)

        existing_study = learner._state.models[_MODEL].optuna_study
        assert existing_study is not None
        returned = learner.reload_study(_MODEL)
        assert returned is existing_study

    def test_reload_study_no_tuned_models(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train([_MODEL])

        with pytest.raises(ValueError, match="not been tuned|No tuned models"):
            learner.reload_study(_MODEL)

    def test_reload_study_missing_db(self, small_data, tmp_path):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.tune(_MODEL, n_trials=2, verbose=False)

        # Save and reload
        path = tmp_path / "learner.joblib"
        learner.save_learner(path)
        loaded = SurvivalLearner.load_learner(path)

        # Point experiment to a non-existent directory instead of deleting
        # (avoids Windows file-locking issues with SQLite)
        loaded._state.experiment_name = "nonexistent_experiment_xyz"

        with pytest.raises(FileNotFoundError, match="Study database not found"):
            loaded.reload_study(_MODEL)

    def test_reload_study_metadata_mismatch_warns(self, small_data, tmp_path):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.tune(_MODEL, n_trials=2, verbose=False)

        # Save (metadata records 2 trials)
        path = tmp_path / "learner.joblib"
        learner.save_learner(path)

        # Add more trials to the study on disk
        learner.tune(_MODEL, n_trials=2, resume=True, force=True, verbose=False)

        # Load the old state (metadata says 2 trials, disk now has 4)
        loaded = SurvivalLearner.load_learner(path)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded.reload_study(_MODEL)

        mismatch_warnings = [x for x in w if "trials" in str(x.message)]
        assert len(mismatch_warnings) >= 1

    def test_optuna_study_meta_preserved(self, small_data, tmp_path):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.tune(_MODEL, n_trials=2, verbose=False)

        path = tmp_path / "learner.joblib"
        learner.save_learner(path)
        loaded = SurvivalLearner.load_learner(path)

        result = loaded._state.models[_MODEL]
        meta = getattr(result, '_optuna_study_meta', None)
        assert meta is not None
        assert meta['n_trials'] == 2
        assert 'study_name' in meta
        assert 'best_params' in meta

    def test_reload_study_default_model(self, small_data, tmp_path):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.tune(_MODEL, n_trials=2, verbose=False)

        path = tmp_path / "learner.joblib"
        learner.save_learner(path)
        loaded = SurvivalLearner.load_learner(path)

        study = loaded.reload_study()  # No model arg
        assert study is not None
        assert len(study.trials) == 2

    def test_reload_study_type_error(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        with pytest.raises(TypeError, match="Expected model name"):
            learner.reload_study(12345)

    def test_reload_study_meta_fallback_experiment_name(self, small_data, tmp_path):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.tune(_MODEL, n_trials=2, verbose=False)

        # Save and reload
        path = tmp_path / "learner.joblib"
        learner.save_learner(path)
        loaded = SurvivalLearner.load_learner(path)

        # Clear experiment_name — force fallback to meta
        loaded._state.experiment_name = None

        study = loaded.reload_study(_MODEL)
        assert study is not None
        assert len(study.trials) == 2


@pytest.mark.integration
class TestNTrialsZeroEpvWarning:
    """EPV warning should be suppressed for n_trials=0."""

    def test_n_trials_zero_no_epv_warning(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        # First tune normally to create study
        learner.tune(_MODEL, n_trials=2, verbose=False)

        # Reload with n_trials=0 — no EPV warning expected
        learner2 = SurvivalLearner(X, y)
        learner2.setup(random_state=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner2.tune(
                _MODEL, n_trials=0, resume=True, verbose=False,
                experiment_name=learner._state.experiment_name,
            )

        epv_warnings = [x for x in w if "EPV" in str(x.message)]
        assert len(epv_warnings) == 0


# ============================================================================
# tune() print box shows resolved storage path when auto-generated
# ============================================================================

class TestDefaultExperimentName:
    """`_default_experiment_name()` produces a parseable timestamp string."""

    def test_format_matches_pattern(self):
        import re
        from mlsurv.tuning.base import _default_experiment_name
        name = _default_experiment_name()
        assert re.fullmatch(r"exp_\d{8}_\d{6}", name) is not None


@pytest.mark.integration
class TestTuneStorageDisplay:
    """tune() pre-resolves experiment_name so the Storage row shows a real path."""

    def test_storage_row_shows_resolved_path(self, small_data, capsys, tmp_path, monkeypatch):
        import re
        # Run inside tmp_path so ./optuna_studies/ doesn't pollute repo root.
        monkeypatch.chdir(tmp_path)

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        capsys.readouterr()  # discard setup output
        learner.tune(_MODEL, n_trials=1, verbose=True)

        out = capsys.readouterr().out
        # Storage row should show the resolved experiment dir, not the
        # placeholder "(auto-generated)". The test conftest redirects
        # DEFAULT_STORAGE_BASE to a tmp dir, so we match on the timestamped
        # exp_ component rather than the literal ./optuna_studies/ prefix.
        assert "Storage:" in out
        assert "(auto-generated)" not in out
        match = re.search(r"Storage:\s*\S*exp_\d{8}_\d{6}/", out)
        assert match is not None, f"Storage row missing resolved path: {out!r}"
        assert re.fullmatch(r"exp_\d{8}_\d{6}", str(learner._state.experiment_name))

    def test_storage_row_respects_explicit_storage(self, small_data, capsys, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        storage_url = f"sqlite:///{tmp_path / 'explicit.db'}"
        learner.tune(_MODEL, n_trials=1, verbose=True, storage=storage_url)

        out = capsys.readouterr().out
        assert "(auto-generated)" not in out
        assert storage_url in out

    def test_explicit_experiment_name_not_overwritten(self, small_data, capsys, tmp_path, monkeypatch):
        """User-supplied experiment_name= must appear verbatim and persist to state."""
        monkeypatch.chdir(tmp_path)

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.tune(_MODEL, n_trials=1, verbose=True, experiment_name="my_custom_exp")

        out = capsys.readouterr().out
        assert "my_custom_exp" in out
        assert "(auto-generated)" not in out
        # _state.experiment_name is stored as Path (BaseTuner.tune wraps it).
        assert str(learner._state.experiment_name) == "my_custom_exp"

    def test_state_experiment_name_preserved_across_tunes(
        self, small_data, capsys, tmp_path, monkeypatch
    ):
        """Second tune() call must reuse experiment_name from prior call,
        not regenerate a fresh timestamp (would orphan the resume path)."""
        monkeypatch.chdir(tmp_path)

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.tune(_MODEL, n_trials=1, verbose=False)
        first = learner._state.experiment_name
        assert first is not None and str(first).startswith("exp_")

        # Second tune: pre-resolve guard must see _state.experiment_name is set
        # and skip regeneration.
        capsys.readouterr()
        learner.tune(_MODEL, n_trials=1, verbose=True, resume=True)
        out = capsys.readouterr().out
        assert str(first) in out
        assert learner._state.experiment_name == first

    def test_setup_experiment_name_preserved(self, small_data, capsys, tmp_path, monkeypatch):
        """experiment_name set in setup() must survive an auto-tune call."""
        monkeypatch.chdir(tmp_path)

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, experiment_name="from_setup")
        learner.tune(_MODEL, n_trials=1, verbose=True)

        out = capsys.readouterr().out
        assert "from_setup" in out
        assert str(learner._state.experiment_name) == "from_setup"


# ============================================================================
# Dataset fingerprint guards resume-on-different-data
# ============================================================================

def _fingerprint_warning(records):
    """Return warning records whose message is about the dataset fingerprint."""
    needles = ("different data", "cannot verify", "dataset fingerprint")
    return [r for r in records
            if any(n in str(r.message).lower() for n in needles)]


@pytest.mark.integration
class TestStudyResumeFingerprint:
    """A study stamps the dataset fingerprint on first tune and verifies it on
    resume, so reusing a study on different data/split/seed is not silent."""

    @pytest.fixture(autouse=True)
    def _isolate_storage(self, monkeypatch, tmp_path):
        """Redirect Optuna storage to tmp_path (mirror of TestReloadStudy)."""
        import mlsurv.tuning.storage as storage_mod
        import mlsurv.tuning.base as base_mod
        import mlsurv.tuning.studies as studies_mod
        import mlsurv.tuning as tuning_pkg
        import mlsurv.learner._format as format_mod
        monkeypatch.setattr(storage_mod, "DEFAULT_STORAGE_BASE", tmp_path)
        monkeypatch.setattr(base_mod, "DEFAULT_STORAGE_BASE", tmp_path)
        monkeypatch.setattr(studies_mod, "DEFAULT_STORAGE_BASE", tmp_path)
        monkeypatch.setattr(tuning_pkg, "DEFAULT_STORAGE_BASE", tmp_path)
        monkeypatch.setattr(format_mod, "DEFAULT_STORAGE_BASE", tmp_path)

    def test_first_tune_stamps_fingerprint(self, small_data):
        """A fresh study carries the dataset fingerprint after tuning."""
        from mlsurv.tuning.base import _FINGERPRINT_KEY
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        result = learner.tune(_MODEL, n_trials=2, verbose=False)
        stored = result.optuna_study.user_attrs.get(_FINGERPRINT_KEY)
        assert stored is not None
        # Matches the fingerprint recomputed from the tuner's own train data.
        tuner = learner._create_tuner_for_model(_MODEL)
        assert stored == tuner._compute_data_fingerprint()

    def test_resume_same_data_silent(self, small_data):
        """Resuming on identical data/split/seed emits no fingerprint warning."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.tune(_MODEL, n_trials=2, verbose=False)
        exp_name = learner._state.experiment_name

        learner2 = SurvivalLearner(X, y)
        learner2.setup(random_state=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner2.tune(
                _MODEL, n_trials=0, resume=True, verbose=False,
                experiment_name=exp_name,
            )
        assert _fingerprint_warning(w) == []

    def test_resume_changed_data_warns(self, small_data):
        """Resuming a study tuned on different data emits a loud
        MethodologicalWarning (same split/seed, perturbed feature values)."""
        from mlsurv._warnings import MethodologicalWarning
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.tune(_MODEL, n_trials=2, verbose=False)
        exp_name = learner._state.experiment_name

        # Same columns/shape/index and same random_state (so the train/test
        # split indices are unchanged) but shifted feature values -> the
        # train-fold data differs -> fingerprint mismatch.
        X_mod = X.copy()
        X_mod.iloc[:, 0] = X_mod.iloc[:, 0] + 100.0
        learner2 = SurvivalLearner(X_mod, y)
        learner2.setup(random_state=42)
        with pytest.warns(MethodologicalWarning, match="different data"):
            learner2.tune(
                _MODEL, n_trials=0, resume=True, verbose=False,
                experiment_name=exp_name,
            )

    def test_resume_changed_random_state_warns(self, small_data):
        """A different random_state (different split) also trips the guard."""
        from mlsurv._warnings import MethodologicalWarning
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.tune(_MODEL, n_trials=2, verbose=False)
        exp_name = learner._state.experiment_name

        learner2 = SurvivalLearner(X, y)
        learner2.setup(random_state=7)
        with pytest.warns(MethodologicalWarning, match="different data"):
            learner2.tune(
                _MODEL, n_trials=0, resume=True, verbose=False,
                experiment_name=exp_name,
            )

    def test_absent_fingerprint_soft_warns(self, small_data):
        """A resumed study with prior trials but no fingerprint (external or
        pre-fingerprint DB) emits a soft 'cannot verify' SuggestionWarning."""
        import optuna
        from mlsurv._warnings import SuggestionWarning
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        tuner = learner._create_tuner_for_model(_MODEL)

        # In-memory study with a completed trial but no fingerprint attr.
        study = optuna.create_study(direction="maximize")
        study.add_trial(optuna.trial.create_trial(
            params={}, distributions={}, value=0.5,
        ))
        with pytest.warns(SuggestionWarning, match="[Cc]annot verify"):
            tuner._verify_study_fingerprint(study, resume=True)

    def test_matching_fingerprint_helper_silent(self, small_data):
        """_verify_study_fingerprint is silent when the stored fingerprint
        matches, even with prior trials present."""
        import optuna
        from mlsurv.tuning.base import _FINGERPRINT_KEY
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        tuner = learner._create_tuner_for_model(_MODEL)

        study = optuna.create_study(direction="maximize")
        study.set_user_attr(_FINGERPRINT_KEY, tuner._compute_data_fingerprint())
        study.add_trial(optuna.trial.create_trial(
            params={}, distributions={}, value=0.5,
        ))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tuner._verify_study_fingerprint(study, resume=True)
        assert _fingerprint_warning(w) == []

    def test_resume_positive_trials_changed_data_warns(self, small_data):
        """The guard fires on the common resume path (n_trials > 0), not only
        the n_trials=0 reload path."""
        from mlsurv._warnings import MethodologicalWarning
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.tune(_MODEL, n_trials=2, verbose=False)
        exp_name = learner._state.experiment_name

        X_mod = X.copy()
        X_mod.iloc[:, 0] = X_mod.iloc[:, 0] + 100.0
        learner2 = SurvivalLearner(X_mod, y)
        learner2.setup(random_state=42)
        with pytest.warns(MethodologicalWarning, match="different data"):
            learner2.tune(
                _MODEL, n_trials=2, resume=True, verbose=False,
                experiment_name=exp_name,
            )

    def test_resume_positive_trials_same_data_silent(self, small_data):
        """Continuing tuning (n_trials > 0, resume) on identical data is
        silent."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.tune(_MODEL, n_trials=2, verbose=False)
        exp_name = learner._state.experiment_name

        learner2 = SurvivalLearner(X, y)
        learner2.setup(random_state=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner2.tune(
                _MODEL, n_trials=2, resume=True, verbose=False,
                experiment_name=exp_name,
            )
        assert _fingerprint_warning(w) == []

    def test_resume_false_rerun_changed_data_restamps_silently(self, small_data):
        """Re-running with the default resume=False on changed data but the
        same experiment_name deletes the prior study and stamps the new
        fingerprint with no warning (the study is not 'resumed')."""
        from mlsurv.tuning.base import _FINGERPRINT_KEY
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.tune(_MODEL, n_trials=2, verbose=False)
        exp_name = learner._state.experiment_name

        X_mod = X.copy()
        X_mod.iloc[:, 0] = X_mod.iloc[:, 0] + 100.0
        learner2 = SurvivalLearner(X_mod, y)
        learner2.setup(random_state=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result2 = learner2.tune(
                _MODEL, n_trials=2, resume=False, verbose=False,
                experiment_name=exp_name,
            )
        assert _fingerprint_warning(w) == []
        # New study carries the new data's fingerprint, not the old one.
        tuner2 = learner2._create_tuner_for_model(_MODEL)
        stored = result2.optuna_study.user_attrs.get(_FINGERPRINT_KEY)
        assert stored == tuner2._compute_data_fingerprint()

    def test_resume_changed_columns_warns(self, small_data):
        """Dropping a feature column (different column set) trips the guard."""
        from mlsurv._warnings import MethodologicalWarning
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.tune(_MODEL, n_trials=2, verbose=False)
        exp_name = learner._state.experiment_name

        X_dropped = X.drop(columns=[X.columns[-1]])
        learner2 = SurvivalLearner(X_dropped, y)
        learner2.setup(random_state=42)
        with pytest.warns(MethodologicalWarning, match="different data"):
            learner2.tune(
                _MODEL, n_trials=0, resume=True, verbose=False,
                experiment_name=exp_name,
            )

    def test_fingerprint_is_content_based(self, small_data):
        """Two equal-but-distinct DataFrames produce the same fingerprint, so
        resume verification keys on data content, not object identity."""
        X, y = small_data
        learner_a = SurvivalLearner(X, y)
        learner_a.setup(random_state=42)
        fp_a = learner_a._create_tuner_for_model(_MODEL)._compute_data_fingerprint()

        learner_b = SurvivalLearner(X.copy(), y.copy())
        learner_b.setup(random_state=42)
        fp_b = learner_b._create_tuner_for_model(_MODEL)._compute_data_fingerprint()

        assert fp_a is not None
        assert fp_a == fp_b
