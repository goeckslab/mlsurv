"""Tests for EnvironmentSection."""

import logging
from unittest.mock import MagicMock

import pandas as pd

from mlsurv import SurvivalLearner
from mlsurv.reports.sections.environment import EnvironmentSection, _KEY_DEPENDENCIES
from types import SimpleNamespace


def test_environment_section_is_always_available(small_data):
    """EnvironmentSection.is_available() should always return True."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = EnvironmentSection(learner)
    assert section.is_available() is True


def test_environment_section_generates_version_table(small_data):
    """generate() should produce a versions DataFrame with mlsurv, Python, and deps."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = EnvironmentSection(learner)
    result = section.generate()

    # Basic structure
    assert 'markdown' in result
    assert 'figures' in result
    assert isinstance(result['markdown'], str)
    assert len(result['markdown']) > 0

    # Check versions tidy data
    assert 'versions' in section.data
    versions_df = section.data['versions']
    assert isinstance(versions_df, pd.DataFrame)
    assert 'Package' in versions_df.columns
    assert 'Version' in versions_df.columns

    # mlsurv and Python should be present
    packages = versions_df['Package'].tolist()
    assert 'mlsurv' in packages
    assert 'Python' in packages

    # Key dependencies should all be present
    for dep in _KEY_DEPENDENCIES:
        assert dep in packages


def test_environment_section_generates_config(small_data):
    """generate() should store config as a DataFrame with Key, Value, and Category."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = EnvironmentSection(learner)
    section.generate()

    config_df = section.data['config']
    assert isinstance(config_df, pd.DataFrame)
    assert 'Key' in config_df.columns
    assert 'Value' in config_df.columns
    assert 'Category' in config_df.columns
    keys = config_df['Key'].tolist()
    assert 'test_fraction' in keys
    assert 'cv_folds' in keys
    assert 'random_state' in keys


def test_environment_section_markdown_has_expected_content(small_data):
    """Markdown output should contain version table, platform, reproducibility, config, and timestamp."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = EnvironmentSection(learner)
    result = section.generate()
    md = result['markdown']

    assert '### Software Versions' in md
    assert '### Platform' in md
    assert '### Reproducibility' in md
    assert '### Full Pipeline Configuration' in md
    assert '<details class="environment-info">' in md
    assert 'Report generated:' in md
    assert 'UTC' in md


def test_environment_section_handles_missing_package_gracefully(small_data, monkeypatch):
    """Version lookup should return 'not installed' for missing packages."""
    import importlib.metadata

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    original_version = importlib.metadata.version

    def _mock_version(name):
        if name == 'xgboost':
            raise importlib.metadata.PackageNotFoundError(name)
        return original_version(name)

    monkeypatch.setattr(importlib.metadata, 'version', _mock_version)

    section = EnvironmentSection(learner)
    section.generate()

    versions_df = section.data['versions']
    xgb_row = versions_df[versions_df['Package'] == 'xgboost']
    assert len(xgb_row) == 1
    assert xgb_row['Version'].iloc[0] == 'not installed'


def test_environment_section_handles_config_failure(small_data, monkeypatch, caplog):
    """If reading learner.config raises, section should log and continue."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    monkeypatch.setattr(
        type(learner), 'config',
        property(lambda self: (_ for _ in ()).throw(RuntimeError("config boom"))),
    )

    section = EnvironmentSection(learner)

    with caplog.at_level(logging.ERROR, logger="mlsurv.reports.sections.environment"):
        result = section.generate()

    assert 'environment: reading learner.config failed' in caplog.text
    assert isinstance(result['markdown'], str)
    assert isinstance(section.data['config'], pd.DataFrame)
    assert section.data['config'].empty


def test_environment_section_handles_unexpected_version_error(small_data, monkeypatch, caplog):
    """Unexpected errors in version lookup should produce 'unknown' and log."""
    import importlib.metadata

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    original_version = importlib.metadata.version

    def _mock_version(name):
        if name == 'numpy':
            raise OSError("disk read failure")
        return original_version(name)

    monkeypatch.setattr(importlib.metadata, 'version', _mock_version)

    section = EnvironmentSection(learner)
    with caplog.at_level(logging.ERROR, logger="mlsurv.reports.sections.environment"):
        section.generate()

    versions_df = section.data['versions']
    numpy_row = versions_df[versions_df['Package'] == 'numpy']
    assert numpy_row['Version'].iloc[0] == 'unknown'
    assert 'environment: version lookup for numpy' in caplog.text


def test_environment_section_no_figures(small_data):
    """EnvironmentSection should produce no figures."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = EnvironmentSection(learner)
    result = section.generate()
    assert result['figures'] == []


def test_reproducibility_subsection_present(small_data):
    """### Reproducibility should appear in markdown."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = EnvironmentSection(learner)
    result = section.generate()
    assert '### Reproducibility' in result['markdown']


def test_reproducibility_random_seed(small_data):
    """Reproducibility data should contain the random seed value."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = EnvironmentSection(learner)
    section.generate()

    repro_df = section.data['reproducibility']
    seed_row = repro_df[repro_df['Property'] == 'Random Seed']
    assert len(seed_row) == 1
    assert seed_row['Value'].iloc[0] == '42'


def test_reproducibility_data_key(small_data):
    """section.data['reproducibility'] should exist with correct columns."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = EnvironmentSection(learner)
    section.generate()

    assert 'reproducibility' in section.data
    repro_df = section.data['reproducibility']
    assert isinstance(repro_df, pd.DataFrame)
    assert 'Property' in repro_df.columns
    assert 'Value' in repro_df.columns


def test_grouped_config(small_data):
    """Config should be grouped by category with a Category column."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = EnvironmentSection(learner)
    section.generate()

    config_df = section.data['config']
    assert 'Category' in config_df.columns
    categories = config_df['Category'].unique().tolist()
    # Should have at least Data, Cross-Validation, Evaluation, Reproducibility,
    # Diagnostics, Subpopulations groups
    assert 'Data' in categories
    assert 'Cross-Validation' in categories
    assert 'Evaluation' in categories
    assert 'Reproducibility' in categories
    assert 'Diagnostics' in categories
    assert 'Subpopulations' in categories


def test_deep_learning_deps_conditional(small_data, monkeypatch):
    """When DL models are present, torch/pycox should appear in versions."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    # Mock a deepsurv model in state
    mock_result = MagicMock()
    learner._state.models['deepsurv'] = mock_result

    section = EnvironmentSection(learner)
    section.generate()

    packages = section.data['versions']['Package'].tolist()
    assert 'torch' in packages
    assert 'pycox' in packages


def test_no_deep_learning_deps_without_dl_models(small_data):
    """Without DL models, torch/pycox should NOT appear in versions."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = EnvironmentSection(learner)
    section.generate()

    packages = section.data['versions']['Package'].tolist()
    assert 'torch' not in packages
    assert 'pycox' not in packages


def test_format_grouped_config_empty():
    """Empty config should produce fallback message."""
    result = EnvironmentSection._format_grouped_config({})
    assert 'No configuration available' in result


def test_config_unknown_key_goes_to_other():
    """Config keys not in _CONFIG_GROUPS should be categorized as 'Other'."""
    df = EnvironmentSection._build_config_dataframe({'unknown_key': 'val'})
    assert df.loc[0, 'Category'] == 'Other'


def test_run_settings_empty_when_no_models(small_data):
    """Run Settings DataFrame should be empty before any training."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = EnvironmentSection(learner)
    result = section.generate()

    run_df = section.data['run_settings']
    assert isinstance(run_df, pd.DataFrame)
    assert run_df.empty
    # No cohorts configured -> no cohort heading rendered
    assert 'External validation cohorts' not in result['markdown']
    # Run Settings header itself is gated by non-empty df
    assert '### Run Settings' not in result['markdown']


def test_count_optuna_trials_branches():
    """_count_optuna_trials covers live, meta-fallback, and missing branches."""
    section = EnvironmentSection.__new__(EnvironmentSection)
    section.learner = MagicMock()
    section.section_data = {}
    section.data = {}

    # Live study with N trials
    mr = MagicMock()
    mr.optuna_study = MagicMock()
    mr.optuna_study.trials = [MagicMock(), MagicMock()]
    assert section._count_optuna_trials(mr) == 2

    # Empty study (boundary)
    mr.optuna_study.trials = []
    assert section._count_optuna_trials(mr) == 0

    # Live study missing -> falls back to metadata, marked as cached
    mr2 = MagicMock(spec=['_optuna_study_meta'])
    mr2._optuna_study_meta = {'n_trials': 7}
    result = section._count_optuna_trials(mr2)
    assert '7' in str(result)
    assert 'metadata' in str(result)

    # Nothing recorded
    mr3 = MagicMock(spec=[])
    assert section._count_optuna_trials(mr3) == '-'


def test_count_bootstrap_iterations_branches():
    """_count_bootstrap_iterations covers None, empty, missing-col, populated."""
    section = EnvironmentSection.__new__(EnvironmentSection)
    section.learner = MagicMock()
    section.section_data = {}
    section.data = {}

    # bootstrap is None
    mr = MagicMock()
    mr.bootstrap = None
    assert section._count_bootstrap_iterations(mr) == '-'

    # iterations is None
    mr.bootstrap = MagicMock()
    mr.bootstrap.iterations = None
    assert section._count_bootstrap_iterations(mr) == '-'

    # iterations is empty DataFrame
    mr.bootstrap.iterations = pd.DataFrame()
    assert section._count_bootstrap_iterations(mr) == '-'

    # iterations is missing 'iteration' column
    mr.bootstrap.iterations = pd.DataFrame({'metric': ['c']})
    assert section._count_bootstrap_iterations(mr) == '-'

    # populated
    mr.bootstrap.iterations = pd.DataFrame({'iteration': [0, 1, 1, 2]})
    assert section._count_bootstrap_iterations(mr) == 3


def test_format_horizons_branches():
    """_format_horizons distinguishes None, empty, single, multi."""
    assert EnvironmentSection._format_horizons(None) == '-'
    assert 'no horizons' in EnvironmentSection._format_horizons([])
    assert EnvironmentSection._format_horizons([5.0]) == '5'
    multi = EnvironmentSection._format_horizons([1.0, 2.0])
    assert '1' in multi and '2' in multi


def test_environment_section_renders_new_group_headings(small_data):
    """Diagnostics, Subpopulations groups should render by default."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = EnvironmentSection(learner)
    result = section.generate()
    md = result['markdown']

    assert '**Diagnostics**' in md
    assert '**Reproducibility**' in md
    assert '**Subpopulations**' in md


def test_run_settings_populated_after_training(small_data):
    """Run Settings should record per-model tuning, bootstrap, recalibration."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    coxph_result = MagicMock()
    coxph_result.is_tuned = False
    coxph_result.optuna_study = None
    coxph_result._optuna_study_meta = None
    boot = MagicMock()
    boot.iterations = pd.DataFrame({'iteration': [0, 1], 'metric': ['c', 'c']})
    coxph_result.bootstrap = boot
    # MagicMock auto-creates a truthy `.calibrator`; force None so coxph
    # renders as uncalibrated.
    coxph_result.calibrator = None

    rsf_result = MagicMock()
    rsf_result.is_tuned = True
    study = MagicMock()
    study.trials = [MagicMock(), MagicMock(), MagicMock()]
    rsf_result.optuna_study = study
    rsf_result.bootstrap = None
    rsf_result.calibrator = SimpleNamespace(
        method='isotonic', horizons=[1.0, 2.0],
    )

    learner._state.models['coxph'] = coxph_result
    learner._state.models['rsf'] = rsf_result
    learner._state.validation_results['Rotterdam'] = MagicMock()

    section = EnvironmentSection(learner)
    result = section.generate()

    run_df = section.data['run_settings']
    assert not run_df.empty
    assert set(run_df['Model']) == {'coxph', 'rsf'}

    rsf_row = run_df[run_df['Model'] == 'rsf'].iloc[0]
    assert bool(rsf_row['Tuned']) is True
    assert rsf_row['n_trials'] == 3
    assert bool(rsf_row['Calibrated']) is True
    assert rsf_row['Cal. method'] == 'isotonic'

    coxph_row = run_df[run_df['Model'] == 'coxph'].iloc[0]
    assert coxph_row['n_bootstrap'] == 2
    assert bool(coxph_row['Calibrated']) is False

    # External cohort listed in markdown
    assert 'Rotterdam' in result['markdown']
    assert 'Run Settings' in result['markdown']


def test_reproducibility_random_seed_fallback(small_data, monkeypatch, caplog):
    """Broken _state access should fall back to 'unknown' and log."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    # Replace _state with an object that raises on attribute access
    broken_state = MagicMock()
    type(broken_state).random_state = property(
        lambda self: (_ for _ in ()).throw(TypeError("broken"))
    )
    # Keep models working for _has_deep_learning_models
    broken_state.models = {}
    monkeypatch.setattr(learner, '_state', broken_state)

    section = EnvironmentSection(learner)
    with caplog.at_level(logging.ERROR, logger="mlsurv.reports.sections.environment"):
        section.generate()

    repro_df = section.data['reproducibility']
    seed_row = repro_df[repro_df['Property'] == 'Random Seed']
    assert seed_row['Value'].iloc[0] == 'unknown'


def test_reproducibility_includes_artifact_path(small_data):
    """When artifact_path is set, it should appear in reproducibility data."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)
    learner._state.artifact_path = "/tmp/my_artifact"

    section = EnvironmentSection(learner)
    section.generate()

    repro_df = section.data['reproducibility']
    artifact_row = repro_df[repro_df['Property'] == 'Artifact Path']
    assert len(artifact_row) == 1
    assert artifact_row['Value'].iloc[0] == '/tmp/my_artifact'


def test_reproducibility_omits_artifact_path_when_none(small_data):
    """Default state (no artifact saved) should omit Artifact Path row."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = EnvironmentSection(learner)
    section.generate()

    repro_df = section.data['reproducibility']
    artifact_row = repro_df[repro_df['Property'] == 'Artifact Path']
    assert len(artifact_row) == 0
