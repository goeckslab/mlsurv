"""Guard test for the config/persistence/studies/export mixin split.

After ``ConfigMixin`` was split into focused mixins, each moved method must
live in exactly its new owner and be gone from ``ConfigMixin`` — while
``SurvivalLearner`` (which composes them all) must still expose every method,
proving the MRO wiring in ``_core.py`` is intact. A future careless move that
re-homes a method (or drops it from the chain) breaks one of these asserts.
"""

from mlsurv.learner import SurvivalLearner
from mlsurv.learner import _config, _persistence, _studies, _export
from mlsurv.learner._config import ConfigMixin
from mlsurv.learner._persistence import PersistenceMixin
from mlsurv.learner._studies import StudyMixin
from mlsurv.learner._export import ExportMixin
from mlsurv.learner._training import TrainingMixin


# Map each moved method to the mixin that should now own it (i.e. define it in
# that class's own __dict__, not merely inherit it).
OWNERS = {
    PersistenceMixin: [
        'save_artifact', 'load_artifact', '_update_artifact_path',
        'save_learner', 'load_learner',
    ],
    StudyMixin: [
        'experiment_path', 'list_studies', 'export_study', 'reload_study',
    ],
    ExportMixin: [
        'export_results', '_collect_all_dataframes', '_truncate_sheet_names',
        '_write_xlsx', '_offload_oversized', '_write_csv_dir',
    ],
    TrainingMixin: [
        '_check_cv_vs_holdout_best', '_print_comparison_table',
    ],
}

# Module-level helpers/classes that moved alongside their cluster's methods.
# Each must live in exactly its new module and be gone from _config.
HELPER_HOMES = {
    _persistence: ['_environment_fingerprint', '_ArtifactEncoder'],
    _studies: ['_resolve_tuned_model'],
    _export: [
        '_audit_env_enabled', '_collect_model_eval_dfs',
        '_resolve_paired_bootstrap', '_INVALID_SHEET_CHARS',
        '_oversize_reason', '_csv_filenames', '_sidecar_dir', '_warn_stale_csvs',
    ],
}


def test_moved_methods_live_in_their_new_owner():
    for mixin, names in OWNERS.items():
        for name in names:
            assert name in vars(mixin), (
                f"{name} should be defined on {mixin.__name__}"
            )


def test_moved_methods_not_duplicated_into_other_mixins():
    # A careless future move could re-home a method while leaving a copy
    # behind on the wrong mixin. Assert each method is defined on its owner
    # ONLY, not on any of the other split mixins.
    all_mixins = list(OWNERS)
    for owner, names in OWNERS.items():
        for other in all_mixins:
            if other is owner:
                continue
            for name in names:
                assert name not in vars(other), (
                    f"{name} should live only on {owner.__name__}, "
                    f"but is also defined on {other.__name__}"
                )


def test_config_mixin_no_longer_defines_moved_methods():
    moved = [name for names in OWNERS.values() for name in names]
    for name in moved:
        assert name not in vars(ConfigMixin), (
            f"{name} should have moved out of ConfigMixin"
        )


def test_module_level_helpers_moved_to_their_new_module():
    for module, names in HELPER_HOMES.items():
        for name in names:
            assert hasattr(module, name), (
                f"{name} should live in {module.__name__}"
            )
            assert not hasattr(_config, name), (
                f"{name} should have moved out of _config"
            )


def test_config_mixin_keeps_only_config():
    public = [n for n in vars(ConfigMixin) if not n.startswith('__')]
    assert public == ['config']


def test_learner_still_exposes_every_moved_method():
    for names in OWNERS.values():
        for name in names:
            assert hasattr(SurvivalLearner, name), (
                f"SurvivalLearner lost access to {name} after the split"
            )
