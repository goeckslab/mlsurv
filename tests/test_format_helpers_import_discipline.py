"""Guard test for the format-helper move out of the mixin modules.

The shared display/formatting helpers were moved out of the mixin modules into
``mlsurv/learner/_format.py``. This test pins that move: the 16 helpers must
resolve from ``_format`` and must NOT be re-defined in the old home modules
(``_setup``, ``_tuning``, ``_evaluation_bootstrap``, ``_evaluation_core``).

Importing a helper in those modules is fine — *defining* one is the regression
to block, because it would re-create the cross-mixin formatting coupling this
refactor removed.

Mirrors the AST style of ``tests/test_import_discipline.py``.
"""

import ast
from pathlib import Path

import mlsurv.learner._format as _format

# The 16 helpers that now live in _format.py.
MOVED_HELPERS = [
    # Tier 1 — shared across mixins
    "_print_param_box",
    "_format_model_display",
    "_format_state_overrides",
    "_format_scoring_name",
    # Tier 2 — setup-only
    "_format_pipeline_str",
    "_format_per_model_preprocessing",
    "_format_data_str",
    "_format_cv_str",
    "_format_subpopulation_summary",
    # Tier 2 — tuning-only
    "_format_trials_display",
    "_format_scoring_display",
    "_format_search_display",
    "_format_storage_display",
    # Tier 2 — eval-only
    "_format_eval_times_row",
    "_format_importance_row",
    "_format_bootstrap_row",
]

# Modules that used to define these helpers and must no longer do so.
OLD_HOME_MODULES = [
    "_setup.py",
    "_tuning.py",
    "_evaluation_bootstrap.py",
    "_evaluation_core.py",
]

_LEARNER_DIR = Path(_format.__file__).parent


def _top_level_function_names(path: Path) -> set[str]:
    """Return the names of all module-level function definitions in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def test_helpers_resolve_from_format_module():
    """All 16 moved helpers are defined in _format.py."""
    format_defs = _top_level_function_names(_LEARNER_DIR / "_format.py")
    missing = [name for name in MOVED_HELPERS if name not in format_defs]
    assert not missing, (
        "These helpers should be defined in learner/_format.py but are not:\n  - "
        + "\n  - ".join(missing)
    )
    # Belt-and-suspenders: they are importable attributes too.
    for name in MOVED_HELPERS:
        assert hasattr(_format, name), f"_format.{name} is not importable"


def test_helpers_not_redefined_in_old_modules():
    """The old home modules no longer DEFINE any of the moved helpers."""
    offenders = []
    moved = set(MOVED_HELPERS)
    for module_name in OLD_HOME_MODULES:
        defs = _top_level_function_names(_LEARNER_DIR / module_name)
        for name in sorted(defs & moved):
            offenders.append(f"{module_name}: {name}")
    assert not offenders, (
        "Formatting helper(s) re-defined in an old home module — they must live "
        "only in learner/_format.py (importing them is fine, defining them is "
        "the regression). Offenders:\n  - " + "\n  - ".join(offenders)
    )


def test_no_new_format_helpers_in_old_modules():
    """No NEW box/banner formatter sneaks into an old home module either.

    Directional guard (beyond the frozen 16-name list): any module-level
    ``_format_*`` or ``_print_param_box`` definition in a mixin module re-creates
    the cross-mixin formatting coupling this refactor removed — it belongs in _format.py.
    """
    offenders = []
    for module_name in OLD_HOME_MODULES:
        for name in sorted(_top_level_function_names(_LEARNER_DIR / module_name)):
            if name.startswith("_format_") or name == "_print_param_box":
                offenders.append(f"{module_name}: {name}")
    assert not offenders, (
        "Box/banner formatting helper(s) defined in a mixin module — move them to "
        "learner/_format.py (and add to MOVED_HELPERS if shared). Offenders:\n  - "
        + "\n  - ".join(offenders)
    )
