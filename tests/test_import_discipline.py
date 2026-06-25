"""Import-discipline guardrail for the ``learner -> viz`` layering.

``mlsurv.viz`` (and the other lower layers ``evaluation``, ``models``,
``preprocessing``, ``results``) must sit *below* ``mlsurv.learner``: code in
those packages may be imported by ``learner`` but must not import ``learner``
back at runtime. The old ``viz <-> learner`` cycle was dissolved by
relocating the viz data-prep layer (``_visualization_impl`` -> ``viz/_data``,
``_visualization_helpers`` -> ``viz/_resolve``) and moving the lone
``_warn_unused_dict_keys`` helper down into ``mlsurv._shared``.

This test turns that architecture rule into a regression net: it AST-scans
every module under the lower-layer packages and fails if any of them imports
``mlsurv.learner`` at runtime. ``TYPE_CHECKING``-only imports (for type hints)
are allowed, since they never execute.

Uses AST rather than a regex (cf. ``test_resolve_model_results.py``) so it can tell a
runtime import from one guarded by ``if TYPE_CHECKING:``, and so it catches
every import spelling (absolute, relative, and ``from mlsurv import learner``).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_MLSURV_ROOT = Path(__file__).resolve().parent.parent / "mlsurv"
_REPO_ROOT = _MLSURV_ROOT.parent  # dir that contains the ``mlsurv`` package

# Lower layers that must never import ``mlsurv.learner`` at runtime.
_GUARDED_DIRS = ("viz", "evaluation", "models", "preprocessing", "results")

# Per-file allowlist of runtime ``mlsurv.learner.*`` import prefixes. Empty
# since a later refactor moved the last shared recalibration helpers
# (``_apply_recalibrators_to_array`` / ``_collect_oof_survival`` /
# ``_fit_isotonic_recalibrators``) down into ``evaluation/_recalibration.py``,
# closing the ``_prediction_bootstrap -> learner._recalibration`` edge — the
# lower layers are now zero-tolerance. The mechanism stays for any future
# tracked edge: map a file to a tuple of allowed prefixes; an import of any
# OTHER learner module in that file still trips the guardrail (the allowance
# is per-target, not a blanket per-file skip).
_ALLOWED_LEARNER_PREFIXES: dict = {}


def _is_type_checking_test(test: ast.expr) -> bool:
    """True if *test* is ``TYPE_CHECKING`` or ``<mod>.TYPE_CHECKING``."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _type_checking_import_ids(tree: ast.Module) -> set[int]:
    """Collect ``id()`` of every import nested under an ``if TYPE_CHECKING:``."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_test(node.test):
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        guarded.add(id(sub))
    return guarded


def _package_parts(path: Path) -> tuple[str, ...]:
    """Dotted package parts used to resolve relative imports in *path*.

    For ``mlsurv/evaluation/foo.py`` (and ``__init__.py``) this is
    ``("mlsurv", "evaluation")``. All guarded modules live exactly three
    levels deep (``mlsurv/<pkg>/<file>.py``), so the deepest legal relative
    import is two dots (``from ..x import y``); the resolution below assumes
    that depth.
    """
    parts = path.relative_to(_REPO_ROOT).with_suffix("").parts
    return parts[:-1]


def _resolved_targets(node: ast.AST, pkg_parts: tuple[str, ...]) -> list[str]:
    """Return the dotted module target(s) an import node refers to.

    For ``ImportFrom`` the imported *names* are appended to the base module so
    that subpackage-by-name imports (``from mlsurv import learner``,
    ``from .. import learner``) are resolved to ``mlsurv.learner``, not just
    the bare ``mlsurv`` package.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.level == 0:
            base = node.module or ""
        else:
            up = pkg_parts[: len(pkg_parts) - (node.level - 1)]
            suffix = tuple(node.module.split(".")) if node.module else ()
            base = ".".join((*up, *suffix))
        targets = [base] if base else []
        for alias in node.names:
            targets.append(f"{base}.{alias.name}" if base else alias.name)
        return targets
    return []


def _imports_learner(target: str) -> bool:
    return target == "mlsurv.learner" or target.startswith("mlsurv.learner.")


def _runtime_learner_imports(path: Path) -> list[tuple[int, str]]:
    """Return ``(lineno, target)`` for each runtime ``mlsurv.learner`` import."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    guarded = _type_checking_import_ids(tree)
    pkg_parts = _package_parts(path)

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if id(node) in guarded:
            continue  # TYPE_CHECKING-only import — never executes
        for target in _resolved_targets(node, pkg_parts):
            if _imports_learner(target):
                hits.append((node.lineno, target))
    return hits


def _iter_guarded_files():
    for sub in _GUARDED_DIRS:
        yield from sorted((_MLSURV_ROOT / sub).rglob("*.py"))


def test_lower_layers_do_not_import_learner_at_runtime():
    """No runtime ``mlsurv.learner`` import under the lower-layer packages.

    A small per-file allowance exists for known, tracked pre-existing edges
    (see ``_ALLOWED_LEARNER_PREFIXES``); any import of a learner module outside
    that allowance is reported, including new ones added to an allow-listed
    file.
    """
    offenders: list[str] = []

    for path in _iter_guarded_files():
        allowed = _ALLOWED_LEARNER_PREFIXES.get(path, ())
        for lineno, target in _runtime_learner_imports(path):
            if any(target.startswith(prefix) for prefix in allowed):
                continue
            rel = path.relative_to(_MLSURV_ROOT)
            offenders.append(f"{rel.as_posix()}:{lineno} -> {target}")

    assert not offenders, (
        "Runtime `mlsurv.learner` imports found in lower-layer packages "
        "(must be TYPE_CHECKING-only to keep the `learner -> viz` layering):\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Self-tests: lock in the detection logic so a future simplification of the
# helpers can't silently neuter the guardrail (it would still pass on a clean
# tree, but these positive cases would fail).
# ---------------------------------------------------------------------------

# Each module lives at ``mlsurv/viz/<file>.py`` for these cases, so the package
# is ``("mlsurv", "viz")``.
_VIZ_PKG = ("mlsurv", "viz")


@pytest.mark.parametrize(
    "src, should_flag",
    [
        # Runtime learner imports — every spelling must be caught.
        ("from mlsurv.learner._core import SurvivalLearner", True),
        ("import mlsurv.learner._core", True),
        ("from ..learner._recalibration import foo", True),       # relative
        ("from .. import learner", True),                          # subpkg by name (relative)
        ("from mlsurv import learner", True),                      # subpkg by name (absolute)
        # Non-learner imports — must NOT be flagged.
        ("from mlsurv.viz._data import get_survival_test_data", False),
        ("from ._resolve import resolve_eval_result", False),
        ("from ..models import get_model_metadata", False),
        ("import numpy as np", False),
    ],
)
def test_detection_flags_runtime_learner_imports(src, should_flag):
    node = ast.parse(src).body[0]
    targets = _resolved_targets(node, _VIZ_PKG)
    assert any(_imports_learner(t) for t in targets) is should_flag


def test_type_checking_guard_exempts_learner_import():
    src = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from mlsurv.learner._core import SurvivalLearner\n"
    )
    tree = ast.parse(src)
    guarded = _type_checking_import_ids(tree)
    import_node = tree.body[1].body[0]  # the import inside the if-block
    assert id(import_node) in guarded


def test_allowlisted_file_still_flags_a_different_learner_module(tmp_path):
    """The per-target allowance must not mask a new, different learner edge."""
    pkg_parts = ("mlsurv", "evaluation")
    new_edge = ast.parse("from ..learner._core import SurvivalLearner").body[0]
    targets = _resolved_targets(new_edge, pkg_parts)
    allowed = ("mlsurv.learner._recalibration",)
    learner_targets = [t for t in targets if _imports_learner(t)]
    assert learner_targets  # it IS a learner import
    assert not any(
        t.startswith(p) for t in learner_targets for p in allowed
    ), "a _core import must not be covered by the _recalibration allowance"
