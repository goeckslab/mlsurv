"""Smoke tests for demo scripts — verifies each runs without error and produces HTML."""
import importlib.util
import types
import warnings
from pathlib import Path

import pytest

pytestmark = [pytest.mark.demo, pytest.mark.integration]

_SCRIPTS = Path(__file__).parent.parent / "scripts"

pytest.importorskip("sksurv", reason="scikit-survival not installed")


def _load(path: Path) -> types.ModuleType:
    """Import a demo script by file path and return its module."""
    if not path.exists():
        pytest.skip(f"Demo script not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_demo(script_path: Path, tmp_path: Path, **kwargs) -> None:
    """Load a demo module, run main() with warnings suppressed, and verify HTML output."""
    mod = _load(script_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod.main(tmp_path, **kwargs)
    files = list(tmp_path.glob("*.html"))
    assert files, "No HTML files written"
    for f in files:
        content = f.read_text(encoding="utf-8")
        assert len(content) > 1_000, f"{f.name} too small ({len(content)} bytes)"
        assert "plotly" in content.lower(), f"{f.name} contains no Plotly content"


class TestDemoSmoke:
    def test_feature_importance(self, tmp_path):
        _run_demo(_SCRIPTS / "demo_feature_importance.py", tmp_path, fast=True)
        assert (tmp_path / "demo_feature_importance.html").exists()

    @pytest.mark.requires_shap
    def test_feature_groups(self, tmp_path):
        _run_demo(_SCRIPTS / "demo_feature_groups.py", tmp_path)
        assert (tmp_path / "feature_groups_demo.html").exists()

    def test_multi_panel(self, tmp_path):
        _run_demo(_SCRIPTS / "demo_multi_panel.py", tmp_path, fast=True)
        assert (tmp_path / "demo_multi_panel.html").exists()

    def test_sksurv_replication(self, tmp_path):
        mod = _load(_SCRIPTS / "sksurv_replication.py")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows = mod.main(output_dir=tmp_path, fast=True)
        assert len(rows) == 4, f"Expected 4 rows, got {len(rows)}"
        html = tmp_path / "sksurv_replication.html"
        assert html.exists()
        content = html.read_text(encoding="utf-8")
        assert len(content) > 1_000, f"HTML too small ({len(content)} bytes)"
        assert "plotly" in content.lower(), "HTML missing Plotly content"
        for r in rows:
            assert r["replicated"] > 0.5, (
                f"{r['dataset']}/{r['model']} replicated C-index {r['replicated']:.3f} "
                f"not above random"
            )
            assert r["tuned"] > 0.5, (
                f"{r['dataset']}/{r['model']} tuned C-index {r['tuned']:.3f} "
                f"not above random"
            )
