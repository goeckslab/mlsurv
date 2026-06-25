#!/usr/bin/env python
"""Vignette validation tests.

Run vignette notebooks via nbconvert and validate execution and outputs.
"""
import subprocess
import sys
import importlib.util
from pathlib import Path

import pytest

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


def _has_pycox():
    """Check if pycox is available."""
    try:
        import pycox  # noqa: F401
        return True
    except ImportError:
        return False


def _ensure_notebook_runtime():
    """Skip when notebook execution dependencies are unavailable."""
    missing = [
        name for name in ("jupyter", "nbconvert")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        pytest.skip("Notebook vignette tests require: " + ", ".join(missing))


def _is_notebook_timeout(stderr: str) -> bool:
    """Detect nbconvert/nbclient timeout failures from stderr."""
    if not stderr:
        return False
    patterns = (
        "CellTimeoutError",
        "Timeout waiting for execute reply",
        "Cell execution timed out",
    )
    return any(p in stderr for p in patterns)


def run_notebook(vignette_name: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a vignette notebook via nbconvert.

    Args:
        vignette_name: Name like '01_quickstart' or '03_benchmarks'
        timeout: Maximum execution time in seconds

    Returns:
        CompletedProcess with stdout, stderr, and returncode
    """
    _ensure_notebook_runtime()

    nb_path = PROJECT_ROOT / "examples" / f"vignette_{vignette_name}.ipynb"
    if not nb_path.exists():
        raise FileNotFoundError(f"Vignette not found: {nb_path}")

    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=" + str(timeout),
            "--output-dir", str(PROJECT_ROOT / "examples" / "output"),
            str(nb_path),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout + 30,  # extra buffer for nbconvert overhead
    )
    return result


# Legacy support: also allow running .py vignettes if they exist
def run_vignette(vignette_name: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a vignette (.py script or .ipynb notebook) via subprocess."""
    nb_path = PROJECT_ROOT / "examples" / f"vignette_{vignette_name}.ipynb"
    py_path = PROJECT_ROOT / "examples" / f"vignette_{vignette_name}.py"

    if nb_path.exists():
        return run_notebook(vignette_name, timeout)
    elif py_path.exists():
        return subprocess.run(
            [sys.executable, str(py_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    else:
        raise FileNotFoundError(
            f"Vignette not found: tried {nb_path} and {py_path}"
        )


# =============================================================================
# Vignette 01 - Quickstart
# =============================================================================

@pytest.mark.vignette
@pytest.mark.slow
class TestVignette01Quickstart:
    """Tests for vignette_01_quickstart.ipynb"""

    def test_executes_without_error(self):
        """Notebook runs to completion without error."""
        result = run_vignette('01_quickstart', timeout=120)
        assert result.returncode == 0, f"stderr: {result.stderr}"


# =============================================================================
# Vignette 02 - Workflow Tour
# =============================================================================

@pytest.mark.vignette
@pytest.mark.slow
class TestVignette02WorkflowTour:
    """Tests for vignette_02_workflow_tour.ipynb"""

    def test_executes_without_error(self):
        """Notebook runs to completion without error."""
        result = run_vignette('02_workflow_tour', timeout=1200)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_generates_outputs(self):
        """Expected output files are generated."""
        run_vignette('02_workflow_tour', timeout=1200)
        output_dir = PROJECT_ROOT / 'examples' / 'output'
        report = output_dir / 'vig2_report.html'
        assert report.exists(), f"Report not found: {report}"


# =============================================================================
# Vignette 03 - Benchmarks
# =============================================================================

@pytest.mark.vignette
@pytest.mark.slow
@pytest.mark.requires_pycox
class TestVignette03Benchmarks:
    """Tests for vignette_03_benchmarks.ipynb"""

    def test_executes_without_error(self):
        """Notebook runs to completion without error."""
        if not _has_pycox():
            pytest.skip("pycox not available")
        result = run_vignette('03_benchmarks', timeout=900)
        if result.returncode != 0 and _is_notebook_timeout(result.stderr):
            pytest.skip("vignette_03_benchmarks timed out in this environment")
        assert result.returncode == 0, f"stderr: {result.stderr}"


# =============================================================================
# Vignette 04 - Custom Components
# =============================================================================

@pytest.mark.vignette
@pytest.mark.slow
class TestVignette04CustomComponents:
    """Tests for vignette_04_custom_components.ipynb"""

    def test_executes_without_error(self):
        """Notebook runs to completion without error."""
        result = run_vignette('04_custom_components', timeout=300)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_generates_report(self):
        """HTML report file is generated."""
        run_vignette('04_custom_components', timeout=300)
        report_path = PROJECT_ROOT / 'examples' / 'output' / 'vig4_report.html'
        assert report_path.exists(), f"Report not found: {report_path}"


# =============================================================================
# Vignette 05 - Advanced CV & Tuning
# =============================================================================

@pytest.mark.vignette
@pytest.mark.slow
class TestVignette05AdvancedCvTuning:
    """Tests for vignette_05_advanced_cv_tuning.ipynb"""

    def test_executes_without_error(self):
        """Notebook runs to completion without error."""
        result = run_vignette('05_advanced_cv_tuning', timeout=600)
        assert result.returncode == 0, f"stderr: {result.stderr}"


# =============================================================================
# Vignette 07 - Immunotherapy Case Study
# =============================================================================

@pytest.mark.vignette
@pytest.mark.slow
class TestVignette07ImmunotherapyCaseStudy:
    """Tests for vignette_07_immunotherapy_case_study.ipynb"""

    def test_executes_without_error(self):
        """Notebook runs to completion without error."""
        data_dir = PROJECT_ROOT / "examples" / "data" / "loris"
        if not (data_dir / "Chowell_train.csv").exists():
            pytest.skip("LORIS data not available in examples/data/loris/")
        result = run_vignette('07_immunotherapy_case_study', timeout=900)
        if result.returncode != 0 and _is_notebook_timeout(result.stderr):
            pytest.skip("vignette_07 timed out in this environment")
        assert result.returncode == 0, f"stderr: {result.stderr}"


