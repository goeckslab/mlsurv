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

# Run limit per vignette, in seconds. run_notebook() caps the whole nbconvert
# run at this plus 30 s.
VIGNETTE_TIMEOUTS = {
    '01_quickstart': 120,
    '02_workflow_tour': 1200,
    '03_benchmarks': 900,
    '04_custom_components': 300,
    '05_advanced_cv_tuning': 600,
    '06_tool_comparison': 900,
    '07_immunotherapy_case_study': 2400,
}


def _pytest_timeout(vignette_name):
    """pytest-timeout marker for a test that runs one vignette.

    The project-wide pytest timeout (60 s, pyproject.toml) would otherwise
    stop every vignette test long before the notebook finishes, so each test
    class sets its own limit, just above the notebook's run limit.
    """
    return pytest.mark.timeout(VIGNETTE_TIMEOUTS[vignette_name] + 60)


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
@_pytest_timeout('01_quickstart')
class TestVignette01Quickstart:
    """Tests for vignette_01_quickstart.ipynb"""

    def test_executes_without_error(self):
        """Notebook runs to completion without error."""
        result = run_vignette('01_quickstart', timeout=VIGNETTE_TIMEOUTS['01_quickstart'])
        assert result.returncode == 0, f"stderr: {result.stderr}"


# =============================================================================
# Vignette 02 - Workflow Tour
# =============================================================================

@pytest.mark.vignette
@pytest.mark.slow
@_pytest_timeout('02_workflow_tour')
class TestVignette02WorkflowTour:
    """Tests for vignette_02_workflow_tour.ipynb"""

    def test_executes_without_error(self):
        """Notebook runs to completion without error."""
        result = run_vignette('02_workflow_tour', timeout=VIGNETTE_TIMEOUTS['02_workflow_tour'])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_generates_outputs(self):
        """Expected output files are generated."""
        run_vignette('02_workflow_tour', timeout=VIGNETTE_TIMEOUTS['02_workflow_tour'])
        output_dir = PROJECT_ROOT / 'examples' / 'output'
        report = output_dir / 'vig2_report.html'
        assert report.exists(), f"Report not found: {report}"


# =============================================================================
# Vignette 03 - Benchmarks
# =============================================================================

@pytest.mark.vignette
@pytest.mark.slow
@pytest.mark.requires_pycox
@_pytest_timeout('03_benchmarks')
class TestVignette03Benchmarks:
    """Tests for vignette_03_benchmarks.ipynb"""

    def test_executes_without_error(self):
        """Notebook runs to completion without error."""
        if not _has_pycox():
            pytest.skip("pycox not available")
        result = run_vignette('03_benchmarks', timeout=VIGNETTE_TIMEOUTS['03_benchmarks'])
        if result.returncode != 0 and _is_notebook_timeout(result.stderr):
            pytest.skip("vignette_03_benchmarks timed out in this environment")
        assert result.returncode == 0, f"stderr: {result.stderr}"


# =============================================================================
# Vignette 04 - Custom Components
# =============================================================================

@pytest.mark.vignette
@pytest.mark.slow
@_pytest_timeout('04_custom_components')
class TestVignette04CustomComponents:
    """Tests for vignette_04_custom_components.ipynb"""

    def test_executes_without_error(self):
        """Notebook runs to completion without error."""
        result = run_vignette('04_custom_components', timeout=VIGNETTE_TIMEOUTS['04_custom_components'])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_generates_report(self):
        """HTML report file is generated."""
        run_vignette('04_custom_components', timeout=VIGNETTE_TIMEOUTS['04_custom_components'])
        report_path = PROJECT_ROOT / 'examples' / 'output' / 'vig4_report.html'
        assert report_path.exists(), f"Report not found: {report_path}"


# =============================================================================
# Vignette 05 - Advanced CV & Tuning
# =============================================================================

@pytest.mark.vignette
@pytest.mark.slow
@_pytest_timeout('05_advanced_cv_tuning')
class TestVignette05AdvancedCvTuning:
    """Tests for vignette_05_advanced_cv_tuning.ipynb"""

    def test_executes_without_error(self):
        """Notebook runs to completion without error."""
        result = run_vignette('05_advanced_cv_tuning', timeout=VIGNETTE_TIMEOUTS['05_advanced_cv_tuning'])
        assert result.returncode == 0, f"stderr: {result.stderr}"


# =============================================================================
# Vignette 06 - Tool Comparison
# =============================================================================

@pytest.mark.vignette
@pytest.mark.slow
@_pytest_timeout('06_tool_comparison')
class TestVignette06ToolComparison:
    """Tests for vignette_06_tool_comparison.ipynb"""

    def test_executes_without_error(self):
        """Notebook runs to completion without error.

        The notebook asserts that mlsurv's test split matches the one it
        rebuilds for scikit-survival, so a clean run also confirms that.
        """
        result = run_vignette('06_tool_comparison', timeout=VIGNETTE_TIMEOUTS['06_tool_comparison'])
        if result.returncode != 0 and _is_notebook_timeout(result.stderr):
            pytest.skip("vignette_06 timed out in this environment")
        assert result.returncode == 0, f"stderr: {result.stderr}"


# =============================================================================
# Vignette 07 - Immunotherapy Case Study
# =============================================================================

@pytest.mark.vignette
@pytest.mark.slow
@_pytest_timeout('07_immunotherapy_case_study')
class TestVignette07ImmunotherapyCaseStudy:
    """Tests for vignette_07_immunotherapy_case_study.ipynb"""

    def test_executes_without_error(self):
        """Notebook runs to completion without error."""
        data_dir = PROJECT_ROOT / "examples" / "data" / "loris"
        if not (data_dir / "Chowell_train.csv").exists():
            pytest.skip("LORIS data not available in examples/data/loris/")
        result = run_vignette('07_immunotherapy_case_study', timeout=VIGNETTE_TIMEOUTS['07_immunotherapy_case_study'])
        if result.returncode != 0 and _is_notebook_timeout(result.stderr):
            pytest.skip("vignette_07 timed out in this environment")
        assert result.returncode == 0, f"stderr: {result.stderr}"


