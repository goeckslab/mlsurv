"""Test fixtures for mlsurv."""

import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv.utils import SurvivalArray


@pytest.fixture(autouse=True)
def _isolate_warning_state():
    """Prevent warning state from leaking between tests."""
    saved_filters = warnings.filters[:]
    yield
    warnings.filters[:] = saved_filters
    for mod in sys.modules.values():
        if mod is not None and getattr(mod, "__name__", "").startswith("mlsurv"):
            reg = getattr(mod, "__warningregistry__", None)
            if reg is not None:
                reg.clear()


@pytest.fixture(autouse=True, scope="session")
def _isolate_optuna_storage(tmp_path_factory):
    """Redirect Optuna SQLite storage to a session-unique tmpdir.

    Prevents worktree collisions when parallel test sessions write to
    the same relative ``./optuna_studies/`` directory.
    """
    isolated_base = tmp_path_factory.mktemp("optuna_studies")

    # Each module holds its own reference to DEFAULT_STORAGE_BASE
    # (Python's ``from X import Y`` creates an independent binding in
    # the importing module's namespace, so patching one module does not
    # affect the others).  We must patch every module that reads the
    # value at runtime.  _config.py uses a function-local
    # ``from ..tuning import DEFAULT_STORAGE_BASE`` which re-reads the
    # attribute from the already-patched ``mlsurv.tuning`` module object
    # each call, so it doesn't need a separate patch target.
    # If you add a new top-level import of DEFAULT_STORAGE_BASE
    # elsewhere, add the module here.
    import mlsurv.tuning as _tuning_pkg
    import mlsurv.tuning.storage as _storage_mod
    import mlsurv.tuning.base as _base_mod
    import mlsurv.tuning.studies as _studies_mod
    import mlsurv.tuning.optuna_tuner as _optuna_tuner_mod
    import mlsurv.learner._format as _format_mod

    _targets = [
        _tuning_pkg, _storage_mod, _base_mod, _studies_mod,
        _optuna_tuner_mod, _format_mod,
    ]
    originals = [(mod, mod.DEFAULT_STORAGE_BASE) for mod in _targets]

    for mod in _targets:
        mod.DEFAULT_STORAGE_BASE = isolated_base

    yield isolated_base

    for mod, original in originals:
        mod.DEFAULT_STORAGE_BASE = original


# ---------------------------------------------------------------------------
# Marker sets excluded by each tier (see tests/README.md for full details).
#   Fast (default) -> _FAST_EXCLUDED
#   Full           -> _FULL_EXCLUDED   (vignettes only)
#   Medium (hidden)-> _MEDIUM_EXCLUDED
#   All / -m "..." -> no exclusion
# ---------------------------------------------------------------------------

_FAST_EXCLUDED = {"slow", "vignette", "integration", "bootstrap_heavy", "parallel", "shap", "full", "validation", "demo", "requires_pycox", "requires_shap"}
_FULL_EXCLUDED = {"slow", "vignette", "validation"}
_MEDIUM_EXCLUDED = {"slow", "vignette", "integration_heavy", "bootstrap_heavy", "parallel", "shap", "full", "validation", "requires_pycox", "requires_shap"}


def _marker_names(item):
    """Return the set of marker names on a test item."""
    return {m.name for m in item.iter_markers()}


def _skip_items_with_markers(items, excluded, reason):
    """Add a skip marker to every item whose markers overlap *excluded*."""
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if _marker_names(item) & excluded:
            item.add_marker(skip)


def pytest_addoption(parser):
    parser.addoption(
        "--run-all",
        action="store_true",
        default=False,
        help="Run all tests including vignettes.",
    )
    parser.addoption(
        "--run-full",
        action="store_true",
        default=False,
        help="Run full suite (excludes vignettes).",
    )
    parser.addoption(
        "--run-vignettes",
        action="store_true",
        default=False,
        help="Run only vignette notebook tests.",
    )
    parser.addoption(
        "--run-medium",
        action="store_true",
        default=False,
        help="(internal) Medium tier: fast + light integration.",
    )


def pytest_collection_modifyitems(config, items):
    # --run-all: everything, no filtering
    if config.getoption("--run-all"):
        return
    # Explicit -m on CLI: respect user's marker filter
    if config.getoption("-m"):
        return
    # --run-vignettes: only keep vignette-marked tests
    if config.getoption("--run-vignettes"):
        skip = pytest.mark.skip(reason="not a vignette test")
        for item in items:
            if "vignette" not in _marker_names(item):
                item.add_marker(skip)
        return
    # --run-full: everything except slow and vignette tests
    if config.getoption("--run-full"):
        _skip_items_with_markers(
            items, _FULL_EXCLUDED, "vignette test (use --run-all to include)"
        )
        return
    # --run-medium (hidden): fast + light integration
    if config.getoption("--run-medium"):
        _skip_items_with_markers(
            items,
            _MEDIUM_EXCLUDED,
            "excluded in medium tier (use --run-full or --run-all to include)",
        )
        return
    # Default (fast): skip tests with excluded markers
    _skip_items_with_markers(
        items,
        _FAST_EXCLUDED,
        "excluded by default (use --run-full or --run-all to include)",
    )


@pytest.fixture
def synthetic_data():
    """
    Generate synthetic survival data for testing.

    Returns
    -------
    tuple
        (X, y) where X is a DataFrame and y is a SurvivalArray.
    """
    np.random.seed(42)
    n_samples, n_features = 200, 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Add some missing values for imputation testing
    X.iloc[10:15, 0] = np.nan
    X.iloc[20:25, 5] = np.nan

    # Generate survival times (exponential + noise)
    true_coef = np.zeros(n_features)
    true_coef[:5] = [0.5, -0.3, 0.4, -0.2, 0.3]  # First 5 features are predictive
    linear_pred = X.fillna(0).values @ true_coef
    times = np.exp(-linear_pred + np.random.randn(n_samples) * 0.5)
    times = np.clip(times, 0.1, 10)

    # Generate censoring (~30% censored)
    events = np.random.binomial(1, 0.7, n_samples).astype(bool)

    # Create SurvivalArray
    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)]
    ))

    return X, y


@pytest.fixture
def synthetic_data_clean():
    """
    Generate synthetic survival data without missing values.

    Returns
    -------
    tuple
        (X, y) where X is a DataFrame and y is a SurvivalArray.
    """
    np.random.seed(42)
    n_samples, n_features = 200, 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Generate survival times
    true_coef = np.zeros(n_features)
    true_coef[:5] = [0.5, -0.3, 0.4, -0.2, 0.3]
    linear_pred = X.values @ true_coef
    times = np.exp(-linear_pred + np.random.randn(n_samples) * 0.5)
    times = np.clip(times, 0.1, 10)

    # Generate events (~30% censored)
    events = np.random.binomial(1, 0.7, n_samples).astype(bool)

    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)]
    ))

    return X, y


@pytest.fixture
def small_data():
    """
    Generate small dataset for quick tests.

    Returns
    -------
    tuple
        (X, y) where X is a DataFrame and y is a SurvivalArray.
    """
    np.random.seed(42)
    n_samples = 50

    X = pd.DataFrame({
        'age': np.random.uniform(30, 80, n_samples),
        'stage': np.random.randint(1, 5, n_samples),
        'grade': np.random.uniform(1, 4, n_samples),
        'tumor_size': np.random.uniform(0.5, 10, n_samples),
        'biomarker': np.random.randn(n_samples),
    })

    times = np.random.exponential(2, n_samples).clip(0.1, 10)
    events = np.random.binomial(1, 0.6, n_samples).astype(bool)

    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)]
    ))

    return X, y


@pytest.fixture
def train_test_data(synthetic_data_clean):
    """
    Split synthetic data into train/test sets.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X, y = synthetic_data_clean
    split_idx = int(len(X) * 0.7)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def _gbsg2_data_raw():
    """Load GBSG2 dataset once per session."""
    try:
        from sksurv.datasets import load_gbsg2
        return load_gbsg2()
    except ImportError:
        pytest.skip("scikit-survival not installed")


@pytest.fixture
def gbsg2_data(_gbsg2_data_raw):
    """
    Load real GBSG2 dataset for integration testing.

    Returns fresh copies per test to prevent mutation issues.

    Returns
    -------
    tuple
        (X, y) from scikit-survival's GBSG2 dataset.
    """
    X, y = _gbsg2_data_raw
    return X.copy(), y.copy()


@pytest.fixture(scope="session")
def _veterans_data_raw():
    """Load Veterans dataset once per session."""
    try:
        from sksurv.datasets import load_veterans_lung_cancer
        return load_veterans_lung_cancer()
    except ImportError:
        pytest.skip("scikit-survival not installed")


@pytest.fixture
def veterans_data(_veterans_data_raw):
    """
    Load Veterans lung cancer dataset for integration testing.

    Returns fresh copies per test to prevent mutation issues.

    Returns
    -------
    tuple
        (X, y) from scikit-survival's Veterans dataset.
    """
    X, y = _veterans_data_raw
    return X.copy(), y.copy()


@pytest.fixture
def large_synthetic_data():
    """
    Generate large synthetic dataset (1000+ samples) for performance/scale testing.

    Returns
    -------
    tuple
        (X, y) where X is a DataFrame and y is a SurvivalArray.
    """
    np.random.seed(42)
    n_samples, n_features = 1000, 30

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Generate survival times
    true_coef = np.zeros(n_features)
    true_coef[:5] = [0.5, -0.3, 0.4, -0.2, 0.3]
    linear_pred = X.values @ true_coef
    times = np.exp(-linear_pred + np.random.randn(n_samples) * 0.5)
    times = np.clip(times, 0.1, 10)

    # Generate events (~30% censored)
    events = np.random.binomial(1, 0.7, n_samples).astype(bool)

    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)]
    ))

    return X, y


@pytest.fixture
def temporal_data():
    """
    Generate dataset with temporal structure for time-series CV tests.

    Simulates patients enrolled over time, where later patients have
    different characteristics.

    Returns
    -------
    tuple
        (X, y, enrollment_time) for time-series cross-validation testing.
    """
    np.random.seed(42)
    n_samples = 200

    # Enrollment time (days from study start)
    enrollment_time = np.sort(np.random.uniform(0, 365, n_samples))

    # Features - later enrollees have slightly different characteristics
    time_effect = enrollment_time / 365
    X = pd.DataFrame({
        'age': np.random.uniform(30, 80, n_samples),
        'biomarker1': np.random.randn(n_samples) + 0.5 * time_effect,
        'biomarker2': np.random.randn(n_samples),
        'treatment_era': (enrollment_time > 180).astype(int),
    })

    times = np.random.exponential(2, n_samples).clip(0.1, 10)
    events = np.random.binomial(1, 0.6, n_samples).astype(bool)

    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)]
    ))

    return X, y, enrollment_time


@pytest.fixture
def multi_site_data():
    """
    Generate dataset with site/group labels for GroupKFold tests.

    Returns
    -------
    tuple
        (X, y, groups) for GroupKFold cross-validation testing.
    """
    np.random.seed(42)
    n_samples = 200
    n_sites = 5

    # Assign samples to sites
    groups = np.repeat(np.arange(n_sites), n_samples // n_sites)
    np.random.shuffle(groups)

    # Site-specific effects
    site_effect = {i: np.random.randn() * 0.3 for i in range(n_sites)}

    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples) + [site_effect[g] for g in groups],
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
    })

    times = np.random.exponential(2, n_samples).clip(0.1, 10)
    events = np.random.binomial(1, 0.6, n_samples).astype(bool)

    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)]
    ))

    return X, y, groups


@pytest.fixture
def repeated_subject_data():
    """Clustered data: 60 subjects x 3 rows each (180 rows).

    Each subject contributes several rows, so a row-level holdout would put
    the same subject in both train and test. Used to test subject-level
    (group-aware) holdout splitting.

    Returns
    -------
    tuple
        (X, y, subject_id) where subject_id is the per-row group label.
    """
    np.random.seed(7)
    n_subjects, rows_per = 60, 3
    n = n_subjects * rows_per
    subject_id = np.repeat(np.arange(n_subjects), rows_per)

    X = pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
        'f3': np.random.randn(n),
    })

    times = np.random.exponential(2, n).clip(0.1, 10)
    events = np.random.binomial(1, 0.6, n).astype(bool)

    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)]
    ))

    return X, y, subject_id


@pytest.fixture
def deterministic_data():
    """
    Generate deterministic synthetic data (no randomness in fixture itself).

    This fixture is specifically for reproducibility tests where we need
    to ensure the data itself is always identical.

    Returns
    -------
    tuple
        (X, y) where X is a DataFrame and y is a SurvivalArray.
    """
    # Deterministic features (no randomness)
    n_samples = 100
    X = pd.DataFrame({
        'feature_0': np.linspace(0, 1, n_samples),
        'feature_1': np.sin(np.linspace(0, 2*np.pi, n_samples)),
        'feature_2': np.cos(np.linspace(0, 2*np.pi, n_samples)),
        'feature_3': np.arange(n_samples) / n_samples,
        'feature_4': (np.arange(n_samples) % 10) / 10,
    })

    # Deterministic survival times (based on features)
    times = 1 + X['feature_0'].values + 0.5 * X['feature_1'].values
    times = np.clip(times, 0.1, 10)

    # Deterministic events (every other sample is event)
    events = (np.arange(n_samples) % 2 == 0).astype(bool)

    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)]
    ))

    return X, y


@pytest.fixture
def high_dimensional_data():
    """
    Generate high-dimensional dataset (100 samples, 100 features) for edge case testing.

    Tests SHAP computation when p ~ n (features similar to samples).

    Returns
    -------
    tuple
        (X, y) where X is a DataFrame and y is a SurvivalArray.
    """
    np.random.seed(42)
    n_samples, n_features = 100, 100

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feat_{i}' for i in range(n_features)]
    )

    times = np.random.exponential(2, n_samples).clip(0.1, 10)
    events = np.random.binomial(1, 0.6, n_samples).astype(bool)

    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)]
    ))

    return X, y


@pytest.fixture
def single_feature_data():
    """
    Generate single-feature dataset for edge case testing.

    Tests SHAP computation with minimal feature dimensionality.

    Returns
    -------
    tuple
        (X, y) where X is a DataFrame and y is a SurvivalArray.
    """
    np.random.seed(42)
    n_samples = 100

    X = pd.DataFrame({'single_feat': np.random.randn(n_samples)})

    times = np.random.exponential(2, n_samples).clip(0.1, 10)
    events = np.random.binomial(1, 0.6, n_samples).astype(bool)

    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)]
    ))

    return X, y
