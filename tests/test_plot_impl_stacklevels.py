"""Lock in user-code attribution for warnings emitted by the
``_plot_X_impl`` family.

After the R3 source-dispatcher refactor, warnings emitted from
deep within the dispatch chain must still attribute to the user's
code (not framework internals) so ``warnings.filterwarnings(..., module=...)``
keeps working and the "no silent data loss" contract from
``CONTRIBUTING.md`` is preserved.

These tests guard the stacklevels in:
- ``_iter_subpopulation_levels`` (mlsurv/viz/_plot_impl_shared.py) — used
  by calibration and ROC subpop helpers.
- ``_compute_roc_ci_data_main`` (mlsurv/viz/_plot_impl_roc.py) — emits
  the "Bootstrap CI bands could not be computed" InternalWarning.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import InternalWarning, SuggestionWarning


def _make_subpop_learner(synthetic_data_clean):
    """Build a learner with one subpopulation variable and a single
    trained CoxPH; corrupt half the test-set labels with NaN to force
    the drop-and-warn path inside _iter_subpopulation_levels.
    """
    X, y = synthetic_data_clean
    median_val = X["feature_0"].median()
    labels = np.where(X["feature_0"] > median_val, "high", "low")
    learner = SurvivalLearner(X, y)
    learner.setup(subpopulations={"risk_group": labels}, random_state=42)
    learner.run(models=["coxph"])
    learner.evaluate()

    test_labels = learner._state.subpopulations_test["risk_group"]
    corrupted = test_labels.astype(object).copy()
    corrupted[: len(corrupted) // 2] = np.nan
    learner._state.subpopulations_test["risk_group"] = corrupted
    return learner


def test_iter_subpopulation_levels_nan_drop_warns_at_user_code(synthetic_data_clean):
    """_iter_subpopulation_levels NaN-drop SuggestionWarning must
    attribute to the test file (user code), not framework internals."""
    learner = _make_subpop_learner(synthetic_data_clean)
    view = learner.subpopulation("risk_group")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # plot_calibration may raise downstream (per-level failures on
        # the corrupted fixture); the NaN-drop warning fires first
        # inside _iter_subpopulation_levels and is what we're locking
        # in here.
        try:
            view.plot_calibration()
        except ValueError:
            pass

    drop_warnings = [
        w for w in caught
        if issubclass(w.category, SuggestionWarning)
        and "Dropped" in str(w.message)
        and "risk_group" in str(w.message)
    ]
    assert drop_warnings, (
        f"expected SuggestionWarning mentioning 'Dropped' and "
        f"'risk_group'; got: "
        f"{[(w.category.__name__, str(w.message)) for w in caught]}"
    )
    warn_filename = drop_warnings[0].filename.replace("\\", "/")
    assert warn_filename.endswith("test_plot_impl_stacklevels.py"), (
        f"expected warning attributed to test_plot_impl_stacklevels.py; "
        f"got {warn_filename!r}"
    )


def test_compute_roc_ci_data_main_warns_at_user_code(
    synthetic_data_clean, monkeypatch,
):
    """_compute_roc_ci_data_main InternalWarning must attribute to the
    test file (user code), not framework internals.

    The fallback path fires when bootstrap_roc_curves raises ValueError;
    we force it via monkeypatch rather than relying on a contrived data
    setup.
    """
    X, y = synthetic_data_clean
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42)
    learner.run(models=["coxph"])
    learner.evaluate()
    learner.bootstrap(n_bootstrap=2, include_predictions=True)

    def _raise_value_error(*args, **kwargs):
        raise ValueError("forced failure for stacklevel test")

    # Patch the symbol that _compute_roc_ci_data_main calls. The function
    # is imported at call time inside _plot_roc_impl_main, so we patch
    # the canonical location (mlsurv.viz.metrics).
    import mlsurv.viz.metrics as _metrics
    monkeypatch.setattr(_metrics, "bootstrap_roc_curves", _raise_value_error)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        learner.plot_roc(show_ci=True)

    ci_warnings = [
        w for w in caught
        if issubclass(w.category, InternalWarning)
        and "Bootstrap CI bands could not be computed" in str(w.message)
    ]
    assert ci_warnings, (
        f"expected InternalWarning about Bootstrap CI bands; got: "
        f"{[(w.category.__name__, str(w.message)) for w in caught]}"
    )
    warn_filename = ci_warnings[0].filename.replace("\\", "/")
    assert warn_filename.endswith("test_plot_impl_stacklevels.py"), (
        f"expected warning attributed to test_plot_impl_stacklevels.py; "
        f"got {warn_filename!r}"
    )


@pytest.fixture(scope="module")
def _no_ci_learner():
    """Module-scoped trained + evaluated learner whose bootstrap collected
    neither predictions nor time-resolved AUC — so an explicit show_ci=True
    request cannot be honored and must warn. Inlined data (module scope can't
    use the function-scoped synthetic_data_clean fixture)."""
    rng = np.random.RandomState(0)
    n, p = 160, 6
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    lin = X["f0"] * 0.8 - X["f1"] * 0.5
    t = rng.exponential(np.exp(-lin) * 12 + 1.0)
    c = rng.exponential(20, size=n)
    y = np.array(list(zip(t <= c, np.minimum(t, c))),
                 dtype=[("event", bool), ("time", float)])
    learner = SurvivalLearner(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(random_state=42)
        learner.run(models=["coxph"], tune=False, n_importance_repeats=0,
                    verbose=False)
        learner.evaluate(verbose=False)
        learner.bootstrap(n_bootstrap=2, verbose=False, show_progress=False)
        # Validation cohort with no bootstrap → no cohort CI data either.
        learner.validate(X.iloc[:40], y[:40], name="V", bootstrap=False,
                         verbose=False)
    return learner


def _assert_attributed_to_user_code(caught, needle, category=SuggestionWarning):
    matched = [
        w for w in caught
        if issubclass(w.category, category) and needle in str(w.message)
    ]
    assert matched, (
        f"expected {category.__name__} matching {needle!r}; got: "
        f"{[(w.category.__name__, str(w.message)) for w in caught]}"
    )
    warn_filename = matched[0].filename.replace("\\", "/")
    assert warn_filename.endswith("test_plot_impl_stacklevels.py"), (
        f"expected warning attributed to test_plot_impl_stacklevels.py; "
        f"got {warn_filename!r}"
    )


@pytest.mark.integration
def test_roc_missing_ci_suggestion_warns_at_user_code(_no_ci_learner):
    """The explicit-show_ci=True 'no bootstrap predictions' SuggestionWarning
    on the main ROC path must attribute to user code."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _no_ci_learner.plot_roc(show_ci=True)
    _assert_attributed_to_user_code(
        caught, "No bootstrap predictions for ROC CI bands")


@pytest.mark.integration
def test_metric_time_missing_ci_suggestion_warns_at_user_code(_no_ci_learner):
    """The explicit-show_ci=True 'no time-resolved CI' SuggestionWarning on
    the main plot_metric_time path must attribute to user code.

    Uses metric='brier': AUC(t) is collected on every bootstrap (cheap-auto),
    so Brier(t) is the metric whose ribbon is missing without
    include_time_resolved. metric='brier' routes through the multi-panel
    builder, exercising the _warn_missing_metric_ribbons stacklevel path."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _no_ci_learner.plot_metric_time(metric="brier", show_ci=True)
    _assert_attributed_to_user_code(
        caught, "No bootstrap CI data for the Brier(t) ribbon")


@pytest.mark.integration
def test_validation_roc_missing_ci_warns_at_user_code(_no_ci_learner):
    """The validation-path ROC SuggestionWarning (stacklevel=6) must
    attribute to user code, not the framework."""
    view = _no_ci_learner.validation("V")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        view.plot_roc(show_ci=True)
    _assert_attributed_to_user_code(
        caught, "No bootstrap predictions for ROC CI bands")


@pytest.mark.integration
def test_validation_metric_time_missing_ci_warns_at_user_code(_no_ci_learner):
    """The validation-path metric_time SuggestionWarning (stacklevel=6) must
    attribute to user code."""
    view = _no_ci_learner.validation("V")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        view.plot_metric_time(show_ci=True)
    _assert_attributed_to_user_code(
        caught, "No bootstrap CI data for the AUC(t) ribbon")
