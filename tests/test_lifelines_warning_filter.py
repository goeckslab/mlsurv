"""Regression tests for the lifelines pandas-4.0 warning filter.

lifelines (verified on v0.30.3) calls pandas Series ``.var/.mean/.std`` with
positional arguments internally during ``WeibullAFTFitter.fit`` and
``CoxPHFitter.fit``, which emits ``pandas.errors.Pandas4Warning`` (a
``DeprecationWarning`` subclass).  These are not user-actionable from
mlsurv; ``mlsurv/__init__.py`` registers a process-wide filter so they
don't surface to users in vignettes or scripts.

Two layers of protection are tested:

1. The filter is actually registered when ``mlsurv`` is imported.
2. End-to-end: training a lifelines-backed model under default warning
   filters (no ``simplefilter('always')`` reset) does not surface
   pandas-4.0 warnings.
"""

import re
import warnings

import pytest

from mlsurv.learner import SurvivalLearner


def test_pandas4_filter_is_registered():
    """The import-time filter for lifelines pandas-4.0 must be present."""
    pattern = re.compile(r"Starting with pandas version 4")
    matching = [
        f for f in warnings.filters
        if f[0] == "ignore"
        and f[1] is not None
        and pattern.search(f[1].pattern)
    ]
    assert matching, (
        "Expected mlsurv/__init__.py to register a filter for "
        "'Starting with pandas version 4.0' DeprecationWarnings from "
        "lifelines."
    )


@pytest.mark.integration
def test_lifelines_pandas4_warnings_do_not_leak(small_data):
    """Training weibull_aft must not surface lifelines pandas-4.0 warnings."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(verbose=False, cv_folds=2, n_jobs=1)

    # Intentionally do NOT call simplefilter('always') — that would wipe
    # the import-time filter we're testing.  catch_warnings() saves and
    # restores the filter list but leaves it active inside the block.
    with warnings.catch_warnings(record=True) as caught:
        learner.train("weibull_aft")

    leaked = [
        w for w in caught
        if "Starting with pandas version 4.0" in str(w.message)
    ]
    assert leaked == [], (
        f"lifelines pandas-4.0 warnings leaked: "
        f"{[str(w.message) for w in leaked]}"
    )


@pytest.mark.integration
def test_lifelines_pandas4_warnings_do_not_leak_during_tune(small_data):
    """Tuning weibull_aft must not surface lifelines pandas-4.0 warnings.

    The tuning path uses ``catch_warnings(record=True)`` +
    ``simplefilter('always')`` in ``mlsurv.tuning.base._evaluate_model``,
    which wipes the import-time filter — separate per-block filter is
    needed there too.
    """
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(verbose=False, cv_folds=2, n_jobs=1)

    with warnings.catch_warnings(record=True) as caught:
        learner.tune("weibull_aft", n_trials=2)

    leaked = [
        w for w in caught
        if "Starting with pandas version 4.0" in str(w.message)
    ]
    assert leaked == [], (
        f"lifelines pandas-4.0 warnings leaked from tune(): "
        f"{[str(w.message) for w in leaked]}"
    )
