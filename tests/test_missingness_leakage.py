"""Tests for missingness-outcome leakage detection."""

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import MethodologicalWarning
from mlsurv.learner import SurvivalLearner
from mlsurv.utils import SurvivalArray

from tests._helpers import make_survival_array


def _make_leaky_data(n=200, seed=42):
    """Create data where one feature's missingness perfectly correlates with event.

    Returns (X, y) where ``leaky_feature`` is NaN for all censored patients.
    """
    rng = np.random.RandomState(seed)
    events = rng.binomial(1, 0.5, n).astype(bool)
    times = rng.exponential(5, n)
    y = make_survival_array(events, times)

    X = pd.DataFrame({
        'clean_feature': rng.randn(n),
        'leaky_feature': rng.randn(n),
    })
    # missingness = censored → perfect leakage
    X.loc[~events, 'leaky_feature'] = np.nan
    return X, y


# ---------------------------------------------------------------------------
# Integration tests: setup() leakage detection
# ---------------------------------------------------------------------------

class TestMissingnessLeakageDetection:
    """Tests for missingness-outcome leakage detection in setup()."""

    def test_perfect_leakage_warns(self):
        X, y = _make_leaky_data()
        learner = SurvivalLearner(X, y)
        with pytest.warns(MethodologicalWarning, match="missingness"):
            learner.setup(verbose=False)

    def test_no_missing_no_warning(self):
        """Clean data should not trigger a missingness leakage warning."""
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame({
            'f1': rng.randn(n),
            'f2': rng.randn(n),
        })
        events = rng.binomial(1, 0.5, n).astype(bool)
        y = make_survival_array(events, rng.exponential(5, n))

        learner = SurvivalLearner(X, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner.setup(verbose=False)

        leakage_warnings = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "missingness" in str(w.message)
        ]
        assert leakage_warnings == []

    def test_random_missingness_no_warning(self):
        """Random NaN uncorrelated with event should not trigger warning."""
        rng = np.random.RandomState(42)
        n = 500  # larger sample to stabilize chi-squared
        X = pd.DataFrame({
            'f1': rng.randn(n),
            'f2': rng.randn(n),
        })
        events = rng.binomial(1, 0.5, n).astype(bool)
        y = make_survival_array(events, rng.exponential(5, n))

        # Random 30% missing, independent of event
        mask = rng.rand(n) < 0.3
        X.loc[mask, 'f1'] = np.nan

        learner = SurvivalLearner(X, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner.setup(verbose=False)

        leakage_warnings = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "missingness" in str(w.message)
        ]
        assert leakage_warnings == []

    def test_check_leakage_false_suppresses(self):
        """check_leakage=False should suppress the warning."""
        X, y = _make_leaky_data()
        learner = SurvivalLearner(X, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner.setup(verbose=False, check_leakage=False)

        leakage_warnings = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "missingness" in str(w.message)
        ]
        assert leakage_warnings == []

    def test_warning_contains_feature_name_and_stats(self):
        X, y = _make_leaky_data()
        learner = SurvivalLearner(X, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner.setup(verbose=False)

        leakage_warnings = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "missingness" in str(w.message)
        ]
        assert len(leakage_warnings) == 1
        msg = str(leakage_warnings[0].message)
        assert "leaky_feature" in msg
        assert "Cramer's V=" in msg
        assert "missing)" in msg

    def test_multiple_leaky_features(self):
        """Two leaky features should both appear in a single warning."""
        rng = np.random.RandomState(42)
        n = 200
        events = rng.binomial(1, 0.5, n).astype(bool)
        times = rng.exponential(5, n)
        y = make_survival_array(events, times)

        X = pd.DataFrame({
            'clean': rng.randn(n),
            'leaky_a': rng.randn(n),
            'leaky_b': rng.randn(n),
        })
        X.loc[~events, 'leaky_a'] = np.nan
        X.loc[~events, 'leaky_b'] = np.nan

        learner = SurvivalLearner(X, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner.setup(verbose=False)

        leakage_warnings = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "missingness" in str(w.message)
        ]
        assert len(leakage_warnings) == 1
        msg = str(leakage_warnings[0].message)
        assert "leaky_a" in msg
        assert "leaky_b" in msg
        assert "2 features" in msg

    def test_all_censored_no_warning(self):
        """All-censored data (no events) should skip the check entirely."""
        from mlsurv.learner._setup import _check_missingness_leakage

        rng = np.random.RandomState(42)
        n = 200
        events = np.zeros(n, dtype=bool)
        times = rng.exponential(5, n)
        y = make_survival_array(events, times)

        X = pd.DataFrame({'f1': rng.randn(n), 'f2': rng.randn(n)})
        X.iloc[:50, 1] = np.nan

        assert _check_missingness_leakage(X, y) == []

    def test_all_events_no_warning(self):
        """All-event data (no censoring) should skip the check entirely."""
        from mlsurv.learner._setup import _check_missingness_leakage

        rng = np.random.RandomState(42)
        n = 200
        events = np.ones(n, dtype=bool)
        times = rng.exponential(5, n)
        y = make_survival_array(events, times)

        X = pd.DataFrame({'f1': rng.randn(n), 'f2': rng.randn(n)})
        X.iloc[:50, 1] = np.nan

        assert _check_missingness_leakage(X, y) == []

    def test_all_nan_column_skipped(self):
        """A column that is entirely NaN should be skipped, not flagged."""
        rng = np.random.RandomState(42)
        n = 200
        events = rng.binomial(1, 0.5, n).astype(bool)
        times = rng.exponential(5, n)
        y = make_survival_array(events, times)

        X = pd.DataFrame({
            'clean': rng.randn(n),
            'all_nan': np.full(n, np.nan),
            'leaky': rng.randn(n),
        })
        X.loc[~events, 'leaky'] = np.nan

        learner = SurvivalLearner(X, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner.setup(verbose=False)

        leakage_warnings = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "missingness" in str(w.message)
        ]
        assert len(leakage_warnings) == 1
        msg = str(leakage_warnings[0].message)
        assert "leaky" in msg
        assert "all_nan" not in msg

    def test_singular_grammar(self):
        """Single leaky feature produces '1 feature has' (not 'features have')."""
        X, y = _make_leaky_data()
        learner = SurvivalLearner(X, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner.setup(verbose=False)

        leakage_warnings = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "missingness" in str(w.message)
        ]
        msg = str(leakage_warnings[0].message)
        assert "1 feature has" in msg

    def test_presplit_data(self):
        """Pre-split data should also detect leakage (on training data)."""
        rng = np.random.RandomState(42)
        n = 200
        events = rng.binomial(1, 0.5, n).astype(bool)
        times = rng.exponential(5, n)

        X = pd.DataFrame({
            'clean': rng.randn(n),
            'leaky': rng.randn(n),
        })
        X.loc[~events, 'leaky'] = np.nan

        # Split manually
        split = int(0.7 * n)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_all = make_survival_array(events, times)
        y_train = SurvivalArray(y_all[:split])
        y_test = SurvivalArray(y_all[split:])

        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        )
        with pytest.warns(MethodologicalWarning, match="missingness"):
            learner.setup(verbose=False)
