"""Quick test to verify get_metric() matches verbose patterns.

Refactored from module-level code into proper test functions so markers
can control when these tests run (they train models).
"""
import pytest
import numpy as np


pytestmark = pytest.mark.integration

# RSF tuning cost scales with forest size, so this test passes a tiny Optuna space.
# Partial spaces merge per-param with the defaults. The asserted equality is between
# two views of the same metric, so it is independent of forest size.
_SMALL_TREE_SPACE = {'n_estimators': (10, 20), 'max_depth': (3, 5)}


def _make_learner():
    """Create a fresh learner with GBSG2 subset."""
    from sksurv.datasets import load_gbsg2
    from mlsurv import SurvivalLearner
    from mlsurv.utils import to_structured_array

    X, y_raw = load_gbsg2()
    X = X.head(100).copy()
    X['horTh'] = (X['horTh'] == 'yes').astype(int)
    X['menostat'] = (X['menostat'] == 'Post').astype(int)
    X['tgrade'] = X['tgrade'].map({'I': 1, 'II': 2, 'III': 3}).astype(int)
    y = to_structured_array(y_raw['time'][:100], y_raw['cens'][:100])

    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=42)
    return learner


@pytest.fixture(scope="module")
def trained_learner():
    """Learner with coxph trained for read-only tests (shared, not mutated)."""
    learner = _make_learner()
    learner.run(models=['coxph'], verbose=False, n_trials=1)
    return learner


class TestGetMetricVignettes:
    """Verify get_metric() matches verbose patterns from vignettes."""

    def test_model_get_metric_cv(self, trained_learner):
        """ModelResult.get_metric for CV matches verbose pattern (vignette_05)."""
        metrics = trained_learner.summary()
        row = metrics[(metrics['model'] == 'coxph')
                      & (metrics['metric'] == 'harrell')
                      & (metrics['phase'] == 'cv')]
        verbose_cv = row['mean'].values[0]
        simple_cv = trained_learner.models['coxph'].get_metric('harrell', 'cv')

        assert np.isclose(verbose_cv, simple_cv), (
            f"CV mismatch: {verbose_cv} vs {simple_cv}"
        )

    def test_model_get_metric_test(self, trained_learner):
        """ModelResult.get_metric for test matches verbose pattern (vignette_04)."""
        eval_result = trained_learner.evaluation_result
        conc_df = eval_result.concordance
        verbose_test = conc_df[conc_df['model_type'] == 'coxph']['c_index_test'].iloc[0]
        simple_test = trained_learner.models['coxph'].get_metric('harrell', 'test')

        assert np.isclose(verbose_test, simple_test), (
            f"Test mismatch: {verbose_test} vs {simple_test}"
        )

    @pytest.mark.integration_heavy
    def test_evaluation_result_property(self):
        """EvaluationResult properties match verbose pattern (vignette_02)."""
        learner = _make_learner()

        tuned = learner.tune('rsf', n_trials=1, search_space=_SMALL_TREE_SPACE)
        eval_result = learner.evaluate(tuned)
        verbose_eval = eval_result.concordance['c_index_test'].iloc[0]
        simple_eval = eval_result.c_index_test

        assert np.isclose(verbose_eval, simple_eval), (
            f"Eval mismatch: {verbose_eval} vs {simple_eval}"
        )
