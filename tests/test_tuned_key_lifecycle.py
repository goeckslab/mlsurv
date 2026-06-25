"""Lifecycle tests for tuned-model storage keys.

``learner.tune()`` returns a dict whose keys agree with ``_state.models``
keys. Tuned results are stored under the plain model name (the ``_tuned``
suffix was removed); ``tune()`` overwrites the untuned model in place. These
tests pin that the returned dict and the write-back targets agree on those
keys.
"""

import warnings

import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import SuggestionWarning

# Tree-model tuning cost scales with forest size, so these key-lifecycle plumbing tests
# pass a tiny Optuna space. Partial spaces merge per-param with the defaults. For a
# multi-model tune() (no model arg) the per-model-name form mis-parses, so the space is
# keyed under the 'model' component, which applies it to every tuned model (here both
# RSF and GBSA are forests sharing n_estimators/max_depth).
_SMALL_TREE_SPACE = {'n_estimators': (10, 20), 'max_depth': (3, 5)}


@pytest.mark.integration
class TestValidateDoesNotClobberTestEvaluation:
    """validate(model=mr, X=X_ext) on an external cohort must NOT overwrite
    the trained-on-test evaluation stored on mr.evaluation. Discovered while
    testing the storage-key fixes end-to-end on vignette 2: with the storage-key fix in
    place, downstream plot_roc(tuned) crashed with a length mismatch because
    Rotterdam (n=2982) predictions had silently replaced the GBSG2 test-set
    (n=206) predictions on rsf_tuned.evaluation.test_set."""

    def test_validate_preserves_test_predictions(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)
        learner.run(models=['rsf'], tune=False,
                    evaluate=True, bootstrap=False,
                    n_importance_repeats=1, verbose=False)

        rsf = learner._state.models['rsf']
        baseline_test_set_len = len(rsf.evaluation.test_set)
        baseline_first_risk = rsf.evaluation.test_set['risk_score'].iloc[0]

        # Build a synthetic external cohort with a different sample count
        X_ext = X.iloc[: len(X) // 2].copy()
        y_ext = y[: len(X) // 2]
        learner.validate(X_ext, y_ext, models=rsf, name='Ext',
                         bootstrap=False, n_importance_repeats=0,
                         verbose=False)

        # rsf.evaluation.test_set must still describe the held-out GBSG2 test
        # set, NOT the external cohort.
        assert len(rsf.evaluation.test_set) == baseline_test_set_len, (
            f"validate() clobbered rsf.evaluation.test_set: "
            f"len went {baseline_test_set_len} -> {len(rsf.evaluation.test_set)}"
        )
        assert rsf.evaluation.test_set['risk_score'].iloc[0] == baseline_first_risk


class TestEvaluateBootstrapWarnOnUnknownKey:
    """Passing a dict whose key is not in ``learner.models`` must emit a
    user-visible ``SuggestionWarning`` rather than silently dropping the
    write-back ("No Silent Data Loss" rule)."""

    @pytest.mark.integration
    def test_evaluate_unknown_key_warns(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)
        learner.run(models=['rsf'], tune=False, evaluate=False,
                    bootstrap=False, verbose=False)

        # Pass the real model under a typo'd key
        rsf = learner._state.models['rsf']
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.evaluate({'rsf_typo': rsf}, n_importance_repeats=1,
                             verbose=False)
        msgs = [str(wi.message) for wi in w
                if issubclass(wi.category, SuggestionWarning)]
        assert any("rsf_typo" in m and "not found" in m for m in msgs), msgs


@pytest.mark.integration
class TestTuneDictKeyAlignment:
    """tune() returned dict keys agree with _state.models keys."""

    def test_tune_all_returns_plain_keys(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=2, random_state=42)
        learner.run(models=['rsf', 'gbsa'], tune=False,
                    evaluate=False, bootstrap=False, verbose=False)

        tuned = learner.tune(n_trials=2, search_spaces={'model': _SMALL_TREE_SPACE},
                             verbose=False)

        assert set(tuned) == {'rsf', 'gbsa'}
        for key in tuned:
            assert tuned[key] is learner._state.models[key]
            assert tuned[key].is_tuned
