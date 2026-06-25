"""Importance-test results invalidate alongside model mutations.

Stored ``importance_test()`` results describe a specific fitted pipeline, so
they ride on the model's ``ModelResult`` (``result.pimp_importance_test`` /
``result.bootstrap_importance_test``). Re-training or re-tuning a model
replaces the ``ModelResult`` and drops its (now stale) tables for free. The
historical ``pimp_importance_test_results`` /
``bootstrap_importance_test_results`` names survive as read-only views.

Re-setup invalidation is covered here too: clearing ``_state.models``
drops everything riding on the results. The uniform overwrite warning emitted by
``_warn_overwrite`` is covered by the fast-tier
``TestWarnOverwriteHelper`` and the integration-marked retrain / run(tune=True)
classes below.

Tests that only call ``setup()`` on small synthetic data (no training) stay in
the fast tier.
"""

import numpy as np
import pandas as pd
import pytest

from mlsurv.learner import SurvivalLearner
from mlsurv.results.containers import ModelResult
from mlsurv.utils import SurvivalArray


@pytest.fixture
def setup_learner():
    """A learner that is set up but has trained nothing."""
    rng = np.random.default_rng(42)
    n = 60
    X = pd.DataFrame({
        "age": rng.standard_normal(n) * 10 + 60,
        "biomarker": rng.standard_normal(n),
    })
    times = np.clip(np.exp(rng.standard_normal(n)), 0.1, 10)
    events = rng.binomial(1, 0.7, n).astype(bool)
    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[("event", bool), ("time", float)],
    ))
    learner = SurvivalLearner(X, y)
    learner.setup(verbose=False)
    return learner


def _fake_model_result(name, is_tuned=False):
    return ModelResult(
        name=name, model=None,
        cv_metric_scores=np.array([0.7]),
        cv_metric_mean=0.7, cv_metric_std=0.0,
        is_tuned=is_tuned,
    )


def _seed_importance_dicts(learner, name="coxph"):
    """Attach one sentinel table per method to *name*'s ModelResult.

    Creates the ModelResult if absent so the per-model views surface it.
    Returns the two seeded DataFrames.
    """
    pimp_df = pd.DataFrame({"feature": ["age"], "p_value": [0.01]})
    pimp_df.attrs["model_name"] = name
    boot_df = pd.DataFrame({"feature": ["age"], "p_value": [0.02]})
    boot_df.attrs["model_name"] = name
    result = learner._state.models.get(name) or _fake_model_result(name)
    result.pimp_importance_test = pimp_df
    result.bootstrap_importance_test = boot_df
    learner._state.models[name] = result
    return pimp_df, boot_df


class TestInstallTunedResultInvalidation:
    """_install_tuned_result drops importance tests with the overwritten model."""

    def test_overwrite_drops_entries(self, setup_learner):
        _seed_importance_dicts(setup_learner)

        tuned = _fake_model_result("coxph", is_tuned=True)
        setup_learner._install_tuned_result("coxph", tuned)

        assert dict(setup_learner._state.pimp_importance_test_results) == {}
        assert dict(setup_learner._state.bootstrap_importance_test_results) == {}

    def test_overwrite_leaves_other_models(self, setup_learner):
        _seed_importance_dicts(setup_learner, "coxph")
        other_pimp, _ = _seed_importance_dicts(setup_learner, "rsf")

        tuned = _fake_model_result("coxph", is_tuned=True)
        setup_learner._install_tuned_result("coxph", tuned)

        # Overwriting coxph leaves rsf's tables untouched.
        assert (setup_learner._state.pimp_importance_test_results["rsf"]
                is other_pimp)


class TestResetupInvalidation:
    """Re-running setup() clears stale per-model result state.

    Covers both importance-test tables and the recalibration fit:
    clearing ``_state.models`` drops everything riding on each
    ModelResult in one step.
    """

    def test_resetup_clears_both_dicts(self, setup_learner):
        from mlsurv._warnings import SuggestionWarning

        _seed_importance_dicts(setup_learner)
        with pytest.warns(SuggestionWarning, match="already been called"):
            setup_learner.setup(verbose=False)
        assert dict(setup_learner._state.pimp_importance_test_results) == {}
        assert dict(setup_learner._state.bootstrap_importance_test_results) == {}

    def test_resetup_clears_calibrator_state(self, setup_learner):
        """Re-setup drops the calibrator fit on the old split.

        Without this, a stale calibrator survives setup() and is silently
        applied to the next model trained under the same name. The calibrator
        rides on the ModelResult, so clearing ``_state.models`` drops it.
        """
        from mlsurv._warnings import SuggestionWarning

        result = _fake_model_result("coxph")
        result.calibrator = object()  # sentinel calibrator
        setup_learner._state.models["coxph"] = result
        with pytest.warns(SuggestionWarning, match="already been called"):
            setup_learner.setup(verbose=False)
        assert not setup_learner._has_calibrator("coxph")
        assert dict(setup_learner._state.models) == {}


class TestWarnOverwriteHelper:
    """_warn_overwrite emits one uniform message, no state inspection."""

    def test_uniform_overwrite_message(self, setup_learner):
        from mlsurv._warnings import SuggestionWarning

        with pytest.warns(
            SuggestionWarning,
            match="all results associated with it will be overwritten",
        ):
            setup_learner._warn_overwrite("coxph")

    def test_message_does_not_vary_with_state(self, setup_learner):
        """The message is the same whether or not derived state is present."""
        from mlsurv._warnings import SuggestionWarning

        _seed_importance_dicts(setup_learner, "coxph")
        with pytest.warns(SuggestionWarning) as rec:
            setup_learner._warn_overwrite("coxph")
        msgs = [str(w.message) for w in rec]
        assert any(
            "all results associated with it will be overwritten" in m
            for m in msgs
        )
        # No method-specific "re-run recalibrate()" tail anymore.
        assert all("re-run" not in m for m in msgs)


@pytest.mark.integration
class TestRetrainInvalidation:
    """Re-training a model drops its stale importance-test entries."""

    def test_retrain_drops_entries(self, small_data):
        from mlsurv._warnings import SuggestionWarning

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, random_state=42, verbose=False)
        learner.train("coxph", verbose=False)

        _seed_importance_dicts(learner, "coxph")
        other_pimp, _ = _seed_importance_dicts(learner, "other")

        with pytest.warns(
            SuggestionWarning,
            match="all results associated with it will be overwritten",
        ):
            learner.train("coxph", verbose=False)

        # Stale entries for the re-trained model are gone from both views;
        # other models' entries survive.
        assert "coxph" not in learner._state.pimp_importance_test_results
        assert "coxph" not in learner._state.bootstrap_importance_test_results
        assert learner._state.pimp_importance_test_results["other"] is other_pimp


@pytest.mark.integration
@pytest.mark.integration_heavy
class TestRunTuneInvalidation:
    """run(tune=True) over an existing model drops its stale entries.

    The successful-tuning path builds its result with ``_store=False`` and
    stores it in the run() loop — outside ``_finalize_model_result`` /
    ``_install_tuned_result`` — so the loop carries its own overwrite warning.
    """

    def test_run_tune_drops_entries(self, small_data):
        from mlsurv._warnings import SuggestionWarning

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=3, random_state=42,
                      n_jobs=1, verbose=False)
        learner.train("coxnet", verbose=False)

        _seed_importance_dicts(learner, "coxnet")

        # The run loop's store guard must warn before replacing the model
        # with the uniform overwrite message.
        with pytest.warns(
            SuggestionWarning,
            match="all results associated with it will be overwritten",
        ):
            learner.run(
                models=["coxnet"], n_trials=2, bootstrap=False,
                n_importance_repeats=0, show_eval_progress=False,
                verbose=False, show_progress_bar=False,
            )

        # Confirm the real tuning path ran (not the skip/fallback paths,
        # which store through the already-guarded _finalize_model_result).
        tuned = learner._lookup_model_by_name("coxnet")
        assert tuned.is_tuned and tuned.best_params_

        assert "coxnet" not in learner._state.pimp_importance_test_results
        assert "coxnet" not in learner._state.bootstrap_importance_test_results
