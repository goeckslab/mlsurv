"""Tests for tuning custom BaseSurvivalModel instances.

Custom models (BaseSurvivalModel subclasses passed as instances) can be tuned
either via a model-declared ``get_search_space()`` hook or a per-call
``search_spaces=`` argument. These tests cover both routes, the merge
precedence (per-call wins), the empty-space guard, and the downstream
evaluate/bootstrap path on a tuned custom model.
"""

import numpy as np
import pytest
from sksurv.linear_model import CoxPHSurvivalAnalysis

from mlsurv import SurvivalLearner
from mlsurv.models.base import BaseSurvivalModel


class _RegularizedCox(BaseSurvivalModel):
    """Ridge-penalized Cox wrapper that declares a search space on ``alpha``."""

    name = "reg_cox"
    display_name = "Regularized Cox"
    color = "#17becf"
    supports_random_state = False

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, X, y):
        self.model_ = CoxPHSurvivalAnalysis(alpha=self.alpha)
        self.model_.fit(X, y)
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.to_numpy()
        return self

    def predict(self, X):
        self._check_is_fitted()
        return self.model_.predict(X)

    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **params):
        if "alpha" in params:
            self.alpha = params["alpha"]
        return self

    def get_search_space(self):
        return {"alpha": (0.001, 1.0, "log")}


class _UntunableCox(_RegularizedCox):
    """Same model but declares no search space (default empty dict)."""

    name = "untunable_cox"

    def get_search_space(self):
        return {}


class _WideCox(_RegularizedCox):
    """Declares a wide alpha range, used to test per-call override precedence."""

    name = "wide_cox"

    def get_search_space(self):
        return {"alpha": (0.5, 1.0)}


class _KParamCox(_RegularizedCox):
    """Custom model that declares its OWN ``k`` hyperparameter.

    Previously, a bare ``k`` in a search space was unconditionally stolen by
    the feature selector. This model proves a model's ``k`` now reaches the
    model, while the selector's ``k`` is tuned only via ``feature_selector__k``.
    """

    name = "kparam_cox"

    def __init__(self, alpha=0.1, k=1):
        self.alpha = alpha
        self.k = k  # stored, unused by fit — purely to verify routing

    def get_params(self, deep=True):
        return {"alpha": self.alpha, "k": self.k}

    def set_params(self, **params):
        if "alpha" in params:
            self.alpha = params["alpha"]
        if "k" in params:
            self.k = params["k"]
        return self

    def get_search_space(self):
        return {"k": [2, 3, 4]}


class _SeededCox(_RegularizedCox):
    """Stochastic custom model used to test seed propagation.

    Exposes ``random_state`` (default ``None``) and uses it to add tiny seeded
    noise to the features before fitting, so its fit — and thus its predictions
    — depend on the seed. ``fit()`` records every ``random_state`` it sees in
    the class-level ``seeds_seen`` list, letting a test assert the learner seed
    reaches every tuning trial *and* the final rebuild.
    """

    name = "seeded_cox"
    supports_random_state = True

    # Class-level: collects the random_state of every fit during tuning (one
    # per CV fold per trial) plus the final rebuild. Tests clear it before
    # tuning. Safe because the seed tests run with n_jobs=1 (no cross-process
    # workers that wouldn't share the list).
    seeds_seen = []

    def __init__(self, alpha=0.1, random_state=None):
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        type(self).seeds_seen.append(self.random_state)
        rng = np.random.RandomState(self.random_state)
        X_arr = np.asarray(X, dtype=float)
        X_noisy = X_arr + rng.normal(0.0, 1e-6, size=X_arr.shape)
        self.model_ = CoxPHSurvivalAnalysis(alpha=self.alpha)
        self.model_.fit(X_noisy, y)
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.to_numpy()
        return self

    def predict(self, X):
        # fit() trains on a plain ndarray, so predict on one too — keeps sksurv
        # from warning about a feature-name mismatch.
        self._check_is_fitted()
        return self.model_.predict(np.asarray(X, dtype=float))

    def get_params(self, deep=True):
        return {"alpha": self.alpha, "random_state": self.random_state}

    def set_params(self, **params):
        if "alpha" in params:
            self.alpha = params["alpha"]
        if "random_state" in params:
            self.random_state = params["random_state"]
        return self


def _setup_learner(small_data):
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=42, n_jobs=1)
    return learner


def _best_alpha(result):
    """Tuned alpha regardless of component prefix."""
    return result.best_params_.get("model__alpha", result.best_params_.get("alpha"))


@pytest.mark.integration
def test_tune_custom_model_declared_space(small_data):
    """A custom model with get_search_space() can be tuned via train(tune=True)."""
    learner = _setup_learner(small_data)
    result = learner.train(_RegularizedCox(), tune=True, n_trials=4, verbose=False)

    assert result.is_tuned is True
    assert result.best_params_, "best_params_ should be populated"
    alpha = _best_alpha(result)
    assert alpha is not None
    assert 0.001 <= alpha <= 1.0, f"tuned alpha {alpha} outside declared range"


@pytest.mark.integration
def test_tune_custom_model_per_call_space_only(small_data):
    """A model with no declared space is tunable via per-call search_spaces."""
    learner = _setup_learner(small_data)
    result = learner.train(
        _UntunableCox(), tune=True, n_trials=4, verbose=False,
        search_spaces={"model": {"alpha": (0.01, 0.5)}},
    )

    assert result.is_tuned is True
    alpha = _best_alpha(result)
    assert alpha is not None
    assert 0.01 <= alpha <= 0.5, f"tuned alpha {alpha} outside per-call range"


@pytest.mark.integration
def test_tune_custom_model_per_call_overrides_declared(small_data):
    """When both are present, the per-call search_spaces dict wins."""
    learner = _setup_learner(small_data)
    # Declared range (0.5, 1.0) is disjoint from the per-call range (0.001, 0.01),
    # so the tuned alpha proves which space was actually used.
    result = learner.train(
        _WideCox(), tune=True, n_trials=6, verbose=False,
        search_spaces={"model": {"alpha": (0.001, 0.01)}},
    )

    alpha = _best_alpha(result)
    assert alpha is not None
    assert 0.001 <= alpha <= 0.01, (
        f"per-call space should win, but tuned alpha={alpha} is in the "
        f"declared range (0.5, 1.0)"
    )


@pytest.mark.integration
def test_tune_custom_model_empty_space_raises(small_data):
    """No declared space and no per-call space → clear, actionable ValueError."""
    learner = _setup_learner(small_data)
    with pytest.raises(ValueError) as exc:
        learner.train(_UntunableCox(), tune=True, n_trials=3, verbose=False)

    msg = str(exc.value)
    # The message should point at all remedies: declared hook, per-call model
    # space, and per-call component space.
    assert "get_search_space()" in msg
    assert "search_spaces" in msg
    assert "component" in msg


@pytest.mark.integration
def test_tuned_custom_model_evaluate_and_bootstrap(small_data):
    """evaluate() and bootstrap() work on a tuned custom model."""
    learner = _setup_learner(small_data)
    result = learner.train(_RegularizedCox(), tune=True, n_trials=4, verbose=False)

    learner.evaluate(result, n_importance_repeats=1, show_progress=False)
    learner.bootstrap(result, n_bootstrap=2)

    assert np.isfinite(result.evaluation.c_index_test)
    assert result.bootstrap is not None


@pytest.mark.integration
def test_tune_method_accepts_custom_instance(small_data):
    """learner.tune(custom_instance) tunes the instance directly."""
    learner = _setup_learner(small_data)
    result = learner.tune(_RegularizedCox(), n_trials=4, verbose=False)

    assert result.is_tuned is True
    alpha = _best_alpha(result)
    assert alpha is not None and 0.001 <= alpha <= 1.0


@pytest.mark.integration
def test_tuned_alpha_reaches_fitted_model(small_data):
    """The tuned value is applied to the final fitted estimator, not just recorded.

    best_params_ is a recorded dict; the model is rebuilt separately via
    clone().set_params(). This guards the gap between the two.
    """
    learner = _setup_learner(small_data)
    result = learner.train(_RegularizedCox(), tune=True, n_trials=4, verbose=False)

    tuned_alpha = _best_alpha(result)
    fitted = result.model.named_steps["model"]
    assert fitted.alpha == tuned_alpha, "tuned alpha not applied to wrapper"
    # And it propagated into the inner sksurv estimator that was actually fit.
    assert fitted.model_.alpha == tuned_alpha


@pytest.mark.integration
def test_tuned_custom_model_overwrites_in_place(small_data):
    """Tuning a custom model overwrites its untuned entry at the plain key."""
    learner = _setup_learner(small_data)
    baseline = learner.train(_RegularizedCox(), verbose=False)
    assert baseline.name == "reg_cox"

    # Tuned overwrites the untuned model in place, no '_tuned' key.
    tuned = learner.tune(_RegularizedCox(), n_trials=4, verbose=False)
    assert tuned.name == "reg_cox"
    assert tuned.is_tuned is True
    assert "reg_cox_tuned" not in learner.models
    assert set(learner.models) == {"reg_cox"}
    assert learner.models["reg_cox"].is_tuned is True


@pytest.mark.integration
def test_tune_custom_model_with_component_space(small_data):
    """A custom model can tune preprocessing components alongside its own params."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=42, n_jobs=1,
                  feature_selector="topk")
    result = learner.train(
        _RegularizedCox(), tune=True, n_trials=5, verbose=False,
        search_spaces={"model": {"alpha": (0.01, 1.0)}, "feature_selector": {"k": (2, 4)}},
    )

    assert result.is_tuned is True
    tuned_k = (
        result.best_params_.get("feature_selector__k")
        or result.best_params_.get("k")
    )
    assert tuned_k is not None and 2 <= tuned_k <= 4


@pytest.mark.integration
def test_tune_custom_model_component_space_only(small_data):
    """Empty model space + a component space tunes the selector, model fixed.

    A custom model with no tunable hyperparameters (empty get_search_space() and
    no per-call model space) is still tunable on the preprocessing pipeline alone,
    matching built-ins like CoxPH. Previously this raised a ValueError.
    """
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=42, n_jobs=1,
                  feature_selector="topk")
    result = learner.train(
        _UntunableCox(), tune=True, n_trials=4, verbose=False,
        search_spaces={"feature_selector": {"k": (2, 4)}},
    )

    assert result.is_tuned is True
    tuned_k = (
        result.best_params_.get("feature_selector__k")
        or result.best_params_.get("k")
    )
    assert tuned_k is not None and 2 <= tuned_k <= 4
    # No model param was tuned: the model stays at its default alpha.
    assert "model__alpha" not in result.best_params_
    assert "alpha" not in result.best_params_


@pytest.mark.integration
def test_tune_custom_model_scaler_space_only(small_data):
    """Component-only tuning works for the scaler axis too, not just the selector.

    has_searchable_params gates feature_selector / imputer / scaler; this locks
    a second axis so a future narrowing of that tuple would be caught.
    """
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=42, n_jobs=1,
                  scaler="standard")
    result = learner.train(
        _UntunableCox(), tune=True, n_trials=4, verbose=False,
        search_spaces={"scaler": {"with_mean": [True, False]}},
    )

    assert result.is_tuned is True
    assert "scaler__with_mean" in result.best_params_
    assert "model__alpha" not in result.best_params_
    assert "alpha" not in result.best_params_


@pytest.mark.integration
def test_tune_custom_model_categorical_space(small_data):
    """Categorical search-space entries work for custom models."""

    class _CategoricalCox(_RegularizedCox):
        name = "cat_cox"
        def get_search_space(self):
            return {"alpha": [0.01, 0.1, 1.0]}

    learner = _setup_learner(small_data)
    result = learner.train(_CategoricalCox(), tune=True, n_trials=4, verbose=False)

    alpha = _best_alpha(result)
    assert alpha in (0.01, 0.1, 1.0), f"categorical alpha {alpha} not from declared set"


@pytest.mark.integration
def test_tune_custom_model_reproducible(small_data):
    """Same random_state → same tuned params (n_jobs=1, fixed seed)."""
    r1 = _setup_learner(small_data).train(
        _RegularizedCox(), tune=True, n_trials=5, verbose=False)
    r2 = _setup_learner(small_data).train(
        _RegularizedCox(), tune=True, n_trials=5, verbose=False)
    assert _best_alpha(r1) == _best_alpha(r2)


@pytest.mark.integration
def test_custom_model_tuning_injects_learner_seed(small_data):
    """The learner seed reaches every tuning trial AND the final rebuild.

    The custom model instance defaults to random_state=None; clone() preserves
    that, so without the fix the trial/final fits would run unseeded. Both the
    objective() injection and the final-rebuild injection should stamp the
    learner's configured seed (42) onto every fit.
    """
    _SeededCox.seeds_seen.clear()
    learner = _setup_learner(small_data)  # random_state=42, n_jobs=1
    learner.tune(_SeededCox(), n_trials=4, verbose=False)

    assert _SeededCox.seeds_seen, "model was never fit — test setup is broken"
    assert set(_SeededCox.seeds_seen) == {42}, (
        f"every trial/final fit should see the learner seed 42, got "
        f"{sorted(set(_SeededCox.seeds_seen), key=lambda v: (v is None, v))}"
    )


@pytest.mark.integration
def test_custom_model_tuning_learner_seed_overrides_instance_seed(small_data):
    """The learner seed overrides a seed the user baked into the instance.

    The fix injects the learner seed unconditionally (matching built-ins like
    rsf.py:27), so it must *replace* any instance seed — not merely fill in a
    missing one. With the model constructed as random_state=7 but the learner
    seeded 42, every fit should see 42. This is the test that distinguishes
    "inject the learner seed" from a weaker "only seed when None" variant.
    """
    _SeededCox.seeds_seen.clear()
    learner = _setup_learner(small_data)  # random_state=42, n_jobs=1
    result = learner.tune(_SeededCox(random_state=7), n_trials=4, verbose=False)

    assert set(_SeededCox.seeds_seen) == {42}, (
        f"learner seed 42 should override the instance seed 7, got "
        f"{sorted(set(_SeededCox.seeds_seen), key=lambda v: (v is None, v))}"
    )
    assert result.model.named_steps["model"].random_state == 42


@pytest.mark.integration
def test_custom_model_tuning_seed_reproducible(small_data):
    """A stochastic custom model tunes reproducibly end-to-end.

    Same learner seed → identical tuned params AND identical predictions from
    the stored final model. The prediction check is what proves the
    final-rebuild injection: without it the served pipeline is refit unseeded
    and predictions drift run-to-run.
    """
    r1 = _setup_learner(small_data).tune(_SeededCox(), n_trials=5, verbose=False)
    r2 = _setup_learner(small_data).tune(_SeededCox(), n_trials=5, verbose=False)

    assert _best_alpha(r1) == _best_alpha(r2)
    X_test = _setup_learner(small_data).X_test
    np.testing.assert_array_equal(r1.model.predict(X_test), r2.model.predict(X_test))


@pytest.mark.integration
def test_custom_model_without_random_state_still_tunes(small_data):
    """The get_params() guard skips models that have no random_state.

    _RegularizedCox exposes only 'alpha' (no random_state) — the injection must
    not attempt set_params(random_state=...) on it, which its set_params would
    silently ignore (and a strict set_params would reject).
    """
    learner = _setup_learner(small_data)
    result = learner.tune(_RegularizedCox(), n_trials=3, verbose=False)

    assert result.is_tuned is True
    assert "random_state" not in result.model.named_steps["model"].get_params()


@pytest.mark.integration
def test_custom_model_set_params_ignoring_seed_warns(small_data):
    """A model that lists random_state but drops it on write warns once.

    The seed injection is silent if a custom set_params accepts random_state in
    get_params() yet never writes it — tuning then looks seeded but isn't. The
    final-rebuild read-back catches this and emits a SuggestionWarning so the
    silent reproducibility failure surfaces to the user.
    """
    import warnings as w
    from mlsurv._warnings import SuggestionWarning

    class _LyingSeedCox(_SeededCox):
        """Exposes random_state via get_params() but ignores it in set_params."""

        name = "lying_seed_cox"

        def set_params(self, **params):
            if "alpha" in params:
                self.alpha = params["alpha"]
            # Deliberately drops random_state on write.
            return self

    learner = _setup_learner(small_data)
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        learner.tune(_LyingSeedCox(), n_trials=3, verbose=False)

    msgs = [str(c.message) for c in caught
            if issubclass(c.category, SuggestionWarning)
            and "did not apply the learner seed" in str(c.message)]
    assert msgs, "expected a read-back warning when set_params ignores the seed"


@pytest.mark.integration
def test_custom_model_unknown_param_warns(small_data, recwarn):
    """A declared param absent from get_params() warns about the no-op axis."""
    import warnings as w
    from mlsurv._warnings import SuggestionWarning

    class _BadParamCox(_RegularizedCox):
        name = "bad_param_cox"
        def get_search_space(self):
            # 'beta' is never exposed by get_params() (only 'alpha' is).
            return {"beta": (0.1, 1.0)}

    learner = _setup_learner(small_data)
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        learner.train(_BadParamCox(), tune=True, n_trials=2, verbose=False)

    msgs = [str(c.message) for c in caught
            if issubclass(c.category, SuggestionWarning) and "no-op" in str(c.message)]
    assert msgs, "expected a no-op-axis warning for the unknown 'beta' param"


@pytest.mark.integration
def test_builtin_tuning_path_unchanged(small_data):
    """Built-in string-name tuning still works (guards against regression)."""
    learner = _setup_learner(small_data)
    result = learner.tune("coxnet", n_trials=2, verbose=False)

    assert result.is_tuned is True
    assert result.best_params_ is not None
    assert result.name == "coxnet"


@pytest.mark.integration
def test_custom_model_k_is_model_param(small_data):
    """A custom model's own 'k' tunes the model, not the selector."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=42, n_jobs=1,
                  feature_selector="topk")
    default_k = learner._state.feature_selector.k

    result = learner.train(_KParamCox(), tune=True, n_trials=3, verbose=False)

    # The tuned k reached the MODEL wrapper from its declared {'k': [2, 3, 4]}.
    fitted_model = result.model.named_steps["model"]
    assert fitted_model.k in (2, 3, 4)
    # It was recorded as the model's own param, NOT rerouted to the selector.
    assert "feature_selector__k" not in result.best_params_
    # ...and the selector kept its default k (the model's k did not leak).
    assert result.model.named_steps["selector"].k == default_k


@pytest.mark.integration
def test_custom_model_k_and_selector_k_coexist(small_data):
    """Model 'k' and selector 'k' tune independently, no collision."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=42, n_jobs=1,
                  feature_selector="topk")
    result = learner.train(
        _KParamCox(), tune=True, n_trials=4, verbose=False,
        search_spaces={"model": {"k": [2, 3]}, "feature_selector": {"k": [2, 4]}},
    )

    # Selector k routed via feature_selector__k and applied to the selector.
    sel_k = result.best_params_.get("feature_selector__k")
    assert sel_k in (2, 4)
    assert result.model.named_steps["selector"].k == sel_k
    # Model k routed to the model wrapper, from its own (per-call) space.
    assert result.model.named_steps["model"].k in (2, 3)
