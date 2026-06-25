"""Tests for CoxLassoSelector and CoxNetSelector."""

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from mlsurv.preprocessing import (
    CoxLassoSelector,
    CoxNetSelector,
    get_selector,
)
from mlsurv._warnings import (
    FitFailureWarning,
    SuggestionWarning,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def survival_data():
    """Synthetic survival data with some predictive features."""
    np.random.seed(42)
    n_samples, n_features = 200, 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Make first 5 features predictive
    true_coef = np.zeros(n_features)
    true_coef[:5] = [0.5, -0.3, 0.4, -0.2, 0.6]
    linear_pred = X.values @ true_coef
    times = np.exp(-linear_pred + np.random.randn(n_samples) * 0.5)
    times = np.clip(times, 0.1, 10)
    events = np.random.binomial(1, 0.7, n_samples).astype(bool)

    y = np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


@pytest.fixture
def train_test_split(survival_data):
    """70/30 train/test split."""
    X, y = survival_data
    n = int(len(X) * 0.7)
    return X.iloc[:n], X.iloc[n:], y[:n], y[n:]


# ---------------------------------------------------------------------------
# CoxLassoSelector
# ---------------------------------------------------------------------------

class TestCoxLassoSelector:

    def test_fit_transform_basic(self, survival_data):
        X, y = survival_data
        selector = CoxLassoSelector(alpha="min")
        X_sel = selector.fit_transform(X, y)
        assert X_sel.shape[0] == X.shape[0]
        assert X_sel.shape[1] <= X.shape[1]
        assert X_sel.shape[1] == selector.n_selected_

    def test_selected_features_set(self, survival_data):
        X, y = survival_data
        selector = CoxLassoSelector(alpha="min")
        selector.fit(X, y)
        assert hasattr(selector, "selected_features_")
        assert hasattr(selector, "feature_names_in_")
        assert hasattr(selector, "n_features_in_")
        assert hasattr(selector, "coefficients_")
        assert hasattr(selector, "selected_alpha_")
        assert hasattr(selector, "n_selected_")
        assert selector.n_features_in_ == X.shape[1]

    def test_l1_ratio_is_one(self, survival_data):
        X, y = survival_data
        selector = CoxLassoSelector()
        assert selector._l1_ratio == 1.0

    def test_alpha_auto(self, survival_data):
        X, y = survival_data
        selector = CoxLassoSelector(alpha="auto", cv_folds=3)
        selector.fit(X, y)
        assert selector.selected_alpha_ is not None
        assert isinstance(selector.selected_alpha_, (float, np.floating))
        assert selector.n_selected_ > 0

    def test_alpha_min(self, survival_data):
        X, y = survival_data
        selector = CoxLassoSelector(alpha="min")
        selector.fit(X, y)
        assert selector.n_selected_ > 0

    def test_alpha_float(self, survival_data):
        X, y = survival_data
        selector = CoxLassoSelector(alpha=0.01)
        selector.fit(X, y)
        assert selector.selected_alpha_ == 0.01

    def test_invalid_alpha_raises(self, survival_data):
        X, y = survival_data
        selector = CoxLassoSelector(alpha="invalid")
        with pytest.raises(ValueError, match="alpha must be"):
            selector.fit(X, y)

    def test_get_support_boolean(self, survival_data):
        X, y = survival_data
        selector = CoxLassoSelector(alpha="min")
        selector.fit(X, y)
        mask = selector.get_support(indices=False)
        assert mask.dtype == bool
        assert len(mask) == X.shape[1]
        assert mask.sum() == selector.n_selected_

    def test_get_support_indices(self, survival_data):
        X, y = survival_data
        selector = CoxLassoSelector(alpha="min")
        selector.fit(X, y)
        indices = selector.get_support(indices=True)
        assert len(indices) == selector.n_selected_
        assert all(0 <= i < X.shape[1] for i in indices)

    def test_get_feature_names_out(self, survival_data):
        X, y = survival_data
        selector = CoxLassoSelector(alpha="min")
        selector.fit(X, y)
        names = selector.get_feature_names_out()
        assert names == selector.selected_features_

    def test_coefficients_series(self, survival_data):
        X, y = survival_data
        selector = CoxLassoSelector(alpha="min")
        selector.fit(X, y)
        assert isinstance(selector.coefficients_, pd.Series)
        assert len(selector.coefficients_) == X.shape[1]
        # Non-zero coefficients should match selected features
        nonzero = selector.coefficients_[selector.coefficients_ != 0].index.tolist()
        assert set(nonzero) == set(selector.selected_features_)

    def test_repr(self, survival_data):
        X, y = survival_data
        selector = CoxLassoSelector()
        r = repr(selector)
        assert "CoxLassoSelector" in r
        selector.fit(X, y)
        r = repr(selector)
        assert "n_selected=" in r


# ---------------------------------------------------------------------------
# CoxNetSelector
# ---------------------------------------------------------------------------

class TestCoxNetSelector:

    def test_fit_transform_basic(self, survival_data):
        X, y = survival_data
        selector = CoxNetSelector(l1_ratio=0.8, alpha="min")
        X_sel = selector.fit_transform(X, y)
        assert X_sel.shape[0] == X.shape[0]
        assert X_sel.shape[1] <= X.shape[1]

    def test_l1_ratio_stored(self, survival_data):
        X, y = survival_data
        selector = CoxNetSelector(l1_ratio=0.7)
        assert selector.l1_ratio == 0.7
        assert selector._l1_ratio == 0.7

    def test_l1_ratio_affects_sparsity(self, survival_data):
        """Higher l1_ratio should generally produce sparser solutions."""
        X, y = survival_data
        sel_high = CoxNetSelector(l1_ratio=1.0, alpha="min")
        sel_low = CoxNetSelector(l1_ratio=0.1, alpha="min")
        sel_high.fit(X, y)
        sel_low.fit(X, y)
        # Low l1_ratio (more ridge) should retain at least as many features
        assert sel_low.n_selected_ >= sel_high.n_selected_

    def test_alpha_auto(self, survival_data):
        X, y = survival_data
        selector = CoxNetSelector(l1_ratio=0.5, alpha="auto", cv_folds=3)
        selector.fit(X, y)
        assert selector.n_selected_ > 0

    def test_repr_includes_l1_ratio(self):
        selector = CoxNetSelector(l1_ratio=0.3)
        r = repr(selector)
        assert "l1_ratio=0.3" in r


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class TestPresets:

    def test_coxlasso_preset(self):
        selector = get_selector("coxlasso")
        assert isinstance(selector, CoxLassoSelector)

    def test_coxnet_preset(self):
        selector = get_selector("coxnet")
        assert isinstance(selector, CoxNetSelector)

    def test_preset_fresh_instances(self):
        s1 = get_selector("coxlasso")
        s2 = get_selector("coxlasso")
        assert s1 is not s2


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:

    def test_in_sklearn_pipeline(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", CoxLassoSelector(alpha="min")),
        ])
        X_sel_train = pipe.fit_transform(X_train, y_train)
        X_sel_test = pipe.transform(X_test)
        assert X_sel_train.shape[1] == X_sel_test.shape[1]
        assert X_sel_train.shape[1] > 0

    def test_coxnet_in_pipeline(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", CoxNetSelector(l1_ratio=0.5, alpha="min")),
        ])
        X_sel_train = pipe.fit_transform(X_train, y_train)
        X_sel_test = pipe.transform(X_test)
        assert X_sel_train.shape[1] == X_sel_test.shape[1]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_feature(self):
        np.random.seed(99)
        X = pd.DataFrame({"only_feat": np.random.randn(50)})
        y = np.array(
            [(True, t) for t in np.random.exponential(2, 50)],
            dtype=[("event", bool), ("time", float)],
        )
        selector = CoxLassoSelector(alpha="min")
        selector.fit(X, y)
        assert selector.n_selected_ == 1

    def test_cv_folds_too_few_warns(self, survival_data):
        X, y = survival_data
        selector = CoxLassoSelector(alpha="auto", cv_folds=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            selector.fit(X, y)
            fit_warnings = [x for x in w if issubclass(x.category, FitFailureWarning)]
            assert len(fit_warnings) >= 1

    def test_few_samples_warns(self):
        np.random.seed(7)
        X = pd.DataFrame(np.random.randn(3, 5), columns=[f"f{i}" for i in range(5)])
        y = np.array(
            [(True, 1.0), (False, 2.0), (True, 3.0)],
            dtype=[("event", bool), ("time", float)],
        )
        selector = CoxLassoSelector(alpha="auto", cv_folds=5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            selector.fit(X, y)
            fit_warnings = [x for x in w if issubclass(x.category, FitFailureWarning)]
            assert len(fit_warnings) >= 1

    def test_numpy_input(self, survival_data):
        X, y = survival_data
        selector = CoxLassoSelector(alpha="min")
        X_sel = selector.fit_transform(X.values, y)
        assert X_sel.shape[1] > 0

    def test_verbose_output(self, survival_data, capsys):
        X, y = survival_data
        selector = CoxLassoSelector(alpha="min", verbose=True)
        selector.fit(X, y)
        captured = capsys.readouterr()
        assert "Selected" in captured.out

    def test_unfitted_transform_raises(self, survival_data):
        X, y = survival_data
        selector = CoxLassoSelector()
        with pytest.raises(NotFittedError):
            selector.transform(X)

    def test_unfitted_get_support_raises(self):
        selector = CoxLassoSelector()
        with pytest.raises(NotFittedError):
            selector.get_support()

    def test_all_zero_fallback_warns(self, survival_data):
        """Over-regularization triggers fallback to alpha='min'."""
        X, y = survival_data
        selector = CoxLassoSelector(alpha=1e6)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            selector.fit(X, y)
            suggestion_warnings = [
                x for x in w if issubclass(x.category, SuggestionWarning)
            ]
            assert len(suggestion_warnings) >= 1
        assert selector.n_selected_ > 0

    def test_all_zero_at_min_alpha_fallback(self, survival_data):
        """When even alpha='min' gives all zeros, all features are kept."""
        X, y = survival_data
        selector = CoxLassoSelector(alpha="min")

        # Mock _fit_and_extract_coef to always return zeros
        with patch.object(
            type(selector), "_fit_and_extract_coef",
            return_value=np.zeros(X.shape[1]),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                selector.fit(X, y)
                fit_warnings = [
                    x for x in w if issubclass(x.category, FitFailureWarning)
                ]
                assert len(fit_warnings) >= 1

        assert selector.n_selected_ == X.shape[1]

    def test_sklearn_clone(self, survival_data):
        X, y = survival_data
        selector = CoxLassoSelector(alpha="min", n_alphas=50)
        cloned = clone(selector)
        assert cloned.alpha == "min"
        assert cloned.n_alphas == 50
        cloned.fit(X, y)
        assert cloned.n_selected_ > 0

    def test_sklearn_get_set_params(self):
        selector = CoxLassoSelector(alpha="min", cv_folds=3)
        params = selector.get_params()
        assert params["alpha"] == "min"
        assert params["cv_folds"] == 3
        selector.set_params(cv_folds=10)
        assert selector.cv_folds == 10


# ---------------------------------------------------------------------------
# CoxNetSelector — additional tests
# ---------------------------------------------------------------------------

class TestCoxNetSelectorValidation:

    @pytest.mark.parametrize("bad_ratio", [0.0, -0.5, 1.5])
    def test_l1_ratio_out_of_range_raises(self, bad_ratio):
        with pytest.raises(ValueError, match="l1_ratio must be in"):
            CoxNetSelector(l1_ratio=bad_ratio)

    def test_set_params_syncs_l1_ratio(self, survival_data):
        """set_params(l1_ratio=...) is picked up by fit()."""
        X, y = survival_data
        selector = CoxNetSelector(l1_ratio=0.5, alpha="min")
        selector.set_params(l1_ratio=0.9)
        selector.fit(X, y)
        assert selector._l1_ratio == 0.9

    def test_sklearn_clone_coxnet(self, survival_data):
        X, y = survival_data
        selector = CoxNetSelector(l1_ratio=0.7, alpha="min")
        cloned = clone(selector)
        assert cloned.l1_ratio == 0.7
        assert cloned._l1_ratio == 0.7
        cloned.fit(X, y)
        assert cloned.n_selected_ > 0


# ---------------------------------------------------------------------------
# random_state threading
# ---------------------------------------------------------------------------

class TestSelectorRandomState:
    """The configured seed reaches the alpha-selection K-fold CV.

    Comparisons across seeds are deterministic here: the fixture data is
    generated under a fixed numpy seed, so the only varying input is the
    selector's own ``random_state``. Seeds 1 and 5 were chosen because they
    land on different alphas for this fixture (verified empirically).
    """

    def test_default_random_state_is_none(self):
        assert CoxLassoSelector().random_state is None
        assert CoxNetSelector().random_state is None

    def test_coxlasso_stores_random_state(self):
        selector = CoxLassoSelector(random_state=7)
        assert selector.random_state == 7
        assert selector.get_params()["random_state"] == 7

    def test_coxnet_stores_random_state(self):
        selector = CoxNetSelector(random_state=7)
        assert selector.random_state == 7
        assert selector.get_params()["random_state"] == 7

    def test_clone_preserves_random_state(self):
        cloned = clone(CoxNetSelector(l1_ratio=0.5, random_state=7))
        assert cloned.random_state == 7

    def test_same_seed_reproducible(self, survival_data):
        X, y = survival_data
        a = CoxNetSelector(l1_ratio=0.5, alpha="auto", cv_folds=3, random_state=7)
        b = CoxNetSelector(l1_ratio=0.5, alpha="auto", cv_folds=3, random_state=7)
        a.fit(X, y)
        b.fit(X, y)
        assert a.selected_alpha_ == b.selected_alpha_
        assert a.selected_features_ == b.selected_features_

    def test_different_seed_changes_coxnet_selection(self, survival_data):
        X, y = survival_data
        s1 = CoxNetSelector(l1_ratio=0.5, alpha="auto", cv_folds=3, random_state=1)
        s5 = CoxNetSelector(l1_ratio=0.5, alpha="auto", cv_folds=3, random_state=5)
        s1.fit(X, y)
        s5.fit(X, y)
        assert s1.selected_alpha_ != s5.selected_alpha_

    def test_different_seed_changes_coxlasso_selection(self, survival_data):
        X, y = survival_data
        s1 = CoxLassoSelector(alpha="auto", cv_folds=3, random_state=1)
        s5 = CoxLassoSelector(alpha="auto", cv_folds=3, random_state=5)
        s1.fit(X, y)
        s5.fit(X, y)
        assert s1.selected_alpha_ != s5.selected_alpha_

    def test_seed_irrelevant_without_cv(self, survival_data):
        """alpha='min' skips the CV, so the seed cannot affect selection."""
        X, y = survival_data
        s1 = CoxNetSelector(l1_ratio=0.5, alpha="min", random_state=1)
        s5 = CoxNetSelector(l1_ratio=0.5, alpha="min", random_state=5)
        s1.fit(X, y)
        s5.fit(X, y)
        assert s1.selected_alpha_ == s5.selected_alpha_
        assert s1.selected_features_ == s5.selected_features_


class TestPresetsRandomState:
    """get_selector threads the seed into the penalized-Cox string presets."""

    def test_coxnet_preset_threads_seed(self):
        assert get_selector("coxnet", random_state=7).random_state == 7

    def test_coxlasso_preset_threads_seed(self):
        assert get_selector("coxlasso", random_state=7).random_state == 7

    def test_preset_default_random_state_none(self):
        assert get_selector("coxnet").random_state is None
        assert get_selector("coxlasso").random_state is None

    def test_seed_threaded_through_list(self):
        pipe = get_selector(["coxnet"], random_state=7)
        assert isinstance(pipe, Pipeline)
        assert pipe[-1].random_state == 7

    def test_deterministic_presets_ignore_seed(self):
        # topk/vif/correlation take no random_state; passing one must not raise.
        for name in ("topk", "vif", "correlation"):
            get_selector(name, random_state=7)


class TestCoxNetModelRandomState:
    """CoxNetModel opts into seed injection and threads it to its alpha CV.

    The model and the penalized selectors now share one fold-scoring helper
    (``mlsurv._shared.select_alpha_by_cv_concordance``), so these behavioral
    fits guard the model's seed path through that helper directly rather than
    relying on the selector tests.
    """

    def test_supports_random_state_flag(self):
        from mlsurv.models.coxnet import CoxNetModel
        assert CoxNetModel.supports_random_state is True

    def test_in_models_with_random_state(self):
        from mlsurv.learner._core import MODELS_WITH_RANDOM_STATE
        assert "coxnet" in MODELS_WITH_RANDOM_STATE

    def test_stores_random_state(self):
        from mlsurv.models.coxnet import CoxNetModel
        assert CoxNetModel(random_state=7).random_state == 7
        assert CoxNetModel().random_state is None

    def test_clone_preserves_random_state(self):
        from mlsurv.models.coxnet import CoxNetModel
        assert clone(CoxNetModel(random_state=7)).random_state == 7

    def test_same_seed_reproducible(self, survival_data):
        from mlsurv.models.coxnet import CoxNetModel
        X, y = survival_data
        a = CoxNetModel(l1_ratio=0.5, alpha="auto", cv_folds=3,
                        fit_baseline_model=False, random_state=7)
        b = CoxNetModel(l1_ratio=0.5, alpha="auto", cv_folds=3,
                        fit_baseline_model=False, random_state=7)
        a.fit(X, y)
        b.fit(X, y)
        assert a.selected_alpha_ == b.selected_alpha_

    def test_different_seed_changes_alpha(self, survival_data):
        from mlsurv.models.coxnet import CoxNetModel
        X, y = survival_data
        m1 = CoxNetModel(l1_ratio=0.5, alpha="auto", cv_folds=3,
                         fit_baseline_model=False, random_state=1)
        m5 = CoxNetModel(l1_ratio=0.5, alpha="auto", cv_folds=3,
                         fit_baseline_model=False, random_state=5)
        m1.fit(X, y)
        m5.fit(X, y)
        assert m1.selected_alpha_ != m5.selected_alpha_

    def test_seed_irrelevant_without_cv(self, survival_data):
        """alpha='min' skips the CV, so the seed cannot affect selection."""
        from mlsurv.models.coxnet import CoxNetModel
        X, y = survival_data
        m1 = CoxNetModel(l1_ratio=0.5, alpha="min",
                         fit_baseline_model=False, random_state=1)
        m5 = CoxNetModel(l1_ratio=0.5, alpha="min",
                         fit_baseline_model=False, random_state=5)
        m1.fit(X, y)
        m5.fit(X, y)
        assert m1.selected_alpha_ == m5.selected_alpha_

    @pytest.mark.integration
    def test_learner_injects_seed_into_trained_coxnet(self, survival_data):
        """setup(random_state=...) reaches the trained CoxNetModel via injection."""
        from mlsurv import SurvivalLearner
        X, y = survival_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=7, feature_selector=None)
        result = learner.train("coxnet", tune=False)
        assert result.model.named_steps["model"].random_state == 7


class TestLearnerSelectorSeed:
    """setup() passes the learner seed to the resolved selectors."""

    def test_setup_threads_seed_to_selector(self, survival_data):
        from mlsurv import SurvivalLearner
        X, y = survival_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector="coxnet", random_state=7)
        assert learner._state.feature_selector.random_state == 7

    def test_setup_threads_seed_to_per_model_selector(self, survival_data):
        from mlsurv import SurvivalLearner
        X, y = survival_data
        learner = SurvivalLearner(X, y)
        learner.setup(
            random_state=7,
            preprocessing={"coxnet": {"feature_selector": "coxlasso"}},
        )
        sel = learner._state.model_preprocessing["coxnet"]["feature_selector"]
        assert sel.random_state == 7

    def test_setup_threads_seed_to_imputer_and_scaler(self, survival_data):
        """setup() threads its seed into the iterative imputer + pca PCA."""
        from mlsurv import SurvivalLearner
        X, y = survival_data
        learner = SurvivalLearner(X, y)
        learner.setup(imputer="iterative", scaler="pca", random_state=99)
        assert learner._state.imputer.random_state == 99
        assert dict(learner._state.scaler.steps)["pca"].random_state == 99

    def test_setup_threads_seed_to_per_model_imputer_scaler(self, survival_data):
        """Per-model preprocessing threads the seed into imputer/scaler too."""
        from mlsurv import SurvivalLearner
        X, y = survival_data
        learner = SurvivalLearner(X, y)
        learner.setup(
            random_state=11,
            preprocessing={"coxnet": {"imputer": "iterative", "scaler": "pca"}},
        )
        cfg = learner._state.model_preprocessing["coxnet"]
        assert cfg["imputer"].random_state == 11
        assert dict(cfg["scaler"].steps)["pca"].random_state == 11


class TestPerModelPreprocessingSeed:
    """_resolve_per_model_preprocessing threads the seed.

    This is the shared resolver that setup(), train(), and run() all call with
    the learner seed, so a direct test guards the train()/run() call sites too.
    """

    def test_resolver_threads_seed_to_selector(self):
        from mlsurv.learner._setup import _resolve_per_model_preprocessing
        resolved = _resolve_per_model_preprocessing(
            {"coxph": {"feature_selector": "coxnet"}}, random_state=7
        )
        assert resolved["coxph"]["feature_selector"].random_state == 7

    def test_resolver_default_seed_none(self):
        from mlsurv.learner._setup import _resolve_per_model_preprocessing
        resolved = _resolve_per_model_preprocessing(
            {"coxph": {"feature_selector": "coxlasso"}}
        )
        assert resolved["coxph"]["feature_selector"].random_state is None

    def test_resolver_threads_seed_to_imputer_and_scaler(self):
        """The imputer/scaler resolvers thread the seed too, not just selector."""
        from mlsurv.learner._setup import _resolve_per_model_preprocessing
        resolved = _resolve_per_model_preprocessing(
            {"rsf": {"imputer": "iterative", "scaler": "pca"}}, random_state=11
        )
        assert resolved["rsf"]["imputer"].random_state == 11
        assert dict(resolved["rsf"]["scaler"].steps)["pca"].random_state == 11
