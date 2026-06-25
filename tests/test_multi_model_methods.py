"""predict()/importance()/rmst_test()/logrank_test() accept multiple models.

These four methods resolve their ``models=`` selection through the shared
arity-aware resolver: a name (or None with exactly one trained model) keeps
the single-model result shape, while a list/dict (or None with several
trained models) covers every selected model — predict() returns a dict keyed
by model name, the other three return one stacked DataFrame with
``model_type`` as the first column. Per-model failures in a multi-model
selection skip the model with a warning instead of aborting the others.
"""

import copy
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import MethodologicalWarning, SuggestionWarning
from mlsurv.utils import SurvivalArray


def _make_data(n=60, seed=42):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        'age': rng.uniform(30, 80, n),
        'stage': rng.randint(1, 5, n).astype(float),
        'grade': rng.uniform(1, 4, n),
        'tumor_size': rng.uniform(0.5, 10, n),
        'biomarker': rng.randn(n),
    })
    t = rng.exponential(2, n).clip(0.1, 10)
    e = rng.binomial(1, 0.6, n).astype(bool)
    y = SurvivalArray(np.array(
        [(a, b) for a, b in zip(e, t)],
        dtype=[('event', bool), ('time', float)],
    ))
    return X, y


@pytest.fixture(scope="module")
def _evaluated_template():
    """Two cheap models trained + evaluated — reused via deepcopy per test."""
    X, y = _make_data()
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None, n_jobs=1)
    learner.run(["coxph", "coxnet"], tune=False, verbose=False)
    learner.evaluate(
        n_importance_repeats=1, verbose=False, show_progress=False,
    )
    return learner


@pytest.fixture
def trained(_evaluated_template):
    """Per-test deep copy — the methods under test mutate stored results."""
    return copy.deepcopy(_evaluated_template)


@pytest.fixture(scope="module")
def _single_model_template():
    """Exactly one trained model — for the None→bare-result contract."""
    X, y = _make_data()
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None, n_jobs=1)
    learner.run(["coxph"], tune=False, verbose=False)
    learner.evaluate(
        n_importance_repeats=1, verbose=False, show_progress=False,
    )
    return learner


@pytest.fixture
def single_trained(_single_model_template):
    return copy.deepcopy(_single_model_template)


def _x_new(learner):
    return learner.X_test.iloc[:3]


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPredictMultiModel:
    def test_str_returns_bare_result(self, trained):
        pred = trained.predict(_x_new(trained), models="coxph", verbose=False)
        assert type(pred).__name__ == "PredictionResult"
        assert pred.model_name == "coxph"
        assert len(pred.risk_scores) == 3

    def test_none_with_one_model_returns_bare_result(self, single_trained):
        pred = single_trained.predict(_x_new(single_trained), verbose=False)
        assert type(pred).__name__ == "PredictionResult"
        assert pred.model_name == "coxph"

    def test_none_with_multiple_models_returns_dict_of_all(self, trained):
        preds = trained.predict(_x_new(trained), verbose=False)
        assert isinstance(preds, dict)
        assert set(preds) == {"coxph", "coxnet"}
        for name, pred in preds.items():
            assert pred.model_name == name

    def test_list_returns_dict_even_for_one_model(self, trained):
        preds = trained.predict(
            _x_new(trained), models=["coxph"], verbose=False,
        )
        assert isinstance(preds, dict)
        assert list(preds) == ["coxph"]

    def test_dict_input_returns_dict(self, trained):
        preds = trained.predict(
            _x_new(trained), models=trained.models, verbose=False,
        )
        assert isinstance(preds, dict)
        assert set(preds) == {"coxph", "coxnet"}

    def test_multi_result_is_cv_sorted(self, trained):
        preds = trained.predict(_x_new(trained), verbose=False)
        expected = trained._cv_sorted_keys(list(trained.models))
        assert list(preds) == expected

    def test_duplicate_selection_dedupes(self, trained):
        preds = trained.predict(
            _x_new(trained),
            models=["coxph", "coxph", trained.models["coxph"]],
            verbose=False,
        )
        assert list(preds) == ["coxph"]

    def test_empty_dict_raises(self, trained):
        with pytest.raises(ValueError, match="Empty model dict"):
            trained.predict(_x_new(trained), models={}, verbose=False)

    def test_multi_values_match_single_model_calls(self, trained):
        X_new = _x_new(trained)
        preds = trained.predict(X_new, verbose=False)
        for name in ("coxph", "coxnet"):
            single = trained.predict(X_new, models=name, verbose=False)
            np.testing.assert_allclose(
                preds[name].risk_scores, single.risk_scores,
            )

    def test_per_model_failure_skips_with_warning(self, trained, monkeypatch):
        def _boom(X):
            raise RuntimeError("synthetic prediction failure")

        monkeypatch.setattr(
            trained.models["coxnet"].model, "predict", _boom, raising=False,
        )
        with pytest.warns(MethodologicalWarning,
                          match="Prediction failed for model 'coxnet'"):
            preds = trained.predict(_x_new(trained), verbose=False)
        assert set(preds) == {"coxph"}

    def test_all_models_failing_raises(self, trained, monkeypatch):
        def _boom(X):
            raise RuntimeError("synthetic prediction failure")

        for name in ("coxph", "coxnet"):
            monkeypatch.setattr(
                trained.models[name].model, "predict", _boom, raising=False,
            )
        with pytest.warns(MethodologicalWarning):
            with pytest.raises(RuntimeError, match="failed for all models"):
                trained.predict(_x_new(trained), verbose=False)

    def test_raw_pipeline_raises_typeerror(self, trained):
        raw = trained.models["coxph"].model
        with pytest.raises(TypeError, match="Raw models/pipelines"):
            trained.predict(_x_new(trained), models=raw, verbose=False)

    def test_empty_list_raises(self, trained):
        with pytest.raises(ValueError, match="Empty model list"):
            trained.predict(_x_new(trained), models=[], verbose=False)

    def test_singular_model_keyword_rejected(self, trained):
        with pytest.raises(TypeError, match="unexpected keyword"):
            trained.predict(_x_new(trained), model="coxph", verbose=False)


# ---------------------------------------------------------------------------
# importance()
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestImportanceMultiModel:
    def test_str_single_model_schema(self, trained):
        out = trained.importance(
            "coxph", n_importance_repeats=1, show_progress=False,
        )
        assert list(out.columns)[0] == "model_type"
        assert set(out["model_type"]) == {"coxph"}

    def test_none_stacks_all_models(self, trained):
        out = trained.importance(n_importance_repeats=1, show_progress=False)
        assert list(out.columns)[0] == "model_type"
        assert set(out["model_type"]) == {"coxph", "coxnet"}
        # One block of rows (one row per feature) per model.
        n_features = trained.X_train.shape[1]
        assert (out.groupby("model_type").size() == n_features).all()

    def test_list_selects_subset(self, trained):
        out = trained.importance(
            ["coxnet"], n_importance_repeats=1, show_progress=False,
        )
        assert set(out["model_type"]) == {"coxnet"}

    def test_none_with_one_model_matches_str_form(self, single_trained):
        out_none = single_trained.importance(
            n_importance_repeats=1, random_state=7, show_progress=False,
        )
        out_str = single_trained.importance(
            "coxph", n_importance_repeats=1, random_state=7,
            show_progress=False,
        )
        pd.testing.assert_frame_equal(out_none, out_str)

    def test_multi_skips_unevaluated_model_with_warning(self, trained):
        trained.models["coxnet"].evaluation = None
        with pytest.warns(SuggestionWarning, match="no evaluation result"):
            out = trained.importance(
                n_importance_repeats=1, show_progress=False,
            )
        assert set(out["model_type"]) == {"coxph"}

    def test_single_unevaluated_model_still_raises(self, trained):
        trained.models["coxnet"].evaluation = None
        with pytest.raises(ValueError, match="no evaluation result"):
            trained.importance(
                "coxnet", n_importance_repeats=1, show_progress=False,
            )

    def test_kept_prior_without_model_type_gets_labelled(
        self, trained, monkeypatch,
    ):
        """A preserved prior table predating the model_type column is
        labelled in the stack rather than contributing NaN model_type rows."""
        import mlsurv.evaluation.importance as imp_mod

        prior = trained.models["coxnet"].evaluation.importance
        if "model_type" in prior.columns:
            trained.models["coxnet"].evaluation.importance = (
                prior.drop(columns=["model_type"])
            )

        original = imp_mod._compute_permutation_importance

        def _fail_for_coxnet(*args, **kwargs):
            if kwargs.get("model_name") == "coxnet":
                empty = pd.DataFrame()
                return empty, empty, None
            return original(*args, **kwargs)

        monkeypatch.setattr(
            imp_mod, "_compute_permutation_importance", _fail_for_coxnet,
        )
        with pytest.warns(SuggestionWarning, match="kept"):
            out = trained.importance(
                n_importance_repeats=1, show_progress=False,
            )
        assert set(out["model_type"]) == {"coxph", "coxnet"}
        assert out["model_type"].notna().all()

    def test_recompute_failure_with_no_prior_returns_empty_schema(
        self, trained, monkeypatch,
    ):
        """A failed recompute with no prior importance to keep returns an
        empty 0-row table with the documented schema, never None."""
        import mlsurv.evaluation.importance as imp_mod

        trained.models["coxph"].evaluation.importance = None

        def _always_fail(*args, **kwargs):
            empty = pd.DataFrame()
            return empty, empty, None

        monkeypatch.setattr(
            imp_mod, "_compute_permutation_importance", _always_fail,
        )
        with pytest.warns(SuggestionWarning, match="computed no"):
            out = trained.importance(
                "coxph", n_importance_repeats=1, show_progress=False,
            )
        assert isinstance(out, pd.DataFrame)
        assert out.empty
        assert list(out.columns) == [
            "model_type", "feature", "coefficient", "perm_mean", "perm_std",
        ]

    def test_recompute_failure_with_empty_prior_returns_empty_schema(
        self, trained, monkeypatch,
    ):
        """An empty (0-row) prior — e.g. from evaluate with
        n_importance_repeats=0 — counts as "no prior to keep": the failure
        path returns the schema'd sentinel, not the malformed empty prior."""
        import mlsurv.evaluation.importance as imp_mod

        trained.models["coxph"].evaluation.importance = pd.DataFrame(
            columns=["feature", "coefficient", "perm_mean", "perm_std",
                     "model_type"],
        )

        def _always_fail(*args, **kwargs):
            empty = pd.DataFrame()
            return empty, empty, None

        monkeypatch.setattr(
            imp_mod, "_compute_permutation_importance", _always_fail,
        )
        with pytest.warns(SuggestionWarning, match="no prior importance"):
            out = trained.importance(
                "coxph", n_importance_repeats=1, show_progress=False,
            )
        assert out.empty
        assert list(out.columns) == [
            "model_type", "feature", "coefficient", "perm_mean", "perm_std",
        ]

    def test_singular_model_keyword_rejected(self, trained):
        with pytest.raises(TypeError, match="unexpected keyword"):
            trained.importance(model="coxph", show_progress=False)


@pytest.mark.integration
@pytest.mark.shap
def test_importance_shap_stacks_all_models(trained):
    pytest.importorskip("shap")
    out = trained.importance(
        method="shap", max_explain=5, max_background=5, show_progress=False,
    )
    assert list(out.columns) == ["model_type", "feature", "mean_abs_shap"]
    assert set(out["model_type"]) == {"coxph", "coxnet"}
    # Sorted by mean_abs_shap descending within each model block.
    for _, block in out.groupby("model_type"):
        vals = block["mean_abs_shap"].to_numpy()
        assert (np.diff(vals) <= 1e-12).all()


# ---------------------------------------------------------------------------
# rmst_test() / logrank_test()
# ---------------------------------------------------------------------------

def _force_constant_risk(learner, name):
    """Make one model's test-set risk scores constant so group formation fails."""
    test_set = learner.models[name].evaluation.test_set
    test_set.loc[:, "risk_score"] = 1.0


@pytest.mark.integration
class TestRiskGroupTestsMultiModel:
    def test_rmst_str_has_model_type_first(self, trained):
        out = trained.rmst_test("coxph", verbose=False)
        assert list(out.columns)[0] == "model_type"
        assert set(out["model_type"]) == {"coxph"}
        # Single-form returns the stored table itself.
        assert trained.models["coxph"].rmst_test is out

    def test_rmst_none_stacks_all_models(self, trained):
        out = trained.rmst_test(verbose=False)
        assert list(out.columns)[0] == "model_type"
        assert set(out["model_type"]) == {"coxph", "coxnet"}
        # Per-model storage keeps one table per model, stamped and typed.
        for name in ("coxph", "coxnet"):
            stored = trained.models[name].rmst_test
            assert stored.attrs["model_name"] == name
            assert set(stored["model_type"]) == {name}

    def test_logrank_str_single_model(self, trained):
        out = trained.logrank_test("coxnet", verbose=False)
        assert list(out.columns)[0] == "model_type"
        assert set(out["model_type"]) == {"coxnet"}
        assert (out["comparison"] == "omnibus").any()

    def test_logrank_none_stacks_all_models(self, trained):
        out = trained.logrank_test(verbose=False)
        assert set(out["model_type"]) == {"coxph", "coxnet"}
        # One omnibus row per model.
        omnibus = out[out["comparison"] == "omnibus"]
        assert len(omnibus) == 2

    def test_multi_skips_failed_model_with_warning(self, trained):
        _force_constant_risk(trained, "coxnet")
        with pytest.warns(MethodologicalWarning,
                          match="skipped model 'coxnet'"):
            out = trained.rmst_test(verbose=False)
        assert set(out["model_type"]) == {"coxph"}
        # The failed model stores an empty table (absent-vs-empty preserved).
        stored = trained.models["coxnet"].rmst_test
        assert stored.empty
        assert list(stored.columns)[0] == "model_type"
        assert stored.attrs["model_name"] == "coxnet"

    def test_single_form_failure_still_raises(self, trained):
        _force_constant_risk(trained, "coxnet")
        with pytest.raises(ValueError, match="unique"):
            trained.rmst_test("coxnet", verbose=False)

    def test_logrank_multi_skips_failed_model(self, trained):
        _force_constant_risk(trained, "coxph")
        with pytest.warns(MethodologicalWarning,
                          match="skipped model 'coxph'"):
            out = trained.logrank_test(verbose=False)
        assert set(out["model_type"]) == {"coxnet"}

    def test_requires_evaluate_first(self):
        X, y = _make_data()
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, feature_selector=None, n_jobs=1)
        # train() (unlike run()) does not auto-evaluate.
        learner.train("coxph", verbose=False)
        with pytest.raises(ValueError, match="evaluate"):
            learner.rmst_test(verbose=False)
        with pytest.raises(ValueError, match="evaluate"):
            learner.logrank_test(verbose=False)

    def test_singular_model_keyword_rejected(self, trained):
        with pytest.raises(TypeError, match="unexpected keyword"):
            trained.rmst_test(model="coxph", verbose=False)
        with pytest.raises(TypeError, match="unexpected keyword"):
            trained.logrank_test(model="coxph", verbose=False)


# ---------------------------------------------------------------------------
# Stale ModelResult references (passed object is not the registered one)
# ---------------------------------------------------------------------------

_STALE_FRAGMENT = "not the model currently registered"


def _stale_warnings(records):
    """Filter a catch_warnings record list to the stale-reference warnings."""
    return [
        w for w in records
        if issubclass(w.category, SuggestionWarning)
        and _STALE_FRAGMENT in str(w.message)
    ]


@pytest.mark.integration
class TestStaleModelResultWarning:
    """A passed ModelResult that is not the registered object warns once
    per stale object; the registered model is still the one computed on."""

    def test_predict_direct_stale_modelresult_warns(self, trained):
        stale = copy.deepcopy(trained.models["coxph"])
        with pytest.warns(SuggestionWarning, match=_STALE_FRAGMENT):
            pred = trained.predict(_x_new(trained), models=stale,
                                   verbose=False)
        # Single-form contract kept, and computed on the registered model.
        assert type(pred).__name__ == "PredictionResult"
        assert pred.model_name == "coxph"
        registered = trained.predict(_x_new(trained), models="coxph",
                                     verbose=False)
        np.testing.assert_allclose(pred.risk_scores, registered.risk_scores)

    def test_importance_list_with_stale_element_warns(self, trained):
        stale = copy.deepcopy(trained.models["coxph"])
        before_stale = stale.evaluation.importance
        before_registered = trained.models["coxph"].evaluation.importance
        with pytest.warns(SuggestionWarning, match=_STALE_FRAGMENT):
            out = trained.importance(
                [stale], n_importance_repeats=1, show_progress=False,
            )
        assert set(out["model_type"]) == {"coxph"}
        # The recompute attached to the registered object, not the stale one.
        assert stale.evaluation.importance is before_stale
        assert (trained.models["coxph"].evaluation.importance
                is not before_registered)

    def test_rmst_dict_with_stale_value_warns_per_object(self, trained):
        stale = copy.deepcopy(trained.models["coxph"])
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            out = trained.rmst_test(
                models={"coxph": stale, "coxnet": trained.models["coxnet"]},
                verbose=False,
            )
        emitted = _stale_warnings(records)
        assert len(emitted) == 1
        assert "'coxph'" in str(emitted[0].message)
        assert "'coxnet'" not in str(emitted[0].message)
        assert set(out["model_type"]) == {"coxph", "coxnet"}

    def test_logrank_direct_stale_warns(self, trained):
        stale = copy.deepcopy(trained.models["coxnet"])
        with pytest.warns(SuggestionWarning, match=_STALE_FRAGMENT):
            out = trained.logrank_test(models=stale, verbose=False)
        assert set(out["model_type"]) == {"coxnet"}

    def test_registered_objects_stay_silent(self, trained):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            trained.predict(_x_new(trained),
                            models=trained.models["coxph"], verbose=False)
            trained.predict(_x_new(trained),
                            models=[trained.models["coxph"]], verbose=False)
            trained.predict(_x_new(trained),
                            models=trained.models, verbose=False)
        assert _stale_warnings(records) == []

    def test_duplicate_stale_element_warns_once(self, trained):
        stale = copy.deepcopy(trained.models["coxph"])
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            preds = trained.predict(
                _x_new(trained), models=[stale, stale], verbose=False,
            )
        assert len(_stale_warnings(records)) == 1
        assert list(preds) == ["coxph"]

    def test_stale_after_registered_duplicate_still_warns(self, trained):
        # The staleness check must not be order-dependent: a stale object
        # following a clean occurrence of the same name still warns.
        stale = copy.deepcopy(trained.models["coxph"])
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            preds = trained.predict(
                _x_new(trained),
                models=[trained.models["coxph"], stale],
                verbose=False,
            )
        assert len(_stale_warnings(records)) == 1
        assert list(preds) == ["coxph"]

    def test_two_stale_objects_warn_once_each(self, trained):
        stale_a = copy.deepcopy(trained.models["coxph"])
        stale_b = copy.deepcopy(trained.models["coxnet"])
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            trained.predict(
                _x_new(trained), models=[stale_a, stale_b], verbose=False,
            )
        emitted = _stale_warnings(records)
        assert len(emitted) == 2
        named = {str(w.message).split("'")[1] for w in emitted}
        assert named == {"coxph", "coxnet"}

    def test_tune_overwrite_makes_held_reference_stale(self, trained):
        # coxnet (not coxph) — coxph has no hyperparameters to search, so
        # tune() would skip it without overwriting the registry entry.
        held = trained.models["coxnet"]
        # The registered object is silent before the overwrite.
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            trained.predict(_x_new(trained), models=held, verbose=False)
        assert _stale_warnings(records) == []

        # tune() overwrites the registry entry (its own overwrite warning
        # is expected and stays outside the assertion below).
        trained.tune(models="coxnet", n_trials=2, verbose=False)
        assert trained.models["coxnet"] is not held

        with pytest.warns(SuggestionWarning, match=_STALE_FRAGMENT):
            pred = trained.predict(_x_new(trained), models=held,
                                   verbose=False)
        # Predictions come from the registered (tuned) model.
        tuned = trained.predict(_x_new(trained), models="coxnet",
                                verbose=False)
        np.testing.assert_allclose(pred.risk_scores, tuned.risk_scores)
