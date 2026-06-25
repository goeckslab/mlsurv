"""evaluate()/bootstrap()/validate() accept a list of model names.

`_resolve_model_input` previously branched on dict/None/str/ModelResult but not
list/tuple, so ``learner.evaluate(["coxph", "coxnet"])`` fell through and the raw
list was treated as a single model, crashing with
``AttributeError: 'list' object has no attribute 'predict'``. A list/tuple is now
resolved to the existing ``{name: pipeline}`` multi-model form, consistent with
``run()``/``tune()``.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv.utils import SurvivalArray


@pytest.fixture(scope="module")
def _trained_template():
    """Learner with two cheap models trained — reused via deepcopy per test."""
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        'age': np.random.uniform(30, 80, n),
        'stage': np.random.randint(1, 5, n).astype(float),
        'grade': np.random.uniform(1, 4, n),
        'tumor_size': np.random.uniform(0.5, 10, n),
        'biomarker': np.random.randn(n),
    })
    t = np.random.exponential(2, n).clip(0.1, 10)
    e = np.random.binomial(1, 0.6, n).astype(bool)
    y = SurvivalArray(np.array(
        [(a, b) for a, b in zip(e, t)],
        dtype=[('event', bool), ('time', float)],
    ))
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None, n_jobs=1)
    learner.run(["coxph", "coxnet"], tune=False, verbose=False)
    return learner


@pytest.fixture
def trained(_trained_template):
    """Per-test deep copy — evaluate()/bootstrap() mutate the learner."""
    return copy.deepcopy(_trained_template)


def _cindex_by_model(eval_result):
    df = eval_result.concordance
    return dict(zip(df['model_type'], df['c_index_test']))


@pytest.mark.integration
def test_evaluate_list_matches_all(trained):
    """evaluate([...]) reproduces evaluate()-all on the same models."""
    all_eval = trained.evaluate(n_importance_repeats=1)
    list_eval = trained.evaluate(["coxph", "coxnet"], n_importance_repeats=1)

    all_c = _cindex_by_model(all_eval)
    list_c = _cindex_by_model(list_eval)

    assert set(list_c) == {"coxph", "coxnet"}
    assert set(list_c) == set(all_c)
    for name in list_c:
        assert list_c[name] == pytest.approx(all_c[name])


@pytest.mark.integration
def test_evaluate_tuple_form(trained):
    """A tuple resolves through the same multi-model path as a list."""
    res = trained.evaluate(("coxph",), n_importance_repeats=1)
    assert set(res.concordance['model_type']) == {"coxph"}


@pytest.mark.integration
def test_evaluate_modelresult_in_list(trained):
    """A list of ModelResult objects (not just names) resolves correctly."""
    res = trained.evaluate(
        [trained.models["coxph"], trained.models["coxnet"]],
        n_importance_repeats=1,
    )
    assert set(res.concordance['model_type']) == {"coxph", "coxnet"}


@pytest.mark.integration
def test_evaluate_duplicate_names_dedup(trained):
    """Duplicate names collapse to a single model (dict keys by name)."""
    res = trained.evaluate(["coxph", "coxph"], n_importance_repeats=1)
    assert list(res.concordance['model_type']) == ["coxph"]


@pytest.mark.integration
def test_evaluate_raw_model_in_list_raises_clear_error(trained):
    """A raw model/pipeline inside a list raises ValueError, not AttributeError."""
    raw = trained.models["coxph"].model
    with pytest.raises(ValueError, match="raw models/pipelines"):
        trained.evaluate([raw])


@pytest.mark.integration
def test_evaluate_unknown_name_in_list_raises(trained):
    """An unknown name inside a list surfaces the standard lookup error."""
    with pytest.raises(ValueError, match="not found"):
        trained.evaluate(["coxph", "nonexistent_model"])


@pytest.mark.integration
def test_evaluate_empty_list_raises(trained):
    """An empty list is a clear ValueError, not a silent empty evaluation."""
    with pytest.raises(ValueError, match="Empty model list"):
        trained.evaluate([])


@pytest.mark.integration
def test_bootstrap_list_stores_per_model(trained):
    """bootstrap([...]) runs and stores per-model results on each ModelResult."""
    trained.bootstrap(["coxph", "coxnet"], n_bootstrap=2)
    assert trained.models["coxph"].bootstrap is not None
    assert trained.models["coxnet"].bootstrap is not None


@pytest.mark.integration
def test_validate_list(trained):
    """validate([...]) covers the _resolve_eval_model_names list branch."""
    val = trained.validate(
        trained.X_test, trained.y_test,
        models=["coxph", "coxnet"], name="Holdout",
    )
    assert val is not None
    # Both requested models appear in the validation concordance.
    assert {"coxph", "coxnet"}.issubset(set(val.concordance['model_type']))


@pytest.mark.integration
def test_validate_raw_model_in_list_raises(trained):
    """A raw model in validate()'s list raises ValueError (resolver parity)."""
    raw = trained.models["coxph"].model
    with pytest.raises(ValueError, match="raw models/pipelines"):
        trained.validate(trained.X_test, trained.y_test, models=[raw])
