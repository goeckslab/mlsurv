"""Tests for mlsurv.tuning.optuna_tuner module.

Covers parallelism computation, search space parsing, storage resolution,
BaseTuner initialization, has_searchable_params, _suggest_value dispatch,
_evaluate_model exception guards, CoxNetTuner refactored score validation,
per-trial warning accumulation/deduplication, and
storage debug logging.
"""

import inspect
import os
import threading
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import SuggestionWarning
from mlsurv.models import _is_deep_learning_available
from mlsurv.tuning.optuna_tuner import (
    _compute_parallelism_split,
    _is_class_name,
    _parse_component_search_space,
    _get_step_class_mapping,
    get_storage_url,
    BaseTuner,
    CoxNetTuner,
)
from mlsurv.tuning.tuners import TUNER_REGISTRY

from tests._helpers import make_survival_array


@pytest.fixture
def tuner_data():
    """Minimal survival data for tuner construction."""
    np.random.seed(42)
    n = 50
    X = pd.DataFrame({'f1': np.random.randn(n), 'f2': np.random.randn(n)})
    y = make_survival_array(
        np.random.binomial(1, 0.6, n),
        np.random.exponential(2, n),
    )
    return X, y


# ---------------------------------------------------------------------------
# Declared-parameter coverage (regression guard)
# ---------------------------------------------------------------------------

class _RecordingTrial:
    """Fake Optuna trial: records every suggested param name and returns a
    valid value from each spec so model construction succeeds.

    Returning the lower bound / first choice guarantees a buildable model
    (e.g. n_layers=1, n_nodes=8 -> layers=[8]); we never fit, so the exact
    values don't matter — only which names get suggested.
    """

    def __init__(self):
        self.suggested = set()

    def suggest_int(self, name, low, high, log=False):
        self.suggested.add(name)
        return low

    def suggest_float(self, name, low, high, log=False):
        self.suggested.add(name)
        return low

    def suggest_categorical(self, name, choices):
        self.suggested.add(name)
        return choices[0]


class TestEveryDeclaredParamIsTuned:
    """Every parameter declared in a tuner's search space must be suggested.

    Regression guard. A hand-maintained fallback dict inside
    an Optuna tuner's ``objective()`` can drift out of sync with
    ``DEFAULT_SPACES`` and silently omit a declared parameter — that is exactly
    how ``xgbcox`` stopped tuning ``max_delta_step``. Iterating
    ``self.search_space`` makes the omission impossible; this test locks the
    invariant for every registered tuner so a future regression fails loudly.
    """

    # coxph is the only tuner with no hyperparameters; every other tuner must
    # declare a non-empty space. A tuner that silently lost its whole space
    # would skip tuning entirely — a worse failure than the per-param omission
    # this test guards against — so we assert against that too.
    _TUNERS_WITHOUT_HYPERPARAMS = {'coxph'}

    # Deep tuners construct pycox/torch models, so they only run when those
    # libraries are installed (see the visible skip below).
    _DEEP_TUNERS = {'deepsurv', 'deephit'}

    @pytest.mark.parametrize("name", list(TUNER_REGISTRY), ids=list(TUNER_REGISTRY))
    def test_all_declared_params_suggested(self, name, tuner_data):
        # Skip deep tuners *visibly* (reported as SKIPPED, not a silent
        # ``continue``) when pycox/torch is absent, so a no-pycox runner sees
        # the dropped coverage instead of a misleading green. When the
        # libraries are present, deep tuners run in the fast tier like the rest.
        if name in self._DEEP_TUNERS and not _is_deep_learning_available():
            pytest.skip(
                f"{name}: pycox/torch not installed — deep tuner "
                f"param-coverage check skipped"
            )

        X, y = tuner_data
        tuner = TUNER_REGISTRY[name](X, y, model_name=name)

        if name not in self._TUNERS_WITHOUT_HYPERPARAMS:
            assert tuner.search_space, (
                f"{name}: search_space is unexpectedly empty — the model "
                f"would silently skip all hyperparameter tuning"
            )

        trial = _RecordingTrial()
        # Patch evaluation so the trial builds model params + a model but
        # never fits — we only assert on which params were suggested.
        with patch.object(BaseTuner, '_evaluate_model', return_value=0.5):
            tuner.objective(trial)

        missing = set(tuner.search_space) - trial.suggested
        assert not missing, (
            f"{name}: parameters declared in the search space but never "
            f"suggested on the default tuning path: {sorted(missing)}"
        )

    def test_recording_trial_detects_missing_param(self):
        """Self-test: the guard must flag a declared param that is never suggested.

        Without this, a future change that broke `_RecordingTrial` (so it stopped
        recording) would make every tuner pass vacuously.
        """
        trial = _RecordingTrial()
        trial.suggest_int('kept', 1, 10)
        declared = {'kept', 'dropped'}
        assert declared - trial.suggested == {'dropped'}

    @pytest.mark.requires_pycox
    def test_malformed_deep_space_fails_loudly(self, tuner_data):
        """A deep custom search space missing n_nodes must fail loudly.

        deepsurv/deephit map ``n_layers`` + ``n_nodes`` -> ``layers`` only when
        BOTH keys are present. A custom space with only ``n_layers`` leaves an
        orphan ``n_layers`` kwarg; the model constructor (no ``**kwargs``) must
        reject it with a ``TypeError`` rather than silently dropping it. The
        TypeError is raised at construction, before any fit, so this stays fast.
        """
        from mlsurv.tuning.tuners import TUNER_REGISTRY

        X, y = tuner_data
        deepsurv_tuner = TUNER_REGISTRY['deepsurv'](
            X, y, model_name='deepsurv', search_spaces={'n_layers': (1, 2)}
        )
        trial = _RecordingTrial()
        with pytest.raises(TypeError, match='n_layers'):
            deepsurv_tuner.objective(trial)


# ---------------------------------------------------------------------------
# Parallelism configuration tests
# ---------------------------------------------------------------------------

class TestComputeParallelismSplit:
    """Tests for _compute_parallelism_split() core-allocation heuristic.

    Correct parallelism splitting affects tuning wall-clock time by 2-5x.
    Wrong splits silently degrade performance without errors.
    """

    def test_single_core(self):
        """n_jobs=1 should give (1, 1) for any model."""
        assert _compute_parallelism_split(1, 10, 500, 'rsf') == (1, 1)

    def test_non_rsf_model_gets_trial_only(self):
        """Non-RSF/XGB models (excluding thread-unsafe) get all cores for trial parallelism."""
        assert _compute_parallelism_split(8, 10, 500, 'gbsa') == (8, 1)

    def test_large_dataset_model_only(self):
        """Large datasets (>=2000 samples) should use model-only parallelism."""
        trial_j, model_j = _compute_parallelism_split(16, 10, 2000, 'rsf')
        assert trial_j == 1
        assert model_j == 16

    def test_medium_dataset_balanced(self):
        """Medium datasets should use balanced approach."""
        trial_j, model_j = _compute_parallelism_split(16, 10, 1000, 'rsf')
        assert trial_j > 1
        assert model_j > 1
        assert trial_j * model_j <= 16

    def test_many_trials_balanced(self):
        """Many trials (>=20) should shift toward trial parallelism."""
        trial_j, model_j = _compute_parallelism_split(16, 25, 500, 'rsf')
        assert trial_j >= 2
        assert model_j >= 1

    def test_minus_one_resolves_to_cpu_count(self):
        """n_jobs=-1 should resolve to actual CPU count."""
        trial_j, model_j = _compute_parallelism_split(-1, 10, 500, 'gbsa')
        cpu_count = os.cpu_count() or 1
        assert trial_j == cpu_count
        assert model_j == 1

    def test_xgbcox_supports_model_parallelism(self):
        """XGBCox should also get model parallelism for large data."""
        trial_j, model_j = _compute_parallelism_split(16, 10, 2000, 'xgbcox')
        assert trial_j == 1
        assert model_j == 16

    def test_thread_unsafe_model_forces_sequential(self):
        """Thread-unsafe models must always get (1, 1)."""
        for model_name in ['weibull_aft', 'coxnet']:
            for n_jobs in [1, 2, 4, 8, 16, -1]:
                trial_j, model_j = _compute_parallelism_split(
                    n_jobs, 25, 500, model_name,
                )
                assert (trial_j, model_j) == (1, 1), (
                    f"{model_name} n_jobs={n_jobs}: expected (1, 1), got ({trial_j}, {model_j})"
                )


# ---------------------------------------------------------------------------
# Search space parsing tests
# ---------------------------------------------------------------------------

class TestIsClassName:
    """Tests for _is_class_name() class name detection."""

    def test_class_name(self):
        assert _is_class_name('VIFSelector') is True

    def test_parameter_name(self):
        assert _is_class_name('threshold') is False

    def test_empty_string(self):
        assert _is_class_name('') is False

    def test_non_string(self):
        assert _is_class_name(42) is False


class TestParseComponentSearchSpace:
    """Tests for _parse_component_search_space() which splits broadcast
    vs class-specific parameters."""

    def test_broadcast_only(self):
        """Flat params should all be broadcast."""
        space = {'k': (5, 20), 'threshold': (0.5, 0.95)}
        class_spaces, broadcast = _parse_component_search_space(space)
        assert class_spaces == {}
        assert broadcast == {'k': (5, 20), 'threshold': (0.5, 0.95)}

    def test_class_specific_only(self):
        """Uppercase keys should be class-specific."""
        space = {'VIFSelector': {'threshold': (5, 15)}}
        class_spaces, broadcast = _parse_component_search_space(space)
        assert class_spaces == {'VIFSelector': {'threshold': (5, 15)}}
        assert broadcast == {}

    def test_mixed_space(self):
        """Mixed broadcast and class-specific params."""
        space = {
            'k': (5, 20),
            'VIFSelector': {'threshold': (5, 15)},
        }
        class_spaces, broadcast = _parse_component_search_space(space)
        assert 'VIFSelector' in class_spaces
        assert 'k' in broadcast

    def test_non_dict_class_value_becomes_broadcast_with_warning(self):
        """Uppercase key with non-dict value is treated as broadcast with warning."""
        space = {'VIFSelector': (5, 15)}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            class_spaces, broadcast = _parse_component_search_space(space)
        assert class_spaces == {}
        assert broadcast == {'VIFSelector': (5, 15)}
        assert len(caught) == 1
        assert "looks like a class name" in str(caught[0].message)
        assert "VIFSelector" in str(caught[0].message)


class TestGetStepClassMapping:
    """Tests for _get_step_class_mapping() with single and Pipeline components."""

    def test_none_component(self):
        """None component should return empty dict."""
        assert _get_step_class_mapping(None) == {}

    def test_single_transformer(self):
        """Single transformer returns class name mapped to [None]."""
        from sklearn.preprocessing import StandardScaler
        result = _get_step_class_mapping(StandardScaler())
        assert result == {'StandardScaler': [None]}

    def test_pipeline_component(self):
        """Pipeline returns class names mapped to step names."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        pipe = Pipeline([('scaler', StandardScaler()), ('imputer', SimpleImputer())])
        result = _get_step_class_mapping(pipe)
        assert 'StandardScaler' in result
        assert result['StandardScaler'] == ['scaler']
        assert 'SimpleImputer' in result
        assert result['SimpleImputer'] == ['imputer']


# ---------------------------------------------------------------------------
# Storage resolution tests
# ---------------------------------------------------------------------------

class TestGetStorageUrl:
    """Tests for get_storage_url() multi-source resolution."""

    def test_explicit_storage_wins(self):
        """Explicit parameter takes priority over everything."""
        result = get_storage_url('postgresql://localhost/db')
        assert result == 'postgresql://localhost/db'

    def test_env_var_fallback(self, monkeypatch):
        """Environment variable is used when no explicit param."""
        monkeypatch.setenv('MLSURV_OPTUNA_STORAGE', 'sqlite:///test.db')
        result = get_storage_url()
        assert result == 'sqlite:///test.db'

    def test_returns_none_when_no_storage(self, monkeypatch):
        """Returns None when no storage configured."""
        monkeypatch.delenv('MLSURV_OPTUNA_STORAGE', raising=False)
        result = get_storage_url()
        assert result is None


# ---------------------------------------------------------------------------
# BaseTuner initialization and search space parsing
# ---------------------------------------------------------------------------

class TestBaseTunerInit:
    """Tests for BaseTuner initialization and search space parsing."""

    def test_default_search_space(self, tuner_data):
        """Tuner with no search_spaces should have empty model space."""
        X, y = tuner_data
        tuner = BaseTuner(X, y)
        assert tuner._search_spaces == {'model': {}}

    def test_flat_search_space_backward_compat(self, tuner_data):
        """Flat dict should be treated as model search space."""
        X, y = tuner_data
        tuner = BaseTuner(X, y, search_spaces={'n_estimators': (100, 500)})
        assert 'n_estimators' in tuner._search_spaces['model']

    def test_structured_search_space(self, tuner_data):
        """Structured dict with component keys should be parsed correctly."""
        X, y = tuner_data
        spaces = {
            'model': {'n_estimators': (100, 500)},
            'feature_selector': {'k': (3, 10)},
        }
        tuner = BaseTuner(X, y, search_spaces=spaces)
        assert 'n_estimators' in tuner._search_spaces['model']
        assert 'k' in tuner._search_spaces['feature_selector']

    def test_search_space_alias(self, tuner_data):
        """Deprecated search_space kwarg should work as alias."""
        X, y = tuner_data
        tuner = BaseTuner(X, y, search_space={'lr': (0.01, 0.1)})
        assert 'lr' in tuner._search_spaces['model']

    def test_cv_splitter_override(self, tuner_data):
        """Providing cv_splitter should override default StratifiedKFold."""
        from sklearn.model_selection import KFold
        X, y = tuner_data
        custom_cv = KFold(n_splits=3)
        tuner = BaseTuner(X, y, cv_splitter=custom_cv)
        assert tuner._cv is custom_cv

    def test_objective_not_implemented(self, tuner_data):
        """BaseTuner.objective() should raise NotImplementedError."""
        X, y = tuner_data
        tuner = BaseTuner(X, y)
        with pytest.raises(NotImplementedError):
            tuner.objective(MagicMock())

    def test_rsf_tuner_small_epv_constrained(self):
        """RSFTuner with small-EPV data gets constrained search space."""
        from mlsurv.tuning.optuna_tuner import RSFTuner

        np.random.seed(42)
        n = 50
        X = pd.DataFrame({'f1': np.random.randn(n), 'f2': np.random.randn(n)})
        # 10 events / 2 features = EPV 5 (low)
        y = make_survival_array(
            [i < 10 for i in range(n)], [float(i + 1) for i in range(n)]
        )
        tuner = RSFTuner(X, y, model_name='rsf')
        assert tuner.search_space['bootstrap'] == [True]
        assert tuner.search_space['max_depth'] == (3, 8)
        assert tuner.search_space['min_samples_leaf'] == (5, 30)

    def test_user_search_spaces_override_epv(self):
        """User search_spaces override wins over EPV constraint."""
        from mlsurv.tuning.optuna_tuner import RSFTuner

        np.random.seed(42)
        n = 50
        X = pd.DataFrame({'f1': np.random.randn(n), 'f2': np.random.randn(n)})
        y = make_survival_array(
            [i < 10 for i in range(n)], [float(i + 1) for i in range(n)]
        )
        tuner = RSFTuner(
            X, y, model_name='rsf',
            search_spaces={'model': {'bootstrap': [True, False]}},
        )
        assert tuner.search_space['bootstrap'] == [True, False]


class TestBaseTunerSuggestValue:
    """Tests for BaseTuner._suggest_value() parameter dispatch.

    This method translates search space specs (tuples/lists) into
    Optuna suggest_* calls. Incorrect dispatch causes tuning to fail
    silently with wrong parameter types.
    """

    @pytest.fixture
    def tuner(self, tuner_data):
        """Minimal tuner for suggest_value testing."""
        X, y = tuner_data
        return BaseTuner(X, y)

    def test_categorical_list(self, tuner):
        """List spec should call suggest_categorical."""
        trial = MagicMock()
        trial.suggest_categorical.return_value = 'relu'
        result = tuner._suggest_value(trial, 'activation', ['relu', 'selu'])
        trial.suggest_categorical.assert_called_once_with('activation', ('relu', 'selu'))
        assert result == 'relu'

    def test_int_range_tuple(self, tuner):
        """Tuple of two ints should call suggest_int."""
        trial = MagicMock()
        trial.suggest_int.return_value = 200
        result = tuner._suggest_value(trial, 'n_estimators', (100, 500))
        trial.suggest_int.assert_called_once_with('n_estimators', 100, 500)
        assert result == 200

    def test_float_range_tuple(self, tuner):
        """Tuple with a float should call suggest_float."""
        trial = MagicMock()
        trial.suggest_float.return_value = 0.5
        result = tuner._suggest_value(trial, 'dropout', (0.1, 0.7))
        trial.suggest_float.assert_called_once_with('dropout', 0.1, 0.7)
        assert result == 0.5

    def test_log_scale_tuple(self, tuner):
        """Tuple ending in 'log' should call suggest_float with log=True."""
        trial = MagicMock()
        trial.suggest_float.return_value = 0.01
        result = tuner._suggest_value(trial, 'lr', (1e-4, 1e-1, 'log'))
        trial.suggest_float.assert_called_once_with('lr', 1e-4, 1e-1, log=True)
        assert result == 0.01

    def test_none_upper_bound_uses_n_features(self, tuner):
        """Tuple with None upper bound should use n_features."""
        trial = MagicMock()
        trial.suggest_int.return_value = 2
        tuner._suggest_value(trial, 'k', (1, None))
        # tuner.X has 2 columns
        trial.suggest_int.assert_called_once_with('k', 1, 2)

    def test_invalid_spec_raises(self, tuner):
        """Invalid spec format should raise ValueError."""
        trial = MagicMock()
        with pytest.raises(ValueError, match="Invalid search space format"):
            tuner._suggest_value(trial, 'bad', "not_a_spec")


# ---------------------------------------------------------------------------
# has_searchable_params tests
# ---------------------------------------------------------------------------

class TestHasSearchableParams:
    """Tests for BaseTuner.has_searchable_params property.

    This property gates whether tuning is skipped entirely — a false
    negative skips tuning for a model that has params to search, while
    a false positive silently runs N identical Optuna trials (wasteful).
    """

    def test_empty_search_space(self, tuner_data):
        """Tuner with no search space should return False."""
        X, y = tuner_data
        tuner = BaseTuner(X, y)
        assert tuner.has_searchable_params is False

    def test_model_search_space(self, tuner_data):
        """Tuner with model search space should return True."""
        X, y = tuner_data
        tuner = BaseTuner(X, y, search_spaces={'model': {'n_estimators': (100, 500)}})
        assert tuner.has_searchable_params is True

    def test_preprocessing_only_search_space(self, tuner_data):
        """Tuner with only preprocessing search space should return True."""
        X, y = tuner_data
        tuner = BaseTuner(X, y, search_spaces={'feature_selector': {'k': (3, 10)}})
        assert tuner.has_searchable_params is True

    def test_empty_model_with_preprocessing(self, tuner_data):
        """Empty model space but non-empty preprocessing should return True."""
        X, y = tuner_data
        tuner = BaseTuner(X, y, search_spaces={
            'model': {},
            'imputer': {'n_neighbors': (3, 10)},
        })
        assert tuner.has_searchable_params is True

    def test_coxph_tuner_no_components(self, tuner_data):
        """CoxPHTuner with no components should have no searchable params."""
        from mlsurv.tuning.optuna_tuner import CoxPHTuner
        X, y = tuner_data
        tuner = CoxPHTuner(X, y)
        assert tuner.has_searchable_params is False

    def test_subclass_with_default_space(self, tuner_data):
        """Tuner subclass with DEFAULT_SPACE should have searchable params."""
        X, y = tuner_data

        class MockTuner(BaseTuner):
            DEFAULT_SPACE = {'n_estimators': (100, 500)}
            def objective(self, trial):
                return 0.5

        tuner = MockTuner(X, y)
        assert tuner.has_searchable_params is True


# ---------------------------------------------------------------------------
# Unconsumed model-space guard (no silent data loss)
# ---------------------------------------------------------------------------

class TestUnconsumedModelSpaceWarning:
    """A model search-space key a tuner cannot apply must warn, not vanish."""

    def test_coxph_unsupported_model_space_warns_and_not_searchable(self, tuner_data):
        """CoxPH applies only ``alpha``: any other model key warns and does
        not count as searchable."""
        from mlsurv.tuning.optuna_tuner import CoxPHTuner
        X, y = tuner_data
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tuner = CoxPHTuner(
                X, y, search_spaces={'model': {'penalizer': (0.0, 1.0)}},
            )
        suggestion = [w for w in caught
                      if issubclass(w.category, SuggestionWarning)]
        assert len(suggestion) == 1
        assert 'penalizer' in str(suggestion[0].message)
        assert tuner.has_searchable_params is False

    def test_coxph_alpha_space_consumed_no_warning(self, tuner_data):
        """``alpha`` is a consumable CoxPH model key: it neither warns nor
        disables tuning."""
        from mlsurv.tuning.optuna_tuner import CoxPHTuner
        X, y = tuner_data
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tuner = CoxPHTuner(
                X, y, search_spaces={'model': {'alpha': (1e-4, 10.0, 'log')}},
            )
        suggestion = [w for w in caught
                      if issubclass(w.category, SuggestionWarning)]
        assert suggestion == []
        assert tuner.has_searchable_params is True

    def test_passthrough_tuner_does_not_warn(self, tuner_data):
        """A standard tuner consumes its whole model space — no warning, and
        it stays searchable."""
        X, y = tuner_data
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tuner = CoxNetTuner(
                X, y, model_name='coxnet',
                search_spaces={'model': {'l1_ratio': (0.1, 0.9)}},
            )
        suggestion = [w for w in caught
                      if issubclass(w.category, SuggestionWarning)]
        assert suggestion == []
        assert tuner.has_searchable_params is True

    def test_coxph_mixed_space_warns_for_unconsumed_only(self, tuner_data):
        """With a mix of consumed (alpha) and unconsumed (penalizer) model keys,
        the guard warns about the unconsumed key only; alpha is tuned."""
        from mlsurv.tuning.optuna_tuner import CoxPHTuner
        X, y = tuner_data
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tuner = CoxPHTuner(
                X, y,
                search_spaces={
                    'model': {'alpha': (1e-4, 10.0, 'log'), 'penalizer': (0.0, 1.0)},
                    'feature_selector': {'k': (2, 4)},
                },
            )
        suggestion = [w for w in caught
                      if issubclass(w.category, SuggestionWarning)]
        assert len(suggestion) == 1
        msg = str(suggestion[0].message)
        assert 'penalizer' in msg and 'alpha' not in msg
        # alpha (consumed) and the component space both keep tuning on.
        assert tuner.has_searchable_params is True

    def test_coxph_component_space_still_searchable(self, tuner_data):
        """CoxPH can still tune preprocessing: a component-only space neither
        warns nor disables tuning."""
        from mlsurv.tuning.optuna_tuner import CoxPHTuner
        X, y = tuner_data
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tuner = CoxPHTuner(
                X, y, search_spaces={'feature_selector': {'k': (2, 4)}},
            )
        suggestion = [w for w in caught
                      if issubclass(w.category, SuggestionWarning)]
        assert suggestion == []
        assert tuner.has_searchable_params is True


# ---------------------------------------------------------------------------
# _decode_best_params tests
# ---------------------------------------------------------------------------

class TestDecodeBestParams:
    """Tests for _decode_best_params categorical index decoding."""

    def _make_tuner(self, categorical_choices):
        tuner = MagicMock(spec=BaseTuner)
        tuner._categorical_choices = categorical_choices
        tuner._decode_best_params = BaseTuner._decode_best_params.__get__(tuner)
        return tuner

    def test_normal_decode(self):
        tuner = self._make_tuner({'layers': [[16], [32, 32], [64, 64, 64]]})
        result = tuner._decode_best_params({'layers': 1, 'lr': 0.01})
        assert result == {'layers': [32, 32], 'lr': 0.01}

    def test_out_of_range_index_warns(self):
        tuner = self._make_tuner({'layers': [[16], [32, 32]]})
        with patch('mlsurv.tuning.base.logger') as mock_logger:
            result = tuner._decode_best_params({'layers': 5})
        assert result == {'layers': 5}  # raw index returned
        mock_logger.warning.assert_called_once()
        assert "out of range" in mock_logger.warning.call_args[0][0]


# ---------------------------------------------------------------------------
# _evaluate_model exception guard tests
# ---------------------------------------------------------------------------

class TestEvaluateModelExceptionGuards:
    """Tests for _evaluate_model exception handling.

    Verifies that TrialPruned and MemoryError/RecursionError are re-raised
    directly, while other exceptions are converted to TrialPruned.
    """

    def test_trial_pruned_reraises_directly(self, tuner_data):
        """TrialPruned from _score_model is re-raised, not double-wrapped."""
        import optuna

        X, y = tuner_data

        class MockTuner(BaseTuner):
            DEFAULT_SPACE = {'n_estimators': (100, 500)}
            def objective(self, trial):
                return self._evaluate_model(trial, MagicMock())

        tuner = MockTuner(X, y)

        with patch.object(BaseTuner, '_score_model', side_effect=optuna.TrialPruned("inner")):
            with pytest.raises(optuna.TrialPruned, match="inner"):
                tuner._evaluate_model(MagicMock(), MagicMock())

    def test_memory_error_reraises(self, tuner_data):
        """MemoryError during fit propagates instead of becoming TrialPruned."""
        from sklearn.pipeline import Pipeline

        X, y = tuner_data

        class MockTuner(BaseTuner):
            DEFAULT_SPACE = {'n_estimators': (100, 500)}
            def objective(self, trial):
                return 0.5

        tuner = MockTuner(X, y)

        with patch.object(Pipeline, 'fit', side_effect=MemoryError("OOM")):
            with pytest.raises(MemoryError):
                tuner._evaluate_model(MagicMock(), MagicMock())

    def test_import_error_reraises(self, tuner_data):
        """ImportError (e.g. missing pycox) propagates instead of becoming TrialPruned."""
        from sklearn.pipeline import Pipeline

        X, y = tuner_data

        class MockTuner(BaseTuner):
            DEFAULT_SPACE = {'n_estimators': (100, 500)}
            def objective(self, trial):
                return 0.5

        tuner = MockTuner(X, y)

        with patch.object(Pipeline, 'fit', side_effect=ImportError("No module named 'pycox'")):
            with pytest.raises(ImportError):
                tuner._evaluate_model(MagicMock(), MagicMock())

    def test_regular_exception_becomes_trial_pruned(self, tuner_data):
        """Regular exceptions (e.g. LinAlgError) are converted to TrialPruned."""
        import optuna
        from sklearn.pipeline import Pipeline

        X, y = tuner_data

        class MockTuner(BaseTuner):
            DEFAULT_SPACE = {'n_estimators': (100, 500)}
            def objective(self, trial):
                return 0.5

        tuner = MockTuner(X, y)

        with patch.object(Pipeline, 'fit', side_effect=np.linalg.LinAlgError("singular")):
            with pytest.raises(optuna.TrialPruned):
                tuner._evaluate_model(MagicMock(), MagicMock())


# ---------------------------------------------------------------------------
# CoxNetTuner refactor tests
# ---------------------------------------------------------------------------

class TestCoxNetTunerRefactor:
    """Tests for CoxNetTuner unification with BaseTuner._evaluate_model
    and first-fold pruning fix."""

    def test_coxnet_delegates_to_evaluate_model(self, tuner_data):
        """CoxNetTuner.objective should delegate to _evaluate_model."""
        from mlsurv.tuning.optuna_tuner import CoxNetTuner

        X, y = tuner_data
        tuner = CoxNetTuner(X, y)

        with patch.object(BaseTuner, '_evaluate_model', return_value=0.72) as mock_eval:
            trial = MagicMock()
            trial.suggest_float.return_value = 0.5
            trial.suggest_int.return_value = 50
            result = tuner.objective(trial)
            mock_eval.assert_called_once()
            assert result == 0.72

    def test_validate_fold_scores_prunes_all_half(self, tuner_data):
        """All 0.5 scores should raise TrialPruned."""
        import optuna
        from mlsurv.tuning.optuna_tuner import CoxNetTuner

        X, y = tuner_data
        tuner = CoxNetTuner(X, y)

        with pytest.raises(optuna.TrialPruned):
            tuner._validate_fold_scores([0.5, 0.5, 0.5])

    def test_validate_fold_scores_prunes_near_half(self, tuner_data):
        """Near-0.5 scores within epsilon should also be pruned."""
        import optuna
        from mlsurv.tuning.optuna_tuner import CoxNetTuner

        X, y = tuner_data
        tuner = CoxNetTuner(X, y)

        near_half = [0.5 + 1e-12, 0.5 - 1e-15, 0.5]
        with pytest.raises(optuna.TrialPruned):
            tuner._validate_fold_scores(near_half)

    def test_validate_fold_scores_boundary_not_pruned(self, tuner_data):
        """Score at exactly epsilon boundary should NOT be pruned."""
        from mlsurv.tuning.optuna_tuner import CoxNetTuner

        X, y = tuner_data
        tuner = CoxNetTuner(X, y)

        # abs(0.5 + 1e-10 - 0.5) == 1e-10, which is NOT < 1e-10 (strict <)
        scores = [0.5 + 1e-10, 0.5 + 1e-10]
        result = tuner._validate_fold_scores(scores)
        assert result == scores

    def test_validate_fold_scores_passes_mixed(self, tuner_data):
        """Mixed scores (not all 0.5) should pass through."""
        from mlsurv.tuning.optuna_tuner import CoxNetTuner

        X, y = tuner_data
        tuner = CoxNetTuner(X, y)

        scores = [0.5, 0.65, 0.5]
        result = tuner._validate_fold_scores(scores)
        assert result == scores

    def test_base_validate_fold_scores_passthrough(self, tuner_data):
        """BaseTuner._validate_fold_scores is identity (pass-through)."""
        X, y = tuner_data
        tuner = BaseTuner(X, y)

        scores = [0.6, 0.7, 0.8]
        assert tuner._validate_fold_scores(scores) is scores

    def test_validate_fold_scores_empty_list(self, tuner_data):
        """Empty scores list should pass through without pruning."""
        from mlsurv.tuning.optuna_tuner import CoxNetTuner

        X, y = tuner_data
        tuner = CoxNetTuner(X, y)

        result = tuner._validate_fold_scores([])
        assert result == []

    def test_evaluate_model_prunes_on_empty_scores(self, tuner_data):
        """_evaluate_model raises TrialPruned when hook returns empty list."""
        import optuna

        X, y = tuner_data

        class EmptyHookTuner(BaseTuner):
            DEFAULT_SPACE = {'n_estimators': (100, 500)}
            def _validate_fold_scores(self, scores):
                return []  # Simulate filtering all scores
            def objective(self, trial):
                return 0.5

        tuner = EmptyHookTuner(X, y)

        with pytest.raises(optuna.TrialPruned):
            tuner._evaluate_model(MagicMock(), MagicMock())

    def test_base_validate_fold_scores_prunes_all_nan(self, tuner_data):
        """BaseTuner._validate_fold_scores raises TrialPruned when all scores are NaN."""
        import optuna

        X, y = tuner_data
        tuner = BaseTuner(X, y)

        with pytest.raises(optuna.TrialPruned, match="All CV folds returned NaN"):
            tuner._validate_fold_scores([float("nan"), float("nan"), float("nan")])

    def test_base_validate_fold_scores_passes_mixed_nan(self, tuner_data):
        """Mixed NaN/finite scores pass through without pruning."""
        X, y = tuner_data
        tuner = BaseTuner(X, y)

        scores = [0.6, float("nan"), 0.7]
        result = tuner._validate_fold_scores(scores)
        assert len(result) == 3
        assert result[0] == 0.6
        assert result[2] == 0.7

    def test_coxnet_validate_fold_scores_inherits_nan_pruning(self, tuner_data):
        """CoxNetTuner inherits all-NaN pruning from BaseTuner via super()."""
        import optuna

        X, y = tuner_data
        tuner = CoxNetTuner(X, y)

        with pytest.raises(optuna.TrialPruned, match="All CV folds returned NaN"):
            tuner._validate_fold_scores([float("nan"), float("nan")])

    def test_base_validate_fold_scores_empty_still_passthrough(self, tuner_data):
        """Empty list passes through (not treated as all-NaN)."""
        X, y = tuner_data
        tuner = BaseTuner(X, y)

        result = tuner._validate_fold_scores([])
        assert result == []

    def test_coxnet_warning_suppression_via_patterns(self, tuner_data):
        """CoxNet 'all coefficients are zero' is filtered by _SUPPRESSED_WARNING_PATTERNS."""
        X, y = tuner_data
        tuner = CoxNetTuner(X, y)
        tuner._trial_warnings = {}
        tuner._trial_warnings_lock = threading.Lock()

        # Simulate a caught warning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("all coefficients are zero")

        tuner._accumulate_trial_warnings(caught)
        # Should be filtered out — not accumulated
        assert len(tuner._trial_warnings) == 0

    def test_coxnet_all_half_through_evaluate_model(self, tuner_data):
        """CoxNet all-0.5 pruning fires through the real _evaluate_model path."""
        import optuna

        X, y = tuner_data
        tuner = CoxNetTuner(X, y)

        # Patch _score_model to always return 0.5 (all-zero coefficients)
        with patch.object(BaseTuner, '_score_model', return_value=0.5):
            trial = MagicMock()
            trial.suggest_float.return_value = 0.5
            trial.suggest_int.return_value = 50
            with pytest.raises(optuna.TrialPruned):
                tuner.objective(trial)

    def test_coxnet_objective_translates_alpha_index(self, tuner_data):
        """CoxNetTuner.objective translates alpha_index to integer alpha."""
        X, y = tuner_data
        tuner = CoxNetTuner(X, y)

        with patch.object(BaseTuner, '_evaluate_model', return_value=0.72) as mock_eval:
            trial = MagicMock()
            trial.suggest_float.return_value = 0.5
            trial.suggest_int.return_value = 42
            tuner.objective(trial)

            # Verify the model received alpha=42 (int), not alpha_index
            model_arg = mock_eval.call_args[0][1]
            assert model_arg.alpha == 42
            assert isinstance(model_arg.alpha, int)


class TestExtractTunedParamsCoxNet:
    """Test _extract_tuned_params translates alpha_index for CoxNet."""

    def test_alpha_index_translated_to_alpha(self):
        """_extract_tuned_params should convert alpha_index to alpha for coxnet."""
        from mlsurv.learner._tuning import TuningMixin

        mixin = TuningMixin()
        best_params = {'l1_ratio': 0.5, 'alpha_index': 42}
        model_params, _, _, _, _ = mixin._extract_tuned_params(best_params, 'coxnet')

        assert 'alpha' in model_params
        assert model_params['alpha'] == 42
        assert 'alpha_index' not in model_params

    def test_alpha_index_not_translated_for_other_models(self):
        """_extract_tuned_params should NOT translate alpha_index for non-coxnet models."""
        from mlsurv.learner._tuning import TuningMixin

        mixin = TuningMixin()
        best_params = {'n_estimators': 100, 'max_depth': 5}
        model_params, _, _, _, _ = mixin._extract_tuned_params(best_params, 'rsf')

        assert 'alpha' not in model_params
        assert 'alpha_index' not in model_params

    def test_alpha_index_translated_with_model_prefix(self):
        """model__alpha_index should be extracted and translated to alpha."""
        from mlsurv.learner._tuning import TuningMixin

        mixin = TuningMixin()
        best_params = {'model__l1_ratio': 0.5, 'model__alpha_index': 42}
        model_params, _, _, _, _ = mixin._extract_tuned_params(best_params, 'coxnet')

        assert model_params['alpha'] == 42
        assert 'alpha_index' not in model_params


# ---------------------------------------------------------------------------
# random_state propagation guard
# ---------------------------------------------------------------------------

class TestTunerForwardsRandomState:
    """Every tuner must forward ``random_state`` to a model that supports it.

    ``CoxNetTuner`` was the one seeded tuner that built its model without
    passing ``random_state`` — harmless under the default search space (which
    supplies ``alpha_index``, pinning each trial's model to one path index and
    bypassing the seeded inner CV) but nondeterministic when a custom space
    omits ``alpha_index``. This guard runs every tuner's
    ``objective()``, captures the constructed model, and asserts the seed
    reached any model that declares ``supports_random_state``. coxph/weibull_aft
    (flag ``False``) pass without a hand-maintained exclusion list.
    """

    # Deep tuners construct pycox/torch models, so they only run when those
    # libraries are installed (visible skip, mirroring TestEveryDeclaredParamIsTuned).
    _DEEP_TUNERS = {'deepsurv', 'deephit'}

    @pytest.mark.parametrize("name", list(TUNER_REGISTRY), ids=list(TUNER_REGISTRY))
    def test_random_state_forwarded(self, name, tuner_data):
        if name in self._DEEP_TUNERS and not _is_deep_learning_available():
            pytest.skip(
                f"{name}: pycox/torch not installed — deep tuner "
                f"random_state check skipped"
            )

        X, y = tuner_data
        seed = 12345
        tuner = TUNER_REGISTRY[name](X, y, model_name=name, random_state=seed)

        trial = _RecordingTrial()
        # Patch evaluation so the trial builds a model but never fits — we only
        # assert on the constructed model's random_state.
        with patch.object(BaseTuner, '_evaluate_model', return_value=0.5) as mock_eval:
            tuner.objective(trial)

        model = mock_eval.call_args[0][1]
        if model.supports_random_state:
            assert model.random_state == seed, (
                f"{name}: tuner did not forward random_state to its model"
            )

    def test_coxnet_explicitly_forwarded(self, tuner_data):
        """Pin coxnet — the actual regression target — non-parametrized.

        The parametrized test above guards the whole registry, but it would
        pass vacuously if the registry ever dropped coxnet or every captured
        model's ``supports_random_state`` flipped to ``False`` (every case would
        skip the assertion and stay green). This sentinel asserts coxnet's model
        both declares it supports a seed and actually receives it, so the
        original bug cannot silently reappear behind a green parametrization.
        """
        X, y = tuner_data
        seed = 12345
        tuner = CoxNetTuner(X, y, model_name='coxnet', random_state=seed)

        trial = _RecordingTrial()
        with patch.object(BaseTuner, '_evaluate_model', return_value=0.5) as mock_eval:
            tuner.objective(trial)

        model = mock_eval.call_args[0][1]
        assert model.supports_random_state is True
        assert model.random_state == seed


# ---------------------------------------------------------------------------
# Warning accumulation and deduplication tests
# ---------------------------------------------------------------------------

class TestWarningAccumulation:
    """Tests for per-trial warning capture, dedup, and emission.

    Verifies that tune() no longer blanket-suppresses UserWarnings and
    instead captures/deduplicates warnings from _evaluate_model CV folds.
    """

    def test_accumulate_trial_warnings_deduplicates(self, tuner_data):
        """Duplicate (category, message) pairs are counted, not repeated."""
        X, y = tuner_data
        tuner = BaseTuner(X, y)
        tuner._trial_warnings = {}
        tuner._trial_warnings_lock = threading.Lock()

        # Simulate two caught warnings with the same message
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("convergence not reached")
            warnings.warn("convergence not reached")

        tuner._accumulate_trial_warnings(caught)
        assert len(tuner._trial_warnings) == 1
        key = (UserWarning, "convergence not reached")
        assert tuner._trial_warnings[key] == 2

    def test_accumulate_different_warnings(self, tuner_data):
        """Different messages produce separate entries."""
        X, y = tuner_data
        tuner = BaseTuner(X, y)
        tuner._trial_warnings = {}
        tuner._trial_warnings_lock = threading.Lock()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("warning A")
            warnings.warn("warning B")

        tuner._accumulate_trial_warnings(caught)
        assert len(tuner._trial_warnings) == 2

    def test_emit_trial_warnings_annotates_count(self, tuner_data):
        """Warnings with count > 1 are annotated with '(xN)'."""
        X, y = tuner_data
        tuner = BaseTuner(X, y)
        tuner._trial_warnings = {
            (UserWarning, "convergence not reached"): 5,
            (UserWarning, "single occurrence"): 1,
        }
        tuner._trial_warnings_lock = threading.Lock()

        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            tuner._emit_trial_warnings()

        messages = [str(w.message) for w in emitted]
        assert any("convergence not reached (x5)" in m for m in messages)
        assert any("single occurrence" in m and "(x" not in m for m in messages)
        # Category should be preserved
        assert all(w.category is UserWarning for w in emitted)
        # Dict should be cleared after emission
        assert len(tuner._trial_warnings) == 0

    def test_suppressed_warning_patterns_filters(self, tuner_data):
        """_SUPPRESSED_WARNING_PATTERNS filters matching warnings."""
        X, y = tuner_data

        class FilteringTuner(BaseTuner):
            _SUPPRESSED_WARNING_PATTERNS = ("noisy warning",)
            def objective(self, trial):
                return 0.5

        tuner = FilteringTuner(X, y)
        tuner._trial_warnings = {}
        tuner._trial_warnings_lock = threading.Lock()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("this is a noisy warning from model")
            warnings.warn("important convergence warning")

        tuner._accumulate_trial_warnings(caught)
        assert len(tuner._trial_warnings) == 1
        assert (UserWarning, "important convergence warning") in tuner._trial_warnings

    def test_tune_does_not_blanket_suppress(self):
        """Regression: tune() source must not contain blanket UserWarning suppression."""
        source = inspect.getsource(BaseTuner.tune)
        assert "simplefilter" not in source
        # Ensure "ignore" and "UserWarning" don't co-occur (blanket suppression)
        assert not ("ignore" in source and "UserWarning" in source)

    def test_accumulate_works_without_tune(self, tuner_data):
        """_accumulate_trial_warnings works via __init__ without calling tune()."""
        X, y = tuner_data
        tuner = BaseTuner(X, y)
        # _trial_warnings is initialized in __init__ — accumulation should work
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("some warning")
        tuner._accumulate_trial_warnings(caught)
        assert len(tuner._trial_warnings) == 1

    def test_evaluate_model_captures_fold_warnings(self, tuner_data):
        """Warnings emitted during fit/score in _evaluate_model are accumulated."""
        X, y = tuner_data

        class WarnTuner(BaseTuner):
            DEFAULT_SPACE = {'n_estimators': (100, 500)}
            def objective(self, trial):
                return 0.5

        tuner = WarnTuner(X, y)

        def score_with_warning(self_, est, X, y, y_train):
            warnings.warn("convergence not reached")
            return 0.7

        with patch.object(BaseTuner, '_score_model', score_with_warning):
            tuner._evaluate_model(MagicMock(), MagicMock())

        assert any("convergence" in msg for (_, msg) in tuner._trial_warnings)

    def test_evaluate_model_preserves_warnings_on_exception(self, tuner_data):
        """Warnings from successful folds are preserved when a later fold raises."""
        import optuna

        X, y = tuner_data

        class WarnTuner(BaseTuner):
            DEFAULT_SPACE = {'n_estimators': (100, 500)}
            def objective(self, trial):
                return 0.5

        tuner = WarnTuner(X, y)
        call_count = 0

        def score_warn_then_fail(self_, est, X, y, y_train):
            nonlocal call_count
            call_count += 1
            warnings.warn("fold warning before failure")
            if call_count >= 3:
                raise ValueError("simulated fold failure")
            return 0.7

        with patch.object(BaseTuner, '_score_model', score_warn_then_fail):
            with pytest.raises(optuna.TrialPruned):
                tuner._evaluate_model(MagicMock(), MagicMock())

        # Warnings from folds before the failure should still be accumulated
        assert any("fold warning" in msg for (_, msg) in tuner._trial_warnings)

    def test_emit_preserves_non_user_warning_category(self, tuner_data):
        """Non-UserWarning categories are preserved through emit."""
        X, y = tuner_data
        tuner = BaseTuner(X, y)
        tuner._trial_warnings = {
            (DeprecationWarning, "old API"): 2,
            (UserWarning, "convergence"): 1,
        }

        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            tuner._emit_trial_warnings()

        categories = {w.category for w in emitted}
        assert DeprecationWarning in categories
        assert UserWarning in categories
        depr = [w for w in emitted if w.category is DeprecationWarning]
        assert "old API (x2)" in str(depr[0].message)

    def test_coxnet_alpha_index_warning_suppressed_behaviorally(self, tuner_data):
        """CoxNet alpha_index= warnings are filtered by _accumulate_trial_warnings."""
        X, y = tuner_data
        tuner = CoxNetTuner(X, y)
        tuner._trial_warnings = {}
        tuner._trial_warnings_lock = threading.Lock()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("alpha_index=5 is out of range, clamping to 3")

        tuner._accumulate_trial_warnings(caught)
        assert len(tuner._trial_warnings) == 0

    def test_non_matching_warning_not_suppressed(self, tuner_data):
        """Warnings that don't match suppressed patterns pass through."""
        X, y = tuner_data
        tuner = CoxNetTuner(X, y)
        tuner._trial_warnings = {}
        tuner._trial_warnings_lock = threading.Lock()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("convergence failed after 1000 iterations")

        tuner._accumulate_trial_warnings(caught)
        assert len(tuner._trial_warnings) == 1

    def test_base_tuner_no_suppressed_patterns(self):
        """BaseTuner has empty suppressed patterns by default."""
        assert BaseTuner._SUPPRESSED_WARNING_PATTERNS == ()


# ---------------------------------------------------------------------------
# Storage resolution debug logging
# ---------------------------------------------------------------------------

class TestStorageDebugLogging:
    """Tests for debug logging in _resolve_study_storage."""

    def test_delete_study_keyerror_logs_debug(self, tuner_data):
        """KeyError from delete_study should emit a debug log, not silently pass."""
        X, y = tuner_data
        tuner = BaseTuner(X, y)

        with patch('mlsurv.tuning.base.optuna.delete_study',
                   side_effect=KeyError("study_not_found")):
            with patch('mlsurv.tuning.base.logger') as mock_logger:
                result = tuner._resolve_study_storage(
                    storage="sqlite:///test.db",
                    experiment_name="test",
                    study_name="test/study",
                    resume=False,
                )
                assert result == ("sqlite:///test.db", "test/study")
                mock_logger.debug.assert_called_once()
                fmt_string = mock_logger.debug.call_args[0][0]
                assert "not found" in fmt_string
                # Verify study_name passed as format argument
                assert mock_logger.debug.call_args[0][1] == "test/study"


# ── Fatal error re-raise tests ────────────────────────────────────────────────


class TestFatalErrorReRaise:
    """Verify MemoryError/RecursionError propagate through broad except blocks."""

    def test_validate_storage_connection_reraises_memory_error(self):
        from mlsurv.tuning.storage import validate_storage_connection
        with patch('mlsurv.tuning.storage.optuna.storages.RDBStorage',
                   side_effect=MemoryError("OOM")):
            with pytest.raises(MemoryError):
                validate_storage_connection("sqlite:///test.db")

    def test_list_studies_from_storage_reraises_memory_error(self):
        from mlsurv.tuning.studies import _list_studies_from_storage
        with patch('mlsurv.tuning.studies.optuna.study.get_all_study_summaries',
                   side_effect=MemoryError("OOM")):
            with pytest.raises(MemoryError):
                _list_studies_from_storage("sqlite:///test.db")


class TestListStudiesContract:
    """Pin the dict-return contract of module-level mlsurv.tuning.list_studies().

    Guards the documented global-view workflow (docs/evaluation.md) after the
    instance method was rescoped to a flat single-experiment list.
    """

    def test_file_mode_returns_dict_keyed_by_experiment(self, tmp_path):
        import mlsurv.tuning as tuning

        # Two experiment dirs with dummy study files.
        exp_a = tmp_path / "gbsg2_analysis"
        exp_b = tmp_path / "flchain_cohort"
        exp_a.mkdir()
        exp_b.mkdir()
        (exp_a / "rsf_study.db").write_text("")
        (exp_a / "gbsa_study.db").write_text("")
        (exp_b / "coxnet_study.db").write_text("")

        result = tuning.list_studies(base_dir=tmp_path)

        assert isinstance(result, dict)
        assert set(result) == {"gbsg2_analysis", "flchain_cohort"}
        assert sorted(result["gbsg2_analysis"]) == ["gbsa_study.db", "rsf_study.db"]
        assert result["flchain_cohort"] == ["coxnet_study.db"]

    def test_file_mode_ignores_non_study_files_and_empty_dirs(self, tmp_path):
        import mlsurv.tuning as tuning

        exp = tmp_path / "exp1"
        exp.mkdir()
        (exp / "rsf_study.db").write_text("")
        (exp / "notes.txt").write_text("")          # not *_study.db -> excluded
        (tmp_path / "empty_exp").mkdir()             # no db files -> key omitted
        (tmp_path / "stray.db").write_text("")       # file at base, not a dir -> skipped

        result = tuning.list_studies(base_dir=tmp_path)

        assert set(result) == {"exp1"}
        assert result["exp1"] == ["rsf_study.db"]

    def test_file_mode_missing_base_dir_returns_empty_dict(self, tmp_path):
        import mlsurv.tuning as tuning
        result = tuning.list_studies(base_dir=tmp_path / "does_not_exist")
        assert result == {}

    @staticmethod
    def _summary(study_name, n_trials, best_value, has_best_trial=True):
        s = MagicMock()
        s.study_name = study_name
        s.n_trials = n_trials
        # best_trial must be a real None (not a MagicMock) to exercise the
        # `s.best_trial.value if s.best_trial else None` guard in studies.py.
        s.best_trial = MagicMock(value=best_value) if has_best_trial else None
        s.datetime_start = None
        return s

    @staticmethod
    def _list_with_summaries(summaries):
        import mlsurv.tuning as tuning
        with patch(
            "mlsurv.tuning.studies.optuna.study.get_all_study_summaries",
            return_value=summaries,
        ):
            return tuning.list_studies(storage="sqlite:///fake.db")

    def test_storage_mode_returns_dict_of_study_info(self):
        result = self._list_with_summaries([
            self._summary("gbsg2_analysis/rsf_tuning", 50, 0.85),
            self._summary("flchain_cohort/rsf_tuning", 30, 0.72),
        ])

        assert isinstance(result, dict)
        assert set(result) == {"gbsg2_analysis", "flchain_cohort"}
        info = result["gbsg2_analysis"][0]
        # Pin the full key set, not just a spot check.
        assert set(info) == {
            "name", "full_name", "n_trials", "best_value", "datetime_start",
        }
        assert info["name"] == "rsf_tuning"
        assert info["full_name"] == "gbsg2_analysis/rsf_tuning"
        assert info["n_trials"] == 50
        assert info["best_value"] == 0.85

    def test_storage_mode_groups_multiple_studies_under_one_experiment(self):
        result = self._list_with_summaries([
            self._summary("gbsg2_analysis/rsf_tuning", 50, 0.85),
            self._summary("gbsg2_analysis/gbsa_tuning", 40, 0.80),
        ])

        assert set(result) == {"gbsg2_analysis"}
        names = sorted(s["name"] for s in result["gbsg2_analysis"])
        assert names == ["gbsa_tuning", "rsf_tuning"]

    def test_storage_mode_default_experiment_and_missing_best_trial(self):
        # Slash-less study name -> '_default' bucket; no best_trial -> best_value None.
        result = self._list_with_summaries([
            self._summary("orphan_study", 0, None, has_best_trial=False),
        ])

        assert set(result) == {"_default"}
        info = result["_default"][0]
        assert info["name"] == "orphan_study"
        assert info["full_name"] == "orphan_study"
        assert info["best_value"] is None


# ── Module structure smoke tests ──────────────────────────────────────────────


class TestBackwardCompatShim:
    """Verify the backward-compat shim re-exports all expected names."""

    def test_shim_exports_all_names(self):
        import mlsurv.tuning.optuna_tuner as shim
        for name in shim.__all__:
            assert hasattr(shim, name), f"{name} missing from optuna_tuner shim"

    def test_shim_identity(self):
        """Objects from shim are identical to objects from canonical modules."""
        from mlsurv.tuning.optuna_tuner import BaseTuner as ShimBT
        from mlsurv.tuning.base import BaseTuner as CanonicalBT
        assert ShimBT is CanonicalBT

        from mlsurv.tuning.optuna_tuner import get_tuner as shim_gt
        from mlsurv.tuning.tuners import get_tuner as canonical_gt
        assert shim_gt is canonical_gt

        from mlsurv.tuning.optuna_tuner import export_study as shim_es
        from mlsurv.tuning.studies import export_study as canonical_es
        assert shim_es is canonical_es

        from mlsurv.tuning.optuna_tuner import get_storage_url as shim_gsu
        from mlsurv.tuning.storage import get_storage_url as canonical_gsu
        assert shim_gsu is canonical_gsu


class TestTunerRegistryCompleteness:
    """Verify TUNER_REGISTRY has an entry for every supported model."""

    def test_registry_has_all_models(self):
        from mlsurv.tuning.tuners import TUNER_REGISTRY
        expected = {
            'coxph', 'coxnet', 'rsf', 'gbsa', 'fssvm',
            'deepsurv', 'xgbcox', 'xgbaft', 'deephit', 'weibull_aft',
        }
        assert set(TUNER_REGISTRY.keys()) == expected

    def test_registry_entries_are_basetuner_subclasses(self):
        from mlsurv.tuning.tuners import TUNER_REGISTRY
        from mlsurv.tuning.base import BaseTuner
        for name, cls in TUNER_REGISTRY.items():
            assert issubclass(cls, BaseTuner), (
                f"{name} -> {cls.__name__} is not a BaseTuner subclass"
            )

    def test_package_level_exports(self):
        """All names in mlsurv.tuning.__all__ are importable."""
        import mlsurv.tuning as pkg
        for name in pkg.__all__:
            assert hasattr(pkg, name), f"{name} missing from mlsurv.tuning"


# ---------------------------------------------------------------------------
# Warm-start tests
# ---------------------------------------------------------------------------

class TestWarmStart:
    """Tests for _get_warm_start_params() hook and enqueue_trial integration."""

    def test_base_tuner_returns_none(self, tuner_data):
        """BaseTuner._get_warm_start_params() returns None by default."""
        X, y = tuner_data
        tuner = BaseTuner(X, y)
        assert tuner._get_warm_start_params() is None

    def test_deephit_warm_start_keys(self, tuner_data):
        """DeepHitTuner returns warm-start params with correct keys."""
        from mlsurv.tuning.tuners.deephit import DeepHitTuner

        X, y = tuner_data
        tuner = DeepHitTuner(X, y, model_name='deephit')
        params = tuner._get_warm_start_params()
        assert params is not None
        expected_keys = {
            'n_layers', 'n_nodes', 'dropout', 'activation', 'alpha', 'sigma',
            'learning_rate', 'weight_decay', 'batch_size', 'num_durations',
        }
        assert set(params.keys()) == expected_keys
        assert isinstance(params['n_layers'], int)
        assert isinstance(params['n_nodes'], int)

    def test_deepsurv_warm_start_keys(self, tuner_data):
        """DeepSurvTuner returns warm-start params with correct keys."""
        from mlsurv.tuning.tuners.deepsurv import DeepSurvTuner

        X, y = tuner_data
        tuner = DeepSurvTuner(X, y, model_name='deepsurv')
        params = tuner._get_warm_start_params()
        assert params is not None
        expected_keys = {
            'n_layers', 'n_nodes', 'dropout', 'activation',
            'weight_decay', 'batch_size', 'lr',
        }
        assert set(params.keys()) == expected_keys

    def test_warm_start_skipped_on_resume(self, tuner_data):
        """Warm-start enqueue should be skipped when resume=True."""
        X, y = tuner_data

        class WarmTuner(BaseTuner):
            DEFAULT_SPACE = {'n_estimators': (100, 500)}

            def _get_warm_start_params(self):
                return {'n_estimators': 200}

            def objective(self, trial):
                return 0.5

        tuner = WarmTuner(X, y, model_name='test')
        # Patch _get_warm_start_params to track calls
        with patch.object(WarmTuner, '_get_warm_start_params', return_value={'n_estimators': 200}) as mock_warm:
            with patch('optuna.study.Study.optimize'):
                with patch('optuna.study.Study.best_params', new_callable=lambda: property(lambda self: {'n_estimators': 200})):
                    tuner.tune(n_trials=1, resume=True, optuna_verbose=False)
            # _get_warm_start_params should NOT have been called during resume
            mock_warm.assert_not_called()

    def test_enqueue_trial_called_on_fresh_start(self, tuner_data):
        """enqueue_trial should be called with warm-start params when not resuming."""
        X, y = tuner_data

        class WarmTuner(BaseTuner):
            DEFAULT_SPACE = {'n_estimators': (100, 500)}

            def _get_warm_start_params(self):
                return {'n_estimators': 200}

            def objective(self, trial):
                return 0.5

        tuner = WarmTuner(X, y, model_name='test')
        with patch('optuna.study.Study.optimize'):
            with patch('optuna.study.Study.enqueue_trial') as mock_enqueue:
                with patch('optuna.study.Study.best_params', new_callable=lambda: property(lambda self: {'n_estimators': 200})):
                    tuner.tune(n_trials=1, resume=False, optuna_verbose=False)
                mock_enqueue.assert_called_once_with({'n_estimators': 200})

    def test_deephit_warm_start_within_search_space(self):
        """Warm-start params should fall within (possibly constrained) search space."""
        from mlsurv.tuning.tuners.deephit import DeepHitTuner
        from mlsurv.tuning.search_spaces import get_search_space

        for n, n_feat, n_ev in [(300, 5, 200), (1000, 10, 600), (5000, 10, 3500)]:
            X = pd.DataFrame(np.random.randn(n, n_feat))
            y = make_survival_array(
                [i < n_ev for i in range(n)], [float(i + 1) for i in range(n)]
            )
            tuner = DeepHitTuner(X, y, model_name='deephit')
            params = tuner._get_warm_start_params()
            space = get_search_space(
                'deephit', n_samples=n, n_events=n_ev, n_features=n_feat,
            )
            # n_nodes and n_layers must be within search space range
            lo, hi = space['n_nodes']
            assert lo <= params['n_nodes'] <= hi, (
                f"n_nodes={params['n_nodes']} outside ({lo}, {hi}) for n={n}"
            )
            lo, hi = space['n_layers']
            assert lo <= params['n_layers'] <= hi, (
                f"n_layers={params['n_layers']} outside ({lo}, {hi}) for n={n}"
            )

    def test_clamp_warm_start_categorical_snap(self, tuner_data):
        """Out-of-range numeric categorical should snap to nearest choice."""
        X, y = tuner_data
        tuner = BaseTuner(X, y)
        tuner.search_space = {'batch_size': [64, 128, 256]}
        result = tuner._clamp_warm_start({'batch_size': 32})
        assert result == {'batch_size': 64}

    def test_clamp_warm_start_categorical_exact(self, tuner_data):
        """Valid categorical value should pass through unchanged."""
        X, y = tuner_data
        tuner = BaseTuner(X, y)
        tuner.search_space = {'batch_size': [64, 128, 256]}
        result = tuner._clamp_warm_start({'batch_size': 128})
        assert result == {'batch_size': 128}

    def test_clamp_warm_start_categorical_string_fallback(self, tuner_data):
        """Non-numeric categorical mismatch should fall back to first choice."""
        X, y = tuner_data
        tuner = BaseTuner(X, y)
        tuner.search_space = {'activation': ['selu', 'relu']}
        result = tuner._clamp_warm_start({'activation': 'tanh'})
        assert result == {'activation': 'selu'}

    def test_deepsurv_warm_start_small_data_valid(self):
        """DeepSurv warm-start batch_size must be in search space for small data."""
        from mlsurv.tuning.tuners.deepsurv import DeepSurvTuner
        from mlsurv.tuning.search_spaces import DEFAULT_SPACES

        np.random.seed(42)
        n = 100  # Small dataset where n_samples//8 < 64
        X = pd.DataFrame(np.random.randn(n, 2), columns=['f1', 'f2'])
        y = make_survival_array(
            np.random.binomial(1, 0.6, n), np.random.exponential(2, n)
        )
        tuner = DeepSurvTuner(X, y, model_name='deepsurv')
        params = tuner._get_warm_start_params()
        clamped = tuner._clamp_warm_start(params)
        valid = DEFAULT_SPACES['deepsurv']['batch_size']
        assert clamped['batch_size'] in valid, (
            f"batch_size={clamped['batch_size']} not in {valid}"
        )


# ---------------------------------------------------------------------------
# Static prefix precomputation tests
# ---------------------------------------------------------------------------

class TestPrecomputeStaticPrefix:
    """Tests for _precompute_static_prefix() and precomputed-fold evaluation."""

    def test_all_static_prefix(self, tuner_data):
        """When no component has tunable params, all are pre-computed."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        X, y = tuner_data
        tuner = BaseTuner(
            X, y,
            imputer=SimpleImputer(),
            scaler=StandardScaler(),
            cv_folds=3,
            random_state=42,
        )
        tuner._precompute_static_prefix()

        assert tuner._precomputed_folds is not None
        assert len(tuner._precomputed_folds) == 3
        # Dynamic suffix should be empty (only model left)
        assert tuner._dynamic_component_defs == []

        # Precomputed X should be different from raw (scaled)
        X_pre_train = tuner._precomputed_folds[0][0]
        assert not np.allclose(X_pre_train.mean(axis=0), X.mean(axis=0))

    def test_partial_static_prefix(self, tuner_data):
        """When scaler has tunable params, only imputer is pre-computed."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        X, y = tuner_data
        tuner = BaseTuner(
            X, y,
            imputer=SimpleImputer(),
            scaler=StandardScaler(),
            search_spaces={'scaler': {'with_mean': [True, False]}},
            cv_folds=3,
            random_state=42,
        )
        tuner._precompute_static_prefix()

        assert tuner._precomputed_folds is not None
        assert len(tuner._precomputed_folds) == 3
        # Dynamic suffix should include scaler
        dynamic_names = [name for name, _, _ in tuner._dynamic_component_defs]
        assert 'scaler' in dynamic_names
        assert 'imputer' not in dynamic_names

    def test_no_static_prefix(self, tuner_data):
        """When the first component is tuned, no precomputation occurs."""
        from sklearn.impute import KNNImputer

        X, y = tuner_data
        tuner = BaseTuner(
            X, y,
            imputer=KNNImputer(),
            search_spaces={'imputer': {'n_neighbors': (3, 10)}},
            cv_folds=3,
            random_state=42,
        )
        tuner._precompute_static_prefix()

        assert tuner._precomputed_folds is None
        assert tuner._dynamic_component_defs is None

    def test_no_components_uses_defaults(self, tuner_data):
        """Backward-compat path: default imputer+scaler are pre-computed."""
        X, y = tuner_data
        tuner = BaseTuner(X, y, cv_folds=3, random_state=42)
        tuner._precompute_static_prefix()

        assert tuner._precomputed_folds is not None
        assert len(tuner._precomputed_folds) == 3
        assert tuner._dynamic_component_defs == []

    def test_skipped_none_components(self, tuner_data):
        """Components that are None are skipped gracefully."""
        from sklearn.preprocessing import StandardScaler

        X, y = tuner_data
        tuner = BaseTuner(
            X, y,
            imputer=None,
            scaler=StandardScaler(),
            cv_folds=3,
            random_state=42,
        )
        tuner._precompute_static_prefix()

        assert tuner._precomputed_folds is not None
        assert len(tuner._precomputed_folds) == 3

    def test_precomputed_fits_per_fold(self, tuner_data):
        """Precomputed fold data matches independent per-fold fitting.

        The static prefix is fit independently on each fold's training split,
        not on the full training set.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.base import clone

        X, y = tuner_data
        imputer = SimpleImputer()
        scaler = StandardScaler()

        tuner = BaseTuner(
            X, y,
            imputer=imputer,
            scaler=scaler,
            cv_folds=3,
            random_state=42,
        )
        tuner._precompute_static_prefix()

        # Manually compute what per-fold fitting produces
        for fold_idx, (train_idx, val_idx) in enumerate(tuner._cv_split()):
            imp = clone(imputer)
            sc = clone(scaler)
            X_tr = imp.fit_transform(X.iloc[train_idx])
            X_tr = sc.fit_transform(X_tr)
            X_v = imp.transform(X.iloc[val_idx])
            X_v = sc.transform(X_v)

            X_pre_tr, X_pre_v, _, _ = tuner._precomputed_folds[fold_idx]
            np.testing.assert_array_almost_equal(X_pre_tr, X_tr)
            np.testing.assert_array_almost_equal(X_pre_v, X_v)

    def test_evaluate_model_skips_static_prefix(self, tuner_data):
        """When precomputed, static components are NOT re-fitted during eval."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        X, y = tuner_data
        imputer = SimpleImputer()
        scaler = StandardScaler()

        tuner = BaseTuner(
            X, y,
            imputer=imputer,
            scaler=scaler,
            cv_folds=3,
            random_state=42,
        )
        tuner._precompute_static_prefix()

        # Create a mock trial
        mock_trial = MagicMock()
        mock_trial.suggest_int = MagicMock(return_value=100)
        mock_trial.suggest_float = MagicMock(return_value=0.5)

        # Create a mock model that returns predictions
        mock_model = MagicMock()
        mock_model.fit = MagicMock()
        mock_model.predict = MagicMock(return_value=np.random.randn(len(X) // 3))

        # Spy on the original imputer's fit_transform
        original_fit_transform = SimpleImputer.fit_transform
        call_count = 0

        def counting_fit_transform(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_fit_transform(self, *args, **kwargs)

        with patch.object(SimpleImputer, 'fit_transform', counting_fit_transform):
            with patch.object(tuner, '_score_model', return_value=0.7):
                tuner._evaluate_model(mock_trial, mock_model)

        # SimpleImputer.fit_transform should NOT have been called
        # (precomputed path doesn't touch the imputer)
        assert call_count == 0

    def test_cleanup_after_tune(self, tuner_data):
        """Precomputed data is freed after tune() completes."""
        X, y = tuner_data

        class DummyTuner(BaseTuner):
            DEFAULT_SPACE = {}

            def objective(self, trial):
                return 0.5

        tuner = DummyTuner(X, y, model_name='test', cv_folds=3, random_state=42)
        with patch('optuna.study.Study.optimize'):
            with patch(
                'optuna.study.Study.best_params',
                new_callable=lambda: property(lambda self: {}),
            ):
                tuner.tune(n_trials=1, optuna_verbose=False)

        assert tuner._precomputed_folds is None
        assert tuner._dynamic_component_defs is None

    def test_precomputed_scores_are_valid(self, tuner_data):
        """Precomputed path produces valid (finite) scores."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from mlsurv.models import CoxPHModel

        X, y = tuner_data
        tuner = BaseTuner(
            X, y,
            imputer=SimpleImputer(),
            scaler=StandardScaler(),
            cv_folds=3,
            random_state=42,
        )

        mock_trial = MagicMock()
        model = CoxPHModel()

        # Score WITH precomputation (fits on full X, then slices per fold)
        tuner._precompute_static_prefix()
        score_precomputed = tuner._evaluate_model(mock_trial, model)

        assert np.isfinite(score_precomputed)
        # C-index should be in a reasonable range
        assert 0.0 <= score_precomputed <= 1.0

    def test_static_selector_with_y(self, tuner_data):
        """Static selector that uses y in fit() works in the precomputed path."""
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from mlsurv.preprocessing.selection import TopKCoxSelector
        from mlsurv.models import CoxPHModel

        X, y = tuner_data
        tuner = BaseTuner(
            X, y,
            imputer=SimpleImputer(),
            scaler=StandardScaler(),
            selector=TopKCoxSelector(k=1),
            cv_folds=3,
            random_state=42,
        )
        tuner._precompute_static_prefix()

        assert tuner._precomputed_folds is not None
        # Selector reduces features — precomputed X should have k=1 columns
        X_pre_train = tuner._precomputed_folds[0][0]
        assert X_pre_train.shape[1] == 1

        # Each fold selects k=1 feature independently
        for X_tr, X_v, _, _ in tuner._precomputed_folds:
            assert X_tr.shape[1] == 1
            assert X_v.shape[1] == 1

        # Precomputed score should be valid (not NaN)
        mock_trial = MagicMock()
        model = CoxPHModel()
        score_pre = tuner._evaluate_model(mock_trial, model)
        assert np.isfinite(score_pre)

    def test_precomputed_exception_parity_memory_error(self, tuner_data):
        """MemoryError propagates from _evaluate_model_precomputed."""
        from sklearn.impute import SimpleImputer

        X, y = tuner_data
        tuner = BaseTuner(
            X, y, imputer=SimpleImputer(), cv_folds=3, random_state=42,
        )
        tuner._precompute_static_prefix()

        mock_model = MagicMock()
        mock_model.fit = MagicMock(side_effect=MemoryError("OOM"))

        with pytest.raises(MemoryError):
            tuner._evaluate_model(MagicMock(), mock_model)

    def test_precomputed_exception_parity_generic(self, tuner_data):
        """Generic exceptions become TrialPruned in precomputed path."""
        import optuna
        from sklearn.impute import SimpleImputer

        X, y = tuner_data
        tuner = BaseTuner(
            X, y, imputer=SimpleImputer(), cv_folds=3, random_state=42,
        )
        tuner._precompute_static_prefix()

        mock_model = MagicMock()
        mock_model.fit = MagicMock(side_effect=ValueError("bad param"))

        with pytest.raises(optuna.TrialPruned, match="Fold fit/score failed"):
            tuner._evaluate_model(MagicMock(), mock_model)

    def test_cleanup_on_precompute_failure(self, tuner_data):
        """If _precompute_static_prefix raises, cleanup still runs."""
        X, y = tuner_data

        class DummyTuner(BaseTuner):
            DEFAULT_SPACE = {}
            def objective(self, trial):
                return 0.5

        tuner = DummyTuner(X, y, model_name='test', cv_folds=3, random_state=42)
        # Force precomputation to fail
        with patch.object(
            BaseTuner, '_precompute_static_prefix',
            side_effect=RuntimeError("imputer failed"),
        ):
            with patch('optuna.study.Study.optimize'):
                with patch(
                    'optuna.study.Study.best_params',
                    new_callable=lambda: property(lambda self: {}),
                ):
                    with pytest.raises(RuntimeError, match="imputer failed"):
                        tuner.tune(n_trials=1, optuna_verbose=False)

        # Cleanup should have run in finally block
        assert tuner._precomputed_folds is None
        assert tuner._dynamic_component_defs is None

    def test_partial_prefix_end_to_end(self, tuner_data):
        """Partial precompute (static imputer, tuned scaler) produces valid scores."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from mlsurv.models import CoxPHModel

        X, y = tuner_data
        tuner = BaseTuner(
            X, y,
            imputer=SimpleImputer(),
            scaler=StandardScaler(),
            search_spaces={'scaler': {'with_mean': [True, False]}},
            cv_folds=3,
            random_state=42,
        )
        tuner._precompute_static_prefix()

        # Build a real trial via Optuna to exercise _suggest_component_params
        import optuna
        study = optuna.create_study()

        def objective(trial):
            model = CoxPHModel()
            return tuner._evaluate_model(trial, model)

        study.optimize(objective, n_trials=2)
        # Should complete without error and produce valid scores
        assert study.best_value > 0.0

    # ------------------------------------------------------------------
    # Per-fold dynamic preprocessing
    # ------------------------------------------------------------------

    def test_dynamic_selector_fit_per_fold(self, tuner_data):
        """When selector has tunable k, fit() is called once per fold."""
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from mlsurv.preprocessing.selection import TopKCoxSelector
        from mlsurv.models import CoxPHModel

        X, y = tuner_data
        tuner = BaseTuner(
            X, y,
            imputer=SimpleImputer(),
            scaler=StandardScaler(),
            selector=TopKCoxSelector(k=1),
            search_spaces={'feature_selector': {'k': [1, 2]}},
            cv_folds=3,
            random_state=42,
        )
        tuner._precompute_static_prefix()

        # Track how many times TopKCoxSelector.fit is called
        fit_call_count = 0
        original_fit = TopKCoxSelector.fit

        def counting_fit(self, X, y=None):
            nonlocal fit_call_count
            fit_call_count += 1
            return original_fit(self, X, y)

        import optuna
        study = optuna.create_study()

        def objective(trial):
            model = CoxPHModel()
            return tuner._evaluate_model(trial, model)

        with patch.object(TopKCoxSelector, 'fit', counting_fit):
            study.optimize(objective, n_trials=1)

        # Selector should be fit once per fold (3 folds)
        assert fit_call_count == 3, (
            f"Expected selector.fit() called once per fold (3), got {fit_call_count}"
        )

    def test_fallback_path_fits_per_fold(self, tuner_data):
        """Fallback path (no static prefix): full pipeline fit per fold."""
        from sklearn.impute import KNNImputer
        from mlsurv.models import CoxPHModel

        X, y = tuner_data
        # Imputer is tuned → first component is dynamic → no static prefix
        tuner = BaseTuner(
            X, y,
            imputer=KNNImputer(),
            search_spaces={'imputer': {'n_neighbors': [3, 5]}},
            cv_folds=3,
            random_state=42,
        )
        tuner._precompute_static_prefix()
        assert tuner._precomputed_folds is None  # no static prefix

        fit_call_count = 0
        original_fit = KNNImputer.fit

        def counting_fit(self, X, y=None):
            nonlocal fit_call_count
            fit_call_count += 1
            return original_fit(self, X, y)

        import optuna
        study = optuna.create_study()

        def objective(trial):
            model = CoxPHModel()
            return tuner._evaluate_model(trial, model)

        with patch.object(KNNImputer, 'fit', counting_fit):
            study.optimize(objective, n_trials=1)

        # Imputer should be fit once per fold (3 folds)
        assert fit_call_count == 3, (
            f"Expected imputer.fit() called once per fold (3), got {fit_call_count}"
        )

    def test_no_dynamic_components_unchanged(self, tuner_data):
        """All-static case: behavior unchanged, model-only per fold."""
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from mlsurv.models import CoxPHModel

        X, y = tuner_data
        tuner = BaseTuner(
            X, y,
            imputer=SimpleImputer(),
            scaler=StandardScaler(),
            cv_folds=3,
            random_state=42,
        )
        tuner._precompute_static_prefix()

        # All static, no dynamic components
        assert tuner._dynamic_component_defs == []

        mock_trial = MagicMock()
        model = CoxPHModel()
        score = tuner._evaluate_model(mock_trial, model)

        assert np.isfinite(score)
        assert 0.0 <= score <= 1.0

    def test_memory_error_in_fallback_path_propagates(self, tuner_data):
        """MemoryError during per-fold pipeline.fit() propagates, not TrialPruned."""
        from sklearn.impute import KNNImputer
        from mlsurv.models import CoxPHModel
        import optuna

        X, y = tuner_data
        tuner = BaseTuner(
            X, y,
            imputer=KNNImputer(),
            search_spaces={'imputer': {'n_neighbors': [3, 5]}},
            cv_folds=3,
            random_state=42,
        )
        tuner._precompute_static_prefix()
        assert tuner._precomputed_folds is None  # no static prefix

        # Use a real Optuna trial so parameter suggestion works
        study = optuna.create_study()
        trial = study.ask()
        model = CoxPHModel()
        with patch.object(CoxPHModel, 'fit', side_effect=MemoryError("OOM")):
            with pytest.raises(MemoryError):
                tuner._evaluate_model(trial, model)

    def test_memory_error_in_precomputed_path_propagates(self, tuner_data):
        """MemoryError during precomputed per-fold pipeline.fit() propagates."""
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from mlsurv.models import CoxPHModel

        X, y = tuner_data
        tuner = BaseTuner(
            X, y,
            imputer=SimpleImputer(),
            scaler=StandardScaler(),
            cv_folds=3,
            random_state=42,
        )
        tuner._precompute_static_prefix()
        assert tuner._precomputed_folds is not None

        model = CoxPHModel()
        with patch.object(CoxPHModel, 'fit', side_effect=MemoryError("OOM")):
            with pytest.raises(MemoryError):
                tuner._evaluate_model(MagicMock(), model)
