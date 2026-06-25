"""Tests for mlsurv.evaluation.helpers and mlsurv.evaluation.evaluate utilities.

Covers resolve_n_bootstrap() auto-computation, resolve_model_name() with
various model types (direct models, pipelines, anonymous objects),
_format_elapsed() time formatting, parameter change detection helpers
(_find_changed_params, _warn_unused_shap_params, _warn_unused_bootstrap_params),
EvalConfig validation, and phase helpers
(_run_importance_phase, _run_shap_phase, _run_interactions_phase).
"""

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import numpy as np
import pandas as pd

from mlsurv._warnings import SuggestionWarning
from mlsurv.evaluation.helpers import resolve_n_bootstrap, resolve_model_name, _format_elapsed
from mlsurv.evaluation._config import EvalConfig
from mlsurv.evaluation.evaluate import (
    _find_changed_params,
    _warn_unused_shap_params,
    _warn_unused_bootstrap_params,
    _run_importance_phase,
    _run_shap_phase,
    _run_interactions_phase,
    _SHAP_EXCEPTIONS,
)


class TestResolveNBootstrap:
    """Tests for auto-resolving 'auto' bootstrap iterations."""

    def test_auto_resolves_to_formula(self):
        """'auto' should use min(500, max(100, 10*sqrt(n)))."""
        result = resolve_n_bootstrap("auto", 100, show_progress=False)
        assert result == 100  # 10*sqrt(100) = 100

    def test_auto_small_sample(self):
        """Small samples should floor at 100 iterations."""
        result = resolve_n_bootstrap("auto", 10, show_progress=False)
        assert result == 100

    def test_auto_large_sample(self):
        """Large samples should cap at 500 iterations."""
        result = resolve_n_bootstrap("auto", 10000, show_progress=False)
        assert result == 500

    def test_auto_medium_sample(self):
        """Medium samples should scale with sqrt(n)."""
        result = resolve_n_bootstrap("auto", 400, show_progress=False)
        assert result == 200  # 10*sqrt(400) = 200

    def test_int_passthrough(self):
        """Integer values pass through unchanged."""
        result = resolve_n_bootstrap(200, 100, show_progress=False)
        assert result == 200

    def test_auto_prints_message(self, capsys):
        """Auto resolution prints message when show_progress=True and bootstrap=True."""
        resolve_n_bootstrap("auto", 100, show_progress=True, bootstrap=True)
        captured = capsys.readouterr()
        assert "automatically set to 100" in captured.out

    def test_auto_no_print_when_show_progress_false(self, capsys):
        """No message when show_progress=False."""
        resolve_n_bootstrap("auto", 100, show_progress=False, bootstrap=True)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_auto_no_print_when_bootstrap_false(self, capsys):
        """No message when bootstrap=False."""
        resolve_n_bootstrap("auto", 100, show_progress=True, bootstrap=False)
        captured = capsys.readouterr()
        assert captured.out == ""


class TestResolveModelName:
    """Tests for extracting model names from various model types.

    Models may be raw instances, sklearn Pipelines, or objects with a
    'name' attribute. This function must handle all cases correctly
    to ensure evaluation results are labeled properly.
    """

    def test_model_with_name_attribute(self):
        """Models with a 'name' attribute should return that name."""
        model = SimpleNamespace(name="rsf")
        assert resolve_model_name(model) == "rsf"

    def test_model_without_name_falls_back_to_class_name(self):
        """Models without 'name' should return class name."""

        class RandomSurvivalForest:
            pass

        assert resolve_model_name(RandomSurvivalForest()) == "RandomSurvivalForest"

    def test_pipeline_with_named_last_step(self):
        """Pipeline should return name from last step's 'name' attribute."""
        last_step = SimpleNamespace(name="coxph")
        model = SimpleNamespace(name=None, named_steps={"imputer": object(), "model": last_step})
        assert resolve_model_name(model) == "coxph"

    def test_pipeline_last_step_no_name(self):
        """Pipeline with unnamed last step should return class name."""

        class CoxPHSurvivalAnalysis:
            pass

        model = SimpleNamespace(name=None, named_steps={"model": CoxPHSurvivalAnalysis()})
        assert resolve_model_name(model) == "CoxPHSurvivalAnalysis"

    def test_none_name_attribute(self):
        """Model where name is explicitly None should check pipeline path."""

        class TestModel:
            pass

        model = TestModel()
        model.name = None
        assert resolve_model_name(model) == "TestModel"


class TestFindChangedParams:
    """Tests for _find_changed_params helper."""

    def test_no_changes_returns_empty(self):
        """Matching actual and defaults should return empty list."""
        defaults = {'a': 1, 'b': 'x'}
        assert _find_changed_params({'a': 1, 'b': 'x'}, defaults) == []

    def test_one_changed(self):
        """One changed param should return single-element list."""
        defaults = {'a': 1, 'b': 'x'}
        result = _find_changed_params({'a': 99, 'b': 'x'}, defaults)
        assert result == ['a=99']

    def test_multiple_changed(self):
        """Multiple changed params should return all of them."""
        defaults = {'a': 1, 'b': 'x'}
        result = _find_changed_params({'a': 99, 'b': 'y'}, defaults)
        assert result == ['a=99', 'b=y']

    def test_none_vs_value(self):
        """None default changed to a value should be detected."""
        defaults = {'a': None}
        result = _find_changed_params({'a': [1, 2]}, defaults)
        assert result == ['a=[1, 2]']


class TestEvalConfigValidation:
    """Tests for EvalConfig __post_init__ validation."""

    def test_defaults_succeed(self):
        """Default EvalConfig should be created without errors."""
        config = EvalConfig()
        assert config.shap_max_background == 50
        assert config.shap_max_explain == 50

    def test_shap_max_background_zero_raises(self):
        """shap_max_background=0 should raise ValueError when SHAP enabled."""
        with pytest.raises(ValueError, match="shap_max_background must be >= 1"):
            EvalConfig(include_shap=True, shap_max_background=0)

    def test_shap_max_background_negative_raises(self):
        """Negative shap_max_background should raise ValueError when SHAP enabled."""
        with pytest.raises(ValueError, match="shap_max_background must be >= 1"):
            EvalConfig(include_shap=True, shap_max_background=-5)

    def test_shap_max_explain_zero_raises(self):
        """shap_max_explain=0 should raise ValueError when SHAP enabled."""
        with pytest.raises(ValueError, match="shap_max_explain must be >= 1"):
            EvalConfig(include_shap=True, shap_max_explain=0)

    def test_shap_max_explain_negative_raises(self):
        """Negative shap_max_explain should raise ValueError when SHAP enabled."""
        with pytest.raises(ValueError, match="shap_max_explain must be >= 1"):
            EvalConfig(include_shap=True, shap_max_explain=-1)

    def test_invalid_shap_params_accepted_when_shap_disabled(self):
        """Invalid SHAP params should be accepted when include_shap=False."""
        config = EvalConfig(include_shap=False, shap_max_background=0, shap_max_explain=-1)
        assert config.shap_max_background == 0
        assert config.shap_max_explain == -1


class TestWarnUnusedShapParams:
    """Tests for _warn_unused_shap_params."""

    def test_shap_disabled_with_non_default_warns(self):
        """include_shap=False + non-default SHAP params should emit UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_unused_shap_params(
                include_shap=False,
                shap_max_background=100,
                shap_max_explain=50,
                stacklevel=2,
            )
            assert len(w) == 1
            assert "shap_max_background=100" in str(w[0].message)
            assert "include_shap=True" in str(w[0].message)

    def test_shap_enabled_no_warning(self):
        """include_shap=True should not emit a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_unused_shap_params(
                include_shap=True,
                shap_max_background=100,
                shap_max_explain=200,
                stacklevel=2,
            )
            assert len(w) == 0

    def test_shap_disabled_defaults_no_warning(self):
        """include_shap=False + default values should not emit a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_unused_shap_params(
                include_shap=False,
                shap_max_background=50,
                shap_max_explain=50,
                stacklevel=2,
            )
            assert len(w) == 0

    def test_shap_disabled_both_params_changed(self):
        """Both SHAP params changed should list both in warning message."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_unused_shap_params(
                include_shap=False,
                shap_max_background=100,
                shap_max_explain=200,
                stacklevel=2,
            )
            assert len(w) == 1
            msg = str(w[0].message)
            assert "shap_max_background=100" in msg
            assert "shap_max_explain=200" in msg


class TestWarnUnusedBootstrapParams:
    """Tests for _warn_unused_bootstrap_params."""

    def test_bootstrap_disabled_with_non_default_warns(self):
        """bootstrap=False + non-default params should emit UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_unused_bootstrap_params(
                bootstrap=False,
                n_bootstrap=500,
                bootstrap_ci=0.95,
                bootstrap_metrics=None,
                bootstrap_stratify=False,
                stacklevel=2,
            )
            assert len(w) == 1
            assert "n_bootstrap=500" in str(w[0].message)
            assert "bootstrap=True" in str(w[0].message)

    def test_bootstrap_enabled_no_warning(self):
        """bootstrap=True should not emit a warning regardless of params."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_unused_bootstrap_params(
                bootstrap=True,
                n_bootstrap=500,
                bootstrap_ci=0.99,
                bootstrap_metrics=['harrell'],
                bootstrap_stratify=True,
                stacklevel=2,
            )
            assert len(w) == 0

    def test_bootstrap_disabled_defaults_no_warning(self):
        """bootstrap=False + all default values should not emit a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_unused_bootstrap_params(
                bootstrap=False,
                n_bootstrap="auto",
                bootstrap_ci=0.95,
                bootstrap_metrics=None,
                bootstrap_stratify=False,
                stacklevel=2,
            )
            assert len(w) == 0


class TestFormatElapsed:
    """Tests for _format_elapsed() time formatting helper."""

    def test_zero_seconds(self):
        """0s should format as '0.0s'."""
        assert _format_elapsed(0) == "0.0s"

    def test_sub_second(self):
        """Fractional seconds should format with 1 decimal."""
        assert _format_elapsed(0.3) == "0.3s"

    def test_seconds(self):
        """Whole-ish seconds should format as 'N.Ns'."""
        assert _format_elapsed(3.21) == "3.2s"

    def test_just_under_minute(self):
        """59.9s stays in seconds format."""
        assert _format_elapsed(59.9) == "59.9s"

    def test_exact_minute(self):
        """60.0s switches to minute format."""
        assert _format_elapsed(60.0) == "1m00.0s"

    def test_minutes_and_seconds(self):
        """125.3s should format as '2m05.3s'."""
        assert _format_elapsed(125.3) == "2m05.3s"

    def test_large_time(self):
        """>10 minutes should still format correctly."""
        assert _format_elapsed(915.2) == "15m15.2s"

    def test_negative_clamped_to_zero(self):
        """Negative seconds should be clamped to 0.0s."""
        assert _format_elapsed(-1.5) == "0.0s"


@pytest.mark.integration
class TestEvalProgressOutput:
    """Tests for timing parentheticals in evaluation progress output."""

    def test_metrics_timing_shown(self, train_test_data, capsys):
        """Metrics phase should show timing in parentheses."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data
        model = get_model('coxph')
        model.fit(X_train, y_train)

        evaluate(
            model, X_test, y_test,
            X_train=X_train, y_train=y_train,
            n_importance_repeats=0,
            random_state=42,
            show_progress=True,
            n_jobs=1,
        )
        captured = capsys.readouterr()
        # Check metrics phase has timing parenthetical
        assert "done (" in captured.out
        assert "s)." in captured.out

    def test_no_output_when_progress_false(self, train_test_data, capsys):
        """No progress output when show_progress=False."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data
        model = get_model('coxph')
        model.fit(X_train, y_train)

        evaluate(
            model, X_test, y_test,
            n_importance_repeats=0,
            random_state=42,
            show_progress=False,
            n_jobs=1,
        )
        captured = capsys.readouterr()
        assert "Evaluating" not in captured.out
        assert "Computing permutation" not in captured.out

    def test_multimodel_njobs_gt1_shows_per_model_messages(self, train_test_data, capsys):
        """Multi-model with n_jobs>1 should still show per-model progress."""
        from mlsurv.models import get_model
        from mlsurv.evaluation import evaluate

        X_train, X_test, y_train, y_test = train_test_data

        coxph = get_model('coxph')
        coxph.fit(X_train, y_train)

        rsf = get_model('rsf', n_jobs=1)
        rsf.fit(X_train, y_train)

        models = {'coxph': coxph, 'rsf': rsf}

        evaluate(
            models, X_test, y_test,
            X_train=X_train, y_train=y_train,
            n_importance_repeats=0,
            random_state=42,
            show_progress=True,
            n_jobs=4,
        )
        captured = capsys.readouterr()
        # Per-model messages should be visible (previously hidden in parallel)
        assert "Evaluating coxph" in captured.out
        assert "Evaluating rsf" in captured.out
        assert "done (" in captured.out


class TestEvaluateInputValidation:
    """Tests for evaluate() input validation of n_importance_repeats."""

    def test_invalid_string_raises(self):
        """Non-'auto' string should raise ValueError."""
        from mlsurv.evaluation.evaluate import evaluate
        with pytest.raises(ValueError, match="n_importance_repeats must be"):
            evaluate(SimpleNamespace(), None, None, n_importance_repeats="invalid")

    def test_bool_raises_type_error(self):
        """Boolean n_importance_repeats should raise TypeError."""
        from mlsurv.evaluation.evaluate import evaluate
        with pytest.raises(TypeError, match="n_importance_repeats must be"):
            evaluate(SimpleNamespace(), None, None, n_importance_repeats=True)

    def test_float_raises_type_error(self):
        """Float n_importance_repeats should raise TypeError."""
        from mlsurv.evaluation.evaluate import evaluate
        with pytest.raises(TypeError, match="n_importance_repeats must be"):
            evaluate(SimpleNamespace(), None, None, n_importance_repeats=10.5)

    def test_negative_int_raises_value_error(self):
        """Negative int should raise ValueError."""
        from mlsurv.evaluation.evaluate import evaluate
        with pytest.raises(ValueError, match="n_importance_repeats must be >= 0"):
            evaluate(SimpleNamespace(), None, None, n_importance_repeats=-1)


# ---------------------------------------------------------------------------
# Phase helper tests
# ---------------------------------------------------------------------------


class TestShapExceptionsConstant:
    """Validate the _SHAP_EXCEPTIONS module constant."""

    def test_is_tuple(self):
        assert isinstance(_SHAP_EXCEPTIONS, tuple)

    def test_all_elements_are_exception_types(self):
        """Every element must be a BaseException subclass — no empty tuples."""
        for item in _SHAP_EXCEPTIONS:
            assert isinstance(item, type), f"Expected exception type, got {type(item).__name__}: {item!r}"
            assert issubclass(item, BaseException), f"{item} does not inherit from BaseException"

    def test_core_types_present(self):
        """The 7 core exception types must always be present."""
        core = {ImportError, ValueError, TypeError, np.linalg.LinAlgError,
                KeyError, NotImplementedError, RuntimeError}
        present = {t for t in _SHAP_EXCEPTIONS if isinstance(t, type)}
        assert core <= present


class TestRunImportancePhase:
    """Boundary tests for _run_importance_phase."""

    def test_zero_repeats_returns_empty(self):
        """Config with n_importance_repeats=0 returns empty DataFrames."""
        config = EvalConfig(n_importance_repeats=0)
        model = SimpleNamespace(predict=lambda X: np.zeros(len(X)))
        X_test = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        y_test = np.array([(True, 1.0), (False, 2.0)],
                          dtype=[('event', bool), ('time', float)])

        imp_df, iter_df, per_col_df = _run_importance_phase(
            model, X_test, y_test, config,
            model_name='test_model', importance_n_jobs=1,
        )
        assert 'model_type' in imp_df.columns
        assert len(imp_df) == 0, "Zero repeats should produce an empty DataFrame"
        assert 'model_type' in iter_df.columns
        assert per_col_df is None


class TestRunShapPhase:
    """Boundary tests for _run_shap_phase."""

    def test_shap_disabled_returns_none(self):
        """include_shap=False returns (None, None) without side effects."""
        config = EvalConfig(include_shap=False)
        shap_df, shap_per_col_df = _run_shap_phase(
            model=None, X_test=None, config=config,
            model_name='m', display_name='m',
        )
        assert shap_df is None
        assert shap_per_col_df is None

    @patch('mlsurv.evaluation.evaluate._compute_shap')
    def test_shap_reraises(self, mock_compute):
        """Exceptions from _compute_shap are always re-raised."""
        mock_compute.side_effect = ValueError("SHAP failed")
        config = EvalConfig(include_shap=True, show_progress=False)
        X_test = pd.DataFrame({'a': [1, 2]})

        with pytest.raises(ValueError, match="SHAP failed"):
            _run_shap_phase(
                model=MagicMock(), X_test=X_test, config=config,
                model_name='m', display_name='m',
            )


class TestRunInteractionsPhase:
    """Boundary tests for _run_interactions_phase."""

    def test_interactions_disabled_returns_none(self):
        """include_interactions=False returns all-None dict."""
        config = EvalConfig(include_interactions=False)
        result = _run_interactions_phase(
            model=None, X_test=None, importance_df=None, config=config,
            model_name='m', display_name='m',
        )
        assert result['interactions'] is None
        assert result['interaction_surfaces'] is None
        assert result['interaction_values_raw'] is None
        assert result['interaction_feature_names'] is None
        assert result['interaction_method'] is None

    @patch('mlsurv.evaluation.evaluate._extract_tree_estimator',
           return_value=(None, None))
    def test_nontree_shap_warns(self, mock_extract):
        """Non-tree model + method='shap' emits SuggestionWarning."""
        config = EvalConfig(
            include_interactions=True, interaction_method='shap',
            show_progress=False,
        )
        imp_df = pd.DataFrame({'feature': ['a'], 'perm_mean': [0.5]})
        model = MagicMock()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _run_interactions_phase(
                model=model, X_test=pd.DataFrame({'a': [1]}),
                importance_df=imp_df, config=config,
                model_name='m', display_name='m',
            )
        suggestion_warnings = [x for x in w if issubclass(x.category, SuggestionWarning)]
        assert len(suggestion_warnings) == 1
        assert "SHAP interaction values require a tree-based model" in str(suggestion_warnings[0].message)
        assert result['interactions'] is None

    def test_insufficient_features_warns(self):
        """Single feature emits SuggestionWarning about need >= 2."""
        config = EvalConfig(
            include_interactions=True, interaction_method='hstat',
            interaction_top_k=5, show_progress=False,
        )
        # Importance with only 1 feature
        imp_df = pd.DataFrame({
            'model_type': ['m'], 'feature': ['a'], 'perm_mean': [0.5],
        })
        X_test = pd.DataFrame({'a': [1, 2, 3]})
        model = MagicMock()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _run_interactions_phase(
                model=model, X_test=X_test, importance_df=imp_df,
                config=config, model_name='m', display_name='m',
            )
        insufficient_warnings = [
            x for x in w
            if issubclass(x.category, SuggestionWarning)
            and "feature(s) available" in str(x.message)
        ]
        assert len(insufficient_warnings) == 1
        assert "only 1 feature(s) available" in str(insufficient_warnings[0].message)
        assert result['interactions'] is None

    def test_interaction_warning_stacklevel(self):
        """Warnings from _run_interactions_phase point to caller, not helper."""
        config = EvalConfig(
            include_interactions=True, interaction_method='hstat',
            interaction_top_k=5, show_progress=False,
        )
        imp_df = pd.DataFrame({
            'model_type': ['m'], 'feature': ['a'], 'perm_mean': [0.5],
        })
        X_test = pd.DataFrame({'a': [1, 2, 3]})
        model = MagicMock()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # _warn_stacklevel=2 points to the caller of the helper — i.e., this test
            _run_interactions_phase(
                model=model, X_test=X_test, importance_df=imp_df,
                config=config, model_name='m', display_name='m',
                _warn_stacklevel=2,
            )
        # Filter for the insufficient-features warning emitted by the helper
        insufficient_warnings = [
            x for x in w
            if issubclass(x.category, SuggestionWarning)
            and "feature(s) available" in str(x.message)
        ]
        assert len(insufficient_warnings) == 1
        # Warning filename should be this test file, not evaluate.py
        assert 'test_evaluation_helpers' in insufficient_warnings[0].filename

    @patch('mlsurv.evaluation.evaluate._is_linear_model', return_value=True)
    def test_linear_model_hstat_skip(self, mock_linear):
        """Linear model + method='hstat' skips interactions (H-stat always zero)."""
        config = EvalConfig(
            include_interactions=True, interaction_method='hstat',
            show_progress=False,
        )
        result = _run_interactions_phase(
            model=MagicMock(), X_test=pd.DataFrame({'a': [1]}),
            importance_df=pd.DataFrame(), config=config,
            model_name='m', display_name='m',
        )
        assert result['interactions'] is None
        assert result['interaction_method'] is None

    @patch('mlsurv.evaluation.evaluate._is_linear_model', return_value=False)
    @patch('mlsurv.evaluation.evaluate._select_interaction_features',
           return_value=['a', 'b'])
    @patch('mlsurv.evaluation.evaluate._compute_h_statistic')
    def test_interactions_reraises(self, mock_hstat, mock_select, mock_linear):
        """Exceptions from _compute_h_statistic are always re-raised."""
        mock_hstat.side_effect = ValueError("H-stat failed")
        config = EvalConfig(
            include_interactions=True, interaction_method='hstat',
            show_progress=False,
        )
        imp_df = pd.DataFrame({
            'model_type': ['m', 'm'], 'feature': ['a', 'b'], 'perm_mean': [0.5, 0.3],
        })
        X_test = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

        with pytest.raises(ValueError, match="H-stat failed"):
            _run_interactions_phase(
                model=MagicMock(), X_test=X_test, importance_df=imp_df,
                config=config, model_name='m', display_name='m',
            )
