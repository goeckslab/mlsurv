"""Tests for training-setup-hardening fixes."""

import pickle
import warnings
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from mlsurv._warnings import MethodologicalWarning, SuggestionWarning
from mlsurv.tuning.exceptions import NoTunerError
from mlsurv.tuning import get_tuner
from mlsurv.learner import SurvivalLearner
from mlsurv.results.containers import ModelResult


def _fake_model_result(name="coxph"):
    return ModelResult(
        name=name, model=None, cv_metric_scores=np.array([0.7]),
        cv_metric_mean=0.7, cv_metric_std=0.0,
    )
from mlsurv.learner._cv_training import (
    _is_degenerate_error, _DEGENERATE_KEYWORDS, _train_cv_fold,
)


class TestNoTunerError:
    """Tests for NoTunerError."""

    def test_is_valueerror_subclass(self):
        assert issubclass(NoTunerError, ValueError)

    def test_caught_by_except_valueerror(self):
        """Backward compat: except ValueError still catches NoTunerError."""
        with pytest.raises(ValueError):
            raise NoTunerError("fake_model")

    def test_carries_model_name(self):
        err = NoTunerError("fake_model", available=["rsf", "coxph"])
        assert err.model_name == "fake_model"
        assert err.available == ["rsf", "coxph"]

    def test_message_includes_model_name(self):
        err = NoTunerError("fake_model")
        assert "fake_model" in str(err)

    def test_get_tuner_raises_for_unknown_model(self):
        with pytest.raises(NoTunerError) as exc_info:
            get_tuner("nonexistent_model")
        assert exc_info.value.model_name == "nonexistent_model"
        assert len(exc_info.value.available) > 0

    def test_pickle_roundtrip(self):
        err = NoTunerError("fake_model", available=["rsf", "coxph"])
        restored = pickle.loads(pickle.dumps(err))
        assert restored.model_name == "fake_model"
        assert restored.available == ["rsf", "coxph"]
        assert str(restored) == str(err)


class TestNoTunerFallback:
    """Tests for NoTunerError fallback in tune/train paths."""

    def test_tune_and_train_warns_on_no_tuner(self, small_data):
        """_tune_and_train falls back to defaults when no tuner exists."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        with patch.object(
            learner, "_create_tuner_for_model", side_effect=NoTunerError("fake")
        ):
            mock_result = MagicMock()
            with patch.object(
                learner, "_train_model_internal", return_value=mock_result
            ) as train_mock:
                with pytest.warns(SuggestionWarning, match="No tuner registered"):
                    result = learner._tune_and_train("fake", n_trials=5, search_spaces=None)
                train_mock.assert_called_once_with("fake", _preprocessing=None)
                assert result is mock_result

    def test_tune_single_model_warns_on_no_tuner(self, small_data):
        """_tune_single_model returns baseline when no tuner exists."""
        from mlsurv.learner._tuning import TuneConfig
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        with patch.object(
            learner, "_create_tuner_for_model", side_effect=NoTunerError("fake")
        ):
            mock_result = MagicMock()
            with patch.object(
                learner, "_train_model_internal", return_value=mock_result
            ) as train_mock:
                with pytest.warns(SuggestionWarning, match="No tuner registered"):
                    result = learner._tune_single_model("fake", TuneConfig())
                # No existing baseline → trains with defaults (not marked tuned)
                train_mock.assert_called_once_with("fake", _preprocessing=None)
                assert result is mock_result


class TestDegenerateFoldWarning:
    """Tests for degenerate fold detection."""

    def test_is_degenerate_error_detects_keywords(self):
        """_is_degenerate_error matches known degenerate error messages."""
        assert _is_degenerate_error(ValueError("All times are identical"))
        assert _is_degenerate_error(ValueError("Degenerate survival data"))
        assert _is_degenerate_error(ValueError("t_min equals t_max"))
        assert _is_degenerate_error(ValueError("Empty array after filtering"))

    def test_is_degenerate_error_rejects_non_degenerate(self):
        """_is_degenerate_error does not match unrelated errors."""
        assert not _is_degenerate_error(ValueError("negative value encountered"))
        assert not _is_degenerate_error(ValueError("matrix is singular"))

    def test_keywords_constant_matches_function(self):
        """_DEGENERATE_KEYWORDS is used by _is_degenerate_error."""
        # Ensure the constant is non-empty and the function uses it
        assert len(_DEGENERATE_KEYWORDS) > 0
        for kw in _DEGENERATE_KEYWORDS:
            assert _is_degenerate_error(ValueError(kw))

    def test_aggregate_warns_on_degenerate_folds(self, small_data):
        """_aggregate_fold_results warns when folds have degenerate data."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        # Build fold results with degenerate flag
        fold_results = [
            {
                "exception": None, "score": 0.6, "harrell": 0.6,
                "ipcw": float("nan"), "ipcw_failed": True,
                "ipcw_degenerate": True,
                "integrated_brier": None,
            },
            {
                "exception": None, "score": 0.7, "harrell": 0.7,
                "ipcw": 0.65, "ipcw_failed": False,
                "integrated_brier": None,
            },
        ]
        with pytest.warns(MethodologicalWarning, match="Degenerate survival data detected"):
            learner._aggregate_fold_results(fold_results, 2)


class TestAggregateFoldScoresAlignment:
    """Tests for _aggregate_fold_scores metric length alignment."""

    def test_failed_fold_appends_nan_to_optional_metrics(self, small_data):
        """Failed fold should append NaN to integrated_brier/ctd if present."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        fold_results = [
            {
                'exception': None, 'score': 0.7, 'harrell': 0.7,
                'ipcw': 0.65, 'ipcw_failed': False,
                'integrated_brier': 0.2, 'ctd': 0.68,
                'warnings': [],
            },
            {
                'exception': 'ValueError: test error', 'score': np.nan,
                'harrell': np.nan, 'ipcw': np.nan, 'ipcw_failed': False,
                'integrated_brier': None, 'ctd': None,
                'warnings': [],
            },
            {
                'exception': None, 'score': 0.72, 'harrell': 0.72,
                'ipcw': 0.68, 'ipcw_failed': False,
                'integrated_brier': 0.19, 'ctd': 0.70,
                'warnings': [],
            },
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores, cv_metrics = learner._aggregate_fold_results(fold_results, 3)

        assert len(scores) == 3
        assert len(cv_metrics['harrell']) == 3
        assert len(cv_metrics['ipcw']) == 3
        assert len(cv_metrics['integrated_brier']) == 3
        assert len(cv_metrics['ctd']) == 3
        assert np.isnan(cv_metrics['integrated_brier'][1])
        assert np.isnan(cv_metrics['ctd'][1])

    def test_optional_metric_none_appends_nan(self, small_data):
        """Fold where optional metric is None should get NaN, not be skipped."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        fold_results = [
            {
                'exception': None, 'score': 0.7, 'harrell': 0.7,
                'ipcw': 0.65, 'ipcw_failed': False,
                'integrated_brier': 0.2, 'ctd': 0.68,
                'warnings': [],
            },
            {
                'exception': None, 'score': 0.68, 'harrell': 0.68,
                'ipcw': 0.63, 'ipcw_failed': False,
                'integrated_brier': None, 'ctd': None,
                'warnings': [],
            },
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores, cv_metrics = learner._aggregate_fold_results(fold_results, 2)

        assert len(scores) == 2
        assert len(cv_metrics['integrated_brier']) == 2
        assert len(cv_metrics['ctd']) == 2
        assert np.isnan(cv_metrics['integrated_brier'][1])
        assert np.isnan(cv_metrics['ctd'][1])

    def test_late_optional_metric_backfills_nan(self, small_data):
        """If optional metric first appears on fold N>0, earlier folds get NaN."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        fold_results = [
            {
                'exception': None, 'score': 0.7, 'harrell': 0.7,
                'ipcw': 0.65, 'ipcw_failed': False,
                'integrated_brier': None, 'ctd': None,
                'warnings': [],
            },
            {
                'exception': None, 'score': 0.72, 'harrell': 0.72,
                'ipcw': 0.68, 'ipcw_failed': False,
                'integrated_brier': 0.2, 'ctd': 0.70,
                'warnings': [],
            },
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores, cv_metrics = learner._aggregate_fold_results(fold_results, 2)

        assert len(scores) == 2
        assert len(cv_metrics['integrated_brier']) == 2
        assert len(cv_metrics['ctd']) == 2
        # First fold (before key existed) should be NaN via backfill
        assert np.isnan(cv_metrics['integrated_brier'][0])
        assert np.isnan(cv_metrics['ctd'][0])
        # Second fold should have the real value
        assert cv_metrics['integrated_brier'][1] == 0.2
        assert cv_metrics['ctd'][1] == 0.70


class TestAutoSetupWarning:
    """Tests for auto-setup warning."""

    def test_ensure_setup_warns_when_not_setup(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        with pytest.warns(SuggestionWarning, match="setup\\(\\) was not called explicitly"):
            learner._ensure_setup()

    def test_ensure_setup_no_warning_after_explicit_setup(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should not warn — setup was already called
            learner._ensure_setup()


class TestDoubleSetupGuard:
    """Tests for double setup() warning."""

    def test_first_setup_does_not_warn_about_double_setup(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner.setup(verbose=False)
        double_setup_warnings = [
            x for x in w if "already been called" in str(x.message)
        ]
        assert len(double_setup_warnings) == 0

    def test_second_setup_warns(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        with pytest.warns(SuggestionWarning, match="setup\\(\\) has already been called"):
            learner.setup(verbose=False)

    def test_double_setup_clears_all_state(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        # Set dummy values for all result containers. The per-model risk-group
        # tables ride on the ModelResult, so seed a real one.
        mr = _fake_model_result("coxph")
        mr.rmst_test = "old_rmst"
        mr.logrank_test = "old_logrank"
        learner._state.models["coxph"] = mr
        learner._state.validation_results["ext"] = "old_validation"
        # Split/model-dependent results with no per-model retrain guard
        # and split-scoped warning-dedup state.
        learner._state.benchmark_result = "old_benchmark"
        learner._state.kang_comparison_result = "old_kang"
        learner._state.ipcw_support_flag = {"tau_used": 1.0}
        learner._state.ipcw_truncation_warned["main"] = True
        learner._state.timeseries_ci_warned = True
        # match= pins the warning's field inventory, so silently dropping
        # a category from the message fails here.
        with pytest.warns(SuggestionWarning,
                          match="benchmark, and statistical-test"):
            learner.setup(verbose=False)
        assert len(learner._state.models) == 0
        assert len(learner._state.validation_results) == 0
        assert learner._state.benchmark_result is None
        assert learner._state.kang_comparison_result is None
        # Risk-group tables died with the cleared ModelResult.
        assert learner._state.rmst_test_results == {}
        assert learner._state.logrank_test_results == {}
        assert learner._state.ipcw_support_flag is None
        assert learner._state.ipcw_truncation_warned == {}
        assert learner._state.timeseries_ci_warned is False


class TestFailedResetupPreservesState:
    """A re-setup() that fails validation must not clear results.

    Preset names are resolved in _resolve_preprocessing, which runs
    BEFORE _reset_state_if_needed — a typo'd preset raises without
    destroying trained models or downstream results.
    """

    @staticmethod
    def _seeded_learner(small_data):
        """Set-up learner with placeholders in a representative set of cleared fields."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        # Per-model derived state rides on the ModelResult; seed a real
        # one carrying sentinels for every riding field.
        mr = _fake_model_result("fake")
        mr.pimp_importance_test = "old_pimp"
        mr.bootstrap_importance_test = "old_boot"
        mr.rmst_test = "old_rmst"
        mr.logrank_test = "old_logrank"
        mr.calibrator = "old_calibrator"
        learner._state.models["fake"] = mr
        learner._state.validation_results["ext"] = "old_validation"
        learner._state.benchmark_result = "old_benchmark"
        learner._state.kang_comparison_result = "old_kang"
        return learner

    @staticmethod
    def _assert_results_intact(learner):
        assert learner._state.models["fake"].name == "fake"
        assert learner._state.pimp_importance_test_results["fake"] == "old_pimp"
        assert (learner._state.bootstrap_importance_test_results["fake"]
                == "old_boot")
        assert learner._state.rmst_test_results["fake"] == "old_rmst"
        assert learner._state.logrank_test_results["fake"] == "old_logrank"
        assert learner._state.models["fake"].calibrator == "old_calibrator"
        assert learner._state.validation_results["ext"] == "old_validation"
        assert learner._state.benchmark_result == "old_benchmark"
        assert learner._state.kang_comparison_result == "old_kang"
        assert learner._state.is_setup is True

    @pytest.mark.parametrize("kwargs, match", [
        ({"imputer": "typo"}, "Unknown imputer preset"),
        ({"scaler": "typo"}, "Unknown scaler preset"),
        ({"feature_selector": "nonsense"}, "Unknown selector preset"),
        # List form resolves through a separate branch of _resolve_component
        ({"feature_selector": ["nonsense"]}, "Unknown selector preset"),
        ({"preprocessing": {"coxph": {"scaler": "typo"}}},
         r"preprocessing\['coxph'\]\['scaler'\]"),
    ])
    def test_typo_resetup_raises_and_preserves_results(
            self, small_data, kwargs, match):
        learner = self._seeded_learner(small_data)
        with pytest.raises(ValueError, match=match):
            learner.setup(verbose=False, **kwargs)
        self._assert_results_intact(learner)

    def test_failed_resetup_emits_no_resetup_warning(self, small_data):
        """Reset never runs, so the 'already been called' warning must not fire."""
        learner = self._seeded_learner(small_data)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="Unknown selector preset"):
                learner.setup(verbose=False, feature_selector="nonsense")
        resetup_warnings = [
            x for x in w if "already been called" in str(x.message)
        ]
        assert len(resetup_warnings) == 0

    def test_failed_resetup_no_partial_assignment(self, small_data):
        """Valid components + one bad preset must not swap any old component."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False, feature_selector="topk",
                      preprocessing={"coxph": {"scaler": "robust"}})
        old_imputer = learner._state.imputer
        old_selector = learner._state.feature_selector
        old_model_preprocessing = learner._state.model_preprocessing
        with pytest.raises(ValueError, match="Unknown scaler preset"):
            learner.setup(verbose=False, imputer="median", scaler="typo")
        assert learner._state.imputer is old_imputer
        assert learner._state.feature_selector is old_selector
        assert learner._state.model_preprocessing is old_model_preprocessing
        assert "coxph" in learner._state.model_preprocessing

    def test_failed_resetup_preserves_config_scalars(self, small_data):
        """scoring/random_state/cv_ci/test_fraction must survive a failed re-setup."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False, scoring="ipcw", random_state=7,
                      cv_ci=0.9, test_fraction=0.25)
        with pytest.raises(ValueError, match="Unknown selector preset"):
            learner.setup(verbose=False, feature_selector="nonsense")
        assert learner.scoring == "ipcw"
        assert learner.random_state == 7
        assert learner.cv_ci == 0.9
        assert learner.test_fraction == 0.25

    def test_first_setup_typo_still_raises(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        with pytest.raises(ValueError, match="Unknown selector preset"):
            learner.setup(verbose=False, feature_selector="nonsense")
        assert learner._state.is_setup is False

    def test_successful_resetup_still_clears(self, small_data):
        """A valid re-setup still clears results (only failed validation preserves them)."""
        learner = self._seeded_learner(small_data)
        with pytest.warns(SuggestionWarning, match="already been called"):
            learner.setup(verbose=False)
        assert len(learner._state.models) == 0
        assert learner._state.benchmark_result is None


class TestFailedSetupClearsIsSetup:
    """A setup() that fails AFTER _reset_state_if_needed() must clear is_setup.

    _configure_cv splits the data and builds CV splitters in place and can raise
    (bad stratify/groups column, cv_folds > events, unknown cv_strategy). When it
    does on a re-setup, the old results are already cleared and _finalize_setup
    never runs, so is_setup would otherwise stay True from the prior setup and the
    learner would silently train on mixed old/new state. The fix flips is_setup to
    False right after the reset; _finalize_setup sets it back to True only on
    success.

    Unlike TestFailedResetupPreservesState (preprocessing typos that raise BEFORE
    the reset and preserve everything), these failures are post-reset.
    """

    def test_resetup_too_many_cv_folds_clears_is_setup(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        with pytest.raises(ValueError):
            learner.setup(verbose=False, cv_folds=1000)  # > training events
        assert learner._state.is_setup is False

    def test_resetup_unknown_cv_strategy_clears_is_setup(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        with pytest.raises(ValueError):
            learner.setup(verbose=False, cv_strategy="stratified")  # typo
        assert learner._state.is_setup is False

    def test_resetup_bad_stratify_column_clears_is_setup(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        with pytest.raises(KeyError):
            learner.setup(verbose=False, stratify="nonexistent_column")
        assert learner._state.is_setup is False

    def test_successful_resetup_leaves_is_setup_true(self, small_data):
        """No regression: a valid re-setup ends fully configured."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        with pytest.warns(SuggestionWarning, match="already been called"):
            learner.setup(verbose=False)
        assert learner._state.is_setup is True

    def test_first_setup_bad_cv_strategy_leaves_is_setup_false(self, small_data):
        """First-time post-reset failure must not flip is_setup to True."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        with pytest.raises(ValueError):
            learner.setup(verbose=False, cv_strategy="stratified")
        assert learner._state.is_setup is False

    def test_failed_resetup_makes_ensure_setup_distrust_learner(self, small_data):
        """After a failed re-setup, _ensure_setup() forces a fresh auto-setup."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        with pytest.raises(ValueError):
            learner.setup(verbose=False, cv_strategy="stratified")
        assert learner._state.is_setup is False
        with pytest.warns(SuggestionWarning, match="setup\\(\\) was not called"):
            learner._ensure_setup()
        # auto-setup completed a fresh, coherent configuration
        assert learner._state.is_setup is True


# ---------------------------------------------------------------------------
# Worker warning capture tests
# ---------------------------------------------------------------------------

class TestWorkerWarningCapture:
    """Tests for warning capture in joblib worker processes."""

    def test_fold_result_has_warnings_key(self, small_data):
        """_train_cv_fold result should contain 'warnings' list."""
        X, y = small_data
        n = len(X)
        train_idx = np.arange(n // 2)
        val_idx = np.arange(n // 2, n)

        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from mlsurv.models import CoxPHModel
        from mlsurv.evaluation import score_survival_model

        pipeline = Pipeline([
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler()),
            ('model', CoxPHModel()),
        ])

        result = _train_cv_fold(
            fold_idx=0,
            train_idx=train_idx,
            val_idx=val_idx,
            X_train=X,
            y_train=y,
            pipeline=pipeline,
            effective_scoring='harrell',
            score_survival_model_fn=score_survival_model,
        )
        assert 'warnings' in result
        assert isinstance(result['warnings'], list)

    def test_fold_captures_warnings_as_tuples(self, small_data):
        """Warnings emitted during fold execution are captured as 4-tuples."""
        X, y = small_data
        n = len(X)
        train_idx = np.arange(n // 2)
        val_idx = np.arange(n // 2, n)

        from sklearn.pipeline import Pipeline

        # Create a pipeline whose fit emits a warning
        mock_pipeline = MagicMock(spec=Pipeline)
        def fit_with_warning(*args, **kwargs):
            warnings.warn("test fold warning")
        mock_pipeline.fit = fit_with_warning
        mock_pipeline.predict.return_value = np.zeros(len(val_idx))

        with patch('mlsurv.learner._cv_training.clone', return_value=mock_pipeline):
            with patch('mlsurv.learner._cv_training.adjust_model_njobs',
                       return_value=mock_pipeline):
                result = _train_cv_fold(
                    fold_idx=0,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    X_train=X,
                    y_train=y,
                    pipeline=mock_pipeline,
                    effective_scoring='harrell',
                    score_survival_model_fn=lambda *a, **kw: 0.7,
                )

        # Should have captured the warning
        warn_tuples = [w for w in result['warnings'] if 'test fold warning' in w[1]]
        assert len(warn_tuples) >= 1
        cat_name, msg, fname, lineno = warn_tuples[0]
        assert cat_name == 'UserWarning'
        assert 'test fold warning' in msg

    def test_aggregate_reemits_worker_warnings(self, small_data):
        """_aggregate_fold_results should re-emit warnings with original category."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        fold_results = [
            {
                'exception': None, 'score': 0.6, 'harrell': 0.6,
                'ipcw': 0.6, 'ipcw_failed': False,
                'integrated_brier': None,
                'warnings': [
                    ('MethodologicalWarning', 'some model warning', 'model.py', 42),
                ],
            },
            {
                'exception': None, 'score': 0.7, 'harrell': 0.7,
                'ipcw': 0.7, 'ipcw_failed': False,
                'integrated_brier': None,
                'warnings': [],
            },
        ]

        with pytest.warns(MethodologicalWarning, match="some model warning"):
            learner._aggregate_fold_results(fold_results, 2)

    def test_aggregate_deduplicates_warnings(self, small_data):
        """Same warning from N folds should appear once with fold count."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        shared_warning = ('MethodologicalWarning', 'convergence warning', 'solver.py', 100)
        fold_results = [
            {
                'exception': None, 'score': 0.6, 'harrell': 0.6,
                'ipcw': 0.6, 'ipcw_failed': False,
                'integrated_brier': None,
                'warnings': [shared_warning],
            },
            {
                'exception': None, 'score': 0.65, 'harrell': 0.65,
                'ipcw': 0.65, 'ipcw_failed': False,
                'integrated_brier': None,
                'warnings': [shared_warning],
            },
            {
                'exception': None, 'score': 0.7, 'harrell': 0.7,
                'ipcw': 0.7, 'ipcw_failed': False,
                'integrated_brier': None,
                'warnings': [shared_warning],
            },
        ]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner._aggregate_fold_results(fold_results, 3)

        convergence_warnings = [
            w for w in caught if 'convergence warning' in str(w.message)
        ]
        assert len(convergence_warnings) == 1
        assert 'across 3 CV folds' in str(convergence_warnings[0].message)
        assert convergence_warnings[0].category is MethodologicalWarning

    def test_dedup_format_uses_occurrences_phrasing(self, small_data):
        """Dedup format reads as cardinality, not a fraction.

        Previously printed `(occurred in 40/5 CV folds)` which scans as a
        malformed fraction.  Verify the new wording explicitly separates
        occurrence count from fold count.
        """
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        shared_warning = ('UserWarning', 'lifelines noise', 'lifelines.py', 10)
        fold_results = [
            {
                'exception': None, 'score': 0.7, 'harrell': 0.7,
                'ipcw': 0.7, 'ipcw_failed': False,
                'integrated_brier': None,
                # Two occurrences in the single fold result -> count=2
                'warnings': [shared_warning, shared_warning],
            },
        ]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner._aggregate_fold_results(fold_results, 5)

        matched = [w for w in caught if 'lifelines noise' in str(w.message)]
        assert len(matched) == 1
        text = str(matched[0].message)
        assert '2 occurrences across 5 CV folds' in text
        # Old format must not return.
        assert '2/5 CV folds' not in text

    def test_dedup_format_singular_is_plain_message(self, small_data):
        """count==1 must NOT append a parenthetical suffix."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        fold_results = [
            {
                'exception': None, 'score': 0.7, 'harrell': 0.7,
                'ipcw': 0.7, 'ipcw_failed': False,
                'integrated_brier': None,
                'warnings': [('UserWarning', 'singleton', 'a.py', 1)],
            },
        ]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner._aggregate_fold_results(fold_results, 5)

        matched = [w for w in caught if 'singleton' in str(w.message)]
        assert len(matched) == 1
        # Bare message; no occurrence/fold suffix.
        assert str(matched[0].message) == 'singleton'

    def test_aggregate_unknown_category_falls_back_to_user_warning(self, small_data):
        """Unknown category string falls back to UserWarning."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        fold_results = [
            {
                'exception': None, 'score': 0.6, 'harrell': 0.6,
                'ipcw': 0.6, 'ipcw_failed': False,
                'integrated_brier': None,
                'warnings': [
                    ('SomeUnknownWarning', 'unknown category test', 'unknown.py', 1),
                ],
            },
        ]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner._aggregate_fold_results(fold_results, 1)

        unknown_warnings = [
            w for w in caught if 'unknown category test' in str(w.message)
        ]
        assert len(unknown_warnings) == 1
        assert unknown_warnings[0].category is UserWarning

    def test_aggregate_handles_missing_warnings_key(self, small_data):
        """Backward compat: fold results without 'warnings' key should work."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        fold_results = [
            {
                'exception': None, 'score': 0.6, 'harrell': 0.6,
                'ipcw': 0.6, 'ipcw_failed': False,
                'integrated_brier': None,
                # No 'warnings' key — old-style result
            },
        ]

        # Should not raise
        scores, cv_metrics = learner._aggregate_fold_results(fold_results, 1)
        assert scores == [0.6]

    def test_fold_preserves_warnings_on_exception(self, small_data):
        """Warnings emitted before a fold exception are still captured."""
        X, y = small_data
        n = len(X)
        train_idx = np.arange(n // 2)
        val_idx = np.arange(n // 2, n)

        from sklearn.pipeline import Pipeline

        # Pipeline that emits a warning during fit, then scoring raises
        mock_pipeline = MagicMock(spec=Pipeline)
        def fit_with_warning(*args, **kwargs):
            warnings.warn("pre-crash warning")
        mock_pipeline.fit = fit_with_warning

        def score_that_raises(*args, **kwargs):
            raise ValueError("scoring exploded")

        with patch('mlsurv.learner._cv_training.clone', return_value=mock_pipeline):
            with patch('mlsurv.learner._cv_training.adjust_model_njobs',
                       return_value=mock_pipeline):
                result = _train_cv_fold(
                    fold_idx=0,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    X_train=X,
                    y_train=y,
                    pipeline=mock_pipeline,
                    effective_scoring='harrell',
                    score_survival_model_fn=score_that_raises,
                )

        # Fold should have failed
        assert result['exception'] is not None
        assert 'scoring exploded' in result['exception']
        # But warnings emitted before the crash should still be captured
        warn_tuples = [w for w in result['warnings'] if 'pre-crash warning' in w[1]]
        assert len(warn_tuples) >= 1
