"""Tests for eval_times display in report sections."""

import warnings
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import InternalWarning
from mlsurv.reports.sections.evaluation._section import EvaluationSection


def _make_learner(user_eval_times=None, time_unit=None):
    """Create a minimal mock learner for EvaluationSection."""
    learner = MagicMock()
    learner._state.user_eval_times = user_eval_times
    learner._state.time_unit = time_unit
    learner._state.outcome_name = "event"
    learner._state.models = {
        "coxph": MagicMock(
            cv_metric_mean=0.75,
            cv_metric_std=0.02,
            is_tuned=False,
            evaluation=None,
            bootstrap=None,
        )
    }
    learner._state.subpopulation_results = None
    learner._state.feature_groups = None
    learner._state.group_colors = None
    learner.scoring = "harrell_c"
    return learner


def _make_eval_result(eval_times):
    """Create a minimal eval_result with auc_t containing given eval_times."""
    result = MagicMock()
    result.auc_t = pd.DataFrame([
        {"model_type": "coxph", "eval_time": t, "auc": 0.75}
        for t in eval_times
    ])
    result.concordance = pd.DataFrame(
        [{"model_type": "coxph", "c_index_train": 0.8, "c_index_test": 0.75}]
    )
    result.importance = pd.DataFrame()
    result.test_set = None
    return result


class TestEvalTimesNoteInEvaluationSection:
    """Eval_times note appears in the Evaluation section intro."""

    def test_auto_eval_times_note(self):
        """Auto-generated eval_times show source description."""
        learner = _make_learner(user_eval_times=None, time_unit="months")
        eval_times = [1.0, 3.0, 6.0, 12.0, 24.0]
        eval_result = _make_eval_result(eval_times)
        section = EvaluationSection(learner, eval_result=eval_result)
        result = section.generate()
        md = result["markdown"]
        assert "Evaluation Timepoints" in md
        assert "auto-generated" in md
        assert "5 points" in md
        assert "months" in md

    def test_user_specified_eval_times_note(self):
        """User-specified eval_times show 'user-specified' source."""
        user_times = np.array([1.0, 5.0, 10.0])
        learner = _make_learner(user_eval_times=user_times)
        eval_result = _make_eval_result([1.0, 5.0, 10.0])
        section = EvaluationSection(learner, eval_result=eval_result)
        result = section.generate()
        md = result["markdown"]
        assert "Evaluation Timepoints" in md
        assert "user-specified" in md
        assert "3 points" in md

    def test_no_time_unit_omits_unit(self):
        """When time_unit is None, no unit label appears."""
        learner = _make_learner(user_eval_times=None, time_unit=None)
        eval_result = _make_eval_result([1.0, 2.0, 3.0])
        section = EvaluationSection(learner, eval_result=eval_result)
        result = section.generate()
        md = result["markdown"]
        assert "Evaluation Timepoints" in md
        assert "months" not in md

    def test_no_auc_t_skips_note(self):
        """When eval_result has no auc_t, eval_times note is skipped."""
        learner = _make_learner()
        eval_result = MagicMock()
        eval_result.auc_t = None
        eval_result.concordance = pd.DataFrame(
            [{"model_type": "coxph", "c_index_train": 0.8, "c_index_test": 0.75}]
        )
        eval_result.importance = pd.DataFrame()
        eval_result.test_set = None
        section = EvaluationSection(learner, eval_result=eval_result)
        result = section.generate()
        md = result["markdown"]
        assert "Evaluation Timepoints" not in md

    def test_eval_times_values_displayed(self):
        """The actual timepoint values appear in the note."""
        learner = _make_learner(user_eval_times=None, time_unit="years")
        eval_result = _make_eval_result([0.5, 1.0, 2.0])
        section = EvaluationSection(learner, eval_result=eval_result)
        result = section.generate()
        md = result["markdown"]
        assert "0.5" in md
        assert "2" in md


# -----------------------------------------------------------------------
# format_eval_time unit tests
# -----------------------------------------------------------------------

from mlsurv.reports._helpers import (
    eval_grid_with_percentile_fallback,
    extract_eval_times_from_auc_t,
    format_eval_time,
)


class TestFormatEvalTime:
    """Unit tests for format_eval_time helper."""

    def test_integer_valued_float(self):
        assert format_eval_time(12.0) == "12"

    def test_true_float(self):
        assert format_eval_time(2.5) == "2.5"

    def test_zero(self):
        assert format_eval_time(0.0) == "0"

    def test_small_float(self):
        assert format_eval_time(0.3) == "0.3"


class TestExtractEvalTimesFromAucT:
    """Unit tests for extract_eval_times_from_auc_t helper."""

    def test_none_returns_empty(self):
        assert extract_eval_times_from_auc_t(None) == []

    def test_empty_df_returns_empty(self):
        assert extract_eval_times_from_auc_t(pd.DataFrame()) == []

    def test_missing_column_returns_empty(self):
        df = pd.DataFrame({"auc": [0.75]})
        assert extract_eval_times_from_auc_t(df) == []

    def test_filters_nan_and_inf(self):
        df = pd.DataFrame({"eval_time": [1.0, np.nan, np.inf, 5.0]})
        assert extract_eval_times_from_auc_t(df) == [1.0, 5.0]

    def test_returns_sorted_unique(self):
        df = pd.DataFrame({"eval_time": [5.0, 1.0, 5.0, 3.0]})
        assert extract_eval_times_from_auc_t(df) == [1.0, 3.0, 5.0]


# -----------------------------------------------------------------------
# Environment section tests
# -----------------------------------------------------------------------

from mlsurv.reports.sections.environment import EnvironmentSection, _CONFIG_GROUPS


class TestEvalTimesInEnvironmentSection:
    """eval_times appear in the Environment section's config dump."""

    def test_config_groups_includes_eval_times_keys(self):
        """_CONFIG_GROUPS['Evaluation'] includes eval_times keys."""
        eval_keys = _CONFIG_GROUPS["Evaluation"]
        assert "eval_times" in eval_keys
        assert "eval_times_source" in eval_keys
        assert "n_eval_times" in eval_keys

    def test_auto_eval_times_in_config(self, small_data):
        """Auto-generated eval_times appear in config with 'auto' source."""
        X, y = small_data
        from mlsurv import SurvivalLearner

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train("coxph")
        learner.evaluate()

        section = EnvironmentSection(learner)
        section.generate()

        config_df = section.data["config"]
        keys = config_df["Key"].tolist()
        assert "eval_times_source" in keys

        source_row = config_df[config_df["Key"] == "eval_times_source"]
        assert "auto" in source_row["Value"].iloc[0].lower()

        assert "eval_times" in keys
        category = config_df[config_df["Key"] == "eval_times"]["Category"].iloc[0]
        assert category == "Evaluation"

    def test_user_eval_times_in_config(self, small_data):
        """User-specified eval_times show 'user-specified' source."""
        X, y = small_data
        from mlsurv import SurvivalLearner

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train("coxph")
        learner.evaluate(eval_times=[1.0, 5.0, 10.0])

        section = EnvironmentSection(learner)
        section.generate()

        config_df = section.data["config"]
        source_row = config_df[config_df["Key"] == "eval_times_source"]
        assert "user-specified" in source_row["Value"].iloc[0].lower()

    def test_no_evaluation_omits_eval_times(self, small_data):
        """When no evaluation has been run, eval_times keys are absent."""
        X, y = small_data
        from mlsurv import SurvivalLearner

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        section = EnvironmentSection(learner)
        section.generate()

        config_df = section.data["config"]
        keys = config_df["Key"].tolist()
        assert "eval_times" not in keys
        assert "eval_times_source" not in keys


def _y_val(times, events):
    return np.array(list(zip(events, times)),
                    dtype=[("event", bool), ("time", float)])


def _val_result_with_auc(eval_times):
    """Minimal val_result exposing .evaluation.auc_t with given eval_times."""
    val = MagicMock()
    if eval_times is None:
        val.evaluation = None
    else:
        val.evaluation = MagicMock()
        val.evaluation.auc_t = pd.DataFrame(
            [{"model_type": "coxph", "eval_time": t, "auc": 0.7}
             for t in eval_times]
        )
    return val


class TestSelectReportHorizons:
    """Validation report calibration sources horizons from the cohort auc_t."""

    def test_uses_auc_t_grid_and_samples_three(self):
        from mlsurv.reports.sections.validation._calibration import (
            _select_report_horizons,
        )
        grid = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        y = _y_val(list(range(1, 21)), [True] * 20)
        horizons, from_auc_t = _select_report_horizons(
            _val_result_with_auc(grid), y,
        )
        assert from_auc_t is True
        assert len(horizons) == 3
        # All chosen horizons come from the auc_t grid (a subset).
        assert set(horizons) <= set(grid)

    def test_small_grid_returned_whole(self):
        from mlsurv.reports.sections.validation._calibration import (
            _select_report_horizons,
        )
        y = _y_val(list(range(1, 21)), [True] * 20)
        horizons, from_auc_t = _select_report_horizons(
            _val_result_with_auc([2.0, 5.0]), y,
        )
        assert from_auc_t is True
        assert horizons == [2.0, 5.0]

    def test_falls_back_to_percentiles_when_no_auc_t(self):
        from mlsurv.reports.sections.validation._calibration import (
            _select_report_horizons,
        )
        y = _y_val(list(range(1, 21)), [True] * 20)
        horizons, from_auc_t = _select_report_horizons(
            _val_result_with_auc(None), y,
        )
        assert from_auc_t is False
        assert len(horizons) == 3  # 25/50/75 event-time percentiles


def _auc_t(eval_times):
    """A minimal auc_t DataFrame carrying the given eval_times."""
    return pd.DataFrame(
        [{"model_type": "coxph", "eval_time": t, "auc": 0.7}
         for t in eval_times]
    )


class TestEvalGridWithPercentileFallback:
    """Shared realized-grid resolver with event-time percentile fallback."""

    def test_first_candidate_grid_wins(self):
        grid, from_auc_t = eval_grid_with_percentile_fallback(
            [_auc_t([1.0, 3.0, 5.0]), _auc_t([2.0])],
            _y_val([1, 2, 3], [True, True, True]),
        )
        assert from_auc_t is True
        assert grid == [1.0, 3.0, 5.0]

    def test_skips_empty_candidates(self):
        grid, from_auc_t = eval_grid_with_percentile_fallback(
            [None, pd.DataFrame(), _auc_t([4.0, 8.0])],
            _y_val([1, 2, 3], [True, True, True]),
        )
        assert from_auc_t is True
        assert grid == [4.0, 8.0]

    def test_percentile_fallback_uses_event_times(self):
        y = _y_val(list(range(1, 21)), [True] * 20)
        grid, from_auc_t = eval_grid_with_percentile_fallback([None], y)
        assert from_auc_t is False
        expected = sorted(set(
            np.percentile(np.arange(1, 21, dtype=float), [25, 50, 75]).tolist()
        ))
        assert grid == expected

    def test_fallback_uses_all_times_when_no_events(self):
        # All censored -> percentile fallback runs over all times, not events.
        y = _y_val([2.0, 4.0, 6.0, 8.0], [False, False, False, False])
        grid, from_auc_t = eval_grid_with_percentile_fallback([None], y)
        assert from_auc_t is False
        expected = sorted(set(
            np.percentile(np.array([2.0, 4.0, 6.0, 8.0]), [25, 50, 75]).tolist()
        ))
        assert grid == expected

    def test_fallback_dedups_coinciding_percentiles(self):
        # Tiny cohort whose 25/50/75 percentiles coincide -> deduped grid.
        y = _y_val([5.0, 5.0], [True, True])
        grid, from_auc_t = eval_grid_with_percentile_fallback([None], y)
        assert from_auc_t is False
        assert grid == [5.0]

    def test_respects_custom_percentiles(self):
        y = _y_val(list(range(1, 21)), [True] * 20)
        grid, _ = eval_grid_with_percentile_fallback([None], y, percentiles=(50,))
        assert grid == [float(np.percentile(np.arange(1, 21, dtype=float), 50))]

    def test_empty_candidate_list_uses_fallback(self):
        # No candidates at all -> loop yields nothing -> percentile fallback.
        y = _y_val(list(range(1, 21)), [True] * 20)
        grid, from_auc_t = eval_grid_with_percentile_fallback([], y)
        assert from_auc_t is False
        assert len(grid) == 3


def _eval_subsection(eval_result, all_models, y_test):
    """Bare _CalibrationSubsection wired with just what _get_eval_times reads."""
    from mlsurv.reports.sections.evaluation._calibration import (
        _CalibrationSubsection,
    )
    sub = _CalibrationSubsection.__new__(_CalibrationSubsection)
    s = MagicMock()
    s.eval_result = eval_result
    s._get_all_models.return_value = all_models
    s.learner.y_test = y_test
    sub._s = s
    return sub


class TestGetEvalTimesFallbackWarning:
    """The evaluation caller warns only when it falls back off the auc_t grid."""

    def test_warns_on_percentile_fallback(self):
        y = _y_val(list(range(1, 21)), [True] * 20)
        sub = _eval_subsection(eval_result=None, all_models={}, y_test=y)
        with pytest.warns(InternalWarning):
            times = sub._get_eval_times()
        assert len(times) == 3

    def test_no_warning_when_auc_t_grid_available(self):
        y = _y_val(list(range(1, 21)), [True] * 20)
        eval_result = MagicMock()
        eval_result.auc_t = _auc_t([1.0, 2.0, 3.0])
        sub = _eval_subsection(eval_result, all_models={}, y_test=y)
        with warnings.catch_warnings():
            warnings.simplefilter("error", InternalWarning)
            times = sub._get_eval_times()
        assert times == [1.0, 2.0, 3.0]
