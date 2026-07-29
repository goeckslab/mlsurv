"""Focused tests for report-section recoverable exception handling."""

import logging
import warnings

import numpy as np
import pandas as pd

from mlsurv import SurvivalLearner
from mlsurv.reports.sections.environment import EnvironmentSection
from mlsurv.reports.sections.evaluation import EvaluationSection
from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection
from mlsurv.reports.sections.summary import SummarySection


def test_summary_logs_and_recovers_when_data_overview_fails(small_data, caplog):
    """SummarySection should log and continue when overview data access fails."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)
    learner.X_train = None  # break data access in generate()

    section = SummarySection(learner)

    with caplog.at_level(logging.ERROR, logger="mlsurv.reports.sections.summary"):
        result = section.generate()

    assert "summary: could not access train/test data" in caplog.text
    assert isinstance(result["markdown"], str)
    # Graceful degradation: generate() returns valid result even when data is broken
    assert isinstance(result["figures"], list)


def test_feature_analysis_warns_and_logs_on_importance_plot_failure(
    small_data, monkeypatch, caplog
):
    """FeatureAnalysisSection should warn+log and keep fallback markdown on chart errors."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    trained = learner.train("coxph")
    eval_result = learner.evaluate(trained, n_importance_repeats=5, bootstrap=False)
    section = FeatureAnalysisSection(learner, eval_result=eval_result)

    def _raise_importance_failure(*args, **kwargs):
        raise RuntimeError("synthetic importance failure")

    import mlsurv.viz.features as features_viz

    monkeypatch.setattr(features_viz, "plot_importance", _raise_importance_failure)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(
            logging.ERROR, logger="mlsurv.reports.sections.feature_analysis"
        ):
            result = section.generate()

    assert any("Could not generate importance chart" in str(w.message) for w in caught)
    assert "feature_analysis: importance chart generation for coxph" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "*[Figure unavailable: synthetic importance failure]*" in result["markdown"]


def test_feature_analysis_recovers_from_overflow_in_a_chart(
    small_data, monkeypatch, caplog
):
    """An OverflowError costs one figure, not the whole section.

    OverflowError was simply never listed in the section's exception tuples,
    so a degenerate value reaching int() or math.ceil() escaped the per-figure
    guard and the report generator replaced all of Feature Analysis with a
    placeholder.
    """
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    trained = learner.train("coxph")
    # Importance must actually be computed, or the chart this test monkeypatches
    # is never reached and the assertions below pass vacuously.
    eval_result = learner.evaluate(trained, n_importance_repeats=2, bootstrap=False)
    section = FeatureAnalysisSection(learner, eval_result=eval_result)

    def _raise_overflow(*args, **kwargs):
        raise OverflowError("cannot convert float infinity to integer")

    import mlsurv.viz.features as features_viz

    monkeypatch.setattr(features_viz, "plot_importance", _raise_overflow)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(
            logging.ERROR, logger="mlsurv.reports.sections.feature_analysis"
        ):
            result = section.generate()

    assert any("Could not generate importance chart" in str(w.message) for w in caught)
    assert "OverflowError" in caplog.text
    assert (
        "*[Figure unavailable: cannot convert float infinity to integer]*"
        in result["markdown"]
    )


def test_overflow_error_is_recoverable_outside_feature_analysis(
    small_data, monkeypatch, caplog
):
    """The widened tuple is shared by every report section, not just one.

    `RUNTIME_EXCEPTIONS` feeds `ALL_EXCEPTIONS`, `CHART_EXCEPTIONS`,
    `PLATFORM_EXCEPTIONS` and `IMPORT_EXCEPTIONS`, which ~23 report modules
    import. Pin the behavior somewhere other than the section that motivated
    the change.
    """
    from mlsurv.reports.sections.base import (
        ALL_EXCEPTIONS, DATA_EXCEPTIONS, IMPORT_EXCEPTIONS,
        PLATFORM_EXCEPTIONS, RUNTIME_EXCEPTIONS,
    )
    from mlsurv.reports.sections.feature_analysis import CHART_EXCEPTIONS

    assert OverflowError in RUNTIME_EXCEPTIONS
    for tup in (ALL_EXCEPTIONS, PLATFORM_EXCEPTIONS, IMPORT_EXCEPTIONS,
                CHART_EXCEPTIONS):
        assert OverflowError in tup
    # DATA_EXCEPTIONS is about bad data shapes, not arithmetic; leave it alone.
    assert OverflowError not in DATA_EXCEPTIONS

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    def _raise_overflow(*args, **kwargs):
        raise OverflowError("cannot convert float infinity to integer")

    monkeypatch.setattr(learner, "plot_metric", _raise_overflow)

    from mlsurv.reports.sections.evaluation._performance import (
        _PerformanceSubsection,
    )

    class _DummyBootstrap:
        def __init__(self):
            self.iterations = pd.DataFrame({
                "iteration": [0, 1, 2], "metric": ["harrell"] * 3,
                "value": [0.6, 0.62, 0.61],
            })
            self.summary = pd.DataFrame({"metric": ["harrell"]})

        @staticmethod
        def get_distribution(metric):
            if metric == "harrell":
                return np.array([0.6, 0.62, 0.61])
            return np.array([])

    section = EvaluationSection(
        learner, bootstrap_results={"coxph": _DummyBootstrap()}
    )

    md_lines = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.ERROR,
                             logger="mlsurv.reports.sections.evaluation"):
            _PerformanceSubsection(section)._add_bootstrap_rainclouds(
                md_lines, [], [],
            )

    assert any("Could not generate bootstrap rainclouds" in str(w.message)
               for w in caught)
    assert "OverflowError" in caplog.text
    assert "cannot convert float infinity to integer" in "\n".join(md_lines)


def test_evaluation_bootstrap_plot_failure_warns_and_logs(
    small_data, monkeypatch, caplog
):
    """_PerformanceSubsection should log+warn and append markdown fallback on bootstrap plot failure."""
    from mlsurv.reports.sections.evaluation._performance import _PerformanceSubsection

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    class _DummyBootstrap:
        def __init__(self):
            self.iterations = pd.DataFrame(
                {"iteration": [0, 1, 2], "metric": ["harrell"] * 3, "value": [0.6, 0.62, 0.61]}
            )
            self.summary = pd.DataFrame({"metric": ["harrell"]})

        @staticmethod
        def get_distribution(metric):
            if metric == "harrell":
                return np.array([0.6, 0.62, 0.61])
            return np.array([])

    section = EvaluationSection(learner, bootstrap_results={"coxph": _DummyBootstrap()})

    def _raise_bootstrap_plot_failure(*args, **kwargs):
        raise RuntimeError("synthetic bootstrap plotting failure")

    monkeypatch.setattr(learner, "plot_metric", _raise_bootstrap_plot_failure)

    md_lines = []
    figures = []
    figure_info = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.ERROR, logger="mlsurv.reports.sections.evaluation"):
            _PerformanceSubsection(section)._add_bootstrap_rainclouds(
                md_lines, figures, figure_info,
            )

    assert any("Could not generate bootstrap rainclouds" in str(w.message) for w in caught)
    assert "evaluation: bootstrap raincloud generation" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "*[Bootstrap distribution plot unavailable: synthetic bootstrap plotting failure]*" in "\n".join(
        md_lines
    )


def test_environment_logs_and_recovers_when_config_raises_attribute_error(
    small_data, monkeypatch, caplog
):
    """EnvironmentSection should handle AttributeError from learner.config gracefully."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    monkeypatch.setattr(
        type(learner), 'config',
        property(lambda self: (_ for _ in ()).throw(AttributeError("no config attr"))),
    )

    section = EnvironmentSection(learner)

    with caplog.at_level(logging.ERROR, logger="mlsurv.reports.sections.environment"):
        result = section.generate()

    assert "environment: reading learner.config failed" in caplog.text
    assert "AttributeError" in caplog.text
    assert isinstance(result["markdown"], str)
    assert isinstance(section.data["config"], pd.DataFrame)
    assert section.data["config"].empty
