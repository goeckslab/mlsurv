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
