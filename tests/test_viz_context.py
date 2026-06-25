"""Unit tests for :mod:`mlsurv.viz._context`.

Covers :class:`PlotContext` and :class:`SubpopulationAxis`:
- direct construction with explicit fields
- frozen-ness (assignment raises)
- ``__post_init__`` validation of ``source``
- the three ``PlotContext`` classmethods (``from_main``,
  ``from_validation``, ``from_subpopulation``)
- ``SubpopulationAxis.from_result`` against a real
  :class:`SubpopulationResult`
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv.results.containers import SubpopulationResult
from mlsurv.viz._context import PlotContext, SubpopulationAxis


# ------------------------------------------------------------------
# PlotContext: direct construction + frozen / validation contract
# ------------------------------------------------------------------

class TestPlotContextConstruction:
    """Direct-construction contract tests that don't need a learner."""

    def test_minimal_construction(self):
        ctx = PlotContext(name="main", source="main")
        assert ctx.name == "main"
        assert ctx.source == "main"
        assert ctx.eval_result is None
        assert ctx.test_data is None
        assert ctx.state_ref is None

    def test_is_frozen(self):
        ctx = PlotContext(name="main", source="main")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.name = "other"  # type: ignore[misc]

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="PlotContext.source must be"):
            PlotContext(name="x", source="bogus")  # type: ignore[arg-type]

    @pytest.mark.parametrize("source", ["main", "subpopulation"])
    def test_valid_source_values_without_eval(self, source):
        # 'validation' requires eval_result — covered in its own test below.
        ctx = PlotContext(name="ctx", source=source)  # type: ignore[arg-type]
        assert ctx.source == source

    def test_validation_source_requires_eval_result(self):
        """Cross-field invariant: validation contexts must carry an eval_result."""
        with pytest.raises(ValueError, match="requires eval_result"):
            PlotContext(name="cohort", source="validation", eval_result=None)


# ------------------------------------------------------------------
# PlotContext: constructors from a real learner
# ------------------------------------------------------------------

@pytest.fixture
def trained_learner(small_data):
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.train("coxph")
    return learner


class TestPlotContextFromMain:
    """``PlotContext.from_main`` snapshots the learner's main state."""

    def test_labels_and_source(self, trained_learner):
        ctx = PlotContext.from_main(trained_learner)
        assert ctx.name == "main"
        assert ctx.source == "main"

    def test_references_main_splits(self, trained_learner):
        ctx = PlotContext.from_main(trained_learner)
        state = trained_learner._state
        assert ctx.X_data is state.X_test
        assert ctx.y_data is state.y_test
        assert ctx.y_train is state.y_train
        assert ctx.state_ref is state

    def test_fitted_models_populated(self, trained_learner):
        ctx = PlotContext.from_main(trained_learner)
        assert ctx.fitted_models is not None
        assert "coxph" in ctx.fitted_models
        assert ctx.fitted_models["coxph"] is trained_learner._state.models["coxph"].model

    def test_inherits_scoring_and_time_unit(self, trained_learner):
        ctx = PlotContext.from_main(trained_learner)
        assert ctx.scoring == trained_learner._state.scoring


class TestPlotContextFromValidation:
    """``PlotContext.from_validation`` substitutes cohort data but keeps
    fitted models and ``y_train`` from the main state."""

    @pytest.fixture
    def validated_learner(self, trained_learner, small_data):
        X, y = small_data
        # Use the first 25 samples as a cheap pseudo-validation cohort
        trained_learner.validate(X.iloc[:25], y[:25], name="Holdout")
        return trained_learner

    def test_cohort_labels(self, validated_learner):
        val_result = validated_learner._state.validation_results["Holdout"]
        ctx = PlotContext.from_validation(validated_learner, val_result)
        assert ctx.name == "Holdout"
        assert ctx.source == "validation"

    def test_eval_and_bootstrap_from_cohort(self, validated_learner):
        val_result = validated_learner._state.validation_results["Holdout"]
        ctx = PlotContext.from_validation(validated_learner, val_result)
        assert ctx.eval_result is val_result.evaluation
        assert ctx.bootstrap_result is val_result.bootstrap

    def test_cohort_x_y_substituted(self, validated_learner):
        val_result = validated_learner._state.validation_results["Holdout"]
        ctx = PlotContext.from_validation(validated_learner, val_result)
        assert ctx.X_data is val_result.X_val
        assert ctx.y_data is val_result.y_val

    def test_y_train_still_from_main_state(self, validated_learner):
        """IPCW baseline (y_train) is fixed across cohorts."""
        val_result = validated_learner._state.validation_results["Holdout"]
        ctx = PlotContext.from_validation(validated_learner, val_result)
        assert ctx.y_train is validated_learner._state.y_train

    def test_fitted_models_from_main(self, validated_learner):
        val_result = validated_learner._state.validation_results["Holdout"]
        ctx = PlotContext.from_validation(validated_learner, val_result)
        assert ctx.fitted_models is not None
        assert "coxph" in ctx.fitted_models


class TestPlotContextFromSubpopulation:
    """``PlotContext.from_subpopulation`` keeps main X/y (filtering happens
    inside impls) but sets ``source="subpopulation"``."""

    @pytest.fixture
    def subpopulation_learner(self, small_data):
        X, y = small_data
        labels = np.where(X["age"] >= 55, "old", "young")
        learner = SurvivalLearner(X, y)
        learner.setup(
            random_state=42,
            feature_selector=None,
            subpopulations={"age_group": labels},
        )
        learner.train("coxph")
        learner.evaluate()
        return learner

    def test_labels_and_source(self, subpopulation_learner):
        ctx = PlotContext.from_subpopulation(subpopulation_learner, "age_group")
        assert ctx.name == "age_group"
        assert ctx.source == "subpopulation"

    def test_eval_result_is_none(self, subpopulation_learner):
        """SubpopulationResult stores aggregate frames, not a full EvaluationResult."""
        ctx = PlotContext.from_subpopulation(subpopulation_learner, "age_group")
        assert ctx.eval_result is None
        assert ctx.bootstrap_result is None

    def test_x_y_are_main_test_split(self, subpopulation_learner):
        ctx = PlotContext.from_subpopulation(subpopulation_learner, "age_group")
        state = subpopulation_learner._state
        assert ctx.X_data is state.X_test
        assert ctx.y_data is state.y_test

    def test_state_ref_available_for_mask(self, subpopulation_learner):
        """Impls filter using state_ref.subpopulations_test[subpop_var]."""
        ctx = PlotContext.from_subpopulation(subpopulation_learner, "age_group")
        assert ctx.state_ref is subpopulation_learner._state
        assert "age_group" in ctx.state_ref.subpopulations_test


# ------------------------------------------------------------------
# SubpopulationAxis
# ------------------------------------------------------------------

class TestSubpopulationAxis:
    """``SubpopulationAxis.from_result`` wraps the four frames on
    SubpopulationResult."""

    def _make_sp_result(self, levels=("pre", "post")):
        metrics = pd.DataFrame(
            {
                "model_type": ["coxph"] * len(levels),
                "subpopulation": list(levels),
                "metric": ["harrell"] * len(levels),
                "value": [0.70, 0.65][: len(levels)],
            }
        )
        bootstrap_summary = pd.DataFrame(
            {
                "model_type": ["coxph"] * len(levels),
                "subpopulation": list(levels),
                "metric": ["harrell"] * len(levels),
                "ci_lower": [0.60, 0.55][: len(levels)],
                "ci_upper": [0.80, 0.75][: len(levels)],
            }
        )
        auc_t = pd.DataFrame(
            {
                "model_type": ["coxph"] * (2 * len(levels)),
                "subpopulation": list(levels) * 2,
                "eval_time": [1.0] * len(levels) + [2.0] * len(levels),
                "auc": [0.68, 0.62, 0.65, 0.60][: 2 * len(levels)],
            }
        )
        summary = pd.DataFrame(
            {"subpopulation": list(levels), "n": [30, 20][: len(levels)]}
        )
        return SubpopulationResult(
            group_variable="menostat",
            summary=summary,
            metrics=metrics,
            bootstrap_summary=bootstrap_summary,
            auc_t=auc_t,
            auc_t_bootstrap=None,
            warnings=[],
        )

    def test_from_result_basic(self):
        sp_result = self._make_sp_result()
        axis = SubpopulationAxis.from_result(sp_result)
        assert axis.group_variable == "menostat"
        assert axis.levels == ["post", "pre"]  # sorted alphabetically
        assert axis.metrics_long is sp_result.metrics
        assert axis.bootstrap_summary is sp_result.bootstrap_summary
        assert axis.auc_t is sp_result.auc_t
        assert axis.auc_t_bootstrap is None

    def test_from_result_empty_metrics(self):
        empty_metrics = pd.DataFrame(
            columns=["model_type", "subpopulation", "metric", "value"]
        )
        sp_result = SubpopulationResult(
            group_variable="empty",
            summary=pd.DataFrame(),
            metrics=empty_metrics,
            bootstrap_summary=None,
            auc_t=None,
            auc_t_bootstrap=None,
            warnings=[],
        )
        axis = SubpopulationAxis.from_result(sp_result)
        assert axis.levels == []
        assert axis.bootstrap_summary is None

    def test_is_frozen(self):
        sp_result = self._make_sp_result()
        axis = SubpopulationAxis.from_result(sp_result)
        with pytest.raises(dataclasses.FrozenInstanceError):
            axis.group_variable = "other"  # type: ignore[misc]
