"""Tests for feature_groups / group_colors pass-through to report plots."""

from unittest.mock import patch, MagicMock

import pytest

from mlsurv._warnings import SuggestionWarning
from mlsurv.learner._state import LearnerState


SAMPLE_GROUPS = {'feature_0': 'A', 'feature_1': 'B', 'feature_2': 'A'}
SAMPLE_COLORS = {'A': '#ff0000', 'B': '#0000ff'}


class TestStateFields:
    """feature_groups / group_colors on LearnerState."""

    def test_defaults_none(self):
        state = LearnerState()
        assert state.feature_groups is None
        assert state.group_colors is None

    def test_assign_and_read(self):
        state = LearnerState(feature_groups=SAMPLE_GROUPS, group_colors=SAMPLE_COLORS)
        assert state.feature_groups == SAMPLE_GROUPS
        assert state.group_colors == SAMPLE_COLORS

    def test_setstate_forward_compat(self):
        """Unpickling old state (missing fields) fills defaults."""
        state = LearnerState()
        old_dict = {k: v for k, v in state.__dict__.items()
                    if k not in ('feature_groups', 'group_colors')}
        new_state = LearnerState.__new__(LearnerState)
        new_state.__setstate__(old_dict)
        assert new_state.feature_groups is None
        assert new_state.group_colors is None


class TestSetupStoresGroups:
    """setup() persists feature_groups / group_colors on state."""

    def test_setup_stores_groups(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False, feature_groups=SAMPLE_GROUPS, group_colors=SAMPLE_COLORS)
        assert learner._state.feature_groups == SAMPLE_GROUPS
        assert learner._state.group_colors == SAMPLE_COLORS

    def test_setup_defaults_none(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        assert learner._state.feature_groups is None
        assert learner._state.group_colors is None

    def test_setup_rejects_invalid_feature_groups_type(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        with pytest.raises(TypeError, match="feature_groups must be"):
            learner.setup(verbose=False, feature_groups=42)

    def test_setup_rejects_invalid_group_colors_type(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        with pytest.raises(TypeError, match="group_colors must be"):
            learner.setup(verbose=False, feature_groups=SAMPLE_GROUPS, group_colors=["red"])

    def test_setup_warns_group_colors_without_feature_groups(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        with pytest.warns(SuggestionWarning, match="group_colors has no effect"):
            learner.setup(verbose=False, group_colors=SAMPLE_COLORS)


class TestGenerateReportOverride:
    """generate_report() temporarily overrides and restores state."""

    def test_override_and_restore(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False, feature_groups=SAMPLE_GROUPS, group_colors=SAMPLE_COLORS)

        orig_fg = learner._state.feature_groups
        orig_gc = learner._state.group_colors

        override_fg = {'feature_0': 'X'}
        override_gc = {'X': '#00ff00'}

        # generate_report will fail because no models are trained,
        # but the override/restore still fires via the finally block.
        try:
            learner.generate_report(
                feature_groups=override_fg, group_colors=override_gc,
            )
        except Exception:
            pass

        # State must be restored
        assert learner._state.feature_groups is orig_fg
        assert learner._state.group_colors is orig_gc

    def test_none_override_keeps_original(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False, feature_groups=SAMPLE_GROUPS)

        try:
            learner.generate_report(feature_groups=None)
        except Exception:
            pass

        assert learner._state.feature_groups == SAMPLE_GROUPS

    def test_generate_report_rejects_invalid_feature_groups(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner
        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        with pytest.raises(TypeError, match="feature_groups must be"):
            learner.generate_report(feature_groups="bad")


class TestBaseSectionProperties:
    """BaseSection exposes feature_groups / group_colors as properties."""

    def test_properties_reflect_state(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner
        from mlsurv.reports.sections.base import BaseSection

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False, feature_groups=SAMPLE_GROUPS, group_colors=SAMPLE_COLORS)

        # Create a concrete subclass for testing
        class DummySection(BaseSection):
            name = "dummy"
            title = "Dummy"
            def is_available(self): return True
            def generate(self): return {}

        section = DummySection(learner)
        assert section.feature_groups == SAMPLE_GROUPS
        assert section.group_colors == SAMPLE_COLORS


class TestGroupPlotKwargs:
    """FeatureAnalysisSection._group_plot_kwargs() helper."""

    def test_empty_when_none(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis._section import FeatureAnalysisSection

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)

        section = FeatureAnalysisSection(learner)
        assert section._group_plot_kwargs() == {}

    def test_includes_groups_when_set(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis._section import FeatureAnalysisSection

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False, feature_groups=SAMPLE_GROUPS, group_colors=SAMPLE_COLORS)

        section = FeatureAnalysisSection(learner)
        kw = section._group_plot_kwargs()
        assert kw['feature_groups'] == SAMPLE_GROUPS
        assert kw['group_colors'] == SAMPLE_COLORS

    def test_groups_only_no_colors(self, synthetic_data_clean):
        from mlsurv.learner import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis._section import FeatureAnalysisSection

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False, feature_groups=SAMPLE_GROUPS)

        section = FeatureAnalysisSection(learner)
        kw = section._group_plot_kwargs()
        assert 'feature_groups' in kw
        assert 'group_colors' not in kw

    def test_colors_without_groups_excluded(self, synthetic_data_clean):
        """group_colors alone (no feature_groups) yields empty kwargs."""
        from mlsurv.learner import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis._section import FeatureAnalysisSection

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        # Set group_colors directly on state (bypassing setup validation)
        learner.setup(verbose=False)
        learner._state.group_colors = SAMPLE_COLORS

        section = FeatureAnalysisSection(learner)
        assert section._group_plot_kwargs() == {}


class TestPlotCallPassThrough:
    """Verify kwargs actually reach plot_importance / plot_importance_heatmap."""

    def test_single_model_passes_kwargs(self, synthetic_data_clean):
        """_generate_single_model_content forwards feature_groups to plot_importance."""
        import pandas as pd
        from mlsurv.learner import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis._section import FeatureAnalysisSection

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False, feature_groups=SAMPLE_GROUPS, group_colors=SAMPLE_COLORS)

        # Build minimal importance data
        importance_df = pd.DataFrame({
            'model_type': ['coxph'] * 3,
            'feature': ['feature_0', 'feature_1', 'feature_2'],
            'perm_mean': [0.05, 0.03, 0.01],
            'perm_std': [0.01, 0.01, 0.01],
        })

        mock_fig = MagicMock()
        section = FeatureAnalysisSection(learner, n_features=3)
        section.eval_result = MagicMock(importance=importance_df)

        with patch.object(learner, 'plot_importance', return_value=mock_fig) as mock_plot:
            # summary=None: the combined table degrades to a note, but the
            # chart call (what this test asserts on) must still happen.
            section._generate_single_model_content(
                'coxph', importance_df, None, {}, None,
            )
            mock_plot.assert_called_once()
            call_kwargs = mock_plot.call_args
            assert call_kwargs.kwargs.get('feature_groups') == SAMPLE_GROUPS
            assert call_kwargs.kwargs.get('group_colors') == SAMPLE_COLORS
