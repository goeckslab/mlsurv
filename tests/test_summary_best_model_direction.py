"""Regression tests: SummarySection direction-handling for ``cv_metric_mean``.

``cv_metric_mean`` is always stored in higher-is-better
space — ``score_survival_model`` negates lower-is-better scorers
(e.g. integrated Brier returns ``-ibs``). The summary section was
using ``min if scoring in LOWER_IS_BETTER else max`` on
``cv_metric_mean``, which inverted the selection on negated values
and named the worst model as best.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from mlsurv.reports.sections.summary import _display_cv_score


class TestDisplayCvScore:
    def test_higher_is_better_passthrough(self):
        assert _display_cv_score(0.7, 'harrell') == 0.7

    def test_lower_is_better_negates(self):
        # Stored cv_metric_mean = -IBS; display should show raw IBS.
        assert _display_cv_score(-0.15, 'integrated_brier') == pytest.approx(0.15)

    def test_non_finite_passthrough(self):
        assert np.isnan(_display_cv_score(float('nan'), 'integrated_brier'))
        assert _display_cv_score(None, 'harrell') is None


class TestSelectBestModelDirection:
    """``_select_best_model_name`` must pick the largest ``cv_metric_mean``
    regardless of metric direction (since ``cv_metric_mean`` is
    pre-normalized to higher-is-better)."""

    @pytest.mark.parametrize('scoring', ['harrell', 'integrated_brier'])
    def test_picks_highest_cv_metric_mean(self, scoring):
        from mlsurv.reports.sections.summary import SummarySection

        # cv values use the storage convention: positive for harrell,
        # negative for integrated_brier (since score_survival_model
        # returns ``-ibs``).
        if scoring == 'harrell':
            best_cv, worse_cv = 0.70, 0.60
        else:
            best_cv, worse_cv = -0.15, -0.25  # raw IBS 0.15 vs 0.25

        models = {
            'worse': SimpleNamespace(cv_metric_mean=worse_cv),
            'best': SimpleNamespace(cv_metric_mean=best_cv),
        }
        # Build a minimal section instance without invoking __init__.
        section = SummarySection.__new__(SummarySection)
        section.learner = SimpleNamespace(scoring=scoring)
        assert section._select_best_model_name(models) == 'best'

    def test_returns_none_for_empty_dict(self):
        from mlsurv.reports.sections.summary import SummarySection

        section = SummarySection.__new__(SummarySection)
        section.learner = SimpleNamespace(scoring='harrell')
        assert section._select_best_model_name({}) is None
