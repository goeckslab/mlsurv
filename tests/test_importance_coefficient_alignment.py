"""Permutation importance must survive a coefficient problem, and say so.

Pinned here:

1. A failure while attaching coefficients degrades the coefficient column to
   NaN. It must never discard the permutation importance that was already
   computed successfully. The grouped (categorical) branch does its own
   coefficient lookup, so it is covered separately.
2. When permutation importance genuinely fails, the user gets a visible
   warning naming the model and the exception type, not a bare ``failed``
   line and a warning category that is suppressed by default.
3. The coefficient column lands on the right features, even for a model that
   returns its coefficients in a different order than the selector emitted
   them.
4. The ``importance_max_samples`` row cap: the draw itself, its validation,
   its seeding, and both plumbing paths that reach it.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from mlsurv import SurvivalLearner
from mlsurv._warnings import MetricFailureWarning
from mlsurv.evaluation import importance as imp_mod
from mlsurv.models.coxph import CoxPHModel
from mlsurv.preprocessing.selection import TopKCoxSelector


def _fitted_pipeline(X, y):
    """Smallest real fitted pipeline that produces coefficients."""
    pipe = Pipeline([('model', CoxPHModel())])
    pipe.fit(X, y)
    return pipe


class TestCoefficientFailureKeepsImportance:

    def test_coefficient_failure_preserves_perm_importance(
            self, small_data, monkeypatch):
        """A raising _get_coefficients leaves perm_mean intact."""
        X, y = small_data
        pipe = _fitted_pipeline(X, y)

        def _boom(*args, **kwargs):
            raise ValueError("coefficient extraction exploded")

        monkeypatch.setattr(imp_mod, "_get_coefficients", _boom)

        with pytest.warns(MetricFailureWarning):
            summary, iterations, _ = imp_mod._compute_permutation_importance(
                pipe, X, y, n_repeats=2, random_state=42, show_progress=False,
                model_name="coxph",
            )

        assert len(summary) == len(X.columns)
        assert summary['perm_mean'].notna().all()
        assert summary['coefficient'].isna().all()
        assert len(iterations) == len(X.columns) * 2

    def test_opaque_selector_still_aligns_by_model_names(self, small_data):
        """A selector that reports nothing is recoverable when the model
        knows its own feature names: importance and coefficients both survive.

        This shape (a selector that shortens the coefficient vector without
        exposing selected_features_) is what used to return a wrong-length
        list and take the whole importance result down with it.
        """
        X, y = small_data

        class _OpaqueSelector:
            """Drops a column without reporting which ones it kept."""

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X.iloc[:, :-1]

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        pipe = Pipeline([
            ('selector', _OpaqueSelector()),
            ('model', CoxPHModel()),
        ])
        pipe.fit(X, y)

        summary, _, _ = imp_mod._compute_permutation_importance(
            pipe, X, y, n_repeats=2, random_state=42, show_progress=False,
            model_name="coxph",
        )

        assert len(summary) == len(X.columns)
        assert summary['perm_mean'].notna().all()
        # Every kept feature has a coefficient; the dropped one is NaN.
        assert summary['coefficient'].notna().sum() == len(X.columns) - 1
        dropped = X.columns[-1]
        assert np.isnan(
            summary.set_index('feature')['coefficient'][dropped]
        )


class TestGroupedImportanceKeepsCoefficients:
    """The grouped branch does its own positional coefficient lookup and
    builds the per-column summary, so it can fail independently of the
    ungrouped path.
    """

    @staticmethod
    def _grouped_pipeline(X, y):
        class _OpaqueSelector:
            """Shortens the coefficient vector without reporting how."""

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X.iloc[:, :-1]

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        pipe = Pipeline([
            ('selector', _OpaqueSelector()),
            ('model', CoxPHModel()),
        ])
        pipe.fit(X, y)
        return pipe

    def test_grouped_importance_survives_a_short_coefficient_vector(
            self, small_data):
        X, y = small_data
        pipe = self._grouped_pipeline(X, y)
        groups = {'grp': [X.columns[0], X.columns[1]]}

        summary, _, per_column = imp_mod._compute_permutation_importance(
            pipe, X, y, n_repeats=2, random_state=42, show_progress=False,
            model_name="coxph", categorical_groups=groups,
        )

        assert not summary.empty
        assert summary['perm_mean'].notna().all()
        # The group row aggregates its columns, so it has no coefficient.
        grouped_row = summary[summary['feature'] == 'grp']
        assert len(grouped_row) == 1
        assert np.isnan(grouped_row['coefficient'].iloc[0])
        # The per-column table survives alongside it.
        assert per_column is not None
        assert len(per_column) == len(X.columns)

    def test_grouped_importance_survives_a_coefficient_failure(
            self, small_data, monkeypatch):
        X, y = small_data
        pipe = self._grouped_pipeline(X, y)
        groups = {'grp': [X.columns[0], X.columns[1]]}

        def _boom(*args, **kwargs):
            raise ValueError("coefficient extraction exploded")

        monkeypatch.setattr(imp_mod, "_get_coefficients", _boom)

        with pytest.warns(MetricFailureWarning):
            summary, _, _ = imp_mod._compute_permutation_importance(
                pipe, X, y, n_repeats=2, random_state=42, show_progress=False,
                model_name="coxph", categorical_groups=groups,
            )

        assert not summary.empty
        assert summary['perm_mean'].notna().all()
        assert summary['coefficient'].isna().all()


class TestImportanceFailureIsVisible:

    def test_failure_emits_warning_under_default_filters(
            self, small_data, monkeypatch, capsys):
        """The warning must survive mlsurv's default filters.

        It used to resolve to InternalWarning, which
        configure_default_filters() ignores, so the user saw nothing.
        """
        X, y = small_data
        pipe = _fitted_pipeline(X, y)

        def _boom(*args, **kwargs):
            raise ValueError("permutation exploded")

        monkeypatch.setattr(imp_mod, "permutation_importance", _boom)

        with pytest.warns(MetricFailureWarning, match="coxph"):
            summary, iterations, per_column = (
                imp_mod._compute_permutation_importance(
                    pipe, X, y, n_repeats=2, random_state=42,
                    show_progress=True, model_name="coxph",
                )
            )

        assert summary.empty
        assert iterations.empty
        assert per_column is None
        assert "ValueError" in capsys.readouterr().out


class TestImportanceMaxSamples:
    """The row cap is opt-in, seeded, and validated at the edges."""

    def test_none_uses_every_row(self, small_data):
        X, y = small_data
        X_out, y_out = imp_mod._subsample_for_importance(X, y, None, 42)
        assert X_out is X
        assert y_out is y

    def test_cap_at_or_above_cohort_size_is_a_no_op(self, small_data):
        X, y = small_data
        for cap in (len(X), len(X) + 10):
            X_out, y_out = imp_mod._subsample_for_importance(X, y, cap, 42)
            assert X_out is X
            assert y_out is y

    def test_cap_reduces_rows_and_keeps_X_y_aligned(self, small_data):
        X, y = small_data
        X_out, y_out = imp_mod._subsample_for_importance(X, y, 10, 42)

        assert len(X_out) == 10
        assert len(y_out) == 10
        # Same rows on both sides: every drawn index keeps its own outcome.
        for pos, orig in enumerate(X_out.index):
            orig_pos = X.index.get_loc(orig)
            assert y_out[pos] == y[orig_pos]

    def test_accepts_a_randomstate_instance(self, small_data):
        """A RandomState instance is a supported random_state module-wide."""
        X, y = small_data
        X_out, _ = imp_mod._subsample_for_importance(
            X, y, 10, np.random.RandomState(42),
        )
        assert len(X_out) == 10

    def test_unseeded_draw_warns_about_comparability(self, small_data):
        """An unseeded cap warns that draws are not comparable across models."""
        from mlsurv._warnings import MethodologicalWarning

        X, y = small_data
        with pytest.warns(MethodologicalWarning, match="random_state"):
            imp_mod._subsample_for_importance(X, y, 10, None)

    @pytest.mark.parametrize("bad", ["100", 10.5, True])
    def test_non_integer_raises(self, bad, small_data):
        X, y = small_data
        with pytest.raises(ValueError, match="importance_max_samples"):
            imp_mod._subsample_for_importance(X, y, bad, 42)

    def test_same_seed_draws_the_same_rows(self, small_data):
        X, y = small_data
        first, _ = imp_mod._subsample_for_importance(X, y, 10, 42)
        second, _ = imp_mod._subsample_for_importance(X, y, 10, 42)
        other, _ = imp_mod._subsample_for_importance(X, y, 10, 7)

        assert list(first.index) == list(second.index)
        assert list(first.index) != list(other.index)

    def test_capped_run_is_reproducible(self, small_data):
        X, y = small_data
        pipe = _fitted_pipeline(X, y)

        runs = [
            imp_mod._compute_permutation_importance(
                pipe, X, y, n_repeats=2, random_state=42, show_progress=False,
                model_name="coxph", max_samples=20,
            )[0]
            for _ in range(2)
        ]
        pd.testing.assert_frame_equal(runs[0], runs[1])
        assert len(runs[0]) == len(X.columns)

    @pytest.mark.parametrize("bad", [0, -1, 1, 2.5, True, "10"])
    def test_invalid_values_raise_at_both_choke_points(self, bad, small_data):
        """1 is rejected too: the C-index needs a comparable pair, so a
        single row would cost the model its whole importance table."""
        from mlsurv.evaluation._config import EvalConfig

        with pytest.raises(ValueError, match="importance_max_samples"):
            EvalConfig(importance_max_samples=bad)

        # learner.importance() bypasses EvalConfig, so the shared choke
        # point has to reject the value too.
        X, y = small_data
        with pytest.raises(ValueError, match="importance_max_samples"):
            imp_mod._subsample_for_importance(X, y, bad, 42)

    def test_two_is_the_minimum_accepted(self, small_data):
        from mlsurv.evaluation._config import EvalConfig

        assert EvalConfig(importance_max_samples=2).importance_max_samples == 2
        X, y = small_data
        X_out, _ = imp_mod._subsample_for_importance(X, y, 2, 42)
        assert len(X_out) == 2

    def test_end_to_end_cap_reaches_the_permutation_call(
            self, small_data, monkeypatch):
        """evaluate(importance_max_samples=...) must reach sklearn's call."""
        X, y = small_data
        seen = {}
        real = imp_mod.permutation_importance

        def _spy(estimator, X_arg, y_arg, **kwargs):
            seen['n_rows'] = len(X_arg)
            return real(estimator, X_arg, y_arg, **kwargs)

        monkeypatch.setattr(imp_mod, "permutation_importance", _spy)

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph', verbose=False)
        learner.evaluate(n_importance_repeats=2, importance_max_samples=5,
                         show_progress=False, verbose=False)

        assert seen['n_rows'] == 5
        importance = learner.evaluation_result.importance
        assert len(importance) == len(X.columns)
        assert importance['perm_mean'].notna().all()

    def test_cap_reaches_the_standalone_importance_recompute(
            self, small_data, monkeypatch):
        """learner.importance() has its own kwarg chain, bypassing EvalConfig."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train('coxph', verbose=False)
        learner.evaluate(n_importance_repeats=2, show_progress=False,
                         verbose=False)

        seen = {}
        real = imp_mod.permutation_importance

        def _spy(estimator, X_arg, y_arg, **kwargs):
            seen['n_rows'] = len(X_arg)
            return real(estimator, X_arg, y_arg, **kwargs)

        monkeypatch.setattr(imp_mod, "permutation_importance", _spy)

        result = learner.importance(
            'coxph', n_importance_repeats=2, importance_max_samples=5,
            show_progress=False,
        )

        assert seen['n_rows'] == 5
        assert len(result) == len(X.columns)
        assert result['perm_mean'].notna().all()


class TestCoefficientColumnMatchesTheModel:
    """Fast-tier guard for the silent mis-assignment.

    Models that return coefficients in a different order than the selector
    emitted them used to get every value attached to the wrong feature. That
    is the more dangerous of the bugs fixed here, so its regression guard
    runs by default rather than under --run-full.

    `weibull_aft` is the case that carries the bug: lifelines sorts its
    covariates, so the selector's order and the model's order disagree.
    `coxph` indexes by the selector's own order and was never affected; it is
    here as the control.
    """

    @pytest.mark.parametrize("model_name", ["weibull_aft", "coxph"])
    def test_coefficient_column_matches_the_model(self, small_data, model_name):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector=TopKCoxSelector(k=3), random_state=42)
        learner.train(model_name, verbose=False)
        learner.evaluate(n_importance_repeats=2, show_progress=False,
                         verbose=False)

        importance = learner.evaluation_result.importance
        assert len(importance) == len(learner.X_test.columns)

        inner = learner.models[model_name].model.named_steps['model']
        truth = inner.get_coefficients()

        reported = importance.set_index('feature')['coefficient']
        for feature, value in truth.items():
            assert reported[str(feature)] == pytest.approx(value)

        # Features the selector dropped carry no coefficient.
        selected = {str(f) for f in truth.index}
        for feature in learner.X_test.columns:
            if feature not in selected:
                assert np.isnan(reported[feature])


@pytest.mark.integration
class TestImportanceEndToEnd:

    def test_chained_selector_produces_importance(self, small_data):
        """feature_selector=[...] wraps a nested Pipeline with no
        selected_features_ of its own, so importance used to come back empty."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(
            feature_selector=['correlation', TopKCoxSelector(k=2)],
            random_state=42,
        )
        learner.train('coxph', verbose=False)
        learner.evaluate(n_importance_repeats=2, show_progress=False,
                         verbose=False)

        importance = learner.evaluation_result.importance
        assert len(importance) == len(learner.X_test.columns)
        assert importance['perm_mean'].notna().all()
        assert importance['coefficient'].notna().any()
