"""Tests for persisting statistical tests on LearnerState."""

import numpy as np
import pandas as pd
import pytest

from mlsurv.utils import SurvivalArray
from mlsurv.learner import SurvivalLearner

pytestmark = pytest.mark.integration

# RSF tuning cost scales with forest size, so the tuned plumbing tests below pass a
# tiny Optuna space. Partial spaces merge per-param with the defaults, so the unlisted
# RSF params keep their default ranges. These tests assert risk-group-test state
# invalidation, which is independent of forest size.
_SMALL_TREE_SPACE = {'n_estimators': (10, 20), 'max_depth': (3, 5)}


@pytest.fixture
def evaluated_learner():
    """Create a learner with CoxPH trained and evaluated on synthetic data."""
    np.random.seed(42)
    n = 150
    X = pd.DataFrame({
        'age': np.random.randn(n) * 10 + 60,
        'biomarker': np.random.randn(n),
        'treatment': np.random.binomial(1, 0.5, n).astype(float),
    })
    linear_pred = 0.3 * X['age'] / 10 - 0.5 * X['biomarker'] + 0.2 * X['treatment']
    times = np.exp(-linear_pred / 5 + np.random.randn(n) * 0.5)
    times = np.clip(times, 0.1, 10)
    events = np.random.binomial(1, 0.7, n).astype(bool)
    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)],
    ))

    learner = SurvivalLearner(X, y)
    learner.setup(verbose=False)
    learner.train('coxph', verbose=False)
    learner.evaluate(verbose=False)
    return learner


@pytest.fixture
def setup_only_learner():
    """Create a learner that is set up but not evaluated."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        'age': np.random.randn(n) * 10 + 60,
        'biomarker': np.random.randn(n),
    })
    times = np.clip(np.exp(np.random.randn(n)), 0.1, 10)
    events = np.random.binomial(1, 0.7, n).astype(bool)
    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)],
    ))
    learner = SurvivalLearner(X, y)
    learner.setup(verbose=False)
    return learner


# -- ph_test -----------------------------------------------------------------

class TestPhTest:
    def test_ph_test_persists(self, evaluated_learner):
        result = evaluated_learner.ph_test(verbose=False)
        assert isinstance(result, pd.DataFrame)
        assert evaluated_learner._state.ph_test_result is result
        assert evaluated_learner.ph_test_result is result

    def test_ph_test_columns(self, evaluated_learner):
        result = evaluated_learner.ph_test(verbose=False)
        assert 'feature' in result.columns
        assert 'test_statistic' in result.columns
        assert 'p_value' in result.columns
        assert 'p_adjusted' in result.columns
        assert 'significant' in result.columns

    def test_ph_test_in_export(self, evaluated_learner):
        evaluated_learner.ph_test(verbose=False)
        dfs = evaluated_learner._collect_all_dataframes()
        assert 'ph_test' in dfs

    def test_ph_test_works_without_evaluate(self, setup_only_learner):
        """ph_test uses training data — does not require evaluate()."""
        setup_only_learner.train('coxph', verbose=False)
        result = setup_only_learner.ph_test(verbose=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# -- linearity_test ----------------------------------------------------------


class TestLinearityTest:
    def test_linearity_test_persists(self, evaluated_learner):
        result = evaluated_learner.linearity_test(verbose=False)
        assert isinstance(result, pd.DataFrame)
        assert evaluated_learner._state.linearity_test_result is result
        assert evaluated_learner.linearity_test_result is result

    def test_linearity_test_columns(self, evaluated_learner):
        result = evaluated_learner.linearity_test(verbose=False)
        assert "feature" in result.columns
        assert "test_statistic" in result.columns
        assert "p_value" in result.columns
        assert "p_adjusted" in result.columns
        assert "significant" in result.columns

    def test_linearity_test_in_export(self, evaluated_learner):
        evaluated_learner.linearity_test(verbose=False)
        dfs = evaluated_learner._collect_all_dataframes()
        assert "linearity_test" in dfs

    def test_linearity_test_works_without_evaluate(self, setup_only_learner):
        """linearity_test uses training data — does not require evaluate()."""
        setup_only_learner.train("coxph", verbose=False)
        result = setup_only_learner.linearity_test(verbose=False)
        assert isinstance(result, pd.DataFrame)
        # setup_only_learner has 2 continuous features (age, biomarker),
        # so we expect per-feature rows + a GLOBAL row.
        assert "GLOBAL" in result["feature"].values, (
            "GLOBAL joint Wald row should always be present on success"
        )
        per_feature = result[result["feature"] != "GLOBAL"]
        assert len(per_feature) >= 2, (
            f"expected ≥2 continuous features tested, got "
            f"{per_feature['feature'].tolist()}"
        )

    def test_linearity_test_n_knots_forwarded(self, evaluated_learner, monkeypatch):
        """The n_knots kwarg is forwarded to the underlying metrics function."""
        captured = {}

        def _spy(X, y, n_knots=4):
            captured["n_knots"] = n_knots
            return pd.DataFrame({
                "feature": ["x", "GLOBAL"],
                "test_statistic": [1.0, 1.0],
                "p_value": [0.5, 0.5],
                "p_adjusted": [0.5, float("nan")],
                "significant": [False, False],
            })

        monkeypatch.setattr("mlsurv.evaluation.metrics.linearity_test", _spy)
        evaluated_learner.linearity_test(n_knots=5, verbose=False)
        assert captured["n_knots"] == 5

    def test_linearity_test_verbose_prints(self, evaluated_learner, capsys):
        evaluated_learner.linearity_test(verbose=True)
        captured = capsys.readouterr()
        assert "Log-Hazard Linearity Test" in captured.out

    def test_linearity_test_verbose_false_silent(self, evaluated_learner, capsys):
        evaluated_learner.linearity_test(verbose=False)
        captured = capsys.readouterr()
        assert "Log-Hazard Linearity Test" not in captured.out

    def test_linearity_test_exception_emits_warning(
        self, evaluated_learner, monkeypatch,
    ):
        """When the underlying function raises, the method emits a warning,
        returns an empty DataFrame with the canonical columns, AND persists
        that empty result on _state.linearity_test_result."""
        import warnings as _warnings
        from mlsurv._warnings import MethodologicalWarning

        def _boom(X, y, n_knots=4):
            raise ValueError("simulated failure")

        monkeypatch.setattr("mlsurv.evaluation.metrics.linearity_test", _boom)
        with _warnings.catch_warnings(record=True) as w_list:
            _warnings.simplefilter("always")
            result = evaluated_learner.linearity_test(verbose=False)

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert list(result.columns) == [
            "feature", "test_statistic", "p_value", "p_adjusted", "significant",
        ]
        # State persistence: a regression that swallows the exception but
        # forgets to set _state would fail this assertion.
        assert evaluated_learner._state.linearity_test_result is result
        meth_warns = [
            w for w in w_list
            if issubclass(w.category, MethodologicalWarning)
            and "Linearity test failed" in str(w.message)
        ]
        assert len(meth_warns) == 1


# -- rmst_test ---------------------------------------------------------------

class TestRmstTest:
    def test_rmst_test_persists(self, evaluated_learner):
        result = evaluated_learner.rmst_test(verbose=False)
        assert isinstance(result, pd.DataFrame)
        # Keyed per-model, not singular.
        assert evaluated_learner._state.rmst_test_results['coxph'] is result
        assert evaluated_learner.rmst_test_results['coxph'] is result

    def test_rmst_test_in_export(self, evaluated_learner):
        evaluated_learner.rmst_test(verbose=False)
        dfs = evaluated_learner._collect_all_dataframes()
        assert 'rmst_coxph' in dfs

    def test_rmst_attrs_stamped(self, evaluated_learner):
        result = evaluated_learner.rmst_test(n_groups=3, verbose=False)
        assert result.attrs['model_name'] == 'coxph'
        assert result.attrs['n_groups'] == 3
        assert result.attrs['p_adjust'] == 'holm'
        assert result.attrs['break_type'] == 'quantile'
        assert 'tau' in result.attrs

    def test_rmst_rerun_overwrites_same_model(self, evaluated_learner):
        evaluated_learner.rmst_test(verbose=False)
        evaluated_learner.rmst_test(n_groups=3, verbose=False)
        # Re-running for the same model replaces its entry in place.
        results = evaluated_learner.rmst_test_results
        assert list(results.keys()) == ['coxph']
        assert results['coxph'].attrs['n_groups'] == 3

    def test_rmst_requires_evaluate(self, setup_only_learner):
        with pytest.raises(ValueError, match="No evaluation result"):
            setup_only_learner.rmst_test()

    def test_rmst_has_finite_values(self, evaluated_learner):
        result = evaluated_learner.rmst_test(verbose=False)
        assert len(result) > 0
        assert result['rmst_0'].notna().all()
        assert result['rmst_1'].notna().all()
        assert result['p_value'].notna().all()

    def test_rmst_three_groups(self, evaluated_learner):
        result = evaluated_learner.rmst_test(n_groups=3, verbose=False)
        assert isinstance(result, pd.DataFrame)
        assert 'p_adjusted' in result.columns
        assert len(result) >= 1


# -- logrank_test ------------------------------------------------------------

class TestLogrank:
    def test_logrank_persists(self, evaluated_learner):
        result = evaluated_learner.logrank_test(verbose=False)
        assert isinstance(result, pd.DataFrame)
        # Keyed per-model, not singular.
        assert evaluated_learner._state.logrank_test_results['coxph'] is result
        assert evaluated_learner.logrank_test_results['coxph'] is result

    def test_logrank_in_export(self, evaluated_learner):
        evaluated_learner.logrank_test(verbose=False)
        dfs = evaluated_learner._collect_all_dataframes()
        assert 'logrank_coxph' in dfs

    def test_logrank_attrs_stamped(self, evaluated_learner):
        result = evaluated_learner.logrank_test(n_groups=3, verbose=False)
        assert result.attrs['model_name'] == 'coxph'
        assert result.attrs['n_groups'] == 3
        assert result.attrs['p_adjust'] == 'holm'
        assert result.attrs['break_type'] == 'quantile'
        # logrank has no tau (that is RMST-only).
        assert 'tau' not in result.attrs

    def test_logrank_requires_evaluate(self, setup_only_learner):
        with pytest.raises(ValueError, match="No evaluation result"):
            setup_only_learner.logrank_test()

    def test_logrank_has_omnibus_row(self, evaluated_learner):
        result = evaluated_learner.logrank_test(verbose=False)
        assert 'omnibus' in result['comparison'].values

    def test_logrank_three_groups(self, evaluated_learner):
        result = evaluated_learner.logrank_test(n_groups=3, verbose=False)
        assert isinstance(result, pd.DataFrame)
        # Should have omnibus + pairwise rows
        assert 'omnibus' in result['comparison'].values
        assert len(result) > 1


# -- custom breaks -----------------------------------------------------------

class TestCustomBreaks:
    """rmst_test / logrank_test custom quantile / score / maxstat breaks."""

    def test_rmst_quantile_breaks_group_count_and_labels(self, evaluated_learner):
        result = evaluated_learner.rmst_test(breaks=[0.25, 0.75], verbose=False)
        # Three groups -> three pairwise rows, labelled by percentile range.
        assert len(result) == 3
        labels = set(result['group_1']) | set(result['group_2'])
        assert labels == {'< 25%', '25–75%', '≥ 75%'}

    def test_logrank_quantile_breaks_group_count_and_labels(self, evaluated_learner):
        result = evaluated_learner.logrank_test(breaks=[0.25, 0.75], verbose=False)
        assert int(result.loc[0, 'n_groups']) == 3
        pairwise = result[result['comparison'] != 'omnibus']['comparison'].tolist()
        assert pairwise == [
            '< 25% vs 25–75%', '< 25% vs ≥ 75%', '25–75% vs ≥ 75%',
        ]

    def test_rmst_score_breaks(self, evaluated_learner):
        scores = evaluated_learner.evaluation_result.test_set['risk_score']
        cut = float(scores.median())
        result = evaluated_learner.rmst_test(
            breaks=[cut], break_type='score', verbose=False,
        )
        # Two score bins -> single comparison with "< cut" / ">= cut" labels.
        assert len(result) == 1
        g1, g2 = result.loc[0, 'group_1'], result.loc[0, 'group_2']
        assert g1.startswith('<') and g2.startswith('≥')

    def test_maxstat_runs_end_to_end(self, evaluated_learner):
        result = evaluated_learner.rmst_test(break_type='maxstat', verbose=False)
        assert not result.empty
        # maxstat 2-group labels mirror plot_km.
        assert set(result['group_1']) | set(result['group_2']) == {
            'Low risk', 'High risk',
        }

    def test_breaks_none_matches_n_groups(self, evaluated_learner):
        """breaks=None reproduces the integer n_groups behaviour."""
        default = evaluated_learner.rmst_test(verbose=False)
        explicit = evaluated_learner.rmst_test(n_groups=2, verbose=False)
        assert len(default) == len(explicit) == 1
        three = evaluated_learner.rmst_test(n_groups=3, verbose=False)
        assert len(three) == 3  # pairwise rows for 3 groups

    def test_parity_with_plot_survival(self, evaluated_learner):
        """Same breaks produce the same groups in the test path and the plot."""
        from mlsurv.viz._survival_helpers import _create_risk_groups

        mr = evaluated_learner.models['coxph']
        rg, _ = evaluated_learner._get_test_risk_groups(
            mr, breaks=[0.5], break_type='quantile',
        )
        test_set = evaluated_learner.evaluation_result.test_set
        df = test_set[test_set['model_type'] == 'coxph'].copy()
        df, _, _ = _create_risk_groups(df, 'risk_score', 2, breaks=[0.5],
                                       break_type='quantile')
        assert (rg['risk_group'].values == df['risk_group'].values).all()

    @pytest.mark.parametrize('bad', [[0.75, 0.25], [1.5], [-0.1], []])
    def test_invalid_quantile_breaks_raise(self, evaluated_learner, bad):
        with pytest.raises(ValueError):
            evaluated_learner.rmst_test(breaks=bad, verbose=False)

    def test_invalid_break_type_raises(self, evaluated_learner):
        with pytest.raises(ValueError, match='break_type'):
            evaluated_learner.logrank_test(
                breaks=[0.5], break_type='nonsense', verbose=False,
            )

    def test_both_n_groups_and_breaks_warns(self, evaluated_learner):
        from mlsurv._warnings import SuggestionWarning

        with pytest.warns(SuggestionWarning, match='n_groups is ignored'):
            evaluated_learner.rmst_test(n_groups=4, breaks=[0.5], verbose=False)
        with pytest.warns(SuggestionWarning, match='n_groups is ignored'):
            evaluated_learner.logrank_test(n_groups=4, breaks=[0.5], verbose=False)

    def test_branch_fix_n_groups2_with_three_group_breaks(self, evaluated_learner):
        """breaks win over an explicit n_groups=2: 3 groups -> 3 pairwise rows.

        Pins the group-count branch fix: the old ``n_groups == 2`` short-circuit
        would wrongly route this through the two-group branch.
        """
        with pytest.warns(Warning):  # precedence SuggestionWarning
            result = evaluated_learner.rmst_test(
                n_groups=2, breaks=[0.25, 0.75], verbose=False,
            )
        assert len(result) == 3
        assert 'p_adjusted' in result.columns

    def test_two_group_branch_column_contract(self, evaluated_learner):
        """The 2-group RMST branch populates rmst_0 / rmst_1."""
        result = evaluated_learner.rmst_test(breaks=[0.5], verbose=False)
        assert len(result) == 1
        assert {'rmst_0', 'rmst_1'}.issubset(result.columns)
        assert result['rmst_0'].notna().all()
        assert result['rmst_1'].notna().all()

    def test_score_break_outside_range_collapse_warns(self, evaluated_learner):
        """A score cut above all scores leaves an empty bin -> visible warning."""
        from mlsurv._warnings import MethodologicalWarning

        scores = evaluated_learner.evaluation_result.test_set['risk_score']
        breaks = [float(scores.median()), float(scores.max()) + 100.0]
        with pytest.warns(MethodologicalWarning, match='only 2 could be formed'):
            result = evaluated_learner.rmst_test(
                breaks=breaks, break_type='score', verbose=False,
            )
        # Collapsed to 2 groups -> single comparison row.
        assert len(result) == 1

    def test_maxstat_with_breaks_warns_breaks_ignored(self, evaluated_learner):
        """Passing breaks with break_type='maxstat' warns that breaks is ignored."""
        from mlsurv._warnings import SuggestionWarning

        with pytest.warns(SuggestionWarning, match="breaks is ignored"):
            evaluated_learner.rmst_test(
                breaks=[0.5], break_type='maxstat', verbose=False,
            )


# -- absent when not computed ------------------------------------------------

class TestAbsentWhenNotComputed:
    def test_stat_tests_absent_when_not_computed(self, evaluated_learner):
        """Fresh evaluated learner has ph_test (from setup) but not rmst/logrank."""
        dfs = evaluated_learner._collect_all_dataframes()
        # ph_test now runs automatically during setup()
        assert 'ph_test' in dfs
        # No per-model risk-group keys before the tests run.
        assert not any(k.startswith('rmst_') for k in dfs)
        assert not any(k.startswith('logrank_') for k in dfs)


# -- per-model keying & invalidation -----------------------------------------

@pytest.fixture
def two_model_learner():
    """A learner with coxph + rsf trained and evaluated (no tuning)."""
    np.random.seed(7)
    n = 160
    X = pd.DataFrame({
        'age': np.random.randn(n) * 10 + 60,
        'biomarker': np.random.randn(n),
        'treatment': np.random.binomial(1, 0.5, n).astype(float),
    })
    linear_pred = 0.3 * X['age'] / 10 - 0.5 * X['biomarker']
    times = np.clip(np.exp(-linear_pred / 5 + np.random.randn(n) * 0.5), 0.1, 10)
    events = np.random.binomial(1, 0.7, n).astype(bool)
    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)],
    ))
    learner = SurvivalLearner(X, y)
    learner.setup(verbose=False)
    learner.run(['coxph', 'rsf'], tune=False, verbose=False)
    learner.evaluate(verbose=False)
    return learner


class TestPerModelKeying:
    def test_two_models_keyed_independently(self, two_model_learner):
        cox = two_model_learner.logrank_test('coxph', verbose=False)
        rsf = two_model_learner.logrank_test('rsf', verbose=False)
        results = two_model_learner.logrank_test_results
        assert set(results) == {'coxph', 'rsf'}
        # Each entry is the result computed for that model (no overwrite).
        assert results['coxph'] is cox
        assert results['rsf'] is rsf
        assert results['coxph'].attrs['model_name'] == 'coxph'
        assert results['rsf'].attrs['model_name'] == 'rsf'

    def test_export_has_both_models(self, two_model_learner):
        two_model_learner.logrank_test('coxph', verbose=False)
        two_model_learner.rmst_test('rsf', verbose=False)
        dfs = two_model_learner._collect_all_dataframes()
        assert 'logrank_coxph' in dfs
        assert 'rmst_rsf' in dfs


class TestRetrainInvalidation:
    def test_retrain_drops_only_that_models_entries(self, two_model_learner):
        two_model_learner.logrank_test('coxph', verbose=False)
        two_model_learner.rmst_test('coxph', verbose=False)
        two_model_learner.logrank_test('rsf', verbose=False)
        two_model_learner.rmst_test('rsf', verbose=False)

        # Re-training rsf invalidates its stale risk-group tests (its risk
        # scores came from the now-replaced pipeline) but leaves coxph intact.
        two_model_learner.train('rsf', verbose=False)
        assert 'rsf' not in two_model_learner.logrank_test_results
        assert 'rsf' not in two_model_learner.rmst_test_results
        assert 'coxph' in two_model_learner.logrank_test_results
        assert 'coxph' in two_model_learner.rmst_test_results

    @pytest.mark.integration
    def test_tune_overwrite_drops_entries(self, two_model_learner):
        """tune() overwrites the model in place and drops its stale risk-group
        tests (the _install_tuned_result drop branch)."""
        two_model_learner.logrank_test('rsf', verbose=False)
        two_model_learner.rmst_test('rsf', verbose=False)
        two_model_learner.logrank_test('coxph', verbose=False)

        two_model_learner.tune('rsf', n_trials=1, search_space=_SMALL_TREE_SPACE,
                               verbose=False)
        assert 'rsf' not in two_model_learner.logrank_test_results
        assert 'rsf' not in two_model_learner.rmst_test_results
        # An unrelated model's entry survives.
        assert 'coxph' in two_model_learner.logrank_test_results

    @pytest.mark.integration
    def test_run_tune_store_drops_entries(self, two_model_learner):
        """run(tune=True) re-stores a tuned result over an existing model and
        drops its stale risk-group tests (the _cv_training store-drop branch)."""
        two_model_learner.logrank_test('rsf', verbose=False)
        two_model_learner.rmst_test('rsf', verbose=False)

        two_model_learner.run(['rsf'], tune=True, n_trials=1,
                              search_spaces={'rsf': _SMALL_TREE_SPACE}, verbose=False)
        assert 'rsf' not in two_model_learner.logrank_test_results
        assert 'rsf' not in two_model_learner.rmst_test_results

    def test_setup_reinit_clears_dicts(self, two_model_learner):
        """Re-running setup() clears the per-model dicts (the results describe
        models trained on the now-replaced train/test split)."""
        two_model_learner.logrank_test('coxph', verbose=False)
        two_model_learner.rmst_test('rsf', verbose=False)
        assert two_model_learner.logrank_test_results
        assert two_model_learner.rmst_test_results

        two_model_learner.setup(verbose=False)
        assert two_model_learner.logrank_test_results == {}
        assert two_model_learner.rmst_test_results == {}
