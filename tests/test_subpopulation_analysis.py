"""Tests for subpopulation analysis."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sksurv.datasets import load_gbsg2

from mlsurv.evaluation._subpopulation import (
    _validate_subpopulations,
    _compute_subpopulation_summary,
    _safe_ibs,
    compute_subpopulation_metrics,
    extract_subpopulation_bootstrap_summary,
)
from mlsurv.results.containers import SubpopulationResult

from tests._helpers import make_random_survival_array, make_survival_array


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Load GBSG2 and create subpopulation labels."""
    X, y = load_gbsg2()
    from mlsurv.utils import to_structured_array
    y = to_structured_array(y)
    # Encode categoricals for model fitting
    X = pd.get_dummies(X, drop_first=True)
    return X, y


@pytest.fixture
def subpopulation_labels(sample_data):
    """Create cancer-type-like subpopulation labels."""
    X, y = sample_data
    n = len(X)
    rng = np.random.RandomState(42)
    labels = rng.choice(['TypeA', 'TypeB', 'TypeC'], size=n, p=[0.5, 0.3, 0.2])
    return labels


# ---------------------------------------------------------------------------
# _validate_subpopulations
# ---------------------------------------------------------------------------

class TestValidateSubpopulations:
    def test_valid_input(self):
        labels = np.array(['A', 'B', 'A', 'C'])
        result = _validate_subpopulations({'grp': labels}, n_samples=4)
        assert 'grp' in result
        assert len(result['grp']) == 4

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="has length 3.*expected 4"):
            _validate_subpopulations({'grp': np.array(['A', 'B', 'C'])}, n_samples=4)

    def test_not_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            _validate_subpopulations(['A', 'B'], n_samples=2)

    def test_empty_dict(self):
        with pytest.raises(ValueError, match="empty"):
            _validate_subpopulations({}, n_samples=5)


# ---------------------------------------------------------------------------
# _compute_subpopulation_summary
# ---------------------------------------------------------------------------

class TestComputeSubpopulationSummary:
    def test_basic_summary(self):
        labels = np.array(['A', 'A', 'B', 'B', 'B'])
        y = make_survival_array(
            [True, False, True, True, False], [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        result = _compute_subpopulation_summary(labels, y)
        assert len(result) == 2
        assert set(result['subpopulation']) == {'A', 'B'}
        a_row = result[result['subpopulation'] == 'A'].iloc[0]
        assert a_row['n_samples'] == 2
        assert a_row['n_events'] == 1


# ---------------------------------------------------------------------------
# compute_subpopulation_metrics
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestComputeSubpopulationMetrics:
    def test_returns_per_subpopulation_cindices(self, sample_data, subpopulation_labels):
        """Test that compute_subpopulation_metrics returns C-indices per subpopulation."""
        X, y = sample_data

        # Create a mock eval_result with test_set DataFrame
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False, random_state=42)
        learner.train('coxph', verbose=False)
        result = learner.evaluate(verbose=False)

        # Slice subpopulation labels to test set
        test_positions = X.index.get_indexer(learner.X_test.index)
        test_labels = subpopulation_labels[test_positions]
        subpopulations = {'cancer_type': test_labels}

        sg_results = compute_subpopulation_metrics(result, subpopulations, learner.y_test)
        assert 'cancer_type' in sg_results
        sg = sg_results['cancer_type']
        assert isinstance(sg, SubpopulationResult)
        assert len(sg.summary) == 3  # TypeA, TypeB, TypeC
        assert len(sg.metrics) > 0
        assert all(sg.metrics['metric'] == 'harrell')

    def test_small_subpopulation_warning(self):
        """Test warning for subpopulations with few events."""
        # Create tiny data with subpopulations
        n = 20
        rng = np.random.RandomState(42)
        y = make_survival_array(rng.binomial(1, 0.5, n), rng.exponential(10, n))
        labels = np.array(['Rare'] * 3 + ['Common'] * 17)

        # Create mock eval_result
        test_set = pd.DataFrame({
            'model_type': ['coxph'] * n,
            'risk_score': rng.randn(n),
            'time': [row[1] for row in y],
            'event': [int(row[0]) for row in y],
        })

        class MockEvalResult:
            pass

        mock_result = MockEvalResult()
        mock_result.test_set = test_set

        result = compute_subpopulation_metrics(
            mock_result, {'grp': labels}, y, min_events=10
        )
        assert 'grp' in result
        # 'Rare' group should generate a warning
        assert any('Rare' in w for w in result['grp'].warnings)

    def test_small_subpopulation_warnings_coalesced(self):
        """Many small subpopulations emit a single aggregate warning per
        group_var, but every per-label message is still captured in
        SubpopulationResult.warnings.
        """
        from mlsurv._warnings import SmallSubpopulationWarning

        # 5 small strata + 1 large stratum → 5 low-event subpopulations
        n = 60
        rng = np.random.RandomState(0)
        y = make_survival_array(rng.binomial(1, 0.5, n), rng.exponential(10, n))
        labels = np.array(
            ['s1'] * 5 + ['s2'] * 5 + ['s3'] * 5
            + ['s4'] * 5 + ['s5'] * 5 + ['big'] * 35
        )
        test_set = pd.DataFrame({
            'model_type': ['coxph'] * n,
            'risk_score': rng.randn(n),
            'time': [row[1] for row in y],
            'event': [int(row[0]) for row in y],
        })

        class MockEvalResult:
            pass

        mock_result = MockEvalResult()
        mock_result.test_set = test_set

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SmallSubpopulationWarning)
            result = compute_subpopulation_metrics(
                mock_result, {'grp': labels}, y, min_events=10
            )

        small_warns = [
            w for w in caught
            if issubclass(w.category, SmallSubpopulationWarning)
        ]
        # max_emit=0 → only the aggregate summary fires (1 warning).
        assert len(small_warns) == 1
        summary_msg = str(small_warns[0].message)
        assert "5/6" in summary_msg
        assert "'grp' subpopulations have insufficient events (< 10)" in summary_msg
        for name in ('s1', 's2', 's3', 's4', 's5'):
            assert name in summary_msg
        assert "Metric estimates may be unreliable." in summary_msg

        # All 5 per-label messages are still captured in SubpopulationResult.warnings.
        result_warnings = result['grp'].warnings
        assert len(result_warnings) == 5
        for name in ('s1', 's2', 's3', 's4', 's5'):
            assert any(f"'{name}'" in w for w in result_warnings)


# ---------------------------------------------------------------------------
# Risk/mask length-mismatch warning collapse — fast tier (MockEvalResult)
# ---------------------------------------------------------------------------

class TestRiskMaskLengthMismatchWarning:
    """Regression tests for risk/mask length-mismatch warning collapse.

    Pre-fix: ``_compute_model_subpopulation_metrics`` emitted a
    ``logger.warning`` for every (level, model) pair when
    ``len(risk_all) != len(mask)``. With N levels and a mismatched
    model, that produced N lines per cohort.
    Post-fix: ``compute_subpopulation_metrics`` detects the mismatch
    once per (model, group_var) and emits ONE coalesced
    ``MethodologicalWarning`` per group_var.

    These tests use ``MockEvalResult`` + tiny synthetic data (no model
    training), so they belong in the fast tier (not the integration
    class above).
    """

    @staticmethod
    def _make_y(n, rng):
        return make_random_survival_array(
            n, event_rate=0.7, time_scale=10.0,
            event_dist="binomial", rng=rng,
        )

    def test_single_group_var_coalesces_to_one_warning(self):
        from mlsurv._warnings import MethodologicalWarning

        n = 30
        rng = np.random.RandomState(0)
        y = self._make_y(n, rng)
        labels = np.array(['lvl1', 'lvl2', 'lvl3', 'lvl4', 'lvl5', 'lvl6'] * 5)

        # Model 'short' has 20 test_set rows (mismatch); 'ok' has 30 rows.
        test_set = pd.DataFrame({
            'model_type': ['short'] * 20 + ['ok'] * n,
            'risk_score': np.concatenate([rng.randn(20), rng.randn(n)]),
        })

        class MockEvalResult:
            pass

        mock_result = MockEvalResult()
        mock_result.test_set = test_set

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", MethodologicalWarning)
            result = compute_subpopulation_metrics(
                mock_result, {'grp': labels}, y, min_events=1,
            )

        mismatch_warns = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "risk_score length" in str(w.message)
        ]
        assert len(mismatch_warns) == 1, (
            f"expected 1 mismatch warning, got {len(mismatch_warns)}: "
            f"{[str(w.message) for w in mismatch_warns]}"
        )
        msg = str(mismatch_warns[0].message)
        assert "short" in msg
        assert "n_risk=20" in msg
        assert "1/2" in msg
        assert "'grp' subpopulation models" in msg
        assert str(n) in msg

        metrics = result['grp'].metrics
        short_rows = metrics[metrics['model_type'] == 'short']
        assert len(short_rows) == 6
        assert short_rows['value'].isna().all()
        ok_rows = metrics[metrics['model_type'] == 'ok']
        assert len(ok_rows) == 6
        assert ok_rows['value'].notna().any()

    def test_multi_group_var_warns_once_per_group(self):
        """Same mismatched model across two group_vars produces two
        warnings (one per group_var), proving the dedup keying is
        (model, group_var) and not just (model)."""
        from mlsurv._warnings import MethodologicalWarning

        n = 24
        rng = np.random.RandomState(1)
        y = self._make_y(n, rng)
        labels_a = np.array(['a1', 'a2', 'a3', 'a4'] * 6)
        labels_b = np.array(['b1', 'b2', 'b3'] * 8)

        test_set = pd.DataFrame({
            'model_type': ['short'] * 10 + ['ok'] * n,
            'risk_score': np.concatenate([rng.randn(10), rng.randn(n)]),
        })

        class MockEvalResult:
            pass

        mock_result = MockEvalResult()
        mock_result.test_set = test_set

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", MethodologicalWarning)
            compute_subpopulation_metrics(
                mock_result,
                {'grpA': labels_a, 'grpB': labels_b},
                y, min_events=1,
            )

        mismatch_warns = [
            str(w.message) for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "risk_score length" in str(w.message)
        ]
        assert len(mismatch_warns) == 2
        assert any("'grpA'" in m for m in mismatch_warns)
        assert any("'grpB'" in m for m in mismatch_warns)

    def test_no_mismatch_emits_no_warning(self):
        """When all models' risk_score length matches labels length,
        no length-mismatch MethodologicalWarning is emitted."""
        from mlsurv._warnings import MethodologicalWarning

        n = 15
        rng = np.random.RandomState(2)
        y = self._make_y(n, rng)
        labels = np.array(['x', 'y', 'z'] * 5)

        test_set = pd.DataFrame({
            'model_type': ['m1'] * n + ['m2'] * n,
            'risk_score': np.concatenate([rng.randn(n), rng.randn(n)]),
        })

        class MockEvalResult:
            pass

        mock_result = MockEvalResult()
        mock_result.test_set = test_set

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", MethodologicalWarning)
            compute_subpopulation_metrics(
                mock_result, {'grp': labels}, y, min_events=1,
            )

        mismatch_warns = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "risk_score length" in str(w.message)
        ]
        assert mismatch_warns == []

    def test_helper_direct_call_emits_user_visible_warning(self):
        """Calling _compute_model_subpopulation_metrics directly (without
        the parent pre-check) on a length-mismatched model still emits
        a MethodologicalWarning, per the "No Silent Data Loss" rule.
        This guards the defensive fallback path."""
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.evaluation._subpopulation import (
            _compute_model_subpopulation_metrics,
        )

        n = 12
        rng = np.random.RandomState(3)
        y = self._make_y(n, rng)
        # test_set has only 6 rows for 'm', but mask is length n=12.
        test_set = pd.DataFrame({
            'model_type': ['m'] * 6,
            'risk_score': rng.randn(6),
        })
        mask = np.ones(n, dtype=bool)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", MethodologicalWarning)
            metric_rows, auc_t_rows = _compute_model_subpopulation_metrics(
                'm', test_set, mask, y[mask], None, None, None, 'lbl',
            )

        warns = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "Risk score length" in str(w.message)
        ]
        assert len(warns) == 1
        assert len(metric_rows) == 1
        row = metric_rows[0]
        assert row['subpopulation'] == 'lbl'
        assert row['model_type'] == 'm'
        assert row['metric'] == 'harrell'
        assert np.isnan(row['value'])
        assert auc_t_rows == []


# ---------------------------------------------------------------------------
# extract_subpopulation_bootstrap_summary
# ---------------------------------------------------------------------------

class TestExtractBootstrapSummary:
    def test_extracts_subpopulation_metrics(self):
        boot_summary = pd.DataFrame({
            'model_type': ['coxph', 'coxph', 'coxph'],
            'metric': ['harrell', 'harrell__subpopulation_A', 'harrell__subpopulation_B'],
            'point': [0.75, 0.72, 0.78],
            'mean': [0.74, 0.71, 0.77],
            'std': [0.02, 0.03, 0.04],
            'ci_lower': [0.70, 0.65, 0.70],
            'ci_upper': [0.78, 0.77, 0.84],
        })
        result = extract_subpopulation_bootstrap_summary(boot_summary)
        assert result is not None
        assert len(result) == 2
        assert set(result['subpopulation']) == {'A', 'B'}
        assert all(result['metric'] == 'harrell')

    def test_no_subpopulation_metrics(self):
        boot_summary = pd.DataFrame({
            'model_type': ['coxph'],
            'metric': ['harrell'],
            'point': [0.75],
        })
        result = extract_subpopulation_bootstrap_summary(boot_summary)
        assert result is None


# ---------------------------------------------------------------------------
# Integration: setup() with subpopulations
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSetupWithSubpopulations:
    def test_subpopulations_split_correctly(self, sample_data, subpopulation_labels):
        """Subpopulation labels are split into train/test along with X."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(X, y)
        learner.setup(
            verbose=False,
            subpopulations={'ct': subpopulation_labels},
            random_state=42,
        )

        assert learner._state.subpopulations_train is not None
        assert learner._state.subpopulations_test is not None
        assert len(learner._state.subpopulations_train['ct']) == len(learner.y_train)
        assert len(learner._state.subpopulations_test['ct']) == len(learner.y_test)

    def test_presplit_subpopulations(self, sample_data, subpopulation_labels):
        """Pre-split data uses explicit subpopulations_train/subpopulations_test."""
        X, y = sample_data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )

        n_train = len(y_train)
        n_test = len(y_test)
        rng = np.random.RandomState(42)

        learner.setup(
            verbose=False,
            subpopulations_train={'ct': rng.choice(['A', 'B'], n_train)},
            subpopulations_test={'ct': rng.choice(['A', 'B'], n_test)},
        )

        assert len(learner._state.subpopulations_train['ct']) == n_train
        assert len(learner._state.subpopulations_test['ct']) == n_test

    def test_presplit_rejects_subpopulations(self, sample_data):
        """Pre-split data should reject 'subpopulations' param."""
        X, y = sample_data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )

        with pytest.raises(ValueError, match="subpopulations_train"):
            learner.setup(
                verbose=False,
                subpopulations={'ct': np.array(['A'] * len(y_train))},
            )

    def test_length_mismatch_raises(self, sample_data):
        """Wrong-length subpopulation array should raise."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(X, y)

        with pytest.raises(ValueError, match="has length"):
            learner.setup(
                verbose=False,
                subpopulations={'ct': np.array(['A', 'B', 'C'])},
            )


# ---------------------------------------------------------------------------
# Integration: evaluate() auto-populates subpopulation_results
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEvaluateWithSubpopulations:
    def test_evaluate_populates_subpopulation_results(self, sample_data, subpopulation_labels):
        """evaluate() should auto-populate subpopulation_results when subpopulations set."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(X, y)
        learner.setup(
            verbose=False,
            subpopulations={'ct': subpopulation_labels},
            random_state=42,
        )
        learner.train('coxph', verbose=False)
        learner.evaluate(verbose=False)

        assert learner._state.subpopulation_results
        assert 'ct' in learner._state.subpopulation_results
        sg = learner._state.subpopulation_results['ct']
        assert isinstance(sg, SubpopulationResult)
        assert len(sg.metrics) > 0

    @pytest.mark.timeout(120)
    def test_bootstrap_includes_subpopulation_cis(self, sample_data, subpopulation_labels):
        """evaluate(bootstrap=True) should produce per-subpopulation CIs."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(X, y)
        learner.setup(
            verbose=False,
            subpopulations={'ct': subpopulation_labels},
            random_state=42,
        )
        learner.train('coxph', verbose=False)
        learner.evaluate(
            verbose=False,
            bootstrap=True,
            n_bootstrap=20,  # Small for speed
        )

        sg = learner._state.subpopulation_results['ct']
        assert sg.bootstrap_summary is not None
        assert len(sg.bootstrap_summary) > 0
        assert 'ci_lower' in sg.bootstrap_summary.columns

    @pytest.mark.timeout(120)
    def test_evaluate_without_bootstrap_preserves_subpopulation_cis(
        self, sample_data, subpopulation_labels,
    ):
        """evaluate(bootstrap=False) must not erase prior subpopulation bootstrap CIs."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(X, y)
        learner.setup(
            verbose=False,
            subpopulations={'ct': subpopulation_labels},
            random_state=42,
        )
        learner.train('coxph', verbose=False)

        # First: evaluate with bootstrap to populate CIs
        learner.evaluate(verbose=False, bootstrap=True, n_bootstrap=20)
        sg = learner._state.subpopulation_results['ct']
        assert sg.bootstrap_summary is not None
        original_ci = sg.bootstrap_summary.copy()

        # Second: evaluate without bootstrap (should NOT erase CIs)
        learner.evaluate(verbose=False, n_importance_repeats=0)
        sg_after = learner._state.subpopulation_results['ct']
        assert sg_after.bootstrap_summary is not None
        pd.testing.assert_frame_equal(sg_after.bootstrap_summary, original_ci)

    @pytest.mark.timeout(120)
    def test_new_bootstrap_overwrites_old_subpopulation_cis(
        self, sample_data, subpopulation_labels,
    ):
        """evaluate(bootstrap=True) must replace prior subpopulation CIs, not preserve stale ones."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(X, y)
        learner.setup(
            verbose=False,
            subpopulations={'ct': subpopulation_labels},
            random_state=42,
        )
        learner.train('coxph', verbose=False)

        # First bootstrap
        learner.evaluate(verbose=False, bootstrap=True, n_bootstrap=10)
        sg_first = learner._state.subpopulation_results['ct']
        assert sg_first.bootstrap_summary is not None

        # Second bootstrap (should replace, not preserve)
        learner.evaluate(verbose=False, bootstrap=True, n_bootstrap=10)
        sg_second = learner._state.subpopulation_results['ct']
        assert sg_second.bootstrap_summary is not None
        # CIs should come from the new bootstrap, not the old one
        # (different resamples → different summary stats)
        assert sg_second.bootstrap_summary is not sg_first.bootstrap_summary


# ---------------------------------------------------------------------------
# Report section
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSubpopulationAnalysisSection:
    def test_is_available_false_without_subpopulations(self, sample_data):
        """Section should not be available without subpopulations."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        from mlsurv.reports.sections.subpopulation_analysis import SubpopulationAnalysisSection
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        section = SubpopulationAnalysisSection(learner)
        assert not section.is_available()

    def test_is_available_with_subpopulations(self, sample_data, subpopulation_labels):
        """Section should be available when subpopulations are present."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        from mlsurv.reports.sections.subpopulation_analysis import SubpopulationAnalysisSection
        learner = SurvivalLearner(X, y)
        learner.setup(
            verbose=False,
            subpopulations={'ct': subpopulation_labels},
            random_state=42,
        )
        learner.train('coxph', verbose=False)
        learner.evaluate(verbose=False)

        section = SubpopulationAnalysisSection(learner)
        assert section.is_available()

    def test_generate_produces_markdown(self, sample_data, subpopulation_labels):
        """generate() should produce valid markdown."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        from mlsurv.reports.sections.subpopulation_analysis import SubpopulationAnalysisSection
        learner = SurvivalLearner(X, y)
        learner.setup(
            verbose=False,
            subpopulations={'ct': subpopulation_labels},
            random_state=42,
        )
        learner.train('coxph', verbose=False)
        learner.evaluate(verbose=False)

        section = SubpopulationAnalysisSection(learner)
        result = section.generate()

        assert 'markdown' in result
        assert 'ct' in result['markdown']
        assert 'Subpopulation' in result['markdown']
        assert 'figures' in result
        # Verify cohort summary table is present
        assert 'feature-summary' in result['markdown']


# ---------------------------------------------------------------------------
# Warning consolidation and grouped table rendering
# ---------------------------------------------------------------------------

class TestWarningConsolidation:
    """Warnings should render as a single callout block, not per-warning."""

    @staticmethod
    def _make_section_and_render(warnings_list):
        """Helper: render a SubpopulationResult with given warnings into md_lines."""
        import logging
        from unittest.mock import MagicMock
        from mlsurv.reports.sections.subpopulation_analysis import (
            SubpopulationAnalysisSection,
        )
        sg_result = SubpopulationResult(
            group_variable='grp',
            summary=pd.DataFrame({
                'subpopulation': ['A', 'B'],
                'n': [50, 50],
                'events': [20, 20],
                'event_rate': [0.4, 0.4],
            }),
            metrics=pd.DataFrame({
                'subpopulation': ['A', 'B'],
                'model_type': ['coxph', 'coxph'],
                'metric': ['harrell', 'harrell'],
                'value': [0.7, 0.6],
            }),
            warnings=warnings_list,
        )
        # Create a minimal section instance with mock learner
        section = SubpopulationAnalysisSection.__new__(SubpopulationAnalysisSection)
        section.data = {}
        section.learner = MagicMock()
        section.learner.scoring = 'harrell'
        section._logger = logging.getLogger('test')

        md_lines = []
        figures = []
        figure_infos = []
        data_infos = []
        section._render_subpopulation(
            'grp', sg_result, 95,
            md_lines, figures, figure_infos, data_infos,
        )
        return '\n'.join(md_lines)

    def test_no_warnings_no_callout(self):
        """No callout rendered when warnings list is empty."""
        output = self._make_section_and_render([])
        assert 'mlsurv-callout-info' not in output

    def test_single_warning_consolidated(self):
        """Single warning still uses consolidated block."""
        output = self._make_section_and_render(["Subpopulation 'A' has only 3 events"])
        assert output.count('mlsurv-callout-info') == 1
        assert '<li>' in output
        assert "Subpopulation &#x27;A&#x27; has only 3 events" in output

    def test_multiple_warnings_single_block(self):
        """Multiple warnings render in one callout block, not separate ones."""
        warnings_list = [
            "Subpopulation 'A' has only 3 events (< 30).",
            "Subpopulation 'B' has only 5 events (< 30).",
        ]
        output = self._make_section_and_render(warnings_list)
        # Single consolidated callout (not multiple blockquotes)
        assert output.count('mlsurv-callout-info') == 1
        # Both warnings present as list items
        assert output.count('<li>') == 2
        assert 'Subpopulation &#x27;A&#x27;' in output
        assert 'Subpopulation &#x27;B&#x27;' in output

    def test_html_escaped(self):
        """Warning text is HTML-escaped."""
        output = self._make_section_and_render(["Value <script>alert(1)</script>"])
        assert '<script>' not in output
        assert '&lt;script&gt;' in output


# ---------------------------------------------------------------------------
# SubpopulationResult container
# ---------------------------------------------------------------------------

class TestSubpopulationResult:
    def test_repr(self):
        sg = SubpopulationResult(
            group_variable='ct',
            summary=pd.DataFrame({'subpopulation': ['A', 'B']}),
            metrics=pd.DataFrame(),
        )
        assert "ct" in repr(sg)
        assert "n_subpopulations=2" in repr(sg)

    def test_warnings_default(self):
        sg = SubpopulationResult(
            group_variable='ct',
            summary=pd.DataFrame(),
            metrics=pd.DataFrame(),
        )
        assert sg.warnings == []


# ---------------------------------------------------------------------------
# Train/test split stratification by subpopulation
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSplitStratificationBySubpopulation:
    def test_subpopulation_proportions_preserved(self, sample_data, subpopulation_labels):
        """Subpopulation proportions should be approximately preserved in train/test."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner

        learner = SurvivalLearner(X, y)
        learner.setup(
            verbose=False,
            subpopulations={'ct': subpopulation_labels},
            stratify_subpopulations=True,
            random_state=42,
        )

        # Compare proportions
        full_counts = pd.Series(subpopulation_labels).value_counts(normalize=True)
        test_counts = pd.Series(
            learner._state.subpopulations_test['ct']
        ).value_counts(normalize=True)

        for label in full_counts.index:
            diff = abs(full_counts[label] - test_counts.get(label, 0))
            assert diff < 0.1, (
                f"Subpopulation '{label}' proportion differs by {diff:.2%}: "
                f"full={full_counts[label]:.2%}, test={test_counts.get(label, 0):.2%}"
            )

    def test_small_cell_fallback(self, sample_data):
        """When an (event x subpopulation) cell has < 2 samples and the merged
        __other__ bucket also fails, fall back to event-only with a
        MethodologicalWarning that names the merged levels."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        from mlsurv._warnings import MethodologicalWarning

        n = len(X)
        # Single sample in 'Rare' — even after merge, __other__ has only 1 sample
        labels = np.array(['Common'] * (n - 1) + ['Rare'])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner = SurvivalLearner(X, y)
            learner.setup(
                verbose=False,
                subpopulations={'ct': labels},
                stratify_subpopulations=True,
                random_state=42,
            )

        fallback = [
            rec for rec in w
            if issubclass(rec.category, MethodologicalWarning)
            and "Falling back to event-only" in str(rec.message)
            and "'Rare'" in str(rec.message)
            and "'__other__'" in str(rec.message)
        ]
        assert fallback, (
            "Expected MethodologicalWarning naming the merged 'Rare' level and "
            f"'__other__'; got: {[str(r.message) for r in w]}"
        )
        assert learner._state.is_setup

    def test_presplit_skips_split_stratification(self, sample_data, subpopulation_labels):
        """Pre-split data should not augment stratification (no split to stratify)."""
        X, y = sample_data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        from mlsurv.learner import SurvivalLearner

        rng = np.random.RandomState(42)
        learner = SurvivalLearner(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )
        learner.setup(
            verbose=False,
            subpopulations_train={'ct': rng.choice(['A', 'B'], len(y_train))},
            subpopulations_test={'ct': rng.choice(['A', 'B'], len(y_test))},
            stratify_subpopulations=True,
        )
        assert learner._state.is_setup
        assert learner._state.subpopulations_test is not None

    def test_stratify_subpopulations_false_skips(self, sample_data, subpopulation_labels):
        """stratify_subpopulations=False should not augment split stratification."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner

        learner = SurvivalLearner(X, y)
        learner.setup(
            verbose=False,
            subpopulations={'ct': subpopulation_labels},
            stratify_subpopulations=False,
            random_state=42,
        )
        # Setup should succeed without subpopulation-augmented stratification
        assert learner._state.is_setup
        assert learner._state.subpopulations_test is not None


# ---------------------------------------------------------------------------
# Rare-level collapse helper
# ---------------------------------------------------------------------------

class TestCollapseRareLevels:
    """Unit tests for _collapse_rare_levels — covers the three branches:
    no merge / partial merge / full fallback."""

    def test_all_levels_large_no_merge(self):
        from mlsurv.learner._setup import _collapse_rare_levels

        events = np.array([0]*10 + [1]*10 + [0]*10 + [1]*10)
        sg = np.array(['A']*10 + ['A']*10 + ['B']*10 + ['B']*10)

        combined, merged, failed = _collapse_rare_levels(events, sg, threshold=5)

        assert merged == []
        assert failed is False
        assert set(np.unique(combined)) == {'0_A', '1_A', '0_B', '1_B'}

    def test_partial_merge_into_other(self):
        from mlsurv.learner._setup import _collapse_rare_levels

        # A is large; B, C, D are each below threshold but together cross it
        events = np.array([0]*10 + [1]*10 + [0,1]*3 + [0,1]*3 + [0,1]*3)
        sg = np.array(['A']*10 + ['A']*10 + ['B']*6 + ['C']*6 + ['D']*6)

        combined, merged, failed = _collapse_rare_levels(events, sg, threshold=5)

        assert sorted(merged) == ['B', 'C', 'D']
        assert failed is False
        # Dominant level preserved, rare levels merged into __other__
        assert set(np.unique(combined)) == {'0_A', '1_A', '0___other__', '1___other__'}
        # Counts: __other__ now has 9 of each event (3 levels x 3 each)
        # spot check
        assert int((combined == '0___other__').sum()) == 9
        assert int((combined == '1___other__').sum()) == 9

    def test_collapse_also_fails_full_fallback(self):
        from mlsurv.learner._setup import _collapse_rare_levels

        # Only one rare sample — even __other__ can't satisfy threshold=2
        events = np.array([0]*10 + [1]*10 + [0])
        sg = np.array(['A']*10 + ['A']*10 + ['B'])

        combined, merged, failed = _collapse_rare_levels(events, sg, threshold=2)

        assert merged == ['B']
        assert failed is True
        assert combined is None


@pytest.mark.integration
class TestSetupAugmentStratify:
    """Integration check: setup() emits SuggestionWarning (not MethodologicalWarning)
    when only the rare levels need merging."""

    def test_partial_merge_emits_suggestion_warning(self, sample_data):
        from mlsurv.learner import SurvivalLearner
        from mlsurv._warnings import SuggestionWarning, MethodologicalWarning

        X, y = sample_data
        n = len(X)
        events = np.asarray(y['event']).astype(bool)
        event_idx = np.flatnonzero(events)
        censored_idx = np.flatnonzero(~events)

        # Deliberately balance rare groups across event status so that the
        # collapsed `__other__` bucket clears both thresholds (split: 2,
        # CV with cv_folds=2: 2). Each rare group itself fails (only 1 of
        # each event class) so all three must be merged.
        labels = np.array(['Common'] * n, dtype=object)
        for k, (e_pick, c_pick) in enumerate([
            (event_idx[0:1], censored_idx[0:1]),
            (event_idx[1:2], censored_idx[1:2]),
            (event_idx[2:3], censored_idx[2:3]),
        ]):
            labels[e_pick] = f'Rare{k+1}'
            labels[c_pick] = f'Rare{k+1}'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner = SurvivalLearner(X, y)
            learner.setup(
                verbose=False,
                subpopulations={'ct': labels},
                stratify_subpopulations=True,
                cv_folds=2,
                random_state=42,
            )

        suggestion_msgs = [
            str(rec.message) for rec in w
            if issubclass(rec.category, SuggestionWarning)
            and "merged into '__other__'" in str(rec.message)
        ]
        methodological_fallback = [
            str(rec.message) for rec in w
            if issubclass(rec.category, MethodologicalWarning)
            and "Falling back to event-only" in str(rec.message)
        ]
        # Both call sites (split path and CV path) should emit a partial-merge
        # SuggestionWarning naming all three rare levels.
        split_msgs = [m for m in suggestion_msgs if "train/test split" in m]
        cv_msgs = [m for m in suggestion_msgs if "for CV stratification" in m]
        assert split_msgs, (
            f"Expected split-path partial-merge SuggestionWarning; got: {suggestion_msgs}"
        )
        assert cv_msgs, (
            f"Expected CV-path partial-merge SuggestionWarning; got: {suggestion_msgs}"
        )
        for msg in suggestion_msgs:
            assert "'Rare1'" in msg and "'Rare2'" in msg and "'Rare3'" in msg, (
                f"Expected all rare levels in merged list; got: {msg}"
            )
        assert not methodological_fallback, (
            f"Did not expect full event-only fallback; got: {methodological_fallback}"
        )
        assert learner._state.is_setup

    def test_cv_path_full_fallback_via_presplit(self, sample_data):
        """When CV stratification can't be saved by merging (presplit so the
        split-path is skipped), the CV path emits MethodologicalWarning."""
        from mlsurv.learner import SurvivalLearner
        from mlsurv._warnings import MethodologicalWarning
        from sklearn.model_selection import train_test_split

        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Train labels: dominant 'Common' + a single 'Rare' sample. Even after
        # merging Rare into __other__, the bucket has only 1 sample so the CV
        # threshold (cv_folds=5) can't be met.
        n_train = len(y_train)
        train_labels = np.array(['Common'] * (n_train - 1) + ['Rare'], dtype=object)
        test_labels = np.array(['Common'] * len(y_test), dtype=object)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner = SurvivalLearner(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
            )
            learner.setup(
                verbose=False,
                subpopulations_train={'ct': train_labels},
                subpopulations_test={'ct': test_labels},
                stratify_subpopulations=True,
                cv_folds=5,
                random_state=42,
            )

        cv_fallback = [
            rec for rec in w
            if issubclass(rec.category, MethodologicalWarning)
            and "Falling back to event-only stratification for CV" in str(rec.message)
            and "cv_folds=5" in str(rec.message)
        ]
        assert cv_fallback, (
            f"Expected CV-path MethodologicalWarning; got: {[str(r.message) for r in w]}"
        )
        assert learner._state.is_setup


# ---------------------------------------------------------------------------
# Per-subpopulation multi-metric (IPCW + AUC(t))
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMultiMetricSubpopulations:
    def test_compute_subpopulation_metrics_with_ytrain(self, sample_data, subpopulation_labels):
        """compute_subpopulation_metrics with y_train produces IPCW metrics."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False, random_state=42)
        learner.train('coxph', verbose=False)
        result = learner.evaluate(verbose=False)

        test_positions = X.index.get_indexer(learner.X_test.index)
        test_labels = subpopulation_labels[test_positions]

        sg_results = compute_subpopulation_metrics(
            result,
            {'ct': test_labels},
            learner.y_test,
            y_train=learner.y_train,
        )

        sg = sg_results['ct']
        available = set(sg.metrics['metric'].unique())
        assert 'harrell' in available
        assert 'ipcw' in available

    def test_compute_subpopulation_metrics_with_auc_t(self, sample_data, subpopulation_labels):
        """compute_subpopulation_metrics with y_train + eval_times produces auc_t."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        from mlsurv.utils import auto_eval_times
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False, random_state=42)
        learner.train('coxph', verbose=False)
        result = learner.evaluate(verbose=False)

        test_positions = X.index.get_indexer(learner.X_test.index)
        test_labels = subpopulation_labels[test_positions]

        eval_times = auto_eval_times(learner.y_test, y_train=learner.y_train)
        sg_results = compute_subpopulation_metrics(
            result,
            {'ct': test_labels},
            learner.y_test,
            y_train=learner.y_train,
            eval_times=eval_times,
        )

        sg = sg_results['ct']
        assert sg.auc_t is not None
        assert len(sg.auc_t) > 0
        assert set(sg.auc_t.columns) == {'subpopulation', 'model_type', 'eval_time', 'auc'}

    def test_evaluate_auto_computes_ipcw(self, sample_data, subpopulation_labels):
        """evaluate() should auto-compute IPCW for subpopulations."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(X, y)
        learner.setup(
            verbose=False,
            subpopulations={'ct': subpopulation_labels},
            random_state=42,
        )
        learner.train('coxph', verbose=False)
        learner.evaluate(verbose=False)

        sg = learner._state.subpopulation_results['ct']
        available = set(sg.metrics['metric'].unique())
        assert 'ipcw' in available

    def test_extract_multi_metric_bootstrap(self):
        """extract_subpopulation_bootstrap_summary handles multiple metrics."""
        boot_summary = pd.DataFrame({
            'model_type': ['coxph'] * 5,
            'metric': [
                'harrell', 'harrell__subpopulation_A', 'ipcw__subpopulation_A',
                'harrell__subpopulation_B', 'ipcw__subpopulation_B',
            ],
            'point': [0.75, 0.72, 0.70, 0.78, 0.76],
            'mean': [0.74, 0.71, 0.69, 0.77, 0.75],
            'std': [0.02, 0.03, 0.03, 0.04, 0.04],
            'ci_lower': [0.70, 0.65, 0.63, 0.70, 0.68],
            'ci_upper': [0.78, 0.77, 0.75, 0.84, 0.82],
        })
        result = extract_subpopulation_bootstrap_summary(boot_summary)
        assert result is not None
        assert len(result) == 4  # 2 subpopulations x 2 metrics
        assert set(result['metric'].unique()) == {'harrell', 'ipcw'}
        assert set(result['subpopulation'].unique()) == {'A', 'B'}


# ---------------------------------------------------------------------------
# Benchmark subpopulation integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBenchmarkSubpopulations:
    @pytest.mark.timeout(120)
    def test_benchmark_populates_subpopulation_incremental(self, sample_data, subpopulation_labels):
        """benchmark() with subpopulations should populate subpopulation_incremental."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(X, y)
        learner.setup(
            verbose=False,
            subpopulations={'ct': subpopulation_labels},
            random_state=42,
        )
        learner.train('coxph', verbose=False)

        # Create reference scores
        rng = np.random.RandomState(42)
        ref_scores = rng.randn(len(y))

        result = learner.benchmark(
            references={'ref': ref_scores},
            bootstrap=False,
            verbose=False,
        )

        assert result.subpopulation_incremental is not None
        assert len(result.subpopulation_incremental) > 0
        assert 'subpopulation' in result.subpopulation_incremental.columns
        assert 'group_variable' in result.subpopulation_incremental.columns

    @pytest.mark.timeout(120)
    def test_benchmark_populates_subpopulation_results(self, sample_data, subpopulation_labels):
        """benchmark() should populate subpopulation_results (C-index for models + refs)."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(X, y)
        learner.setup(
            verbose=False,
            subpopulations={'ct': subpopulation_labels},
            random_state=42,
        )
        learner.train('coxph', verbose=False)

        rng = np.random.RandomState(42)
        ref_scores = rng.randn(len(y))

        learner.benchmark(
            references={'ref': ref_scores},
            bootstrap=False,
            verbose=False,
        )

        # subpopulation_results should include both model AND reference
        assert learner._state.subpopulation_results
        sg = learner._state.subpopulation_results['ct']
        model_types = set(sg.metrics['model_type'].unique())
        assert 'coxph' in model_types
        assert 'ref' in model_types


# ---------------------------------------------------------------------------
# _safe_ibs unit tests
# ---------------------------------------------------------------------------

class TestSafeIbs:
    def test_returns_float(self):
        """_safe_ibs returns a float in (0, 1) for valid synthetic data."""
        rng = np.random.RandomState(42)
        n_train, n_test = 100, 50
        n_times = 10

        y_train = make_survival_array(
            rng.binomial(1, 0.6, n_train), rng.exponential(10, n_train)
        )
        y_test = make_survival_array(
            rng.binomial(1, 0.6, n_test), rng.exponential(10, n_test)
        )
        eval_times = np.linspace(1, 8, n_times)
        # Decreasing survival probs over time
        surv_probs = np.column_stack([
            np.clip(1 - eval_times[j] / 15 + rng.randn(n_test) * 0.05, 0.01, 0.99)
            for j in range(n_times)
        ])

        result = _safe_ibs(y_test, surv_probs, eval_times, y_train)
        assert isinstance(result, float)
        assert 0 < result < 1

    def test_returns_nan_on_failure(self):
        """_safe_ibs returns NaN when given invalid inputs."""
        # Empty arrays should cause failure
        y_empty = np.array([], dtype=[('event', bool), ('time', float)])
        y_train = make_survival_array([True, False], [1.0, 2.0])
        result = _safe_ibs(y_empty, np.array([]).reshape(0, 2),
                           np.array([1.0, 2.0]), y_train)
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# IBS in compute_subpopulation_metrics
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSubpopulationIBS:
    def test_compute_subpopulation_metrics_with_ibs(self, sample_data, subpopulation_labels):
        """When surv_probs_by_model is passed, IBS appears in metrics."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False, random_state=42)
        learner.train('coxph', verbose=False)
        result = learner.evaluate(verbose=False)

        test_positions = X.index.get_indexer(learner.X_test.index)
        test_labels = subpopulation_labels[test_positions]

        # Extract cached surv_probs
        surv_cache = getattr(result, '_surv_probs_cache', None)
        assert surv_cache is not None, "surv_probs_cache should be set after evaluate()"
        # IBS is now computed over the dense, train-anchored grid, so the
        # subpopulation path needs the dense cache (not the display surv_cache).
        ibs_cache = getattr(result, '_ibs_surv_probs_cache', None)
        assert ibs_cache is not None, "_ibs_surv_probs_cache should be set after evaluate()"

        from mlsurv.utils import auto_eval_times
        et = auto_eval_times(learner.y_test, y_train=learner.y_train)

        sg_results = compute_subpopulation_metrics(
            result, {'ct': test_labels}, learner.y_test,
            y_train=learner.y_train,
            eval_times=et,
            surv_probs_by_model=surv_cache,
            dense_surv_probs_by_model=ibs_cache,
        )

        sg = sg_results['ct']
        available = set(sg.metrics['metric'].unique())
        assert 'integrated_brier' in available
        # IBS values should be in (0, 1) range
        ibs_vals = sg.metrics[sg.metrics['metric'] == 'integrated_brier']['value']
        for v in ibs_vals:
            if not np.isnan(v):
                assert 0 < v < 1, f"IBS value {v} out of expected range"

    def test_backward_compat_no_surv_probs(self, sample_data, subpopulation_labels):
        """Without surv_probs_by_model, only harrell/ipcw are computed (no IBS)."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False, random_state=42)
        learner.train('coxph', verbose=False)
        result = learner.evaluate(verbose=False)

        test_positions = X.index.get_indexer(learner.X_test.index)
        test_labels = subpopulation_labels[test_positions]

        sg_results = compute_subpopulation_metrics(
            result, {'ct': test_labels}, learner.y_test,
            y_train=learner.y_train,
            # surv_probs_by_model NOT passed
        )

        sg = sg_results['ct']
        available = set(sg.metrics['metric'].unique())
        assert 'harrell' in available
        assert 'ipcw' in available
        assert 'integrated_brier' not in available

    def test_surv_probs_cache_populated(self, sample_data):
        """evaluate() populates _surv_probs_cache on EvaluationResult."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False, random_state=42)
        learner.train('coxph', verbose=False)
        result = learner.evaluate(verbose=False)

        cache = getattr(result, '_surv_probs_cache', None)
        assert cache is not None
        assert 'coxph' in cache
        surv_probs, eval_times = cache['coxph']
        assert surv_probs.shape[0] == len(learner.y_test)
        assert len(eval_times) == surv_probs.shape[1]

    def test_evaluate_auto_computes_ibs(self, sample_data, subpopulation_labels):
        """evaluate() with subpopulations auto-computes IBS when surv_probs available."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(X, y)
        learner.setup(
            verbose=False,
            subpopulations={'ct': subpopulation_labels},
            random_state=42,
        )
        learner.train('coxph', verbose=False)
        learner.evaluate(verbose=False)

        sg = learner._state.subpopulation_results['ct']
        available = set(sg.metrics['metric'].unique())
        assert 'integrated_brier' in available


# ---------------------------------------------------------------------------
# Bootstrap summary filter fix
# ---------------------------------------------------------------------------

class TestBootstrapSummaryFilter:
    def test_summary_includes_all_subpopulation_metrics(self):
        """_compute_summary_statistics should include ipcw/ibs subpopulation metrics."""
        from mlsurv.evaluation.bootstrap import _compute_summary_statistics

        # Mock iterations with multiple subpopulation metric types
        rows = []
        for i in range(10):
            for metric in ['harrell', 'harrell__subpopulation_A', 'ipcw__subpopulation_A',
                           'integrated_brier__subpopulation_A']:
                rows.append({
                    'iteration': i,
                    'metric_name': metric,
                    'value': np.random.RandomState(i).uniform(0.5, 0.9),
                    'model_type': 'coxph',
                })
        iterations_df = pd.DataFrame(rows)

        summary = _compute_summary_statistics(
            iterations_df,
            metrics=['harrell'],
            custom_metrics={},
            model_name='coxph',
            ci=0.95,
        )

        summary_metrics = set(summary['metric'].values)
        assert 'harrell__subpopulation_A' in summary_metrics
        assert 'ipcw__subpopulation_A' in summary_metrics
        assert 'integrated_brier__subpopulation_A' in summary_metrics


# ---------------------------------------------------------------------------
# Subpopulation AUC(t) bootstrap CI ribbons
# ---------------------------------------------------------------------------

class TestSubpopulationAucBootstrapHelper:
    """Unit tests for _compute_subpopulation_auc_for_iter."""

    def test_correct_row_structure(self):
        """Returned rows have correct columns."""
        from mlsurv.evaluation._bootstrap_iteration import _compute_subpopulation_auc_for_iter

        rng = np.random.RandomState(42)
        n = 50
        y_train = np.array(
            [(bool(rng.rand() > 0.5), rng.exponential(10)) for _ in range(n)],
            dtype=[('event', bool), ('time', float)],
        )
        y_sub = np.array(
            [(bool(rng.rand() > 0.5), rng.exponential(10)) for _ in range(20)],
            dtype=[('event', bool), ('time', float)],
        )
        risk_sub = rng.randn(20)
        eval_times = np.array([2.0, 4.0, 6.0, 8.0])

        rows = _compute_subpopulation_auc_for_iter(
            y_sub, risk_sub, y_train, eval_times,
            group_name='TypeA', model_name='coxph', iteration=0,
        )

        assert len(rows) == len(eval_times)
        for row in rows:
            assert set(row.keys()) == {'subpopulation', 'model_type', 'iteration', 'eval_time', 'auc'}
            assert row['subpopulation'] == 'TypeA'
            assert row['model_type'] == 'coxph'
            assert row['iteration'] == 0

    def test_edge_case_too_few_samples(self):
        """With <2 samples, should still produce NaN rows gracefully."""
        from mlsurv.evaluation._bootstrap_iteration import _compute_subpopulation_auc_for_iter

        y_train = make_survival_array([True, False, True], [1.0, 2.0, 3.0])
        # Only 1 sample — too few for AUC
        y_sub = make_survival_array([True], [1.5])
        risk_sub = np.array([0.5])
        eval_times = np.array([1.0, 2.0])

        rows = _compute_subpopulation_auc_for_iter(
            y_sub, risk_sub, y_train, eval_times,
            group_name='Tiny', model_name='coxph', iteration=0,
        )

        assert len(rows) == len(eval_times)
        for row in rows:
            assert np.isnan(row['auc'])

    def test_nan_for_out_of_range_times(self):
        """Eval times outside subpopulation range should get NaN."""
        from mlsurv.evaluation._bootstrap_iteration import _compute_subpopulation_auc_for_iter

        rng = np.random.RandomState(42)
        n = 50
        y_train = np.array(
            [(bool(rng.rand() > 0.5), rng.uniform(1, 20)) for _ in range(n)],
            dtype=[('event', bool), ('time', float)],
        )
        # Subpopulation times: 2-8
        y_sub = np.array(
            [(bool(rng.rand() > 0.5), rng.uniform(2, 8)) for _ in range(20)],
            dtype=[('event', bool), ('time', float)],
        )
        risk_sub = rng.randn(20)
        # Include times well outside subpopulation range
        eval_times = np.array([0.5, 3.0, 5.0, 7.0, 15.0])

        rows = _compute_subpopulation_auc_for_iter(
            y_sub, risk_sub, y_train, eval_times,
            group_name='MidRange', model_name='coxph', iteration=0,
        )

        assert len(rows) == len(eval_times)
        # Times 0.5 and 15.0 are outside the [min, max] of subpopulation times
        row_by_time = {r['eval_time']: r for r in rows}
        assert np.isnan(row_by_time[0.5]['auc'])
        assert np.isnan(row_by_time[15.0]['auc'])


class TestBootstrapResultSubpopulationAucT:
    """Test that BootstrapResult.subpopulation_auc_t field works correctly."""

    def test_to_dataframes_includes_subpopulation_auc(self):
        """to_dataframes() should export subpopulation_auc_t."""
        from mlsurv.results.containers import BootstrapResult

        sg_auc = pd.DataFrame({
            'subpopulation': ['A', 'B'],
            'model_type': ['coxph', 'coxph'],
            'iteration': [0, 0],
            'eval_time': [1.0, 1.0],
            'auc': [0.7, 0.6],
        })
        br = BootstrapResult(
            summary=pd.DataFrame({'model_type': ['coxph'], 'metric': ['harrell'],
                                  'point': [0.7], 'mean': [0.7], 'std': [0.01],
                                  'ci_lower': [0.68], 'ci_upper': [0.72]}),
            iterations=pd.DataFrame({'model_type': ['coxph'], 'iteration': [0],
                                     'metric_name': ['harrell'], 'value': [0.7]}),
            subpopulation_auc_t=sg_auc,
        )
        dfs = br.to_dataframes()
        assert 'subpopulation_auc_over_time' in dfs
        assert len(dfs['subpopulation_auc_over_time']) == 2


class TestToDataframesSubpopulationFilter:
    """Test the fixed subpopulation metric filter in to_dataframes."""

    def test_filters_all_subpopulation_metric_types(self):
        """to_dataframes should filter ipcw__subpopulation_* and ibs__subpopulation_* too."""
        from mlsurv.results.containers import BootstrapResult

        summary = pd.DataFrame({
            'model_type': ['coxph'] * 4,
            'metric': ['harrell', 'harrell__subpopulation_A',
                        'ipcw__subpopulation_A', 'integrated_brier__subpopulation_A'],
            'point': [0.7, 0.65, 0.68, 0.2],
            'mean': [0.7, 0.65, 0.68, 0.2],
            'std': [0.01] * 4,
            'ci_lower': [0.68, 0.63, 0.66, 0.18],
            'ci_upper': [0.72, 0.67, 0.70, 0.22],
        })
        iterations = pd.DataFrame({
            'model_type': ['coxph'] * 4,
            'iteration': [0] * 4,
            'metric_name': ['harrell', 'harrell__subpopulation_A',
                            'ipcw__subpopulation_A', 'integrated_brier__subpopulation_A'],
            'value': [0.7, 0.65, 0.68, 0.2],
        })
        br = BootstrapResult(summary=summary, iterations=iterations)
        dfs = br.to_dataframes()

        # Only 'harrell' should remain after filtering
        assert list(dfs['summary']['metric']) == ['harrell']
        assert list(dfs['iterations']['metric_name']) == ['harrell']


@pytest.mark.integration
class TestBootstrapSubpopulationAucIntegration:
    """Integration: bootstrap with subpopulations populates subpopulation_auc_t."""

    @pytest.mark.timeout(120)
    def test_bootstrap_populates_subpopulation_auc_t(self, sample_data, subpopulation_labels):
        """evaluate(bootstrap=True) with subpopulations should populate auc_t_bootstrap."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(X, y)
        learner.setup(
            verbose=False,
            subpopulations={'ct': subpopulation_labels},
            random_state=42,
        )
        learner.train('coxph', verbose=False)
        learner.evaluate(
            verbose=False,
            bootstrap=True,
            n_bootstrap=15,
        )

        # Check BootstrapResult has subpopulation_auc_t
        # evaluation_result is now a computed property combining per-model data
        boot = learner.evaluation_result.bootstrap
        assert boot is not None
        assert boot.subpopulation_auc_t is not None
        assert len(boot.subpopulation_auc_t) > 0
        expected_cols = {'subpopulation', 'model_type', 'iteration', 'eval_time', 'auc'}
        assert expected_cols.issubset(set(boot.subpopulation_auc_t.columns))

        # Check SubpopulationResult has auc_t_bootstrap
        sg = learner._state.subpopulation_results['ct']
        assert sg.auc_t_bootstrap is not None
        assert len(sg.auc_t_bootstrap) > 0

        # All subpopulations in bootstrap data should be from this group variable
        boot_subpopulations = set(sg.auc_t_bootstrap['subpopulation'].unique())
        point_subpopulations = set(str(s) for s in sg.summary['subpopulation'].unique())
        assert boot_subpopulations.issubset(point_subpopulations)


class TestSubpopulationAucTPlotSizing:
    """Tests for adaptive panel sizing in subpopulation AUC(t) plots."""

    def test_height_for_all_layout_sizes(self):
        """Verify exact heights for common subpopulation counts."""
        expected = {1: 460, 2: 460, 3: 820, 4: 820, 5: 1180, 6: 1180}
        for n_sg, want in expected.items():
            n_cols = min(2, n_sg)
            n_rows = (n_sg + n_cols - 1) // n_cols
            fig_height = max(450, n_rows * 360 + 100)
            assert fig_height == want, f"n_sg={n_sg}: got {fig_height}, want {want}"

    def test_fig_height_minimum_floor(self):
        """Single-row layouts (1-2 subpopulations) hit the 450px floor."""
        for n_sg in (1, 2):
            n_cols = min(2, n_sg)
            n_rows = (n_sg + n_cols - 1) // n_cols
            assert n_rows == 1
            fig_height = max(450, n_rows * 360 + 100)
            assert fig_height == 460

    def test_fig_height_scales_with_rows(self):
        """Multi-row layouts scale above the floor."""
        n_sg = 6
        n_cols = min(2, n_sg)
        n_rows = (n_sg + n_cols - 1) // n_cols
        assert n_rows == 3
        fig_height = max(450, n_rows * 360 + 100)
        assert fig_height == 1180

    def test_figure_width_two_columns(self):
        """2-column layout should use wider figure (900px)."""
        n_sg = 4
        n_cols = min(2, n_sg)
        fig_width = 900 if n_cols == 2 else 550
        assert fig_width == 900

    def test_figure_width_single_column(self):
        """1-column layout should use narrower figure (550px)."""
        n_sg = 1
        n_cols = min(2, n_sg)
        fig_width = 900 if n_cols == 2 else 550
        assert fig_width == 550


# ---------------------------------------------------------------------------
# Merge on re-evaluate
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSubpopulationMergeOnReEvaluate:
    """Single-model re-evaluate must not discard other models' subpopulation data."""

    @pytest.fixture
    def learner_two_models(self, sample_data, subpopulation_labels):
        """Train coxph + rsf with subpopulations, evaluate with bootstrap."""
        X, y = sample_data
        from mlsurv.learner import SurvivalLearner
        learner = SurvivalLearner(X, y)
        learner.setup(
            verbose=False,
            subpopulations={'ct': subpopulation_labels},
            random_state=42,
        )
        learner.train('coxph', verbose=False)
        learner.train('rsf', verbose=False)
        learner.evaluate(
            verbose=False, bootstrap=True, n_bootstrap=5,
        )
        return learner

    @pytest.mark.timeout(180)
    def test_single_model_reevaluate_preserves_all_models(
        self, learner_two_models,
    ):
        """evaluate(model='rsf') must keep coxph data in subpopulation metrics."""
        learner = learner_two_models
        sg = learner._state.subpopulation_results['ct']
        models_before = set(sg.metrics['model_type'].unique())
        assert models_before == {'coxph', 'rsf'}

        # Re-evaluate only RSF (e.g. to add SHAP interactions)
        learner.evaluate(models='rsf', verbose=False, n_importance_repeats=0)

        sg_after = learner._state.subpopulation_results['ct']
        models_after = set(sg_after.metrics['model_type'].unique())
        assert models_after == {'coxph', 'rsf'}, (
            f"Expected both models, got {models_after}"
        )

    @pytest.mark.timeout(180)
    def test_bootstrap_ci_preserved_after_reevaluate(
        self, learner_two_models,
    ):
        """CIs for non-re-evaluated models must survive single-model evaluate."""
        learner = learner_two_models
        sg = learner._state.subpopulation_results['ct']
        assert sg.bootstrap_summary is not None
        # CIs exist for both models
        ci_models_before = set(sg.bootstrap_summary['model_type'].unique())
        assert 'coxph' in ci_models_before

        # Re-evaluate only RSF without bootstrap
        learner.evaluate(models='rsf', verbose=False, n_importance_repeats=0)

        sg_after = learner._state.subpopulation_results['ct']
        assert sg_after.bootstrap_summary is not None
        ci_models_after = set(sg_after.bootstrap_summary['model_type'].unique())
        assert 'coxph' in ci_models_after, (
            f"CoxPH CIs lost after RSF re-evaluate; models: {ci_models_after}"
        )
        # RSF CIs are also preserved — no new bootstrap was produced,
        # so old CIs remain until explicitly replaced by a new bootstrap.
        assert 'rsf' in ci_models_after, (
            f"RSF CIs lost after RSF re-evaluate; models: {ci_models_after}"
        )

    @pytest.mark.timeout(180)
    def test_auc_t_preserved_after_reevaluate(
        self, learner_two_models,
    ):
        """AUC(t) for all models must survive single-model re-evaluate."""
        learner = learner_two_models
        sg = learner._state.subpopulation_results['ct']
        if sg.auc_t is None:
            pytest.skip("AUC(t) not computed (eval_times may be unavailable)")
        auc_models_before = set(sg.auc_t['model_type'].unique())
        assert len(auc_models_before) == 2

        learner.evaluate(models='rsf', verbose=False, n_importance_repeats=0)

        sg_after = learner._state.subpopulation_results['ct']
        assert sg_after.auc_t is not None
        auc_models_after = set(sg_after.auc_t['model_type'].unique())
        assert auc_models_after == auc_models_before, (
            f"AUC(t) models changed: {auc_models_before} → {auc_models_after}"
        )


# ---------------------------------------------------------------------------
# Subpopulation AUC(t) plot rendering
# ---------------------------------------------------------------------------

class TestSubpopulationAucTPlot:
    """Forward-looking invariant tests for the subpopulation AUC(t)
    ribbon renderer. Originally guarded the inline
    ``SubpopulationAnalysisSection._build_auc_t_plot``; the
    rendering now lives in ``mlsurv.viz._metrics_time.plot_grouped_metric_time``
    and the tests call that function directly. Ribbon / fill
    assertions are preserved byte-identically from before that move —
    the move was byte-identical so the assertions must keep passing.
    """

    def _build_axis(self):
        """Construct a :class:`SubpopulationAxis` with synthetic AUC(t) data."""
        import pandas as pd

        from mlsurv.viz._context import SubpopulationAxis

        # 2 models x 3 subpopulations x 4 eval times
        rows = []
        for model in ["coxph", "rsf"]:
            for sg in ["TypeA", "TypeB", "TypeC"]:
                for t in [1.0, 2.0, 3.0, 4.0]:
                    rows.append({
                        "model_type": model,
                        "subpopulation": sg,
                        "eval_time": t,
                        "auc": 0.7 + 0.02 * t,
                    })
        auc_df = pd.DataFrame(rows)

        # Bootstrap iterations (10 iters per model/subpopulation/time)
        boot_rows = []
        for model in ["coxph", "rsf"]:
            for sg in ["TypeA", "TypeB", "TypeC"]:
                for t in [1.0, 2.0, 3.0, 4.0]:
                    for i in range(10):
                        boot_rows.append({
                            "model_type": model,
                            "subpopulation": sg,
                            "iteration": i,
                            "eval_time": t,
                            "auc": 0.7 + 0.02 * t + 0.01 * (i - 5),
                        })
        boot_df = pd.DataFrame(boot_rows)

        # metrics_long is unused by plot_grouped_metric_time but must
        # still satisfy the SubpopulationAxis contract.
        metrics_long = pd.DataFrame(
            columns=["model_type", "subpopulation", "metric", "value"]
        )

        return SubpopulationAxis(
            group_variable="test_group",
            levels=["TypeA", "TypeB", "TypeC"],
            metrics_long=metrics_long,
            bootstrap_summary=None,
            auc_t=auc_df,
            auc_t_bootstrap=boot_df,
        )

    def _build_fig(self):
        """Render the faceted AUC(t) figure via the moved viz helper."""
        from mlsurv.viz._metrics_time import plot_grouped_metric_time

        return plot_grouped_metric_time(
            self._build_axis(),
            time_unit="months",
            ref_names=set(),
        )

    def test_ribbons_rendered_on_correct_panels(self):
        """Each (model, subpopulation) with CI data gets a visible ribbon on its panel."""
        import re

        fig = self._build_fig()
        assert fig is not None, "plot_grouped_metric_time returned None on valid data"

        # Ribbons may be rendered as a single ``fill='toself'`` polygon
        # or as a ``fill='tonexty'`` upper trace paired with an invisible
        # lower trace; either shape satisfies the test as long as the
        # expected number of fills is present.
        fill_traces = [
            t for t in fig.data
            if getattr(t, "fill", None) in ("toself", "tonexty")
        ]

        n_models = 2
        n_subpopulations = 3
        expected_fills = n_models * n_subpopulations  # one per (model, subpopulation)

        assert len(fill_traces) >= expected_fills, (
            f"Expected >= {expected_fills} ribbon fill traces "
            f"(one per model x subpopulation), got {len(fill_traces)}. "
            f"All trace fills: {[getattr(t, 'fill', None) for t in fig.data]}"
        )

        # Every fill trace must be assigned to a subplot axis. In a
        # make_subplots layout, subplot traces have xaxis='x', 'x2', 'x3',
        # ... and similarly for yaxis. A fill trace without a subplot axis
        # would render outside any panel.
        for t in fill_traces:
            assert getattr(t, "xaxis", None), (
                f"Fill trace has no xaxis assignment: {t}"
            )
            assert getattr(t, "yaxis", None), (
                f"Fill trace has no yaxis assignment: {t}"
            )

        # Every fill trace must have a non-empty, non-transparent fillcolor.
        # Alpha=0 would make the ribbon invisible even if the fill spec is
        # otherwise correct.
        rgba_re = re.compile(
            r"rgba?\s*\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*([\d.]+)\s*\)"
        )
        for t in fill_traces:
            fc = getattr(t, "fillcolor", None) or ""
            m = rgba_re.match(fc)
            assert m is not None, f"Expected rgba fillcolor, got: {fc!r}"
            alpha = float(m.group(1))
            assert alpha > 0.0, f"Fill trace is fully transparent: {fc}"

    def test_ribbons_per_panel_count(self):
        """Each subplot panel has at least one ribbon per model."""
        fig = self._build_fig()
        assert fig is not None

        # Group fill traces by (xaxis, yaxis) to count per-panel ribbons.
        panel_counts = {}
        for t in fig.data:
            if getattr(t, "fill", None) not in ("toself", "tonexty"):
                continue
            key = (getattr(t, "xaxis", None), getattr(t, "yaxis", None))
            panel_counts[key] = panel_counts.get(key, 0) + 1

        # 3 subpopulations = 3 panels, each with 2 models = 2 ribbons per panel.
        assert len(panel_counts) == 3, (
            f"Expected ribbons spread across 3 panels, got {len(panel_counts)}: "
            f"{panel_counts}"
        )
        for panel, count in panel_counts.items():
            assert count >= 2, (
                f"Panel {panel} has only {count} ribbon(s), expected >= 2 (one per model)"
            )

    def test_model_missing_from_first_panel_keeps_legend_entry(self):
        """A model absent from the first subpopulation panel still gets exactly
        one legend entry.

        Regression: legend visibility was tied to ``sg_idx == 0``, so a model
        with no data in the first panel (but curves in later panels) was drawn
        without ever appearing in the legend.
        """
        import pandas as pd

        from mlsurv.viz._context import SubpopulationAxis
        from mlsurv.viz._metrics_time import plot_grouped_metric_time

        levels = ["TypeA", "TypeB", "TypeC"]
        rows = []
        for model in ["coxph", "rsf"]:
            for sg in levels:
                # rsf has no data in the first panel ("TypeA").
                if model == "rsf" and sg == "TypeA":
                    continue
                for t in [1.0, 2.0, 3.0, 4.0]:
                    rows.append({
                        "model_type": model, "subpopulation": sg,
                        "eval_time": t, "auc": 0.7 + 0.02 * t,
                    })
        axis = SubpopulationAxis(
            group_variable="grp",
            levels=levels,
            metrics_long=pd.DataFrame(
                columns=["model_type", "subpopulation", "metric", "value"]
            ),
            bootstrap_summary=None,
            auc_t=pd.DataFrame(rows),
            auc_t_bootstrap=None,
        )

        fig = plot_grouped_metric_time(axis, ref_names=set())
        assert fig is not None

        legend_names = [t.name for t in fig.data if t.showlegend]
        # Exactly one legend entry per model, no duplicates — rsf must appear
        # even though it is missing from the first panel.
        assert sorted(legend_names) == sorted(set(legend_names)), legend_names
        assert len(legend_names) == 2, legend_names


# ---------------------------------------------------------------------------
# Subpopulation single non-AUC metric layout unification
# ---------------------------------------------------------------------------

class TestSubpopulationSingleMetricLayout:
    """A single non-AUC metric must share the AUC facet grid and dimensions.

    Guards against the subpopulation surface rendering ``metric="brier"``
    through the wider multi-metric grid (different wrapping and width) than
    ``metric="auc"``. Both single-metric paths now flow through the same
    ``_grouped_metric_grid_dims`` sizing.
    """

    def _make_metric_frames(self, n_sg, metric, *, with_std=True):
        """Synthetic long-format frame (+ optional std) for *n_sg* levels."""
        import pandas as pd

        levels = [f"L{i}" for i in range(n_sg)]
        base = 0.18 if metric == "brier" else 0.72
        rows, srows = [], []
        for model in ["coxph", "rsf"]:
            for sg in levels:
                for t in [1.0, 2.0, 3.0, 4.0]:
                    rows.append({
                        "model_type": model, "subpopulation": sg,
                        "eval_time": t, "value": base + 0.01 * t,
                        "metric_name": metric,
                    })
                    srows.append({
                        "model_type": model, "subpopulation": sg,
                        "eval_time": t, "value_std": 0.02,
                        "metric_name": metric,
                    })
        df = pd.DataFrame(rows)
        std = pd.DataFrame(srows) if with_std else None
        return levels, df, std

    def _make_auc_axis(self, n_sg):
        """SubpopulationAxis with AUC(t) data for *n_sg* levels."""
        import pandas as pd

        from mlsurv.viz._context import SubpopulationAxis

        levels = [f"L{i}" for i in range(n_sg)]
        rows = []
        for model in ["coxph", "rsf"]:
            for sg in levels:
                for t in [1.0, 2.0, 3.0, 4.0]:
                    rows.append({
                        "model_type": model, "subpopulation": sg,
                        "eval_time": t, "auc": 0.72 + 0.01 * t,
                    })
        return SubpopulationAxis(
            group_variable="grp",
            levels=levels,
            metrics_long=pd.DataFrame(
                columns=["model_type", "subpopulation", "metric", "value"]
            ),
            bootstrap_summary=None,
            auc_t=pd.DataFrame(rows),
            auc_t_bootstrap=None,
        )

    def test_grid_dims_helper_matches_auc_sizing(self):
        """_grouped_metric_grid_dims reproduces the documented AUC sizing."""
        from mlsurv.viz._metrics_time import _grouped_metric_grid_dims

        # (n_cols, n_rows, fig_width, fig_height) for representative counts.
        expected = {
            1: (1, 1, 550, 460),
            2: (2, 1, 900, 460),
            3: (2, 2, 900, 820),
            6: (2, 3, 900, 1180),
        }
        for n_sg, (n_cols, n_rows, w, h) in expected.items():
            dims = _grouped_metric_grid_dims(n_sg)
            assert (dims[0], dims[1], dims[3], dims[4]) == (n_cols, n_rows, w, h)

    @pytest.mark.parametrize("n_sg", [1, 3])
    def test_single_brier_matches_auc_dimensions(self, n_sg):
        """Single brier and single auc share width/height for the same levels."""
        from mlsurv.viz._metrics_time import (
            plot_grouped_metric_time,
            plot_grouped_single_metric_time,
        )

        fig_auc = plot_grouped_metric_time(self._make_auc_axis(n_sg))
        levels, df, std = self._make_metric_frames(n_sg, "brier")
        fig_brier = plot_grouped_single_metric_time(
            df, std, "brier", group_variable="grp", subpopulations=levels,
        )
        assert fig_auc is not None and fig_brier is not None
        assert fig_auc.layout.width == fig_brier.layout.width
        assert fig_auc.layout.height == fig_brier.layout.height

    def test_single_brier_uses_wrapped_grid(self):
        """3 levels wrap into a 2-col grid with level-name subplot titles."""
        from mlsurv.viz._metrics_time import plot_grouped_single_metric_time

        levels, df, std = self._make_metric_frames(3, "brier")
        fig = plot_grouped_single_metric_time(
            df, std, "brier", group_variable="grp", subpopulations=levels,
        )
        # 2-col wrap → 4 subplot cells (one empty) → 4 x-axes.
        n_xaxes = sum(
            1 for k in fig.layout if k == "xaxis" or k.startswith("xaxis")
        )
        assert n_xaxes == 4, f"expected 4 axes (2x2 wrap), got {n_xaxes}"
        # Subplot titles are the level names, not "Brier — L0".
        titles = {a.text for a in fig.layout.annotations}
        assert set(levels) <= titles
        assert not any("—" in (a.text or "") for a in fig.layout.annotations)

    def test_single_brier_lower_is_better_yaxis(self):
        """Brier panels floor the y-axis at 0 (lower-is-better)."""
        from mlsurv.viz._metrics_time import plot_grouped_single_metric_time

        levels, df, std = self._make_metric_frames(3, "brier", with_std=False)
        fig = plot_grouped_single_metric_time(
            df, std, "brier", group_variable="grp", subpopulations=levels,
        )
        assert fig.layout.yaxis.range[0] == 0.0

    def test_single_brier_renders_ribbons(self):
        """Std data still produces per-panel ±1 SD ribbons."""
        from mlsurv.viz._metrics_time import plot_grouped_single_metric_time

        levels, df, std = self._make_metric_frames(3, "brier")
        fig = plot_grouped_single_metric_time(
            df, std, "brier", group_variable="grp", subpopulations=levels,
        )
        fills = [
            t for t in fig.data
            if getattr(t, "fill", None) in ("toself", "tonexty")
        ]
        # 2 models x 3 subpopulations = 6 ribbons.
        assert len(fills) >= 6, f"got {len(fills)} ribbon fills"
        assert "(±1 SD)" in fig.layout.title.text

    def test_metric_filter_emptying_frame_returns_none(self):
        """A non-empty frame with no rows matching the metric yields None.

        This is the contract the dispatcher relies on to raise a clear
        ValueError instead of handing back a blank figure.
        """
        from mlsurv.viz._metrics_time import plot_grouped_single_metric_time

        levels, df, _ = self._make_metric_frames(3, "brier", with_std=False)
        fig = plot_grouped_single_metric_time(
            df, None, "auc", group_variable="grp", subpopulations=levels,
        )
        assert fig is None

    def test_all_nan_values_do_not_crash(self):
        """An all-NaN value column falls back to a label-only y-axis."""
        import numpy as np

        from mlsurv.viz._metrics_time import plot_grouped_single_metric_time

        levels, df, _ = self._make_metric_frames(3, "brier", with_std=False)
        df["value"] = np.nan
        fig = plot_grouped_single_metric_time(
            df, None, "brier", group_variable="grp", subpopulations=levels,
        )
        # Traces still added (NaN lines), so a figure is returned with no
        # explicit y-range set (label only).
        assert fig is not None
        assert fig.layout.yaxis.range is None

    def test_higher_is_better_metric_caps_at_one(self):
        """A not-lower-better metric uses the [y_min, 1.0] y-range branch."""
        from mlsurv.viz._metrics_time import plot_grouped_single_metric_time

        # 'auc' is higher-is-better; render it through the metric-agnostic
        # renderer directly to exercise the else branch.
        levels, df, std = self._make_metric_frames(3, "auc")
        fig = plot_grouped_single_metric_time(
            df, std, "auc", group_variable="grp", subpopulations=levels,
        )
        assert fig.layout.yaxis.range[1] == 1.0
        assert fig.layout.yaxis.range[0] > 0.0

    def test_model_missing_from_first_panel_keeps_legend_entry(self):
        """A model absent from the first subpopulation panel still gets exactly
        one legend entry (same regression as the AUC renderer).
        """
        from mlsurv.viz._metrics_time import plot_grouped_single_metric_time

        levels, df, std = self._make_metric_frames(3, "brier")
        # rsf has no data in the first panel ("L0").
        keep = ~((df["model_type"] == "rsf") & (df["subpopulation"] == "L0"))
        df = df[keep].reset_index(drop=True)
        std = std[
            ~((std["model_type"] == "rsf") & (std["subpopulation"] == "L0"))
        ].reset_index(drop=True)

        fig = plot_grouped_single_metric_time(
            df, std, "brier", group_variable="grp", subpopulations=levels,
        )
        assert fig is not None

        legend_names = [t.name for t in fig.data if t.showlegend]
        # Exactly one legend entry per model, no duplicates — rsf must appear
        # even though it is missing from the first panel.
        assert sorted(legend_names) == sorted(set(legend_names)), legend_names
        assert len(legend_names) == 2, legend_names
