"""Tests for extended analysis one-liners in the executive summary."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from mlsurv.reports.sections.summary import SummarySection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_learner():
    """Minimal mock learner for SummarySection tests."""
    learner = MagicMock()
    learner._state.is_setup = True
    learner.validation_results = {}
    learner.subpopulation_results = {}
    learner.benchmark_result = None
    learner.cv_strategy = 'stratifiedkfold'
    learner.cv_folds = 5
    learner.cv_repeats = 1
    learner.scoring = 'harrell'

    rng = np.random.RandomState(42)
    n_train, n_test = 80, 20
    learner.X_train = pd.DataFrame({
        'age': rng.normal(60, 10, n_train),
        'bmi': rng.normal(25, 5, n_train),
    })
    learner.X_test = pd.DataFrame({
        'age': rng.normal(60, 10, n_test),
        'bmi': rng.normal(25, 5, n_test),
    })
    learner.y_train = pd.DataFrame({
        'time': rng.exponential(10, n_train),
        'event': rng.binomial(1, 0.6, n_train),
    })
    learner.y_test = pd.DataFrame({
        'time': rng.exponential(10, n_test),
        'event': rng.binomial(1, 0.6, n_test),
    })
    learner.models = {}
    return learner


def _add_model(learner, name='coxph', cv_score=0.72):
    result = MagicMock()
    result.cv_metric_mean = cv_score
    result.cv_metric_std = 0.03
    result.is_tuned = False
    result.optuna_study = None
    result.scoring_metric = 'harrell'
    result.model = MagicMock()
    learner.models[name] = result
    return result


def _make_subpopulation_result(group_var, subpopulation_names):
    """Create a mock SubpopulationResult with the given group variable and subpopulation names."""
    sg = MagicMock()
    sg.group_variable = group_var
    sg.summary = pd.DataFrame({
        'subpopulation': subpopulation_names,
        'n_samples': [50] * len(subpopulation_names),
        'n_events': [25] * len(subpopulation_names),
    })
    return sg


def _make_interactions_df(n_pairs, method='hstat'):
    """Create a mock interactions DataFrame."""
    score_col = 'shap_interaction' if method == 'shap' else 'h_statistic'
    rows = [
        {
            'model_type': 'coxph',
            'feature_1': f'feat_{i}',
            'feature_2': f'feat_{i + n_pairs}',
            score_col: 0.1 * (n_pairs - i),
            'method': method,
        }
        for i in range(n_pairs)
    ]
    return pd.DataFrame(rows)


def _make_eval_result(interactions=None):
    """Create a minimal mock EvaluationResult for interaction tests."""
    eval_result = MagicMock()
    eval_result.interactions = interactions
    eval_result.importance = None
    eval_result.auc_t = None
    eval_result.test_set = None
    eval_result.bootstrap = None
    return eval_result


def _generate_md(learner, eval_result=None):
    """Generate summary markdown."""
    _add_model(learner)
    section = SummarySection(learner, eval_result=eval_result)
    return section.generate()['markdown']


# ---------------------------------------------------------------------------
# Tests: no extended analyses
# ---------------------------------------------------------------------------

class TestNoExtendedAnalyses:

    def test_no_extended_analyses_no_extra_bullets(self):
        """When no extended analyses exist, no extra bullets appear."""
        learner = _make_mock_learner()
        md = _generate_md(learner)

        assert '**Subpopulation analysis:**' not in md
        assert '**External validation:**' not in md
        assert '**Benchmarking:**' not in md
        assert '**Feature interactions:**' not in md


# ---------------------------------------------------------------------------
# Tests: subpopulation analysis
# ---------------------------------------------------------------------------

class TestSubpopulationSummaryLine:

    def test_single_variable(self):
        """Concise line for a single subpopulation variable."""
        learner = _make_mock_learner()
        learner.subpopulation_results = {
            'cancer_type': _make_subpopulation_result('cancer_type', ['Breast', 'Lung', 'Colon']),
        }
        md = _generate_md(learner)

        assert '**Subpopulation analysis:** 3 Cancer Type groups' in md

    def test_multiple_variables(self):
        """Multiple subpopulation variables joined by semicolon."""
        learner = _make_mock_learner()
        learner.subpopulation_results = {
            'cancer_type': _make_subpopulation_result('cancer_type', ['Breast', 'Lung']),
            'stage': _make_subpopulation_result('stage', ['I', 'II', 'III']),
        }
        md = _generate_md(learner)

        assert '**Subpopulation analysis:** 2 Cancer Type groups; 3 Stage groups' in md

    def test_no_verbose_sample_sizes(self):
        """The concise line does not contain per-group sample sizes."""
        learner = _make_mock_learner()
        learner.subpopulation_results = {
            'cancer_type': _make_subpopulation_result('cancer_type', ['Breast', 'Lung']),
        }
        md = _generate_md(learner)

        # The old verbose format had "(n=" — should not appear in subpopulation bullet
        subpopulation_line = [
            line for line in md.split('\n')
            if '**Subpopulation analysis:**' in line
        ]
        assert len(subpopulation_line) == 1
        assert '(n=' not in subpopulation_line[0]


# ---------------------------------------------------------------------------
# Tests: external validation
# ---------------------------------------------------------------------------

class TestValidationSummaryLine:

    def test_single_cohort(self):
        """Singular 'cohort' for one validation result."""
        learner = _make_mock_learner()
        learner.validation_results = {'TCGA': MagicMock()}
        md = _generate_md(learner)

        assert '**External validation:** 1 cohort' in md

    def test_multiple_cohorts(self):
        """Plural 'cohorts' for multiple validation results."""
        learner = _make_mock_learner()
        learner.validation_results = {
            'TCGA': MagicMock(),
            'ICGC': MagicMock(),
            'GEO': MagicMock(),
        }
        md = _generate_md(learner)

        assert '**External validation:** 3 cohorts' in md


# ---------------------------------------------------------------------------
# Tests: benchmarking
# ---------------------------------------------------------------------------

class TestBenchmarkSummaryLine:

    def test_benchmark_with_references(self):
        """Benchmark line lists reference names."""
        learner = _make_mock_learner()
        bench = MagicMock()
        bench.reference_names = ['LORIS', 'TMB', 'PD-L1']
        learner.benchmark_result = bench
        md = _generate_md(learner)

        assert '**Benchmarking:** compared against 3 reference scores (LORIS, TMB, PD-L1)' in md

    def test_benchmark_single_reference(self):
        """Singular 'score' for one reference."""
        learner = _make_mock_learner()
        bench = MagicMock()
        bench.reference_names = ['LORIS']
        learner.benchmark_result = bench
        md = _generate_md(learner)

        assert '**Benchmarking:** compared against 1 reference score (LORIS)' in md

    def test_benchmark_empty_references(self):
        """No bullet when reference_names is empty."""
        learner = _make_mock_learner()
        bench = MagicMock()
        bench.reference_names = []
        learner.benchmark_result = bench
        md = _generate_md(learner)

        assert '**Benchmarking:**' not in md


# ---------------------------------------------------------------------------
# Tests: feature interactions
# ---------------------------------------------------------------------------

class TestInteractionsSummaryLine:

    def test_hstat_method(self):
        """H-statistic method displayed correctly."""
        learner = _make_mock_learner()
        eval_result = _make_eval_result(_make_interactions_df(5, method='hstat'))
        md = _generate_md(learner, eval_result=eval_result)

        assert '**Feature interactions:** 5 pairs analyzed via H-statistic' in md

    def test_shap_method(self):
        """SHAP interaction method displayed correctly."""
        learner = _make_mock_learner()
        eval_result = _make_eval_result(_make_interactions_df(3, method='shap'))
        md = _generate_md(learner, eval_result=eval_result)

        assert '**Feature interactions:** 3 pairs analyzed via SHAP interaction' in md

    def test_single_pair(self):
        """Singular 'pair' for one interaction."""
        learner = _make_mock_learner()
        eval_result = _make_eval_result(_make_interactions_df(1, method='hstat'))
        md = _generate_md(learner, eval_result=eval_result)

        assert '1 pair analyzed via' in md

    def test_dual_method(self):
        """Both H-statistic and SHAP methods displayed."""
        learner = _make_mock_learner()
        hstat_df = _make_interactions_df(3, method='hstat')
        shap_df = _make_interactions_df(2, method='shap')
        combined = pd.concat([hstat_df, shap_df], ignore_index=True)
        eval_result = _make_eval_result(combined)
        md = _generate_md(learner, eval_result=eval_result)

        assert '5 pairs analyzed via H-statistic & SHAP' in md

    def test_empty_interactions(self):
        """Empty interactions DataFrame produces no bullet."""
        learner = _make_mock_learner()
        eval_result = _make_eval_result(pd.DataFrame(columns=[
            'model_type', 'feature_1', 'feature_2', 'h_statistic', 'method',
        ]))
        md = _generate_md(learner, eval_result=eval_result)

        assert '**Feature interactions:**' not in md

    def test_no_method_column(self):
        """Interactions without method column default to H-statistic."""
        learner = _make_mock_learner()
        df = pd.DataFrame({
            'model_type': ['coxph', 'coxph'],
            'feature_1': ['a', 'b'],
            'feature_2': ['c', 'd'],
            'h_statistic': [0.1, 0.05],
        })
        eval_result = _make_eval_result(df)
        md = _generate_md(learner, eval_result=eval_result)

        assert '2 pairs analyzed via H-statistic' in md


# ---------------------------------------------------------------------------
# Tests: graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:

    def test_all_extended_analyses_present(self):
        """All four bullets appear when all analyses were performed."""
        learner = _make_mock_learner()
        learner.subpopulation_results = {
            'stage': _make_subpopulation_result('stage', ['I', 'II']),
        }
        learner.validation_results = {'TCGA': MagicMock()}
        bench = MagicMock()
        bench.reference_names = ['LORIS']
        learner.benchmark_result = bench
        eval_result = _make_eval_result(_make_interactions_df(2, method='hstat'))
        md = _generate_md(learner, eval_result=eval_result)

        assert '**Subpopulation analysis:**' in md
        assert '**External validation:**' in md
        assert '**Benchmarking:**' in md
        assert '**Feature interactions:**' in md

    def test_acronym_group_variable(self):
        """Mixed-case variable names are preserved (not title-cased)."""
        learner = _make_mock_learner()
        learner.subpopulation_results = {
            'PFS_status': _make_subpopulation_result('PFS_status', ['low', 'high']),
        }
        md = _generate_md(learner)

        # PFS is uppercase — title_if_lower should preserve it
        assert '2 PFS status groups' in md

    def test_broken_subpopulation_state(self):
        """Broken subpopulation_results doesn't crash the summary."""
        learner = _make_mock_learner()
        # Set to something iterable but with broken items
        learner.subpopulation_results = {'bad': 'not_a_subpopulation_result'}
        md = _generate_md(learner)

        # Should not crash; subpopulation line just absent
        assert '> **Key Findings**' in md
        assert '**Subpopulation analysis:**' not in md

    def test_broken_benchmark_state(self):
        """Broken benchmark_result doesn't crash the summary."""
        learner = _make_mock_learner()
        bench = MagicMock()
        bench.reference_names = None  # Will fail on len()
        learner.benchmark_result = bench
        md = _generate_md(learner)

        assert '> **Key Findings**' in md
        assert '**Benchmarking:**' not in md
