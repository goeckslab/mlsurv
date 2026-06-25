"""Tests for interpretation text and quality-label helpers across report sections."""

import numpy as np
import pandas as pd
import pytest

from mlsurv.reports.sections.base import BaseSection
from mlsurv.reports.sections.summary import SummarySection
from mlsurv.reports.sections.evaluation import EvaluationSection


# ---------------------------------------------------------------------------
# Quality-label shared utility (BaseSection._interpret_quality)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score,expected", [
    (0.95, "excellent"),
    (0.85, "excellent"),
    (0.80, "excellent"),   # boundary
    (0.79, "good"),
    (0.75, "good"),
    (0.70, "good"),        # boundary
    (0.69, "moderate"),
    (0.65, "moderate"),
    (0.60, "moderate"),    # boundary
    (0.59, "poor"),
    (0.55, "poor"),
    (0.50, "poor"),
    (0.0, "poor"),
])
def test_interpret_quality_thresholds(score, expected):
    """Shared quality mapper should classify correctly at boundaries."""
    assert BaseSection._interpret_quality(score) == expected


# ---------------------------------------------------------------------------
# Summary section: C-index interpretation presence
# ---------------------------------------------------------------------------


def test_summary_includes_cindex_interpretation(small_data):
    """Summary should include plain-language C-index interpretation when trained."""
    from mlsurv import SurvivalLearner

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)
    learner.train('coxph')

    section = SummarySection(learner)
    result = section.generate()
    md = result['markdown']

    assert "correctly ranking risk" in md
    assert "discrimination" in md


# ---------------------------------------------------------------------------
# Data preprocessing: interpretation paragraph
# ---------------------------------------------------------------------------


def test_data_preprocessing_interpretation_small_sample(small_data):
    """DataPreprocessingSection should warn about small sample size."""
    from mlsurv import SurvivalLearner
    from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

    X, y = small_data  # 50 samples
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = DataPreprocessingSection(learner)
    result = section.generate()
    md = result['markdown']

    assert "**Interpretation:**" in md
    assert "relatively small" in md


# ---------------------------------------------------------------------------
# Feature analysis: interpretation paragraph
# ---------------------------------------------------------------------------


def test_feature_analysis_includes_interpretation(small_data):
    """FeatureAnalysisSection should include permutation importance interpretation."""
    from mlsurv import SurvivalLearner
    from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)
    trained = learner.train('coxph')
    eval_result = learner.evaluate(trained, n_importance_repeats=5, bootstrap=False)

    section = FeatureAnalysisSection(learner, eval_result=eval_result)
    result = section.generate()
    md = result['markdown']

    assert "Permutation feature importance measures" in md
    assert "importance does not imply causation" in md


def test_feature_analysis_coefficient_intro(small_data):
    """Feature coefficients section should explain log hazard ratios."""
    from mlsurv import SurvivalLearner
    from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)
    trained = learner.train('coxph')
    eval_result = learner.evaluate(trained, n_importance_repeats=5, bootstrap=False)

    section = FeatureAnalysisSection(learner, eval_result=eval_result)
    result = section.generate()
    md = result['markdown']

    assert "log hazard ratio" in md.lower()


# ---------------------------------------------------------------------------
# Evaluation: stratification interpretation
# ---------------------------------------------------------------------------


def test_stratification_interpretation_block(small_data):
    """Stratification subsection should include interpretation with log-rank and risk groups."""
    from mlsurv import SurvivalLearner

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)
    trained = learner.train('coxph')
    eval_result = learner.evaluate(trained, bootstrap=False)

    section = EvaluationSection(learner, eval_result=eval_result)
    result = section.generate()
    md = result['markdown']

    # Stratification intro explains quantile grouping
    assert "quantile groups" in md.lower()
    assert "risk scores" in md.lower()

    # Interpretation block explains log-rank and RMST (if stratification rendered)
    # Note: may not render if risk score extraction fails on small data
    if "log-rank" in md.lower():
        assert "rmst" in md.lower()


# ---------------------------------------------------------------------------
# Benchmarking: interpretation paragraphs and table formatting
# ---------------------------------------------------------------------------


class TestBenchmarkingInterpretation:
    """Tests for narrative explanations in the benchmarking report section."""

    @staticmethod
    def _make_section():
        """Create a BenchmarkSection with a minimal mock learner."""
        from unittest.mock import MagicMock
        from mlsurv.reports.sections.benchmarking import BenchmarkSection
        learner = MagicMock()
        learner.scoring = "concordance_index"
        section = BenchmarkSection(learner)
        return section

    @staticmethod
    def _make_incremental_df(ci_excludes_zero=True):
        """Build a minimal incremental DataFrame."""
        rows = [
            {"model": "coxph", "reference": "ref_a", "metric": "delta_c_index",
             "point": 0.05, "ci_lower": 0.01 if ci_excludes_zero else -0.02,
             "ci_upper": 0.09},
            {"model": "coxph", "reference": "ref_a", "metric": "nri",
             "point": 0.10, "ci_lower": -0.05, "ci_upper": 0.25},
        ]
        return pd.DataFrame(rows)

    def test_incremental_interpretation_with_significant_ci(self):
        """CIs excluding zero should note significance."""
        section = self._make_section()
        inc = self._make_incremental_df(ci_excludes_zero=True)
        md_lines = []
        section._add_incremental_interpretation(md_lines, inc)
        text = "\n".join(md_lines)

        assert "**Interpretation:**" in text
        assert "Delta C-index" in text
        assert "NRI" in text
        assert "IDI" in text
        assert "exclude zero" in text
        assert "statistically significant" in text

    def test_incremental_interpretation_no_significant_ci(self):
        """CIs all including zero should report no significance."""
        section = self._make_section()
        inc = self._make_incremental_df(ci_excludes_zero=False)
        md_lines = []
        section._add_incremental_interpretation(md_lines, inc)
        text = "\n".join(md_lines)

        assert "**Interpretation:**" in text
        assert "No metric showed statistically significant" in text

    def test_incremental_interpretation_no_ci(self):
        """Missing CIs should produce static explanation only."""
        section = self._make_section()
        inc = self._make_incremental_df()
        inc["ci_lower"] = np.nan
        inc["ci_upper"] = np.nan
        md_lines = []
        section._add_incremental_interpretation(md_lines, inc)
        text = "\n".join(md_lines)

        assert "**Interpretation:**" in text
        # Should not contain significance language
        assert "Significant:" not in text
        assert "No metric showed" not in text

    def test_univariate_cox_interpretation(self):
        """Univariate Cox section should explain HR and report significance."""
        from unittest.mock import MagicMock

        cox = pd.DataFrame([
            {"reference": "ref_a", "hazard_ratio": 0.8,
             "hr_lower_95": 0.6, "hr_upper_95": 1.0, "p_value": 0.03},
            {"reference": "ref_b", "hazard_ratio": 1.2,
             "hr_lower_95": 0.9, "hr_upper_95": 1.6, "p_value": 0.20},
        ])
        br = MagicMock()
        br.univariate_cox = cox

        section = self._make_section()
        md_lines = []
        section._add_univariate_cox(md_lines, br)
        text = "\n".join(md_lines)

        assert "HR < 1" in text
        assert "protective" in text.lower()
        assert "statistically significant" in text

    def test_correlation_matrix_format(self):
        """Correlation section should use matrix format with raw HTML."""
        from unittest.mock import MagicMock

        corr = pd.DataFrame([
            {"model": "coxph", "reference": "ref_a",
             "spearman_rho": -0.85, "spearman_p": 0.0001,
             "pearson_r": -0.82, "pearson_p": 0.0003},
            {"model": "rsf", "reference": "ref_a",
             "spearman_rho": -0.72, "spearman_p": 0.002,
             "pearson_r": -0.70, "pearson_p": 0.004},
            # Positive correlation row — verifies the symbol prefix
            # renders cleanly without a leading minus sign.
            {"model": "coxph", "reference": "ref_b",
             "spearman_rho": 0.63, "spearman_p": 0.01,
             "pearson_r": 0.58, "pearson_p": 0.02},
            {"model": "rsf", "reference": "ref_b",
             "spearman_rho": 0.55, "spearman_p": 0.03,
             "pearson_r": 0.50, "pearson_p": 0.04},
        ])
        br = MagicMock()
        br.correlation = corr

        section = self._make_section()
        md_lines = []
        section._add_correlation(md_lines, br)
        text = "\n".join(md_lines)

        # Should use raw HTML markers (matrix format)
        assert "<!-- MLSURV_RAW_HTML -->" in text
        # Should NOT use the old flat "Pair" column
        assert "Pair" not in text
        # Should have significance stars
        assert "***" in text
        # Should have Spearman and Pearson sub-headings
        assert "Spearman" in text
        assert "Pearson" in text
        # Should have interpretation before tables
        assert "higher-is-worse" in text
        # Interpretation should appear before the first matrix
        interp_pos = text.index("higher-is-worse")
        matrix_pos = text.index("<!-- MLSURV_RAW_HTML -->")
        assert interp_pos < matrix_pos

        # Correct notation — rho for Spearman, lowercase r for
        # Pearson, and no capital R= anywhere (reserved for regression).
        spearman_block = text[
            text.index("#### Spearman"):text.index("#### Pearson")
        ]
        pearson_block = text[text.index("#### Pearson"):]
        assert "\u03c1=" in spearman_block  # rho=...
        assert "r=" in pearson_block
        assert "R=" not in text
        # Symbol should also appear in the per-table heading
        assert "(\u03c1)" in spearman_block
        assert "(r)" in pearson_block
        # Positive-correlation row must render without a leading minus
        # (exercises the symbol prefix on non-negative values).
        assert "\u03c1=0.630" in spearman_block
        assert "r=0.580" in pearson_block

    def test_significance_stars(self):
        """_significance_stars should return correct markers."""
        from mlsurv.reports.sections.benchmarking import BenchmarkSection
        assert BenchmarkSection._significance_stars(0.0001) == "***"
        assert BenchmarkSection._significance_stars(0.005) == "**"
        assert BenchmarkSection._significance_stars(0.03) == "*"
        assert BenchmarkSection._significance_stars(0.1) == ""
        assert BenchmarkSection._significance_stars(np.nan) == ""
        assert BenchmarkSection._significance_stars(-np.inf) == ""
        assert BenchmarkSection._significance_stars(np.inf) == ""
        assert BenchmarkSection._significance_stars(-0.01) == ""

    def test_subpopulation_toggle_large_table(self):
        """Subpopulation table with >20 rows should include toggle HTML."""
        from mlsurv.reports.sections.benchmarking import BenchmarkSection

        # Build 25 rows of table data
        table_data = [
            {"Subpopulation": f"grp_{i}", "Model vs Reference": "coxph vs ref_a",
             "Delta C-index": "0.05", "NRI": "0.10", "IDI": "0.02"}
            for i in range(25)
        ]

        lines = BenchmarkSection._render_subpopulation_table_html(table_data, 20)
        text = "\n".join(lines)

        assert "<!-- MLSURV_RAW_HTML -->" in text
        assert "ft-overflow-row" in text
        assert "ft-expand-btn" in text
        assert "feature-table-container" in text
        assert 'data-collapsed-text="Show remaining 5 rows"' in text
        assert 'data-expanded-text="Show first 20 rows"' in text

    def test_subpopulation_no_toggle_small_table(self):
        """Subpopulation table with <=20 rows should use markdown (no toggle)."""
        from unittest.mock import MagicMock

        # Build a small subpopulation incremental DataFrame (2 subpopulations × 1 pair × 1 metric)
        sg_inc = pd.DataFrame([
            {"subpopulation": "grp_a", "model": "coxph", "reference": "ref_a",
             "metric": "delta_c_index", "point": 0.05,
             "ci_lower": np.nan, "ci_upper": np.nan},
            {"subpopulation": "grp_b", "model": "coxph", "reference": "ref_a",
             "metric": "delta_c_index", "point": 0.03,
             "ci_lower": np.nan, "ci_upper": np.nan},
        ])
        br = MagicMock()
        br.subpopulation_incremental = sg_inc

        section = self._make_section()
        md_lines = []
        section._add_subpopulation_incremental(md_lines, br)
        text = "\n".join(md_lines)

        # Should NOT have toggle HTML
        assert "ft-overflow-row" not in text
        assert "ft-expand-btn" not in text
        # Should have interpretation before the table
        assert "2 subpopulations" in text
        assert "Consistent positive metrics" in text
