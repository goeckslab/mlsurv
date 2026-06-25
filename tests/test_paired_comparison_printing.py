"""Tests for paired comparison printing and report integration."""

import numpy as np
import pandas as pd
import pytest

from mlsurv.results.printing import (
    print_paired_comparison,
    _format_pvalue,
    _significance_stars,
)
from mlsurv.results.containers import BootstrapResult, EvaluationResult

from tests._helpers import make_random_survival_array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_comparison_df(
    pairs=None,
    method="paired_bootstrap",
    with_adjusted=True,
):
    """Build a comparison DataFrame for testing."""
    if pairs is None:
        pairs = [
            ("rsf", "coxph", 0.032, -0.011, 0.074, 0.14),
            ("rsf", "deepsurv", 0.045, 0.008, 0.081, 0.02),
        ]
    rows = []
    for model_a, model_b, delta, ci_lo, ci_hi, p_val in pairs:
        row = {
            "model_a": model_a,
            "model_b": model_b,
            "metric": "harrell",
            "method": method,
            "delta": delta,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "p_value": p_val,
        }
        if with_adjusted:
            row["p_adjusted"] = p_val  # simplified: same as raw
        rows.append(row)
    return pd.DataFrame(rows)


def _make_bootstrap_result(
    models,
    n_bootstrap=200,
):
    """Build a BootstrapResult from {model: {metric: values}}."""
    iter_rows = []
    summary_rows = []

    for model_name, metrics in models.items():
        for metric_name, values in metrics.items():
            for i, v in enumerate(values):
                iter_rows.append({
                    "model_type": model_name,
                    "iteration": i,
                    "metric_name": metric_name,
                    "value": v,
                })
            summary_rows.append({
                "model_type": model_name,
                "metric": metric_name,
                "point": float(np.mean(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "ci_lower": float(np.percentile(values, 2.5)),
                "ci_upper": float(np.percentile(values, 97.5)),
            })

    return BootstrapResult(
        summary=pd.DataFrame(summary_rows),
        iterations=pd.DataFrame(iter_rows),
    )


# ===========================================================================
# _format_pvalue tests
# ===========================================================================

class TestFormatPvalue:
    @pytest.mark.parametrize("value, expected", [
        (0.14, "0.140"),         # large p-value → decimal
        (0.0001, "1.00e-04"),    # small p-value → scientific
        (0.0, "0.00e+00"),       # zero
        (float('nan'), "NaN"),   # NaN
        (None, "NaN"),           # None
        (0.001, "0.001"),        # borderline: 0.001 is NOT < 0.001 → decimal
        (0.0009, "9.00e-04"),    # just below 0.001 → scientific
    ])
    def test_format_pvalue(self, value, expected):
        assert _format_pvalue(value) == expected


# ===========================================================================
# _significance_stars tests
# ===========================================================================

class TestSignificanceStars:
    @pytest.mark.parametrize("value, expected", [
        (0.0001, " ***"),   # highly significant
        (0.005, " **"),     # very significant
        (0.03, " *"),       # significant
        (0.10, ""),         # not significant
        (float('nan'), ""), # NaN
        (None, ""),         # None
        (0.05, ""),         # exact boundary: not < 0.05
        (0.01, " *"),       # exact boundary: not < 0.01
        (0.001, " **"),     # exact boundary: not < 0.001
    ])
    def test_significance_stars(self, value, expected):
        assert _significance_stars(value) == expected


# ===========================================================================
# print_paired_comparison tests
# ===========================================================================

class TestPrintPairedComparison:
    def test_basic_output(self, capsys):
        df = _make_comparison_df()
        print_paired_comparison(df)
        out = capsys.readouterr().out
        assert "Paired Model Comparison" in out
        assert "rsf vs coxph" in out
        assert "rsf vs deepsurv" in out
        assert "+0.032" in out
        assert "* p < 0.05" in out

    def test_kang_method_label(self, capsys):
        df = _make_comparison_df(method="kang")
        print_paired_comparison(df, method_label="Kang z-test")
        out = capsys.readouterr().out
        assert "Kang z-test" in out

    def test_custom_precision(self, capsys):
        df = _make_comparison_df(
            pairs=[("a", "b", 0.12345, 0.01, 0.20, 0.05)],
        )
        print_paired_comparison(df, precision=4)
        out = capsys.readouterr().out
        assert "0.1235" in out  # rounded to 4 decimal places

    def test_empty_df(self, capsys):
        df = pd.DataFrame()
        print_paired_comparison(df)
        out = capsys.readouterr().out
        assert out == ""

    def test_none_df(self, capsys):
        print_paired_comparison(None)
        out = capsys.readouterr().out
        assert out == ""

    def test_nan_pvalues(self, capsys):
        df = _make_comparison_df(
            pairs=[("a", "b", np.nan, np.nan, np.nan, np.nan)],
        )
        print_paired_comparison(df)
        out = capsys.readouterr().out
        assert "N/A" in out
        assert "NaN" in out

    def test_display_names(self, capsys):
        df = _make_comparison_df(
            pairs=[("rsf", "coxph", 0.05, 0.01, 0.09, 0.03)],
        )
        print_paired_comparison(
            df,
            display_names={"rsf": "Random Forest", "coxph": "CoxPH"},
        )
        out = capsys.readouterr().out
        assert "Random Forest vs CoxPH" in out

    def test_significance_stars_in_output(self, capsys):
        df = _make_comparison_df(
            pairs=[
                ("a", "b", 0.01, -0.01, 0.03, 0.0001),  # ***
                ("a", "c", 0.01, -0.01, 0.03, 0.005),   # **
                ("b", "c", 0.01, -0.01, 0.03, 0.03),    # *
            ],
        )
        print_paired_comparison(df)
        out = capsys.readouterr().out
        assert "***" in out
        assert "**" in out
        assert "*" in out

    def test_negative_delta(self, capsys):
        df = _make_comparison_df(
            pairs=[("a", "b", -0.05, -0.10, 0.01, 0.10)],
        )
        print_paired_comparison(df)
        out = capsys.readouterr().out
        assert "-0.050" in out

    def test_p_adjusted_preferred(self, capsys):
        """Should use p_adjusted column when available and display its value."""
        df = _make_comparison_df(
            pairs=[("a", "b", 0.05, 0.01, 0.09, 0.01)],
            with_adjusted=True,
        )
        # Set p_adjusted to a different value than p_value
        df['p_adjusted'] = 0.99
        print_paired_comparison(df)
        out = capsys.readouterr().out
        assert "p-adj" in out
        assert "0.99" in out  # Should show the adjusted value, not raw

    def test_p_value_fallback(self, capsys):
        """Should fall back to p_value when p_adjusted not available."""
        df = _make_comparison_df(with_adjusted=False)
        print_paired_comparison(df)
        out = capsys.readouterr().out
        assert "p-value" in out

    def test_auto_method_detection(self, capsys):
        """Auto-detects method from DataFrame column."""
        df = _make_comparison_df(method="kang")
        print_paired_comparison(df)  # No method_label override
        out = capsys.readouterr().out
        assert "Kang z-test" in out

    def test_single_pair(self, capsys):
        df = _make_comparison_df(
            pairs=[("rsf", "coxph", 0.032, -0.011, 0.074, 0.14)],
        )
        print_paired_comparison(df)
        out = capsys.readouterr().out
        assert "rsf vs coxph" in out
        assert "=" in out


# ===========================================================================
# BootstrapResult.print_comparison tests
# ===========================================================================

class TestBootstrapResultPrintComparison:
    def test_print_comparison(self, capsys):
        rng = np.random.RandomState(42)
        boot = _make_bootstrap_result({
            "rsf": {"harrell": rng.normal(0.72, 0.03, 100)},
            "coxph": {"harrell": rng.normal(0.69, 0.03, 100)},
        }, n_bootstrap=100)
        df = boot.print_comparison()
        out = capsys.readouterr().out
        assert "Paired Model Comparison" in out
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_print_comparison_returns_df(self):
        rng = np.random.RandomState(42)
        boot = _make_bootstrap_result({
            "rsf": {"harrell": rng.normal(0.72, 0.03, 100)},
            "coxph": {"harrell": rng.normal(0.69, 0.03, 100)},
        }, n_bootstrap=100)
        df = boot.print_comparison()
        assert "model_a" in df.columns
        assert "delta" in df.columns


# ===========================================================================
# EvaluationResult.print_metrics auto-comparison tests
# ===========================================================================

class TestEvaluationResultAutoComparison:
    @staticmethod
    def _make_eval(concordance, bootstrap=None):
        """Build an EvaluationResult with required fields."""
        empty = pd.DataFrame()
        return EvaluationResult(
            concordance=concordance,
            test_set=empty,
            auc_t=empty,
            importance=empty,
            importance_iterations=empty,
            bootstrap=bootstrap,
        )

    def _make_eval_result_with_bootstrap(self):
        """Build an EvaluationResult with multi-model bootstrap."""
        rng = np.random.RandomState(42)
        boot = _make_bootstrap_result({
            "rsf": {"harrell": rng.normal(0.72, 0.03, 50)},
            "coxph": {"harrell": rng.normal(0.69, 0.03, 50)},
        }, n_bootstrap=50)

        concordance = pd.DataFrame({
            "model_type": ["rsf", "coxph"],
            "c_index_test": [0.72, 0.69],
        })

        return self._make_eval(concordance, bootstrap=boot)

    def test_print_metrics_no_auto_comparison(self, capsys):
        """print_metrics() does not auto-print paired comparison.

        The paired comparison table must be requested explicitly via
        BootstrapResult.print_comparison() or compare_bootstrap().
        """
        result = self._make_eval_result_with_bootstrap()
        result.print_metrics()
        out = capsys.readouterr().out
        assert "Paired Model Comparison" not in out

        # Sanity: the on-demand path still works.
        result.bootstrap.print_comparison()
        on_demand = capsys.readouterr().out
        assert "Paired Model Comparison" in on_demand

    def test_single_model_no_comparison(self, capsys):
        """Single model should not print comparison."""
        concordance = pd.DataFrame({
            "model_type": ["rsf"],
            "c_index_test": [0.72],
        })
        result = self._make_eval(concordance)
        result.print_metrics()
        out = capsys.readouterr().out
        assert "Paired Model Comparison" not in out

    def test_no_bootstrap_no_comparison(self, capsys):
        """Multi-model without bootstrap does not auto-compare."""
        concordance = pd.DataFrame({
            "model_type": ["rsf", "coxph"],
            "c_index_test": [0.72, 0.69],
        })
        result = self._make_eval(concordance)
        result.print_metrics()
        out = capsys.readouterr().out
        assert "Paired Model Comparison" not in out


# ===========================================================================
# Report subsection tests
# ===========================================================================

class TestPairedComparisonSubsection:
    def _make_mock_section(self, n_models=2, with_bootstrap=True):
        """Create a mock EvaluationSection for testing."""
        from unittest.mock import MagicMock

        section = MagicMock()
        section.metric = "harrell"
        section.data = {}

        # Models
        models = {}
        for i in range(n_models):
            name = f"model_{i}"
            m = MagicMock()
            m.evaluation = True
            models[name] = m
        section._get_all_models.return_value = models

        # Bootstrap
        if with_bootstrap and n_models >= 2:
            rng = np.random.RandomState(42)
            boot_results = {}
            for i in range(n_models):
                name = f"model_{i}"
                boot_results[name] = _make_bootstrap_result(
                    {name: {"harrell": rng.normal(0.70 + i * 0.02, 0.03, 50)}},
                    n_bootstrap=50,
                )
            section.bootstrap_results = boot_results
        else:
            section.bootstrap_results = None

        # Learner: no Kang comparison by default
        section.learner.kang_comparison_result = None

        return section

    def test_is_available_with_bootstrap(self):
        from mlsurv.reports.sections.evaluation._paired_comparison import (
            _PairedComparisonSubsection,
        )
        section = self._make_mock_section(n_models=2, with_bootstrap=True)
        sub = _PairedComparisonSubsection(section)
        assert sub.is_available()

    def test_not_available_single_model(self):
        from mlsurv.reports.sections.evaluation._paired_comparison import (
            _PairedComparisonSubsection,
        )
        section = self._make_mock_section(n_models=1, with_bootstrap=False)
        sub = _PairedComparisonSubsection(section)
        assert not sub.is_available()

    def test_add_generates_markdown(self):
        from mlsurv.reports.sections.evaluation._paired_comparison import (
            _PairedComparisonSubsection,
        )
        section = self._make_mock_section(n_models=2, with_bootstrap=True)
        sub = _PairedComparisonSubsection(section)

        md_lines = []
        figures = []
        figure_info = []
        sub.add(md_lines, figures, figure_info)

        text = "\n".join(md_lines)
        assert "Pair" in text
        assert "Method" in text
        assert "Delta" in text
        assert "p-value" in text

    def test_add_stores_data(self):
        from mlsurv.reports.sections.evaluation._paired_comparison import (
            _PairedComparisonSubsection,
        )
        section = self._make_mock_section(n_models=2, with_bootstrap=True)
        sub = _PairedComparisonSubsection(section)

        md_lines = []
        sub.add(md_lines, [], [])

        assert 'paired_comparison' in section.data

    def test_not_available_without_bootstrap_or_kang(self):
        from mlsurv.reports.sections.evaluation._paired_comparison import (
            _PairedComparisonSubsection,
        )
        section = self._make_mock_section(n_models=2, with_bootstrap=False)
        sub = _PairedComparisonSubsection(section)
        assert not sub.is_available()

    def test_available_with_kang_data(self):
        from mlsurv.reports.sections.evaluation._paired_comparison import (
            _PairedComparisonSubsection,
        )
        section = self._make_mock_section(n_models=2, with_bootstrap=False)
        # Add Kang data to state
        section.learner.kang_comparison_result = _make_comparison_df(method="kang")
        sub = _PairedComparisonSubsection(section)
        assert sub.is_available()

    def test_kang_only_generates_markdown(self):
        """Kang-only path (no bootstrap) generates valid markdown table."""
        from mlsurv.reports.sections.evaluation._paired_comparison import (
            _PairedComparisonSubsection,
        )
        section = self._make_mock_section(n_models=2, with_bootstrap=False)
        section.learner.kang_comparison_result = _make_comparison_df(method="kang")
        sub = _PairedComparisonSubsection(section)

        md_lines = []
        sub.add(md_lines, [], [])

        text = "\n".join(md_lines)
        assert "Kang z-test" in text
        assert "Pair" in text
        assert 'paired_comparison' in section.data

    def test_interpretation_significant(self):
        """Significant pairs produce specific interpretation text."""
        from mlsurv.reports.sections.evaluation._paired_comparison import (
            _PairedComparisonSubsection,
        )
        section = self._make_mock_section(n_models=2, with_bootstrap=False)
        # p=0.02 is significant
        section.learner.kang_comparison_result = _make_comparison_df(
            pairs=[("rsf", "coxph", 0.05, 0.01, 0.09, 0.02)],
            method="kang",
        )
        sub = _PairedComparisonSubsection(section)
        md_lines = []
        sub.add(md_lines, [], [])
        text = "\n".join(md_lines)
        assert "Statistically significant" in text

    def test_interpretation_not_significant(self):
        """Non-significant pairs produce appropriate interpretation."""
        from mlsurv.reports.sections.evaluation._paired_comparison import (
            _PairedComparisonSubsection,
        )
        section = self._make_mock_section(n_models=2, with_bootstrap=False)
        # p=0.50 is not significant
        section.learner.kang_comparison_result = _make_comparison_df(
            pairs=[("rsf", "coxph", 0.01, -0.05, 0.07, 0.50)],
            method="kang",
        )
        sub = _PairedComparisonSubsection(section)
        md_lines = []
        sub.add(md_lines, [], [])
        text = "\n".join(md_lines)
        assert "No statistically significant" in text

    def test_no_comparison_data_fallback(self):
        """When bootstrap comparison fails and no Kang data, shows fallback message."""
        from mlsurv.reports.sections.evaluation._paired_comparison import (
            _PairedComparisonSubsection,
        )
        # 2 models with bootstrap that will fail (empty iterations)
        section = self._make_mock_section(n_models=2, with_bootstrap=False)
        # Provide a bootstrap_results dict with broken data (empty DataFrames)
        from mlsurv.results.containers import BootstrapResult
        broken_boot = BootstrapResult(
            summary=pd.DataFrame(),
            iterations=pd.DataFrame(),
            lower_is_better=False,
        )
        section.bootstrap_results = {"model_0": broken_boot, "model_1": broken_boot}
        sub = _PairedComparisonSubsection(section)

        md_lines = []
        sub.add(md_lines, [], [])
        text = "\n".join(md_lines)
        assert "No comparison data available" in text


# ===========================================================================
# _auto_kang_comparison tests
# ===========================================================================

class TestAutoKangComparison:
    """Test the _auto_kang_comparison method on EvaluationMixin."""

    @staticmethod
    def _make_test_set(models, n_samples=50, seed=42):
        """Build a test_set DataFrame with risk scores for multiple models."""
        rng = np.random.RandomState(seed)
        rows = []
        for model_name, base_risk in models.items():
            for i in range(n_samples):
                rows.append({
                    'model_type': model_name,
                    'index': i,
                    'risk_score': base_risk + rng.normal(0, 0.1),
                    'time': rng.exponential(5.0),
                    'event': rng.random() < 0.5,
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _make_survival_array(n=50, seed=42):
        return make_random_survival_array(
            n, event_rate=0.5, time_scale=5.0,
            event_dist="uniform", seed=seed,
        )

    def test_two_models_returns_dataframe(self):
        """_auto_kang_comparison returns a DataFrame with 2 models."""
        from mlsurv.learner._evaluation import EvaluationMixin

        test_set = self._make_test_set({"rsf": 0.7, "coxph": 0.6})
        y_test = self._make_survival_array(n=50)

        result = type('MockResult', (), {'test_set': test_set})()
        mixin = EvaluationMixin()
        df = mixin._auto_kang_comparison(result, y_test, ["rsf", "coxph"])

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df['model_a'].iloc[0] == "rsf"
        assert df['model_b'].iloc[0] == "coxph"
        assert 'delta' in df.columns
        assert 'p_value' in df.columns

    def test_three_models_returns_three_pairs(self):
        """Three models produce 3 pairwise comparisons."""
        from mlsurv.learner._evaluation import EvaluationMixin

        test_set = self._make_test_set({"a": 0.7, "b": 0.6, "c": 0.5})
        y_test = self._make_survival_array(n=50)

        result = type('MockResult', (), {'test_set': test_set})()
        mixin = EvaluationMixin()
        df = mixin._auto_kang_comparison(result, y_test, ["a", "b", "c"])

        assert df is not None
        assert len(df) == 3  # C(3,2) = 3 pairs

    def test_empty_test_set_returns_none(self):
        """Empty test_set returns None."""
        from mlsurv.learner._evaluation import EvaluationMixin

        result = type('MockResult', (), {'test_set': pd.DataFrame()})()
        mixin = EvaluationMixin()
        df = mixin._auto_kang_comparison(result, None, ["rsf", "coxph"])
        assert df is None

    def test_single_model_returns_none(self):
        """Only one model with risk scores returns None."""
        from mlsurv.learner._evaluation import EvaluationMixin

        test_set = self._make_test_set({"rsf": 0.7})
        y_test = self._make_survival_array(n=50)

        result = type('MockResult', (), {'test_set': test_set})()
        mixin = EvaluationMixin()
        df = mixin._auto_kang_comparison(result, y_test, ["rsf"])
        assert df is None

    def test_nan_scores_excluded(self):
        """Models with all-NaN scores are excluded."""
        from mlsurv.learner._evaluation import EvaluationMixin

        test_set = self._make_test_set({"rsf": 0.7, "coxph": 0.6})
        # Replace coxph scores with NaN
        test_set.loc[test_set['model_type'] == 'coxph', 'risk_score'] = np.nan
        y_test = self._make_survival_array(n=50)

        result = type('MockResult', (), {'test_set': test_set})()
        mixin = EvaluationMixin()
        df = mixin._auto_kang_comparison(result, y_test, ["rsf", "coxph"])
        assert df is None  # Only one model has valid scores

    def test_mismatched_lengths_returns_none(self):
        """Models with different score array lengths after NaN drop return None."""
        from mlsurv.learner._evaluation import EvaluationMixin

        test_set = self._make_test_set({"rsf": 0.7, "coxph": 0.6}, n_samples=50)
        # Set some coxph scores to NaN so dropna() produces fewer values
        coxph_mask = test_set['model_type'] == 'coxph'
        coxph_indices = test_set[coxph_mask].index[:5]
        test_set.loc[coxph_indices, 'risk_score'] = np.nan
        y_test = self._make_survival_array(n=50)

        result = type('MockResult', (), {'test_set': test_set})()
        mixin = EvaluationMixin()
        df = mixin._auto_kang_comparison(result, y_test, ["rsf", "coxph"])
        assert df is None  # Different lengths → skipped


# ===========================================================================
# End-to-end no-auto-print + cache population
# ===========================================================================

class TestNoAutoPrint:
    """Lock in the no-auto-print contract at the learner level.

    These complement the unit-level flipped test on EvaluationResult by
    exercising the SurvivalLearner paths that were modified.
    """

    @staticmethod
    def _make_data(n=80, p=6, seed=0):
        rng = np.random.RandomState(seed)
        X = pd.DataFrame(rng.randn(n, p), columns=[f"x{i}" for i in range(p)])
        risk = X["x0"] - X["x1"]
        times = rng.exponential(scale=np.exp(-risk + 1.5))
        events = rng.binomial(1, 0.7, n).astype(bool)
        y = np.array(
            list(zip(events, times)),
            dtype=[("event", bool), ("time", float)],
        )
        return X, y

    @pytest.mark.integration
    def test_run_does_not_auto_print_paired_comparison(self, capsys):
        """`learner.run([m1, m2])` does not auto-print the comparison table."""
        from mlsurv import SurvivalLearner

        X, y = self._make_data()
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, feature_selector=None)
        learner.run(
            ["coxph", "rsf"],
            n_bootstrap=2,
            n_importance_repeats=1,
            tune=False,
        )
        out = capsys.readouterr().out
        assert "Paired Model Comparison" not in out
        assert "Kang z-test" not in out

        # On-demand path still works.
        cmp_df = learner.bootstrap_result.compare(metric="harrell")
        assert isinstance(cmp_df, pd.DataFrame) and len(cmp_df) >= 1

    @pytest.mark.integration
    def test_evaluate_populates_kang_cache_when_not_verbose(self):
        """`evaluate(verbose=False)` still populates `state.kang_comparison_result`.

        Previously the cache was a side-effect of the verbose print path; the
        decoupled helper now populates it unconditionally so HTML reports
        and ``export_results`` work for non-verbose workflows.
        """
        from mlsurv import SurvivalLearner

        X, y = self._make_data()
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, feature_selector=None)
        learner.run(
            ["coxph", "rsf"],
            evaluate=False,
            bootstrap=False,
            tune=False,
        )
        learner.evaluate(bootstrap=False, n_importance_repeats=1, verbose=False)

        kang = learner._state.kang_comparison_result
        assert isinstance(kang, pd.DataFrame) and not kang.empty
        assert {"model_a", "model_b"}.issubset(kang.columns)
        # Public read-only property exposes the same stashed result.
        assert learner.kang_comparison_result is kang

    @pytest.mark.integration
    def test_evaluate_single_model_leaves_cache_untouched(self):
        """Single-model evaluate() does not populate kang_comparison_result."""
        from mlsurv import SurvivalLearner

        X, y = self._make_data()
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, feature_selector=None)
        learner.train("coxph")
        learner.evaluate(bootstrap=False, n_importance_repeats=1, verbose=False)
        assert learner._state.kang_comparison_result is None
