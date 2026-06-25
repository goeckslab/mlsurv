"""Tests for mlsurv.viz.export HTML export utilities.

Covers save_html(), save_html_report(), and figures_to_dict().
The export module is at 94% but missing lines are in the report
assembly path (save_html_report with TOC) and the figures_to_dict
error handling.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

import plotly.graph_objects as go

from mlsurv._warnings import SuggestionWarning
from mlsurv.viz.export import save_html, save_html_report, figures_to_dict


LEARNER_PLOT_METHODS = [
    'plot_metric',
    'plot_metric_time',
    'plot_survival',
    'plot_importance',
    'plot_calibration',
    'plot_hazard_ratios',
]


class TestSaveHtml:
    """Tests for save_html() single-figure export."""

    def test_creates_html_file(self, tmp_path):
        """Should create a valid HTML file."""
        fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
        output = tmp_path / "test.html"

        save_html(fig, output)

        assert output.exists()
        content = output.read_text(encoding='utf-8')
        assert '<html' in content.lower()
        assert 'plotly' in content.lower()

    def test_custom_title(self, tmp_path):
        """save_html should include custom title in generated HTML."""
        fig = go.Figure()
        output = tmp_path / "test.html"

        save_html(fig, output, title="My Analysis")

        content = output.read_text(encoding='utf-8')
        assert '<title>My Analysis</title>' in content
        assert output.stat().st_size > 0

    def test_title_escaping_xss(self, tmp_path):
        """XSS payload in title should be escaped, not rendered as HTML."""
        fig = go.Figure()
        output = tmp_path / "test.html"

        save_html(fig, output, title="<script>alert('xss')</script>")

        content = output.read_text(encoding='utf-8')
        assert "<script>alert('xss')</script>" not in content
        assert '&lt;script&gt;' in content

    def test_title_special_chars(self, tmp_path):
        """Special characters (&, <, >, quotes) in title should be escaped."""
        fig = go.Figure()
        output = tmp_path / "test.html"

        save_html(fig, output, title='Test & "quotes" <tags>')

        content = output.read_text(encoding='utf-8')
        assert '&amp;' in content
        assert '&lt;tags&gt;' in content

    def test_cdn_mode(self, tmp_path):
        """CDN mode should produce smaller file (no embedded plotly.js)."""
        fig = go.Figure(data=go.Scatter(x=[1], y=[1]))
        output_full = tmp_path / "full.html"
        output_cdn = tmp_path / "cdn.html"

        save_html(fig, output_full, include_plotlyjs=True)
        save_html(fig, output_cdn, include_plotlyjs='cdn')

        # CDN version should be much smaller
        assert output_cdn.stat().st_size < output_full.stat().st_size

    def test_creates_missing_parent_dir(self, tmp_path):
        """save_html should create a missing parent directory."""
        fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
        output = tmp_path / "no" / "dir" / "test.html"

        save_html(fig, output)

        assert output.exists()


class TestSaveHtmlReport:
    """Tests for save_html_report() multi-figure assembly."""

    def test_multi_figure_report(self, tmp_path):
        """Should combine multiple figures into one HTML file."""
        fig1 = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
        fig2 = go.Figure(data=go.Bar(x=['a', 'b'], y=[5, 6]))
        output = tmp_path / "report.html"

        save_html_report(
            [("Section 1", fig1), ("Section 2", fig2)],
            output,
            title="Test Report"
        )

        assert output.exists()
        content = output.read_text(encoding='utf-8')
        assert 'Section 1' in content
        assert 'Section 2' in content
        assert 'Test Report' in content

    def test_creates_missing_parent_dir(self, tmp_path):
        """save_html_report should create a missing parent directory."""
        fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
        output = tmp_path / "no" / "dir" / "report.html"

        save_html_report([("Section 1", fig)], output)

        assert output.exists()

    def test_toc_generated(self, tmp_path):
        """Table of contents should be generated for >1 figures."""
        fig1 = go.Figure()
        fig2 = go.Figure()
        output = tmp_path / "report.html"

        save_html_report(
            [("First", fig1), ("Second", fig2)],
            output,
            include_toc=True,
        )

        content = output.read_text(encoding='utf-8')
        assert 'Contents' in content
        assert 'First' in content
        assert 'Second' in content

    def test_report_title_and_name_escaping(self, tmp_path):
        """Malicious title and figure names should be escaped at all 4 injection points."""
        fig1 = go.Figure()
        fig2 = go.Figure()
        output = tmp_path / "report.html"

        malicious_title = '<img src=x onerror=alert(1)>'
        malicious_name = '<b>bold</b><script>evil()</script>'

        save_html_report(
            [(malicious_name, fig1), ("Safe Name", fig2)],
            output,
            title=malicious_title,
            include_toc=True,
        )

        content = output.read_text(encoding='utf-8')

        # Title should be escaped in <title> and <h1> (2 occurrences)
        assert '<img src=x' not in content
        assert content.count('&lt;img src=x onerror=alert(1)&gt;') == 2

        # Figure name should be escaped in <h2> and TOC <a> (2 occurrences)
        assert '<script>evil()</script>' not in content
        assert content.count('&lt;script&gt;evil()&lt;/script&gt;') == 2

    def test_no_toc_single_figure(self, tmp_path):
        """Single figure should not get TOC nav block even if enabled."""
        fig = go.Figure()
        output = tmp_path / "report.html"

        save_html_report(
            [("Only One", fig)],
            output,
            include_toc=True,
        )

        content = output.read_text(encoding='utf-8')
        # TOC nav block only generated when len(figures) > 1
        assert '<nav class="toc">' not in content

    def test_timestamp_included(self, tmp_path):
        """Timestamp should be present when include_timestamp=True."""
        fig = go.Figure()
        output = tmp_path / "report.html"

        save_html_report([("Test", fig)], output, include_timestamp=True)

        content = output.read_text(encoding='utf-8')
        assert 'Generated:' in content

    def test_no_timestamp(self, tmp_path):
        """Timestamp should be absent when include_timestamp=False."""
        fig = go.Figure()
        output = tmp_path / "report.html"

        save_html_report([("Test", fig)], output, include_timestamp=False)

        content = output.read_text(encoding='utf-8')
        assert 'Generated:' not in content


class TestFiguresToDict:
    """Tests for figures_to_dict() learner figure generation."""

    def test_empty_learner(self):
        """Learner with all plot methods raising should return empty list."""
        learner = MagicMock(spec=LEARNER_PLOT_METHODS)
        model_result = SimpleNamespace(name='coxph', model=object())

        # All plot methods raise expected exceptions
        learner.plot_metric.side_effect = ValueError("No results")
        learner.plot_survival.side_effect = ValueError("No results")
        learner.plot_importance.side_effect = ValueError("No results")

        # Each skipped plot must emit a SuggestionWarning
        with pytest.warns(SuggestionWarning) as record:
            result = figures_to_dict(learner, model_result=model_result)
        assert result == []
        skips = [w for w in record if issubclass(w.category, SuggestionWarning)]
        assert len(skips) == 3

        # All three methods should have been called
        learner.plot_metric.assert_called_once()
        learner.plot_survival.assert_called_once_with(model_result.name)
        learner.plot_importance.assert_called_once_with(model_result.name)

    def test_default_includes(self):
        """Default includes should be cindex, survival, importance."""
        learner = MagicMock(spec=LEARNER_PLOT_METHODS)
        learner.plot_metric.return_value = go.Figure()

        result = figures_to_dict(learner)
        assert len(result) == 1
        assert result[0][0] == 'C-Index Performance'

        # survival and importance are skipped when model_result is None
        learner.plot_survival.assert_not_called()
        learner.plot_importance.assert_not_called()

    def test_default_includes_with_model_result(self):
        """Default includes should generate cindex, survival, and importance."""
        learner = MagicMock(spec=LEARNER_PLOT_METHODS)
        model_result = SimpleNamespace(name='coxph', model=object())
        learner.plot_metric.return_value = go.Figure()
        learner.plot_survival.return_value = go.Figure()
        learner.plot_importance.return_value = go.Figure()

        result = figures_to_dict(learner, model_result=model_result)

        assert [name for name, _ in result] == [
            'C-Index Performance',
            'Survival Curves by Risk Score',
            'Feature Importance',
        ]
        learner.plot_metric.assert_called_once()
        learner.plot_survival.assert_called_once_with(model_result.name)
        learner.plot_importance.assert_called_once_with(model_result.name)

    def test_custom_includes(self):
        """Custom include list should only generate specified plots."""
        learner = MagicMock(spec=LEARNER_PLOT_METHODS)
        learner.plot_metric.return_value = go.Figure()

        result = figures_to_dict(learner, include=['cindex'])
        assert len(result) == 1
        assert result[0][0] == 'C-Index Performance'

    def test_expected_exception_in_plot_skipped_with_warning(self):
        """Expected exceptions are skipped but emit SuggestionWarning."""
        learner = MagicMock(spec=LEARNER_PLOT_METHODS)
        learner.plot_metric.side_effect = ValueError("No results")

        with pytest.warns(SuggestionWarning, match="'cindex'.*ValueError.*No results"):
            result = figures_to_dict(learner, include=['cindex'])
        assert result == []

    def test_keyerror_warning_names_plot_and_error(self):
        """The warning names the plot key and the original exception."""
        learner = MagicMock(spec=LEARNER_PLOT_METHODS)
        learner.plot_hazard_ratios.side_effect = KeyError("hazard_ratio")

        with pytest.warns(SuggestionWarning, match="'forest'.*KeyError.*hazard_ratio"):
            result = figures_to_dict(learner, include=['forest'])
        assert result == []

    def test_partial_failure_keeps_survivors_and_warns(self):
        """One failing plot warns; the others still make it into the result."""
        learner = MagicMock(spec=LEARNER_PLOT_METHODS)
        learner.plot_metric.side_effect = ValueError("No results")
        learner.plot_hazard_ratios.return_value = go.Figure()

        with pytest.warns(SuggestionWarning, match="'cindex'"):
            result = figures_to_dict(learner, include=['cindex', 'forest'])

        assert [name for name, _ in result] == ['Hazard Ratios']

    def test_missing_model_result_warns_and_skips(self):
        """Model-specific plots without model_result warn and are skipped."""
        learner = MagicMock(spec=LEARNER_PLOT_METHODS)
        learner.plot_metric.return_value = go.Figure()

        with pytest.warns(SuggestionWarning,
                          match="'survival', 'importance'.*require model_result"):
            result = figures_to_dict(learner)  # default include

        assert [name for name, _ in result] == ['C-Index Performance']
        learner.plot_survival.assert_not_called()
        learner.plot_importance.assert_not_called()

    def test_all_keys_without_model_result(self):
        """All 7 keys with model_result=None: the 3 model-specific keys are
        skipped (named in one warning), the rest still generate.

        Also guards the needs_model_result set against drifting from the
        dispatch table: a new model-dereferencing key missing from the set
        would crash here with AttributeError on model_result.name.
        """
        learner = MagicMock(spec=LEARNER_PLOT_METHODS)
        learner.plot_metric.return_value = go.Figure()
        learner.plot_metric_time.return_value = go.Figure()
        learner.plot_hazard_ratios.return_value = go.Figure()

        all_keys = ['cindex', 'auc', 'survival', 'importance',
                    'calibration', 'brier', 'forest']
        with pytest.warns(SuggestionWarning,
                          match="'survival', 'importance', 'calibration'"):
            result = figures_to_dict(learner, include=all_keys)

        assert [name for name, _ in result] == [
            'C-Index Performance',
            'Time-Dependent AUC',
            'Brier Score Curve',
            'Hazard Ratios',
        ]

    def test_skip_warning_names_model(self):
        """Failures of model-specific plots name the model, so different
        models' failures don't dedup under the 'once' filter."""
        learner = MagicMock(spec=LEARNER_PLOT_METHODS)
        model_result = SimpleNamespace(name='coxph', model=object())
        learner.plot_calibration.side_effect = ValueError("No results")

        with pytest.warns(SuggestionWarning,
                          match="'calibration' for model 'coxph'.*ValueError"):
            result = figures_to_dict(learner, model_result=model_result,
                                     include=['calibration'])
        assert result == []

    def test_unexpected_exception_propagates(self):
        """Unexpected exceptions (TypeError, RuntimeError, etc.) should propagate."""
        learner = MagicMock(spec=LEARNER_PLOT_METHODS)
        learner.plot_metric.side_effect = TypeError("Unexpected bug")

        with pytest.raises(TypeError, match="Unexpected bug"):
            figures_to_dict(learner, include=['cindex'])

    def test_auc_include_uses_plot_metric_time(self):
        """'auc' generates a figure via plot_metric_time(metric='auc')."""
        learner = MagicMock(spec=LEARNER_PLOT_METHODS)
        learner.plot_metric_time.return_value = go.Figure()

        result = figures_to_dict(learner, include=['auc'])

        assert [name for name, _ in result] == ['Time-Dependent AUC']
        learner.plot_metric_time.assert_called_once_with(metric='auc')

    def test_brier_include_uses_plot_metric_time(self):
        """'brier' generates a figure via plot_metric_time(metric='brier')."""
        learner = MagicMock(spec=LEARNER_PLOT_METHODS)
        learner.plot_metric_time.return_value = go.Figure()

        result = figures_to_dict(learner, include=['brier'])

        assert [name for name, _ in result] == ['Brier Score Curve']
        learner.plot_metric_time.assert_called_once_with(metric='brier')

    def test_unknown_include_key_raises(self):
        """Unknown include keys raise ValueError listing valid options."""
        learner = MagicMock(spec=LEARNER_PLOT_METHODS)

        with pytest.raises(ValueError, match=r"'bogus'.*Valid options:.*auc.*cindex"):
            figures_to_dict(learner, include=['cindex', 'bogus'])

        # No plot should have been attempted before validation
        learner.plot_metric.assert_not_called()
