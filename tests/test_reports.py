"""Tests for HTML report rendering."""

import pytest
import mistune

from tests._helpers import make_survival_array

# Skip all tests if plotly not available
plotly = pytest.importorskip("plotly")
import plotly.graph_objects as go

# Feature-analysis subsection helpers (production calls these module functions
# directly; tests exercise them here rather than through any section wrapper).
from mlsurv.reports.sections.feature_analysis._interactive_table import (
    build_multi_model_column_groups,
    build_single_model_column_groups,
    format_interactive_table,
)
from mlsurv.reports.sections.feature_analysis._importance import (
    build_feature_row,
    get_sorted_top_features,
)
from mlsurv.reports.sections.feature_analysis._coefficients import (
    get_all_coefficient_models,
)
from mlsurv.reports.sections.feature_analysis._audit import (
    generate_feature_selection_audit,
)
from mlsurv.reports.sections.feature_analysis._correlation import (
    generate_correlation_interpretation,
    generate_correlation_warning_callout,
)
from mlsurv.reports.sections.evaluation._calibration import _CalibrationSubsection

# Tree-model fit cost scales with forest size; these report fixtures only need trained
# models (not tuned), so they train with tune=False and a tiny n_estimators.
_SMALL_TREE_PARAMS = {'n_estimators': 20}


class _CountingCalibrationModel:
    """Simple model stub that counts survival-function prediction calls."""

    def __init__(self, variant: str = "base"):
        self.variant = variant
        self.predict_calls = 0

    def predict_survival_function(self, X):
        import numpy as np
        import pandas as pd

        self.predict_calls += 1
        n_samples = len(X)

        base = np.linspace(0.1, 0.9, n_samples)
        if self.variant == "shifted":
            base = np.clip(base * 0.9 + 0.05, 0.05, 0.95)

        times = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        curves = [np.clip(1.0 - (t / 10.0) * (1.0 - base), 0.01, 1.0) for t in times]
        return pd.DataFrame(curves, index=times, columns=np.arange(n_samples))


class TestFigureAccessibility:
    """Embedded figures expose a screen-reader name (role + aria-label)."""

    def _embedder(self):
        from mlsurv.reports.renderers._figures import FigureEmbedder
        return FigureEmbedder(include_plotlyjs=False)

    def _fig(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        return fig

    def test_static_figure_has_img_role_and_label(self):
        """A non-evolution figure wrapper carries role="img" + aria-label."""
        html = self._embedder()._wrap(
            '<div class="plotly-graph-div"></div>',
            include_trace_filters=False,
            label='Time-Dependent AUC Curves',
        )
        assert 'class="figure-container"' in html
        assert 'role="img"' in html
        assert 'aria-label="Time-Dependent AUC Curves"' in html

    def test_trace_filter_figure_uses_group_role_and_keeps_controls(self):
        """Evolution figures use role="group" (not "img") so the curve-
        visibility controls stay reachable by assistive tech."""
        html = self._embedder()._wrap(
            '<div class="plotly-graph-div"></div>',
            include_trace_filters=True,
            label='CV Score Evolution',
        )
        assert 'role="group"' in html
        assert 'role="img"' not in html
        assert 'aria-label="CV Score Evolution"' in html
        # Controls must remain present (role="group" does not flatten them).
        assert '<div class="trace-filter-controls"' in html

    def test_embed_uses_figure_info_title(self):
        """embed() threads the FigureInfo title into the aria-label."""
        from mlsurv.reports.sections.base import FigureInfo
        info = FigureInfo(
            figure_id='auct_curves',
            title='Time-Dependent AUC Curves',
            data_description='Tidy AUC(t) frame',
        )
        html = self._embedder().embed(self._fig(), info, figure_id='auct_curves')
        assert 'role="img"' in html
        assert 'aria-label="Time-Dependent AUC Curves"' in html

    def test_embed_falls_back_to_humanized_figure_id(self):
        """Without FigureInfo, the figure_id becomes a non-empty name."""
        html = self._embedder().embed(self._fig(), None, figure_id='custom_plot')
        assert 'aria-label="Custom Plot"' in html

    def test_empty_label_emits_no_role_or_aria(self):
        """A falsy label leaves the wrapper unannotated — an empty
        aria-label="" is worse for screen readers than none, so emit neither
        role nor aria-label rather than an empty name."""
        for kwargs in ({}, {'include_trace_filters': True}):
            html = self._embedder()._wrap(
                '<div></div>', label='', **kwargs,
            )
            assert 'role=' not in html
            assert 'aria-label="' not in html.replace(
                'aria-label="Curve visibility controls"', '',
            )

    def test_aria_label_is_html_escaped(self):
        """A title with quotes/markup is escaped inside the attribute."""
        html = self._embedder()._wrap(
            '<div></div>', label='A "quote" & <tag>',
        )
        assert 'aria-label="A &quot;quote&quot; &amp; &lt;tag&gt;"' in html
        assert '<tag>' not in html

    def test_render_remaining_static_figure_is_labelled(self, tmp_path):
        """End-to-end: a rendered static figure gets role="img" + a name."""
        from mlsurv.reports.renderers import HTMLRenderer

        renderer = HTMLRenderer(str(tmp_path / "r.html"), title="A11y")
        sections = [{
            'title': 'Overview',
            'markdown': 'Body.',
            'figures': [('survival_curves.png', self._fig())],
        }]
        content = renderer.render(sections)
        assert 'role="img"' in content
        assert 'aria-label="Survival Curves"' in content

    def test_render_remaining_evolution_figure_is_group(self, tmp_path):
        """End-to-end: a rendered evolution figure gets role="group"."""
        from mlsurv.reports.renderers import HTMLRenderer

        renderer = HTMLRenderer(str(tmp_path / "r.html"), title="A11y")
        sections = [{
            'title': 'Evolution',
            'markdown': 'Body.',
            'figures': [('train_cv_evolution.png', self._fig())],
        }]
        content = renderer.render(sections)
        assert 'role="group"' in content
        assert 'aria-label="Train Cv Evolution"' in content
        assert '<div class="trace-filter-controls"' in content

    def test_figure_accessible_name_helper(self):
        """The name resolver prefers title, falls back to humanized id."""
        from mlsurv.reports.renderers._figures import _figure_accessible_name
        from mlsurv.reports.sections.base import FigureInfo

        info = FigureInfo(figure_id='x', title='Real Title', data_description='d')
        assert _figure_accessible_name(info, 'x') == 'Real Title'
        # Empty title falls back to the humanized figure_id.
        empty = FigureInfo(figure_id='x', title='', data_description='d')
        assert _figure_accessible_name(empty, 'corr_heatmap') == 'Corr Heatmap'
        assert _figure_accessible_name(None, 'risk_groups') == 'Risk Groups'
        assert _figure_accessible_name(None, '') == ''


class TestHTMLRenderer:
    """Tests for HTMLRenderer class."""

    def test_render_empty_sections(self, tmp_path):
        """HTMLRenderer handles empty sections list."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "report.html"
        renderer = HTMLRenderer(str(output_file), title="Empty Report")

        html = renderer.render([])
        renderer.write(html)

        assert output_file.exists()
        content = output_file.read_text(encoding='utf-8')
        assert '<h1 class="report-title">Empty Report</h1>' in content
        assert '<div class="phase-tabs report-sections-tabs"' not in content
        assert '</html>' in content

    def test_render_section_with_figure(self, tmp_path):
        """Plotly figure is embedded as div in HTML."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "report.html"
        renderer = HTMLRenderer(str(output_file), title="Figure Test")

        # Create a simple Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))

        sections = [{
            'title': 'Test Section',
            'markdown': 'Some content here.',
            'figures': [('test_fig.png', fig)]
        }]

        html = renderer.render(sections)
        renderer.write(html)

        content = output_file.read_text(encoding='utf-8')

        # Verify Plotly div is embedded
        assert 'class="plotly-graph-div"' in content
        # Verify section title
        assert '<h2>Test Section</h2>' in content
        # Verify content is included
        assert '<p>Some content here.</p>' in content

    def test_render_section_with_trace_filter_controls(self, tmp_path):
        """Run Evolution figures include curve visibility checkbox controls."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "trace_filters.html"
        renderer = HTMLRenderer(str(output_file), title="Trace Filter Test")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 3, 2], name='curve A'))
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 2, 3], name='curve B'))

        sections = [{
            'title': 'Evolution',
            'markdown': 'Compare lines.',
            'figures': [('train_cv_evolution.png', fig)]
        }]

        html = renderer.render(sections)
        renderer.write(html)

        content = output_file.read_text(encoding='utf-8')
        assert 'trace-filter-controls' in content
        assert 'initTraceFilters' in content
        assert 'trace-toggle' in content
        assert "showlegend: false" not in content
        assert 'legendgroup::' in content

    def test_render_non_evolution_figure_without_trace_filter_controls(self, tmp_path):
        """Non-Run-Evolution figures should render without extra checkbox controls."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "no_trace_filters.html"
        renderer = HTMLRenderer(str(output_file), title="No Trace Filter Test")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 3, 2], name='curve A'))
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 2, 3], name='curve B'))

        sections = [{
            'title': 'Overview',
            'markdown': 'Compare lines.',
            'figures': [('survival_curves.png', fig)]
        }]

        html = renderer.render(sections)
        renderer.write(html)

        content = output_file.read_text(encoding='utf-8')
        assert '<div class="trace-filter-controls"' not in content
        assert 'initTraceFilters' not in content

    def test_render_markdown_table(self, tmp_path):
        """Markdown tables are converted to HTML tables."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "report.html"
        renderer = HTMLRenderer(str(output_file), title="Table Test")

        table_markdown = """| Model | Score |
|-------|-------|
| CoxPH | 0.75 |
| RSF | 0.80 |"""

        sections = [{
            'title': 'Results',
            'markdown': table_markdown,
            'figures': []
        }]

        html = renderer.render(sections)
        renderer.write(html)

        content = output_file.read_text(encoding='utf-8')

        # Verify HTML table structure
        assert '<table>' in content
        assert '<thead>' in content
        assert '<th>Model</th>' in content
        assert '<th>Score</th>' in content
        assert '<tbody>' in content
        assert '<td>CoxPH</td>' in content
        assert '<td>0.75</td>' in content
        assert '</table>' in content
        assert 'initSortableTables' in content
        assert 'sort-indicator' in content
        assert 'initPredictionTableScroll' in content
        assert "event.target.closest('.tab-button')" in content
        assert 'setTimeout(initPredictionTableScroll, 0);' in content

    def test_renderer_escapes_titles_and_section_labels(self, tmp_path):
        """Renderer escapes title and section/tab labels in output HTML."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "escape_titles.html"
        renderer = HTMLRenderer(str(output_file), title='Bad <script>alert("x")</script>')

        sections = [{
            'title': 'Section <img src=x onerror=alert(1)>',
            'markdown': 'Safe body.',
            'figures': [],
            'tabs': [
                {'key': 'tab-1', 'title': 'Tab <svg onload=alert(1)>', 'markdown': 'Body'},
            ],
        }]

        html = renderer.render(sections)
        renderer.write(html)
        content = output_file.read_text(encoding='utf-8')

        assert '<script>alert("x")</script>' not in content
        assert 'Section <img src=x onerror=alert(1)>' not in content
        assert 'Tab <svg onload=alert(1)>' not in content
        assert '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;' in content
        assert '&lt;img src=x onerror=alert(1)&gt;' in content
        assert '&lt;svg onload=alert(1)&gt;' in content

    def test_renderer_sanitizes_markdown_links_and_html(self, tmp_path):
        """Markdown rendering blocks javascript: links and escapes inline HTML."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "sanitize_markdown.html"
        renderer = HTMLRenderer(str(output_file), title="Sanitize Markdown")

        sections = [{
            'title': 'Security',
            'markdown': (
                '[good](https://example.com) '
                '[bad](javascript:alert(1)) '
                '**<img src=x onerror=alert(1)>** '
                '| Col |\n|---|\n| <script>alert(1)</script> |'
            ),
            'figures': [],
        }]

        html = renderer.render(sections)
        renderer.write(html)
        content = output_file.read_text(encoding='utf-8')

        assert 'href="https://example.com"' in content
        assert 'href="javascript:alert(1)"' not in content
        assert 'href="#"' in content
        assert '<img src=x onerror=alert(1)>' not in content
        assert '&lt;img src=x onerror=alert(1)&gt;' in content
        assert '<script>alert(1)</script>' not in content
        assert '&lt;script&gt;alert(1)&lt;/script&gt;' in content

    def test_renderer_supports_level4_markdown_headings(self, tmp_path):
        """#### headings render as heading tags in HTML output."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "level4_heading.html"
        renderer = HTMLRenderer(str(output_file), title="Heading Test")
        sections = [{
            'title': 'Section',
            'markdown': '#### Risk Stratification Comparison',
            'figures': [],
        }]

        html = renderer.render(sections)
        renderer.write(html)
        content = output_file.read_text(encoding='utf-8')

        assert '<h5 id="risk-stratification-comparison">Risk Stratification Comparison</h5>' in content

    def test_html_no_images_directory(self, tmp_path):
        """HTML renderer does not create images/ directory."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "report.html"
        renderer = HTMLRenderer(str(output_file), title="No Images Test")

        # Create a figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]))

        sections = [{
            'title': 'Section',
            'markdown': 'Content',
            'figures': [('fig.png', fig)]
        }]

        html = renderer.render(sections)
        renderer.save_figures([('fig.png', fig)])  # This should be a no-op
        renderer.write(html)

        # Verify no images directory created
        images_dir = tmp_path / "images"
        assert not images_dir.exists()

    def test_plotlyjs_embedded(self, tmp_path):
        """Full Plotly.js library is embedded when include_plotlyjs=True."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "report.html"
        renderer = HTMLRenderer(str(output_file), title="JS Test", include_plotlyjs=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]))

        sections = [{
            'title': 'Section',
            'markdown': 'Content',
            'figures': [('fig.png', fig)]
        }]

        html = renderer.render(sections)
        renderer.write(html)

        content = output_file.read_text(encoding='utf-8')

        # When include_plotlyjs=True, the full library is embedded
        # Look for Plotly library code (a characteristic string)
        assert 'Plotly' in content

    def test_plotlyjs_cdn(self, tmp_path):
        """CDN link used when include_plotlyjs='cdn'."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "report.html"
        renderer = HTMLRenderer(str(output_file), title="CDN Test", include_plotlyjs='cdn')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]))

        sections = [{
            'title': 'Section',
            'markdown': 'Content',
            'figures': [('fig.png', fig)]
        }]

        html = renderer.render(sections)
        renderer.write(html)

        content = output_file.read_text(encoding='utf-8')

        # When include_plotlyjs='cdn', a script tag with CDN URL is used
        assert 'cdn.plot.ly' in content or 'plotly' in content.lower()

    def test_multiple_figures_single_plotlyjs(self, tmp_path):
        """Plotly.js is only included once for multiple figures."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "report.html"
        renderer = HTMLRenderer(str(output_file), title="Multi-fig Test", include_plotlyjs=True)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=[1, 2], y=[1, 2]))
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=[1, 2], y=[3, 4]))

        sections = [
            {
                'title': 'Section 1',
                'markdown': 'First section',
                'figures': [('fig1.png', fig1)]
            },
            {
                'title': 'Section 2',
                'markdown': 'Second section',
                'figures': [('fig2.png', fig2)]
            }
        ]

        html = renderer.render(sections)
        renderer.write(html)

        content = output_file.read_text(encoding='utf-8')

        # Both figures should be present
        assert content.count('class="plotly-graph-div"') == 2

    def test_inline_markdown_conversion(self, tmp_path):
        """Bold, italic, and code are converted to HTML."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "report.html"
        renderer = HTMLRenderer(str(output_file), title="Inline Test")

        sections = [{
            'title': 'Formatting',
            'markdown': 'This has **bold**, *italic*, and `code` text.',
            'figures': []
        }]

        html = renderer.render(sections)
        renderer.write(html)

        content = output_file.read_text(encoding='utf-8')

        assert '<strong>bold</strong>' in content
        assert '<em>italic</em>' in content
        assert '<code>code</code>' in content

    def test_image_references_stripped(self, tmp_path):
        """Image markdown references are stripped (figures embedded instead)."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "report.html"
        renderer = HTMLRenderer(str(output_file), title="Image Strip Test")

        sections = [{
            'title': 'Images',
            'markdown': '![Alt text](images/fig.png)\n\nSome text after.',
            'figures': []
        }]

        html = renderer.render(sections)
        renderer.write(html)

        content = output_file.read_text(encoding='utf-8')

        # Image reference should not appear
        assert '![' not in content
        assert 'images/fig.png' not in content
        # But other text should
        assert 'Some text after' in content

    def test_image_reference_embeds_matching_figure(self, tmp_path):
        """Image markdown placeholders embed matching section figures inline."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "inline_figure.html"
        renderer = HTMLRenderer(str(output_file), title="Inline Figure Test")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))

        sections = [{
            'title': 'Inline Figure',
            'markdown': 'Before figure.\n\n![Inline](images/inline_fig.png)\n\nAfter figure.',
            'figures': [('inline_fig.png', fig)],
        }]

        html = renderer.render(sections)
        renderer.write(html)
        content = output_file.read_text(encoding='utf-8')

        assert 'Before figure.' in content
        assert 'After figure.' in content
        assert 'class="plotly-graph-div"' in content
        assert 'data-figure="inline_fig.png"' not in content

    def test_prediction_table_scroll_div_preserved(self, tmp_path):
        """Prediction-table scroll wrappers survive markdown conversion."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "prediction_scroll.html"
        renderer = HTMLRenderer(str(output_file), title="Prediction Scroll Test")

        sections = [{
            'title': 'Predictions',
            'markdown': (
                '<div class="prediction-table-scroll" data-visible-rows="10">\n'
                '| A | B |\n'
                '|---|---|\n'
                '| x | 1 |\n'
                '</div>'
            ),
            'figures': [],
        }]

        html = renderer.render(sections)
        renderer.write(html)
        content = output_file.read_text(encoding='utf-8')

        assert '<div class="prediction-table-scroll" data-visible-rows="10">' in content
        assert '&lt;div class=&quot;prediction-table-scroll&quot;' not in content
        assert '<table>' in content

    def test_section_tabs_targets_match_section_anchors(self, tmp_path):
        """Top-level section tabs target rendered section anchors."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "report.html"
        renderer = HTMLRenderer(str(output_file), title="Section Tabs Test")

        sections = [
            {'title': 'Dataset Summary', 'markdown': 'Data info', 'figures': []},
            {'title': 'Model Results', 'markdown': 'Results info', 'figures': []},
        ]

        html = renderer.render(sections)
        renderer.write(html)

        content = output_file.read_text(encoding='utf-8')

        # Verify top-level section tab buttons
        assert 'class="phase-tabs report-sections-tabs"' in content
        assert 'data-tab-group="report-sections"' in content
        assert 'data-tab-target="dataset-summary"' in content
        assert 'data-tab-target="model-results"' in content
        # Verify section anchors
        assert 'id="dataset-summary"' in content
        assert 'id="model-results"' in content

    def test_cross_section_anchor_link_has_matching_target(self, tmp_path):
        """Cross-section anchor links target existing tab panels and nav script is included."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "report.html"
        renderer = HTMLRenderer(str(output_file), title="Anchor Nav Test")

        sections = [
            {
                'title': 'Model Evaluation',
                'markdown': (
                    'See [Subpopulation Analysis](#subpopulation-analysis) for details.'
                ),
                'figures': [],
            },
            {
                'title': 'Subpopulation Analysis',
                'markdown': 'Subpopulation content here.',
                'figures': [],
            },
        ]

        html = renderer.render(sections)
        renderer.write(html)
        content = output_file.read_text(encoding='utf-8')

        # The link exists in the rendered HTML
        assert 'href="#subpopulation-analysis"' in content
        # The target section exists as a tab panel with matching id
        assert 'id="subpopulation-analysis"' in content
        # The tab button for the target exists
        assert 'data-tab-target="subpopulation-analysis"' in content
        # The anchor nav script is included
        assert "closest('a[href^=" in content

    def test_anchor_nav_script_included_when_sections_present(self, tmp_path):
        """Anchor navigation script is included when report has sections."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "report.html"
        renderer = HTMLRenderer(str(output_file), title="Script Test")

        sections = [
            {'title': 'Section A', 'markdown': 'Content A', 'figures': []},
            {'title': 'Section B', 'markdown': 'Content B', 'figures': []},
        ]

        html = renderer.render(sections)

        # Tab script present
        assert 'tab-button' in html
        # Anchor nav script present alongside tab script
        assert "closest('a[href^=" in html

    def test_render_section_with_tabs(self, tmp_path):
        """Renderer outputs tab controls and panels for section tab payload."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "tabs.html"
        renderer = HTMLRenderer(str(output_file), title="Tabs Test")

        sections = [{
            'title': 'Tabbed Section',
            'markdown': 'Intro text.',
            'figures': [],
            'tabs': [
                {'key': 'train', 'title': 'Train', 'markdown': '| A | B |\n|---|---|\n| x | 1 |'},
                {'key': 'evaluation', 'title': 'Evaluation', 'markdown': 'Evaluation content.'},
                {'key': 'test', 'title': 'Test', 'markdown': 'Test content.'},
            ],
        }]

        html = renderer.render(sections)
        renderer.write(html)
        content = output_file.read_text(encoding='utf-8')

        assert 'class="phase-tabs"' in content
        assert 'class="tab-button active"' in content
        assert '>Train<' in content
        assert '>Evaluation<' in content
        assert '>Test<' in content
        assert 'class="tab-panel active"' in content


class TestReportGeneratorFormats:
    """Tests for ReportGenerator format inference from file extension."""

    def test_invalid_extension_raises(self, tmp_path):
        """Invalid file extension raises ValueError."""
        from mlsurv.reports import ReportGenerator
        from unittest.mock import MagicMock

        # Create mock learner
        learner = MagicMock()

        generator = ReportGenerator(learner)

        output_file = tmp_path / "report.pdf"

        with pytest.raises(ValueError) as exc_info:
            generator.generate(str(output_file))

        assert ".pdf" in str(exc_info.value).lower()
        assert ".md" in str(exc_info.value).lower()
        assert ".html" in str(exc_info.value).lower()

    def test_html_extension_accepted(self, tmp_path):
        """HTML extension is accepted (may fail later due to no data, but extension is valid)."""
        from mlsurv.reports import ReportGenerator
        from unittest.mock import MagicMock

        # Create mock learner with no data
        learner = MagicMock()
        # generate() auto-wires eval/bootstrap from the learner; honor
        # the real _gather_report_data() -> (eval, bootstrap) contract.
        learner._gather_report_data.return_value = (None, None)

        generator = ReportGenerator(learner)

        output_file = tmp_path / "report.html"

        # With graceful error handling, the report should generate
        # successfully even with a mock learner (sections degrade
        # gracefully). The key assertion is that .html is accepted
        # as a valid extension, not rejected as "Unknown file extension".
        generator.generate(str(output_file))
        assert output_file.exists()


class TestMarkdownRendererRegression:
    """Regression tests to ensure MarkdownRenderer still works."""

    def test_markdown_renderer_still_works(self, tmp_path):
        """MarkdownRenderer continues to work after changes."""
        from mlsurv.reports.renderers import MarkdownRenderer

        output_file = tmp_path / "report.md"
        renderer = MarkdownRenderer(str(output_file), title="Markdown Test")

        sections = [{
            'title': 'Test Section',
            'markdown': 'Some **bold** content.',
            'figures': []
        }]

        md_content = renderer.render(sections)
        renderer.write(md_content)

        assert output_file.exists()
        content = output_file.read_text(encoding='utf-8')
        assert '# Markdown Test' in content
        assert '## Test Section' in content
        assert 'Some **bold** content.' in content

    def test_markdown_renderer_renders_tab_fallback(self, tmp_path):
        """Markdown renderer appends tab content as subheadings."""
        from mlsurv.reports.renderers import MarkdownRenderer

        output_file = tmp_path / "tabs.md"
        renderer = MarkdownRenderer(str(output_file), title="Markdown Tabs")

        sections = [{
            'title': 'Tabbed Section',
            'markdown': 'Intro text.',
            'figures': [],
            'tabs': [
                {'key': 'train', 'title': 'Train', 'markdown': 'Train metrics table'},
                {'key': 'evaluation', 'title': 'Evaluation', 'markdown': 'Evaluation metrics table'},
                {'key': 'test', 'title': 'Test', 'markdown': 'Test metrics table'},
            ],
        }]

        md_content = renderer.render(sections)
        renderer.write(md_content)
        content = output_file.read_text(encoding='utf-8')

        assert '## Tabbed Section' in content
        assert '### Train' in content
        assert '### Evaluation' in content
        assert '### Test' in content


class TestBindVar:
    """Unit tests for the _bind_var snippet-normalization helper."""

    def test_rewrites_existing_assignment_target(self):
        from mlsurv.reports.renderers._figures import _bind_var
        assert _bind_var('fig = learner.plot_x()', 'fig') == 'fig = learner.plot_x()'
        # Rebinds a differently-named target (figure return_data -> df).
        assert _bind_var('fig = learner.plot_x(return_data=True)', 'df') == \
            'df = learner.plot_x(return_data=True)'

    def test_prepends_for_bare_expression(self):
        from mlsurv.reports.renderers._figures import _bind_var
        # The motivating case: chained subpopulation call with no assignment.
        code = "learner.subpopulation('g').plot_metric(style='forest')"
        assert _bind_var(code, 'fig') == \
            "fig = learner.subpopulation('g').plot_metric(style='forest')"

    def test_multiline_rewrites_only_first_line(self):
        from mlsurv.reports.renderers._figures import _bind_var
        code = "fig = learner.plot_metric(\n    style='forest',\n)"
        assert _bind_var(code, 'df') == \
            "df = learner.plot_metric(\n    style='forest',\n)"

    def test_comparison_is_not_treated_as_assignment(self):
        from mlsurv.reports.renderers._figures import _bind_var
        # A leading '==' must NOT be mistaken for an assignment target.
        assert _bind_var('learner.foo(x == 1)', 'df') == 'df = learner.foo(x == 1)'

    def test_result_is_executable(self):
        """_bind_var output must compile and bind the requested name."""
        from mlsurv.reports.renderers._figures import _bind_var
        ns = {'learner': type('L', (), {'plot_x': staticmethod(lambda: 7)})()}
        exec(compile(_bind_var('learner.plot_x()', 'fig'), '<t>', 'exec'), ns)
        assert ns['fig'] == 7


class TestTidyDataFrames:
    """Tests for tidy DataFrame outputs from report sections."""

    def test_base_section_has_data_dict(self):
        """BaseSection initializes with empty data dict."""
        from mlsurv.reports.sections.base import BaseSection
        from unittest.mock import MagicMock

        # Create a concrete subclass for testing
        class TestSection(BaseSection):
            name = "test"
            title = "Test"

            def is_available(self):
                return True

            def generate(self):
                return {'markdown': '', 'figures': [], 'figure_info': []}

        learner = MagicMock()
        section = TestSection(learner)
        assert hasattr(section, 'data')
        assert isinstance(section.data, dict)
        assert len(section.data) == 0

    def test_base_section_format_table_hides_all_empty_columns(self):
        """BaseSection._format_table drops fully empty columns."""
        from mlsurv.reports.sections.base import BaseSection
        from unittest.mock import MagicMock
        import pandas as pd

        class TestSection(BaseSection):
            name = "test"
            title = "Test"

            def is_available(self):
                return True

            def generate(self):
                return {'markdown': '', 'figures': [], 'figure_info': []}

        section = TestSection(MagicMock())
        df = pd.DataFrame(
            {
                'model': ['RSF', 'CoxPH'],
                'empty_col': [pd.NA, pd.NA],
                'score': [0.71, 0.69],
            }
        )
        table_md = section._format_table(df)
        header = table_md.splitlines()[0]

        assert 'empty_col' not in header
        assert 'model' in header
        assert 'score' in header

    def test_base_section_format_table_handles_array_cells(self):
        """BaseSection._format_table handles array-like cells without ambiguity errors."""
        from mlsurv.reports.sections.base import BaseSection
        from unittest.mock import MagicMock
        import numpy as np
        import pandas as pd

        class TestSection(BaseSection):
            name = "test"
            title = "Test"

            def is_available(self):
                return True

            def generate(self):
                return {'markdown': '', 'figures': [], 'figure_info': []}

        section = TestSection(MagicMock())
        df = pd.DataFrame(
            {
                'model': ['RSF', 'CoxPH'],
                'vector_metric': [np.array([0.1, 0.2]), np.array([0.3, 0.4])],
                'empty_col': [pd.NA, pd.NA],
            }
        )

        table_md = section._format_table(df)
        header = table_md.splitlines()[0]

        assert 'vector_metric' in header
        assert 'empty_col' not in header

    def test_figure_info_has_data_key(self):
        """FigureInfo dataclass includes data_key attribute."""
        from mlsurv.reports.sections.base import FigureInfo

        info = FigureInfo(
            figure_id='test_fig',
            title='Test Figure',
            data_description='Test description',
            data_key='test_key',
        )

        assert info.data_key == 'test_key'

    def test_figure_info_data_key_optional(self):
        """FigureInfo data_key is optional and defaults to None."""
        from mlsurv.reports.sections.base import FigureInfo

        info = FigureInfo(
            figure_id='test_fig',
            title='Test Figure',
            data_description='Test description',
        )

        assert info.data_key is None

    def test_figure_info_learner_code_optional(self):
        """FigureInfo learner_code and standalone_import default to None."""
        from mlsurv.reports.sections.base import FigureInfo

        info = FigureInfo(
            figure_id='test_fig',
            title='Test Figure',
            data_description='Test description',
        )
        assert info.learner_code is None
        assert info.standalone_import is None

    def test_figure_info_learner_code_set(self):
        """FigureInfo accepts learner_code parameter."""
        from mlsurv.reports.sections.base import FigureInfo

        info = FigureInfo(
            figure_id='test_fig',
            title='Test Figure',
            data_description='Test description',
            learner_code='fig = learner.plot_feature_correlation()',
        )
        assert info.learner_code == 'fig = learner.plot_feature_correlation()'

    def test_data_info_dataclass(self):
        """DataInfo dataclass works correctly."""
        from mlsurv.reports.sections.base import DataInfo

        di = DataInfo(data_key='corr', description='Correlation matrix')
        assert di.data_key == 'corr'
        assert di.description == 'Correlation matrix'
        assert di.learner_code is None
        assert di.results_code is None

    def test_consolidated_access_html_renders(self):
        """render_consolidated_access produces section-access dropdown."""
        from mlsurv.reports.renderers._figures import FigureEmbedder
        from mlsurv.reports.sections.base import FigureInfo

        infos = [
            FigureInfo(
                figure_id='corr_heatmap',
                title='Correlation Heatmap',
                data_description='Tidy correlation data',
                data_key='correlation',
                learner_code='fig = learner.plot_feature_correlation()',
            ),
        ]
        html = FigureEmbedder.render_consolidated_access(
            section_name='feature_analysis',
            figure_info_list=infos,
            data_keys=['correlation'],
        )
        assert '<details class="section-access">' in html
        assert 'Programmatic Access' in html
        # Figure: reproduce via learner.plot_* then export with Plotly's
        # native write_image (filename uses the bare figure_id).
        assert 'fig = learner.plot_feature_correlation()' in html
        assert "fig.write_image(&#x27;corr_heatmap.svg&#x27;)" in html
        # Data: figure-derived tidy frame via return_data=True (no report object).
        assert 'df = learner.plot_feature_correlation(return_data=True)' in html
        # The unreachable `report` object must be gone entirely.
        assert 'report.figures[' not in html
        assert 'report.export_figure' not in html
        assert 'report.data[' not in html
        assert 'report.export_data' not in html

    def test_consolidated_access_empty(self):
        """render_consolidated_access returns empty string when no content."""
        from mlsurv.reports.renderers._figures import FigureEmbedder

        html = FigureEmbedder.render_consolidated_access(
            section_name='test',
            figure_info_list=[],
            data_keys=[],
        )
        assert html == ""

    def test_consolidated_access_data_only(self):
        """render_consolidated_access handles sections with data but no figures."""
        from mlsurv.reports.renderers._figures import FigureEmbedder
        from mlsurv.reports.sections.base import DataInfo

        data_infos = [
            DataInfo(data_key='train_predictions', description='Train predictions',
                     results_code='learner.eval_result.train_predictions'),
            DataInfo(data_key='test_predictions', description='Test predictions',
                     results_code='learner.eval_result.test_predictions'),
        ]
        html = FigureEmbedder.render_consolidated_access(
            section_name='evaluation',
            figure_info_list=[],
            data_keys=['train_predictions', 'test_predictions'],
            data_info_list=data_infos,
        )
        assert '<details class="section-access">' in html
        # Data accessor shown directly (no unreachable report object).
        assert 'learner.eval_result.train_predictions' in html
        assert 'learner.eval_result.test_predictions' in html
        assert 'report.data[' not in html
        assert 'report.export_data' not in html

    def test_consolidated_access_with_data_info(self):
        """render_consolidated_access uses DataInfo descriptions."""
        from mlsurv.reports.renderers._figures import FigureEmbedder
        from mlsurv.reports.sections.base import DataInfo

        data_infos = [
            DataInfo(
                data_key='train_predictions',
                description='Per-sample train predictions',
                learner_code='learner.predict(X_train)',
            ),
        ]
        html = FigureEmbedder.render_consolidated_access(
            section_name='evaluation',
            figure_info_list=[],
            data_keys=['train_predictions'],
            data_info_list=data_infos,
        )
        assert 'Per-sample train predictions' in html
        assert 'learner.predict(X_train)' in html

    def test_consolidated_access_return_data_multiline(self):
        """return_data=True appended correctly for multiline learner calls."""
        from mlsurv.reports.renderers._figures import FigureEmbedder
        from mlsurv.reports.sections.base import FigureInfo

        multiline_code = (
            "fig = learner.plot_metric(\n"
            "    style='forest', source='bootstrap',\n"
            "    metric=['c_index'],\n"
            ")"
        )
        infos = [FigureInfo(
            figure_id='forest', title='Forest',
            data_description='Bootstrap data',
            data_key='bootstrap', learner_code=multiline_code,
        )]
        html = FigureEmbedder.render_consolidated_access(
            'eval', infos, ['bootstrap'],
        )
        # Must have return_data=True inserted correctly, no double comma
        assert 'return_data=True)' in html
        assert '(,' not in html
        assert ',,' not in html

    def test_markdown_consolidated_access(self, tmp_path):
        """MarkdownRenderer renders consolidated access dropdown."""
        from mlsurv.reports.renderers.markdown import MarkdownRenderer
        from mlsurv.reports.sections.base import FigureInfo

        renderer = MarkdownRenderer(str(tmp_path / 'report.md'))
        sections = [{
            'title': 'Test Section',
            'markdown': 'Content here.',
            'figures': [],
            'figure_info': [
                FigureInfo(
                    figure_id='test_fig',
                    title='Test Figure',
                    data_description='Test data',
                    learner_code='fig = learner.plot_test()',
                ),
            ],
            'tabs': None,
            'section_name': 'test_section',
            'data_keys': ['some_data'],
            'data_info': [],
        }]
        content = renderer.render(sections)
        assert '<details>' in content
        assert 'Programmatic Access' in content
        # Figure reproduced via learner.plot_* + fig.write_image (bare id).
        assert 'fig = learner.plot_test()' in content
        assert "fig.write_image('test_fig.svg')" in content
        # No unreachable report object; 'some_data' has no accessor so it's dropped.
        assert 'report.figures' not in content
        assert 'report.export_figure' not in content
        assert 'report.data' not in content

    def test_consolidated_access_return_data_no_args(self):
        """return_data=True appended correctly for no-arg learner calls."""
        from mlsurv.reports.renderers._figures import FigureEmbedder
        from mlsurv.reports.sections.base import FigureInfo

        infos = [FigureInfo(
            figure_id='corr', title='Correlation',
            data_description='Correlation data',
            data_key='correlation', learner_code='fig = learner.plot_feature_correlation()',
        )]
        html = FigureEmbedder.render_consolidated_access(
            'feat', infos, ['correlation'],
        )
        # Must NOT have stray comma: plot_feature_correlation(, return_data=True)
        assert 'plot_feature_correlation(return_data=True)' in html
        assert '(,' not in html

    def test_consolidated_access_return_data_with_args(self):
        """return_data=True appended correctly for learner calls with args."""
        from mlsurv.reports.renderers._figures import FigureEmbedder
        from mlsurv.reports.sections.base import FigureInfo

        infos = [FigureInfo(
            figure_id='imp', title='Importance',
            data_description='Importance data',
            data_key='importance', learner_code="fig = learner.plot_importance(n=20)",
        )]
        html = FigureEmbedder.render_consolidated_access(
            'feat', infos, ['importance'],
        )
        assert 'plot_importance(n=20, return_data=True)' in html

    def test_consolidated_access_standalone_import(self):
        """A figure with only standalone_import shows the import, no export line.

        Import-only figures (no learner_code, e.g. the validation metrics
        heatmap) have nothing to bind ``fig`` to, so the renderer must show the
        bare import pointer and emit no ``fig.write_image(...)`` / ``fig =``
        line that would NameError when run.
        """
        from mlsurv.reports.renderers._figures import FigureEmbedder
        from mlsurv.reports.sections.base import FigureInfo

        infos = [FigureInfo(
            figure_id='corr', title='Correlation',
            data_description='Correlation data',
            standalone_import='from mlsurv.viz import plot_feature_correlation',
        )]
        html = FigureEmbedder.render_consolidated_access(
            'feat', infos, [],
        )
        assert 'from mlsurv.viz import plot_feature_correlation' in html
        # No fig binding / export line for an import-only figure.
        assert 'fig.write_image' not in html
        assert 'fig =' not in html
        assert 'report.export_figure' not in html

    def test_consolidated_access_all_report_only_suppressed(self):
        """A section with no runnable access at all renders no dropdown.

        Report-only figures (no learner_code, no standalone_import) and data
        keys with no accessor are dropped; when nothing reachable remains, the
        whole Programmatic Access block is suppressed rather than rendered empty.
        """
        from mlsurv.reports.renderers._figures import FigureEmbedder
        from mlsurv.reports.sections.base import FigureInfo

        infos = [FigureInfo(
            figure_id='flowchart', title='Pipeline Flowchart',
            data_description='Pipeline diagram', data_key='dataset_summary',
        )]
        # Figure has no learner_code/standalone_import; data key has no accessor.
        html = FigureEmbedder.render_consolidated_access(
            'data_preprocessing', infos, ['dataset_summary'],
        )
        assert html == ""

    def test_consolidated_access_results_code(self):
        """DataInfo.results_code rendered when provided."""
        from mlsurv.reports.renderers._figures import FigureEmbedder
        from mlsurv.reports.sections.base import DataInfo

        data_infos = [DataInfo(
            data_key='metrics', description='Evaluation metrics',
            results_code='learner.eval_result.metrics',
        )]
        html = FigureEmbedder.render_consolidated_access(
            'eval', [], ['metrics'], data_info_list=data_infos,
        )
        assert 'learner.eval_result.metrics' in html

    def test_data_preprocessing_section_dataset_summary(self):
        """DataPreprocessingSection produces dataset_summary tidy DataFrame."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection
        from unittest.mock import MagicMock
        import pandas as pd

        # Create mock learner with data
        learner = MagicMock()
        learner._state.is_setup = True
        learner.model_recommendations = None
        learner.X_train = pd.DataFrame({
            'feature_a': [1, 2, 3, 4, 5],
            'feature_b': [2, 4, 6, 8, 10],
            'feature_c': [5, 4, 3, 2, 1],
        })
        learner.X_test = pd.DataFrame({
            'feature_a': [6, 7],
            'feature_b': [12, 14],
            'feature_c': [0, -1],
        })
        learner.y_train = pd.DataFrame({
            'time': [10, 20, 30, 40, 50],
            'event': [1, 0, 1, 1, 0],
        })
        learner.y_test = pd.DataFrame({
            'time': [60, 70],
            'event': [1, 0],
        })

        section = DataPreprocessingSection(learner)
        assert section.is_available()

        section.generate()

        # Check that dataset_summary DataFrame was created
        assert 'dataset_summary' in section.data
        summary_df = section.data['dataset_summary']

        # Check tidy format: metric, value
        assert 'metric' in summary_df.columns
        assert 'value' in summary_df.columns
        assert len(summary_df) > 0

    def _make_recs_learner(self, rec):
        """Build a mock learner with `model_recommendations` attached."""
        from unittest.mock import MagicMock
        import pandas as pd
        learner = MagicMock()
        learner._state.is_setup = True
        learner.model_recommendations = rec
        learner._state.subpopulations_test = None
        learner._state.subpopulations_train = None
        learner._state.presplit = False
        learner._state.stratify_array = None
        learner.cv_strategy = 'stratifiedkfold'
        learner.X_train = pd.DataFrame({
            'a': [1, 2, 3, 4, 5], 'b': [2, 4, 6, 8, 10], 'c': [5, 4, 3, 2, 1],
        })
        learner.X_test = pd.DataFrame({'a': [6, 7], 'b': [12, 14], 'c': [0, -1]})
        learner.y_train = pd.DataFrame(
            {'time': [10, 20, 30, 40, 50], 'event': [1, 0, 1, 1, 0]}
        )
        learner.y_test = pd.DataFrame({'time': [60, 70], 'event': [1, 0]})
        return learner

    def test_data_preprocessing_section_includes_model_recommendations(self):
        """Tier table + tidy DataFrame appear when state.model_recommendations is set."""
        from mlsurv.guidance import RecommendedModels
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        rec = RecommendedModels(
            recommended=['coxph', 'rsf'],
            cautionary=['coxnet'],
            not_recommended=[],
            rationale='', n_samples=200, n_events=80, n_features=10, epv=8.0,
            details={'coxph': 'EPV ok', 'rsf': 'tree-based',
                     'coxnet': 'EPV=8.0 below 10'},
            ph_test_ran=True, ph_violations=(),
            linearity_test_ran=True, linearity_violations=(),
            n_linearity_tested=4,
        )
        section = DataPreprocessingSection(self._make_recs_learner(rec))
        result = section.generate()

        assert '### Model Recommendations' in result['markdown']
        # Column headers — color-coded HTML table emitted as raw HTML
        for header in ('Sample size', 'EPV', 'PH', 'Linearity', 'Recommendation'):
            assert header in result['markdown'], f"Missing column header: {header}"
        # EPV=8 < 10 → feature-reduction note
        assert 'below 10' in result['markdown']
        # Both tests ran clean → both summary lines present
        assert 'PH test' in result['markdown']
        assert 'Linearity test' in result['markdown']

        df = section.data['model_recommendations']
        assert list(df.columns) == [
            'model', 'sample_size', 'epv', 'ph', 'linearity', 'tier'
        ]
        assert len(df) == 3
        assert set(df['tier']) == {'Recommended', 'Cautionary'}

    def test_model_recommendations_omits_lines_when_tests_did_not_run(self):
        """No PH/linearity summary line when *_test_ran is False."""
        from mlsurv.guidance import RecommendedModels
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        rec = RecommendedModels(
            recommended=['coxph'], cautionary=[], not_recommended=[],
            rationale='', n_samples=200, n_events=100, n_features=5, epv=20.0,
            details={'coxph': 'EPV ok'},
            ph_test_ran=False, linearity_test_ran=False,
        )
        section = DataPreprocessingSection(self._make_recs_learner(rec))
        result = section.generate()
        # Check only the recommendations subsection — other diagnostics
        # subsections may emit their own (unrelated) "Linearity test" text.
        recs_block = result['markdown'].split('### Model Recommendations', 1)[1]
        recs_block = recs_block.split('### Missing Data', 1)[0]
        assert 'Non-PH features' not in recs_block
        assert 'PH test:' not in recs_block
        assert 'Linearity test' not in recs_block
        assert 'Log-hazard non-linear features' not in recs_block

    def test_model_recommendations_renders_violations(self):
        """Violator names surface verbatim in markdown."""
        from mlsurv.guidance import RecommendedModels
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        rec = RecommendedModels(
            recommended=['rsf'], cautionary=['coxph'], not_recommended=[],
            rationale='', n_samples=200, n_events=100, n_features=5, epv=20.0,
            details={'rsf': 'ok', 'coxph': 'PH violations'},
            ph_test_ran=True, ph_violations=('age', 'grade'),
            linearity_test_ran=True, linearity_violations=('biomarker_x',),
            n_linearity_tested=3,
        )
        section = DataPreprocessingSection(self._make_recs_learner(rec))
        result = section.generate()

        assert 'age' in result['markdown'] and 'grade' in result['markdown']
        assert 'biomarker_x' in result['markdown']
        assert 'Non-PH features (2)' in result['markdown']
        assert 'Log-hazard non-linear features (1)' in result['markdown']

    def test_model_recommendations_extreme_low_epv_warning(self):
        """EPV < TOO_SMALL_EPV (=2) emits the bold WARNING form, not the milder Note."""
        from mlsurv.guidance import RecommendedModels
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        rec = RecommendedModels(
            recommended=[], cautionary=[], not_recommended=['coxph'],
            rationale='', n_samples=20, n_events=2, n_features=10, epv=0.2,
            details={'coxph': 'EPV=0.2 < 5'},
        )
        section = DataPreprocessingSection(self._make_recs_learner(rec))
        result = section.generate()

        recs_block = result['markdown'].split('### Model Recommendations', 1)[1]
        recs_block = recs_block.split('### Missing Data', 1)[0]
        assert '**WARNING:' in recs_block
        assert 'extremely low' in recs_block
        assert '*Note:' not in recs_block

    def test_model_recommendations_inf_epv_formatting(self):
        """epv=inf renders as 'inf' (not 'inf.0' or a TypeError)."""
        from mlsurv.guidance import RecommendedModels
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        rec = RecommendedModels(
            recommended=['coxph'], cautionary=[], not_recommended=[],
            rationale='', n_samples=200, n_events=0, n_features=5,
            epv=float('inf'),
            details={'coxph': 'no events'},
        )
        section = DataPreprocessingSection(self._make_recs_learner(rec))
        result = section.generate()

        assert 'EPV=inf' in result['markdown']
        assert 'EPV=inf.0' not in result['markdown']

    def test_model_recommendations_empty_tiers(self):
        """All-empty tier lists produce no markdown table but still render advisories."""
        from mlsurv.guidance import RecommendedModels
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        rec = RecommendedModels(
            recommended=[], cautionary=[], not_recommended=[],
            rationale='', n_samples=200, n_events=100, n_features=5, epv=20.0,
            details={},
            ph_test_ran=True, ph_violations=(),
        )
        section = DataPreprocessingSection(self._make_recs_learner(rec))
        result = section.generate()

        df = section.data['model_recommendations']
        assert df.empty
        assert list(df.columns) == [
            'model', 'sample_size', 'epv', 'ph', 'linearity', 'tier'
        ]
        recs_block = result['markdown'].split('### Model Recommendations', 1)[1]
        recs_block = recs_block.split('### Missing Data', 1)[0]
        # Table should NOT be emitted when there are no rows
        assert '<table' not in recs_block
        assert 'MLSURV_RAW_HTML' not in recs_block
        # PH summary line still renders
        assert 'PH test:' in recs_block

    def test_model_recommendations_tier_ordering_and_singular(self):
        """Tidy DataFrame rows preserve Recommended → Cautionary → Not recommended order.

        Also exercises ``n_linearity_tested=1`` (singular phrasing).
        """
        from mlsurv.guidance import RecommendedModels
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        rec = RecommendedModels(
            recommended=['rsf'],
            cautionary=['coxph'],
            not_recommended=['deepsurv'],
            rationale='', n_samples=200, n_events=100, n_features=5, epv=20.0,
            details={'rsf': 'r', 'coxph': 'c', 'deepsurv': 'n'},
            linearity_test_ran=True, linearity_violations=(),
            n_linearity_tested=1,
        )
        section = DataPreprocessingSection(self._make_recs_learner(rec))
        result = section.generate()

        df = section.data['model_recommendations']
        assert list(df['tier']) == ['Recommended', 'Cautionary', 'Not recommended']
        # Singular form: "all 1 continuous feature" (no trailing 's')
        assert 'all 1 continuous feature pass' in result['markdown']
        assert 'all 1 continuous features' not in result['markdown']

    def test_model_recommendations_threshold_labels(self):
        """Sample-size and EPV cells label the band met (or the threshold missed)."""
        from mlsurv.guidance import RecommendedModels
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        # tree_ensemble class: n_rec=500, n_caut=200, epv_rec=50, epv_caut=20.
        # N=300 → cautionary band; EPV=10 → below cautionary.
        rec = RecommendedModels(
            recommended=[], cautionary=[], not_recommended=['rsf'],
            rationale='', n_samples=300, n_events=30, n_features=3, epv=10.0,
            details={'rsf': 'tree caut'},
        )
        section = DataPreprocessingSection(self._make_recs_learner(rec))
        section.generate()
        row = section.data['model_recommendations'].iloc[0]
        # tree_ensemble: n_rec=500, n_caut=200, epv_rec=50, epv_caut=20.
        # N=300 lands in cautionary band → chained inequality showing both bounds.
        assert row['sample_size'] == '200 ≤ N=300 < 500'
        # EPV=10 falls below cautionary → only the missed bound shown.
        assert row['epv'] == 'EPV=10.0 < 20'

    def test_model_recommendations_per_criterion_status(self):
        """PH/Linearity columns show N/A for assumption-free models."""
        from mlsurv.guidance import RecommendedModels
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        rec = RecommendedModels(
            recommended=['coxph'],          # assumes PH and linearity
            cautionary=['rsf'],             # assumption-free
            not_recommended=[],
            rationale='', n_samples=200, n_events=100, n_features=5, epv=20.0,
            details={'coxph': 'EPV ok', 'rsf': 'caut'},
            ph_test_ran=True, ph_violations=(),
            linearity_test_ran=True, linearity_violations=(),
            n_linearity_tested=3,
        )
        section = DataPreprocessingSection(self._make_recs_learner(rec))
        section.generate()
        df = section.data['model_recommendations']

        coxph_row = df[df['model'] == 'CoxPH'].iloc[0]
        rsf_row = df[df['model'] == 'RSF'].iloc[0]
        # CoxPH (linear class, no n_min) shows raw N — no threshold to compare
        assert coxph_row['sample_size'] == 'N=200'
        # EPV=20.0 meets the linear recommended threshold (≥20)
        assert coxph_row['epv'] == 'EPV=20.0 ≥ 20'
        assert coxph_row['ph'] == 'OK'
        assert coxph_row['linearity'] == 'OK'
        # RSF assumes neither PH nor linearity
        assert rsf_row['ph'] == 'N/A'
        assert rsf_row['linearity'] == 'N/A'

    def test_model_recommendations_assumption_unknown_when_test_did_not_run(self):
        """PH column shows 'not tested' for PH-assuming models when test didn't run."""
        from mlsurv.guidance import RecommendedModels
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        rec = RecommendedModels(
            recommended=['coxph'], cautionary=[], not_recommended=[],
            rationale='', n_samples=200, n_events=100, n_features=5, epv=20.0,
            details={'coxph': 'EPV ok'},
            ph_test_ran=False, linearity_test_ran=False,
        )
        section = DataPreprocessingSection(self._make_recs_learner(rec))
        section.generate()
        df = section.data['model_recommendations']
        row = df.iloc[0]
        assert row['ph'] == 'not tested'
        assert row['linearity'] == 'not tested'

    def test_model_recommendations_handles_missing_state(self):
        """No header / no data when model_recommendations is None."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        section = DataPreprocessingSection(self._make_recs_learner(None))
        result = section.generate()

        assert '### Model Recommendations' not in result['markdown']
        assert 'model_recommendations' not in section.data

    def test_report_generator_data_property(self):
        """ReportGenerator aggregates data from sections."""
        from mlsurv.reports import ReportGenerator
        from mlsurv.reports.sections.base import BaseSection
        from unittest.mock import MagicMock
        import pandas as pd

        learner = MagicMock()
        learner._state.is_setup = True
        learner.models = {}

        generator = ReportGenerator(learner)

        # Create mock sections with data
        section1 = MagicMock(spec=BaseSection)
        section1.name = 'section1'
        section1.data = {'key1': pd.DataFrame({'a': [1, 2]})}

        section2 = MagicMock(spec=BaseSection)
        section2.name = 'section2'
        section2.data = {'key2': pd.DataFrame({'b': [3, 4]})}

        generator._sections = [section1, section2]
        generator._data = None  # Clear cache

        # Access data property
        data = generator.data

        assert 'section1_key1' in data
        assert 'section2_key2' in data
        assert len(data['section1_key1']) == 2
        assert len(data['section2_key2']) == 2

    def test_report_generator_data_empty_before_generate(self):
        """ReportGenerator.data is empty before generate() is called."""
        from mlsurv.reports import ReportGenerator
        from unittest.mock import MagicMock

        learner = MagicMock()
        generator = ReportGenerator(learner)

        # Before generate(), _sections is empty so data should be empty
        assert len(generator.data) == 0


class TestEvaluationSectionStratification:
    """Tests for EvaluationSection stratification (replaces SurvivalCurvesSection tests)."""

    def test_evaluation_section_has_risk_scores(self):
        """Test EvaluationSection._has_risk_scores with single model."""
        from mlsurv.reports.sections.evaluation import EvaluationSection
        from unittest.mock import MagicMock
        import pandas as pd

        learner = MagicMock()
        learner._state.is_setup = True
        learner.models = {}

        eval_result = MagicMock()
        eval_result.test_set = pd.DataFrame({
            'time': [10, 20, 30, 40, 50],
            'event': [1, 0, 1, 1, 0],
            'risk_score': [0.1, 0.3, 0.5, 0.7, 0.9],
        })
        eval_result.concordance = pd.DataFrame({
            'model_type': ['coxph'],
            'c_index': [0.75],
        })
        eval_result.auc_t = None
        eval_result.bootstrap = None

        section = EvaluationSection(learner, eval_result=eval_result)
        assert section._has_risk_scores()

    def test_evaluation_section_get_test_data_with_risk_multi_model(self):
        """Test EvaluationSection._get_test_data_with_risk returns dict for multiple models."""
        from mlsurv.reports.sections.evaluation import EvaluationSection
        from unittest.mock import MagicMock
        import pandas as pd

        learner = MagicMock()
        learner._state.is_setup = True
        learner.models = {}

        eval_result = MagicMock()
        eval_result.test_set = pd.DataFrame({
            'model_type': ['coxph', 'coxph', 'rsf', 'rsf'],
            'time': [10, 20, 10, 20],
            'event': [1, 0, 1, 0],
            'risk_score': [0.2, 0.8, 0.3, 0.7],
        })
        eval_result.concordance = pd.DataFrame({
            'model_type': ['coxph', 'rsf'],
            'c_index': [0.72, 0.75],
        })
        eval_result.auc_t = None
        eval_result.bootstrap = None

        section = EvaluationSection(learner, eval_result=eval_result)
        result = section._get_test_data_with_risk()

        assert isinstance(result, dict)
        assert len(result) == 2
        assert 'coxph' in result
        assert 'rsf' in result


class TestEvaluationSectionRMST:
    """Tests for RMST tables in the stratification report subsection."""

    @staticmethod
    def _make_section(n_samples=100, n_groups=4, multi=False):
        """Create an EvaluationSection with synthetic risk-score data."""
        import numpy as np
        import pandas as pd
        from mlsurv.reports.sections.evaluation import EvaluationSection
        from unittest.mock import MagicMock

        rng = np.random.RandomState(42)

        if multi:
            models = ['coxph', 'rsf']
            model_types = np.repeat(models, n_samples)
            times = np.concatenate([
                rng.exponential(50, n_samples) for _ in models
            ])
            events = np.concatenate([
                rng.binomial(1, 0.7, n_samples) for _ in models
            ])
            risk_scores = np.concatenate([
                rng.uniform(0, 1, n_samples) for _ in models
            ])
            test_set = pd.DataFrame({
                'model_type': model_types,
                'time': times,
                'event': events.astype(int),
                'risk_score': risk_scores,
            })
        else:
            times = rng.exponential(50, n_samples)
            events = rng.binomial(1, 0.7, n_samples)
            risk_scores = rng.uniform(0, 1, n_samples)
            test_set = pd.DataFrame({
                'time': times,
                'event': events.astype(int),
                'risk_score': risk_scores,
            })

        learner = MagicMock()
        learner._state.is_setup = True
        learner._state.outcome_name = "event"
        learner._state.time_unit = None
        learner.models = {}

        eval_result = MagicMock()
        eval_result.test_set = test_set
        eval_result.concordance = pd.DataFrame({
            'model_type': ['coxph'], 'c_index': [0.75],
        })
        eval_result.auc_t = None
        eval_result.bootstrap = None

        return EvaluationSection(
            learner, eval_result=eval_result, n_groups=n_groups,
        )

    def test_rmst_table_single_model(self):
        """Single-model path renders RMST tables and stores data."""
        from mlsurv.reports.sections.evaluation._stratification import (
            _StratificationSubsection,
        )

        section = self._make_section(n_samples=100, n_groups=4)
        sub = _StratificationSubsection(section)

        test_data_by_model = sub._get_test_data_with_risk()
        model_type = next(iter(test_data_by_model))
        test_data = test_data_by_model[model_type]

        test_data, actual_groups = section._create_risk_groups(
            test_data, section.n_groups,
        )
        assert test_data is not None

        md_lines = []
        sub._append_rmst_table(test_data, actual_groups, md_lines, model_type)

        md_text = '\n'.join(md_lines)
        assert 'RMST by Risk Group' in md_text
        assert 'Pairwise RMST Differences' in md_text
        assert '\u03c4' in md_text or 'τ' in md_text

        assert 'rmst_comparison' in section.data
        rmst_df = section.data['rmst_comparison']
        assert 'risk_group' in rmst_df.columns
        assert 'rmst' in rmst_df.columns
        assert 'se' in rmst_df.columns
        assert 'tau' in rmst_df.columns
        assert len(rmst_df) == actual_groups

        assert 'rmst_pairwise' in section.data
        pw_df = section.data['rmst_pairwise']
        assert 'group_0' in pw_df.columns
        assert 'group_1' in pw_df.columns
        assert 'diff' in pw_df.columns
        assert 'adjusted_p' in pw_df.columns

    def test_rmst_comparison_multi_model(self):
        """Multi-model path renders RMST summary table with all model names."""
        from mlsurv.reports.sections.evaluation._stratification import (
            _StratificationSubsection,
        )

        section = self._make_section(n_samples=100, n_groups=4, multi=True)
        sub = _StratificationSubsection(section)

        test_data_by_model = sub._get_test_data_with_risk()
        grouped_by_model = {}
        for mt, td in test_data_by_model.items():
            gd, ag = section._create_risk_groups(td, section.n_groups)
            if gd is not None:
                grouped_by_model[mt] = (gd, ag)

        assert len(grouped_by_model) >= 2

        md_lines = []
        sub._append_rmst_comparison_multi(grouped_by_model, md_lines)

        md_text = '\n'.join(md_lines)
        assert 'RMST Comparison' in md_text
        for mt in grouped_by_model:
            from mlsurv.viz._palettes import get_model_display_name
            assert get_model_display_name(mt) in md_text

        assert 'rmst_pairwise' in section.data
        pw_df = section.data['rmst_pairwise']
        assert 'model_type' in pw_df.columns
        assert set(pw_df['model_type'].unique()) == set(grouped_by_model.keys())

        # Multi-model RMST table includes the per-model τ caption.
        assert 'longest follow-up at which RMST is comparable' in md_text

    def test_rmst_degenerate_groups_handled(self):
        """All-censored group doesn't crash; RMST is computed or skipped."""
        import numpy as np
        import pandas as pd
        from mlsurv.reports.sections.evaluation._stratification import (
            _StratificationSubsection,
        )

        section = self._make_section(n_samples=60, n_groups=3)
        sub = _StratificationSubsection(section)

        # Create data where one group is all-censored
        rng = np.random.RandomState(99)
        test_data = pd.DataFrame({
            'time': rng.exponential(50, 60),
            'event': ([0] * 20) + list(rng.binomial(1, 0.8, 40)),
            'risk_score': np.linspace(0, 1, 60),
        })

        test_data, actual_groups = section._create_risk_groups(
            test_data, section.n_groups,
        )
        assert test_data is not None

        md_lines = []
        # Should not raise
        sub._append_rmst_table(
            test_data, actual_groups, md_lines, model_type='coxph',
        )

    def test_rmst_narrative_extreme_pair(self):
        """Narrative mentions Q1 and Qn with correct difference direction."""
        from mlsurv.reports.sections.evaluation._stratification import (
            _StratificationSubsection,
        )

        section = self._make_section(n_samples=200, n_groups=4)
        sub = _StratificationSubsection(section)

        test_data_by_model = sub._get_test_data_with_risk()
        model_type = next(iter(test_data_by_model))
        test_data = test_data_by_model[model_type]

        test_data, actual_groups = section._create_risk_groups(
            test_data, section.n_groups,
        )

        md_lines = []
        sub._append_rmst_table(test_data, actual_groups, md_lines, model_type)

        md_text = '\n'.join(md_lines)
        # Narrative should mention both extreme groups
        assert 'Q1' in md_text
        assert f'Q{actual_groups}' in md_text
        # Should contain "time units" and a significance statement
        assert 'time units' in md_text
        assert 'statistically significant' in md_text

    def test_rmst_narrative_sign_conditional_expected(self):
        """When diff < 0 (low-risk survived longer), narrative says 'low-risk group'."""
        from mlsurv.reports.sections.evaluation._stratification import (
            _build_rmst_narrative,
        )

        extreme = {
            'groups': ('Q1', 'Q4'),
            'diff': -5.0,
            'adjusted_p': 0.01,
        }
        group_map = {'Q1': 'Q1 (Low)', 'Q4': 'Q4 (High)'}
        md_lines = []
        _build_rmst_narrative(md_lines, extreme, group_map, tau=60.0, time_unit="months")

        text = '\n'.join(md_lines)
        assert 'low-risk group (Q1 (Low))' in text
        assert 'high-risk group (Q4 (High))' in text
        assert 'Unexpectedly' not in text
        assert 'statistically significant' in text
        assert 'months' in text

    def test_rmst_narrative_sign_conditional_unexpected(self):
        """When diff > 0 (high-risk survived longer), narrative says 'Unexpectedly'."""
        from mlsurv.reports.sections.evaluation._stratification import (
            _build_rmst_narrative,
        )

        extreme = {
            'groups': ('Q1', 'Q4'),
            'diff': 3.0,
            'adjusted_p': 0.20,
        }
        group_map = {'Q1': 'Q1 (Low)', 'Q4': 'Q4 (High)'}
        md_lines = []
        _build_rmst_narrative(md_lines, extreme, group_map, tau=60.0, time_unit="days")

        text = '\n'.join(md_lines)
        assert 'Unexpectedly' in text
        assert 'high-risk group (Q4 (High))' in text
        assert 'low-risk group (Q1 (Low))' in text
        assert 'miscalibration' in text
        assert 'not statistically significant' in text
        assert 'days' in text

    def test_rmst_narrative_equivalent_groups(self):
        """When diff == 0, narrative says groups have equivalent RMST."""
        from mlsurv.reports.sections.evaluation._stratification import (
            _build_rmst_narrative,
        )

        extreme = {
            'groups': ('Q1', 'Q4'),
            'diff': 0.0,
            'adjusted_p': 1.0,
        }
        group_map = {'Q1': 'Q1 (Low)', 'Q4': 'Q4 (High)'}
        md_lines = []
        _build_rmst_narrative(md_lines, extreme, group_map, tau=60.0, time_unit="months")

        text = '\n'.join(md_lines)
        assert 'equivalent' in text
        assert 'Unexpectedly' not in text
        assert 'survived on average' not in text

    def test_validation_rmst_narrative_uses_time_unit(self):
        """Validation _build_rmst_narrative uses time_unit parameter, not hardcoded."""
        from unittest.mock import MagicMock
        from mlsurv.reports.sections.validation._stratification import (
            _build_rmst_narrative,
        )

        group_keys = ['Q1', 'Q4']
        group_map = {'Q1': 'Q1', 'Q4': 'Q4'}

        # Expected case (diff < 0): low-risk survived longer
        result = MagicMock()
        result.tau = 60.0
        result.pairwise = [{
            'groups': ('Q1', 'Q4'),
            'diff': -5.0,
            'adjusted_p': 0.01,
        }]
        text = '\n'.join(_build_rmst_narrative(result, group_keys, group_map, time_unit="years"))
        assert 'years' in text
        assert 'time units' not in text

        # Unexpected case (diff > 0): high-risk survived longer
        result.pairwise = [{
            'groups': ('Q1', 'Q4'),
            'diff': 3.0,
            'adjusted_p': 0.20,
        }]
        text = '\n'.join(_build_rmst_narrative(result, group_keys, group_map, time_unit="years"))
        assert 'Unexpectedly' in text
        assert 'years' in text
        assert 'time units' not in text
        assert 'miscalibration' in text

        # Default parameter should produce "time units"
        text_default = '\n'.join(_build_rmst_narrative(result, group_keys, group_map))
        assert 'time units' in text_default

    def test_importance_significance_narrative_paragraph_breaks(self):
        """Narrative joins lines with paragraph breaks, not single spaces."""
        from unittest.mock import MagicMock
        import pandas as pd
        from mlsurv.reports.sections.feature_analysis._importance import (
            generate_importance_significance_narrative,
        )

        section = MagicMock()
        section._pvalue_source = "pimp"
        models = ['CoxPH']
        imp_pvals = pd.DataFrame({
            'feature': ['age', 'stage'],
            'p_adjusted': [0.01, 0.80],
        })

        text = generate_importance_significance_narrative(section, models, {'CoxPH': imp_pvals})
        assert '\n\n' in text
        # Each model header should start on its own paragraph
        parts = text.split('\n\n')
        assert len(parts) >= 2

    def test_no_redundant_create_risk_groups_multi(self):
        """Multi-model path calls _create_risk_groups once per model, not 2x."""
        from unittest.mock import patch
        from mlsurv.reports.sections.evaluation._stratification import (
            _StratificationSubsection,
        )

        section = self._make_section(n_samples=80, n_groups=3, multi=True)
        sub = _StratificationSubsection(section)

        test_data_by_model = sub._get_test_data_with_risk()
        n_models = len(test_data_by_model)

        original_create = type(section)._create_risk_groups

        with patch.object(
            type(section), '_create_risk_groups',
            side_effect=original_create,
        ) as mock_create:
            md_lines = []
            figures = []
            figure_info = []
            sub._add_multi_model_stratification(
                test_data_by_model, md_lines, figures, figure_info,
            )
            assert mock_create.call_count == n_models


class TestEvaluationSectionRMSTErrorPaths:
    """Tests for RMST error handling when rmst_test_pairwise fails."""

    @staticmethod
    def _make_section(n_samples=100, n_groups=4, multi=False):
        """Reuse factory from TestEvaluationSectionRMST."""
        return TestEvaluationSectionRMST._make_section(n_samples, n_groups, multi)

    def test_rmst_single_model_error_no_crash(self, monkeypatch):
        """Single-model: rmst_test_pairwise raises → section renders without RMST."""
        from mlsurv.reports.sections.evaluation._stratification import (
            _StratificationSubsection,
        )

        section = self._make_section(n_samples=100, n_groups=4)
        sub = _StratificationSubsection(section)

        test_data_by_model = sub._get_test_data_with_risk()
        model_type = next(iter(test_data_by_model))
        test_data = test_data_by_model[model_type]
        test_data, actual_groups = section._create_risk_groups(
            test_data, section.n_groups,
        )
        assert test_data is not None

        monkeypatch.setattr(
            "mlsurv.evaluation.metrics.rmst_test_pairwise",
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("mock RMST fail")),
        )

        md_lines = []
        sub._append_rmst_table(test_data, actual_groups, md_lines, model_type)
        # Should not crash; no RMST table rendered
        assert 'RMST by Risk Group' not in '\n'.join(md_lines)

    def test_rmst_multi_model_partial_error(self, monkeypatch):
        """Multi-model: one model raises → other models' RMST rows still appear."""
        from mlsurv.reports.sections.evaluation._stratification import (
            _StratificationSubsection,
        )
        from mlsurv.evaluation.metrics import rmst_test_pairwise as real_rmst

        section = self._make_section(n_samples=100, n_groups=3, multi=True)
        sub = _StratificationSubsection(section)

        test_data_by_model = sub._get_test_data_with_risk()
        grouped_by_model = {}
        for model_type, test_data in test_data_by_model.items():
            grouped_data, actual_groups = section._create_risk_groups(
                test_data, section.n_groups,
            )
            if grouped_data is not None:
                grouped_by_model[model_type] = (grouped_data, actual_groups)

        assert len(grouped_by_model) >= 2
        call_count = [0]

        def patched_rmst(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("mock first-model RMST fail")
            return real_rmst(*args, **kwargs)

        monkeypatch.setattr(
            "mlsurv.evaluation.metrics.rmst_test_pairwise",
            patched_rmst,
        )

        md_lines = []
        sub._append_rmst_comparison_multi(grouped_by_model, md_lines)
        md_text = '\n'.join(md_lines)
        # At least one model's RMST should appear
        assert 'RMST Comparison' in md_text

    def test_rmst_error_logs_with_warn_user(self, monkeypatch):
        """Verify _log_recoverable_exception is called with warn_user=True."""
        from unittest.mock import MagicMock
        from mlsurv.reports.sections.evaluation._stratification import (
            _StratificationSubsection,
        )

        section = self._make_section(n_samples=100, n_groups=4)
        sub = _StratificationSubsection(section)

        test_data_by_model = sub._get_test_data_with_risk()
        model_type = next(iter(test_data_by_model))
        test_data = test_data_by_model[model_type]
        test_data, actual_groups = section._create_risk_groups(
            test_data, section.n_groups,
        )

        monkeypatch.setattr(
            "mlsurv.evaluation.metrics.rmst_test_pairwise",
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("mock fail")),
        )

        log_mock = MagicMock()
        section._log_recoverable_exception = log_mock

        md_lines = []
        sub._append_rmst_table(test_data, actual_groups, md_lines, model_type)

        log_mock.assert_called_once()
        call_kwargs = log_mock.call_args
        assert call_kwargs[1].get('warn_user') is True


class TestGetRiskFromLearnerWarnUser:
    """Tests for _get_risk_from_learner narrowed except + warn_user."""

    def test_value_error_caught_and_warns(self):
        """ValueError from model.predict() is caught and warns user."""
        import pandas as pd
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from mlsurv.reports.sections.evaluation import EvaluationSection
        from mlsurv.reports.sections.evaluation._stratification import (
            _StratificationSubsection,
        )

        bad_model = MagicMock()
        bad_model.predict.side_effect = ValueError("bad input")
        model_result = SimpleNamespace(model=bad_model, is_tuned=False)

        learner = MagicMock()
        learner._state.is_setup = True
        learner.X_test = pd.DataFrame({"x": [1, 2, 3]})
        learner.y_test = pd.DataFrame({"time": [1.0, 2.0, 3.0], "event": [1, 0, 1]})
        learner.models = {"bad": model_result}

        eval_result = MagicMock()
        eval_result.test_set = None
        eval_result.concordance = pd.DataFrame({"model_type": ["bad"], "c_index": [0.5]})
        eval_result.auc_t = None
        eval_result.bootstrap = None

        section = EvaluationSection(learner, eval_result=eval_result)
        sub = _StratificationSubsection(section)

        log_mock = MagicMock()
        section._log_recoverable_exception = log_mock

        result = sub._get_risk_from_learner()
        assert result == {}  # No models succeeded
        log_mock.assert_called_once()
        assert log_mock.call_args[1].get('warn_user') is True

    def test_key_error_propagates(self):
        """KeyError (not in narrowed tuple) propagates instead of being caught."""
        import pandas as pd
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from mlsurv.reports.sections.evaluation import EvaluationSection
        from mlsurv.reports.sections.evaluation._stratification import (
            _StratificationSubsection,
        )

        bad_model = MagicMock()
        bad_model.predict.side_effect = KeyError("unexpected")
        model_result = SimpleNamespace(model=bad_model, is_tuned=False)

        learner = MagicMock()
        learner._state.is_setup = True
        learner.X_test = pd.DataFrame({"x": [1, 2, 3]})
        learner.y_test = pd.DataFrame({"time": [1.0, 2.0, 3.0], "event": [1, 0, 1]})
        learner.models = {"bad": model_result}

        eval_result = MagicMock()
        eval_result.test_set = None
        eval_result.concordance = pd.DataFrame({"model_type": ["bad"], "c_index": [0.5]})
        eval_result.auc_t = None
        eval_result.bootstrap = None

        section = EvaluationSection(learner, eval_result=eval_result)
        sub = _StratificationSubsection(section)

        with pytest.raises(KeyError, match="unexpected"):
            sub._get_risk_from_learner()


class TestEvaluationSectionCalibrationDedup:
    """Tests for calibration interactive dropdown in EvaluationSection."""

    def test_single_model_calibration_single_predict_call(self, small_data):
        """Single-model calibration calls predict_survival_function once for all times."""
        from types import SimpleNamespace

        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.evaluation import EvaluationSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        model = _CountingCalibrationModel()
        learner._state.models = {
            'mock': SimpleNamespace(model=model, is_tuned=False, evaluation=None),
        }

        section = EvaluationSection(learner, n_bins=3)
        md_lines = []
        figures = []
        figure_info = []

        _CalibrationSubsection(section).add(md_lines, figures, figure_info)

        # compute_calibration calls predict once for all eval_times
        assert model.predict_calls == 1
        assert len(figures) == 1
        assert 'calibration' in section.data
        assert (section.data['calibration']['model_type'] == 'mock').all()

    def test_multi_model_calibration_predict_calls(self, small_data):
        """Multi-model calibration calls each model once per eval_time."""
        from types import SimpleNamespace

        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.evaluation import EvaluationSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        model_a = _CountingCalibrationModel()
        model_b = _CountingCalibrationModel(variant="shifted")
        learner._state.models = {
            'mock_a': SimpleNamespace(model=model_a, is_tuned=False, evaluation=None),
            'mock_b': SimpleNamespace(model=model_b, is_tuned=False, evaluation=None),
        }

        section = EvaluationSection(learner, n_bins=3)
        md_lines = []
        figures = []
        figure_info = []

        _CalibrationSubsection(section).add(md_lines, figures, figure_info)

        # Fallback to 3 percentile times; each model called once per time
        assert model_a.predict_calls == 3
        assert model_b.predict_calls == 3
        assert len(figures) == 1
        assert 'calibration' in section.data
        assert set(section.data['calibration']['model_type'].unique()) == {'mock_a', 'mock_b'}

    def test_single_model_calibration_keeps_empty_data_key(self, small_data, monkeypatch):
        """Single-model calibration keeps empty data key when no bins qualify."""
        from types import SimpleNamespace

        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.evaluation import EvaluationSection
        import mlsurv.viz.calibration as calibration_viz

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        monkeypatch.setattr(
            calibration_viz,
            "_compute_bin_calibration",
            lambda pred_probs, y_test, t, n_bins, bin_strategy="width", context="": (
                [], [], [], []
            ),
        )

        model = _CountingCalibrationModel()
        learner._state.models = {
            'mock': SimpleNamespace(model=model, is_tuned=False, evaluation=None),
        }

        section = EvaluationSection(learner, n_bins=100)
        md_lines = []
        figures = []
        figure_info = []

        _CalibrationSubsection(section).add(md_lines, figures, figure_info)

        expected_columns = [
            'model_type', 'eval_time',
            'bin_predicted', 'bin_observed', 'bin_ci_lower', 'bin_ci_upper',
        ]

        assert model.predict_calls == 1
        assert len(figures) == 1
        assert 'calibration' in section.data
        assert len(section.data['calibration']) == 0
        assert list(section.data['calibration'].columns) == expected_columns

    def test_multi_model_calibration_keeps_empty_data_key(self, small_data, monkeypatch):
        """Multi-model calibration keeps empty data key when no bins qualify."""
        from types import SimpleNamespace

        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.evaluation import EvaluationSection
        import mlsurv.viz.calibration as calibration_viz

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        monkeypatch.setattr(
            calibration_viz,
            "_compute_bin_calibration",
            lambda pred_probs, y_test, t, n_bins, bin_strategy="width", context="": (
                [], [], [], []
            ),
        )

        model_a = _CountingCalibrationModel()
        model_b = _CountingCalibrationModel(variant="shifted")
        learner._state.models = {
            'mock_a': SimpleNamespace(model=model_a, is_tuned=False, evaluation=None),
            'mock_b': SimpleNamespace(model=model_b, is_tuned=False, evaluation=None),
        }

        section = EvaluationSection(learner, n_bins=100)
        md_lines = []
        figures = []
        figure_info = []

        _CalibrationSubsection(section).add(md_lines, figures, figure_info)

        expected_columns = [
            'model_type', 'eval_time',
            'bin_predicted', 'bin_observed', 'bin_ci_lower', 'bin_ci_upper',
        ]

        assert model_a.predict_calls == 3
        assert model_b.predict_calls == 3
        assert len(figures) == 1
        assert 'calibration' in section.data
        assert len(section.data['calibration']) == 0
        assert list(section.data['calibration'].columns) == expected_columns


class TestTuningSectionTests:
    """Tests for tuning content within ModelDevelopmentSection."""

    def test_tuning_content_with_tuned_model(self, small_data, tmp_path):
        """Test ModelDevelopmentSection includes tuning content for tuned model."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        learner.tune('coxnet', n_trials=2, evaluate=False, bootstrap=False)

        section = ModelDevelopmentSection(learner)
        if section.is_available():
            result = section.generate()
            assert 'markdown' in result
            # Tuning content should appear in the markdown
            assert 'Hyperparameter Tuning' in result['markdown']

    def test_no_tuning_content_without_tuned(self, small_data):
        """Test ModelDevelopmentSection omits tuning when no models are tuned."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('coxph')

        section = ModelDevelopmentSection(learner)
        result = section.generate()
        assert 'Hyperparameter Tuning' not in result['markdown']

    def test_tuning_content_after_run(self, small_data):
        """Test ModelDevelopmentSection includes tuning when models are tuned via run()."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.run(models=['coxnet'], n_trials=2, evaluate=False, bootstrap=False, verbose=False)

        section = ModelDevelopmentSection(learner)
        assert section.is_available()
        result = section.generate()
        assert 'Selected Hyperparameters' in result['markdown']

    def test_tuning_alias_resolves(self):
        """Test that sections=['tuning'] resolves to model_development."""
        from mlsurv.reports.generator import ReportGenerator

        resolved = ReportGenerator._resolve_sections(['tuning'])
        assert resolved == ['model_development']

    def test_tuning_alias_deduplicates(self):
        """Test that ['model_development', 'tuning'] deduplicates."""
        from mlsurv.reports.generator import ReportGenerator

        resolved = ReportGenerator._resolve_sections(['model_development', 'tuning'])
        assert resolved == ['model_development']

    def test_tuning_config_no_cv_duplicates(self, small_data):
        """Test tuning config table has only Models Tuned / Metric / Trials."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.tune('coxnet', n_trials=2, evaluate=False, bootstrap=False)

        section = ModelDevelopmentSection(learner)
        result = section.generate()
        md = result['markdown']

        # Tuning config should NOT duplicate CV Strategy / CV Folds
        # Find the tuning config table and check its content
        lines = md.split('\n')
        in_tuning_config = False
        tuning_config_lines = []
        for line in lines:
            if 'Tuning Configuration' in line:
                in_tuning_config = True
                continue
            if in_tuning_config and line.startswith('|'):
                tuning_config_lines.append(line)
            elif in_tuning_config and not line.startswith('|') and line.strip():
                break

        # Filter data rows (skip header + separator)
        data_rows = [l for l in tuning_config_lines if '---' not in l and 'Setting' not in l]
        settings = [row.split('|')[1].strip() for row in data_rows if '|' in row]
        assert 'CV Strategy' not in settings
        assert 'CV Folds' not in settings
        assert 'Models Tuned' in settings


class TestModelDevelopmentSectionTests:
    """Tests for ModelDevelopmentSection (replaces CVSection tests)."""

    def test_model_development_with_cv_results(self, small_data):
        """Test ModelDevelopmentSection with trained model CV results."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        learner.train('coxph')

        section = ModelDevelopmentSection(learner)

        if section.is_available():
            result = section.generate()
            assert 'markdown' in result
            # Check cv_scores tidy DataFrame was created
            assert 'cv_scores' in section.data
            cv_df = section.data['cv_scores']
            assert 'model' in cv_df.columns
            assert 'fold' in cv_df.columns
            assert 'metric' in cv_df.columns
            assert 'score' in cv_df.columns


    def test_cv_config_stratification_default(self, small_data):
        """Default stratify='event' shows 'event' label + matching prose."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3, verbose=False)

        section = ModelDevelopmentSection(learner)
        md_lines = []
        section._add_cv_config_section(md_lines, {}, False)
        md = "\n".join(md_lines)

        # Pandas pads markdown cells with spaces; use regex on the row.
        import re
        assert re.search(r"\|\s*Stratification\s*\|\s*event\s*\|", md)
        assert "maintains event composition" in md
        # Strategy row should reference the same label
        assert "maintains event composition in each fold" in md

    def test_cv_config_stratification_custom_column(self, small_data):
        """Custom stratify column name appears verbatim in row + prose."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3, stratify="stage", verbose=False)

        section = ModelDevelopmentSection(learner)
        md_lines = []
        section._add_cv_config_section(md_lines, {}, False)
        md = "\n".join(md_lines)

        import re
        assert re.search(r"\|\s*Stratification\s*\|\s*stage\s*\|", md)
        assert "maintains stage composition" in md

    def test_cv_config_stratification_with_subpopulations(self, small_data):
        """Subpopulations + stratify_subpopulations=True shows composite label."""
        import numpy as np
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        X, y = small_data
        rng = np.random.RandomState(0)
        race = rng.choice(["A", "B"], size=len(X))

        learner = SurvivalLearner(X, y)
        learner.setup(
            random_state=42, cv_folds=3,
            subpopulations={"race": race},
            stratify_subpopulations=True,
            verbose=False,
        )

        section = ModelDevelopmentSection(learner)
        md_lines = []
        section._add_cv_config_section(md_lines, {}, False)
        md = "\n".join(md_lines)

        import re
        assert re.search(
            r"\|\s*Stratification\s*\|\s*event × subpopulation\(race\)\s*\|", md
        )
        assert "maintains event × subpopulation(race) composition" in md

    def test_cv_config_stratification_custom_array(self, small_data):
        """Unnamed array stratify renders without angle brackets in markdown."""
        import numpy as np
        import re
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        X, y = small_data
        rng = np.random.RandomState(0)
        custom = rng.randint(0, 2, size=len(X))

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3, stratify=custom, verbose=False)

        section = ModelDevelopmentSection(learner)
        md_lines = []
        section._add_cv_config_section(md_lines, {}, False)
        md = "\n".join(md_lines)

        assert "<" not in md.replace("</", "")  # no stray HTML angles
        assert re.search(r"\|\s*Stratification\s*\|\s*custom stratify array\s*\|", md)
        assert "maintains custom stratify array composition" in md

    def test_cv_config_stratification_subpops_disabled(self, small_data):
        """subpopulations present + stratify_subpopulations=False: label = spec only."""
        import numpy as np
        import re
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        X, y = small_data
        rng = np.random.RandomState(0)
        race = rng.choice(["A", "B"], size=len(X))

        learner = SurvivalLearner(X, y)
        learner.setup(
            random_state=42, cv_folds=3,
            subpopulations={"race": race},
            stratify_subpopulations=False,
            verbose=False,
        )

        section = ModelDevelopmentSection(learner)
        md_lines = []
        section._add_cv_config_section(md_lines, {}, False)
        md = "\n".join(md_lines)

        assert re.search(r"\|\s*Stratification\s*\|\s*event\s*\|", md)
        assert "subpopulation(" not in md  # no composite label
        assert "maintains event composition" in md

    def test_cv_config_stratification_repeated_cv(self, small_data):
        """cv_repeats > 1 still emits Stratification row + label-aware prose."""
        import re
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(
            random_state=42, cv_folds=3, cv_repeats=2, stratify="stage",
            verbose=False,
        )

        section = ModelDevelopmentSection(learner)
        md_lines = []
        section._add_cv_config_section(md_lines, {}, False)
        md = "\n".join(md_lines)

        # Row remains present
        assert re.search(r"\|\s*Stratification\s*\|\s*stage\s*\|", md)
        # Repeated-CV prose mentions the label
        assert "Repeated Stratified K-Fold" in md
        assert "maintains the same stage composition" in md

    def test_cv_config_no_stratification_row_for_kfold(self, small_data):
        """Non-stratified strategies omit the Stratification row."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3, cv_strategy="kfold", verbose=False)

        section = ModelDevelopmentSection(learner)
        md_lines = []
        section._add_cv_config_section(md_lines, {}, False)
        md = "\n".join(md_lines)

        assert "Stratification" not in md
        assert "Standard K-Fold" in md

    def test_compact_params_table_uses_raw_html(self):
        """Compact params table should use MLSURV_RAW_HTML markers."""
        from unittest.mock import MagicMock
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        # Build a mock result with an Optuna study
        mock_result = MagicMock()
        mock_result.optuna_study.best_params = {
            "model__n_estimators": 100,
            "model__max_depth": 5,
        }
        mock_result.optuna_study.best_trial.value = 0.72
        mock_result.scoring_metric = "c_index"

        section = ModelDevelopmentSection.__new__(ModelDevelopmentSection)
        output = section._build_compact_params_table({"xgbcox": mock_result})

        assert "<!-- MLSURV_RAW_HTML -->" in output
        assert "<!-- /MLSURV_RAW_HTML -->" in output
        assert '<table class="hp-params-table">' in output
        assert "<br>" in output
        assert "&lt;br&gt;" not in output

    def test_compact_params_table_escapes_special_chars(self):
        """HTML special characters in param values should be escaped."""
        from unittest.mock import MagicMock
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        mock_result = MagicMock()
        mock_result.optuna_study.best_params = {"model__threshold": "<0.5>"}
        mock_result.optuna_study.best_trial.value = 0.8
        mock_result.scoring_metric = "c_index"

        section = ModelDevelopmentSection.__new__(ModelDevelopmentSection)
        output = section._build_compact_params_table({"coxph": mock_result})

        # The <0.5> value should be escaped so it doesn't break HTML
        assert "&lt;0.5&gt;" in output
        # But the <br> separator should NOT be escaped
        assert "<table" in output

    def test_compact_params_table_br_renders_in_html(self, tmp_path):
        """<br> in compact params table should render as HTML, not escaped."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "test_br.html"
        renderer = HTMLRenderer(str(output_file), title="Test")

        md = (
            "<!-- MLSURV_RAW_HTML -->\n"
            '<table class="hp-params-table">'
            "<thead><tr><th>Model</th><th>Params</th></tr></thead>"
            "<tbody><tr><td>XGB</td><td>a=1<br>b=2</td></tr></tbody>"
            "</table>\n"
            "<!-- /MLSURV_RAW_HTML -->"
        )
        result = renderer._markdown_to_html(md)
        assert "<br>" in result
        assert "&lt;br&gt;" not in result


class TestFeatureAnalysisSectionTests:
    """Tests for FeatureAnalysisSection (replaces ImportanceSection tests)."""

    @staticmethod
    def _make_synthetic_bootstrap(features, model_name='coxph', n_iter=20, seed=42):
        """Build a minimal BootstrapResult with synthetic importance data."""
        import numpy as np
        import pandas as pd
        from mlsurv.results.containers import BootstrapResult

        rng = np.random.RandomState(seed)
        imp_rows = [
            {'model_type': model_name, 'iteration': it, 'feature': feat,
             'importance': rng.normal(0.05, 0.02)}
            for it in range(n_iter)
            for feat in features
        ]
        return BootstrapResult(
            summary=pd.DataFrame({
                'model_type': [model_name], 'metric': ['harrell'],
                'point': [0.7], 'mean': [0.7], 'std': [0.01],
                'ci_lower': [0.68], 'ci_upper': [0.72],
            }),
            iterations=pd.DataFrame({
                'model_type': [model_name] * n_iter,
                'iteration': list(range(n_iter)),
                'metric_name': ['harrell'] * n_iter,
                'value': rng.normal(0.7, 0.01, n_iter).tolist(),
            }),
            importance=pd.DataFrame(imp_rows),
        )

    def test_feature_analysis_generates_figure(self, small_data):
        """Test FeatureAnalysisSection generates importance figure."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=5, bootstrap=False)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)

        if section.is_available():
            result = section.generate()
            assert 'figures' in result
            # Check importance tidy DataFrame was created
            assert 'importance' in section.data

    def test_format_significance_thresholds(self):
        """Test _format_significance star notation at all threshold boundaries."""
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        assert FeatureAnalysisSection._format_significance(0.0) == "***"
        assert FeatureAnalysisSection._format_significance(0.0001) == "***"
        assert FeatureAnalysisSection._format_significance(0.0009) == "***"
        assert FeatureAnalysisSection._format_significance(0.001) == "**"  # exact boundary
        assert FeatureAnalysisSection._format_significance(0.005) == "**"
        assert FeatureAnalysisSection._format_significance(0.009) == "**"
        assert FeatureAnalysisSection._format_significance(0.01) == "*"  # exact boundary
        assert FeatureAnalysisSection._format_significance(0.03) == "*"
        assert FeatureAnalysisSection._format_significance(0.049) == "*"
        assert FeatureAnalysisSection._format_significance(0.05) == ""  # exact boundary
        assert FeatureAnalysisSection._format_significance(0.10) == ""
        assert FeatureAnalysisSection._format_significance(float('nan')) == ""
        assert FeatureAnalysisSection._format_significance(-0.01) == ""  # negative guard
        assert FeatureAnalysisSection._format_significance(float('inf')) == ""

    def test_sig_column_present_with_bootstrap_importance(self, small_data):
        """Test Sig. column appears when bootstrap importance data is available."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=5, bootstrap=False)

        features = eval_result.importance['feature'].unique()
        eval_result.bootstrap = self._make_synthetic_bootstrap(features)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        gen_result = section.generate()
        md = gen_result['markdown']

        assert 'Sig.' in md
        assert 'bootstrap importance significance' in md
        assert 'importance_pvalues' in section.data
        pvals_df = section.data['importance_pvalues']
        assert 'p_adjusted' in pvals_df.columns
        assert 'model_type' in pvals_df.columns

    def test_no_sig_column_without_bootstrap(self, small_data):
        """Test Sig. column absent when bootstrap is not available."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=5, bootstrap=False)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        assert section.is_available(), "Section should be available with importance data"

        gen_result = section.generate()
        md = gen_result['markdown']
        assert 'Sig.' not in md
        assert 'importance_pvalues' not in section.data

    def test_narrative_present_with_bootstrap(self, small_data):
        """Test importance significance narrative appears with bootstrap data."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=5, bootstrap=False)

        features = eval_result.importance['feature'].unique()
        eval_result.bootstrap = self._make_synthetic_bootstrap(features)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        gen_result = section.generate()
        md = gen_result['markdown']

        assert 'Importance Significance (Bootstrap)' in md
        assert 'features show statistically significant importance' in md or \
               'No features reached statistical significance' in md

    def test_format_adjusted_pvalue(self):
        """Test _format_adjusted_pvalue formatting at various magnitudes."""
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        # Unified via mlsurv.reports._constants.format_pvalue:
        # - p >= 0.01 → 3 decimals
        # - p < 0.01  → scientific with 2-decimal mantissa
        # - NaN / non-finite / out-of-range → empty string (unchanged
        #   convention for feature-analysis tables; see docstring on
        #   _format_adjusted_pvalue)
        assert FeatureAnalysisSection._format_adjusted_pvalue(0.0001) == "1.00e-04"
        assert FeatureAnalysisSection._format_adjusted_pvalue(0.005) == "5.00e-03"
        assert FeatureAnalysisSection._format_adjusted_pvalue(0.05) == "0.050"
        assert FeatureAnalysisSection._format_adjusted_pvalue(0.234) == "0.234"
        assert FeatureAnalysisSection._format_adjusted_pvalue(float('nan')) == ""
        assert FeatureAnalysisSection._format_adjusted_pvalue(float('inf')) == ""
        assert FeatureAnalysisSection._format_adjusted_pvalue(-0.01) == ""

    def test_short_column_name(self):
        """Test _short_column_name strips parenthetical model suffix."""
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        assert FeatureAnalysisSection._short_column_name("PFI (CoxPH)") == "PFI"
        assert FeatureAnalysisSection._short_column_name("Sig. (RSF)") == "Sig."
        assert FeatureAnalysisSection._short_column_name("Feature") == "Feature"
        assert FeatureAnalysisSection._short_column_name("p (adj.) (CoxPH)") == "p (adj.)"

    def test_interactive_table_html_structure(self):
        """Test _format_interactive_table generates proper HTML structure."""
        import pandas as pd

        df = pd.DataFrame({
            'Feature': ['age', 'size'],
            'p-value': ['0.01', '0.05'],
            'HR': ['1.5', '0.8'],
            'Effect': ['Risk (+)', 'Protective (-)'],
            'PFI (CoxPH)': ['0.05', '0.03'],
        })
        groups = [
            ("feature", "Feature", ["Feature"], True),
            ("univariate-cox", "Univariate Cox", ["p-value", "HR", "Effect"]),
            ("coxph", "CoxPH", ["PFI (CoxPH)"]),
        ]
        html = format_interactive_table(df, groups)

        assert '<!-- MLSURV_RAW_HTML -->' in html
        assert '<!-- /MLSURV_RAW_HTML -->' in html
        assert 'feature-table-container' in html
        assert 'feature-table-controls' in html
        assert 'data-ft-model' in html
        assert 'data-ft-type' in html
        assert 'data-ft-group-id' in html
        assert 'rowspan="2"' in html
        assert 'sticky-col' in html
        assert 'data-grouped-header="1"' in html
        assert 'data-col-idx' in html

    def test_column_groups_single_model(self):
        """Test _build_single_model_column_groups returns correct structure."""
        import pandas as pd

        df = pd.DataFrame({
            'Feature': ['a', 'b'],
            'p-value': ['0.01', '0.05'],
            'HR': ['1.5', '0.8'],
            'Effect': ['Risk (+)', 'Protective (-)'],
            'PFI (CoxPH)': ['0.05', '0.03'],
            'p (adj.)': ['0.02', '0.10'],
            'Sig.': ['*', ''],
            'Coef. (CoxPH)': ['0.3', '-0.2'],
            'HR (mv) (CoxPH)': ['1.35', '0.82'],
        })
        groups = build_single_model_column_groups(df, 'coxph', coef_models=['coxph'])

        group_ids = [g[0] for g in groups]
        assert "feature" in group_ids
        assert "univariate-cox" in group_ids
        assert "coxph" in group_ids

        # Feature group should be sticky (4th element = True)
        feature_group = [g for g in groups if g[0] == "feature"][0]
        assert len(feature_group) >= 4
        assert feature_group[3] is True

        # Single merged model group should contain PFI + Sig + Coef + HR(mv)
        coxph_group = [g for g in groups if g[0] == "coxph"][0]
        assert "PFI (CoxPH)" in coxph_group[2]
        assert "Coef. (CoxPH)" in coxph_group[2]
        assert "HR (mv) (CoxPH)" in coxph_group[2]

    def test_raw_html_block_renderer_passthrough(self):
        """Test raw HTML blocks pass through mistune unescaped."""
        from mlsurv.reports.renderers.html import HTMLRenderer

        renderer = HTMLRenderer.__new__(HTMLRenderer)
        renderer._markdown_parser = mistune.create_markdown(
            renderer=mistune.HTMLRenderer(),
            plugins=[],
        )

        raw_block = (
            '<!-- MLSURV_RAW_HTML -->\n'
            '<div class="feature-table-container">\n'
            '<table><tr><td>hello</td></tr></table>\n'
            '</div>\n'
            '<!-- /MLSURV_RAW_HTML -->'
        )
        result = renderer._markdown_to_html(raw_block)
        assert '<div class="feature-table-container">' in result
        assert '<table><tr><td>hello</td></tr></table>' in result

    def test_padj_column_present_with_bootstrap(self, small_data):
        """Test p (adj.) column appears in output when bootstrap data is available."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=5, bootstrap=False)

        features = eval_result.importance['feature'].unique()
        eval_result.bootstrap = self._make_synthetic_bootstrap(features)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        gen_result = section.generate()
        md = gen_result['markdown']

        assert 'p (adj.)' in md

    @pytest.mark.parametrize("models", [['coxph'], ['coxph', 'coxnet']])
    def test_feature_summary_failure_does_not_kill_section(self, small_data, monkeypatch, models):
        """feature_summary() raising must degrade to a note, not crash.

        Covers both the single-model and multi-model table paths.
        """
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.run(models, tune=False, verbose=False)
        eval_result = learner.evaluate(n_importance_repeats=2, bootstrap=False)

        def _boom(*args, **kwargs):
            raise ValueError("synthetic feature_summary failure")

        # The section assembles the summary via compute_feature_summary
        # (with its own eval_result frames), so inject the failure there.
        monkeypatch.setattr(
            'mlsurv.evaluation._feature_summary.compute_feature_summary',
            _boom,
        )

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        gen_result = section.generate()
        md = gen_result['markdown']

        # The combined table is replaced by a note...
        assert 'Combined feature table unavailable' in md
        assert 'synthetic feature_summary failure' in md
        # ...but the rest of the section still renders.
        assert 'Univariate Cox and Permutation Importance' in md
        assert 'Feature Correlation Analysis' in md

    def test_summary_missing_feature_degrades_not_crashes(self, small_data, monkeypatch):
        """Features absent from the summary block are skipped, not fatal.

        The summary assembly and the section's eval_result can diverge
        in edge cases; the table must skip the divergent feature, not
        KeyError away the whole section.
        """
        from mlsurv import SurvivalLearner
        from mlsurv.evaluation._feature_summary import compute_feature_summary
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=2, bootstrap=False)

        dropped = eval_result.importance['feature'].iloc[0]

        def _dropping(*args, **kwargs):
            summary = compute_feature_summary(*args, **kwargs)
            return summary[summary['feature'] != dropped]

        monkeypatch.setattr(
            'mlsurv.evaluation._feature_summary.compute_feature_summary',
            _dropping,
        )

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        md = section.generate()['markdown']

        # Section and table still render; only the divergent row is gone.
        assert 'Univariate Cox and Permutation Importance' in md
        assert 'feature-analysis-table' in md
        remaining = [f for f in eval_result.importance['feature'] if f != dropped]
        assert any(f in md for f in remaining)

    def test_stale_eval_result_renders_consistent_values(self, small_data):
        """A caller-supplied eval result feeds the table, charts, and exports alike.

        Regression for the later mixed-sources issue: the combined
        table used to render live learner state while charts/exports
        used the passed eval result.
        """
        import copy
        import re
        import numpy as np
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection
        from mlsurv.reports.sections.feature_analysis._helpers import (
            _format_importance_value,
        )

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('coxph')

        # Evaluation A (kept), then evaluation B with a different repeat
        # count so live perm_mean values diverge from A.
        learner.evaluate(n_importance_repeats=2, bootstrap=False)
        old_eval = copy.deepcopy(learner.evaluation_result)
        learner.evaluate(n_importance_repeats=7, bootstrap=False)

        old_imp = old_eval.importance.set_index('feature')['perm_mean']
        new_imp = learner.evaluation_result.importance.set_index('feature')['perm_mean']
        diff_feats = [f for f in old_imp.index
                      if not np.isclose(old_imp[f], new_imp[f])]
        assert diff_feats, "need diverging importance values for the check"
        feat = diff_feats[0]

        section = FeatureAnalysisSection(learner, eval_result=old_eval)
        md = section.generate()['markdown']

        c_index = dict(zip(old_eval.concordance['model_type'],
                           old_eval.concordance['c_index_test']))['coxph']
        old_str = _format_importance_value(old_imp[feat], c_index)
        new_str = _format_importance_value(new_imp[feat], c_index)

        row_match = re.search(
            rf'<td class="sticky-col">{feat}[^<]*</td>\s*'
            rf'(?:<td[^>]*>[^<]*</td>\s*){{3}}'      # p-value, HR, Effect
            rf'<td[^>]*data-ft-type="pfi"[^>]*>([^<]*)</td>',
            md,
        )
        assert row_match, "PFI cell not found in rendered table"
        # The table must show the PASSED eval result's value, matching
        # the charts and the importance data export.
        assert row_match.group(1) == old_str
        if old_str != new_str:
            assert row_match.group(1) != new_str

    def test_correlation_computed_once_per_report(self, small_data, monkeypatch):
        """generate() runs detect_correlated_features exactly once.

        Regression for the later duplicate: the correlation
        subsection and the combined table each used to compute it.
        """
        import mlsurv.evaluation.importance as imp_mod
        import mlsurv.reports.sections.feature_analysis._correlation as corr_mod
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=2, bootstrap=False)

        calls = {'n': 0}
        real = imp_mod.detect_correlated_features

        def counting(*args, **kwargs):
            calls['n'] += 1
            return real(*args, **kwargs)

        # _correlation.py binds the name at import time; patch both it
        # and the source module (covers any lazy importers).
        monkeypatch.setattr(imp_mod, 'detect_correlated_features', counting)
        monkeypatch.setattr(corr_mod, 'detect_correlated_features', counting)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        section.generate()
        assert calls['n'] == 1

    def test_correlation_failure_renders_na_not_false(self, small_data, monkeypatch):
        """When correlation fails, the summary's correlated column is NA."""
        import mlsurv.reports.sections.feature_analysis._correlation as corr_mod
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=2, bootstrap=False)

        def _boom(*args, **kwargs):
            raise ValueError("corr fail")

        monkeypatch.setattr(corr_mod, 'detect_correlated_features', _boom)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        captured = {}
        orig = FeatureAnalysisSection._get_feature_summary

        def capturing(self, correlated_features):
            captured['set'] = correlated_features
            captured['summary'] = orig(self, correlated_features)
            return captured['summary']

        monkeypatch.setattr(
            FeatureAnalysisSection, '_get_feature_summary', capturing,
        )
        md = section.generate()['markdown']

        # Failure is "unknown", not "no correlated pairs".
        assert captured['set'] is None
        assert captured['summary']['correlated'].isna().all()
        # Section still renders (table + unavailable correlation note).
        assert 'feature-analysis-table' in md

    @pytest.mark.integration
    def test_categorical_group_rows_render(self):
        """Group parent row + child rows render through the summary path."""
        import numpy as np
        import pandas as pd
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        rng = np.random.default_rng(0)
        n = 60
        cat = rng.integers(0, 3, size=n)
        X = pd.DataFrame({
            'cont1': rng.normal(size=n),
            'grp_a': (cat == 0).astype(float),
            'grp_b': (cat == 1).astype(float),
            'grp_c': (cat == 2).astype(float),
        })
        risk = 0.8 * X['cont1'] + 0.5 * X['grp_a']
        time = rng.exponential(scale=np.exp(-risk))
        event = rng.random(n) < 0.7
        y = np.array([(bool(e), float(t)) for e, t in zip(event, time)],
                     dtype=[('event', bool), ('time', float)])

        learner = SurvivalLearner(X, y)
        learner.setup(
            feature_selector=None, random_state=42, cv_folds=3,
            categorical_groups={'grp': ['grp_a', 'grp_b', 'grp_c']},
        )
        learner.train('coxph')
        eval_result = learner.evaluate(n_importance_repeats=2, bootstrap=False)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        md = section.generate()['markdown']

        # Grouped-features note, parent row, and indented child rows.
        assert 'Grouped features' in md
        assert '>grp<' in md  # parent row cell for the group feature
        for col in ('grp_a', 'grp_b', 'grp_c'):
            assert f'↳ {col}' in md  # child rows survive the refactor

    def test_column_groups_multi_model(self):
        """Test _build_multi_model_column_groups returns per-model groups."""
        import pandas as pd

        df = pd.DataFrame({
            'Feature': ['a', 'b'],
            'p-value': ['0.01', '0.05'],
            'HR': ['1.5', '0.8'],
            'Effect': ['Risk (+)', 'Protective (-)'],
            'PFI (CoxPH)': ['0.05', '0.03'],
            'p (adj.) (CoxPH)': ['0.02', '0.10'],
            'Sig. (CoxPH)': ['*', ''],
            'PFI (RSF)': ['0.04', '0.02'],
            'p (adj.) (RSF)': ['0.03', '0.15'],
            'Sig. (RSF)': ['*', ''],
            'Coef. (CoxPH)': ['0.3', '-0.2'],
            'HR (mv) (CoxPH)': ['1.35', '0.82'],
        })
        groups = build_multi_model_column_groups(df, ['coxph', 'rsf'], coef_models=['coxph'])

        group_ids = [g[0] for g in groups]
        assert "feature" in group_ids
        assert "univariate-cox" in group_ids
        # Each model gets a single merged group (PFI + Sig + Coef under one header)
        coxph_groups = [g for g in groups if g[0] == 'coxph']
        rsf_groups = [g for g in groups if g[0] == 'rsf']
        assert len(coxph_groups) == 1
        assert len(rsf_groups) == 1
        # CoxPH: PFI + p(adj.) + Sig. + Coef. + HR(mv) = 5 columns
        assert len(coxph_groups[0][2]) == 5
        assert 'Coef. (CoxPH)' in coxph_groups[0][2]
        # RSF: PFI + p(adj.) + Sig. = 3 columns (no coefficients)
        assert len(rsf_groups[0][2]) == 3

    def test_interactive_table_col_idx_correctness(self):
        """Test data-col-idx values are correct for sorting."""
        import re
        import pandas as pd

        df = pd.DataFrame({
            'Feature': ['age', 'size'],
            'p-value': ['0.01', '0.05'],
            'HR': ['1.5', '0.8'],
            'PFI (CoxPH)': ['0.05', '0.03'],
        })
        groups = [
            ("feature", "Feature", ["Feature"], True),
            ("univariate-cox", "Univariate Cox", ["p-value", "HR"]),
            ("coxph", "CoxPH", ["PFI (CoxPH)"]),
        ]
        html = format_interactive_table(df, groups)

        # Extract all data-col-idx values from column-header-row
        col_indices = re.findall(r'data-col-idx="(\d+)"', html)
        # Feature (sticky, idx 0) is skipped in column-header-row
        # p-value=1, HR=2, PFI=3
        assert col_indices == ['1', '2', '3']

        # Verify tbody has correct number of cells per row (4: Feature + 3 data cols)
        row_cells = re.findall(r'<tr data-original-index="0">(.*?)</tr>', html, re.DOTALL)
        assert row_cells
        td_count = row_cells[0].count('<td')
        assert td_count == 4


    def test_expandable_rows_hidden_when_exceeds_n_features(self):
        """Test ft-overflow-row class on rows beyond n_visible."""
        import pandas as pd

        df = pd.DataFrame({
            'Feature': [f'feat_{i}' for i in range(15)],
            'PFI (CoxPH)': [f'{0.1 - i * 0.005:.3f}' for i in range(15)],
        })
        groups = [
            ("feature", "Feature", ["Feature"], True),
            ("coxph", "CoxPH", ["PFI (CoxPH)"]),
        ]
        html = format_interactive_table(df, groups, n_visible=10)

        assert html.count('ft-overflow-row') == 5
        assert 'ft-expand-btn' in html
        assert 'Show all 15 features' in html

    def test_expandable_rows_absent_when_few_features(self):
        """Test no expand button when n_total <= n_visible."""
        import pandas as pd

        df = pd.DataFrame({
            'Feature': ['a', 'b', 'c'],
            'PFI (CoxPH)': ['0.05', '0.03', '0.01'],
        })
        groups = [
            ("feature", "Feature", ["Feature"], True),
            ("coxph", "CoxPH", ["PFI (CoxPH)"]),
        ]
        html = format_interactive_table(df, groups, n_visible=10)

        assert 'ft-overflow-row' not in html
        assert 'ft-expand-btn' not in html

    # ── direction from univariate Cox HR ────────────────────────────────

    def test_build_importance_tidy_direction_from_univar_hr(self):
        """Direction labels in tidy importance come from univariate Cox HR."""
        import pandas as pd
        from unittest.mock import MagicMock
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        section = MagicMock(spec=FeatureAnalysisSection)
        section._build_importance_tidy = FeatureAnalysisSection._build_importance_tidy.__get__(section)

        importance_df = pd.DataFrame({
            'model_type': ['rsf', 'rsf'],
            'feature': ['age', 'size'],
            'perm_mean': [0.05, 0.03],
            'perm_std': [0.01, 0.005],
        })
        univar_cox = pd.DataFrame({
            'feature': ['age', 'size'],
            'p_value': [0.001, 0.05],
            'hazard_ratio': [1.5, 0.7],  # age=Risk, size=Protective
        })

        tidy = section._build_importance_tidy(importance_df, univar_cox)

        assert tidy.loc[tidy['feature'] == 'age', 'direction'].values[0] == 'Risk (+)'
        assert tidy.loc[tidy['feature'] == 'size', 'direction'].values[0] == 'Protective (-)'

    def test_build_importance_tidy_no_univar_cox(self):
        """When univar_cox is None, direction defaults to 'Unknown'."""
        import pandas as pd
        from unittest.mock import MagicMock
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        section = MagicMock(spec=FeatureAnalysisSection)
        section._build_importance_tidy = FeatureAnalysisSection._build_importance_tidy.__get__(section)

        importance_df = pd.DataFrame({
            'model_type': ['rsf'],
            'feature': ['age'],
            'perm_mean': [0.05],
            'perm_std': [0.01],
        })

        tidy = section._build_importance_tidy(importance_df, None)

        assert 'direction' in tidy.columns
        assert (tidy['direction'] == 'Unknown').all()

    def test_per_model_coefficients_in_multi_model_table(self):
        """Test Coef/HR columns appear for Cox models only in multi-model table."""
        import pandas as pd

        df = pd.DataFrame({
            'Feature': ['a', 'b'],
            'PFI (CoxPH)': ['0.05', '0.03'],
            'PFI (RSF)': ['0.04', '0.02'],
            'Coef. (CoxPH)': ['0.300', '-0.200'],
            'HR (mv) (CoxPH)': ['1.350', '0.819'],
        })
        groups = [
            ("feature", "Feature", ["Feature"], True),
            ("coxph", "CoxPH", ["PFI (CoxPH)", "Coef. (CoxPH)", "HR (mv) (CoxPH)"]),
            ("rsf", "RSF", ["PFI (RSF)"]),
        ]
        html = format_interactive_table(df, groups)

        assert 'Coef.' in html
        assert 'HR (mv)' in html
        # CoxPH group header spans PFI + Coef + HR columns
        assert 'CoxPH' in html
        # RSF should not have coefficient columns
        assert 'Coef. (RSF)' not in html
        assert 'HR (mv) (RSF)' not in html

    def test_sorted_top_features_returns_all(self):
        """Test _get_sorted_top_features returns ALL features, not just top N."""
        import pandas as pd

        importance_df = pd.DataFrame({
            'model_type': ['coxph'] * 5 + ['rsf'] * 5,
            'feature': ['a', 'b', 'c', 'd', 'e'] * 2,
            'perm_mean': [0.10, 0.08, 0.06, 0.04, 0.02,
                          0.09, 0.07, 0.05, 0.03, 0.01],
        })
        result = get_sorted_top_features(['coxph', 'rsf'], importance_df)
        assert len(result) == 5  # all features, not just top 3
        assert result[0] == 'a'  # highest avg: (0.10+0.09)/2=0.095
        assert result[-1] == 'e'  # lowest avg

    def test_per_model_coef_blocks_from_summary(self):
        """per_model_coef_blocks keeps only models with fitted coefficients."""
        import numpy as np
        import pandas as pd
        from mlsurv.reports.sections.feature_analysis._importance import (
            per_model_coef_blocks,
        )

        summary = pd.DataFrame({
            'model_type': ['coxph', 'coxph', 'rsf', 'rsf'],
            'feature': ['age', 'size', 'age', 'size'],
            'mv_coefficient': [0.5, -0.3, np.nan, np.nan],
            'mv_hazard_ratio': [np.exp(0.5), np.exp(-0.3), np.nan, np.nan],
            'mv_hr_lower_95': [np.nan] * 4,
            'mv_hr_upper_95': [np.nan] * 4,
            'mv_p_value': [0.02, 0.4, np.nan, np.nan],
        })
        blocks = per_model_coef_blocks(summary)
        assert 'coxph' in blocks
        assert 'rsf' not in blocks  # all-NaN coefficients
        coef_df = blocks['coxph']
        age_hr = coef_df.loc[coef_df['feature'] == 'age', 'mv_hazard_ratio'].values[0]
        assert np.isclose(age_hr, np.exp(0.5))
        assert per_model_coef_blocks(None) == {}

    def test_get_all_coefficient_models_skips_non_linear(self):
        """Test _get_all_coefficient_models excludes models without coefficients."""
        from unittest.mock import MagicMock
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection
        import pandas as pd

        mock_cox = MagicMock()
        mock_cox.get_coefficients.return_value = pd.Series([0.5], index=['age'])
        mock_rsf = MagicMock(spec=[])  # no get_coefficients

        mock_learner = MagicMock()
        mock_learner.models = {
            'coxph': MagicMock(model=mock_cox),
            'rsf': MagicMock(model=mock_rsf),
        }
        section = FeatureAnalysisSection(mock_learner, eval_result=None)
        result = get_all_coefficient_models(section)

        assert 'coxph' in result
        assert 'rsf' not in result

    def test_get_all_coefficient_models_handles_exception(self):
        """Test _get_all_coefficient_models logs and skips on extraction error."""
        from unittest.mock import MagicMock, patch
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        mock_model = MagicMock()
        mock_model.get_coefficients.side_effect = ValueError("extraction error")

        mock_learner = MagicMock()
        mock_learner.models = {'coxph': MagicMock(model=mock_model)}

        section = FeatureAnalysisSection(mock_learner, eval_result=None)
        with patch.object(section, '_log_recoverable_exception') as mock_log:
            result = get_all_coefficient_models(section)

        assert result == {}  # gracefully empty, not a crash
        mock_log.assert_called_once()
        # Verify warn_user=True is passed
        assert mock_log.call_args[1].get('warn_user') is True

    def test_expandable_rows_boundary_exact_n_visible(self):
        """Test no expand button when n_total equals exactly n_visible."""
        import pandas as pd

        df = pd.DataFrame({
            'Feature': [f'f{i}' for i in range(10)],
            'PFI (CoxPH)': ['0.05'] * 10,
        })
        groups = [
            ("feature", "Feature", ["Feature"], True),
            ("coxph", "CoxPH", ["PFI (CoxPH)"]),
        ]
        html = format_interactive_table(df, groups, n_visible=10)

        assert 'ft-overflow-row' not in html
        assert 'ft-expand-btn' not in html

    def test_column_group_labels_no_importance_suffix(self):
        """Test group labels use model name without 'Importance' suffix."""
        import pandas as pd

        df = pd.DataFrame({
            'Feature': ['a'],
            'PFI (CoxPH)': ['0.05'],
        })
        groups = build_single_model_column_groups(df, 'coxph')
        pfi_group = [g for g in groups if g[0] == 'coxph'][0]
        assert pfi_group[1] == 'CoxPH'
        assert 'Importance' not in pfi_group[1]

    def test_classify_column_type_mappings(self):
        """Test _classify_column_type returns correct type slugs."""
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        # Exact matches
        assert FeatureAnalysisSection._classify_column_type("p-value") == "univariate-cox"
        assert FeatureAnalysisSection._classify_column_type("HR") == "univariate-cox"
        assert FeatureAnalysisSection._classify_column_type("Effect") == "univariate-cox"
        assert FeatureAnalysisSection._classify_column_type("Mean |SHAP|") == "shap"

        # Prefix matches
        assert FeatureAnalysisSection._classify_column_type("PFI (CoxPH)") == "pfi"
        assert FeatureAnalysisSection._classify_column_type("PFI (RSF)") == "pfi"
        assert FeatureAnalysisSection._classify_column_type("p (adj.) (CoxPH)") == "significance"
        assert FeatureAnalysisSection._classify_column_type("Sig. (RSF)") == "significance"
        assert FeatureAnalysisSection._classify_column_type("Coef. (CoxPH)") == "coefficients"
        assert FeatureAnalysisSection._classify_column_type("HR (mv) (CoxNet)") == "coefficients"

        # No match
        assert FeatureAnalysisSection._classify_column_type("Feature") is None
        assert FeatureAnalysisSection._classify_column_type("Unknown Column") is None

    def test_data_ft_attributes_on_body_cells(self):
        """Test data-ft-model and data-ft-type attributes are correct on td cells."""
        import pandas as pd

        df = pd.DataFrame({
            'Feature': ['age'],
            'p-value': ['0.01'],
            'PFI (CoxPH)': ['0.05'],
            'Coef. (CoxPH)': ['0.3'],
        })
        groups = [
            ("feature", "Feature", ["Feature"], True),
            ("univariate-cox", "Univariate Cox", ["p-value"]),
            ("coxph", "CoxPH", ["PFI (CoxPH)", "Coef. (CoxPH)"]),
        ]
        html = format_interactive_table(df, groups)

        # Feature column: no data-ft-model or data-ft-type (sticky)
        assert 'class="sticky-col"' in html

        # Univariate Cox: data-ft-type but no data-ft-model
        assert 'data-ft-type="univariate-cox"' in html

        # PFI (CoxPH): both model and type
        assert 'data-ft-model="coxph" data-ft-type="pfi"' in html

        # Coef (CoxPH): both model and type
        assert 'data-ft-model="coxph" data-ft-type="coefficients"' in html

    def test_format_interactive_table_empty_df(self):
        """Test _format_interactive_table returns fallback for None/empty df."""
        import pandas as pd

        result_none = format_interactive_table(None, [])
        assert "No features met the display threshold" in result_none

        result_empty = format_interactive_table(pd.DataFrame(), [])
        assert "No features met the display threshold" in result_empty

    def test_append_coef_columns_feature_not_found(self):
        """Test _append_coef_columns returns dashes when feature is absent."""
        import numpy as np
        import pandas as pd
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        coef_block = pd.DataFrame({
            'feature': ['age'], 'mv_coefficient': [0.5],
            'mv_hazard_ratio': [1.65], 'mv_hr_lower_95': [np.nan],
            'mv_hr_upper_95': [np.nan], 'mv_p_value': [np.nan],
        })
        row_data = {}
        FeatureAnalysisSection._append_coef_columns(row_data, 'missing_feat', 'CoxPH', coef_block)

        assert row_data['Coef. (CoxPH)'] == '-'
        assert row_data['HR (mv) (CoxPH)'] == '-'

    def test_append_coef_columns_nan_renders_dash(self):
        """NaN HR/coef render as '-'; CI inline; p column only with inference."""
        import numpy as np
        import pandas as pd
        from mlsurv.reports.sections.feature_analysis._importance import (
            append_coef_columns,
        )

        # CoxPH-style: full inference, finite values -> HR with inline CI + p.
        coxph_df = pd.DataFrame({
            'feature': ['age'], 'mv_coefficient': [0.5],
            'mv_hazard_ratio': [1.649], 'mv_hr_lower_95': [1.20],
            'mv_hr_upper_95': [2.27], 'mv_p_value': [0.002],
        })
        row = {}
        append_coef_columns(row, 'age', 'CoxPH', coxph_df)
        assert row['Coef. (CoxPH)'] == '0.500'
        assert row['HR (mv) (CoxPH)'] == '1.649 (1.200-2.270)'
        assert 'p (mv) (CoxPH)' in row

        # CoxNet-style: gated NaN HR/CI/p -> HR dash, no p column.
        coxnet_df = pd.DataFrame({
            'feature': ['age'], 'mv_coefficient': [0.5],
            'mv_hazard_ratio': [np.nan], 'mv_hr_lower_95': [np.nan],
            'mv_hr_upper_95': [np.nan], 'mv_p_value': [np.nan],
        })
        row = {}
        append_coef_columns(row, 'age', 'CoxNet', coxnet_df)
        assert row['Coef. (CoxNet)'] == '0.500'
        assert row['HR (mv) (CoxNet)'] == '-'
        assert 'p (mv) (CoxNet)' not in row

        # Singular-Hessian CoxPH: HR finite but CI NaN -> bare HR, p dash.
        singular_df = pd.DataFrame({
            'feature': ['age'], 'mv_coefficient': [0.5],
            'mv_hazard_ratio': [1.649], 'mv_hr_lower_95': [np.nan],
            'mv_hr_upper_95': [np.nan], 'mv_p_value': [np.nan],
        })
        row = {}
        append_coef_columns(row, 'age', 'CoxPH', singular_df)
        assert row['HR (mv) (CoxPH)'] == '1.649'  # no parenthetical CI
        # p_value is all-NaN here, so no p column is emitted.
        assert 'p (mv) (CoxPH)' not in row

    def test_feature_summary_gating(self, small_data):
        """CoxNet HR column blanks (NaN); CoxPH gets finite HR + CI/p."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis._importance import (
            per_model_coef_blocks,
        )

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector=None, random_state=42, cv_folds=3)
        learner.run(['coxph', 'coxnet'], tune=False, verbose=False)

        per_model = per_model_coef_blocks(learner.feature_summary())

        assert per_model['coxph']['mv_hazard_ratio'].notna().all()
        assert per_model['coxph']['mv_p_value'].notna().all()
        assert per_model['coxnet']['mv_hazard_ratio'].isna().all()
        assert per_model['coxnet']['mv_p_value'].isna().all()

    def test_column_groups_include_p_mv(self):
        """Interactive table includes the p (mv) column when present."""
        import pandas as pd
        from mlsurv.reports.sections.feature_analysis._interactive_table import (
            _classify_column_type,
            build_multi_model_column_groups,
        )

        # CoxPH carries p (mv); CoxNet does not (gated).
        table_df = pd.DataFrame(columns=[
            'Feature', 'PFI (CoxPH)', 'Coef. (CoxPH)', 'HR (mv) (CoxPH)',
            'p (mv) (CoxPH)', 'PFI (CoxNet)', 'Coef. (CoxNet)',
            'HR (mv) (CoxNet)',
        ])
        groups = build_multi_model_column_groups(
            table_df, ['coxph', 'coxnet'], coef_models=['coxph', 'coxnet'],
        )
        all_cols = [c for _, _, cols, *_ in groups for c in cols]
        assert 'p (mv) (CoxPH)' in all_cols
        assert 'p (mv) (CoxNet)' not in all_cols  # gated model has no p column
        # p (mv) classifies as a coefficients-type column.
        assert _classify_column_type('p (mv) (CoxPH)') == 'coefficients'


class TestSummarySectionTests:
    """Tests for SummarySection."""

    def test_summary_section_basic(self, small_data):
        """Test SummarySection generates executive summary."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.summary import SummarySection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        # Train a model
        learner.train('coxph')

        section = SummarySection(learner)

        if section.is_available():
            result = section.generate()
            assert 'markdown' in result

    def test_summary_cv_raincloud_figure(self, small_data):
        """Summary includes CV raincloud figure when models are trained."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.summary import SummarySection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('coxph')

        section = SummarySection(learner)
        result = section.generate()

        filenames = [f[0] for f in result['figures']]
        assert 'cv_raincloud.png' in filenames

        figure_info = result.get('figure_info', [])
        ids = [fi.figure_id for fi in figure_info]
        assert 'cv_raincloud' in ids

    def test_summary_bootstrap_raincloud_figure(self, small_data):
        """Summary includes bootstrap raincloud when bootstrap results exist."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.summary import SummarySection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('coxph')
        learner.evaluate()
        learner.bootstrap(n_bootstrap=3)

        boot_results = {
            name: result.bootstrap
            for name, result in learner.models.items()
            if result.bootstrap is not None
        }

        section = SummarySection(
            learner,
            bootstrap_results=boot_results,
            eval_result=learner.evaluation_result,
        )
        result = section.generate()

        filenames = [f[0] for f in result['figures']]
        assert 'bootstrap_raincloud.png' in filenames

        ids = [fi.figure_id for fi in result.get('figure_info', [])]
        assert 'bootstrap_raincloud' in ids

    def test_summary_no_bootstrap_raincloud_without_bootstrap(self, small_data):
        """Summary omits bootstrap raincloud when no bootstrap results."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.summary import SummarySection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('coxph')

        section = SummarySection(learner)
        result = section.generate()

        filenames = [f[0] for f in result['figures']]
        assert 'bootstrap_raincloud.png' not in filenames

    @pytest.mark.integration
    def test_summary_importance_heatmap_multi_model(self, small_data):
        """Summary shows importance heatmap when >=2 models are trained."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.summary import SummarySection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.run(['coxph', 'rsf'], tune=False,
                    model_params={'rsf': _SMALL_TREE_PARAMS})
        learner.evaluate(n_importance_repeats=1)

        section = SummarySection(
            learner,
            eval_result=learner.evaluation_result,
        )
        result = section.generate()

        filenames = [f[0] for f in result['figures']]
        assert 'importance_heatmap.png' in filenames

        ids = [fi.figure_id for fi in result.get('figure_info', [])]
        assert 'importance_heatmap' in ids

    def test_summary_importance_bar_single_model(self, small_data):
        """Summary shows importance bar chart for single model."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.summary import SummarySection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('coxph')
        learner.evaluate(n_importance_repeats=1)

        section = SummarySection(
            learner,
            eval_result=learner.evaluation_result,
        )
        result = section.generate()

        filenames = [f[0] for f in result['figures']]
        assert 'importance.png' in filenames

        ids = [fi.figure_id for fi in result.get('figure_info', [])]
        assert 'importance' in ids

    def test_summary_no_importance_without_eval(self, small_data):
        """Summary omits importance figure when no evaluation data."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.summary import SummarySection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('coxph')

        section = SummarySection(learner)
        result = section.generate()

        filenames = [f[0] for f in result['figures']]
        assert 'importance_heatmap.png' not in filenames
        assert 'importance.png' not in filenames

    def test_summary_cv_raincloud_graceful_on_error(self, small_data):
        """CV raincloud degrades gracefully when plot_metric raises."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.summary import SummarySection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('coxph')

        original = learner.plot_metric

        def raise_on_cv(*args, **kwargs):
            if kwargs.get('source') == 'cv':
                raise RuntimeError("test error")
            return original(*args, **kwargs)

        learner.plot_metric = raise_on_cv

        section = SummarySection(learner)
        result = section.generate()

        filenames = [f[0] for f in result['figures']]
        assert 'cv_raincloud.png' not in filenames
        assert '*[CV performance plot unavailable:' in result['markdown']

    def test_summary_importance_graceful_on_error(self, small_data):
        """Importance figure degrades gracefully when plot raises."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.summary import SummarySection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('coxph')
        learner.evaluate(n_importance_repeats=1)

        learner.plot_importance = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("test error")
        )

        section = SummarySection(
            learner,
            eval_result=learner.evaluation_result,
        )
        result = section.generate()

        filenames = [f[0] for f in result['figures']]
        assert 'importance.png' not in filenames
        assert '*[Importance plot unavailable:' in result['markdown']


@pytest.mark.integration
class TestReportGeneratorIntegration:
    """Integration tests for ReportGenerator."""

    def test_report_generator_with_learner(self, synthetic_data_clean, tmp_path):
        """Test full report generation with actual learner."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        # Train a model
        result = learner.train('coxph')

        # Evaluate
        learner.evaluate(result, n_importance_repeats=5, bootstrap=False)

        # Generate report
        report_path = tmp_path / "full_report.html"
        learner.generate_report(str(report_path))

        assert report_path.exists()

        # Check content
        content = report_path.read_text(encoding='utf-8')
        assert 'Survival Analysis Report' in content
        assert '</html>' in content

    def test_report_generator_markdown_format(self, synthetic_data_clean, tmp_path):
        """Test report generation in markdown format."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        learner.train('coxph')

        # Generate markdown report
        report_path = tmp_path / "report.md"
        learner.generate_report(str(report_path))

        assert report_path.exists()

        content = report_path.read_text(encoding='utf-8')
        assert '# ' in content  # Markdown header

    def test_report_generator_all_sections(self, small_data, tmp_path):
        """Test report with all sections."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        # Compare models
        learner.run(
            models=['coxph', 'rsf'],
            tune=False,
            model_params={'rsf': _SMALL_TREE_PARAMS},
            verbose=False,
            evaluate=True,
            bootstrap=False
        )

        # Generate report
        report_path = tmp_path / "full_sections.html"
        learner.generate_report(str(report_path))

        assert report_path.exists()

    def test_report_includes_environment_section(self, small_data, tmp_path):
        """Test that report includes the Environment & Reproducibility section."""
        from mlsurv import SurvivalLearner

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('coxph')

        report_path = tmp_path / "env_report.html"
        learner.generate_report(str(report_path))

        content = report_path.read_text(encoding='utf-8')
        assert 'Environment' in content
        assert 'Reproducibility' in content
        assert 'Software Versions' in content
        assert 'mlsurv' in content

    # Environment data-generation (versions/config DataFrames) is covered by the
    # focused unit tests in test_report_environment_section.py; the full-report
    # wiring smoke above is the one integration check kept here.

    def test_report_data_export(self, synthetic_data_clean, tmp_path):
        """Test that report generator creates exportable data."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports import ReportGenerator

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=5, bootstrap=False)

        generator = ReportGenerator(learner)
        generator.set_evaluation_result(eval_result)

        # Generate to populate sections
        report_path = tmp_path / "report_data.html"
        generator.generate(str(report_path))

        # Check that data is accessible
        data = generator.data
        assert isinstance(data, dict)

    def test_report_generator_section_tabs_start_with_executive_summary(self, synthetic_data_clean, tmp_path):
        """Top-level report tabs should open on Executive Summary first."""
        import re
        from mlsurv import SurvivalLearner
        from mlsurv.reports import ReportGenerator

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        model_result = learner.train('coxph')
        eval_result = learner.evaluate(model_result, n_importance_repeats=5, bootstrap=False)

        generator = ReportGenerator(learner)
        generator.set_evaluation_result(eval_result)
        report_path = tmp_path / "section_tabs_order.html"
        generator.generate(str(report_path))

        content = report_path.read_text(encoding='utf-8')
        match = re.search(
            r'<div class="phase-tab-buttons report-tab-buttons">.*?'
            r'<button type="button" class="tab-button active"[^>]*>([^<]+)</button>',
            content,
            re.DOTALL,
        )
        assert match is not None
        assert match.group(1).strip() == "Executive Summary"


class TestHTMLRendererAdvanced:
    """Advanced tests for HTML rendering."""

    def test_html_renderer_special_characters(self, tmp_path):
        """Test HTML renderer handles special characters."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "special.html"
        renderer = HTMLRenderer(str(output_file), title="Test <>&")

        sections = [{
            'title': 'Special Characters',
            'markdown': 'Testing < > & " \' characters',
            'figures': []
        }]

        html = renderer.render(sections)
        renderer.write(html)

        content = output_file.read_text(encoding='utf-8')
        assert '</html>' in content

    def test_html_renderer_unicode(self, tmp_path):
        """Test HTML renderer handles unicode."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "unicode.html"
        renderer = HTMLRenderer(str(output_file), title="Test Unicode")

        sections = [{
            'title': 'Unicode Test',
            'markdown': 'Testing unicode: α β γ δ ε μ σ',
            'figures': []
        }]

        html = renderer.render(sections)
        renderer.write(html)

        content = output_file.read_text(encoding='utf-8')
        assert 'α' in content or 'unicode' in content.lower()

    def test_html_renderer_large_table(self, tmp_path):
        """Test HTML renderer handles large tables."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "large_table.html"
        renderer = HTMLRenderer(str(output_file), title="Large Table Test")

        # Create large table markdown
        rows = ['| Col1 | Col2 | Col3 |', '|------|------|------|']
        for i in range(100):
            rows.append(f'| Row{i} | {i*2} | {i*3} |')

        table_markdown = '\n'.join(rows)

        sections = [{
            'title': 'Large Table',
            'markdown': table_markdown,
            'figures': []
        }]

        html = renderer.render(sections)
        renderer.write(html)

        content = output_file.read_text(encoding='utf-8')
        assert '<table>' in content
        assert 'Row99' in content  # Last row should be present


# ---------------------------------------------------------------------------
# _sanitize_markdown_link_url tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://example.com", "https://example.com"),
        ("http://example.com", "http://example.com"),
        ("#fragment", "#fragment"),
        ("", "#"),
        ("javascript:alert(1)", "#"),
        ("JAVASCRIPT:alert(1)", "#"),
        ("data:text/html,<h1>hi</h1>", "#"),
        ("ftp://example.com", "#"),
        ("file:///etc/passwd", "#"),
        ("vbscript:MsgBox", "#"),
    ],
)
def test_sanitize_markdown_link_url(url, expected):
    """Verify that only http(s) and fragment URLs are allowed."""
    from mlsurv.reports.renderers.html import _sanitize_markdown_link_url

    assert _sanitize_markdown_link_url(url) == expected


# ---------------------------------------------------------------------------
# _configure_matplotlib_cache_dir tests
# ---------------------------------------------------------------------------


class TestConfigureMatplotlibCacheDir:
    """Tests for the matplotlib cache dir fallback in mlsurv.__init__."""

    def test_early_return_when_mplconfigdir_set(self, monkeypatch):
        """If MPLCONFIGDIR is already set, function returns immediately."""
        monkeypatch.setenv("MPLCONFIGDIR", "/tmp/existing")
        from mlsurv import _configure_matplotlib_cache_dir

        _configure_matplotlib_cache_dir()
        # Should not change it
        import os
        assert os.environ["MPLCONFIGDIR"] == "/tmp/existing"

    def test_early_return_when_matplotlib_dir_writable(self, monkeypatch, tmp_path):
        """If ~/.matplotlib is writable, no fallback needed."""
        monkeypatch.delenv("MPLCONFIGDIR", raising=False)
        mpl_dir = tmp_path / ".matplotlib"
        mpl_dir.mkdir()

        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))
        from mlsurv import _configure_matplotlib_cache_dir

        _configure_matplotlib_cache_dir()
        import os
        assert os.environ.get("MPLCONFIGDIR") is None or os.environ.get("MPLCONFIGDIR") == ""

    def test_fallback_to_tempdir_when_home_unwritable(self, monkeypatch, tmp_path):
        """When HOME is not writable, falls back to tempdir."""
        monkeypatch.delenv("MPLCONFIGDIR", raising=False)

        fake_home = tmp_path / "unwritable_home"
        fake_home.mkdir()
        # No .matplotlib dir exists and home is "unwritable"
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))
        monkeypatch.setattr("os.access", lambda path, mode: False)

        from mlsurv import _configure_matplotlib_cache_dir

        _configure_matplotlib_cache_dir()

        import os
        result = os.environ.get("MPLCONFIGDIR", "")
        assert "mlsurv-matplotlib" in result

        # Cleanup
        monkeypatch.delenv("MPLCONFIGDIR", raising=False)

    def test_graceful_oserror(self, monkeypatch, tmp_path):
        """OSError during fallback dir creation does not crash."""
        monkeypatch.delenv("MPLCONFIGDIR", raising=False)

        fake_home = tmp_path / "bad_home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))
        monkeypatch.setattr("os.access", lambda path, mode: False)

        # Make mkdir raise OSError
        original_mkdir = type(fake_home).mkdir

        def _raise_oserror(*args, **kwargs):
            raise OSError("Permission denied")

        monkeypatch.setattr("pathlib.Path.mkdir", _raise_oserror)

        import warnings
        from mlsurv import _configure_matplotlib_cache_dir

        # Should not raise, but should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _configure_matplotlib_cache_dir()
            cache_warnings = [x for x in w if "matplotlib cache" in str(x.message).lower()]
            assert len(cache_warnings) == 1

        # Restore mkdir for cleanup
        monkeypatch.setattr("pathlib.Path.mkdir", original_mkdir)


# ---------------------------------------------------------------------------
# validate_survival_data warn_missing tests
# ---------------------------------------------------------------------------


class TestValidateSurvivalDataWarnMissing:
    """Tests for the warn_missing parameter."""

    def test_warn_missing_false_suppresses_warning(self):
        """warn_missing=False should suppress missing-value warnings."""
        import warnings
        import numpy as np
        import pandas as pd
        from mlsurv.utils import validate_survival_data

        X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        y = np.array(
            [(True, 1.0), (False, 2.0), (True, 3.0)],
            dtype=[("event", bool), ("time", float)],
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_survival_data(X, y, warn_missing=False, min_samples=0, min_events=0)
            missing_warnings = [x for x in w if "missing" in str(x.message).lower()]
            assert len(missing_warnings) == 0

    def test_warn_missing_true_emits_warning(self):
        """warn_missing=True (default) should emit a missing-value warning."""
        import warnings
        import numpy as np
        import pandas as pd
        from mlsurv.utils import validate_survival_data

        X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        y = np.array(
            [(True, 1.0), (False, 2.0), (True, 3.0)],
            dtype=[("event", bool), ("time", float)],
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_survival_data(X, y, warn_missing=True, min_samples=0, min_events=0)
            missing_warnings = [x for x in w if "missing" in str(x.message).lower()]
            assert len(missing_warnings) == 1


# ---------------------------------------------------------------------------
# Mistune parser fallback test
# ---------------------------------------------------------------------------


class TestMarkdownParserFallback:
    """Test that _markdown_to_html degrades gracefully on parser errors."""

    def test_fallback_to_escaped_text_on_parser_error(self, tmp_path):
        """When the markdown parser raises, output should be HTML-escaped raw text."""
        from mlsurv.reports.renderers import HTMLRenderer

        output_file = tmp_path / "fallback.html"
        renderer = HTMLRenderer(str(output_file), title="Fallback Test")

        # Replace parser with one that always raises
        original_parser = renderer._markdown_parser
        renderer._markdown_parser = lambda text: (_ for _ in ()).throw(
            ValueError("parser boom")
        )

        result = renderer._markdown_to_html("**bold** <script>alert(1)</script>")
        # Should contain escaped text, not rendered markdown
        assert "&lt;script&gt;" in result
        assert "**bold**" in result  # raw markdown, not <strong>
        assert "<!-- mlsurv: markdown parsing failed" in result

        # Restore
        renderer._markdown_parser = original_parser


# ---------------------------------------------------------------------------
# Tuning data key tests
# ---------------------------------------------------------------------------


class TestTuningDataKeys:
    """Verify tuning data keys produce correct composite keys under model_development."""

    def test_tuning_history_key(self):
        """Section stores 'tuning_history', generator exposes as 'model_development_tuning_history'."""
        from unittest.mock import MagicMock

        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        learner = MagicMock()
        learner.models = {}
        section = ModelDevelopmentSection(learner)

        # Simulate what _store_tuning_tidy_data does (keys now include 'tuning_' prefix)
        import pandas as pd
        section.data['tuning_history'] = pd.DataFrame({'trial': [1, 2]})
        section.data['tuning_param_importance'] = pd.DataFrame({'param': ['a']})

        # Simulate generator composite key construction
        composite = {f"{section.name}_{k}": v for k, v in section.data.items()}
        assert 'model_development_tuning_history' in composite
        assert 'model_development_tuning_param_importance' in composite


# ---------------------------------------------------------------------------
# CLI eval_result warning test
# ---------------------------------------------------------------------------


class TestCLIEvalResultWarning:
    """Test that CLI warns when eval_result is None and output is requested."""

    def test_warns_on_stderr_when_no_eval_result(self, monkeypatch, tmp_path, capsys):
        """cmd_run should print warning to stderr when evaluation is missing."""
        import mlsurv
        from mlsurv.cli import main
        from pathlib import Path

        data_path = tmp_path / "sample.csv"
        data_path.write_text("a,b,event,time\n1,2,1,3.0\n4,5,0,6.0\n7,8,1,9.0\n")
        report_path = tmp_path / "report.html"

        class FakeModelResult:
            name = "coxph"
            cv_metric_mean = 0.70
            cv_metric_std = 0.05
            evaluation = None  # No evaluation

        class FakeLearner:
            def __init__(self, X, y):
                pass

            def setup(self, **kwargs):
                pass

            def run(self, **kwargs):
                # run() now returns dict of {str: ModelResult}
                return {"coxph": FakeModelResult()}

            @property
            def best_model(self):
                return FakeModelResult()

            @property
            def evaluation_result(self):
                return None  # Key: no evaluation available

            def generate_report(self, output_path, eval_result=None, **kwargs):
                Path(output_path).write_text("<html>ok</html>")

        monkeypatch.setattr(mlsurv, "SurvivalLearner", FakeLearner)

        main(["run", str(data_path), "--output", str(report_path)])

        captured = capsys.readouterr()
        assert "No evaluation result" in captured.err


class TestDiagnosticsSubsection:
    """Tests for PH diagnostics in DataPreprocessingSection."""

    def test_diagnostics_in_data_preprocessing(self, small_data):
        """Schoenfeld residuals and PH test appear in DataPreprocessingSection."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('coxph')

        section = DataPreprocessingSection(learner)
        result = section.generate()

        fig_names = [name for name, _ in result['figures']]
        assert 'schoenfeld_residuals.png' in fig_names

        assert 'schoenfeld' in section.data

        sch_df = section.data['schoenfeld']
        assert set(sch_df.columns) == {'feature', 'time', 'residual'}

    def test_diagnostics_not_in_evaluation(self, small_data):
        """Diagnostics no longer appears in EvaluationSection."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.evaluation import EvaluationSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('coxph')

        section = EvaluationSection(learner)
        result = section.generate()

        fig_names = [name for name, _ in result['figures']]
        assert 'schoenfeld_residuals.png' not in fig_names

    def test_diagnostics_not_in_model_development(self, small_data):
        """Diagnostics no longer appears in ModelDevelopmentSection."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('coxph')

        section = ModelDevelopmentSection(learner)
        result = section.generate()
        fig_names = [name for name, _ in result['figures']]
        assert 'schoenfeld_residuals.png' not in fig_names

    def test_diagnostics_available_without_cox_training(self, small_data):
        """PH diagnostics work even when only RSF is trained (re-fits CoxPH internally)."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('rsf')

        section = DataPreprocessingSection(learner)
        result = section.generate()
        # Schoenfeld diagnostics re-fit CoxPH internally, so they work
        # regardless of which models were trained
        fig_names = [name for name, _ in result['figures']]
        assert 'schoenfeld_residuals.png' in fig_names


class TestDiagnosticsNaNData:
    """Tests for PH diagnostics with NaN in X_train."""

    def test_diagnostics_with_nan_produces_schoenfeld(self):
        """_add_diagnostics succeeds when X_train has NaN values."""
        import numpy as np
        import pandas as pd
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        np.random.seed(42)
        n = 80
        X = pd.DataFrame({
            'age': np.random.uniform(30, 80, n),
            'stage': np.random.randint(1, 5, n).astype(float),
            'grade': np.random.uniform(1, 4, n),
        })
        # Inject NaN
        X.loc[0:9, 'age'] = np.nan
        X.loc[15:19, 'grade'] = np.nan

        times = np.random.exponential(2, n).clip(0.1, 10)
        events = np.random.binomial(1, 0.7, n).astype(bool)
        y = make_survival_array(events, times)

        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('rsf')

        section = DataPreprocessingSection(learner)
        result = section.generate()

        fig_names = [name for name, _ in result['figures']]
        assert 'schoenfeld_residuals.png' in fig_names
        assert 'schoenfeld' in section.data
        assert '[unavailable]' not in result['markdown'].lower()


class TestTrainedModelDetails:
    """Tests for trained model details in ModelDevelopmentSection."""

    def test_coxph_convergence_info(self, small_data):
        """Trained details includes convergence info for CoxPH."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('coxph')

        section = ModelDevelopmentSection(learner)
        result = section.generate()

        assert 'Convergence' in result['markdown']

    def test_feature_selection_info(self, small_data):
        """Feature selection info reported for models with selectors in pipeline."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3, feature_selector='topk')
        learner.train('coxph')

        section = ModelDevelopmentSection(learner)
        result = section.generate()
        # Should mention selected features in the markdown
        assert 'Selected features' in result['markdown']

    def test_model_development_includes_tuning_content(self, small_data):
        """ModelDevelopmentSection generates tuning markdown when models are tuned."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.run(models=['coxnet'], n_trials=2, evaluate=False, bootstrap=False, verbose=False)

        section = ModelDevelopmentSection(learner)
        result = section.generate()
        md = result['markdown']
        assert 'Hyperparameter Tuning' in md
        assert 'Selected Hyperparameters' in md

    def test_training_evolution_removed(self, small_data):
        """Training Evolution figure should no longer be generated."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('coxph')

        section = ModelDevelopmentSection(learner)
        result = section.generate()
        fig_names = [name for name, _ in result['figures']]
        assert 'train_cv_evolution.png' not in fig_names
        assert 'Training Evolution' not in result['markdown']


class TestDeepLearningEpochs:
    """Tests for actual_epochs_ and converged_ attributes."""

    def test_deep_learning_epochs_attributes(self):
        """DeepSurv and DeepHit models should have epoch tracking attributes."""
        from mlsurv.models.deepsurv import DeepSurvModel
        from mlsurv.models.deephit import DeepHitModel

        ds = DeepSurvModel()
        assert ds.actual_epochs_ is None
        assert ds.converged_ is None

        dh = DeepHitModel()
        assert dh.actual_epochs_ is None
        assert dh.converged_ is None


class TestImportanceHeatmap:
    """Tests for importance heatmap in multi-model feature analysis."""

    def test_heatmap_generated_multi_model(self, small_data):
        """Importance heatmap is generated when multiple models are trained."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection
        import pandas as pd

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        # Train two models and evaluate each separately, then combine importance
        result_1 = learner.train('coxph')
        result_2 = learner.train('rsf')
        eval_1 = learner.evaluate(result_1, n_importance_repeats=3, bootstrap=False)
        eval_2 = learner.evaluate(result_2, n_importance_repeats=3, bootstrap=False)

        # Combine importance DataFrames to simulate multi-model eval
        combined_importance = pd.concat(
            [eval_1.importance, eval_2.importance], ignore_index=True,
        )
        eval_1.importance = combined_importance

        section = FeatureAnalysisSection(learner, eval_result=eval_1)
        assert section.is_available()
        result = section.generate()
        fig_names = [name for name, _ in result['figures']]
        assert 'importance_heatmap.png' in fig_names


class TestHazardRatiosForest:
    """Tests for hazard ratio forest plot in feature analysis."""

    def test_hazard_ratios_with_coxph(self, small_data):
        """Hazard ratio forest plots are generated when CoxPH model is available."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=3, bootstrap=False)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        assert section.is_available()
        gen_result = section.generate()
        fig_names = [name for name, _ in gen_result['figures']]
        assert 'hazard_ratios.png' in fig_names
        assert 'hazard_ratios' in section.data
        # Univariate HR forest plot also present
        assert 'univariate_hazard_ratios.png' in fig_names
        assert 'univariate_hazard_ratios' in section.data

        hr_df = section.data['hazard_ratios']
        assert 'feature' in hr_df.columns
        assert 'hr' in hr_df.columns
        assert 'ci_lower' in hr_df.columns
        assert 'ci_upper' in hr_df.columns

        # HR comparison (univariate vs multivariate)
        assert 'hazard_ratios_comparison.png' in fig_names

        # Coefficient barplot (includes univariate + trained Cox)
        assert 'coefficients.png' in fig_names

    def test_hazard_ratios_skipped_without_cox(self, small_data):
        """Hazard ratio section skips when no Cox-type model is available."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        result = learner.train('rsf')
        eval_result = learner.evaluate(result, n_importance_repeats=3, bootstrap=False)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        assert section.is_available()
        gen_result = section.generate()
        # RSF doesn't expose coefficients, so no hazard ratio forest plot
        # (unless auto-fit CoxPH kicks in -- in which case it would exist)
        # The key assertion is that it doesn't crash
        assert 'figures' in gen_result


_UNPLOTTABLE_MSG = 'cannot be placed on a log axis'


class TestHazardRatiosNonFinite:
    """The section survives a feature whose univariate CI is unbounded.

    A separated or badly scaled feature overflows lifelines'
    exp(coef + 1.96*se) to +inf and underflows exp(coef - 1.96*se) to 0.0.
    That infinity used to reach the log-decade tick computation and raise
    OverflowError, which the report generator turned into
    '*[Section unavailable: ...]*' for all of Feature Analysis.
    """

    @staticmethod
    def _degenerate_data(n=80, seed=42):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(seed)
        half = n // 2
        # Complete separation: every event sits in the sep == 1 group. That is
        # the likeliest source of the infinity; `tiny` is scaled small enough
        # that its coefficient can overflow exp() too. Neither is guaranteed by
        # construction, which is why the tests assert non-finiteness directly.
        sep = np.concatenate([np.ones(half), np.zeros(n - half)])
        event = sep.astype(bool)
        time = np.where(event, rng.uniform(1, 20, n), rng.uniform(60, 100, n))
        X = pd.DataFrame({
            'age': rng.uniform(30, 80, n),
            'biomarker': rng.normal(0, 1, n),
            'sep': sep,
            'tiny': rng.normal(0, 1, n) * 1e-8,
        })
        y = np.array(list(zip(event, time)),
                     dtype=[('event', bool), ('time', float)])
        return X, y

    @classmethod
    def _degenerate_learner(cls):
        from mlsurv import SurvivalLearner

        X, y = cls._degenerate_data()
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector=None, random_state=42, cv_folds=3,
                      n_jobs=1)
        return learner

    @staticmethod
    def _assert_really_degenerate(learner):
        """Guard against the fixture quietly going benign on an upgrade."""
        import numpy as np

        cols = ['hazard_ratio', 'hr_lower_95', 'hr_upper_95']
        vals = learner.univariate_cox[cols].to_numpy(dtype=float)
        assert not (np.isfinite(vals) & (vals > 0)).all()

    @pytest.mark.integration
    def test_section_generates_with_infinite_hazard_ratio(self):
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        learner = self._degenerate_learner()
        # The test is only meaningful if the design really produced an
        # unplottable hazard ratio; assert that before anything else.
        self._assert_really_degenerate(learner)

        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=0,
                                       bootstrap=False)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        gen_result = section.generate()

        fig_names = [name for name, _ in gen_result['figures']]
        assert 'univariate_hazard_ratios.png' in fig_names
        assert 'Section unavailable' not in gen_result['markdown']
        assert 'OverflowError' not in gen_result['markdown']

    def test_plot_warns_once_about_the_unbounded_interval(self):
        """One warning per figure, carrying the total — not one per value."""
        import warnings

        from mlsurv._warnings import ClampWarning

        learner = self._degenerate_learner()
        self._assert_really_degenerate(learner)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            learner.plot_hazard_ratios(kind='univariate')

        # A list, not a set: a set would collapse duplicates and pass however
        # many times the warning actually fired. Filtered by message because
        # _safe_exp raises a different ClampWarning from the same module.
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, ClampWarning)
                and _UNPLOTTABLE_MSG in str(w.message)]
        assert len(msgs) == 1

    def test_forest_plot_hands_plotly_no_infinities(self):
        """Plotly serializes a numpy array through a binary channel that
        carries inf straight to the browser, so the figure itself must be
        clean -- not merely free of exceptions."""
        import numpy as np

        learner = self._degenerate_learner()
        self._assert_really_degenerate(learner)

        fig = learner.plot_hazard_ratios(kind='univariate')
        for trace in fig.data:
            for values in (getattr(trace, 'x', None),
                           getattr(getattr(trace, 'error_x', None), 'array', None),
                           getattr(getattr(trace, 'error_x', None), 'arrayminus', None)):
                if values is None:
                    continue
                arr = np.asarray(values, dtype=float)
                assert not np.isinf(arr).any()

    def test_hover_still_reports_the_true_bound(self):
        """customdata is deliberately NOT sanitized.

        The plot cannot draw an unbounded interval, but the hover is where the
        reader learns the real number. Stripping it would remove the last
        honest signal.
        """
        import numpy as np

        learner = self._degenerate_learner()
        self._assert_really_degenerate(learner)

        fig = learner.plot_hazard_ratios(kind='univariate')
        customdata = [t.customdata for t in fig.data
                      if getattr(t, 'customdata', None) is not None]
        assert customdata, 'expected CI bounds in customdata'
        assert any(np.isinf(np.asarray(cd, dtype=float)).any()
                   for cd in customdata)

    def test_report_text_explains_the_missing_marker(self):
        """The ClampWarning reaches the caller; the report reaches the reader."""
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        learner = self._degenerate_learner()
        self._assert_really_degenerate(learner)
        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=0,
                                       bootstrap=False)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        markdown = section.generate()['markdown']
        assert 'could not be drawn in full' in markdown

    def test_comparison_plot_survives_and_draws_no_infinite_whisker(self):
        import numpy as np

        learner = self._degenerate_learner()
        self._assert_really_degenerate(learner)
        learner.train('coxph')

        fig = learner.plot_hazard_ratios_comparison()
        for trace in fig.data:
            err = getattr(trace, 'error_x', None)
            for values in (getattr(err, 'array', None),
                           getattr(err, 'arrayminus', None)):
                if values is None:
                    continue
                assert not np.isinf(np.asarray(values, dtype=float)).any()


class TestFeatureAnalysisEdgeCases:
    """Tests for feature analysis edge cases."""

    def test_correlation_interpretation_zero_features(self):
        """n_features=0 with non-empty corr_pairs → no ZeroDivisionError."""
        result = generate_correlation_interpretation(
            [('a', 'b', 0.95)], n_features=0,
            correlated_features={'a', 'b'}, severity='none', recommendation='',
        )
        assert 'No features available' in result

    def test_md_escape_special_chars(self):
        """_md_escape escapes * and _ characters."""
        from mlsurv.reports.sections.feature_analysis import _md_escape

        assert _md_escape("x*y") == r"x\*y"
        assert _md_escape("gene_1") == r"gene\_1"
        assert _md_escape("normal") == "normal"
        assert _md_escape("a*b_c") == r"a\*b\_c"

    def test_correlation_interpretation_escapes_names(self):
        """Feature names with * or _ are escaped in interpretation."""
        result = generate_correlation_interpretation(
            [('gene_1', 'x*y', 0.95)], n_features=10,
            correlated_features={'gene_1', 'x*y'}, severity='minor', recommendation='',
        )
        assert r'gene\_1' in result
        assert r'x\*y' in result

    @pytest.mark.parametrize("n_missing, expect_lead, expect_extent", [
        # Both format strings, each with a finite and an infinite maximum.
        ([0, 3, 12], 'in 2 of 3 features', 'up to 12 observations'),
        ([4, 3, 12], 'in all 3 features', 'up to 12 observations'),
        ([0, 3, float('inf')], 'in 2 of 3 features', None),
        ([4, 3, float('inf')], 'in all 3 features', None),
    ])
    def test_missing_data_note(self, n_missing, expect_lead, expect_extent):
        """The count is dropped when it is not finite: int(inf) raises."""
        import pandas as pd

        from mlsurv.reports.sections.feature_analysis._helpers import (
            _missing_data_note,
        )

        univar = pd.DataFrame({
            'feature': ['a', 'b', 'c'],
            'n_missing': n_missing,
        })
        note = _missing_data_note(univar)
        assert expect_lead in note
        if expect_extent is None:
            assert 'up to' not in note
        else:
            assert expect_extent in note
        assert note.rstrip().endswith('*')

    def test_missing_data_note_warns_on_a_corrupt_count(self):
        """A missing count is a non-negative integer; inf means bad input."""
        import warnings

        import pandas as pd

        from mlsurv._warnings import InternalWarning
        from mlsurv.reports.sections.feature_analysis._helpers import (
            _missing_data_note,
        )

        univar = pd.DataFrame({
            'feature': ['a', 'b'],
            'n_missing': [float('inf'), 3],
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _missing_data_note(univar)
        assert any(issubclass(w.category, InternalWarning) for w in caught)

    def test_unplottable_hr_note_names_the_feature(self):
        import numpy as np
        import pandas as pd

        from mlsurv.reports.sections.feature_analysis._helpers import (
            _unplottable_hr_note,
        )

        hr_data = pd.DataFrame({
            'feature': ['good', 'overflowed', 'underflowed'],
            'hr': [1.5, np.inf, 2.0],
            'ci_lower': [1.2, 1.0, 0.0],
            'ci_upper': [1.8, np.inf, 3.0],
        })
        note = _unplottable_hr_note(hr_data)
        assert 'overflowed' in note and 'underflowed' in note
        assert 'good' not in note

    def test_unplottable_hr_note_matches_what_the_plot_draws(self):
        """The note must describe the drawn rows, not the whole table.

        plot_hazard_ratios(return_data=True) hands back the untruncated frame,
        so scanning it directly would name a feature the reader cannot see.
        """
        import numpy as np
        import pandas as pd

        from mlsurv.reports.sections.feature_analysis._helpers import (
            _unplottable_hr_note,
        )
        from mlsurv.viz._features_hazard_ratios import (
            select_plotted_hr_features,
        )

        # 'weak' has an unplottable bound but sits closest to HR = 1, so it
        # ranks last and falls outside the top 3 the figure draws.
        hr_data = pd.DataFrame({
            'feature': ['big', 'mid', 'small', 'tiny', 'weak'],
            'hr': [8.0, 4.0, 0.25, 0.125, 1.01],
            'ci_lower': [7.0, 3.0, 0.2, 0.1, 0.99],
            'ci_upper': [9.0, 5.0, 0.3, 0.15, np.inf],
        })
        drawn = select_plotted_hr_features(hr_data, 3)
        assert 'weak' not in drawn['feature'].tolist()
        assert _unplottable_hr_note(drawn) == ""
        # Scanning the full table would have named it.
        assert 'weak' in _unplottable_hr_note(hr_data)

    def test_unplottable_hr_note_silent_on_clean_data(self):
        import pandas as pd

        from mlsurv.reports.sections.feature_analysis._helpers import (
            _unplottable_hr_note,
        )

        hr_data = pd.DataFrame({
            'feature': ['a', 'b'],
            'hr': [1.5, 2.0],
            'ci_lower': [1.2, 1.6], 'ci_upper': [1.8, 2.4],
        })
        assert _unplottable_hr_note(hr_data) == ""

    def test_unplottable_hr_note_ignores_plain_missing_data(self):
        """An all-NaN row is ordinary missing data, not an overflow."""
        import numpy as np
        import pandas as pd

        from mlsurv.reports.sections.feature_analysis._helpers import (
            _unplottable_hr_note,
        )

        hr_data = pd.DataFrame({
            'feature': ['a', 'b'],
            'hr': [np.nan, 2.0],
            'ci_lower': [np.nan, 1.6], 'ci_upper': [np.nan, 2.4],
        })
        assert _unplottable_hr_note(hr_data) == ""


class TestFeatureAnalysisNoImportance:
    """Tests for feature analysis when importance is empty."""

    def test_is_available_without_importance(self, small_data):
        """is_available() returns True when eval_result exists but importance is empty."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=0, bootstrap=False)

        assert eval_result.importance.empty
        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        assert section.is_available()

    def test_is_available_without_eval_result(self, small_data):
        """is_available() returns False when eval_result is None."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        section = FeatureAnalysisSection(learner, eval_result=None)
        assert not section.is_available()

    def test_generate_without_importance_produces_content(self, small_data):
        """generate() returns non-empty markdown with placeholder note."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=0, bootstrap=False)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        output = section.generate()
        md = output['markdown']

        assert len(md) > 0
        assert 'n\\_importance\\_repeats=0' in md
        assert 'Permutation Feature Importance' in md

    def test_independent_subsections_present_without_importance(self, small_data):
        """Correlation, hazard ratios, and coefficients still appear."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=0, bootstrap=False)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        output = section.generate()
        md = output['markdown']

        # Coefficient analysis and hazard ratios should appear for CoxPH
        assert 'Feature Coefficients' in md
        assert 'Hazard Ratios' in md

    def test_importance_content_absent_without_importance(self, small_data):
        """PFI interpretation text should NOT appear when importance is empty."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=0, bootstrap=False)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        output = section.generate()
        md = output['markdown']

        assert 'Permutation feature importance measures how much' not in md

    def test_data_dict_has_empty_importance(self, small_data):
        """section.data['importance'] should be an empty DataFrame."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection
        import pandas as pd

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=0, bootstrap=False)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        section.generate()

        assert isinstance(section.data['importance'], pd.DataFrame)
        assert section.data['importance'].empty

    def test_preamble_adapts_without_importance(self, small_data):
        """Preamble should not mention 'permutation importance' as active analysis."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=0, bootstrap=False)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        output = section.generate()
        md = output['markdown']

        # The preamble summary should not list permutation importance as a method
        assert 'permutation importance, and' not in md
        # PIMP/bootstrap guidance should also be absent
        assert 'importance_test()' not in md

    def test_multi_model_without_importance(self, small_data):
        """Multi-model evaluation without importance derives models correctly."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        r1 = learner.train('coxph')
        r2 = learner.train('rsf')
        eval_result = learner.evaluate(
            {'coxph': r1, 'rsf': r2}, n_importance_repeats=0, bootstrap=False,
        )

        assert eval_result.importance.empty

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        output = section.generate()
        md = output['markdown']

        assert len(md) > 0
        assert 'Permutation Feature Importance' in md
        assert 'n\\_importance\\_repeats=0' in md


def _make_mock_learner(with_nan=False):
    """Create a mock learner suitable for section tests."""
    from unittest.mock import MagicMock
    import numpy as np
    import pandas as pd

    learner = MagicMock()
    learner._state.is_setup = True
    learner._state.imputer = None
    learner._state.scaler = None
    learner._state.feature_selector = None
    learner._state.presplit = False
    learner._state.subpopulations_train = None
    learner._state.subpopulations_test = None
    learner._state.stratify_array = None
    # Public accessors the sections read directly — a bare MagicMock
    # auto-attr is truthy, so pin the "nothing computed" values explicitly.
    learner.is_setup = True
    learner.validation_results = {}
    learner.subpopulation_results = {}
    learner.benchmark_result = None
    learner.cv_strategy = 'stratifiedkfold'
    learner.cv_folds = 5
    learner.cv_repeats = 1
    learner.scoring = 'harrell'

    rng = np.random.RandomState(42)
    n_train, n_test = 80, 20
    X_train_data = {
        'age': rng.normal(60, 10, n_train),
        'bmi': rng.normal(25, 5, n_train),
        'stage': rng.choice(['I', 'II', 'III'], n_train),
        'grade': rng.choice([1, 2, 3], n_train),
        'score': rng.uniform(0, 1, n_train),
    }
    X_test_data = {
        'age': rng.normal(60, 10, n_test),
        'bmi': rng.normal(25, 5, n_test),
        'stage': rng.choice(['I', 'II', 'III'], n_test),
        'grade': rng.choice([1, 2, 3], n_test),
        'score': rng.uniform(0, 1, n_test),
    }
    if with_nan:
        X_train_data['age'][0] = np.nan
        X_train_data['bmi'][1] = np.nan
        X_test_data['age'][0] = np.nan

    learner.X_train = pd.DataFrame(X_train_data)
    learner.X_test = pd.DataFrame(X_test_data)
    learner.X = pd.concat([learner.X_train, learner.X_test], ignore_index=True)
    learner.y_train = pd.DataFrame({
        'time': rng.exponential(10, n_train),
        'event': rng.binomial(1, 0.6, n_train),
    })
    learner.y_test = pd.DataFrame({
        'time': rng.exponential(10, n_test),
        'event': rng.binomial(1, 0.6, n_test),
    })
    learner.y = pd.concat([learner.y_train, learner.y_test], ignore_index=True)
    learner.models = {}
    return learner


def _add_mock_model(learner, name='coxph', cv_score=0.72, is_tuned=False):
    """Add a mock model result to the learner."""
    from unittest.mock import MagicMock

    result = MagicMock()
    result.cv_metric_mean = cv_score
    result.cv_metric_std = 0.03
    result.is_tuned = is_tuned
    result.optuna_study = None
    result.scoring_metric = 'harrell'
    result.model = MagicMock()
    # Match the real ModelResult contract — algorithm identity equals name.
    result.name = name
    result.base_name = name
    learner.models[name] = result
    return result


class TestPhase1ConfigDefaults:
    """Tests for pipeline config showing defaults not tuned values."""

    def test_config_table_shows_initial_not_untuned(self):
        """Config table column is 'Initial Parameters', not 'Untuned Parameters'."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'rsf', 0.75)

        section = DataPreprocessingSection(learner)
        result = section.generate()
        md = result['markdown']
        assert 'Initial Parameters' in md
        assert 'Untuned Parameters' not in md

    def test_default_params_uses_fresh_model(self):
        """_get_default_params returns params from fresh model, not live model."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'rsf', 0.75)

        section = DataPreprocessingSection(learner)
        defaults = section._get_default_params(learner.models['rsf'])
        # Should succeed and return a dict (not None)
        assert defaults is not None
        assert isinstance(defaults, dict)


class TestPhase1Table1:
    """Tests for the feature summary table."""

    def test_table1_continuous_and_categorical(self):
        """Table 1 includes both continuous and categorical features."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        section.generate()

        assert 'table1' in section.data
        table1 = section.data['table1']
        assert 'Variable' in table1.columns
        assert 'Statistic' in table1.columns
        # Should have continuous stats (Mean (SD)) and categorical counts
        stats = table1['Statistic'].unique()
        assert any('Mean' in str(s) for s in stats)

    def test_table1_train_test_split(self):
        """Table 1 has both Train and Test in Split column."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        section.generate()

        table1 = section.data['table1']
        splits = table1['Split'].unique()
        assert 'Train' in splits
        assert 'Test' in splits

    def test_table1_other_category_nonnegative(self):
        """Categorical with >10 levels shows non-negative 'Other' count."""
        from unittest.mock import MagicMock
        import numpy as np
        import pandas as pd
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        rng = np.random.RandomState(42)
        n_train, n_test = 80, 20
        # Feature with 12 unique levels → triggers "Other" row
        levels = [f"cat_{i}" for i in range(12)]
        X_train = pd.DataFrame({
            'age': rng.normal(60, 10, n_train),
            'many_levels': rng.choice(levels, n_train),
        })
        X_test = pd.DataFrame({
            'age': rng.normal(60, 10, n_test),
            'many_levels': rng.choice(levels, n_test),
        })
        y_train = make_survival_array(
            rng.binomial(1, 0.6, n_train), rng.exponential(100, n_train)
        )
        y_test = make_survival_array(
            rng.binomial(1, 0.6, n_test), rng.exponential(100, n_test)
        )

        learner = MagicMock()
        learner.X = pd.concat([X_train, X_test], ignore_index=True)
        learner.X_train = X_train
        learner.X_test = X_test
        learner.y_train = y_train
        learner.y_test = y_test
        learner._state.is_setup = True
        learner._state.imputer = None
        learner._state.scaler = None
        learner._state.presplit = False
        learner._state.subpopulations_train = None
        learner._state.subpopulations_test = None
        learner._state.stratify_array = None
        learner.model_recommendations = None
        learner.scoring = 'harrell'
        learner.cv_strategy = 'stratifiedkfold'
        learner.cv_folds = 5
        learner.cv_repeats = 1
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        section.generate()

        table1 = section.data['table1']
        other_rows = table1[table1['Statistic'].str.startswith('Other')]
        assert len(other_rows) > 0, "Should have 'Other' rows for >10 levels"
        # Check all values are non-negative counts
        for _, row in other_rows.iterrows():
            count_str = row['Value'].split('(')[0].strip()
            assert int(count_str) >= 0, f"Other count should be non-negative, got {count_str}"


def _make_mock_learner_many_features(n_features=15):
    """Create a mock learner with many numeric features for overflow tests."""
    from unittest.mock import MagicMock
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(42)
    n_train, n_test = 80, 20
    X_train_data = {f'feat_{i:02d}': rng.normal(0, 1, n_train) for i in range(n_features)}
    X_test_data = {f'feat_{i:02d}': rng.normal(0, 1, n_test) for i in range(n_features)}

    learner = MagicMock()
    learner._state.is_setup = True
    learner._state.imputer = None
    learner._state.scaler = None
    learner._state.feature_selector = None
    learner._state.feature_groups = None
    learner._state.group_colors = None
    learner._state.presplit = False
    learner._state.subpopulations_train = None
    learner._state.subpopulations_test = None
    learner._state.stratify_array = None
    # Public accessors the sections read directly — a bare MagicMock
    # auto-attr is truthy, so pin the "nothing computed" values explicitly.
    learner.is_setup = True
    learner.validation_results = {}
    learner.subpopulation_results = {}
    learner.benchmark_result = None
    learner.cv_strategy = 'stratifiedkfold'
    learner.cv_folds = 5
    learner.cv_repeats = 1
    learner.scoring = 'harrell'

    learner.X_train = pd.DataFrame(X_train_data)
    learner.X_test = pd.DataFrame(X_test_data)
    learner.X = pd.concat([learner.X_train, learner.X_test], ignore_index=True)
    y_train = make_survival_array(
        rng.binomial(1, 0.6, n_train), rng.exponential(100, n_train)
    )
    y_test = make_survival_array(
        rng.binomial(1, 0.6, n_test), rng.exponential(100, n_test)
    )
    learner.y_train = y_train
    learner.y_test = y_test

    return learner


class TestTable1Overflow:
    """Tests for feature summary table collapsible overflow."""

    def test_table1_overflow_details_toggle(self):
        """With >10 features, overflow renders with a button toggle."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner_many_features(n_features=15)
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        result = section.generate()
        md = result['markdown']

        assert 'ft-expand-btn' in md
        assert 'ft-overflow-row' in md
        assert 'Show all 15 features' in md
        assert '<!-- MLSURV_RAW_HTML -->' in md
        # Overflow features appear in hidden rows
        assert 'feat_14' in md
        assert 'feat_10' in md

    def test_table1_no_overflow_when_few_features(self):
        """With <=10 features, no expand button appears."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        result = section.generate()
        md = result['markdown']

        assert 'ft-expand-btn' not in md
        assert 'ft-overflow-row' not in md

    def test_table1_no_overflow_at_boundary(self):
        """With exactly 10 features, no expand button appears."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner_many_features(n_features=10)
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        result = section.generate()
        md = result['markdown']

        assert 'ft-expand-btn' not in md

    def test_table1_minimal_overflow(self):
        """With 11 features, exactly 1 feature overflows."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner_many_features(n_features=11)
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        result = section.generate()
        md = result['markdown']

        assert 'ft-expand-btn' in md
        assert 'Show all 11 features' in md

    def test_table1_stores_all_features(self):
        """self.data['table1'] contains ALL features, not just visible ones."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner_many_features(n_features=15)
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        section.generate()

        table1 = section.data['table1']
        features_in_data = table1['Variable'].unique()
        assert len(features_in_data) == 15


class TestPhase1MissingData:
    """Tests for the missing data breakdown."""

    def test_missing_data_with_nan(self):
        """Per-variable missing data table appears when NaN present."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner(with_nan=True)
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        section.generate()

        assert 'missing_data' in section.data
        missing = section.data['missing_data']
        assert len(missing) > 0
        assert 'variable' in missing.columns

    def test_empty_variable_is_called_out(self):
        """A 100%-missing variable reads as "100.0%" in the table; the note
        says what actually happens to it."""
        import numpy as np

        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner(with_nan=True)
        learner.X_train['dz_stage'] = np.nan
        learner.X_test['dz_stage'] = np.nan
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        markdown = section.generate()['markdown']

        assert 'dz_stage' in markdown
        assert 'no observed values' in markdown
        assert 'Nothing can be learned from them' in markdown

    def test_no_empty_variable_note_when_all_have_data(self):
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner(with_nan=True)
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        markdown = section.generate()['markdown']

        assert 'no observed values' not in markdown

    def test_missing_data_clean(self):
        """Clean dataset shows 'No missing values' message."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner(with_nan=False)
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        result = section.generate()

        md = result['markdown']
        assert 'No missing values' in md
        assert 'missing_data' not in section.data


class TestPhase1EPVCensoring:
    """Tests for EPV and censoring rate."""

    def test_epv_shown_and_flagged(self):
        """EPV row appears in dataset summary."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        section.generate()

        summary = section.data['dataset_summary']
        metrics = summary['metric'].tolist()
        assert 'Events per variable (EPV)' in metrics

    def test_censoring_rate_shown(self):
        """Censoring rate rows appear in dataset summary."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        section.generate()

        summary = section.data['dataset_summary']
        metrics = summary['metric'].tolist()
        assert any('Censored' in m for m in metrics)


class TestPhase1KMCurve:
    """Tests for the overall KM figure."""

    def test_km_figure_generated(self):
        """Overall KM figure appears in figures list."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        result = section.generate()

        figure_names = [name for name, _ in result['figures']]
        assert 'overall_km.png' in figure_names


class TestPhase1Summary:
    """Tests for the executive summary overhaul."""

    def test_summary_no_boilerplate(self):
        """Summary no longer contains generic 'Survival analysis predicts' text."""
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        section = SummarySection(learner)
        result = section.generate()
        md = result['markdown']
        assert 'Survival analysis predicts the time' not in md

    def test_summary_pipeline_facts(self):
        """Summary contains factual observation count and model info."""
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        section = SummarySection(learner)
        result = section.generate()
        md = result['markdown']
        assert '*100*' in md  # 80 + 20 observations
        assert '*80*' in md   # training
        assert '*20*' in md   # test

    def test_key_findings_box(self):
        """Key findings blockquote is present in output."""
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        section = SummarySection(learner)
        result = section.generate()
        md = result['markdown']
        assert '> **Key Findings**' in md
        assert '> - **Best model:**' in md
        assert '> - **Sample size:**' in md

    def test_summary_prose_no_subheadings(self):
        """Summary uses flowing prose, not ### subheadings."""
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        section = SummarySection(learner)
        result = section.generate()
        md = result['markdown']
        assert '### ' not in md
        assert '**Model Training:**' not in md
        assert '**Model Comparison:**' not in md

    def test_censoring_rate_in_summary(self):
        """Summary mentions censoring."""
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        section = SummarySection(learner)
        result = section.generate()
        md = result['markdown']
        assert 'censored' in md.lower()


class TestPhase1ConsolidatedFeatures:
    """Tests for feature importance in summary."""

    def test_top_features_in_summary(self):
        """Top features from importance appear in summary prose."""
        from mlsurv.reports.sections.summary import SummarySection
        from unittest.mock import MagicMock
        import pandas as pd

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)
        _add_mock_model(learner, 'rsf', 0.75)

        eval_result = MagicMock()
        eval_result.importance = pd.DataFrame({
            'model_type': ['coxph'] * 3 + ['rsf'] * 3,
            'feature': ['age', 'bmi', 'score'] * 2,
            'perm_mean': [0.05, 0.03, 0.02] * 2,
        })
        eval_result.auc_t = None
        eval_result.test_set = None
        eval_result.bootstrap = None

        section = SummarySection(learner, eval_result=eval_result)
        result = section.generate()
        md = result['markdown']
        assert '*age*' in md
        assert '*bmi*' in md
        assert 'top predictive features' in md.lower()


class TestPhase1LowerIsBetter:
    """Tests for lower-is-better metric handling in summary.

    ``cv_metric_mean`` is stored in higher-is-better space — for IBS,
    ``score_survival_model`` returns ``-ibs``. So the *better* model
    has the *less negative* ``cv_metric_mean`` (mathematically the
    larger value), and the displayed score is un-negated back to the
    raw IBS.
    """

    def test_methods_paragraph_selects_lower_score(self):
        """With IBS scoring, best model has the lower raw IBS."""
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        learner.scoring = 'integrated_brier'
        # cv_metric_mean = -ibs convention.
        _add_mock_model(learner, 'coxph', -0.25)  # raw IBS 0.25
        _add_mock_model(learner, 'rsf', -0.15)    # raw IBS 0.15 (better)

        section = SummarySection(learner)
        result = section.generate()
        md = result['markdown']
        # RSF (raw IBS 0.15) should be selected as best, not CoxPH (0.25)
        assert '*0.150*' in md
        assert '*Random Survival Forest*' in md or '*RSF*' in md

    def test_key_findings_selects_lower_score(self):
        """Key findings box selects model with lower raw IBS."""
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        learner.scoring = 'integrated_brier'
        _add_mock_model(learner, 'coxph', -0.25)
        _add_mock_model(learner, 'rsf', -0.15)

        section = SummarySection(learner)
        result = section.generate()
        md = result['markdown']
        assert '0.150' in md
        # Best model line should reference RSF and the un-negated value
        for line in md.split('\n'):
            if '**Best model:**' in line:
                assert '0.150' in line
                break

    def test_build_test_set_sentence_multi_model_lower_is_better(self):
        """Multi-model bootstrap selects lower score for IBS."""
        from unittest.mock import MagicMock
        import pandas as pd
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        learner.scoring = 'integrated_brier'
        # cv_metric_mean = -ibs convention; -0.15 (raw IBS 0.15) is better.
        _add_mock_model(learner, 'coxph', -0.25)
        _add_mock_model(learner, 'rsf', -0.15)

        # Build mock bootstrap results
        boot_coxph = MagicMock()
        boot_coxph.ci_level = 0.95
        boot_coxph.summary = pd.DataFrame({
            'metric': ['integrated_brier'], 'mean': [0.22],
            'ci_lower': [0.18], 'ci_upper': [0.26],
        })
        boot_coxph.iterations = pd.DataFrame({
            'iteration': list(range(100)),
        })
        boot_rsf = MagicMock()
        boot_rsf.ci_level = 0.95
        boot_rsf.summary = pd.DataFrame({
            'metric': ['integrated_brier'], 'mean': [0.12],
            'ci_lower': [0.09], 'ci_upper': [0.15],
        })
        boot_rsf.iterations = pd.DataFrame({
            'iteration': list(range(100)),
        })

        section = SummarySection(
            learner, bootstrap_results={'coxph': boot_coxph, 'rsf': boot_rsf}
        )
        result = section.generate()
        md = result['markdown']
        # RSF with 0.12 should be selected as best on test set
        assert '0.120' in md

    def test_build_test_set_sentence_uses_cv_best_not_bootstrap_best(self):
        """Test-set sentence reports CV-selected best, not bootstrap-best."""
        from unittest.mock import MagicMock
        import pandas as pd
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        learner.scoring = 'harrell'
        # CV: RSF wins (0.68 > 0.65)
        _add_mock_model(learner, 'coxph', 0.65)
        _add_mock_model(learner, 'rsf', 0.68)

        # Bootstrap: CoxPH wins (0.72 > 0.70)
        boot_coxph = MagicMock()
        boot_coxph.ci_level = 0.95
        boot_coxph.summary = pd.DataFrame({
            'metric': ['harrell'], 'mean': [0.72],
            'ci_lower': [0.68], 'ci_upper': [0.76],
        })
        boot_coxph.iterations = pd.DataFrame({'iteration': list(range(100))})
        boot_rsf = MagicMock()
        boot_rsf.ci_level = 0.95
        boot_rsf.summary = pd.DataFrame({
            'metric': ['harrell'], 'mean': [0.70],
            'ci_lower': [0.66], 'ci_upper': [0.74],
        })
        boot_rsf.iterations = pd.DataFrame({'iteration': list(range(100))})

        section = SummarySection(
            learner, bootstrap_results={'coxph': boot_coxph, 'rsf': boot_rsf}
        )
        trained_models = dict(learner.models)
        sentence = section._build_test_set_sentence(trained_models)
        assert sentence is not None
        # Should report RSF (CV-best), not CoxPH (bootstrap-best)
        assert 'RSF' in sentence or '0.700' in sentence
        assert '0.720' not in sentence


class TestPhase1MultiModelSummary:
    """Tests for multi-model summary formatting."""

    def test_two_models_listed(self):
        """Summary with 2 models says 'compared *X* and *Y*' (no Oxford comma)."""
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)
        _add_mock_model(learner, 'rsf', 0.75)

        section = SummarySection(learner)
        result = section.generate()
        md = result['markdown']
        assert 'compared' in md.lower()
        # Both names should appear, joined by ' and ' without an Oxford comma.
        assert '*CoxPH* and *RSF*' in md or '*RSF* and *CoxPH*' in md
        assert ', and *' not in md

    def test_three_models_listed(self):
        """Summary with 3 models lists all three names."""
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)
        _add_mock_model(learner, 'rsf', 0.75)
        _add_mock_model(learner, 'gbsa', 0.74)

        section = SummarySection(learner)
        result = section.generate()
        md = result['markdown']
        assert '*3*' in md

    def test_test_set_sentence_uses_point_estimate(self):
        """Multi-model test-set sentence uses bootstrap 'point', not 'mean'."""
        from unittest.mock import MagicMock
        import pandas as pd
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)
        _add_mock_model(learner, 'rsf', 0.75)

        boot_rsf = MagicMock()
        boot_rsf.ci_level = 0.95
        boot_rsf.summary = pd.DataFrame({
            'metric': ['harrell'],
            'point': [0.753], 'mean': [0.741],
            'ci_lower': [0.70], 'ci_upper': [0.79],
        })
        boot_rsf.iterations = pd.DataFrame({'iteration': range(2)})
        boot_coxph = MagicMock()
        boot_coxph.ci_level = 0.95
        boot_coxph.summary = pd.DataFrame({
            'metric': ['harrell'],
            'point': [0.721], 'mean': [0.715],
            'ci_lower': [0.67], 'ci_upper': [0.77],
        })
        boot_coxph.iterations = pd.DataFrame({'iteration': range(2)})

        section = SummarySection(
            learner, bootstrap_results={'rsf': boot_rsf, 'coxph': boot_coxph},
        )
        result = section.generate()
        md = result['markdown']
        # RSF wins on cv_metric_mean (0.75 > 0.72) and gets reported.
        assert '*0.753*' in md
        assert '*0.741*' not in md
        assert '(bootstrap mean)' not in md

    def test_test_set_sentence_falls_back_to_mean_with_label(self):
        """When 'point' is missing, fall back to 'mean', change the label,
        AND emit an InternalWarning (project rule: No Silent Data Loss)."""
        import warnings as _warnings
        from unittest.mock import MagicMock
        import pandas as pd
        from mlsurv._warnings import InternalWarning
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        boot = MagicMock()
        boot.ci_level = 0.95
        boot.summary = pd.DataFrame({
            'metric': ['harrell'], 'mean': [0.715],
            'ci_lower': [0.67], 'ci_upper': [0.77],
        })
        boot.iterations = pd.DataFrame({'iteration': range(2)})

        section = SummarySection(learner, bootstrap_results={'coxph': boot})
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            result = section.generate()
        md = result['markdown']
        assert '*0.715*' in md
        assert '(bootstrap mean)' in md
        fallback_warnings = [
            w for w in caught
            if issubclass(w.category, InternalWarning)
            and "falls back to the bootstrap mean" in str(w.message)
        ]
        assert fallback_warnings, "expected an InternalWarning on point→mean fallback"

    def test_top_features_box_labeled_with_best_model(self):
        """Key Findings 'Top features' bullet names the best model and uses
        that model's PFI ranking, not a cross-model average."""
        from unittest.mock import MagicMock
        import pandas as pd
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        coxph = _add_mock_model(learner, 'coxph', 0.72)
        rsf = _add_mock_model(learner, 'rsf', 0.85)

        eval_result = MagicMock()
        eval_result.importance = pd.DataFrame({
            'model_type': ['rsf', 'rsf', 'coxph', 'coxph'],
            'feature':    ['rsf_only_top', 'shared', 'cox_only_top', 'shared'],
            'perm_mean':  [0.50,           0.10,    0.40,           0.10],
        })
        coxph.evaluation = eval_result
        rsf.evaluation = eval_result

        section = SummarySection(learner, eval_result=eval_result)
        result = section.generate()
        md = result['markdown']
        # RSF wins on cv_metric_mean and should label the bullet.
        assert '**Top features (RSF):**' in md
        # Best model's top feature appears; the cox-only winner does not lead.
        rsf_idx = md.find('rsf_only_top')
        cox_idx = md.find('cox_only_top')
        assert rsf_idx != -1
        assert cox_idx == -1 or rsf_idx < cox_idx


class TestPhase1MissingDataHeatmap:
    """Tests for missing data heatmap figure generation."""

    def test_heatmap_figure_generated_with_nan(self):
        """Heatmap figure appears when >= 2 features have NaN."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner(with_nan=True)
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        result = section.generate()

        figure_names = [name for name, _ in result['figures']]
        assert 'missing_data_heatmap.png' in figure_names

    def test_no_heatmap_without_nan(self):
        """No heatmap figure when data is clean."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner(with_nan=False)
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        result = section.generate()

        figure_names = [name for name, _ in result['figures']]
        assert 'missing_data_heatmap.png' not in figure_names


class TestPhase1KeyFindingsCVMetrics:
    """Tests for CV metrics display in key findings box."""

    def test_key_findings_shows_cv_std(self):
        """Key findings box shows +/- std from CV, not bootstrap CI."""
        from unittest.mock import MagicMock
        import pandas as pd
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        model = _add_mock_model(learner, 'coxph', 0.72)
        model.cv_metric_std = 0.035

        boot = MagicMock()
        boot.summary = pd.DataFrame({
            'metric': ['harrell'], 'mean': [0.71],
            'ci_lower': [0.65], 'ci_upper': [0.77],
        })

        section = SummarySection(learner, bootstrap_results={'coxph': boot})
        result = section.generate()
        md = result['markdown']
        # Key findings should show CV std, not bootstrap CI
        assert '\u00b1 0.035' in md
        # Bootstrap CI should NOT appear in the key findings line
        key_findings_end = md.find('**Sample size:**')
        key_findings_box = md[:key_findings_end] if key_findings_end != -1 else md
        assert '95% CI' not in key_findings_box

    def test_key_findings_shows_cv_description(self):
        """Key findings box labels score with CV description and 'mean'."""
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        section = SummarySection(learner)
        result = section.generate()
        md = result['markdown']
        assert '5-fold stratified cross-validation mean' in md

    def test_methods_paragraph_says_cv_mean(self):
        """Methods paragraph labels score as CV mean."""
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        section = SummarySection(learner)
        result = section.generate()
        md = result['markdown']
        assert '5-fold stratified cross-validation mean' in md
        assert '*0.720*' in md

    def test_key_findings_nan_std_omits_plusminus(self):
        """When cv_metric_std is NaN, +/- is omitted gracefully."""
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        model = _add_mock_model(learner, 'coxph', 0.72)
        model.cv_metric_std = float('nan')

        section = SummarySection(learner)
        result = section.generate()
        md = result['markdown']
        assert '\u00b1' not in md
        assert '*0.720*' in md

    def test_cv_description_with_repeats(self):
        """CV description includes repeat count when cv_repeats > 1."""
        from mlsurv.reports.sections.summary import SummarySection

        learner = _make_mock_learner()
        learner.cv_repeats = 3
        _add_mock_model(learner, 'coxph', 0.72)

        section = SummarySection(learner)
        result = section.generate()
        md = result['markdown']
        assert '3\u00d7 5-fold' in md


class TestFeatureAnalysisReorder:
    """Tests for feature analysis section reorder and new content."""

    def test_correlation_before_importance(self, small_data):
        """Correlation analysis appears before importance tables in output."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=3, bootstrap=False)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        gen_result = section.generate()
        md = gen_result['markdown']

        # Correlation heading must precede importance/PFI content
        corr_idx = md.find('### Feature Correlation Analysis')
        # The importance table is inside raw HTML; look for the table container
        imp_idx = md.find('feature-analysis-table')
        assert corr_idx != -1, "Correlation heading not found"
        assert imp_idx != -1, "Importance table not found"
        assert corr_idx < imp_idx, "Correlation must come before importance"

    def test_feature_selection_audit_shown(self, small_data):
        """Feature selection audit appears as HTML table when TopKCoxSelector drops features."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        # Use k=3 to force feature dropping (small_data has 5 features)
        learner.setup(random_state=42, cv_folds=3, feature_selector=TopKCoxSelector(k=3))

        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=3, bootstrap=False)

        # Test the audit method directly (full generate() may fail on int
        # feature names from pipeline, a pre-existing issue)
        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        audit_html = generate_feature_selection_audit(section)

        assert '### Feature Selection Audit' in audit_html
        assert 'TopKCoxSelector' in audit_html
        # Single model: "Selected In" column shows model name and em-dash
        assert 'Selected In' in audit_html
        assert 'CoxPH' in audit_html
        assert '\u2014' in audit_html  # em-dash for dropped features
        # HTML table structure
        assert 'feature-analysis-table' in audit_html
        assert '<!-- MLSURV_RAW_HTML -->' in audit_html
        assert '<!-- /MLSURV_RAW_HTML -->' in audit_html
        assert 'feature_selection' in section.data
        # Original feature names should be resolved from integer indices
        df = section.data['feature_selection']
        assert all(isinstance(f, str) for f in df['feature'])
        # Backward-compatible 'selected' bool + new 'selected_in' list
        assert 'selected' in df.columns
        assert 'selected_in' in df.columns
        assert df['selected'].dtype == bool
        # 5 features, default n_features=10 -> no expand button needed
        assert 'ft-expand-btn' not in audit_html
        assert 'ft-overflow-row' not in audit_html

    def test_feature_selection_audit_expandable(self, small_data):
        """Audit table shows expand button when features exceed n_features."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        # Use k=3 to force feature dropping (small_data has 5 features)
        learner.setup(random_state=42, cv_folds=3, feature_selector=TopKCoxSelector(k=3))

        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=3, bootstrap=False)

        # Set n_features=2 so 5 features triggers the expand button
        section = FeatureAnalysisSection(learner, eval_result=eval_result, n_features=2)
        audit_html = generate_feature_selection_audit(section)

        assert 'ft-expand-btn' in audit_html
        assert 'ft-overflow-row' in audit_html
        # 5 total - 2 visible = 3 overflow rows
        assert audit_html.count('ft-overflow-row') == 3
        assert 'data-n-visible="2"' in audit_html
        assert 'data-n-total="5"' in audit_html
        assert 'Show all 5 features' in audit_html

    def test_feature_selection_audit_hidden(self, small_data):
        """Feature selection audit not shown when no selector in pipeline."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3, feature_selector=None)

        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=3, bootstrap=False)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        gen_result = section.generate()
        md = gen_result['markdown']

        assert '### Feature Selection Audit' not in md

    def test_feature_selection_audit_multi_model_same_k(self, small_data):
        """Audit shows 'All models' when multiple models share the same selector."""
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3, feature_selector=TopKCoxSelector(k=3))

        learner.train('coxph')
        learner.train('rsf')
        # Use coxph result for eval (just need a valid eval_result)
        eval_result = learner.evaluate(
            learner.models['coxph'], n_importance_repeats=3, bootstrap=False,
        )

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        audit_html = generate_feature_selection_audit(section)

        assert 'Selected In' in audit_html
        assert 'All models' in audit_html
        assert 'All 2 models share the same selection' in audit_html
        assert '\u2014' in audit_html  # em-dash for dropped features

        df = section.data['feature_selection']
        selected_rows = df[df['selected']]
        for names in selected_rows['selected_in']:
            assert 'CoxPH' in names
            assert 'RSF' in names

    def test_feature_selection_audit_multi_model_different_k(self, small_data):
        """Audit shows per-model names when models have different effective_k."""
        from unittest.mock import MagicMock
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3, feature_selector=TopKCoxSelector(k=3))

        # Train a real model to get a valid pipeline
        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=3, bootstrap=False)

        # Get the real selector's sorted_features_ for the mock
        real_selector = result.model.named_steps['selector']
        sorted_feats = real_selector.sorted_features_

        # Create a mock second model with k=4 (selects one more feature)
        mock_selector = MagicMock()
        mock_selector.sorted_features_ = sorted_feats
        mock_selector.effective_k_ = 4
        mock_selector.n_features_in_ = real_selector.n_features_in_
        mock_selector.selected_features_ = [f for f, _ in sorted_feats[:4]]

        mock_pipeline = MagicMock()
        mock_pipeline.named_steps = {'selector': mock_selector}

        mock_result = MagicMock()
        mock_result.model = mock_pipeline

        # Inject the mock model
        learner.models['rsf'] = mock_result

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        audit_html = generate_feature_selection_audit(section)

        assert 'Selected In' in audit_html
        # Description should mention the range
        assert 'between 3 and 4' in audit_html
        # The 4th feature (selected by RSF only) should show just 'RSF'
        assert 'RSF' in audit_html
        # Should NOT say "All models" since they disagree
        assert 'All models' not in audit_html

        df = section.data['feature_selection']
        # First 3 features: selected by both
        for names in df.iloc[:3]['selected_in']:
            assert len(names) == 2
        # 4th feature: selected by RSF only
        assert df.iloc[3]['selected_in'] == ['RSF']
        # 5th feature: selected by neither
        assert df.iloc[4]['selected_in'] == []
        assert not df.iloc[4]['selected']

    def test_feature_selection_audit_no_drop_selector_excluded(self, small_data):
        """Selector that keeps all features is excluded from the audit."""
        from unittest.mock import MagicMock
        from mlsurv import SurvivalLearner
        from mlsurv.preprocessing import TopKCoxSelector
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        # k=3 drops features for the real model
        learner.setup(random_state=42, cv_folds=3, feature_selector=TopKCoxSelector(k=3))

        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=3, bootstrap=False)
        real_selector = result.model.named_steps['selector']

        # Mock a second model whose selector keeps ALL features (no drop)
        mock_selector = MagicMock()
        mock_selector.sorted_features_ = real_selector.sorted_features_
        mock_selector.selected_features_ = [f for f, _ in real_selector.sorted_features_]
        mock_selector.effective_k_ = real_selector.n_features_in_  # keeps all
        mock_selector.n_features_in_ = real_selector.n_features_in_

        mock_pipeline = MagicMock()
        mock_pipeline.named_steps = {'selector': mock_selector}
        mock_result = MagicMock()
        mock_result.model = mock_pipeline
        learner.models['rsf'] = mock_result

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        audit_html = generate_feature_selection_audit(section)

        # Only CoxPH should appear (RSF keeps all features, excluded)
        assert 'CoxPH' in audit_html
        assert 'RSF' not in audit_html
        # Should be single-model description, not multi-model
        assert 'All models' not in audit_html

    def test_feature_selection_audit_unified_ranking_different_pvalues(self):
        """Unified ranking picks min p-value when selectors disagree on rankings."""
        from mlsurv.reports.sections.feature_analysis._audit import (
            _build_unified_ranking,
        )
        from unittest.mock import MagicMock

        sel_a = MagicMock()
        sel_a.sorted_features_ = [('age', 0.01), ('bmi', 0.05), ('wbc', 0.10)]
        sel_b = MagicMock()
        sel_b.sorted_features_ = [('bmi', 0.02), ('age', 0.03), ('wbc', 0.08)]

        ranking = _build_unified_ranking({'coxph': sel_a, 'rsf': sel_b})

        # age: min(0.01, 0.03)=0.01, bmi: min(0.05, 0.02)=0.02, wbc: min(0.10, 0.08)=0.08
        assert ranking[0] == ('age', 0.01)
        assert ranking[1] == ('bmi', 0.02)
        assert ranking[2] == ('wbc', 0.08)

    def test_correlation_warning_moderate(self):
        """Callout HTML appears for moderate multicollinearity."""
        result = generate_correlation_warning_callout('moderate', 'Review correlated pairs.')
        assert 'mlsurv-callout-warning' in result
        assert 'Multicollinearity Warning' in result
        assert 'Review correlated pairs.' in result

    def test_correlation_warning_minor_hidden(self):
        """No callout for minor multicollinearity."""
        result = generate_correlation_warning_callout('minor', 'Some recommendation.')
        assert result == ''

    def test_univariate_hr_forest_plot(self, small_data):
        """Univariate HR forest plot is generated when CoxPH model is available."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=3, bootstrap=False)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        gen_result = section.generate()
        fig_names = [name for name, _ in gen_result['figures']]

        assert 'univariate_hazard_ratios.png' in fig_names
        assert 'univariate_hazard_ratios' in section.data

        hr_df = section.data['univariate_hazard_ratios']
        assert 'feature' in hr_df.columns
        assert 'hr' in hr_df.columns
        assert 'ci_lower' in hr_df.columns
        assert 'ci_upper' in hr_df.columns

    def test_correlation_severity_stored(self, small_data):
        """correlation_severity is stored in section.data."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=3, bootstrap=False)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        section.generate()

        assert 'correlation_severity' in section.data
        assert section.data['correlation_severity'] in ('none', 'minor', 'moderate', 'substantial')

    def test_hazard_ratios_multivariate_still_present(self, small_data):
        """Multivariate HR forest plot still generated alongside univariate."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.feature_analysis import FeatureAnalysisSection

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)

        result = learner.train('coxph')
        eval_result = learner.evaluate(result, n_importance_repeats=3, bootstrap=False)

        section = FeatureAnalysisSection(learner, eval_result=eval_result)
        gen_result = section.generate()
        fig_names = [name for name, _ in gen_result['figures']]

        assert 'hazard_ratios.png' in fig_names
        assert 'hazard_ratios' in section.data

    def test_assess_correlation_severity_boundaries(self):
        """_assess_correlation_severity returns correct labels at boundaries."""
        from mlsurv.reports.sections.feature_analysis import _assess_correlation_severity

        # Substantial: n_pairs >= 10 OR pct_involved > 50
        sev, _ = _assess_correlation_severity(n_pairs=10, pct_involved=30, max_corr=0.85)
        assert sev == "substantial"
        sev, _ = _assess_correlation_severity(n_pairs=3, pct_involved=60, max_corr=0.85)
        assert sev == "substantial"

        # Moderate: n_pairs >= 5 OR max_corr > 0.9
        sev, _ = _assess_correlation_severity(n_pairs=5, pct_involved=30, max_corr=0.85)
        assert sev == "moderate"
        sev, _ = _assess_correlation_severity(n_pairs=3, pct_involved=20, max_corr=0.95)
        assert sev == "moderate"

        # Minor: below all thresholds
        sev, _ = _assess_correlation_severity(n_pairs=3, pct_involved=20, max_corr=0.85)
        assert sev == "minor"

    def test_correlated_feature_dagger_marker(self):
        """Features in correlated pairs get † marker in importance tables."""
        import numpy as np
        import pandas as pd

        def _summary_row(correlated):
            return pd.Series({
                'feature': 'age', 'perm_mean': 0.05, 'perm_std': 0.01,
                'uni_p_value': 0.01, 'uni_hazard_ratio': 1.5,
                'p_adjusted': np.nan, 'sig_stars': None,
                'mean_abs_shap': np.nan, 'correlated': correlated,
            })

        # Not correlated — no marker
        result = build_feature_row(_summary_row(False), 'coxph', 0.7)
        assert result['Feature'] == 'age'

        # Correlated flag set — marker added
        result = build_feature_row(_summary_row(True), 'coxph', 0.7)
        assert result['Feature'] == 'age †'

        # Flag unknown (NA) — no marker
        result = build_feature_row(_summary_row(pd.NA), 'coxph', 0.7)
        assert result['Feature'] == 'age'


# =====================================================================
# Evaluation section restructure tests
# =====================================================================


class TestEvaluationRestructure:
    """Tests for the evaluation section restructure."""

    @staticmethod
    def _make_section(with_bootstrap=False, with_cv=True, multi_metric=False):
        """Create an EvaluationSection with mocked data."""
        from mlsurv.reports.sections.evaluation import EvaluationSection
        from unittest.mock import MagicMock
        import numpy as np
        import pandas as pd

        learner = MagicMock()
        learner._state.is_setup = True
        # No IPCW support violation by default; without this the auto-speccing
        # MagicMock returns a truthy mock for the flag, so the performance
        # subsection's caveat path runs min(mock, mock) and raises TypeError.
        learner._state.ipcw_support_flag = None
        learner.scoring = 'harrell'
        # Prevent calibration subsection from activating
        learner.X_test = None
        learner.y_test = None

        # Create model results with cv_metric_mean
        coxph_result = MagicMock()
        coxph_result.is_tuned = False
        coxph_result.cv_metric_mean = 0.78 if with_cv else None
        coxph_result.cv_metric_std = 0.03
        coxph_eval = MagicMock(spec=[])
        coxph_eval.c_index_test = 0.75
        coxph_eval.concordance = pd.DataFrame({
            'c_index_test': [0.75],
            'integrated_brier_test': [0.18],
        })
        coxph_result.evaluation = coxph_eval

        rsf_result = MagicMock()
        rsf_result.is_tuned = True
        rsf_result.cv_metric_mean = 0.82 if with_cv else None
        rsf_result.cv_metric_std = 0.02
        rsf_eval = MagicMock(spec=[])
        rsf_eval.c_index_test = 0.80
        rsf_eval.concordance = pd.DataFrame({
            'c_index_test': [0.80],
            'integrated_brier_test': [0.16],
        })
        rsf_result.evaluation = rsf_eval

        learner.models = {'coxph': coxph_result, 'rsf': rsf_result}

        eval_result = MagicMock()
        eval_result.test_set = pd.DataFrame({
            'model_type': ['coxph'] * 5 + ['rsf'] * 5,
            'time': [10, 20, 30, 40, 50, 10, 20, 30, 40, 50],
            'event': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
            'risk_score': np.random.default_rng(42).uniform(0, 1, 10).tolist(),
        })
        eval_result.concordance = pd.DataFrame({
            'model_type': ['coxph', 'rsf'],
            'c_index': [0.75, 0.80],
        })
        eval_result.auc_t = None
        eval_result.bootstrap = None

        bootstrap_results = None
        if with_bootstrap:
            boot_coxph = MagicMock()
            boot_coxph.ci_level = 0.95
            metrics = ['harrell']
            if multi_metric:
                metrics = ['harrell', 'integrated_brier']
            boot_coxph.summary = pd.DataFrame({
                'metric': metrics,
                'mean': [0.75, 0.22][:len(metrics)],
                'point': [0.74, 0.21][:len(metrics)],
                'ci_lower': [0.70, 0.18][:len(metrics)],
                'ci_upper': [0.80, 0.26][:len(metrics)],
                'model_type': ['coxph'] * len(metrics),
            })
            boot_coxph.iterations = pd.DataFrame({
                'model_type': ['coxph'] * 10,
                'model': ['Cox PH'] * 10,
                'iteration': list(range(1, 11)),
                'metric_name': ['harrell'] * 10,
                'metric': ['harrell'] * 10,
                'value': np.random.default_rng(42).normal(0.75, 0.03, 10).tolist(),
            })
            boot_coxph.get_distribution = MagicMock(
                side_effect=lambda m: np.random.default_rng(42).normal(0.75, 0.03, 10)
                if m == 'harrell' else None
            )

            boot_rsf = MagicMock()
            boot_rsf.ci_level = 0.95
            boot_rsf.summary = pd.DataFrame({
                'metric': metrics,
                'mean': [0.80, 0.19][:len(metrics)],
                'point': [0.79, 0.18][:len(metrics)],
                'ci_lower': [0.76, 0.15][:len(metrics)],
                'ci_upper': [0.84, 0.23][:len(metrics)],
                'model_type': ['rsf'] * len(metrics),
            })
            boot_rsf.iterations = pd.DataFrame({
                'model_type': ['rsf'] * 10,
                'model': ['RSF'] * 10,
                'iteration': list(range(1, 11)),
                'metric_name': ['harrell'] * 10,
                'metric': ['harrell'] * 10,
                'value': np.random.default_rng(43).normal(0.80, 0.02, 10).tolist(),
            })
            boot_rsf.get_distribution = MagicMock(
                side_effect=lambda m: np.random.default_rng(43).normal(0.80, 0.02, 10)
                if m == 'harrell' else None
            )

            bootstrap_results = {'coxph': boot_coxph, 'rsf': boot_rsf}

        section = EvaluationSection(
            learner, eval_result=eval_result,
            bootstrap_results=bootstrap_results,
        )
        return section

    @staticmethod
    def _make_mock_subsections(
        disc=False, roc=False, cal=False, strat=False, paired=False,
    ):
        """Create mock subsection objects with the given availability flags."""
        from unittest.mock import MagicMock

        def _mock(available):
            m = MagicMock()
            m.is_available.return_value = available
            return m

        return _mock(disc), _mock(roc), _mock(cal), _mock(strat), _mock(paired)

    def _call_outline(self, section, **subsection_flags):
        """Call _add_outline and return the joined text."""
        md_lines = []
        mocks = self._make_mock_subsections(**subsection_flags)
        section._add_outline(md_lines, *mocks)
        return '\n'.join(md_lines)

    def test_evaluation_toc_has_anchor_links(self):
        """TOC contains [label](#anchor) format entries."""
        section = self._make_section(with_cv=True)
        text = self._call_outline(
            section, paired=True, disc=True, cal=True, strat=True,
        )
        assert '[Methodology](#methodology)' in text
        assert '[Overview](#overview)' in text
        assert '[Paired Model Comparison](#paired-model-comparison)' in text
        assert '[Calibration](#calibration)' in text
        assert '[Risk Stratification](#risk-stratification)' in text

    def test_evaluation_toc_omits_unavailable_subsections(self):
        """TOC absent when fewer than 3 subsections are available."""
        section = self._make_section(with_cv=True)
        # Only 2 entries (Methodology + Overview) — below threshold
        text = self._call_outline(section)
        assert '**Contents:**' not in text

    def test_evaluation_toc_appears_with_three_plus_subsections(self):
        """TOC appears when 3+ subsections are available."""
        section = self._make_section(with_cv=True)
        text = self._call_outline(section, disc=True)
        assert '**Contents:**' in text
        assert '[Methodology](#methodology)' in text
        assert '[Overview](#overview)' in text
        assert '[Discrimination](#discrimination)' in text
        assert 'Paired Model Comparison' not in text

    def test_evaluation_no_prediction_tabs(self):
        """generate() returns tabs=None (no prediction tabs)."""
        section = self._make_section(with_cv=True)
        result = section.generate()
        assert result['tabs'] is None

    def test_evaluation_prediction_access_via_consolidated_dropdown(self):
        """Prediction data keys available for consolidated access dropdown."""
        section = self._make_section(with_cv=True)
        result = section.generate()
        # Prediction data is stored in section.data (rendered by consolidated dropdown)
        assert 'train_predictions' in section.data
        assert 'test_predictions' in section.data
        # No standalone prediction snippet in markdown
        assert 'environment-info' not in result['markdown']

    def test_evaluation_prediction_data_still_stored(self):
        """generate() still populates data['train_predictions'] and data['test_predictions']."""
        import pandas as pd
        section = self._make_section(with_cv=True)
        section.generate()
        assert 'train_predictions' in section.data
        assert 'test_predictions' in section.data
        assert isinstance(section.data['train_predictions'], pd.DataFrame)
        assert isinstance(section.data['test_predictions'], pd.DataFrame)

    def test_evaluation_raincloud_uses_learner_plot_metric(self):
        """Bootstrap rainclouds call learner.plot_metric with all available metrics."""
        from unittest.mock import MagicMock, patch
        from mlsurv.reports.sections.evaluation._performance import _PerformanceSubsection
        section = self._make_section(with_bootstrap=True, multi_metric=True)
        perf = _PerformanceSubsection(section)
        md_lines, figures, figure_info = [], [], []

        mock_fig = MagicMock()
        with patch.object(section.learner, 'plot_metric', return_value=mock_fig) as mock_pm:
            perf._add_bootstrap_rainclouds(md_lines, figures, figure_info)

        # Should call learner.plot_metric with style="raincloud", source="bootstrap"
        mock_pm.assert_called_once()
        call_kwargs = mock_pm.call_args
        assert call_kwargs.kwargs.get('style') == 'raincloud'
        assert call_kwargs.kwargs.get('source') == 'bootstrap'
        # Should pass all available metrics via the unified metric= parameter
        assert 'metric' in call_kwargs.kwargs
        # Figure stored as single multi-panel image
        assert any('bootstrap_metrics' in f[0] for f in figures)

    def test_evaluation_bootstrap_methodology_always_visible(self):
        """Bootstrap methodology table is always visible (no toggle button)."""
        from mlsurv.reports.sections.evaluation._methodology import _MethodologySubsection
        section = self._make_section(with_bootstrap=True)
        meth = _MethodologySubsection(section)
        md_lines = []
        meth._add_bootstrap_methodology_details(md_lines)
        text = '\n'.join(md_lines)
        assert 'Bootstrap Iterations' in text
        assert 'CI Method' in text
        assert 'ft-expand-btn' not in text
        assert 'ft-overflow-row' not in text

    def test_evaluation_no_bootstrap_methodology_without_bootstrap(self):
        """No bootstrap methodology when no bootstrap results."""
        from mlsurv.reports.sections.evaluation._methodology import _MethodologySubsection
        section = self._make_section(with_bootstrap=False)
        meth = _MethodologySubsection(section)
        md_lines = []
        meth._add_bootstrap_methodology_details(md_lines)
        assert len(md_lines) == 0

    def test_evaluation_methodology_heading(self):
        """Section includes a Methodology subsection."""
        section = self._make_section(with_cv=True)
        result = section.generate()
        assert '### Methodology' in result['markdown']

    def test_evaluation_overview_heading(self):
        """Section uses 'Overview' heading for the metrics subsection."""
        section = self._make_section(with_cv=True)
        result = section.generate()
        assert '### Overview' in result['markdown']
        assert '### Performance Summary' not in result['markdown']

    def test_evaluation_risk_stratification_heading(self):
        """Section uses 'Risk Stratification' heading instead of 'Survival Stratification'."""
        section = self._make_section(with_cv=True)
        result = section.generate()
        # If stratification is available, heading should be 'Risk Stratification'
        assert 'Survival Stratification' not in result['markdown']

    def test_evaluation_cv_disclaimer(self):
        """CV-vs-test disclaimer appears in performance section."""
        section = self._make_section(with_cv=True)
        result = section.generate()
        assert 'cross-validation performance' in result['markdown']
        assert 'not test set performance alone' in result['markdown']

    def test_evaluation_no_highlight_box(self):
        """Regression: CV-based 'Evaluation Highlights' blockquote stays removed."""
        section = self._make_section(with_cv=True, with_bootstrap=True)
        result = section.generate()
        assert 'Evaluation Highlights' not in result['markdown']
        assert 'Best model:' not in result['markdown']

    def test_evaluation_performance_table_excludes_calibration_diagnostics(self):
        """Performance table shows only discrimination/overall metrics, not calibration."""
        import pandas as pd
        from mlsurv.reports.sections.evaluation._performance import _PerformanceSubsection
        section = self._make_section()
        # Give eval results a concordance DataFrame with all metrics
        for result in section._get_all_models().values():
            result.evaluation.concordance = pd.DataFrame({
                'c_index_test': [0.75],
                'integrated_brier_test': [0.18],
                'cal_slope_test': [0.95],
                'cal_intercept_test': [0.02],
                'gnd_test': [0.45],
            })

        md_lines, figures, figure_info = [], [], []
        _PerformanceSubsection(section).add(md_lines, figures, figure_info)
        md = '\n'.join(md_lines)
        # Discrimination metrics should be present
        assert "Harrell" in md
        # Calibration diagnostics should NOT appear
        assert 'Cal Slope' not in md
        assert 'Cal Int' not in md
        assert 'GND' not in md
        # Check stored data too. A 'Calibrated' provenance column is allowed;
        # only calibration DIAGNOSTIC columns (slope/intercept/GND) must be out.
        perf_df = section.data.get('test_performance')
        if perf_df is not None:
            cols = ' '.join(perf_df.columns)
            assert 'Slope' not in cols
            assert 'Intercept' not in cols
            assert 'GND' not in cols

    def test_evaluation_paired_comparison_ns_marker(self):
        """Non-significant pairs show 'ns' and are excluded from interpretation."""
        from mlsurv.reports.sections.evaluation._paired_comparison import (
            _significance_marker,
        )
        # Non-significant p-value produces "ns"
        assert _significance_marker(0.50) == "ns"
        assert _significance_marker(0.06) == "ns"
        # Significant p-values produce stars
        assert _significance_marker(0.04) == "\\*"
        assert _significance_marker(0.005) == "\\*\\*"
        assert _significance_marker(0.0005) == "\\*\\*\\*"
        # NaN produces empty string
        assert _significance_marker(float('nan')) == ""

    def test_evaluation_paired_comparison_interpretation_ns(self):
        """Interpretation says 'no significant differences' when all pairs are ns."""
        from mlsurv.reports.sections.evaluation._paired_comparison import (
            _PairedComparisonSubsection,
        )
        import pandas as pd

        section = self._make_section(with_cv=True)
        # Provide Kang comparison data where all p-values are non-significant
        section.learner.kang_comparison_result = pd.DataFrame({
            'model_a': ['coxph'],
            'model_b': ['rsf'],
            'delta': [0.02],
            'ci_lower': [-0.05],
            'ci_upper': [0.09],
            'p_value': [0.65],
        })

        subsection = _PairedComparisonSubsection(section)
        md_lines, figures, figure_info = [], [], []
        subsection.add(md_lines, figures, figure_info)
        text = '\n'.join(md_lines)
        # Should say no significant differences
        assert 'No statistically significant differences' in text
        # Should NOT list pairs as significant
        assert 'Statistically significant differences:' not in text

    def test_evaluation_paired_comparison_method_intro(self):
        """Intro paragraph mentions comparison methods."""
        from mlsurv.reports.sections.evaluation._paired_comparison import (
            _PairedComparisonSubsection,
        )
        import pandas as pd

        section = self._make_section(with_cv=True)
        section.learner.kang_comparison_result = pd.DataFrame({
            'model_a': ['coxph'],
            'model_b': ['rsf'],
            'delta': [0.02],
            'ci_lower': [-0.05],
            'ci_upper': [0.09],
            'p_value': [0.65],
        })

        subsection = _PairedComparisonSubsection(section)
        md_lines, figures, figure_info = [], [], []
        subsection.add(md_lines, figures, figure_info)
        text = '\n'.join(md_lines)
        assert 'bootstrap permutation tests' in text
        assert 'Kang z-test' in text


class TestCTDReferenceValues:
    """Antolini's C-index (ctd) should have reference value 0.5."""

    def test_panel_reference_includes_ctd(self):
        from mlsurv.viz._metrics_helpers import _PANEL_REFERENCE_VALUES
        assert _PANEL_REFERENCE_VALUES["ctd"] == 0.5

    def test_metric_reference_ctd(self):
        from mlsurv.viz._metrics_helpers import _metric_reference
        ref_val, ref_label = _metric_reference("ctd")
        assert ref_val == 0.5
        assert ref_label == "Random Chance"


class TestBuildSectionOutline:
    """Tests for BaseSection._build_section_outline."""

    def test_basic_outline(self):
        """Numbered outline with Contents heading."""
        from mlsurv.reports.sections.base import BaseSection

        entries = [('A', 'a'), ('B', 'b'), ('C', 'c')]
        lines = BaseSection._build_section_outline(entries)
        assert lines[0] == '**Contents:**'
        assert '1. [A](#a)' in lines
        assert '2. [B](#b)' in lines
        assert '3. [C](#c)' in lines

    def test_below_threshold_returns_empty(self):
        """Fewer than min_entries returns empty list."""
        from mlsurv.reports.sections.base import BaseSection

        lines = BaseSection._build_section_outline([('A', 'a'), ('B', 'b')])
        assert lines == []

    def test_custom_min_entries(self):
        """Custom min_entries threshold is respected."""
        from mlsurv.reports.sections.base import BaseSection

        lines = BaseSection._build_section_outline([('A', 'a')], min_entries=1)
        assert '**Contents:**' in lines
        assert '1. [A](#a)' in lines


class TestMetadataBar:
    """Tests for report metadata header bar."""

    def test_metadata_bar_in_html(self):
        """HTML report contains metadata bar with version and dimensions."""
        from mlsurv.reports.renderers.html import HTMLRenderer

        meta = {
            'version': '1.2.3',
            'timestamp': '2026-01-01 00:00:00 UTC',
            'n_samples': 1000,
            'n_features': 20,
            'n_events': 450,
        }
        bar = HTMLRenderer._render_metadata_bar(meta)
        assert 'report-metadata' in bar
        assert 'v1.2.3' in bar
        assert '2026-01-01' in bar
        assert '1,000 samples' in bar
        assert '20 features' in bar
        assert '450 events' in bar

    def test_metadata_bar_escapes_html(self):
        """Metadata values are HTML-escaped."""
        from mlsurv.reports.renderers.html import HTMLRenderer

        meta = {'version': '<script>alert(1)</script>'}
        bar = HTMLRenderer._render_metadata_bar(meta)
        assert '<script>' not in bar
        assert '&lt;script&gt;' in bar

    def test_metadata_in_markdown(self):
        """Markdown renderer includes metadata line."""
        from mlsurv.reports.renderers.markdown import MarkdownRenderer

        renderer = MarkdownRenderer.__new__(MarkdownRenderer)
        renderer.title = "Test"
        renderer.output_path = __import__('pathlib').Path('/tmp/test.md')
        renderer.images_dir = renderer.output_path.parent / 'images'

        meta = {'version': '1.0.0', 'n_samples': 500}
        content = renderer.render([], metadata=meta)
        assert 'v1.0.0' in content
        assert '500 samples' in content


class TestCohortSummaryGuard:
    """Test that _add_cohort_summary_table handles exceptions gracefully."""

    def test_cohort_summary_recovers_from_error(self):
        """_add_cohort_summary_table catches DATA_EXCEPTIONS and logs warning."""
        from unittest.mock import MagicMock, PropertyMock

        from mlsurv.reports.sections.validation import ValidationSection

        # Create a mock validation result where event_rate raises AttributeError
        mock_val = MagicMock()
        mock_val.n_samples = 100
        mock_val.n_events = 30
        type(mock_val).event_rate = PropertyMock(
            side_effect=AttributeError("no event_rate")
        )

        # Build a minimal ValidationSection with mocked internals
        section = ValidationSection.__new__(ValidationSection)
        section.validation_results = {'test_cohort': mock_val}
        section.data = {}
        section._log_recoverable_exception = MagicMock()

        md_lines = []
        # Should NOT raise — the guard catches AttributeError
        section._add_cohort_summary_table(md_lines)

        # Verify _log_recoverable_exception was called with the right context
        section._log_recoverable_exception.assert_called_once()
        context = section._log_recoverable_exception.call_args[0][1]
        assert "cohort summary" in context


class TestValidationCalibrationModelScope:
    """Test that calibration plots only include validated models."""

    @staticmethod
    def _make_model_result():
        """Create a mock ModelResult whose model has predict_survival_function."""
        from unittest.mock import MagicMock

        result = MagicMock()
        result.model.predict_survival_function = MagicMock()
        return result

    @staticmethod
    def _make_subsection(all_models):
        """Create a _CalibrationSubsection with the given all_models dict."""
        from unittest.mock import MagicMock

        from mlsurv.reports.sections.validation._calibration import (
            _CalibrationSubsection,
        )

        mock_section = MagicMock()
        mock_section._get_all_models.return_value = all_models
        return _CalibrationSubsection(mock_section)

    def test_get_calibration_models_scopes_to_validated(self):
        """_get_calibration_models returns only models present in val_result.concordance."""
        from unittest.mock import MagicMock

        import pandas as pd

        result_coxph = self._make_model_result()
        result_rsf = self._make_model_result()
        result_gbsa = self._make_model_result()

        sub = self._make_subsection({
            'coxph': result_coxph, 'rsf': result_rsf, 'gbsa': result_gbsa,
        })

        # val_result.concordance has only 'rsf'
        mock_val_result = MagicMock()
        mock_val_result.concordance = pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_test': [0.72],
        })

        md_lines = []
        models = sub._get_calibration_models(
            mock_val_result, MagicMock(), MagicMock(), md_lines,
        )

        assert models is not None
        assert list(models.keys()) == ['rsf']
        assert models['rsf'] is result_rsf.model
        assert md_lines == []

    def test_get_calibration_models_returns_all_when_all_validated(self):
        """When all models are validated, all should appear in calibration."""
        from unittest.mock import MagicMock

        import pandas as pd

        sub = self._make_subsection({
            'coxph': self._make_model_result(),
            'rsf': self._make_model_result(),
        })

        mock_val_result = MagicMock()
        mock_val_result.concordance = pd.DataFrame({
            'model_type': ['coxph', 'rsf'],
            'c_index_test': [0.70, 0.72],
        })

        md_lines = []
        models = sub._get_calibration_models(
            mock_val_result, MagicMock(), MagicMock(), md_lines,
        )

        assert models is not None
        assert set(models.keys()) == {'coxph', 'rsf'}

    def test_get_calibration_models_none_when_no_concordance(self):
        """Returns None with message when concordance is empty."""
        from unittest.mock import MagicMock

        import pandas as pd

        sub = self._make_subsection({})

        mock_val_result = MagicMock()
        mock_val_result.concordance = pd.DataFrame()

        md_lines = []
        result = sub._get_calibration_models(
            mock_val_result, MagicMock(), MagicMock(), md_lines,
        )

        assert result is None
        assert any("validation metrics" in line for line in md_lines)

    def test_get_calibration_models_none_when_concordance_is_none(self):
        """Returns None with message when concordance is None (not just empty)."""
        from unittest.mock import MagicMock

        sub = self._make_subsection({})

        mock_val_result = MagicMock()
        mock_val_result.concordance = None

        md_lines = []
        result = sub._get_calibration_models(
            mock_val_result, MagicMock(), MagicMock(), md_lines,
        )

        assert result is None
        assert any("validation metrics" in line for line in md_lines)

    def test_get_calibration_models_none_when_model_lacks_survival_function(self):
        """Returns None when validated model doesn't support predict_survival_function."""
        from unittest.mock import MagicMock

        import pandas as pd

        # Create model without predict_survival_function
        result_fssvm = MagicMock(spec=[])
        result_fssvm.model = MagicMock(spec=[])  # no predict_survival_function

        sub = self._make_subsection({'fssvm': result_fssvm})

        mock_val_result = MagicMock()
        mock_val_result.concordance = pd.DataFrame({
            'model_type': ['fssvm'],
            'c_index_test': [0.65],
        })

        md_lines = []
        result = sub._get_calibration_models(
            mock_val_result, MagicMock(), MagicMock(), md_lines,
        )

        assert result is None
        assert any("survival function prediction" in line for line in md_lines)


# ── Data & Preprocessing table improvements ──────────────────────────────


class TestCategoricalValuePrefix:
    """Tests that categorical values in Table 1 get value= prefix."""

    def test_categorical_rows_have_value_prefix(self):
        """Categorical feature values use value= prefix in Statistic column."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        result = section.generate()

        table1 = section.data['table1']
        # 'stage' is categorical (values: I, II, III)
        stage_rows = table1[table1['Variable'] == 'stage']
        for _, row in stage_rows.iterrows():
            stat = row['Statistic']
            if not stat.startswith('Other'):
                assert stat.startswith('value='), (
                    f"Categorical stat should start with 'value=', got '{stat}'"
                )
        # Prefix survives rendering to markdown
        assert 'value=' in result['markdown']

    def test_continuous_rows_unchanged(self):
        """Continuous features still show Mean (SD) and Median, no value= prefix."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        section.generate()

        table1 = section.data['table1']
        age_rows = table1[table1['Variable'] == 'age']
        stats = age_rows['Statistic'].unique().tolist()
        assert 'Mean (SD)' in stats
        assert any('Median' in s for s in stats)
        assert not any(s.startswith('value=') for s in stats)


class TestResolvedKDisplay:
    """Tests that pipeline config shows resolved k for auto selector."""

    def test_k_auto_shows_resolved(self):
        """k='auto' with effective_k_=15 renders as k='auto' -> 15."""
        from unittest.mock import MagicMock
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        selector = MagicMock()
        selector.k = 'auto'
        selector.effective_k_ = 15
        type(selector).__name__ = 'TopKCoxSelector'
        learner._state.feature_selector = selector

        section = DataPreprocessingSection(learner)
        formatted = section._format_untuned_params(selector, 'selector')
        assert formatted == "TopKCoxSelector(k='auto' \u2192 15)"

    def test_k_integer_unchanged(self):
        """k=10 (not auto) renders as k=10 without arrow."""
        from unittest.mock import MagicMock
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        selector = MagicMock(spec=['k'])
        selector.k = 10
        type(selector).__name__ = 'TopKCoxSelector'

        section = DataPreprocessingSection(learner)
        formatted = section._format_untuned_params(selector, 'selector')
        assert formatted == "TopKCoxSelector(k=10)"

    def test_k_auto_no_effective_k(self):
        """k='auto' without effective_k_ (unfitted) renders as k=auto."""
        from unittest.mock import MagicMock
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)

        selector = MagicMock(spec=['k'])
        selector.k = 'auto'
        type(selector).__name__ = 'TopKCoxSelector'

        section = DataPreprocessingSection(learner)
        formatted = section._format_untuned_params(selector, 'selector')
        assert formatted == "TopKCoxSelector(k=auto)"

    def test_format_untuned_params_model_requires_result(self):
        """component_type='model' without result= must raise ValueError, not
        silently skip the model-params formatting path. Pins the invariant
        guard at the boundary."""
        from unittest.mock import MagicMock
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner()
        _add_mock_model(learner, 'coxph', 0.72)
        section = DataPreprocessingSection(learner)
        with pytest.raises(ValueError, match="requires result="):
            section._format_untuned_params(MagicMock(), 'model')


class TestSelectorKRouting:
    """Report display routes selector k only via feature_selector__k.

    A bare 'k' in best_params is the model's own hyperparameter and must NOT be
    rendered as the feature selector's k.
    """

    def test_tuned_selector_str_uses_prefixed_key(self):
        from unittest.mock import MagicMock
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner()
        result = _add_mock_model(learner, 'rsf', 0.72, is_tuned=True)
        result.optuna_study = MagicMock()
        result.optuna_study.best_params = {'feature_selector__k': 6, 'model__n_estimators': 200}

        section = DataPreprocessingSection(learner)
        assert section._get_tuned_selector_str() == "TopKCoxSelector(k=6)"

    def test_tuned_selector_str_ignores_bare_model_k(self):
        from unittest.mock import MagicMock
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner()
        result = _add_mock_model(learner, 'kparam_cox', 0.70, is_tuned=True)
        result.optuna_study = MagicMock()
        # Bare 'k' is the model's own hyperparameter, not the selector's.
        result.optuna_study.best_params = {'k': 3}

        section = DataPreprocessingSection(learner)
        assert section._get_tuned_selector_str() is None

    def test_parse_param_name_bare_k_is_model(self):
        from mlsurv.reports.sections.model_development import ModelDevelopmentSection

        learner = _make_mock_learner()
        section = ModelDevelopmentSection(learner)
        assert section._parse_param_name('k') == ('k', 'model')
        assert section._parse_param_name('feature_selector__k') == ('k', 'selector')
        assert section._parse_param_name('model__alpha') == ('alpha', 'model')


def _make_mock_learner_many_missing(n_features=15):
    """Create a mock learner where every feature has missing data."""
    from unittest.mock import MagicMock
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(42)
    n_train, n_test = 80, 20
    X_train_data = {f'feat_{i:02d}': rng.normal(0, 1, n_train) for i in range(n_features)}
    X_test_data = {f'feat_{i:02d}': rng.normal(0, 1, n_test) for i in range(n_features)}

    # Inject NaN into every feature
    for i in range(n_features):
        col = f'feat_{i:02d}'
        X_train_data[col][i % n_train] = np.nan
        X_test_data[col][i % n_test] = np.nan

    learner = MagicMock()
    learner._state.is_setup = True
    learner._state.imputer = None
    learner._state.scaler = None
    learner._state.feature_selector = None
    learner._state.feature_groups = None
    learner._state.group_colors = None
    learner._state.presplit = False
    learner._state.subpopulations_train = None
    learner._state.subpopulations_test = None
    learner._state.stratify_array = None
    # Public accessors the sections read directly — a bare MagicMock
    # auto-attr is truthy, so pin the "nothing computed" values explicitly.
    learner.is_setup = True
    learner.validation_results = {}
    learner.subpopulation_results = {}
    learner.benchmark_result = None
    learner.cv_strategy = 'stratifiedkfold'
    learner.cv_folds = 5
    learner.cv_repeats = 1
    learner.scoring = 'harrell'

    learner.X_train = pd.DataFrame(X_train_data)
    learner.X_test = pd.DataFrame(X_test_data)
    learner.X = pd.concat([learner.X_train, learner.X_test], ignore_index=True)
    y_train = make_survival_array(
        rng.binomial(1, 0.6, n_train), rng.exponential(100, n_train)
    )
    y_test = make_survival_array(
        rng.binomial(1, 0.6, n_test), rng.exponential(100, n_test)
    )
    learner.y_train = y_train
    learner.y_test = y_test

    return learner


class TestMissingDataOverflow:
    """Tests for missing data table collapsible overflow."""

    def test_overflow_details_toggle(self):
        """With >10 features missing, overflow renders with a button toggle."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner_many_missing(n_features=15)
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        result = section.generate()
        md = result['markdown']

        assert 'ft-expand-btn' in md
        assert 'ft-overflow-row' in md
        assert 'Show all 15 features' in md
        assert '<!-- MLSURV_RAW_HTML -->' in md
        assert 'feat_14' in md
        assert 'feat_10' in md

    def test_no_overflow_when_few_missing(self):
        """With <=10 features missing, no expand button for missing data."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner_many_missing(n_features=5)
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        result = section.generate()
        md = result['markdown']

        missing_section = md.split('### Missing Data')[1].split('###')[0]
        assert 'ft-expand-btn' not in missing_section

    def test_no_overflow_at_boundary(self):
        """With exactly 10 features missing, no overflow appears."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner_many_missing(n_features=10)
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        result = section.generate()
        md = result['markdown']

        missing_section = md.split('### Missing Data')[1].split('###')[0]
        assert 'ft-expand-btn' not in missing_section

    def test_minimal_overflow(self):
        """With 11 features missing, exactly 1 overflows."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner_many_missing(n_features=11)
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        result = section.generate()
        md = result['markdown']

        assert 'ft-expand-btn' in md
        assert 'Show all 11 features' in md

    def test_stores_all_missing_features(self):
        """self.data['missing_data'] contains ALL features, not just visible."""
        from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection

        learner = _make_mock_learner_many_missing(n_features=15)
        _add_mock_model(learner, 'coxph', 0.72)

        section = DataPreprocessingSection(learner)
        section.generate()

        missing = section.data['missing_data']
        assert len(missing) == 15


# ------------------------------------------------------------------
# Validation Table 1 — interactive HTML
# ------------------------------------------------------------------


def _make_mock_validation_section(n_features=5, n_cohorts=2, cohort_names=None):
    """Build a minimal ValidationSection for testing Table 1 HTML rendering.

    Creates mock dev data (X_train, X_test) and mock validation cohorts
    with ``n_features`` numeric columns, then returns the section ready
    for ``add_distribution_comparison(section, md_lines)``.
    """
    from unittest.mock import MagicMock

    import numpy as np
    import pandas as pd

    from mlsurv.reports.sections.validation import ValidationSection

    rng = np.random.RandomState(42)
    n_dev = 100
    cols = [f"feat_{i}" for i in range(n_features)]
    X_train = pd.DataFrame(rng.randn(n_dev // 2, n_features), columns=cols)
    X_test = pd.DataFrame(rng.randn(n_dev // 2, n_features), columns=cols)

    section = ValidationSection.__new__(ValidationSection)
    section.data = {}
    section.learner = MagicMock()
    section.learner.X_train = X_train
    section.learner.X_test = X_test
    section._log_recoverable_exception = MagicMock()

    if cohort_names is None:
        cohort_names = [f"Cohort_{i}" for i in range(1, n_cohorts + 1)]

    validation_results = {}
    for name in cohort_names:
        mock_val = MagicMock()
        n_val = 40 + len(name)  # vary sizes slightly
        mock_val.X_val = pd.DataFrame(rng.randn(n_val, n_features), columns=cols)
        validation_results[name] = mock_val

    section.validation_results = validation_results
    return section


class TestValidationTable1Interactive:
    """Tests for validation Table 1 scroll, toggle, and cohort filters."""

    def test_raw_html_sentinels(self):
        """Output contains MLSURV_RAW_HTML sentinel markers."""
        from mlsurv.reports.sections.validation._table1 import add_distribution_comparison

        section = _make_mock_validation_section(n_features=5)
        md_lines = []
        add_distribution_comparison(section, md_lines)
        md = '\n'.join(md_lines)

        assert '<!-- MLSURV_RAW_HTML -->' in md
        assert '<!-- /MLSURV_RAW_HTML -->' in md

    def test_horizontal_scroll_wrapper(self):
        """Table is wrapped in a feature-table-scroll container."""
        from mlsurv.reports.sections.validation._table1 import add_distribution_comparison

        section = _make_mock_validation_section(n_features=5)
        md_lines = []
        add_distribution_comparison(section, md_lines)
        md = '\n'.join(md_lines)

        assert 'feature-table-scroll' in md

    def test_feature_analysis_table_class(self):
        """Table uses feature-analysis-table class for interactive CSS/JS."""
        from mlsurv.reports.sections.validation._table1 import add_distribution_comparison

        section = _make_mock_validation_section(n_features=5)
        md_lines = []
        add_distribution_comparison(section, md_lines)
        md = '\n'.join(md_lines)

        assert 'feature-analysis-table' in md

    def test_expand_collapse_with_overflow(self):
        """With > TABLE1_MAX_FEATURES, overflow rows and expand button appear."""
        from mlsurv.reports._constants import TABLE1_MAX_FEATURES
        from mlsurv.reports.sections.validation._table1 import add_distribution_comparison

        n_features = TABLE1_MAX_FEATURES + 5
        section = _make_mock_validation_section(n_features=n_features)
        md_lines = []
        add_distribution_comparison(section, md_lines)
        md = '\n'.join(md_lines)

        assert 'ft-overflow-row' in md
        assert 'ft-expand-btn' in md
        assert f'Show all {n_features} features' in md
        # ALL features appear in the HTML (not just visible ones)
        for i in range(n_features):
            assert f'feat_{i}' in md

    def test_no_expand_without_overflow(self):
        """With <= TABLE1_MAX_FEATURES, no overflow elements appear."""
        from mlsurv.reports.sections.validation._table1 import add_distribution_comparison

        section = _make_mock_validation_section(n_features=5)
        md_lines = []
        add_distribution_comparison(section, md_lines)
        md = '\n'.join(md_lines)

        assert 'ft-overflow-row' not in md
        assert 'ft-expand-btn' not in md
        # Table container and scroll still present
        assert 'feature-table-container' in md
        assert 'feature-table-scroll' in md

    def test_boundary_exact_max_features(self):
        """With exactly TABLE1_MAX_FEATURES, no overflow button appears."""
        from mlsurv.reports._constants import TABLE1_MAX_FEATURES
        from mlsurv.reports.sections.validation._table1 import add_distribution_comparison

        section = _make_mock_validation_section(n_features=TABLE1_MAX_FEATURES)
        md_lines = []
        add_distribution_comparison(section, md_lines)
        md = '\n'.join(md_lines)

        assert 'ft-expand-btn' not in md
        assert 'ft-overflow-row' not in md

    def test_cohort_filter_checkboxes(self):
        """Cohort filter checkboxes with data-ft-model attributes are present."""
        from mlsurv.reports.sections.validation._table1 import add_distribution_comparison

        section = _make_mock_validation_section(
            n_features=5, cohort_names=["TCGA", "METABRIC"],
        )
        md_lines = []
        add_distribution_comparison(section, md_lines)
        md = '\n'.join(md_lines)

        assert 'feature-table-controls' in md
        assert 'ft-control-label' in md
        # Development + 2 validation cohorts = 3 checkboxes
        assert 'data-ft-model="development"' in md
        assert 'data-ft-model="tcga"' in md
        assert 'data-ft-model="metabric"' in md
        # data-ft-model on cells too
        assert md.count('data-ft-model="development"') > 1  # th + td cells

    def test_sticky_variable_column(self):
        """Variable column uses sticky-col class on th and td."""
        from mlsurv.reports.sections.validation._table1 import add_distribution_comparison

        section = _make_mock_validation_section(n_features=3)
        md_lines = []
        add_distribution_comparison(section, md_lines)
        md = '\n'.join(md_lines)

        assert '<th class="sticky-col">Variable</th>' in md
        assert '<td class="sticky-col">' in md

    def test_tidy_data_stores_all_features(self):
        """section.data['table1'] contains ALL features, not just visible ones."""
        from mlsurv.reports._constants import TABLE1_MAX_FEATURES
        from mlsurv.reports.sections.validation._table1 import add_distribution_comparison

        n_features = TABLE1_MAX_FEATURES + 5
        section = _make_mock_validation_section(n_features=n_features)
        md_lines = []
        add_distribution_comparison(section, md_lines)

        table1 = section.data['table1']
        features_in_data = table1['Variable'].unique()
        assert len(features_in_data) == n_features

    def test_html_escaping(self):
        """HTML-sensitive cohort names are properly escaped."""
        from mlsurv.reports.sections.validation._table1 import add_distribution_comparison

        section = _make_mock_validation_section(
            n_features=3, cohort_names=["Cohort <A>"],
        )
        md_lines = []
        add_distribution_comparison(section, md_lines)
        md = '\n'.join(md_lines)

        # Escaped form should appear, not raw HTML
        assert 'Cohort &lt;A&gt;' in md
        assert 'Cohort <A>' not in md.replace('Cohort &lt;A&gt;', '')


class TestReportPolish:
    """Tests for report polish: exec summary, tuning, raincloud, validation heatmap."""

    def test_display_name_idempotent_on_resolved_tuned(self):
        """get_model_display_name is idempotent on already-resolved '(tuned)' strings."""
        from mlsurv.viz._palettes import get_model_display_name

        # An already-resolved label passes through unchanged.
        assert get_model_display_name('RSF (tuned)') == 'RSF (tuned)'
        # Uppercase-artifact form collapses to canonical form.
        assert get_model_display_name('RSF (TUNED)') == 'RSF (tuned)'
        # Non-tuned display names pass through unchanged.
        assert get_model_display_name('RSF') == 'RSF'

    def test_raincloud_labels_use_display_names(self):
        """_prepare_model_styling resolves storage keys through get_model_display_name."""
        import pandas as pd

        from mlsurv.viz._metrics_helpers import _prepare_model_styling

        df = pd.DataFrame({
            'model': ['rsf_tuned', 'coxph', 'RSF (TUNED)'],
            'value': [0.7, 0.65, 0.71],
        })
        models, display_names, _colors = _prepare_model_styling(
            df, 'model', 'value', sort='alpha', palette=None,
        )
        # Every tuned label uses canonical '(tuned)' casing (never '(TUNED)').
        for label in display_names:
            if '(' in label:
                assert '(tuned)' in label, f"non-canonical tuned label: {label!r}"
                assert '(TUNED)' not in label

    @pytest.mark.integration
    def test_summary_quality_paragraph_names_best_model(self, small_data):
        """Multi-model summary quality paragraph names the CV-best model explicitly."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.summary import SummarySection
        from mlsurv.viz._palettes import get_model_display_name

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        # Model identity is incidental: the paragraph consumes only the
        # explicitly passed bootstrap/eval results, and the best model is
        # CV-sorted. Two fast linear models keep the multi-model shape.
        learner.run(['coxph', 'coxnet'], tune=False, evaluate=False,
                    verbose=False)
        learner.evaluate()
        learner.bootstrap(n_bootstrap=2)

        boot_results = {
            name: result.bootstrap
            for name, result in learner.models.items()
            if result.bootstrap is not None
        }
        section = SummarySection(
            learner,
            bootstrap_results=boot_results,
            eval_result=learner.evaluation_result,
        )

        best_name = section._select_best_model_name(learner.models)
        assert best_name is not None
        best_display = get_model_display_name(best_name)

        md_lines = []
        section._add_quality_paragraph(md_lines, learner.models)
        md = '\n'.join(md_lines)
        # The quality paragraph should reference the best model (via the
        # bootstrap test-set sentence) and its risk-stratification result,
        # which are both implicitly scoped to best_name.
        assert best_display in md

    @pytest.mark.integration
    def test_summary_tuning_paragraph_names_only_best_model(self, small_data):
        """Tuning paragraph lists influential params only for the CV-best tuned model."""
        from mlsurv import SurvivalLearner
        from mlsurv.reports.sections.summary import SummarySection
        from mlsurv.viz._palettes import get_model_display_name

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        # run(tune=False) so the explicit tune() calls below are the real
        # tuning passes (a bare run() would auto-tune both and turn them
        # into "already tuned" no-ops). The tiny rsf search space keeps the
        # forest cheap; the narrative only needs two tuned models.
        learner.run(['coxnet', 'rsf'], tune=False, evaluate=False,
                    verbose=False)
        learner.tune('coxnet', n_trials=3, evaluate=False, bootstrap=False)
        learner.tune('rsf', n_trials=3, evaluate=False, bootstrap=False,
                     search_spaces={'n_estimators': (10, 20),
                                    'max_depth': (3, 5)})

        section = SummarySection(learner)
        tuned_models = section._get_tuned_models()
        cv_desc = section._build_cv_description()
        narrative = section._generate_tuning_narrative(
            learner.models, tuned_models, cv_desc,
        )

        best_name = section._select_best_model_name(learner.models)
        best_display = get_model_display_name(best_name) if best_name else ''
        assert narrative is not None
        if 'Most influential parameters' in narrative:
            # Should name exactly one model (the best), not a per-model dump.
            assert f"Most influential parameters for *{best_display}*" in narrative
            # No other tuned model should be mentioned in the params clause.
            params_clause = narrative.split('Most influential parameters')[1]
            other_tuned = [n for n in tuned_models if n != best_name]
            for name in other_tuned:
                assert get_model_display_name(name) not in params_clause

    def test_validation_single_cohort_falls_back_to_table(self):
        """With a single validation cohort, the heatmap is suppressed; a metrics table is emitted."""
        from unittest.mock import MagicMock
        import pandas as pd

        from mlsurv.reports.sections.validation import ValidationSection

        section = ValidationSection.__new__(ValidationSection)
        section.data = {}
        section._log_recoverable_exception = MagicMock()

        val_result = MagicMock()
        val_result.concordance = pd.DataFrame([
            {
                'model_type': 'coxph',
                'c_index_test': 0.70,
                'integrated_brier_test': 0.18,
            },
            {
                'model_type': 'rsf',
                'c_index_test': 0.72,
                'integrated_brier_test': 0.17,
            },
        ])
        section.validation_results = {'CohortA': val_result}

        md_lines, figures, figure_info = [], [], []
        section._add_performance_heatmap(md_lines, figures, figure_info)
        md = '\n'.join(md_lines)

        assert not any('validation_metrics_heatmap' in f[0] for f in figures)
        assert not any(
            fi.figure_id == 'validation_metrics_heatmap' for fi in figure_info
        )
        # A table representation should be present in markdown.
        assert '|' in md
        assert 'metrics' in section.data


class TestSummaryStratificationRows:
    """Direct branch coverage for _summary_stratification_rows.

    The helper reads the two labels setup() records (``stratify_spec`` for the
    split, ``cv_stratify_spec`` for the folds), so a SimpleNamespace stub is
    enough.
    """

    @staticmethod
    def _learner(presplit=False, stratify_spec="event", cv_stratify_spec="event"):
        from types import SimpleNamespace
        state = SimpleNamespace(
            presplit=presplit,
            stratify_spec=stratify_spec,
            cv_stratify_spec=cv_stratify_spec,
        )
        return SimpleNamespace(_state=state, cv_strategy='stratifiedkfold')

    def test_default_split_row_only(self):
        """Folds matching the split add nothing, so the CV row is left out."""
        from mlsurv.reports.sections.data_preprocessing import (
            _summary_stratification_rows,
        )
        rows = _summary_stratification_rows(self._learner())
        assert rows == [["Split stratification", "Event status"]]

    def test_presplit_skips_split_row_but_keeps_the_cv_row(self):
        """Pre-split data has no split to describe, but the folds still stratify.

        The CV row is dropped only when it would restate the split row. There is
        no split row here, so the CV row is the only place the reader learns how
        the folds were built.
        """
        from mlsurv.reports.sections.data_preprocessing import (
            _summary_stratification_rows,
        )
        rows = _summary_stratification_rows(self._learner(presplit=True))
        assert rows == [["CV stratification", "Event status"]]

    def test_cv_row_when_folds_differ_from_split(self):
        from mlsurv.reports.sections.data_preprocessing import (
            _summary_stratification_rows,
        )
        rows = _summary_stratification_rows(self._learner(
            stratify_spec="event",
            cv_stratify_spec="event × subpopulation(race)",
        ))
        assert ["Split stratification", "Event status"] in rows
        assert ["CV stratification", "Event × subpopulation(race)"] in rows

    def test_list_stratify_label(self):
        """A stratify= list is reported as the joined variable names."""
        from mlsurv.reports.sections.data_preprocessing import (
            _summary_stratification_rows,
        )
        rows = _summary_stratification_rows(self._learner(
            stratify_spec="event × stage", cv_stratify_spec="event × stage",
        ))
        assert rows == [["Split stratification", "Event × stage"]]

    def test_column_stratify_label_not_reported_as_event(self):
        """A named column must not be reported as event stratification."""
        from mlsurv.reports.sections.data_preprocessing import (
            _summary_stratification_rows,
        )
        rows = _summary_stratification_rows(self._learner(
            stratify_spec="stage", cv_stratify_spec="stage",
        ))
        assert rows == [["Split stratification", "stage"]]

    def test_non_stratified_cv_skips_cv_row(self):
        from mlsurv.reports.sections.data_preprocessing import (
            _summary_stratification_rows,
        )
        rows = _summary_stratification_rows(self._learner(cv_stratify_spec=None))
        assert not any(r[0] == "CV stratification" for r in rows)
