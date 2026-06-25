"""Tests for ManuscriptRenderer — pandoc-compatible markdown output."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlsurv._warnings import SuggestionWarning
from mlsurv.reports._figure_export import FigureExportTimeout
from mlsurv.reports.renderers.manuscript import (
    ManuscriptRenderer,
    _html_table_to_markdown,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def renderer(tmp_path):
    return ManuscriptRenderer(str(tmp_path / "report.md"), title="Test Report")


@pytest.fixture
def metadata():
    return {
        "version": "0.42.0",
        "timestamp": "2026-03-09 12:00:00 UTC",
        "n_samples": 1903,
        "n_features": 9,
        "n_events": 801,
    }


@pytest.fixture
def sections_basic():
    """Minimal sections with no HTML artifacts."""
    return [
        {
            "title": "Executive Summary",
            "markdown": "This is a summary of the analysis.",
            "figures": [],
            "figure_info": [],
            "tabs": None,
        },
        {
            "title": "Model Evaluation",
            "markdown": "Models were evaluated using C-index.",
            "figures": [],
            "figure_info": [],
            "tabs": None,
        },
    ]


@pytest.fixture
def sections_with_html():
    """Sections containing HTML artifacts that must be stripped."""
    return [
        {
            "title": "Data Section",
            "markdown": (
                "<!-- MLSURV_RAW_HTML -->\n"
                '<div class="mlsurv-callout-warning">\n'
                "&#9888; Some features have high missingness.\n"
                "</div>\n"
                "<!-- /MLSURV_RAW_HTML -->\n"
                "\n"
                '<details class="environment-info">\n'
                "<summary>Full details (click to expand)</summary>\n"
                "\n"
                "Some expanded content here.\n"
                "\n"
                "</details>\n"
                "\n"
                '<div class="figure-placeholder" data-figure="missing_heatmap.png"></div>\n'
                "\n"
                '<div class="prediction-table-scroll" data-visible-rows="20">\n'
                "| Col A | Col B |\n"
                "|-------|-------|\n"
                "| 1     | 2     |\n"
                "</div>\n"
                "\n"
                "<button class=\"ft-expand-btn\">Show all</button>\n"
                "\n"
                "Normal paragraph with <strong>bold text</strong> and <br> line break.\n"
                "\n"
                "<details>\n"
                "<summary>BibTeX</summary>\n"
                "\n"
                "```bibtex\n"
                "@article{foo, title={bar}}\n"
                "```\n"
                "\n"
                "</details>"
            ),
            "figures": [],
            "figure_info": [],
            "tabs": [
                {
                    "title": "Train Results",
                    "markdown": "Training data <strong>summary</strong>.",
                }
            ],
        },
    ]


@pytest.fixture
def sections_with_figures():
    """Section with image references for cross-ref testing."""
    return [
        {
            "title": "Feature Analysis",
            "markdown": (
                "![Correlation Heatmap](images/correlation_heatmap.png)\n"
                "\n"
                "![](images/importance_bar.png)\n"
            ),
            "figures": [],
            "figure_info": [],
            "tabs": None,
        },
    ]


@pytest.fixture
def sections_with_html_table():
    """Section with an HTML table inside RAW_HTML markers."""
    return [
        {
            "title": "PH Diagnostics",
            "markdown": (
                "<!-- MLSURV_RAW_HTML -->\n"
                "<table>\n"
                "<tr><th>Feature</th><th>p-value</th></tr>\n"
                "<tr><td>age</td><td>0.032</td></tr>\n"
                "<tr><td>grade</td><td>0.891</td></tr>\n"
                "</table>\n"
                "<!-- /MLSURV_RAW_HTML -->"
            ),
            "figures": [],
            "figure_info": [],
            "tabs": None,
        },
    ]


# ---------------------------------------------------------------------------
# YAML front matter
# ---------------------------------------------------------------------------

class TestYamlFrontMatter:
    def test_yaml_present(self, renderer, sections_basic, metadata):
        content = renderer.render(sections_basic, metadata=metadata)
        assert content.startswith("---\n")
        assert 'title: "Test Report"' in content

    def test_yaml_date(self, renderer, sections_basic, metadata):
        content = renderer.render(sections_basic, metadata=metadata)
        assert 'date: "2026-03-09"' in content

    def test_yaml_abstract(self, renderer, sections_basic, metadata):
        content = renderer.render(sections_basic, metadata=metadata)
        assert "abstract:" in content
        assert "1,903 samples" in content

    def test_yaml_without_metadata(self, renderer, sections_basic):
        content = renderer.render(sections_basic, metadata=None)
        assert content.startswith("---\n")
        assert 'title: "Test Report"' in content


# ---------------------------------------------------------------------------
# Section structure
# ---------------------------------------------------------------------------

class TestSectionStructure:
    def test_sections_have_headings(self, renderer, sections_basic, metadata):
        content = renderer.render(sections_basic, metadata=metadata)
        assert "## Executive Summary" in content
        assert "## Model Evaluation" in content

    def test_page_breaks(self, renderer, sections_basic, metadata):
        content = renderer.render(sections_basic, metadata=metadata)
        assert content.count("\\newpage") == len(sections_basic)

    def test_tabs_flattened(self, renderer, sections_with_html, metadata):
        content = renderer.render(sections_with_html, metadata=metadata)
        assert "### Train Results" in content


# ---------------------------------------------------------------------------
# HTML artifact stripping
# ---------------------------------------------------------------------------

class TestHtmlStripping:
    @pytest.mark.parametrize("open_tag, close_tag", [
        ("<details", "</details>"),
        ("<div", "</div>"),
        ("<button", "</button>"),
    ])
    def test_no_html_tags(self, renderer, sections_with_html, metadata, open_tag, close_tag):
        content = renderer.render(sections_with_html, metadata=metadata)
        assert open_tag not in content
        assert close_tag not in content

    def test_no_raw_html_markers(self, renderer, sections_with_html, metadata):
        content = renderer.render(sections_with_html, metadata=metadata)
        assert "MLSURV_RAW_HTML" not in content

    def test_callout_to_blockquote(self, renderer, sections_with_html, metadata):
        content = renderer.render(sections_with_html, metadata=metadata)
        assert "> **Warning:**" in content

    def test_figure_placeholder_converted(self, renderer, sections_with_html, metadata):
        content = renderer.render(sections_with_html, metadata=metadata)
        # data-figure carries a bare basename; the ref must point at images/
        # (where save_figures writes the PNG), not the report root.
        assert "![](images/missing_heatmap.png)" in content
        assert "![](missing_heatmap.png)" not in content

    def test_label_input_toggles_stripped(self, renderer, metadata):
        """Interactive feature-table toggles must not leak into markdown."""
        sections = [
            {
                "title": "Feature Analysis",
                "markdown": (
                    'Models:\n'
                    '<label class="feature-group-toggle">'
                    '<input type="checkbox" data-ft-model="coxph" checked>'
                    ' CoxPH</label>\n'
                ),
                "figures": [],
                "figure_info": [],
                "tabs": None,
            },
        ]
        content = renderer.render(sections, metadata=metadata)
        assert "<label" not in content
        assert "<input" not in content
        assert "feature-group-toggle" not in content
        # The visible model name is kept.
        assert "CoxPH" in content

    def test_prediction_table_no_scroll(self, renderer, sections_with_html, metadata):
        content = renderer.render(sections_with_html, metadata=metadata)
        assert "prediction-table-scroll" not in content
        # Inner table content preserved
        assert "Col A" in content

    def test_clean_html_strong(self, renderer, sections_with_html, metadata):
        content = renderer.render(sections_with_html, metadata=metadata)
        assert "**bold text**" in content
        assert "<strong>" not in content

    def test_html_entity_decode(self, renderer, sections_with_html, metadata):
        content = renderer.render(sections_with_html, metadata=metadata)
        # &#9888; -> Unicode warning sign
        assert "\u26a0" in content
        assert "&#9888;" not in content

    def test_br_to_newline(self, renderer, sections_with_html, metadata):
        content = renderer.render(sections_with_html, metadata=metadata)
        assert "<br>" not in content

    def test_bibtex_details_dropped(self, renderer, sections_with_html, metadata):
        content = renderer.render(sections_with_html, metadata=metadata)
        assert "@article" not in content

    def test_non_bibtex_details_expanded(self, renderer, sections_with_html, metadata):
        content = renderer.render(sections_with_html, metadata=metadata)
        # The "Full details" details block should be expanded inline
        assert "Some expanded content here." in content

    def test_tab_markdown_cleaned(self, renderer, sections_with_html, metadata):
        content = renderer.render(sections_with_html, metadata=metadata)
        # Tab content should also have HTML stripped
        assert "**summary**" in content
        assert "<strong>" not in content


# ---------------------------------------------------------------------------
# Figure cross-references
# ---------------------------------------------------------------------------

class TestFigureCrossRefs:
    def test_figure_cross_refs(self, renderer, sections_with_figures, metadata):
        content = renderer.render(sections_with_figures, metadata=metadata)
        assert "{#fig:correlation_heatmap}" in content
        assert "{#fig:importance_bar}" in content

    def test_figure_info_stripped(self, renderer, metadata):
        """FigureInfo blocks should not appear in manuscript output."""
        sections = [
            {
                "title": "Test",
                "markdown": "![fig](images/test.png)",
                "figures": [],
                "figure_info": [
                    MagicMock(
                        figure_id="test",
                        title="Test Figure",
                        data_description="Test data",
                    )
                ],
                "tabs": None,
            },
        ]
        content = renderer.render(sections, metadata=metadata)
        # ManuscriptRenderer does not inject the HTML consolidated-access
        # panel. These assertions pin strings that ``FigureEmbedder.
        # render_consolidated_access`` does emit (_figures.py:256, 65, 71)
        # so the test would fail loudly if ManuscriptRenderer ever
        # started delegating to that renderer.
        assert "Programmatic Access" not in content
        assert "report.figures[" not in content
        assert "report.export_figure" not in content


# ---------------------------------------------------------------------------
# HTML table conversion
# ---------------------------------------------------------------------------

class TestHtmlTableConversion:
    def test_html_table_to_markdown(self):
        html = (
            "<table>"
            "<tr><th>Name</th><th>Value</th></tr>"
            "<tr><td>alpha</td><td>0.5</td></tr>"
            "<tr><td>beta</td><td>1.2</td></tr>"
            "</table>"
        )
        result = _html_table_to_markdown(html)
        assert "| Name" in result
        assert "| alpha" in result
        assert "---" in result

    def test_html_table_in_section(self, renderer, sections_with_html_table, metadata):
        content = renderer.render(sections_with_html_table, metadata=metadata)
        assert "<table>" not in content
        assert "| Feature" in content
        assert "| age" in content

    def test_empty_table(self):
        assert _html_table_to_markdown("<table></table>") == ""


# ---------------------------------------------------------------------------
# save_figures (inherited from MarkdownRenderer)
# ---------------------------------------------------------------------------

class TestSaveFigures:
    def test_save_figures_png(self, renderer, tmp_path):
        mock_fig = MagicMock()
        failed = renderer.save_figures([("test_fig.png", mock_fig)])
        mock_fig.write_image.assert_called_once()
        call_args = mock_fig.write_image.call_args
        assert call_args[0][0].endswith("test_fig.png")
        # A successful export reports nothing failed.
        assert failed == set()

    def test_save_figures_empty(self, renderer):
        assert renderer.save_figures([]) == set()

    def test_save_figures_timeout_reports_remaining(self, renderer):
        """A timeout skips the current figure and all the ones after it."""
        figs = [(f"fig{i}.png", MagicMock()) for i in range(3)]

        def _raise_on_second(fig, filepath, **kwargs):
            if str(filepath).endswith("fig1.png"):
                raise FigureExportTimeout("wedged")

        with patch(
            "mlsurv.reports._figure_export.write_image_safe",
            side_effect=_raise_on_second,
        ):
            failed = renderer.save_figures(figs)

        # fig0 exported; fig1 timed out and fig2 was never attempted.
        assert failed == {"fig1.png", "fig2.png"}

    @pytest.mark.parametrize("exc", [
        ValueError("kaleido engine error"),
        OSError("disk full"),
    ])
    def test_save_figures_single_failure_continues(self, renderer, exc):
        """A per-figure error marks only that figure and keeps going."""
        figs = [("good1.png", MagicMock()),
                ("bad.png", MagicMock()),
                ("good2.png", MagicMock())]

        def _raise_on_bad(fig, filepath, **kwargs):
            if str(filepath).endswith("bad.png"):
                raise exc

        with patch(
            "mlsurv.reports._figure_export.write_image_safe",
            side_effect=_raise_on_bad,
        ):
            failed = renderer.save_figures(figs)

        assert failed == {"bad.png"}

    def test_save_figures_timeout_warns_user(self, renderer):
        """A skipped figure must reach the user, not just the return value."""
        figs = [("fig0.png", MagicMock()), ("fig1.png", MagicMock())]

        def _timeout(fig, filepath, **kwargs):
            raise FigureExportTimeout("wedged")

        with patch(
            "mlsurv.reports._figure_export.write_image_safe",
            side_effect=_timeout,
        ):
            with pytest.warns(SuggestionWarning):
                renderer.save_figures(figs)


# ---------------------------------------------------------------------------
# strip_missing_figures — broken image-link cleanup
# ---------------------------------------------------------------------------

class TestStripMissingFigures:
    def _write_png(self, renderer, name):
        renderer.images_dir.mkdir(parents=True, exist_ok=True)
        (renderer.images_dir / name).write_bytes(b"\x89PNG\r\n")

    def test_present_ref_untouched_missing_replaced(self, renderer):
        self._write_png(renderer, "correlation_heatmap.png")
        content = (
            "![Correlation](images/correlation_heatmap.png)"
            "{#fig:correlation_heatmap}\n\n"
            "![Importance](images/importance_bar.png){#fig:importance_bar}\n"
        )
        result = renderer.strip_missing_figures(content)

        # Good ref (and its cross-ref) survive intact.
        assert "![Correlation](images/correlation_heatmap.png)" in result
        assert "{#fig:correlation_heatmap}" in result
        # Missing ref is replaced, leaving no dangling cross-ref.
        assert "importance_bar.png" not in result
        assert "{#fig:importance_bar}" not in result
        assert "*Figure unavailable: static image export failed.*" in result

    def test_bare_basename_ref(self, renderer):
        """The data-figure-converted form uses a bare basename, no images/."""
        content = "![](missing_heatmap.png){#fig:missing_heatmap}"
        result = renderer.strip_missing_figures(content)
        assert "missing_heatmap.png" not in result
        assert "{#fig:missing_heatmap}" not in result
        assert "*Figure unavailable: static image export failed.*" in result

    def test_title_variant(self, renderer):
        self._write_png(renderer, "present.png")
        missing = '![alt](images/gone.png "A caption"){#fig:gone}'
        present = '![alt](images/present.png "A caption"){#fig:present}'
        assert "*Figure unavailable" in renderer.strip_missing_figures(missing)
        assert renderer.strip_missing_figures(present) == present

    def test_empty_file_treated_as_missing(self, renderer):
        renderer.images_dir.mkdir(parents=True, exist_ok=True)
        (renderer.images_dir / "zero.png").write_bytes(b"")
        result = renderer.strip_missing_figures("![](images/zero.png)")
        assert "*Figure unavailable" in result

    def test_plain_missing_ref_no_suffix_no_title(self, renderer):
        """The base case: bare ![alt](images/x.png) with no cross-ref."""
        result = renderer.strip_missing_figures("![alt](images/gone.png)")
        assert result == "*Figure unavailable: static image export failed.*"

    def test_multiple_missing_refs_all_replaced(self, renderer):
        """Every missing ref is replaced; interleaved good refs survive."""
        self._write_png(renderer, "good1.png")
        self._write_png(renderer, "good2.png")
        content = (
            "![](images/good1.png){#fig:good1}\n"
            "![](images/bad1.png){#fig:bad1}\n"
            "![](images/bad2.png){#fig:bad2}\n"
            "![](images/good2.png){#fig:good2}\n"
        )
        result = renderer.strip_missing_figures(content)

        assert "![](images/good1.png){#fig:good1}" in result
        assert "![](images/good2.png){#fig:good2}" in result
        assert "bad1.png" not in result and "{#fig:bad1}" not in result
        assert "bad2.png" not in result and "{#fig:bad2}" not in result
        assert result.count(
            "*Figure unavailable: static image export failed.*") == 2

    def test_crossref_consumed_in_place(self, renderer):
        """The cross-ref is consumed with the ref; surrounding text is kept."""
        content = "before ![](images/gone.png){#fig:gone} after"
        result = renderer.strip_missing_figures(content)
        assert result == (
            "before *Figure unavailable: static image export failed.* after"
        )

    def test_failed_set_replaces_even_when_file_present(self, renderer):
        """A stale PNG from a prior run must not mask a current-run failure."""
        self._write_png(renderer, "stale.png")  # valid file on disk
        content = "![](images/stale.png){#fig:stale}"
        # save_figures reported stale.png as not written this run.
        result = renderer.strip_missing_figures(content, {"stale.png"})
        assert "stale.png" not in result
        assert "*Figure unavailable" in result

    def test_missing_file_not_in_failed_still_replaced(self, renderer):
        """The on-disk check still catches misses not in the failed set."""
        result = renderer.strip_missing_figures(
            "![](images/gone.png)", failed=set()
        )
        assert "*Figure unavailable" in result

    def test_strip_on_rendered_manuscript_body(
        self, renderer, sections_with_figures, metadata
    ):
        """End to end on a real render() body: cross-refs must be consumed.

        render() appends ``{#fig:stem}`` to every ref, so this confirms the
        strip regex agrees with the cross-ref form the manuscript renderer
        actually emits.
        """
        body = renderer.render(sections_with_figures, metadata=metadata)
        assert "{#fig:correlation_heatmap}" in body  # sanity: cross-ref added

        # correlation_heatmap exported; importance_bar was skipped this run.
        self._write_png(renderer, "correlation_heatmap.png")
        result = renderer.strip_missing_figures(body, {"importance_bar.png"})

        assert "images/correlation_heatmap.png" in result
        assert "{#fig:correlation_heatmap}" in result
        assert "importance_bar.png" not in result
        assert "{#fig:importance_bar}" not in result
        assert "*Figure unavailable" in result

    def test_no_image_refs_unchanged(self, renderer):
        content = "Just some prose with no figures.\n\n## Heading\n"
        assert renderer.strip_missing_figures(content) == content


# ---------------------------------------------------------------------------
# HTML renderer parity (no-op figure handling)
# ---------------------------------------------------------------------------

class TestHtmlRendererFigureParity:
    def test_save_figures_returns_empty_set(self, tmp_path):
        from mlsurv.reports.renderers.html import HTMLRenderer
        r = HTMLRenderer(str(tmp_path / "report.html"), title="T")
        assert r.save_figures([("x.png", MagicMock())]) == set()

    def test_strip_missing_figures_is_noop(self, tmp_path):
        from mlsurv.reports.renderers.html import HTMLRenderer
        r = HTMLRenderer(str(tmp_path / "report.html"), title="T")
        content = '<img src="images/x.png">'
        assert r.strip_missing_figures(content, {"x.png"}) == content


# ---------------------------------------------------------------------------
# Integration with generator
# ---------------------------------------------------------------------------

def _mock_generator(tmp_path):
    """Create a ReportGenerator with a mocked learner and single section."""
    from mlsurv.reports.generator import ReportGenerator

    mock_learner = MagicMock()
    mock_learner._state.is_setup = True
    mock_learner._state.X_train = None
    mock_learner._state.models = {}
    # generate() auto-wires eval/bootstrap from the learner; honor the
    # real _gather_report_data() -> (eval_result, bootstrap_results) contract.
    mock_learner._gather_report_data.return_value = (None, None)

    gen = ReportGenerator(mock_learner)

    mock_section = MagicMock()
    mock_section.is_available.return_value = True
    mock_section.name = "summary"
    mock_section.title = "Executive Summary"
    mock_section.generate.return_value = {
        "markdown": "Test content",
        "figures": [],
        "figure_info": [],
        "tabs": None,
    }
    mock_section.data = {}
    return gen, mock_section


class TestGeneratorIntegration:
    def test_md_extension_uses_manuscript(self, tmp_path):
        """'.md' extension should route to ManuscriptRenderer (YAML front matter)."""
        gen, mock_section = _mock_generator(tmp_path)
        md_path = tmp_path / "report.md"

        with patch.object(gen, '_create_sections', return_value=[mock_section]):
            gen.generate(str(md_path), title="Test")

        content = md_path.read_text(encoding="utf-8")
        # YAML front matter is unique to ManuscriptRenderer
        assert content.startswith("---\n")
        assert 'title: "Test"' in content
        assert "\\newpage" in content

    def test_dual_output(self, tmp_path):
        """manuscript=True with HTML format should create both files."""
        gen, mock_section = _mock_generator(tmp_path)
        html_path = tmp_path / "report.html"

        with patch.object(gen, '_create_sections', return_value=[mock_section]):
            gen.generate(str(html_path), title="Test", manuscript=True)

        assert html_path.exists()
        ms_path = tmp_path / "report_manuscript.md"
        assert ms_path.exists()
        ms_content = ms_path.read_text(encoding="utf-8")
        assert ms_content.startswith("---\n")
        assert "## Executive Summary" in ms_content

    def test_manuscript_false_no_companion(self, tmp_path):
        """manuscript=False (default) should NOT create a companion .md file."""
        gen, mock_section = _mock_generator(tmp_path)
        html_path = tmp_path / "report.html"

        with patch.object(gen, '_create_sections', return_value=[mock_section]):
            gen.generate(str(html_path), title="Test", manuscript=False)

        ms_path = tmp_path / "report_manuscript.md"
        assert not ms_path.exists()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_sections(self, renderer, metadata):
        content = renderer.render([], metadata=metadata)
        assert "---" in content  # YAML front matter
        assert "Generated by mlsurv" in content

    def test_no_excessive_blank_lines(self, renderer, metadata):
        sections = [
            {
                "title": "Test",
                "markdown": "Line 1\n\n\n\n\n\nLine 2",
                "figures": [],
                "figure_info": [],
                "tabs": None,
            },
        ]
        content = renderer.render(sections, metadata=metadata)
        assert "\n\n\n" not in content

    def test_summary_becomes_bold(self, renderer, metadata):
        sections = [
            {
                "title": "Test",
                "markdown": "<summary>Click me</summary>",
                "figures": [],
                "figure_info": [],
                "tabs": None,
            },
        ]
        content = renderer.render(sections, metadata=metadata)
        assert "**Click me**" in content
        assert "<summary>" not in content

    def test_write_returns_path(self, renderer):
        result = renderer.write("test content")
        assert isinstance(result, Path)
        assert result.exists()

    def test_nested_details(self, renderer, metadata):
        """Nested <details> blocks should be fully expanded."""
        sections = [
            {
                "title": "Test",
                "markdown": (
                    "<details><summary>Outer</summary>"
                    "<details><summary>Inner</summary>"
                    "Nested content"
                    "</details>"
                    "Outer content"
                    "</details>"
                ),
                "figures": [],
                "figure_info": [],
                "tabs": None,
            },
        ]
        content = renderer.render(sections, metadata=metadata)
        assert "<details" not in content
        assert "Nested content" in content
        assert "Outer content" in content

    def test_em_to_italic(self, renderer, metadata):
        sections = [
            {
                "title": "Test",
                "markdown": "This is <em>italic</em> text.",
                "figures": [],
                "figure_info": [],
                "tabs": None,
            },
        ]
        content = renderer.render(sections, metadata=metadata)
        assert "*italic*" in content
        assert "<em>" not in content

    def test_code_to_backtick(self, renderer, metadata):
        sections = [
            {
                "title": "Test",
                "markdown": "Use <code>predict()</code> method.",
                "figures": [],
                "figure_info": [],
                "tabs": None,
            },
        ]
        content = renderer.render(sections, metadata=metadata)
        assert "`predict()`" in content
        assert "<code>" not in content

    def test_span_stripped(self, renderer, metadata):
        sections = [
            {
                "title": "Test",
                "markdown": '<span class="highlight">text</span>',
                "figures": [],
                "figure_info": [],
                "tabs": None,
            },
        ]
        content = renderer.render(sections, metadata=metadata)
        assert "text" in content
        assert "<span" not in content

    def test_partial_metadata(self, renderer, metadata):
        """Partial metadata (only version) should not produce date or abstract."""
        partial = {"version": "1.0.0"}
        content = renderer.render([], metadata=partial)
        assert 'date:' not in content
        # abstract should still be generated from version
        assert 'abstract: "mlsurv v1.0.0"' in content

    def test_yaml_title_with_quotes(self, tmp_path, metadata):
        """Double quotes in title should be escaped in YAML."""
        renderer = ManuscriptRenderer(
            str(tmp_path / "report.md"), title='My "Best" Analysis'
        )
        content = renderer.render([], metadata=metadata)
        assert 'title: "My \\"Best\\" Analysis"' in content

    def test_ragged_table_rows(self):
        """Table rows with fewer cells than header should be padded."""
        html = (
            "<table>"
            "<tr><th>A</th><th>B</th><th>C</th></tr>"
            "<tr><td>1</td></tr>"
            "</table>"
        )
        result = _html_table_to_markdown(html)
        assert "| A" in result
        assert "| 1" in result
        # Should have proper formatting despite missing cells
        lines = result.strip().split("\n")
        assert len(lines) == 3  # header + separator + 1 body row

    def test_br_in_table_cell(self):
        """<br> inside table cells should become spaces."""
        html = (
            "<table>"
            "<tr><th>Col</th></tr>"
            "<tr><td>foo<br>bar</td></tr>"
            "</table>"
        )
        result = _html_table_to_markdown(html)
        assert "foo bar" in result

    def test_inline_tag_multiline(self, renderer, metadata):
        """<strong> spanning a newline should still be converted."""
        sections = [
            {
                "title": "Test",
                "markdown": "<strong>line1\nline2</strong>",
                "figures": [],
                "figure_info": [],
                "tabs": None,
            },
        ]
        content = renderer.render(sections, metadata=metadata)
        assert "**line1\nline2**" in content
        assert "<strong>" not in content
