"""HTML renderer for report generation with embedded Plotly figures."""

import html as html_module
import re
from pathlib import Path
from typing import Dict, Any, List, Set, Union, Optional
from urllib.parse import urlparse

import mistune
from mistune.plugins.table import table as mistune_table

from mlsurv.logging_utils import get_logger, log_exception
from mlsurv.reports.renderers._figures import FigureEmbedder
from mlsurv.reports.renderers._scripts import (
    TABS_SCRIPT,
    ANCHOR_NAV_SCRIPT,
    TRACE_FILTER_SCRIPT,
    TABLE_SORT_SCRIPT,
    PREDICTION_TABLE_SCROLL_SCRIPT,
    FEATURE_TABLE_TOGGLE_SCRIPT,
    wrap_script,
)
from mlsurv.reports.renderers._styles import LOGO_SVG, REPORT_CSS


def _sanitize_markdown_link_url(url: str) -> str:
    """Allow only http(s) and fragment links in markdown anchors."""
    if not url:
        return "#"
    if url.startswith("#"):
        return url
    parsed = urlparse(url)
    if parsed.scheme.lower() in {"http", "https"}:
        return url
    return "#"


class _SafeMarkdownRenderer(mistune.HTMLRenderer):
    """Mistune renderer with safe links and report-specific heading levels.

    Sections may emit a fenced code block with the language tag
    ``mlsurv-html`` to inject trusted HTML (e.g. color-coded delta
    tables). Any other HTML in markdown is still escaped per the
    ``escape=True`` constructor argument.
    """

    _TRUSTED_HTML_LANG = "mlsurv-html"

    def block_code(self, code: str, info: Optional[str] = None) -> str:
        if info and info.strip() == self._TRUSTED_HTML_LANG:
            return code
        return super().block_code(code, info)

    def heading(self, text: str, level: int, **attrs: Any) -> str:
        """Shift markdown headings down one level and add id for anchor links."""
        adjusted_level = min(max(level + 1, 2), 6)
        # Generate slug from text (strip HTML tags first)
        plain = re.sub(r'<[^>]+>', '', text)
        slug = re.sub(r'[^\w\s-]', '', plain).strip().lower()
        slug = re.sub(r'[\s]+', '-', slug)
        return f'<h{adjusted_level} id="{slug}">{text}</h{adjusted_level}>'

    def link(self, text: str, url: str, title: Optional[str] = None) -> str:
        """Render links with a strict URL allowlist."""
        safe_url = _sanitize_markdown_link_url(url)
        href = html_module.escape(safe_url, quote=True)
        if title is None:
            return f'<a href="{href}">{text}</a>'
        safe_title = html_module.escape(title, quote=True)
        return f'<a href="{href}" title="{safe_title}">{text}</a>'


class HTMLRenderer:
    """
    Render report content to HTML format with embedded interactive Plotly figures.

    Creates a single self-contained HTML file with no external dependencies.
    Plotly figures are embedded directly in the HTML using to_html().

    Parameters
    ----------
    output_path : str or Path
        Path to output HTML file.
    title : str
        Report title.
    include_plotlyjs : bool or str
        How to include Plotly.js library:
        - True: Embed full library (~3MB, works offline)
        - 'cdn': Use CDN link (smaller file, requires internet)
    """

    def __init__(
        self,
        output_path: str,
        title: str = "Survival Analysis Report",
        include_plotlyjs: Union[bool, str] = True,
    ):
        self.output_path = Path(output_path)
        self.title = title
        self.include_plotlyjs = include_plotlyjs
        self._figures = FigureEmbedder(include_plotlyjs)
        self._markdown_parser = mistune.create_markdown(
            renderer=_SafeMarkdownRenderer(escape=True),
            plugins=[mistune_table],
        )

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def render(self, sections: List[Dict[str, Any]], metadata: Optional[dict] = None) -> str:
        """
        Render all sections to HTML.

        Parameters
        ----------
        sections : list
            List of section dictionaries with 'title', 'markdown', 'figures' keys.
        metadata : dict, optional
            Report metadata (version, timestamp, dimensions) for the header bar.

        Returns
        -------
        str
            Complete HTML content.
        """
        html_parts = []
        self._figures.reset()
        safe_report_title = html_module.escape(str(self.title))

        # HTML header with CSS styling
        html_parts.append(self._get_html_header())

        # Header with logo and title
        html_parts.append('<div class="report-header">')
        html_parts.append(LOGO_SVG)
        html_parts.append(f'<h1 class="report-title">{safe_report_title}</h1>')
        if metadata:
            html_parts.append(self._render_metadata_bar(metadata))
        html_parts.append('</div>')

        if sections:
            section_tab_group = "report-sections"
            html_parts.append(
                f'<div class="phase-tabs report-sections-tabs" data-tab-group="{section_tab_group}">'
            )

            # Build tab button items: (target_id, label)
            tab_items = []
            for section in sections:
                section_title = str(section.get("title", "Section"))
                safe_section_title = html_module.escape(section_title)
                anchor = self._make_anchor(section_title)
                tab_items.append((anchor, safe_section_title))
            html_parts.append(
                self._build_tab_buttons(tab_items, section_tab_group, extra_class="report-tab-buttons")
            )

            for idx, section in enumerate(sections):
                section_title = str(section.get("title", "Section"))
                safe_section_title = html_module.escape(section_title)
                anchor = self._make_anchor(section_title)
                active_class = " active" if idx == 0 else ""
                html_parts.append(
                    f'<section class="report-section tab-panel{active_class}" '
                    f'id="{anchor}" data-tab-group="{section_tab_group}">'
                )
                html_parts.append(f'<h2>{safe_section_title}</h2>')

                # Convert markdown content to HTML
                content_html = self._markdown_to_html(section['markdown'])

                # Build figure_info lookup by figure_id
                figure_info_map = {
                    info.figure_id: info
                    for info in section.get('figure_info', [])
                }

                # Replace figure placeholders with embedded figures
                figures_dict = dict(section.get('figures', []))
                content_html, embedded = self._figures.replace_placeholders(
                    content_html, figures_dict, figure_info_map
                )

                html_parts.append(f'<div class="section-content">{content_html}</div>')

                tabs = section.get('tabs') or []
                if tabs:
                    html_parts.append(self._render_tabs(tabs, anchor))

                # Embed remaining figures that weren't referenced inline
                self._figures.render_remaining(
                    html_parts, section.get('figures', []), embedded, figure_info_map
                )

                # Consolidated programmatic access dropdown (one per section)
                section_name = section.get('section_name', '')
                if section_name:
                    access_html = FigureEmbedder.render_consolidated_access(
                        section_name=section_name,
                        figure_info_list=list(figure_info_map.values()),
                        data_keys=section.get('data_keys', []),
                        data_info_list=section.get('data_info'),
                    )
                    if access_html:
                        html_parts.append(access_html)

                html_parts.append('</section>')

            html_parts.append('</div>')

        if sections:
            html_parts.append(wrap_script(TABS_SCRIPT))
            html_parts.append(wrap_script(ANCHOR_NAV_SCRIPT))
        if self._figures.has_trace_filter_figures:
            html_parts.append(wrap_script(TRACE_FILTER_SCRIPT))
        html_parts.append(wrap_script(TABLE_SORT_SCRIPT))
        html_parts.append(wrap_script(PREDICTION_TABLE_SCROLL_SCRIPT))
        html_parts.append(wrap_script(FEATURE_TABLE_TOGGLE_SCRIPT))

        # Footer
        html_parts.append('<footer class="report-footer">')
        html_parts.append('<hr>')
        html_parts.append('<p><em>Generated by mlsurv</em></p>')
        html_parts.append('</footer>')

        # HTML closing tags
        html_parts.append('</div>')  # container
        html_parts.append('</body>')
        html_parts.append('</html>')

        return '\n'.join(html_parts)

    def _render_tabs(self, tabs: List[Dict[str, str]], section_anchor: str) -> str:
        """Render tabbed markdown blocks as an interactive HTML widget."""
        if not tabs:
            return ""

        group_id = f"{section_anchor}-tabs"
        html_parts = [f'<div class="phase-tabs" data-tab-group="{group_id}">']

        tab_items = []
        panel_html = []
        for idx, tab in enumerate(tabs):
            tab_key = self._sanitize_tab_key(tab.get("key", f"tab-{idx}"))
            panel_id = f"{group_id}-{tab_key}"
            active_class = " active" if idx == 0 else ""
            title = tab.get("title", tab_key.title())
            safe_title = html_module.escape(str(title))
            tab_markdown = tab.get("markdown", "")
            content_html = self._markdown_to_html(tab_markdown)

            tab_items.append((panel_id, safe_title))
            panel_html.append(
                f'<div id="{panel_id}" class="tab-panel{active_class}" data-tab-group="{group_id}">{content_html}</div>'
            )

        html_parts.append(self._build_tab_buttons(tab_items, group_id))
        html_parts.extend(panel_html)
        html_parts.append('</div>')
        return "\n".join(html_parts)

    @staticmethod
    def _build_tab_buttons(
        items: List[tuple], group_id: str, *, extra_class: str = ""
    ) -> str:
        """Generate tab button bar HTML.

        Parameters
        ----------
        items : list of (target_id, label)
            Tab targets and their display labels (labels must be pre-escaped).
        group_id : str
            The ``data-tab-group`` value shared by buttons and panels.
        extra_class : str
            Additional CSS class(es) for the button container div.
        """
        css_class = "phase-tab-buttons"
        if extra_class:
            css_class += f" {extra_class}"
        parts = [f'<div class="{css_class}">']
        for idx, (target_id, label) in enumerate(items):
            active_class = " active" if idx == 0 else ""
            parts.append(
                f'<button type="button" class="tab-button{active_class}" '
                f'data-tab-group="{group_id}" data-tab-target="{target_id}">{label}</button>'
            )
        parts.append('</div>')
        return "\n".join(parts)

    @staticmethod
    def _sanitize_tab_key(key: str) -> str:
        """Create a CSS/HTML-id safe tab key."""
        value = str(key).strip().lower().replace(" ", "-")
        return re.sub(r"[^a-z0-9_-]", "", value) or "tab"

    @staticmethod
    def _render_metadata_bar(metadata: dict) -> str:
        """Render the report metadata bar (version, timestamp, dimensions)."""
        parts = []
        if 'version' in metadata:
            parts.append(f'mlsurv v{html_module.escape(str(metadata["version"]))}')
        if 'timestamp' in metadata:
            parts.append(html_module.escape(str(metadata['timestamp'])))
        if 'n_samples' in metadata:
            parts.append(html_module.escape(f'{metadata["n_samples"]:,} samples'))
        if 'n_features' in metadata:
            parts.append(html_module.escape(f'{metadata["n_features"]:,} features'))
        if 'n_events' in metadata:
            parts.append(html_module.escape(f'{metadata["n_events"]:,} events'))
        return f'<div class="report-metadata">{" &middot; ".join(parts)}</div>'

    def save_figures(self, figures: List[tuple]) -> Set[str]:
        """
        No-op for HTML renderer (figures embedded inline).

        Parameters
        ----------
        figures : list
            List of (filename, plotly.Figure) tuples (ignored).

        Returns
        -------
        set of str
            Always empty: HTML embeds figures directly, so none can be a
            missing file. Mirrors ``MarkdownRenderer.save_figures``.
        """
        # Figures are embedded directly in HTML, no separate files needed
        return set()

    def strip_missing_figures(
        self, content: str, failed: Optional[Set[str]] = None
    ) -> str:
        """
        No-op for HTML renderer: figures are embedded, never broken file links.

        Parameters
        ----------
        content : str
            Rendered HTML content.
        failed : set of str, optional
            Ignored. Present to match ``MarkdownRenderer``.

        Returns
        -------
        str
            ``content`` unchanged.
        """
        return content

    def write(self, content: str) -> Path:
        """
        Write HTML content to file.

        Parameters
        ----------
        content : str
            HTML content to write.

        Returns
        -------
        Path
            Path to written file.
        """
        self.output_path.write_text(content, encoding='utf-8')
        return self.output_path

    def _get_html_header(self) -> str:
        """Generate HTML header with CSS styling."""
        safe_title = html_module.escape(str(self.title))
        return (
            '<!DOCTYPE html>\n'
            '<html lang="en">\n'
            '<head>\n'
            '    <meta charset="UTF-8">\n'
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'    <title>{safe_title}</title>\n'
            '    <style>\n'
            f'{REPORT_CSS}\n'
            '    </style>\n'
            '</head>\n'
            '<body>\n'
            '<div class="container">'
        )

    def _make_anchor(self, title: str) -> str:
        """Convert section title to URL-safe anchor."""
        slug = re.sub(r"[^a-z0-9_-]+", "-", str(title).lower()).strip("-")
        return slug or "section"

    def _markdown_to_html(self, markdown: str) -> str:
        """
        Convert markdown to HTML using mistune with strict sanitization.

        Parameters
        ----------
        markdown : str
            Markdown content.

        Returns
        -------
        str
            HTML content.
        """
        markdown_text = str(markdown or "")
        prepared_markdown, replacements = self._extract_special_markdown_blocks(markdown_text)
        try:
            rendered_html = self._markdown_parser(prepared_markdown)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RecursionError) as e:
            log_exception(
                get_logger("reports.renderers.html"),
                e,
                "markdown parsing",
                warn_user=True,
                user_message=f"Markdown parsing failed ({type(e).__name__}: {e}); falling back to escaped text",
            )
            rendered_html = (
                "<!-- mlsurv: markdown parsing failed, showing escaped text -->\n"
                + html_module.escape(prepared_markdown)
            )
        return self._restore_special_markdown_blocks(rendered_html, replacements)

    # ------------------------------------------------------------------
    # Special markdown block extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_token(replacements: Dict[str, str], html_content: str) -> str:
        """Create a unique placeholder token and register *html_content*."""
        token = f"@@MLSURV_SPECIAL_{len(replacements)}@@"
        replacements[token] = html_content
        return token

    @staticmethod
    def _match_figure_placeholder(stripped: str) -> Optional[str]:
        """Return figure-placeholder HTML if *stripped* is a markdown or HTML image."""
        img_match = re.match(r'!\[.*?\]\(.*?([^/]+\.png)\)', stripped)
        if img_match:
            filename = html_module.escape(img_match.group(1), quote=True)
            return f'<div class="figure-placeholder" data-figure="{filename}"></div>'

        html_img_match = re.search(r'<img\s+src="images/([^"]+\.png)"', stripped)
        if html_img_match:
            safe_filename = html_module.escape(html_img_match.group(1), quote=True)
            return f'<div class="figure-placeholder" data-figure="{safe_filename}"></div>'

        return None

    @staticmethod
    def _handle_passthrough_line(
        stripped: str, prediction_div_depth: int
    ) -> Optional[tuple]:
        """Check *stripped* against passthrough patterns.

        Returns ``(html_content, depth_delta)`` if the line matches, else ``None``.
        """
        if stripped.startswith('<details') or stripped == '</details>':
            return stripped, 0
        if stripped.startswith('<summary') and stripped.endswith('</summary>'):
            return stripped, 0

        if stripped.startswith('<div class="prediction-table-scroll"'):
            rows_match = re.search(r'data-visible-rows="(\d+)"', stripped)
            visible_rows = rows_match.group(1) if rows_match else "30"
            return (
                '<div class="prediction-table-scroll" '
                f'data-visible-rows="{int(visible_rows)}">'
            ), 1

        if stripped == "</div>" and prediction_div_depth > 0:
            return "</div>", -1

        placeholder = HTMLRenderer._match_figure_placeholder(stripped)
        if placeholder is not None:
            return placeholder, 0

        return None

    def _extract_special_markdown_blocks(self, markdown: str) -> tuple[str, Dict[str, str]]:
        """Replace report-specific raw HTML/image lines with placeholder tokens."""
        replacements: Dict[str, str] = {}
        processed_lines: List[str] = []
        prediction_div_depth = 0
        raw_html_buffer: Optional[List[str]] = None

        for line in markdown.split("\n"):
            stripped = line.strip()

            # --- Raw HTML passthrough blocks ---
            if stripped == "<!-- MLSURV_RAW_HTML -->":
                raw_html_buffer = []
                continue
            if stripped == "<!-- /MLSURV_RAW_HTML -->" and raw_html_buffer is not None:
                processed_lines.append(
                    self._make_token(replacements, "\n".join(raw_html_buffer))
                )
                raw_html_buffer = None
                continue
            if raw_html_buffer is not None:
                raw_html_buffer.append(line)
                continue

            # --- Passthrough patterns (details, summary, prediction divs, images) ---
            result = self._handle_passthrough_line(stripped, prediction_div_depth)
            if result is not None:
                html_content, depth_delta = result
                processed_lines.append(self._make_token(replacements, html_content))
                prediction_div_depth += depth_delta
                continue

            processed_lines.append(line)

        # Flush unclosed raw HTML buffer to prevent silent data loss
        if raw_html_buffer is not None:
            get_logger("reports.renderers.html").warning(
                "Unclosed <!-- MLSURV_RAW_HTML --> block; "
                "%d buffered lines included as raw HTML",
                len(raw_html_buffer),
            )
            processed_lines.append(
                self._make_token(replacements, "\n".join(raw_html_buffer))
            )

        return "\n".join(processed_lines), replacements

    @staticmethod
    def _restore_special_markdown_blocks(content_html: str, replacements: Dict[str, str]) -> str:
        """Restore placeholder tokens to safe HTML blocks after markdown parsing."""
        restored = content_html
        for token, replacement in replacements.items():
            token_pattern = re.escape(token)
            restored = re.sub(
                rf"<p>\s*{token_pattern}\s*</p>",
                lambda _m, rep=replacement: rep,
                restored,
            )
            restored = restored.replace(token, replacement)
        return restored
