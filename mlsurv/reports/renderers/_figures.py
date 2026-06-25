"""Figure embedding logic for the HTML report renderer."""

import html as html_module
import re
from typing import Dict, List, Optional, Union

from mlsurv.reports.sections.base import DataInfo, FigureInfo


_TRACE_FILTER_FIGURE_IDS = frozenset({
    "train_cv_evolution",
    "bootstrap_metric_evolution",
    "tuning_trial_evolution",
})


# ------------------------------------------------------------------
# Consolidated-access helpers (extracted from render_consolidated_access)
# ------------------------------------------------------------------


def _inject_return_data(learner_code: str) -> str:
    """Transform learner code to add ``return_data=True``.

    Handles three cases: no-args ``()``, has-args ``(a, b)``, no-parens.
    """
    lc = learner_code.rstrip()
    if lc.endswith('()'):
        return lc[:-1] + 'return_data=True)'
    if '(' in lc and lc.endswith(')'):
        idx = lc.rfind(')')
        prefix = lc[:idx].rstrip().rstrip(',')
        return prefix + ', return_data=True)'
    return lc + '(return_data=True)'


def _bind_var(code: str, var: str) -> str:
    """Normalize a snippet so its first statement binds *var*.

    Some ``learner_code`` snippets already assign (``fig = learner.plot_x()``);
    others are a bare expression (``learner.subpopulation('g').plot_metric()``).
    To follow them with a ``fig.write_image(...)`` / ``df.to_csv(...)`` line we
    need the figure/frame bound to a known name. If the first line is an
    assignment, rewrite its target to *var*; otherwise prepend ``var = ``.
    """
    first_line = code.split('\n', 1)[0]
    if re.match(r'^\s*\w+\s*=(?!=)', first_line):
        return re.sub(r'^(\s*)\w+\s*=(?!=)', rf'\g<1>{var} =', code, count=1)
    return f'{var} = {code}'


def _resolve_data_label(dk, di, fig_match):
    """Resolve a human-readable label for a data key.

    Checks DataInfo first, then matching FigureInfo, falls back to title-cased key.

    Parameters
    ----------
    dk : str
        The raw data key.
    di : DataInfo or None
        Pre-resolved DataInfo for this key.
    fig_match : FigureInfo or None
        Pre-resolved FigureInfo whose data_key matches *dk*.
    """
    if di:
        return di.description
    if fig_match and fig_match.data_description:
        return fig_match.data_description
    return dk.replace('_', ' ').title()


def _figure_accessible_name(info, figure_id):
    """Resolve a non-empty accessible name for a figure wrapper.

    Prefers the ``FigureInfo`` title; falls back to a humanized ``figure_id``
    whenever the title is missing or empty (including when no ``FigureInfo``
    exists at all), so every embedded figure exposes a non-empty screen-reader
    name.

    Parameters
    ----------
    info : FigureInfo or None
        Per-figure metadata, when available.
    figure_id : str or None
        Figure id (filename without extension), used for the fallback name.
    """
    if info is not None and info.title:
        return info.title
    if figure_id:
        return figure_id.replace('_', ' ').strip().title()
    return ""


def _render_figure_items(figure_info_list, esc):
    """Build HTML lines for the Figures group in the access dropdown.

    A figure gets an entry only when it offers a runnable hint: a
    ``learner_code`` call (reproduce via ``learner.plot_*(...)`` then export
    with Plotly's native ``fig.write_image(...)``) or, failing that, a
    ``standalone_import`` pointer to the viz function. Report-only figures with
    neither are skipped rather than advertising a line that can't run. Returns
    ``[]`` when no figure has a hint.
    """
    items = []
    for info in figure_info_list:
        if not info.learner_code and not info.standalone_import:
            continue
        items.append('      <div class="access-item">')
        items.append(f'        <h6>{esc(info.title)}</h6>')
        if info.learner_code:
            plot_code = _bind_var(info.learner_code, 'fig')
            items.append(f'        <pre><code>{esc(plot_code)}</code></pre>')
            if info.standalone_import:
                items.append(f'        <pre><code>{esc(info.standalone_import)}</code></pre>')
            items.append(f'        <pre><code>'
                         f'fig.write_image(&#x27;{esc(info.figure_id)}.svg&#x27;)</code></pre>')
        else:
            # Import-only pointer: no learner call to bind ``fig``, so show the
            # import on its own rather than a broken export line.
            items.append(f'        <pre><code>{esc(info.standalone_import)}</code></pre>')
        items.append('      </div>')
    if not items:
        return []
    return ['    <div class="access-group">', '      <h5>Figures</h5>', *items, '    </div>']


def _find_matching_figure(figure_info_list, data_key):
    """Find the first FigureInfo whose data_key matches, or None."""
    return next(
        (fi for fi in figure_info_list if fi.data_key == data_key), None
    )


def _data_accessor_lines(dk, di, fig_match):
    """Code line(s) that reach the tidy data for *dk*, or ``[]``.

    Shows the learner accessor that produces the data, never the unreachable
    ``report`` object. No ``df.to_csv(...)`` export line is appended: the
    accessors return heterogeneous types (DataFrame, ndarray, result object,
    or a dict of frames for multi-panel plots), so a blanket ``to_csv`` would
    not always run. The export idiom lives in ``docs/reporting.md`` instead.

    - An explicit ``DataInfo.learner_code`` / ``results_code`` is shown as-is.
    - Otherwise a matching figure's ``learner_code`` is reused with
      ``return_data=True``, bound to ``df`` for a readable variable name.
    """
    if di and di.learner_code:
        lines = [di.learner_code]
        if di.results_code:
            lines.append(di.results_code)
        return lines
    if fig_match and fig_match.learner_code:
        lines = [_bind_var(_inject_return_data(fig_match.learner_code), 'df')]
        if di and di.results_code:
            lines.append(di.results_code)
        return lines
    if di and di.results_code:
        return [di.results_code]
    return []


def _render_data_items(data_keys, data_info_map, figure_info_list, esc):
    """Build HTML lines for the Data group in the access dropdown.

    Only data keys reachable through a learner/results accessor get an entry.
    Keys with no runnable accessor are skipped rather than advertising an
    unreachable ``report.data[...]``. Returns ``[]`` when no key is reachable.
    """
    items = []
    for dk in data_keys:
        di = data_info_map.get(dk)
        fig_match = _find_matching_figure(figure_info_list, dk)

        lines = _data_accessor_lines(dk, di, fig_match)
        if not lines:
            continue

        items.append('      <div class="access-item">')
        items.append(f'        <h6>{esc(_resolve_data_label(dk, di, fig_match))}</h6>')
        for line in lines:
            items.append(f'        <pre><code>{esc(line)}</code></pre>')
        items.append('      </div>')
    if not items:
        return []
    return ['    <div class="access-group">', '      <h5>Data</h5>', *items, '    </div>']


class FigureEmbedder:
    """Manage Plotly figure embedding and trace-filter state for a single render pass.

    Parameters
    ----------
    include_plotlyjs : bool or str
        How to include Plotly.js library (True for inline, 'cdn' for CDN).
    """

    def __init__(self, include_plotlyjs: Union[bool, str] = True) -> None:
        self.include_plotlyjs = include_plotlyjs
        self._plotlyjs_included = False
        self._figure_counter = 0
        self.has_trace_filter_figures = False

    def reset(self) -> None:
        """Reset per-render state (call at the start of each ``render()``)."""
        self.has_trace_filter_figures = False

    def replace_placeholders(
        self,
        content_html: str,
        figures_dict: Dict[str, object],
        figure_info_map: Dict[str, FigureInfo],
    ) -> tuple[str, set[str]]:
        """Replace ``<div class="figure-placeholder" ...>`` tags with embedded figures.

        Parameters
        ----------
        content_html : str
            Report HTML containing figure-placeholder ``<div>`` tags.
        figures_dict : dict of {str: object}
            Mapping from figure filename to the figure object to embed.
        figure_info_map : dict of {str: FigureInfo}
            Per-figure metadata keyed by figure id (filename without extension).

        Returns
        -------
        tuple of (content_html, embedded)
            Updated HTML and set of filenames that were embedded inline.
        """
        embedded: set[str] = set()
        for filename in figures_dict:
            placeholder = f'<div class="figure-placeholder" data-figure="{filename}"></div>'
            if placeholder in content_html:
                figure_id = filename.rsplit('.', 1)[0]
                fig_html = self.embed(
                    figures_dict[filename],
                    figure_info_map.get(figure_id),
                    figure_id=figure_id,
                )
                content_html = content_html.replace(placeholder, fig_html)
                embedded.add(filename)
        return content_html, embedded

    def render_remaining(
        self,
        html_parts: list,
        figures: list,
        embedded: set,
        figure_info_map: Dict[str, FigureInfo],
    ) -> None:
        """Embed figures that were not referenced by inline placeholders."""
        for filename, fig in figures:
            if filename in embedded:
                continue
            try:
                fig_html = fig.to_html(
                    full_html=False,
                    include_plotlyjs=self._resolve_plotlyjs(),
                )
            except (MemoryError, RecursionError):
                raise
            except Exception as exc:
                from mlsurv.logging_utils import get_logger, log_exception
                log_exception(
                    get_logger("reports.renderers.figures"), exc,
                    f"figure serialization: {filename}",
                    warn_user=True,
                    user_message=f"Could not render figure: {exc}",
                )
                fig_html = '<em>[Figure rendering failed]</em>'
            figure_id = filename.rsplit('.', 1)[0]
            html_parts.append(
                self._wrap(
                    fig_html,
                    include_trace_filters=self.supports_trace_filters(figure_id),
                    label=_figure_accessible_name(
                        figure_info_map.get(figure_id), figure_id,
                    ),
                )
            )

    def embed(
        self,
        fig: object,
        info: Optional[FigureInfo] = None,
        figure_id: Optional[str] = None,
    ) -> str:
        """Embed a single Plotly figure as HTML (without per-figure access dropdown)."""
        try:
            fig_html = fig.to_html(full_html=False, include_plotlyjs=self._resolve_plotlyjs())
        except (MemoryError, RecursionError):
            raise
        except Exception as exc:
            from mlsurv.logging_utils import get_logger, log_exception
            log_exception(
                get_logger("reports.renderers.figures"), exc,
                f"figure serialization: {figure_id or 'unknown'}",
                warn_user=True,
                user_message=f"Could not render figure: {exc}",
            )
            fig_html = '<em>[Figure rendering failed]</em>'
        return self._wrap(
            fig_html,
            include_trace_filters=self.supports_trace_filters(figure_id or ""),
            label=_figure_accessible_name(info, figure_id),
        )

    # ------------------------------------------------------------------
    # Consolidated access dropdown
    # ------------------------------------------------------------------

    @staticmethod
    def render_consolidated_access(
        section_name: str,
        figure_info_list: List[FigureInfo],
        data_keys: List[str],
        data_info_list: Optional[List[DataInfo]] = None,
    ) -> str:
        """Render a single consolidated programmatic-access dropdown for a section.

        Parameters
        ----------
        section_name : str
            Section registry name (e.g. ``'feature_analysis'``).
        figure_info_list : list of FigureInfo
            All figure metadata for this section.
        data_keys : list of str
            All keys in ``section.data`` (without section prefix).
        data_info_list : list of DataInfo, optional
            Additional standalone data descriptions.

        Returns
        -------
        str
            An HTML ``<details>`` block with programmatic-access examples, or an
            empty string when nothing in the section is reachable through a
            runnable learner accessor.
        """
        esc = html_module.escape

        # --- Figures ---
        figure_parts = (
            _render_figure_items(figure_info_list, esc)
            if figure_info_list else []
        )

        # --- Data ---
        data_info_map: Dict[str, DataInfo] = {
            di.data_key: di for di in (data_info_list or [])
        }

        all_data_keys = list(dict.fromkeys(data_keys))  # preserve order, dedupe
        data_parts = (
            _render_data_items(all_data_keys, data_info_map, figure_info_list, esc)
            if all_data_keys else []
        )

        if not figure_parts and not data_parts:
            return ""

        parts = [
            '<details class="section-access">',
            '  <summary>Programmatic Access</summary>',
            '  <div class="access-content">',
            *figure_parts,
            *data_parts,
            '  </div>',
            '</details>',
        ]
        return '\n'.join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wrap(
        self,
        fig_html: str,
        *,
        include_trace_filters: bool = False,
        label: Optional[str] = None,
    ) -> str:
        """Wrap figure HTML and optionally include trace-visibility controls.

        ``label`` is the figure's accessible name. Static figures get
        ``role="img"`` so assistive tech announces the chart as a single image.
        Trace-filter figures instead get ``role="group"`` — ``role="img"`` would
        flatten the subtree and hide the curve-visibility controls from screen
        readers, so a labelled group keeps those controls reachable.
        """
        name = html_module.escape(label) if label else ""
        if include_trace_filters:
            self.has_trace_filter_figures = True
            self._figure_counter += 1
            role_attr = f' role="group" aria-label="{name}"' if name else ""
            return (
                f'<div class="figure-container has-trace-filters" '
                f'data-figure-id="fig-{self._figure_counter}"{role_attr}>'
                '<div class="trace-filter-controls" aria-label="Curve visibility controls"></div>'
                f'{fig_html}</div>'
            )
        role_attr = f' role="img" aria-label="{name}"' if name else ""
        return f'<div class="figure-container"{role_attr}>{fig_html}</div>'

    def _resolve_plotlyjs(self) -> Union[bool, str]:
        """Return the plotlyjs include value, ensuring it is only included once."""
        if not self._plotlyjs_included:
            self._plotlyjs_included = True
            return self.include_plotlyjs
        return False

    @staticmethod
    def supports_trace_filters(figure_id: str) -> bool:
        """Return True only for Run Evolution figures."""
        return figure_id in _TRACE_FILTER_FIGURE_IDS
