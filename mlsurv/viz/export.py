"""HTML export utilities for Plotly figures."""

import warnings

import plotly.graph_objects as go
from pathlib import Path
from typing import List, Tuple, Union, Optional
from datetime import datetime
from html import escape

from mlsurv._warnings import SuggestionWarning

#: Plots generated when ``include`` is not given.
_DEFAULT_INCLUDE = ('cindex', 'survival', 'importance')


def save_html(
    fig: go.Figure,
    output_path: Union[str, Path],
    title: str = "mlsurv Visualization",
    include_plotlyjs: Union[bool, str] = True,
) -> None:
    """
    Save Plotly figure as self-contained HTML.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to save.
    output_path : str or Path
        Output HTML file path.
    title : str
        HTML page title.
    include_plotlyjs : bool or str
        How to include plotly.js:
        - True: embed ~3MB plotly.js (works offline)
        - 'cdn': link to CDN (smaller file, requires internet)
        - False: don't include (requires plotly.js loaded elsewhere)
    """
    safe_title = escape(title)
    fig_html = fig.to_html(
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }
    )

    html = '\n'.join([
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>{safe_title}</title>",
        "<meta charset='utf-8'>",
        "</head>",
        "<body>",
        fig_html,
        "</body>",
        "</html>",
    ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding='utf-8')

def save_html_report(
    figures: List[Tuple[str, go.Figure]],
    output_path: Union[str, Path],
    title: str = "mlsurv Results",
    include_timestamp: bool = True,
    include_toc: bool = True,
) -> None:
    """
    Combine multiple figures into a single HTML report.

    Parameters
    ----------
    figures : list of (name, fig) tuples
        Each tuple contains a section title and Plotly Figure.
    output_path : str or Path
        Output HTML file path.
    title : str
        Report title.
    include_timestamp : bool
        If True, add generation timestamp.
    include_toc : bool
        If True, add table of contents.
    """
    safe_title = escape(title)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build table of contents
    toc_html = ""
    if include_toc and len(figures) > 1:
        toc_items = []
        for i, (name, _) in enumerate(figures):
            section_id = f"section-{i}"
            toc_items.append(f'<li><a href="#{section_id}">{escape(name)}</a></li>')
        toc_html = f"""
        <nav class="toc">
            <h3>Contents</h3>
            <ul>{"".join(toc_items)}</ul>
        </nav>
        """

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>{safe_title}</title>",
        "<meta charset='utf-8'>",
        "<style>",
        """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fafafa;
        }
        h1 {
            color: #333;
            text-align: center;
            border-bottom: 2px solid #4a90d9;
            padding-bottom: 15px;
        }
        h2 {
            color: #555;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
            margin-top: 40px;
        }
        .figure-container {
            margin: 20px 0;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .timestamp {
            color: #888;
            font-size: 0.9em;
            text-align: center;
            margin-bottom: 20px;
        }
        .toc {
            background: white;
            padding: 15px 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .toc h3 {
            margin-top: 0;
            color: #333;
        }
        .toc ul {
            padding-left: 20px;
        }
        .toc li {
            margin: 5px 0;
        }
        .toc a {
            color: #4a90d9;
            text-decoration: none;
        }
        .toc a:hover {
            text-decoration: underline;
        }
        .footer {
            text-align: center;
            color: #888;
            font-size: 0.8em;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }
        """,
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{safe_title}</h1>",
    ]

    if include_timestamp:
        html_parts.append(f'<p class="timestamp">Generated: {timestamp}</p>')

    html_parts.append(toc_html)

    for i, (name, fig) in enumerate(figures):
        section_id = f"section-{i}"
        # First figure includes plotly.js, rest use reference
        include_js = True if i == 0 else 'cdn'

        html_parts.append(f'<div class="figure-container" id="{section_id}">')
        html_parts.append(f'<h2>{escape(name)}</h2>')
        html_parts.append(fig.to_html(
            include_plotlyjs=include_js,
            full_html=False,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d']
            }
        ))
        html_parts.append('</div>')

    html_parts.append("""
    <div class="footer">
        Generated by mlsurv - Survival Machine Learning Package
    </div>
    """)
    html_parts.append("</body>")
    html_parts.append("</html>")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text('\n'.join(html_parts), encoding='utf-8')


def figures_to_dict(
    learner,
    model_result=None,
    include: Optional[List[str]] = None,
) -> List[Tuple[str, go.Figure]]:
    """
    Generate standard figures from a learner for export.

    Parameters
    ----------
    learner : SurvivalLearner
        Fitted learner with comparison results.
    model_result : ModelResult, optional
        Specific model for detailed plots.
    include : list, optional
        Which plots to include. Options:
        - 'cindex': C-index heatmap
        - 'auc': Time-dependent AUC curves
        - 'survival': KM curves
        - 'importance': Feature importance
        - 'calibration': Calibration plot
        - 'brier': Brier score curve
        - 'forest': Hazard ratio forest plot
        Default: ['cindex', 'survival', 'importance']
        'survival', 'importance', and 'calibration' need ``model_result``
        and are skipped with a warning without it. Unknown keys raise
        ``ValueError``.

    Returns
    -------
    list of (name, Figure) tuples
        Ready for save_html_report().

    Raises
    ------
    ValueError
        If ``include`` contains an unknown plot key.

    Warns
    -----
    SuggestionWarning
        When a requested plot is skipped, either because its plot method
        failed or because it needs ``model_result`` and none was given.
    """
    if include is None:
        include = list(_DEFAULT_INCLUDE)

    needs_model_result = {'survival', 'importance', 'calibration'}

    # stacklevel=3 walks warn -> _plot -> figures_to_dict, so the warning
    # points at the caller of figures_to_dict. The model name is included
    # so failures for different models don't dedup into one warning under
    # the "once" SuggestionWarning filter.
    def _plot(key, gen):
        try:
            return gen()
        except (ValueError, KeyError, IndexError) as exc:
            target = (f" for model {model_result.name!r}"
                      if key in needs_model_result else "")
            warnings.warn(
                f"Skipping plot {key!r}{target}: {type(exc).__name__}: {exc}",
                SuggestionWarning,
                stacklevel=3,
            )
            return None

    # Dispatch table: plot_key -> (title, generator_func)
    # plot_* methods select by model name, so pass model_result.name
    # (a TypeError or AttributeError here means a real bug and is allowed
    # to propagate).
    plot_generators = {
        'cindex': ('C-Index Performance', lambda: learner.plot_metric(style='forest')),
        'auc': ('Time-Dependent AUC', lambda: learner.plot_metric_time(metric='auc')),
        'survival': ('Survival Curves by Risk Score',
                     lambda: learner.plot_survival(model_result.name)),
        'importance': ('Feature Importance', lambda: learner.plot_importance(model_result.name)),
        'calibration': ('Calibration Plot', lambda: learner.plot_calibration(model_result.name)),
        'brier': ('Brier Score Curve', lambda: learner.plot_metric_time(metric='brier')),
        'forest': ('Hazard Ratios', lambda: learner.plot_hazard_ratios(kind='multivariate')),
    }

    unknown = [k for k in include if k not in plot_generators]
    if unknown:
        raise ValueError(
            f"Unknown include key(s): {', '.join(map(repr, unknown))}. "
            f"Valid options: {', '.join(sorted(plot_generators))}"
        )

    skipped = []
    if model_result is None:
        skipped = [k for k in include if k in needs_model_result]
        if skipped:
            warnings.warn(
                f"Skipping plot(s) {', '.join(map(repr, skipped))}: "
                "these require model_result, which was not provided.",
                SuggestionWarning,
                stacklevel=2,
            )

    figures = []
    for key in include:
        if key in skipped:
            continue
        title, gen = plot_generators[key]
        fig = _plot(key, gen)
        if fig is not None:
            figures.append((title, fig))

    return figures


