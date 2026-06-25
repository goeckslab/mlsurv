"""Pipeline flowchart visualization for reports."""

import plotly.graph_objects as go
from typing import Optional, List


# Colors for pipeline components
COLORS = {
    'preprocessing': '#e3f2fd',  # Light blue
    'models': '#e8f5e9',         # Light green
    'evaluation': '#fff3e0',     # Light orange
    'tuned_badge': '#c8e6c9',    # Green
    'untuned_badge': '#eeeeee',  # Gray
    'border': '#666666',
    'disabled_border': '#cccccc',
    'disabled_fill': '#f5f5f5',
    'text': '#333333',
    'disabled_text': '#999999',
    'arrow': '#666666',
}


def pipeline_flowchart(
    imputer: Optional[str],
    scaler: Optional[str],
    selector: Optional[str],
    models: List[str],
    show_bracket: bool = True,
    bracket_label: Optional[str] = None,
    has_bootstrap: bool = False,
    n_bootstrap: Optional[int] = None,
) -> go.Figure:
    """
    Create a Plotly flowchart showing the data analysis pipeline.

    Parameters
    ----------
    imputer : str or None
        Imputer description, e.g., "SimpleImputer(median)" or None if disabled.
    scaler : str or None
        Scaler description, e.g., "StandardScaler" or None if disabled.
    selector : str or None
        Feature selector description, e.g., "TopKCoxSelector(k='auto')" or None.
    models : list of str
        List of model names, e.g., ["CoxPH", "RSF", "GBT"].
    show_bracket : bool, default True
        Whether to show the CV/tuning bracket below the pipeline.
    bracket_label : str, optional
        Label for the bracket (e.g., "Cross Validation", "Cross Validation & Tuning").
    has_bootstrap : bool, default False
        Whether bootstrap evaluation was performed.
    n_bootstrap : int, optional
        Number of bootstrap iterations.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive pipeline flowchart.
    """
    fig = go.Figure()

    # Layout constants
    node_height = 60
    node_width = 140
    y_center = 100
    x_positions = [50, 220, 390, 560, 730]  # Imputer, Scaler, Selector, Models, Evaluation

    # Calculate Models box height based on number of models (extra padding at bottom)
    models_height = max(node_height, 36 + len(models) * 14)  # Increased from 30

    # Create nodes: (title, content, x_pos, fill_color, height)
    nodes = [
        ('Imputation', imputer, x_positions[0], COLORS['preprocessing'], node_height),
        ('Scaling', scaler, x_positions[1], COLORS['preprocessing'], node_height),
        ('Feature Selection', selector, x_positions[2], COLORS['preprocessing'], node_height),
        ('Models', _format_models(models), x_positions[3], COLORS['models'], models_height),
        ('Evaluation', _format_evaluation(has_bootstrap, n_bootstrap), x_positions[4], COLORS['evaluation'], node_height),
    ]

    for title, content, x_pos, fill_color, height in nodes:
        is_disabled = content is None and title != 'Models' and title != 'Evaluation'
        _add_node(
            fig, x_pos, y_center, node_width, height,
            title, content, fill_color, is_disabled
        )

    # Add arrows between nodes
    for i in range(len(x_positions) - 1):
        _add_arrow(
            fig,
            x_positions[i] + node_width/2, y_center,
            x_positions[i+1] - node_width/2, y_center
        )

    # Add horizontal CV/tuning bracket below pipeline (Imputation through Models)
    if show_bracket and bracket_label:
        bracket_y = y_center - models_height/2 - 15
        bracket_x_start = x_positions[0]  # Imputation
        bracket_x_end = x_positions[3]    # Models
        tick_height = 8
        bracket_color = COLORS['border']  # Match box outline color

        # Horizontal line
        fig.add_shape(
            type="line",
            x0=bracket_x_start, y0=bracket_y,
            x1=bracket_x_end, y1=bracket_y,
            line=dict(color=bracket_color, width=2),
        )
        # Left vertical tick
        fig.add_shape(
            type="line",
            x0=bracket_x_start, y0=bracket_y,
            x1=bracket_x_start, y1=bracket_y + tick_height,
            line=dict(color=bracket_color, width=2),
        )
        # Right vertical tick
        fig.add_shape(
            type="line",
            x0=bracket_x_end, y0=bracket_y,
            x1=bracket_x_end, y1=bracket_y + tick_height,
            line=dict(color=bracket_color, width=2),
        )

        # Bracket label text centered below bracket
        fig.add_annotation(
            x=(bracket_x_start + bracket_x_end) / 2,
            y=bracket_y - 12,
            text=f"<b>{bracket_label}</b>",
            showarrow=False,
            font=dict(size=10, color=bracket_color),
        )

    # Configure layout — flowchart needs hidden axes, so override template grid
    fig.update_layout(
        template="mlsurv_minimal",
        title=dict(
            text="<b>Analysis Pipeline</b>",
            y=0.95,
            font=dict(size=16),
        ),
        showlegend=False,
        xaxis=dict(visible=False, range=[-40, 820], showgrid=False),
        yaxis=dict(visible=False, range=[0, 200], scaleanchor="x",
                   scaleratio=1, showgrid=False),
        width=850,
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def _add_node(
    fig: go.Figure,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    content: Optional[str],
    fill_color: str,
    is_disabled: bool = False,
) -> None:
    """Add a rounded rectangle node with title and content."""
    # Determine styling based on disabled state
    if is_disabled:
        actual_fill = COLORS['disabled_fill']
        border_color = COLORS['disabled_border']
        text_color = COLORS['disabled_text']
        line_dash = 'dash'
        display_content = "Disabled"
    else:
        actual_fill = fill_color
        border_color = COLORS['border']
        text_color = COLORS['text']
        line_dash = 'solid'
        display_content = content if content else ""

    # Add rounded rectangle
    fig.add_shape(
        type="rect",
        x0=x - width/2,
        y0=y - height/2,
        x1=x + width/2,
        y1=y + height/2,
        fillcolor=actual_fill,
        line=dict(color=border_color, width=2, dash=line_dash),
    )

    # Title text (bold, at top of node) - fixed offset from top
    fig.add_annotation(
        x=x,
        y=y + height/2 - 12,  # Fixed 12px from top edge
        text=f"<b>{title}</b>",
        showarrow=False,
        font=dict(size=11, color=text_color),
    )

    # Content text (smaller, below title) - anchored at top so multi-line content flows down
    fig.add_annotation(
        x=x,
        y=y + height/2 - 28,  # 16px below title
        text=display_content,
        showarrow=False,
        font=dict(size=9, color=text_color),
        align='center',
        yanchor='top',  # Anchor at top so text flows downward
    )


def _add_arrow(
    fig: go.Figure,
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
) -> None:
    """Add an arrow annotation between two points."""
    fig.add_annotation(
        x=x_end,
        y=y_end,
        ax=x_start,
        ay=y_start,
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor=COLORS['arrow'],
    )


def _format_models(models: List[str]) -> str:
    """Format model list for display."""
    if not models:
        return "None"
    return "<br>".join(models)


def _format_evaluation(has_bootstrap: bool, n_bootstrap: Optional[int]) -> str:
    """Format evaluation box content."""
    lines = ["Test Set"]
    if has_bootstrap:
        if n_bootstrap:
            lines.append(f"Bootstrap (n={n_bootstrap})")
        else:
            lines.append("Bootstrap")
    return "<br>".join(lines)
