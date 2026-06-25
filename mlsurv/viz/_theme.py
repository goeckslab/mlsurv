"""Centralized theme and figure helpers for mlsurv visualizations."""

from typing import Optional, Union

import plotly.graph_objects as go
import plotly.io as pio

from mlsurv.viz._palettes import MANUSCRIPT_DEFAULTS, hex_to_rgba


# =============================================================================
# Plotly Template — registered at import time, never set as global default
# =============================================================================

_d = MANUSCRIPT_DEFAULTS
_MLSURV_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=_d["font_size_axis"]),
        title=dict(x=0.5, xanchor="center", font=dict(size=_d["font_size_title"])),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(200,200,200,0.5)",
            showline=False,
            ticks="",
            zeroline=False,
            tickfont=dict(size=_d["font_size_tick"]),
            title=dict(font=dict(size=_d["font_size_axis"])),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(200,200,200,0.5)",
            showline=False,
            ticks="",
            zeroline=False,
            tickfont=dict(size=_d["font_size_tick"]),
            title=dict(font=dict(size=_d["font_size_axis"])),
        ),
        legend=dict(
            font=dict(size=_d["font_size_legend"]),
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(
            l=_d["margin_l"], r=_d["margin_r"],
            t=_d["margin_t"], b=_d["margin_b"], pad=_d["margin_pad"],
        ),
    )
)
del _d

pio.templates["mlsurv_minimal"] = _MLSURV_TEMPLATE

# Named size shortcuts → pixel values
_SIZE_MAP = {
    "single": MANUSCRIPT_DEFAULTS["width_single"],   # 500
    "double": MANUSCRIPT_DEFAULTS["width_double"],   # 800
    "small": MANUSCRIPT_DEFAULTS["height_small"],    # 300
    "medium": MANUSCRIPT_DEFAULTS["height_medium"],  # 400
    "large": MANUSCRIPT_DEFAULTS["height_large"],    # 500
}


# =============================================================================
# create_figure — replaces the 8+ line boilerplate in every viz function
# =============================================================================

def create_figure(
    title: Optional[str] = None,
    width: Union[str, int] = "single",
    height: Union[str, int] = "medium",
    **overrides,
) -> go.Figure:
    """Create a themed ``go.Figure`` with manuscript-ready defaults.

    Parameters
    ----------
    title : str, optional
        Figure title.  Rendered as ``<b>title</b>`` for bold centering.
    width : str or int
        Named size (``"single"``, ``"double"``) or pixel value.
    height : str or int
        Named size (``"small"``, ``"medium"``, ``"large"``) or pixel value.
    **overrides
        Any extra ``fig.update_layout(...)`` keyword arguments.

    Returns
    -------
    go.Figure
    """
    def _resolve(value, param):
        if isinstance(value, str):
            if value not in _SIZE_MAP:
                raise ValueError(
                    f"Unknown {param} '{value}'. "
                    f"Must be one of: {sorted(_SIZE_MAP)} or an integer."
                )
            return _SIZE_MAP[value]
        return value

    w = _resolve(width, "width")
    h = _resolve(height, "height")

    layout_kwargs = dict(
        template="mlsurv_minimal",
        width=w,
        height=h,
    )

    if title is not None:
        layout_kwargs["title"] = dict(text=f"<b>{title}</b>")
        # Add top margin for multi-line titles (subtitles via <br>)
        if "<br>" in title:
            layout_kwargs["margin"] = dict(t=MANUSCRIPT_DEFAULTS["margin_t"] + 45)

    layout_kwargs.update(overrides)

    fig = go.Figure()
    fig.update_layout(**layout_kwargs)
    return fig


def apply_size_overrides(
    fig: go.Figure,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """Apply user width/height overrides to a figure (in-place)."""
    overrides = {}
    if width is not None:
        overrides["width"] = width
    if height is not None:
        overrides["height"] = height
    if overrides:
        fig.update_layout(**overrides)


def adaptive_size(
    n: int,
    *,
    per_item: int = 30,
    base: int = 100,
    min_size: int = 300,
    max_size: int = 1200,
) -> int:
    """Compute a figure dimension that scales with the number of items.

    Parameters
    ----------
    n : int
        Number of variable-length items (features, models, rows, etc.).
    per_item : int
        Pixels added per item.
    base : int
        Fixed pixel baseline before per-item scaling.
    min_size : int
        Minimum returned value (floor clamp).
    max_size : int
        Maximum returned value (ceiling clamp).

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If ``min_size`` exceeds ``max_size``.
    """
    if min_size > max_size:
        raise ValueError(
            f"min_size ({min_size}) must be <= max_size ({max_size})."
        )
    n = max(0, n)
    return max(min_size, min(max_size, base + n * per_item))


def truncate_labels(names, max_chars=30):
    """Truncate long labels and return ``(display_names, full_names)``.

    Parameters
    ----------
    names : list of str
        Original label strings.
    max_chars : int
        Maximum characters before truncation. Labels exceeding this
        are shortened to ``name[:max_chars-3] + "..."``.

    Returns
    -------
    display_names : list of str
        Truncated labels for axis tick text.
    full_names : list of str
        Original labels (for hover / tickvals).

    Raises
    ------
    ValueError
        If ``max_chars`` is less than 4 (too short to leave a visible
        character plus the ``"..."`` ellipsis).
    """
    if max_chars < 4:
        raise ValueError(
            f"max_chars must be >= 4 (at least 1 visible char + '...'), "
            f"got {max_chars}."
        )
    full_names = [str(n) for n in names]
    display_names = [
        (n[:max_chars - 3] + "...") if len(n) > max_chars else n
        for n in full_names
    ]
    return display_names, full_names


def margin_for_labels(names, *, per_char=7, padding=10, min_margin=60,
                      max_margin=250):
    """Compute adaptive margin from the longest display label.

    Parameters
    ----------
    names : list of str
        Display labels (already truncated if applicable).
    per_char : int
        Pixels per character estimate.
    padding : int
        Extra padding pixels.
    min_margin : int
        Floor clamp.
    max_margin : int
        Ceiling clamp (safety net after truncation).

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If ``min_margin`` exceeds ``max_margin``.
    """
    if min_margin > max_margin:
        raise ValueError(
            f"min_margin ({min_margin}) must be <= max_margin ({max_margin})."
        )
    if not names:
        return min_margin
    longest = max(len(str(n)) for n in names)
    return max(min_margin, min(max_margin, longest * per_char + padding))


# =============================================================================
# Trace helpers
# =============================================================================

def add_ci_ribbon(
    fig: go.Figure,
    x,
    upper,
    lower,
    color: str,
    *,
    alpha: float = 0.2,
    row: Optional[int] = None,
    col: Optional[int] = None,
    name: str = "",
    legendgroup: Optional[str] = None,
) -> None:
    """Add a confidence-interval ribbon (``fill='toself'``) to *fig*.

    Parameters
    ----------
    fig : go.Figure
    x : array-like
        Shared x-values.
    upper : array-like
        Upper bound of the ribbon.
    lower : array-like
        Lower bound of the ribbon.
    color : str
        Hex color (e.g. ``"#1f77b4"``).
    alpha : float
        Fill opacity (default 0.2).
    row : int, optional
        Subplot row position (pass both *row* and *col* or neither).
    col : int, optional
        Subplot column position (pass both *row* and *col* or neither).
    name : str
        Trace name (default hidden from legend).
    legendgroup : str, optional
        Legend group to link with a parent trace for toggle.
    """
    fill_color = hex_to_rgba(color, alpha) if "#" in color else color

    add_kw = {}
    if row is not None and col is not None:
        add_kw = dict(row=row, col=col)

    # Use two-trace tonexty approach instead of single toself polygon —
    # avoids white fringe artifacts from SVG anti-aliasing on polygon edges.
    # Lower bound (invisible line)
    fig.add_trace(go.Scatter(
        x=list(x), y=list(lower),
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
        legendgroup=legendgroup,
    ), **add_kw)
    # Upper bound with fill down to lower
    fig.add_trace(go.Scatter(
        x=list(x), y=list(upper),
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor=fill_color,
        showlegend=False,
        hoverinfo="skip",
        name=name,
        legendgroup=legendgroup,
    ), **add_kw)


class OrientedAxes:
    """Encapsulates orientation-dependent axis mappings.

    Eliminates repetitive ``if horiz:`` branching in metric renderers.
    """

    __slots__ = ("is_horiz", "val", "cat", "error_key")

    def __init__(self, orientation: str):
        self.is_horiz = orientation == "horizontal"
        self.val = "x" if self.is_horiz else "y"
        self.cat = "y" if self.is_horiz else "x"
        self.error_key = "error_x" if self.is_horiz else "error_y"

    def scatter_kw(self, val, cat):
        """Return ``{val_axis: val, cat_axis: cat}``."""
        return {self.val: val, self.cat: cat}

    def bar_orientation(self):
        """Return ``"h"`` or ``"v"``."""
        return "h" if self.is_horiz else "v"

    def add_reference(self, fig, ref_val, label=None):
        """Add a dashed reference line on the *value* axis."""
        if self.is_horiz:
            add_reference_vline(fig, x=ref_val, label=label)
        else:
            add_reference_hline(fig, y=ref_val, label=label)

    def model_ref(self):
        """Hover placeholder for the categorical axis."""
        return "%{y}" if self.is_horiz else "%{x}"

    def val_ref(self, fmt=".3f"):
        """Hover placeholder for the value axis."""
        return f"%{{{self.val}:{fmt}}}"

    def value_axis_layout(self, title, range_vals, **extra):
        """Return layout dict for the value axis."""
        key = "xaxis" if self.is_horiz else "yaxis"
        return {f"{key}_title": title, key: {"range": range_vals, **extra}}

    def cat_axis_layout(self, title="", **extra):
        """Return layout dict for the categorical axis."""
        key = "yaxis" if self.is_horiz else "xaxis"
        return {f"{key}_title": title, key: extra}


def add_reference_hline(
    fig: go.Figure,
    y: float,
    label: Optional[str] = None,
) -> None:
    """Add a dashed gray horizontal reference line."""
    kwargs = dict(y=y, line_dash="dash", line_color="gray", line_width=1.5)
    if label:
        kwargs.update(
            annotation_text=label,
            annotation_position="bottom left",
            annotation_font_size=MANUSCRIPT_DEFAULTS["font_size_annotation"],
        )
    fig.add_hline(**kwargs)


def add_reference_vline(
    fig: go.Figure,
    x: float,
    label: Optional[str] = None,
) -> None:
    """Add a dashed gray vertical reference line."""
    kwargs = dict(x=x, line_dash="dash", line_color="gray", line_width=1.5)
    if label:
        kwargs.update(
            annotation_text=label,
            annotation_position="top right",
            annotation_font_size=MANUSCRIPT_DEFAULTS["font_size_annotation"],
            annotation_textangle=90,
        )
    fig.add_vline(**kwargs)
