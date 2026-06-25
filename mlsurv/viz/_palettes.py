"""Color palettes and figure styling for mlsurv visualizations."""

from typing import Dict, List, Optional, Any, Sequence, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import pandas as pd

FeatureGroupsType = Optional[Union[Dict[str, str], Sequence[str], "pd.Series"]]


# =============================================================================
# Manuscript-Ready Figure Defaults
# =============================================================================
# Standard sizing for publication-quality figures
# Single column: ~3.5" (88mm), Double column: ~7" (178mm)
# At scale=2 export, use half the target pixel width

MANUSCRIPT_DEFAULTS: Dict[str, Any] = {
    # Figure dimensions (pixels at scale=1, doubled on export)
    "width_single": 500,      # Single column figure
    "width_double": 800,      # Double column figure
    "height_small": 300,      # Compact height
    "height_medium": 400,     # Standard height
    "height_large": 500,      # Taller figures

    # Font sizes (pt)
    "font_size_title": 16,
    "font_size_axis": 12,
    "font_size_tick": 12,
    "font_size_legend": 10,
    "font_size_annotation": 12,

    # Line widths
    "line_width_main": 1.5,
    "line_width_secondary": 1,
    "line_width_reference": 1.5,

    # Marker sizes
    "marker_size_main": 6,
    "marker_size_small": 4,

    # Margins (pixels) - compact for manuscript
    "margin_l": 60,
    "margin_r": 15,
    "margin_t": 40,
    "margin_b": 50,
    "margin_pad": 4,

    # Export settings
    "export_scale": 2,  # 2x for high-DPI
}


def get_manuscript_layout(
    width: str = "single",
    height: str = "medium",
    title: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Get manuscript-ready layout configuration.

    .. deprecated::
        Use :func:`mlsurv.viz._theme.create_figure` instead.

    Parameters
    ----------
    width : str
        Figure width: 'single' or 'double'
    height : str
        Figure height: 'small', 'medium', or 'large'
    title : str, optional
        Figure title
    **kwargs
        Additional layout overrides

    Returns
    -------
    dict
        Plotly layout configuration
    """
    import warnings
    warnings.warn(
        "get_manuscript_layout() is deprecated. "
        "Use mlsurv.viz._theme.create_figure() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    d = MANUSCRIPT_DEFAULTS

    w = d["width_double"] if width == "double" else d["width_single"]
    h_map = {"small": d["height_small"], "medium": d["height_medium"], "large": d["height_large"]}
    h = h_map.get(height, d["height_medium"])

    layout = {
        "width": w,
        "height": h,
        "margin": dict(
            l=d["margin_l"],
            r=d["margin_r"],
            t=d["margin_t"],
            b=d["margin_b"],
            pad=d["margin_pad"],
        ),
        "font": dict(size=d["font_size_axis"], family="Arial, sans-serif"),
        "title": dict(
            text=title,
            x=0.5,
            font=dict(size=d["font_size_title"]),
        ) if title else None,
        "legend": dict(
            font=dict(size=d["font_size_legend"]),
            bgcolor="rgba(255,255,255,0.8)",
        ),
        "xaxis": dict(
            tickfont=dict(size=d["font_size_tick"]),
            title=dict(font=dict(size=d["font_size_axis"])),
        ),
        "yaxis": dict(
            tickfont=dict(size=d["font_size_tick"]),
            title=dict(font=dict(size=d["font_size_axis"])),
        ),
    }

    # Remove None values
    layout = {k: v for k, v in layout.items() if v is not None}

    # Apply overrides
    layout.update(kwargs)

    return layout

# Disease/cancer type colors (colorblind-friendly)
DISEASE_COLORS: Dict[str, str] = {
    "breast": "#E69F00",      # Orange
    "lung": "#56B4E9",        # Sky blue
    "prostate": "#009E73",    # Bluish green
    "colon": "#F0E442",       # Yellow
    "melanoma": "#0072B2",    # Blue
    "leukemia": "#D55E00",    # Vermillion
    "lymphoma": "#CC79A7",    # Reddish purple
    "pancreatic": "#999999",  # Gray
    "ovarian": "#882255",     # Wine
    "kidney": "#44AA99",      # Teal
    "default": "#666666",     # Default gray
}

# =============================================================================
# Named Palettes
# =============================================================================
# Okabe-Ito is the default — universally distinguishable for all common forms
# of color vision deficiency (deuteranopia, protanopia, tritanopia).
# Tableau 10 preserved as a named option for backward compatibility.

NAMED_PALETTES: Dict[str, Dict[str, str]] = {
    "okabe_ito": {
        "coxph":       "#0072B2",  # Blue
        "coxnet":      "#E69F00",  # Orange
        "rsf":         "#D55E00",  # Vermillion
        "gbsa":        "#009E73",  # Bluish green
        "fssvm":       "#CC79A7",  # Reddish purple
        "deepsurv":    "#56B4E9",  # Sky blue
        "deephit":     "#F0E442",  # Yellow
        "weibull_aft": "#000000",  # Black
        "xgbcox":     "#882255",  # Wine (Paul Tol extension)
        "ensemble":    "#44AA99",  # Teal (Paul Tol extension)
        "baseline":    "#999999",  # Gray
    },
    "tableau10": {
        "coxph":       "#1f77b4",
        "coxnet":      "#ff7f0e",
        "rsf":         "#2ca02c",
        "gbsa":        "#d62728",
        "fssvm":       "#9467bd",
        "deepsurv":    "#8c564b",
        "deephit":     "#aa4499",
        "weibull_aft": "#bcbd22",
        "xgbcox":     "#17becf",
        "ensemble":    "#e377c2",
        "baseline":    "#7f7f7f",
    },
}

# =============================================================================
# Module-level palette state + public API
# =============================================================================

_active_palette: Optional[Dict[str, str]] = None


def set_palette(palette: Union[str, Dict[str, str]]) -> None:
    """Set the active model color palette.

    Parameters
    ----------
    palette : str or dict
        Named palette (e.g. ``"okabe_ito"``, ``"tableau10"``) or a
        ``{model_name: hex_color}`` dict for custom overrides.

    Raises
    ------
    TypeError
        If *palette* is neither a string nor a dict.
    ValueError
        If *palette* is a string that doesn't match any named palette.
    """
    if not isinstance(palette, (str, dict)):
        raise TypeError(
            f"palette must be a str or dict, got {type(palette).__name__}"
        )
    global _active_palette
    if isinstance(palette, str):
        if palette not in NAMED_PALETTES:
            raise ValueError(
                f"Unknown palette '{palette}'. "
                f"Available: {sorted(NAMED_PALETTES)}"
            )
        _active_palette = NAMED_PALETTES[palette].copy()
    else:
        _active_palette = {k.lower(): v for k, v in palette.items()}


def reset_palette() -> None:
    """Reset to default model colors (no global override)."""
    global _active_palette
    _active_palette = None


def _resolve_palette(palette: Optional[Union[str, Dict[str, str]]]) -> Optional[Dict[str, str]]:
    """Resolve a per-call palette argument to a dict (or None)."""
    if palette is None:
        if _active_palette is not None:
            return _active_palette.copy()
        return None
    if isinstance(palette, str):
        if palette not in NAMED_PALETTES:
            raise ValueError(
                f"Unknown palette '{palette}'. "
                f"Available: {sorted(NAMED_PALETTES)}"
            )
        return NAMED_PALETTES[palette].copy()
    if not isinstance(palette, dict):
        raise TypeError(
            f"palette must be a str or dict, got {type(palette).__name__}"
        )
    return {k.lower(): v for k, v in palette.items()}


# =============================================================================
# Model Colors and Display Names (Lazy-Loading from Model Metadata)
# =============================================================================
# These dicts auto-populate from model class metadata on first access.
# This eliminates the need to manually update palettes when adding new models.


class _LazyDict(dict):
    """Dict that auto-populates via *loader* on first access."""

    def __init__(self, loader):
        super().__init__()
        self._loader = loader
        self._initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            self.update(self._loader())
            self._initialized = True

    def __contains__(self, key):
        self._ensure_initialized()
        return super().__contains__(key)

    def __getitem__(self, key):
        self._ensure_initialized()
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._ensure_initialized()
        return super().get(key, default)

    def items(self):
        self._ensure_initialized()
        return super().items()

    def keys(self):
        self._ensure_initialized()
        return super().keys()

    def values(self):
        self._ensure_initialized()
        return super().values()

    def __iter__(self):
        self._ensure_initialized()
        return super().__iter__()

    def __len__(self):
        self._ensure_initialized()
        return super().__len__()


def _load_model_colors():
    from mlsurv.models import get_model_colors
    return get_model_colors()


def _load_model_display_names():
    from mlsurv.models import get_model_display_names
    return get_model_display_names()


# Auto-populated from model class metadata on first access
MODEL_COLORS: Dict[str, str] = _LazyDict(_load_model_colors)
MODEL_DISPLAY_NAMES: Dict[str, str] = _LazyDict(_load_model_display_names)

# Reference comparison palette — visually distinct from model colors
# so that references are immediately distinguishable in benchmark plots.
REFERENCE_PALETTE: List[str] = [
    "#7570B3",  # Slate purple
    "#66A61E",  # Olive green
    "#E6AB02",  # Gold
    "#A6761D",  # Brown
    "#1B9E77",  # Teal
    "#D95F02",  # Burnt orange
    "#E7298A",  # Hot pink
    "#666666",  # Gray
]


def get_reference_color(name: str, index: int = 0) -> str:
    """Get a color for a reference by name or index.

    Cycles through ``REFERENCE_PALETTE`` when the index exceeds the
    palette length.

    Parameters
    ----------
    name : str
        Reference name (currently unused; reserved for future overrides).
    index : int
        Ordinal position among references.

    Returns
    -------
    str
        Hex colour code.
    """
    return REFERENCE_PALETTE[index % len(REFERENCE_PALETTE)]


# =============================================================================
# Subpopulation Palette
# =============================================================================
# Jewel-toned, colorblind-friendly colours for subpopulation KM curves and forest
# plots.  Medium-dark saturation sits between the saturated Okabe-Ito model
# palette and the muted feature-group palette.

SUBPOPULATION_PALETTE: List[str] = [
    "#7B2D8E",  # Amethyst
    "#1A7F6D",  # Jade
    "#C45B28",  # Burnt sienna
    "#2B65A3",  # Cobalt
    "#B5922D",  # Antique gold
]


def get_subpopulation_color(index: int) -> str:
    """Return a subpopulation colour by ordinal index (cycles)."""
    return SUBPOPULATION_PALETTE[index % len(SUBPOPULATION_PALETTE)]


# =============================================================================
# Qualitative Palette for Feature Groups
# =============================================================================
# Muted, colorblind-friendly colours for auto-assigning feature-group
# colours.  Designed for maximum pairwise contrast — each colour occupies
# a different hue band and stays clearly pastel/muted so it won't clash
# with the saturated Okabe-Ito palette used for MODEL_COLORS.

QUALITATIVE_PALETTE: List[str] = [
    "#c47ec0",  # Orchid
    "#7ec8c8",  # Aqua
    "#c9b643",  # Olive gold
    "#d4a76a",  # Caramel
    "#6a9fd4",  # Periwinkle
    "#d48787",  # Blush
    "#5db89e",  # Seafoam
    "#b8964e",  # Bronze
    "#9b8ec4",  # Wisteria
    "#c7cf5a",  # Chartreuse
    "#d97f4a",  # Terracotta
    "#6bc4d4",  # Sky
]


def _normalize_feature_groups(feature_groups, features):
    """Normalize *feature_groups* (dict, list, tuple, or Series) to a raw dict.

    Performs type dispatch and validation.  Returns a raw ``{feature: group}``
    dict that may contain non-string values.

    Parameters
    ----------
    feature_groups : dict, list, tuple, or Series
        Input feature-to-group mapping.
    features : list of str
        Feature names to match against.

    Returns
    -------
    dict
        ``{feature_name: group_value}`` (may contain non-string values).
    """
    import pandas as pd

    if isinstance(feature_groups, dict):
        feat_to_grp = {
            f: feature_groups[f]
            for f in features
            if f in feature_groups
        }
        if not feat_to_grp and feature_groups:
            import warnings

            from mlsurv._warnings import SuggestionWarning

            warnings.warn(
                f"feature_groups dict has {len(feature_groups)} entries but "
                f"none matched the {len(features)} feature names. Check for "
                f"naming mismatches.",
                SuggestionWarning,
                stacklevel=3,
            )
        return feat_to_grp
    if isinstance(feature_groups, pd.Series):
        vals = feature_groups.values.tolist()
        if len(vals) != len(features):
            raise ValueError(
                f"feature_groups Series length ({len(vals)}) != "
                f"features length ({len(features)})"
            )
        return dict(zip(features, vals))
    if isinstance(feature_groups, (list, tuple)):
        if len(feature_groups) != len(features):
            raise ValueError(
                f"feature_groups length ({len(feature_groups)}) != "
                f"features length ({len(features)})"
            )
        return dict(zip(features, feature_groups))
    raise TypeError(
        f"feature_groups must be a dict, list, or Series, "
        f"got {type(feature_groups).__name__}"
    )


def resolve_feature_group_colors(
    feature_groups: Union[Dict[str, str], Sequence[str], "pd.Series"],
    features: Sequence[str],
    group_colors: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Resolve feature→group mapping and group→color mapping.

    Parameters
    ----------
    feature_groups : dict, list, or Series
        - **dict** ``{feature_name: group_name}`` — features not in dict
          are excluded from group coloring.
        - **list** — parallel to *features* (same length required).
        - **Series** — index ignored; values treated like a list.
    features : sequence of str
        Feature names (e.g. from the DataFrame being plotted).
    group_colors : dict, optional
        Explicit ``{group_name: hex_color}`` overrides.

    Returns
    -------
    feature_to_group : dict
        ``{feature_name: group_name}`` for features that have a group.
    group_to_color : dict
        ``{group_name: hex_color}`` for every group.

    Raises
    ------
    ValueError
        If *feature_groups* is a list/Series whose length doesn't match
        *features*.
    TypeError
        If *feature_groups* is not a dict, list, or Series.
    """
    features = list(features)
    feat_to_grp = _normalize_feature_groups(feature_groups, features)

    # --- Filter out non-string group values (e.g. None, NaN, int) -----------
    import warnings

    from mlsurv._warnings import SuggestionWarning

    non_string = {f: g for f, g in feat_to_grp.items() if not isinstance(g, str)}
    if non_string:
        examples = list(non_string.items())[:5]
        warnings.warn(
            f"{len(non_string)} feature(s) have non-string group values and "
            f"will be excluded from group coloring: {examples}. "
            f"Ensure all group values are strings.",
            SuggestionWarning,
            stacklevel=2,
        )
    feat_to_grp = {
        f: g for f, g in feat_to_grp.items()
        if isinstance(g, str)
    }

    # --- Determine unique groups (sorted for deterministic colour assignment) -
    unique_groups = sorted(dict.fromkeys(feat_to_grp.values()))

    # --- Build group→color ---------------------------------------------------
    grp_to_color: Dict[str, str] = {}
    qi = 0  # index into QUALITATIVE_PALETTE

    for grp in unique_groups:
        if group_colors and grp in group_colors:
            grp_to_color[grp] = group_colors[grp]
        else:
            grp_to_color[grp] = QUALITATIVE_PALETTE[qi % len(QUALITATIVE_PALETTE)]
            qi += 1

    return feat_to_grp, grp_to_color


def _resolve_groups(
    feature_groups: FeatureGroupsType,
    features: Sequence[str],
    group_colors: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
    """Convenience wrapper: returns ``(None, None)`` when *feature_groups* is None.

    Avoids repeating the two-line guard clause at every call site.
    """
    if feature_groups is None:
        return None, None
    return resolve_feature_group_colors(feature_groups, features, group_colors)


def _build_discrete_colorscale(
    group_to_color: Dict[str, str],
) -> Tuple[List[str], Dict[str, int], List[List]]:
    """Build a Plotly discrete colorscale from a group-to-color mapping.

    Returns
    -------
    groups_ordered : list of str
        Group names in insertion order.
    group_to_idx : dict
        ``{group_name: integer_index}`` for encoding groups as Z values.
    colorscale : list of [float, str] pairs
        Plotly-compatible discrete colorscale.
    """
    if not group_to_color:
        return [], {}, []
    groups_ordered = list(group_to_color)
    n_grps = len(groups_ordered)
    group_to_idx = {g: i for i, g in enumerate(groups_ordered)}
    colorscale: List[List] = []
    for i, grp in enumerate(groups_ordered):
        lo = i / n_grps
        hi = (i + 1) / n_grps
        colorscale.append([lo, group_to_color[grp]])
        colorscale.append([hi, group_to_color[grp]])
    return groups_ordered, group_to_idx, colorscale


def _make_group_strip_trace(features, feat_to_grp, grp_to_color, orientation="vertical"):
    """Build a ``go.Heatmap`` trace encoding feature-group membership as a color strip.

    Parameters
    ----------
    features : list of str
        Feature names in display order.
    feat_to_grp : dict
        ``{feature_name: group_name}`` mapping.
    grp_to_color : dict
        ``{group_name: hex_color}`` mapping.
    orientation : {"vertical", "horizontal"}
        ``"vertical"`` produces a single-column strip (features on y-axis),
        ``"horizontal"`` produces a single-row strip (features on x-axis).

    Returns
    -------
    plotly.graph_objects.Heatmap
        Ready to add via ``fig.add_trace(trace, row=..., col=...)``.
    """
    import plotly.graph_objects as go

    # Build discrete colorscale, adding "Ungrouped" bucket when needed
    ungrouped = [f for f in features if f not in feat_to_grp]
    strip_colors = dict(grp_to_color)
    if ungrouped:
        strip_colors["Ungrouped"] = "#cccccc"
    _, group_to_idx, discrete_cs = _build_discrete_colorscale(strip_colors)

    if orientation == "horizontal":
        z = [[group_to_idx.get(feat_to_grp.get(f, "Ungrouped"), 0) for f in features]]
        return go.Heatmap(
            z=z, x=features, y=[""],
            colorscale=discrete_cs, showscale=False,
            hovertemplate="%{x}: %{customdata}<extra></extra>",
            customdata=[[feat_to_grp.get(f, "Ungrouped") for f in features]],
        )
    else:
        z = [[group_to_idx.get(feat_to_grp.get(f, "Ungrouped"), 0)] for f in features]
        return go.Heatmap(
            z=z, x=["Group"], y=features,
            colorscale=discrete_cs, showscale=False,
            hovertemplate="<b>%{y}</b><br>Group: %{customdata}<extra></extra>",
            customdata=[[feat_to_grp.get(f, "Ungrouped")] for f in features],
        )


def _add_group_legend_traces(fig, group_to_color, row=None, col=None):
    """Add invisible scatter traces for a group color legend.

    Uses ``legendgrouptitle`` so groups appear under a separate
    "Feature Group" header, visually distinct from model traces.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Target figure.
    group_to_color : dict
        ``{group_name: hex_color}`` mapping.
    row, col : int, optional
        Subplot coordinates (for ``make_subplots`` figures).
    """
    import plotly.graph_objects as go

    for i, (grp, color) in enumerate(group_to_color.items()):
        kw = dict(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color),
            name=grp, showlegend=True,
            legendgroup="_feature_groups_",
            hoverinfo="skip",
        )
        if i == 0:
            kw["legendgrouptitle"] = dict(text="Feature Group")
        if row is not None:
            fig.add_trace(go.Scatter(**kw), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(**kw))


def _narrow_groups(
    feat_to_grp: Optional[Dict[str, str]],
    grp_to_color: Optional[Dict[str, str]],
    feature_subset: Sequence[str],
) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
    """Restrict group mappings to only the features in *feature_subset*.

    Returns ``(None, None)`` when *feat_to_grp* is falsy.
    """
    if not feat_to_grp or not grp_to_color:
        return None, None
    narrowed = {f: feat_to_grp[f] for f in feature_subset if f in feat_to_grp}
    if not narrowed:
        return None, None
    active = set(narrowed.values())
    return narrowed, {g: c for g, c in grp_to_color.items() if g in active}


def _sort_features_by_group(
    feature_list: Sequence[str],
    feat_to_grp: Dict[str, str],
    grp_to_color: Dict[str, str],
    metric_values: Optional[Dict[str, float]] = None,
    reverse: bool = False,
) -> List[str]:
    """Reorder features: primary sort by group, secondary by metric within group.

    Groups are ordered by their first occurrence in *grp_to_color* (which
    preserves insertion order from :func:`resolve_feature_group_colors`).
    Within each group, features are sorted by *metric_values* descending
    (if provided) or preserve their original order.  Ungrouped features
    are appended at the end.

    Parameters
    ----------
    feature_list : sequence of str
        Features in their current order.
    feat_to_grp : dict
        ``{feature: group}`` mapping.
    grp_to_color : dict
        ``{group: color}`` mapping (defines group order).
    metric_values : dict, optional
        ``{feature: float}`` for within-group sorting (descending).
    reverse : bool, default False
        If ``True``, reverse the final order.  Useful for Plotly horizontal
        plots where ``categoryarray[0]`` maps to the bottom of the y-axis.

    Returns
    -------
    list of str
        Reordered feature names.
    """
    grouped: Dict[str, List[str]] = {}
    ungrouped: List[str] = []
    for feat in feature_list:
        grp = feat_to_grp.get(feat)
        if grp is not None:
            grouped.setdefault(grp, []).append(feat)
        else:
            ungrouped.append(feat)

    # Sort within each group by metric (descending) if provided
    if metric_values is not None:
        for grp in grouped:
            grouped[grp].sort(key=lambda f: metric_values.get(f, 0), reverse=True)

    ordered: List[str] = []
    for grp in grp_to_color:
        ordered.extend(grouped.pop(grp, []))
    # Features in groups not present in grp_to_color (e.g. typos)
    for remaining in grouped.values():
        ordered.extend(remaining)
    ordered.extend(ungrouped)
    return ordered[::-1] if reverse else ordered


def _add_group_separator_shapes(
    fig,
    feature_list: Sequence[str],
    feat_to_grp: Dict[str, str],
    axis: str = "y",
) -> None:
    """Add dotted horizontal (or vertical) lines between feature groups.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Target figure.
    feature_list : sequence of str
        Features in display order (bottom-to-top for y-axis).
    feat_to_grp : dict
        ``{feature: group}`` mapping.
    axis : {"y", "x"}
        Which axis features are on.
    """
    prev_grp = None
    for i, feat in enumerate(feature_list):
        grp = feat_to_grp.get(feat, "")
        if prev_grp is not None and grp != prev_grp:
            pos = i - 0.5
            if axis == "y":
                fig.add_hline(
                    y=pos, line_dash="dot",
                    line_color="lightgray", line_width=1,
                )
            else:
                fig.add_vline(
                    x=pos, line_dash="dot",
                    line_color="lightgray", line_width=1,
                )
        prev_grp = grp


def _iter_group_segments(
    feature_list: Sequence[str],
    feat_to_grp: Dict[str, str],
    grp_to_color: Dict[str, str],
    ungrouped_color: str = "#7f7f7f",
) -> List[Tuple[str, str, List[str]]]:
    """Return ``(group_name, color, features)`` segments for group-based traces.

    Groups appear in first-occurrence order. Features not in *feat_to_grp*
    are collected into a final ``"Ungrouped"`` segment (if any).
    """
    group_order = list(dict.fromkeys(
        feat_to_grp[f] for f in feature_list if f in feat_to_grp
    ))
    segments: List[Tuple[str, str, List[str]]] = []
    for grp in group_order:
        color = grp_to_color.get(grp, ungrouped_color)
        grp_feats = [f for f in feature_list if feat_to_grp.get(f) == grp]
        segments.append((grp, color, grp_feats))

    ungrouped = [f for f in feature_list if f not in feat_to_grp]
    if ungrouped:
        segments.append(("Ungrouped", ungrouped_color, ungrouped))
    return segments


# C-index colorscale (diverging from 0.5)
# Values: 0.0 (poor) -> 0.5 (random) -> 1.0 (perfect)
CINDEX_COLORSCALE: List[List] = [
    [0.0, "#d73027"],   # Red (poor)
    [0.4, "#fc8d59"],   # Light red
    [0.5, "#fee090"],   # Yellow (random)
    [0.6, "#91bfdb"],   # Light blue
    [0.7, "#4575b4"],   # Blue (good)
    [1.0, "#313695"],   # Dark blue (excellent)
]

# AUC colorscale (similar to C-index)
AUC_COLORSCALE: List[List] = [
    [0.0, "#d73027"],   # Red (poor)
    [0.5, "#fee090"],   # Yellow (random)
    [0.75, "#91bfdb"],  # Light blue
    [1.0, "#313695"],   # Dark blue (excellent)
]

# Risk quartile colors (for KM curves)
RISK_QUARTILE_COLORS: Dict[str, str] = {
    "Q1 (Low)": "#2166ac",    # Blue (good prognosis)
    "Q2": "#67a9cf",          # Light blue
    "Q3": "#ef8a62",          # Light red
    "Q4 (High)": "#b2182b",   # Red (poor prognosis)
}

def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert hex color to rgba string.

    Parameters
    ----------
    hex_color : str
        Hex color code (e.g., '#1f77b4')
    alpha : float
        Alpha transparency value (0.0 to 1.0)

    Returns
    -------
    str
        RGBA color string (e.g., 'rgba(31, 119, 180, 0.2)')
    """
    h = hex_color.lstrip('#')
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    """Parse a ``#rrggbb`` hex string into an ``(r, g, b)`` int tuple."""
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Format an ``(r, g, b)`` int tuple as a ``#rrggbb`` hex string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def get_model_color(
    model_name: str,
    palette: Optional[Union[str, Dict[str, str]]] = None,
) -> str:
    """Get color for a model type.

    Override chain (first match wins):

    1. Per-call *palette* (dict or named string).
    2. Module-level ``_active_palette`` (set via :func:`set_palette`).
    3. ``MODEL_COLORS`` defaults (loaded from model class metadata).

    Parameters
    ----------
    model_name : str
        Model name (e.g., 'coxph', 'rsf').
    palette : str or dict, optional
        Per-call palette override — a named palette string or a
        ``{model_name: hex_color}`` dict.

    Returns
    -------
    str
        Hex color code
    """
    name_lower = model_name.lower()

    # Strip recalibration/tuning suffixes and resolve via the base name.
    # Without this, models whose display name and internal key differ
    # (e.g. "XGBoost-AFT" / "xgbaft") can't be matched after suffixing —
    # the reverse-display and partial-match steps below both fail because
    # "xgbaft" isn't a substring of "xgboost-aft".
    for suffix in (' (raw)', ' (recalibrated)', ' (tuned)'):
        if name_lower.endswith(suffix):
            base = model_name[: -len(suffix)]
            return get_model_color(base, palette=palette)

    # Resolve palette override (per-call > module-level)
    resolved = _resolve_palette(palette)
    if resolved is not None and name_lower in resolved:
        return resolved[name_lower]

    # Fall through to MODEL_COLORS defaults
    if name_lower in MODEL_COLORS:
        return MODEL_COLORS[name_lower]

    # Reverse display-name lookup (e.g. "XGBoost-Cox" → "xgbcox")
    for key, display in MODEL_DISPLAY_NAMES.items():
        if display == model_name:
            if key in MODEL_COLORS:
                return MODEL_COLORS[key]

    # Partial matches
    for key, color in MODEL_COLORS.items():
        if key in name_lower or name_lower in key:
            return color

    return MODEL_COLORS.get("baseline", "#999999")


def build_benchmark_palette(
    model_names: List[str],
    reference_names: List[str],
) -> Dict[str, str]:
    """Return ``{name: color}`` for every model and reference.

    Builds a fresh palette dictionary using :func:`get_model_color` for
    models and :func:`get_reference_color` for references.  Callers that
    need to honour user overrides should merge afterward::

        base = build_benchmark_palette(model_names, ref_names)
        base.update(user_palette)   # user overrides win

    Parameters
    ----------
    model_names : list[str]
        Model identifiers (e.g. ``BenchmarkResult.model_names``).
    reference_names : list[str]
        Reference identifiers (e.g. ``BenchmarkResult.reference_names``).

    Returns
    -------
    dict[str, str]
        Mapping from name to hex color code.

    Notes
    -----
    If a name appears in both *model_names* and *reference_names*, the
    reference color wins (references are assigned second).
    """
    palette: Dict[str, str] = {}
    for name in model_names:
        palette[name] = get_model_color(name)
    for i, name in enumerate(reference_names):
        palette[name] = get_reference_color(name, i)
    return palette


def get_model_display_name(model_name: str) -> str:
    """Get display name for a model type.

    Accepts both internal names (``'xgbcox'``) and display names
    (``'XGBoost-Cox'``).  Display names pass through unchanged to
    avoid double-mapping (e.g. ``'XGBoost-Cox'`` → ``'XGBOOST-COX'``).

    Parameters
    ----------
    model_name : str
        Model name (e.g., 'coxph', 'rsf', or already-formatted 'CoxPH')

    Returns
    -------
    str
        Properly formatted display name (e.g., 'CoxPH', 'RSF')
    """
    name_lower = model_name.lower()
    # Idempotent passthrough for already-resolved " (tuned)" strings such as
    # "RSF (tuned)" — strip the suffix, resolve the base display, re-append.
    if name_lower.endswith(' (tuned)'):
        base_display_in = model_name[: -len(' (tuned)')]
        known_display = set(MODEL_DISPLAY_NAMES.values())
        if base_display_in in known_display:
            return f"{base_display_in} (tuned)"
        return f"{get_model_display_name(base_display_in)} (tuned)"
    # Same idempotent passthrough for the recalibration variants used by the
    # recalibration report section: "<base> (raw)" and "<base> (recalibrated)".
    for suffix in (' (raw)', ' (recalibrated)'):
        if name_lower.endswith(suffix):
            base_in = model_name[: -len(suffix)]
            return f"{get_model_display_name(base_in)}{suffix}"
    # Direct lookup by internal name
    if name_lower in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[name_lower]
    # Already a known display name — pass through unchanged
    known_display = set(MODEL_DISPLAY_NAMES.values())
    if model_name in known_display:
        return model_name
    return model_name.upper()


def get_disease_color(disease: str) -> str:
    """Get color for a disease type.

    Parameters
    ----------
    disease : str
        Disease name (e.g., 'breast', 'lung')

    Returns
    -------
    str
        Hex color code
    """
    # Normalize disease name
    name_lower = disease.lower()

    # Check direct match
    if name_lower in DISEASE_COLORS:
        return DISEASE_COLORS[name_lower]

    # Check partial matches
    for key, color in DISEASE_COLORS.items():
        if key in name_lower or name_lower in key:
            return color

    return DISEASE_COLORS.get("default", "#666666")


def get_risk_colors(n_groups: int = 4) -> List[str]:
    """Get colors for risk groups.

    Parameters
    ----------
    n_groups : int
        Number of risk groups (2-4)

    Returns
    -------
    List[str]
        List of hex color codes from low to high risk
    """
    if n_groups == 2:
        return ["#2166ac", "#b2182b"]
    elif n_groups == 3:
        return ["#2166ac", "#fee090", "#b2182b"]
    elif n_groups == 4:
        return list(RISK_QUARTILE_COLORS.values())
    else:
        # Generate gradient for other values without importing matplotlib.
        anchors = ["#2166ac", "#fee090", "#b2182b"]

        def _lerp(a: int, b: int, t: float) -> int:
            return int(round(a + (b - a) * t))

        anchor_rgbs = [_hex_to_rgb(c) for c in anchors]
        out = []
        for i in range(n_groups):
            pos = i / (n_groups - 1)
            if pos <= 0.5:
                t = pos / 0.5
                c0, c1 = anchor_rgbs[0], anchor_rgbs[1]
            else:
                t = (pos - 0.5) / 0.5
                c0, c1 = anchor_rgbs[1], anchor_rgbs[2]
            rgb = tuple(_lerp(c0[j], c1[j], t) for j in range(3))
            out.append(_rgb_to_hex(rgb))
        return out

