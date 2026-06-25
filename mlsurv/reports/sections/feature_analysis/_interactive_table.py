"""Interactive HTML table rendering for feature analysis.

Provides the two-dimensional toggle table used by the feature importance
section, with model × column-type filtering and expandable rows.
"""

import html as html_module
from typing import Dict, Iterable, List, Optional, Tuple

from mlsurv.reports.sections.base import BaseSection
from mlsurv.viz._palettes import get_model_display_name

from ._helpers import _short_column_name

# ── Column-type classification constants ──────────────────────────────────────

_COLUMN_TYPE_PREFIXES = (
    ('PFI', 'pfi'),
    ('p (adj.)', 'significance'),
    ('Sig.', 'significance'),
    ('Coef.', 'coefficients'),
    ('HR (mv)', 'coefficients'),
    ('p (mv)', 'coefficients'),
)

_COLUMN_TYPE_EXACT = {
    'p-value': 'univariate-cox',
    'HR': 'univariate-cox',
    'Effect': 'univariate-cox',
    'Mean |SHAP|': 'shap',
}

_MODEL_INDEPENDENT_GROUPS = frozenset({'feature', 'univariate-cox', 'shap'})

_TYPE_DISPLAY_NAMES = {
    'univariate-cox': 'Univariate Cox',
    'pfi': 'PFI',
    'significance': 'Significance',
    'coefficients': 'Coefficients',
    'shap': 'SHAP',
}


_slugify = BaseSection._slugify


def _classify_column_type(col_name: str) -> Optional[str]:
    """Return the type slug for a column name."""
    if col_name in _COLUMN_TYPE_EXACT:
        return _COLUMN_TYPE_EXACT[col_name]
    for prefix, type_slug in _COLUMN_TYPE_PREFIXES:
        if col_name.startswith(prefix):
            return type_slug
    return None


# ── Column group builders ─────────────────────────────────────────────────────


def build_single_model_column_groups(
    table_df, model_name: str,
    coef_models: Optional[Iterable[str]] = None,
) -> List[Tuple]:
    """Build column group descriptors for a single-model table.

    Returns list of (group_id, group_label, [columns], sticky?) tuples.
    Each model gets a single group spanning PFI + significance + coefficients.
    *coef_models* names the models with coefficient columns in the table.
    """
    display = get_model_display_name(model_name)
    cols = set(table_df.columns)

    groups: List[Tuple] = [("feature", "Feature", ["Feature"], True)]

    ucox_cols = [c for c in ["p-value", "HR", "Effect"] if c in cols]
    if ucox_cols:
        groups.append(("univariate-cox", "Univariate Cox", ucox_cols))

    slug = _slugify(display)
    model_cols = [c for c in [f"PFI ({display})", "p (adj.)", "Sig."] if c in cols]
    if coef_models:
        for m_name in coef_models:
            m_display = get_model_display_name(m_name)
            model_cols.extend(
                c for c in [
                    f"Coef. ({m_display})",
                    f"HR (mv) ({m_display})",
                    f"p (mv) ({m_display})",
                ]
                if c in cols
            )
    if model_cols:
        groups.append((slug, display, model_cols))

    if "Mean |SHAP|" in cols:
        groups.append(("shap", "SHAP", ["Mean |SHAP|"]))

    return groups


def build_multi_model_column_groups(
    table_df, models,
    coef_models: Optional[Iterable[str]] = None,
) -> List[Tuple]:
    """Build column group descriptors for a multi-model table.

    Each model gets a single group spanning PFI + significance + coefficients.
    *coef_models* names the models with coefficient columns in the table.
    """
    cols = set(table_df.columns)
    coef_models = set(coef_models or ())

    groups: List[Tuple] = [("feature", "Feature", ["Feature"], True)]

    ucox_cols = [c for c in ["p-value", "HR", "Effect"] if c in cols]
    if ucox_cols:
        groups.append(("univariate-cox", "Univariate Cox", ucox_cols))

    for model_name in models:
        display = get_model_display_name(model_name)
        slug = _slugify(display)

        model_cols = [
            c for c in [
                f"PFI ({display})",
                f"p (adj.) ({display})",
                f"Sig. ({display})",
            ]
            if c in cols
        ]
        if model_name in coef_models:
            model_cols.extend(
                c for c in [
                    f"Coef. ({display})",
                    f"HR (mv) ({display})",
                    f"p (mv) ({display})",
                ]
                if c in cols
            )
        if model_cols:
            groups.append((slug, display, model_cols))

    return groups


# ── Rendering pipeline ────────────────────────────────────────────────────────


def _normalize_column_groups(
    column_groups: List[Tuple], live_cols: set,
) -> List[Tuple]:
    """Filter dropped columns and normalize to consistent 4-tuples."""
    groups = []
    for gid, label, cols, *rest in column_groups:
        kept = [c for c in cols if c in live_cols]
        if kept:
            groups.append((gid, label, kept, bool(rest and rest[0])))
    return groups


def _build_column_metadata(
    groups: List[Tuple],
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """Build per-column metadata: (model_slug | None, type_slug | None)."""
    col_meta: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for gid, _label, cols, sticky in groups:
        if sticky:
            continue
        model = None if gid in _MODEL_INDEPENDENT_GROUPS else gid
        for col in cols:
            col_meta[col] = (model, _classify_column_type(col))
    return col_meta


def _collect_toggle_dimensions(
    col_meta: Dict[str, Tuple[Optional[str], Optional[str]]],
) -> Tuple[List[str], List[str]]:
    """Return ordered lists of unique model slugs and type slugs for toggles."""
    models_seen: List[str] = []
    types_seen: List[str] = []
    models_set: set = set()
    types_set: set = set()
    for model, ctype in col_meta.values():
        if model and model not in models_set:
            models_seen.append(model)
            models_set.add(model)
        if ctype and ctype not in types_set:
            types_seen.append(ctype)
            types_set.add(ctype)
    return models_seen, types_seen


def _render_toggle_controls(
    groups: List[Tuple],
    models_seen: List[str],
    types_seen: List[str],
) -> List[str]:
    """Render two-dimensional toggle checkbox HTML."""
    if not models_seen and not types_seen:
        return []
    lines = ['<div class="feature-table-controls">']
    if models_seen:
        lines.append('<span class="ft-control-label">Models:</span>')
        for m in models_seen:
            safe_m = html_module.escape(m, quote=True)
            display = next(
                (label for gid, label, _, _ in groups if gid == m), m
            )
            safe_label = html_module.escape(display)
            lines.append(
                f'<label class="feature-group-toggle">'
                f'<input type="checkbox" data-ft-model="{safe_m}" checked> {safe_label}'
                f'</label>'
            )
    if models_seen and types_seen:
        lines.append('<span class="ft-control-separator"></span>')
    if types_seen:
        lines.append('<span class="ft-control-label">Columns:</span>')
        for t in types_seen:
            safe_t = html_module.escape(t, quote=True)
            safe_label = html_module.escape(_TYPE_DISPLAY_NAMES.get(t, t))
            lines.append(
                f'<label class="feature-group-toggle">'
                f'<input type="checkbox" data-ft-type="{safe_t}" checked> {safe_label}'
                f'</label>'
            )
    lines.append('</div>')
    return lines


def _render_thead(
    groups: List[Tuple],
    col_meta: Dict[str, Tuple[Optional[str], Optional[str]]],
) -> List[str]:
    """Render group header + column header rows."""
    lines = ['<thead>']

    # Group header row
    lines.append('<tr class="group-header-row">')
    for gid, label, cols, sticky in groups:
        safe_gid = html_module.escape(gid, quote=True)
        safe_label = html_module.escape(label)
        if sticky:
            lines.append(f'<th rowspan="2" class="sticky-col">{safe_label}</th>')
        else:
            model_attr = ''
            if gid not in _MODEL_INDEPENDENT_GROUPS:
                model_attr = f' data-ft-model="{safe_gid}"'
            lines.append(
                f'<th colspan="{len(cols)}" data-ft-group-id="{safe_gid}"{model_attr}>'
                f'{safe_label}</th>'
            )
    lines.append('</tr>')

    # Column header row
    col_idx = 0
    lines.append('<tr class="column-header-row">')
    for gid, _label, cols, sticky in groups:
        if sticky:
            col_idx += len(cols)
            continue
        safe_gid = html_module.escape(gid, quote=True)
        for col in cols:
            model, ctype = col_meta.get(col, (None, None))
            short = html_module.escape(_short_column_name(col))
            attrs = f'data-ft-group-id="{safe_gid}" data-col-idx="{col_idx}"'
            if model:
                attrs += f' data-ft-model="{html_module.escape(model, quote=True)}"'
            if ctype:
                attrs += f' data-ft-type="{html_module.escape(ctype, quote=True)}"'
            lines.append(f'<th {attrs}>{short}</th>')
            col_idx += 1
    lines.append('</tr>')
    lines.append('</thead>')
    return lines


def _render_tbody(
    df, groups: List[Tuple],
    col_meta: Dict[str, Tuple[Optional[str], Optional[str]]],
    n_visible: int,
) -> List[str]:
    """Render data rows with overflow support."""
    use_overflow = n_visible > 0 and len(df) > n_visible
    lines = ['<tbody>']
    for row_idx, (_, row) in enumerate(df.iterrows()):
        tr_cls = ' class="ft-overflow-row"' if use_overflow and row_idx >= n_visible else ''
        lines.append(f'<tr data-original-index="{row_idx}"{tr_cls}>')
        for gid, _label, cols, sticky in groups:
            for col in cols:
                val = html_module.escape(str(row.get(col, '')))
                if sticky:
                    lines.append(f'<td class="sticky-col">{val}</td>')
                else:
                    model, ctype = col_meta.get(col, (None, None))
                    attrs = ''
                    if model:
                        attrs += f' data-ft-model="{html_module.escape(model, quote=True)}"'
                    if ctype:
                        attrs += f' data-ft-type="{html_module.escape(ctype, quote=True)}"'
                    lines.append(f'<td{attrs}>{val}</td>')
        lines.append('</tr>')
    lines.append('</tbody>')
    return lines


def _render_expand_button(n_visible: int, n_total: int) -> List[str]:
    """Render the show-more button HTML."""
    return [
        '<div class="feature-table-expand">',
        f'<button class="ft-expand-btn" data-n-visible="{n_visible}" '
        f'data-n-total="{n_total}">Show all {n_total} features</button>',
        '</div>',
    ]


def format_interactive_table(
    df, column_groups: List[Tuple], n_visible: int = 0,
) -> str:
    """Generate raw HTML table with grouped headers, toggles, and sticky Feature column.

    Toggle controls use a two-dimensional filter: model checkboxes and
    column-type checkboxes.  A cell is visible when both its model and
    its type are checked.

    When *n_visible* > 0 and the table has more rows, rows beyond
    *n_visible* are hidden with ``ft-overflow-row`` and a toggle button
    is appended.
    """
    if df is None or df.empty:
        return "*No features met the display threshold.*"
    df = BaseSection._drop_all_empty_columns(df)

    groups = _normalize_column_groups(column_groups, set(df.columns))
    col_meta = _build_column_metadata(groups)
    models_seen, types_seen = _collect_toggle_dimensions(col_meta)

    lines = ['<!-- MLSURV_RAW_HTML -->', '<div class="feature-table-container">']
    lines.extend(_render_toggle_controls(groups, models_seen, types_seen))
    lines.append('<div class="feature-table-scroll">')
    lines.append('<table class="feature-analysis-table" data-grouped-header="1">')
    lines.extend(_render_thead(groups, col_meta))
    lines.extend(_render_tbody(df, groups, col_meta, n_visible))
    lines.append('</table>')
    lines.append('</div>')  # feature-table-scroll

    if n_visible > 0 and len(df) > n_visible:
        lines.extend(_render_expand_button(n_visible, len(df)))

    lines.append('</div>')  # feature-table-container
    lines.append('<!-- /MLSURV_RAW_HTML -->')

    return '\n'.join(lines)
