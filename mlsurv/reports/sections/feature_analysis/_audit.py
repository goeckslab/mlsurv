"""Feature selection audit for feature analysis section."""

import html as html_module
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from mlsurv.reports.sections.base import ALL_EXCEPTIONS
from mlsurv.viz._palettes import get_model_display_name

from mlsurv.reports._constants import format_pvalue


def generate_feature_selection_audit(section) -> str:
    """Generate feature selection audit if a TopKCoxSelector dropped features.

    Returns empty string if no selector found or no features were dropped.
    """
    try:
        return _generate_feature_selection_audit_impl(section)
    except ALL_EXCEPTIONS as e:
        section._log_recoverable_exception(
            e, "feature_analysis: feature selection audit",
            warn_user=True,
            user_message=f"Feature selection audit could not be generated: {e}",
        )
        return ''


def _generate_feature_selection_audit_impl(section) -> str:
    """Build the feature selection audit table (implementation)."""
    selectors = _collect_selectors(section)
    if not selectors:
        return ''

    first_sel = next(iter(selectors.values()))
    n_features_in = getattr(first_sel, 'n_features_in_', None)
    if n_features_in is None:
        return ''

    unified_ranking = _build_unified_ranking(selectors)
    if not unified_ranking:
        return ''

    return _build_audit_table(section, selectors, unified_ranking, n_features_in)


def _collect_selectors(section) -> Dict[str, Any]:
    """Collect TopKCoxSelectors from all trained models that drop features.

    Returns dict mapping model name to fitted selector, including only
    selectors where ``effective_k_ < n_features_in_`` (i.e., features
    were actually dropped).
    """
    trained = section._get_all_models()
    selectors: Dict[str, Any] = {}
    for name, model_result in trained.items():
        pipeline = model_result.model
        if hasattr(pipeline, 'named_steps') and 'selector' in pipeline.named_steps:
            selector = pipeline.named_steps['selector']
            if (hasattr(selector, 'sorted_features_')
                    and hasattr(selector, 'selected_features_')):
                effective_k = getattr(selector, 'effective_k_', None)
                n_features_in = getattr(selector, 'n_features_in_', None)
                if (effective_k is not None and n_features_in is not None
                        and effective_k < n_features_in):
                    selectors[name] = selector
    return selectors


def _build_unified_ranking(
    selectors: Dict[str, Any],
) -> List[Tuple[Any, float]]:
    """Build a unified feature ranking from multiple selectors.

    Takes the union of all features across selectors, using the minimum
    p-value per feature.  When all selectors share the same ranking
    (common case), the result is identical to any single selector's.
    """
    pval_map: Dict[Any, float] = {}
    for selector in selectors.values():
        for feat_id, pval in selector.sorted_features_:
            if feat_id not in pval_map or pval < pval_map[feat_id]:
                pval_map[feat_id] = pval
    return sorted(pval_map.items(), key=lambda x: (x[1], str(x[0])))


def _all_selectors_agree(selectors: Dict[str, Any]) -> bool:
    """Return True when all selectors selected the same features."""
    sel_sets = [frozenset(s.selected_features_) for s in selectors.values()]
    return len(set(sel_sets)) <= 1


def _format_selected_in(
    model_names: List[str],
    all_agree: bool,
    total_models: int,
) -> str:
    """Format the 'Selected In' cell value.

    Parameters
    ----------
    model_names : list of str
        Display names of models that selected this feature.
    all_agree : bool
        Whether all selectors have the same selected features.
    total_models : int
        Total number of models with selectors.
    """
    if not model_names:
        return "\u2014"  # em-dash
    if total_models == 1:
        return model_names[0]
    if all_agree:
        return "All models"
    return ", ".join(model_names)


def _build_audit_table(
    section,
    selectors: Dict[str, Any],
    unified_ranking: List[Tuple[Any, float]],
    n_features_in: int,
) -> str:
    """Build the audit table as HTML with expandable rows."""
    original_cols = list(getattr(section.learner, 'X_train', pd.DataFrame()).columns)

    def _resolve_name(feat_id):
        if isinstance(feat_id, (int, np.integer)) and feat_id < len(original_cols):
            return str(original_cols[feat_id])
        return str(feat_id)

    n_visible = getattr(section, 'n_features', 10)
    agree = _all_selectors_agree(selectors)
    total_models = len(selectors)

    # Pre-compute selected-features sets per model
    selected_sets: Dict[str, set] = {
        name: set(sel.selected_features_)
        for name, sel in selectors.items()
    }

    # Collect effective_k values for description
    k_values = [sel.effective_k_ for sel in selectors.values()]

    # Markdown heading + description
    lines: List[str] = ["### Feature Selection Audit", ""]
    lines.append(_build_description(selectors, k_values, n_features_in, agree))
    lines.append("")

    # HTML table with expand/collapse support
    n_total = len(unified_ranking)
    use_overflow = n_visible > 0 and n_total > n_visible

    lines.append("<!-- MLSURV_RAW_HTML -->")
    lines.append('<div class="feature-table-container">')
    lines.append('<div class="feature-table-scroll">')
    lines.append('<table class="feature-analysis-table compact">')

    # Header
    lines.append("<thead>")
    lines.append("<tr>")
    lines.append("<th>Feature</th><th>p-value</th><th>Selected In</th>")
    lines.append("</tr>")
    lines.append("</thead>")

    # Body
    lines.append("<tbody>")
    audit_rows: List[dict] = []
    for row_idx, (feat_id, pval) in enumerate(unified_ranking):
        feat_name = html_module.escape(_resolve_name(feat_id))
        pval_str = html_module.escape(
            format_pvalue(pval) if np.isfinite(pval) else "N/A"
        )

        # Which models selected this feature?
        selecting_names = sorted(
            get_model_display_name(name)
            for name, sel_set in selected_sets.items()
            if feat_id in sel_set
        )
        cell_text = html_module.escape(
            _format_selected_in(selecting_names, agree, total_models)
        )

        tr_cls = ' class="ft-overflow-row"' if use_overflow and row_idx >= n_visible else ''
        lines.append(f"<tr{tr_cls}>")
        lines.append(
            f"<td>{feat_name}</td><td>{pval_str}</td><td>{cell_text}</td>"
        )
        lines.append("</tr>")

        audit_rows.append({
            'feature': _resolve_name(feat_id),
            'p_value': pval,
            'selected': len(selecting_names) > 0,
            'selected_in': selecting_names,
        })
    lines.append("</tbody>")

    lines.append("</table>")
    lines.append("</div>")  # feature-table-scroll

    if use_overflow:
        lines.append('<div class="feature-table-expand">')
        lines.append(
            f'<button class="ft-expand-btn" data-n-visible="{n_visible}" '
            f'data-n-total="{n_total}">Show all {n_total} features</button>'
        )
        lines.append("</div>")

    lines.append("</div>")  # feature-table-container
    lines.append("<!-- /MLSURV_RAW_HTML -->")
    lines.append("")

    # Store data for programmatic access
    section.data['feature_selection'] = pd.DataFrame(audit_rows)

    return '\n'.join(lines)


def _build_description(
    selectors: Dict[str, Any],
    k_values: List[int],
    n_features_in: int,
    agree: bool,
) -> str:
    """Build the description paragraph below the audit heading."""
    if len(selectors) == 1:
        name = next(iter(selectors))
        display = get_model_display_name(name)
        k = k_values[0]
        return (
            f"The preprocessing pipeline includes a **TopKCoxSelector** that "
            f"retained {k} of {n_features_in} features for **{display}** based "
            f"on univariate Cox p-values (fitted on training data within each "
            f"CV fold)."
        )
    if agree:
        k = k_values[0]
        n = len(selectors)
        return (
            f"The preprocessing pipeline includes a **TopKCoxSelector** that "
            f"retained {k} of {n_features_in} features based on univariate Cox "
            f"p-values (fitted on training data within each CV fold). All {n} "
            f"models share the same selection."
        )
    min_k, max_k = min(k_values), max(k_values)
    return (
        f"The preprocessing pipeline includes a **TopKCoxSelector**. Models "
        f"retained between {min_k} and {max_k} of {n_features_in} features "
        f"based on univariate Cox p-values (fitted on training data within "
        f"each CV fold)."
    )
