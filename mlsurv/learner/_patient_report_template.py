"""HTML template + CSS for :meth:`SurvivalLearner.patient_report`.

The template is intentionally standalone rather than reusing the cohort
report renderer (:mod:`mlsurv.reports.renderers.html`): a patient-level
report has a different layout (alerts banner, narrative-forward, per-patient
framing) and the CSS is namespaced under ``.mlsurv-patient-report`` to avoid
collisions if both kinds of report are inlined in the same page.
"""

from __future__ import annotations

from html import escape
from typing import Optional


PATIENT_REPORT_CSS = """
.mlsurv-patient-report {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1100px;
    margin: 0 auto;
    padding: 1.5rem;
    color: #212529;
    line-height: 1.5;
}
.mlsurv-patient-report h1 {
    font-size: 1.5rem;
    margin: 0 0 0.25rem 0;
}
.mlsurv-patient-report h2 {
    font-size: 1.15rem;
    margin: 1.75rem 0 0.5rem 0;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid #dee2e6;
}
.mlsurv-patient-report .prov {
    color: #495057;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}
.mlsurv-patient-report .alerts {
    background: #fff8e1;
    border-left: 4px solid #f2b705;
    padding: 0.75rem 1rem;
    margin: 1rem 0 0.5rem 0;
    border-radius: 4px;
}
.mlsurv-patient-report .recal-notice {
    background: #e7f5ff;
    border-left: 4px solid #1864ab;
    padding: 0.6rem 1rem;
    margin: 0.75rem 0 0.5rem 0;
    border-radius: 4px;
    font-size: 0.9rem;
    color: #0b3d68;
}
.mlsurv-patient-report .recal-notice ul {
    margin: 0;
    padding-left: 1.25rem;
}
.mlsurv-patient-report .calibration-legend {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 0.25rem 0.75rem;
    margin: 0.6rem 0 0.2rem 0;
    font-size: 0.85rem;
    color: #495057;
}
.mlsurv-patient-report .calibration-legend dt {
    font-weight: 600;
    color: #212529;
}
.mlsurv-patient-report .calibration-legend dd {
    margin: 0;
}
.mlsurv-patient-report .bias-positive {
    color: #c92a2a;
    font-weight: 600;
}
.mlsurv-patient-report .bias-negative {
    color: #1864ab;
    font-weight: 600;
}
.mlsurv-patient-report .bias-zero {
    color: #868e96;
}
.mlsurv-patient-report .alerts h3 {
    margin: 0 0 0.35rem 0;
    font-size: 1rem;
    color: #664d03;
}
.mlsurv-patient-report .alerts ul {
    margin: 0;
    padding-left: 1.25rem;
}
.mlsurv-patient-report .summary-row {
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    align-items: flex-start;
}
.mlsurv-patient-report .summary-narrative {
    flex: 2 1 400px;
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 4px;
    white-space: pre-wrap;
    font-family: "SF Mono", Consolas, "Courier New", monospace;
    font-size: 0.9rem;
}
.mlsurv-patient-report .stat-grid {
    flex: 1 1 260px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem 1rem;
    align-content: start;
}
.mlsurv-patient-report .stat-grid dt {
    color: #6c757d;
    font-size: 0.85rem;
}
.mlsurv-patient-report .stat-grid dd {
    margin: 0;
    font-weight: 600;
    font-size: 1rem;
}
.mlsurv-patient-report .cohort-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}
.mlsurv-patient-report .cohort-row > div {
    flex: 1 1 400px;
}
.mlsurv-patient-report table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.9rem;
    margin: 0.5rem 0;
}
.mlsurv-patient-report th, .mlsurv-patient-report td {
    text-align: left;
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid #e9ecef;
}
.mlsurv-patient-report th {
    background: #f1f3f5;
    font-weight: 600;
}
.mlsurv-patient-report tr.ood td {
    background: #fff3cd;
}
.mlsurv-patient-report .badge {
    display: inline-block;
    padding: 0.1rem 0.4rem;
    font-size: 0.75rem;
    border-radius: 3px;
    background: #e7f5ff;
    color: #1864ab;
}
.mlsurv-patient-report .badge-impute {
    background: #fff3cd;
    color: #664d03;
}
.mlsurv-patient-report .direction-up {
    color: #c92a2a;
    font-weight: 600;
}
.mlsurv-patient-report .direction-down {
    color: #2f9e44;
    font-weight: 600;
}
.mlsurv-patient-report footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #dee2e6;
    color: #6c757d;
    font-size: 0.85rem;
}
.mlsurv-patient-report .methodology {
    color: #6c757d;
    font-size: 0.85em;
    margin-top: 0.4em;
    margin-bottom: 0.8em;
    line-height: 1.4;
}
.mlsurv-patient-report .disclaimer {
    font-style: italic;
    margin-top: 0.5rem;
}
""".strip()


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
{plotlyjs}
</head>
<body>
<div class="mlsurv-patient-report">
<header>
  <h1>{title}</h1>
  <div class="prov">{provenance}</div>
  <div class="prov">Generated {date}</div>
</header>
{recalibration_notice_html}
{alerts_html}

<section>
  <h2>Prediction summary</h2>
  <div class="summary-row">
    <div class="summary-narrative">{narrative}</div>
    <dl class="stat-grid">{stat_grid}</dl>
  </div>
</section>

<section>
  <h2>Individualized survival curve</h2>
  {survival_curve}
</section>

<section>
  <h2>Cohort context</h2>
  <div class="cohort-row">
    <div>{risk_histogram}</div>
    <div>{km_overlay}</div>
  </div>
</section>

<section>
  <h2>Feature contributions</h2>
  {shap_waterfall}
  {feature_table}
</section>

{knn_section}

{calibration_panel_section}

<footer>
  {footer_meta}
  <div class="disclaimer">{disclaimer}</div>
</footer>
</div>
</body>
</html>
"""


def render_html(
    *,
    title: str,
    provenance: str,
    date: str,
    narrative: str,
    stat_grid_items: list,
    survival_curve_html: str,
    risk_histogram_html: str,
    km_overlay_html: str,
    shap_waterfall_html: str,
    feature_table_html: str,
    knn_table_html: Optional[str],
    alerts: list,
    footer_meta: str,
    disclaimer: str,
    recalibration_notices: Optional[list] = None,
    calibration_panel_html: str = "",
    plotlyjs_include: str = "cdn",
) -> str:
    """Render the assembled patient report as a complete HTML document.

    Parameters
    ----------
    title : str
        Report title shown in the header and the document ``<title>``.
    provenance : str
        Provenance/attribution line shown beneath the title.
    date : str
        Report-generation date string shown in the header.
    narrative : str
        Pre-rendered HTML narrative/summary paragraph.
    stat_grid_items : list of (label, value) tuples
        Rendered as ``<dt>label</dt><dd>value</dd>`` pairs.
    survival_curve_html : str
        Embedded HTML for the predicted survival-curve figure.
    risk_histogram_html : str
        Embedded HTML for the cohort risk-distribution histogram.
    km_overlay_html : str
        Embedded HTML for the Kaplan-Meier overlay figure.
    shap_waterfall_html : str
        Embedded HTML for the SHAP waterfall figure.
    feature_table_html : str
        Pre-rendered HTML table of the patient's feature values.
    knn_table_html : str or None
        Pre-rendered HTML nearest-neighbour table, or ``None`` to omit it.
    alerts : list of str
        Rendered as bullet list items under the alerts banner. Pass ``[]``
        to hide the banner entirely.
    footer_meta : str
        Footer metadata line (e.g. model name/version provenance).
    disclaimer : str
        Disclaimer text rendered in the footer.
    recalibration_notices : list of str, optional
        Recalibration notices rendered as a banner. ``None`` or empty hides it.
    calibration_panel_html : str, default=""
        Embedded HTML for the calibration panel. Empty string hides it.
    plotlyjs_include : str
        ``'cdn'`` to load plotly.js from the CDN once in the ``<head>``,
        or ``'inline'`` to rely on each figure's own ``include_plotlyjs=True``.

    Returns
    -------
    str
        A complete, self-contained HTML document string (head, body,
        embedded CSS, and all rendered components) ready for file output.
    """
    stat_grid = "".join(
        f"<dt>{escape(str(label))}</dt><dd>{escape(str(value))}</dd>"
        for label, value in stat_grid_items
    )

    if alerts:
        items = "".join(f"<li>{escape(a)}</li>" for a in alerts)
        alerts_html = (
            f'<div class="alerts"><h3>Alerts</h3><ul>{items}</ul></div>'
        )
    else:
        alerts_html = ""

    if recalibration_notices:
        items = "".join(
            f"<li>{escape(n)}</li>" for n in recalibration_notices
        )
        recalibration_notice_html = (
            f'<div class="recal-notice"><ul>{items}</ul></div>'
        )
    else:
        recalibration_notice_html = ""

    if calibration_panel_html:
        calibration_panel_section = (
            "<section><h2>Calibration</h2>"
            f"{calibration_panel_html}</section>"
        )
    else:
        calibration_panel_section = ""

    if knn_table_html:
        knn_section = (
            "<section><h2>Similar training patients</h2>"
            f"{knn_table_html}</section>"
        )
    else:
        knn_section = ""

    if plotlyjs_include == "cdn":
        plotlyjs = (
            '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" '
            'charset="utf-8"></script>'
        )
    else:
        plotlyjs = ""

    return _PAGE_TEMPLATE.format(
        title=escape(title),
        css=PATIENT_REPORT_CSS,
        plotlyjs=plotlyjs,
        provenance=escape(provenance),
        date=escape(date),
        narrative=escape(narrative),
        stat_grid=stat_grid,
        survival_curve=survival_curve_html,
        risk_histogram=risk_histogram_html,
        km_overlay=km_overlay_html,
        shap_waterfall=shap_waterfall_html,
        feature_table=feature_table_html,
        knn_section=knn_section,
        alerts_html=alerts_html,
        recalibration_notice_html=recalibration_notice_html,
        calibration_panel_section=calibration_panel_section,
        footer_meta=footer_meta,  # trusted (built from PredictionResult)
        disclaimer=escape(disclaimer),
    )
