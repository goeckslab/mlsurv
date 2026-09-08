"""Data & Preprocessing section for report generation.

Merges pipeline flowchart and dataset summary into a single comprehensive section.
"""

import html as html_module
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from mlsurv.utils import (
    empty_feature_columns,
    format_column_list,
    unwrap_pipeline,
)
from mlsurv.reports._constants import P_THRESHOLD_SIGNIFICANT, format_pvalue
from mlsurv.reports._text import pluralize
from mlsurv.reports.sections.base import (
    BaseSection,
    FigureInfo,
    ALL_EXCEPTIONS,
    IMPORT_EXCEPTIONS,
)
from mlsurv.viz.flowchart import pipeline_flowchart
from mlsurv.viz._palettes import get_model_display_name
from mlsurv.tuning.search_spaces import DEFAULT_SPACES
from mlsurv.constants import (
    LINEARITY_ASSUMPTION_MODELS,
    PH_ASSUMPTION_MODELS,
    RECOMMEND_THRESHOLDS,
)
from mlsurv.guidance import RecommendedModels
from mlsurv.models import get_model_complexity, get_model_display_names

if TYPE_CHECKING:
    from mlsurv.results.containers import EvaluationResult, ModelResult


# ------------------------------------------------------------------
# Dataset summary row helpers (extracted from _add_dataset_summary)
# ------------------------------------------------------------------


def _summary_event_rows(n_train, n_test, train_events, test_events, outcome_name):
    """Build event/censoring summary rows."""
    train_rate = train_events / n_train
    test_rate = test_events / n_test
    return [
        [f"Training {pluralize(outcome_name)}", f"{int(train_events):,} ({train_rate:.1%})"],
        [f"Test {pluralize(outcome_name)}", f"{int(test_events):,} ({test_rate:.1%})"],
        ["Censored (training)", f"{int(n_train - train_events):,} ({1 - train_rate:.1%})"],
        ["Censored (test)", f"{int(n_test - test_events):,} ({1 - test_rate:.1%})"],
    ]


def _summary_time_rows(all_times, time_unit):
    """Build survival time min/median/max rows."""
    prefix = f"Survival time, {time_unit}" if time_unit else "Survival time"
    return [
        [f"{prefix} (min)", f"{all_times.min():.1f}"],
        [f"{prefix} (median)", f"{np.median(all_times):.1f}"],
        [f"{prefix} (max)", f"{all_times.max():.1f}"],
    ]


def _pretty_stratify_label(spec):
    """Render a recorded stratify label for the report table.

    Only the leading ``event`` token is capitalized: the rest of a composite
    label is column names, which must stay as the user spelled them. ``"none"``
    is capitalized too, so it reads alongside the ``"None (...)"`` descriptions
    the caller builds for splits that never stratify.
    """
    if spec == "event":
        return "Event status"
    if spec == "none":
        return "None"
    if spec.startswith("event "):
        return "Event" + spec[len("event"):]
    return spec


def _summary_stratification_rows(learner):
    """Build split/CV stratification rows from the labels setup recorded.

    Both labels come from state (`stratify_spec`, `cv_stratify_spec`), which
    setup() writes after any binning, subpopulation augmentation, or rare-cell
    fallback.

    The CV row is dropped only when it would restate the split row word for
    word. It must survive the cases where the split row says something else:
    a group-aware or temporal split is not stratified at all, yet the folds
    still can be, and suppressing the row there would hide a balance that WAS
    applied. Pre-split data has no split row to repeat, so the CV row always
    stands on its own.
    """
    split_spec = getattr(learner._state, 'stratify_spec', None) or "event"
    presplit = getattr(learner._state, 'presplit', False)

    rows = []
    split_desc = None
    if not presplit:
        # Two split strategies ignore stratification entirely, so reporting the
        # resolved labels there would claim a balance that was never applied.
        if getattr(learner._state, 'group_aware_holdout', False):
            split_desc = "None (group-aware holdout keeps groups intact)"
        elif learner.cv_strategy == 'timeseries':
            split_desc = "None (temporal split, ordered not shuffled)"
        else:
            split_desc = _pretty_stratify_label(split_spec)
        rows.append(["Split stratification", split_desc])

    cv_spec = getattr(learner._state, 'cv_stratify_spec', None)
    if cv_spec is not None:
        cv_desc = _pretty_stratify_label(cv_spec)
        if cv_desc != split_desc:
            rows.append(["CV stratification", cv_desc])
    return rows


def _summary_epv_row(train_events, n_features):
    """Build the Events Per Variable (EPV) row."""
    epv = train_events / n_features if n_features > 0 else 0
    epv_str = f"{epv:.1f}"
    if epv < 10:
        epv_str += " (LOW — consider reducing features)"
    elif epv < 20:
        epv_str += " (borderline)"
    return ["Events per variable (EPV)", epv_str]


_TIER_LABELS = {
    "recommended": "Recommended",
    "cautionary": "Cautionary",
    "not_recommended": "Not recommended",
}

# Per-cell visual styles. Inline so the report does not depend on extra CSS
# being injected by the renderer.
_STATUS_STYLES = {
    "ok":      ("✓", "#d4edda", "#155724"),  # check, green
    "warn":    ("⚠", "#fff3cd", "#856404"),  # warn sign, amber
    "fail":    ("✗", "#f8d7da", "#721c24"),  # cross, red
    "na":      ("—", "#e9ecef", "#6c757d"),  # em-dash, gray
    "unknown": ("?",      "#f1f3f5", "#6c757d"),  # gray
}

_TIER_STATUS = {
    "recommended": "ok",
    "cautionary": "warn",
    "not_recommended": "fail",
}


def _sample_size_status(n_samples, n_rec, n_caut):
    """(status, label) for the Sample size column.

    ``n_rec``/``n_caut`` are the recommended/cautionary minimum sample
    sizes (or ``None`` when the class has no sample-size requirement,
    e.g. linear Cox).
    Labels surface the relevant threshold inequality only — the band
    name is implicit in the cell color and the Recommendation column.
    """
    if n_rec is None and n_caut is None:
        return "na", f"N={n_samples}"
    if n_rec is not None and n_samples >= n_rec:
        return "ok", f"N={n_samples} ≥ {n_rec}"
    # Cautionary band: lower (cautionary) met, upper (recommended) missed.
    if n_caut is not None and n_samples >= n_caut:
        return "warn", f"{n_caut} ≤ N={n_samples} < {n_rec}"
    # Below cautionary: the cautionary (lower) threshold was missed.
    threshold = n_caut if n_caut is not None else n_rec
    return "fail", f"N={n_samples} < {threshold}"


def _epv_status(epv, epv_rec, epv_caut):
    """(status, label) for the EPV column.

    Labels surface the relevant threshold inequality only — the band
    name is implicit in the cell color and the Recommendation column.
    """
    epv_str = "inf" if epv == float("inf") else f"{epv:.1f}"
    if epv >= epv_rec:
        return "ok", f"EPV={epv_str} ≥ {epv_rec}"
    if epv >= epv_caut:
        return "warn", f"{epv_caut} ≤ EPV={epv_str} < {epv_rec}"
    return "fail", f"EPV={epv_str} < {epv_caut}"


def _assumption_status(name, violations, test_ran, assumption_models):
    """(status, label) for a PH or linearity column.

    ``violations`` is the global tuple of feature names violating the
    assumption; ``test_ran`` is whether the diagnostic ran. ``assumption_models``
    selects which models care about this assumption.
    """
    if name not in assumption_models:
        return "na", "N/A"
    if not test_ran:
        return "unknown", "not tested"
    if not violations:
        return "ok", "OK"
    n = len(violations)
    plural = "s" if n != 1 else ""
    label = f"{n} violation{plural}"
    # Mirror `_violation_demote` thresholds so column color matches tier
    # demotion: tiny ratio => warn (yellow), larger => fail (red).
    return ("fail" if n >= 3 else "warn"), label


def _status_cell_html(status, label):
    icon, bg, fg = _STATUS_STYLES[status]
    style = (
        f"background-color: {bg}; color: {fg}; padding: 4px 8px; "
        "text-align: left; border-bottom: 1px solid #dee2e6;"
    )
    safe = html_module.escape(label)
    return f'<td style="{style}"><strong>{icon}</strong>&nbsp; {safe}</td>'


def _build_recommendations_block(rec: RecommendedModels):
    """Render `RecommendedModels` to (markdown_lines, tidy_df).

    Mirrors the stdout output from `RecommendedModels._format_table()` so the
    advisory text never diverges. The table is per-criterion (Sample size,
    EPV, PH, Linearity, Recommendation) with color-coded cells.
    """
    display_names = get_model_display_names()
    epv_str = f"{rec.epv:.1f}" if rec.epv != float("inf") else "inf"

    md_lines: list = []
    md_lines.append(
        f"*N={rec.n_samples}, events={rec.n_events}, "
        f"features={rec.n_features}, EPV={epv_str}*"
    )
    md_lines.append("")

    rows = []
    for tier_key, tier_label in _TIER_LABELS.items():
        for name in getattr(rec, tier_key):
            complexity = get_model_complexity(name)
            n_rec, epv_rec, n_caut, epv_caut = RECOMMEND_THRESHOLDS[complexity]
            ss = _sample_size_status(rec.n_samples, n_rec, n_caut)
            ev = _epv_status(rec.epv, epv_rec, epv_caut)
            ph = _assumption_status(
                name, rec.ph_violations, rec.ph_test_ran, PH_ASSUMPTION_MODELS,
            )
            ln = _assumption_status(
                name, rec.linearity_violations, rec.linearity_test_ran,
                LINEARITY_ASSUMPTION_MODELS,
            )
            tier_status = _TIER_STATUS[tier_key]
            rows.append({
                "model": display_names.get(name, name),
                "sample_size": ss,
                "epv": ev,
                "ph": ph,
                "linearity": ln,
                "tier": (tier_status, tier_label),
            })

    tidy_df = pd.DataFrame(
        [
            {
                "model": r["model"],
                "sample_size": r["sample_size"][1],
                "epv": r["epv"][1],
                "ph": r["ph"][1],
                "linearity": r["linearity"][1],
                "tier": r["tier"][1],
            }
            for r in rows
        ],
        columns=["model", "sample_size", "epv", "ph", "linearity", "tier"],
    )

    if rows:
        # Emit raw HTML — the renderer's ``MLSURV_RAW_HTML`` markers preserve
        # it verbatim. Markdown tables cannot color individual cells.
        header_style = (
            "text-align:left; padding: 6px 8px; "
            "border-bottom: 2px solid #adb5bd;"
        )
        header_cells = "".join(
            f'<th style="{header_style}">{h}</th>'
            for h in ("Model", "Sample size", "EPV", "PH", "Linearity",
                      "Recommendation")
        )
        body_rows = []
        for r in rows:
            cells = (
                f'<td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; '
                f'font-weight: 600;">{html_module.escape(r["model"])}</td>'
                + _status_cell_html(*r["sample_size"])
                + _status_cell_html(*r["epv"])
                + _status_cell_html(*r["ph"])
                + _status_cell_html(*r["linearity"])
                + _status_cell_html(*r["tier"])
            )
            body_rows.append(f"<tr>{cells}</tr>")
        table_html = (
            '<table class="model-recommendations-table" '
            'style="border-collapse: collapse; margin: 0.5em 0;">'
            f"<thead><tr>{header_cells}</tr></thead>"
            f'<tbody>{"".join(body_rows)}</tbody>'
            "</table>"
        )
        md_lines.append("<!-- MLSURV_RAW_HTML -->")
        md_lines.append(table_html)
        md_lines.append("<!-- /MLSURV_RAW_HTML -->")
        md_lines.append("")

    epv_line = rec.epv_advisory_line()
    if epv_line is not None:
        # Bold the WARNING form so it stands out in HTML; leave the milder
        # Note form as plain text.
        md_lines.append(
            f"**{epv_line}**" if epv_line.startswith("WARNING") else f"*{epv_line}*"
        )
        md_lines.append("")

    ph_line = rec.ph_summary_line()
    if ph_line is not None:
        md_lines.append(ph_line)
        md_lines.append("")

    linearity_line = rec.linearity_summary_line()
    if linearity_line is not None:
        md_lines.append(linearity_line)
        md_lines.append("")

    md_lines.append(
        "Column legend: ✓ pass, ⚠ marginal, ✗ fail, "
        "— not applicable, ? not tested. EPV thresholds: ≥10 "
        "conventional (Peduzzi 1995); ≥50 stricter (Riley 2019). "
        "Lower EPV penalises Cox-family models toward *cautionary* / "
        "*not recommended*."
    )
    md_lines.append("")

    return md_lines, tidy_df


def _add_feature_group_column(display_df, feature_groups, all_cols):
    """Insert a Group column into the display DataFrame when feature_groups is set."""
    if feature_groups is None:
        return display_df
    if isinstance(feature_groups, dict):
        fg_map = feature_groups
    elif isinstance(feature_groups, pd.Series):
        fg_map = feature_groups.to_dict()
    else:
        fg_map = dict(zip(all_cols, feature_groups))
    display_df.insert(1, "Group", display_df["Variable"].map(fg_map).fillna(""))
    return display_df


def _reorder_and_rename_columns(display_df, n_train, n_test):
    """Reorder columns and rename Train/Test with sample counts."""
    col_order = ["Variable"]
    if "Group" in display_df.columns:
        col_order.append("Group")
    col_order.append("Statistic")
    if "Train" in display_df.columns:
        col_order.append("Train")
    if "Test" in display_df.columns:
        col_order.append("Test")
    display_df = display_df[col_order]
    rename_map = {"Train": f"Train (n={n_train})", "Test": f"Test (n={n_test})"}
    display_df = display_df.rename(columns={c: rename_map.get(c, c) for c in display_df.columns})
    return display_df


def _build_expandable_html_table(df, n_visible_rows, table_class, n_total_items):
    """Build an HTML table with overflow rows hidden behind a 'Show all' button."""
    cols = df.columns
    header = '<thead><tr>' + ''.join(
        f'<th>{html_module.escape(str(c))}</th>' for c in cols
    ) + '</tr></thead>'
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        cls = ' class="ft-overflow-row"' if i >= n_visible_rows else ''
        cells = ''.join(f'<td>{html_module.escape(str(v))}</td>' for v in row)
        rows.append(f'<tr{cls}>{cells}</tr>')
    table = (
        f'<table class="{table_class}">{header}'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )
    btn = (
        '<div class="feature-table-expand">'
        f'<button class="ft-expand-btn" '
        f'data-n-visible="{n_visible_rows}" '
        f'data-n-total="{n_total_items}">'
        f'Show all {n_total_items} features</button></div>'
    )
    return (
        '<div class="feature-table-container">'
        f'<div class="feature-table-scroll">{table}</div>'
        f'{btn}</div>'
    )


def _pivot_for_display(table1_df, feature_groups, all_cols, n_train, n_test):
    """Pivot tidy table1 rows into wide display format."""
    display_df = table1_df.pivot_table(
        index=["Variable", "Statistic"],
        columns="Split",
        values="Value",
        aggfunc="first",
    ).reset_index()
    display_df.columns.name = None
    display_df = _add_feature_group_column(display_df, feature_groups, all_cols)
    display_df = _reorder_and_rename_columns(display_df, n_train, n_test)
    return display_df


class DataPreprocessingSection(BaseSection):
    """
    Data & Preprocessing section showing dataset summary and pipeline configuration.

    Combines:
    - Dataset summary: train/test split, event rates, survival time statistics
    - Pipeline flowchart: preprocessing steps and model workflow
    - Pipeline configuration: parameters and search spaces
    - Proportional hazards assessment: Schoenfeld residuals, PH test (Grambsch-Therneau)

    Maps to API method: setup()
    """

    name = "data_preprocessing"
    title = "Data & Preprocessing"

    def __init__(
        self,
        learner,
        eval_result: Optional['EvaluationResult'] = None,
    ):
        """
        Initialize data preprocessing section.

        Parameters
        ----------
        learner : SurvivalLearner
            The learner instance.
        eval_result : EvaluationResult, optional
            Evaluation result to check for bootstrap presence.
        """
        super().__init__(learner)
        self.eval_result = eval_result

    def is_available(self) -> bool:
        """Data preprocessing section is available if setup has been called."""
        return self.learner.is_setup

    def generate(self) -> Dict[str, Any]:
        """Generate data preprocessing section content."""
        md_lines = []
        figures = []
        figure_info = []

        # Section introduction
        md_lines.append(
            "This section summarizes the dataset characteristics and preprocessing "
            "pipeline used in this survival analysis."
        )
        md_lines.append("")

        # Section outline
        has_recommendations = self.learner.model_recommendations is not None
        toc_entries = [
            ('Dataset Overview', 'dataset-overview'),
        ]
        if has_recommendations:
            toc_entries.append(('Model Recommendations', 'model-recommendations'))
        toc_entries += [
            ('Missing Data', 'missing-data'),
            ('Pipeline Configuration', 'pipeline-configuration'),
            ('Analysis Pipeline', 'analysis-pipeline'),
            ('Feature Summary', 'feature-summary'),
            ('Feature Distributions', 'feature-distributions'),
        ]
        if self.learner.X_train is not None and self.learner.y_train is not None:
            toc_entries.append(('Proportional Hazards Assessment', 'proportional-hazards-assessment'))
            toc_entries.append(('Linearity Assumption Assessment', 'linearity-assumption-assessment'))
        md_lines.extend(self._build_section_outline(toc_entries))

        # === 2.1 Dataset Overview (summary table + KM curve side-by-side) ===
        md_lines.append("### Dataset Overview")
        md_lines.append("")
        try:
            self._add_dataset_summary(md_lines)
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "data_preprocessing: dataset summary",
                warn_user=True,
                user_message=f"Could not generate dataset summary: {exc}",
            )
            md_lines.append(f"*[Dataset summary unavailable: {exc}]*")
            md_lines.append("")
        try:
            self._add_overall_km(md_lines, figures, figure_info)
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "data_preprocessing: overall KM curve",
                warn_user=True,
                user_message=f"Could not generate overall KM curve: {exc}",
            )
            md_lines.append(f"*[Overall survival curve unavailable: {exc}]*")
            md_lines.append("")

        # === 2.1b Model Recommendations ===
        if has_recommendations:
            md_lines.append("### Model Recommendations")
            md_lines.append("")
            try:
                self._add_model_recommendations(md_lines)
            except ALL_EXCEPTIONS as exc:
                self._log_recoverable_exception(
                    exc, "data_preprocessing: model recommendations",
                    warn_user=True,
                    user_message=f"Could not generate model recommendations: {exc}",
                )
                md_lines.append(f"*[Model recommendations unavailable: {exc}]*")
                md_lines.append("")

        # === 2.2 Missing Data ===
        md_lines.append("### Missing Data")
        md_lines.append("")
        try:
            self._add_missing_data_breakdown(md_lines, figures, figure_info)
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "data_preprocessing: missing data breakdown",
                warn_user=True,
                user_message=f"Could not generate missing data breakdown: {exc}",
            )
            md_lines.append(f"*[Missing data breakdown unavailable: {exc}]*")
            md_lines.append("")

        # === 2.3 Pipeline Configuration ===
        md_lines.append("### Pipeline Configuration")
        md_lines.append("")
        try:
            config_table = self._generate_config_table()
            md_lines.append(config_table)
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "data_preprocessing: pipeline configuration",
                warn_user=True,
                user_message=f"Could not generate pipeline configuration: {exc}",
            )
            md_lines.append(f"*[Pipeline configuration unavailable: {exc}]*")
        md_lines.append("")

        # === 2.4 Analysis Pipeline ===
        md_lines.append("### Analysis Pipeline")
        md_lines.append("")
        md_lines.append(
            "The flowchart below illustrates the data analysis pipeline "
            "used in this survival analysis."
        )
        md_lines.append("")
        try:
            self._add_pipeline_flowchart(md_lines, figures, figure_info)
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "data_preprocessing: pipeline flowchart",
                warn_user=True,
                user_message=f"Could not generate pipeline flowchart: {exc}",
            )
            md_lines.append(f"*[Pipeline flowchart unavailable: {exc}]*")
            md_lines.append("")

        # === 2.5 Feature Summary ===
        md_lines.append("### Feature Summary")
        md_lines.append("")
        try:
            self._add_table1(md_lines)
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "data_preprocessing: feature summary",
                warn_user=True,
                user_message=f"Could not generate feature summary: {exc}",
            )
            md_lines.append(f"*[Feature summary unavailable: {exc}]*")
            md_lines.append("")

        # === 2.5b Feature Distributions ===
        try:
            self._add_feature_distributions(md_lines, figures, figure_info)
        except (ImportError,) + ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "data_preprocessing: feature distributions",
                warn_user=True,
                user_message=f"Could not generate feature distributions: {exc}",
            )
            md_lines.append(f"*[Feature distributions unavailable: {exc}]*")
            md_lines.append("")

        # === 2.6 Proportional Hazards Assessment ===
        try:
            self._add_diagnostics(md_lines, figures, figure_info)
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "data_preprocessing: PH diagnostics",
                warn_user=True,
                user_message=f"Could not generate PH diagnostics: {exc}",
            )
            md_lines.append(f"*[PH diagnostics unavailable: {exc}]*")
            md_lines.append("")

        # === 2.7 Linearity Assumption Assessment ===
        try:
            self._add_linearity_assessment(md_lines, figures, figure_info)
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "data_preprocessing: linearity diagnostics",
                warn_user=True,
                user_message=f"Could not generate linearity diagnostics: {exc}",
            )
            md_lines.append(f"*[Linearity diagnostics unavailable: {exc}]*")
            md_lines.append("")

        return {
            'markdown': '\n'.join(md_lines),
            'figures': figures,
            'figure_info': figure_info,
        }

    def _add_dataset_summary(self, md_lines: list):
        """Add dataset summary statistics."""
        n_train = len(self.learner.X_train)
        n_test = len(self.learner.X_test)
        n_total = n_train + n_test
        split_ratio = n_test / n_total

        train_events = self.learner.y_train['event'].sum()
        test_events = self.learner.y_test['event'].sum()
        all_times = np.concatenate([
            self.learner.y_train['time'],
            self.learner.y_test['time']
        ])
        n_features = len(self.learner.X_train.columns)

        summary_data = [
            ["Total cohort size", f"{n_total:,}"],
            ["Training samples", f"{n_train:,}"],
            ["Test samples", f"{n_test:,}"],
            ["Test fraction", f"{split_ratio:.1%}"],
        ]

        # Subpopulation rows (if configured)
        sg_test = getattr(self.learner._state, 'subpopulations_test', None)
        if sg_test:
            for group_var, labels in sg_test.items():
                unique, counts = np.unique(labels[~pd.isna(labels)], return_counts=True)
                parts = [f"{u}: {c}" for u, c in zip(unique, counts)]
                summary_data.append([f"Subpopulations ({group_var})", ', '.join(parts)])

        summary_data.extend(
            _summary_event_rows(n_train, n_test, train_events, test_events, self.outcome_name)
        )
        summary_data.extend(_summary_time_rows(all_times, self.time_unit))
        summary_data.append(["Number of features", f"{n_features:,}"])
        summary_data.extend(_summary_stratification_rows(self.learner))
        summary_data.append(_summary_epv_row(train_events, n_features))

        # Missing data percentage
        X_full = pd.concat([self.learner.X_train, self.learner.X_test])
        missing_total = X_full.isnull().sum().sum()
        missing_pct = missing_total / (X_full.shape[0] * X_full.shape[1]) * 100
        if missing_pct > 0:
            summary_data.append(["Missing values", f"{missing_pct:.1f}%"])

        summary_df = pd.DataFrame(summary_data, columns=["Metric", "Value"])

        self.data['dataset_summary'] = pd.DataFrame({
            'metric': [row[0] for row in summary_data],
            'value': [row[1] for row in summary_data],
        })

        md_lines.append(self._format_table(summary_df))
        md_lines.append("")

        self._dataset_summary_interpretation(
            md_lines, n_total, train_events, test_events,
        )

    def _add_model_recommendations(self, md_lines: list) -> None:
        """Append the Model Recommendations subsection.

        Reads `learner.model_recommendations` (populated by `setup()`)
        and renders the same tier table that `recommend_models()` prints to
        stdout, plus the EPV / PH / linearity advisory lines.
        """
        rec = self.learner.model_recommendations
        block_lines, tidy_df = _build_recommendations_block(rec)
        self.data['model_recommendations'] = tidy_df
        md_lines.extend(block_lines)

    def _dataset_summary_interpretation(
        self, md_lines: list, n_total: int, train_events: float, test_events: float,
    ) -> None:
        """Append interpretation paragraph based on sample-size and event-rate thresholds."""
        overall_event_rate = (train_events + test_events) / n_total
        interpretation_parts = []
        if n_total < 100:
            interpretation_parts.append(
                f"The sample size ({n_total}) is relatively small, which may "
                "limit model complexity and statistical power"
            )
        elif n_total < 500:
            interpretation_parts.append(
                f"The sample size ({n_total}) is moderate"
            )

        if overall_event_rate < 0.1:
            interpretation_parts.append(
                f"the {self.outcome_name} rate ({overall_event_rate:.1%}) is low, which "
                "reduces the effective information available for model training "
                "and may lead to wider confidence intervals"
            )
        elif overall_event_rate < 0.3:
            interpretation_parts.append(
                f"the {self.outcome_name} rate ({overall_event_rate:.1%}) is moderate"
            )

        if interpretation_parts:
            md_lines.append(
                "**Interpretation:** " + "; ".join(interpretation_parts) + "."
            )
            md_lines.append("")

    _TABLE1_MAX_FEATURES = 10
    _MISSING_DATA_MAX_FEATURES = 10

    def _add_table1(self, md_lines: list) -> None:
        """Add feature summary table by train/test split.

        Shows up to ``_TABLE1_MAX_FEATURES`` features in the visible table.
        When more features exist, the remainder is rendered inside a
        collapsible ``<details>`` dropdown.
        """
        X_train = self.learner.X_train
        X_test = self.learner.X_test
        n_train, n_test = len(X_train), len(X_test)

        all_cols = list(X_train.columns)
        n_total_features = len(all_cols)

        # Build rows for ALL features (not just visible) so
        # self.data['table1'] contains the complete tidy DataFrame.
        rows: List[List[str]] = []
        for col in all_cols:
            is_continuous = (
                pd.api.types.is_numeric_dtype(X_train[col]) and X_train[col].nunique() > 10
            )
            if is_continuous:
                rows.extend(self._table1_continuous_rows(col, X_train, X_test))
            else:
                rows.extend(self._table1_categorical_rows(
                    col, X_train, X_test, n_train, n_test,
                ))

        if not rows:
            md_lines.append("No feature summary available.")
            md_lines.append("")
            return

        table1_df = pd.DataFrame(
            rows, columns=["Variable", "Statistic", "Split", "Value"]
        )

        # Store complete tidy DataFrame
        self.data['table1'] = table1_df

        # Split into visible and overflow feature sets
        visible_cols = all_cols[:self._TABLE1_MAX_FEATURES]
        overflow_cols = all_cols[self._TABLE1_MAX_FEATURES:]

        if not overflow_cols:
            visible_tidy = table1_df[table1_df["Variable"].isin(visible_cols)]
            visible_display = _pivot_for_display(
                visible_tidy, self.feature_groups, all_cols, n_train, n_test,
            )
            md_lines.append(self._format_table(visible_display))
        else:
            # Single HTML table with overflow rows hidden via button toggle
            visible_tidy = table1_df[table1_df["Variable"].isin(visible_cols)]
            visible_display = _pivot_for_display(
                visible_tidy, self.feature_groups, all_cols, n_train, n_test,
            )
            n_visible_rows = len(visible_display)

            full_display = _pivot_for_display(
                table1_df, self.feature_groups, all_cols, n_train, n_test,
            )
            table_html = _build_expandable_html_table(
                full_display, n_visible_rows, 'feature-summary',
                n_total_features,
            )
            md_lines.append("")
            md_lines.append("<!-- MLSURV_RAW_HTML -->")
            md_lines.append(table_html)
            md_lines.append("<!-- /MLSURV_RAW_HTML -->")
        md_lines.append("")

    def _table1_continuous_rows(
        self, col: str, X_train: pd.DataFrame, X_test: pd.DataFrame,
    ) -> List[List[str]]:
        """Build Table 1 rows for a continuous feature (Mean/SD + Median)."""
        rows: List[List[str]] = []
        for label, series in [("Train", X_train[col]), ("Test", X_test[col])]:
            valid = series.dropna()
            if len(valid) == 0:
                rows.append([col, "Mean (SD)", label, "N/A"])
                continue
            rows.append([
                col, "Mean (SD)", label,
                f"{valid.mean():.2f} ({valid.std():.2f})",
            ])
            rows.append([
                col, "Median [min, max]", label,
                f"{valid.median():.2f} [{valid.min():.2f}, {valid.max():.2f}]",
            ])
        return rows

    def _table1_categorical_rows(
        self, col: str, X_train: pd.DataFrame, X_test: pd.DataFrame,
        n_train: int, n_test: int,
    ) -> List[List[str]]:
        """Build Table 1 rows for a categorical feature (top-10 levels + Other)."""
        rows: List[List[str]] = []
        combined = pd.concat([X_train[col], X_test[col]])
        top_levels = combined.value_counts().head(10).index.tolist()
        for level in top_levels:
            for label, series, n in [
                ("Train", X_train[col], n_train),
                ("Test", X_test[col], n_test),
            ]:
                count = int((series == level).sum())
                pct = count / n * 100 if n > 0 else 0
                rows.append([
                    col, f"value={level}", label,
                    f"{count} ({pct:.1f}%)",
                ])
        # Show "Other" if there are more levels
        n_other = combined.nunique() - len(top_levels)
        if n_other > 0:
            for label, series, n in [
                ("Train", X_train[col], n_train),
                ("Test", X_test[col], n_test),
            ]:
                other_count = int((~series.isin(top_levels)).sum()) if len(series) > 0 else 0
                pct = other_count / n * 100 if n > 0 else 0
                rows.append([col, f"Other ({n_other} levels)", label, f"{other_count} ({pct:.1f}%)"])
        return rows

    def _add_feature_distributions(
        self, md_lines: list, figures: list, figure_info: list,
    ) -> None:
        """Add interactive feature distribution plots with dropdown selector."""
        X_train = self.learner.X_train
        X_test = self.learner.X_test
        if X_train is None or X_test is None:
            return

        md_lines.append("### Feature Distributions")
        md_lines.append("")

        if X_train.empty or X_test.empty:
            md_lines.append(
                "*[Feature distributions skipped: train/test data is empty]*"
            )
            md_lines.append("")
            return
        md_lines.append("")
        md_lines.append(
            "Per-feature distribution plots comparing train and test splits. "
            "Use the dropdown to inspect individual features. "
            "Detected one-hot groups are shown as combined bar charts."
        )
        md_lines.append("")

        from mlsurv.viz.distributions import plot_feature_distributions

        fig, fi, err = self._generate_figure_safe(
            plot_feature_distributions,
            'feature_distributions.png',
            'Feature Distributions',
            figure_info_params={
                'figure_id': 'feature_distributions',
                'title': 'Feature Distributions by Train/Test Split',
                'data_description': (
                    'Train/test feature DataFrames used for distribution plots'
                ),
                # data_key=None: this section does not populate
                # self.data['feature_distributions']; users access the
                # underlying data via X_train/X_test directly.
                'data_key': None,
            },
            error_prefix="feature distributions",
            X_train=X_train,
            X_test=X_test,
        )

        if fig is not None:
            figures.append(('feature_distributions.png', fig))
            md_lines.append(
                "![Feature Distributions](images/feature_distributions.png)"
            )
            md_lines.append("")
            if fi is not None:
                figure_info.append(fi)
        elif err:
            md_lines.append(err)
            md_lines.append("")

    def _add_missing_data_breakdown(
        self, md_lines: list, figures: list, figure_info: list,
    ) -> None:
        """Add per-variable missing data breakdown (conditional on any NaN)."""
        X_train = self.learner.X_train
        X_test = self.learner.X_test
        X_full = pd.concat([X_train, X_test])

        total_missing = X_full.isnull().sum()
        if total_missing.sum() == 0:
            md_lines.append("No missing values detected in the dataset.")
            md_lines.append("")
            return

        # Build per-variable table
        rows = []
        for col in X_full.columns:
            n_miss = int(total_missing[col])
            if n_miss == 0:
                continue
            pct = n_miss / len(X_full) * 100
            train_miss = int(X_train[col].isnull().sum())
            test_miss = int(X_test[col].isnull().sum())
            rows.append([col, n_miss, f"{pct:.1f}%", train_miss, test_miss])

        columns = ["Variable", "N Missing", "% Missing", "Train", "Test"]
        n_total = len(rows)

        if n_total <= self._MISSING_DATA_MAX_FEATURES:
            missing_df = pd.DataFrame(rows, columns=columns)
            md_lines.append(self._format_table(missing_df))
        else:
            full_df = pd.DataFrame(rows, columns=columns)
            table_html = _build_expandable_html_table(
                full_df, self._MISSING_DATA_MAX_FEATURES, 'missing-data',
                n_total,
            )
            md_lines.append("")
            md_lines.append("<!-- MLSURV_RAW_HTML -->")
            md_lines.append(table_html)
            md_lines.append("<!-- /MLSURV_RAW_HTML -->")
        md_lines.append("")

        # A 100%-missing variable shows up in the table above as "100.0%" but
        # its fate does not: imputation has nothing to fill it from, so it is
        # dropped before fitting and never reaches the models, the assumption
        # diagnostics, or the importance tables.
        #
        # Checked against X_train, not X_full: the imputer is fit on the
        # training split, so that is the frame that decides what gets
        # dropped. A variable empty across the whole cohort is empty in
        # train too, and one empty only in test is still imputed from the
        # training fit rather than dropped.
        empty_cols = empty_feature_columns(X_train)
        if empty_cols:
            n = len(empty_cols)
            noun = "variable" if n == 1 else pluralize("variable")
            verb = "has" if n == 1 else "have"
            md_lines.append(
                f"**{n} {noun} ({format_column_list(empty_cols)}) {verb} no "
                "observed values in the training data.** Nothing can be "
                "learned from them: an imputer drops them before fitting, "
                "and they are excluded from the assumption diagnostics. "
                "They still appear in the feature importance table, scoring "
                "zero, which is not the same as a feature that was measured "
                "and found unimportant."
            )
            md_lines.append("")

        # Store tidy data
        self.data['missing_data'] = pd.DataFrame(
            rows,
            columns=["variable", "n_missing", "pct_missing", "train", "test"],
        )

        # Optional heatmap when >= 2 features have missing data
        n_features_missing = len(rows)
        if n_features_missing >= 2:
            from mlsurv.viz.missing import plot_missing as _plot_missing

            fig, fi, err = self._generate_figure_safe(
                _plot_missing,
                'missing_data_heatmap.png',
                'Missing Data Pattern',
                figure_info_params={
                    'figure_id': 'missing_data_heatmap',
                    'title': 'Missing Data Pattern',
                    'data_description': 'Per-variable missing data counts',
                    'data_key': 'missing_data',
                },
                error_prefix="missing data heatmap",
                X=X_full,
                columns=[r[0] for r in rows],
            )
            if fig is not None:
                md_lines.append("![Missing Data Heatmap](images/missing_data_heatmap.png)")
                md_lines.append("")
                figures.append(('missing_data_heatmap.png', fig))
                if fi is not None:
                    figure_info.append(fi)
            elif err:
                md_lines.append(err)
                md_lines.append("")

    def _add_overall_km(
        self, md_lines: list, figures: list, figure_info: list,
    ) -> None:
        """Add overall Kaplan-Meier survival curve by train/test split.

        When subpopulations are configured, also adds a per-subpopulation KM curve
        on the test set.
        """
        from mlsurv.reports._text import title_if_lower

        y_train = self.learner.y_train
        y_test = self.learner.y_test

        # y_train/y_test may be structured numpy arrays or DataFrames
        def _to_df(y, label):
            if isinstance(y, np.ndarray) and y.dtype.names:
                return pd.DataFrame({
                    'time': y['time'],
                    'event': y['event'].astype(int),
                    'Split': label,
                })
            df = pd.DataFrame({'time': y['time'], 'event': y['event']})
            df['Split'] = label
            return df

        km_data = pd.concat([_to_df(y_train, 'Train'), _to_df(y_test, 'Test')], ignore_index=True)

        from mlsurv.viz.survival import plot_km

        outcome_label = title_if_lower(self.outcome_name)
        km_title = f"Time to {outcome_label} Distribution"

        def _make_km():
            return plot_km(
                km_data, group_col='Split', show_ci=True,
                show_risk_table=True,
                title=km_title,
                time_unit=self.time_unit,
            )

        fig, fi, err = self._generate_figure_safe(
            _make_km,
            'overall_km.png',
            km_title,
            figure_info_params={
                'figure_id': 'overall_km',
                'title': km_title,
                'data_description': 'KM curve by train/test split',
                'data_key': 'dataset_summary',
            },
            error_prefix="overall KM curve",
        )
        if fig is not None:
            figures.append(('overall_km.png', fig))
            if fi is not None:
                figure_info.append(fi)
            md_lines.append("![Overall KM](images/overall_km.png)")
            md_lines.append("")
        elif err:
            md_lines.append(err)
            md_lines.append("")

        # Per-subpopulation KM curve (test set only)
        sg_test = getattr(self.learner._state, 'subpopulations_test', None)
        if sg_test:
            self._add_subpopulation_km(md_lines, figures, figure_info, sg_test, km_title)

    def _add_subpopulation_km(
        self, md_lines, figures, figure_info, sg_test, base_title,
    ):
        """Add per-subpopulation KM curve on the test set."""
        from mlsurv.viz.survival import plot_km

        y_test = self.learner.y_test
        for group_var, labels in sg_test.items():
            def _to_sg_df(y, sg_labels):
                if isinstance(y, np.ndarray) and y.dtype.names:
                    return pd.DataFrame({
                        'time': y['time'],
                        'event': y['event'].astype(int),
                        'Subpopulation': sg_labels,
                    })
                return pd.DataFrame({
                    'time': y['time'], 'event': y['event'],
                    'Subpopulation': sg_labels,
                })

            sg_title = f"{base_title} by {group_var}"
            sg_data = _to_sg_df(y_test, labels)
            # Drop NaN subpopulation labels
            sg_data = sg_data.dropna(subset=['Subpopulation'])

            # Build color map from subpopulation palette
            from mlsurv.viz._palettes import get_subpopulation_color
            unique_groups = sorted(sg_data['Subpopulation'].unique())
            sg_colors = {g: get_subpopulation_color(i) for i, g in enumerate(unique_groups)}

            def _make_sg_km(data=sg_data, title=sg_title, colors=sg_colors):
                return plot_km(
                    data, group_col='Subpopulation', show_ci=True,
                    show_risk_table=True,
                    title=title,
                    time_unit=self.time_unit,
                    colors=colors,
                )

            fname = f'subpopulation_km_{group_var}.png'
            fig, fi, err = self._generate_figure_safe(
                _make_sg_km,
                fname,
                sg_title,
                figure_info_params={
                    'figure_id': f'subpopulation_km_{group_var}',
                    'title': sg_title,
                    'data_description': f'KM curve by {group_var} (test set)',
                },
                error_prefix=f"subpopulation KM curve ({group_var})",
            )
            if fig is not None:
                figures.append((fname, fig))
                if fi is not None:
                    figure_info.append(fi)
                md_lines.append(f"![Subpopulation KM](images/{fname})")
                md_lines.append("")
            elif err:
                md_lines.append(err)
                md_lines.append("")

    def _add_pipeline_flowchart(self, md_lines: list, figures: list, figure_info: list):
        """Add pipeline flowchart visualization."""
        # Extract pipeline components from learner
        imputer_str = self._get_component_str(
            self.imputer, 'imputer'
        )
        scaler_str = self._get_component_str(
            self.scaler, 'scaler'
        )
        selector_str = self._get_component_str(
            self.feature_selector, 'selector'
        )

        # Check if feature selection was tuned (even if not explicitly configured)
        if selector_str is None:
            selector_str = self._get_tuned_selector_str()

        # Get model names
        all_models = self._get_all_models()
        model_names = [
            get_model_display_name(name) for name in all_models.keys()
        ]

        # Detect CV and tuning status for bracket
        show_bracket, bracket_label = self._detect_cv_and_tuning()

        # Detect bootstrap
        has_bootstrap = (
            self.eval_result is not None and
            self.eval_result.bootstrap is not None
        )
        n_bootstrap = None
        if has_bootstrap:
            n_bootstrap = len(
                self.eval_result.bootstrap.iterations['iteration'].unique()
            )

        # Generate flowchart
        fig = pipeline_flowchart(
            imputer=imputer_str,
            scaler=scaler_str,
            selector=selector_str,
            models=model_names,
            show_bracket=show_bracket,
            bracket_label=bracket_label,
            has_bootstrap=has_bootstrap,
            n_bootstrap=n_bootstrap,
        )
        figures.append(('pipeline_flowchart.png', fig))

        figure_info.append(FigureInfo(
            figure_id='pipeline_flowchart',
            title='Data Analysis Pipeline Flowchart',
            data_description='Dataset summary table (metric, value)',
            data_key='dataset_summary',
        ))
        md_lines.append("![Pipeline Flowchart](images/pipeline_flowchart.png)")
        md_lines.append("")

    @staticmethod
    def _get_component_str(component, component_type: str) -> Optional[str]:
        """Get a display string for a pipeline component, or None if disabled."""
        if component is None:
            return None
        return type(component).__name__

    def _detect_cv_and_tuning(self) -> tuple:
        """
        Detect CV and tuning status for bracket display.

        Returns
        -------
        tuple of (bool, str)
            (show_bracket: always True since all models do CV,
             bracket_label: formatted label for bracket)
        """
        all_models = self._get_all_models()
        all_model_names = [
            get_model_display_name(name) for name in all_models.keys()
        ]

        # Collect tuned model names from both optuna studies and is_tuned flag
        tuned_names_set = set()
        for name, result in all_models.items():
            if hasattr(result, 'optuna_study') and result.optuna_study is not None:
                tuned_names_set.add(get_model_display_name(name))
        for name in self._get_tuned_models().keys():
            tuned_names_set.add(get_model_display_name(name))

        # Always show bracket (all models go through CV)
        bracket_label = self._format_bracket_label(all_model_names, sorted(tuned_names_set))
        return True, bracket_label

    @staticmethod
    def _format_bracket_label(all_models: list, tuned_models: list) -> str:
        """Format the bracket label based on CV and tuning status."""
        if not tuned_models:
            return "Cross Validation"
        if len(tuned_models) == len(all_models):
            return "Cross Validation & Tuning"
        return f"Cross Validation (all models) & Tuning ({', '.join(tuned_models)})"

    def _generate_config_table(self) -> str:
        """
        Generate markdown table of pipeline configuration.

        Returns
        -------
        str
            Markdown table showing each pipeline step with its
            untuned parameters and tuned parameter search spaces.
        """
        rows = []

        # Row 1: Scaler
        scaler = self.scaler
        scaler_untuned = self._format_untuned_params(scaler, 'scaler')
        rows.append({
            'Step': 'Scaler',
            'Initial Parameters': scaler_untuned,
            'Parameter Search Spaces for Tuning': '-',
        })

        # Row 2: Imputer
        imputer = self.imputer
        imputer_untuned = self._format_untuned_params(imputer, 'imputer')
        rows.append({
            'Step': 'Imputer',
            'Initial Parameters': imputer_untuned,
            'Parameter Search Spaces for Tuning': '-',
        })

        # Row 3: Selector
        # Use fitted selector from a trained pipeline if available
        # (the configured selector is the unfitted template without effective_k_)
        selector = self.feature_selector
        if selector is not None and not hasattr(selector, 'effective_k_'):
            for mr in self.learner.models.values():
                pipe = getattr(mr, 'model', None)
                fitted_sel = (
                    getattr(pipe, 'named_steps', {}).get('selector')
                    if pipe is not None else None
                )
                if fitted_sel is not None and hasattr(fitted_sel, 'effective_k_'):
                    selector = fitted_sel
                    break
        selector_untuned = self._format_untuned_params(selector, 'selector')
        selector_tuned = self._get_selector_tuned_params()
        rows.append({
            'Step': 'Selector',
            'Initial Parameters': selector_untuned,
            'Parameter Search Spaces for Tuning': selector_tuned,
        })

        # Row 4+: Models
        all_models = self._get_all_models()
        tuned_models = self._get_tuned_models()

        for model_key, model_result in all_models.items():
            display_name = get_model_display_name(model_key)
            model_untuned = self._format_untuned_params(
                model_result.model, 'model', result=model_result
            )
            # Check if this model was tuned
            is_tuned = model_key in tuned_models
            model_tuned = self._format_search_space(model_result) if is_tuned else '-'
            rows.append({
                'Step': display_name,
                'Initial Parameters': model_untuned,
                'Parameter Search Spaces for Tuning': model_tuned,
            })

        df = pd.DataFrame(rows)
        return self._format_table(df, index=False)

    def _format_untuned_params(
        self,
        component,
        component_type: str,
        result: Optional["ModelResult"] = None,
    ) -> str:
        """
        Format untuned parameters for display.

        Parameters
        ----------
        component : object or None
            The sklearn-like component (imputer, scaler, selector, or model).
        component_type : str
            Type of component ('imputer', 'scaler', 'selector', 'model').
        result : ModelResult, optional
            Required when ``component_type='model'``; identifies the model so its
            base name (e.g., 'rsf') can drive default-parameter lookup.

        Returns
        -------
        str
            Formatted display string with key parameters.
        """
        if component is None:
            return "Not configured"

        class_name = type(component).__name__

        if component_type == 'scaler':
            return class_name
        elif component_type == 'imputer':
            strategy = getattr(component, 'strategy', None)
            if strategy:
                return f"{class_name}(strategy={strategy})"
            return class_name
        elif component_type == 'selector':
            k = getattr(component, 'k', None)
            if k is not None:
                effective = getattr(component, 'effective_k_', None)
                if k == 'auto' and effective is not None:
                    return f"{class_name}(k='auto' \u2192 {effective})"
                return f"{class_name}(k={k})"
            return class_name
        elif component_type == 'model':
            if result is None:
                raise ValueError(
                    "_format_untuned_params(component_type='model') requires result="
                )
            return self._format_model_untuned_params(component, result)
        else:
            return class_name

    def _format_model_untuned_params(self, component, result: "ModelResult") -> str:
        """Format untuned parameters for a model component.

        Uses a fresh default model to show true defaults; falls back to live model
        if fresh instantiation fails.
        """
        base_name = result.base_name
        default_params = self._get_default_params(result)

        # Non-empty dict: fresh model produced key params
        if default_params:
            return ', '.join(f"{k}={v}" for k, v in default_params.items())

        # None: fresh instantiation failed -- fall back to live model params
        if default_params is None:
            actual_model = unwrap_pipeline(component)
            key_params = self._get_model_key_params(actual_model, base_name)
            if key_params:
                return ', '.join(f"{k}={v}" for k, v in key_params.items())
            return type(actual_model).__name__

        # Empty dict: model has no key params -- show class name
        actual_model = unwrap_pipeline(component)
        return type(actual_model).__name__

    def _get_default_params(self, result: "ModelResult") -> Optional[dict]:
        """Get key parameters from a freshly instantiated default model.

        Returns dict of formatted params, empty dict if model has no key params,
        or None if fresh instantiation fails.
        """
        base_name = result.base_name
        try:
            from mlsurv.models import get_model, MODEL_REGISTRY, _LAZY_MODEL_IMPORTS
            # Custom (unregistered) models can't be freshly instantiated via
            # the registry — skip silently and fall back to live model params.
            if base_name not in MODEL_REGISTRY and base_name not in _LAZY_MODEL_IMPORTS:
                return None
            n_train = len(self.learner.X_train)
            n_features = len(self.learner.X_train.columns)
            n_events = int(self.learner.y_train['event'].sum())
            fresh_model = get_model(
                base_name,
                n_samples=n_train,
                n_features=n_features,
                n_events=n_events,
            )
            return self._get_model_key_params(fresh_model, base_name)
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, f"data_preprocessing: fresh model defaults for {base_name}",
            )
            return None

    def _get_model_key_params(self, model, model_name: str) -> dict:
        """
        Get key parameters for a model for display.

        Parameters
        ----------
        model : object
            The fitted model instance.
        model_name : str
            Model key name.

        Returns
        -------
        dict
            Dictionary of key parameter names and values.
        """
        # Define key params per model type (most important ones)
        # These match actual model attribute names, not search space names
        key_param_names = {
            'coxph': [],
            'coxnet': ['l1_ratio', 'n_alphas'],
            'rsf': ['n_estimators', 'max_depth', 'min_samples_split'],
            'gbsa': ['n_estimators', 'learning_rate', 'max_depth'],
            'fssvm': ['alpha', 'rank_ratio'],
            'deepsurv': ['layers', 'dropout', 'learning_rate'],
            'deephit': ['layers', 'dropout', 'learning_rate'],
            'xgbcox': ['n_estimators', 'learning_rate', 'max_depth'],
            'weibull_aft': ['penalizer'],
        }

        param_names = key_param_names.get(model_name, [])
        if not param_names:
            return {}

        result = {}
        for param in param_names:
            val = getattr(model, param, None)
            if val is not None:
                # Format different types nicely
                if isinstance(val, float):
                    if val < 0.01:
                        result[param] = f"{val:.1e}"
                    else:
                        result[param] = f"{val:.3g}"
                elif isinstance(val, list):
                    # Format lists like [64, 64]
                    result[param] = str(val)
                else:
                    result[param] = val
        return result

    def _get_tuned_selector_str(self) -> Optional[str]:
        """
        Get selector string from tuned models if feature selection was tuned.

        Returns
        -------
        str or None
            Selector description like 'TopKCoxSelector(k=6)' if tuned, None otherwise.
        """
        tuned_models = self._get_tuned_models()
        if not tuned_models:
            return None

        # The selector's k is tuned only via feature_selector__k;
        # a bare 'k' in best_params belongs to the model, not the selector.
        for model_name, result in tuned_models.items():
            if hasattr(result, 'optuna_study') and result.optuna_study is not None:
                best_params = result.optuna_study.best_params
                k_value = best_params.get('feature_selector__k')
                if k_value is not None:
                    return f"TopKCoxSelector(k={k_value})"

        return None

    def _get_selector_tuned_params(self) -> str:
        """
        Get tuned parameter string for selector based on model search spaces.

        Returns
        -------
        str
            Tuned parameter string like 'k: 5-n_features' or '-'.
        """
        tuned_models = self._get_tuned_models()
        if not tuned_models:
            return '-'

        # Check if any tuned model has 'k' in its search space
        for model_name, result in tuned_models.items():
            base_name = result.base_name
            if base_name in DEFAULT_SPACES:
                space = DEFAULT_SPACES[base_name]
                if 'k' in space:
                    k_range = space['k']
                    if isinstance(k_range, tuple) and len(k_range) >= 2:
                        low, high = k_range[0], k_range[1]
                        high_str = 'n_features' if high is None else str(high)
                        return f"k: {low}-{high_str}"
        return '-'

    def _format_search_space(self, result: "ModelResult") -> str:
        """
        Format search space for model.

        Parameters
        ----------
        result : ModelResult
            The model's result; ``result.base_name`` keys the DEFAULT_SPACES lookup.

        Returns
        -------
        str
            Formatted search space string or '-' if no tuning.
        """
        base_name = result.base_name
        if base_name not in DEFAULT_SPACES:
            return '-'

        space = DEFAULT_SPACES[base_name]
        if not space:
            return '-'

        # Skip 'k' as it's shown in selector row
        formatted_params = []
        for param, value in space.items():
            if param == 'k':
                continue
            formatted_params.append(self._format_param_range(param, value))

        if not formatted_params:
            return '-'

        return ', '.join(formatted_params)

    # ------------------------------------------------------------------
    # PH diagnostics (Schoenfeld, PH test)
    # ------------------------------------------------------------------

    def _add_diagnostics(self, md_lines, figures, figure_info):
        """Schoenfeld residuals and PH test.

        These assess the proportional hazards assumption of the covariates
        themselves, independent of which models were trained. The Schoenfeld
        scatter and the Grambsch-Therneau p-value table come from a **single**
        internally re-fit CoxPH model (``ph_test_with_schoenfeld``), so
        the plot and the table are guaranteed consistent.
        """
        if self.learner.X_train is None or self.learner.y_train is None:
            return

        y_train = self.learner.y_train

        md_lines.append("### Proportional Hazards Assessment")
        md_lines.append("")
        md_lines.append(
            "These diagnostics test whether the covariates satisfy the proportional "
            "hazards (PH) assumption — a key requirement for Cox-family models. "
            "All residuals are computed from an internally re-fit CoxPH model."
        )
        md_lines.append("")

        # Schoenfeld residuals (scatter) and the formal PH test (table) share
        # one Cox fit; a single failure here degrades both sub-sections. The
        # imputation runs inside the guard too: it can raise (an all-missing
        # matrix, an imputer whose output cannot be labelled), and outside it
        # that message would escape the section instead of landing in the
        # report where the reader is.
        try:
            from mlsurv.evaluation.metrics import ph_test_with_schoenfeld

            X_train = self.learner._get_imputed_X(
                self.learner.X_train, raise_on_error=True,
            )
            ph_df, schoenfeld_df = ph_test_with_schoenfeld(
                X_train, y_train, n_features=X_train.shape[1],
            )
        except (ImportError,) + ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e, "data_preprocessing: PH / Schoenfeld computation",
                warn_user=True,
                user_message=f"Could not compute PH diagnostics: {e}",
            )
            md_lines.append(f"*[Schoenfeld residuals unavailable: {e}]*")
            md_lines.append("")
            md_lines.append(f"*[PH test unavailable: {e}]*")
            md_lines.append("")
            return

        # --- Schoenfeld residuals ---
        # The fit succeeded (the early return above did not fire) but the
        # scaled-residual computation can still fail, leaving an empty frame.
        # Surface that instead of rendering the prose over a blank figure.
        if schoenfeld_df.empty:
            md_lines.append(
                "*[Schoenfeld residual plot unavailable — scaled residual "
                "computation failed for this fit; the PH test table below "
                "is still valid.]*"
            )
            md_lines.append("")
        else:
            self._add_schoenfeld_plot(schoenfeld_df, md_lines, figures, figure_info)

        # --- PH formal test (Grambsch-Therneau) ---
        try:
            if not ph_df.empty:
                self.data['ph_test'] = ph_df

                md_lines.append(
                    "**Proportional Hazards Test (Grambsch-Therneau):** "
                    "This test regresses scaled Schoenfeld residuals on time for each "
                    "covariate. The null hypothesis is that the coefficient (hazard ratio) "
                    "is constant over time. Per-feature p-values are FDR-corrected "
                    "(Benjamini-Hochberg); significance is based on adjusted p < "
                    f"{P_THRESHOLD_SIGNIFICANT}. "
                    "Non-significant results support the PH assumption. "
                    "Significant results indicate the covariate's "
                    "effect changes over the follow-up period, which may warrant "
                    "time-varying coefficients or stratification. "
                    "The GLOBAL row tests the joint hypothesis across all covariates "
                    "(not adjusted for multiplicity)."
                )
                md_lines.append("")
                self._build_ph_table(md_lines, ph_df)
            else:
                md_lines.append(
                    "*[PH test unavailable — computation returned no results. "
                    "Check logs for details.]*"
                )
                md_lines.append("")
        except (ImportError,) + ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e, "data_preprocessing: PH test computation",
                warn_user=True,
                user_message=f"Could not compute PH test: {e}",
            )
            md_lines.append(f"*[PH test unavailable: {e}]*")
            md_lines.append("")

    def _add_schoenfeld_plot(self, schoenfeld_df, md_lines, figures, figure_info):
        """Render the scaled Schoenfeld residual scatter from a tidy frame."""
        try:
            from mlsurv.viz.diagnostics import plot_schoenfeld_interactive

            self.data['schoenfeld'] = schoenfeld_df

            md_lines.append(
                "**Schoenfeld residuals.** A Cox model assumes each covariate's "
                "effect — its *hazard ratio* (multiplicative effect on the "
                "instantaneous event rate) — stays constant over follow-up. "
                "This is the *proportional hazards* (PH) assumption. Schoenfeld "
                "residuals check it one covariate at a time: at every event "
                "time, take the covariate value of the patient who just had "
                "the event and subtract the hazard-weighted average covariate "
                "value among everyone still at risk. If the model is right, "
                "these residuals should scatter flat around zero over follow-up. "
                "A systematic upward or downward LOWESS trend means the "
                "covariate's effect is drifting — e.g., strong early and "
                "fading later — and the constant-hazard-ratio assumption does "
                "not hold for that feature. The residuals shown here are "
                "*scaled* so that the slope against time is directly "
                "interpretable as how much the effect is changing; the formal "
                "Grambsch-Therneau test below turns that slope into a p-value. "
                "Use the dropdown to inspect individual features."
            )
            md_lines.append("")

            fig_sch = plot_schoenfeld_interactive(schoenfeld_df, time_unit=self.time_unit)
            figures.append(('schoenfeld_residuals.png', fig_sch))
            md_lines.append("![Schoenfeld Residuals](images/schoenfeld_residuals.png)")
            md_lines.append("")

            figure_info.append(FigureInfo(
                figure_id='schoenfeld_residuals',
                title='Scaled Schoenfeld Residuals',
                data_description=(
                    'Tidy DataFrame with scaled Schoenfeld residuals '
                    '(feature, time, residual) — all features'
                ),
                data_key='schoenfeld',
                learner_code="fig = learner.plot_diagnostics()",
            ))
        except (ImportError,) + ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e, "data_preprocessing: Schoenfeld residual plot",
                warn_user=True,
                user_message=f"Could not generate Schoenfeld residual plot: {e}",
            )
            md_lines.append(f"*[Schoenfeld residuals unavailable: {e}]*")
            md_lines.append("")

    @staticmethod
    def _build_ph_table(md_lines, ph_df, n_show: int = 10):
        """Emit a compact PH test table sorted by p-value with toggle for overflow."""

        sorted_df = ph_df.sort_values('p_value').reset_index(drop=True)
        n_total = len(sorted_df)

        def _row_html(row, overflow=False):
            sig = "Yes" if row['significant'] else "No"
            sig_cls = ' class="ph-sig"' if row['significant'] else ''
            feat = html_module.escape(str(row['feature']))
            tr_cls = ' class="ft-overflow-row"' if overflow else ''
            return (
                f'<tr{tr_cls}><td>{feat}</td>'
                f'<td style="text-align:right">{row["test_statistic"]:.3f}</td>'
                f'<td style="text-align:right">{format_pvalue(row["p_value"])}</td>'
                f'<td style="text-align:right">{format_pvalue(row["p_adjusted"])}</td>'
                f'<td style="text-align:center"{sig_cls}>{sig}</td></tr>'
            )

        header = (
            '<table class="ph-test-table">'
            '<thead><tr>'
            '<th>Feature</th><th style="text-align:right">Statistic</th>'
            '<th style="text-align:right">p-value</th>'
            '<th style="text-align:right">p-adj (BH)</th>'
            '<th style="text-align:center">Sig.</th>'
            '</tr></thead><tbody>'
        )
        all_rows = [
            _row_html(sorted_df.iloc[i], overflow=(i >= n_show))
            for i in range(n_total)
        ]

        md_lines.append("<!-- MLSURV_RAW_HTML -->")
        if n_total <= n_show:
            md_lines.append(header + ''.join(all_rows) + '</tbody></table>')
        else:
            table_html = header + ''.join(all_rows) + '</tbody></table>'
            btn_html = (
                '<div class="feature-table-expand">'
                f'<button class="ft-expand-btn" '
                f'data-n-visible="{n_show}" data-n-total="{n_total}">'
                f'Show all {n_total} features</button></div>'
            )
            md_lines.append(
                '<div class="feature-table-container">'
                f'<div class="feature-table-scroll">{table_html}</div>'
                f'{btn_html}</div>'
            )
        md_lines.append("<!-- /MLSURV_RAW_HTML -->")
        md_lines.append("")

    # ------------------------------------------------------------------
    # Linearity diagnostics — martingale plot + Wald test table
    # ------------------------------------------------------------------

    def _add_linearity_assessment(self, md_lines, figures, figure_info):
        """Martingale residual plot and Harrell/rms Wald linearity test.

        Tests whether the log-hazard is linear in each continuous covariate
        (a key Cox-family assumption distinct from PH).  Re-uses the cached
        ``learner.linearity_test_result`` from setup when available so the
        RCS Cox fit runs at most once per analysis.
        """
        if self.learner.X_train is None or self.learner.y_train is None:
            return

        y_train = self.learner.y_train

        md_lines.append("### Linearity Assumption Assessment")
        md_lines.append("")
        md_lines.append(
            "These diagnostics test whether the log-hazard is linear in each "
            "continuous covariate — a Cox-family assumption distinct from "
            "(and equal in importance to) proportional hazards.  Violations "
            "produce biased coefficient estimates and miscalibrated risk "
            "predictions even when the PH assumption itself holds."
        )
        md_lines.append("")

        # Both subsections below need the imputed matrix, and imputation can
        # raise (an all-missing matrix, an imputer whose output cannot be
        # labelled). Guarded here rather than in each block, so one failure
        # reports once and neither block runs on an undefined frame.
        try:
            X_train = self.learner._get_imputed_X(
                self.learner.X_train, raise_on_error=True,
            )
        except (ImportError,) + ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e, "data_preprocessing: linearity imputation",
                warn_user=True,
                user_message=f"Could not compute linearity diagnostics: {e}",
            )
            md_lines.append(f"*[Martingale residual plot unavailable: {e}]*")
            md_lines.append("")
            md_lines.append(f"*[Linearity test unavailable: {e}]*")
            md_lines.append("")
            return

        # --- Martingale residual plot (Therneau-Grambsch-Fleming 1990) ---
        try:
            from mlsurv.viz.diagnostics import (
                compute_martingale_data,
                plot_martingale_interactive,
            )

            martingale_df = compute_martingale_data(X_train, y_train)
            if martingale_df.empty:
                md_lines.append(
                    "*[Martingale residual plot unavailable — no continuous "
                    "covariates or null-model fit failed.]*"
                )
                md_lines.append("")
            else:
                self.data['martingale'] = martingale_df

                md_lines.append(
                    "**Martingale residuals:** Null-model martingale residuals "
                    "(observed events minus cumulative hazard, estimated from "
                    "Nelson-Aalen; Therneau, Grambsch & Fleming 1990) plotted "
                    "against each continuous covariate.  A **straight** LOWESS "
                    "trend (of any slope) is consistent with a linear log-hazard "
                    "effect for that covariate.  A **curved** pattern (U-shape, "
                    "monotone curve, step, etc.) reveals the functional form of "
                    "the covariate's effect on the hazard and indicates a "
                    "violation of the linearity assumption.  Use the dropdown to "
                    "inspect individual features."
                )
                md_lines.append("")

                fig_mart = plot_martingale_interactive(martingale_df)
                figures.append(('martingale_residuals.png', fig_mart))
                md_lines.append(
                    "![Martingale Residuals](images/martingale_residuals.png)"
                )
                md_lines.append("")

                figure_info.append(FigureInfo(
                    figure_id='martingale_residuals',
                    title='Null-Model Martingale Residuals',
                    data_description=(
                        'Tidy DataFrame with null-model martingale residuals '
                        '(feature, x_value, residual) for every continuous '
                        'covariate'
                    ),
                    data_key='martingale',
                    learner_code="fig = learner.plot_diagnostics(kind='martingale')",
                ))
        except IMPORT_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e, "data_preprocessing: martingale residual plot",
                warn_user=True,
                user_message=f"Could not generate martingale residual plot: {e}",
            )
            md_lines.append(f"*[Martingale residual plot unavailable: {e}]*")
            md_lines.append("")

        # --- Formal linearity test (Harrell/rms Wald on stacked RCS) ---
        try:
            # Prefer the cached result from setup to avoid a refit
            lin_df = self.learner.linearity_test_result
            if lin_df is None or lin_df.empty:
                from mlsurv.evaluation.metrics import linearity_test
                lin_df = linearity_test(X_train, y_train)

            if not lin_df.empty:
                self.data['linearity_test'] = lin_df

                md_lines.append(
                    "**Log-Hazard Linearity Test (Harrell/rms Wald):** "
                    "A single unpenalized Cox model is fit with 4-knot "
                    "restricted-cubic-spline expansions for every continuous "
                    "covariate (knot quantiles at 5/35/65/95%, matching "
                    "Harrell 2015 Table 2.3). For each feature, the joint "
                    "Wald chi-square on the non-linear spline terms tests "
                    "whether the log-hazard deviates from linearity; with "
                    "4 knots the per-feature test has 2 degrees of freedom. "
                    "Per-feature p-values are FDR-corrected "
                    "(Benjamini & Hochberg 1995); significance is based on "
                    f"adjusted p < {P_THRESHOLD_SIGNIFICANT}. "
                    "Non-significant results support the linearity assumption. "
                    "Significant results indicate the covariate enters the "
                    "log-hazard non-linearly, suggesting a transformation "
                    "(log, Box-Cox) or a flexible model (RSF, GBSA, DeepSurv). "
                    "The GLOBAL row tests the joint non-linearity across all "
                    "covariates (not adjusted for multiplicity).  Matches "
                    "R's ``rms::anova.cph`` output (Harrell 2015 §5.5)."
                )
                md_lines.append("")
                self._build_linearity_table(md_lines, lin_df)
            else:
                md_lines.append(
                    "*[Linearity test unavailable — no continuous covariates "
                    "or RCS Cox fit did not converge.  Check logs for details.]*"
                )
                md_lines.append("")
        except IMPORT_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e, "data_preprocessing: linearity test computation",
                warn_user=True,
                user_message=f"Could not compute linearity test: {e}",
            )
            md_lines.append(f"*[Linearity test unavailable: {e}]*")
            md_lines.append("")

    @staticmethod
    def _build_linearity_table(md_lines, lin_df, n_show: int = 10):
        """Emit a compact linearity test table sorted by p-value with overflow toggle.

        Mirrors :meth:`_build_ph_table` exactly — reuses the existing
        ``ph-test-table`` and ``ph-sig`` CSS classes since they describe
        the visual treatment of a "feature × significance" table, not
        PH-specific semantics.
        """
        sorted_df = lin_df.sort_values('p_value').reset_index(drop=True)
        n_total = len(sorted_df)

        def _row_html(row, overflow=False):
            sig = "Yes" if row['significant'] else "No"
            sig_cls = ' class="ph-sig"' if row['significant'] else ''
            feat = html_module.escape(str(row['feature']))
            tr_cls = ' class="ft-overflow-row"' if overflow else ''
            return (
                f'<tr{tr_cls}><td>{feat}</td>'
                f'<td style="text-align:right">{row["test_statistic"]:.3f}</td>'
                f'<td style="text-align:right">{format_pvalue(row["p_value"])}</td>'
                f'<td style="text-align:right">{format_pvalue(row["p_adjusted"])}</td>'
                f'<td style="text-align:center"{sig_cls}>{sig}</td></tr>'
            )

        header = (
            '<table class="ph-test-table">'
            '<thead><tr>'
            '<th>Feature</th><th style="text-align:right">Wald &chi;&sup2;</th>'
            '<th style="text-align:right">p-value</th>'
            '<th style="text-align:right">p-adj (BH)</th>'
            '<th style="text-align:center">Sig.</th>'
            '</tr></thead><tbody>'
        )
        all_rows = [
            _row_html(sorted_df.iloc[i], overflow=(i >= n_show))
            for i in range(n_total)
        ]

        md_lines.append("<!-- MLSURV_RAW_HTML -->")
        if n_total <= n_show:
            md_lines.append(header + ''.join(all_rows) + '</tbody></table>')
        else:
            table_html = header + ''.join(all_rows) + '</tbody></table>'
            btn_html = (
                '<div class="feature-table-expand">'
                f'<button class="ft-expand-btn" '
                f'data-n-visible="{n_show}" data-n-total="{n_total}">'
                f'Show all {n_total} features</button></div>'
            )
            md_lines.append(
                '<div class="feature-table-container">'
                f'<div class="feature-table-scroll">{table_html}</div>'
                f'{btn_html}</div>'
            )
        md_lines.append("<!-- /MLSURV_RAW_HTML -->")
        md_lines.append("")

    @staticmethod
    def _format_param_range(param: str, value) -> str:
        """Format a single parameter range as 'param: min-max' or 'param: [a, b, c]'."""
        if isinstance(value, list):
            vals = [str(v) if v is not None else 'None' for v in value]
            return f"{param}: [{', '.join(vals)}]"

        if isinstance(value, tuple):
            if len(value) >= 2:
                low, high = value[0], value[1]
                high_str = 'n_features' if high is None else str(high)
                scale = value[2] if len(value) == 3 else None
                suffix = " (log)" if scale == 'log' else ""
                return f"{param}: {low}-{high_str}{suffix}"

        return f"{param}: {value}"

