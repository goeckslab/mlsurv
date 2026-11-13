"""TRIPOD+AI Checklist appendix for report generation.

Cross-references all 52 TRIPOD+AI items against the mlsurv report,
showing which are auto-filled, which need researcher input, and where
to find each piece of information.

Reference: Collins GS et al. (2024) TRIPOD+AI statement. BMJ 385:e078378.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mlsurv.reports.sections.base import BaseSection


# Status constants
_AUTO = "Auto"
_SCAFFOLD = "Scaffold"
_RESEARCHER = "Researcher"
_NA = "N/A"

_FALLBACK = "See Data & Preprocessing"

# ---------------------------------------------------------------------------
# TRIPOD+AI checklist item definitions
# Each tuple: (item_id, requirement, status, reference_or_guidance)
# reference_or_guidance is a static string; dynamic values are injected at
# generate-time for Auto items whose reference starts with "$".
# ---------------------------------------------------------------------------

_CHECKLIST_ITEMS: list[tuple[str, str, str, str]] = [
    # TITLE & ABSTRACT
    ("1", "Identify study type, target population, outcome",
     _RESEARCHER, "Provide in manuscript title"),
    ("2", "Structured abstract",
     _SCAFFOLD, "Adapt Executive Summary for manuscript abstract"),

    # INTRODUCTION
    ("3a", "Healthcare context, rationale, existing models",
     _RESEARCHER, "Provide clinical background"),
    ("3b", "Target population, intended purpose, intended users",
     _RESEARCHER, "Provide study context"),
    ("3c", "Known health inequalities",
     _RESEARCHER, "Provide equity considerations"),
    ("4", "Study objectives (development vs. validation)",
     _SCAFFOLD, "See Executive Summary — adapt objective statement"),

    # METHODS — Data
    ("5a", "Data sources, rationale, representativeness",
     _RESEARCHER, "Describe data provenance"),
    ("5b", "Dates of participant data",
     _RESEARCHER, "Provide collection dates"),

    # METHODS — Participants
    ("6a", "Study setting",
     _RESEARCHER, "Describe clinical setting"),
    ("6b", "Eligibility criteria",
     _RESEARCHER, "Describe inclusion/exclusion criteria"),
    ("6c", "Treatment details",
     _RESEARCHER, "Describe treatments received"),

    # METHODS — Data Preparation
    ("7", "Data preprocessing and quality checking",
     _AUTO, "See Data & Preprocessing"),

    # METHODS — Outcome
    ("8a", "Outcome definition and time horizon",
     _SCAFFOLD, "See Data & Preprocessing — add clinical outcome definition"),
    ("8b", "Outcome assessor qualifications",
     _RESEARCHER, "Describe assessment procedures"),
    ("8c", "Blinding of outcome assessment",
     _RESEARCHER, "Describe blinding procedures"),

    # METHODS — Predictors
    ("9a", "Choice of initial predictors, pre-selection rationale",
     _SCAFFOLD, "See Feature Analysis — add selection rationale"),
    ("9b", "Predictor definitions and measurement",
     _SCAFFOLD, "See Feature Analysis — add clinical definitions"),
    ("9c", "Predictor assessor qualifications",
     _RESEARCHER, "Describe measurement procedures"),

    # METHODS — Sample Size
    ("10", "Sample size justification",
     _AUTO, "$epv"),

    # METHODS — Missing Data
    ("11", "Missing data handling",
     _AUTO, "See Data & Preprocessing"),

    # METHODS — Analytical Methods
    ("12a", "Data partitioning strategy",
     _AUTO, "$split"),
    ("12b", "Predictor handling (rescaling, transformation)",
     _AUTO, "See Data & Preprocessing"),
    ("12c", "Model type, building steps, tuning, internal validation",
     _AUTO, "See Model Development + Feature Analysis (SHAP, importance)"),
    ("12d", "Heterogeneity across clusters",
     _SCAFFOLD, "Use subpopulation analysis to evaluate per-site performance"),
    ("12e", "Performance measures and plots with rationale",
     _AUTO, "See Model Evaluation"),
    ("12f", "Model updating / recalibration",
     _NA, "Not applicable to current analysis"),
    ("12g", "How predictions were calculated",
     _AUTO, "See Environment & Reproducibility"),

    # METHODS — Class Imbalance, Fairness, Output
    ("13", "Class imbalance handling (censoring)",
     _AUTO, "$censoring"),
    ("14", "Fairness approaches",
     _SCAFFOLD, "See Subpopulation Analysis for per-group discrimination and calibration metrics"),
    ("15", "Model output specification",
     _AUTO, "See Model Evaluation — risk scores"),

    # METHODS — Training vs. Evaluation, Ethics
    ("16", "Differences between dev and eval data",
     _AUTO, "$validation_comparison"),
    ("17", "Ethical approval",
     _RESEARCHER, "Provide IRB/ethics approval details"),

    # OPEN SCIENCE
    ("18a", "Funding",
     _RESEARCHER, "Provide funding sources"),
    ("18b", "Conflicts of interest",
     _RESEARCHER, "Declare conflicts of interest"),
    ("18c", "Protocol access",
     _RESEARCHER, "Provide protocol availability"),
    ("18d", "Registration",
     _RESEARCHER, "Provide study registration details"),
    ("18e", "Data sharing",
     _RESEARCHER, "Describe data availability"),
    ("18f", "Code sharing",
     _AUTO, "See Environment & Reproducibility"),

    # PATIENT & PUBLIC INVOLVEMENT
    ("19", "Patient and public involvement",
     _RESEARCHER, "Describe PPI activities"),

    # RESULTS
    ("20a", "Participant flow (N, events, follow-up)",
     _AUTO, "$flow"),
    ("20b", "Participant characteristics table",
     _AUTO, "See Data & Preprocessing"),
    ("20c", "Dev vs. eval data comparison",
     _AUTO, "$validation_comparison"),
    ("21", "N participants and events per analysis phase",
     _AUTO, "$phase_n"),
    ("22", "Full model specification",
     _AUTO, "See Model Development (parameters) and Environment & Reproducibility (pipeline)"),
    ("23a", "Performance with CIs, subpopulation results",
     _AUTO, "See Model Evaluation"),
    ("23b", "Heterogeneity across clusters",
     _SCAFFOLD, "Use subpopulation analysis to evaluate per-site performance"),
    ("24", "Model updating results",
     _NA, "Not applicable to current analysis"),

    # DISCUSSION
    ("25", "Overall interpretation",
     _SCAFFOLD, "See Executive Summary — expand with clinical context"),
    ("26", "Limitations",
     _SCAFFOLD, "See Limitations & Caveats — add study-specific limitations"),
    ("27a", "Handling poor input data at deployment",
     _RESEARCHER, "Describe deployment data quality plans"),
    ("27b", "User expertise requirements",
     _RESEARCHER, "Describe intended user qualifications"),
    ("27c", "Next steps, generalizability",
     _RESEARCHER, "Discuss external applicability"),
]

# Category boundaries for grouping in the table
_CATEGORIES: list[tuple[str, list[str]]] = [
    ("Title & Abstract", ["1", "2"]),
    ("Introduction", ["3a", "3b", "3c", "4"]),
    ("Methods — Data", ["5a", "5b"]),
    ("Methods — Participants", ["6a", "6b", "6c"]),
    ("Methods — Data Preparation", ["7"]),
    ("Methods — Outcome", ["8a", "8b", "8c"]),
    ("Methods — Predictors", ["9a", "9b", "9c"]),
    ("Methods — Sample Size", ["10"]),
    ("Methods — Missing Data", ["11"]),
    ("Methods — Analytical Methods",
     ["12a", "12b", "12c", "12d", "12e", "12f", "12g"]),
    ("Methods — Imbalance, Fairness, Output", ["13", "14", "15"]),
    ("Methods — Dev vs. Eval, Ethics", ["16", "17"]),
    ("Open Science", ["18a", "18b", "18c", "18d", "18e", "18f"]),
    ("Patient & Public Involvement", ["19"]),
    ("Results", ["20a", "20b", "20c", "21", "22", "23a", "23b", "24"]),
    ("Discussion", ["25", "26", "27a", "27b", "27c"]),
]


@dataclass
class _SplitCounts:
    """Pre-computed counts from the learner for dynamic checklist values."""

    n_train: int
    n_test: int
    n_features: int
    n_events_train: int
    n_events_test: int
    test_fraction: float


class TripodChecklistSection(BaseSection):
    """TRIPOD+AI Checklist appendix.

    Cross-references all TRIPOD+AI items against the mlsurv report,
    auto-filling dynamic values where possible and providing guidance
    for researcher-supplied items.
    """

    name: str = "tripod_checklist"
    title: str = "TRIPOD+AI Checklist"

    def is_available(self) -> bool:
        """Available whenever the learner has been set up."""
        return self.learner.is_setup

    def generate(self) -> dict[str, Any]:
        """Generate the TRIPOD+AI checklist appendix."""
        dynamic = self._compute_dynamic_values()

        # 12f (model updating / recalibration) flips from N/A to Auto when an
        # S(t|x) calibrator has been attached via train(calibrate=...). Partial
        # coverage (some models calibrated, others not) is reflected in the value.
        all_models = self.learner.models or {}
        calibrated = [
            n for n in all_models if self.learner._has_calibrator(n)
        ]
        recal_override = None
        if calibrated:
            n_recal = len(calibrated)
            n_total = len(all_models)
            if n_recal == n_total and n_total > 0:
                recal_value = (
                    f"Calibrated all {n_total} models via train(calibrate=...)."
                )
            else:
                recal_value = (
                    f"Partial: {n_recal}/{n_total} models calibrated via "
                    f"train(calibrate=...)."
                )
            recal_override = (_AUTO, recal_value)

        rows: list[dict[str, str]] = []
        for item_id, requirement, status, reference in _CHECKLIST_ITEMS:
            if item_id == "12f" and recal_override is not None:
                status, ref = recal_override
            else:
                ref = self._resolve_reference(reference, dynamic)
            rows.append({
                "Item": item_id,
                "Requirement": requirement,
                "Status": status,
                "Reference / Value": ref,
            })

        checklist_df = pd.DataFrame(rows)
        self.data["checklist"] = checklist_df

        md_lines = self._build_markdown(checklist_df)

        return {
            "markdown": "\n".join(md_lines),
            "figures": [],
        }

    # ------------------------------------------------------------------
    # Dynamic value computation
    # ------------------------------------------------------------------

    def _extract_split_counts(self) -> _SplitCounts | None:
        """Extract train/test counts from the learner's public accessors.

        Returns None if the data attributes are unset or inconsistent.
        """
        learner = self.learner
        try:
            n_train = int(len(learner.X_train))
            n_test = int(len(learner.X_test))
            n_features = int(learner.X_train.shape[1])
            n_events_train = int(learner.y_train["event"].sum())
            n_events_test = int(learner.y_test["event"].sum())

            total = n_train + n_test
            pct = learner.test_fraction
            if pct is None and total > 0:
                pct = n_test / total
            pct = float(pct)

            return _SplitCounts(
                n_train=n_train,
                n_test=n_test,
                n_features=n_features,
                n_events_train=n_events_train,
                n_events_test=n_events_test,
                test_fraction=pct,
            )
        except (AttributeError, TypeError, ZeroDivisionError) as exc:
            self._log_recoverable_exception(
                exc, "tripod: extracting split counts",
                warn_user=True,
                user_message="TRIPOD checklist: could not extract split counts.",
            )
            return None

    def _compute_dynamic_values(self) -> dict[str, str]:
        """Compute dynamic checklist values from the learner."""
        values: dict[str, str] = {}
        counts = self._extract_split_counts()

        # EPV (events-per-variable)
        if counts is not None and counts.n_features > 0:
            epv = counts.n_events_train / counts.n_features
            values["epv"] = (
                f"EPV = {epv:.1f} "
                f"({counts.n_events_train:,} events / "
                f"{counts.n_features} features). {_FALLBACK}"
            )
        else:
            values["epv"] = _FALLBACK

        # Train/test split
        if counts is not None:
            pct = counts.test_fraction
            values["split"] = (
                f"{1 - pct:.0%} / {pct:.0%} train/test split "
                f"({counts.n_train:,} / {counts.n_test:,}). {_FALLBACK}"
            )
        else:
            values["split"] = _FALLBACK

        # Censoring rate
        if counts is not None and counts.n_train > 0:
            n_censored = counts.n_train - counts.n_events_train
            censoring_rate = n_censored / counts.n_train
            values["censoring"] = (
                f"Censoring rate: {censoring_rate:.1%} "
                f"({n_censored:,} censored / "
                f"{counts.n_train:,} training samples). {_FALLBACK}"
            )
        else:
            values["censoring"] = _FALLBACK

        # Participant flow
        if counts is not None:
            n_total = counts.n_train + counts.n_test
            n_events_total = counts.n_events_train + counts.n_events_test
            event_rate = n_events_total / n_total if n_total > 0 else 0
            values["flow"] = (
                f"N = {n_total:,}, "
                f"events = {n_events_total:,} "
                f"(rate {event_rate:.1%}). {_FALLBACK}"
            )
        else:
            values["flow"] = _FALLBACK

        # Phase N (train/test counts)
        if counts is not None:
            values["phase_n"] = (
                f"Train: {counts.n_train:,} "
                f"({counts.n_events_train:,} events), "
                f"Test: {counts.n_test:,} "
                f"({counts.n_events_test:,} events). {_FALLBACK}"
            )
        else:
            values["phase_n"] = _FALLBACK

        # Validation comparison (conditional)
        validation_results = self.learner.validation_results
        if validation_results:
            cohort_names = list(validation_results.keys())
            values["validation_comparison"] = (
                f"See External Validation ({', '.join(cohort_names)})"
            )
        else:
            values["validation_comparison"] = (
                "No external validation performed"
            )

        return values

    # ------------------------------------------------------------------
    # Reference resolution
    # ------------------------------------------------------------------

    def _resolve_reference(self, reference: str, dynamic: dict[str, str]) -> str:
        """Replace $key placeholders with computed dynamic values."""
        if reference.startswith("$"):
            key = reference[1:]
            value = dynamic.get(key)
            if value is None:
                self._logger.warning(
                    "TRIPOD checklist: no dynamic value for '$%s'", key
                )
                return "See report sections"
            return value
        return reference

    # ------------------------------------------------------------------
    # Markdown rendering
    # ------------------------------------------------------------------

    def _build_markdown(self, checklist_df: pd.DataFrame) -> list[str]:
        """Build full section markdown."""
        lines: list[str] = []

        # Intro
        lines.append(
            "This appendix maps the TRIPOD+AI reporting checklist "
            "(Collins et al., BMJ 2024) against this report. "
            "Each item shows its coverage status and where to find "
            "the information — either auto-filled from the analysis "
            "or flagged for the researcher to provide."
        )
        lines.append("")

        # Status legend
        lines.append("**Status key:**")
        lines.append("")
        lines.append(
            f"- **{_AUTO}** — Value or section reference provided by mlsurv"
        )
        lines.append(
            f"- **{_SCAFFOLD}** — Placeholder with guidance; "
            f"researcher completes for manuscript"
        )
        lines.append(
            f"- **{_RESEARCHER}** — Outside mlsurv scope; "
            f"researcher provides for manuscript"
        )
        lines.append(
            f"- **{_NA}** — Not applicable to this analysis"
        )
        lines.append("")

        # Summary counts
        status_counts = checklist_df["Status"].value_counts()
        lines.append("**Coverage summary:** ")
        parts = []
        for status in [_AUTO, _SCAFFOLD, _RESEARCHER, _NA]:
            count = status_counts.get(status, 0)
            parts.append(f"{status}: {count}")
        lines.append(" | ".join(parts))
        lines.append("")

        # Grouped tables
        item_lookup = {
            row["Item"]: row
            for _, row in checklist_df.iterrows()
        }

        # Warn about mismatched IDs between categories and checklist items
        category_ids = {iid for _, ids in _CATEGORIES for iid in ids}
        missing = category_ids - set(item_lookup)
        if missing:
            self._logger.warning(
                "TRIPOD categories reference undefined items: %s", missing
            )

        for category, item_ids in _CATEGORIES:
            lines.append(f"### {category}")
            lines.append("")

            cat_rows = [item_lookup[iid] for iid in item_ids if iid in item_lookup]
            if not cat_rows:
                continue

            cat_df = pd.DataFrame(cat_rows)
            lines.append(self._format_table(cat_df))
            lines.append("")

        return lines
