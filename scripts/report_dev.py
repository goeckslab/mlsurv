"""Dev report generation and reading script.

Runs a configurable survival analysis pipeline on METABRIC data and
generates an HTML report.  The ``read`` subcommand lets you inspect
report content from the CLI (useful for CI and automated verification).

Usage
-----
    # Generate (default, backward-compatible)
    conda run -n mlsurv python scripts/report_dev.py
    conda run -n mlsurv python scripts/report_dev.py generate --expand-features
    conda run -n mlsurv python scripts/report_dev.py generate --inject-missing

    # Read
    conda run -n mlsurv python scripts/report_dev.py read
    conda run -n mlsurv python scripts/report_dev.py read path/to/report.html
    conda run -n mlsurv python scripts/report_dev.py read --sections summary evaluation
    conda run -n mlsurv python scripts/report_dev.py read --list-sections

Output
------
    examples/output/dev_report.html
"""

from __future__ import annotations

import argparse
import re
import sys
import webbrowser
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mlsurv import SurvivalLearner
from mlsurv.utils import load_survival_data

# -- Feature flags ------------------------------------------------------------
ENABLE_TUNING = True
ENABLE_BOOTSTRAP = True
ENABLE_SHAP = False
ENABLE_VALIDATION = True
N_TRIALS = 5
N_BOOTSTRAP = 50

# -- Paths --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "examples" / "data"
OUTPUT_DIR = ROOT / "examples" / "output"
REPORT_PATH = OUTPUT_DIR / "dev_report.html"


# =============================================================================
# Generate subcommand (existing logic)
# =============================================================================

def _expand_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to reach 20+ columns for testing wide-feature reports.

    Generates interaction terms, polynomial features, and binned versions
    of continuous columns from the base METABRIC feature set.
    """
    X = X.copy()
    continuous = [c for c in X.columns if X[c].nunique() > 5]
    binary = [c for c in X.columns if X[c].nunique() <= 5]

    # Squared terms for continuous features
    for col in continuous:
        X[f"{col}_sq"] = X[col] ** 2

    # Pairwise interactions among continuous features (first 3 pairs)
    for i, c1 in enumerate(continuous):
        for c2 in continuous[i + 1:]:
            X[f"{c1}_x_{c2}"] = X[c1] * X[c2]
            if X.shape[1] >= 25:
                break
        if X.shape[1] >= 25:
            break

    # Log(1 + |x|) for positive-skewed continuous features
    for col in continuous[:3]:
        X[f"{col}_log"] = np.log1p(X[col].abs())

    # Bin age into quartile indicator
    if "age_at_diagnosis" in X.columns:
        X["age_quartile"] = pd.qcut(
            X["age_at_diagnosis"], q=4, labels=False, duplicates="drop",
        )

    # Interaction of binary with top continuous
    if binary and continuous:
        for b in binary[:2]:
            X[f"{b}_x_{continuous[0]}"] = X[b] * X[continuous[0]]

    print(f"  Expanded to {X.shape[1]} features")
    return X


def _inject_missing(X: pd.DataFrame, fraction: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Randomly set *fraction* of numeric cells to NaN."""
    rng = np.random.default_rng(seed)
    X = X.copy()
    numeric_cols = X.select_dtypes(include="number").columns
    n_cells = len(X) * len(numeric_cols)
    n_missing = int(n_cells * fraction)
    rows = rng.integers(0, len(X), size=n_missing)
    cols = rng.choice(numeric_cols, size=n_missing)
    for r, c in zip(rows, cols):
        X.at[X.index[r], c] = np.nan
    total = X[numeric_cols].isna().sum().sum()
    print(f"  Injected {total} NaN values ({total / n_cells:.1%} of numeric cells)")
    return X


def load_data(expand: bool = False, inject_missing: bool = False):
    """Load METABRIC and optionally hold out a validation split."""
    print("Loading METABRIC dataset...")
    X, y = load_survival_data(
        str(DATA_DIR / "metabric.csv"),
        time_col="duration",
        event_col="event",
    )

    if expand:
        X = _expand_features(X)

    print(f"  n={len(X)}, features={X.shape[1]}, events={y['event'].sum()}")

    if inject_missing:
        X = _inject_missing(X)

    if not ENABLE_VALIDATION:
        return X, y, None, None

    idx = np.arange(len(X))
    idx_main, idx_val = train_test_split(idx, test_size=0.2, random_state=0)
    X_val = X.iloc[idx_val].reset_index(drop=True)
    y_val = y[idx_val]
    X = X.iloc[idx_main].reset_index(drop=True)
    y = y[idx_main]
    print(f"  Held out {len(X_val)} samples for validation")
    return X, y, X_val, y_val


def run_pipeline(X, y, X_val, y_val, outcome_name="event", time_unit=None,
                 manuscript=False, output_format="html"):
    """Set up learner, train models, optionally validate, and generate report."""
    print("\nSetting up learner...")
    learner = SurvivalLearner(X, y)

    learner.setup(
        test_fraction=0.25, random_state=42, n_jobs=-1,
        outcome_name=outcome_name, time_unit=time_unit,
    )

    print("\nRunning pipeline...")
    learner.run(
        models=["coxph", "coxnet", "xgbcox"],
        tune=ENABLE_TUNING,
        n_trials=N_TRIALS,
        bootstrap=ENABLE_BOOTSTRAP,
        n_bootstrap=N_BOOTSTRAP,
        include_shap=ENABLE_SHAP,
        verbose=True,
    )

    if ENABLE_VALIDATION and X_val is not None:
        print("\nRunning validation on held-out data...")
        learner.validate(X_val, y_val, name="Holdout")

    OUTPUT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_PATH
    feature_groups = {
        "MKI67": "Genomic", "EGFR": "Genomic", "PGR": "Genomic", "ERBB2": "Genomic",
        "hormone_therapy": "Treatment", "radiotherapy": "Treatment", "chemotherapy": "Treatment",
        "ER_positive": "Clinical", "age_at_diagnosis": "Clinical",
    }
    generate_kwargs = {"title": "Dev Report", "feature_groups": feature_groups}

    if output_format == "md":
        report_path = OUTPUT_DIR / "dev_report_manuscript.md"
    if manuscript:
        generate_kwargs["manuscript"] = True

    print(f"\nGenerating report -> {report_path}")
    learner.generate_report(str(report_path), **generate_kwargs)

    print(f"Report saved to: {report_path}")
    try:
        webbrowser.open(report_path.as_uri())
        print("Opened in browser.")
    except Exception as e:
        print(f"Could not open browser ({e}). Open manually: {report_path.as_uri()}")


def cmd_generate(args: argparse.Namespace) -> None:
    """Run the generate pipeline."""
    X, y, X_val, y_val = load_data(
        expand=args.expand_features,
        inject_missing=args.inject_missing,
    )
    run_pipeline(
        X, y, X_val, y_val,
        outcome_name=args.outcome_name,
        time_unit=args.time_unit,
        manuscript=args.manuscript,
        output_format=args.format,
    )


# =============================================================================
# Read subcommand
# =============================================================================

def _make_anchor(title: str) -> str:
    """Convert a section title to URL-safe anchor (mirrors HTMLRenderer)."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", str(title).lower()).strip("-")
    return slug or "section"


def _build_section_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build lookup dicts from registry keys and anchor IDs to canonical anchors.

    Returns (alias_to_anchor, anchor_to_title) where alias_to_anchor maps
    both registry keys and anchors to the canonical anchor, and
    anchor_to_title maps anchors to human-readable titles.
    """
    from mlsurv.reports.generator import SECTION_REGISTRY

    alias_to_anchor: Dict[str, str] = {}
    anchor_to_title: Dict[str, str] = {}

    for key, cls in SECTION_REGISTRY.items():
        title = cls.title
        anchor = _make_anchor(title)
        alias_to_anchor[key] = anchor
        alias_to_anchor[anchor] = anchor
        anchor_to_title[anchor] = title

    return alias_to_anchor, anchor_to_title


class ReportReader(HTMLParser):
    """State-machine HTML parser that extracts readable text per section."""

    def __init__(self, max_table_rows: int = 20):
        super().__init__()
        self.max_table_rows = max_table_rows

        # Output: anchor_id -> extracted text
        self.sections: Dict[str, str] = {}

        # Current section tracking
        self._current_anchor: Optional[str] = None
        self._current_title: Optional[str] = ""
        self._parts: List[str] = []

        # Skip-depth counters for suppressed elements
        self._skip_script = 0
        self._skip_style = 0
        self._skip_figure = 0
        self._skip_details = 0

        # Table accumulation
        self._in_table = False
        self._table_rows: List[List[str]] = []
        self._current_row: List[str] = []
        self._in_cell = False
        self._cell_text: List[str] = []

        # List tracking
        self._in_ol = 0
        self._in_ul = 0
        self._ol_counter = 0
        self._li_needs_prefix = False

        # Heading tracking
        self._in_heading: Optional[str] = None
        self._heading_text: List[str] = []

        # Tab button tracking
        self._tab_buttons: Dict[str, str] = {}  # target -> label
        self._in_tab_button = False
        self._tab_button_target: Optional[str] = None
        self._tab_button_text: List[str] = []

        # Paragraph tracking
        self._in_p = False

    @property
    def _skipping(self) -> bool:
        return (self._skip_script > 0 or self._skip_style > 0
                or self._skip_figure > 0 or self._skip_details > 0)

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attr_dict = dict(attrs)
        classes = attr_dict.get("class", "").split()

        # --- Skip zones ---
        if tag == "script":
            self._skip_script += 1
            return
        if tag == "style":
            self._skip_style += 1
            return
        if tag == "div" and "figure-container" in classes:
            self._skip_figure += 1
            return
        if self._skip_figure and tag == "div":
            self._skip_figure += 1
            return
        if tag == "details" and "figure-access" in classes:
            self._skip_details += 1
            return
        if self._skip_details and tag in ("details", "div", "summary", "pre", "code"):
            if tag == "details":
                self._skip_details += 1
            return

        if self._skipping:
            return

        # --- Section boundaries ---
        if tag == "section" and "report-section" in classes:
            anchor = attr_dict.get("id", "")
            self._current_anchor = anchor
            self._parts = []
            return

        if self._current_anchor is None:
            # Tab buttons at report level (section navigation)
            if tag == "button" and "tab-button" in classes:
                target = attr_dict.get("data-tab-target", "")
                self._in_tab_button = True
                self._tab_button_target = target
                self._tab_button_text = []
            return

        # --- Inside a section ---

        # Tab buttons within sections
        if tag == "button" and "tab-button" in classes:
            target = attr_dict.get("data-tab-target", "")
            self._in_tab_button = True
            self._tab_button_target = target
            self._tab_button_text = []
            return

        # Tab panels within sections
        if tag == "div" and "tab-panel" in classes:
            panel_id = attr_dict.get("id", "")
            label = self._tab_buttons.get(panel_id, panel_id)
            if label:
                self._parts.append(f"\n[Tab: {label}]\n")
            return

        # Headings
        if tag in ("h2", "h3", "h4", "h5", "h6"):
            self._in_heading = tag
            self._heading_text = []
            return

        # Paragraph
        if tag == "p":
            self._in_p = True
            return

        # Tables
        if tag == "table":
            self._in_table = True
            self._table_rows = []
            return
        if self._in_table:
            if tag == "tr":
                self._current_row = []
            elif tag in ("td", "th"):
                self._in_cell = True
                self._cell_text = []
            elif tag == "br" and self._in_cell:
                self._cell_text.append(" | ")
            return

        # Lists
        if tag == "ul":
            self._in_ul += 1
            return
        if tag == "ol":
            self._in_ol += 1
            self._ol_counter = 0
            return
        if tag == "li":
            if self._in_ol:
                self._ol_counter += 1
            self._li_needs_prefix = True
            return

        # Line break
        if tag == "br":
            self._parts.append("\n")
            return

        # Horizontal rule
        if tag == "hr":
            self._parts.append("\n---\n")
            return

    def handle_startendtag(self, tag: str, attrs) -> None:
        """Route self-closing tags (e.g. <br/>) through handle_starttag."""
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        # --- Skip zone exits ---
        if tag == "script" and self._skip_script > 0:
            self._skip_script -= 1
            return
        if tag == "style" and self._skip_style > 0:
            self._skip_style -= 1
            return
        if tag == "div" and self._skip_figure > 0:
            self._skip_figure -= 1
            if self._skip_figure == 0:
                if self._current_anchor is not None:
                    self._parts.append("\n[Figure]\n")
            return
        if tag == "details" and self._skip_details > 0:
            self._skip_details -= 1
            return

        if self._skipping:
            return

        # --- Tab button end ---
        if tag == "button" and self._in_tab_button:
            label = "".join(self._tab_button_text).strip()
            if self._tab_button_target:
                self._tab_buttons[self._tab_button_target] = label
            self._in_tab_button = False
            self._tab_button_target = None
            return

        if self._current_anchor is None:
            return

        # --- Section end ---
        if tag == "section":
            text = "".join(self._parts).strip()
            if text:
                self.sections[self._current_anchor] = text
            self._current_anchor = None
            self._parts = []
            return

        # --- Headings ---
        if tag == self._in_heading:
            text = "".join(self._heading_text).strip()
            if text:
                if tag == "h2":
                    pass  # Title already shown in section header
                else:
                    level = int(tag[1])
                    prefix = "#" * level
                    self._parts.append(f"\n{prefix} {text}\n")
            self._in_heading = None
            return

        # --- Paragraph ---
        if tag == "p" and self._in_p:
            self._in_p = False
            self._parts.append("\n")
            return

        # --- Tables ---
        if tag == "table" and self._in_table:
            self._in_table = False
            self._format_table()
            return
        if self._in_table:
            if tag in ("td", "th"):
                self._in_cell = False
                self._current_row.append("".join(self._cell_text).strip())
            elif tag == "tr":
                if self._current_row:
                    self._table_rows.append(self._current_row)
            return

        # --- Lists ---
        if tag == "ul":
            self._in_ul = max(0, self._in_ul - 1)
            self._parts.append("\n")
            return
        if tag == "ol":
            self._in_ol = max(0, self._in_ol - 1)
            self._parts.append("\n")
            return
        if tag == "li":
            self._parts.append("\n")
            return

    def handle_data(self, data: str) -> None:
        if self._skipping:
            return

        # Tab button text
        if self._in_tab_button:
            self._tab_button_text.append(data)
            return

        if self._current_anchor is None:
            return

        # Heading text
        if self._in_heading:
            self._heading_text.append(data)
            return

        # Table cell text
        if self._in_table and self._in_cell:
            self._cell_text.append(data)
            return

        # List item prefix — emit only on the first text chunk per <li>;
        # nested inline tags (e.g. <strong>, <em>) split a single <li>'s
        # text into multiple handle_data calls.
        if self._in_ul or self._in_ol:
            text = data.strip()
            if not text:
                return
            if self._li_needs_prefix:
                prefix = f"{self._ol_counter}. " if self._in_ol else "- "
                self._parts.append(f"{prefix}{text}")
                self._li_needs_prefix = False
            else:
                if self._parts and self._parts[-1] and not self._parts[-1][-1].isspace():
                    self._parts.append(" ")
                self._parts.append(text)
            return

        # Regular text — add a space separator when adjacent to previous text
        # to avoid "on1903patients" when <strong>1903</strong> is between words
        text = data.strip()
        if text:
            if self._parts and self._parts[-1] and not self._parts[-1][-1].isspace():
                self._parts.append(" ")
            self._parts.append(text)

    def _format_table(self) -> None:
        """Format accumulated table rows as a pipe-delimited markdown table."""
        if not self._table_rows:
            return

        header = self._table_rows[0]
        body = self._table_rows[1:]

        # Calculate column widths
        all_rows = [header] + body[:self.max_table_rows]
        n_cols = max(len(r) for r in all_rows) if all_rows else 0
        if n_cols == 0:
            return

        # Pad short rows
        for row in all_rows:
            while len(row) < n_cols:
                row.append("")

        widths = [
            max(len(row[i]) for row in all_rows)
            for i in range(n_cols)
        ]
        widths = [max(w, 3) for w in widths]  # minimum 3 for separator

        def fmt_row(row: List[str]) -> str:
            cells = [cell.ljust(w) for cell, w in zip(row, widths)]
            return "| " + " | ".join(cells) + " |"

        lines = [fmt_row(header)]
        lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")

        shown = body[:self.max_table_rows]
        for row in shown:
            while len(row) < n_cols:
                row.append("")
            lines.append(fmt_row(row))

        remaining = len(body) - len(shown)
        if remaining > 0:
            lines.append(f"... ({remaining} more rows)")

        self._parts.append("\n" + "\n".join(lines) + "\n")


def cmd_read(args: argparse.Namespace) -> None:
    """Read and display report content."""
    alias_to_anchor, anchor_to_title = _build_section_maps()

    if args.list_sections:
        print("Available sections:\n")
        # Reverse map: anchor -> list of aliases
        anchor_aliases: Dict[str, List[str]] = {}
        for alias, anchor in alias_to_anchor.items():
            if alias != anchor:
                anchor_aliases.setdefault(anchor, []).append(alias)
        for anchor, title in anchor_to_title.items():
            aliases = anchor_aliases.get(anchor, [])
            alias_str = f"  (aliases: {', '.join(aliases)})" if aliases else ""
            print(f"  {anchor}{alias_str}")
            print(f"    Title: {title}")
        return

    report_path = Path(args.path)
    if not report_path.exists():
        print(f"Error: Report file not found: {report_path}", file=sys.stderr)
        sys.exit(1)

    html = report_path.read_text(encoding="utf-8")
    reader = ReportReader(max_table_rows=args.max_table_rows)
    reader.feed(html)

    # Determine which sections to show
    if args.sections:
        requested_anchors = []
        for name in args.sections:
            anchor = alias_to_anchor.get(name) or alias_to_anchor.get(
                _make_anchor(name)
            )
            if anchor is None:
                # Try direct match against parsed sections
                if name in reader.sections:
                    anchor = name
                else:
                    print(f"Warning: Unknown section '{name}', skipping.",
                          file=sys.stderr)
                    continue
            requested_anchors.append(anchor)
        section_ids = requested_anchors
    else:
        section_ids = list(reader.sections.keys())

    if not section_ids:
        print("No sections to display.", file=sys.stderr)
        sys.exit(1)

    for anchor in section_ids:
        text = reader.sections.get(anchor)
        if text is None:
            print(f"Warning: Section '{anchor}' not found in report.",
                  file=sys.stderr)
            continue
        title = anchor_to_title.get(anchor, anchor)
        print(f"\n=== {title} ===\n")
        print(text)
        print()


# =============================================================================
# CLI entry point
# =============================================================================

def _add_generate_args(parser: argparse.ArgumentParser) -> None:
    """Add generate-subcommand arguments to a parser."""
    parser.add_argument(
        "--expand-features", action="store_true",
        help="Add derived features (interactions, polynomials) to reach 20+ columns",
    )
    parser.add_argument(
        "--inject-missing", action="store_true",
        help="Inject ~15%% NaN values into numeric cells to test imputation handling",
    )
    parser.add_argument(
        "--outcome-name", default="event",
        help="Outcome name for display labels (default: 'event')",
    )
    parser.add_argument(
        "--time-unit", default=None,
        help="Time unit for axis labels (e.g., 'months', 'years')",
    )
    parser.add_argument(
        "--manuscript", action="store_true",
        help="Also generate manuscript markdown alongside HTML",
    )
    parser.add_argument(
        "--format", choices=["html", "md"], default="html",
        help="Output format (default: html)",
    )


def main():
    """Entry point with subcommands: generate (default) and read."""
    parser = argparse.ArgumentParser(
        description="Dev report generation and reading tool",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- generate subcommand ---
    gen_parser = subparsers.add_parser(
        "generate", help="Generate a dev report (default)",
    )
    _add_generate_args(gen_parser)

    # --- read subcommand ---
    read_parser = subparsers.add_parser(
        "read", help="Read and display report content",
    )
    read_parser.add_argument(
        "path", nargs="?", default=str(REPORT_PATH),
        help=f"Path to report HTML (default: {REPORT_PATH})",
    )
    read_parser.add_argument(
        "--sections", "-s", nargs="+", metavar="NAME",
        help="Filter by section registry key or HTML anchor ID",
    )
    read_parser.add_argument(
        "--list-sections", "-l", action="store_true",
        help="List available section names and exit",
    )
    read_parser.add_argument(
        "--max-table-rows", type=int, default=20,
        help="Maximum table body rows to display (default: 20)",
    )

    args = parser.parse_args()

    if args.command is None or args.command == "generate":
        if args.command is None:
            standalone = argparse.ArgumentParser(description="Generate dev report")
            _add_generate_args(standalone)
            args = standalone.parse_args()
        cmd_generate(args)
    elif args.command == "read":
        cmd_read(args)


if __name__ == "__main__":
    main()
