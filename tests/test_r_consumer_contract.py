"""Regression: R consumer scripts must reference xlsx sheet names that exist.

Catches the class of bug where a producer-side rename
(``subgroup`` → ``subpopulation``) shifted Excel's 31-char sheet-name
truncation point and silently broke hardcoded ``sheet = "..."`` literals
in ``scripts/figures/*.R``.

The coverage here has two layers:

1. **Fast-tier unit test** ``test_r_literals_match_producer_truncation``
   hardcodes the full untruncated producer-side keys the R script expects,
   runs them through :func:`ExportMixin._truncate_sheet_names`, and asserts
   the result is a superset of the parsed R literals. This catches a
   producer-side rename the moment it happens, without requiring
   ``examples/output/vig7/results_OS.xlsx`` to be regenerated —
   the stale-xlsx blind spot flagged in PR review.

2. **Integration-tier test** ``test_r_sheet_references_exist`` additionally
   confirms the committed xlsx on disk still contains the R-referenced
   sheets. This catches out-of-band xlsx regeneration that would otherwise
   drift from the R script.

When a new R consumer script ships, add a ``(r_script, xlsx, producer_keys)``
tuple to :data:`R_SCRIPTS`. Both tests will automatically parametrize over
the new entry. The ``producer_keys`` list must enumerate every full
(untruncated) key the script's sheet literals originate from — if you
can't name them, you're probably missing a producer-side invariant the
rename audit is meant to protect.
"""
import re
import warnings
from pathlib import Path

import pandas as pd
import pytest

from mlsurv._warnings import SuggestionWarning
from mlsurv.learner._export import ExportMixin

REPO_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Consumer contract registry. Adding a new R consumer:
#
#   1. Append a dict with r_script, xlsx, and producer_keys (full
#      untruncated keys the producer generates for this consumer).
#   2. Keep producer_keys in lockstep with the actual producer code —
#      if you rename a constant in mlsurv.learner._export, update this
#      list and the R script in the same commit.
# --------------------------------------------------------------------------

R_SCRIPTS = [
    pytest.param(
        REPO_ROOT / "scripts" / "figures" / "example_r_xlsx_consumer.R",
        REPO_ROOT / "examples" / "output" / "vig7" / "results_OS.xlsx",
        # Full untruncated producer keys corresponding to the four
        # sheet = "..." literals in example_r_xlsx_consumer.R. These are built
        # by mlsurv.learner._export._collect_subpopulation_dfs (subpopulation
        # prefix) and _collect_model_eval_dfs / benchmark_ prefix for the
        # bootstrap-based variants. If you rename any of these producer-side,
        # update this list AND example_r_xlsx_consumer.R in the same commit.
        [
            "subpopulation_Cancer Type_cohort",
            "subpopulation_Cancer Type_auc_t",
            "eval_bootstrap_subpopulation_auc_over_time",
            "benchmark_eval_bootstrap_subpopulation_auc_over_time",
        ],
        id="example_r_xlsx_consumer.R",
    ),
]

# Matches `sheet = "..."` and `sheet = '...'` (R accepts both quote styles).
# Positional-arg calls like `read_excel(path, "name")` are intentionally
# out of scope — the test's empty-literals guard below fails loudly if a
# future R script moves to a form this regex can't parse.
_SHEET_LITERAL_RE = re.compile(r'''sheet\s*=\s*["']([^"']+)["']''')


def _parse_r_literals(r_file: Path) -> set[str]:
    """Return the set of sheet-name literals referenced in an R script."""
    return set(_SHEET_LITERAL_RE.findall(r_file.read_text(encoding="utf-8")))


@pytest.mark.parametrize("r_file, xlsx_file, producer_keys", R_SCRIPTS)
def test_r_literals_match_producer_truncation(
    r_file: Path,
    xlsx_file: Path,  # unused here; kept for parametrize symmetry
    producer_keys: list[str],
) -> None:
    """Producer-side keys, after 31-char truncation, must cover every R literal.

    Fast-tier unit guard. Runs with no I/O beyond reading the R script,
    so it catches a rename the moment the producer constants change — no
    xlsx regeneration required.
    """
    assert r_file.exists(), f"R script not found: {r_file}"

    expected = _parse_r_literals(r_file)
    assert expected, (
        f"{r_file.name} contains no `sheet = \"...\"` literals — either the "
        f"script changed form (e.g. now uses excel_sheets() dynamically) or "
        f"should be removed from R_SCRIPTS."
    )

    with warnings.catch_warnings():
        # Producer keys intentionally include names that trip truncation;
        # the warning from _truncate_sheet_names is not the subject of this
        # test (that's covered in tests/test_export_results.py).
        warnings.simplefilter("ignore", SuggestionWarning)
        mapping = ExportMixin._truncate_sheet_names(producer_keys)
    produced = set(mapping.values())

    missing = sorted(expected - produced)
    assert not missing, (
        f"{r_file.name} references sheets {missing} that the current "
        f"producer code does NOT generate after 31-char truncation of "
        f"{producer_keys!r}. Full truncation result: "
        f"{sorted(produced)}. This usually means a producer-side rename "
        f"shifted the truncation point — update the R script and this "
        f"test's hardcoded producer_keys together."
    )


@pytest.mark.integration
@pytest.mark.parametrize("r_file, xlsx_file, producer_keys", R_SCRIPTS)
def test_r_sheet_references_exist(
    r_file: Path,
    xlsx_file: Path,
    producer_keys: list[str],  # unused here; kept for parametrize symmetry
) -> None:
    """Every ``sheet = "..."`` literal in an R consumer must exist in its xlsx.

    Integration-tier guard that catches drift between the local xlsx
    and the R script — e.g. an xlsx regenerated from a stale branch. Runs
    under ``--run-full`` only. The xlsx is a gitignored local artifact
    (``examples/output/vig7/results_OS.xlsx``) produced by running
    vignette 07; when it is absent (e.g. a fresh clone that has not run the
    vignette) the drift check is skipped rather than failed.
    """
    assert r_file.exists(), f"R script not found: {r_file}"
    if not xlsx_file.exists():
        pytest.skip(
            f"{xlsx_file} not present — gitignored local artifact. "
            f"Regenerate via vignette 07 "
            f"(examples/vignette_07_immunotherapy_case_study.ipynb) "
            f"to run this drift check."
        )

    expected = _parse_r_literals(r_file)
    assert expected, (
        f"{r_file.name} contains no `sheet = \"...\"` literals — either the "
        f"script changed form (e.g. now uses excel_sheets() dynamically) or "
        f"should be removed from R_SCRIPTS."
    )

    actual = set(pd.ExcelFile(xlsx_file).sheet_names)
    missing = sorted(expected - actual)
    assert not missing, (
        f"{r_file.name} references sheets not in {xlsx_file.name}: {missing}. "
        f"Actual sheets (first 10): {sorted(actual)[:10]}. This usually means "
        f"a producer-side rename shifted Excel's 31-char truncation point."
    )
