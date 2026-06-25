"""Build the combined supplementary table for Vignette 03 (public benchmarks).

Vignette 03 saves its results to three CSVs in ``examples/output/``:

- ``vig3_jmlr.csv``   — C^td (5-fold CV) vs Kvamme targets
- ``vig3_sksurv.csv`` — Harrell C (test) vs scikit-survival targets
- ``vig3_tuned.csv``  — tuned DeepSurv/DeepHit C^td on SUPPORT (optional)

This script reads those CSVs and writes one long-format table, one row per
model/dataset, ready to paste into a Word table. The tuned SUPPORT values are
folded into their default rows as a "default / tuned" add-in. Re-run it after a
fresh Vignette 03 run to refresh the numbers: every score, target, and gap comes
from the saved CSVs. Only the display names, row order, and target-source
citations are defined here in the script.

Usage:
    python scripts/figures/make_vignette_03_supp_table.py
    python scripts/figures/make_vignette_03_supp_table.py --output-dir path/to/output --out table.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Repo root is two levels up from scripts/figures/.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples" / "output"

# Human-readable model names, in the order Vignette 03 reports them.
MODEL_DISPLAY = {
    "coxph": "CoxPH",
    "coxnet": "CoxNet",
    "rsf": "RSF",
    "deepsurv": "DeepSurv",
    "deephit": "DeepHit",
    "fssvm": "FSSVM",
}

# Where each published target comes from. The JMLR suite draws on two papers:
# CoxPH/DeepSurv/RSF from Kvamme et al. (2019), DeepHit from Kvamme & Borgan (2021).
JMLR_SOURCE = {
    "coxph": "Kvamme et al. (2019)",
    "deepsurv": "Kvamme et al. (2019)",
    "rsf": "Kvamme et al. (2019)",
    "deephit": "Kvamme & Borgan (2021)",
}
SKSURV_SOURCE = "scikit-survival recipe, 30-seed mean"

# Row order in the final table: scikit-survival (held-out test) first, then the
# JMLR datasets with SUPPORT last (its PH-violation models are the exception case).
JMLR_DATASET_ORDER = ["METABRIC", "FLCHAIN", "SUPPORT"]
JMLR_MODEL_ORDER = ["coxph", "deepsurv", "rsf", "deephit"]
SKSURV_DATASET_ORDER = ["GBSG2", "Veterans"]


def fmt_score(value):
    """Format a C-index to 3 decimals, or 'N/A' if missing."""
    return f"{value:.3f}" if pd.notna(value) else "N/A"


def fmt_gap(value):
    """Format a gap to 3 decimals with an explicit + or - sign."""
    return f"{value:+.3f}" if pd.notna(value) else "N/A"


def fmt_mean_std(mean, std):
    """Format 'mean ± std' to 3 decimals, dropping '± std' when std is missing.

    The sksurv suite reports a mean over seeds with its spread; FSSVM's target is
    a single replicated value with no std, so it shows as the bare mean.
    """
    if pd.isna(mean):
        return "N/A"
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def load_csv(path, required):
    """Read a results CSV; return None if an optional file is missing."""
    if not path.exists():
        if required:
            sys.exit(f"ERROR: required input not found: {path}\n"
                     "Run Vignette 03 first to generate it.")
        print(f"NOTE: optional input not found, skipping tuned add-in: {path}")
        return None
    return pd.read_csv(path)


def require_cols(df, cols, name):
    """Exit with a clear message if df is missing any expected column.

    Turns a downstream KeyError deep in the build loop into an actionable
    load-time error naming the file and the missing columns.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {name} is missing column(s) {missing}; "
                 f"found {list(df.columns)}. Re-run Vignette 03 to regenerate it.")


def index_by_key(df, name):
    """Index df by (dataset, model), erroring if that key is not unique.

    A duplicate key would make .loc return a DataFrame instead of a Series and
    silently corrupt the table, so we refuse rather than guess.
    """
    indexed = df.set_index(["dataset", "model"])
    if indexed.index.has_duplicates:
        dupes = sorted(set(indexed.index[indexed.index.duplicated()]))
        sys.exit(f"ERROR: {name} has duplicate (dataset, model) rows: {dupes}.")
    return indexed


def build_table(output_dir):
    """Assemble the combined long-format supplementary table."""
    jmlr = load_csv(output_dir / "vig3_jmlr.csv", required=True)
    sksurv = load_csv(output_dir / "vig3_sksurv.csv", required=True)
    tuned = load_csv(output_dir / "vig3_tuned.csv", required=False)

    require_cols(jmlr, ["dataset", "model", "ctd", "target", "gap"],
                 "vig3_jmlr.csv")
    require_cols(sksurv,
                 ["dataset", "model", "harrell_mean", "harrell_std",
                  "target", "target_std", "gap"],
                 "vig3_sksurv.csv")

    # (dataset, model) -> (tuned score, tuned gap) for the SUPPORT add-ins.
    # Held to the same column and uniqueness checks as the required files.
    tuned_lookup = {}
    if tuned is not None:
        require_cols(tuned, ["dataset", "model", "ctd_tuned", "gap_tuned"],
                     "vig3_tuned.csv")
        tuned_indexed = index_by_key(tuned, "vig3_tuned.csv")
        for (dataset, model), row in tuned_indexed.iterrows():
            tuned_lookup[(dataset, model)] = (row["ctd_tuned"], row["gap_tuned"])
    matched_tuned = set()

    rows = []

    # scikit-survival suite first: Harrell C, test set, mean over seeds. Both
    # mlsurv and the target are means ± std (the FSSVM target has no std). No
    # tuned add-in.
    sksurv_indexed = index_by_key(sksurv, "vig3_sksurv.csv")
    for dataset in SKSURV_DATASET_ORDER:
        sub = sksurv[sksurv["dataset"] == dataset]
        for model in sub["model"]:
            r = sksurv_indexed.loc[(dataset, model)]
            rows.append({
                "Dataset": dataset,
                "Model": MODEL_DISPLAY.get(model, model),
                "mlsurv (default / tuned)": fmt_mean_std(r["harrell_mean"],
                                                         r["harrell_std"]),
                "Target": fmt_mean_std(r["target"], r["target_std"]),
                "Gap (default / tuned)": fmt_gap(r["gap"]),
                "Target source": SKSURV_SOURCE,
            })

    # JMLR suite: C^td, 5-fold CV. SUPPORT comes last (see JMLR_DATASET_ORDER).
    jmlr_indexed = index_by_key(jmlr, "vig3_jmlr.csv")
    skipped = []
    for dataset in JMLR_DATASET_ORDER:
        for model in JMLR_MODEL_ORDER:
            if (dataset, model) not in jmlr_indexed.index:
                skipped.append((dataset, model))
                continue
            r = jmlr_indexed.loc[(dataset, model)]
            score = fmt_score(r["ctd"])
            gap = fmt_gap(r["gap"])
            tuned_vals = tuned_lookup.get((dataset, model))
            if tuned_vals is not None:
                # Fold the tuned result into the default row: "default / tuned".
                matched_tuned.add((dataset, model))
                score = f"{score} / {fmt_score(tuned_vals[0])}"
                gap = f"{gap} / {fmt_gap(tuned_vals[1])}"
            rows.append({
                "Dataset": dataset,
                "Model": MODEL_DISPLAY.get(model, model),
                "mlsurv (default / tuned)": score,
                "Target": fmt_score(r["target"]),
                "Gap (default / tuned)": gap,
                "Target source": JMLR_SOURCE.get(model, ""),
            })

    # Surface anything missing so a short or add-in-less table is never silent.
    if skipped:
        print(f"NOTE: {len(skipped)} expected JMLR (dataset, model) cell(s) "
              f"absent from the CSV, omitted from the table: {skipped}")
    unused_tuned = sorted(set(tuned_lookup) - matched_tuned)
    if unused_tuned:
        print(f"NOTE: {len(unused_tuned)} tuned row(s) matched no JMLR row "
              f"(check dataset/model naming), so no add-in was applied: "
              f"{unused_tuned}")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory holding the Vignette 03 result CSVs "
                             f"(default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write the combined table "
                             "(default: <output-dir>/vignette_03_supp_table.csv).")
    args = parser.parse_args()

    out_path = args.out or (args.output_dir / "vignette_03_supp_table.csv")

    table = build_table(args.output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    print(f"\nWrote {len(table)} rows to {out_path}\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
