"""Compare Vignette 07 models with LORIS on the metrics used in the LORIS paper.

The LORIS paper (Chang et al., Nature Cancer 2024) evaluates its score with
ROC AUC rather than the C-index:

- Response AUC (LORIS Fig 2a): responders vs non-responders, per cohort.
- Fixed-time OS AUC (LORIS Fig 5a): alive vs dead at 6, 12, 24 and 36 months,
  on Chowell test + MSK1 pooled (n=968).

This script first reproduces the published LORIS values, then scores the
Vignette 07 survival models the same way. Lower predicted risk counts as the
better outcome (responder / alive).

For fixed-time AUC the LORIS paper labels patients 0 if they died before t and
1 if followed to t or beyond, and drops patients censored before t, then takes
a plain ROC AUC (code/07_3 in github.com/rootchang/LORIS). We report that
"LORIS method" and, because it drops censored patients, the censoring-adjusted
cumulative/dynamic AUC (IPCW; the AUC(t) mlsurv reports) alongside it.

LORIS is computed with scripts/loris_score_calculator.py. Patients with unknown
systemic therapy history (4 in MSK2) are dropped, as in the LORIS paper.

Confidence intervals: 1,000 patient-level bootstrap resamples (percentile), as
in the LORIS paper. The same resamples are used for every score, so the p-value
against LORIS is paired (two-sided bootstrap test on the AUC difference).

Inputs (from a Vignette 07 run, OS and PFS):
    examples/output/vig7/results_OS.xlsx, results_PFS.xlsx
    examples/data/loris/<cohort>.csv   (Response labels and LORIS inputs)

Outputs (in --output-dir):
    vignette_07_loris_metrics.csv          every number, long format
    vignette_07_loris_metrics_table.csv    paste-ready table (panels A-C)

Usage:
    python scripts/figures/make_vignette_07_loris_metrics_table.py
    python scripts/figures/make_vignette_07_loris_metrics_table.py --n-boot 20   # quick check
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

# Repo root is two levels up from scripts/figures/.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples" / "output" / "vig7"
DATA_DIR = REPO_ROOT / "examples" / "data" / "loris"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from loris_score_calculator import calculate_loris  # noqa: E402

SEED = 42
MODELS = ["coxph", "coxnet", "rsf", "xgbcox", "xgbaft", "fssvm"]
MODEL_DISPLAY = {"coxph": "CoxPH", "coxnet": "CoxNet", "rsf": "RSF",
                 "xgbcox": "XGBoost-Cox", "xgbaft": "XGBoost-AFT", "fssvm": "FSSVM"}
SCORES = ["LORIS", "TMB"] + MODELS

# Response AUC cohorts, in LORIS Fig 2a order, with the published LORIS and
# TMB AUC (95% CI) and cohort size read from that figure.
RESPONSE_COHORTS = {
    "Chowell_test":     {"label": "Chowell test", "n": 515,
                         "LORIS": (0.75, 0.70, 0.80), "TMB": (0.65, 0.59, 0.71)},
    "MSK1":             {"label": "MSK1", "n": 453,
                         "LORIS": (0.70, 0.65, 0.75), "TMB": (0.54, 0.48, 0.60)},
    "MSK2":             {"label": "MSK2", "n": 100,
                         "LORIS": (0.64, 0.47, 0.80), "TMB": (0.52, 0.34, 0.69)},
    "Kato_panCancer":   {"label": "Kato", "n": 35,
                         "LORIS": (0.79, 0.61, 0.94), "TMB": (0.66, 0.35, 0.96)},
    "Pradat_panCancer": {"label": "Pradat", "n": 57,
                         "LORIS": (0.71, 0.56, 0.86), "TMB": (0.41, 0.26, 0.56)},
}

# Fixed-time OS AUC: LORIS Fig 5a, ICB cohort (Chowell test + MSK1, n=968).
FIXED_TIME_COHORTS = ["Chowell_test", "MSK1"]
TIMES = [6, 12, 24, 36]  # months
FIXED_TIME_PUBLISHED = {6: (0.73, 0.71, 0.76), 12: (0.74, 0.73, 0.76),
                        24: (0.81, 0.79, 0.85), 36: (0.83, 0.81, 0.87)}


# ── Loading ────────────────────────────────────────────────────────────────────

def predictions_sheet(xlsx, cohort):
    """Name of the per-patient predictions sheet for a cohort, or None.

    The internal test set is in 'benchmark_eval_predictions'; external cohorts
    are in 'val_<cohort>_predictions'. Excel caps sheet names at 31 characters,
    so external sheets are matched on a prefix.
    """
    prefix = ("benchmark_eval_predictions" if cohort == "Chowell_test"
              else f"val_{cohort}_pred")[:31]
    names = [s for s in pd.ExcelFile(xlsx).sheet_names if s.startswith(prefix)]
    if len(names) > 1:
        sys.exit(f"ERROR: {xlsx.name}/{cohort}: several predictions sheets {names}")
    return names[0] if names else None


def load_cohort(output_dir, outcome, cohort):
    """One row per patient: time, event, Response and a risk column per score.

    Risk columns are oriented so that higher = worse outcome. Returns None when
    the cohort has no data for this outcome (e.g. Pradat has no PFS).
    """
    xlsx = output_dir / f"results_{outcome}.xlsx"
    if not xlsx.exists():
        sys.exit(f"ERROR: {xlsx} not found. Run Vignette 07 for {outcome} first.")
    sheet = predictions_sheet(xlsx, cohort)
    if sheet is None:
        return None
    pred = pd.read_excel(xlsx, sheet_name=sheet)

    # Prediction rows are indexed by row position in the cohort CSV after
    # dropping rows with a missing outcome, as Vignette 07 does.
    df = pd.read_csv(DATA_DIR / f"{cohort}.csv")
    df = df.dropna(subset=[f"{outcome}_Months", f"{outcome}_Event"]).reset_index(drop=True)

    wide = pred.pivot(index="index", columns="model_type", values="risk_score")
    missing = [m for m in MODELS + ["TMB"] if m not in wide.columns]
    if missing:
        sys.exit(f"ERROR: {sheet} in {xlsx.name} has no scores for {missing}")
    base = pred.drop_duplicates("index").set_index("index")[["time", "event"]]
    wide = base.join(wide[MODELS + ["TMB"]])
    if not np.allclose(wide["time"], df.loc[wide.index, f"{outcome}_Months"]):
        sys.exit(f"ERROR: {outcome}/{cohort}: prediction rows do not match the CSV")

    wide["LORIS"] = -calculate_loris(df).to_numpy()[wide.index]  # higher LORIS = better
    wide["Response"] = df.loc[wide.index, "Response"].to_numpy()
    # The LORIS paper drops patients with unknown therapy history (4 in MSK2).
    known = df.loc[wide.index, "Systemic_therapy_history"].notna().to_numpy()
    return wide[known].reset_index(drop=True)


# ── Metrics ────────────────────────────────────────────────────────────────────

def _auc(y, score):
    """ROC AUC; raise if only one class is present (newer scikit-learn returns
    NaN with a warning instead, which would leak NaN into the bootstrap)."""
    if y.nunique() < 2:
        raise ValueError("only one class present")
    return roc_auc_score(y, score)


def response_auc(d, s, t=None):
    ok = d["Response"].notna()
    return _auc(d.loc[ok, "Response"].astype(int), -d.loc[ok, s])


def loris_method_auc(d, s, t):
    """Alive at t (1) vs died before t (0); censored before t dropped."""
    died = (d["time"] < t) & (d["event"] == 1)
    alive = d["time"] >= t
    keep = died | alive
    return _auc(alive[keep].astype(int), -d.loc[keep, s])


def ipcw_auc(d, s, t):
    y = Surv.from_arrays(d["event"].astype(bool), d["time"])
    return cumulative_dynamic_auc(y, y, d[s].to_numpy(float), [t])[0][0]


def evaluate(d, metric, t, n_boot, rng, panel, cohort):
    """Point AUC, bootstrap CI and paired p vs LORIS for every score."""
    point = {s: metric(d, s, t) for s in SCORES}
    boots = {s: [] for s in SCORES}
    for _ in range(n_boot):
        b = d.iloc[rng.integers(0, len(d), len(d))]
        try:
            vals = {s: metric(b, s, t) for s in SCORES}
        except ValueError:  # resample with one class, or t past follow-up
            continue
        for s in SCORES:
            boots[s].append(vals[s])
    loris = np.array(boots["LORIS"])
    rows = []
    for s in SCORES:
        b = np.array(boots[s])
        diff = b - loris
        p = np.nan if s == "LORIS" else min(1.0, 2 * min((diff <= 0).mean(),
                                                         (diff >= 0).mean()))
        rows.append(dict(panel=panel, cohort=cohort, months=t, score=s,
                         n=int(d["Response"].notna().sum()) if t is None else len(d),
                         auc=point[s], ci_lower=np.percentile(b, 2.5),
                         ci_upper=np.percentile(b, 97.5),
                         diff_vs_loris=point[s] - point["LORIS"], p_vs_loris=p,
                         n_boot=len(b)))
    return rows


# ── Build ──────────────────────────────────────────────────────────────────────

def compute(output_dir, n_boot):
    rows = []
    # A: response AUC, scored with the OS models and with the PFS models.
    for outcome in ["OS", "PFS"]:
        for cohort in RESPONSE_COHORTS:
            d = load_cohort(output_dir, outcome, cohort)
            if d is None:
                continue
            rng = np.random.default_rng(SEED)
            for r in evaluate(d, response_auc, None, n_boot, rng,
                              f"A_response_{outcome}", cohort):
                rows.append(r)

    # B, C: fixed-time OS AUC on the pooled cohort.
    pooled = pd.concat([load_cohort(output_dir, "OS", c) for c in FIXED_TIME_COHORTS],
                       ignore_index=True)
    for panel, metric in [("B_fixed_time_LORIS_method", loris_method_auc),
                          ("C_fixed_time_IPCW", ipcw_auc)]:
        for t in TIMES:
            rng = np.random.default_rng(SEED)
            rows += evaluate(pooled, metric, t, n_boot, rng, panel, "Chowell_test+MSK1")
    return pd.DataFrame(rows)


def fmt(auc, lo, hi):
    return f"{auc:.2f} ({lo:.2f}–{hi:.2f})"


def cell(r):
    """AUC (95% CI), with * when the paired bootstrap p vs LORIS is < 0.05."""
    star = "*" if pd.notna(r.p_vs_loris) and r.p_vs_loris < 0.05 else ""
    return fmt(r.auc, r.ci_lower, r.ci_upper) + star


def build_table(res):
    """Paste-ready table: one block per panel, a column per cohort or time."""
    blocks = []

    # Panel A: response AUC. LORIS and TMB once, then OS- and PFS-model rows.
    a = res[res.panel.str.startswith("A_")]
    cohorts = list(RESPONSE_COHORTS)
    labels = [f"{v['label']} (n={v['n']})" for v in RESPONSE_COHORTS.values()]
    rows = [{"Score": "A. Response AUC (responders vs non-responders)"}]
    rows.append({"Score": "LORIS (published)",
                 **{lab: fmt(*RESPONSE_COHORTS[c]["LORIS"]) for c, lab in zip(cohorts, labels)}})
    os_ = a[a.panel == "A_response_OS"]
    for s, name in [("LORIS", "LORIS (this study)"), ("TMB", "TMB")]:
        r_ = {"Score": name}
        for c, lab in zip(cohorts, labels):
            hit = os_[(os_.cohort == c) & (os_.score == s)]
            r_[lab] = cell(hit.iloc[0]) if len(hit) else "–"
        rows.append(r_)
    for outcome in ["OS", "PFS"]:
        rows.append({"Score": f"{outcome} models"})
        g = a[a.panel == f"A_response_{outcome}"]
        for m in MODELS:
            r_ = {"Score": MODEL_DISPLAY[m]}
            for c, lab in zip(cohorts, labels):
                hit = g[(g.cohort == c) & (g.score == m)]
                r_[lab] = cell(hit.iloc[0]) if len(hit) else "–"
            rows.append(r_)
    blocks.append(pd.DataFrame(rows))

    # Panels B and C: fixed-time OS AUC, one column per time point.
    tlabels = ["6 months", "1 year", "2 years", "3 years"]
    for panel, title, published in [
            ("B_fixed_time_LORIS_method",
             "B. Fixed-time OS AUC, LORIS method (Chowell test + MSK1, n=968)",
             FIXED_TIME_PUBLISHED),
            ("C_fixed_time_IPCW",
             "C. Fixed-time OS AUC, censoring-adjusted (IPCW)", {})]:
        g = res[res.panel == panel]
        rows = [{"Score": title}]
        if published:
            rows.append({"Score": "LORIS (published)",
                         **{lab: fmt(*published[t]) for t, lab in zip(TIMES, tlabels)}})
        for s in SCORES:
            r_ = {"Score": "LORIS (this study)" if s == "LORIS" else MODEL_DISPLAY.get(s, s)}
            for t, lab in zip(TIMES, tlabels):
                r_[lab] = cell(g[(g.months == t) & (g.score == s)].iloc[0])
            rows.append(r_)
        blocks.append(pd.DataFrame(rows))

    # Stack the panels. Their columns differ, so each panel gets its own header
    # row (cohorts or time points) right under its title row.
    out = []
    for b in blocks:
        cols = [c for c in b.columns if c != "Score"]
        b = b.rename(columns={c: f"col{i+1}" for i, c in enumerate(cols)})
        header = pd.DataFrame([{"Score": "", **{f"col{i+1}": c for i, c in enumerate(cols)}}])
        out += [b.iloc[:1], header, b.iloc[1:]]
    table = pd.concat(out, ignore_index=True).fillna("")
    return table[["Score"] + [c for c in table.columns if c.startswith("col")]]


def check_reproduction(res):
    """Print our LORIS / TMB values next to the published ones."""
    print("Reproduction of published LORIS values (point estimates):")
    a = res[res.panel == "A_response_OS"]
    for c, v in RESPONSE_COHORTS.items():
        for s in ["LORIS", "TMB"]:
            r = a[(a.cohort == c) & (a.score == s)].iloc[0]
            print(f"  response {v['label']:13s} {s:5s} n={r.n:<4d} ours {r.auc:.3f} "
                  f"published {v[s][0]:.2f}  diff {r.auc - v[s][0]:+.3f}")
    b = res[(res.panel == "B_fixed_time_LORIS_method") & (res.score == "LORIS")]
    for t in TIMES:
        r = b[b.months == t].iloc[0]
        print(f"  OS AUC {t:>2d} mo       LORIS n={r.n:<4d} ours {r.auc:.3f} "
              f"published {FIXED_TIME_PUBLISHED[t][0]:.2f}  "
              f"diff {r.auc - FIXED_TIME_PUBLISHED[t][0]:+.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Vignette 07 output folder (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap resamples (default 1000, as in the LORIS paper).")
    args = parser.parse_args()

    res = compute(args.output_dir, args.n_boot)
    check_reproduction(res)

    long_path = args.output_dir / "vignette_07_loris_metrics.csv"
    table_path = args.output_dir / "vignette_07_loris_metrics_table.csv"
    res.to_csv(long_path, index=False)
    table = build_table(res)
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    print(f"\nWrote {long_path}\nWrote {table_path}\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
