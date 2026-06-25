"""Benchmark DeepHit against Kvamme & Borgan (2021) Table 2 targets.

Reference: Kvamme & Borgan (2021), PMC8536596
Datasets: SUPPORT, METABRIC, FLCHAIN (same as vignette 03 JMLR section)
Protocol: 5-fold CV, matching the paper's evaluation setup.

Usage:
    conda run -n mlsurv python tests/benchmarks/bench_deephit_kvamme.py
    conda run -n mlsurv python tests/benchmarks/bench_deephit_kvamme.py --n-trials 20
    conda run -n mlsurv python tests/benchmarks/bench_deephit_kvamme.py --dataset SUPPORT
"""

import argparse
import time
import warnings

import numpy as np

from mlsurv import SurvivalLearner
from mlsurv.utils import to_structured_array

# Kvamme & Borgan (2021) Table 2 — DeepHit, 5-fold CV
DEEPHIT_TARGETS = {
    "SUPPORT": 0.639,
    "METABRIC": 0.675,
    "FLCHAIN": 0.791,
}

# Also include original JMLR (2019) targets for context
JMLR_TARGETS = {
    "SUPPORT": {"coxph": 0.598, "deepsurv": 0.611, "rsf": 0.628},
    "METABRIC": {"coxph": 0.628, "deepsurv": 0.636, "rsf": 0.649},
    "FLCHAIN": {"coxph": 0.790, "deepsurv": 0.790, "rsf": 0.784},
}


def load_pycox_dataset(name):
    """Load a pycox dataset and convert to mlsurv format."""
    from pycox.datasets import flchain, metabric, support

    loaders = {"SUPPORT": support, "METABRIC": metabric, "FLCHAIN": flchain}
    loader = loaders[name]

    # FLCHAIN requires special handling due to pycox bug
    if name == "FLCHAIN":
        df = loader.read_df(processed=False)
        df = (
            df.drop(["chapter", "rownames"], axis=1, errors="ignore")
            .loc[lambda x: x["creatinine"].notna()]
            .reset_index(drop=True)
            .assign(sex=lambda x: (x["sex"] == "M").astype(float))
            .rename(columns={"futime": "duration", "death": "event"})
        )
        for col in df.columns:
            if col not in ["duration", "event"]:
                df[col] = df[col].astype("float32")
    else:
        # SUPPORT's x1-x6 are nominal categoricals that Kvamme one-hot encodes
        # only for the classical Cox baseline. This script benchmarks the
        # embedding-free neural nets, which are tuned on the raw integer codes,
        # so we keep them as-is (one-hot is reserved for the Cox path; see
        # examples/vignette_03_benchmarks.ipynb).
        df = loader.read_df()

    df = df[df["duration"] > 0]
    X = df.drop(columns=["duration", "event"])
    y = to_structured_array(df["duration"].values, df["event"].values)
    return X, y


def run_benchmark(datasets, n_trials, random_state, verbose):
    """Run DeepHit benchmark on specified datasets."""
    results = []

    for ds_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"  {ds_name}")
        print(f"{'=' * 60}")

        X, y = load_pycox_dataset(ds_name)
        n = len(X)
        n_events = int(y["event"].sum())
        print(f"  n={n}, features={X.shape[1]}, events={n_events}")

        learner = SurvivalLearner(X, y)
        learner.setup(
            test_fraction=0.2,
            cv_folds=5,
            random_state=random_state,
            n_jobs=-1,
        )

        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            comparison = learner.run(
                models=["deephit"],
                verbose=verbose,
                n_trials=n_trials,
                n_importance_repeats=10,
                n_bootstrap=10,
            )
        elapsed = time.time() - t0

        result = comparison["deephit"]
        cv = result.cv_metric_mean
        ctd_cv = result.get_metric("ctd", phase="cv", stat="mean")
        target = DEEPHIT_TARGETS[ds_name]
        # Kvamme targets are C^td, so compare C^td vs target
        ctd_diff = (ctd_cv - target) if ctd_cv is not None else float("nan")

        results.append(
            {
                "dataset": ds_name,
                "n": n,
                "harrell": cv,
                "ctd": ctd_cv,
                "target": target,
                "diff": ctd_diff,
                "time_s": elapsed,
            }
        )

        ctd_str = f"{ctd_cv:.3f}" if ctd_cv is not None else "N/A"
        print(f"  Harrell: {cv:.3f}  C^td: {ctd_str}  (target: {target:.3f}, diff: {ctd_diff:+.3f})")
        print(f"  Time: {elapsed:.1f}s")

    return results


def print_summary(results):
    """Print comparison table."""
    print(f"\n{'=' * 60}")
    print("  DeepHit Benchmark — Kvamme & Borgan (2021) Table 2")
    print(f"{'=' * 60}")
    print(f"  {'Dataset':<10} {'n':>6}  {'Harrell':>8}  {'C^td':>8}  {'target':>7}  {'diff':>7}  {'time':>6}")
    print(f"  {'-' * 62}")

    for r in results:
        ctd_str = f"{r['ctd']:>8.3f}" if r['ctd'] is not None else "     N/A"
        print(
            f"  {r['dataset']:<10} {r['n']:>6}  {r['harrell']:>8.3f}  "
            f"{ctd_str}  {r['target']:>7.3f}  {r['diff']:>+7.3f}  {r['time_s']:>5.1f}s"
        )

    diffs = [abs(r["diff"]) for r in results if not np.isnan(r["diff"])]
    print(f"  {'-' * 62}")
    print(f"  Mean |diff|: {np.mean(diffs):.3f}  (C^td vs Kvamme target)")
    print(f"  Max  |diff|: {np.max(diffs):.3f}")

    # Context: JMLR targets for other models
    print(f"\n  Reference context (Kvamme 2019 JMLR):")
    print(f"  {'Dataset':<10} {'CoxPH':>7} {'DeepSurv':>9} {'RSF':>7} {'DeepHit':>8}")
    print(f"  {'-' * 45}")
    for ds_name in ["SUPPORT", "METABRIC", "FLCHAIN"]:
        jmlr = JMLR_TARGETS[ds_name]
        dh = DEEPHIT_TARGETS[ds_name]
        print(
            f"  {ds_name:<10} {jmlr['coxph']:>7.3f} {jmlr['deepsurv']:>9.3f} "
            f"{jmlr['rsf']:>7.3f} {dh:>8.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepHit against Kvamme & Borgan (2021) targets"
    )
    parser.add_argument(
        "--dataset",
        choices=["SUPPORT", "METABRIC", "FLCHAIN"],
        help="Run a single dataset (default: all three)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="Optuna tuning trials (default: 3, paper used full search)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show learner.run() output",
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DEEPHIT_TARGETS.keys())

    print("DeepHit Benchmark (Kvamme & Borgan 2021)")
    print(f"  Datasets: {datasets}")
    print(f"  Tuning trials: {args.n_trials}")
    print(f"  Random state: {args.random_state}")

    results = run_benchmark(datasets, args.n_trials, args.random_state, args.verbose)
    print_summary(results)


if __name__ == "__main__":
    main()
