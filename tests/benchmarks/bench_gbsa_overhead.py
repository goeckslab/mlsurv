"""Benchmark: GBSA wrapper overhead and parallelization scaling.

Quantifies the overhead mlsurv's wrapper layers add vs calling scikit-survival
directly, and measures parallel tuning efficiency via Optuna n_jobs sweep.

Part A — Wrapper Overhead (4 levels):
  1. Raw sksurv fit        — GradientBoostingSurvivalAnalysis.fit() directly
  2. mlsurv wrapper fit    — GBSAModel.fit() (adds _validate_data + attribute bookkeeping)
  3. Pipeline fit           — Pipeline([Imputer, Scaler, GBSAModel]).fit()
  4. Tuning trial           — Single Optuna trial: param → Pipeline → 3-fold CV → score

Part B — Parallelization Scaling:
  Runs N tuning trials with varying n_jobs to measure parallel efficiency.
  GBSA has no internal n_jobs, so all parallelism comes from Optuna's
  study.optimize(n_jobs=N) running concurrent trials.

Datasets:
  - GBSG2 (686 patients, 8 features) — real clinical data
  - Synthetic (200 samples, 20 features) — matches test fixtures

Configs:
  - n_estimators=100 (fast baseline)
  - n_estimators=300 (matches auto_gbsa_defaults for GBSG2-size data)

Usage:
    python tests/benchmarks/bench_gbsa_overhead.py
"""

import csv
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from mlsurv.models.gbsa import GBSAModel

N_WARMUP = 1
N_REPEATS = 3
N_FOLDS = 3
N_TUNING_TRIALS = 8
RANDOM_STATE = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def time_fn(fn, n_warmup=N_WARMUP, n_repeats=N_REPEATS):
    """Run *fn* with warmup, return (median_seconds, all_times)."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), times


def load_gbsg2():
    """Load and encode GBSG2 dataset.

    Returns X (categoricals → float codes) and y with fields
    renamed to ('event', 'time') for mlsurv / sksurv compatibility.
    """
    from sksurv.datasets import load_gbsg2
    X, y = load_gbsg2()
    X = X.copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].cat.codes.astype(float)
    y_out = np.array(
        [(e, t) for e, t in y],
        dtype=[("event", bool), ("time", float)],
    )
    return X, y_out


def make_synthetic(n_samples=200, n_features=20, seed=RANDOM_STATE):
    """Generate synthetic survival data matching the test fixture."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    # Inject NaN at same positions as conftest.synthetic_data
    X.iloc[10:15, 0] = np.nan
    X.iloc[20:25, 5] = np.nan

    true_coef = np.zeros(n_features)
    true_coef[:5] = [0.5, -0.3, 0.4, -0.2, 0.3]
    linear_pred = X.fillna(0).values @ true_coef
    times = np.exp(-linear_pred + rng.randn(n_samples) * 0.5)
    times = np.clip(times, 0.1, 10)
    events = rng.binomial(1, 0.7, n_samples).astype(bool)
    y = np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


# ---------------------------------------------------------------------------
# Part A: Wrapper Overhead Benchmarks
# ---------------------------------------------------------------------------

def bench_raw_sksurv(X, y, **params):
    """Level 1: raw sksurv GradientBoostingSurvivalAnalysis.fit()."""
    X_clean = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=X.columns,
    )

    def run():
        model = GradientBoostingSurvivalAnalysis(
            random_state=RANDOM_STATE, **params
        )
        model.fit(X_clean, y)

    return time_fn(run)


def bench_mlsurv_wrapper(X, y, **params):
    """Level 2: mlsurv GBSAModel.fit()."""
    X_clean = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=X.columns,
    )

    def run():
        model = GBSAModel(random_state=RANDOM_STATE, **params)
        model.fit(X_clean, y)

    return time_fn(run)


def bench_pipeline(X, y, **params):
    """Level 3: Pipeline([Imputer, Scaler, GBSAModel]).fit()."""

    def run():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GBSAModel(random_state=RANDOM_STATE, **params)),
        ])
        pipe.fit(X, y)

    return time_fn(run)


def bench_tuning_trial(X, y, **params):
    """Level 4: single Optuna-style tuning trial (Pipeline + 3-fold CV + score).

    Simplified approximation of _evaluate_model(): for each fold, fit Pipeline
    and score with Harrell's C-index, then average.  Omits trial pruning,
    warning capture, and _validate_fold_scores hooks.
    """
    events = y["event"]
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def run():
        scores = []
        for train_idx, val_idx in cv.split(X, events):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", GBSAModel(random_state=RANDOM_STATE, **params)),
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_val)
            c_index = concordance_index_censored(
                y_val["event"], y_val["time"], preds
            )[0]
            scores.append(c_index)
        return np.mean(scores)

    return time_fn(run)


# ---------------------------------------------------------------------------
# Part B: Parallelization Scaling
# ---------------------------------------------------------------------------

def bench_parallel_tuning(X, y, n_trials, n_jobs_list, **params):
    """Run *n_trials* Optuna trials at each n_jobs level.

    Returns list of dicts with n_jobs, n_trials, wall_time, cpu_time,
    speedup, per_trial_time.

    Note: cpu_time only measures main-process CPU via time.process_time();
    it underestimates total CPU at n_jobs > 1 when work runs in threads
    that release the GIL (e.g. sksurv Cython code).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    events = y["event"]
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X, events))

    def objective(trial):
        """Single trial: fixed params (no hyperparameter search), 3-fold CV, C-index."""
        scores = []
        for train_idx, val_idx in folds:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", GBSAModel(random_state=RANDOM_STATE, **params)),
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_val)
            c_index = concordance_index_censored(
                y_val["event"], y_val["time"], preds
            )[0]
            scores.append(c_index)
        return np.mean(scores)

    results = []
    baseline_wall = None

    for nj in n_jobs_list:
        study = optuna.create_study(direction="maximize")

        t_wall_start = time.perf_counter()
        t_cpu_start = time.process_time()

        study.optimize(objective, n_trials=n_trials, n_jobs=nj)

        wall_time = time.perf_counter() - t_wall_start
        cpu_time = time.process_time() - t_cpu_start

        if baseline_wall is None:
            baseline_wall = wall_time

        speedup = baseline_wall / wall_time if wall_time > 0 else float("nan")
        per_trial = wall_time / n_trials

        results.append({
            "n_jobs": nj,
            "n_trials": n_trials,
            "wall_time": wall_time,
            "cpu_time": cpu_time,
            "speedup": speedup,
            "per_trial_time": per_trial,
        })

    return results


# ---------------------------------------------------------------------------
# Suite Runners
# ---------------------------------------------------------------------------

LEVEL_FUNCS = [
    ("1. Raw sksurv fit", bench_raw_sksurv),
    ("2. mlsurv wrapper fit", bench_mlsurv_wrapper),
    ("3. Pipeline fit", bench_pipeline),
    ("4. Tuning trial (3-fold CV)", bench_tuning_trial),
]


def run_overhead_suite(dataset_name, X, y, params_list):
    """Run Part A for all param configs, return list of result dicts."""
    rows = []
    for params in params_list:
        n_est = params.get("n_estimators", "?")
        label = f"{dataset_name} (n_est={n_est})"
        print(f"\n--- {label} ---")

        baseline_time = None
        for level_name, fn in LEVEL_FUNCS:
            print(f"  {level_name}...", end="", flush=True)
            median_s, all_times = fn(X, y, **params)
            print(f" {median_s:.3f}s")

            if baseline_time is None:
                baseline_time = median_s

            overhead_pct = (
                (median_s - baseline_time) / baseline_time * 100
                if baseline_time > 0
                else float("nan")
            )
            row = {
                "part": "A",
                "dataset": dataset_name,
                "n_estimators": n_est,
                "level": level_name,
                "median_s": round(median_s, 4),
                "overhead_pct": round(overhead_pct, 2),
                "all_times": ";".join(f"{t:.4f}" for t in all_times),
            }
            # Approximate per-fold time (total / N_FOLDS) for rough
            # comparison to single-fit levels; includes scoring overhead.
            if "CV" in level_name:
                row["per_fold_s"] = round(median_s / N_FOLDS, 4)
            rows.append(row)

    return rows


def run_parallel_suite(dataset_name, X, y, params):
    """Run Part B parallel scaling, return list of result dicts."""
    cpu_count = os.cpu_count() or 4
    n_jobs_list = sorted({1, 2, min(4, cpu_count), cpu_count})

    n_est = params.get("n_estimators", "?")
    print(f"\n--- Parallel Scaling: {dataset_name} (n_est={n_est}) ---")
    print(f"  Trials: {N_TUNING_TRIALS}, n_jobs sweep: {n_jobs_list}")

    results = bench_parallel_tuning(X, y, N_TUNING_TRIALS, n_jobs_list, **params)

    rows = []
    for r in results:
        rows.append({
            "part": "B",
            "dataset": dataset_name,
            "n_estimators": n_est,
            "n_jobs": r["n_jobs"],
            "n_trials": r["n_trials"],
            "wall_time": round(r["wall_time"], 3),
            "cpu_time": round(r["cpu_time"], 3),
            "speedup": round(r["speedup"], 2),
            "per_trial_time": round(r["per_trial_time"], 3),
        })

    return rows


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------

def print_overhead_table(rows):
    """Print Part A results as a formatted table."""
    print("\n" + "=" * 80)
    print("Part A: Wrapper Overhead")
    print("=" * 80)
    header = (
        f"{'Dataset':<15} {'n_est':>5} {'Level':<30} "
        f"{'Median (s)':>10} {'vs baseline':>11} {'Per-fold':>10}"
    )
    print(header)
    print("-" * 80)

    current_group = None
    for r in rows:
        group = (r["dataset"], r["n_estimators"])
        if group != current_group:
            if current_group is not None:
                print("-" * 80)
            current_group = group

        overhead_str = (
            "baseline" if r["overhead_pct"] == 0 else f"{r['overhead_pct']:+.1f}%"
        )
        per_fold_str = (
            f"{r['per_fold_s']:.3f}s" if "per_fold_s" in r else ""
        )
        print(
            f"{r['dataset']:<15} {r['n_estimators']:>5} "
            f"{r['level']:<30} {r['median_s']:>10.3f} {overhead_str:>11} "
            f"{per_fold_str:>10}"
        )


def print_parallel_table(rows):
    """Print Part B results as a formatted table."""
    print("\n" + "=" * 80)
    print("Part B: Parallelization Scaling")
    print("=" * 80)
    header = (
        f"{'Dataset':<15} {'n_est':>5} {'n_jobs':>6} {'Trials':>6} "
        f"{'Wall (s)':>8} {'CPU (s)':>8} {'Speedup':>8} {'Per-trial':>10}"
    )
    print(header)
    print("-" * 80)
    for r in rows:
        print(
            f"{r['dataset']:<15} {r['n_estimators']:>5} {r['n_jobs']:>6} "
            f"{r['n_trials']:>6} {r['wall_time']:>8.2f} {r['cpu_time']:>8.2f} "
            f"{r['speedup']:>7.2f}x {r['per_trial_time']:>9.3f}s"
        )


def write_csv(overhead_rows, parallel_rows, base_path):
    """Write Part A and Part B results to separate CSV files."""
    base_path.parent.mkdir(parents=True, exist_ok=True)

    a_path = base_path.with_name("bench_gbsa_overhead_partA.csv")
    b_path = base_path.with_name("bench_gbsa_overhead_partB.csv")

    with open(a_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "n_estimators", "level",
            "median_s", "overhead_pct", "all_times",
        ])
        for r in overhead_rows:
            writer.writerow([
                r["dataset"], r["n_estimators"], r["level"],
                r["median_s"], r["overhead_pct"], r["all_times"],
            ])

    with open(b_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "n_estimators", "n_jobs", "n_trials",
            "wall_time", "cpu_time", "speedup", "per_trial_time",
        ])
        for r in parallel_rows:
            writer.writerow([
                r["dataset"], r["n_estimators"], r["n_jobs"],
                r["n_trials"], r["wall_time"], r["cpu_time"],
                r["speedup"], r["per_trial_time"],
            ])

    print(f"\nResults written to {a_path} and {b_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*trapz.*")

    cpu_count = os.cpu_count() or 4
    print("=" * 75)
    print("GBSA Wrapper Overhead & Parallelization Benchmark")
    print("=" * 75)
    print(f"CPUs: {cpu_count} | Warmup: {N_WARMUP} | Repeats: {N_REPEATS}")
    print(f"CV folds: {N_FOLDS} | Tuning trials (Part B): {N_TUNING_TRIALS}")

    # -- Load datasets --
    print("\nLoading datasets...")
    X_gbsg2, y_gbsg2 = load_gbsg2()
    X_synth, y_synth = make_synthetic()
    print(f"  GBSG2:     {X_gbsg2.shape[0]} samples, {X_gbsg2.shape[1]} features")
    print(f"  Synthetic: {X_synth.shape[0]} samples, {X_synth.shape[1]} features")

    # -- Param configs --
    configs = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.1},
    ]

    # ===================== Part A =====================
    print("\n" + "#" * 75)
    print("# Part A: Wrapper Overhead")
    print("#" * 75)

    overhead_rows = []
    overhead_rows.extend(run_overhead_suite("GBSG2", X_gbsg2, y_gbsg2, configs))
    overhead_rows.extend(run_overhead_suite("Synthetic", X_synth, y_synth, configs))

    print_overhead_table(overhead_rows)

    # ===================== Part B =====================
    print("\n" + "#" * 75)
    print("# Part B: Parallelization Scaling")
    print("#" * 75)

    # Use n_estimators=100 for Part B to keep total runtime reasonable
    parallel_params = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}
    parallel_rows = []
    parallel_rows.extend(
        run_parallel_suite("GBSG2", X_gbsg2, y_gbsg2, parallel_params)
    )
    parallel_rows.extend(
        run_parallel_suite("Synthetic", X_synth, y_synth, parallel_params)
    )

    print_parallel_table(parallel_rows)

    # ===================== Summary =====================
    print("\n" + "=" * 75)
    print("Summary")
    print("=" * 75)

    # Wrapper overhead = Levels 2-3 vs Level 1 (same operation, just wrapper cost)
    # CV overhead = Level 4 per-fold vs Level 3 (adds scoring + data splitting)
    print("\nWrapper overhead (Levels 2-3 vs raw sksurv):")
    for cfg in configs:
        n_est = cfg["n_estimators"]
        wrapper_rows = [
            r for r in overhead_rows
            if r["n_estimators"] == n_est and "CV" not in r["level"]
            and r["overhead_pct"] != 0  # skip baseline itself
        ]
        if wrapper_rows:
            max_overhead = max(abs(r["overhead_pct"]) for r in wrapper_rows)
            # On fast configs (<0.2s), percentage is noisy — check absolute diff
            max_abs_diff = max(
                abs(r["median_s"] - [
                    b["median_s"] for b in overhead_rows
                    if b["n_estimators"] == n_est and b["dataset"] == r["dataset"]
                    and b["overhead_pct"] == 0
                ][0])
                for r in wrapper_rows
            )
            note = ""
            if max_abs_diff < 0.05:
                note = " (within timing noise)"
            target_met = max_overhead < 10
            status = "PASS" if target_met else "FAIL"
            print(
                f"  n_estimators={n_est}: max |overhead| = {max_overhead:.1f}%, "
                f"max |diff| = {max_abs_diff:.3f}s  "
                f"[{status} — target <10%]{note}"
            )

    print("\nCV per-fold vs full-data pipeline (Level 4/{} vs Level 3):".format(N_FOLDS))
    print("  (Per-fold < pipeline is expected: each fold trains on ~{:.0f}% of data)".format(
        100 * (N_FOLDS - 1) / N_FOLDS
    ))
    for cfg in configs:
        n_est = cfg["n_estimators"]
        for ds in ["GBSG2", "Synthetic"]:
            cv_row = [
                r for r in overhead_rows
                if r["n_estimators"] == n_est and r["dataset"] == ds
                and "CV" in r["level"]
            ]
            pipe_row = [
                r for r in overhead_rows
                if r["n_estimators"] == n_est and r["dataset"] == ds
                and "Pipeline" in r["level"]
            ]
            if cv_row and pipe_row:
                per_fold = cv_row[0].get("per_fold_s", cv_row[0]["median_s"] / N_FOLDS)
                pipe_time = pipe_row[0]["median_s"]
                print(
                    f"  {ds} n_est={n_est}: per-fold={per_fold:.3f}s, "
                    f"full-data={pipe_time:.3f}s"
                )

    # Parallel efficiency
    for r in parallel_rows:
        if r["n_jobs"] == cpu_count:
            eff = r["speedup"] / r["n_jobs"] * 100
            print(
                f"  {r['dataset']} parallel efficiency at n_jobs={cpu_count}: "
                f"{r['speedup']:.2f}x speedup ({eff:.0f}% efficient)"
            )

    # -- Write CSVs --
    write_csv(overhead_rows, parallel_rows, RESULTS_DIR / "bench_gbsa_overhead.csv")


if __name__ == "__main__":
    main()
