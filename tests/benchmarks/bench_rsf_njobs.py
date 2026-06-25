"""Benchmark: mlsurv RSF importance vs raw sksurv RSF importance.

Compares importance timing across:
  1. mlsurv RSF (fixed)  — SurvivalLearner.setup(n_jobs=N) -> train -> evaluate
  2. Raw sksurv RSF      — RandomSurvivalForest(n_jobs=N) -> permutation_importance
  3. mlsurv RSF (broken) — get_model('rsf', n_jobs=1) -> evaluate(n_jobs=1)

where N = min(4, cpu_count).

Note: Overhead comparison is approximate — mlsurv evaluate() computes
concordance, IPCW, AUC(t), Brier, IBS beyond just permutation importance,
so some overhead vs raw sksurv is expected and unrelated to the fix.

Uses GBSG2 dataset (686 patients, 8 features).

Usage:
    python tests/benchmarks/bench_rsf_njobs.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

N_IMPORTANCE_REPEATS = 30
N_JOBS = min(4, os.cpu_count() or 1)
N_WARMUP = 1
N_REPEATS = 3


def time_fn(fn, n_warmup=N_WARMUP, n_repeats=N_REPEATS):
    """Run fn with warmup, return wall-time median and all times."""
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

    Returns X with categorical columns encoded as floats,
    and y with fields renamed to ('event', 'time') for mlsurv compatibility.
    """
    from sksurv.datasets import load_gbsg2
    X, y = load_gbsg2()
    X = X.copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].cat.codes.astype(float)
    # sksurv GBSG2 uses ('cens', 'time'); mlsurv expects ('event', 'time')
    y_mlsurv = np.array(
        [(e, t) for e, t in y],
        dtype=[('event', bool), ('time', float)],
    )
    return X, y_mlsurv


def bench_mlsurv_fixed(X_train, X_test, y_train, y_test):
    """mlsurv RSF with n_jobs propagation (the fix)."""
    from mlsurv import SurvivalLearner

    learner = SurvivalLearner(X_train, y_train)
    learner.setup(cv_folds=3, random_state=42, n_jobs=N_JOBS)
    result = learner.train('rsf')

    # Verify the fix: inner model should have n_jobs == N_JOBS
    inner = result.model.steps[-1][1]
    assert inner.n_jobs == N_JOBS, (
        f"Fix verification failed: expected n_jobs={N_JOBS}, got {inner.n_jobs}"
    )

    from mlsurv.evaluation import evaluate

    def run():
        evaluate(
            result.model, X_test, y_test,
            X_train=X_train, y_train=y_train,
            n_importance_repeats=N_IMPORTANCE_REPEATS,
            random_state=42,
            show_progress=False,
            n_jobs=N_JOBS,
        )

    return time_fn(run)


def bench_raw_sksurv(X_train, X_test, y_train, y_test):
    """Raw sksurv RSF with sklearn permutation_importance."""
    from sksurv.ensemble import RandomSurvivalForest
    from sklearn.inspection import permutation_importance

    rsf = RandomSurvivalForest(
        n_estimators=100, random_state=42, n_jobs=N_JOBS,
    )
    rsf.fit(X_train, y_train)

    def run():
        permutation_importance(
            rsf, X_test, y_test,
            n_repeats=N_IMPORTANCE_REPEATS,
            random_state=42,
            n_jobs=N_JOBS,
        )

    return time_fn(run)


def bench_mlsurv_broken(X_train, X_test, y_train, y_test):
    """mlsurv RSF simulating the old bug (n_jobs=1 everywhere)."""
    from mlsurv.models import get_model
    from mlsurv.evaluation import evaluate

    model = get_model('rsf', n_jobs=1, random_state=42)
    model.fit(X_train, y_train)

    # Verify broken baseline: model must be single-threaded
    assert model.n_jobs == 1, (
        f"Broken baseline verification failed: expected n_jobs=1, got {model.n_jobs}"
    )

    def run():
        evaluate(
            model, X_test, y_test,
            X_train=X_train, y_train=y_train,
            n_importance_repeats=N_IMPORTANCE_REPEATS,
            random_state=42,
            show_progress=False,
            n_jobs=1,
        )

    return time_fn(run)


def main():
    print("=== RSF n_jobs Propagation Benchmark ===")
    print(f"GBSG2 dataset | n_importance_repeats={N_IMPORTANCE_REPEATS} | n_jobs={N_JOBS}")
    print(f"Warmup: {N_WARMUP} | Repeats: {N_REPEATS}")
    print()

    X, y = load_gbsg2()
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {X.shape[1]}")
    print()

    # --- Broken baseline (single-threaded) ---
    print("1. mlsurv RSF (broken — n_jobs=1)...", end="", flush=True)
    broken_median, broken_times = bench_mlsurv_broken(X_train, X_test, y_train, y_test)
    print(f" {broken_median:.2f}s")

    # --- Raw sksurv RSF ---
    print(f"2. Raw sksurv RSF (n_jobs={N_JOBS})...", end="", flush=True)
    raw_median, raw_times = bench_raw_sksurv(X_train, X_test, y_train, y_test)
    print(f" {raw_median:.2f}s")

    # --- Fixed mlsurv RSF ---
    print(f"3. mlsurv RSF (fixed — n_jobs={N_JOBS})...", end="", flush=True)
    fixed_median, fixed_times = bench_mlsurv_fixed(X_train, X_test, y_train, y_test)
    print(f" {fixed_median:.2f}s")

    # --- Results ---
    print()
    print("=== Results ===")
    print(f"{'Scenario':<35} {'Median (s)':>10} {'Speedup':>10} {'Overhead':>10}")
    print("-" * 67)

    speedup_fixed = broken_median / fixed_median if fixed_median > 0 else float("nan")
    speedup_raw = broken_median / raw_median if raw_median > 0 else float("nan")
    overhead = (fixed_median - raw_median) / raw_median * 100 if raw_median > 0 else float("nan")

    print(f"{'mlsurv RSF (broken, n_jobs=1)':<35} {broken_median:>10.2f} {'1.00x':>10} {'baseline':>10}")
    print(f"{'Raw sksurv RSF (n_jobs=' + str(N_JOBS) + ')':<35} {raw_median:>10.2f} {speedup_raw:>9.2f}x {'-':>10}")
    print(f"{'mlsurv RSF (fixed, n_jobs=' + str(N_JOBS) + ')':<35} {fixed_median:>10.2f} {speedup_fixed:>9.2f}x {overhead:>9.1f}%")

    print()
    target_met = overhead < 10
    print(f"Overhead target (<10%): {'PASS' if target_met else 'FAIL'} ({overhead:.1f}%)")
    print(f"  Note: Overhead includes mlsurv evaluate() extras (IPCW, AUC(t), Brier)")
    print(f"  vs raw sklearn permutation_importance (concordance only).")


if __name__ == "__main__":
    main()
