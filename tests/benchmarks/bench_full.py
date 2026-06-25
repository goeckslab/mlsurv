"""Chunked full-runtime benchmark across all 9 models.

Runs each pipeline step as an independent chunk, appending results to a shared
CSV so that partial progress is preserved if a chunk fails or is killed.

Usage:
    python bench_full.py --step cv         # CV training chunk
    python bench_full.py --step evaluate   # Evaluate chunk
    python bench_full.py --step bootstrap  # Bootstrap chunk
    python bench_full.py --step tuning     # Tuning chunk
    python bench_full.py --step shap       # SHAP chunk
    python bench_full.py --step all        # All chunks sequentially
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse utilities from bench_parallel
from bench_parallel import (
    RESULTS_DIR,
    make_synthetic,
    compute_speedup_table,
)

# Add project root to path (bench_parallel does this too, but be explicit)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_MODELS = [
    "coxph", "coxnet", "rsf", "gbsa", "fssvm",
    "xgbcox", "weibull_aft", "deepsurv", "deephit",
]

DATASETS = [
    ("synth_200_20", 200, 20),
    ("synth_1000_30", 1000, 30),
    ("synth_3000_50", 3000, 50),
]

N_JOBS_LIST = [1, 4]
N_WARMUP = 1
N_REPEATS = 3

DEFAULT_OUTPUT = RESULTS_DIR / "bench_full.csv"

# Unified CSV columns (superset of all steps)
ALL_COLUMNS = [
    "step", "dataset", "model", "n_jobs", "wall_median", "cpu_median",
    "n_importance_repeats", "n_bootstrap", "n_trials", "n_samples",
    "speedup", "efficiency",
]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def append_rows_to_csv(rows, output_path):
    """Append benchmark rows to CSV, computing speedup first."""
    if not rows:
        print("  No rows to save.")
        return

    new_df = pd.DataFrame(rows)
    # compute_speedup_table requires n_jobs column; skip for steps without it
    if "n_jobs" in new_df.columns:
        new_df = compute_speedup_table(rows)
    else:
        new_df["speedup"] = np.nan
        new_df["efficiency"] = np.nan

    # Ensure all columns exist
    for col in ALL_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df = new_df[ALL_COLUMNS]

    # Append to existing CSV or create new
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(output_path, index=False)
    print(f"  Saved {len(new_df)} rows -> {output_path} (total: {len(combined)})")


def print_chunk_summary(rows, step_name):
    """Print a concise summary of the chunk results."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    print(f"\n--- {step_name} Summary ---")
    summary_cols = [c for c in ["dataset", "model", "n_jobs", "wall_median"] if c in df.columns]
    print(df[summary_cols].to_string(index=False))
    total_wall = df["wall_median"].sum()
    print(f"  Total wall time (sum of medians): {total_wall:.1f}s")


# ---------------------------------------------------------------------------
# Dataset and model helpers
# ---------------------------------------------------------------------------

def generate_datasets():
    """Generate all synthetic datasets."""
    print("=== Generating datasets ===")
    datasets = {}
    for name, n, f in DATASETS:
        print(f"  {name}: n={n}, features={f}")
        datasets[name] = make_synthetic(n, f)
    return datasets


def split_datasets(datasets):
    """70/30 train/test split for each dataset."""
    splits = {}
    for name, (X, y) in datasets.items():
        idx = int(len(X) * 0.7)
        splits[name] = (X.iloc[:idx], X.iloc[idx:], y[:idx], y[idx:])
    return splits


def train_models(splits, model_names=None):
    """Train fitted models on each dataset's training split."""
    from mlsurv.models import get_model

    model_names = model_names or ALL_MODELS
    fitted = {}
    for ds_name, (X_train, X_test, y_train, y_test) in splits.items():
        fitted[ds_name] = {}
        for model_name in model_names:
            print(f"  Training {model_name} on {ds_name}...", end="", flush=True)
            try:
                model = get_model(model_name)
                model.fit(X_train, y_train)
                fitted[ds_name][model_name] = model
                print(" done.")
            except Exception as e:
                print(f" FAILED: {e}")
    return fitted


# ---------------------------------------------------------------------------
# Chunk runners
# ---------------------------------------------------------------------------

def run_cv(output_path):
    """Chunk 1: CV training — 9 models x 3 datasets x 2 n_jobs = 54 configs."""
    from bench_parallel import bench_cv_training

    print("\n" + "=" * 60)
    print("CHUNK 1: CV Training")
    print("=" * 60)

    datasets = generate_datasets()
    t0 = time.perf_counter()
    rows = bench_cv_training(
        ALL_MODELS, datasets, N_JOBS_LIST,
        n_warmup=N_WARMUP, n_repeats=N_REPEATS,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nCV chunk completed in {elapsed:.1f}s")
    print_chunk_summary(rows, "CV Training")
    append_rows_to_csv(rows, output_path)
    return rows


def run_evaluate(output_path):
    """Chunk 2: Evaluate — 9 models x 3 datasets x 2 n_jobs x 1 repeat = 54 configs."""
    from bench_parallel import bench_evaluate

    print("\n" + "=" * 60)
    print("CHUNK 2: Evaluate")
    print("=" * 60)

    datasets = generate_datasets()
    splits = split_datasets(datasets)
    print("\n--- Training models for evaluate ---")
    fitted = train_models(splits)

    t0 = time.perf_counter()
    rows = bench_evaluate(
        fitted, splits, N_JOBS_LIST,
        n_importance_repeats_list=[5],
        n_warmup=N_WARMUP, n_repeats=N_REPEATS,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nEvaluate chunk completed in {elapsed:.1f}s")
    print_chunk_summary(rows, "Evaluate")
    append_rows_to_csv(rows, output_path)
    return rows


def run_bootstrap(output_path):
    """Chunk 3: Bootstrap — 9 models x 3 datasets x 2 n_jobs x 2 iters = 108 configs."""
    from bench_parallel import bench_bootstrap

    print("\n" + "=" * 60)
    print("CHUNK 3: Bootstrap")
    print("=" * 60)

    datasets = generate_datasets()
    splits = split_datasets(datasets)
    print("\n--- Training models for bootstrap ---")
    fitted = train_models(splits)

    t0 = time.perf_counter()
    rows = bench_bootstrap(
        fitted, splits,
        n_bootstrap_list=[50, 200],
        n_jobs_list=N_JOBS_LIST,
        n_warmup=N_WARMUP, n_repeats=N_REPEATS,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nBootstrap chunk completed in {elapsed:.1f}s")
    print_chunk_summary(rows, "Bootstrap")
    append_rows_to_csv(rows, output_path)
    return rows


def run_tuning(output_path):
    """Chunk 4: Tuning — 9 models x 1 dataset x 2 n_jobs = 18 configs."""
    from bench_parallel import bench_tuning

    print("\n" + "=" * 60)
    print("CHUNK 4: Tuning")
    print("=" * 60)

    # Only smallest dataset for tuning
    datasets = generate_datasets()
    tuning_datasets = {DATASETS[0][0]: datasets[DATASETS[0][0]]}

    t0 = time.perf_counter()
    rows = bench_tuning(
        tuning_datasets, ALL_MODELS, N_JOBS_LIST,
        n_trials=5, n_warmup=N_WARMUP, n_repeats=N_REPEATS,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nTuning chunk completed in {elapsed:.1f}s")
    print_chunk_summary(rows, "Tuning")
    append_rows_to_csv(rows, output_path)
    return rows


def run_shap(output_path):
    """Chunk 5: SHAP — 9 models x 3 datasets = 27 configs."""
    from bench_parallel import bench_shap

    print("\n" + "=" * 60)
    print("CHUNK 5: SHAP")
    print("=" * 60)

    datasets = generate_datasets()
    splits = split_datasets(datasets)
    print("\n--- Training models for SHAP ---")
    fitted = train_models(splits)

    t0 = time.perf_counter()
    rows = bench_shap(
        fitted, splits,
        n_warmup=N_WARMUP, n_repeats=N_REPEATS,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nSHAP chunk completed in {elapsed:.1f}s")
    print_chunk_summary(rows, "SHAP")
    append_rows_to_csv(rows, output_path)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STEP_RUNNERS = {
    "cv": run_cv,
    "evaluate": run_evaluate,
    "bootstrap": run_bootstrap,
    "tuning": run_tuning,
    "shap": run_shap,
}


def main():
    parser = argparse.ArgumentParser(description="Chunked full-runtime benchmark")
    parser.add_argument(
        "--step", required=True,
        choices=list(STEP_RUNNERS.keys()) + ["all"],
        help="Which benchmark step to run",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    print(f"bench_full.py | step={args.step} | output={args.output}")
    print(f"Models: {len(ALL_MODELS)} | Datasets: {len(DATASETS)} | n_jobs: {N_JOBS_LIST}")
    print(f"Warmup: {N_WARMUP} | Repeats: {N_REPEATS}")
    print()

    t_total = time.perf_counter()

    if args.step == "all":
        for step_name, runner in STEP_RUNNERS.items():
            runner(args.output)
    else:
        STEP_RUNNERS[args.step](args.output)

    elapsed_total = time.perf_counter() - t_total
    print(f"\n{'=' * 60}")
    print(f"Total elapsed: {elapsed_total:.1f}s ({elapsed_total / 60:.1f} min)")

    # Final summary
    output_path = Path(args.output)
    if output_path.exists():
        df = pd.read_csv(output_path)
        print(f"Total rows in {output_path.name}: {len(df)}")
        print(f"Steps covered: {sorted(df['step'].unique())}")


if __name__ == "__main__":
    main()
