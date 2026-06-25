"""Step-by-step timing benchmarks for parallelization.

Times each workflow step in isolation across models, dataset sizes, and n_jobs values.

Usage:
    python tests/benchmarks/bench_parallel.py           # Full benchmark suite
    python tests/benchmarks/bench_parallel.py --quick   # Reduced sweep for CI/quick checks
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def make_synthetic(n_samples, n_features, seed=42):
    """Generate synthetic survival data for benchmarking."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    true_coef = np.zeros(n_features)
    true_coef[: min(5, n_features)] = [0.5, -0.3, 0.4, -0.2, 0.3][: min(5, n_features)]
    linear_pred = X.values @ true_coef
    times = np.exp(-linear_pred + rng.randn(n_samples) * 0.5)
    times = np.clip(times, 0.1, 10)
    events = rng.binomial(1, 0.7, n_samples).astype(bool)
    y = np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


def load_gbsg2():
    """Load GBSG2 real dataset."""
    try:
        from sksurv.datasets import load_gbsg2
        X, y = load_gbsg2()
        return X, y
    except ImportError:
        print("WARNING: scikit-survival not installed, skipping GBSG2")
        return None, None


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

def time_fn(fn, n_warmup=2, n_repeats=5):
    """Run fn with warmup, return per-iteration wall and CPU times.

    Returns
    -------
    dict with keys: wall_median, wall_times, cpu_median, cpu_times
    """
    # Warmup
    for _ in range(n_warmup):
        fn()

    wall_times = []
    cpu_times = []
    for _ in range(n_repeats):
        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()
        fn()
        wall_times.append(time.perf_counter() - t0_wall)
        cpu_times.append(time.process_time() - t0_cpu)

    return {
        "wall_median": float(np.median(wall_times)),
        "wall_times": wall_times,
        "cpu_median": float(np.median(cpu_times)),
        "cpu_times": cpu_times,
    }


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def bench_cv_training(models, datasets, n_jobs_list, n_warmup=2, n_repeats=5):
    """Benchmark CV training across models, datasets, and n_jobs values."""
    from mlsurv import SurvivalLearner

    rows = []
    for ds_name, (X, y) in datasets.items():
        for model_name in models:
            for n_jobs in n_jobs_list:
                label = f"CV | {ds_name} | {model_name} | n_jobs={n_jobs}"
                print(f"  {label}...", end="", flush=True)

                def run():
                    learner = SurvivalLearner(X, y)
                    learner.setup(cv_folds=3, random_state=42, n_jobs=n_jobs)
                    learner.train(model_name)

                try:
                    result = time_fn(run, n_warmup=n_warmup, n_repeats=n_repeats)
                    rows.append({
                        "step": "cv_training",
                        "dataset": ds_name,
                        "model": model_name,
                        "n_jobs": n_jobs,
                        "wall_median": result["wall_median"],
                        "cpu_median": result["cpu_median"],
                    })
                    print(f" {result['wall_median']:.3f}s")
                except Exception as e:
                    print(f" FAILED: {e}")
                    rows.append({
                        "step": "cv_training",
                        "dataset": ds_name,
                        "model": model_name,
                        "n_jobs": n_jobs,
                        "wall_median": float("nan"),
                        "cpu_median": float("nan"),
                    })
    return rows


def bench_evaluate(models_to_eval, datasets, n_jobs_list,
                   n_importance_repeats_list, n_warmup=2, n_repeats=5):
    """Benchmark evaluate() across models, datasets, n_jobs, importance repeats."""
    from mlsurv.evaluation import evaluate

    rows = []
    for ds_name, (X_train, X_test, y_train, y_test) in datasets.items():
        for model_name, model in models_to_eval[ds_name].items():
            for n_repeats_imp in n_importance_repeats_list:
                for n_jobs in n_jobs_list:
                    label = f"evaluate | {ds_name} | {model_name} | repeats={n_repeats_imp} | n_jobs={n_jobs}"
                    print(f"  {label}...", end="", flush=True)

                    def run():
                        evaluate(
                            model, X_test, y_test,
                            X_train=X_train, y_train=y_train,
                            n_importance_repeats=n_repeats_imp,
                            random_state=42,
                            show_progress=False,
                            n_jobs=n_jobs,
                        )

                    try:
                        result = time_fn(run, n_warmup=n_warmup, n_repeats=n_repeats)
                        rows.append({
                            "step": "evaluate",
                            "dataset": ds_name,
                            "model": model_name,
                            "n_importance_repeats": n_repeats_imp,
                            "n_jobs": n_jobs,
                            "wall_median": result["wall_median"],
                            "cpu_median": result["cpu_median"],
                        })
                        print(f" {result['wall_median']:.3f}s")
                    except Exception as e:
                        print(f" FAILED: {e}")
                        rows.append({
                            "step": "evaluate",
                            "dataset": ds_name,
                            "model": model_name,
                            "n_importance_repeats": n_repeats_imp,
                            "n_jobs": n_jobs,
                            "wall_median": float("nan"),
                            "cpu_median": float("nan"),
                        })
    return rows


def bench_bootstrap(models_to_boot, datasets, n_bootstrap_list, n_jobs_list,
                    n_warmup=2, n_repeats=5):
    """Benchmark bootstrap() across models, iterations, and n_jobs values."""
    from mlsurv.evaluation import bootstrap

    rows = []
    for ds_name, (X_train, X_test, y_train, y_test) in datasets.items():
        for model_name, model in models_to_boot[ds_name].items():
            for n_iter in n_bootstrap_list:
                for n_jobs in n_jobs_list:
                    label = f"bootstrap | {ds_name} | {model_name} | iters={n_iter} | n_jobs={n_jobs}"
                    print(f"  {label}...", end="", flush=True)

                    def run():
                        bootstrap(
                            model, X_test, y_test,
                            y_train=y_train,
                            n_bootstrap=n_iter,
                            random_state=42,
                            show_progress=False,
                            n_jobs=n_jobs,
                            force_parallel=True,  # Bypass auto-disable for benchmarking
                        )

                    try:
                        result = time_fn(run, n_warmup=n_warmup, n_repeats=n_repeats)
                        rows.append({
                            "step": "bootstrap",
                            "dataset": ds_name,
                            "model": model_name,
                            "n_bootstrap": n_iter,
                            "n_jobs": n_jobs,
                            "wall_median": result["wall_median"],
                            "cpu_median": result["cpu_median"],
                        })
                        print(f" {result['wall_median']:.3f}s")
                    except Exception as e:
                        print(f" FAILED: {e}")
                        rows.append({
                            "step": "bootstrap",
                            "dataset": ds_name,
                            "model": model_name,
                            "n_bootstrap": n_iter,
                            "n_jobs": n_jobs,
                            "wall_median": float("nan"),
                            "cpu_median": float("nan"),
                        })
    return rows


def bench_tuning(datasets, models, n_jobs_list, n_trials=10, n_warmup=1, n_repeats=3):
    """Benchmark tuning across models, datasets, and n_jobs values."""
    from mlsurv import SurvivalLearner

    rows = []
    for ds_name, (X, y) in datasets.items():
        for model_name in models:
            for n_jobs in n_jobs_list:
                label = f"tuning | {ds_name} | {model_name} | trials={n_trials} | n_jobs={n_jobs}"
                print(f"  {label}...", end="", flush=True)

                def run():
                    learner = SurvivalLearner(X, y)
                    learner.setup(cv_folds=3, random_state=42, n_jobs=n_jobs)
                    learner.train(model_name)
                    learner.tune(model_name, n_trials=n_trials)

                try:
                    result = time_fn(run, n_warmup=n_warmup, n_repeats=n_repeats)
                    rows.append({
                        "step": "tuning",
                        "dataset": ds_name,
                        "model": model_name,
                        "n_trials": n_trials,
                        "n_jobs": n_jobs,
                        "wall_median": result["wall_median"],
                        "cpu_median": result["cpu_median"],
                    })
                    print(f" {result['wall_median']:.3f}s")
                except Exception as e:
                    print(f" FAILED: {e}")
                    rows.append({
                        "step": "tuning",
                        "dataset": ds_name,
                        "model": model_name,
                        "n_trials": n_trials,
                        "n_jobs": n_jobs,
                        "wall_median": float("nan"),
                        "cpu_median": float("nan"),
                    })
    return rows


def bench_shap(models_shap, datasets, n_warmup=1, n_repeats=3):
    """Benchmark SHAP computation."""
    from mlsurv.evaluation._shap import _compute_shap

    rows = []
    for ds_name, (X_train, X_test, y_train, y_test) in datasets.items():
        for model_name, model in models_shap[ds_name].items():
            label = f"shap | {ds_name} | {model_name} | n={len(X_test)}"
            print(f"  {label}...", end="", flush=True)

            def run():
                _compute_shap(model, X_test)

            try:
                result = time_fn(run, n_warmup=n_warmup, n_repeats=n_repeats)
                rows.append({
                    "step": "shap",
                    "dataset": ds_name,
                    "model": model_name,
                    "n_samples": len(X_test),
                    "wall_median": result["wall_median"],
                    "cpu_median": result["cpu_median"],
                })
                print(f" {result['wall_median']:.3f}s")
            except Exception as e:
                print(f" FAILED: {e}")
                rows.append({
                    "step": "shap",
                    "dataset": ds_name,
                    "model": model_name,
                    "n_samples": len(X_test),
                    "wall_median": float("nan"),
                    "cpu_median": float("nan"),
                })
    return rows


# ---------------------------------------------------------------------------
# Speedup analysis
# ---------------------------------------------------------------------------

def compute_speedup_table(rows):
    """Compute speedup and efficiency relative to n_jobs=1 baseline."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Group by everything except n_jobs and timing columns
    id_cols = [c for c in df.columns if c not in ("n_jobs", "wall_median", "cpu_median")]
    enriched = []

    for _, group in df.groupby(id_cols, dropna=False):
        baseline = group[group["n_jobs"] == 1]
        if baseline.empty:
            for _, row in group.iterrows():
                row_dict = row.to_dict()
                row_dict["speedup"] = float("nan")
                row_dict["efficiency"] = float("nan")
                enriched.append(row_dict)
            continue

        base_wall = baseline["wall_median"].values[0]
        for _, row in group.iterrows():
            row_dict = row.to_dict()
            if row["n_jobs"] == 1 or np.isnan(row["wall_median"]) or np.isnan(base_wall):
                row_dict["speedup"] = 1.0 if row["n_jobs"] == 1 else float("nan")
                row_dict["efficiency"] = 1.0 if row["n_jobs"] == 1 else float("nan")
            else:
                speedup = base_wall / row["wall_median"]
                n = row["n_jobs"] if row["n_jobs"] > 0 else os.cpu_count() or 1
                row_dict["speedup"] = round(speedup, 2)
                row_dict["efficiency"] = round(speedup / n, 2)
            enriched.append(row_dict)

    return pd.DataFrame(enriched)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parallelization benchmarks for mlsurv")
    parser.add_argument("--quick", action="store_true", help="Reduced sweep for quick checks")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    from mlsurv.models import get_model

    # -----------------------------------------------------------------------
    # Configuration: full vs quick
    # -----------------------------------------------------------------------
    if args.quick:
        cv_models = ["coxph", "rsf"]
        cv_n_jobs = [1, 4]
        synthetic_specs = [("synth_200_20", 200, 20)]
        boot_n_bootstrap = [20, 100]
        boot_n_jobs = [1, 4]
        eval_n_jobs = [1, 4]
        eval_importance_repeats = [5]
        tuning_models = ["rsf"]
        tuning_n_jobs = [1, 4]
        tuning_n_trials = 5
        eval_models_names = ["coxph", "rsf"]
        boot_models_names = ["coxph", "rsf"]
        shap_models_names = ["coxph"]
        n_warmup = 1
        n_repeats = 3
    else:
        cv_models = ["coxph", "coxnet", "rsf", "gbsa", "fssvm", "xgbcox", "weibull_aft",
                     "deepsurv", "deephit"]
        cv_n_jobs = [1, 2, 4, 8, 16]
        synthetic_specs = [
            ("synth_200_20", 200, 20),
            ("synth_1000_30", 1000, 30),
            ("synth_3000_50", 3000, 50),
        ]
        boot_n_bootstrap = [20, 50, 100, 200, 500]
        boot_n_jobs = [1, 4, 8, 16]
        eval_n_jobs = [1, 4, 8, 16]
        eval_importance_repeats = [5, 10]
        tuning_models = ["rsf", "xgbcox", "coxph", "deepsurv"]
        tuning_n_jobs = [1, 4, 8, 16]
        tuning_n_trials = 10
        eval_models_names = ["coxph", "coxnet", "rsf", "gbsa", "fssvm", "xgbcox",
                             "weibull_aft", "deepsurv", "deephit"]
        boot_models_names = ["coxph", "coxnet", "rsf", "gbsa", "fssvm", "xgbcox",
                             "weibull_aft", "deepsurv", "deephit"]
        shap_models_names = ["coxph", "coxnet", "rsf", "gbsa", "fssvm", "xgbcox",
                             "weibull_aft", "deepsurv", "deephit"]
        n_warmup = 2
        n_repeats = 5

    # -----------------------------------------------------------------------
    # Generate datasets
    # -----------------------------------------------------------------------
    print("=== Generating datasets ===")
    cv_datasets = {}
    for name, n, f in synthetic_specs:
        print(f"  {name}: n={n}, features={f}")
        cv_datasets[name] = make_synthetic(n, f)

    # Split datasets for evaluate/bootstrap/shap
    split_datasets = {}
    for name, (X, y) in cv_datasets.items():
        split_idx = int(len(X) * 0.7)
        split_datasets[name] = (X.iloc[:split_idx], X.iloc[split_idx:],
                                y[:split_idx], y[split_idx:])

    # Train fitted models for evaluate/bootstrap/shap benchmarks
    print("\n=== Training models for evaluate/bootstrap/shap benchmarks ===")
    eval_fitted = {}
    boot_fitted = {}
    shap_fitted = {}
    for ds_name, (X_train, X_test, y_train, y_test) in split_datasets.items():
        eval_fitted[ds_name] = {}
        boot_fitted[ds_name] = {}
        shap_fitted[ds_name] = {}
        for model_name in set(eval_models_names + boot_models_names + shap_models_names):
            print(f"  Training {model_name} on {ds_name}...", end="", flush=True)
            try:
                model = get_model(model_name)
                model.fit(X_train, y_train)
                if model_name in eval_models_names:
                    eval_fitted[ds_name][model_name] = model
                if model_name in boot_models_names:
                    boot_fitted[ds_name][model_name] = model
                if model_name in shap_models_names:
                    shap_fitted[ds_name][model_name] = model
                print(" done.")
            except Exception as e:
                print(f" FAILED: {e}")

    # -----------------------------------------------------------------------
    # Run benchmarks
    # -----------------------------------------------------------------------
    all_rows = []

    print("\n=== CV Training Benchmarks ===")
    all_rows.extend(bench_cv_training(
        cv_models, cv_datasets, cv_n_jobs, n_warmup=n_warmup, n_repeats=n_repeats,
    ))

    print("\n=== Evaluate Benchmarks ===")
    all_rows.extend(bench_evaluate(
        eval_fitted, split_datasets, eval_n_jobs, eval_importance_repeats,
        n_warmup=n_warmup, n_repeats=n_repeats,
    ))

    print("\n=== Bootstrap Benchmarks ===")
    all_rows.extend(bench_bootstrap(
        boot_fitted, split_datasets, boot_n_bootstrap, boot_n_jobs,
        n_warmup=n_warmup, n_repeats=n_repeats,
    ))

    print("\n=== Tuning Benchmarks ===")
    tuning_datasets = {k: v for k, v in cv_datasets.items()
                       if k in [synthetic_specs[0][0]]}  # Only smallest dataset for tuning
    all_rows.extend(bench_tuning(
        tuning_datasets, tuning_models, tuning_n_jobs,
        n_trials=tuning_n_trials, n_warmup=max(1, n_warmup - 1), n_repeats=max(2, n_repeats - 2),
    ))

    print("\n=== SHAP Benchmarks ===")
    all_rows.extend(bench_shap(
        shap_fitted, split_datasets, n_warmup=max(1, n_warmup), n_repeats=max(2, n_repeats),
    ))

    # -----------------------------------------------------------------------
    # Compute speedup and save results
    # -----------------------------------------------------------------------
    print("\n=== Computing speedup analysis ===")
    result_df = compute_speedup_table(all_rows)

    output_path = args.output or str(RESULTS_DIR / "bench_parallel.csv")
    result_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n=== Summary ===")
    if not result_df.empty and "speedup" in result_df.columns:
        summary_cols = ["step", "dataset", "model", "n_jobs", "wall_median", "speedup", "efficiency"]
        avail_cols = [c for c in summary_cols if c in result_df.columns]
        print(result_df[avail_cols].to_string(index=False))
    else:
        print(result_df.to_string(index=False))

    print(f"\nTotal benchmark rows: {len(result_df)}")


if __name__ == "__main__":
    main()
