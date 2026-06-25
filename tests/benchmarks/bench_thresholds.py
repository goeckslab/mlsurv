"""Crossover-point finder for parallelization thresholds.

For each auto-disable threshold, sweeps the controlling variable to find where
parallel becomes faster than sequential.

Usage:
    python tests/benchmarks/bench_thresholds.py           # Full sweep
    python tests/benchmarks/bench_thresholds.py --quick   # Reduced sweep for CI
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic(n_samples, n_features, seed=42):
    """Generate synthetic survival data."""
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


def timed_median(fn, n_reps=3):
    """Run fn n_reps times, return median wall time."""
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Threshold sweeps
# ---------------------------------------------------------------------------

def sweep_bootstrap_iterations(model, X_test, y_test, y_train, sweep_range,
                               precompute, n_reps=3):
    """Sweep bootstrap iterations to find parallel crossover point.

    Parameters
    ----------
    precompute : bool
        If True, simulates precomputed mode (no custom_metrics/importance).
    """
    from mlsurv.evaluation import bootstrap

    results = []
    for n_iter in sweep_range:
        label = f"bootstrap iters={n_iter} precompute={precompute}"
        print(f"  {label}...", end="", flush=True)

        def run_seq():
            bootstrap(
                model, X_test, y_test, y_train=y_train,
                n_bootstrap=n_iter, random_state=42, show_progress=False,
                n_jobs=1, force_parallel=True,
                include_importance=not precompute,
            )

        def run_par():
            bootstrap(
                model, X_test, y_test, y_train=y_train,
                n_bootstrap=n_iter, random_state=42, show_progress=False,
                n_jobs=-1, force_parallel=True,
                include_importance=not precompute,
            )

        t_seq = timed_median(run_seq, n_reps)
        t_par = timed_median(run_par, n_reps)
        faster = "PAR" if t_par < t_seq else "SEQ"
        print(f" seq={t_seq:.3f}s par={t_par:.3f}s [{faster}]")

        results.append({
            "threshold": "MIN_ITERATIONS_FOR_PARALLEL" + ("_PRECOMPUTED" if precompute else ""),
            "value": n_iter,
            "time_seq": t_seq,
            "time_par": t_par,
            "faster": faster,
        })

    return results


def sweep_model_iterations_outer(models, X_test, y_test, y_train, sweep_range, n_reps=3):
    """Sweep n_models * n_bootstrap for outer parallelization crossover."""
    from mlsurv.evaluation import bootstrap

    n_models = len(models)
    results = []
    for n_iter in sweep_range:
        total_work = n_models * n_iter
        label = f"outer parallel total_work={total_work} ({n_models}×{n_iter})"
        print(f"  {label}...", end="", flush=True)

        def run_seq():
            bootstrap(
                models, X_test, y_test, y_train=y_train,
                n_bootstrap=n_iter, random_state=42, show_progress=False,
                n_jobs=1, n_jobs_models=1, force_parallel=True,
            )

        def run_par():
            bootstrap(
                models, X_test, y_test, y_train=y_train,
                n_bootstrap=n_iter, random_state=42, show_progress=False,
                n_jobs=1, n_jobs_models=-1, force_parallel=True,
            )

        t_seq = timed_median(run_seq, n_reps)
        t_par = timed_median(run_par, n_reps)
        faster = "PAR" if t_par < t_seq else "SEQ"
        print(f" seq={t_seq:.3f}s par={t_par:.3f}s [{faster}]")

        results.append({
            "threshold": "MIN_MODEL_ITERATIONS_FOR_OUTER_PARALLEL",
            "value": total_work,
            "n_bootstrap": n_iter,
            "time_seq": t_seq,
            "time_par": t_par,
            "faster": faster,
        })

    return results


def sweep_features_univariate_cox(X, y, sweep_range, n_reps=3):
    """Sweep n_features for univariate Cox parallelization crossover."""
    from mlsurv.evaluation.importance import compute_univariate_cox

    results = []
    for n_feat in sweep_range:
        label = f"univariate_cox n_features={n_feat}"
        print(f"  {label}...", end="", flush=True)

        X_sub = X.iloc[:, :n_feat]

        def run_seq():
            compute_univariate_cox(X_sub, y, n_jobs=1, force_parallel=True)

        def run_par():
            compute_univariate_cox(X_sub, y, n_jobs=-1, force_parallel=True)

        t_seq = timed_median(run_seq, n_reps)
        t_par = timed_median(run_par, n_reps)
        faster = "PAR" if t_par < t_seq else "SEQ"
        print(f" seq={t_seq:.3f}s par={t_par:.3f}s [{faster}]")

        results.append({
            "threshold": "MIN_FEATURES_FOR_PARALLEL",
            "value": n_feat,
            "time_seq": t_seq,
            "time_par": t_par,
            "faster": faster,
        })

    return results


def sweep_sequential_cv_models(models_to_test, X, y, n_reps=3):
    """Test whether models belong in SEQUENTIAL_CV_MODELS.

    Compares n_jobs=1 vs n_jobs=16 for CV training.
    """
    from mlsurv import SurvivalLearner

    results = []
    for model_name in models_to_test:
        label = f"cv_sequential_test {model_name}"
        print(f"  {label}...", end="", flush=True)

        def run_seq():
            learner = SurvivalLearner(X, y)
            learner.setup(cv_folds=3, random_state=42, n_jobs=1)
            learner.train(model_name)

        def run_par():
            learner = SurvivalLearner(X, y)
            learner.setup(cv_folds=3, random_state=42, n_jobs=-1)
            learner.train(model_name)

        try:
            t_seq = timed_median(run_seq, n_reps)
            t_par = timed_median(run_par, n_reps)
            speedup = t_seq / t_par if t_par > 0 else float("inf")
            recommend = "SEQUENTIAL" if speedup < 1.1 else "PARALLEL"
            print(f" seq={t_seq:.3f}s par={t_par:.3f}s speedup={speedup:.2f}x [{recommend}]")

            results.append({
                "threshold": "SEQUENTIAL_CV_MODELS",
                "model": model_name,
                "time_seq": t_seq,
                "time_par": t_par,
                "speedup": speedup,
                "recommendation": recommend,
            })
        except Exception as e:
            print(f" FAILED: {e}")
            results.append({
                "threshold": "SEQUENTIAL_CV_MODELS",
                "model": model_name,
                "time_seq": float("nan"),
                "time_par": float("nan"),
                "speedup": float("nan"),
                "recommendation": "ERROR",
            })

    return results


def sweep_bootstrap_backend(model, X_test, y_test, y_train, n_bootstrap_list, n_reps=3):
    """Compare loky vs threading backend for precomputed bootstrap.

    NumPy releases the GIL, so threading avoids serialization overhead.
    """
    from mlsurv.evaluation._bootstrap_workers import _precompute_predictions
    from mlsurv.evaluation.bootstrap import _generate_seeds
    from mlsurv.evaluation._config import BootstrapConfig
    from mlsurv.utils import auto_eval_times, adjust_model_njobs
    from mlsurv.constants import ALL_METRICS
    from joblib import Parallel, delayed
    from mlsurv.evaluation._bootstrap_iteration import _bootstrap_single_iteration

    results = []
    eval_times = auto_eval_times(y_test)

    for n_iter in n_bootstrap_list:
        label = f"backend test iters={n_iter}"
        print(f"  {label}...", end="", flush=True)

        # Set up the bootstrap infrastructure once
        seeds = _generate_seeds(42, n_iter)
        config = BootstrapConfig(
            metrics=ALL_METRICS,
            custom_metrics={},
            eval_times=eval_times,
            ci=0.95,
            include_predictions=False,
            include_auc=False,
            include_importance=False,
            n_importance_repeats=10,
            stratify=False,
        )

        test_model = adjust_model_njobs(model, -1)
        (precomputed_risk, precomputed_surv_probs,
         precomputed_ibs_surv_probs, ibs_times, can_precompute) = _precompute_predictions(
            test_model, X_test, ALL_METRICS, eval_times, {}, False, "bench",
        )
        feature_names = X_test.columns.tolist()

        iter_kwargs = dict(
            model=test_model, X_test=X_test, y_test=y_test, y_train=y_train,
            config=config, model_name="bench", feature_names=feature_names,
            precomputed_risk=precomputed_risk, precomputed_surv_probs=precomputed_surv_probs,
        )

        n_jobs = os.cpu_count() or 4

        for backend in ["loky", "threading"]:
            def run_backend():
                jobs = (
                    delayed(_bootstrap_single_iteration)(i=i, seed=seeds[i], **iter_kwargs)
                    for i in range(n_iter)
                )
                Parallel(n_jobs=n_jobs, backend=backend)(jobs)

            try:
                t = timed_median(run_backend, n_reps)
                print(f" {backend}={t:.3f}s", end="")
            except Exception as e:
                t = float("nan")
                print(f" {backend}=FAILED({e})", end="")

            results.append({
                "threshold": "BOOTSTRAP_BACKEND",
                "value": n_iter,
                "backend": backend,
                "time": t,
            })

        print()

    return results


def sweep_large_dataset_threshold(X, y, sweep_range, n_reps=3):
    """Sweep n_samples for tuning heuristic crossover."""
    from mlsurv import SurvivalLearner

    results = []
    for n_samples in sweep_range:
        label = f"tuning threshold n_samples={n_samples}"
        print(f"  {label}...", end="", flush=True)

        X_sub = X.iloc[:n_samples]
        y_sub = y[:n_samples]

        def run_seq():
            learner = SurvivalLearner(X_sub, y_sub)
            learner.setup(cv_folds=3, random_state=42, n_jobs=1)
            learner.train("rsf")
            learner.tune("rsf", n_trials=5)

        def run_par():
            learner = SurvivalLearner(X_sub, y_sub)
            learner.setup(cv_folds=3, random_state=42, n_jobs=-1)
            learner.train("rsf")
            learner.tune("rsf", n_trials=5)

        try:
            t_seq = timed_median(run_seq, n_reps)
            t_par = timed_median(run_par, n_reps)
            faster = "PAR" if t_par < t_seq else "SEQ"
            print(f" seq={t_seq:.3f}s par={t_par:.3f}s [{faster}]")

            results.append({
                "threshold": "LARGE_DATASET_THRESHOLD",
                "value": n_samples,
                "time_seq": t_seq,
                "time_par": t_par,
                "faster": faster,
            })
        except Exception as e:
            print(f" FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_results(all_results):
    """Analyze sweep results and print crossover analysis."""
    from mlsurv.evaluation._bootstrap_workers import (
        MIN_ITERATIONS_FOR_PARALLEL,
        MIN_ITERATIONS_FOR_PARALLEL_PRECOMPUTED,
    )
    from mlsurv.evaluation.bootstrap import MIN_MODEL_ITERATIONS_FOR_OUTER_PARALLEL
    from mlsurv.evaluation.importance import MIN_FEATURES_FOR_PARALLEL

    current_thresholds = {
        "MIN_ITERATIONS_FOR_PARALLEL": MIN_ITERATIONS_FOR_PARALLEL,
        "MIN_ITERATIONS_FOR_PARALLEL_PRECOMPUTED": MIN_ITERATIONS_FOR_PARALLEL_PRECOMPUTED,
        "MIN_MODEL_ITERATIONS_FOR_OUTER_PARALLEL": MIN_MODEL_ITERATIONS_FOR_OUTER_PARALLEL,
        "MIN_FEATURES_FOR_PARALLEL": MIN_FEATURES_FOR_PARALLEL,
    }

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 70)
    print("CROSSOVER ANALYSIS")
    print("=" * 70)

    for threshold_name, current_val in current_thresholds.items():
        subset = df[df["threshold"] == threshold_name]
        if subset.empty:
            print(f"\n{threshold_name}: NO DATA")
            continue

        if "time_seq" in subset.columns and "time_par" in subset.columns:
            crossover_rows = subset[subset["time_par"] < subset["time_seq"]]
            if crossover_rows.empty:
                print(f"\n{threshold_name} (current={current_val}): "
                      f"Sequential always faster in tested range")
            else:
                crossover_val = crossover_rows["value"].min()
                ratio = crossover_val / current_val if current_val > 0 else float("inf")
                flag = " *** NEEDS UPDATE ***" if ratio > 2 or ratio < 0.5 else ""
                print(f"\n{threshold_name} (current={current_val}): "
                      f"Crossover at {crossover_val} (ratio={ratio:.2f}x){flag}")

    # SEQUENTIAL_CV_MODELS analysis
    seq_results = df[df["threshold"] == "SEQUENTIAL_CV_MODELS"]
    if not seq_results.empty:
        print(f"\nSEQUENTIAL_CV_MODELS analysis:")
        for _, row in seq_results.iterrows():
            print(f"  {row['model']}: speedup={row.get('speedup', 'N/A'):.2f}x "
                  f"-> {row.get('recommendation', 'N/A')}")

    # Backend analysis
    backend_results = df[df["threshold"] == "BOOTSTRAP_BACKEND"]
    if not backend_results.empty:
        print(f"\nBOOTSTRAP BACKEND analysis:")
        for n_iter in backend_results["value"].unique():
            iter_data = backend_results[backend_results["value"] == n_iter]
            for _, row in iter_data.iterrows():
                print(f"  iters={n_iter} {row['backend']}: {row['time']:.3f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Threshold crossover-point finder")
    parser.add_argument("--quick", action="store_true", help="Reduced sweep for quick checks")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    from mlsurv.models import get_model

    # Configuration
    if args.quick:
        boot_sweep = range(50, 301, 50)
        boot_precomputed_sweep = range(10, 101, 20)
        outer_sweep = range(50, 201, 50)
        features_sweep = range(5, 31, 5)
        seq_cv_models = ["weibull_aft", "fssvm", "gbsa"]
        backend_iters = [50, 100]
        tuning_sweep = range(500, 2001, 500)
        n_reps = 2
        n_samples = 500
        n_features = 40
    else:
        boot_sweep = range(50, 1001, 50)
        boot_precomputed_sweep = range(10, 201, 10)
        outer_sweep = range(50, 601, 50)
        features_sweep = range(5, 41, 5)
        seq_cv_models = ["weibull_aft", "fssvm", "gbsa"]
        backend_iters = [50, 100, 200]
        tuning_sweep = range(500, 4001, 500)
        n_reps = 3
        n_samples = 1000
        n_features = 50

    # Generate data
    print("=== Generating data ===")
    X, y = make_synthetic(n_samples, n_features)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # For tuning sweep, need larger dataset
    X_large, y_large = make_synthetic(max(n_samples, 4000), n_features, seed=123)

    # Train models
    print("\n=== Training models ===")
    coxph = get_model("coxph")
    coxph.fit(X_train, y_train)
    print("  CoxPH trained.")

    rsf = get_model("rsf")
    rsf.fit(X_train, y_train)
    print("  RSF trained.")

    models_dict = {"coxph": coxph, "rsf": rsf}

    all_results = []

    # Sweep 1: Bootstrap iterations (full model)
    print("\n=== Sweep: MIN_ITERATIONS_FOR_PARALLEL ===")
    all_results.extend(
        sweep_bootstrap_iterations(coxph, X_test, y_test, y_train,
                                   boot_sweep, precompute=False, n_reps=n_reps)
    )

    # Sweep 2: Bootstrap iterations (precomputed)
    print("\n=== Sweep: MIN_ITERATIONS_FOR_PARALLEL_PRECOMPUTED ===")
    all_results.extend(
        sweep_bootstrap_iterations(coxph, X_test, y_test, y_train,
                                   boot_precomputed_sweep, precompute=True, n_reps=n_reps)
    )

    # Sweep 3: Outer parallelization
    print("\n=== Sweep: MIN_MODEL_ITERATIONS_FOR_OUTER_PARALLEL ===")
    all_results.extend(
        sweep_model_iterations_outer(models_dict, X_test, y_test, y_train,
                                     outer_sweep, n_reps=n_reps)
    )

    # Sweep 4: Features for univariate Cox
    print("\n=== Sweep: MIN_FEATURES_FOR_PARALLEL ===")
    all_results.extend(
        sweep_features_univariate_cox(X, y, features_sweep, n_reps=n_reps)
    )

    # Sweep 5: SEQUENTIAL_CV_MODELS candidates
    print("\n=== Sweep: SEQUENTIAL_CV_MODELS candidates ===")
    all_results.extend(
        sweep_sequential_cv_models(seq_cv_models, X, y, n_reps=n_reps)
    )

    # Sweep 6: Bootstrap backend (loky vs threading)
    print("\n=== Sweep: Bootstrap backend (loky vs threading) ===")
    all_results.extend(
        sweep_bootstrap_backend(coxph, X_test, y_test, y_train,
                                backend_iters, n_reps=n_reps)
    )

    # Sweep 7: Large dataset threshold for tuning
    print("\n=== Sweep: LARGE_DATASET_THRESHOLD ===")
    all_results.extend(
        sweep_large_dataset_threshold(X_large, y_large, tuning_sweep, n_reps=n_reps)
    )

    # Save and analyze
    result_df = pd.DataFrame(all_results)
    output_path = args.output or str(RESULTS_DIR / "bench_thresholds.csv")
    result_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    analyze_results(all_results)

    print(f"\nTotal sweep points: {len(all_results)}")


if __name__ == "__main__":
    main()
