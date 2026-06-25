"""Profile evaluation substeps per model to identify bottlenecks.

Reproduces the vignette 2 setup (GBSG2, Cox/tree/AFT models) and times
each eval substep: predict, survival functions, surv_funcs->array
conversion, Brier/IBS computation, and permutation importance.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sksurv.datasets import load_gbsg2

# Suppress noisy warnings during mlsurv imports and model fitting
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="mlsurv")

from mlsurv import SurvivalLearner  # noqa: E402
from mlsurv.evaluation.importance import _compute_permutation_importance  # noqa: E402
from mlsurv.evaluation.metrics import (  # noqa: E402
    _surv_funcs_to_probs,
    _brier_score_at_times,
    _integrated_brier,
)
from mlsurv.utils import (  # noqa: E402
    auto_eval_times,
    clip_eval_times,
    to_structured_array,
    truncate_y_to_train_range,
)


def _timer(func, *args, **kwargs):
    """Run func and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def main():
    print("=" * 70)
    print("Vignette 2 Evaluation Profiler")
    print("=" * 70)

    # --- Setup: same as vignette 2 ---
    X, y_raw = load_gbsg2()
    X = X.copy()
    X['horTh'] = (X['horTh'] == 'yes').astype(int)
    X['menostat'] = (X['menostat'] == 'Post').astype(int)
    X['tgrade'] = X['tgrade'].map({'I': 1, 'II': 2, 'III': 3}).astype(int)
    y = to_structured_array(y_raw['time'], y_raw['cens'])

    learner = SurvivalLearner(X, y)
    learner.setup(
        test_fraction=0.3,
        cv_folds=5,
        imputer='knn',
        scaler='robust',
        feature_selector='topk',
        random_state=42,
        verbose=0,
    )

    model_names = ["coxph", "coxnet", "rsf", "weibull_aft"]
    models = {}
    for name in model_names:
        print(f"Training {name}...", end=" ", flush=True)
        result, t = _timer(learner.train, name)
        models[name] = result.model
        print(f"{t:.1f}s")

    X_test = learner.X_test
    y_test = learner.y_test
    y_train = learner.y_train

    eval_times = auto_eval_times(y_test, y_train=y_train)
    eval_times = clip_eval_times(eval_times, y_train)
    y_test_ipcw = truncate_y_to_train_range(y_test, y_train)

    n_test = len(X_test)
    n_features = X_test.shape[1]
    n_eval_times = len(eval_times)

    print(f"\nTest set: {n_test} samples, {n_features} features, {n_eval_times} eval times")
    print()

    # --- Profile each model ---
    rows = []
    for name, model in models.items():
        print(f"--- Profiling {name} ---")
        row = {"model": name}

        # 1. predict (risk scores)
        _, t = _timer(model.predict, X_test)
        row["predict"] = t
        print(f"  predict:          {t:.4f}s")

        # 2. predict_survival_function
        if hasattr(model, "predict_survival_function"):
            surv_funcs, t = _timer(model.predict_survival_function, X_test)
            row["surv_funcs"] = t
            print(f"  surv_funcs:       {t:.4f}s")

            # 3. _surv_funcs_to_probs conversion
            _, t = _timer(_surv_funcs_to_probs, surv_funcs, eval_times)
            row["to_probs"] = t
            print(f"  to_probs:         {t:.4f}s")

            # 4. _brier_score_at_times (single median time)
            median_time = np.median(eval_times)
            _, t = _timer(_brier_score_at_times, y_train, y_test_ipcw, surv_funcs, [median_time])
            row["brier_1t"] = t
            print(f"  brier(1 time):    {t:.4f}s")

            # 5. _integrated_brier (all eval_times)
            _, t = _timer(_integrated_brier, y_train, y_test_ipcw, surv_funcs, eval_times)
            row["ibs"] = t
            print(f"  ibs(all times):   {t:.4f}s")

            # 6. brier + ibs combined (old wrapper path, before optimization)
            row["brier_ibs_total"] = row["brier_1t"] + row["ibs"]
            print(f"  brier+ibs total:  {row['brier_ibs_total']:.4f}s")
        else:
            for k in ["surv_funcs", "to_probs", "brier_1t", "ibs", "brier_ibs_total"]:
                row[k] = np.nan

        # 7. Permutation importance (100 repeats)
        _, t = _timer(
            _compute_permutation_importance, model, X_test, y_test, 100, 42,
            show_progress=False, model_name=name, n_jobs=1,
        )
        row["importance_100"] = t
        print(f"  importance(100):  {t:.1f}s")

        # 8. Permutation importance (10 repeats for comparison)
        _, t = _timer(
            _compute_permutation_importance, model, X_test, y_test, 10, 42,
            show_progress=False, model_name=name, n_jobs=1,
        )
        row["importance_10"] = t
        print(f"  importance(10):   {t:.1f}s")

        rows.append(row)
        print()

    # --- Summary table ---
    df = pd.DataFrame(rows)
    print("=" * 70)
    print("SUMMARY (seconds)")
    print("=" * 70)

    cols = ["model", "predict", "surv_funcs", "to_probs", "brier_1t", "ibs",
            "brier_ibs_total", "importance_100", "importance_10"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n--- Analysis ---")
    slowest = df.loc[df["importance_100"].idxmax()]
    print(f"Slowest importance (100 reps): {slowest['model']} at {slowest['importance_100']:.1f}s")

    total_brier = df["brier_ibs_total"].sum()
    print(f"Total brier+ibs across all models: {total_brier:.3f}s")


if __name__ == "__main__":
    main()
