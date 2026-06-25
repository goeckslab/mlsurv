"""Re-tune DeepSurv and DeepHit on SUPPORT for Vignette 03.

Vignette 03 embeds tuned hyperparameters for DeepSurv and DeepHit on SUPPORT
(``DEEPSURV_SUPPORT_TUNED`` / ``DEEPHIT_SUPPORT_TUNED``) so the notebook runs
fast. Those literals were produced by 100 Optuna trials each. This script
regenerates them against the current codebase and prints them in ready-to-paste
constructor form.

The Optuna search space and the model ``__init__`` signature do NOT use the same
parameter names: tuning samples ``n_layers`` / ``n_nodes`` / ``lr`` (DeepSurv),
but the constructor takes ``layers=[...]`` / ``learning_rate``. ``best_params_``
comes back in search-space form, so this script translates before printing —
paste the printed dicts straight into the notebook's ``model_params``.

Reproducibility: ``random_state=42`` and ``n_jobs=1`` (the project's determinism
rule — see CONTRIBUTING.md). Neural nets go through torch, so results are as
reproducible as the original run, not guaranteed bit-identical to the last digit.

Usage:
    # Smoke test first (2 trials each, ~1-2 min) to catch API errors:
    python scripts/figures/tune_vignette_03_support.py --smoke

    # Full run (100 trials each, ~1-4+ hours):
    python scripts/figures/tune_vignette_03_support.py

    # Custom trial count:
    python scripts/figures/tune_vignette_03_support.py --n-trials 50
"""

import argparse
import json
import time
from pathlib import Path

from pycox.datasets import support

from mlsurv import SurvivalLearner
from mlsurv.utils import to_structured_array

# Repo root is two levels up from scripts/figures/.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples" / "output"

MODELS = ["deepsurv", "deephit"]


def load_support():
    """Load SUPPORT into mlsurv format, matching the Vignette 03 helper."""
    # SUPPORT's x1-x6 are nominal categoricals that Kvamme one-hot encodes only
    # for the classical Cox baseline. This script tunes the embedding-free
    # neural nets (tuned on the raw integer codes), so we keep them as-is.
    df = support.read_df()
    df = df[df["duration"] > 0]
    X = df.drop(columns=["duration", "event"])
    y = to_structured_array(df["duration"].values, df["event"].values)
    return X, y


def to_constructor_params(best_params):
    """Translate Optuna search-space params to model constructor kwargs.

    The tuners sample ``n_layers`` + ``n_nodes`` and build ``layers`` as
    ``[n_nodes] * n_layers``; DeepSurv also samples ``lr`` and renames it to
    ``learning_rate``. ``best_params_`` records the sampled (search-space) names,
    so we apply the same mapping the tuner's ``objective()`` does. Any other keys
    (dropout, activation, alpha, sigma, weight_decay, batch_size, num_durations)
    already match the constructor signature and pass through unchanged.
    """
    params = dict(best_params)
    if "n_layers" in params and "n_nodes" in params:
        n_layers = params.pop("n_layers")
        n_nodes = params.pop("n_nodes")
        params["layers"] = [n_nodes] * n_layers
    if "lr" in params:
        params["learning_rate"] = params.pop("lr")
    return params


def format_dict_literal(name, params):
    """Render a params dict as a copy-pasteable Python literal."""
    lines = [f"{name} = {{"]
    for key, value in params.items():
        lines.append(f"    {key!r}: {value!r},")
    lines.append("}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Optuna trials per model (default: 100).")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: override --n-trials to 2 to verify the "
                             "tuning path before the long full run.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Where to write the params JSON "
                             f"(default: {DEFAULT_OUTPUT_DIR}).")
    args = parser.parse_args()

    n_trials = 2 if args.smoke else args.n_trials
    mode = "SMOKE TEST" if args.smoke else "FULL RUN"

    print(f"=== Vignette 03 SUPPORT re-tuning ({mode}) ===")
    print(f"n_trials={n_trials} per model, scoring=ctd, seed=42, n_jobs=1\n")

    X, y = load_support()
    print(f"SUPPORT loaded: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{int(y['event'].sum())} events\n")

    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.2, cv_folds=5, random_state=42,
                  scoring="ctd", n_jobs=1, verbose=False)

    constructor_params = {}
    for model in MODELS:
        print(f"--- Tuning {model} ({n_trials} trials) ---")
        t0 = time.time()
        results = learner.run(
            model, n_trials=n_trials, tune=True,
            experiment_name=f"{model}_support_ctd",
            verbose=False, evaluate=False, bootstrap=False,
            n_importance_repeats=0, include_shap=False,
        )
        elapsed = time.time() - t0
        best = results[model].best_params_
        ctd = results[model].get_metric("ctd", phase="cv", stat="mean")
        ctd_str = f"{ctd:.4f}" if ctd is not None else "N/A"
        print(f"  done in {elapsed/60:.1f} min  |  best CV ctd={ctd_str}")
        print(f"  best_params_ (search-space form): {best}")
        constructor_params[model] = to_constructor_params(best)
        print()

    # Persist for the record alongside the vignette output CSVs.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "vignette_03_tuned_params.json"
    out_json.write_text(json.dumps(constructor_params, indent=2))

    print("=" * 64)
    print("Ready-to-paste constructor-form dicts for the notebook:\n")
    print(format_dict_literal("DEEPSURV_SUPPORT_TUNED",
                              constructor_params["deepsurv"]))
    print()
    print(format_dict_literal("DEEPHIT_SUPPORT_TUNED",
                              constructor_params["deephit"]))
    print(f"\nAlso saved to {out_json}")
    if args.smoke:
        print("\nSMOKE TEST PASSED — re-run without --smoke for the full 100-trial run.")


if __name__ == "__main__":
    main()
