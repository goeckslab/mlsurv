"""Generate the scikit-survival target numbers for Vignette 03's sksurv suite.

The vignette compares mlsurv against scikit-survival, but a single published
doc number is a single split at one seed, which is noisy. So instead of copying
a doc number, we make each target the MEAN of scikit-survival's own recipe over
a pinned list of seeds. This script produces those means.

It runs in two steps, in order:

1. replicate_anchors() -- reproduce the EXACT values printed in the
   scikit-survival docs, at their documented seeds. This is the trust anchor:
   the recipes here must hit the published numbers before we average anything.

2. multiseed_targets() -- run each recipe over a pinned seed list and report the
   mean (and std). These means are what the vignette uses as its `target` column.

Run this script and paste the printed SKSURV_TARGETS dict into the vignette to
refresh the targets.

Provenance of each recipe (already nailed down -- do NOT re-derive):
  - GBSG2 CoxNet 0.688 and RSF 0.692 come from the scikit-survival
    "Evaluating Survival Models" user guide. We reproduce the printed values with
    one stratified 75/25 split (sklearn's train_test_split default) at
    random_state=1:
      CoxNet recipe = CoxnetSurvivalAnalysis(l1_ratio=0.99, fit_baseline_model=True)
      RSF recipe    = RandomSurvivalForest(max_depth=2, random_state=1)
    The 0.688 point sits at the 80th percentile of that protocol's own seed
    distribution (protocol mean ~ 0.669). RSF 0.692 is typical (~53rd pct).
  - Veterans FSSVM 0.720 comes from the scikit-survival "Survival Support Vector
    Machine" user guide: ShuffleSplit(n_splits=100, test_size=0.5) + a grid search
    over alpha (13 alphas x 100 splits = 1300 fits). That value is already a
    100-split mean, so we replicate the recipe rather than re-average it across our
    seed list.

At 30 seeds, mlsurv's CoxNet mean matches the scikit-survival recipe mean; an
apparent gap seen at small N was a seed-sampling artifact, not a model defect.

Run:
  python scripts/benchmarks/vignette_03_sksurv_targets.py
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sksurv.column import encode_categorical
from sksurv.datasets import load_gbsg2, load_veterans_lung_cancer
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.preprocessing import OneHotEncoder
from sksurv.svm import FastSurvivalSVM

# Pinned seed list. Use >= 30: the per-recipe mean's third decimal is
# seed-list-sensitive at small N (the CoxNet mean moved ~0.65 -> ~0.67 going
# from 15 to 30 seeds), so a short list gives an unstable target.
SEEDS = range(30)

# The exact single-seed values printed in the scikit-survival docs. Used only as
# trust anchors in replicate_anchors() -- they are NOT the vignette targets.
PUBLISHED = {"GBSG2/coxnet": 0.688, "GBSG2/rsf": 0.692, "Veterans/fssvm": 0.720}


# --------------------------------------------------------------------------- #
# scikit-survival recipes (each model's own documented protocol)              #
# --------------------------------------------------------------------------- #
def gbsg2_onehot():
    """Load GBSG2 and one-hot encode, as the Evaluating-Survival-Models guide does."""
    X, y = load_gbsg2()
    return OneHotEncoder().fit_transform(X), y


def sk_gbsg2_coxnet(seed, Xt, y):
    """CoxNet on one stratified 75/25 split. seed=1 reproduces the doc's 0.688."""
    Xtr, Xte, ytr, yte = train_test_split(Xt, y, stratify=y["cens"], random_state=seed)
    model = CoxnetSurvivalAnalysis(l1_ratio=0.99, fit_baseline_model=True).fit(Xtr, ytr)
    return model.score(Xte, yte)


def sk_gbsg2_rsf(seed, Xt, y):
    """RSF on one stratified 75/25 split. seed=1 reproduces the doc's 0.692.

    Only the split varies with `seed`; the forest RNG is pinned at 1 to match the
    doc, so the multiseed spread reflects split variance alone.
    """
    Xtr, Xte, ytr, yte = train_test_split(Xt, y, stratify=y["cens"], random_state=seed)
    model = RandomSurvivalForest(max_depth=2, random_state=1).fit(Xtr, ytr)
    return model.score(Xte, yte)


def sk_veterans_fssvm():
    """FSSVM grid search over 100 random 50/50 splits; reproduces the doc's ~0.720.

    The doc value is already a 100-split mean, so this returns one number, not a
    per-seed array. error_score='raise' makes a fitting failure crash loudly
    rather than become a silent NaN in best_score_.
    """
    X, y = load_veterans_lung_cancer()
    Xt = encode_categorical(X)

    def score(model, X_, y_):
        pred = model.predict(X_)
        return concordance_index_censored(y_["Status"], y_["Survival_in_days"], pred)[0]

    gcv = GridSearchCV(
        FastSurvivalSVM(max_iter=1000, tol=1e-5, random_state=0),
        {"alpha": 2.0 ** np.arange(-12, 13, 2)},
        scoring=score,
        cv=ShuffleSplit(n_splits=100, test_size=0.5, random_state=0),
        refit=False,
        n_jobs=4,
        error_score="raise",
    ).fit(Xt, y)
    return gcv.best_score_


# --------------------------------------------------------------------------- #
# Step 1: replicate the published anchors exactly                             #
# --------------------------------------------------------------------------- #
def replicate_anchors():
    """Reproduce the exact published doc values; return them keyed by model."""
    Xt, y = gbsg2_onehot()
    anchors = {
        "GBSG2/coxnet": sk_gbsg2_coxnet(1, Xt, y),
        "GBSG2/rsf": sk_gbsg2_rsf(1, Xt, y),
        "Veterans/fssvm": sk_veterans_fssvm(),
    }
    print("Step 1 -- replicate published scikit-survival anchors:")
    for key, value in anchors.items():
        print(f"  {key:16s} {value:.3f}   (published {PUBLISHED[key]:.3f})")
    return anchors


# --------------------------------------------------------------------------- #
# Step 2: average each recipe over the seed list -> the vignette targets       #
# --------------------------------------------------------------------------- #
def multiseed_targets(seeds=SEEDS):
    """Return {model_key: (mean, std)} -- the targets the vignette displays.

    FSSVM's published value is already a 100-split mean, so its std is None
    (it is replicated, not re-averaged over our seed list).
    """
    Xt, y = gbsg2_onehot()
    cox = np.array([sk_gbsg2_coxnet(s, Xt, y) for s in seeds])
    rsf = np.array([sk_gbsg2_rsf(s, Xt, y) for s in seeds])
    # A non-finite score would silently poison the mean; refuse to average it.
    if not (np.isfinite(cox).all() and np.isfinite(rsf).all()):
        raise RuntimeError("A GBSG2 seed produced a non-finite score; "
                           "refusing to build a target mean from it.")
    fssvm = sk_veterans_fssvm()
    return {
        "GBSG2/coxnet": (float(cox.mean()), float(cox.std())),
        "GBSG2/rsf": (float(rsf.mean()), float(rsf.std())),
        "Veterans/fssvm": (float(fssvm), None),
    }


def _print_targets_dict(targets):
    """Print a ready-to-paste SKSURV_TARGETS dict for the vignette."""
    cox_m, cox_s = targets["GBSG2/coxnet"]
    rsf_m, rsf_s = targets["GBSG2/rsf"]
    fssvm_m, _ = targets["Veterans/fssvm"]
    print("\nStep 2 -- paste this into the vignette (means over the seed list):\n")
    print("SKSURV_TARGETS = {")
    print(f"    'GBSG2':    {{'coxnet': ({cox_m:.3f}, {cox_s:.3f}),")
    print(f"                 'rsf':    ({rsf_m:.3f}, {rsf_s:.3f})}},")
    print(f"    'Veterans': {{'fssvm':  ({fssvm_m:.3f}, None)}},")
    print("}")


if __name__ == "__main__":
    import warnings

    # Silence the noisy (and expected) sksurv/pandas warnings for a clean
    # console run. Scoped to script execution so importing this module for the
    # guard test does not mutate global warning state.
    warnings.filterwarnings("ignore")

    replicate_anchors()
    targets = multiseed_targets()
    _print_targets_dict(targets)
