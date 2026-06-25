"""Benchmark post-hoc calibrators for S(t|x) across datasets x models.

Evaluates whether held-out recalibration improves calibration (slope -> 1,
ICI down) WITHOUT worsening the integrated Brier score, and which calibrator
suits each model. All calibrators are fit on the SAME 5-fold out-of-fold (OOF)
predictions of the training cohort and applied to a held-out test set, so the
comparison is leakage-free and apples-to-apples.

Arms
----
  raw            model's native S(t|x)
  lowdof         2-param cloglog (van Houwelingen) recal of S(t|x), IPCW-weighted
  isotonic       per-horizon isotonic on S(t|x)
  breslow        Breslow baseline refit on held-out risk scores, slope fixed at 1
                 (score-based models only: coxph/coxnet/gbsa/xgbcox/fssvm/deepsurv)
  breslow_slope  Breslow baseline + Cox slope refit on held-out scores
                 (full van Houwelingen; score-based models only)

fssvm trains raw (the model has no built-in calibration), so its calibrators are
compared from the RAW SVM score.

Settings
--------
7 datasets (gbsg2, veterans, whas500, flchain, metabric, rotterdam, tcga_brca;
the large ones subsampled to 800 rows for speed), all 10 models, 5-fold OOF.
The fold count matters: 3-fold under-fits the slope refit; 5-fold is needed for
the breslow_slope full van Houwelingen slope refit to settle.

Results (median |slope-1| across the 7 datasets, raw -> best calibrator)
------------------------------------------------------------------------
Only 3 of 10 models benefit from calibration; the other 7 are already adequate:

  fssvm    1.36 -> 0.21  via breslow_slope  (full van Houwelingen slope refit)
  rsf      0.19 -> 0.07  via lowdof
  deephit  2.37 -> 0.69  via isotonic       (low-DoF too weak for its severe,
                                             non-parametric miscalibration)
  coxph, coxnet, gbsa, xgbcox, deepsurv, weibull_aft, xgbaft -> raw (no gain)

Notes
  - For score->baseline models the SLOPE refit is essential: breslow (level
    only) ~= raw; only breslow_slope fixes the score scale.
  - Calibration is monotone, so the C-index is preserved. Rank-safety check
    (max C-index drift vs raw): breslow/breslow_slope 0.000, lowdof <=0.004,
    but isotonic up to 0.164 on gbsa/coxnet -> isotonic is NOT a safe general
    default (it collapses predictions onto plateaus, creating ties). On deephit
    its drift is only 0.014, so it is acceptable there.

These results motivate per-model default calibrators: fssvm->breslow_slope,
rsf->lowdof, deephit->isotonic, everything else raw.

Usage
-----
    conda run -n mlsurv python scripts/benchmark_calibrators.py smoke out.csv
    conda run -n mlsurv python scripts/benchmark_calibrators.py full  out.csv
"""
import os
import sys
import warnings
import traceback

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.simplefilter("ignore")

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sksurv.datasets import (load_gbsg2, load_veterans_lung_cancer,
                             load_whas500, load_flchain)
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.linear_model.coxph import BreslowEstimator
from sksurv.metrics import concordance_index_censored, integrated_brier_score

from mlsurv import SurvivalLearner
from mlsurv.utils import load_survival_data
from mlsurv.evaluation.metrics import (
    predict_survival_functions, surv_funcs_to_probs, calibration_slope_intercept)
from mlsurv.evaluation._recalibration import (
    compute_calibration_indices, _fit_censoring_distribution,
    _ipcw_calibration_weights, _fit_isotonic_recalibrators,
    _apply_recalibrators_to_array)
from mlsurv.models._overflow import breslow_call_with_clamp

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "examples", "data")
MAX_N = 800     # subsample big datasets for speed
EPS = 1e-6


# ---------- dataset loaders -> (X numeric, y structured ['event','time']) ----------
def _norm_y(y_struct):
    names = list(y_struct.dtype.names)
    # event = the boolean field, time = the float field (order-agnostic)
    ev_name = next((n for n in names if y_struct.dtype[n].kind == "b"), names[0])
    tm_name = next(n for n in names if n != ev_name)
    ev = np.asarray(y_struct[ev_name]).astype(bool)
    tm = np.asarray(y_struct[tm_name]).astype(float)
    return np.array(list(zip(ev, tm)), dtype=[("event", bool), ("time", float)])


def _encode(X):
    X = X.copy()
    for c in X.select_dtypes(include=["category", "object"]).columns:
        X[c] = X[c].astype("category").cat.codes.astype(float)
    return X.astype(float)


def _sub(X, y, seed=0):
    if len(X) <= MAX_N:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), MAX_N, replace=False)
    return X.iloc[idx].reset_index(drop=True), y[idx]


def load_gbsg2_ds():
    X, y = load_gbsg2()
    return _encode(X), _norm_y(y)


def load_veterans_ds():
    X, y = load_veterans_lung_cancer()
    return _encode(X), _norm_y(y)


def load_whas500_ds():
    X, y = load_whas500()
    return _encode(X), _norm_y(y)


def load_flchain_ds():
    X, y = load_flchain()
    return _sub(_encode(X), _norm_y(y), seed=1)


def load_csv_ds(fname, time_col, event_col, drop=None):
    X, y = load_survival_data(os.path.join(DATA_DIR, fname),
                              time_col=time_col, event_col=event_col)
    if drop:
        X = X.drop(columns=[c for c in drop if c in X.columns])
    return _sub(_encode(X), _norm_y(y), seed=2)


DATASETS = {
    "gbsg2":    load_gbsg2_ds,
    "veterans": load_veterans_ds,
    "whas500":  load_whas500_ds,
    "flchain":  load_flchain_ds,
    "metabric": lambda: load_csv_ds("metabric.csv", "duration", "event"),
    "rotterdam": lambda: load_csv_ds("rotterdam.csv", "rtime", "recur",
                                     drop=["rownames", "pid", "dtime", "death"]),
    "tcga_brca": lambda: load_csv_ds("tcga_brca.csv", "time", "event"),
}

MODELS = ["coxph", "coxnet", "rsf", "gbsa", "fssvm", "xgbcox", "xgbaft",
          "weibull_aft", "deepsurv", "deephit"]

# Models with a single linear-predictor risk score -> Breslow baseline applies.
SCORE_MODELS = {"coxph", "coxnet", "gbsa", "xgbcox", "fssvm", "deepsurv"}


def collect_oof(pipe, Xtr, ytr, skf, horizons, want_scores):
    """One refit per fold; return OOF S(t|x) and (optionally) OOF risk scores."""
    n = len(Xtr)
    oof_S = np.full((n, len(horizons)), np.nan)
    oof_sc = np.full(n, np.nan) if want_scores else None
    strat = np.asarray(ytr["event"]).astype(int)
    for tr, va in skf.split(Xtr, strat):
        fp = clone(pipe)
        fp.fit(Xtr.iloc[tr], ytr[tr])
        sf = predict_survival_functions(fp, Xtr.iloc[va], horizons)
        oof_S[va, :] = np.asarray(surv_funcs_to_probs(sf, horizons), float)
        if want_scores:
            oof_sc[va] = np.asarray(fp.predict(Xtr.iloc[va]), float)
    return oof_S, oof_sc


def breslow_probs(scores_fit, y_fit, scores_test, horizons, model_name, beta=1.0):
    """Fit a Breslow baseline on scores_fit (scaled by beta), predict S at horizons."""
    bre = breslow_call_with_clamp(beta * scores_fit, BreslowEstimator().fit,
                                  y_fit["event"], y_fit["time"],
                                  model_name=model_name)
    sfns = breslow_call_with_clamp(beta * scores_test, bre.get_survival_function,
                                   model_name=model_name)
    return np.array([[float(fn(h)) for h in horizons] for fn in sfns])


# ---------- low-DoF (a,b) cloglog recalibration ----------
def fit_lowdof(oof_S, horizons, y_train):
    """Fit S' = exp(-exp(a + b*log(-log S))) on pooled OOF, IPCW-weighted."""
    t = np.asarray(y_train["time"]); e = np.asarray(y_train["event"]).astype(bool)
    cde = _fit_censoring_distribution(y_train)
    logH, fail, wts = [], [], []
    for j, h in enumerate(horizons):
        S = np.clip(oof_S[:, j], EPS, 1 - EPS)
        ok = np.isfinite(oof_S[:, j])
        w = _ipcw_calibration_weights(t, e, float(h), cde)
        m = ok & (w > 0)
        logH.append(np.log(-np.log(S[m])))
        fail.append((t[m] <= float(h)).astype(float))   # I[T<=h]
        wts.append(w[m])
    lh = np.concatenate(logH); fl = np.concatenate(fail); w = np.concatenate(wts)

    def nll(ab):
        a, b = ab
        eta = a + b * lh
        mu = np.clip(1 - np.exp(-np.exp(np.clip(eta, -30, 30))), EPS, 1 - EPS)
        return -np.sum(w * (fl * np.log(mu) + (1 - fl) * np.log(1 - mu)))

    res = minimize(nll, x0=[0.0, 1.0], method="L-BFGS-B",
                   bounds=[(-10, 10), (0.0, 10)])
    return float(res.x[0]), float(res.x[1])


def apply_lowdof(ab, S2d, horizons):
    a, b = ab
    S = np.clip(S2d, EPS, 1 - EPS)
    eta = a + b * np.log(-np.log(S))
    out = np.exp(-np.exp(np.clip(eta, -30, 30)))
    out = np.clip(out, 0.0, 1.0)
    order = np.argsort(horizons)
    so = out[:, order]; np.minimum.accumulate(so, axis=1, out=so)
    return so[:, np.argsort(order)]


# ---------- metrics ----------
def metric_row(probs, horizons, y_train, y_test):
    """slope_abs & ICI at central horizon; IBS over valid grid; C at central h."""
    jc = len(horizons) // 2
    hc = float(horizons[jc])
    slope, _ = calibration_slope_intercept(y_test, probs[:, jc], hc)
    ici = compute_calibration_indices(y_test, probs[:, jc], hc).ici
    # IBS over horizons strictly inside test follow-up
    tmax = float(np.asarray(y_test["time"]).max())
    tmin = float(np.asarray(y_test["time"]).min())
    grid_mask = (horizons > tmin) & (horizons < tmax)
    if grid_mask.sum() >= 2:
        g = horizons[grid_mask]
        est = probs[:, grid_mask]
        try:
            ibs = float(integrated_brier_score(y_train, y_test, est, g))
        except Exception:
            ibs = np.nan
    else:
        ibs = np.nan
    risk = 1.0 - probs[:, jc]
    c = concordance_index_censored(np.asarray(y_test["event"]).astype(bool),
                                   np.asarray(y_test["time"]), risk)[0]
    sa = abs(slope - 1.0) if np.isfinite(slope) else np.nan
    return dict(slope_abs=sa, ici=float(ici), ibs=ibs, cindex=float(c))


# ---------- per-cell benchmark ----------
def run_cell(ds_name, X, y, model_name, model_params=None):
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, stratify="event",
                  imputer="knn", scaler="robust", feature_selector="topk",
                  random_state=42, n_jobs=1)
    if model_params:
        learner.train([model_name], model_params={model_name: model_params})
    else:
        learner.train([model_name])
    st = learner._state
    res = st.models[model_name]
    pipe = res.model
    Xtr, ytr = st.X_train, st.y_train
    Xte, yte = st.X_test, st.y_test

    # horizons: deciles of training event times, clipped to test range
    et = np.asarray(ytr["time"])[np.asarray(ytr["event"]).astype(bool)]
    if et.size < 5:
        et = np.asarray(ytr["time"])
    horizons = np.unique(np.quantile(et, np.linspace(0.1, 0.9, 7)))
    horizons = horizons[horizons > 0]

    # OOF S(t|x) (+ risk scores for score models) on the training cohort.
    # 5-fold is load-bearing: fewer folds under-fit the breslow_slope full
    # van Houwelingen slope refit.
    is_score = model_name in SCORE_MODELS
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_S, oof_sc = collect_oof(pipe, Xtr, ytr, skf, horizons, want_scores=is_score)
    # test raw probs
    sf = predict_survival_functions(pipe, Xte, horizons)
    raw = np.asarray(surv_funcs_to_probs(sf, horizons), dtype=float)

    rows = []
    rows.append(dict(dataset=ds_name, model=model_name, arm="raw",
                     **metric_row(raw, horizons, ytr, yte)))
    # low-DoF cloglog (model-agnostic)
    ab = fit_lowdof(oof_S, horizons, ytr)
    low = apply_lowdof(ab, raw, horizons)
    rows.append(dict(dataset=ds_name, model=model_name, arm="lowdof",
                     a=ab[0], b=ab[1], **metric_row(low, horizons, ytr, yte)))
    # isotonic (legacy method)
    iso_map, _ = _fit_isotonic_recalibrators(oof_S, horizons, ytr)
    if iso_map:
        iso = _apply_recalibrators_to_array(iso_map, raw, horizons)
        rows.append(dict(dataset=ds_name, model=model_name, arm="isotonic",
                         **metric_row(iso, horizons, ytr, yte)))
    # Breslow baseline refit on held-out scores (score models only)
    if is_score:
        fin = np.isfinite(oof_sc)
        sc_test = np.asarray(pipe.predict(Xte), float)
        try:
            bp = breslow_probs(oof_sc[fin], ytr[fin], sc_test, horizons, model_name)
            rows.append(dict(dataset=ds_name, model=model_name, arm="breslow",
                             **metric_row(bp, horizons, ytr, yte)))
        except Exception as ex:
            print(f"    [breslow fail] {model_name}: {type(ex).__name__}: {ex}")
        try:  # breslow + Cox slope (full van Houwelingen) = the internal feature
            cox = CoxPHSurvivalAnalysis(alpha=1e-6).fit(oof_sc[fin].reshape(-1, 1),
                                                        ytr[fin])
            beta = float(cox.coef_[0])
            bsp = breslow_probs(oof_sc[fin], ytr[fin], sc_test, horizons,
                                model_name, beta=beta)
            rows.append(dict(dataset=ds_name, model=model_name, arm="breslow_slope",
                             beta=beta, **metric_row(bsp, horizons, ytr, yte)))
        except Exception as ex:
            print(f"    [breslow_slope fail] {model_name}: {type(ex).__name__}: {ex}")
    return rows


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    out = sys.argv[2]
    if mode == "smoke":
        ds_list = ["veterans"]; model_list = ["coxph", "fssvm", "deephit"]
    else:
        ds_list = list(DATASETS); model_list = MODELS

    all_rows = []
    for ds_name in ds_list:
        try:
            X, y = DATASETS[ds_name]()
        except Exception as ex:
            print(f"[SKIP dataset] {ds_name}: {type(ex).__name__}: {ex}")
            continue
        print(f"\n### {ds_name}  n={len(X)} feat={X.shape[1]} "
              f"events={int(np.asarray(y['event']).sum())}")
        for model_name in model_list:
            try:
                rows = run_cell(ds_name, X, y, model_name, model_params=None)
                all_rows.extend(rows)
                r = {x["arm"]: x for x in rows}
                msg = " | ".join(
                    f"{a}: |slope-1|={r[a]['slope_abs']:.2f} ICI={r[a]['ici']:.3f} "
                    f"IBS={r[a]['ibs']:.3f}" for a in r)
                print(f"  {model_name:12} {msg}")
            except Exception as ex:
                print(f"  {model_name:12} FAILED {type(ex).__name__}: {ex}")
                if mode == "smoke":
                    traceback.print_exc()
    df = pd.DataFrame(all_rows)
    df.to_csv(out, index=False)
    print(f"\nwrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
