"""Does hyperparameter tuning change the Vignette 07 results?

Refits the six case-study models with default settings (train(tune=False)),
using the same data, split, preprocessing and seed as Vignette 07, and compares
Harrell's C-index with the tuned models from a saved Vignette 07 run:

- Chowell test set: tuned - default, with a paired bootstrap CI and p-value.
- Each external cohort: tuned - default (no bootstrap).

Inputs (from a Vignette 07 run):
    examples/output/vig7/results_{OS,PFS}.xlsx
    examples/data/loris/<cohort>.csv
Outputs:
    examples/output/vig7/tuned_vs_default_<OUTCOME>_test.csv
    examples/output/vig7/tuned_vs_default_<OUTCOME>_external.csv

usage (from the repo root):
    python scripts/vignette_07_tuned_vs_default.py OS|PFS
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

from mlsurv import SurvivalLearner
from mlsurv.preprocessing import detect_categorical_groups
from mlsurv.utils import to_structured_array

warnings.filterwarnings('ignore')
OUTCOME = sys.argv[1]
TIME, EVENT = f'{OUTCOME}_Months', f'{OUTCOME}_Event'
EX = Path(__file__).resolve().parents[1] / 'examples'
DATA = EX / 'data' / 'loris'
OUT = EX / 'output' / 'vig7'
XL = pd.ExcelFile(OUT / f'results_{OUTCOME}.xlsx')
MODELS = ['coxph', 'coxnet', 'rsf', 'xgbcox', 'xgbaft', 'fssvm']
CLINICAL = ['TMB', 'Age', 'Systemic_therapy_history', 'Albumin', 'NLR']
COHORTS = ['MSK1', 'MSK2', 'Shim_NSCLC', 'Ravi_NSCLC', 'Pradat_panCancer', 'Kato_panCancer']
COHORTS += ['Vanguri_NSCLC'] if OUTCOME == 'PFS' else ['MSK_nonICB']
N_BOOT, SEED = 1000, 42


def prepare(df, cols=None):  # as in Vignette 07
    X = pd.concat([df[CLINICAL], pd.get_dummies(df['CancerType'], prefix='CancerType')], axis=1)
    return X.reindex(columns=cols, fill_value=0) if cols is not None else X


tr = pd.read_csv(DATA / 'Chowell_train.csv')
te = pd.read_csv(DATA / 'Chowell_test.csv')
X_train, y_train = prepare(tr), to_structured_array(tr[TIME], tr[EVENT])
X_test, y_test = prepare(te, X_train.columns), to_structured_array(te[TIME], te[EVENT])

L = SurvivalLearner(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
L.setup(categorical_groups=detect_categorical_groups(X_train),
        subpopulations_train={'Cancer Type': tr['CancerType'].values},
        subpopulations_test={'Cancer Type': te['CancerType'].values},
        stratify_subpopulations=True, cv_folds=5, random_state=42, n_jobs=1,
        verbose=False)
L.train(models=MODELS, tune=False, verbose=False)
ev = L.evaluate(models=MODELS, verbose=False, show_progress=False)

# Risk scores on the test set: default (just fitted) and tuned (final run).
dfs = ev.to_dataframes()
pred_key = [k for k in dfs if k.endswith('predictions')][0]
default_pred = dfs[pred_key]
tuned_pred = XL.parse('eval_predictions')


def harrell(event, time, risk):
    return concordance_index_censored(event.astype(bool), time, risk)[0]


event, time = y_test['event'], y_test['time']
rng = np.random.default_rng(SEED)
boot = [rng.integers(0, len(time), len(time)) for _ in range(N_BOOT)]
rows = []
for m in MODELS:
    d = default_pred[default_pred.model_type == m].sort_values('index')['risk_score'].to_numpy()
    t = tuned_pred[tuned_pred.model_type == m].sort_values('index')['risk_score'].to_numpy()
    cd, ct = harrell(event, time, d), harrell(event, time, t)
    diffs = np.array([harrell(event[i], time[i], t[i]) - harrell(event[i], time[i], d[i])
                      for i in boot])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    rows.append(dict(model=m, default=cd, tuned=ct, gain=ct - cd,
                     ci_lo=np.percentile(diffs, 2.5), ci_hi=np.percentile(diffs, 97.5), p=p))
res = pd.DataFrame(rows).set_index('model')
print(f'\n=== {OUTCOME}: Chowell test, Harrell C (tuned - default; paired bootstrap, {N_BOOT})')
print(res.round(3).to_string())
print('check: tuned C matches final run:',
      np.allclose(res.tuned, XL.parse('eval_metrics').set_index('model_type')
                  .loc[MODELS, 'c_index_test'], atol=1e-6))

# External cohorts: validate the default models the way Vignette 07 does.
ext = {}
for c in COHORTS:
    df = pd.read_csv(DATA / f'{c}.csv').dropna(subset=[TIME, EVENT])
    if len(df) < 10:  # e.g. Pradat has no PFS; the vignette skips it too
        continue
    vr = L.validate(prepare(df, X_train.columns), to_structured_array(df[TIME], df[EVENT]),
                    models=MODELS, name=c, verbose=False, bootstrap=False, show_progress=False)
    dflt = vr.concordance.set_index('model_type')['c_index_test']
    sheet = [s for s in XL.sheet_names if s.startswith(f'val_{c}'[:20]) and 'metri' in s][0]
    tuned = XL.parse(sheet).set_index('model_type')['c_index_test']
    ext[c] = (tuned.loc[MODELS] - dflt.loc[MODELS])
gain = pd.DataFrame(ext)
print(f'\n=== {OUTCOME}: external cohorts, Harrell C gain from tuning (tuned - default)')
print(gain.round(3).to_string())
print('mean gain per cohort:', gain.mean().round(3).to_dict())

res.to_csv(OUT / f'tuned_vs_default_{OUTCOME}_test.csv')
gain.to_csv(OUT / f'tuned_vs_default_{OUTCOME}_external.csv')
print(f'\nwrote tuned_vs_default_{OUTCOME}_test.csv and _external.csv to {OUT}')
