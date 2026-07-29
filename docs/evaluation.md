# Model Development & Evaluation

This guide covers two workflow phases:

- **Model Development** — workflow approaches, training, hyperparameter tuning
- **Model Evaluation** — discrimination and calibration metrics, uncertainty quantification, pairwise comparison, risk stratification, subpopulation analysis, prospective patient prediction

# Model Development

## Workflow Approaches

### All-in-One: `run()`

`run()` executes the full pipeline in a single call — training, hyperparameter tuning, test-set evaluation, and bootstrap confidence intervals are all enabled by default:

```python
model_results = learner.run(['coxph', 'rsf', 'xgbcox'])
```

Returns a dictionary mapping model names to `ModelResult` objects. Each result contains the trained model, CV scores, evaluation metrics, and bootstrap confidence intervals. The same `ModelResult` objects are always available as `learner.models`, so you do not need to keep the return value: `learner.models['coxph']` gives you the same object at any time, including after `load_learner(...)`.

### Granular: Step-by-Step

For fine-grained control, use the individual methods:

```python
learner.train(['coxph', 'rsf'])       # Train with default hyperparameters
learner.tune('rsf', n_trials=50)      # Tune one model
learner.evaluate()                     # Evaluate all trained models
learner.bootstrap(n_bootstrap=200)     # Bootstrap CIs
```

This approach supports evaluating a subset of trained models or adding models to an existing comparison without re-running the entire pipeline.

### Fair Comparison

By default, all models share the same preprocessing configuration, CV folds, and evaluation metrics, enabling fair head-to-head comparison across architectures.

### Leakage-Free Cross-Validation

The full preprocessing pipeline (imputation → scaling → feature selection) is **fitted independently within each CV fold** so that CV scores reflect out-of-sample performance of the complete pipeline. Fitting preprocessing once on the training set and reusing it across folds would leak information from validation folds into the transforms, inflating performance estimates. See [Pipeline Configuration — Preprocessing Pipeline](pipeline.md#preprocessing-pipeline).

## Hyperparameter Tuning

Hyperparameter tuning uses [Optuna](https://optuna.org/)'s Tree-structured Parzen Estimator (TPE) for Bayesian optimization<sup>[1](#references)</sup>, with safeguards tailored to survival analysis.

### Interpreting Tuned-Model CV Scores

A tuned model's reported cross-validation score is selection-optimistic: it is the objective that tuning maximized over many trials, and by default the tuning folds and the reporting folds share the same split, so the score tends to read higher than a true out-of-sample estimate. This is not data leakage (the held-out test set is never touched during tuning, and selecting the best model on CV is sound); the held-out test metric, or nested cross-validation (see [Vignette 05](../examples/vignette_05_advanced_cv_tuning.ipynb)), is the honest out-of-sample estimate.

### Tuning Overwrites In Place

`tune()` **overwrites the model in place**: the tuned variant replaces the entry under its plain algorithm name (e.g. `'rsf'`), and any stale calibrator and importance-test results for that key are dropped — the tuned `S(t|x)` differs, so the old calibrator would miscalibrate, and old p-values describe the old pipeline (re-train with `calibrate=...` / re-run `importance_test()` on the tuned model if needed).

To compare tuned against untuned, build a second learner with the same `setup(random_state=...)`. The identical split, folds, and preprocessing make the comparison apples-to-apples, and the leakage-safe question "did tuning help?" is answered on cross-validation (`cv_metric_mean` before vs after) rather than on the held-out test set:

```python
learner.tune('rsf', n_trials=50)
untuned_cv = base_learner.train('rsf').cv_metric_mean   # same setup(random_state=...)
tuned_cv = learner.models['rsf'].cv_metric_mean
```

Comparing tuned and untuned variants on the **test** set instead would be selecting a model with the test set — the exact leakage the framework otherwise guards against.

### EPV-Aware Search Space Narrowing

On small datasets, mlsurv automatically constrains the search space to prevent overfitting-prone configurations. The constraints apply per-key, so any parameter range you supply yourself always wins — only the parameters you leave unspecified are narrowed.

**Tier selection — ratio *and* absolute events.** The constraint tier (`low` / `mid` / `high`) is chosen from both the events-per-variable ratio (EPV = events / features) and the absolute number of events. A high ratio reached with a tiny feature count *and* few absolute events is still unstable, so the events-per-variable ratio alone is an insufficient sample-size criterion<sup>[Riley 2019; van Smeden 2019]</sup>. A permissive tier therefore requires both a sufficient ratio and enough absolute events; otherwise the tier is demoted so constraints become more aggressive:

| Tier | Requires | Effect |
|------|----------|--------|
| `high` | EPV ≥ 20 **and** ≥ 100 events | defaults preserved (no narrowing) |
| `mid` | EPV ≥ 10 **and** ≥ 50 events | moderate narrowing |
| `low` | otherwise | aggressive narrowing |

The 50 / 100 event floors are pragmatic conventions aligned with the rare-event instability shown for survival models<sup>[Ambler 2012; Pavlou 2016]</sup>; the literature deliberately avoids a single universal cutoff.

**Two opposite mechanisms, by model family:**

- *Capacity-capping models* (RSF, GBSA, XGBCox, DeepSurv, DeepHit): tree depth, ensemble size, and neural-net width / layer count are capped, and dropout floors are raised, to limit capacity at low EPV.
- *Shrinkage-forcing penalized models* (CoxNet, FSSVM): the search is steered toward **stronger** shrinkage — the opposite direction — because the cure for few events is more regularization, not less capacity<sup>[Ambler 2012; Pavlou 2016]</sup>. CoxNet's `l1_ratio` is capped toward the L2 (ridge) side for stability under correlated / high-dimensional predictors<sup>[Zou & Hastie 2005]</sup>, and its penalty-path position (and FSSVM's `alpha`) is kept away from the weakly-penalized, near-unpenalized end.

**References:** Riley et al. 2019 (*Stat Med* 38:1276, [doi:10.1002/sim.7992](https://doi.org/10.1002/sim.7992)); van Smeden et al. 2016 (*BMC Med Res Methodol* 16:163, [doi:10.1186/s12874-016-0267-3](https://doi.org/10.1186/s12874-016-0267-3)) & 2019 (*Stat Methods Med Res* 28:2455, [doi:10.1177/0962280218784726](https://doi.org/10.1177/0962280218784726)); Ambler et al. 2012 (*Stat Med* 31:1150, [doi:10.1002/sim.4371](https://doi.org/10.1002/sim.4371)); Pavlou et al. 2016 (*Stat Med* 35:1159, [doi:10.1002/sim.6782](https://doi.org/10.1002/sim.6782)); Zou & Hastie 2005 (*JRSS-B* 67:301, [doi:10.1111/j.1467-9868.2005.00503.x](https://doi.org/10.1111/j.1467-9868.2005.00503.x)); Pölsterl et al. 2015 (ECML-PKDD, [doi:10.1007/978-3-319-23525-7_15](https://doi.org/10.1007/978-3-319-23525-7_15)).

### Auto-Trial Sizing

The number of optimization trials is automatically sized by model family:

- Linear models: fewer trials (smaller search space)
- Tree ensembles: moderate trials
- Neural networks: more trials (larger, more sensitive search space)

Override with `n_trials` parameter: `learner.tune('rsf', n_trials=100)`.

### Convergence Detection

Tuning stops early when the scoring metric plateaus, preventing wasted computation on models that have already converged.

### Pipeline-Wide Tuning

The search space extends beyond model hyperparameters to include preprocessing parameters — the number of features retained by the selector, number of neighbors for KNN imputation, etc. — enabling joint optimization across the full pipeline. Use the structured `search_spaces` form, with one sub-dict per component:

```python
learner.tune('rsf', search_spaces={
    'model': {'n_estimators': (100, 500)},
    'feature_selector': {'k': (5, 20)},   # number of features retained
})
```

Each component's parameters are namespaced to that component, so a model and a preprocessing step can share a parameter name without collision (e.g. a model with its own `k` hyperparameter is tuned independently of the selector's `k`).

**`search_spaces` in `run()` and multi-model `train()`.** With more than one model, `search_spaces` accepts two forms:

```python
# Component-structured: applied to EVERY model in the run
learner.run(['coxph', 'coxnet'], search_spaces={'feature_selector': {'k': (5, 20)}})

# Per-model: each model gets its own space (top-level keys are model names)
learner.run(['coxph', 'rsf'], search_spaces={
    'coxph': {'feature_selector': {'k': (5, 20)}},
    'rsf':   {'model': {'n_estimators': (100, 500)}},
})
```

The top-level keys decide the form: component names (`model`, `feature_selector`, `imputer`, `scaler`) mean "apply this to all models," while model names mean "per-model." A per-model key that matches no model in the run is reported with a warning rather than dropped silently; mixing the two forms raises `ValueError`.

**Tuning the feature count without a `search_spaces` dict.** Setting the selector's `k` to `'tune'` opts into searching the number of retained features with no `search_spaces` argument at all: the default range `(5, n_features)` is applied automatically:

```python
from mlsurv.preprocessing import TopKCoxSelector

learner.setup(feature_selector=TopKCoxSelector(k='tune'))
learner.run(['coxph', 'coxnet'], tune=True)   # k searched for every model
```

Pass an explicit `search_spaces={'feature_selector': {'k': (lo, hi)}}` to override the default range.

### Custom Configuration

```python
# Custom search space
learner.tune('rsf', search_space={
    'n_estimators': (100, 500),
    'max_depth': (3, 10),
    'min_samples_leaf': (5, 50),
})

# Alternative scoring metric
learner.tune('rsf', scoring='integrated_brier')
```

See [Vignette 05 — Advanced CV & Tuning](../examples/vignette_05_advanced_cv_tuning.ipynb) for nested tuning with persisted Optuna studies.

### Tuning Custom Models

Custom models (`BaseSurvivalModel` subclasses passed as instances) are tunable in two ways:

```python
class MyModel(BaseSurvivalModel):
    name = 'my_model'
    def get_search_space(self):
        # Same format as built-in spaces: (min, max), (min, max, 'log'), [choices]
        return {'alpha': (1e-3, 1.0, 'log')}
    ...

# 1. Model-declared space
learner.train(MyModel(), tune=True, n_trials=30)

# 2. Per-call space (merges with, and overrides, the declared one)
learner.train(MyModel(), tune=True, n_trials=30,
              search_spaces={'model': {'alpha': (0.01, 0.5)}})

# 3. Preprocessing-only: tune the pipeline around a fixed custom model
#    (no model space needed; works even when get_search_space() is empty)
learner.train(MyModel(), tune=True, n_trials=20,
              search_spaces={'feature_selector': {'k': (2, 8)}})
```

If neither a declared `get_search_space()` nor a per-call `search_spaces` supplies any searchable parameter (a model space *or* a component space for the feature selector / imputer / scaler), tuning raises a `ValueError` naming the remedies.

**Limitations.** Custom-instance tuning is programmatic only: it is not available through the CLI (which accepts model *names*, not instances), and a custom instance cannot be passed in a multi-model `run([...])` list. The declared search space is used as-is, so the EPV-aware narrowing applied to built-in tree/neural models (see [EPV-Aware Search Space Narrowing](#epv-aware-search-space-narrowing)) does not constrain custom spaces; size your ranges to the dataset yourself on small cohorts.

### Study Persistence

Tuning studies can be persisted to SQLite or PostgreSQL for later resumption or visualization with [Optuna Dashboard](https://optuna-dashboard.readthedocs.io/en/stable/getting-started.html).

```python
learner.tune('rsf', storage='sqlite:///study.db', study_name='rsf_tuning')
```

#### Locating the default study files

If you don't pass an explicit `storage`, tuning writes SQLite study files under
`./optuna_studies/{experiment_name}/`, where `experiment_name` comes from
`setup(experiment_name=...)` (a timestamped name is auto-generated if you don't
set one). Two helpers let you find these files without knowing the layout:

```python
# Full path to the current experiment's study directory.
# Returns None until experiment_name is set — either via
# setup(experiment_name=...) or auto-generated by the first tuning run.
path = learner.experiment_path

# Study files for THIS learner's experiment (flat list):
# ['rsf_study.db', 'gbsa_study.db']
learner.list_studies()
```

For a global view across every experiment on disk (a dict keyed by experiment
name), call the module-level function instead:

```python
from mlsurv.tuning import list_studies
list_studies()
# {'gbsg2_analysis': ['rsf_study.db', 'gbsa_study.db'], 'flchain_cohort': [...]}
```

You can then point Optuna Dashboard at a study directly:

```bash
optuna-dashboard sqlite:///optuna_studies/<experiment_name>/<model>_study.db
```

### Visualization

```python
learner.plot_optuna('rsf')  # Optimization history, parameter importance
```

### Reproducibility

A single `random_state` parameter (set at `setup()`) seeds all underlying frameworks — scikit-learn<sup>[2](#references)</sup>, XGBoost<sup>[3](#references)</sup>, PyTorch<sup>[4](#references)</sup>, and Optuna<sup>[1](#references)</sup> — for end-to-end reproducibility.

**Bit-exact reproducibility requires `n_jobs=1`.** With `n_jobs>1`, results vary slightly between runs even with a fixed seed — tuned hyperparameters can differ, and CV/bootstrap scores vary at the ~1e-3 level. Two independent sources cause this:

1. **Optuna parallel trials**: When trials run concurrently, the TPE sampler's state updates depend on trial-completion order, which is timing-dependent. Optuna's own docs note this is expected behavior for `n_jobs>1`.
2. **Multi-threaded XGBoost/BLAS reductions**: Floating-point sum order changes with thread count, producing tiny numerical differences that compound through tuning.

This tradeoff is standard across ML libraries (scikit-learn, XGBoost, LightGBM, PyTorch all document similar behavior). For publication-grade reproducibility, set `n_jobs=1`; for everyday workflows, the parallel speedup is worth the small run-to-run variance.

# Model Evaluation

## Evaluation Metrics

The `evaluate()` method assesses models on two dimensions using the held-out test set: **discrimination** (how well the model ranks patients by risk) and **calibration** (how well predicted probabilities match observed outcomes).

### Discrimination Metrics

| Metric | Key | Description | Ref |
|--------|-----|-------------|-----|
| Harrell's C-index | `harrell` | Concordance probability between predicted risk and observed outcome. Default scoring metric. | <sup>[5](#references)</sup> |
| Uno's C-index (IPCW) | `ipcw` | Censoring-adjusted concordance using inverse probability of censoring weighting. More robust under heavy censoring. | <sup>[6](#references)</sup> |
| Antolini's C-index | `ctd` | Time-dependent concordance based on cumulative/dynamic AUC. | <sup>[7](#references)</sup> |
| AUC(t) | — | Time-dependent area under the ROC curve, evaluated at user-specified time points. | <sup>[8](#references)</sup> |

### Calibration Metrics

| Metric | Key | Description | Ref |
|--------|-----|-------------|-----|
| Calibration slope | `cal_slope` | Agreement between predicted and observed survival probabilities. Ideal value: 1.0. | <sup>[9](#references)</sup> |
| Calibration intercept | `cal_intercept` | Systematic bias in predicted probabilities. Ideal value: 0.0. | <sup>[9](#references)</sup> |
| GND test | `gnd` | Greenwood-Nam-D'Agostino goodness-of-fit test comparing predicted survival to Kaplan-Meier estimates. | <sup>[10](#references)</sup> |
| Brier score | — | Mean squared error between predicted survival probabilities and observed outcomes at a given time. | <sup>[11](#references)</sup> |
| Integrated Brier Score | `integrated_brier` | Brier score integrated over a dense grid of event times within the censoring-stable window `[t_min, tau]` (see [§Integrated Brier Score grid](#integrated-brier-score-grid)). | <sup>[11](#references)</sup> |

The scoring metric used for model selection is configured during `setup()` (default: Harrell's C-index). Custom metric functions can be supplied via `setup(scoring=custom_fn)`.

The calibration slope and intercept use a complete-case approximation: subjects censored before the evaluation horizon are dropped and the rest treated as a binary survived/event outcome, rather than inverse-probability-of-censoring (IPCW) weighting. They can therefore be biased under heavy or informative censoring. The GND and Hosmer-Lemeshow tests are censoring-aware (they compare against Kaplan-Meier estimates), so the headline calibration p-value accounts for censoring. An IPCW-weighted slope/intercept/ICI is a planned enhancement.

### Integrated Brier Score grid

The Integrated Brier Score (IBS)<sup>[11](#references)</sup> is the time-average of the Brier score `BS(t)` over an integration window. The underlying integrator (`sksurv.metrics.integrated_brier_score`) is a trapezoid sum over exactly the time points it is given, so the choice of grid affects the value. Following Graf et al., mlsurv integrates over **observed event times within a censoring-stable window** rather than an arbitrary percentile grid:

- **Grid points** — the unique **training** event times that fall within the window. Anchoring to *training* event times (the same source `auto_eval_times` and the IPCW horizon use) keeps the grid fixed across the test split, every bootstrap resample, and every subpopulation within one run, so IBS stays comparable across those cohorts and bootstrap CIs are not inflated by a grid that shifts per resample.
- **Window `[t_min, tau]`** — `t_min` is the lower edge of the (truncated) test follow-up; `tau` is the censoring-stable upper bound `min(train_t_max_event, p90(test_event_times))` — the same cap used for Uno's C-index (see below). This avoids the unstable right tail where the IPCW weights `1/G(t)^2` explode. The window is then clipped per cohort to that cohort's observed range, so a short bootstrap resample integrates only where it has support.
- **Cap** — the grid is capped at 100 points (subsampled to actual event times, evenly by rank, keeping the first and last in-window event time). IBS is a smooth integral, so 100 points is effectively converged; the cap bounds the per-model prediction cost in bootstrap.

This grid is computed once per `evaluate()` and once per `bootstrap()` (bootstrap iterations slice the precomputed matrix rather than re-predicting), so the only added cost is one `S(t|x)` prediction per model on the dense grid. The point Brier score (`BS(t)` at the median time), AUC(t), and calibration metrics are point-in-time quantities and continue to use the user-facing `eval_times`.

**Scoring caveat.** When IBS is used as the *tuning* objective (`setup(scoring='ibs')`), the cross-validation objective is integrated over the per-fold `eval_times`, not this dense grid — the tuning objective only needs a consistent relative ranking across hyperparameters and folds, and a dense prediction per Optuna trial × fold would be needlessly expensive. The IBS *reported* by `evaluate()`/`bootstrap()` always uses the dense grid above.

### Uno's IPCW C-index — estimation choices

Uno's C-index uses inverse probability of censoring weights `1/G(T)^2`, where `G(t)` is the Kaplan-Meier estimator of the censoring distribution. Two choices materially affect both the point estimate and its bootstrap uncertainty; both are exposed as kwargs on `evaluate()`, `bootstrap()`, and `validate()`.

**Evaluation horizon — `ipcw_tau`** (default: `'auto'`)

`G(t)` is unreliable in the right tail of follow-up where few subjects remain at risk; `1/G(t)^2` then explodes and a handful of late pairs dominate the estimate. The `ipcw_tau` kwarg truncates the evaluation horizon:

- `'auto'` (default) — `tau = min(train_t_max_event, p90(test_event_times))`. Trims the tail to where weights remain stable.
- `<float>` — user-specified upper bound.
- `None` — legacy unbounded behavior (sksurv default). Often produces wide bootstrap CIs on cohorts with extended follow-up.

When the resolved horizon, training support, or `min(G(t))` indicate instability, mlsurv emits a `MethodologicalWarning` and surfaces the violation in the report (Limitations section + IPCW row caveat).

**G_hat fitting cohort — `ipcw_g_hat_source`** (default: `'train'` for `evaluate`/`bootstrap`, `'test'` for `validate`)

Uno's weight is consistent for the population whose pairs are evaluated. For the in-distribution test set (split from the training population), training G_hat is correct under exchangeability. For external validation (`validate()`), exchangeability fails by construction — using training G_hat biases the point estimate when the validation cohort's censoring distribution differs from training's.

| Source | When to use |
|--------|-------------|
| `'train'` | Internal test split, or any cohort drawn from the same population as training. |
| `'test'` | External validation cohort with its own (potentially different) censoring distribution. |

`validate()` defaults to `'test'`; pass `ipcw_g_hat_source='train'` to opt back into the legacy behavior. Estimating G on the validation cohort introduces no leakage — `G(t)` uses censoring times only, not events or model predictions. References: <sup>[6](#references)</sup>, Heagerty PJ, Zheng Y. *Biometrics* 2005, Wolbers M et al. *Epidemiology* 2009.

### Training-range cap on IPCW metrics

<a id="training-range-cap"></a>

The Kaplan-Meier estimator `Ĝ(t)` of the censoring distribution is undefined past the training cohort's maximum follow-up time `train_max`. Any IPCW-weighted metric — Uno's C-index, time-dependent AUC(t), and the Integrated Brier Score — silently produces undefined or unstable values when evaluated past that horizon. mlsurv therefore administratively censors test/validation observations beyond `train_max` at `train_max` before computing IPCW metrics.

**Affected metrics**: Uno's C-index, AUC(t), Integrated Brier Score (IBS).

**Unaffected metrics**: Harrell's C-index, Antolini's time-dependent C, calibration plots/slope/intercept, and survival predictions — all of these use the full test/validation set untouched.

mlsurv emits a single `MethodologicalWarning` per cohort when truncation applies (at the top of `evaluate()`/`bootstrap()`/`validate()`), and the HTML report surfaces a footnote under the metrics table giving the cap and the number of truncated observations. The diagnostic is also persisted on `EvaluationResult` as `truncation_t_max` and `truncation_n_beyond`.

**Literature recommendation.** Uno (2011) and Pölsterl (*Evaluating Survival Models* 2019) recommend a more conservative cap at the 80th percentile of observed event times to avoid weight instability in the right tail of `Ĝ`. mlsurv's `train_max` cap is the looser bound that keeps `Ĝ(t)` defined; the `ipcw_tau` kwarg (see above) provides the conservative tail trim independently.

**Overrides.**

- Pass `eval_times=[...]` to `evaluate()` / `bootstrap()` to use a fixed grid for the point/curve metrics — AUC(t), point Brier, calibration (mlsurv still clips to the training range). IBS uses its own dense event-time grid (see the Integrated Brier Score section above).
- Pass `ipcw_tau=<float>` to bound Uno's evaluation horizon explicitly.
- Pass `ipcw_tau=None` for legacy unbounded sksurv behavior.

### Accessing Metrics

```python
result = learner.train('coxph')
learner.evaluate('coxph')

# Access evaluation results
print(result.evaluation.c_index_test)     # Harrell's C on test set
print(result.evaluation.c_index_train)    # Harrell's C on training set
print(result.evaluation.ibs_test)         # Integrated Brier Score on test set
print(result.evaluation.concordance)      # Full metrics DataFrame (one row per model)
```

#### Apparent train-set metrics (optimism diagnostic)

The `concordance` DataFrame also carries the four **apparent (resubstitution)
train-set** metrics — computed on the training set itself, the same scalar set
cross-validation reports per fold:

| Column | Metric |
|--------|--------|
| `c_index_train` | Harrell's C |
| `c_index_ipcw_train` | Uno's IPCW C |
| `c_index_td_train` | Antolini's time-dependent C |
| `integrated_brier_train` | Integrated Brier Score (IBS) |

The gap between apparent-train and CV/test is a direct **optimism / overfitting**
diagnostic. These columns are point values (no distribution). They are included
in `export_results()` and the opt-in `plot_metric(source="train")`, but are
deliberately kept out of printed output and the HTML report. Access them through
the `concordance` table:

```python
conc = result.evaluation.concordance
print(conc[['c_index_train', 'c_index_ipcw_train',
            'c_index_td_train', 'integrated_brier_train']])
```

Apparent-train survival probabilities are always raw (the calibrator is never
applied to them, even when one is attached). Time-dependent curves
(AUC(t)/Brier(t)) are intentionally **not** computed on the training set — those
stay exclusive to the held-out test set.

### Visualization

```python
# Model comparison — style selects layout (forest, bar, raincloud, boxplot)
learner.plot_metric(style='forest', metric='harrell')
learner.plot_metric(models=['coxph', 'rsf'], metric='ipcw', orientation='vertical')

# Apparent train-set metrics (opt-in optimism diagnostic — bare point bars).
# source='train' is never selected by 'auto'; request it explicitly.
learner.plot_metric(metric=['harrell', 'ipcw', 'ctd', 'integrated_brier'],
                    source='train')

# Multi-metric heatmap across all (or selected) models
learner.plot_metric_heatmap(metrics=['harrell', 'ipcw', 'integrated_brier'])

# Time-dependent metric curves (AUC(t), Brier(t))
learner.plot_metric_time(metric=['auc', 'brier'])

# Time-dependent ROC at one timepoint, or with a time-selection dropdown
learner.plot_roc(t=60, show_ci=True)
learner.plot_roc()                          # interactive dropdown across eval times

# Calibration curve at a specific time, with n_bins controlling bin count
learner.plot_calibration(eval_time=60, n_bins=10)

# Overlay predicted vs observed survival curves across the follow-up
learner.plot_calibration_overlay(models=['coxph', 'rsf'])
```

## Uncertainty Quantification

The `bootstrap()` method quantifies uncertainty for all point estimates by randomly resampling the test set with replacement:

```python
learner.bootstrap(n_bootstrap=200)

# Access bootstrap results
result = learner.models['coxph']
print(result.bootstrap.summary)
# DataFrame columns: [model_type, metric, point, mean, std, ci_lower, ci_upper]

# CIs for a specific metric live in the summary DataFrame
harrell_row = result.bootstrap.summary.query("metric == 'harrell'").iloc[0]
print((harrell_row['ci_lower'], harrell_row['ci_upper']))
```

Bootstrap confidence intervals are computed for all discrimination and calibration metrics.

## S(t|x) Calibration

`train()` / `run()` accept a `calibrate=` argument that fits a per-model calibrator (a `SurvivalCalibrator`) on the development cohort's held-out out-of-fold (OOF) predictions and attaches it to `result.calibrator`. That calibrator is the deployed model's `S(t|x)`: it is applied automatically wherever the model produces `S(t|x)` — `evaluate`, `bootstrap`, `predict`, `validate` (external cohort), and subpopulation metrics. The calibration plots (`plot_calibration`, `plot_calibration_overlay`) and the metric-time Brier(t) panel (`plot_metric_time(metric='brier')`) also show the calibrated curve by default, matching those metrics; pass `raw=True` to any of them for the uncalibrated view. Because the calibrator is monotone, the C-index is unchanged (and rank-based AUC(t) is unaffected, so it has no `raw` switch).

Each held-out fold is predicted by a clone of the model that never saw it during fit, so the (predicted, observed) pairs used to fit the calibrator are leakage-free.

### Methods

Three methods (`evaluation/_calibrators.py`):

- **`isotonic`** — per-horizon isotonic regression on `S(t|x)`; model-agnostic.
- **`lowdof`** — a 2-parameter cloglog map `S' = exp(-exp(a + b·log(-log S)))`; monotone, model-agnostic.
- **`breslow_slope`** — refits a Breslow baseline on the model's OOF risk score scaled by a Cox slope (van Houwelingen 2000). Score-based models only (a single linear predictor).

### `calibrate=` grammar

- **unset** — the per-model default registry (`fssvm`→`breslow_slope`, `deephit`→`isotonic`; everything else raw).
- **`False` / `None`** — all models raw.
- **a method string** — that method for every targeted model.
- **a `{model: method}` dict** — per-model control (missing keys stay raw).

`breslow_slope` on a non-score model raises `ValueError`. `rsf` is **not** calibrated by default — its `lowdof` gain is only a cross-dataset median and regresses on several datasets (gbsg2/veterans/whas500), so opt in explicitly with `calibrate={'rsf': 'lowdof'}` when it helps your cohort.

```python
learner.run(['coxph', 'rsf', 'fssvm'])                 # registry defaults: fssvm calibrated
learner.train(['rsf'], calibrate=False)                # uncalibrated rsf
learner.train(['coxph'], calibrate={'coxph': 'isotonic'})  # any model can elect a method
```

With a calibrator attached, `evaluate()` reports the calibrated (deployed) metrics and stores the raw pre-calibration slope on `result.precalibration_slope` so you can read the before/after.

### Calibration-drift warning

After `evaluate()`, mlsurv warns about calibration drift only when the test calibration **slope** falls outside `[0.5, 1.5]` (`|slope - 1| > 0.5`). The GND p-value is reported as context only, not a trigger. The warning's advice recommends re-training with `train(calibrate=...)`.

The names of trained models whose evaluation tripped this slope check are available on the read-only `learner.calibration_flagged_models` property:

```python
learner.calibration_flagged_models   # e.g. ['rsf', 'gbsa']; [] if nothing flagged
```

Calibration drift is a **per-model** diagnostic — each model has its own
`S(t|x)`, so "which models are flagged" is a clean model partition. Contrast
this with the **feature-level** setup diagnostics on
`learner.model_recommendations` (PH / linearity tests; see
[pipeline.md](pipeline.md#model-recommendations)): those name violating
*covariates*, not models, so there is no per-model PH/linearity flag — the
model verdict lives in the recommendation tiers.

## Model Comparison

### Pairwise Statistical Tests

The Kang nonparametric z-test<sup>[12](#references)</sup> assesses whether performance differences between models are statistically significant. It is computed automatically during `evaluate()` when two or more models are trained and cached for programmatic access; it is **not** printed to stdout (request the table on demand to keep `evaluate()` / `bootstrap()` / `run()` output focused on metrics).

```python
# Cached automatically when 2+ models are evaluated without bootstrap
learner.evaluate()
kang_df = learner.kang_comparison_result

# On-demand: paired bootstrap comparison (Holm-corrected)
boot_df = learner.bootstrap_result.compare(metric='harrell')
learner.bootstrap_result.print_comparison()  # also prints the table

# Module-level helpers (mlsurv.evaluation)
from mlsurv.evaluation import compare_bootstrap, compare_kang
```

The Kang comparison continues to appear as a paired-comparison section in the generated HTML report.

### Risk Stratification

Risk stratification evaluates clinical utility by dividing patients into quantile-based risk groups. mlsurv provides dedicated methods for the statistical tests and visualization:

```python
# Log-rank test across tertile-based risk groups (all trained models)
learner.logrank_test(n_groups=3)

# Restricted mean survival time comparison
learner.rmst_test(n_groups=3)

# A subset, or one model
learner.rmst_test(models=['coxph', 'rsf'], n_groups=3)
learner.logrank_test(models='coxph')
```

Both tests run for every trained model by default (`models=None`) and return
one stacked table with `model_type` as the first column — one block of rows
per model — so risk stratification can be compared across models directly.

- **Log-rank test**<sup>[13](#references)</sup>: Tests whether survival curves differ significantly between risk groups
- **Restricted mean survival time (RMST)**<sup>[14](#references),[15](#references)</sup>: Compares expected survival time across groups, providing a clinically interpretable measure of absolute benefit

Visualization:

```python
learner.plot_survival(n_groups=3)  # Kaplan-Meier curves by risk group, log-rank test, at-risk table
```

### Reading results back: properties

Each post-train analysis stashes its result so you can read it back without
re-computing. Use the **read-only properties** — never reach into
`learner._state`:

```python
learner.benchmark_result        # aggregate model comparison (after benchmark())
learner.logrank_test_results    # {model: log-rank table} (after logrank_test())
learner.rmst_test_results       # {model: RMST table} (after rmst_test())
learner.ph_test_result          # PH assumption test (from setup(), or after ph_test())
learner.linearity_test_result   # linearity test (from setup(), or after linearity_test())
learner.kang_comparison_result  # paired Kang comparison (auto after multi-model evaluate())
learner.pimp_importance_test_results       # {model: PIMP table} (after importance_test())
learner.bootstrap_importance_test_results  # {model: bootstrap table} (after importance_test(method='bootstrap'))
```

The singular `*_result` properties return `None` before the relevant step has
run (a failed `ph_test()` / `linearity_test()` re-run stores an empty
DataFrame instead). The per-model dict properties return `{}`, never `None`.
These join the existing `evaluation_result` / `bootstrap_result` /
`best_model` properties.

The PH and linearity DataFrames are self-describing: `df.attrs` records the
result-affecting parameter (`time_transform` / `n_knots`) and a `source`
marker (`'setup'` or `'method'`), so you can always tell whether you are
looking at the setup-default run or a re-run with different parameters.

The risk-group (`logrank`/`rmst`) and importance-test dicts hold one entry per
model; re-running a method on a model replaces only that model's entry, and
re-training or re-tuning a model drops its (now stale) entries. Each stored
DataFrame records its parameters in `.attrs`:

- log-rank: `model_name`, `n_groups`, `p_adjust`, `break_type`, `breaks`, `correction`
- RMST: as log-rank plus `tau`
- PIMP: `n_permutations`, `n_importance_repeats`, `random_state`, `p_adjust`, `model_name`
- importance bootstrap: `p_adjust`, `model_name`

**Naming convention.** A singular `<method>_result` property exposes a test run
on the training data (`ph_test`, `linearity_test`); the Kang `_comparison`
property is an effect without a single p-value. The model-keyed tests expose
per-model dicts instead, because results are stored per model and a singular
accessor would silently reflect only the latest call: `logrank_test_results`
and `rmst_test_results` (one entry per model), and `importance_test()`'s
`<method>_importance_test_results` (one dict per method).

## Subpopulation Analysis

Models with strong overall performance may underperform in specific subpopulations, masking clinically significant disparities. When grouping variables are configured at `setup()`, mlsurv evaluates each model within every subpopulation for every metric.

### Configuration

```python
learner.setup(
    subpopulations={'Cancer_Type': cancer_labels},
    stratify_subpopulations=True,
    random_state=42,
)
```

- **Event x subpopulation stratification**: Automatic balancing in train/test splits and CV folds preserves subpopulation proportions
- **Per-subpopulation metrics**: C-index, IPCW, IBS, and AUC(t) are computed within each subpopulation
- **Bootstrap CIs**: Uncertainty quantification per subpopulation
- **Small subpopulation flagging**: Subpopulations with fewer than 30 events are flagged automatically (see [Reporting — Limitation Flags](reporting.md#limitation-flags))

### Accessing Results

mlsurv exposes subpopulation results through a `subpopulation(group)` view. Use the view's `.result` property to access the underlying `SubpopulationResult` (with `.summary`, `.metrics`, `.bootstrap_summary`, `.auc_t_bootstrap` DataFrames) for export or downstream analysis:

```python
# After run() or evaluate()
sg_result = learner.subpopulation('Cancer_Type').result  # SubpopulationResult

print(sg_result.summary)   # one row per subpopulation: n_samples, n_events, c_index, ...
print(sg_result.metrics)   # long-form (subpopulation, model_type, metric, value, ci_low, ci_high)

# Shorthand when exactly one subpopulation variable is configured:
sg_result = learner.subpopulation().result
```

### Visualization

Plot methods are available on the `subpopulation(group)` view:

```python
learner.subpopulation('Cancer_Type').plot_survival()              # KM curves by group
learner.subpopulation('Cancer_Type').plot_metric(style='forest')  # Forest plot by group
learner.subpopulation('Cancer_Type').plot_metric_time()           # Time-dependent AUC/Brier by group
```

See [Vignette 07 — Immunotherapy Case Study](../examples/vignette_07_immunotherapy_case_study.ipynb) for a worked example across 16 cancer types with bootstrap CIs and small-subpopulation flagging.

## Prospective Patient Prediction

`predict()` generates risk scores, survival probabilities at specified time points, median survival times, and optional bootstrap confidence intervals for new patient data beyond the existing dataset:

```python
prediction = learner.predict(new_patient_data, models='coxph', times=[12, 24, 60], ci=True)

print(prediction.risk_scores)         # Risk score per patient (ndarray)
print(prediction.survival_function)   # Survival function S(t|x) as DataFrame
print(prediction.median_survival)     # Median survival time per patient (ndarray)
print(prediction.median_survival_ci)  # Bootstrap CIs for median survival (DataFrame)
```

A multi-model selection (a list/dict, or `models=None` with several models
trained) returns a dict of per-model results keyed by model name, so the
models' predictions for the same patients can be compared side by side:

```python
preds = learner.predict(new_patient_data, models=['coxph', 'rsf'])
preds['coxph'].risk_scores, preds['rsf'].risk_scores
```

`predict()` also attaches training-cohort context to the result (cohort size, test C-index, per-feature training ranges, the training risk-score distribution, and the training outcome array). This lets `prediction.summary(patient=i)` produce a plain-language narrative that places the patient in cohort context — risk percentile, out-of-distribution feature flags, median/milestone survival with CIs, and imputation flags — without any additional queries to the learner:

```python
prediction = learner.predict(new_patient_data, models='coxph',
                             times=[12, 36, 60], ci=True)
print(prediction.summary(patient=0))
# Patient 0 Prediction (coxph model)
#   Model: coxph | training cohort N=480 | test C-index 0.70 | 8 features
#   Risk score: 0.110 (training range [-4.39, 2.60]; higher = higher risk)
#   Risk percentile: 58% of the training cohort (N=480)
#   Median survival: 1675.6 days [95% CI: 1100.2-2208.9 days]
#   P(survive past 12 days): 100%
#   ...
```

Extra helpers on the result expose the same context programmatically: `prediction.risk_percentile(patient)`, `prediction.ood_features(patient)`, `prediction.nearest_neighbors(patient, k)` (by predicted risk), and `prediction.nearest_neighbors_feature_space(patient, k)` (by z-scored Euclidean distance across features).

`prediction.plot(patient=0)` returns a standalone Plotly figure of one patient's predicted survival curve `S(t|x)` — the same curve as the patient-report panel, with the bootstrap CI band (when produced via `predict(ci=True, ci_band=True)`), a median-survival marker at `S=0.5`, and milestone annotations for any `times` passed to `predict()`:

```python
fig = prediction.plot(patient=0)   # or a 'low_risk' / 'high_risk' / 'median' / 'pNN' spec
fig.write_image('patient_0_survival.png')
```

### Single-Patient HTML Report

`patient_report()` packages `predict()` and `explain()` into a single-page HTML report intended for clinical interpretation:

```python
html = learner.patient_report(
    new_patient_data, patient=0, model='coxph',
    output_path='patient_0_report.html',
    time_unit='days',  # optional; also inherited from setup(time_unit=...)
    # times= defaults to clinically relevant milestones for the unit
)
```

Sections (in order):

1. **Header** — patient index, report date, model provenance one-liner
2. **Alerts** — rendered only when warranted: OOD features, imputed inputs, wide median-survival CI
3. **Prediction summary** — narrative prose (from `PredictionResult.summary`) alongside a key-stat grid (risk score, percentile, median survival with CI, milestone probabilities with CIs)
4. **Individualized survival curve** — S(t|x) with bootstrap CI band (when `ci=True`), milestone markers, and a median-survival dot at S=0.5
5. **Cohort context** — risk-score histogram with patient marker, plus a KM overlay stratified by training risk quartile with the patient's predicted curve overlaid (colors match `plot_survival`)
6. **Feature contributions** — SHAP waterfall (filled arrow bars, `E[f(X)]` and `f(x)` anchored on dashed reference lines) and a feature table with `Train percentile [range]`, SHAP value with direction, and imputation flag
7. **Similar training patients** — two tables: closest by predicted risk, and closest by z-scored feature distance
8. **Footer** — full model metadata and disclaimer

Key options:

| Argument | Purpose |
|----------|---------|
| `patient` | Row index in `X` to report on (default `0`), or a risk-percentile shortcut string resolved against the model's predictions: `'low_risk'` (p10), `'high_risk'` (p90), `'median'` (p50), or `'pNN'` for any percentile NN in 0..100 |
| `times` | Milestone times for P(survive > t). If omitted, defaults from `time_unit` (years → `[1,3,5]`; months → `[12,36,60]`; weeks → `[52,156,260]`; days → `[365,1095,1825]`) |
| `time_unit` | `'years' / 'months' / 'weeks' / 'days'`; singular/abbreviated forms accepted. Falls back to `learner._state.time_unit` set at `setup()` |
| `ci`, `ci_band` | Bootstrap CI on milestone/median (default on) and the shaded band on the survival curve (default on; gated memory use for batch `predict`) |
| `n_neighbors` | `k` for the nearest-neighbor tables (default `10`) |
| `offline` | Embed plotly.js instead of loading from CDN (larger file, works without network) |

To decompose an individual prediction into per-feature SHAP contributions without generating a full report, see [Feature Analysis — Individual Patient Explanation](feature-analysis.md#individual-patient-explanation).

## References

**Hyperparameter tuning**
1. Akiba T, Sano S, Yanase T, Ohta T, Koyama M. Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD '19.* 2019:2623-2631. [doi:10.1145/3292500.3330701](https://doi.org/10.1145/3292500.3330701)

**Frameworks**
2. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python. *J Mach Learn Res.* 2011;12:2825-2830. [jmlr.org/papers/v12/pedregosa11a.html](https://www.jmlr.org/papers/v12/pedregosa11a.html)
3. Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. *KDD '16.* 2016:785-794. [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
4. Paszke A, Gross S, Massa F, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS.* 2019. [proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html)

**Discrimination metrics**
5. Harrell FE, Califf RM, Pryor DB, Lee KL, Rosati RA. Evaluating the yield of medical tests. *JAMA.* 1982;247:2543-2546. [doi:10.1001/jama.1982.03320430047030](https://doi.org/10.1001/jama.1982.03320430047030)
6. Uno H, Cai T, Pencina MJ, D'Agostino RB, Wei LJ. On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data. *Stat Med.* 2011;30:1105-1117. [doi:10.1002/sim.4154](https://doi.org/10.1002/sim.4154)
7. Antolini L, Boracchi P, Biganzoli E. A time-dependent discrimination index for survival data. *Stat Med.* 2005;24:3927-3944. [doi:10.1002/sim.2427](https://doi.org/10.1002/sim.2427)
8. Uno H, Cai T, Tian L, Wei LJ. Evaluating Prediction Rules for t-Year Survivors With Censored Regression Models. *J Am Stat Assoc.* 2007;102:527-537. [doi:10.1198/016214507000000149](https://doi.org/10.1198/016214507000000149)

**Calibration metrics**
9. Steyerberg EW, Vickers AJ, Cook NR, et al. Assessing the performance of prediction models: a framework for traditional and novel measures. *Epidemiology.* 2010;21:128-138. [doi:10.1097/EDE.0b013e3181c30fb2](https://doi.org/10.1097/EDE.0b013e3181c30fb2)
10. Demler OV, Paynter NP, Cook NR. Tests of calibration and goodness-of-fit in the survival setting. *Stat Med.* 2015;34:1659-1680. [doi:10.1002/sim.6428](https://doi.org/10.1002/sim.6428)
11. Graf E, Schmoor C, Sauerbrei W, Schumacher M. Assessment and comparison of prognostic classification schemes for survival data. *Stat Med.* 1999;18:2529-2545. [doi:10.1002/(sici)1097-0258(19990915/30)18:17/18<2529::aid-sim274>3.0.co;2-5](https://doi.org/10.1002/(sici)1097-0258(19990915/30)18:17/18%3C2529::aid-sim274%3E3.0.co;2-5)

**Model comparison & risk stratification**
12. Kang L, Chen W, Petrick NA, Gallas BD. Comparing two correlated C indices with right-censored survival outcome: a one-shot nonparametric approach. *Stat Med.* 2015;34:685-703. [doi:10.1002/sim.6370](https://doi.org/10.1002/sim.6370)
13. Mantel N. Evaluation of survival data and two new rank order statistics arising in its consideration. *Cancer Chemother Rep.* 1966;50:163-170.
14. Royston P, Parmar MKB. Restricted mean survival time: an alternative to the hazard ratio for the design and analysis of randomized trials with a time-to-event outcome. *BMC Med Res Methodol.* 2013;13:152. [doi:10.1186/1471-2288-13-152](https://doi.org/10.1186/1471-2288-13-152)
15. Uno H, Claggett B, Tian L, et al. Moving beyond the hazard ratio in quantifying the between-group difference in survival analysis. *J Clin Oncol.* 2014;32:2380-2385. [doi:10.1200/JCO.2014.55.2208](https://doi.org/10.1200/JCO.2014.55.2208)
