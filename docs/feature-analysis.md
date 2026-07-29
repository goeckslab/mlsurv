# Feature Analysis

Feature analysis quantifies which variables drive predictions and how they interact, spanning global importance across the cohort to local explanations for individual patients.

## Feature Importance

Multiple importance methods provide complementary perspectives on which features drive predictions.

For quick access, `importance()` is a single entry point that wraps both backends. It defaults to permutation importance and switches to SHAP via `method='shap'`. Both return one row per feature per selected model (the columns differ — see each method below), and both require a prior `evaluate()`:

```python
learner.importance('rsf')                  # permutation (default)
learner.importance('rsf', method='shap')   # per-feature mean |SHAP|

learner.importance()                       # all trained models, stacked
learner.importance(models=['coxph', 'rsf'])  # a subset
```

The `method='shap'` summary has columns `[model_type, feature, mean_abs_shap]`; the full per-sample SHAP values remain on `result.evaluation.shap_values`.

A multi-model selection (a list/dict, or `models=None` with several models trained) returns one stacked DataFrame with a block of rows per model, keyed by the `model_type` column — the same model handling as `plot_importance()`. Models without an evaluation result are skipped with a warning.

### Permutation Importance (PIMP)

Permutation importance measures the drop in model performance when a feature's values are randomly shuffled. Statistical significance is assessed via the permuted-response method<sup>[1](#references)</sup>.

```python
learner.evaluate(n_importance_repeats=10)
print(result.evaluation.importance)  # DataFrame: model_type, feature, coefficient, perm_mean, perm_std
```

`evaluate()` computes permutation importance as part of the full metric suite. To recompute it standalone — for example at a different repeat count or seed — without re-running the other metrics, call `importance(method='permutation')`. It requires a prior `evaluate()` and overwrites the importance fields on the existing result in place:

```python
learner.importance('rsf', method='permutation', n_importance_repeats=50)
```

#### Cost on large cohorts

Permutation importance costs `n_features x n_repeats x (predict + C-index over n_test rows)`. The C-index term is superlinear in the number of test rows, so on large cohorts importance can take longer than fitting the models. Two levers:

```python
# 1. Compute importance on a capped, seeded subsample of the test set.
learner.setup(random_state=42)   # the cap draws from this seed
learner.evaluate(importance_max_samples=1000)

# 2. Skip it during an expensive sweep, then compute it on demand later,
#    including from a learner reloaded with load_learner().
learner.evaluate(n_importance_repeats=0)
learner.importance('coxph', method='permutation')
```

`importance_max_samples` defaults to `None`, which uses every row. A value at or above the cohort size is a no-op, and the minimum is 2 (the C-index needs a pair of rows to compare). The draw is seeded from `random_state`, so with a seed set all features within a model, and all models in the call, are scored on the same rows and their importances stay comparable. Without a seed each model draws its own subsample; mlsurv warns when that happens. Permutation importance is a ranking diagnostic that tolerates subsampling well: the cost is a little precision in the importance estimate, not in the model.

### Grouped Importance

For one-hot encoded categorical features, grouped importance permutes all dummy columns jointly and reports a single aggregated score per original variable<sup>[2](#references),[3](#references)</sup>:

```python
learner.setup(
    categorical_groups={'Cancer_Type': ['CT_Breast', 'CT_Lung', 'CT_Melanoma', ...]},
)
```

### Cox Coefficients and Hazard Ratios

For models that support them (CoxPH, CoxNet), coefficients and hazard ratios with confidence intervals are available:

```python
print(result.evaluation.importance)  # Includes 'coefficient' column for Cox models

learner.plot_coefficients('coxph')                   # Coefficient bar/forest plot
learner.plot_hazard_ratios(kind='univariate')        # Univariate HR forest plot
learner.plot_hazard_ratios(kind='multivariate')      # Multivariate HR forest plot
learner.plot_hazard_ratios_comparison()              # Univariate vs multivariate side-by-side
```

### Univariate Cox Statistics

`setup()` automatically fits a univariate Cox model per feature and stores the
results on the `learner.univariate_cox` property — a model-agnostic table of
each feature's marginal association with the outcome, available right after
`setup()` (before any model is trained):

```python
learner.setup()
print(learner.univariate_cox)
#   feature  p_value  hazard_ratio  hr_lower_95  hr_upper_95  coefficient
#       age    0.001         1.045        1.019        1.071        0.044
#     stage    0.003         2.134        1.287        3.539        0.758
#       ...
```

Columns: `feature`, `p_value` (Wald), `hazard_ratio`, `hr_lower_95`,
`hr_upper_95`, and `coefficient` (log hazard ratio), sorted by ascending
`p_value`. These statistics are fit on the **full** cohort and are descriptive
only — they are never used for model fitting or feature selection. The property
is `None` before `setup()` runs or if the fit fails.

It is read/write, so you can recompute it (for example, on the training split
only) and assign it back:

```python
from mlsurv.evaluation import compute_univariate_cox

learner.univariate_cox = compute_univariate_cox(learner.X_train, learner.y_train)
```

#### Unbounded hazard ratios

A separated feature (every event on one side of it) or a badly scaled one
pushes the coefficient far enough that `exp()` leaves the range of a float.
Then `hr_upper_95` comes back as `inf`, `hr_lower_95` as `0.0`, and
`hazard_ratio` itself can be either. The table keeps those values, because they
are true: the interval really is unbounded.

A log-scaled forest plot cannot place them, so it leaves them out of the axis
range and draws nothing for them. An affected feature loses the whisker on the
bad side, or its marker too when the point estimate is the bad value. It keeps
its row either way, so you can still see which feature it is, and the hover
text still shows the true bound. Each plot reports what it dropped through a
`ClampWarning`, and the report's Hazard Ratios section repeats it in writing
for whoever reads the report rather than runs it.

If you see that warning, fix it on the data side: rescale or drop the feature,
or recompute the table with a larger `penalizer`.

## SHAP Analysis

SHAP (Shapley Additive Explanations)<sup>[4](#references)</sup> values decompose each prediction into per-feature contributions, providing both global importance rankings and individual patient-level explanations.

### Global SHAP

```python
learner.plot_shap('rsf', kind='importance')  # Global importance bar chart
learner.plot_shap('rsf', kind='beeswarm')    # Per-sample values colored by feature value
```

SHAP values are computed on demand by the plotting calls above. To compute them standalone after `evaluate()` — without re-running the rest of the metric suite — call `importance(method='shap')`. It returns a global ranking (one row per feature) and stores the per-sample values on `result.evaluation.shap_values` as a side effect, so `plot_shap` and the beeswarm work afterward:

```python
shap_summary = learner.importance('rsf', method='shap')   # columns: [model_type, feature, mean_abs_shap]
learner.plot_shap('rsf', kind='beeswarm')                 # reads the stored per-sample values
```

The raw per-sample SHAP frame stays available on `result.evaluation.shap_values` (where `result = learner.models['rsf']`), mirroring how permutation importance keeps its per-repeat values on `result.evaluation.importance_iterations`.

When categorical groups are defined, SHAP values for one-hot encoded columns are aggregated into a single contribution per original variable.

## Feature Interactions

Machine learning models can capture interaction effects that linear models assume away. mlsurv provides dedicated tools to identify and visualize these interactions.

### H-Statistic

The Friedman-Popescu H-statistic<sup>[6](#references)</sup> quantifies interaction strength via partial dependence for all nonlinear models. It measures how much the joint effect of two features differs from the sum of their individual effects. The method is selected at training time via `interaction_method`:

```python
# Compute H-statistic interactions during run()
learner.run(['rsf'], include_interactions=True, interaction_method='hstat')

learner.plot_interactions('rsf')  # Heatmap of pairwise H-statistics
```

### SHAP Interaction Values

For tree-based models (RSF, GBSA, XGBoost), SHAP interaction values<sup>[5](#references)</sup> decompose pairwise contributions at the per-sample level. Switch methods via `interaction_method='shap'` at training time:

```python
learner.run(['xgbcox'], include_interactions=True, interaction_method='shap')

learner.plot_interactions('xgbcox')  # Heatmap of SHAP-based interactions
```

### 2D Partial Dependence

Two-dimensional partial dependence surfaces visualize the joint effect of a feature pair on predicted risk:

```python
learner.plot_interaction_pdp(feature_1='TMB', feature_2='Albumin', model='rsf')
```

Contour plots for continuous-continuous pairs and heatmaps for categorical pairs reveal non-additive relationships invisible to univariate importance.

For right-skewed features, pass `xscale='log'` or `yscale='log'` to space the grid on a base-10 log scale, sampling densely where the data is dense (`xscale` sets `feature_2` / x-axis, `yscale` sets `feature_1` / y-axis):

```python
learner.plot_interaction_pdp(feature_1='Albumin', feature_2='TMB', model='rsf', xscale='log')
```

Zeros are floored to the smallest positive value; categorical, negative, or all-non-positive features fall back to a linear axis with a warning.

See [Vignette 07 — Immunotherapy Case Study](../examples/vignette_07_immunotherapy_case_study.ipynb) for H-statistic and 2D PDP analysis on top-interacting pairs across clinical and genomic features.

## Individual Patient Explanation

`explain()` decomposes a single prediction into per-feature SHAP contributions, rendered as an interactive waterfall plot. Use it alongside [`predict()`](evaluation.md#prospective-patient-prediction) to understand *why* a model assigned a given risk score. Pass new patient data via the `X=` keyword argument:

```python
learner.explain(X=new_patient_data, model='rsf')

# Or explain a patient already in the test set by index
learner.explain(patient=0, model='rsf')

# patient= also accepts a risk-percentile shortcut, resolved against the
# model's predictions on the test set: 'low_risk' (p10), 'high_risk' (p90),
# 'median' (p50), or 'pNN' for any percentile NN in 0..100
learner.explain(patient='high_risk', model='rsf')
```

When categorical groups are defined, SHAP values for one-hot encoded columns are aggregated into a single contribution per original variable, providing cleaner explanations.

The waterfall plot follows the canonical SHAP convention: features are ordered by |SHAP| with the most influential at the top, each bar is a filled arrow pointing in the direction of its contribution (red → increases risk, blue → decreases), and dashed reference lines mark `E[f(X)]` (cohort baseline) and `f(x)` (the patient's predicted risk) at the bottom and top of the chart. The returned `PatientExplanation` carries `base_value = E[f(X)]` as a field; by construction, `base_value + sum(shap_values) == risk_score`.

To bundle the explanation with a full clinical-style report (narrative summary, survival curve with CI band, cohort context, nearest-neighbor tables), use [`patient_report()`](evaluation.md#single-patient-html-report) — it internally calls `explain()` and `predict()` and assembles a standalone HTML file.

See [Vignette 02 — Workflow Tour](../examples/vignette_02_workflow_tour.ipynb) for a walk-through from global SHAP importance down to individual-patient waterfall plots.

## Visualization Methods

mlsurv provides 23 programmatic visualization methods, all rendered as interactive Plotly figures. Each method is also available as a standalone call for use outside of reports.

### Performance Visualization

| Method | Description |
|--------|-------------|
| `plot_metric` | Model performance comparison with bootstrap CIs (forest, bar, raincloud, box, violin) |
| `plot_metric_time` | Time-dependent AUC or Brier score curves |
| `plot_metric_heatmap` | Multi-metric heatmap across models |
| `plot_roc` | Time-dependent ROC curve with bootstrap CI ribbons |
| `plot_calibration` | Calibration curves (predicted vs observed survival) with Greenwood CIs |
| `plot_calibration_overlay` | Predicted vs. observed survival curves |
| `plot_benchmark_incremental` | Forest plot of delta-C, NRI, IDI for benchmarking |
| `plot_benchmark_scatter` | Scatter of model risk score vs. each reference score |
| `plot_validation_heatmap` | Faceted model x metric heatmap, one panel per external cohort |

### Survival Visualization

| Method | Description |
|--------|-------------|
| `plot_survival` | Kaplan-Meier curves, overall or by risk group with log-rank test |
| `subpopulation(group).plot_survival` | KM curves stratified by subpopulation level |

### Feature Visualization

| Method | Description |
|--------|-------------|
| `plot_importance` | Permutation importance across models (bar, forest, raincloud, boxplot, heatmap) |
| `plot_shap` | SHAP values (importance bar, beeswarm) |
| `plot_coefficients` | Cox coefficients (bar, forest) |
| `plot_hazard_ratios` | Hazard ratio forest (univariate, multivariate) |
| `plot_hazard_ratios_comparison` | Univariate vs multivariate hazard ratio comparison |
| `plot_coefficient_comparison` | Scatter plot comparing coefficients between models |

### Interaction & Diagnostic Visualization

| Method | Description |
|--------|-------------|
| `plot_interactions` | Feature interaction heatmap (H-statistic or SHAP) |
| `plot_interaction_pdp` | 2D partial dependence contour surface for feature pairs |
| `plot_diagnostics` | Diagnostic residuals (Schoenfeld, martingale, deviance) |

### Data Visualization

| Method | Description |
|--------|-------------|
| `plot_feature_correlation` | Feature correlation heatmap |
| `plot_feature_distributions` | Feature distributions for train and test sets |
| `plot_missing` | Missing data heatmap |
| `plot_optuna` | Optuna optimization history and parameter importance |

## References

**Feature importance**
1. Altmann A, Toloşi L, Sander O, Lengauer T. Permutation importance: a corrected feature importance measure. *Bioinformatics.* 2010;26:1340-1347. [doi:10.1093/bioinformatics/btq134](https://doi.org/10.1093/bioinformatics/btq134)
2. Strobl C, Boulesteix A-L, Zeileis A, Hothorn T. Bias in random forest variable importance measures: illustrations, sources and a solution. *BMC Bioinformatics.* 2007;8:25. [doi:10.1186/1471-2105-8-25](https://doi.org/10.1186/1471-2105-8-25)
3. Au Q, Herbinger J, Stachl C, Bischl B, Casalicchio G. Grouped feature importance and combined features effect plot. *Data Min Knowl Discov.* 2022;36:1401-1450. [doi:10.1007/s10618-022-00840-5](https://doi.org/10.1007/s10618-022-00840-5)

**SHAP**
4. Lundberg SM, Lee S-I. A Unified Approach to Interpreting Model Predictions. *NeurIPS.* 2017. [proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
5. Lundberg SM, Erion G, Chen H, et al. From Local Explanations to Global Understanding with Explainable AI for Trees. *Nat Mach Intell.* 2020;2:56-67. [doi:10.1038/s42256-019-0138-9](https://doi.org/10.1038/s42256-019-0138-9)

**Feature interactions**
6. Friedman JH, Popescu BE. Predictive learning via rule ensembles. *Ann Appl Stat.* 2008;2:916-954. [doi:10.1214/07-AOAS148](https://doi.org/10.1214/07-AOAS148)
