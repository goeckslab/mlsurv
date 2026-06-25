# Pipeline Configuration

This guide covers data loading, the `setup()` configuration step, preprocessing, and automated diagnostic checks that run before model training.

## Data Loading

mlsurv provides helper functions for loading and preparing survival data.

### From CSV

```python
from mlsurv.utils import load_survival_data

X, y = load_survival_data('data.csv', time_col='OS_MONTHS', event_col='OS_STATUS')
```

`load_survival_data()` reads a CSV file, drops the time and event columns from the feature matrix, converts events to boolean and times to float, and constructs a structured NumPy array following scikit-survival conventions. It validates event encodings, rejects non-positive survival times, and flags potential competing-risks scenarios (non-binary event columns).

### From Date Columns

When your data has diagnosis and follow-up dates instead of pre-computed survival times:

```python
from mlsurv.utils import compute_survival_time

X, y = compute_survival_time(
    df, start_col='diagnosis_date', end_col='last_followup_date',
    event_col='vital_status', time_unit='months',
    event_values=['deceased'],
)
```

### From Existing DataFrame

```python
from mlsurv.utils import to_structured_array

y = to_structured_array(times=df['OS_MONTHS'], events=df['OS_STATUS'])
X = df.drop(columns=['OS_MONTHS', 'OS_STATUS'])
```

## Setup Configuration

The `setup()` function configures the analysis pipeline. All subsequent training, evaluation, and reporting steps inherit these settings.

```python
learner = SurvivalLearner(X, y)
learner.setup(
    test_fraction=0.3,       # Train/test split ratio
    cv_folds=5,              # Number of CV folds
    scoring='harrell',       # Metric for model optimization
    random_state=42,         # Reproducibility seed
)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `test_fraction` | 0.3 | Fraction of data held out for testing |
| `cv_folds` | 5 | Number of cross-validation folds |
| `scoring` | `'harrell'` | Metric for model optimization (Harrell's C-index<sup>[1](#references)</sup>; see [Evaluation Metrics](evaluation.md#evaluation-metrics)) |
| `random_state` | `None` | Seeds all frameworks (scikit-learn<sup>[2](#references)</sup>, XGBoost, PyTorch) |
| `imputer` | `'median'` | Imputation strategy (`'median'`, `'mean'`, `'knn'`, `'iterative'`, `None`) |
| `scaler` | `'standard'` | Scaling strategy (`'standard'`, `'robust'`, `'minmax'`, `None`) |
| `feature_selector` | `None` | Feature selection method (see [Feature Selection](#feature-selection)) |
| `n_jobs` | 1 | Parallel workers (-1 for all cores) |
| `subpopulations` | `None` | Dict mapping group names to labels for subpopulation analysis |
| `stratify_subpopulations` | `True` | Maintain subpopulation proportions in splits |

The lifecycle is:

```
setup → train / run → evaluate → bootstrap → validate → report
```

`S(t|x)` calibration happens inside `train()` / `run()` via their `calibrate=` argument: it fits a per-model calibrator on the development cohort's held-out OOF predictions, which `evaluate()`, `bootstrap()`, `predict()`, `validate()`, and the reports then apply automatically (see [S(t|x) Calibration](evaluation.md#stx-calibration)).

### Cross-Validation Strategies

| Strategy | Use Case | Configuration |
|----------|----------|---------------|
| Stratified K-Fold | General use (default) | `cv_folds=5` |
| GroupKFold | Clustered observations | `cv_strategy='group'`, `groups=group_labels` |
| TimeSeriesSplit | Temporally ordered data | `cv_strategy='timeseries'` |
| Repeated CV | Small datasets, improved stability | `cv_repeats=3` |
| Pre-split data | External train/test split | `X_test=X_test, y_test=y_test` |
| Custom CV object | Any scikit-learn splitter | `cv_strategy=custom_splitter` |

Stratified splitting maintains event ratios across partitions to prevent imbalanced evaluation.

`cv_strategy='timeseries'` splits the data with `shuffle=False` (earliest rows become train, latest rows become test) and the CV folds forward-chain over the training rows. This only yields a true past→future split if **the rows are already sorted chronologically by entry/diagnosis time** before `setup()`. mlsurv does not sort or verify the order, so it emits a `SuggestionWarning` whenever this strategy is selected; if the rows are not in chronological order, the split and folds are not temporal and the out-of-sample estimate is invalid. Sort your data first (e.g. `X = X.sort_values('year'); y = y[X.index]`).

See [Vignette 05 — Advanced CV & Tuning](../examples/vignette_05_advanced_cv_tuning.ipynb) for worked examples of temporal, repeated stratified, and nested cross-validation.

## Preprocessing Pipeline

Preprocessing follows a fixed stage order: **imputation → scaling → feature selection**. The full pipeline is fitted independently within each CV fold so that scores reflect out-of-sample performance of the complete pipeline, preventing inflated estimates from information leakage across folds.

### Feature Selection

Five built-in selectors are available:

| Selector | Method | Notes |
|----------|--------|-------|
| `'topk'` | Univariate Cox ranking<sup>[7](#references)</sup> | Automatic dimensionality via Harrell's rule (EPV/10)<sup>[4](#references)</sup> |
| `'coxlasso'` | L1-penalized Cox regression<sup>[6](#references)</sup> | Selects features with non-zero coefficients |
| `'coxnet'` | Elastic net Cox regression<sup>[6](#references)</sup> | Combines L1 and L2 penalization |
| `'vif'` | Variance inflation factor | Removes features with high multicollinearity |
| `'correlation'` | Pairwise correlation filter | Removes one of each highly correlated pair |

Cox-based selectors (`'topk'`, `'coxlasso'`, `'coxnet'`) are fitted before imputation using per-feature complete-case analysis. All other selectors operate after imputation.

```python
learner.setup(feature_selector='topk', random_state=42)
```

Custom selector objects (any scikit-learn transformer) can also be supplied directly.

## Automated Setup Checks

During `setup()`, mlsurv runs several automated checks that surface potential issues before model training begins.

### Events Per Variable (EPV)

A low EPV warning is issued when the training set has fewer than 10 events per feature. Low EPV increases the risk of unstable coefficient estimates and overfitting.

### Missingness-Outcome Leakage

A chi-squared test flags features whose missing-data pattern is statistically associated with the outcome (p < 0.001 AND Cramer's V > 0.3). This can indicate that missingness itself carries prognostic information, which may introduce bias.

### Proportional Hazards Testing

The Grambsch-Therneau test<sup>[3](#references)</sup> evaluates whether each feature satisfies the proportional hazards assumption using Schoenfeld residuals. Results are corrected for multiple comparisons using Benjamini-Hochberg FDR adjustment<sup>[5](#references)</sup>. Violations both emit a `MethodologicalWarning` and feed into the unified [model recommendation table](#model-recommendations), demoting PH-assuming models (CoxPH, CoxNet, GBSA, XGBoost-Cox, DeepSurv, Weibull-AFT) in proportion to the violation rate. The full per-feature DataFrame is available as `learner.ph_test_result` (no recompute; `learner.model_recommendations.ph_test_result` keeps the setup-time snapshot), and `learner.ph_test()` re-runs the test on demand and returns the fresh DataFrame.

**Why Weibull-AFT is in the PH-assuming list.** The Weibull distribution is the only one under which the AFT and PH parameterizations coincide<sup>[12](#references)</sup>. Substituting the AFT scale `λ = exp(-β'x)` into the Weibull survival function `S(t) = exp(-(λt)^k)` yields `exp(-t^k · exp(-kβ'x))` — exactly a PH form with constant hazard ratio `exp(-kβ'x)`. mlsurv wraps lifelines' `WeibullAFTFitter` with a single shared shape parameter and no time-varying covariates, so the fit is *unconditionally* PH; Grambsch-Therneau violations imply Weibull-AFT misspecification just as they do for CoxPH. (The other parametric AFT distributions — lognormal, log-logistic, generalized gamma — are AFT but not PH; only Weibull has this duality.)

### Log-Hazard Linearity Testing

A restricted cubic splines (RCS) joint Wald test evaluates whether the log-hazard is linear in each continuous covariate, following Harrell's *Regression Modeling Strategies* methodology<sup>[4](#references)</sup>. Results use BH-FDR correction. As with PH, violations both emit a `MethodologicalWarning` and feed the recommendation table, demoting linearity-assuming models. The full result is available as `learner.linearity_test_result` (no recompute; `learner.model_recommendations.linearity_test_result` keeps the setup-time snapshot), and `learner.linearity_test()` re-runs the test and returns the fresh DataFrame.

## Model Recommendations

`setup()` produces a single unified recommendation table that combines EPV / sample-size checks with the PH and linearity diagnostics. The EPV thresholds are derived from published complexity-specific evidence:

| Complexity | Recommended (N) | Recommended (EPV) | Cautionary (N) | Cautionary (EPV) |
|------------|----------------:|-------------------:|----------------:|------------------:|
| Linear | — | ≥ 20 | — | ≥ 5 |
| Tree ensemble | ≥ 500 | ≥ 50 | ≥ 200 | ≥ 20 |
| Deep learning | ≥ 1,000 | ≥ 100 | ≥ 500 | ≥ 50 |

Models are classified into three tiers:
- **Recommended**: Both sample size and EPV meet the recommended thresholds
- **Cautionary**: Meets cautionary but not recommended thresholds
- **Not recommended**: Below cautionary thresholds

Assumption violations modify these tiers as follows:
- **PH violations** demote PH-assuming models (CoxPH, CoxNet, GBSA, XGBoost-Cox, DeepSurv, Weibull-AFT). Up to 25% of features violating drops one tier; more than 25% drops two.
- **Log-linearity violations** demote linearity-assuming models on the same scale. Demotions stack: a model that assumes both PH and linearity can drop two tiers from a single violation type or from two milder ones.
- **High-dimensionality hard rule** (Hu et al. 2023<sup>[11](#references)</sup>): when `n_features ≥ n_events`, unregularized Cox-family models (`coxph`, `weibull_aft`) are forced to *Not recommended* because the partial likelihood is ill-conditioned. `coxnet` is exempt (regularization handles high-dim).

The trailing `Non-PH features` and `Log-hazard non-linear features` lines under the table list the offending features (first five plus a count suffix).

```python
from mlsurv import recommend_models

# Standalone recommendation from (X, y) — no PH/linearity info passed in,
# so PH/linearity-assuming models will report "test unavailable" in their reason.
rec = recommend_models(X, y)
print(rec)  # .recommended, .cautionary, .not_recommended, .epv, .n_samples

# The unified table is printed automatically by setup() (when verbose=True)
# and exposed on the learner via the read-only model_recommendations
# property — always populated after setup() (the diagnostic always runs).
rec = learner.model_recommendations

# Model-level verdict: read the tiers. PH/linearity are already folded in,
# so the tier (not a per-model PH flag) is the model verdict.
rec.recommended                   # ['rsf', 'gbsa', 'coxnet', ...]
rec.cautionary, rec.not_recommended
rec.details["coxph"]              # per-model reason text, e.g. "...; 2 PH violations"

# Dataset stats (EPV / sample size are already in the bundle).
rec.epv, rec.n_samples, rec.n_events, rec.n_features

# Feature-level evidence. PH and linearity are per-FEATURE tests (they name
# the covariates that violate, not models), so they live as DataFrames and
# violation tuples — not as a per-model partition.
rec.ph_test_result                # DataFrame: feature, test_statistic, p_value, p_adjusted, significant
rec.linearity_test_result         # same columns; None when a test was skipped
rec.ph_violations                 # tuple[str, ...] — violating feature names
rec.linearity_violations          # tuple[str, ...]
```

`ph_test_result` / `linearity_test_result` are the **setup-time snapshot**.
Calling `learner.ph_test()` or `learner.linearity_test()` afterward (e.g.
with a different `time_transform` or `n_knots`) refreshes the live
`learner.ph_test_result` / `learner.linearity_test_result` properties but
does **not** rewrite the bundle's snapshot. Read the tiers for a model
verdict; read `ph_test_result` for the per-feature test.

Evidence: van der Ploeg et al. (2014)<sup>[9](#references)</sup>, Harrell (2015)<sup>[4](#references)</sup>, Ogundimu et al. (2016)<sup>[10](#references)</sup>, Riley et al. (2019)<sup>[8](#references)</sup>.

## References

1. Harrell FE, Califf RM, Pryor DB, Lee KL, Rosati RA. Evaluating the yield of medical tests. *JAMA.* 1982;247:2543-2546. [doi:10.1001/jama.1982.03320430047030](https://doi.org/10.1001/jama.1982.03320430047030)
2. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python. *J Mach Learn Res.* 2011;12:2825-2830. [jmlr.org/papers/v12/pedregosa11a.html](https://www.jmlr.org/papers/v12/pedregosa11a.html)
3. Grambsch PM, Therneau TM. Proportional hazards tests and diagnostics based on weighted residuals. *Biometrika.* 1994;81:515-526. [doi:10.1093/biomet/81.3.515](https://doi.org/10.1093/biomet/81.3.515)
4. Harrell FE. *Regression Modeling Strategies: With Applications to Linear Models, Logistic and Ordinal Regression, and Survival Analysis.* 2nd ed. Springer; 2015. [doi:10.1007/978-3-319-19425-7](https://doi.org/10.1007/978-3-319-19425-7)
5. Benjamini Y, Hochberg Y. Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *J R Stat Soc Ser B.* 1995;57:289-300. [doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
6. Simon N, Friedman J, Hastie T, Tibshirani R. Regularization Paths for Cox's Proportional Hazards Model via Coordinate Descent. *J Stat Softw.* 2011;39:1-13. [doi:10.18637/jss.v039.i05](https://doi.org/10.18637/jss.v039.i05)
7. Davidson-Pilon C. lifelines: survival analysis in Python. *J Open Source Softw.* 2019;4:1317. [doi:10.21105/joss.01317](https://doi.org/10.21105/joss.01317)
8. Riley RD, Snell KI, Ensor J, et al. Minimum sample size for developing a multivariable prediction model: PART II - binary and time-to-event outcomes. *Stat Med.* 2019;38:1276-1296. [doi:10.1002/sim.7992](https://doi.org/10.1002/sim.7992)
9. van der Ploeg T, Austin PC, Steyerberg EW. Modern modelling techniques are data hungry: a simulation study for predicting dichotomous endpoints. *BMC Med Res Methodol.* 2014;14:137. [doi:10.1186/1471-2288-14-137](https://doi.org/10.1186/1471-2288-14-137)
10. Ogundimu EO, Altman DG, Collins GS. Adequate sample size for developing prediction models is not simply related to events per variable. *J Clin Epidemiol.* 2016;76:175-182. [doi:10.1016/j.jclinepi.2016.02.031](https://doi.org/10.1016/j.jclinepi.2016.02.031)
11. Hu Y, Hu W, Liu C-H, et al. Modern statistical methods for analysis of high-dimensional survival data. *Annu Rev Stat Appl.* 2023;10:399-422. [doi:10.1146/annurev-statistics-033021-014739](https://doi.org/10.1146/annurev-statistics-033021-014739)
12. Kalbfleisch JD, Prentice RL. *The Statistical Analysis of Failure Time Data.* 2nd ed. Wiley; 2002. [doi:10.1002/9781118032985](https://doi.org/10.1002/9781118032985)
