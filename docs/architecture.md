# Architecture & Models

mlsurv organizes the full survival machine learning workflow into five stages, each accessible through a single method call while remaining extensible for advanced users.

## Five-Stage Workflow

```
Configure ──► Train ──► Evaluate ──► Validate ──► Report
 setup()      train()   evaluate()   validate()   generate_report()
```

Each stage also exposes optional methods for extended analyses.

**Stage 1 — Configure.** `setup()` configures the preprocessing pipeline, cross-validation strategy, and train/test split. Automated checks validate data integrity and generate model recommendations. See [Pipeline Configuration](pipeline.md).

**Stage 2 — Train.** `train()` fits models with default or user-supplied hyperparameters, and can fold in Optuna-based hyperparameter optimization via `train(tune=True)`. *Optional:* `tune()` runs tuning as a standalone step. See [Model Development & Evaluation](evaluation.md).

**Stage 3 — Evaluate.** `evaluate()` assesses discrimination and calibration on the held-out test set, with optional subpopulation breakdowns, feature importance, SHAP values, and feature interactions. Bootstrap confidence intervals can be folded in via `evaluate(bootstrap=True)`. *Optional:* `bootstrap()` runs test-set resampling as a standalone step, `predict()` generates prospective patient-level risk scores and survival probabilities, and `explain()` decomposes individual predictions into per-feature SHAP contributions. See [Model Development & Evaluation](evaluation.md) and [Feature Analysis](feature-analysis.md).

**Stage 4 — Validate.** `validate()` evaluates trained models on independent external cohorts using the same discrimination and calibration metrics as internal evaluation. *Optional:* `benchmark()` quantifies the added value of trained models over reference predictors (e.g., established clinical scores or biomarkers) using delta-C, NRI, and IDI. See [Validation & Benchmarking](validation.md).

**Stage 5 — Report.** `generate_report()` assembles all results into a standalone interactive HTML report with limitation flags and a TRIPOD+AI compliance checklist. *Optional:* `export_results()` writes structured results to spreadsheet or CSV, and `save_learner()` serializes the full analysis state for later reproduction. See [Reporting & Compliance](reporting.md).

### Two Workflow Tracks

**All-in-one** — `run()` wraps stages 2 and 3 (train + tune + evaluate + bootstrap) in a single call. External validation (Stage 4) requires a separate `validate()` call with the external cohort data.

```python
learner = SurvivalLearner(X, y)
learner.setup(random_state=42)
model_results = learner.run(['coxph', 'rsf', 'xgbcox'])  # same dict as learner.models
learner.generate_report('report.html')
```

See [Vignette 01 — Quickstart](../examples/vignette_01_quickstart.ipynb) for a runnable version of this workflow.

**Granular** — individual methods for fine-grained control:

```python
learner = SurvivalLearner(X, y)
learner.setup(random_state=42)

# Train with defaults, then tune
learner.train(['coxph', 'rsf'])
learner.tune('rsf', n_trials=50)

# Evaluate and compare
learner.evaluate()
learner.bootstrap(n_bootstrap=200)

# Add another model without re-running the pipeline
learner.train('xgbcox')
learner.evaluate('xgbcox')

# Validate and benchmark
learner.validate(X_external, y_external, name='External')
learner.benchmark(references={'Clinical Score': clinical_scores})

# Report and export
learner.generate_report('report.html')
learner.save_learner('analysis.pkl')
```

The granular approach supports workflows such as training all models then selectively tuning only the ones that benefit (`tune()` overwrites in place), evaluating a subset of trained models, or adding models to an existing comparison. See [Vignette 02 — Workflow Tour](../examples/vignette_02_workflow_tour.ipynb) for a comprehensive walk-through of the granular interface.

### Views

After training, mlsurv exposes two scoped views for analyzing subsets of results. Both take a user-supplied name as argument — the same name passed to `validate(..., name=...)` or to `setup(subpopulations={<name>: ...})`:

```python
# Earlier in the pipeline:
learner.validate(X_rotterdam, y_rotterdam, name='Rotterdam')
learner.setup(subpopulations={'CancerType': cancer_type_labels})

# Re-scope plots to the 'Rotterdam' external validation cohort
learner.validation('Rotterdam').plot_metric()
learner.validation('Rotterdam').plot_calibration(eval_time=365)

# Re-scope plots to the 'CancerType' subpopulation variable
learner.subpopulation('CancerType').plot_survival()
learner.subpopulation('CancerType').plot_metric_time()
```

`validation(cohort)` returns a `ValidationView` that reads from stored validation results instead of the internal test set. `subpopulation(group)` returns a `SubpopulationView` that reads from pre-aggregated per-subpopulation results. Both views expose a subset of the main `plot_*` methods with data pre-filtered to the cohort or subpopulation.

### Parameter Sweeps

For exploratory hyperparameter analysis outside of Optuna-based tuning, `parameter_sweep()` runs cross-validated training across a grid of parameter values and returns all metrics:

```python
result = learner.parameter_sweep(
    param='n_estimators', values=[100, 250, 500, 1000],
    target='model', models='rsf',
)
```

The sweep covers model parameters (`target='model'`) or preprocessing parameters (`target='imputer'`, `'scaler'`, `'feature_selector'`), and does not modify learner state. Useful for sensitivity analyses and didactic plots. See [Vignette 05 — Advanced CV & Tuning](../examples/vignette_05_advanced_cv_tuning.ipynb).

## Model Inventory

mlsurv integrates ten survival models spanning four families. All models share a unified interface for fitting, prediction, survival function estimation, scoring, and feature importance extraction.

Each row lists the **library** that implements the model (with the software citation) and the **model paper** that originally introduced the method. Both link to the full citation in [References](#references).

| Model | Abbrev. | Family | PH | Linearity | Library (software citation) | Model paper |
|-------|---------|--------|----|-----------|----------------------------|-------------|
| Cox Proportional Hazards | CoxPH | Linear | Yes | Yes | [scikit-survival](https://scikit-survival.readthedocs.io/) ([Pölsterl 2020](#references)) | [Cox 1972](#references) |
| Elastic Net Cox Regression | CoxNet | Linear (regularized) | Yes | Yes | [scikit-survival](https://scikit-survival.readthedocs.io/) ([Pölsterl 2020](#references)) | [Simon et al. 2011](#references) |
| Weibull Accelerated Failure Time | Weibull AFT | Linear (parametric) | Yes\* | Yes | [lifelines](https://lifelines.readthedocs.io/) ([Davidson-Pilon 2019](#references)) | [Kalbfleisch & Prentice 2002](#references) |
| Random Survival Forest | RSF | Ensemble | No | No | [scikit-survival](https://scikit-survival.readthedocs.io/) ([Pölsterl 2020](#references)) | [Ishwaran et al. 2008](#references) |
| Gradient Boosted Survival Analysis | GBSA | Ensemble | Yes† | No | [scikit-survival](https://scikit-survival.readthedocs.io/) ([Pölsterl 2020](#references)) | [Hothorn et al. 2006](#references) |
| XGBoost Cox Regression | XGBoost-Cox | Ensemble | Yes‡ | No | [XGBoost](https://xgboost.readthedocs.io/) ([Chen & Guestrin 2016](#references)) | [Chen & Guestrin 2016](#references) |
| XGBoost Accelerated Failure Time | XGBoost-AFT | Ensemble | No | No | [XGBoost](https://xgboost.readthedocs.io/) ([Chen & Guestrin 2016](#references)) | [Barnwal et al. 2022](#references) |
| Fast Survival Support Vector Machine | FSSVM | Kernel | No§ | No§ | [scikit-survival](https://scikit-survival.readthedocs.io/) ([Pölsterl 2020](#references)) | [Pölsterl et al. 2015](#references) |
| DeepSurv | DeepSurv | Neural | Yes‡ | No | [pycox](https://github.com/havakv/pycox) ([Kvamme & Borgan 2021](#references)) | [Katzman et al. 2018](#references) |
| DeepHit | DeepHit | Neural | No | No | [pycox](https://github.com/havakv/pycox) ([Kvamme & Borgan 2021](#references)) | [Lee et al. 2018](#references) |

\*Weibull is the unique distribution under which AFT and PH parameterizations coincide; the AFT-form Weibull is structurally a PH model. †GBSA's default loss `coxph` fits a Cox partial likelihood; with `loss='ipcwls'` the PH assumption no longer applies. ‡XGBoost-Cox and DeepSurv optimize the Cox partial likelihood, so the PH assumption sits in the loss function — the trees and network can partially absorb non-PH effects via non-linear capacity, but mlsurv's tier logic flags them like other Cox-loss models. §FSSVM optimizes a concordance-ranking objective rather than fitting a hazard model; there is no PH or linearity parameterization to violate.

**Columns**: "Yes" / "No" reflect whether mlsurv's tier-recommendation logic demotes the model when the corresponding PH or linearity test detects violations. The flags live on the model class as `assumes_ph` / `assumes_linearity` and feed `PH_ASSUMPTION_MODELS` / `LINEARITY_ASSUMPTION_MODELS` in `mlsurv.constants`.

When proportional hazards or log-linearity violations are detected at `setup()` time, mlsurv folds them into the unified [model recommendation table](pipeline.md#model-recommendations) — assumption-dependent models are demoted from *Recommended* to *Cautionary* to *Not recommended* in proportion to the share of features violating, and unregularized Cox-family models are forced to *Not recommended* outright when `n_features ≥ n_events`. See [Pipeline Configuration — Automated Checks](pipeline.md#automated-setup-checks).

## Complexity Tiers

Each model is assigned a complexity tier that drives the [model recommendation system](pipeline.md#model-recommendations):

| Tier | Models | Characteristics |
|------|--------|-----------------|
| Linear | CoxPH, CoxNet, Weibull AFT, FSSVM | Few parameters, low overfitting risk, interpretable coefficients |
| Tree ensemble | RSF, GBSA, XGBoost-Cox, XGBoost-AFT | Moderate parameters, captures non-linear effects, requires more data |
| Deep learning | DeepSurv, DeepHit | Many parameters, highest capacity, requires large datasets |

FSSVM is listed under the kernel family in the [model inventory](#model-inventory) above, but for recommendation purposes it is treated as linear-tier because its effective complexity (and sample-size requirements) matches linear models rather than tree ensembles or deep learning.

Model recommendations are based on events-per-variable (EPV) and absolute sample size using published thresholds. See [Pipeline Configuration — Model Recommendations](pipeline.md#model-recommendations) for the threshold table.

## Dynamic Defaults

mlsurv adapts hyperparameters to dataset characteristics so that reasonable results are achievable without manual tuning:

- Tree depth and ensemble size are reduced on small datasets
- Neural network width and depth scale with sample size
- Tuning trial counts are sized by model family
- Feature selector dimensionality follows Harrell's rule when set to automatic

Advanced users can override any default by supplying custom parameters to `train()` or custom search spaces to `tune()`:

```python
# Override default hyperparameters at train time
learner.train('rsf', n_estimators=500, max_depth=8, min_samples_leaf=10)

# Override the default tuning search space
learner.tune('rsf', search_space={
    'n_estimators': (100, 1000),
    'max_depth': (3, 12),
    'min_samples_leaf': (5, 50),
}, n_trials=100)
```

See [Model Development & Evaluation — Hyperparameter Tuning](evaluation.md#hyperparameter-tuning).

## Extensibility

mlsurv accepts custom components at each stage:

- **Custom models**: Any estimator implementing the `BaseSurvivalModel` interface (scikit-learn compatible). Tunable by declaring a `get_search_space()` hook or passing `search_spaces=` (see [Tuning Custom Models](evaluation.md#tuning-custom-models))
- **Custom CV strategies**: Any scikit-learn cross-validator or pre-split indices
- **Custom preprocessors**: Any scikit-learn transformer for imputation, scaling, or feature selection
- **Custom metrics**: Any scoring function accepting `(y_true, risk_scores)` for model optimization
- **Custom search spaces**: Optuna search space dictionaries for hyperparameter tuning

See [Vignette 04 — Custom Components](../examples/vignette_04_custom_components.ipynb) for a complete worked example of plugging custom models, preprocessors, and metrics into the mlsurv pipeline.

## References

**Source libraries**
1. Pölsterl S. scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn. *J Mach Learn Res.* 2020;21(212):1-6. [jmlr.org/papers/v21/20-729.html](https://www.jmlr.org/papers/v21/20-729.html)
2. Davidson-Pilon C. lifelines: survival analysis in Python. *J Open Source Softw.* 2019;4:1317. [doi:10.21105/joss.01317](https://doi.org/10.21105/joss.01317)
3. Kvamme H, Borgan Ø. Continuous and discrete-time survival prediction with neural networks. *Lifetime Data Anal.* 2021;27:710-736. [doi:10.1007/s10985-021-09532-6](https://doi.org/10.1007/s10985-021-09532-6)
4. Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. *KDD '16.* 2016:785-794. [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)

**Models**
5. Cox DR. Regression Models and Life-Tables. *J R Stat Soc Ser B.* 1972;34:187-202. [doi:10.1111/j.2517-6161.1972.tb00899.x](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x)
6. Simon N, Friedman J, Hastie T, Tibshirani R. Regularization Paths for Cox's Proportional Hazards Model via Coordinate Descent. *J Stat Softw.* 2011;39:1-13. [doi:10.18637/jss.v039.i05](https://doi.org/10.18637/jss.v039.i05)
7. Kalbfleisch JD, Prentice RL. *The Statistical Analysis of Failure Time Data.* Wiley; 2002. [doi:10.1002/9781118032985](https://doi.org/10.1002/9781118032985)
8. Ishwaran H, Kogalur UB, Blackstone EH, Lauer MS. Random survival forests. *Ann Appl Stat.* 2008;2:841-860. [doi:10.1214/08-AOAS169](https://doi.org/10.1214/08-AOAS169)
9. Hothorn T, Bühlmann P, Dudoit S, Molinaro A, Van Der Laan MJ. Survival ensembles. *Biostatistics.* 2006;7:355-373. [doi:10.1093/biostatistics/kxj011](https://doi.org/10.1093/biostatistics/kxj011)
10. Barnwal A, Cho H, Hocking T. Survival Regression with Accelerated Failure Time Model in XGBoost. *J Comput Graph Stat.* 2022;31:1292-1302. [doi:10.1080/10618600.2022.2067548](https://doi.org/10.1080/10618600.2022.2067548)
11. Pölsterl S, Navab N, Katouzian A. Fast Training of Support Vector Machines for Survival Analysis. In: *Machine Learning and Knowledge Discovery in Databases.* Springer; 2015:243-259. [doi:10.1007/978-3-319-23525-7_15](https://doi.org/10.1007/978-3-319-23525-7_15)
12. Katzman JL, Shaham U, Cloninger A, Bates J, Jiang T, Kluger Y. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Med Res Methodol.* 2018;18:24. [doi:10.1186/s12874-018-0482-1](https://doi.org/10.1186/s12874-018-0482-1)
13. Lee C, Zame W, Yoon J, van der Schaar M. DeepHit: A Deep Learning Approach to Survival Analysis With Competing Risks. *AAAI.* 2018;32. [doi:10.1609/aaai.v32i1.11842](https://doi.org/10.1609/aaai.v32i1.11842)
