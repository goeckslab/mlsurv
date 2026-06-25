# Validation & Benchmarking

External validation and comparative benchmarking are essential steps for assessing whether a model generalizes beyond its training population and whether it adds value over existing baselines<sup>[1](#references),[2](#references)</sup>.

## External Validation

The `validate()` method evaluates all trained models on one or more independent held-out datasets, computing the same discrimination and calibration metrics used during internal evaluation.

```python
# Train on GBSG2, validate on Rotterdam
learner = SurvivalLearner(X_gbsg2, y_gbsg2)
learner.setup(random_state=42)
learner.run(['coxph', 'rsf'])

result = learner.validate(X_rotterdam, y_rotterdam, name='Rotterdam')
print(result.evaluation.c_index_test)   # C-index on external cohort
print(result.bootstrap.summary)         # Bootstrap CIs
```

### Key Features

- **Same metrics**: Discrimination and calibration metrics match internal evaluation for direct comparison
- **Automatic column alignment**: Validation datasets are automatically aligned to the training schema, minimizing manual data wrangling
- **Multiple cohorts**: Call `validate()` repeatedly with different `name` values to validate on several independent datasets
- **Bootstrap CIs**: Confidence intervals on external performance are computed automatically
- **Validation-cohort IPCW**: For Uno's IPCW C-index, `validate()` defaults `ipcw_g_hat_source='test'` so the censoring distribution is estimated on the validation cohort itself rather than the training cohort. This avoids the train-vs-validation exchangeability assumption that would otherwise bias the point estimate when the two cohorts differ in censoring patterns. Pass `ipcw_g_hat_source='train'` to opt back into the legacy training-cohort G_hat. See [Uno's IPCW C-index — estimation choices](evaluation.md#unos-ipcw-c-index--estimation-choices).

### Multiple Validation Cohorts

```python
learner.validate(X_cohort_1, y_cohort_1, name='Hospital A')
learner.validate(X_cohort_2, y_cohort_2, name='Hospital B')
learner.validate(X_cohort_3, y_cohort_3, name='Negative Control')
```

All validation results are included in the generated HTML report.

### Scoring Reference Biomarkers

Like [`benchmark()`](#benchmarking-against-references), `validate()` can score external reference predictors (clinical scores, single biomarkers) head-to-head with the trained models on the validation cohort:

```python
result = learner.validate(
    X_external, y_external, name='MSK',
    references={'LORIS': loris_scores, 'TMB': tmb_values},
    higher_is_worse={'LORIS': False, 'TMB': False},
)
```

Each reference array must be aligned with `X_external` — one value per validation patient. Reference rows are appended to the cohort's concordance / AUC(t) / Brier tables and bootstrap CIs with the same columns as model rows, and flow into the xlsx export. `higher_is_worse` accepts a single bool (applied to every reference) or a per-reference dict; set it to `False` for scores where higher = better outcome (e.g. LORIS, TMB, PD-L1), which are sign-flipped internally. Non-finite reference values are filtered per reference, with a `MethodologicalWarning` reporting the effective sample size.

Unlike `benchmark()`, only array-aligned references are accepted — column-name lookups against `X_external` are not supported, because external cohorts may not share column semantics with the training data.

### Applying the Calibrator to the External Cohort

When a model carries a calibrator (from `train(calibrate=...)`), `validate(X_external, y_external)` applies it to the external cohort by default. The calibrator is a fixed development-cohort transform the external rows never saw, so this is leakage-free, and it shows whether the development-cohort calibration transports to the new cohort. Pass `use_recalibrated=False` to force raw external metrics instead.

## Benchmarking Against References

The `benchmark()` method quantifies the added value of trained models over established baselines such as clinical scores or single biomarkers. Rather than simply comparing C-indices side by side, it computes incremental improvement metrics.

```python
learner.benchmark(
    references={'LORIS': loris_scores, 'TMB': tmb_values},
    higher_is_worse={'LORIS': False, 'TMB': True},
)
```

### Incremental Metrics

| Metric | Description | Ref |
|--------|-------------|-----|
| Delta C-index | Difference in concordance between the mlsurv model and the reference | — |
| NRI | Net Reclassification Improvement — measures how many patients are correctly reclassified into higher or lower risk categories | <sup>[3](#references)</sup> |
| IDI | Integrated Discrimination Improvement — measures the improvement in separation between event and non-event patients | <sup>[3](#references)</sup> |

All metrics include bootstrap confidence intervals and significance testing<sup>[3](#references),[4](#references)</sup>.

### Missing Value Handling

When a reference predictor contains missing values (e.g., PD-L1 is not measured for all patients), mlsurv automatically:

1. Evaluates the reference on its valid subset only
2. Evaluates the mlsurv model on the same subset for fair comparison
3. Annotates the effective sample size for each comparison

### Per-Subpopulation Benchmarking

When subpopulations are configured, incremental metrics are computed within each subpopulation as well as overall.

### Visualization

```python
learner.plot_benchmark_incremental()  # Forest plot of delta-C, NRI, IDI
```

The forest plot displays each reference predictor with its incremental metrics and bootstrap CIs, providing a clear visual summary of added value.

## Workflow Example

A complete validation and benchmarking workflow:

```python
learner = SurvivalLearner(X_train, y_train)
learner.setup(random_state=42)
learner.run(['coxph', 'rsf', 'xgbcox'])

# External validation on independent cohorts
learner.validate(X_cohort_1, y_cohort_1, name='Cohort 1')
learner.validate(X_cohort_2, y_cohort_2, name='Cohort 2')

# Benchmark against established references
learner.benchmark(
    references={'Clinical Score': clinical_scores},
    higher_is_worse={'Clinical Score': False},
)

# Generate report with validation and benchmarking sections
learner.generate_report('report.html')
```

See [Vignette 07 — Immunotherapy Case Study](../examples/vignette_07_immunotherapy_case_study.ipynb) for a complete worked example with seven external cohorts and three reference predictors.

## References

1. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ.* 2024;385:e078378. [doi:10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378)
2. Steyerberg EW, Harrell FE. Prediction models need appropriate internal, internal-external, and external validation. *J Clin Epidemiol.* 2016;69:245-247. [doi:10.1016/j.jclinepi.2015.04.005](https://doi.org/10.1016/j.jclinepi.2015.04.005)
3. Pencina MJ, D'Agostino RB Sr, D'Agostino RB Jr, Vasan RS. Evaluating the added predictive ability of a new marker: from area under the ROC curve to reclassification and beyond. *Stat Med.* 2008;27:157-172. [doi:10.1002/sim.2929](https://doi.org/10.1002/sim.2929)
4. Pencina MJ, D'Agostino RB, Steyerberg EW. Extensions of net reclassification improvement calculations to measure usefulness of new biomarkers. *Stat Med.* 2011;30:11-21. [doi:10.1002/sim.4085](https://doi.org/10.1002/sim.4085)
