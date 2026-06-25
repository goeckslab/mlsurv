# Reporting & Compliance

mlsurv automates methodological checks, comprehensive reporting, and reproducibility through limitation flags, structured HTML reports, and TRIPOD+AI compliance scaffolding.

## Limitation Flags

Throughout the analysis pipeline, mlsurv automatically evaluates 14 limitation flags that surface potential methodological concerns. Each flag provides three components: **what** was detected, **why** it matters, and a **suggested action**.

| # | Flag | Trigger | Risk |
|---|------|---------|------|
| 1 | Small Sample Size | N < 200 (train + test) | Overfitting, unstable performance estimates |
| 2 | Low EPV | EPV < 10 | Biased and unstable coefficient estimates |
| 3 | High Censoring Rate | Censoring > 80% | Reduced effective sample size, biased survival estimates |
| 4 | Extensive Missing Data | Overall > 20% OR any variable > 50% | Bias if not MCAR, imputation assumptions may not hold |
| 5 | PH Violations | Grambsch-Therneau FDR-adjusted p < 0.05 | Cox-family models assume constant hazard ratios; violations bias predictions |
| 6 | Log-Hazard Linearity Violations | RCS Wald FDR-adjusted p < 0.05 | Cox-family models assume linear effects; non-linearity biases coefficients |
| 7 | Train/Test Performance Gap | CV mean - test metric > 0.05 | Suggests overfitting |
| 8 | Low Discrimination | Test C-index < 0.6 | Near-random discrimination; insufficient prognostic information |
| 9 | Poor Calibration | Slope outside [0.8, 1.2] OR GND p < 0.05 | Systematic over- or under-prediction of risk |
| 10 | Substantial Multicollinearity | Pairs with \|R\| >= 0.8, severity above minor | Inflates coefficient variance, distorts importance |
| 11 | No External Validation | No external cohort provided | Internal validation may overestimate performance |
| 12 | Small Subpopulations | Any subpopulation with < 30 events | Unreliable performance estimates |
| 13 | Model Appropriateness | Trained model not recommended for dataset size/EPV | High overfitting risk for complex models |
| 14 | IPCW Training-Support Violation | Uno's C-index, AUC(t), or IBS evaluated beyond reliable censoring support: observations past `train_max`, min(Ĝ) < 0.05, or too few events/censored to estimate Ĝ | IPCW weights `1/Ĝ(t)²` blow up, so these metrics are undefined or unstable in the right tail |

### Warning Tiers

Flags are delivered through a three-tier warning system:

| Tier | Behavior | Use Case |
|------|----------|----------|
| Methodological | Always surfaces to the user | Critical methodological concerns (PH violations, leakage, low EPV) |
| Suggestion | Appears once per issue | Actionable recommendations (feature reduction, model alternatives) |
| Internal | Suppressed by default | Debugging information, re-enable with `enable_internal_warnings()` |

Each tier is a public warning category in `mlsurv.exceptions`, so you can silence
or escalate any of them with the standard `warnings` module. See
[Filtering warnings](warnings.md).

## HTML Report

A single `generate_report()` call produces a standalone interactive HTML report:

```python
learner.generate_report('report.html')
```

### Output Formats

The output format is inferred from the file extension:

| Extension | Best for | Figures |
|-----------|----------|---------|
| `.html` | Interactive human viewing | Interactive Plotly charts embedded in one standalone file |
| `.md` | Programmatic or AI-agent consumption, and pandoc to PDF/Word | Static PNGs in an adjacent `images/` folder |

```python
learner.generate_report('report.html')   # interactive, for people
learner.generate_report('report.md')     # plain text, for agents or pandoc
```

For an AI agent (or any automated step) that reads the report back, prefer `.md`. It is plain text, much smaller than the HTML (which embeds every figure inline), and easy to read or grep without first stripping embedded JSON, CSS, and scripts.

### Report Sections

The report contains up to 12 sections, populated based on the analyses performed:

| # | Section | Content |
|---|---------|---------|
| 1 | Executive Summary | Model rankings, best model, key metrics |
| 2 | Data & Preprocessing | Dataset overview, missing data, feature selection, preprocessing steps |
| 3 | Model Development | Training configuration, CV results, tuning history |
| 4 | Model Evaluation | Discrimination and calibration metrics, risk stratification |
| 5 | Feature Analysis | Permutation importance, SHAP, interactions |
| 6 | Subpopulation Analysis | Per-subpopulation metrics with bootstrap CIs |
| 7 | External Validation | Validation cohort results |
| 8 | Benchmarking | Incremental metrics vs reference predictors |
| 9 | Limitations & Caveats | Triggered limitation flags with explanations |
| 10 | Environment & Reproducibility | Python version, package versions, random seeds |
| 11 | References | Literature citations for methods used |
| 12 | TRIPOD+AI Compliance | Pre-populated checklist (see below) |

The patient report shows a **calibration banner** naming the calibration method when the shown model is calibrated (from `train(calibrate=...)`).

Visualizations are rendered as interactive Plotly figures embedded in the report. The report is fully standalone — no external dependencies are needed to view it.

See [Vignette 01 — Quickstart](../examples/vignette_01_quickstart.ipynb) for the minimal `generate_report()` path, and [Vignette 07 — Immunotherapy Case Study](../examples/vignette_07_immunotherapy_case_study.ipynb) for a full report with all 12 sections, triggered limitation flags, and TRIPOD+AI coverage.

### Reproducing Report Figures

Each section has a **Programmatic Access** dropdown listing the `learner` call that reproduces every figure, so you can regenerate and export any figure outside the report. The figures come straight from the learner — no report object is needed, and the calls survive `save_learner()` / `load_learner()`. Use Plotly's native `write_image()` to save:

```python
fig = learner.plot_metric(style='raincloud', source='cv', metric='harrell')
fig.write_image('cv_raincloud.svg')   # .svg, .png, or .pdf
```

The same dropdown shows how to pull the tidy data behind a figure via `return_data=True`. Most plots return a single `DataFrame` (a few multi-panel plots return a dict of frames), which you can write to CSV:

```python
df = learner.plot_metric(style='raincloud', source='cv', metric='harrell', return_data=True)
df.to_csv('cv_raincloud.csv')
```

## TRIPOD+AI Compliance

The report includes a pre-populated TRIPOD+AI compliance checklist mapped against the 52-item reporting standard for AI-based prediction models<sup>[1](#references)</sup>.

| Category | Count | Description |
|----------|------:|-------------|
| Auto-filled | 18 | Populated from analysis metadata (sample size, model type, metrics, software version) |
| Scaffolded | 10 | Guidance provided referencing the relevant report section |
| Researcher-provided | 22 | Flagged for the researcher (clinical rationale, intended use, ethical approval) |
| Not applicable | 2 | Model updating/recalibration (outside mlsurv scope) |

This reduces the reporting burden from a blank 52-item checklist to 32 items requiring researcher input, 10 of which are pre-scaffolded with specific guidance.

## Data Export

### Structured Results

`export_results()` writes all results to spreadsheet or CSV formats:

```python
learner.export_results('results.xlsx')                  # Excel workbook (default format)
learner.export_results('results_dir', format='csv')     # directory of CSVs, one per table
```

The Excel workbook contains separate sheets for concordance metrics, bootstrap summaries, feature importance, subpopulation results, and validation metrics.

### Session Serialization

`save_learner()` serializes the full analysis state for later reproduction:

```python
learner.save_learner('analysis.pkl')

# Later: reload and resume
from mlsurv import load_learner
learner = load_learner('analysis.pkl')
learner.predict(new_data, models='coxph')
```

### Model Artifacts

For deployment scenarios, `save_artifact()` exports a lightweight model pipeline with metadata:

```python
learner.save_artifact('coxph', 'model_artifact/', step='train')
```

The `step` argument records which pipeline step produced the artifact (`'train'`, `'tune'`, or `'run'`) and is written into the artifact metadata.

See the [CLI Guide](CLI.md) for command-line equivalents of all export operations.

## References

1. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ.* 2024;385:e078378. [doi:10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378)
