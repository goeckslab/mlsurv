# mlsurv CLI Guide

mlsurv provides a command-line interface for survival analysis workflows.
The CLI is designed for scripting, HPC pipelines, and quick one-off analyses
without writing Python code.

## Installation

```bash
pip install mlsurv

# Optional: shell completion (Bash/Zsh)
pip install mlsurv[completion]
eval "$(register-python-argcomplete mlsurv)"
```

## Quick Start

```bash
# Run a full comparison pipeline on a CSV file
mlsurv run patients.csv --time-col OS_MONTHS --event-col OS_STATUS \
    --models coxph rsf gbsa --output report.html

# List available models
mlsurv models --verbose
```

Every command accepts `--help` for full usage details:

```bash
mlsurv run --help
mlsurv evaluate --help
```

## Core Workflow

The typical analysis pipeline follows these steps:

```
setup --> train --> tune --> evaluate --> bootstrap --> predict
                      \                      |
                       \----> run (all-in-one shortcut)
```

### 1. Inspect your data

```bash
mlsurv setup data.csv --time-col time --event-col status
```

Reports sample counts, feature count, event rate, missing values, and
train/test split sizes.

### 2. Train a model

```bash
mlsurv train data.csv --time-col time --event-col status --model rsf
```

Trains with default hyperparameters and reports cross-validated concordance
index.

### 3. Tune hyperparameters

```bash
mlsurv tune data.csv --time-col time --event-col status \
    --model rsf --n-trials 100
```

Uses Optuna to search hyperparameter space. Reports the best parameters and
CV score.

### 4. Evaluate on the test set

```bash
mlsurv evaluate data.csv --time-col time --event-col status \
    --model rsf --shap --report eval_report.html \
    --save-learner session.learner
```

Computes concordance index, Brier score, feature importance, and optionally
SHAP values. Use `--save-learner` to enable downstream commands (statistical
tests, explain, plot).

### 5. Bootstrap confidence intervals

```bash
mlsurv bootstrap data.csv --time-col time --event-col status \
    --model rsf --n-bootstrap 200
```

Resamples the test set to produce confidence intervals on all metrics.

### 6. Predict on new patients

```bash
mlsurv predict new_patients.csv --load-learner session.learner \
    --times 12 24 36 --ci
```

Generates risk scores and survival/risk probabilities at specified time
horizons. Use `--ci` with `--load-learner` for bootstrap confidence
intervals on predictions.

### 7. All-in-one shortcut

```bash
mlsurv run data.csv --time-col time --event-col status \
    --models coxph rsf gbsa --compare --output report.html
```

Runs the full pipeline (train, tune, evaluate, bootstrap) for all specified
models and produces a comparison report.

## Subcommand Reference

### Core Workflow

#### `run`

Full pipeline: train all models, optionally tune, evaluate, and bootstrap.

```bash
mlsurv run data.csv --models coxph rsf gbsa \
    --n-bootstrap 200 --compare --output report.html
```

Key flags:
- `--models` -- Models to compare (default: coxph rsf)
- `--n-bootstrap` -- Bootstrap iterations (default: auto)
- `--compare` -- Print paired model comparison table
- `--compare-output FILE` -- Save comparison CSV
- `--resume` -- Resume existing Optuna studies (requires `--experiment-name`)
- `--output` -- HTML report path

#### `train`

Train a single model with default hyperparameters.

```bash
mlsurv train data.csv --time-col time --event-col status --model coxph
```

Key flags:
- `--model` -- Model name (required)
- `--cv-folds` -- Number of CV folds (default: 5)
- `--model-params FILE` -- JSON file of constructor kwargs, nested as
  `{model_name: {param: value}}`, applied to the untuned model

#### `tune`

Hyperparameter optimization via Optuna.

```bash
mlsurv tune data.csv --model rsf --n-trials 100 --timeout 3600
```

Key flags:
- `--model` -- Model to tune (required)
- `--n-trials` -- Number of Optuna trials (default: 50)
- `--timeout` -- Maximum tuning time in seconds
- `--resume` -- Resume an existing study
- `--experiment-name` -- Name for Optuna study (required for `--resume`)
- `--search-spaces` -- Path to JSON file with custom search spaces
- `--optuna-verbose` -- Show trial-by-trial logs

#### `evaluate`

Evaluate a trained model on the test set.

```bash
mlsurv evaluate data.csv --model rsf --shap --bootstrap \
    --interactions --report eval.html --save-learner session.learner
```

Key flags:
- `--model` -- Model to evaluate (required)
- `--shap` -- Compute SHAP values
- `--bootstrap` -- Also run bootstrap CIs
- `--n-bootstrap` -- Bootstrap iterations (default: auto)
- `--n-importance-repeats` -- Permutation importance repeats (default: 10)
- `--interactions` -- Compute pairwise feature interactions
- `--interaction-top-k` -- Features for interaction testing (default: 5)
- `--subgroups COLUMN` -- Subpopulation analysis by column(s)
- `--eval-times` -- Time points for time-dependent metrics
- `--plot TYPE` -- Generate inline plots (e.g., `--plot metric importance roc`)
- `--report` -- HTML report path
- `--save-learner` / `--load-learner` -- Save/load full learner state
- Statistical test flags: `--ph-test`, `--logrank-test`, `--rmst-test`,
  `--importance-test`, `--linearity-test`

#### `calibrate`

Attach an `S(t|x)` calibrator to one or more trained models, using the same path as `train(calibrate=...)`. It supports all three methods (`isotonic`, `lowdof`, `breslow_slope`). Its main use is calibrating an already-saved learner from the command line; load it with `--load-learner` (recommended) or train the named models from scratch.

```bash
mlsurv calibrate --load-learner session.learner --models rsf coxph \
    --method isotonic --save-learner session.learner
```

Key flags:
- `--load-learner FILE` -- Load a saved learner (skips data loading and training)
- `--models` -- Models to calibrate (default: all trained on the loaded learner)
- `--method` -- Calibration method (required): `isotonic`, `lowdof`, or `breslow_slope`. `breslow_slope` is only valid for score-based models (a single linear-predictor risk score); use `isotonic` or `lowdof` otherwise.
- `--output` -- CSV summary file (one row per calibrated model)
- `--save-learner` -- Persist the calibrators back to the learner file

The calibrator is fit on held-out OOF of the training cohort and attached to `result.calibrator`, so it rides on the model and round-trips through `--save-learner`.

#### `bootstrap`

Bootstrap confidence intervals on the test set.

```bash
mlsurv bootstrap data.csv --model rsf --n-bootstrap 500 --plot metric
```

Key flags:
- `--model` -- Model to bootstrap (required)
- `--n-bootstrap` -- Number of iterations (default: auto)
- `--eval-times` -- Time points for time-dependent metrics
- `--plot TYPE` -- Generate inline plots after bootstrap

#### `validate`

External validation on an independent cohort.

```bash
mlsurv validate train.csv --model coxph \
    --validation-data external.csv --cohort-name "Rotterdam"
```

Key flags:
- `--model` -- Model to validate (required)
- `--validation-data` -- Path to external validation CSV (required)
- `--cohort-name` -- Display name for the external cohort (default: External)
- `--val-time-col`, `--val-event-col` -- Column names in validation data
  (default: `time` / `event`). Set these explicitly when your validation
  data uses different column names.
- `--val-time-unit` -- Time unit for validation data (defaults to `--time-unit`)
- `--val-event-values` -- Event indicator values in validation data
- `--bootstrap` / `--no-bootstrap` -- Toggle bootstrap CIs (default: enabled)
- `--references-file FILE` -- CSV of reference biomarker scores (one row per
  validation patient, same order as `--validation-data`); reference rows are
  scored alongside the models
- `--reference-cols` -- Columns to use from the references file (default: all)
- `--higher-is-worse` / `--no-higher-is-worse` -- Whether higher reference
  values mean worse outcome. Use `--no-higher-is-worse` for scores where higher
  = better (e.g. LORIS, TMB, PD-L1); they are sign-flipped internally (default:
  `--higher-is-worse`)

#### `predict`

Generate risk scores and survival/risk predictions for new patients.

```bash
# From a saved artifact (no CI support)
mlsurv predict patients.csv --load-artifact artifact_dir/ --times 12 24

# From a saved learner (with CI support)
mlsurv predict patients.csv --load-learner session.learner \
    --times 12 24 --ci --n-bootstrap 200

# All trained models at once (omit --model): combined table,
# one block of rows per model, model_name as the first column
mlsurv predict patients.csv --load-learner session.learner --times 12 24
```

Key flags:
- `--load-artifact DIR` -- Load model artifact (no CI support)
- `--load-learner FILE` -- Load full learner state (enables `--ci`)
- `--model` -- Model name; omit to predict with all trained models
  (combined table when several are trained)
- `--times` -- Time horizons for survival/risk columns
- `--ci` -- Compute bootstrap CIs (requires `--load-learner`)
- `--ci-band` -- Also retain bootstrap survival CI bands for plotting
  (requires `--ci`)
- `--n-bootstrap` -- CI iterations (default: 200)
- `--ci-level` -- Confidence level (default: 0.95)
- `--strict` -- Exit with error if any selected model fails to predict or
  cannot produce the requested survival columns; with several models, any
  one model failing fails the command
- `--output` -- Output CSV path

Output columns: `sample_idx`, `model_name`, `risk_score`, `median_survival`,
`survival_<time>`, `risk_<time>`. With `--ci`: additional `*_lower`/`*_upper`
columns. With several models, `model_name` moves to the front and the table
stacks one block per model in CV-sorted order (best first). A model that
cannot produce the survival columns keeps its `risk_score` rows with the
survival columns left empty; a model whose prediction fails entirely is
omitted from the table with a warning (either case fails the command under
`--strict`).

#### `setup`

Inspect dataset and data split/CV configuration without training.

```bash
mlsurv setup data.csv --time-col time --event-col status --cv-folds 10
```

### Analysis

#### `benchmark`

Compare trained models against external reference scores.

```bash
mlsurv benchmark data.csv --reference-file published_scores.csv \
    --models coxph rsf --n-bootstrap 200 --subgroups stage
```

Key flags:
- `--reference-file` -- CSV with reference score columns (required)
- `--reference-cols` -- Specific columns to use (default: all)
- `--models` -- Models to compare (default: coxph rsf)
- `--dataset` -- Data split to score references on: test, train, or full
  (default: test). Models always use the held-out test set.
- `--n-bootstrap` / `--no-bootstrap` -- Bootstrap CIs
- `--subgroups` -- Subpopulation analysis

#### `sweep`

Parameter sweep: vary one parameter and measure CV performance.

```bash
mlsurv sweep data.csv --models rsf gbsa \
    --param n_estimators --values 50 100 200 500 --include-test
```

Key flags:
- `--models` -- Models to sweep (required)
- `--param` -- Parameter name (required)
- `--values` -- Values to try (required)
- `--target` -- Pipeline component: model, imputer, scaler, or
  feature_selector (default: model)
- `--include-test` -- Also evaluate on the test set
- `--n-bootstrap` -- Bootstrap iterations for test CIs (requires
  `--include-test`)

#### `explain`

Patient-level SHAP waterfall explanation.

```bash
# Explain a test-set patient by index
mlsurv explain --load-learner session.learner --patient 0 \
    --plot explanation.html

# Explain a new patient from CSV
mlsurv explain --load-learner session.learner --patient-data patient.csv
```

Key flags:
- `--load-learner FILE` -- Saved learner (required)
- `--patient N` or `--patient-data PATH` -- Patient to explain
  (mutually exclusive, one required)
- `--model` -- Model name (auto-resolved if only one trained)
- `--n-features` -- Features in waterfall plot (default: 15)
- `--max-background` -- Background samples for SHAP (default: 50)
- `--plot PATH` -- Save waterfall plot to HTML
- `--output PATH` -- Save SHAP values to CSV

#### `patient-report`

Standalone interactive single-patient HTML report: individualized survival
curve with CI band, milestone probabilities, SHAP waterfall, and a similar-risk
training-cohort anchor.

```bash
# Report on a test-set patient by index
mlsurv patient-report --load-learner session.learner --patient 0 \
    --output report.html

# Report on the highest-risk test-set patient
mlsurv patient-report --load-learner session.learner --patient high_risk \
    --output report.html

# Report on a new patient from CSV
mlsurv patient-report --load-learner session.learner \
    --patient-data patient.csv --output report.html
```

Key flags:
- `--load-learner FILE` -- Saved learner (required)
- `--patient INDEX_OR_SPEC` or `--patient-data PATH` -- Patient to report on
  (mutually exclusive, one required). `INDEX_OR_SPEC` is a 0-based test-set
  index or a risk-percentile spec (`low_risk`/`p10`, `high_risk`/`p90`,
  `median`/`p50`, `pNN`)
- `--output PATH` -- Write the HTML report here (required)
- `--model` -- Model name (auto-resolved if only one trained)
- `--times` -- Milestone times for P(survive > t) (default: from time unit)
- `--time-unit` -- Outcome time unit (days/weeks/months/years)
- `--n-neighbors` -- Neighbors for the cohort anchor (default: 10)
- `--n-bootstrap` -- Bootstrap iterations for CIs (default: 200)
- `--no-ci` -- Skip bootstrap confidence intervals (faster)
- `--no-ci-band` -- Skip the shaded CI band around the survival curve
- `--no-recalibrated` -- Show raw S(t|x) instead of the recalibrated curve
- `--offline` -- Embed plotly.js (works without internet)
- `--random-state` -- Seed for bootstrap reproducibility

### Statistical Tests

All statistical test commands require `--load-learner`. They can also be run
inline as flags on `evaluate` (e.g., `mlsurv evaluate ... --ph-test`).

#### `ph-test`

Proportional hazards test (Schoenfeld residuals).

```bash
mlsurv ph-test --load-learner session.learner --output ph_results.csv
```

Key flags:
- `--time-transform` -- rank, km, identity, or log (default: rank)

#### `logrank-test`

Log-rank test across risk groups.

```bash
mlsurv logrank-test --load-learner session.learner --n-groups 3
```

Key flags:
- `--model` -- Model for risk groups (default: all trained models)
- `--n-groups` -- Number of risk groups (default: 2)
- `--p-adjust` -- P-value adjustment: holm, bonferroni, bh, none (default: holm)

#### `rmst-test`

Restricted mean survival time comparison across risk groups.

```bash
mlsurv rmst-test --load-learner session.learner --n-groups 3 --tau 60
```

Key flags:
- `--model` -- Model for risk groups (default: all trained models)
- `--n-groups` -- Number of risk groups (default: 2)
- `--tau` -- RMST time horizon (default: auto-computed)
- `--p-adjust` -- P-value adjustment (default: holm)

#### `importance-test`

Feature importance p-values via PIMP or bootstrap permutation.

```bash
mlsurv importance-test --load-learner session.learner \
    --method pimp --n-permutations 200 --n-jobs -1
```

Key flags:
- `--model` -- Model to test (default: best model)
- `--method` -- pimp or bootstrap (default: pimp)
- `--n-permutations` -- Response permutations for PIMP (default: 100)
- `--n-importance-repeats` -- Importance repeats per model (default: 10)
- `--p-adjust` -- P-value adjustment (default: bh)

#### `linearity-test`

Test the log-hazard linearity of continuous features (spline-based joint Wald).

```bash
mlsurv linearity-test --load-learner session.learner --n-knots 4
```

Key flags:
- `--n-knots` -- Spline knots for the test (default: 4)

#### `interactions`

Pairwise feature interactions for a trained model. Requires a learner saved
after `evaluate`.

```bash
mlsurv interactions --load-learner session.learner --model rsf --top-k 5
```

Key flags:
- `--model` -- Model to analyse (default: the single trained model)
- `--method` -- hstat (model-agnostic) or shap (tree models only) (default: hstat)
- `--top-k` -- Top features by importance to test (default: 5)
- `--max-samples` -- Samples subsampled from the test set (default: 100)
- `--grid-resolution` -- Grid points per feature for PDP surfaces (default: 25)

### Visualization

#### `plot`

Generate visualizations from a saved learner session.

```bash
# List available plot types
mlsurv plot --list

# Kaplan-Meier survival curves
mlsurv plot survival --load-learner session.learner --n-groups 3

# Forest plot of metrics
mlsurv plot metric --load-learner session.learner --style forest

# Feature importance bar chart
mlsurv plot importance --load-learner session.learner --n-features 15

# SHAP beeswarm
mlsurv plot shap --load-learner session.learner --kind beeswarm

# Export plot data instead of a figure
mlsurv plot metric --load-learner session.learner --return-data -o data.csv
```

Key flags:
- `--load-learner FILE` -- Saved learner (required)
- `--model NAME` -- Model(s) to include (default: all trained)
- `-o PATH` / `--output PATH` -- Output file path
- `-f FORMAT` / `--format FORMAT` -- html, png, or json (default: html)
- `--no-open` -- Do not open HTML in browser
- `--return-data` -- Output data (CSV/JSON) instead of a figure
- `--list` -- List available plot types

Available plot types (23):

| Type | Requires | Description |
|------|----------|-------------|
| `survival` | train | Kaplan-Meier curves stratified by risk score |
| `metric` | train | Metric comparison (forest, bar, raincloud, boxplot) |
| `importance` | evaluate | Feature importance (bar, forest, heatmap) |
| `roc` | evaluate | Time-dependent ROC curve |
| `calibration` | train | Calibration curve (predicted vs. observed) |
| `missing` | setup | Missing data heatmap |
| `diagnostics` | setup | Diagnostic residual plots |
| `coefficients` | train | Feature coefficients from Cox models |
| `hazard-ratios` | train | Hazard ratios (univariate or multivariate) |
| `feature-correlation` | setup | Feature correlation matrix |
| `feature-distributions` | setup | Feature distributions (train/test) |
| `metric-time` | evaluate | Time-dependent metric curves (AUC over time) |
| `shap` | evaluate | SHAP importance or beeswarm |
| `interactions` | evaluate | Feature interaction heatmap |
| `interaction-pdp` | evaluate | 2D partial dependence for a feature pair |
| `calibration-overlay` | evaluate | Predicted vs. observed survival overlay |
| `optuna` | tune | Optuna optimization results |
| `metric-heatmap` | evaluate | Multi-metric heatmap across models |
| `hazard-ratios-comparison` | train | Univariate vs. multivariate comparison |
| `coefficient-comparison` | train | Coefficient scatter across Cox models |
| `benchmark-incremental` | benchmark | Incremental metrics forest plot |
| `benchmark-scatter` | benchmark | Model risk scores vs. reference scores |
| `validation-heatmap` | validate | Cross-cohort validation metric heatmap (2+ cohorts) |

### Artifact Management

#### `artifact-inspect`

View artifact metadata without loading the model.

```bash
mlsurv artifact-inspect artifact_dir/
mlsurv artifact-inspect artifact_dir/ --json
```

#### `artifact-upgrade`

Upgrade artifact metadata to the current schema version.

```bash
# Preview changes
mlsurv artifact-upgrade artifact_dir/ --dry-run

# Upgrade in place
mlsurv artifact-upgrade artifact_dir/

# Upgrade to a new directory
mlsurv artifact-upgrade artifact_dir/ --output upgraded_dir/
```

### Utilities

#### `models`

List available model names.

```bash
mlsurv models
mlsurv models --verbose    # includes tunable status
```

#### `export-study`

Export Optuna tuning history to CSV or JSON.

```bash
mlsurv export-study --experiment my_experiment --model rsf --output trials.csv
mlsurv export-study --study-name custom_study --storage sqlite:///optuna.db \
    --output trials.json
```

#### `list-studies`

List Optuna tuning studies and where their `.db` files live. The table output
prints the resolved experiment directory plus a copy-pasteable
`optuna-dashboard sqlite:///...` command per study.

```bash
mlsurv list-studies                          # all experiments under ./optuna_studies
mlsurv list-studies --experiment my_experiment
mlsurv list-studies --storage sqlite:///optuna.db   # query a backend
mlsurv list-studies --format json            # machine-readable (also: csv)
```

#### `export-results`

Export all available result tables (CV metrics, evaluation, bootstrap, feature
analysis, statistical tests, ...) from a saved learner to a single Excel
workbook or a directory of CSV files.

```bash
mlsurv export-results results.xlsx --load-learner session.learner
mlsurv export-results results_dir/ --load-learner session.learner --format csv
```

Key flags:
- `--load-learner FILE` -- Saved learner (required)
- `--format` -- xlsx or csv (default: inferred from the output extension)
- `--audit` -- Include the full sheet-name truncation mapping (xlsx only)

## Persistence: Artifacts vs. Learners

mlsurv offers two persistence mechanisms with different trade-offs:

| Feature | `--save-artifact` | `--save-learner` |
|---------|-------------------|------------------|
| Contents | Model pipeline + metadata | Entire session (data, models, results) |
| File size | Small (KB-MB) | Large (can be GB+) |
| Predict | Yes | Yes |
| Prediction CIs (`--ci`) | No | Yes |
| Statistical tests | No | Yes |
| SHAP explain | No | Yes |
| Plot generation | No | Yes |
| Use case | Deployment, sharing | Analysis, session resumption |

**Recommendation**: Use `--save-learner` during analysis (enables all
downstream commands). Use `--save-artifact` for deploying a trained model
to production.

```bash
# Save learner for analysis workflow
mlsurv evaluate data.csv --model rsf --save-learner session.learner

# Save artifact for deployment
mlsurv train data.csv --model rsf --save-artifact rsf_artifact/
```

## Workflows

### Full analysis pipeline

Train, evaluate, and run statistical tests in a multi-step workflow using
a saved learner:

```bash
# Step 1: Train, evaluate, and save the learner
mlsurv evaluate data.csv --time-col time --event-col status \
    --model rsf --shap --bootstrap --n-bootstrap 200 \
    --save-learner session.learner --report eval.html

# Step 2: Run statistical tests
mlsurv ph-test --load-learner session.learner
mlsurv logrank-test --load-learner session.learner --n-groups 3
mlsurv importance-test --load-learner session.learner --n-jobs -1

# Step 3: Explain individual patients
mlsurv explain --load-learner session.learner --patient 0 \
    --plot waterfall.html

# Step 4: Generate publication-quality plots
mlsurv plot survival --load-learner session.learner -o km_curves.html
mlsurv plot metric --load-learner session.learner --style forest -o forest.html
mlsurv plot shap --load-learner session.learner --kind beeswarm -o shap.html
```

### Artifact-based prediction

Train a model, save the artifact, and predict on new data:

```bash
# Train and save artifact
mlsurv train data.csv --time-col time --event-col status \
    --model coxph --save-artifact coxph_artifact/

# Predict risk scores
mlsurv predict new_patients.csv --load-artifact coxph_artifact/

# Predict survival at specific time horizons
mlsurv predict new_patients.csv --load-artifact coxph_artifact/ --times 12 24 36

# Inspect artifact metadata
mlsurv artifact-inspect coxph_artifact/
```

### Resumable tuning

Long-running tuning jobs can be interrupted and resumed:

```bash
# Start tuning with a named experiment
mlsurv tune data.csv --model rsf --n-trials 200 \
    --experiment-name my_rsf_study --timeout 3600

# Resume later (adds more trials to the same study)
mlsurv tune data.csv --model rsf --n-trials 100 \
    --experiment-name my_rsf_study --resume

# Export tuning history
mlsurv export-study --experiment my_rsf_study --model rsf --output trials.csv
```

Note: `--experiment-name` is required for `--resume` to locate the prior study.

### External validation

Validate a model trained on one cohort against an independent cohort:

```bash
mlsurv validate train_data.csv --time-col OS_MONTHS --event-col OS_STATUS \
    --model coxph \
    --validation-data external_cohort.csv \
    --val-time-col surv_time --val-event-col event_flag \
    --cohort-name "TCGA" \
    --n-bootstrap 200 --report validation.html
```

If the validation data uses different column names or time units, specify
them with `--val-*` flags. Note: `--val-time-col` and `--val-event-col`
default to `time` and `event` (not to `--time-col`/`--event-col`), so set
them explicitly when your validation file uses non-default column names.

### Model comparison

Compare multiple models with paired statistical tests:

```bash
mlsurv run data.csv --models coxph coxnet rsf gbsa xgbcox \
    --n-bootstrap 500 --compare --compare-output comparison.csv \
    --output full_report.html
```

The `--compare` flag prints a paired comparison table showing which models
are statistically different. `--compare-output` saves the table to CSV.

## Scripting and Automation

### Machine-readable output

Use `--format` and `--quiet` for automation:

```bash
# JSON output for parsing
mlsurv train data.csv --model rsf --format json -q

# CSV output for downstream processing
mlsurv evaluate data.csv --model rsf --format csv -q --output results.csv

# Pipe to jq
mlsurv models --format json -q | jq '.[].name'
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (unexpected failure) |
| 2 | Usage error (invalid arguments) |
| 3 | Data error (file not found, parse failure, validation) |
| 4 | Computation error (model training or evaluation failure) |
| 130 | Interrupted (Ctrl+C) |

Use exit codes in scripts:

```bash
mlsurv train data.csv --model rsf -q
rc=$?
if [ $rc -eq 3 ]; then
    echo "Data error -- check file format"
elif [ $rc -eq 4 ]; then
    echo "Training failed -- check model configuration"
fi
```

### Shell completion

Install optional completion support:

```bash
pip install mlsurv[completion]

# Bash/Zsh: add to your shell profile
eval "$(register-python-argcomplete mlsurv)"
```

Then use Tab to complete subcommands and flags:

```bash
mlsurv ev<Tab>        # completes to 'evaluate'
mlsurv evaluate --sh<Tab>  # completes to '--shap'
```

## Data Format

### Basic CSV

The simplest input is a CSV with numeric features, a time column, and an
event column:

```csv
age,tumor_size,grade,time,event
65,3.2,2,24.5,1
58,1.8,1,36.0,0
72,4.1,3,12.3,1
```

### Column specification

```bash
# Default column names (time, event)
mlsurv train data.csv --model coxph

# Custom column names
mlsurv train data.csv --time-col OS_MONTHS --event-col OS_STATUS --model coxph

# Custom event values (e.g., "deceased" and "progressed" count as events)
mlsurv train data.csv --event-col vital_status \
    --event-values deceased progressed --model coxph
```

### Date-based survival time

When your data has start/end dates instead of pre-computed survival time:

```bash
mlsurv train data.csv --start-col diagnosis_date --end-col last_followup \
    --time-unit months --event-col status --model coxph
```

Supported `--time-unit` values: `days` (default), `months`, `years`.

### Preprocessing options

```bash
mlsurv train data.csv --model rsf \
    --imputer knn \
    --scaler robust \
    --feature-selector topk \
    --scoring ipcw
```

Available presets:
- `--imputer`: median (default), mean, knn, iterative, none
- `--scaler`: standard (default), robust, minmax, pca, none
- `--feature-selector`: none (default), topk, vif, correlation
- `--scoring`: harrell (default), ipcw, integrated_brier, ctd

## Global Flags

These flags are available on all (or most) subcommands:

| Flag | Description |
|------|-------------|
| `-q` / `--quiet` | Suppress non-essential output |
| `--verbose` | Print detailed output |
| `--format` | Output format: table, json, or csv |
| `--random-state` | Random seed (default: 42) |
| `--n-jobs` | Parallel jobs (default: 1; -1 for all cores) |
| `--device` | Device for deep learning: cpu, cuda, auto |

## Troubleshooting

### "Column not found" errors

Verify your column names match the CSV headers exactly:

```bash
# Check what columns exist
head -1 data.csv

# Specify the correct names
mlsurv train data.csv --time-col "Survival Time" --event-col "Event Status" \
    --model coxph
```

### Time unit mismatch

If survival curves or predictions look wrong, check that `--times` values
match the time unit in your training data:

```bash
# If training data has time in months
mlsurv predict patients.csv --load-learner session.learner --times 6 12 24
# These are months, not days
```

### Artifact version mismatch

If you get schema errors loading an old artifact:

```bash
# Check the artifact version
mlsurv artifact-inspect old_artifact/

# Upgrade to current schema
mlsurv artifact-upgrade old_artifact/ --output upgraded_artifact/
```

### Model training failures

Some models have specific requirements:

- **deepsurv/deephit**: Require PyTorch. Install with `pip install mlsurv[deep]`.
- **fssvm**: Slow on large datasets (>5000 samples). Consider using `--n-jobs -1`.
- **xgbcox/xgbaft**: Require XGBoost. Install with `pip install mlsurv[xgboost]`.

### Prediction CI errors

The `--ci` flag requires `--load-learner`, not `--load-artifact`:

```bash
# This will fail:
mlsurv predict data.csv --load-artifact dir/ --ci

# Use --load-learner instead:
mlsurv predict data.csv --load-learner session.learner --ci
```

### Resume not finding prior study

The `--resume` flag requires `--experiment-name` to locate the prior Optuna
study:

```bash
# This warns and starts fresh:
mlsurv tune data.csv --model rsf --resume

# This resumes correctly:
mlsurv tune data.csv --model rsf --resume --experiment-name my_study
```

## Available Models

| Name | Description |
|------|-------------|
| `coxph` | Cox proportional hazards |
| `coxnet` | Elastic-net penalized Cox |
| `rsf` | Random survival forest |
| `gbsa` | Gradient boosted survival analysis |
| `fssvm` | Fast survival SVM |
| `xgbcox` | XGBoost Cox |
| `xgbaft` | XGBoost accelerated failure time |
| `weibull_aft` | Weibull accelerated failure time |
| `deepsurv` | Deep survival network (requires PyTorch) |
| `deephit` | DeepHit competing risks (requires PyTorch) |
