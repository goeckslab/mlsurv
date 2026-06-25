<p align="center">
  <img src="https://raw.githubusercontent.com/goeckslab/mlsurv/main/docs/logo/mlsurv_icon.png" alt="mlsurv logo" width="120">
</p>

<h1 align="center">mlsurv</h1>

<p align="center">
  <em>Low-code machine learning package for survival analysis</em>
</p>

---

## Intro to mlsurv

Survival analysis in clinical research typically requires stitching together various software tools for modeling, visualization, tuning, validation, and reporting. mlsurv unifies these processes into a single package: train, tune, and validate 10 survival models, generate publication-ready reports, and test on independent cohorts — all in under 10 lines of code.

<p align="center">
  <img src="https://raw.githubusercontent.com/goeckslab/mlsurv/main/docs/overview.png" alt="mlsurv overview" width="600">
</p>

## Quick Start

```python
from mlsurv import SurvivalLearner
from mlsurv.utils import load_survival_data

# Load data (pandas DataFrame + structured array with 'event' and 'time' fields)
X, y = load_survival_data('data.csv', time_col='OS_MONTHS', event_col='OS_STATUS')

# Train, tune, evaluate, and bootstrap all models
learner = SurvivalLearner(X, y)
learner.setup(random_state=42)
learner.run()

# Generate a publication-ready interactive HTML report
learner.generate_report('report.html')
```

See [Vignette 01](https://github.com/goeckslab/mlsurv/blob/main/examples/vignette_01_quickstart.ipynb) for a complete walkthrough.

### Accessing results and saving your analysis

The learner stores every result internally, so you don't need to capture what
`run()` returns. Reach any result through these read-only accessors at any time,
including after a reload:

```python
result   = learner.models['coxph']      # ModelResult: .evaluation, .bootstrap, .model, .coefficients
best     = learner.best_model           # winning ModelResult
eval_all = learner.evaluation_result    # combined EvaluationResult (None before evaluate())
boot_all = learner.bootstrap_result     # combined BootstrapResult (None before bootstrap())
```

Export the result tables, or save and reload the whole session:

```python
# Export all result tables to a single Excel workbook (or CSVs with format='csv')
learner.export_results('results.xlsx')

# Save the whole session and reload it later; results survive the round-trip
learner.save_learner('analysis.learner')
from mlsurv import load_learner
learner = load_learner('analysis.learner')
learner.models['coxph']                 # still here after reload
```

## Command-Line Interface

mlsurv also provides a CLI for scripting, HPC pipelines, and quick one-off analyses:

```bash
# Full pipeline in one command
mlsurv run data.csv --time-col OS_MONTHS --event-col OS_STATUS --output report.html

# List available models
mlsurv models

# Train and tune a specific model
mlsurv tune data.csv --time-col OS_MONTHS --event-col OS_STATUS --model rsf --n-trials 50

# Validate on an external cohort
mlsurv validate data.csv --time-col OS_MONTHS --event-col OS_STATUS \
    --model coxph --validation-data external.csv \
    --val-time-col OS_MONTHS --val-event-col OS_STATUS
```

Run `mlsurv --help` for the full list of 26 subcommands, or see the
[CLI Guide](https://github.com/goeckslab/mlsurv/blob/main/docs/CLI.md) for detailed documentation and workflows.

## Data Format

mlsurv expects a feature matrix `X` (pandas DataFrame) and a structured survival array `y` with `event` (bool) and `time` (float) fields.

**From CSV** (recommended):

```python
from mlsurv.utils import load_survival_data

X, y = load_survival_data('data.csv', time_col='OS_MONTHS', event_col='OS_STATUS')
```

`load_survival_data` automatically drops the time/event columns from `X` and converts events to boolean and times to float, following scikit-survival conventions.

**From existing DataFrame**:

```python
from mlsurv.utils import to_structured_array

y = to_structured_array(times=df['OS_MONTHS'], events=df['OS_STATUS'])
X = df.drop(columns=['OS_MONTHS', 'OS_STATUS'])
```

**From date columns** (when your data has diagnosis/follow-up dates instead of pre-computed times):

```python
from mlsurv.utils import compute_survival_time

X, y = compute_survival_time(
    df, start_col='diagnosis_date', end_col='last_followup_date',
    event_col='vital_status', time_unit='months',
    event_values=['deceased'],
)
```

Missing values are handled automatically via median imputation during preprocessing (configurable via `setup(imputer=...)`).

## Installation

Install from PyPI:

```bash
pip install mlsurv
```

With deep learning support (PyTorch, pycox) for DeepSurv / DeepHit:

```bash
pip install "mlsurv[deep]"
```

**macOS note:** the XGBoost models (`xgbaft`, `xgbcox`) need the OpenMP runtime,
which macOS does not ship by default. If you see an error about `libomp.dylib`
failing to load, install it once with Homebrew:

```bash
brew install libomp
```

To install the latest unreleased version from GitHub:

```bash
pip install git+https://github.com/goeckslab/mlsurv.git
```

For development:

```bash
git clone https://github.com/goeckslab/mlsurv.git
cd mlsurv
pip install -e ".[dev]"
```

See [CONTRIBUTING.md](https://github.com/goeckslab/mlsurv/blob/main/CONTRIBUTING.md) for the
architecture overview, coding rules, and test conventions.

## Features

- **Low-code interface**: `setup()` / `run()` / `predict()` / `generate_report()` for a full analysis in under 10 lines of code ([Architecture](https://github.com/goeckslab/mlsurv/blob/main/docs/architecture.md))
- **10 survival models** across 4 families (linear, ensemble, kernel, neural) with unified interface ([Models](https://github.com/goeckslab/mlsurv/blob/main/docs/architecture.md#model-inventory))
- **Configurable preprocessing**: Imputation, scaling, and 5 feature selectors with leak-free CV pipelines ([Pipeline](https://github.com/goeckslab/mlsurv/blob/main/docs/pipeline.md))
- **Automated checks**: EPV warnings, leakage detection, PH and linearity testing, model recommendations ([Pipeline](https://github.com/goeckslab/mlsurv/blob/main/docs/pipeline.md#automated-setup-checks))
- **Optuna tuning**: EPV-aware search spaces, auto-trial sizing, convergence detection, study persistence ([Evaluation](https://github.com/goeckslab/mlsurv/blob/main/docs/evaluation.md#hyperparameter-tuning))
- **Multi-metric evaluation**: Harrell/IPCW/Antolini C-index, Brier/IBS, AUC(t), calibration slope/intercept, GND test ([Metrics](https://github.com/goeckslab/mlsurv/blob/main/docs/evaluation.md#evaluation-metrics))
- **Multi-scale evaluation**: Subpopulation analysis, prospective patient prediction, and single-patient HTML reports with SHAP waterfall + cohort context ([Evaluation](https://github.com/goeckslab/mlsurv/blob/main/docs/evaluation.md)); feature importance, SHAP, interactions, and individual patient explanations ([Feature Analysis](https://github.com/goeckslab/mlsurv/blob/main/docs/feature-analysis.md))
- **External validation & benchmarking**: Independent cohort validation, delta-C/NRI/IDI incremental metrics ([Validation](https://github.com/goeckslab/mlsurv/blob/main/docs/validation.md))
- **S(t|x) calibration**: Opt-in per-model calibration via `train(calibrate=...)` (isotonic / lowdof / breslow_slope) fit on leakage-free OOF predictions and applied automatically across evaluate / predict / validate, with a calibration banner in reports ([Calibration](https://github.com/goeckslab/mlsurv/blob/main/docs/evaluation.md#stx-calibration))
- **Automated reporting**: 12-section interactive HTML report, 14 limitation flags, TRIPOD+AI compliance checklist ([Reporting](https://github.com/goeckslab/mlsurv/blob/main/docs/reporting.md))
- **Parallel computing**: One-line `n_jobs=-1` setup with smart auto-disable and no oversubscription
- **26 CLI subcommands** for scripting and HPC pipelines ([CLI Guide](https://github.com/goeckslab/mlsurv/blob/main/docs/CLI.md))
- **Reproducible by default**: Single `random_state` seeds all frameworks (scikit-learn, XGBoost, PyTorch). Bit-exact reproducibility requires `n_jobs=1`; see [Reproducibility](https://github.com/goeckslab/mlsurv/blob/main/docs/evaluation.md#reproducibility)

## Documentation

| Guide | Description |
|-------|-------------|
| [Architecture & Models](https://github.com/goeckslab/mlsurv/blob/main/docs/architecture.md) | Four-stage workflow, 10 survival models, complexity tiers, unified interface |
| [Pipeline Configuration](https://github.com/goeckslab/mlsurv/blob/main/docs/pipeline.md) | Data loading, `setup()`, cross-validation, preprocessing, automated checks |
| [Model Development & Evaluation](https://github.com/goeckslab/mlsurv/blob/main/docs/evaluation.md) | Training, tuning, metrics, bootstrap CIs, model comparison, subpopulation analysis, patient prediction |
| [Feature Analysis](https://github.com/goeckslab/mlsurv/blob/main/docs/feature-analysis.md) | Feature importance, SHAP, interactions, individual patient explanations |
| [Validation & Benchmarking](https://github.com/goeckslab/mlsurv/blob/main/docs/validation.md) | External cohort validation, incremental benchmarking metrics |
| [Reporting & Compliance](https://github.com/goeckslab/mlsurv/blob/main/docs/reporting.md) | Limitation flags, HTML reports, TRIPOD+AI checklist, data export |
| [CLI Guide](https://github.com/goeckslab/mlsurv/blob/main/docs/CLI.md) | 26 command-line subcommands for scripting and HPC |
| [Tutorial Vignettes](https://github.com/goeckslab/mlsurv/tree/main/examples) | 7 Jupyter notebooks from quickstart to immunotherapy case study |

## Supported Models

mlsurv includes 10 survival models across 4 categories:

**Linear/Parametric (3)**
| Model | Description |
|-------|-------------|
| CoxPH | Cox Proportional Hazards |
| CoxNet | Elastic Net penalized Cox |
| Weibull AFT | Parametric Weibull Accelerated Failure Time model |

**Tree-Based Ensemble (4)**
| Model | Description |
|-------|-------------|
| RSF | Random Survival Forest |
| GBSA | Gradient Boosting Survival Analysis |
| XGBoost-Cox | XGBoost with survival:cox objective |
| XGBoost AFT | XGBoost with Accelerated Failure Time objective |

**Support Vector Machine (1)**
| Model | Description |
|-------|-------------|
| FSSVM | Fast Survival SVM |

**Neural Networks (2)**
| Model | Description |
|-------|-------------|
| DeepSurv | Continuous-time Cox neural network |
| DeepHit | Discrete-time neural network (DeepHitSingle) |

> **Note**: All 10 models assume a single event type. See [Limitations](#limitations) for clinical implications.

## Limitations

- Single-event right-censored outcomes only (no competing risks, no time-varying covariates)
- See [Reporting & Compliance](https://github.com/goeckslab/mlsurv/blob/main/docs/reporting.md#limitation-flags) for the full list of 14 automatically detected limitation flags

## Citation

If you use mlsurv in published work, please cite:

> mlsurv: Low-code machine learning for survival analysis. Goecks Lab, Moffitt Cancer Center. https://github.com/goeckslab/mlsurv

## Changelog

See [CHANGELOG.md](https://github.com/goeckslab/mlsurv/blob/main/CHANGELOG.md) for the release history.

## License

MIT License - see [LICENSE](https://github.com/goeckslab/mlsurv/blob/main/LICENSE)

## Acknowledgments

mlsurv was built in part with [Claude Code](https://claude.com/claude-code), Anthropic's agentic coding tool.
