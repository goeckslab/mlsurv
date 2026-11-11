# mlsurv Documentation

Detailed implementation documentation for the [mlsurv](../README.md) survival machine learning package.

For a quick overview and installation instructions, see the [main README](../README.md). For hands-on tutorials, see the [example vignettes](../examples/).

## Contents

| Guide | Description |
|-------|-------------|
| [Architecture & Models](architecture.md) | Four-stage workflow, 10 survival models, complexity tiers, unified interface |
| [Pipeline Configuration](pipeline.md) | Data loading, `setup()`, cross-validation strategies, preprocessing, automated checks |
| [Model Development & Evaluation](evaluation.md) | Training, hyperparameter tuning, evaluation metrics, bootstrap CIs, model comparison, subpopulation analysis, prospective patient prediction, single-patient HTML reports |
| [Feature Analysis](feature-analysis.md) | Feature importance, SHAP, feature interactions, individual patient explanations |
| [Validation & Benchmarking](validation.md) | External cohort validation, incremental benchmarking metrics (NRI, IDI, delta-C) |
| [Reporting & Compliance](reporting.md) | Limitation flags, HTML report generation, TRIPOD+AI checklist, data export |
| [CLI Guide](CLI.md) | 26 command-line subcommands for scripting and HPC pipelines |

## Tutorial Vignettes

| # | Vignette | Dataset(s) | Runtime |
|---|----------|------------|---------|
| 1 | [Quickstart](../examples/vignette_01_quickstart.ipynb) | METABRIC (n=1,903) | ~1 min |
| 2 | [Workflow Tour](../examples/vignette_02_workflow_tour.ipynb) | GBSG2 (n=686), Rotterdam (n=2,982) | ~10 min |
| 3 | [Benchmarks](../examples/vignette_03_benchmarks.ipynb) | SUPPORT, METABRIC, FLCHAIN, GBSG2, Veterans | ~10-12 min |
| 4 | [Custom Components](../examples/vignette_04_custom_components.ipynb) | TCGA BRCA (n=1,071) | ~2-3 min |
| 5 | [Advanced CV & Tuning](../examples/vignette_05_advanced_cv_tuning.ipynb) | Rotterdam, GBSG2 | ~3 min |
| 6 | [Tool Comparison](../examples/vignette_06_tool_comparison.ipynb) | GBSG2, METABRIC | ~6 min |
| 7 | [Immunotherapy Case Study](../examples/vignette_07_immunotherapy_case_study.ipynb) | Chowell (n=1,479) + 7 external cohorts | ~15-20 min |
