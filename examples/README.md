# mlsurv Examples

Example vignettes demonstrating mlsurv package usage.

## Vignettes

| Vignette | Purpose | Dataset | Runtime |
|----------|---------|---------|---------|
| `vignette_01_quickstart.ipynb` | Minimal "Hello World" | METABRIC | ~2min |
| `vignette_02_workflow_tour.ipynb` | Complete workflow demo (SHAP, external validation, all 7 plots) | GBSG2 + Rotterdam | ~5min |
| `vignette_03_benchmarks.ipynb` | Kvamme et al. (2019) benchmark | SUPPORT, METABRIC, FLCHAIN, GBSG2, Veterans | ~10-12min |
| `vignette_04_custom_components.ipynb` | Custom components demo | TCGA BRCA | ~1min |
| `vignette_05_advanced_cv_tuning.ipynb` | Advanced CV & tuning | GBSG2 + Rotterdam | ~3min |
| `vignette_06_tool_comparison.ipynb` | mlsurv vs scikit-survival vs lifelines: code, fidelity, tuning | GBSG2 + METABRIC | ~6min |
| `vignette_07_immunotherapy_case_study.ipynb` | LORIS immunotherapy case study with benchmarking & external validation | Chowell + 6-7 external cohorts (6 OS / 7 PFS) | ~20-30 min per outcome |

## Helpers

| Helper | Purpose |
|--------|---------|
| `scripts/loris_score_calculator.py` | LORIS score calculation using published coefficients; imported by `vignette_07_immunotherapy_case_study.ipynb` (via a `sys.path` shim, since it lives in the repo's top-level `scripts/`) |

## Layout

```
examples/
├── README.md                    # this file
├── vignette_0N_*.ipynb          # the seven vignettes above
├── data/                        # input datasets shipped with the vignettes
│   ├── metabric.csv, rotterdam.csv, tcga_brca.csv
│   └── loris/                   # LORIS immunotherapy cohort CSVs (vignette 07)
└── output/                      # generated artifacts — git-ignored
```

- **`data/`** holds the input datasets the vignettes read.
- **`output/`** holds everything the vignettes write and is git-ignored. Each vignette
  writes files prefixed with its number (`vig1_*`, `vig2_*`, …); vignette 07 writes into
  an `output/vig7/` subfolder.
- The vignettes use paths relative to this folder, so run them with `examples/` as the
  working directory (notebook execution sets this automatically).

## Quick Start

```bash
conda activate mlsurv
jupyter nbconvert --to notebook --execute examples/vignette_01_quickstart.ipynb
```

## Dependencies

**Core** (installed with `mlsurv`; runs vignettes 01, 02, 04, 05, 06, 07):
- scikit-survival
- lifelines
- optuna
- plotly
- shap

**Jupyter** (to open and run the notebooks):
```bash
pip install "mlsurv[notebooks]"
```

**Optional** (vignette 03 only — DeepSurv/DeepHit benchmarks need PyTorch + pycox):
```bash
pip install "mlsurv[deep]"
```

## Benchmark References

- Kvamme et al. (2019) [Time-to-Event Prediction with Neural Networks](https://jmlr.org/papers/v20/18-424.html) - JMLR
- [scikit-survival User Guide](https://scikit-survival.readthedocs.io/en/stable/user_guide/)
- Chang et al. (2024) [LORIS robustly predicts patient outcomes with immune checkpoint blockade therapy](https://doi.org/10.1038/s43018-024-00772-7) - Nature Cancer
