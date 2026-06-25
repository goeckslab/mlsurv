# `scripts/`

Developer and repo tooling for mlsurv. None of these ship in the package
(`pyproject.toml` packages only `mlsurv*`), so they are free to import from the
repo layout and assume a working dev install.

Run Python scripts from the repo root through the project conda env, e.g.:

```bash
conda run -n mlsurv python scripts/<name>.py
```

Some files are **load-bearing**: a test or vignette depends on them, so renaming
or moving one means updating that consumer in the same commit. Those are marked
below.

## Demos (HTML plot galleries)

Each runs a real analysis on a small dataset and writes an interactive Plotly
gallery, using the same `learner.plot_*()` API the vignettes and docs use. All
three are smoke-tested by `tests/test_demos.py`.

| Script | What it makes | Notes |
|--------|---------------|-------|
| `demo_feature_importance.py` | Importance-plot gallery | `--fast` for a quick run. **Load-bearing** (test_demos). |
| `demo_feature_groups.py` | Feature-group coloring gallery (GBSG2) | Needs SHAP. **Load-bearing** (test_demos). |
| `demo_multi_panel.py` | Multi-panel plot gallery | `--fast` for a quick run. **Load-bearing** (test_demos). |

## Validation and benchmarks

| Script | Purpose | Notes |
|--------|---------|-------|
| `sksurv_replication.py` | Check model wrappers reproduce scikit-survival tutorial results, then compare against Optuna-tuned runs. | **Load-bearing** (test_demos). |
| `benchmark_calibrators.py` | Benchmark post-hoc S(t\|x) calibrators across datasets x models. | Provenance for the recipes in `mlsurv/evaluation/_calibrators.py`. |
| `benchmarks/vignette_03_sksurv_targets.py` | Regenerate the scikit-survival target numbers used by Vignette 03 (means over a pinned seed list). | **Load-bearing** (test_vignette_03_sksurv_targets, vig-03 notebook). |

## Dev tools

| Script | Purpose | Notes |
|--------|---------|-------|
| `report_dev.py` | Generate a dev HTML report from METABRIC, or `read` an existing report from the CLI. | Documented in `CONTRIBUTING.md`. **Load-bearing** (test_report_dev). |

## Vignette support

These regenerate or feed the example notebooks.

| Script | Purpose | Notes |
|--------|---------|-------|
| `loris_score_calculator.py` | Compute LORIS / LR5 scores from published coefficients. | Imported by `vignette_07` via a `sys.path` shim. **Load-bearing** (test_vignette_smoke). |
| `figures/make_vignette_03_supp_table.py` | Build the combined Vignette 03 supplementary table from the saved CSVs. | **Load-bearing** (test_vignette_03_sksurv_targets). |
| `figures/tune_vignette_03_support.py` | Re-tune DeepSurv and DeepHit on SUPPORT and print paste-ready constructor params. | Run `--smoke` first. Slow (hours) at full trials. |
| `figures/example_r_xlsx_consumer.R` | Canonical R consumer of mlsurv's xlsx export, and the fixture that pins the export sheet names. | **Load-bearing** (test_r_consumer_contract). Rename a producer-side sheet key and you must update this file too. |

## Not tracked

The following live under `scripts/` locally but are gitignored, so they will not
appear in a fresh checkout:

- `figures/*.png`, `figures/*.pdf` — generated figure outputs.
- `figures/example_plots/` — local-only manuscript example-panel scripts.
- `__pycache__/` — bytecode caches.

Note: `figures/` here is a misnomer kept for now. The manuscript figure scripts
moved to a separate `mlsurv-manuscript-figures` repo; what remains under
`figures/` is Vignette 03/07 support and the R contract fixture.
