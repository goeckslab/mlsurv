# mlsurv Test Suite

## Quick Reference

```bash
# Fast (default) — unit, functional, CLI tests
pytest

# Full suite — everything except vignettes
pytest --run-full

# All — truly everything including vignettes
pytest --run-all

# Vignettes only — notebook execution tests
pytest --run-vignettes

# Run specific marker categories
pytest -m "integration"          # All integration tests
pytest -m "shap"                 # SHAP importance tests
pytest -m "parallel"             # Parallel execution tests
```

## Test Tiers

| Tier | Command | Tests | Measured Runtime | When to Use |
|------|---------|-------|------------------|-------------|
| Fast | `pytest` | ~5700 | ~5.5 min | Every save / local dev |
| Full | `pytest --run-full` | ~7000 | ~30 min | Before merging / pre-release |
| All | `pytest --run-all` | ~7000 | ~60 min* | Includes vignette notebooks |

\* Fast and full runtimes are measured on a local dev machine; the All tier is estimated (full + vignette notebook execution, not re-measured).

### What each tier includes

- **Fast**: Unit tests, functional tests, CLI parsing, data leakage checks — everything *not* marked with an excluded marker.
- **Full**: Everything except vignette/slow-marked tests. Includes integration, parallel, SHAP, bootstrap-heavy.
- **All**: Truly everything, including vignette notebook execution.
- **Vignettes** (`--run-vignettes`): Runs *only* `vignette`-marked tests (notebook execution). Useful for targeted vignette validation.

## Test Markers

Markers are for targeted `-m` runs, not for choosing tiers.

| Marker | Description | Count |
|--------|-------------|-------|
| `integration` | Full workflow integration tests (10-60s) | ~98 |
| `integration_heavy` | Expensive integration (tuning, bootstrap, DeepSurv) — subset of `integration` | ~43 |
| `bootstrap_heavy` | Tests with expensive bootstrap workloads (20+ iterations) | ~6 |
| `parallel` | Parallel execution tests | ~43 |
| `shap` | SHAP importance tests (CPU-intensive) | ~45 |
| `vignette` | Notebook execution tests (use `--run-vignettes`) | ~7 |
| `slow` | Vignette / long-running tests (> 60s) | ~7 |
| `requires_pycox` | Tests requiring pycox/torch | varies |

## How Default Exclusion Works

Implemented in `conftest.py::pytest_collection_modifyitems` (not via `addopts -m`, which would conflict with CLI `-m` flags).

| Flag | Behavior |
|------|----------|
| *(none)* | Skip `slow`, `vignette`, `integration`, `bootstrap_heavy`, `parallel`, `shap`, `full`, `validation`, `demo`, `requires_pycox`, `requires_shap` |
| `--run-full` | Skip `slow`, `vignette`, `validation` |
| `--run-all` | Run everything |
| `--run-vignettes` | Run *only* `vignette`-marked tests |
| `-m "..."` | User filter respected, no default exclusion applied |

If multiple flags are passed, precedence is: `--run-all` > `-m` > `--run-vignettes` > `--run-full`.
