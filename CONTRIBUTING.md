# Contributing to mlsurv

Thanks for your interest in contributing — bug fixes, new models, docs, and questions are all
welcome. This guide covers how the code is organized, the conventions we follow, and how to run
the tests, so you can get a change landed smoothly. It's written for human contributors and AI
coding agents alike.

If something here is unclear or out of date, opening an issue to flag it is itself a welcome
contribution.

## Project Overview

**mlsurv** is a low-code survival machine learning package that unifies modeling, tuning, evaluation, validation, and reporting for multiple survival models behind a single API.

---

## Correctness: what to double-check

mlsurv is used for clinical survival analysis, so a few classes of bug matter more than usual.
When you touch modeling code, it helps to check:

- **Reproducibility**: seeds set and propagated (numpy, torch, optuna). Results are deterministic when seeds are fixed **and `n_jobs=1`** — with `n_jobs>1` you'll see small run-to-run variance from parallel Optuna trials and multi-threaded XGBoost/BLAS, which is expected (see `docs/evaluation.md#reproducibility`), not a bug. Serialization round-trips should preserve state.
- **Data leakage**: fit preprocessing on train only; keep feature selection from seeing test/validation data; isolate CV folds; respect temporal ordering.
- **ML correctness**: proper train/test/validation splits, careful censored-observation handling, correct C-index/Brier/IBS, tuning on validation rather than test, and bootstrap resampling done right.

If you're unsure whether something is handled correctly, flag it in your PR — we'd much rather talk it through than miss it.

---

## Code Architecture

```
mlsurv/
├── learner/               # SurvivalLearner — main API (mixin package)
│   ├── _core.py           #   Class definition, __init__, MRO
│   ├── _state.py          #   LearnerState dataclass (all instance variables)
│   ├── _setup.py          #   SetupMixin: setup(), data splitting, CV config
│   ├── _training.py       #   TrainingMixin: train(), run()
│   ├── _cv_training.py    #   CvTrainingMixin: CV fold logic
│   ├── _tuning.py         #   TuningMixin: tune(), Optuna integration
│   ├── _evaluation*.py    #   EvaluationMixin: evaluate/bootstrap/interactions/validate (+ _eval_context)
│   ├── _prediction.py     #   PredictionMixin: predict() for new patients
│   ├── _recalibration.py  #   RecalibrationMixin: post-hoc S(t|x) recalibration
│   ├── _sweep.py          #   SweepMixin: parameter_sweep()
│   ├── _benchmarking.py   #   BenchmarkingMixin: benchmark() (_references.py scores ref biomarkers)
│   ├── _explain.py        #   ExplainMixin: explain()
│   ├── _patient_*.py      #   PatientReportMixin: single-patient HTML report (report/html/template/resolve)
│   ├── _visualization.py  #   VisualizationMixin: plot_* methods (data-prep in viz/_data.py + viz/_resolve.py; _views.py proxies)
│   ├── _config.py         #   ConfigMixin: config (property)
│   ├── _persistence.py    #   PersistenceMixin: save/load_artifact, save/load_learner
│   ├── _studies.py        #   StudyMixin: experiment_path, list/export/reload_study
│   ├── _export.py         #   ExportMixin: export_results() (xlsx/csv)
│   └── _reports.py        #   ReportMixin: generate_report()
├── cli/                   # CLI interface (package)
│   ├── __main__.py        #   Entry point for `python -m mlsurv.cli`
│   ├── _parsers.py        #   create_parser(), per-subcommand builders
│   ├── _commands.py       #   cmd_run … cmd_export_results (23 handlers)
│   ├── _data.py           #   load_data, validation, _setup_learner
│   ├── _artifacts.py      #   Artifact load/save/inspect/upgrade
│   ├── _plot.py           #   Plot subcommand
│   ├── _output.py         #   Structured output; _exit_codes.py: exit codes & CLI exceptions
│   └── _helpers.py        #   I/O helpers (_write_csv, _write_json, etc.)
├── utils.py               # Data loading, SurvivalArray utilities
├── constants.py           # METRIC_REGISTRY, model/metric lookups
├── guidance.py            # User-facing guidance messages
├── logging_utils.py       # Logging configuration
├── _shared.py             # Shared internal utilities
├── _warnings.py           # Warning tiers (Methodological, Suggestion, Internal)
├── _iter_warn.py          # Coalescer for repeated IterativeWarnings
├── _calibration_thresholds.py # Shared calibration thresholds
├── _loky_patch.py         # loky/joblib parallel backend patch
├── models/                # 10 model wrappers (BaseSurvivalModel subclasses)
│   ├── base.py            #   BaseSurvivalModel ABC
│   ├── coxph, coxnet, rsf, gbsa, fssvm
│   ├── deepsurv, deephit  #   pycox/torch-based
│   ├── weibull_aft, xgbcox, xgbaft
│   ├── _overflow.py       #   Numeric overflow guards
│   ├── _metadata.py       #   Model metadata registry
│   └── _reference_adapter.py # Reference implementation adapters
├── preprocessing/         # Feature selection (TopKCoxSelector), covariance selectors
├── evaluation/            # Metrics, evaluate(), bootstrap(), SHAP
├── tuning/                # Optuna hyperparameter tuning, search spaces, MetricInfo registry
├── results/               # Result containers (ModelResult, EvaluationResult, BootstrapResult)
├── reports/               # HTML report generation (sections, flowchart)
└── viz/                   # Plotly visualizations (KM curves, importance, SHAP)
```

### Module naming convention (same name, different layer)

The same filename appears in several packages on purpose — each copy does one job at one layer.
When you see a name repeated, read it by its package:

| Package | Role of `X` | Example |
|---------|-------------|---------|
| `evaluation/_X.py` | **computes** X (the numbers) | `evaluation/_explain.py`, `evaluation/metrics.py` |
| `viz/X.py` | **plots** X | `viz/metrics.py`, `viz/interactions.py` |
| `reports/sections/.../_X.py` | **renders** X into a report section | `reports/sections/feature_analysis/_shap.py` |
| `learner/_X.py` | the public `.X()` method (thin wrapper that calls the computer) | `learner/_explain.py` |
| `results/_X.py` | the X result **container** | `results/_prediction.py` |

One concept can span three files (e.g. `_interactions.py` in `evaluation/` compute,
`viz/` plot, `reports/sections/feature_analysis/` render). That's intended layering, so there's
no need to "fix" it by renaming. If `learner/_X.py` doesn't seem to hold the real work, check the
`evaluation/` sibling. Within `reports/sections/`, the same name under `evaluation/` vs
`validation/` (e.g. `_calibration.py`) is the same job for a different cohort (test vs
validation), not a duplicate.

---

## Environment & Commands

Install the package with its development dependencies, then work from the project root:

```bash
pip install -e ".[dev]"          # core + test tooling
pip install -e ".[dev,deep]"     # add torch/pycox for DeepSurv / DeepHit
```

```bash
# Fast tier (default) — unit, functional, CLI tests
pytest

# Full suite — everything except vignettes
pytest --run-full

# All — truly everything including vignettes
pytest --run-all

# Run quickstart vignette
jupyter nbconvert --to notebook --execute examples/vignette_01_quickstart.ipynb

# Run workflow tour vignette
jupyter nbconvert --to notebook --execute examples/vignette_02_workflow_tour.ipynb
```

See [Testing](#testing) below for the full marker list and tips on keeping tests fast.

### Inspecting Reports Programmatically

`learner.generate_report()` infers the format from the output extension. For programmatic
consumption, a markdown report (`.md`) is usually easiest: it's plain text, much smaller
than HTML, easy to read or grep, with figures saved as PNGs in an adjacent `images/` folder
(and it's pandoc-ready for PDF/Word). Use `.html` for interactive human viewing, where figures
are embedded as interactive Plotly charts in one standalone file.

```python
learner.generate_report('report.md')    # markdown — handy for programmatic use
learner.generate_report('report.html')  # HTML — interactive human viewing
```

To inspect an *existing* HTML report without a browser, `scripts/report_dev.py read` strips
it to text:

```bash
# Read the default dev report
python scripts/report_dev.py read

# Read a specific report
python scripts/report_dev.py read path/to/report.html

# List available sections with aliases
python scripts/report_dev.py read --list-sections

# Read specific sections (by registry key or HTML anchor)
python scripts/report_dev.py read -s summary evaluation feature_analysis

# Truncate long tables
python scripts/report_dev.py read --max-table-rows 10
```

Figures are replaced with `[Figure]` placeholders, and Plotly JSON, CSS, and JS are stripped.
Section names accept both registry keys (`summary`, `evaluation`) and HTML anchors (`executive-summary`, `model-evaluation`).

---

## Coding Conventions

### Handling exceptions

We catch the narrowest exception type a call can actually raise rather than a blanket
`except Exception`, so real bugs don't hide behind error handling.

- In report sections, use the exception tuples from `reports/sections/base.py` (`ALL_EXCEPTIONS`, `DATA_EXCEPTIONS`, `IMPORT_EXCEPTIONS`, `PLATFORM_EXCEPTIONS`) rather than a bare catch.
- Elsewhere, catch the specific type(s) the callee can raise.
- One deliberate exception: `_evaluate_model` and `_evaluate_model_precomputed` in `tuning/base.py` use `except Exception` on purpose, to turn arbitrary trial failures into `optuna.TrialPruned`. That's Optuna's contract — a failed trial should prune the study, not crash it. `MemoryError`, `RecursionError`, and `ImportError` are re-raised before the broad catch.
- Each `except` clause should do something visible: re-raise, log, warn, or return a meaningful fallback. The one thing to avoid is swallowing an error silently, since that turns a crash into a confusing wrong answer.

### Verifying warning filters

When you add or change a `warnings.filterwarnings()` call, a few checks save a lot of debugging:

- Confirm the actual warning class by triggering it in a test and checking `type(w)`, rather than guessing from the docs. Python's class hierarchy means a filter for `UserWarning` won't catch a `DeprecationWarning`.
- Add a test that the filter works (`with warnings.catch_warnings(record=True)`).
- Leave a short comment noting the library-version behavior (e.g., `# NumPy 2.x emits DeprecationWarning, not UserWarning`).
- Grep the whole codebase and fix every matching site in one change, so the behavior stays consistent.

### Surfacing problems instead of hiding them

When a computation can't produce a real result, it's better to tell the user than to return a
quiet NaN:

- Emit a user-visible `warnings.warn()` with the appropriate tier from `_warnings.py` (`MethodologicalWarning`, `SuggestionWarning`, `InternalWarning`). A `logger.warning()` alone doesn't reach users.
- On a shape mismatch, raise `ValueError` rather than padding or truncating arrays to fit.
- For a "nothing computed" sentinel DataFrame, return one that's empty (0 rows) rather than NaN-filled — a NaN-filled frame passes nonempty checks and can leak into exports.
- When a user-provided parameter gets ignored (e.g., `model_params` when `tune=True`), let them know with a `SuggestionWarning`.

---

## Learner Internals (`mlsurv/learner/`)

`SurvivalLearner` is composed from mixins (one per `_*.py` file) so all methods stay on
`self` with zero API change. `_core.py` defines the class and the inheritance chain; every
other `_*.py` is one mixin. State lives in a single `LearnerState` dataclass (`_state.py`),
reached via `self._state`.

The full reference — state-field inventory, computed/post-train properties, and the per-mixin
method map — lives in [`docs/learner-internals.md`](docs/learner-internals.md); read it on
demand. This section keeps just the rules you'll want on hand for most changes.

### Where things live

| File | Role |
|------|------|
| `_core.py` | `SurvivalLearner` class, MRO, model-name resolution, computed properties |
| `_state.py` | `LearnerState` dataclass — all instance state |
| `_setup.py` | `setup()`, data split, CV config; `SetupConfig` |
| `_format.py` | shared banner/box display helpers (pure string builders for `setup`/`tune`/`evaluate`/`bootstrap` summary boxes) |
| `_cv_training.py` / `_training.py` | CV fold logic; `train()` / `run()` |
| `_tuning.py` | `tune()`, Optuna; `TuneConfig` |
| `_evaluation.py` | `evaluate`, `bootstrap`, `validate`, `interactions`, `importance`/`importance_test`, `ph_test`/`linearity_test`/`rmst_test`/`logrank_test` |
| `_prediction.py` / `_recalibration.py` | `predict()` for new patients; post-hoc `S(t|x)` recalibration |
| `_benchmarking.py` / `_sweep.py` / `_explain.py` | `benchmark()`; `parameter_sweep()`; `explain()` |
| `_visualization.py` | `plot_*` methods (data-prep lives in `viz/_data.py` + `viz/_resolve.py`) |
| `_patient_report*.py` | single-patient HTML report |
| `_config.py` | `config` (current-configuration snapshot property) |
| `_persistence.py` | `save_artifact`/`load_artifact`/`save_learner`/`load_learner` |
| `_studies.py` | `experiment_path`/`list_studies`/`export_study`/`reload_study` |
| `_export.py` | `export_results` (xlsx/csv of all result tables) |
| `_reports.py` | `generate_report()` |

### State access rules

- For eval/bootstrap data, read via the computed properties (`self.evaluation_result`,
  `self.bootstrap_result`) or per-model (`self._state.models[name].evaluation`). These
  aggregates are computed on each read from the current per-model results (no cache), so
  they always reflect the live state — there's no invalidation step to remember.
- All other state reads/writes inside mixins go through `self._state.<field>` directly.
- External code (reports, tests) uses public properties where available, otherwise
  `learner._state.<field>` — not the legacy `self._<field>` pattern.
- Diagnostic-result fields aren't the public access path: reach them through the
  `<method>_result` property, and keep `_state.<field>` out of user-facing strings and docs.

### Preventive Rules

#### State consistency (after any change to how results are stored in `_state.models`)

- Trace the full pipeline: `train/tune → _state.models[key] → evaluate → bootstrap → report
  → export`. The storage key (e.g. `"coxph"`) needs to stay consistent at every lookup, and
  `ModelResult.name` should match its dict key.
- Pass `ModelResult`/`Pipeline` objects through call chains, not `model_name: str`. The
  public API resolves a string once (`_resolve_model_input`/`_resolve_single_model`); helpers
  below that boundary take the resolved object. Re-resolving a user string in an internal
  helper reintroduces a class of bug that's easy to slip back in — `tests/test_resolve_model_results.py::test_no_unlisted_state_models_access`
  guards against it.
- Single storage namespace: `tune()` overwrites `_state.models[name]` in place. There's no
  `_tuned` suffix — storage keys equal registry names, so `_lookup_model_by_name` is a direct
  dict lookup, and `ModelResult.base_name` equals `name` (the registry/factory lookup key).
- Per-model derived state rides on the `ModelResult`. The internal S(t|x) calibrator
  from `train(calibrate=...)` (`result.calibrator` — a `SurvivalCalibrator`; see
  `evaluation/_calibrators.py`), importance-test tables (`result.pimp_importance_test`,
  `result.bootstrap_importance_test`), and risk-group-test tables (`result.rmst_test`,
  `result.logrank_test`) live on the `ModelResult`, not in side dicts on `_state`. So they
  die with the model the instant `_state.models[name]` is replaced — no separate drop step.
  The historical dict names (`pimp_importance_test_results`,
  `bootstrap_importance_test_results`, `rmst_test_results`, `logrank_test_results`) survive
  as read-only `MappingProxyType` views on both `LearnerState` (so internal
  `self._state.<name>` reads keep working) and `SurvivalLearner` (public API); writes go
  only through the `ModelResult` field. Every `train()`/`run()`/`tune()` overwrite of
  `_state.models[name]` calls `_warn_overwrite(name)` (`_training.py`) for a uniform
  `SuggestionWarning`. The calibrator is applied wherever the deployed model produces
  S(t|x) (evaluate/bootstrap/predict/validate/subpopulation) via
  `self._apply_calibrator(name, surv_probs, eval_times, X=...)`. The shared calibration
  helpers live in `_recalibration.py`.
- Regression net: the lifecycle test in `tests/test_lifecycle_integration.py` traces a
  `ModelResult` reference through every public stage (`setup → train → tune → evaluate →
  bootstrap → generate_report`).

#### API parameter naming

- Count parameters use the `n_<thing>` form (`n_bootstrap`, `n_trials`, `n_jobs`, `n_importance_repeats`).
- Before adding a parameter, `grep -r "n_<similar>"` and reuse the existing name. Internal and
  external APIs use the same name for the same concept.

### Cross-Mixin Rules

1. Mixins don't import each other directly — cross-mixin calls go through `self`.
2. Only `_core.py` imports the mixin classes (it defines the chain). The one exception is a sub-mixin imported by its parent via direct inheritance (e.g. `CvTrainingMixin`).
3. Each mixin imports only the external deps it needs.
4. All state goes through `self._state` — please don't add instance vars directly on `self`.

### TYPE_CHECKING pattern (IDE support)

```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner

class SomeMixin:
    def method(self: SurvivalLearner, ...): ...
```

This gives IDEs full autocomplete on `self` without circular imports.

### Adding a new method

1. Pick the mixin matching the method's purpose (or create `_foo.py` + `FooMixin` and add it
   to the chain in `_core.py`).
2. Add it with the `self: SurvivalLearner` hint, and import only what it needs.
3. For new state, add fields to `LearnerState` in `_state.py`.
4. Update the method map in `docs/learner-internals.md`.

### Module-level functions (needed for `joblib.Parallel` pickling)

These live at module level so they pickle cleanly: `_train_cv_fold()` (`_cv_training.py`);
`_compute_cv_ci()`, `_finite_mean_std()` (`_training.py`).

### Constants (in `_core.py`)

`_NOT_SET` (unset sentinel); `MODELS_WITH_RANDOM_STATE`; `SEQUENTIAL_CV_MODELS` (parallel CV
is counterproductive); `MODELS_WITH_INTERNAL_PARALLELISM` (n_jobs propagated from `setup()`);
`_adjust_model_njobs` (back-compat alias).

---

## Model Wrappers (`mlsurv/models/`)

### File Map

| File | Purpose |
|------|---------|
| `base.py` | `BaseSurvivalModel` ABC, `PycoxSerializationMixin`, `ImpurityImportanceMixin` |
| `coxph.py` | Cox proportional hazards (sksurv) |
| `coxnet.py` | Elastic-net Cox (sksurv) |
| `rsf.py` | Random survival forest (sksurv) |
| `gbsa.py` | Gradient-boosted survival analysis (sksurv) |
| `fssvm.py` | Fast survival SVM (sksurv) |
| `deepsurv.py` | DeepSurv (pycox/torch) |
| `deephit.py` | DeepHit (pycox/torch) |
| `weibull_aft.py` | Weibull AFT (lifelines) |
| `xgbcox.py` | XGBoost survival (Cox objective) |
| `xgbaft.py` | XGBoost AFT (internal/experimental) |
| `_overflow.py` | Numeric overflow guards (`predict_with_clamp`) |
| `_metadata.py` | Model metadata registry |
| `_auto_defaults.py` | Data-driven `auto_*_defaults()` constructor defaults |
| `_reference_adapter.py` | Reference implementation adapters |

### Rank-Based and Score-Based Models

Some models (currently FSSVM) optimize a concordance ranking objective. Their
raw decision values are well-ordered for the C-index but their magnitudes are
not log-hazards, so feeding them directly to a Breslow estimator yields
miscalibrated survival probabilities. Two class attributes handle this:

- **`rank_based: bool`** (`BaseSurvivalModel`, default `False`).
  Set `rank_based = True` on the wrapper to opt in. Future ranking-only models
  inherit `False` until they explicitly opt in.
- **`score_based: bool`** (`BaseSurvivalModel`, default `False`).
  True when the model reduces each subject to a single linear-predictor risk
  score, so a Breslow baseline can be refit on those scores. Set on
  `coxph, coxnet, gbsa, xgbcox, fssvm, deepsurv`. It gates the `breslow_slope`
  calibrator (`train(calibrate=...)`), which raises `ValueError` on a non-score model.

#### Calibration lives in the learner, not the model

Post-hoc `S(t|x)` calibration is controlled by **`train(calibrate=...)`**, which
fits a calibrator on held-out OOF of the training cohort and attaches it to the
`ModelResult` (`result.calibrator`). See `evaluation/_calibrators.py`.

- fssvm trains raw and the learner attaches a `breslow_slope` calibrator by
  default (a van Houwelingen refit fit on held-out OOF). The model itself has no
  calibration arguments.
- A rank-based model with no calibrator and a bad slope still trips the drift
  warning, which points at `train(calibrate='breslow_slope')`.

### Serialization Rules

When wrapping an external ML framework (pycox, torch, xgboost, lifelines), a few habits keep
models picklable and predictions stable:

- Keep classes and closures at module level if the containing object will be serialized — local classes defined inside methods can't be pickled.
- Strip transient training state in `__getstate__` — callbacks, optimizers, metric trackers, training logs, and DataLoaders aren't needed for prediction. The `PycoxSerializationMixin` pattern from `base.py` (`_PYCOX_TRAINING_ATTRS` tuple) does this for you.
- Test serialization right away: round-trip with `joblib.dump(model)` / `joblib.load()` after fitting, confirm predictions match, and check the original model wasn't mutated by `__getstate__`.
- Pickle device-neutral state. A torch tensor pickled outside `torch.save` remembers the device it came from, and `joblib.load` has no `map_location` to undo that — so a GPU-trained model saved as-is cannot be opened on a CPU-only host. `PycoxSerializationMixin` writes CPU tensors in `__getstate__` and moves the model back onto an available device in `__setstate__` (falling back to CPU with a `SuggestionWarning`). Any new framework wrapper holding accelerator state should do the same.
- Validate defaults against search spaces: every `auto_*_defaults()` function (in `_auto_defaults.py`) should produce values within the corresponding `DEFAULT_SPACES[model_name][param]` range (in `tuning/search_spaces.py`).
- Calibrators round-trip on their own: `train(calibrate=...)` attaches a `SurvivalCalibrator` (isotonic / lowdof / breslow_slope from `evaluation/_calibrators.py`) to `result.calibrator`, not to `LearnerState`. It round-trips because `ModelResult` is pickled as part of `_state.models`. All three are pickle-safe (isotonic holds `IsotonicRegression` maps; lowdof holds two floats; breslow_slope holds a float slope plus a fitted `BreslowEstimator`), so there's no need to rebuild them on load. `result.calibrator is None` means uncalibrated — don't auto-fit on load.

---

## Testing

### Running Tests

Install the package with its test dependencies first (`pip install -e ".[dev]"`),
then run from the project root:

```bash
# Fast tier (default) — unit, functional, CLI tests
pytest

# Full suite — everything except vignettes
pytest --run-full

# All — truly everything including vignettes
pytest --run-all

# Vignettes only
pytest --run-vignettes

# By category
pytest -m "shap"              # SHAP tests only
pytest -m "integration"       # Integration tests
pytest -m "slow"              # Slow tests only (notebooks)
pytest -m "validation"        # Reference-implementation validation tests

# With coverage
python -m pytest tests/ --cov=mlsurv --cov-report=term-missing
```

### Test Markers

| Marker | Meaning |
|--------|---------|
| *(none)* | Fast unit tests |
| `integration` | Full workflow tests (10-60s) |
| `integration_heavy` | Tuning + bootstrap + DeepSurv (subset of `integration`) |
| `bootstrap_heavy` | Tests with expensive bootstrap workloads (20+ iterations) |
| `parallel` | Parallel execution tests |
| `shap` | SHAP importance tests (CPU-intensive) |
| `full` | Full-suite-only tests (run with `--run-full` or `--run-all`) |
| `vignette` | Notebook execution tests (use `--run-vignettes`) |
| `slow` | Vignette / long-running (> 60s) |
| `requires_pycox` | Needs pycox/torch installed |
| `requires_shap` | Needs shap library installed |
| `validation` | Reference-impl validation (run with `--run-all` or `-m validation`) |

### Test Files

There are ~100 test files across `tests/` and `tests/validation/`. `pytest --collect-only -q`
lists the current tests and counts. The naming conventions:

- `test_phase{1-4}_*.py` — original phased test suite (imports → models → evaluation → viz/CLI)
- `test_viz_*.py` — visualization-specific tests
- `validation/test_validation_*.py` — reference-implementation validation tests
- Other `test_*.py` — feature-specific or regression tests

### Key Fixtures (conftest.py)

- `synthetic_data` — 200 samples, 20 features, with NaN
- `synthetic_data_clean` — Same without NaN
- `small_data` — 50 samples for fast tests
- `train_test_data` — Pre-split data (from `synthetic_data_clean`)
- `gbsg2_data` — Real breast cancer dataset (686 patients)

### Writing Fast Tests

The fast tier aims to finish in under 6 minutes, and every new test runs in it unless marked
otherwise. A few habits keep it quick.

#### 1. Mark expensive tests `@pytest.mark.integration`

Mark a test `@pytest.mark.integration` if it does any of the following, so the fast tier stays fast:

- Calls `SurvivalLearner.setup()` + `train()` on GBSG2 or Veterans data (686+ samples)
- Calls `tune()`, runs Optuna trials, or creates Optuna studies
- Calls `benchmark()` with real model training
- Calls `bootstrap()` with `n_bootstrap >= 10`
- Calls `parameter_sweep()` with real model training
- Calls `validate()` with real model evaluation
- Takes longer than ~2s in isolation

Tests that only call `setup()` on small synthetic data (50-200 samples) without training are fast enough for the fast tier.

#### 2. Choose the smallest dataset that works

| Fixture | Size | Use when |
|---------|------|----------|
| `small_data` | 50 samples, 5 features | **Default choice** for API contract tests (setup, train, evaluate) |
| `synthetic_data_clean` | 200 samples, 20 features | Tests that need more features (feature selection, high-dimensional edge cases) |
| `synthetic_data` | 200 samples, 20 features, NaN | Tests that specifically test imputation / missing-value handling |
| `gbsg2_data` | 686 patients | Integration tests only — always mark `@pytest.mark.integration` |

Rule of thumb: if the test only checks "does this method return the right type / not crash / have the right keys", use `small_data`.

#### 3. Scope expensive fixtures to module level

When several tests share the same trained learner and none of them mutate it:

```python
@pytest.fixture(scope="module")
def trained_learner():
    X, y = _make_data()  # inline data (can't depend on function-scoped fixtures)
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42)
    learner.train("coxph")
    return learner
```

Two things to keep in mind:
- Module-scoped fixtures can't depend on function-scoped fixtures (like `small_data` from conftest), so inline the data generation instead.
- If any test mutates the learner (sets attributes, adds models, changes state), use the deepcopy template pattern instead:

```python
@pytest.fixture(scope="module")
def _template():
    """Train once, reuse via deepcopy."""
    X, y = _make_data()
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42)
    learner.run(["coxph"])
    return learner

@pytest.fixture
def trained_learner(_template):
    """Per-test deep copy — safe for tests that mutate state."""
    import copy
    return copy.deepcopy(_template)
```

This runs ~20-50x faster than training from scratch per test.

#### 4. Patterns that slow the suite down

- Using `synthetic_data_clean` for setup-only tests — `small_data` is ~4x faster
- Function-scoped fixtures that train models, used by 10+ tests — scope to module or use the deepcopy template
- Unmarked tests that call `tune()` or `benchmark()` — these belong in `@pytest.mark.integration`
- `n_bootstrap=20` in tests that only check structure — `n_bootstrap=2` is plenty
- `n_importance_repeats=5` in tests that don't check importance — pass `n_importance_repeats=0`

#### 5. Calibration tests

- `train(calibrate=...)` and the shared OOF helpers (`_collect_oof_survival`, `_fit_isotonic_recalibrators`) re-run CV folds, so tests that fit a calibrator end-to-end fit best under `@pytest.mark.integration`. To stay in the fast tier, construct a fitted calibrator (or its mappings) directly and attach it to `result.calibrator` instead of fitting via the OOF path.
- Tests asserting on `predict()` / `predict_survival_function()` / `patient_report()` that depend on `S(t|x)` numerics should set `use_recalibrated=True` or `=False` explicitly. The deployed default applies the attached calibrator when one is present, so leaving it implicit makes the assertion's meaning depend on whether a calibrator happens to be on the fixture.

### Boundary Value Testing

For every new numeric parameter, it's worth covering the edges:
- `0` — should either work with meaningful degenerate behavior, or raise a clear `ValueError` (rather than a `KeyError`/`IndexError`).
- `1` — the minimum meaningful value.
- Negative values — should raise `ValueError`.

For functions operating on arrays:
- Guard DataFrame column access with length/key checks before indexing.
- Use `np.isfinite()` rather than `~np.isnan()` — `isfinite` catches both NaN and Inf.
- Test with empty arrays, single-element arrays, and arrays containing NaN/Inf.

For numeric computations:
- Guard `np.exp()` on model outputs with clamping (see `_safe_exp()` in `viz/_features_hazard_ratios.py`, `predict_with_clamp()` in `models/_overflow.py`).
- Watch for double-exponentiation: if `predict()` returns `exp(x)` and a downstream consumer also calls `exp()`, the chain produces `exp(exp(x))`.
- Call `.tolist()` on numpy arrays before passing them to Plotly — its JSON serializer silently drops numpy arrays in nested dicts.

### Acceptable Warnings

These are expected and fine to see in test output:
- Convergence warnings from Cox models
- Optuna experimental feature warnings
- `FutureWarning` from sklearn pipeline (not-yet-fitted check)
- `DeprecationWarning` from `trapz` → `trapezoid` in sksurv
