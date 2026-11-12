# Changelog

All notable changes to mlsurv are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **PyTorch loads only when a deep model is used.** `setup()` and
  `list_models()` used to import torch and pycox just to check whether they
  were installed. On macOS, once PyTorch's copy of OpenMP is in the process,
  a later XGBoost fit can hang or crash, so this hit sessions that never used
  a deep model. The check now looks the packages up without importing them.
- **A torch install that fails to load no longer crashes `setup()`.** The old
  check caught only `ImportError`, so a torch that raised `OSError` on import
  (a blocked or missing DLL on Windows) crashed `setup()` for every model.
  Training DeepSurv or DeepHit now reports a load failure as an `ImportError`
  that keeps the loader's message and tells a broken install apart from a
  missing one, and `run()` drops the deep model with a warning while the
  other models still train.
- **macOS: XGBoost warns when PyTorch is already loaded.** Fitting `xgbcox`
  or `xgbaft` after a deep model in the same session can still hang on macOS.
  mlsurv now warns before the fit and names the fix: set `OMP_NUM_THREADS=1`
  before starting Python, or use separate sessions.

## [0.2.0] - 2026-09-08

### Added

- **`setup(stratify=...)` takes a list, so event and a column can be one
  argument.** A column name used to replace event stratification instead of
  joining it, and the only route to event x variable ran through the
  subpopulation machinery, which wants an array rather than a column name and
  also switches on per-subgroup evaluation and report sections.
  `stratify=['event', 'stage']` now builds composite keys for both the
  train/test split (rare-cell threshold 2, full cohort) and the CV folds
  (threshold `cv_folds`, training rows only), so the two stages can diverge.
  Entries are `'event'`, column names, or label arrays, mixed freely.
  Stratification adapts rather than failing, and warns every time: wide
  columns quantile-bin, entries that separate nothing drop, rare combinations
  merge into `__other__`, and it backs off when even that fails. The scalar,
  array, and `None` forms keep their previous behavior exactly.
- **`TopKCoxSelector(k='tune')`** searches the feature count during tuning,
  with no `search_spaces` argument needed. Fitting it untuned raises rather
  than silently choosing a `k`.
- **`importance_max_samples`**, an opt-in cap on the test rows used for
  permutation importance. The C-index term is superlinear in `n_test`, so on
  a large cohort importance can cost more than fitting the model. The default
  `None` is a no-op.

### Fixed

- **A feature with no observed values no longer breaks the diagnostics.** A
  100%-missing column cannot be imputed, and every sklearn imputer drops it at
  `fit` time. Labelling the transformed output with the input's column names
  then raised a bare pandas "Shape of passed values" error, which took the
  proportional-hazards and linearity diagnostics, their plots, and the
  report's Data Preprocessing section down with it. Such columns are now
  reported through a `MethodologicalWarning` naming them, and the diagnostic
  matrix is narrowed to the features that carry data. The warning is explicit
  that the feature still occupies a row in the importance table, scoring zero
  with no coefficient, which is not the same as a feature that was measured
  and found unimportant. A matrix whose every column is empty raises instead,
  since nothing can be computed from it.
- **A broken xgboost install is reported instead of costing you the model.**
  The xgboost wheel links against an OpenMP runtime that macOS does not ship,
  so a plain `pip install` can leave a package that raises `XGBoostError` on
  import. That class inherits from `ValueError`, not `ImportError`, so the
  wrappers' `except ImportError` guards missed it: the `dlopen` failure
  escaped as a bad-input error, became a pruned tuning trial, and `xgbcox`
  and `xgbaft` were dropped from `run()` with no legible reason, leaving
  `best_model` quietly dependent on the platform. Every xgboost import now
  reports any failure as an `ImportError` naming the remedy (`brew install
  libomp` on macOS), and that reason survives Optuna into the dropped-models
  warning.
- **`run()` no longer drops a component-structured `search_spaces`.** `run()`
  and multi-model `train()` treated `search_spaces` as a per-model map, so a
  component-structured dict was silently discarded: coxph was skipped with
  "no hyperparameters to search" and coxnet tuned only its defaults. A
  component-structured space now broadcasts to every model, the per-model
  form is kept, keys matching no model warn, and a mixed form raises.
- **An oversized result table no longer fails the whole xlsx export.**
  `export_results(format='xlsx')` let openpyxl raise "This sheet is too
  large", so a run exited non-zero even though modeling had succeeded, and a
  caller who exported before saving lost the fitted learner. The table that
  overflows is `eval_bootstrap_predictions`, one row per model x bootstrap
  iteration x resampled sample. Tables past Excel's limits are now written as
  CSV into a sibling `<stem>_large_tables/` directory and recorded in a
  `_large_tables` index sheet, so the workbook keeps every table that fits and
  the run keeps its clean exit.
- **A deep model trained on a GPU can now be loaded on a CPU-only host.**
  mlsurv never calls `torch.save`/`torch.load`: deep models reach disk as
  ordinary pickled objects inside `joblib.dump`, and a torch tensor pickled
  outside `torch.save` embeds the device it lived on, which `joblib.load` has
  no `map_location` to undo. Saving now writes CPU tensors, and loading
  restores onto a device the host actually has, warning whenever it settles
  for less. Files already written on a GPU still cannot be opened on a CPU
  host, so `load_learner` and `load_artifact` now say that plainly instead of
  reporting the file as corrupt.
- **Permutation importance survives a coefficient failure.** Importance came
  back empty for coxph, coxnet, fssvm, and weibull_aft whenever the
  pipeline's selector exposed no `selected_features_`. sklearn's
  `permutation_importance` succeeded every time; the `ValueError` came from
  the coefficient step that ran right after it inside the same `try`, and the
  handler threw the computed importance away. A coefficient problem now
  degrades only the coefficient column.
- **Weibull-AFT coefficients no longer land on the wrong features.**
  Coefficients were keyed positionally by the selector's order, but lifelines
  sorts its covariates, so every `weibull_aft` coefficient was attached to the
  wrong feature, sign flips included, in both the report and the
  `eval_importance` sheet. Coefficients are now keyed by the model's own index.

- **The coefficient table and the importance table can no longer disagree about
  which feature a coefficient belongs to.** Two code paths decided which label
  keys a model's coefficient vector, and they decided it separately. lifelines
  sorts Weibull-AFT covariates, so a model can return its coefficients in an
  order the feature selector never emitted, and the two paths handled that
  differently: the same fitted model reported different coefficients depending
  on whether you read `EvaluationResult.importance` or
  `ModelResult.coefficients`. Both now resolve labels through one shared
  function, so they agree by construction. Two narrower faults went with it: a
  coefficient index whose labels matched no feature produced a table of nothing
  but NaN, with no warning, and an index that was only partly named (a string
  first label, numbers after) kept the first coefficient and quietly dropped
  the rest onto the wrong rows.
- **One model's coefficient problem no longer costs every other model its
  rows.** `SurvivalLearner.coefficients` loops over every trained model with no
  guard, so a `ValueError` from aligning one model's coefficients escaped and
  emptied the whole table, taking the report's feature-analysis table with it.
  A model whose coefficients cannot be placed on its features is now reported
  and skipped, and the rest of the table survives.
- **Coefficient failures now reach the user.** Four ways of failing to produce
  a coefficient table were invisible. A model that raised inside
  `get_coefficients()` warned through a category the default filters drop and
  wrote no log record either, so nothing appeared anywhere. A model that
  returned an empty coefficient vector was treated as though it had no
  coefficients at all, which looks the same as a random survival forest. A
  table of nothing but NaN was returned as a result rather than reported as a
  failure, and it passed the non-empty checks downstream, so it reached
  `feature_summary()` and the exports. Losing the feature names to an unnamed
  array was reported through the same dropped category. All four now warn
  visibly and name the model, and the two that computed nothing usable return
  an empty table rather than a NaN-filled one.
- **An unbounded hazard ratio no longer costs you the whole Feature Analysis
  report section.** A separated or badly scaled feature makes the univariate
  Cox `exp(coef + 1.96 * se)` overflow to infinity. That infinity reached the
  hazard-ratio plot's log-scale tick computation, where `math.ceil` raised
  `OverflowError`, and the report replaced the entire section — correlations,
  importance, coefficients and all — with a placeholder. Hazard ratios and
  bounds that a log axis cannot place (infinity from an overflow, zero from an
  underflow) are now left out of the axis range and reported through a
  `ClampWarning`. The affected feature loses the whisker on the bad side, or
  its marker too when the point estimate is the bad value; every other feature
  draws normally. Stray numpy divide and overflow warnings from the same values
  are gone too.
- **The report now says why a hazard ratio is missing from a forest plot.**
  The warning above reaches whoever ran the code; the report is read by
  someone else, who saw only a labelled row with no marker. The Hazard Ratios
  section now names the affected features and explains the omission.
- **A confidence bound on the wrong side of its hazard ratio no longer draws a
  mirrored whisker.** A swapped `ci_lower_col` / `ci_upper_col` produced a
  whisker to a value that appeared nowhere in the data. No whisker is drawn
  now, and one `InternalWarning` per figure points at the likely swap.
- **A hazard ratio too large for a log axis to label no longer breaks the
  plot.** Past about `1e307` the tick computation built a power of ten too big
  for a float, raising `OverflowError` for a bound that was large but perfectly
  finite. Tick decades are now capped at what a float can hold, and a range
  spanning hundreds of decades is labelled sparsely instead of at every one.
- **A failed figure no longer takes its whole report section with it.**
  `OverflowError` was missing from the shared report-section exception tuples,
  so an arithmetic failure in one chart escaped the per-figure guard. It now
  degrades to `*[Figure unavailable: ...]*` like every other recoverable chart
  error.

- **CoxPH is no longer dropped from cross-validation and tuning on
  near-singular designs.** When the Cox optimizer's solver failed, only some of
  the possible error messages routed into the automatic regularization
  fallback; scipy's "LAPACK reported an illegal value" wording did not, so the
  fit aborted, every tuning trial was pruned, and the model was dropped from
  the results. The classification now also covers scipy's LAPACK
  illegal-value wording, and the fallback escalates through `0.01`, `0.1`, and
  `1.0`, keeping the first penalty that fits. Each rung is floored at any
  `alpha` you supplied, so a user-chosen penalty is never silently weakened.
- **Bad input is no longer reported as a degenerate design.** A NaN or infinite
  value in `X` produced "Data may be too degenerate for Cox PH", hiding
  scikit-learn's far more useful message naming the offending input. Input and
  parameter errors now propagate unchanged.
- **`FitFailureWarning` now surfaces at most once per distinct message per
  learner.** The tuning phase kept its own warning tally, so a `tune()` call
  could report the same numerical-instability message twice: once from the
  Optuna trials and once from the cross-validation re-run. Tuning now shares
  the learner's already-seen set with cross-validation, the final fit, and
  calibration.

### Changed

- **Hazard ratios are suppressed when CoxPH escalates past `alpha=0.01`.** If
  an unstable fit needs `0.1` or `1.0` to converge, the coefficients are shrunk
  enough that hazard ratios, Wald confidence intervals, and p-values are
  conditional on the penalty rather than unbiased multivariable estimates, so
  the coefficient table and forest plot are left out and a
  `MethodologicalWarning` explains why. The C-index and survival probabilities
  are unaffected. Escalating to `0.01`, the only previous fallback, is
  unchanged.
- **The copyright holder is now the Goecks Lab.** The initial release
  named the wrong holder in `LICENSE` and the package metadata.

## [0.1.0] - 2026-06-25

Initial public release of mlsurv, a low-code machine learning package for
survival analysis. mlsurv unifies modeling, tuning, evaluation, validation,
and reporting into a single API.

### Added

- **Low-code survival analysis workflow** — a typical analysis runs `setup()`
  (configure preprocessing and cross-validation) → `run()` (train, tune,
  evaluate, and bootstrap every model in one call, or call `train()`, `tune()`,
  `evaluate()`, and `bootstrap()` individually) → `generate_report()`
  (publication-ready HTML or markdown). Additional steps slot in as needed:
  `predict()` for new patients, `validate()` on an external cohort,
  `train(calibrate=...)` for post-hoc `S(t|x)` calibration, `explain()` for
  SHAP, `benchmark()` against reference biomarkers, and `parameter_sweep()`.
- **Results access, export, and persistence** — every result is stored on the
  learner and reachable through read-only accessors (`models`, `best_model`,
  `evaluation_result`, `bootstrap_result`), so you never need to capture what
  `run()` returns. `export_results()` writes all tables to an Excel workbook or
  CSVs, and `save_learner()` / `load_learner()` round-trip the whole session with
  results intact.
- **10 survival models** across four families with a unified interface:
  CoxPH, CoxNet, and Weibull AFT (linear/parametric); Random Survival Forest,
  Gradient Boosting Survival Analysis, XGBoost-Cox, and XGBoost-AFT
  (tree ensembles); Fast Survival SVM (kernel); DeepSurv and DeepHit (neural,
  optional via the `deep` extra).
- **Configurable preprocessing pipeline** with leak-free cross-validation:
  imputation (median, mean, KNN, iterative), scaling (standard, robust, minmax,
  PCA), and five feature selectors (`topk`, `vif`, `correlation`, `coxlasso`,
  `coxnet`).
- **Automated setup checks**: events-per-variable warnings, leakage detection,
  proportional-hazards and linearity testing, and model recommendations.
- **Hyperparameter tuning** via Optuna with EPV-aware search spaces, automatic
  trial sizing, convergence detection, and study persistence.
- **Multi-metric evaluation**: Harrell, IPCW, and Antolini C-index; Brier score
  and integrated Brier score; time-dependent AUC(t); calibration
  slope/intercept; and the Greenwood-Nam-D'Agostino test. `add_eval_times()`
  registers additive report horizons (e.g. 12- and 60-month) that flow into the
  time-dependent metrics without replacing the default grid.
- **External validation and benchmarking** on independent cohorts, including
  delta-C, NRI, and IDI incremental metrics against reference biomarkers.
- **Post-hoc `S(t|x)` calibration**: opt-in via `train(calibrate=...)`, with
  leakage-free out-of-fold fitting and a choice of methods (isotonic, low
  degrees of freedom, Breslow-slope refit). The calibrator is applied wherever
  the model produces survival probabilities, including cohort and patient
  reports and the `validate()` transferability panel.
- **Feature explanation**: permutation importance, SHAP values and
  interactions, and single-patient explanations with a SHAP waterfall and
  cohort context.
- **Statistical tests**: proportional-hazards (Schoenfeld), log-rank, RMST, and
  permutation-importance significance (PIMP).
- **Automated reporting**: interactive single-file HTML reports and plain-text
  markdown reports (figures as PNGs), with a TRIPOD+AI compliance checklist and
  13 automatically detected limitation flags.
- **Command-line interface** with 26 subcommands for scripting and HPC
  pipelines, covering the full workflow (run, train, tune, evaluate, bootstrap,
  validate, predict, calibrate, explain, benchmark, sweep, plot), statistical
  tests (including feature interactions), patient reports, full result export,
  Optuna study management, artifact inspection/upgrade, and model listing.
- **Parallel execution**: one-line `n_jobs=-1` setup with smart auto-disable
  and oversubscription guards.
- **Reproducible by default**: a single `random_state` seeds scikit-learn,
  XGBoost, and PyTorch; bit-exact runs require `n_jobs=1`.
- **Inline type annotations** shipped with a `py.typed` marker (PEP 561).

[0.2.0]: https://github.com/goeckslab/mlsurv/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/goeckslab/mlsurv/releases/tag/v0.1.0
