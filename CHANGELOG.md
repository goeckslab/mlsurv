# Changelog

All notable changes to mlsurv are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

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

[0.1.0]: https://github.com/goeckslab/mlsurv/releases/tag/v0.1.0
