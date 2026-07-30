# Learner Internals Reference

Detailed reference for the `mlsurv/learner/` mixin package — the full state-field
inventory, computed/post-train properties, and the per-mixin method map. This is the
load-on-demand companion to `CONTRIBUTING.md`, which keeps only the rules an
editor needs on every change. Read this when you need the exhaustive detail.

The `LearnerState` dataclass in `_state.py` is the **single source of truth** for state
fields; the inventory below adds the cross-field semantics (what owns what, when it is
populated) that the bare dataclass does not capture. If the two ever disagree, the
dataclass wins — fix this doc.

## File → Method Map

| File | Mixin Class | Key Methods |
|------|-------------|-------------|
| `_state.py` | `LearnerState` | Centralized dataclass holding all 27+ instance variables |
| `_core.py` | `SurvivalLearner` | `__init__`, `_ensure_setup`, `_resolve_model_input`, `_resolve_single_model`, `_resolve_model_names`, `_resolve_models_multi` *(arity-aware resolver for `predict`/`importance`/`rmst_test`/`logrank_test`)*, `models` / `model_recommendations` properties |
| `_setup.py` | `SetupMixin` | `setup`, `_reset_state_if_needed`, `_validate_setup_inputs`, `_resolve_preprocessing`, `_configure_cv`, `_finalize_setup`, `_run_setup_diagnostics`, `_setup_presplit_data`, `_setup_split_data`, `_get_cv_splitter`, `_build_stratify_keys`, `_guard_stratify_class_count`, `_warn_stratify_list_wins`, `_augment_stratify_for_split`, `_augment_stratify_with_subpopulations`; `SetupConfig` dataclass |
| `_format.py` | *(no mixin — module-level functions)* | Shared banner/box display helpers, pure string builders for the `setup`/`tune`/`evaluate`/`bootstrap` summary boxes: `_print_param_box` (generic renderer), `_format_scoring_name`, `_format_model_display`, `_format_state_overrides`, `_format_pipeline_str`, `_format_per_model_preprocessing`, `_format_data_str`, `_format_cv_str`, `_format_subpopulation_summary`, `_format_trials_display`, `_format_scoring_display`, `_format_search_display`, `_format_storage_display`, `_format_eval_times_row`, `_format_importance_row`, `_format_bootstrap_row` |
| `_cv_training.py` | `CvTrainingMixin` | `_get_cv_folds`, `_run_cv_folds`, `_aggregate_fold_results`, `_run_model_comparison`, `_post_run_evaluation` |
| `_training.py` | `TrainingMixin(CvTrainingMixin)` | `train`, `run`, `_train_model_internal`, `_finalize_model_result`, `_warn_overwrite_and_drop` *(overwrite warning + state drops for the train/run paths)*, `_build_pipeline`, `_check_cv_vs_holdout_best`, `_print_comparison_table` *(model-selection helpers)* |
| `_tuning.py` | `TuningMixin` | `tune`, `_tune_single_model`, `_tune_and_train`, `_tune_all_models`, `_run_post_tune_evaluation`, `_validate_tune_inputs`, `_resolve_tune_models_for_display`; `TuneConfig` dataclass |
| `_evaluation.py` | `EvaluationMixin` | `evaluate`, `bootstrap`, `validate`, `interactions`, `importance` *(magnitude front door; dispatches `method='permutation'` to the private `_permutation_importance` backend and `method='shap'` to the private `_shap` backend)*, `importance_test`, `ph_test`, `linearity_test`, `rmst_test`, `logrank_test` *(both multi-model: loop over `_rmst_test_single` / `_logrank_test_single` per-model workers, stored via `_stamp_and_store_risk_test`)*, `_permutation_importance` *(private permutation backend)*, `_shap` *(private SHAP backend)*, `_get_test_risk_groups`, `_auto_kang_comparison` |
| `_benchmarking.py` | `BenchmarkingMixin` | `benchmark` |
| `_sweep.py` | `SweepMixin` | `parameter_sweep`, `_build_sweep_preprocessing` |
| `_visualization.py` | `VisualizationMixin` | `plot_metric`, `plot_metric_time`, `plot_survival`, `plot_importance` (incl. `style="heatmap"`), `plot_calibration`, `plot_roc`, `plot_benchmark_incremental`, `plot_benchmark_scatter`, `plot_coefficients`, `plot_coefficient_comparison`, `plot_hazard_ratios`, `plot_hazard_ratios_comparison`, `plot_feature_correlation`, `plot_missing`, `plot_shap`, `plot_calibration_overlay`, `plot_interactions`, `plot_interaction_pdp`, `plot_diagnostics`, `plot_feature_distributions`, `plot_metric_heatmap`, `plot_validation_heatmap`, `plot_optuna`, `validation(cohort)`, `subpopulation(group)` |
| *(data-prep moved to `viz/`)* | — | `_visualization.py` delegates to `mlsurv.viz._resolve` (shared resolution utilities) and `mlsurv.viz._data` (domain logic for metrics, features, survival, time, calibration, tuning, benchmarking). Relocated to dissolve the old `viz <-> learner` import cycle |
| `_config.py` | `ConfigMixin` | `config` *(property)* — the current-configuration snapshot; module-level helpers `_groups_display_label`, `_str_or_none` |
| `_persistence.py` | `PersistenceMixin` | `save_artifact`, `load_artifact`, `_update_artifact_path`, `save_learner`, `load_learner`; module-level `_environment_fingerprint()` + `_ArtifactEncoder` |
| `_studies.py` | `StudyMixin` | `experiment_path` *(property)*, `list_studies`, `export_study`, `reload_study`; module-level `_resolve_tuned_model` |
| `_export.py` | `ExportMixin` | `export_results`, `_collect_all_dataframes`, `_truncate_sheet_names`, `_write_xlsx`, `_offload_oversized`, `_write_csv_dir`; module-level `_collect_*`/`_build_*` table collectors, `_oversize_reason`, `_csv_filenames`, `_sidecar_dir`, `_warn_stale_csvs` |
| `_reports.py` | `ReportMixin` | `generate_report`, `_gather_report_data` |
| `_prediction.py` | `PredictionMixin` | `predict` *(multi-model: dict of per-model results for collection selections)*, `_predict_single` *(per-model worker)*, `_validate_predict_input`, `_run_prediction_bootstrap`, `_attach_cohort_context` |
| `_explain.py` | `ExplainMixin` | `explain` |
| `_patient_report.py` | `PatientReportMixin` | `patient_report` (builds standalone HTML report wrapping `predict` + `explain` with survival curve + bootstrap CI band, cohort context, SHAP waterfall, KNN tables; `use_recalibrated=True` toggle, calibration banner, calibration panel) |
| `_recalibration.py` | `RecalibrationMixin` | **Calibrators** (`train(calibrate=...)`): `_has_calibrator`, `_apply_calibrator`, `_attach_calibrators`, `_calibration_splitter`, `_model_is_score_based` — fit a `SurvivalCalibrator` (`evaluation/_calibrators.py`) on the development cohort's held-out OOF and store it on `result.calibrator`. Single-track: evaluate/bootstrap/predict/validate report the calibrated S(t|x), applying `result.calibrator` automatically |
| `_patient_report_template.py` | *(helper module)* | CSS + `render_html()` for `patient_report` |
| `_patient_report_html.py` | *(helper module)* | Pure HTML/data builders: `_stat_grid_items`, `_feature_table_html`, `_knn_table_html`, `_alerts_list` (split out of `_patient_report.py` to isolate the CC-27 hotspot) |

## State Management (`_state.py`)

All mutable learner state lives in a single `LearnerState` dataclass, accessed via `self._state`:

```python
# Reading state
splitter = self._state.cv_splitter
models = self._state.models

# Writing state
self._state.models[name] = model_result
```

**State fields by lifecycle phase:**

| Group | Fields |
|-------|--------|
| Input data | `X`, `y`, `presplit` |
| Data splits | `X_train`, `X_test`, `y_train`, `y_test` |
| Setup config | `is_setup`, `test_fraction`, `random_state`, `scoring`, `cv_ci`, `cv_folds`, `cv_repeats`, `cv_strategy`, `n_jobs`, `experiment_name`, `time_col`, `event_col` |
| Preprocessing | `imputer`, `scaler`, `feature_selector` |
| CV infrastructure | `cv_splitter`, `cv_splitter_single`, `stratify_array`, `stratify_components`, `groups`, `group_aware_holdout` |
| Stratification labels | `stratify_spec` (what the split stratified on), `cv_stratify_spec` (what the folds stratified on) — both recorded after any binning, subpopulation augmentation, or rare-level fallback, so reports state what happened rather than what was requested |
| Descriptive | `univariate_cox` (per-feature univariate Cox table; set in `setup()` on the full cohort, descriptive only — see the `univariate_cox` property docstring) |
| Models & results | `models` (dict — **authoritative store**; each `ModelResult` holds `.evaluation`, `.bootstrap`, `.calibrator` — the `SurvivalCalibrator` attached by `train(calibrate=...)` — and `.calibration_flagged` — the drift flag stamped by `evaluate()` and aggregated by the `calibration_flagged_models` property), `pimp_importance_test_results` (dict), `bootstrap_importance_test_results` (dict — both read-only views assembled from per-model `ModelResult` fields; each table rides on its `ModelResult`, dying with it on overwrite), `validation_results` (dict), `benchmark_result`, `user_eval_times` |
| Paired comparisons | `kang_comparison_result` |
| Statistical tests | `ph_test_result`, `linearity_test_result` (live singular fields — refreshed by `ph_test()` / `linearity_test()`), `rmst_test_results`, `logrank_test_results` (per-model dict views over `ModelResult.rmst_test` / `.logrank_test` — each table rides on its `ModelResult`, dying with it on overwrite) |
| Setup diagnostics | `model_recommendations` (`RecommendedModels` bundle; also keeps a **setup-snapshot** reference to the PH/linearity DataFrames as `.ph_test_result` / `.linearity_test_result` — later `ph_test()`/`linearity_test()` calls rebind the live `_state` fields above but never touch the bundle's reference) |
| Runtime flags | `timeseries_ci_warned`, `epv_warned_at_setup`, `reference_overlap_warned` (set of `frozenset` signatures) |

**Computed properties** on `SurvivalLearner` (in `_core.py`):
- `evaluation_result` — assembles multi-model `EvaluationResult` from per-model `ModelResult.evaluation` via `EvaluationResult.combine()` (uncached; computed on each read, so always reflects live state)
- `bootstrap_result` — assembles multi-model `BootstrapResult` from per-model data via `BootstrapResult.combine()` (uncached; computed on each read, so always reflects live state)
- `best_model` — returns non-synthetic `ModelResult` with highest `cv_metric_mean` (via the shared `mlsurv._shared.best_model_key` selector). Its `.name` backs the `plot_*(models='best')` sentinel, so the property and the sentinel always agree on the top model
- `coefficients` — long-format coefficient DataFrame across all trained models that expose coefficients, with HR/CI/Wald-p gated by the model's `hazard_ratio_valid` flag (uncached; assembled live via `mlsurv.evaluation.compute_model_coefficients`). Per-model accessor: `ModelResult.coefficients`
- `feature_summary(model=None, *, correlation_threshold=...)` — *method, not property* (it takes a model selector). Combined per-feature table joining univariate Cox (`uni_*`), multivariable coefficients (`mv_*`), permutation importance, mean |SHAP|, importance significance (`p_adjusted`/`sig_stars`), and the `correlated` flag; assembled live via `mlsurv.evaluation.compute_feature_summary`. The feature-analysis report section renders this same table. Helper: `_correlated_feature_set(threshold)`
- `calibration_flagged_models` — names of trained models whose evaluation tripped the calibration-drift thresholds. Aggregates the per-`ModelResult.calibration_flagged` flag (stamped by `evaluate()`) in `_state.models` order; `[]` before `evaluate()`.
- `model_recommendations` — read-only proxy to `_state.model_recommendations` (the setup-time `RecommendedModels` bundle); `None` before `setup()`, populated after

**Post-train analysis result properties** on `SurvivalLearner` (in `_core.py`) — read-only passthroughs over the stashed `_state` field (no caching; these are directly stored, unlike the assembled aggregates above). Each is `None`/empty until the relevant step runs (for `ph_test_result` / `linearity_test_result` that step is `setup()` itself, despite the group name):
- `benchmark_result` → `_state.benchmark_result` (after `benchmark()`)
- `logrank_test_results` → `_state.logrank_test_results` and `rmst_test_results` → `_state.rmst_test_results` (per-model dicts, one entry per model from `logrank_test()` / `rmst_test()`; `{}` before any call). **Exception to the `<method>_result` convention:** each stored table is built from one model's risk scores, so these expose only per-model dicts (`_results`), not a singular `_result` — a singular accessor would silently reflect only the latest call. Each stored DataFrame carries `model_type` as its first column and records its result-shaping parameters in `.attrs` (log-rank: `model_name`, `n_groups`, `p_adjust`, `break_type`, `breaks`, `correction`; RMST: those plus `tau`). Each table rides on its `ModelResult` (`result.rmst_test` / `result.logrank_test`), dying with it on overwrite.
- `ph_test_result` → `_state.ph_test_result` (computed during `setup()` when `ph_test=True`; refreshed by `ph_test()`)
- `linearity_test_result` → `_state.linearity_test_result` (computed during `setup()` when `linearity_test=True`; refreshed by `linearity_test()`)
- `kang_comparison_result` → `_state.kang_comparison_result` (auto after multi-model `evaluate()`)
- `pimp_importance_test_results` → `_state.pimp_importance_test_results` and `bootstrap_importance_test_results` → `_state.bootstrap_importance_test_results` (per-model dicts, one per `importance_test()` method; `{}` before any call; "pimp" = the Altmann 2010 PIMP estimator). **Exception to the `<method>_result` convention:** `importance_test()` exposes only per-model dicts (`_results`), not a singular `_result` — it is single-model-per-call, so a singular accessor would silently reflect only the latest model. Each stored DataFrame records its result-shaping parameters in `.attrs` (PIMP: `n_permutations`, `n_importance_repeats`, `random_state`, `p_adjust`, `model_name`; bootstrap: `p_adjust`, `model_name`). PIMP requires a model trained through the learner; a raw pipeline/estimator raises `ValueError` (use `compute_pimp()` for a one-off analysis on an unregistered object).

**Naming convention for diagnostic results.** Property = `<method_name>_result` (or `<analysis_noun>_result` when auto-computed with no method). Suffix encodes kind: `_test` = produces a p-value (assumption tests `ph_test`, `linearity_test`); `_comparison` = an effect/relationship without a single NHST verdict (`kang_comparison`). **Per-model-stored tests use a per-model `_results` dict, not a singular `_result`** — `logrank_test`, `rmst_test`, and `importance_test` build each stored table from one model's predictions, so a singular accessor would silently reflect only the latest call. Keep the vocabulary split by indexing intentional: **model-indexed → "flagged"** (e.g. `calibration_flagged_models`), **feature-indexed → "violations"** (e.g. `ph_violations`), **comparison-indexed → the `_test`/`_comparison` suffix**. Do not unify these words — different words signal different indexing. `learner._state.<field>` must never be the public access path for any of these, and must never appear in user-facing strings or docs.

**Public properties** on `SurvivalLearner` (in `_core.py`) proxy commonly-accessed fields:
- Read/write: `X`, `y`, `X_train`, `X_test`, `y_train`, `y_test`, `univariate_cox`, `scoring`, `random_state`, `test_fraction`, `cv_folds`, `cv_repeats`, `cv_ci`, `cv_strategy`, `n_jobs`
- Read-only: `models`, `evaluation_result`, `bootstrap_result`, `best_model`, `coefficients`, `calibration_flagged_models`, `model_recommendations`, `benchmark_result`, `logrank_test_results`, `rmst_test_results`, `ph_test_result`, `linearity_test_result`, `kang_comparison_result`, `pimp_importance_test_results`, `bootstrap_importance_test_results`
