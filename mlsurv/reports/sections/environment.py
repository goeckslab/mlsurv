"""Environment & Reproducibility section for report generation.

Captures software versions, platform info, reproducibility details,
and pipeline configuration so reviewers can reproduce the analysis environment.
"""

import datetime
import importlib.metadata
import os
import platform
import sys
from typing import Any

import pandas as pd

from mlsurv.reports.sections.base import (
    ALL_EXCEPTIONS,
    BaseSection,
    DATA_EXCEPTIONS,
    PLATFORM_EXCEPTIONS,
)
from mlsurv.reports._helpers import extract_eval_times_from_auc_t, format_eval_time


# Key dependencies whose versions are reported in the environment table.
_KEY_DEPENDENCIES = (
    'scikit-survival',
    'optuna',
    'xgboost',
    'numpy',
    'pandas',
    'lifelines',
    'plotly',
    'scikit-learn',
)

# Mapping from config keys to display groups for the grouped config dump.
# Keys not listed here are assigned to 'Other'.
_CONFIG_GROUPS = {
    'Data': ('n_train', 'n_test', 'n_features', 'n_events_train'),
    'Preprocessing': ('imputer', 'scaler', 'feature_selector'),
    'Cross-Validation': ('cv_folds', 'cv_repeats', 'cv_ci', 'cv_strategy',
                         'stratify', 'groups'),
    'Reproducibility': ('random_state', 'n_jobs', 'presplit'),
    'Tuning': ('tuned_models', 'experiment_name'),
    'Evaluation': ('test_fraction', 'scoring', 'device', 'eval_times',
                   'eval_times_source', 'n_eval_times', 'time_unit',
                   'outcome_name'),
    'Diagnostics': ('check_leakage', 'ph_test', 'linearity_test'),
    'Subpopulations': ('stratify_subpopulations', 'subpopulations_declared',
                       'categorical_groups_declared', 'feature_groups_declared'),
}


# Reverse lookup: config key -> group name. Computed once at import time.
_KEY_TO_GROUP: dict[str, str] = {
    k: group_name
    for group_name, keys in _CONFIG_GROUPS.items()
    for k in keys
}


class EnvironmentSection(BaseSection):
    """
    Environment & Reproducibility section.

    Reports software versions, platform details, reproducibility info,
    and full pipeline configuration so that reviewers can reproduce
    the analysis.
    """

    name: str = "environment"
    title: str = "Environment & Reproducibility"

    def is_available(self) -> bool:
        """Always available."""
        return True

    def generate(self) -> dict[str, Any]:
        """Generate environment and reproducibility content."""
        md_lines: list[str] = []
        md_lines.append(
            "This section records the software environment and pipeline "
            "configuration used to generate this report, supporting "
            "reproducibility and reviewer verification."
        )
        md_lines.append("")

        # Section outline (anchors prefixed to avoid collision with
        # Data & Preprocessing's "Pipeline Configuration" heading)
        md_lines.extend(self._build_section_outline([
            ('Software Versions', 'software-versions'),
            ('Platform', 'platform'),
            ('Run Settings', 'run-settings'),
            ('Full Pipeline Configuration', 'full-pipeline-configuration'),
        ]))

        # --- Version table ---
        versions_df = self._collect_environment_info()
        self.data['versions'] = versions_df

        md_lines.append("### Software Versions")
        md_lines.append("")
        md_lines.append(self._format_table(versions_df))
        md_lines.append("")

        # --- Platform info ---
        md_lines.append("### Platform")
        md_lines.append("")
        platform_df = self._collect_platform_info()
        self.data['platform'] = platform_df
        md_lines.append(self._format_table(platform_df))
        md_lines.append("")

        # --- Reproducibility info ---
        repro_df = self._collect_reproducibility_info()
        self.data['reproducibility'] = repro_df
        md_lines.append("### Reproducibility")
        md_lines.append("")
        md_lines.append(self._format_table(repro_df))
        md_lines.append("")

        # --- Run Settings (per-model run-time provenance) ---
        run_settings_df = self._collect_run_settings()
        self.data['run_settings'] = run_settings_df
        if not run_settings_df.empty:
            md_lines.append("### Run Settings")
            md_lines.append("")
            md_lines.append(self._format_table(run_settings_df))
            md_lines.append("")

        cohort_names = list(self.learner.validation_results.keys())
        if cohort_names:
            md_lines.append("**External validation cohorts:** " +
                            ", ".join(f"`{c}`" for c in cohort_names))
            md_lines.append("")

        # --- Configuration dump (grouped, collapsible) ---
        config = self._collect_config_info()
        config_df = self._build_config_dataframe(config)
        self.data['config'] = config_df

        md_lines.append("### Full Pipeline Configuration")
        md_lines.append("")
        md_lines.append("<details class=\"environment-info\">")
        md_lines.append("<summary>Full configuration dump (click to expand)</summary>")
        md_lines.append("")
        md_lines.append(self._format_grouped_config(config))
        md_lines.append("</details>")
        md_lines.append("")

        # --- Timestamp ---
        md_lines.append(
            f"*Report generated: "
            f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*"
        )
        md_lines.append("")

        return {
            'markdown': '\n'.join(md_lines),
            'figures': [],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_environment_info(self) -> pd.DataFrame:
        """Collect mlsurv, Python, and key dependency versions."""
        rows: list[dict[str, str]] = []

        # mlsurv version
        try:
            import mlsurv
            rows.append({'Package': 'mlsurv', 'Version': mlsurv.__version__})
        except (ImportError, AttributeError):
            rows.append({'Package': 'mlsurv', 'Version': 'unknown'})

        # Python version
        rows.append({'Package': 'Python', 'Version': sys.version.split()[0]})

        # Key dependencies
        for dep in _KEY_DEPENDENCIES:
            rows.append({'Package': dep, 'Version': self._get_package_version(dep)})

        # Conditional DL dependencies
        if self._has_deep_learning_models():
            for pkg in ('torch', 'pycox'):
                rows.append({'Package': pkg, 'Version': self._get_package_version(pkg)})

        return pd.DataFrame(rows)

    def _get_package_version(self, package: str) -> str:
        """Look up an installed package version with graceful fallbacks."""
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return 'not installed'
        except PLATFORM_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc,
                f"environment: version lookup for {package}",
            )
            return 'unknown'

    def _collect_platform_info(self) -> pd.DataFrame:
        """Collect OS / hardware platform details."""
        def _safe(func, fallback='unknown'):
            try:
                return str(func())
            except PLATFORM_EXCEPTIONS as exc:
                self._log_recoverable_exception(
                    exc,
                    f"environment: platform info ({func.__name__})",
                )
                return fallback

        rows = [
            {'Property': 'Platform', 'Value': _safe(platform.platform)},
            {'Property': 'Architecture', 'Value': _safe(platform.machine)},
            {'Property': 'CPU count', 'Value': _safe(os.cpu_count, 'unknown')},
        ]
        return pd.DataFrame(rows)

    def _collect_reproducibility_info(self) -> pd.DataFrame:
        """Collect reproducibility-related details (seed, GPU info)."""
        rows: list[dict[str, str]] = []

        # Random seed
        try:
            seed = self.learner.random_state
            rows.append({'Property': 'Random Seed', 'Value': str(seed)})
        except (AttributeError, TypeError) as exc:
            self._log_recoverable_exception(
                exc,
                "environment: random seed lookup",
            )
            rows.append({'Property': 'Random Seed', 'Value': 'unknown'})

        # Artifact path (getattr guards against old deserialized states)
        artifact_path = getattr(self.learner._state, 'artifact_path', None)
        if artifact_path is not None:
            rows.append({'Property': 'Artifact Path', 'Value': artifact_path})

        # Device + GPU info (only if DL models are used)
        if self._has_deep_learning_models():
            device_setting = getattr(self.learner._state, 'device', None) or 'auto'
            rows.append({'Property': 'Device Setting', 'Value': device_setting})
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                rows.append(
                    {'Property': 'GPU Available', 'Value': str(gpu_available)}
                )
                if gpu_available:
                    cuda_ver = getattr(torch.version, 'cuda', None)
                    if cuda_ver:
                        rows.append(
                            {'Property': 'CUDA Version', 'Value': str(cuda_ver)}
                        )
                    try:
                        device_name = torch.cuda.get_device_name(0)
                        rows.append(
                            {'Property': 'GPU Device', 'Value': device_name}
                        )
                    except PLATFORM_EXCEPTIONS as exc:
                        self._log_recoverable_exception(
                            exc,
                            "environment: GPU device name lookup",
                        )
                        rows.append(
                            {'Property': 'GPU Device',
                             'Value': 'unavailable (query failed)'}
                        )
            except (ImportError, OSError) as exc:
                # OSError: torch is installed but its libraries failed to load.
                self._log_recoverable_exception(
                    exc, "environment: torch import",
                )
                rows.append(
                    {'Property': 'GPU Available', 'Value': 'torch not available'}
                )

        return pd.DataFrame(rows)

    def _collect_run_settings(self) -> pd.DataFrame:
        """Per-model run-time provenance: tuning, bootstrap, calibration."""
        models = self.learner.models or {}
        if not models:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for name, mr in models.items():
            try:
                calibrator = getattr(mr, 'calibrator', None)
                rows.append({
                    'Model': name,
                    'Tuned': bool(getattr(mr, 'is_tuned', False)),
                    'n_trials': self._count_optuna_trials(mr),
                    'n_bootstrap': self._count_bootstrap_iterations(mr),
                    'Calibrated': calibrator is not None,
                    'Cal. method': (
                        calibrator.method if calibrator is not None else '-'
                    ),
                    'Cal. horizons': self._format_horizons(
                        getattr(calibrator, 'horizons', None)
                    ),
                })
            except DATA_EXCEPTIONS as exc:
                self._log_recoverable_exception(
                    exc,
                    f"environment: run settings for model '{name}'",
                    warn_user=True,
                    user_message=(
                        f"Could not collect run settings for model '{name}'."
                    ),
                )
                rows.append({
                    'Model': name, 'Tuned': '-', 'n_trials': '-',
                    'n_bootstrap': '-', 'Calibrated': '-',
                    'Cal. method': '-', 'Cal. horizons': '-',
                })
        return pd.DataFrame(rows)

    def _count_optuna_trials(self, model_result) -> Any:
        """Effective Optuna trial count, with fallback to serialized metadata.

        Live success returns ``int``. Live failure renders as either
        ``"<N> (live lookup failed; from saved metadata)"`` or
        ``"unavailable (live lookup failed)"`` so a corrupted study isn't
        silently presented as a clean serialization snapshot.
        """
        meta = getattr(model_result, '_optuna_study_meta', None)
        meta_count = (
            meta['n_trials'] if isinstance(meta, dict) and 'n_trials' in meta
            else None
        )
        study = getattr(model_result, 'optuna_study', None)
        if study is not None and hasattr(study, 'trials'):
            try:
                return len(study.trials)
            except (AttributeError, TypeError) as exc:
                self._log_recoverable_exception(
                    exc, "environment: optuna study trials lookup",
                )
                if meta_count is not None:
                    return f"{meta_count} (live lookup failed; from saved metadata)"
                return 'unavailable (live lookup failed)'
        if meta_count is not None:
            return f"{meta_count} (from saved metadata)"
        return '-'

    def _count_bootstrap_iterations(self, model_result) -> Any:
        """Per-model bootstrap iteration count from BootstrapResult.iterations."""
        boot = getattr(model_result, 'bootstrap', None)
        if boot is None:
            return '-'
        iters = getattr(boot, 'iterations', None)
        if iters is None:
            return '-'
        if not isinstance(iters, pd.DataFrame):
            # Non-None but wrong type indicates upstream corruption — surface it.
            self._log_recoverable_exception(
                TypeError(
                    f"bootstrap.iterations is {type(iters).__name__}, "
                    f"expected DataFrame"
                ),
                "environment: bootstrap iteration count",
            )
            return '-'
        if iters.empty or 'iteration' not in iters.columns:
            return '-'
        try:
            return int(iters['iteration'].nunique())
        except DATA_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "environment: bootstrap iteration count",
            )
            return '-'

    @staticmethod
    def _format_horizons(horizons) -> str:
        """Format calibration horizons as a comma-separated string."""
        if horizons is None:
            return '-'
        if len(horizons) == 0:
            return '(no horizons recorded)'
        return ', '.join(format_eval_time(t) for t in horizons)

    def _collect_config_info(self) -> dict:
        """Return the learner's pipeline configuration as a plain dict."""
        try:
            config = self.learner.config
        except ALL_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc,
                "environment: reading learner.config failed",
                warn_user=True,
                user_message="Could not retrieve pipeline configuration.",
            )
            return {}

        # Augment with eval_times info from evaluation results
        eval_times = self._extract_resolved_eval_times()
        if eval_times:
            is_auto = getattr(self.learner._state, 'user_eval_times', None) is None
            config['eval_times'] = ', '.join(format_eval_time(t) for t in eval_times)
            config['eval_times_source'] = (
                'auto (10th\u201390th percentile of training event times)'
                if is_auto
                else 'user-specified'
            )
            config['n_eval_times'] = len(eval_times)

        return config

    def _extract_resolved_eval_times(self) -> list:
        """Extract the resolved eval_times from evaluation results, or empty list."""
        try:
            eval_result = self.learner.evaluation_result
            if eval_result is None:
                return []
            return extract_eval_times_from_auc_t(eval_result.auc_t)
        except DATA_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc,
                "environment: eval_times extraction",
            )
            return []

    @staticmethod
    def _build_config_dataframe(config: dict) -> pd.DataFrame:
        """Build a config DataFrame with Key, Value, and Category columns."""
        if not config:
            return pd.DataFrame(columns=['Key', 'Value', 'Category'])

        rows = [
            {'Key': k, 'Value': str(v), 'Category': _KEY_TO_GROUP.get(k, 'Other')}
            for k, v in config.items()
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def _format_grouped_config(config: dict) -> str:
        """Format config as grouped sub-tables in markdown."""
        if not config:
            return "*No configuration available.*"

        # Partition config into groups
        groups: dict[str, list[tuple[str, str]]] = {}
        for k, v in config.items():
            group = _KEY_TO_GROUP.get(k, 'Other')
            groups.setdefault(group, []).append((k, str(v)))

        # Render each group as a small table, in canonical order
        lines: list[str] = []
        ordered_groups = list(_CONFIG_GROUPS.keys()) + ['Other']
        for group_name in ordered_groups:
            items = groups.get(group_name)
            if not items:
                continue
            # Skip groups where all values are None
            if all(v == 'None' for _, v in items):
                continue
            lines.append(f"**{group_name}**")
            lines.append("")
            lines.append("| Setting | Value |")
            lines.append("|---------|-------|")
            for k, v in items:
                lines.append(f"| {k} | `{v}` |")
            lines.append("")

        return '\n'.join(lines)
