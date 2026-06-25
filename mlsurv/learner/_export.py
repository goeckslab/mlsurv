"""ExportMixin - unified export of all results to xlsx/csv."""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Union

import pandas as pd

from .._warnings import InternalWarning, SuggestionWarning

if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner


_INVALID_SHEET_CHARS = re.compile(r'[\\/*?\[\]:]')

_AUDIT_ENV_VAR = 'MLSURV_EXPORT_AUDIT'
_AUDIT_TRUTHY = frozenset({'1', 'true', 'yes', 'on'})


def _audit_env_enabled() -> bool:
    """Return True if MLSURV_EXPORT_AUDIT is set to a recognized truthy value."""
    raw = os.environ.get(_AUDIT_ENV_VAR, '').strip().lower()
    return raw in _AUDIT_TRUTHY


# Declarative list of state fields to export as DataFrames.
# Each tuple: (state_attribute, export_key).
_STATE_DF_FIELDS = [
    ('kang_comparison_result', 'kang_comparison'),
    ('ph_test_result', 'ph_test'),
    ('linearity_test_result', 'linearity_test'),
]


def _build_cv_summary_dfs(state, dfs):
    """Build CV summary DataFrame from per-model data and add to *dfs*."""
    if not state.models:
        return
    cv_rows = []
    for name, mr in state.models.items():
        cv_df = mr.cv_results_df().copy()
        cv_df.insert(0, 'model', name)
        cv_rows.append(cv_df)
    if cv_rows:
        dfs['cv_scores'] = pd.concat(cv_rows, ignore_index=True)


def _build_standalone_bootstrap_dfs(state, evaluated, eval_result, dfs):
    """Add standalone bootstrap DataFrames to *dfs* if not covered by eval."""
    from ..results.containers import BootstrapResult

    combined_eval_has_boot = (
        evaluated and eval_result is not None
        and eval_result.bootstrap is not None
    )
    has_standalone_boot = any(
        m.bootstrap is not None for m in state.models.values()
    )
    if has_standalone_boot and not combined_eval_has_boot:
        boot = BootstrapResult.combine(state.models)
        if boot is not None:
            for k, df in boot.to_dataframes().items():
                dfs[f'bootstrap_{k}'] = df


def _collect_model_eval_dfs(state):
    """Collect eval + bootstrap DataFrames from per-model ModelResult data."""
    from ..results.containers import EvaluationResult

    dfs = {}
    _build_cv_summary_dfs(state, dfs)

    # Combine evaluations
    eval_result = None
    evaluated = {
        n: m for n, m in state.models.items()
        if m.evaluation is not None
    }
    if evaluated:
        eval_result = EvaluationResult.combine(evaluated)
        for k, df in eval_result.to_dataframes().items():
            dfs[f'eval_{k}'] = df

    _build_standalone_bootstrap_dfs(state, evaluated, eval_result, dfs)
    return dfs


def _collect_per_model_importance_tests(state, all_dfs, _is_nonempty_df):
    """Add per-model importance-test DataFrames (PIMP + bootstrap) to all_dfs."""
    for model_name, pimp_df in state.pimp_importance_test_results.items():
        if _is_nonempty_df(pimp_df):
            all_dfs[f'pimp_{model_name}'] = pimp_df
    # 'imp_boot_' keeps the longest built-in key
    # ('imp_boot_weibull_aft', 20 chars) inside Excel's 31-char
    # sheet-name limit; longer keys (custom models) fall back to
    # _truncate_sheet_names, which warns.
    for model_name, boot_df in state.bootstrap_importance_test_results.items():
        if _is_nonempty_df(boot_df):
            all_dfs[f'imp_boot_{model_name}'] = boot_df


def _collect_per_model_risk_group_tests(state, all_dfs, _is_nonempty_df):
    """Add per-model logrank/rmst test DataFrames to all_dfs.

    The ``rmst_`` prefix (not ``rmst_comparison_``) keeps the longest built-in
    key (``rmst_weibull_aft``, 16 chars) inside Excel's 31-char
    sheet-name limit, mirroring the ``imp_boot_`` choice above; longer keys
    (custom models) fall back to _truncate_sheet_names, which warns.
    """
    for model_name, logrank_df in state.logrank_test_results.items():
        if _is_nonempty_df(logrank_df):
            all_dfs[f'logrank_{model_name}'] = logrank_df
    for model_name, rmst_df in state.rmst_test_results.items():
        if _is_nonempty_df(rmst_df):
            all_dfs[f'rmst_{model_name}'] = rmst_df


def _resolve_paired_bootstrap(state):
    """Resolve the bootstrap result for paired comparison export."""
    from mlsurv.viz._data_metrics import _has_any_bootstrap

    if not _has_any_bootstrap(state.models):
        return None
    from ..results.containers import BootstrapResult
    return BootstrapResult.combine(state.models)


def _collect_subpopulation_dfs(state, all_dfs, _is_nonempty_df):
    """Add subpopulation analysis DataFrames to all_dfs."""
    for group_var, sg_result in state.subpopulation_results.items():
        prefix = f'subpopulation_{group_var}'
        if _is_nonempty_df(sg_result.summary):
            all_dfs[f'{prefix}_cohort'] = sg_result.summary
        if _is_nonempty_df(sg_result.metrics):
            all_dfs[f'{prefix}_metrics'] = sg_result.metrics
        if _is_nonempty_df(sg_result.bootstrap_summary):
            all_dfs[f'{prefix}_bootstrap'] = sg_result.bootstrap_summary
        if _is_nonempty_df(getattr(sg_result, 'auc_t', None)):
            all_dfs[f'{prefix}_auc_t'] = sg_result.auc_t


def _collect_paired_comparison(bootstrap, all_dfs):
    """Try to compute paired comparison from bootstrap and add to all_dfs."""
    if bootstrap is None or bootstrap.iterations is None:
        return
    model_types = bootstrap.iterations.get('model_type')
    if model_types is not None and model_types.nunique() >= 2:
        try:
            all_dfs['paired_comparison'] = bootstrap.compare()
        except (ValueError, KeyError, TypeError) as exc:
            warnings.warn(
                f"Paired comparison could not be computed during export: {exc}. "
                "This table will be omitted from the export.",
                InternalWarning,
                stacklevel=4,
            )


class ExportMixin:
    """Mixin for unified export of all results to file(s)."""

    def _collect_all_dataframes(self: SurvivalLearner) -> Dict[str, pd.DataFrame]:
        """Collect DataFrames from all result containers with namespaced keys."""
        from ..results.containers import _is_nonempty_df

        all_dfs: Dict[str, pd.DataFrame] = {}
        s = self._state

        # Per-model eval + bootstrap
        all_dfs.update(_collect_model_eval_dfs(s))

        # Validation cohorts
        for name, val in s.validation_results.items():
            for k, df in val.to_dataframes().items():
                all_dfs[f'val_{name}_{k}'] = df

        # Benchmark
        if s.benchmark_result is not None:
            for k, df in s.benchmark_result.to_dataframes().items():
                all_dfs[f'benchmark_{k}'] = df

        # Per-model importance-test results (PIMP + bootstrap)
        _collect_per_model_importance_tests(s, all_dfs, _is_nonempty_df)

        # Per-model risk-group test results (logrank + rmst)
        _collect_per_model_risk_group_tests(s, all_dfs, _is_nonempty_df)

        # Declarative state fields (paired comparisons + statistical tests)
        for attr, key in _STATE_DF_FIELDS:
            val = getattr(s, attr, None)
            if _is_nonempty_df(val):
                all_dfs[key] = val

        # Paired bootstrap comparison
        bootstrap = _resolve_paired_bootstrap(s)

        # Subpopulation analysis
        _collect_subpopulation_dfs(s, all_dfs, _is_nonempty_df)

        # Paired comparison from bootstrap
        _collect_paired_comparison(bootstrap, all_dfs)

        return all_dfs

    def export_results(
        self: SurvivalLearner,
        path: Union[str, Path],
        *,
        format: str = 'xlsx',
        audit: bool = False,
    ) -> str:
        """Export all available results to file(s).

        Parameters
        ----------
        path : str or Path
            Output path. For 'xlsx', a single file path (e.g., 'results.xlsx').
            For 'csv', a directory path (one file per table).
        format : {'xlsx', 'csv'}
            Output format.
        audit : bool, default False
            If True, the Excel-sheet-name truncation warning includes the full
            ``{original: truncated}`` mapping and the downstream-consumer
            guidance. If False (default), a one-line summary is emitted with
            counts of truncated/sanitized/collided names. Can also be enabled
            globally via the ``MLSURV_EXPORT_AUDIT=1`` environment variable.
            Ignored for ``csv`` (sheet-name limit is xlsx-only).

        Returns
        -------
        str
            Absolute path to the written file or directory.

        Raises
        ------
        ValueError
            If no results are available or ``format`` is unsupported.
        PermissionError
            If the output path cannot be written (e.g. the target file is
            open in another program such as Excel).

        Notes
        -----
        The output xlsx includes a ``_data_dictionary`` sheet describing every
        column. Some columns (e.g., ``coefficient`` in ``eval_importance``) are
        applicable only to a subset of models; the data dictionary indicates
        which models populate each column.
        """
        from ..results.containers import _build_data_dictionary

        all_dfs = self._collect_all_dataframes()
        if not all_dfs:
            raise ValueError("No results available to export. Run analysis first.")

        data_dict = _build_data_dictionary(all_dfs)
        path = Path(path)

        valid_formats = ('xlsx', 'csv')
        if format not in valid_formats:
            raise ValueError(
                f"Unsupported format: {format!r}. Choose from {list(valid_formats)}."
            )
        try:
            if format == 'xlsx':
                self._write_xlsx(path, all_dfs, data_dict, audit=audit)
            else:
                self._write_csv_dir(path, all_dfs, data_dict)
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot write to {path}. If this file is open in another "
                f"program (e.g., Excel), close it and retry."
            ) from exc
        return str(path.resolve())

    @staticmethod
    def _truncate_sheet_names(
        keys: list,
        *,
        audit: bool = False,
    ) -> Dict[str, str]:
        """Map original keys to Excel-safe sheet names (max 31 chars).

        Returns a dict mapping original key -> truncated name. If any key
        was sanitized or truncated (excluding the ``_data_dictionary``
        internal sentinel), emits a single :class:`SuggestionWarning`.

        By default the warning is a one-line summary with counts; pass
        ``audit=True`` (or set ``MLSURV_EXPORT_AUDIT=1``) for the full
        ``{original: truncated}`` mapping plus the downstream-consumer
        guidance — useful when chasing down hardcoded sheet names in R
        scripts under ``scripts/figures/``.
        """
        seen: Dict[str, int] = {}
        mapping: Dict[str, str] = {}
        # Tags are non-exclusive: a single key can be both sanitized and
        # truncated (and/or collided). Counts therefore may sum to more
        # than the number of renamed keys — by design.
        n_sanitized = 0
        n_truncated = 0
        n_collisions = 0
        for key in keys:
            sanitized = _INVALID_SHEET_CHARS.sub('_', key)
            was_sanitized = sanitized != key
            was_truncated = len(sanitized) > 31
            name = sanitized[:31]
            if name in seen:
                n_collisions += 1
                # Bump the suffix until we land on a name not already assigned.
                # Recording the suffixed name in `seen` (below) is essential:
                # otherwise a later key whose truncated name equals this suffix
                # would collide undetected and overwrite this sheet silently.
                base_count = seen[name]
                while True:
                    base_count += 1
                    suffix = f'_{base_count}'
                    candidate = sanitized[:31 - len(suffix)] + suffix
                    if candidate not in seen:
                        break
                seen[name] = base_count
                name = candidate
            seen[name] = seen.get(name, 0) + 1
            mapping[key] = name
            if name != key:
                if was_sanitized:
                    n_sanitized += 1
                if was_truncated:
                    n_truncated += 1

        # Any key whose final sheet name differs from the input was either
        # sanitized (invalid Excel chars replaced) or truncated (>31 chars).
        # The ``_data_dictionary`` sentinel is short enough and char-clean
        # today, but we intentionally do NOT special-case it here: if anyone
        # ever renames it to something longer or containing ``:``/``/``, we
        # want the warning to fire loudly instead of silently masking the
        # mismatch with the later ``name_map['_data_dictionary']`` lookup.
        truncated = {k: v for k, v in mapping.items() if k != v}
        if truncated:
            if audit or _audit_env_enabled():
                warnings.warn(
                    f"Excel sheet names truncated/renamed to fit the 31-char "
                    f"limit: {truncated}. Downstream consumers that hardcode "
                    f"sheet names (e.g. R scripts in scripts/figures/) may "
                    f"break — audit each hardcoded sheet literal "
                    f"after renaming producer constants.",
                    SuggestionWarning,
                    stacklevel=4,
                )
            else:
                example_key, example_name = next(iter(truncated.items()))
                warnings.warn(
                    f"{len(truncated)} Excel sheet name(s) renamed to fit the "
                    f"31-char limit (truncated={n_truncated}, "
                    f"sanitized={n_sanitized}, collisions={n_collisions}; "
                    f"counts may overlap). E.g. {example_key!r} -> "
                    f"{example_name!r}. Downstream consumers that hardcode "
                    f"sheet names (R scripts) may break — "
                    f"pass audit=True or set MLSURV_EXPORT_AUDIT=1 for the "
                    f"full rename map.",
                    SuggestionWarning,
                    stacklevel=4,
                )
        return mapping

    @staticmethod
    def _write_xlsx(
        path: Path,
        dataframes: Dict[str, pd.DataFrame],
        data_dict: pd.DataFrame,
        *,
        audit: bool = False,
    ) -> None:
        """Write all DataFrames to a single Excel file."""
        all_keys = ['_data_dictionary'] + list(dataframes.keys())
        name_map = ExportMixin._truncate_sheet_names(all_keys, audit=audit)

        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            data_dict.to_excel(
                writer, sheet_name=name_map['_data_dictionary'], index=False
            )
            for key, df in dataframes.items():
                df.to_excel(writer, sheet_name=name_map[key], index=False)

    @staticmethod
    def _write_csv_dir(
        path: Path,
        dataframes: Dict[str, pd.DataFrame],
        data_dict: pd.DataFrame,
    ) -> None:
        """Write each DataFrame as a CSV file in a directory."""
        path.mkdir(parents=True, exist_ok=True)
        data_dict.to_csv(path / '_data_dictionary.csv', index=False)
        for key, df in dataframes.items():
            df.to_csv(path / f'{key}.csv', index=False)
