"""ExportMixin - unified export of all results to xlsx/csv."""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple, Union

import pandas as pd

from .._warnings import ExportSplitWarning, InternalWarning, SuggestionWarning

if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner


_INVALID_SHEET_CHARS = re.compile(r'[\\/*?\[\]:]')

# Excel's forbidden sheet-name characters plus the ones only Windows filenames
# reject (< > " |). Export keys embed user-supplied column and cohort names, and
# clinical ones like 'PD-L1 >1%' or 'Stage III|IV' hit exactly that gap.
_INVALID_FILENAME_CHARS = re.compile(r'[\\/*?\[\]:<>"|]')

_AUDIT_ENV_VAR = 'MLSURV_EXPORT_AUDIT'
_AUDIT_TRUTHY = frozenset({'1', 'true', 'yes', 'on'})

# Excel's hard per-worksheet limits. A table past either one cannot be written
# as a sheet at all, so it is offloaded to CSV beside the workbook. Kept at
# module level so tests can shrink them instead of building million-row frames.
_XLSX_MAX_ROWS = 1_048_576
_XLSX_MAX_COLS = 16_384

# Sheet listing the tables that were offloaded, written only when some were.
_OFFLOAD_INDEX_KEY = '_large_tables'


def _audit_env_enabled() -> bool:
    """Return True if MLSURV_EXPORT_AUDIT is set to a recognized truthy value."""
    raw = os.environ.get(_AUDIT_ENV_VAR, '').strip().lower()
    return raw in _AUDIT_TRUTHY


def _oversize_reason(df: pd.DataFrame) -> Optional[str]:
    """Return why *df* cannot fit on one worksheet, or None if it fits.

    The string is user-facing: it goes into the warning and the
    ``_large_tables`` sheet. Both dimensions are reported when both are over,
    so shrinking one does not just uncover the other.
    """
    reasons = []
    # to_excel(index=False) writes one header row above the data.
    if len(df) + 1 > _XLSX_MAX_ROWS:
        reasons.append(f"{len(df):,} rows (max {_XLSX_MAX_ROWS - 1:,})")
    if df.shape[1] > _XLSX_MAX_COLS:
        reasons.append(f"{df.shape[1]:,} columns (max {_XLSX_MAX_COLS:,})")
    return ', '.join(reasons) if reasons else None


def _sidecar_dir(path: Path) -> Path:
    """Return the directory holding *path*'s offloaded CSV tables."""
    return path.parent / f'{path.stem}_large_tables'


def _warn_stale_csvs(directory: Path, written: Iterable[str], *,
                     pointer: str, stacklevel: int) -> None:
    """Warn about CSVs in *directory* that this export did not write.

    A re-export that produces a different set of tables leaves the previous
    run's files behind, and a script globbing the directory then reads stale
    numbers. mlsurv says so rather than deleting: it cannot tell its own
    leftovers from a file the user put there, and a delete that runs before the
    workbook write would destroy the previous export whenever this one fails.
    """
    if not directory.is_dir():
        return
    keep = set(written)
    stale = sorted(p.name for p in directory.glob('*.csv') if p.name not in keep)
    if not stale:
        return
    shown = ', '.join(stale[:3])
    if len(stale) > 3:
        shown += f', and {len(stale) - 3} more'
    warnings.warn(
        f"{directory} holds {len(stale)} CSV file(s) this export did not "
        f"write ({shown}). They are most likely left over from an earlier run, "
        f"and nothing in {pointer} points at them. mlsurv does not delete "
        f"files it did not create, so remove them or export to a clean path "
        f"before anything reads the directory as a whole.",
        ExportSplitWarning,
        stacklevel=stacklevel,
    )


def _csv_filenames(keys: Iterable[str]) -> Dict[str, str]:
    """Map export keys to unique, filesystem-safe ``<key>.csv`` names.

    Sanitizing is lossy: ``a/b``, ``a:b`` and ``a_b`` all become ``a_b``. So
    collisions take a numeric suffix, the same way ``_truncate_sheet_names``
    keeps sheet names distinct. Without it one table would overwrite another on
    disk with nothing to show for it. Matching is case-insensitive because
    Windows and macOS filesystems are.
    """
    seen = set()
    mapping: Dict[str, str] = {}
    for key in keys:
        stem = _INVALID_FILENAME_CHARS.sub('_', key)
        candidate = stem
        suffix = 1
        while candidate.lower() in seen:
            suffix += 1
            candidate = f'{stem}_{suffix}'
        seen.add(candidate.lower())
        mapping[key] = f'{candidate}.csv'
    return mapping


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
        OSError
            If the ``<stem>_large_tables`` directory needed for an oversized
            table cannot be created.

        Notes
        -----
        The output xlsx includes a ``_data_dictionary`` sheet describing every
        column. Some columns (e.g., ``coefficient`` in ``eval_importance``) are
        applicable only to a subset of models; the data dictionary indicates
        which models populate each column.

        Excel caps a worksheet at 1,048,576 rows (one of them the header, so
        1,048,575 rows of data) and 16,384 columns. A table past either limit
        is written as CSV in a sibling ``<stem>_large_tables`` directory
        instead of a sheet, listed in a ``_large_tables`` sheet, and announced
        with an :class:`~mlsurv.exceptions.ExportSplitWarning`. The workbook is
        still written. Pass ``format='csv'`` to keep every table in one place.

        mlsurv never deletes files it did not create. If an earlier export left
        CSVs there that this one did not write, it warns instead; the
        ``_large_tables`` sheet is the authority on which files belong.

        See also :meth:`save_learner`, which is worth calling first.
        """
        from ..results.containers import _build_data_dictionary

        all_dfs = self._collect_all_dataframes()
        if not all_dfs:
            raise ValueError("No results available to export. Run analysis first.")

        # Built from every table, including any the xlsx path later offloads to
        # CSV, so the data dictionary documents the sidecar files too.
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
            # exc.filename points at the file that actually refused, which for
            # an offloaded table is a sidecar CSV, not the workbook.
            raise PermissionError(
                f"Cannot write to {exc.filename or path}. If this file is open "
                f"in another program (e.g., Excel), close it and retry."
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
    def _offload_oversized(
        path: Path,
        dataframes: Dict[str, pd.DataFrame],
        oversized: Dict[str, str],
    ) -> Tuple[Dict[str, Path], pd.DataFrame]:
        """Write the oversized tables to CSV beside the workbook.

        Returns the ``{key: csv path}`` mapping and the index DataFrame that
        becomes the ``_large_tables`` sheet. The index records a workbook-relative
        path, so moving the workbook and its sidecar directory together keeps the
        pointers valid.
        """
        sidecar = _sidecar_dir(path)
        try:
            sidecar.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(
                f"Cannot create {sidecar}, needed to hold {len(oversized)} "
                f"result table(s) too large for a worksheet: {exc.strerror}. "
                f"Free that path, or use export_results(..., format='csv')."
            ) from exc

        written: Dict[str, Path] = {}
        rows = []
        filenames = _csv_filenames(oversized)
        for key, reason in oversized.items():
            filename = filenames[key]
            csv_path = sidecar / filename
            dataframes[key].to_csv(csv_path, index=False)
            written[key] = csv_path
            rows.append({
                'table': key,
                'reason': reason,
                'file': f'{sidecar.name}/{filename}',
            })
        return written, pd.DataFrame(rows)

    @staticmethod
    def _write_xlsx(
        path: Path,
        dataframes: Dict[str, pd.DataFrame],
        data_dict: pd.DataFrame,
        *,
        audit: bool = False,
    ) -> Dict[str, Path]:
        """Write all DataFrames to a single Excel file.

        Tables too large for one worksheet are written as CSV in a sibling
        ``<stem>_large_tables`` directory instead of raising, and listed in a
        ``_large_tables`` sheet. Returns the ``{key: csv path}`` mapping of
        whatever was offloaded (empty when everything fit).
        """
        oversized: Dict[str, str] = {}
        writable: Dict[str, pd.DataFrame] = {}
        for key, df in dataframes.items():
            reason = _oversize_reason(df)
            if reason is None:
                writable[key] = df
            else:
                oversized[key] = reason

        path.parent.mkdir(parents=True, exist_ok=True)

        offloaded: Dict[str, Path] = {}
        index_df = None
        if oversized:
            offloaded, index_df = ExportMixin._offload_oversized(
                path, dataframes, oversized
            )

        # Offloaded keys are left out of the name map on purpose. Warning that a
        # sheet name was truncated is noise when no such sheet is written, and a
        # key with no sheet must not consume a name in the collision map, or a
        # real sheet gets pushed onto a _2 suffix.
        all_keys = ['_data_dictionary']
        if index_df is not None:
            all_keys.append(_OFFLOAD_INDEX_KEY)
        all_keys.extend(writable.keys())
        name_map = ExportMixin._truncate_sheet_names(all_keys, audit=audit)

        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            # Not size-checked: the dictionary has a fixed 6 columns and one row
            # per exported column, so it cannot approach either limit.
            data_dict.to_excel(
                writer, sheet_name=name_map['_data_dictionary'], index=False
            )
            if index_df is not None:
                index_df.to_excel(
                    writer, sheet_name=name_map[_OFFLOAD_INDEX_KEY], index=False
                )
            for key, df in writable.items():
                df.to_excel(writer, sheet_name=name_map[key], index=False)

        # Warned only once the workbook is on disk, so the message cannot
        # claim a layout that a failed write never produced.
        if oversized:
            example_key, example_reason = next(iter(oversized.items()))
            warnings.warn(
                f"{len(oversized)} result table(s) exceed Excel's per-worksheet "
                f"limits and were written as CSV to "
                f"{_sidecar_dir(path)} instead of sheets in {path.name} "
                f"(e.g. {example_key!r}: {example_reason}); the "
                f"{_OFFLOAD_INDEX_KEY!r} sheet lists them. Every other table "
                f"was written normally and no rows were dropped. Use "
                f"export_results(..., format='csv') to keep all tables in one "
                f"place.",
                ExportSplitWarning,
                stacklevel=3,
            )
        _warn_stale_csvs(
            _sidecar_dir(path), [p.name for p in offloaded.values()],
            pointer=path.name, stacklevel=3,
        )
        return offloaded

    @staticmethod
    def _write_csv_dir(
        path: Path,
        dataframes: Dict[str, pd.DataFrame],
        data_dict: pd.DataFrame,
    ) -> None:
        """Write each DataFrame as a CSV file in a directory."""
        path.mkdir(parents=True, exist_ok=True)
        data_dict.to_csv(path / '_data_dictionary.csv', index=False)
        # '_data_dictionary' leads the key list so a table of that name takes
        # the suffixed filename rather than overwriting the dictionary.
        filenames = _csv_filenames(['_data_dictionary', *dataframes])
        for key, df in dataframes.items():
            df.to_csv(path / filenames[key], index=False)

        renamed = {k: filenames[k] for k in dataframes
                   if filenames[k] != f'{k}.csv'}
        if renamed:
            example_key, example_name = next(iter(renamed.items()))
            warnings.warn(
                f"{len(renamed)} table(s) written under a different filename "
                f"because the key contains characters a filename cannot hold, "
                f"or two keys sanitized to the same name. E.g. {example_key!r} "
                f"-> {example_name!r}. Scripts that build a path from the table "
                f"key will miss these files.",
                SuggestionWarning,
                stacklevel=3,
            )

        # The directory is the whole output here, so a leftover file from an
        # earlier export is more misleading than in the xlsx sidecar.
        _warn_stale_csvs(
            path, ['_data_dictionary.csv', *filenames.values()],
            pointer='_data_dictionary.csv', stacklevel=3,
        )
