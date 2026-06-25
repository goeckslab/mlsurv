"""Tests for to_dataframes() on containers and export_results() on learner."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import InternalWarning, SuggestionWarning
from mlsurv.results.containers import (
    BenchmarkResult,
    BootstrapResult,
    EvaluationResult,
    ModelResult,
    ValidationResult,
    _build_data_dictionary,
    _COLUMN_DESCRIPTIONS,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal container instances
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_learner():
    """Create a bare learner-like object (no results) for export tests."""
    from mlsurv.learner._export import ExportMixin
    from mlsurv.learner._state import LearnerState

    class FakeLearner(ExportMixin):
        def __init__(self):
            self._state = LearnerState()

    return FakeLearner()


@pytest.fixture
def bootstrap_result():
    """Minimal BootstrapResult with 2 iterations."""
    return BootstrapResult(
        summary=pd.DataFrame({
            'model_type': ['rsf', 'rsf'],
            'metric': ['harrell', 'ipcw'],
            'point': [0.72, 0.70],
            'mean': [0.71, 0.69],
            'std': [0.02, 0.03],
            'ci_lower': [0.67, 0.63],
            'ci_upper': [0.75, 0.75],
        }),
        iterations=pd.DataFrame({
            'model_type': ['rsf'] * 4,
            'iteration': [0, 0, 1, 1],
            'metric_name': ['harrell', 'ipcw', 'harrell', 'ipcw'],
            'value': [0.70, 0.68, 0.72, 0.70],
        }),
    )


@pytest.fixture
def eval_result(bootstrap_result):
    """Minimal EvaluationResult with bootstrap."""
    return EvaluationResult(
        concordance=pd.DataFrame({
            'model_type': ['rsf'],
            'c_index_train': [0.85],
            'c_index_test': [0.72],
        }),
        test_set=pd.DataFrame({
            'model_type': ['rsf'] * 3,
            'index': [0, 1, 2],
            'risk_score': [0.1, 0.5, 0.9],
            'time': [1.0, 2.0, 3.0],
            'event': [True, False, True],
        }),
        auc_t=pd.DataFrame({
            'model_type': ['rsf'] * 2,
            'eval_time': [1.0, 2.0],
            'auc': [0.70, 0.72],
        }),
        importance=pd.DataFrame({
            'model_type': ['rsf'] * 2,
            'feature': ['age', 'stage'],
            'coefficient': [0.5, 0.3],
            'perm_mean': [0.02, 0.01],
            'perm_std': [0.005, 0.003],
        }),
        importance_iterations=pd.DataFrame({
            'model_type': ['rsf'] * 4,
            'feature': ['age', 'age', 'stage', 'stage'],
            'iteration': [0, 1, 0, 1],
            'importance': [0.02, 0.025, 0.01, 0.012],
        }),
        bootstrap=bootstrap_result,
    )


@pytest.fixture
def model_result(eval_result):
    """Minimal ModelResult."""
    return ModelResult(
        name='rsf',
        model=None,
        cv_metric_scores=np.array([0.70, 0.72, 0.74]),
        cv_metric_mean=0.72,
        cv_metric_std=0.02,
        evaluation=eval_result,
    )


# ---------------------------------------------------------------------------
# BootstrapResult.to_dataframes
# ---------------------------------------------------------------------------

class TestBootstrapToDataframes:

    def test_keys(self, bootstrap_result):
        dfs = bootstrap_result.to_dataframes()
        assert 'summary' in dfs
        assert 'iterations' in dfs

    def test_none_optional_excluded(self):
        br = BootstrapResult(
            summary=pd.DataFrame({'metric': ['harrell'], 'mean': [0.7],
                                  'model_type': ['rsf'], 'point': [0.7],
                                  'std': [0.02], 'ci_lower': [0.6],
                                  'ci_upper': [0.8]}),
            iterations=pd.DataFrame({'iteration': [0], 'metric_name': ['harrell'],
                                     'value': [0.7], 'model_type': ['rsf']}),
            predictions=None,
            auc_t=None,
            importance=None,
        )
        dfs = br.to_dataframes()
        assert 'predictions' not in dfs
        assert 'auc_over_time' not in dfs
        assert 'importance' not in dfs

    def test_empty_df_excluded(self):
        br = BootstrapResult(
            summary=pd.DataFrame({'metric': ['harrell'], 'mean': [0.7],
                                  'model_type': ['rsf'], 'point': [0.7],
                                  'std': [0.02], 'ci_lower': [0.6],
                                  'ci_upper': [0.8]}),
            iterations=pd.DataFrame({'iteration': [0], 'metric_name': ['harrell'],
                                     'value': [0.7], 'model_type': ['rsf']}),
            predictions=pd.DataFrame(),  # empty
        )
        dfs = br.to_dataframes()
        assert 'predictions' not in dfs


# ---------------------------------------------------------------------------
# EvaluationResult.to_dataframes
# ---------------------------------------------------------------------------

class TestEvaluationToDataframes:

    def test_keys(self, eval_result):
        dfs = eval_result.to_dataframes()
        expected = {
            'metrics', 'predictions', 'auc_over_time',
            'importance', 'importance_iterations',
            'bootstrap_summary', 'bootstrap_iterations',
        }
        assert set(dfs.keys()) == expected

    def test_no_bootstrap(self):
        er = EvaluationResult(
            concordance=pd.DataFrame({'model_type': ['rsf'], 'c_index_test': [0.7]}),
            test_set=pd.DataFrame({'model_type': ['rsf'], 'index': [0],
                                   'risk_score': [0.5], 'time': [1.0], 'event': [1]}),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame(),
        )
        dfs = er.to_dataframes()
        assert 'metrics' in dfs
        assert 'predictions' in dfs
        # Empty DFs excluded
        assert 'auc_over_time' not in dfs
        assert 'importance' not in dfs
        # No bootstrap prefix keys
        assert not any(k.startswith('bootstrap_') for k in dfs)

    def test_shap_included_when_present(self, eval_result):
        eval_result.shap_values = pd.DataFrame({
            'model_type': ['rsf'], 'feature': ['age'],
            'sample_idx': [0], 'shap_value': [0.1],
        })
        dfs = eval_result.to_dataframes()
        assert 'shap_values' in dfs


# ---------------------------------------------------------------------------
# ModelResult.to_dataframes
# ---------------------------------------------------------------------------

class TestModelResultToDataframes:

    def test_keys(self, model_result):
        dfs = model_result.to_dataframes()
        assert 'cv_scores' in dfs
        assert 'summary' in dfs
        # eval_ prefixed keys from evaluation
        assert any(k.startswith('eval_') for k in dfs)

    def test_standalone_bootstrap(self, bootstrap_result):
        mr = ModelResult(
            name='rsf', model=None,
            cv_metric_scores=np.array([0.7]),
            cv_metric_mean=0.7, cv_metric_std=0.0,
            bootstrap=bootstrap_result,
        )
        dfs = mr.to_dataframes()
        assert 'bootstrap_summary' in dfs
        assert not any(k.startswith('eval_') for k in dfs)


# ---------------------------------------------------------------------------
# ValidationResult.to_dataframes
# ---------------------------------------------------------------------------

class TestValidationToDataframes:

    def test_keys(self, eval_result):
        vr = ValidationResult(
            name='TCGA', evaluation=eval_result,
            n_samples=100, n_events=40,
        )
        dfs = vr.to_dataframes()
        assert 'cohort_info' in dfs
        ci = dfs['cohort_info']
        assert ci['cohort'].iloc[0] == 'TCGA'
        assert ci['n_samples'].iloc[0] == 100
        assert ci['event_rate'].iloc[0] == pytest.approx(0.4)
        # Also has eval keys (un-prefixed, matching plan)
        assert 'metrics' in dfs


# ---------------------------------------------------------------------------
# BenchmarkResult.to_dataframes
# ---------------------------------------------------------------------------

class TestBenchmarkToDataframes:

    def test_keys(self, eval_result):
        br = BenchmarkResult(
            evaluation=eval_result,
            univariate_cox=pd.DataFrame({
                'reference': ['ref1'], 'coefficient': [0.5],
                'hazard_ratio': [1.65], 'hr_lower_95': [1.2],
                'hr_upper_95': [2.3], 'p_value': [0.01],
            }),
            incremental=pd.DataFrame({
                'model': ['rsf'], 'reference': ['ref1'],
                'metric': ['harrell'], 'point': [0.75],
                'mean': [0.74], 'std': [0.02],
                'ci_lower': [0.70], 'ci_upper': [0.78],
            }),
            correlation=pd.DataFrame({
                'model': ['rsf'], 'reference': ['ref1'],
                'spearman_rho': [0.8], 'spearman_p': [0.001],
                'pearson_r': [0.75], 'pearson_p': [0.002],
            }),
            reference_names=['ref1'],
            model_names=['rsf'],
        )
        dfs = br.to_dataframes()
        assert 'univariate_cox' in dfs
        assert 'incremental' in dfs
        assert 'correlation' in dfs
        assert any(k.startswith('eval_') for k in dfs)
        # The exported correlation DataFrame must use the canonical
        # column name `spearman_rho` (not the old `spearman_r`).
        corr_cols = dfs['correlation'].columns
        assert 'spearman_rho' in corr_cols
        assert 'spearman_r' not in corr_cols

    def test_empty_excluded(self, eval_result):
        br = BenchmarkResult(
            evaluation=eval_result,
            univariate_cox=pd.DataFrame(),
            incremental=pd.DataFrame(),
            correlation=pd.DataFrame(),
            reference_names=[], model_names=['rsf'],
        )
        dfs = br.to_dataframes()
        assert 'univariate_cox' not in dfs
        assert 'incremental' not in dfs
        assert 'correlation' not in dfs


# ---------------------------------------------------------------------------
# _build_data_dictionary
# ---------------------------------------------------------------------------

class TestBuildDataDictionary:

    def test_schema(self):
        dfs = {
            'metrics': pd.DataFrame({
                'model_type': ['rsf'],
                'c_index_test': [0.72],
            }),
        }
        dd = _build_data_dictionary(dfs)
        assert list(dd.columns) == [
            'table', 'column', 'dtype', 'description', 'n_non_null', 'example',
        ]

    def test_known_column_has_description(self):
        dfs = {
            'metrics': pd.DataFrame({'c_index_test': [0.72]}),
        }
        dd = _build_data_dictionary(dfs)
        row = dd[dd['column'] == 'c_index_test'].iloc[0]
        assert row['description'] == _COLUMN_DESCRIPTIONS['c_index_test']

    def test_unknown_column_empty_description(self):
        dfs = {
            'metrics': pd.DataFrame({'custom_col': [1]}),
        }
        dd = _build_data_dictionary(dfs)
        row = dd[dd['column'] == 'custom_col'].iloc[0]
        assert row['description'] == ''

    def test_example_truncated(self):
        long_val = 'x' * 100
        dfs = {'t': pd.DataFrame({'col': [long_val]})}
        dd = _build_data_dictionary(dfs)
        assert len(dd['example'].iloc[0]) == 50

    def test_all_nan_column(self):
        dfs = {'t': pd.DataFrame({'col': [np.nan, np.nan]})}
        dd = _build_data_dictionary(dfs)
        row = dd.iloc[0]
        assert row['n_non_null'] == 0
        assert row['example'] == ''


# ---------------------------------------------------------------------------
# export_results() integration tests
# ---------------------------------------------------------------------------

class TestExportResults:

    @pytest.fixture
    def learner_with_results(self, fake_learner):
        """Populate fake_learner with models for export tests."""
        mr = ModelResult(
            name='rsf', model=None,
            cv_metric_scores=np.array([0.70, 0.72, 0.74]),
            cv_metric_mean=0.72, cv_metric_std=0.02,
        )
        fake_learner._state.models['rsf'] = mr
        return fake_learner

    def test_xlsx_export(self, learner_with_results, tmp_path):
        out = tmp_path / 'results.xlsx'
        result_path = learner_with_results.export_results(out, format='xlsx')
        assert Path(result_path).exists()

        # Read back
        sheets = pd.ExcelFile(out).sheet_names
        assert sheets[0] == '_data_dictionary'
        assert len(sheets) > 1

    def test_audit_flag_propagates_through_export_results(
        self, learner_with_results, tmp_path, monkeypatch,
    ):
        """audit kwarg propagates export_results → _write_xlsx → _truncate.

        Regression net: if a future refactor drops the ``audit=audit`` thread,
        every direct-call ``_truncate_sheet_names`` test still passes but the
        public API silently downgrades to summary-only.
        """
        monkeypatch.delenv('MLSURV_EXPORT_AUDIT', raising=False)

        long_key = 'subpopulation_Cancer Type_cohort_summary_extended'
        original = learner_with_results._collect_all_dataframes
        dummy = pd.DataFrame({'a': [1]})

        def patched():
            dfs = original()
            dfs[long_key] = dummy
            return dfs

        learner_with_results._collect_all_dataframes = patched

        # audit=True → full audit phrasing
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always', SuggestionWarning)
            learner_with_results.export_results(
                tmp_path / 'r_audit.xlsx', audit=True,
            )
        audit_msgs = [
            str(w.message) for w in record
            if issubclass(w.category, SuggestionWarning)
        ]
        assert any(
            long_key in m and 'audit each hardcoded sheet literal' in m
            for m in audit_msgs
        ), audit_msgs

        # default → summary phrasing
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always', SuggestionWarning)
            learner_with_results.export_results(tmp_path / 'r_default.xlsx')
        default_msgs = [
            str(w.message) for w in record
            if issubclass(w.category, SuggestionWarning)
        ]
        assert any(
            'audit=True' in m and 'audit each hardcoded sheet literal' not in m
            for m in default_msgs
        ), default_msgs

    def test_csv_export(self, learner_with_results, tmp_path):
        out = tmp_path / 'csv_results'
        result_path = learner_with_results.export_results(out, format='csv')
        result_dir = Path(result_path)
        assert result_dir.is_dir()
        assert (result_dir / '_data_dictionary.csv').exists()
        csv_files = list(result_dir.glob('*.csv'))
        assert len(csv_files) >= 2  # data_dict + at least one data file
        # Round-trip: verify content matches (new arch exports cv_scores)
        cv_scores = pd.read_csv(result_dir / 'cv_scores.csv')
        assert 'model' in cv_scores.columns
        assert cv_scores['harrell'].iloc[0] == pytest.approx(0.70)

    def test_no_results_raises(self, fake_learner, tmp_path):
        with pytest.raises(ValueError, match="No results available"):
            fake_learner.export_results(tmp_path / 'out.xlsx', format='xlsx')

    def test_unsupported_format_raises(self, learner_with_results, tmp_path):
        with pytest.raises(ValueError, match="Unsupported format"):
            learner_with_results.export_results(
                tmp_path / 'out', format='json'
            )

    def test_xlsx_data_dictionary_content(self, learner_with_results, tmp_path):
        out = tmp_path / 'results.xlsx'
        learner_with_results.export_results(out, format='xlsx')
        dd = pd.read_excel(out, sheet_name='_data_dictionary')
        assert 'table' in dd.columns
        assert 'description' in dd.columns
        assert len(dd) > 0


class TestSheetNameTruncation:

    def test_long_name_truncated(self):
        from mlsurv.learner._export import ExportMixin
        keys = ['a' * 40]
        mapping = ExportMixin._truncate_sheet_names(keys)
        assert len(mapping['a' * 40]) <= 31

    def test_duplicate_after_truncation(self):
        from mlsurv.learner._export import ExportMixin
        # Two keys that truncate to the same 31-char prefix
        prefix = 'a' * 31
        keys = [prefix + '_x', prefix + '_y']
        mapping = ExportMixin._truncate_sheet_names(keys)
        values = list(mapping.values())
        assert len(set(values)) == 2  # all unique
        assert all(len(v) <= 31 for v in values)

    def test_invalid_chars_sanitized(self):
        from mlsurv.learner._export import ExportMixin
        keys = [r'cohort:external/val[1]', 'a\\b*c?d']
        mapping = ExportMixin._truncate_sheet_names(keys)
        for v in mapping.values():
            assert not any(c in v for c in r'\/*?[]:')

    def test_sanitize_then_truncate(self):
        from mlsurv.learner._export import ExportMixin
        # Key with invalid chars near the 31-char boundary
        key = 'a' * 28 + ':b[c'  # 32 chars, : and [ should be replaced
        mapping = ExportMixin._truncate_sheet_names([key])
        name = mapping[key]
        assert len(name) <= 31
        assert ':' not in name
        assert '[' not in name

    def test_sanitize_preserves_dedup(self):
        from mlsurv.learner._export import ExportMixin
        # Two keys that differ only in invalid chars
        keys = ['cohort:a', 'cohort/a']
        mapping = ExportMixin._truncate_sheet_names(keys)
        values = list(mapping.values())
        assert len(set(values)) == 2  # both sanitize to cohort_a but get dedup
        assert values[0] == 'cohort_a'
        assert '_2' in values[1]  # dedup suffix applied

    def test_warns_on_truncation(self):
        """Emits SuggestionWarning with full map when audit=True."""
        from mlsurv.learner._export import ExportMixin

        long_key = 'subpopulation_Cancer Type_cohort_summary_extended'  # > 31
        short_key = 'coxph'

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always', SuggestionWarning)
            mapping = ExportMixin._truncate_sheet_names(
                [long_key, short_key], audit=True
            )

        assert mapping[long_key] == long_key[:31]
        assert mapping[short_key] == short_key
        msgs = [
            str(w.message) for w in record
            if issubclass(w.category, SuggestionWarning)
        ]
        assert len(msgs) == 1
        assert long_key in msgs[0]
        assert '31-char' in msgs[0]

    def test_warns_on_sanitization(self):
        """Emits SuggestionWarning with full map when audit=True."""
        from mlsurv.learner._export import ExportMixin

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always', SuggestionWarning)
            ExportMixin._truncate_sheet_names(['cohort:external'], audit=True)

        msgs = [
            str(w.message) for w in record
            if issubclass(w.category, SuggestionWarning)
        ]
        assert len(msgs) == 1
        assert 'cohort:external' in msgs[0]

    def test_silent_when_all_fit(self):
        """No warning when every name is valid and fits within 31 chars."""
        from mlsurv.learner._export import ExportMixin

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always', SuggestionWarning)
            ExportMixin._truncate_sheet_names(['coxph', 'rsf', 'coxnet'])

        suggestion_warnings = [
            w for w in record if issubclass(w.category, SuggestionWarning)
        ]
        assert not suggestion_warnings

    def test_silent_on_empty_list(self):
        """Empty input returns an empty mapping with no warning."""
        from mlsurv.learner._export import ExportMixin

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always', SuggestionWarning)
            mapping = ExportMixin._truncate_sheet_names([])

        assert mapping == {}
        assert not [
            w for w in record if issubclass(w.category, SuggestionWarning)
        ]

    def test_sentinel_passthrough_is_silent(self):
        """``_data_dictionary`` + short user key leaves everything unchanged."""
        from mlsurv.learner._export import ExportMixin

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always', SuggestionWarning)
            mapping = ExportMixin._truncate_sheet_names(['_data_dictionary', 'rsf'])

        assert mapping == {'_data_dictionary': '_data_dictionary', 'rsf': 'rsf'}
        assert not [
            w for w in record if issubclass(w.category, SuggestionWarning)
        ]

    def test_collision_suffix_warns(self):
        """Collision dedup (`_2`, `_3`) must still fire the warning.

        Two keys that truncate to the same 31-char prefix get dedup suffixes.
        With audit=True the warning must name both keys — a regression here
        would silently swallow the collision case.
        """
        from mlsurv.learner._export import ExportMixin

        prefix = 'a' * 31
        keys = [prefix + '_x', prefix + '_y']  # both truncate to same prefix

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always', SuggestionWarning)
            mapping = ExportMixin._truncate_sheet_names(keys, audit=True)

        # Dedup suffix applied
        assert len(set(mapping.values())) == 2
        assert all(len(v) <= 31 for v in mapping.values())

        msgs = [
            str(w.message) for w in record
            if issubclass(w.category, SuggestionWarning)
        ]
        assert len(msgs) == 1
        # Both colliding keys must appear in the warning payload
        assert keys[0] in msgs[0]
        assert keys[1] in msgs[0]

    def test_warning_includes_long_key_excludes_sentinel(self):
        """audit=True payload mentions the user key, not ``_data_dictionary``.

        ``_data_dictionary`` passes through unchanged, so the audit
        mapping should include only the user-visible truncated key.
        """
        from mlsurv.learner._export import ExportMixin

        long_key = 'subpopulation_Cancer Type_cohort_summary_extended'

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always', SuggestionWarning)
            ExportMixin._truncate_sheet_names(
                ['_data_dictionary', long_key], audit=True
            )

        msgs = [
            str(w.message) for w in record
            if issubclass(w.category, SuggestionWarning)
        ]
        assert len(msgs) == 1
        assert long_key in msgs[0]
        assert '_data_dictionary' not in msgs[0]

    # ---------- summary-by-default + opt-in audit ----------

    def test_default_warning_is_summary(self, monkeypatch):
        """Default (audit=False): summary with counts + one example + brief hint."""
        from mlsurv.learner._export import ExportMixin

        monkeypatch.delenv('MLSURV_EXPORT_AUDIT', raising=False)
        long_key = 'subpopulation_Cancer Type_cohort_summary_extended'

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always', SuggestionWarning)
            ExportMixin._truncate_sheet_names([long_key, 'coxph'])

        msgs = [
            str(w.message) for w in record
            if issubclass(w.category, SuggestionWarning)
        ]
        assert len(msgs) == 1
        msg = msgs[0]
        assert '31-char' in msg
        assert '1 Excel sheet name' in msg
        assert 'truncated=1' in msg
        assert 'audit=True' in msg
        # Brief, actionable: short downstream pointer + one example, but
        # NOT the full dict-literal rename map.
        assert 'downstream' in msg.lower()
        assert long_key in msg  # the single example
        # Full audit phrasing must not appear (that goes with the dict literal)
        assert 'audit each hardcoded sheet literal' not in msg

    def test_audit_flag_emits_full_map(self):
        """audit=True restores full mapping + downstream-consumer guidance."""
        from mlsurv.learner._export import ExportMixin

        long_key = 'subpopulation_Cancer Type_cohort_summary_extended'

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always', SuggestionWarning)
            ExportMixin._truncate_sheet_names([long_key, 'coxph'], audit=True)

        msgs = [
            str(w.message) for w in record
            if issubclass(w.category, SuggestionWarning)
        ]
        assert len(msgs) == 1
        msg = msgs[0]
        assert long_key in msg
        assert 'scripts/figures/' in msg
        # Full-audit-only phrasing — distinguishes from summary form
        assert 'audit each hardcoded sheet literal' in msg

    @pytest.mark.parametrize('value', ['1', 'true', 'TRUE', 'yes', 'on', ' 1 '])
    def test_env_var_enables_audit(self, monkeypatch, value):
        """MLSURV_EXPORT_AUDIT promotes summary to full map for truthy values."""
        from mlsurv.learner._export import ExportMixin

        monkeypatch.setenv('MLSURV_EXPORT_AUDIT', value)
        long_key = 'subpopulation_Cancer Type_cohort_summary_extended'

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always', SuggestionWarning)
            ExportMixin._truncate_sheet_names([long_key])  # audit defaults to False

        msgs = [
            str(w.message) for w in record
            if issubclass(w.category, SuggestionWarning)
        ]
        assert len(msgs) == 1
        assert 'audit each hardcoded sheet literal' in msgs[0]

    @pytest.mark.parametrize('value', ['0', 'false', 'no', '', 'garbage'])
    def test_env_var_falsy_values_keep_summary(self, monkeypatch, value):
        """Only recognized truthy values enable audit; everything else → summary."""
        from mlsurv.learner._export import ExportMixin

        monkeypatch.setenv('MLSURV_EXPORT_AUDIT', value)
        long_key = 'subpopulation_Cancer Type_cohort_summary_extended'

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always', SuggestionWarning)
            ExportMixin._truncate_sheet_names([long_key])

        msgs = [
            str(w.message) for w in record
            if issubclass(w.category, SuggestionWarning)
        ]
        assert len(msgs) == 1
        assert 'audit each hardcoded sheet literal' not in msgs[0]
        assert 'audit=True' in msgs[0]  # summary form's hint

    def test_summary_counts_categories(self, monkeypatch):
        """Summary breaks down truncated / sanitized / collisions counts.

        Tags are non-exclusive: a key can be both sanitized AND truncated,
        so counts may sum to more than the total renamed-keys count.
        """
        from mlsurv.learner._export import ExportMixin

        monkeypatch.delenv('MLSURV_EXPORT_AUDIT', raising=False)
        keys = [
            'a' * 40,                # truncated only
            'cohort:external',       # sanitized only (<=31 after replacement)
            'cohort:' + 'a' * 40,    # sanitized AND truncated (overlap)
            'b' * 31 + '_x',         # truncated only (claims slot)
            'b' * 31 + '_y',         # collision (also truncated)
        ]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always', SuggestionWarning)
            ExportMixin._truncate_sheet_names(keys)

        msgs = [
            str(w.message) for w in record
            if issubclass(w.category, SuggestionWarning)
        ]
        assert len(msgs) == 1
        msg = msgs[0]
        # 5 total renames. Truncated tag: a*40, cohort:+a*40, b*31_x, b*31_y → 4.
        # Sanitized tag: cohort:external, cohort:+a*40 → 2.
        # Collision tag: b*31_y → 1.
        assert '5 Excel sheet name' in msg
        assert 'truncated=4' in msg
        assert 'sanitized=2' in msg
        assert 'collisions=1' in msg
        assert 'counts may overlap' in msg

    def test_triple_collision_yields_distinct_names(self):
        """A key equal to a prior suffixed name must not re-collide.

        k1/k2 both truncate to 'a'*31; k2 becomes 'a'*29+'_2'. k3 is *literally*
        'a'*29+'_2', so it collides with k2's resolved name and must be bumped
        again — not silently mapped to the same sheet.
        """
        from mlsurv.learner._export import ExportMixin
        k1, k2, k3 = 'a' * 31 + 'x', 'a' * 31 + 'y', 'a' * 29 + '_2'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mapping = ExportMixin._truncate_sheet_names([k1, k2, k3])
        values = list(mapping.values())
        assert len(set(values)) == 3, values
        assert all(len(v) <= 31 for v in values)

    def test_triple_collision_write_xlsx_keeps_all_sheets(self, tmp_path):
        """Real xlsx write must not silently drop a colliding sheet."""
        from mlsurv.learner._export import ExportMixin
        k1, k2, k3 = 'a' * 31 + 'x', 'a' * 31 + 'y', 'a' * 29 + '_2'
        dataframes = {
            k1: pd.DataFrame({'v': [1]}),
            k2: pd.DataFrame({'v': [2]}),
            k3: pd.DataFrame({'v': [3]}),
        }
        data_dict = pd.DataFrame({'key': ['d'], 'description': ['x']})
        out = tmp_path / 'out.xlsx'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ExportMixin._write_xlsx(out, dataframes, data_dict)

        xl = pd.ExcelFile(out)
        data_sheets = [s for s in xl.sheet_names if s != '_data_dictionary']
        assert len(data_sheets) == 3  # all three survive — no overwrite
        written = {pd.read_excel(out, sheet_name=s)['v'].iloc[0] for s in data_sheets}
        assert written == {1, 2, 3}

    def test_write_xlsx_creates_parent_dir(self, tmp_path):
        """_write_xlsx must create a missing parent directory."""
        from mlsurv.learner._export import ExportMixin
        out = tmp_path / 'no' / 'dir' / 'out.xlsx'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ExportMixin._write_xlsx(
                out, {'sheet1': pd.DataFrame({'a': [1]})},
                pd.DataFrame({'key': ['d'], 'description': ['x']}),
            )
        assert out.exists()


class TestCollectAllDataframes:
    """Test _collect_all_dataframes with various result configurations."""

    def test_validation_results_prefixed(self, fake_learner):
        eval_r = EvaluationResult(
            concordance=pd.DataFrame({'model_type': ['rsf'], 'c_index_test': [0.7]}),
            test_set=pd.DataFrame({'model_type': ['rsf'], 'index': [0],
                                   'risk_score': [0.5], 'time': [1.0], 'event': [1]}),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame(),
        )
        fake_learner._state.validation_results = {
            'TCGA': ValidationResult(
                name='TCGA', evaluation=eval_r,
                n_samples=100, n_events=40,
            ),
        }
        dfs = fake_learner._collect_all_dataframes()
        assert any(k.startswith('val_TCGA_') for k in dfs)

    def _seed_model(self, fake_learner, name, **fields):
        """Register a bare ModelResult carrying per-model test tables."""
        mr = ModelResult(
            name=name, model=None, cv_metric_scores=np.array([0.7]),
            cv_metric_mean=0.7, cv_metric_std=0.0,
        )
        for attr, df in fields.items():
            setattr(mr, attr, df)
        fake_learner._state.models[name] = mr

    def test_pimp_results_per_model(self, fake_learner):
        """Per-model PIMP results are exported; no singular 'pimp' sheet."""
        self._seed_model(fake_learner, 'rsf', pimp_importance_test=pd.DataFrame(
            {'feature': ['age'], 'p_value': [0.01]}))
        self._seed_model(fake_learner, 'coxph', pimp_importance_test=pd.DataFrame(
            {'feature': ['age'], 'p_value': [0.05]}))
        dfs = fake_learner._collect_all_dataframes()
        assert 'pimp_rsf' in dfs
        assert 'pimp_coxph' in dfs
        # The singular latest-result sheet was removed.
        assert 'pimp' not in dfs

    def test_bootstrap_importance_per_model(self, fake_learner):
        """Per-model bootstrap importance-test results are exported."""
        self._seed_model(fake_learner, 'rsf', bootstrap_importance_test=pd.DataFrame(
            {'feature': ['age'], 'p_value': [0.02]}))
        dfs = fake_learner._collect_all_dataframes()
        assert 'imp_boot_rsf' in dfs

    def test_risk_group_tests_per_model(self, fake_learner):
        """Per-model logrank/rmst results are exported; no singular sheet."""
        self._seed_model(
            fake_learner, 'rsf',
            logrank_test=pd.DataFrame({'comparison': ['omnibus'], 'p_value': [0.01]}),
            rmst_test=pd.DataFrame({'group_1': ['Low'], 'p_value': [0.03]}),
        )
        self._seed_model(fake_learner, 'coxph', logrank_test=pd.DataFrame(
            {'comparison': ['omnibus'], 'p_value': [0.04]}))
        dfs = fake_learner._collect_all_dataframes()
        assert 'logrank_rsf' in dfs
        assert 'logrank_coxph' in dfs
        assert 'rmst_rsf' in dfs
        # The singular latest-result sheets were removed.
        assert 'logrank' not in dfs
        assert 'rmst_comparison' not in dfs

    def test_fallback_eval_without_comparison(self, fake_learner):
        """Standalone evaluation_result collected via per-model ModelResult."""
        eval_r = EvaluationResult(
            concordance=pd.DataFrame({'model_type': ['rsf'], 'c_index_test': [0.7]}),
            test_set=pd.DataFrame({'model_type': ['rsf'], 'index': [0],
                                   'risk_score': [0.5], 'time': [1.0], 'event': [1]}),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame(),
        )
        mr = ModelResult(
            name='rsf', model=None,
            cv_metric_scores=np.array([0.70]),
            cv_metric_mean=0.70, cv_metric_std=0.0,
            evaluation=eval_r,
        )
        fake_learner._state.models['rsf'] = mr
        dfs = fake_learner._collect_all_dataframes()
        assert 'eval_metrics' in dfs
        assert 'eval_predictions' in dfs

    def test_bootstrap_not_duplicated_when_embedded(self, fake_learner):
        """Bootstrap in evaluation appears via eval_ prefix, not duplicated."""
        boot = BootstrapResult(
            summary=pd.DataFrame({'model_type': ['rsf'], 'metric': ['harrell'],
                                  'point': [0.7], 'mean': [0.7], 'std': [0.02],
                                  'ci_lower': [0.6], 'ci_upper': [0.8]}),
            iterations=pd.DataFrame({'model_type': ['rsf'], 'iteration': [0],
                                     'metric_name': ['harrell'], 'value': [0.7]}),
        )
        eval_r = EvaluationResult(
            concordance=pd.DataFrame({'model_type': ['rsf'], 'c_index_test': [0.7]}),
            test_set=pd.DataFrame({'model_type': ['rsf'], 'index': [0],
                                   'risk_score': [0.5], 'time': [1.0], 'event': [1]}),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame(),
            bootstrap=boot,
        )
        # Store via per-model ModelResult (new architecture)
        mr = ModelResult(
            name='rsf', model=None,
            cv_metric_scores=np.array([0.70]),
            cv_metric_mean=0.70, cv_metric_std=0.0,
            evaluation=eval_r,
        )
        fake_learner._state.models['rsf'] = mr
        dfs = fake_learner._collect_all_dataframes()
        # Bootstrap should appear via eval_ prefix only, not duplicated
        assert 'eval_bootstrap_summary' in dfs
        assert 'bootstrap_summary' not in dfs

    def test_kang_comparison(self, fake_learner):
        """Kang comparison DataFrame is collected."""
        fake_learner._state.kang_comparison_result = pd.DataFrame({
            'model_a': ['rsf'], 'model_b': ['coxph'], 'z_statistic': [1.5],
        })
        dfs = fake_learner._collect_all_dataframes()
        assert 'kang_comparison' in dfs

    def test_multiple_validation_cohorts(self, fake_learner):
        """Multiple validation cohorts produce distinct prefixed keys."""
        for name in ('TCGA', 'GEO'):
            eval_r = EvaluationResult(
                concordance=pd.DataFrame({'model_type': ['rsf'], 'c_index_test': [0.7]}),
                test_set=pd.DataFrame({'model_type': ['rsf'], 'index': [0],
                                       'risk_score': [0.5], 'time': [1.0], 'event': [1]}),
                auc_t=pd.DataFrame(),
                importance=pd.DataFrame(),
                importance_iterations=pd.DataFrame(),
            )
            fake_learner._state.validation_results[name] = ValidationResult(
                name=name, evaluation=eval_r, n_samples=50, n_events=20,
            )
        dfs = fake_learner._collect_all_dataframes()
        assert any(k.startswith('val_TCGA_') for k in dfs)
        assert any(k.startswith('val_GEO_') for k in dfs)


class TestPairedComparisonAutoCompute:
    """Test auto-compute of paired_comparison in _collect_all_dataframes."""

    def test_paired_comparison_from_standalone_bootstrap(self, fake_learner):
        """paired_comparison computed when per-model bootstrap has 2+ models."""
        boot_rsf = BootstrapResult(
            summary=pd.DataFrame({
                'model_type': ['rsf'],
                'metric': ['harrell'],
                'point': [0.75], 'mean': [0.74],
                'std': [0.02],
                'ci_lower': [0.70], 'ci_upper': [0.78],
            }),
            iterations=pd.DataFrame({
                'model_type': ['rsf'] * 5,
                'iteration': list(range(5)),
                'metric_name': ['harrell'] * 5,
                'value': [0.73, 0.74, 0.75, 0.76, 0.72],
            }),
        )
        boot_coxph = BootstrapResult(
            summary=pd.DataFrame({
                'model_type': ['coxph'],
                'metric': ['harrell'],
                'point': [0.70], 'mean': [0.69],
                'std': [0.03],
                'ci_lower': [0.63], 'ci_upper': [0.75],
            }),
            iterations=pd.DataFrame({
                'model_type': ['coxph'] * 5,
                'iteration': list(range(5)),
                'metric_name': ['harrell'] * 5,
                'value': [0.68, 0.69, 0.70, 0.71, 0.67],
            }),
        )
        fake_learner._state.models['rsf'] = ModelResult(
            name='rsf', model=None,
            cv_metric_scores=np.array([0.74]),
            cv_metric_mean=0.74, cv_metric_std=0.0,
            bootstrap=boot_rsf,
        )
        fake_learner._state.models['coxph'] = ModelResult(
            name='coxph', model=None,
            cv_metric_scores=np.array([0.69]),
            cv_metric_mean=0.69, cv_metric_std=0.0,
            bootstrap=boot_coxph,
        )
        dfs = fake_learner._collect_all_dataframes()
        assert 'paired_comparison' in dfs

    def test_standalone_bootstrap_reachable_with_eval_no_bootstrap(
        self, fake_learner
    ):
        """Standalone bootstrap on ModelResult used when evaluation has no bootstrap."""
        eval_rsf = EvaluationResult(
            concordance=pd.DataFrame({'model_type': ['rsf'], 'c_index_test': [0.7]}),
            test_set=pd.DataFrame({'model_type': ['rsf'], 'index': [0],
                                   'risk_score': [0.5], 'time': [1.0], 'event': [1]}),
            auc_t=pd.DataFrame(),
            importance=pd.DataFrame(),
            importance_iterations=pd.DataFrame(),
        )
        boot_rsf = BootstrapResult(
            summary=pd.DataFrame({
                'model_type': ['rsf'],
                'metric': ['harrell'],
                'point': [0.75], 'mean': [0.74],
                'std': [0.02],
                'ci_lower': [0.70], 'ci_upper': [0.78],
            }),
            iterations=pd.DataFrame({
                'model_type': ['rsf'] * 5,
                'iteration': list(range(5)),
                'metric_name': ['harrell'] * 5,
                'value': [0.73, 0.74, 0.75, 0.76, 0.72],
            }),
        )
        boot_coxph = BootstrapResult(
            summary=pd.DataFrame({
                'model_type': ['coxph'],
                'metric': ['harrell'],
                'point': [0.70], 'mean': [0.69],
                'std': [0.03],
                'ci_lower': [0.63], 'ci_upper': [0.75],
            }),
            iterations=pd.DataFrame({
                'model_type': ['coxph'] * 5,
                'iteration': list(range(5)),
                'metric_name': ['harrell'] * 5,
                'value': [0.68, 0.69, 0.70, 0.71, 0.67],
            }),
        )
        # evaluation exists but has no bootstrap; standalone bootstrap is set
        fake_learner._state.models['rsf'] = ModelResult(
            name='rsf', model=None,
            cv_metric_scores=np.array([0.74]),
            cv_metric_mean=0.74, cv_metric_std=0.0,
            evaluation=eval_rsf,
            bootstrap=boot_rsf,
        )
        fake_learner._state.models['coxph'] = ModelResult(
            name='coxph', model=None,
            cv_metric_scores=np.array([0.69]),
            cv_metric_mean=0.69, cv_metric_std=0.0,
            bootstrap=boot_coxph,
        )
        dfs = fake_learner._collect_all_dataframes()
        # paired_comparison should still be computed from standalone bootstrap
        assert 'paired_comparison' in dfs

    @pytest.fixture
    def _two_model_boot_learner(self, fake_learner):
        """Set up fake_learner with two models each having standalone bootstrap."""
        boot_rsf = BootstrapResult(
            summary=pd.DataFrame({
                'model_type': ['rsf'],
                'metric': ['harrell'],
                'point': [0.75], 'mean': [0.74],
                'std': [0.02],
                'ci_lower': [0.70], 'ci_upper': [0.78],
            }),
            iterations=pd.DataFrame({
                'model_type': ['rsf'] * 5,
                'iteration': list(range(5)),
                'metric_name': ['harrell'] * 5,
                'value': [0.73, 0.74, 0.75, 0.76, 0.72],
            }),
        )
        boot_coxph = BootstrapResult(
            summary=pd.DataFrame({
                'model_type': ['coxph'],
                'metric': ['harrell'],
                'point': [0.70], 'mean': [0.69],
                'std': [0.03],
                'ci_lower': [0.63], 'ci_upper': [0.75],
            }),
            iterations=pd.DataFrame({
                'model_type': ['coxph'] * 5,
                'iteration': list(range(5)),
                'metric_name': ['harrell'] * 5,
                'value': [0.68, 0.69, 0.70, 0.71, 0.67],
            }),
        )
        fake_learner._state.models['rsf'] = ModelResult(
            name='rsf', model=None,
            cv_metric_scores=np.array([0.74]),
            cv_metric_mean=0.74, cv_metric_std=0.0,
            bootstrap=boot_rsf,
        )
        fake_learner._state.models['coxph'] = ModelResult(
            name='coxph', model=None,
            cv_metric_scores=np.array([0.69]),
            cv_metric_mean=0.69, cv_metric_std=0.0,
            bootstrap=boot_coxph,
        )
        return fake_learner

    @pytest.mark.parametrize("exc", [
        ValueError("value test error"),
        KeyError("key test error"),
        TypeError("type test error"),
    ])
    def test_paired_comparison_compare_raises_emits_user_warning(
        self, monkeypatch, _two_model_boot_learner, exc
    ):
        """bootstrap.compare() raising ValueError, KeyError, or TypeError emits UserWarning, omits table."""
        # Monkeypatch BootstrapResult.combine to return a bootstrap whose
        # compare() raises the desired exception.
        combined = BootstrapResult(
            summary=pd.DataFrame({
                'model_type': ['rsf', 'coxph'],
                'metric': ['harrell', 'harrell'],
                'point': [0.75, 0.70], 'mean': [0.74, 0.69],
                'std': [0.02, 0.03],
                'ci_lower': [0.70, 0.63], 'ci_upper': [0.78, 0.75],
            }),
            iterations=pd.DataFrame({
                'model_type': ['rsf'] * 5 + ['coxph'] * 5,
                'iteration': list(range(5)) * 2,
                'metric_name': ['harrell'] * 10,
                'value': [0.73, 0.74, 0.75, 0.76, 0.72,
                          0.68, 0.69, 0.70, 0.71, 0.67],
            }),
        )
        def _raise():
            raise exc
        monkeypatch.setattr(combined, 'compare', _raise)
        monkeypatch.setattr(
            BootstrapResult, 'combine',
            classmethod(lambda cls, models: combined),
        )
        with pytest.warns(InternalWarning, match="Paired comparison could not be computed during export:"):
            dfs = _two_model_boot_learner._collect_all_dataframes()
        assert 'paired_comparison' not in dfs


class TestToDataframesIdempotent:
    """Calling to_dataframes() twice should not crash."""

    def test_model_result_to_dataframes_twice(self):
        mr = ModelResult(
            name='rsf', model=None,
            cv_metric_scores=np.array([0.70]),
            cv_metric_mean=0.70, cv_metric_std=0.0,
        )
        dfs1 = mr.to_dataframes()
        dfs2 = mr.to_dataframes()
        assert dfs1.keys() == dfs2.keys()
