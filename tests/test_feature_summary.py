"""Tests for learner.feature_summary() and its compute layer.

Fast tier: pure-assembly tests on fabricated frames
(``compute_feature_summary`` takes pre-computed pieces, so no training
is needed). Integration tier: the learner gathering path on small
trained models.
"""

import copy
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import SuggestionWarning
from mlsurv.evaluation._feature_summary import (
    FEATURE_SUMMARY_COLUMNS,
    _p_to_stars,
    compute_feature_summary,
    compute_mean_abs_shap,
    resolve_importance_pvalues,
)

# ---------------------------------------------------------------------------
# Fabricated pieces
# ---------------------------------------------------------------------------


def _uni_frame():
    """Univariate Cox table for features a, b, c (p-sorted)."""
    return pd.DataFrame({
        'feature': ['a', 'b', 'c'],
        'p_value': [0.001, 0.04, 0.8],
        'hazard_ratio': [2.0, 1.5, 1.01],
        'hr_lower_95': [1.5, 1.1, 0.7],
        'hr_upper_95': [2.7, 2.1, 1.4],
        'coefficient': [0.69, 0.41, 0.01],
    })


def _coef_frame():
    """Long coefficients: coxph (full inference) + coxnet (gated).

    Feature ``d`` is a selector-dropped row (NaN values) present only
    in the coefficient table.
    """
    return pd.DataFrame({
        'model_type': ['coxph'] * 4 + ['coxnet'] * 3,
        'feature': ['a', 'b', 'c', 'd', 'a', 'b', 'c'],
        'coefficient': [0.6, 0.3, 0.0, np.nan, 0.5, 0.2, 0.0],
        'hazard_ratio': [1.82, 1.35, 1.0, np.nan] + [np.nan] * 3,
        'hr_lower_95': [1.2, 0.9, 0.6, np.nan] + [np.nan] * 3,
        'hr_upper_95': [2.8, 2.0, 1.6, np.nan] + [np.nan] * 3,
        'p_value': [0.002, 0.2, 0.99, np.nan] + [np.nan] * 3,
    })


def _importance_frame():
    """Importance for coxph and rsf; rsf has a group feature ``grp``."""
    return pd.DataFrame({
        'model_type': ['coxph'] * 3 + ['rsf'] * 4,
        'feature': ['a', 'b', 'c', 'a', 'b', 'c', 'grp'],
        'coefficient': [0.6, 0.3, 0.0] + [np.nan] * 4,
        'perm_mean': [0.10, 0.05, 0.01, 0.08, 0.05, 0.02, 0.12],
        'perm_std': [0.01] * 7,
    })


def _shap_frame():
    """Long SHAP values for coxph only."""
    return pd.DataFrame({
        'model_type': ['coxph'] * 6,
        'feature': ['a', 'a', 'b', 'b', 'c', 'c'],
        'sample_idx': [0, 1] * 3,
        'shap_value': [0.5, -0.3, 0.2, -0.2, 0.0, 0.1],
        'feature_value': [1.0] * 6,
    })


def _pimp_frame(model_name='coxph', p_adjust='bh'):
    df = pd.DataFrame({
        'feature': ['a', 'b', 'c'],
        'observed_importance': [0.10, 0.05, 0.01],
        'null_mean': [0.0] * 3,
        'null_std': [0.01] * 3,
        'p_value': [0.001, 0.02, 0.6],
        'p_adjusted': [0.003, 0.03, 0.6],
    })
    df.attrs['model_name'] = model_name
    df.attrs['p_adjust'] = p_adjust
    return df


class _BootstrapStub:
    """Minimal stand-in for BootstrapResult."""

    def __init__(self, importance=None, summary=None, raise_on=None):
        self.importance = importance
        self._summary = summary
        self._raise_on = raise_on

    def importance_summary(self, model_type=None, p_adjust="bh"):
        if self._raise_on and model_type == self._raise_on:
            raise ValueError(f"no importance for {model_type!r}")
        return self._summary


# ---------------------------------------------------------------------------
# compute_feature_summary — pure assembly
# ---------------------------------------------------------------------------


class TestComputeFeatureSummary:

    def test_schema_and_column_order(self):
        df = compute_feature_summary(
            ['coxph'],
            coefficients=_coef_frame(),
            univariate_cox=_uni_frame(),
            importance=_importance_frame(),
        )
        assert list(df.columns) == FEATURE_SUMMARY_COLUMNS

    def test_empty_sentinel(self):
        """No models and no univariate stats -> 0-row full schema."""
        df = compute_feature_summary([])
        assert list(df.columns) == FEATURE_SUMMARY_COLUMNS
        assert len(df) == 0
        assert df.attrs['significance_source'] == {}

    def test_no_models_univariate_only_block(self):
        """Empty model list with univariate stats -> model_type NaN block."""
        df = compute_feature_summary([], univariate_cox=_uni_frame())
        assert len(df) == 3
        assert df['model_type'].isna().all()
        assert df['uni_p_value'].notna().all()
        assert df['mv_coefficient'].isna().all()

    def test_long_format_blocks_repeat_univariate(self):
        df = compute_feature_summary(
            ['coxph', 'coxnet'],
            coefficients=_coef_frame(),
            univariate_cox=_uni_frame(),
        )
        uni_cols = ['uni_p_value', 'uni_hazard_ratio',
                    'uni_hr_lower_95', 'uni_hr_upper_95']
        a_rows = df[df['feature'] == 'a']
        assert len(a_rows) == 2
        for col in uni_cols:
            assert a_rows[col].nunique() == 1

    def test_row_universe_union(self):
        """Union of univariate, coefficient, importance, SHAP features."""
        df = compute_feature_summary(
            ['coxph', 'rsf'],
            coefficients=_coef_frame(),
            univariate_cox=_uni_frame(),
            importance=_importance_frame(),
        )
        coxph = df[df['model_type'] == 'coxph']
        # Selector-dropped coefficient row survives with NaN mv values.
        assert 'd' in set(coxph['feature'])
        d_row = coxph[coxph['feature'] == 'd'].iloc[0]
        assert pd.isna(d_row['mv_coefficient'])
        assert pd.isna(d_row['uni_p_value'])
        # rsf block includes the importance-only group feature.
        rsf = df[df['model_type'] == 'rsf']
        grp_row = rsf[rsf['feature'] == 'grp'].iloc[0]
        assert pd.isna(grp_row['uni_p_value'])
        assert pd.isna(grp_row['mv_coefficient'])
        assert grp_row['perm_mean'] == 0.12
        # rsf has no coefficients at all.
        assert rsf['mv_coefficient'].isna().all()

    def test_row_order(self):
        """uni_p_value ascending NaN-last, then perm_mean descending."""
        df = compute_feature_summary(
            ['rsf'],
            univariate_cox=_uni_frame(),
            importance=_importance_frame(),
        )
        feats = df['feature'].tolist()
        # a (p=.001), b (.04), c (.8), then grp (NaN uni_p) last.
        assert feats == ['a', 'b', 'c', 'grp']

    def test_gating_columns(self):
        """coxph carries mv inference; coxnet only mv_coefficient."""
        df = compute_feature_summary(
            ['coxph', 'coxnet'],
            coefficients=_coef_frame(),
            univariate_cox=_uni_frame(),
        )
        coxph = df[df['model_type'] == 'coxph']
        coxnet = df[df['model_type'] == 'coxnet']
        assert coxph['mv_p_value'].notna().any()
        assert coxnet['mv_coefficient'].notna().any()
        assert coxnet['mv_hazard_ratio'].isna().all()
        assert coxnet['mv_p_value'].isna().all()

    def test_optional_pieces_absent_no_warning(self):
        """Absent-by-design analyses produce NaN columns, never warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            df = compute_feature_summary(
                ['coxph'],
                univariate_cox=_uni_frame(),
            )
        assert df['perm_mean'].isna().all()
        assert df['mean_abs_shap'].isna().all()
        assert df['p_adjusted'].isna().all()
        assert df['sig_stars'].isna().all()

    def test_mean_abs_shap_column(self):
        df = compute_feature_summary(
            ['coxph'],
            univariate_cox=_uni_frame(),
            shap_values=_shap_frame(),
        )
        a_val = df.loc[df['feature'] == 'a', 'mean_abs_shap'].iloc[0]
        assert a_val == pytest.approx(0.4)  # mean(|0.5|, |-0.3|)

    def test_significance_pimp_preferred(self):
        """PIMP wins over bootstrap; attrs record source and method."""
        df = compute_feature_summary(
            ['coxph'],
            univariate_cox=_uni_frame(),
            pimp_results={'coxph': _pimp_frame(p_adjust='holm')},
            bootstrap_result=_BootstrapStub(
                importance=pd.DataFrame({'model_type': ['coxph']}),
                summary=_pimp_frame(),
            ),
        )
        assert df.attrs['significance_source'] == {'coxph': 'pimp'}
        assert df.attrs['significance_method'] == {'coxph': 'holm'}
        a_row = df[df['feature'] == 'a'].iloc[0]
        assert a_row['p_adjusted'] == 0.003
        assert a_row['sig_stars'] == '**'

    def test_significance_bootstrap_fallback(self):
        df = compute_feature_summary(
            ['coxph'],
            univariate_cox=_uni_frame(),
            bootstrap_result=_BootstrapStub(
                importance=pd.DataFrame({'model_type': ['coxph']}),
                summary=_pimp_frame(),
            ),
        )
        assert df.attrs['significance_source'] == {'coxph': 'bootstrap'}
        assert df.attrs['significance_method'] == {'coxph': 'bh'}
        assert df['p_adjusted'].notna().any()

    def test_correlated_flag_dtype_and_threshold_attr(self):
        df = compute_feature_summary(
            ['coxph'],
            univariate_cox=_uni_frame(),
            correlated_features={'a'},
            correlation_threshold=0.8,
        )
        assert df['correlated'].dtype.name == 'boolean'
        assert bool(df.loc[df['feature'] == 'a', 'correlated'].iloc[0])
        assert not bool(df.loc[df['feature'] == 'b', 'correlated'].iloc[0])
        assert df.attrs['correlation_threshold'] == 0.8

    def test_correlated_unknown_is_na(self):
        """correlated_features=None -> the whole column is NA, not False."""
        df = compute_feature_summary(
            ['coxph'],
            univariate_cox=_uni_frame(),
            correlated_features=None,
        )
        assert df['correlated'].isna().all()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestPToStars:

    @pytest.mark.parametrize("p, expected", [
        (0.0005, '***'),
        (0.005, '**'),
        (0.04, '*'),
        (0.2, ''),
        (0.05, ''),       # strict <, boundary gets no star
        (np.nan, None),
        (None, None),
        (-0.1, None),
    ])
    def test_thresholds(self, p, expected):
        assert _p_to_stars(p) == expected


class TestComputeMeanAbsShap:

    def test_aggregates(self):
        out = compute_mean_abs_shap(_shap_frame())
        assert list(out.columns) == ['model_type', 'feature', 'mean_abs_shap']
        a = out[out['feature'] == 'a']['mean_abs_shap'].iloc[0]
        assert a == pytest.approx(0.4)

    def test_none_and_empty(self):
        assert compute_mean_abs_shap(None) is None
        assert compute_mean_abs_shap(_shap_frame().iloc[0:0]) is None


class TestResolveImportancePvalues:

    def test_no_sources_returns_none(self):
        result, source, method = resolve_importance_pvalues('coxph')
        assert result is None and source is None and method is None

    def test_pimp_dict_hit(self):
        result, source, method = resolve_importance_pvalues(
            'coxph', pimp_results={'coxph': _pimp_frame(p_adjust='bh')},
        )
        assert source == 'pimp' and method == 'bh'
        assert len(result) == 3

    def test_pimp_dict_miss_returns_none_without_bootstrap(self):
        """Dict-only PIMP source: a missing model resolves to nothing."""
        result, source, _ = resolve_importance_pvalues(
            'rsf', pimp_results={'coxph': _pimp_frame(model_name='coxph')},
        )
        assert result is None and source is None

    def test_bootstrap_fallback(self):
        result, source, method = resolve_importance_pvalues(
            'coxph',
            bootstrap_result=_BootstrapStub(
                importance=pd.DataFrame({'model_type': ['coxph']}),
                summary=_pimp_frame(),
            ),
        )
        assert source == 'bootstrap' and method == 'bh'

    def test_bootstrap_failure_warns_and_returns_none(self):
        stub = _BootstrapStub(
            importance=pd.DataFrame({'model_type': ['rsf']}),
            raise_on='coxph',
        )
        with pytest.warns(Warning, match="bootstrap importance"):
            result, source, _ = resolve_importance_pvalues(
                'coxph', bootstrap_result=stub,
            )
        assert result is None and source is None

    def test_empty_dict_hit_falls_through_to_bootstrap(self):
        """An empty PIMP entry must not block the bootstrap fallback."""
        result, source, method = resolve_importance_pvalues(
            'coxph',
            pimp_results={'coxph': pd.DataFrame()},
            bootstrap_result=_BootstrapStub(
                importance=pd.DataFrame({'model_type': ['coxph']}),
                summary=_pimp_frame(),
            ),
        )
        assert source == 'bootstrap' and method == 'bh'
        assert result is not None


class TestDetectCorrelatedFeaturesVectorized:
    """The vectorized pair scan must match the original loop exactly."""

    @staticmethod
    def _reference_loop(X, threshold):
        """The pre-vectorization implementation, kept as the oracle."""
        corr_matrix = X.corr()
        pairs = []
        features = X.columns.tolist()
        for i, f1 in enumerate(features):
            for j, f2 in enumerate(features):
                if i < j:
                    corr = corr_matrix.loc[f1, f2]
                    if abs(corr) >= threshold:
                        pairs.append((f1, f2, corr))
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    def test_matches_reference_with_planted_pairs_and_nan(self):
        from mlsurv.evaluation.importance import detect_correlated_features

        rng = np.random.default_rng(0)
        n = 120
        X = pd.DataFrame(rng.normal(size=(n, 6)),
                         columns=[f"f{i}" for i in range(6)])
        # Planted correlations, including a negative pair and a tie
        # candidate; one constant column gives NaN correlations.
        X['f1'] = X['f0'] * 0.95 + rng.normal(scale=0.1, size=n)
        X['f3'] = -X['f2'] * 0.95 + rng.normal(scale=0.1, size=n)
        X['f5'] = 1.0  # constant -> NaN correlation with everything

        for threshold in (0.5, 0.8, 0.99):
            got = detect_correlated_features(X, threshold=threshold)
            want = self._reference_loop(X, threshold)
            assert len(got) == len(want)
            for (g1, g2, gc), (w1, w2, wc) in zip(got, want):
                assert (g1, g2) == (w1, w2)
                assert gc == pytest.approx(wc)

    def test_boundary_pair_at_exact_threshold_included(self):
        """>= threshold, same as the original loop."""
        from mlsurv.evaluation.importance import detect_correlated_features

        # Two identical columns -> correlation exactly 1.0.
        X = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0],
                          'b': [1.0, 2.0, 3.0, 4.0],
                          'c': [4.0, 1.0, 3.0, 2.0]})
        pairs = detect_correlated_features(X, threshold=1.0)
        assert ('a', 'b', 1.0) in [(f1, f2, round(c, 12)) for f1, f2, c in pairs]

    def test_empty_and_single_column(self):
        from mlsurv.evaluation.importance import detect_correlated_features

        assert detect_correlated_features(
            pd.DataFrame({'a': [1.0, 2.0, 3.0]}), threshold=0.8) == []
        assert detect_correlated_features(
            pd.DataFrame(index=[0, 1, 2]), threshold=0.8) == []


# ---------------------------------------------------------------------------
# Learner integration
# ---------------------------------------------------------------------------


def _make_fs_data(n=80, p=4, seed=0):
    """Small tie-free survival dataset with one correlated pair."""
    from mlsurv.utils import SurvivalArray

    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)),
                     columns=[f"f{i}" for i in range(p)])
    # f3 highly correlated with f0 -> exercises the correlated flag.
    X["f3"] = X["f0"] * 0.95 + rng.normal(scale=0.1, size=n)
    risk = 0.8 * X["f0"] - 0.5 * X["f1"]
    time = rng.exponential(scale=np.exp(-risk))
    event = rng.random(n) < 0.7
    y = SurvivalArray(np.array(
        [(bool(e), float(t)) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)],
    ))
    return X, y


@pytest.fixture(scope="module")
def _fs_template():
    """Train coxph + coxnet + rsf once and evaluate; deep-copied per test."""
    from mlsurv import SurvivalLearner

    X, y = _make_fs_data()
    learner = SurvivalLearner(X, y)
    learner.setup(feature_selector=None, random_state=42, n_jobs=1)
    learner.run(["coxph", "coxnet", "rsf"], tune=False, verbose=False)
    learner.evaluate(n_importance_repeats=2, verbose=False, show_progress=False)
    return learner


@pytest.fixture
def fs_learner(_fs_template):
    return copy.deepcopy(_fs_template)


class TestFeatureSummaryWithoutModels:
    """Fast tier: setup-only paths (no model training)."""

    def test_no_models_univariate_block_warns(self):
        from mlsurv import SurvivalLearner

        X, y = _make_fs_data()
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector=None, random_state=42, n_jobs=1)
        with pytest.warns(SuggestionWarning, match="No trained models"):
            df = learner.feature_summary()
        assert len(df) == len(X.columns)
        assert df['model_type'].isna().all()
        assert df['uni_p_value'].notna().all()

    def test_nothing_to_summarize_raises(self):
        from mlsurv import SurvivalLearner

        X, y = _make_fs_data()
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector=None, random_state=42, n_jobs=1)
        learner.univariate_cox = None
        with pytest.raises(ValueError, match="Nothing to summarize"):
            learner.feature_summary()

    def test_empty_univariate_table_raises(self):
        """A 0-row univariate table counts as nothing to summarize."""
        from mlsurv import SurvivalLearner

        X, y = _make_fs_data()
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector=None, random_state=42, n_jobs=1)
        learner.univariate_cox = learner.univariate_cox.iloc[0:0]
        with pytest.raises(ValueError, match="Nothing to summarize"):
            learner.feature_summary()


@pytest.mark.integration
class TestLearnerFeatureSummary:

    def test_multi_model_blocks_and_gating(self, fs_learner):
        df = fs_learner.feature_summary()
        assert list(df.columns) == FEATURE_SUMMARY_COLUMNS
        assert list(df['model_type'].unique()) == ['coxph', 'coxnet', 'rsf']

        coxph = df[df['model_type'] == 'coxph']
        coxnet = df[df['model_type'] == 'coxnet']
        rsf = df[df['model_type'] == 'rsf']

        # CoxPH: full inference; CoxNet: coefficient only; RSF: no mv.
        assert coxph['mv_p_value'].notna().any()
        assert coxnet['mv_coefficient'].notna().any()
        assert coxnet['mv_hazard_ratio'].isna().all()
        assert coxnet['mv_p_value'].isna().all()
        assert rsf['mv_coefficient'].isna().all()
        assert rsf['perm_mean'].notna().any()

        # Univariate columns populated in every block.
        for block in (coxph, coxnet, rsf):
            assert block['uni_p_value'].notna().any()

        # Correlated pair f0/f3 flagged; others not.
        flags = dict(zip(coxph['feature'], coxph['correlated']))
        assert bool(flags['f0']) and bool(flags['f3'])
        assert not bool(flags['f1'])

    def test_single_model_equals_multi_slice(self, fs_learner):
        full = fs_learner.feature_summary()
        single = fs_learner.feature_summary('coxnet')
        slice_ = full[full['model_type'] == 'coxnet'].reset_index(drop=True)
        pd.testing.assert_frame_equal(single.reset_index(drop=True), slice_)
        assert 'model_type' in single.columns  # kept in single-model form

    def test_unknown_model_raises_valueerror(self, fs_learner):
        with pytest.raises(ValueError, match="not found"):
            fs_learner.feature_summary('nope')

    def test_no_importance_nan_perm_columns(self):
        from mlsurv import SurvivalLearner

        X, y = _make_fs_data()
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector=None, random_state=42, n_jobs=1)
        learner.train('coxph', verbose=False)
        df = learner.feature_summary()
        assert df['perm_mean'].isna().all()
        assert df['mv_coefficient'].notna().any()  # coefficients still there

    def test_attrs_pimp_injected(self, fs_learner):
        """PIMP results in _state surface through attrs and p_adjusted."""
        pimp = pd.DataFrame({
            'feature': ['f0', 'f1'],
            'observed_importance': [0.1, 0.05],
            'null_mean': [0.0, 0.0],
            'null_std': [0.01, 0.01],
            'p_value': [0.001, 0.2],
            'p_adjusted': [0.002, 0.2],
        })
        pimp.attrs['model_name'] = 'coxph'
        pimp.attrs['p_adjust'] = 'bh'
        fs_learner._state.models['coxph'].pimp_importance_test = pimp

        df = fs_learner.feature_summary()
        assert df.attrs['significance_source'] == {'coxph': 'pimp'}
        coxph = df[df['model_type'] == 'coxph']
        f0 = coxph[coxph['feature'] == 'f0'].iloc[0]
        assert f0['p_adjusted'] == 0.002
        assert f0['sig_stars'] == '**'
        # Models without significance data stay out of attrs.
        assert 'rsf' not in df.attrs['significance_source']

        # Mixed blocks: slice equality (incl. dtypes) must hold both
        # for the stars-bearing block and an all-None-stars block —
        # pins the sig_stars astype(object) against string inference.
        for name in ('coxph', 'rsf'):
            single = fs_learner.feature_summary(name)
            pd.testing.assert_frame_equal(
                single.reset_index(drop=True),
                df[df['model_type'] == name].reset_index(drop=True),
            )

    def test_correlation_failure_warns_and_na(self, fs_learner, monkeypatch):
        """Correlation failure -> InternalWarning + all-NA column."""
        import mlsurv.evaluation.importance as imp_mod
        from mlsurv._warnings import InternalWarning

        def _boom(*args, **kwargs):
            raise ValueError("corr fail")

        monkeypatch.setattr(imp_mod, 'detect_correlated_features', _boom)
        with pytest.warns(InternalWarning, match="feature correlations"):
            df = fs_learner.feature_summary()
        assert df['correlated'].isna().all()

    def test_attrs_bootstrap_only(self, fs_learner):
        fs_learner.bootstrap(
            models='coxph', n_bootstrap=2, include_importance=True,
            n_importance_repeats=1, verbose=False, show_progress=False,
        )
        df = fs_learner.feature_summary('coxph')
        assert df.attrs['significance_source'] == {'coxph': 'bootstrap'}
        assert df['p_adjusted'].notna().any()

    def test_reflects_retrain(self, fs_learner):
        """The summary is assembled live from current state."""
        del fs_learner._state.models['rsf']
        df = fs_learner.feature_summary()
        assert 'rsf' not in set(df['model_type'])

    def test_no_warning_spam(self, fs_learner):
        """RSF (no coefficients) must not trigger SuggestionWarnings."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', SuggestionWarning)
            fs_learner.feature_summary()
