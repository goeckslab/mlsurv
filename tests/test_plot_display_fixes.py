"""Display fixes for tuning-history, beeswarm, grouped-forest, and metric-heatmap plots."""

import types

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# plot_optuna
# ---------------------------------------------------------------------------

def _saved_result(values):
    """A ModelResult stand-in as left by load_learner(): no live study."""
    return types.SimpleNamespace(
        optuna_study=None,
        _optuna_tuning_history=[
            {'trial': i, 'value': v, 'state': 'COMPLETE'}
            for i, v in enumerate(values)
        ],
        _optuna_param_importances={'alpha': 0.7, 'l1_ratio': 0.3},
    )


def test_saved_tuning_history_draws_best_value_line():
    """A saved learner's history matches the live-study plot: trial points
    plus a running-best line (studies maximize)."""
    from mlsurv.viz._data import build_combined_optuna_figure

    values = [0.60, 0.58, 0.65, 0.62, 0.70, 0.69]
    fig = build_combined_optuna_figure({'coxnet': _saved_result(values)})

    history = [t for t in fig.data if t.xaxis == 'x2']
    by_name = {t.name: t for t in history}
    assert set(by_name) == {'Objective Value', 'Best Value'}
    assert by_name['Objective Value'].mode == 'markers'
    assert list(by_name['Objective Value'].y) == values
    assert list(by_name['Best Value'].y) == [0.60, 0.60, 0.65, 0.65, 0.70, 0.70]


def test_saved_tuning_history_best_line_skips_nan():
    """A NaN trial value does not poison the running best."""
    from mlsurv.viz._data import build_combined_optuna_figure

    fig = build_combined_optuna_figure(
        {'coxnet': _saved_result([0.60, float('nan'), 0.65])}
    )
    best = next(t for t in fig.data if t.name == 'Best Value')
    assert list(best.y) == [0.60, 0.60, 0.65]


@pytest.mark.parametrize('n_models', [2, 5, 8])
def test_optuna_row_gap_stays_fixed_in_pixels(n_models):
    """The gap between model rows stays ~80 px instead of growing with the
    number of models (a fixed fraction left 60% of a 5-model figure empty)."""
    from mlsurv.viz._data import build_combined_optuna_figure

    results = {f'm{i}': _saved_result([0.6, 0.7]) for i in range(n_models)}
    fig = build_combined_optuna_figure(results)

    row1 = fig.layout.yaxis.domain      # first row, column 1
    row2 = fig.layout.yaxis3.domain     # second row, column 1
    gap_px = (row1[0] - row2[1]) * fig.layout.height
    assert gap_px == pytest.approx(80, abs=1)


# ---------------------------------------------------------------------------
# SHAP beeswarm xlim
# ---------------------------------------------------------------------------

def _shap_frames():
    rng = np.random.default_rng(0)
    n = 30
    X = pd.DataFrame({'a': rng.normal(size=n), 'b': rng.normal(size=n)})
    rows = []
    for i in range(n):
        rows.append({'feature': 'a', 'sample_idx': i, 'shap_value': rng.normal()})
        rows.append({'feature': 'b', 'sample_idx': i, 'shap_value': rng.normal() * 0.2})
    rows[0]['shap_value'] = -8.0   # one extreme value
    return pd.DataFrame(rows), X


def test_beeswarm_xlim_sets_shap_axis_range():
    from mlsurv.viz import plot_shap_beeswarm

    shap_df, X = _shap_frames()
    fig = plot_shap_beeswarm(shap_df, X=X, xlim=(-2.0, 2.0))
    assert tuple(fig.layout.xaxis.range) == (-2.0, 2.0)


def test_beeswarm_default_range_is_unset():
    from mlsurv.viz import plot_shap_beeswarm

    shap_df, X = _shap_frames()
    fig = plot_shap_beeswarm(shap_df, X=X)
    assert fig.layout.xaxis.range is None


@pytest.mark.parametrize('bad', [(1.0, -1.0), (0.0, 0.0), (1.0,)])
def test_beeswarm_rejects_bad_xlim(bad):
    from mlsurv.viz import plot_shap_beeswarm

    shap_df, X = _shap_frames()
    with pytest.raises(ValueError, match="xlim"):
        plot_shap_beeswarm(shap_df, X=X, xlim=bad)


# ---------------------------------------------------------------------------
# Grouped (subpopulation) forest
# ---------------------------------------------------------------------------

def test_grouped_forest_omits_model_with_no_values():
    """A reference with no per-level values (e.g. PD-L1 missing in every
    subgroup) gets no legend entry and no empty row."""
    from mlsurv.viz._context import SubpopulationAxis
    from mlsurv.viz._metrics_helpers import plot_grouped_forest

    metrics_long = pd.DataFrame([
        {'model_type': m, 'subpopulation': sg, 'metric': 'harrell', 'value': v}
        for m, vals in [('coxph', [0.70, 0.66]), ('PD-L1', [np.nan, np.nan])]
        for sg, v in zip(['A', 'B'], vals)
    ])
    axis = SubpopulationAxis(
        group_variable='grp', levels=['A', 'B'], metrics_long=metrics_long,
    )
    fig = plot_grouped_forest(axis, scoring='harrell', ref_names={'PD-L1'})

    names = {t.name for t in fig.data if t.name}
    assert not any('PD-L1' in n for n in names)
    assert any('CoxPH' in n or 'coxph' in n for n in names)


# ---------------------------------------------------------------------------
# Metric heatmaps: missing cells
# ---------------------------------------------------------------------------

def test_heatmap_cell_text_marks_missing_values():
    from mlsurv.viz.metrics import _heatmap_cell_text

    raw = np.array([[0.7254, np.nan], [0.6, 0.0]])
    mask = np.isnan(raw)
    labels = _heatmap_cell_text(raw, mask)
    assert labels.tolist() == [['0.725', '–'], ['0.600', '0.000']]


def test_metric_heatmap_missing_cell_is_not_zero():
    """Plotly renders a None cell as "0.000" under "%{text:.3f}", which reads
    as a real score; missing cells must show a dash instead."""
    from mlsurv.viz.metrics import plot_metric_heatmap

    df = pd.DataFrame({
        'model_type': ['coxph', 'rsf', 'gbsa'],
        'harrell': [0.72, 0.77, 0.75],
        'ipcw': [0.71, np.nan, 0.74],
    })
    with pytest.warns(Warning):
        fig = plot_metric_heatmap(df)
    trace = fig.data[0]
    assert trace.texttemplate == '%{text}'
    text = np.array(trace.text, dtype=object)
    assert '–' in text.ravel().tolist()
    assert '0.000' not in text.ravel().tolist()


def test_faceted_heatmap_missing_cell_is_not_zero():
    from mlsurv.viz.metrics import plot_faceted_metric_heatmap

    cohorts = {
        'A': pd.DataFrame({'model_type': ['coxph', 'rsf'],
                           'harrell': [0.72, 0.70], 'ipcw': [0.71, np.nan]}),
        'B': pd.DataFrame({'model_type': ['coxph', 'rsf'],
                           'harrell': [0.65, 0.66], 'ipcw': [0.64, 0.63]}),
    }
    with pytest.warns(Warning):
        fig = plot_faceted_metric_heatmap(cohorts, metrics=['harrell', 'ipcw'])
    texts = [v for t in fig.data if t.type == 'heatmap'
             for v in np.array(t.text, dtype=object).ravel().tolist()]
    assert '–' in texts
    assert '0.000' not in texts
