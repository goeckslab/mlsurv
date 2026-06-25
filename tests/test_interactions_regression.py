"""Numeric regression tests pinning H-statistic and SHAP-interaction outputs.

Recorded BEFORE the R11 refactor of ``mlsurv/evaluation/_interactions.py``.
Both kernels must produce byte-for-byte identical output
across the refactor — these tests are the safety net.

Goldens are stored in ``tests/_data/r11_goldens.npz`` and checked into
the repository. To regenerate (only when intentionally changing
behavior), delete the file and re-run; the test will write fresh
goldens and pass with a ``GENERATED`` marker.

Row ordering is deterministic: rows are sorted by a tolerance-bucketed
interaction strength (rounded to 1e-12) and ties are broken by feature
name, so near-zero noise pairs cannot reorder across platforms/BLAS
builds. The golden therefore pins the intended stable
order, not platform-specific float-jitter order.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsurv.evaluation._interactions import (
    _compute_h_statistic,
    _compute_shap_interactions,
)

GOLDENS_PATH = Path(__file__).parent / "_data" / "r11_goldens.npz"
GOLDENS_PATH.parent.mkdir(exist_ok=True)


# ── Synthetic predict-only model for H-stat (no real fit) ───────────────


class _MultiplicativeRiskModel:
    """Risk = x0 * x1 + 0.3 * x2 + 0.5 * (g_b - g_c).

    x0/x1 carry the dominant pairwise interaction; the categorical
    group ``g`` (one-hot dummies g_a, g_b, g_c) is additive.
    """

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return (
            X[:, 0] * X[:, 1]
            + 0.3 * X[:, 2]
            + 0.5 * (X[:, 4] - X[:, 5])
        )


def _make_grouped_dataframe(n=80, seed=42):
    rng = np.random.RandomState(seed)
    levels = rng.choice(['a', 'b', 'c'], size=n)
    return pd.DataFrame({
        'x0': rng.randn(n),
        'x1': rng.randn(n),
        'x2': rng.randn(n),
        'g_a': (levels == 'a').astype(float),
        'g_b': (levels == 'b').astype(float),
        'g_c': (levels == 'c').astype(float),
    })


# ── Helpers for goldens I/O ─────────────────────────────────────────────


def _save_goldens(arrays: dict) -> None:
    np.savez(GOLDENS_PATH, **arrays)


def _load_goldens() -> dict:
    with np.load(GOLDENS_PATH, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


# ── H-stat regression ───────────────────────────────────────────────────


def _h_payload():
    """Compute H-stat outputs for the pinned fixture (no surfaces + with surfaces)."""
    X = _make_grouped_dataframe()
    model = _MultiplicativeRiskModel()
    groups = {'g': ['g_a', 'g_b', 'g_c']}
    features = ['x0', 'x1', 'x2', 'g']

    df = _compute_h_statistic(
        model, X, features,
        max_samples=80, random_state=42,
        categorical_groups=groups,
    )
    df_surf, surfaces = _compute_h_statistic(
        model, X, features,
        max_samples=80, random_state=42,
        compute_surfaces=True, grid_resolution=10,
        categorical_groups=groups,
    )
    return df, df_surf, surfaces


def test_h_statistic_regression():
    """Pin H-stat DataFrame + surface arrays for the grouped fixture."""
    df, df_surf, surfaces = _h_payload()

    payload = {
        'h_pairs': df[['feature_1', 'feature_2']].to_numpy(),
        'h_values': df['h_statistic'].to_numpy(dtype=np.float64),
        'h_method': df['method'].to_numpy(),
        'h_surf_pairs': df_surf[['feature_1', 'feature_2']].to_numpy(),
        'h_surf_values': df_surf['h_statistic'].to_numpy(dtype=np.float64),
        'h_n_surfaces': np.array(len(surfaces)),
    }
    for i, s in enumerate(surfaces):
        payload[f'h_surf{i}_f1'] = np.array(s['feature_1'])
        payload[f'h_surf{i}_f2'] = np.array(s['feature_2'])
        payload[f'h_surf{i}_pdp'] = np.asarray(s['pdp_surface'], dtype=np.float64)
        payload[f'h_surf{i}_grid1'] = np.asarray(s['grid_1'], dtype=np.float64)
        payload[f'h_surf{i}_grid2'] = np.asarray(s['grid_2'], dtype=np.float64)
        payload[f'h_surf{i}_grid1_type'] = np.array(s['grid_1_type'])
        payload[f'h_surf{i}_grid2_type'] = np.array(s['grid_2_type'])
        payload[f'h_surf{i}_grid1_labels'] = np.array(
            s['grid_1_labels'] if s['grid_1_labels'] is not None else [],
        )
        payload[f'h_surf{i}_grid2_labels'] = np.array(
            s['grid_2_labels'] if s['grid_2_labels'] is not None else [],
        )

    if not GOLDENS_PATH.exists():
        _save_goldens(payload)
        pytest.skip(f"GENERATED goldens at {GOLDENS_PATH}; rerun to assert.")

    golden = _load_goldens()
    np.testing.assert_array_equal(payload['h_pairs'], golden['h_pairs'])
    np.testing.assert_allclose(
        payload['h_values'], golden['h_values'],
        rtol=0, atol=1e-12, equal_nan=True,
    )
    np.testing.assert_array_equal(payload['h_method'], golden['h_method'])
    np.testing.assert_array_equal(payload['h_surf_pairs'], golden['h_surf_pairs'])
    np.testing.assert_allclose(
        payload['h_surf_values'], golden['h_surf_values'],
        rtol=0, atol=1e-12, equal_nan=True,
    )
    assert int(payload['h_n_surfaces']) == int(golden['h_n_surfaces'])
    n = int(golden['h_n_surfaces'])
    for i in range(n):
        for key in (f'h_surf{i}_f1', f'h_surf{i}_f2',
                    f'h_surf{i}_grid1_type', f'h_surf{i}_grid2_type',
                    f'h_surf{i}_grid1_labels', f'h_surf{i}_grid2_labels'):
            np.testing.assert_array_equal(payload[key], golden[key])
        for key in (f'h_surf{i}_pdp', f'h_surf{i}_grid1', f'h_surf{i}_grid2'):
            np.testing.assert_allclose(
                payload[key], golden[key],
                rtol=0, atol=1e-12,
            )


# ── SHAP regression ─────────────────────────────────────────────────────


def _make_survival_data_for_rsf(n=120, seed=42):
    """Synthetic survival data with the same grouped-feature shape."""
    rng = np.random.RandomState(seed)
    X = _make_grouped_dataframe(n=n, seed=seed)
    risk = X['x0'].values * X['x1'].values + 0.3 * X['x2'].values
    # Exponential times scaled by risk; censoring at random
    times = rng.exponential(scale=np.exp(-risk).clip(0.1, 10.0))
    events = rng.rand(n) < 0.7
    y = np.array(
        list(zip(events.astype(bool), times.astype(float))),
        dtype=[('event', '?'), ('time', '<f8')],
    )
    return X, y


@pytest.mark.shap
@pytest.mark.integration
def test_shap_interactions_regression():
    """Pin SHAP-interaction summary + raw matrix for a small RSF fit."""
    pytest.importorskip("shap")
    pytest.importorskip("sksurv")
    from sksurv.ensemble import RandomSurvivalForest
    from sklearn.pipeline import Pipeline

    X, y = _make_survival_data_for_rsf(n=120, seed=42)
    rsf = RandomSurvivalForest(
        n_estimators=20, max_depth=4, min_samples_leaf=10,
        random_state=42, n_jobs=1,
    )
    # Wrap in a Pipeline that does no preprocessing so X passes through.
    # _decompose_pipeline expects a model wrapper exposing the inner
    # estimator via .model_ — emulate that contract minimally.
    class _PassThroughWrapper:
        def __init__(self, inner):
            self.model_ = inner

        def fit(self, X, y):
            self.model_.fit(X, y)
            return self

        def predict(self, X):
            return self.model_.predict(X)

    wrapper = _PassThroughWrapper(rsf).fit(X, y)
    pipeline = Pipeline([('model', wrapper)])

    groups = {'g': ['g_a', 'g_b', 'g_c']}
    features = ['x0', 'x1', 'x2', 'g']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, raw, names = _compute_shap_interactions(
            pipeline, X, features,
            max_samples=60, random_state=42,
            categorical_groups=groups,
        )

    payload = {
        'shap_pairs': df[['feature_1', 'feature_2']].to_numpy(),
        'shap_values': df['shap_interaction'].to_numpy(dtype=np.float64),
        'shap_method': df['method'].to_numpy(),
        'shap_raw': np.asarray(raw, dtype=np.float64),
        'shap_names': np.array(names),
    }

    # Merge into the existing goldens file (or create if missing).
    existing = _load_goldens() if GOLDENS_PATH.exists() else {}
    if not any(k.startswith('shap_') for k in existing):
        existing.update(payload)
        _save_goldens(existing)
        pytest.skip(f"GENERATED SHAP goldens at {GOLDENS_PATH}; rerun to assert.")

    golden = _load_goldens()
    np.testing.assert_array_equal(payload['shap_pairs'], golden['shap_pairs'])
    np.testing.assert_allclose(
        payload['shap_values'], golden['shap_values'],
        rtol=0, atol=1e-10, equal_nan=True,
    )
    np.testing.assert_array_equal(payload['shap_method'], golden['shap_method'])
    np.testing.assert_array_equal(payload['shap_names'], golden['shap_names'])
    assert payload['shap_raw'].shape == golden['shap_raw'].shape
    np.testing.assert_allclose(
        payload['shap_raw'], golden['shap_raw'],
        rtol=0, atol=1e-10,
    )
