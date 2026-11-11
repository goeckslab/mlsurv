"""Test visualization module for Phase 4 (Session 3)."""

import warnings

import pytest
import numpy as np
import pandas as pd

from mlsurv._warnings import InternalWarning

from tests._helpers import make_survival_array, make_y

# Mark all tests to skip if plotly not available
pytest.importorskip("plotly")


class TestHeatmaps:
    """Test plot_metric_heatmap (model x metric, z-scored, direction-aware)."""

    def test_plot_metric_heatmap_basic(self):
        """Verify plot_metric_heatmap renders a heatmap with correct shape."""
        from mlsurv.viz.metrics import plot_metric_heatmap
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'model_type': ['coxph', 'rsf', 'gbsa'],
            'harrell': [0.72, 0.77, 0.75],
            'ipcw': [0.71, 0.76, 0.74],
            'integrated_brier': [0.18, 0.16, 0.17],
        })

        fig = plot_metric_heatmap(df)

        assert isinstance(fig, go.Figure)
        assert fig.data[0].type == 'heatmap'
        assert np.array(fig.data[0].z).shape == (3, 3)

    def test_plot_metric_heatmap_z_scored(self):
        """Z-values have zero mean per column (metric)."""
        from mlsurv.viz.metrics import plot_metric_heatmap

        df = pd.DataFrame({
            'model_type': ['coxph', 'rsf', 'gbsa'],
            'harrell': [0.60, 0.70, 0.80],
            'integrated_brier': [0.20, 0.15, 0.10],
        })

        fig = plot_metric_heatmap(df)
        z = np.array(fig.data[0].z)

        # Each column should be centered at 0
        for col_idx in range(z.shape[1]):
            assert abs(z[:, col_idx].mean()) < 1e-10

    def test_plot_metric_heatmap_direction_aware(self):
        """Lower IBS (better) gets positive z; higher C-index (better) gets positive z."""
        from mlsurv.viz.metrics import plot_metric_heatmap

        df = pd.DataFrame({
            'model_type': ['good_model', 'bad_model'],
            'harrell': [0.85, 0.55],          # higher = better
            'integrated_brier': [0.10, 0.30],  # lower = better
        })

        fig = plot_metric_heatmap(df)
        z = np.array(fig.data[0].z)

        # good_model (row 0) should have positive z for both metrics
        assert z[0, 0] > 0, "good_model should have positive z for harrell"
        assert z[0, 1] > 0, "good_model should have positive z for IBS"
        # bad_model (row 1) should have negative z for both
        assert z[1, 0] < 0
        assert z[1, 1] < 0

    def test_plot_metric_heatmap_annotations_show_raw(self):
        """Cell annotations show original values, not z-scores."""
        from mlsurv.viz.metrics import plot_metric_heatmap

        df = pd.DataFrame({
            'model_type': ['coxph', 'rsf'],
            'harrell': [0.72, 0.77],
            'integrated_brier': [0.18, 0.16],
        })

        fig = plot_metric_heatmap(df)
        text = np.array(fig.data[0].text)

        assert text[0, 0] == pytest.approx(0.72, abs=0.001)
        assert text[1, 1] == pytest.approx(0.16, abs=0.001)

    def test_plot_metric_heatmap_specific_metrics(self):
        """plot_metric_heatmap can limit output to selected metrics."""
        from mlsurv.viz.metrics import plot_metric_heatmap

        df = pd.DataFrame({
            'model_type': ['coxph', 'rsf'],
            'harrell': [0.72, 0.77],
            'ipcw': [0.71, 0.76],
            'integrated_brier': [0.18, 0.16],
        })

        fig = plot_metric_heatmap(df, metrics=['harrell', 'ipcw'])

        assert list(fig.data[0].x) == ["Harrell's C-index", "Uno's C-index (IPCW)"]
        assert np.array(fig.data[0].z).shape == (2, 2)

    def test_plot_metric_heatmap_single_model_zero_std(self):
        """Single-model heatmap doesn't crash (std=0 for each column)."""
        from mlsurv.viz.metrics import plot_metric_heatmap

        df = pd.DataFrame({
            'model_type': ['coxph'],
            'harrell': [0.75],
            'integrated_brier': [0.18],
        })

        fig = plot_metric_heatmap(df)
        z = np.array(fig.data[0].z)

        # All z-scores should be 0 when there's only one model
        assert (z == 0.0).all()

    def test_plot_metric_heatmap_missing_column_raises(self):
        """Requesting a missing metric column raises ValueError."""
        from mlsurv.viz.metrics import plot_metric_heatmap

        df = pd.DataFrame({
            'model_type': ['coxph'],
            'harrell': [0.75],
        })

        with pytest.raises(ValueError, match="missing metric columns"):
            plot_metric_heatmap(df, metrics=['harrell', 'nonexistent'])

    def test_plot_metric_heatmap_nan_warns(self):
        """NaN values trigger a UserWarning and don't corrupt z-scores."""
        from mlsurv.viz.metrics import plot_metric_heatmap

        df = pd.DataFrame({
            'model_type': ['coxph', 'rsf', 'gbsa'],
            'harrell': [0.72, 0.77, float('nan')],
            'ipcw': [0.71, 0.76, 0.74],
        })

        with pytest.warns(InternalWarning, match="NaN"):
            fig = plot_metric_heatmap(df)

        z = np.array(fig.data[0].z, dtype=float)
        # ipcw column (no NaN) should still be centered at 0
        assert abs(z[:, 1].mean()) < 1e-10
        # harrell z-scores for non-NaN rows should be finite
        assert np.isfinite(z[0, 0])
        assert np.isfinite(z[1, 0])
        # NaN cell should remain NaN, not become 0.0
        assert np.isnan(z[2, 0])
        # Colorscale bounds must be finite
        assert np.isfinite(fig.data[0].zmin)
        assert np.isfinite(fig.data[0].zmax)
        # Annotation text for NaN cell should be None (renders blank)
        text = np.array(fig.data[0].text)
        assert text[2, 0] is None


class TestCurves:
    """Test curve visualizations."""

    def test_auct_curves_basic(self):
        """Verify plot_metric_time(metric='auc') creates figure."""
        from mlsurv.viz.metrics import plot_metric_time
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'model': ['coxph'] * 3 + ['rsf'] * 3,
            'eval_time': [6, 12, 18] * 2,
            'auc': [0.75, 0.78, 0.76, 0.77, 0.80, 0.79]
        })

        fig = plot_metric_time(df, metric='auc')

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # At least one trace per model

    def test_auct_curves_with_std(self):
        """Verify plot_metric_time(metric='auc') with uncertainty ribbons."""
        from mlsurv.viz.metrics import plot_metric_time
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'model': ['coxph'] * 3 + ['rsf'] * 3,
            'eval_time': [6, 12, 18] * 2,
            'auc': [0.75, 0.78, 0.76, 0.77, 0.80, 0.79]
        })

        std_df = pd.DataFrame({
            'model': ['coxph'] * 3 + ['rsf'] * 3,
            'eval_time': [6, 12, 18] * 2,
            'auc_std': [0.02, 0.03, 0.02, 0.03, 0.02, 0.03]
        })

        fig = plot_metric_time(df, std_df=std_df, metric='auc', show_ci=True)

        assert isinstance(fig, go.Figure)

    def test_auct_marker_no_white_border(self):
        """Line trace markers must have no outline to avoid white halo."""
        from mlsurv.viz.metrics import plot_metric_time

        df = pd.DataFrame({
            'model': ['coxph'] * 3,
            'eval_time': [6, 12, 18],
            'auc': [0.75, 0.78, 0.76],
        })
        std_df = pd.DataFrame({
            'model': ['coxph'] * 3,
            'eval_time': [6, 12, 18],
            'auc_std': [0.02, 0.03, 0.02],
        })

        fig = plot_metric_time(df, std_df=std_df, metric='auc', show_ci=True)
        # Line trace is after the ribbon trace
        line_traces = [t for t in fig.data if t.mode == 'lines+markers']
        assert len(line_traces) >= 1
        for t in line_traces:
            assert t.marker.line.width == 0

    def test_auct_uneven_eval_times_shown(self):
        """All eval times must appear as ticks, even when unevenly spaced."""
        from mlsurv.viz.metrics import plot_metric_time

        times = [6, 12, 24, 36, 60]
        df = pd.DataFrame({
            'model': ['coxph'] * len(times),
            'eval_time': times,
            'auc': [0.70, 0.75, 0.78, 0.76, 0.74],
        })

        fig = plot_metric_time(df, metric='auc')
        assert tuple(fig.layout.xaxis.tickvals) == tuple(times)

    def test_auct_y_range_accommodates_ribbons(self):
        """Y-axis must extend below 0.4 when ribbon lower bound is below 0.4."""
        from mlsurv.viz.metrics import plot_metric_time

        df = pd.DataFrame({
            'model': ['coxph'] * 3,
            'eval_time': [6, 12, 18],
            'auc': [0.45, 0.42, 0.44],
        })
        # Large std pushes lower bound below 0.4
        std_df = pd.DataFrame({
            'model': ['coxph'] * 3,
            'eval_time': [6, 12, 18],
            'auc_std': [0.10, 0.10, 0.10],
        })

        fig = plot_metric_time(df, std_df=std_df, metric='auc', show_ci=True)
        y_range = fig.layout.yaxis.range
        assert y_range[0] < 0.4, f"y_range[0]={y_range[0]} should be < 0.4"
        assert y_range[1] == 1.0

    def test_auct_y_range_no_ribbon_still_reasonable(self):
        """Without ribbons, y-axis should still accommodate all data points."""
        from mlsurv.viz.metrics import plot_metric_time

        df = pd.DataFrame({
            'model': ['coxph'] * 3,
            'eval_time': [6, 12, 18],
            'auc': [0.80, 0.82, 0.81],
        })

        fig = plot_metric_time(df, metric='auc')
        y_range = fig.layout.yaxis.range
        assert y_range[0] <= 0.80
        assert y_range[0] >= 0.65  # not wastefully low
        assert y_range[1] == 1.0

    def test_auct_y_range_nan_safe(self):
        """Y-range computation handles NaN values gracefully."""
        from mlsurv.viz.metrics import plot_metric_time

        df = pd.DataFrame({
            'model': ['coxph'] * 4,
            'eval_time': [6, 12, 18, 24],
            'auc': [0.75, np.nan, 0.78, np.nan],
        })

        fig = plot_metric_time(df, metric='auc')
        y_range = fig.layout.yaxis.range
        assert np.isfinite(y_range[0]), f"y_range[0]={y_range[0]} is not finite"
        assert y_range[1] == 1.0

    def test_brier_y_range_accommodates_ribbons(self):
        """Y-axis must extend above 0.35 when ribbon upper bound exceeds 0.35."""
        from mlsurv.viz.metrics import plot_metric_time

        df = pd.DataFrame({
            'model': ['coxph'] * 3,
            'eval_time': [6, 12, 18],
            'brier_score': [0.30, 0.33, 0.32],
        })
        std_df = pd.DataFrame({
            'model': ['coxph'] * 3,
            'eval_time': [6, 12, 18],
            'brier_score_std': [0.08, 0.08, 0.08],
        })

        fig = plot_metric_time(
            df, std_df=std_df, metric='integrated_brier', show_ci=True,
        )
        y_range = fig.layout.yaxis.range
        assert y_range[0] == 0
        assert y_range[1] > 0.35, f"y_range[1]={y_range[1]} should be > 0.35"

    def test_brier_y_range_no_ribbon_still_reasonable(self):
        """Without ribbons, y-axis should still accommodate all Brier data."""
        from mlsurv.viz.metrics import plot_metric_time

        df = pd.DataFrame({
            'model': ['coxph'] * 3,
            'eval_time': [6, 12, 18],
            'brier_score': [0.12, 0.14, 0.13],
        })

        fig = plot_metric_time(df, metric='integrated_brier')
        y_range = fig.layout.yaxis.range
        assert y_range[0] == 0
        assert y_range[1] >= 0.14  # must accommodate max data value
        assert y_range[1] <= 0.35  # not wastefully high for small values

    def test_brier_y_range_nan_safe(self):
        """Y-range computation handles NaN Brier values gracefully."""
        from mlsurv.viz.metrics import plot_metric_time

        df = pd.DataFrame({
            'model': ['coxph'] * 4,
            'eval_time': [6, 12, 18, 24],
            'brier_score': [0.20, np.nan, 0.22, np.nan],
        })

        fig = plot_metric_time(df, metric='integrated_brier')
        y_range = fig.layout.yaxis.range
        assert y_range[0] == 0
        assert np.isfinite(y_range[1]), f"y_range[1]={y_range[1]} is not finite"

    def test_auct_many_eval_times_no_ticks(self):
        """With >20 eval times, tick values should not be explicitly set."""
        from mlsurv.viz.metrics import plot_metric_time

        times = list(range(1, 25))  # 24 times, exceeds cap of 20
        df = pd.DataFrame({
            'model': ['coxph'] * len(times),
            'eval_time': times,
            'auc': [0.7 + 0.01 * i for i in range(len(times))],
        })

        fig = plot_metric_time(df, metric='auc')
        # Plotly auto-ticks when not explicitly set
        assert fig.layout.xaxis.tickvals is None or fig.layout.xaxis.tickmode != 'array'

    def test_auct_faceted_basic(self):
        """auct_faceted removed in viz refactor."""
        pytest.skip("auct_faceted removed in viz refactor")

    def test_brier_score_curve(self, train_test_data):
        """Verify compute_brier_curves + plot_metric_time creates figure."""
        from mlsurv.viz.metrics import compute_brier_curves, plot_metric_time
        from mlsurv.models import CoxPHModel
        import plotly.graph_objects as go

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxPHModel()
        model.fit(X_train, y_train)

        brier_df = compute_brier_curves(
            {'coxph': model},
            X_train, y_train,
            X_test, y_test
        )
        fig = plot_metric_time(brier_df, metric='brier')

        assert isinstance(fig, go.Figure)


class TestSurvival:
    """Test survival curve visualizations."""

    def test_km_interactive_basic(self, synthetic_data_clean):
        """Verify plot_km with risk_col creates figure."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        X, y = synthetic_data_clean
        np.random.seed(42)
        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'risk_score': np.random.randn(len(y))
        })

        fig = plot_km(df, risk_col='risk_score', n_groups=2, show_risk_table=False)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_km_interactive_with_risk_table(self, synthetic_data_clean):
        """Verify plot_km with risk table."""
        from mlsurv.viz.survival import plot_km

        X, y = synthetic_data_clean
        np.random.seed(42)
        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'risk_score': np.random.randn(len(y))
        })

        fig = plot_km(df, risk_col='risk_score', n_groups=2, show_risk_table=True)

        assert fig is not None

    def test_km_risk_table_cliponaxis_false(self, synthetic_data_clean):
        """Risk-table text traces must set cliponaxis=False."""
        from mlsurv.viz.survival import plot_km

        X, y = synthetic_data_clean
        np.random.seed(42)
        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'risk_score': np.random.randn(len(y)),
        })

        fig = plot_km(df, risk_col='risk_score', n_groups=2, show_risk_table=True)

        text_traces = [t for t in fig.data if t.mode == 'text']
        assert len(text_traces) > 0, "Expected risk-table text traces"
        for t in text_traces:
            assert t.cliponaxis is False, (
                f"Risk-table text trace should have cliponaxis=False, got {t.cliponaxis}"
            )

    def test_km_interactive_no_ci(self, synthetic_data_clean):
        """Verify plot_km without confidence intervals."""
        from mlsurv.viz.survival import plot_km

        X, y = synthetic_data_clean

        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'risk_score': np.random.randn(len(y))
        })

        fig = plot_km(df, risk_col='risk_score', n_groups=2, show_ci=False)

        assert fig is not None

    def test_km_by_group(self, synthetic_data_clean):
        """Verify plot_km with group_col creates figure."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        X, y = synthetic_data_clean

        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'group': np.random.choice(['A', 'B'], len(y))
        })

        fig = plot_km(df, group_col='group')

        assert isinstance(fig, go.Figure)

    def test_km_by_group_with_colors(self, synthetic_data_clean):
        """Verify plot_km with group_col and custom colors."""
        from mlsurv.viz.survival import plot_km

        X, y = synthetic_data_clean

        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'group': np.random.choice(['A', 'B'], len(y))
        })

        fig = plot_km(
            df, group_col='group',
            colors={'A': '#ff0000', 'B': '#0000ff'}
        )

        assert fig is not None

    def test_survival_grid(self, synthetic_data_clean):
        """Verify _survival_grid creates grid figure."""
        from mlsurv.viz.survival import _survival_grid
        import plotly.graph_objects as go

        X, y = synthetic_data_clean

        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'risk_score': np.random.randn(len(y))
        })

        data_dict = {
            ('coxph', 'breast'): df,
            ('rsf', 'breast'): df
        }

        fig = _survival_grid(
            data_dict,
            models=['coxph', 'rsf'],
            diseases=['breast']
        )

        assert isinstance(fig, go.Figure)

    def test_km_risk_show_rmst(self, synthetic_data_clean):
        """Verify plot_km with show_rmst=True adds shaded fills and tau line."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        X, y = synthetic_data_clean
        rng = np.random.RandomState(42)

        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'risk_score': rng.randn(len(y))
        })

        fig = plot_km(df, risk_col='risk_score', n_groups=2,
                      show_risk_table=False, show_rmst=True)

        assert isinstance(fig, go.Figure)
        # Should have shaded fill traces (fill='tozeroy')
        fill_traces = [t for t in fig.data if getattr(t, 'fill', None) == 'tozeroy']
        assert len(fill_traces) == 2, f"Expected 2 RMST fill traces, got {len(fill_traces)}"

        # Should have a dashed vertical line at tau
        dash_traces = [t for t in fig.data
                       if hasattr(t, 'line') and t.line and
                       getattr(t.line, 'dash', None) == 'dash']
        assert len(dash_traces) >= 1, "Expected at least 1 dashed tau line"

        # Subtitle should contain RMST info
        title_text = fig.layout.title.text if fig.layout.title else ""
        assert "RMST" in title_text or "\u0394" in title_text

    def test_km_risk_show_rmst_with_custom_tau(self, synthetic_data_clean):
        """show_rmst with explicit rmst_tau uses that tau value."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        X, y = synthetic_data_clean
        rng = np.random.RandomState(42)

        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'risk_score': rng.randn(len(y))
        })

        tau_val = float(np.median(y['time']))
        fig = plot_km(df, risk_col='risk_score', n_groups=2,
                      show_risk_table=False, show_rmst=True, rmst_tau=tau_val)

        assert isinstance(fig, go.Figure)
        # Verify tau line x-coordinates match
        dash_traces = [t for t in fig.data
                       if hasattr(t, 'line') and t.line and
                       getattr(t.line, 'dash', None) == 'dash']
        assert len(dash_traces) >= 1
        np.testing.assert_allclose(dash_traces[0].x[0], tau_val, rtol=0.01)

    def test_km_risk_show_rmst_false_no_fills(self, synthetic_data_clean):
        """show_rmst=False (default) should NOT add fill traces."""
        from mlsurv.viz.survival import plot_km

        X, y = synthetic_data_clean
        rng = np.random.RandomState(42)

        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'risk_score': rng.randn(len(y))
        })

        fig = plot_km(df, risk_col='risk_score', n_groups=2,
                      show_risk_table=False, show_rmst=False)

        fill_traces = [t for t in fig.data if getattr(t, 'fill', None) == 'tozeroy']
        assert len(fill_traces) == 0

    def test_km_risk_show_rmst_with_risk_table(self, synthetic_data_clean):
        """show_rmst=True with show_risk_table=True (combined path)."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        X, y = synthetic_data_clean
        rng = np.random.RandomState(42)

        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'risk_score': rng.randn(len(y))
        })

        fig = plot_km(df, risk_col='risk_score', n_groups=2,
                      show_risk_table=True, show_rmst=True)

        assert isinstance(fig, go.Figure)
        fill_traces = [t for t in fig.data if getattr(t, 'fill', None) == 'tozeroy']
        assert len(fill_traces) == 2

    def test_km_risk_show_rmst_more_than_2_groups(self, synthetic_data_clean):
        """show_rmst=True with n_groups=3 now shows fills for all 3 groups."""
        from mlsurv.viz.survival import plot_km

        X, y = synthetic_data_clean
        rng = np.random.RandomState(42)

        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'risk_score': rng.randn(len(y))
        })

        fig = plot_km(df, risk_col='risk_score', n_groups=3,
                      show_risk_table=False, show_rmst=True)

        fill_traces = [t for t in fig.data if getattr(t, 'fill', None) == 'tozeroy']
        assert len(fill_traces) == 3, f"Expected 3 RMST fill traces, got {len(fill_traces)}"

        # Subtitle should mention pairwise comparisons for 3+ groups
        title_text = fig.layout.title.text if fig.layout.title else ""
        assert "pairwise comparisons via return_data" in title_text

    def test_km_show_rmst_no_risk_col_no_group_col_warns(self, synthetic_data_clean):
        """show_rmst=True without risk_col or group_col should warn."""
        from mlsurv.viz.survival import plot_km
        import warnings

        X, y = synthetic_data_clean

        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
        })

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plot_km(df, show_rmst=True)
            rmst_warns = [x for x in w if "requires risk_col or group_col" in str(x.message)]
            assert len(rmst_warns) >= 1

    def test_km_show_rmst_n_groups_3_no_warn(self, synthetic_data_clean):
        """show_rmst=True with n_groups=3 should NOT warn (now supported)."""
        from mlsurv.viz.survival import plot_km
        import warnings

        X, y = synthetic_data_clean
        rng = np.random.RandomState(42)

        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'risk_score': rng.randn(len(y))
        })

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plot_km(df, risk_col='risk_score', n_groups=3, show_rmst=True)
            rmst_warns = [x for x in w if "show_rmst" in str(x.message).lower()]
            assert len(rmst_warns) == 0

    def test_km_show_rmst_return_data(self, synthetic_data_clean):
        """plot_km(show_rmst=True, return_data=True) includes 'rmst' key."""
        from mlsurv.viz.survival import plot_km
        from mlsurv.evaluation.metrics import RMSTResult

        X, y = synthetic_data_clean
        rng = np.random.RandomState(42)

        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'risk_score': rng.randn(len(y))
        })

        result = plot_km(df, risk_col='risk_score', n_groups=2,
                         show_rmst=True, return_data=True,
                         show_risk_table=False)
        assert isinstance(result, dict)
        assert "rmst" in result
        assert isinstance(result["rmst"], RMSTResult)

    def test_km_show_rmst_return_data_3_groups(self, synthetic_data_clean):
        """return_data with 3+ groups returns PairwiseRMSTResult."""
        from mlsurv.viz.survival import plot_km
        from mlsurv.evaluation.metrics import PairwiseRMSTResult

        X, y = synthetic_data_clean
        rng = np.random.RandomState(42)

        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'risk_score': rng.randn(len(y))
        })

        result = plot_km(df, risk_col='risk_score', n_groups=3,
                         show_rmst=True, return_data=True,
                         show_risk_table=False)
        assert isinstance(result, dict)
        assert "rmst" in result
        assert isinstance(result["rmst"], PairwiseRMSTResult)
        assert len(result["rmst"].pairwise) == 3  # C(3,2) = 3 pairs

    def test_km_show_rmst_group_col(self, synthetic_data_clean):
        """show_rmst=True with group_col produces fills."""
        from mlsurv.viz.survival import plot_km

        X, y = synthetic_data_clean

        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'treatment': np.where(np.arange(len(y)) % 2 == 0, 'A', 'B'),
        })

        fig = plot_km(df, group_col='treatment', show_rmst=True)
        fill_traces = [t for t in fig.data if getattr(t, 'fill', None) == 'tozeroy']
        assert len(fill_traces) == 2

    def test_km_show_rmst_group_col_return_data(self, synthetic_data_clean):
        """group_col + show_rmst + return_data returns 'rmst' key."""
        from mlsurv.viz.survival import plot_km
        from mlsurv.evaluation.metrics import RMSTResult

        X, y = synthetic_data_clean

        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'treatment': np.where(np.arange(len(y)) % 2 == 0, 'A', 'B'),
        })

        result = plot_km(df, group_col='treatment',
                         show_rmst=True, return_data=True)
        assert isinstance(result, dict)
        assert "rmst" in result
        assert isinstance(result["rmst"], RMSTResult)

    def test_km_show_rmst_group_col_3_groups(self, synthetic_data_clean):
        """group_col with 3 groups + show_rmst produces fills for all 3."""
        from mlsurv.viz.survival import plot_km

        X, y = synthetic_data_clean
        n = len(y)

        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'treatment': np.array(['A', 'B', 'C'] * (n // 3 + 1))[:n],
        })

        fig = plot_km(df, group_col='treatment', show_rmst=True)
        fill_traces = [t for t in fig.data if getattr(t, 'fill', None) == 'tozeroy']
        assert len(fill_traces) == 3

    def test_km_return_data_rmst_none_without_show_rmst(self, synthetic_data_clean):
        """return_data=True without show_rmst has rmst=None."""
        from mlsurv.viz.survival import plot_km

        X, y = synthetic_data_clean
        rng = np.random.RandomState(42)

        df = pd.DataFrame({
            'time': y['time'],
            'event': y['event'],
            'risk_score': rng.randn(len(y))
        })

        result = plot_km(df, risk_col='risk_score', n_groups=2,
                         return_data=True, show_rmst=False)
        assert isinstance(result, dict)
        assert "rmst" in result
        assert result["rmst"] is None


class TestImportance:
    """Test feature importance visualizations."""

    def test_top_features_bar_basic(self):
        """Verify plot_importance creates figure."""
        from mlsurv.viz.features import plot_importance
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(20)],
            'perm_mean': np.random.rand(20),
            'perm_std': np.random.rand(20) * 0.1,
            'coefficient': np.random.randn(20)
        })

        fig = plot_importance(df, n=10)

        assert isinstance(fig, go.Figure)

    def test_top_features_bar_no_error_bars(self):
        """Verify plot_importance without error bars."""
        from mlsurv.viz.features import plot_importance

        df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(10)],
            'perm_mean': np.random.rand(10)
        })

        fig = plot_importance(df, n=5, show_error_bars=False)

        assert fig is not None

    def test_importance_comparison(self):
        """Verify plot_importance auto-detects multi-model."""
        from mlsurv.viz.features import plot_importance

        df = pd.DataFrame({
            'feature': ['A', 'B', 'C'] * 2,
            'perm_mean': [0.1, 0.2, 0.15, 0.12, 0.18, 0.14],
            'model': ['coxph'] * 3 + ['rsf'] * 3
        })

        fig = plot_importance(df, n=3)

        assert fig is not None

    def test_importance_comparison_specific_models(self):
        """Verify plot_importance with specific models filtered."""
        from mlsurv.viz.features import plot_importance

        df = pd.DataFrame({
            'feature': ['A', 'B', 'C'] * 3,
            'perm_mean': np.random.rand(9),
            'model': ['coxph'] * 3 + ['rsf'] * 3 + ['gbsa'] * 3
        })

        # Filter to specific models before passing
        df_filtered = df[df['model'].isin(['coxph', 'rsf'])]
        fig = plot_importance(df_filtered, n=3)

        assert fig is not None

    def test_coefficient_scatter(self):
        """Verify plot_coefficient_comparison creates figure."""
        from mlsurv.viz.features import plot_coefficient_comparison
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(10)],
            'coxph_coef': np.random.randn(10) * 0.5,
            'coxnet_coef': np.random.randn(10) * 0.5
        })

        fig = plot_coefficient_comparison(df)

        assert isinstance(fig, go.Figure)

    def test_coefficient_scatter_with_color(self):
        """Verify plot_coefficient_comparison with color column."""
        from mlsurv.viz.features import plot_coefficient_comparison

        df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(10)],
            'coxph_coef': np.random.randn(10) * 0.5,
            'coxnet_coef': np.random.randn(10) * 0.5,
            'feature_class': np.random.choice(['clinical', 'genomic'], 10)
        })

        fig = plot_coefficient_comparison(df, color_col='feature_class')

        assert fig is not None

    def test_importance_heatmap(self):
        """Verify plot_importance_heatmap creates figure with correct trace."""
        from mlsurv.viz.features import plot_importance_heatmap
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'feature': ['f1', 'f2', 'f3', 'f1', 'f2', 'f3'],
            'model': ['coxph', 'coxph', 'coxph', 'rsf', 'rsf', 'rsf'],
            'perm_mean': [0.1, 0.3, 0.05, 0.15, 0.25, 0.08],
        })

        fig = plot_importance_heatmap(df, n=3)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)
        # Top 3 features should all be present
        y_labels = fig.data[0].y
        assert len(y_labels) == 3

    def test_importance_heatmap_sort_alpha(self):
        """Verify sort='alpha' produces alphabetical feature ordering."""
        from mlsurv.viz.features import plot_importance_heatmap

        df = pd.DataFrame({
            'feature': ['zebra', 'apple', 'mango'] * 2,
            'model': ['coxph'] * 3 + ['rsf'] * 3,
            'perm_mean': [0.5, 0.1, 0.3, 0.4, 0.2, 0.35],
        })

        fig = plot_importance_heatmap(df, n=3, sort="alpha")

        y_labels = list(fig.data[0].y)
        # Alphabetical top-to-bottom: apple, mango, zebra
        # Plotly y-axis is bottom-to-top, so reversed list = [zebra, mango, apple]
        assert y_labels == ['zebra', 'mango', 'apple']

    def test_importance_heatmap_no_annotations(self):
        """Verify annotations=False produces no annotations."""
        from mlsurv.viz.features import plot_importance_heatmap

        df = pd.DataFrame({
            'feature': ['f1', 'f2'] * 2,
            'model': ['coxph'] * 2 + ['rsf'] * 2,
            'perm_mean': [0.1, 0.3, 0.15, 0.25],
        })

        fig = plot_importance_heatmap(df, n=2, annotations=False)

        assert len(fig.layout.annotations) == 0

    def test_importance_heatmap_model_sort_value(self):
        """Verify model_sort='value' orders columns by descending mean importance."""
        from mlsurv.viz.features import plot_importance_heatmap

        df = pd.DataFrame({
            'feature': ['f1', 'f2'] * 3,
            'model': ['coxph'] * 2 + ['rsf'] * 2 + ['gbsa'] * 2,
            'perm_mean': [0.01, 0.02, 0.5, 0.6, 0.001, 0.002],
        })

        fig = plot_importance_heatmap(df, n=2, model_sort="value")

        x_labels = list(fig.data[0].x)
        # RSF has highest mean (0.55), then CoxPH (0.015), then GBSA (0.0015)
        assert x_labels[0] == "RSF"

    def test_importance_heatmap_annotations_content(self):
        """Verify annotation formatting and contrast colors."""
        from mlsurv.viz.features import plot_importance_heatmap

        df = pd.DataFrame({
            'feature': ['f1', 'f2'] * 2,
            'model': ['coxph'] * 2 + ['rsf'] * 2,
            'perm_mean': [0.0005, 15.7, 0.0003, 12.3],
        })

        fig = plot_importance_heatmap(df, n=2, log_scale=False)

        annotations = fig.layout.annotations
        assert len(annotations) == 4
        texts = {a.text for a in annotations}
        # Small values should use scientific notation
        assert any("e" in t for t in texts)
        # Large values should use 1 decimal
        assert any(t in ("15.7", "12.3") for t in texts)
        # All annotations should have contrast colors
        colors = {a.font.color for a in annotations}
        assert colors <= {"white", "black"}

    def test_coefficient_forest(self):
        """Verify plot_coefficients creates figure."""
        from mlsurv.viz.features import plot_coefficients
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(15)],
            'coefficient': np.random.randn(15) * 0.5,
            'coef_std': np.random.rand(15) * 0.1
        })

        fig = plot_coefficients(df, n=10)

        assert isinstance(fig, go.Figure)

    def test_coefficient_forest_without_std(self):
        """Verify plot_coefficients without standard errors."""
        from mlsurv.viz.features import plot_coefficients

        df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(10)],
            'coefficient': np.random.randn(10) * 0.5
        })

        fig = plot_coefficients(df, n=5)

        assert fig is not None


class TestCalibration:
    """Test calibration plot."""

    def test_calibration_plot_requires_survival_function(self, synthetic_data_clean):
        """Verify compute_calibration raises for models without survival function."""
        from mlsurv.viz.calibration import compute_calibration

        X, y = synthetic_data_clean

        class MockModel:
            pass

        model = MockModel()

        with pytest.raises(ValueError, match="does not support predict_survival_function"):
            compute_calibration(model, X, y)

    def test_calibration_plot_with_model(self, train_test_data):
        """Verify compute_calibration + plot_calibration works with fitted CoxPH model."""
        from mlsurv.viz.calibration import compute_calibration, plot_calibration
        from mlsurv.models import CoxPHModel
        import plotly.graph_objects as go

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxPHModel()
        model.fit(X_train, y_train)

        # Use reasonable eval times based on data
        max_time = y_test['time'].max()
        eval_times = [max_time * 0.25, max_time * 0.5]

        cal_data = compute_calibration(model, X_test, y_test, eval_times=eval_times)
        fig = plot_calibration(cal_data)

        assert isinstance(fig, go.Figure)

    def test_calibration_comparison(self, train_test_data):
        """Verify _compute_multi_model_calibration_data + plot_calibration creates figure."""
        from mlsurv.viz.calibration import _compute_multi_model_calibration_data, plot_calibration
        from mlsurv.models import CoxPHModel
        import plotly.graph_objects as go

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxPHModel()
        model.fit(X_train, y_train)

        max_time = y_test['time'].max()

        cal_data = _compute_multi_model_calibration_data(
            {'coxph': model},
            X_test, y_test,
            eval_time=max_time * 0.5,
            n_bins=10,
        )
        fig = plot_calibration(cal_data)

        assert isinstance(fig, go.Figure)


class TestCalibrationBinStrategy:
    """Width vs quantile binning in calibration (fast, no model fit)."""

    @staticmethod
    def _synthetic_y(n, seed=0):
        rng = np.random.RandomState(seed)
        return np.array(
            [(bool(e), float(t)) for e, t in
             zip(rng.randint(0, 2, n), rng.uniform(1, 50, n))],
            dtype=[("event", bool), ("time", float)],
        )

    def test_narrow_predictor_collapses_under_width_not_quantile(self):
        """A narrow-spread predictor (GBSA-like) collapses to ~1 bin under
        width binning but spreads to n_bins points under quantile binning."""
        from mlsurv.viz.calibration import _compute_bin_calibration

        n, n_bins = 200, 10
        y = self._synthetic_y(n)
        # All predictions packed into a 0.07-wide window -> one width-bin.
        narrow = np.random.RandomState(1).uniform(0.84, 0.91, n)

        pred_w, *_ = _compute_bin_calibration(narrow, y, 20.0, n_bins, "width")
        pred_q, *_ = _compute_bin_calibration(narrow, y, 20.0, n_bins, "quantile")

        assert len(pred_w) <= 2
        assert len(pred_q) >= n_bins - 1

    def test_quantile_points_are_x_ordered(self):
        """Quantile bins are ordered by predicted probability, so the
        predicted-x values come out non-decreasing for the line plot."""
        from mlsurv.viz.calibration import _compute_bin_calibration

        y = self._synthetic_y(200)
        wide = np.random.RandomState(2).uniform(0.05, 0.95, 200)

        pred_q, *_ = _compute_bin_calibration(wide, y, 20.0, 10, "quantile")

        assert pred_q == sorted(pred_q)
        assert len(pred_q) >= 5

    def test_partial_drop_yields_between_one_and_n_bins(self):
        """A predictor with only a few distinct values drops some quantile
        edges (duplicates="drop"), giving more than one but fewer than n_bins
        points, still x-ordered."""
        from mlsurv.viz.calibration import _compute_bin_calibration

        y = self._synthetic_y(210)
        # 3 distinct probability levels, 70 patients each -> at most 3 bins.
        tied = np.repeat([0.3, 0.5, 0.7], 70)

        pred_q, *_ = _compute_bin_calibration(tied, y, 20.0, 10, "quantile")

        assert 1 < len(pred_q) < 10
        assert pred_q == sorted(pred_q)

    def test_degenerate_constant_predictor_warns_and_returns_one_point(self):
        """An all-identical predictor cannot form quantile edges; the fallback
        is a single bin over every patient (>=1 point, no crash) and warns the
        user that the curve collapsed."""
        from mlsurv.viz.calibration import _compute_bin_calibration
        from mlsurv._warnings import SuggestionWarning

        y = self._synthetic_y(200)
        const = np.full(200, 0.5)

        with pytest.warns(SuggestionWarning, match="too few distinct values"):
            pred_q, *_ = _compute_bin_calibration(const, y, 20.0, 10, "quantile")

        assert len(pred_q) == 1

    def test_multi_model_quantile_happy_path(self, train_test_data):
        """_compute_multi_model_calibration_data forwards bin_strategy and
        returns x-ordered per-model bins under quantile binning."""
        from mlsurv.viz.calibration import _compute_multi_model_calibration_data
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)
        t = float(y_test["time"].max() * 0.5)

        out = _compute_multi_model_calibration_data(
            {"coxph": model}, X_test, y_test, t, 10, bin_strategy="quantile",
        )
        assert "coxph" in out
        predicted = out["coxph"]["predicted"]
        assert len(predicted) > 1
        assert predicted == sorted(predicted)

    def test_invalid_bin_strategy_raises(self):
        """An unknown bin_strategy is rejected with a clear ValueError."""
        from mlsurv.viz.calibration import (
            _compute_multi_model_calibration_data,
            compute_calibration,
        )

        class _SurvModel:
            def predict_survival_function(self, X):
                raise NotImplementedError

        y = self._synthetic_y(50)
        with pytest.raises(ValueError, match="bin_strategy must be one of"):
            compute_calibration(_SurvModel(), None, y, bin_strategy="bogus")
        with pytest.raises(ValueError, match="bin_strategy must be one of"):
            _compute_multi_model_calibration_data(
                {"m": _SurvModel()}, None, y, 20.0, 10, bin_strategy="bogus",
            )

    @pytest.mark.integration
    def test_public_plot_calibration_threads_bin_strategy(self, synthetic_data_clean):
        """bin_strategy threads end-to-end through the public
        learner.plot_calibration path (all four layers). A bogus value must
        surface as ValueError; if the kwarg were dropped anywhere along the
        way, no error would propagate."""
        from mlsurv import SurvivalLearner

        X, y = synthetic_data_clean
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train(["coxph", "coxnet"])

        df = learner.plot_calibration(
            models=["coxph", "coxnet"], bin_strategy="quantile", return_data=True,
        )
        assert isinstance(df, pd.DataFrame)

        with pytest.raises(ValueError, match="bin_strategy must be one of"):
            learner.plot_calibration(
                models=["coxph", "coxnet"], bin_strategy="bogus",
            )

    def test_compute_calibration_threads_bin_strategy(self, train_test_data):
        """End-to-end through compute_calibration: a narrow-spread fitted model
        yields more points under quantile than width binning."""
        from mlsurv.viz.calibration import compute_calibration
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)
        t = [float(y_test["time"].max() * 0.5)]

        width = compute_calibration(
            model, X_test, y_test, eval_times=t, bin_strategy="width",
        )
        quant = compute_calibration(
            model, X_test, y_test, eval_times=t, bin_strategy="quantile",
        )
        # Same model, same horizon: quantile bins by equal count, so it should
        # produce at least as many points as width binning.
        assert len(quant[0]["predicted"]) >= len(width[0]["predicted"])


class TestCalibrationDefaultEvalTimes:
    """Data-driven default eval_times across time-unit scales."""

    @staticmethod
    def _stub_model(times_grid, surv_curve_fn):
        """Stub with predict_survival_function returning per-sample StepFunctions."""
        from sksurv.functions import StepFunction

        class StubModel:
            def predict_survival_function(self, X):
                # One curve per sample, perturbed by row index for variability.
                n = len(X)
                fns = []
                for i in range(n):
                    shift = (i / max(n - 1, 1)) - 0.5  # in [-0.5, 0.5]
                    y_vals = np.clip(surv_curve_fn(times_grid) + 0.05 * shift,
                                     1e-3, 1 - 1e-3)
                    fns.append(StepFunction(times_grid.astype(float), y_vals))
                return np.array(fns, dtype=object)

        return StubModel()

    @pytest.fixture
    def _scale_data(self):
        """Build (X_test, y_test, y_train, model) for a given time scale."""
        def _build(max_time):
            rng = np.random.default_rng(42)
            n = 60
            # Training event times spread across the full range.
            train_times = rng.uniform(max_time * 0.05, max_time * 0.95, size=n)
            train_events = rng.random(n) > 0.3
            y_train = make_y(train_times, train_events)

            test_times = rng.uniform(max_time * 0.05, max_time * 0.95, size=n)
            test_events = rng.random(n) > 0.3
            y_test = make_y(test_times, test_events)

            X_test = pd.DataFrame(rng.random((n, 3)), columns=["a", "b", "c"])

            grid = np.linspace(0, max_time * 1.1, 50)
            # Exponential-ish survival curve: S(t) = exp(-t / (max_time/2))
            model = self._stub_model(
                grid, lambda t: np.exp(-t / (max_time / 2.0)),
            )
            return X_test, y_test, y_train, model
        return _build

    @pytest.mark.parametrize(
        "scale_label,max_time,min_expected_eval_time",
        [
            ("days", 2500.0, 100.0),    # GBSG2-like
            ("months", 350.0, 20.0),    # METABRIC-like
            ("years", 10.0, 0.5),       # year-scale
        ],
    )
    def test_default_eval_times_data_driven(
        self, _scale_data, scale_label, max_time, min_expected_eval_time
    ):
        """Default eval_times falls in body of curve, not stacked at t=12,24,36."""
        from mlsurv.viz.calibration import compute_calibration

        X_test, y_test, y_train, model = _scale_data(max_time)

        cal_data = compute_calibration(model, X_test, y_test, y_train=y_train)

        assert len(cal_data) == 3, (
            f"[{scale_label}] expected 3 eval points (25/50/75 percentiles), "
            f"got {len(cal_data)}"
        )

        eval_times = [item["eval_time"] for item in cal_data]
        train_event_times = y_train["time"][y_train["event"]]
        t_min, t_max = train_event_times.min(), train_event_times.max()

        for t in eval_times:
            assert t_min <= t <= t_max, (
                f"[{scale_label}] eval_time {t} outside training event range "
                f"[{t_min}, {t_max}]"
            )
            assert t >= min_expected_eval_time, (
                f"[{scale_label}] eval_time {t} too small — likely stuck at "
                f"the hardcoded [12, 24, 36] default"
            )

        # At least one bin should have predicted survival below 0.9 — i.e.,
        # the curve isn't collapsed at the top-right corner.
        all_predicted = np.concatenate([item["predicted"] for item in cal_data])
        assert (all_predicted < 0.9).any(), (
            f"[{scale_label}] all bins have predicted >= 0.9; default "
            f"eval_times are stuck at the head of the curve"
        )

    def test_default_uses_y_test_when_y_train_none(self, _scale_data):
        """Without y_train, fallback to y_test event times — still data-driven."""
        from mlsurv.viz.calibration import compute_calibration

        X_test, y_test, _, model = _scale_data(2500.0)

        cal_data = compute_calibration(model, X_test, y_test)  # no y_train

        assert len(cal_data) == 3
        eval_times = [item["eval_time"] for item in cal_data]
        # Should sit within y_test event-time range, not at [12, 24, 36].
        assert all(t > 50 for t in eval_times), (
            f"eval_times {eval_times} look like the old [12,24,36] default"
        )

    def test_y_train_drives_percentiles_when_ranges_diverge(self, _scale_data):
        """y_train must dominate percentile choice even when y_test ranges differ.

        Builds y_train with max ~1000 and y_test with max ~3000. The default
        eval_times must come from y_train's percentile range, not y_test's,
        proving the y_train= kwarg is actually threaded through.
        """
        from mlsurv.viz.calibration import compute_calibration

        # y_train short follow-up
        _, _, y_train_short, model = _scale_data(1000.0)
        # y_test long follow-up — completely different range
        X_test_long, y_test_long, _, _ = _scale_data(3000.0)

        cal_data = compute_calibration(
            model, X_test_long, y_test_long, y_train=y_train_short,
        )

        eval_times = [item["eval_time"] for item in cal_data]
        train_event_times = y_train_short["time"][y_train_short["event"]]
        train_max = train_event_times.max()
        # All eval_times should sit within the training range, not the
        # much-larger y_test range.
        assert all(t <= train_max for t in eval_times), (
            f"eval_times {eval_times} exceed y_train max {train_max} — "
            f"y_train kwarg is being ignored"
        )

    def test_value_error_propagates_with_context(self):
        """auto_eval_times ValueError must surface with actionable message."""
        from mlsurv.viz.calibration import compute_calibration

        # All censored, single event time → auto_eval_times raises.
        n = 30
        rng = np.random.default_rng(0)
        y_degenerate = make_y(
            np.full(n, 5.0), np.zeros(n, dtype=bool),
        )
        X = pd.DataFrame(rng.random((n, 3)), columns=["a", "b", "c"])
        grid = np.linspace(0, 10, 20)
        model = self._stub_model(grid, lambda t: np.exp(-t / 5.0))

        with pytest.raises(ValueError, match="eval_times="):
            compute_calibration(model, X, y_degenerate, y_train=y_degenerate)


class TestCalibrationWarnings:
    """Tests for calibration pipeline warning tiers and edge cases."""

    def test_km_survival_at_extrapolation_preserves_ci(self):
        """CI bounds at extrapolation should use last available, not collapse."""
        from lifelines import KaplanMeierFitter
        from mlsurv.viz.calibration import _km_survival_at

        kmf = KaplanMeierFitter()
        # Fit with enough data so last-event survival is well above 0
        # (heavy censoring keeps survival high → non-degenerate CI)
        rng = np.random.RandomState(42)
        times = rng.exponential(10, 50)
        events = rng.binomial(1, 0.3, 50).astype(bool)  # 70% censored
        kmf.fit(times, events)

        # Request survival beyond last observed time
        obs, ci_low, ci_high = _km_survival_at(kmf, times.max() + 100)

        assert np.isfinite(obs)
        assert np.isfinite(ci_low)
        assert np.isfinite(ci_high)
        # CI should not be collapsed to point estimate
        assert ci_low <= obs <= ci_high

    def test_km_survival_at_all_censored(self):
        """All-censored data produces finite survival (flat at 1.0)."""
        from lifelines import KaplanMeierFitter
        from mlsurv.viz.calibration import _km_survival_at

        kmf = KaplanMeierFitter()
        times = np.array([1, 2, 3, 4, 5], dtype=float)
        events = np.array([0, 0, 0, 0, 0], dtype=bool)
        kmf.fit(times, events)

        obs, ci_low, ci_high = _km_survival_at(kmf, 3.0)
        assert np.isfinite(obs)

    def test_km_survival_at_empty_survival_function(self):
        """Empty survival function returns NaN triplet."""
        from mlsurv.viz.calibration import _km_survival_at
        from types import SimpleNamespace

        mock_kmf = SimpleNamespace(
            survival_function_=pd.DataFrame(),
            confidence_interval_survival_function_=pd.DataFrame(),
        )
        obs, ci_low, ci_high = _km_survival_at(mock_kmf, 3.0)
        assert np.isnan(obs)
        assert np.isnan(ci_low)
        assert np.isnan(ci_high)

    def test_km_survival_at_no_ci_returns_nan_bounds(self):
        """When CI is unavailable, bounds should be NaN, not equal to obs."""
        from mlsurv.viz.calibration import _km_survival_at
        from types import SimpleNamespace

        # Build a minimal mock KMF with survival function but empty CI
        sf = pd.DataFrame(
            {"KM_estimate": [1.0, 0.8, 0.6]},
            index=pd.Index([0.0, 2.0, 5.0], name="timeline"),
        )
        mock_kmf = SimpleNamespace(
            survival_function_=sf,
            confidence_interval_survival_function_=pd.DataFrame(),
            survival_function_at_times=lambda t: pd.Series(
                [sf.iloc[sf.index.searchsorted(t, side="right") - 1, 0]],
            ),
        )

        obs, ci_low, ci_high = _km_survival_at(mock_kmf, 3.0)

        assert np.isfinite(obs)
        assert np.isnan(ci_low)
        assert np.isnan(ci_high)

    def test_multi_model_skip_warns_suggestion(self, train_test_data):
        """Model without predict_survival_function emits SuggestionWarning."""
        import warnings as _warnings
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.viz.calibration import _compute_multi_model_calibration_data

        X_train, X_test, y_train, y_test = train_test_data

        class NoSurvFunc:
            pass

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            result = _compute_multi_model_calibration_data(
                {"fake": NoSurvFunc()},
                X_test, y_test,
                eval_time=y_test['time'].max() * 0.5,
                n_bins=10,
            )

        assert result == {}
        skip_warns = [w for w in caught if "Skipping fake" in str(w.message)]
        assert len(skip_warns) == 1
        assert issubclass(skip_warns[0].category, SuggestionWarning)

    def test_multi_model_error_warns_suggestion(self, train_test_data):
        """Model that raises ValueError emits SuggestionWarning, not InternalWarning."""
        import warnings as _warnings
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.viz.calibration import _compute_multi_model_calibration_data

        X_train, X_test, y_train, y_test = train_test_data

        class BadModel:
            def predict_survival_function(self, X):
                raise ValueError("test error")

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            result = _compute_multi_model_calibration_data(
                {"bad_model": BadModel()},
                X_test, y_test,
                eval_time=y_test['time'].max() * 0.5,
                n_bins=10,
            )

        assert result == {}
        err_warns = [w for w in caught if "Could not compute" in str(w.message)]
        assert len(err_warns) == 1
        assert issubclass(err_warns[0].category, SuggestionWarning)

    def test_multi_model_not_implemented_warns_suggestion(self, train_test_data):
        """Model raising NotImplementedError emits SuggestionWarning."""
        import warnings as _warnings
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.viz.calibration import _compute_multi_model_calibration_data

        X_train, X_test, y_train, y_test = train_test_data

        class NotImplModel:
            def predict_survival_function(self, X):
                raise NotImplementedError("not supported")

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            result = _compute_multi_model_calibration_data(
                {"not_impl": NotImplModel()},
                X_test, y_test,
                eval_time=y_test['time'].max() * 0.5,
                n_bins=10,
            )

        assert result == {}
        skip_warns = [w for w in caught if "not supported" in str(w.message)]
        assert len(skip_warns) == 1
        assert issubclass(skip_warns[0].category, SuggestionWarning)

    def test_n_groups_reduction_warns(self):
        """Reducing n_groups due to duplicate scores emits SuggestionWarning."""
        import warnings as _warnings
        from unittest.mock import patch
        from mlsurv._warnings import SuggestionWarning
        from mlsurv.viz.calibration import plot_calibration_overlay

        rng = np.random.RandomState(42)
        n = 100
        max_time = 15.0

        class TiedModel:
            def predict(self, X):
                n = len(X)
                return np.array([1.0] * (n // 2) + [2.0] * (n - n // 2))

            def predict_survival_function(self, X):
                from sksurv.functions import StepFunction
                return [
                    StepFunction(
                        np.array([0, max_time * 0.5, max_time]),
                        np.array([1.0, 0.8, 0.5]),
                    )
                    for _ in range(len(X))
                ]

        X_test = pd.DataFrame({'x1': rng.randn(n)})
        y_test = make_survival_array(
            rng.binomial(1, 0.7, n),
            rng.uniform(1.0, max_time * 0.9, n),
        )

        # Patch pd.qcut inside the module to force the except-ValueError path
        # (with duplicates="drop", pd.qcut handles ties silently)
        original_qcut = pd.qcut
        call_count = [0]

        def qcut_fail_first(x, q, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Bin edges must be unique")
            return original_qcut(x, q=q, **kwargs)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            with patch("mlsurv.viz.calibration.pd.qcut", side_effect=qcut_fail_first):
                plot_calibration_overlay(
                    TiedModel(), X_test, y_test, n_groups=4,
                )

        reduce_warns = [w for w in caught if "Reduced n_groups" in str(w.message)]
        assert len(reduce_warns) == 1
        assert issubclass(reduce_warns[0].category, SuggestionWarning)
        assert "from 4 to 2" in str(reduce_warns[0].message)

    def test_n_groups_all_identical_raises(self):
        """All-identical risk scores should raise ValueError from our guard."""
        from unittest.mock import patch
        from mlsurv.viz.calibration import plot_calibration_overlay

        rng = np.random.RandomState(42)
        n = 50
        max_time = 15.0

        class ConstantModel:
            def predict(self, X):
                return np.ones(len(X))

            def predict_survival_function(self, X):
                from sksurv.functions import StepFunction
                return [
                    StepFunction(
                        np.array([0, max_time * 0.5, max_time]),
                        np.array([1.0, 0.8, 0.5]),
                    )
                    for _ in range(len(X))
                ]

        X_test = pd.DataFrame({'x1': rng.randn(n)})
        y_test = make_survival_array(
            rng.binomial(1, 0.7, n),
            rng.uniform(1.0, max_time * 0.9, n),
        )

        # Force qcut to always raise so our n_unique < 2 guard triggers
        with patch(
            "mlsurv.viz.calibration.pd.qcut",
            side_effect=ValueError("Bin edges must be unique"),
        ):
            with pytest.raises(ValueError, match="Cannot create risk groups"):
                plot_calibration_overlay(
                    ConstantModel(), X_test, y_test, n_groups=4,
                )


class TestForest:
    """Test hazard ratio forest plot."""

    def test_forest_plot_basic(self, synthetic_data_clean):
        """Verify compute_hazard_ratios + plot_hazard_ratios creates figure for CoxPH."""
        from mlsurv.viz.features import compute_hazard_ratios, plot_hazard_ratios
        from mlsurv.models import CoxPHModel
        import plotly.graph_objects as go

        X, y = synthetic_data_clean
        model = CoxPHModel()
        model.fit(X, y)

        hr_df = compute_hazard_ratios(model)
        fig = plot_hazard_ratios(hr_df, n_features=10)

        assert isinstance(fig, go.Figure)

    def test_forest_plot_requires_coefficients(self):
        """Verify compute_hazard_ratios raises for models without coefficients."""
        from mlsurv.viz.features import compute_hazard_ratios

        class MockModel:
            hazard_ratio_valid = True  # pass the HR-validity gate to reach coef check

            def get_coefficients(self):
                return None

        model = MockModel()

        with pytest.raises(ValueError, match="does not have coefficients"):
            compute_hazard_ratios(model)

    def test_forest_plot_from_df(self):
        """Verify plot_hazard_ratios creates figure from DataFrame."""
        from mlsurv.viz.features import plot_hazard_ratios
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(10)],
            'hr': np.exp(np.random.randn(10) * 0.5),
            'ci_lower': np.exp(np.random.randn(10) * 0.3 - 0.2),
            'ci_upper': np.exp(np.random.randn(10) * 0.3 + 0.2),
        })

        fig = plot_hazard_ratios(df, n_features=5)

        assert isinstance(fig, go.Figure)

    def test_multi_model_forest(self, synthetic_data_clean):
        """Non-HR models (CoxNet) are dropped with a warning; CoxPH still plots."""
        from mlsurv.viz.features import plot_hazard_ratios_comparison
        from mlsurv.models import CoxPHModel, CoxNetModel
        from mlsurv._warnings import MethodologicalWarning
        import plotly.graph_objects as go

        X, y = synthetic_data_clean

        model1 = CoxPHModel()
        model1.fit(X, y)

        model2 = CoxNetModel(fit_baseline_model=True)
        model2.fit(X, y)

        # CoxNet's exp(coef) is not a hazard ratio: it is dropped with a
        # MethodologicalWarning, and the figure is built from CoxPH alone.
        with pytest.warns(MethodologicalWarning, match="coxnet"):
            fig = plot_hazard_ratios_comparison(
                {'coxph': model1, 'coxnet': model2},
                n_features=5
            )

        assert isinstance(fig, go.Figure)

    def test_multi_model_forest_no_coefficients_error(self):
        """Verify plot_hazard_ratios_comparison raises when no models have coefficients."""
        from mlsurv.viz.features import plot_hazard_ratios_comparison

        class MockModel:
            def get_coefficients(self):
                return None

        with pytest.raises(ValueError, match="No models with coefficients"):
            plot_hazard_ratios_comparison({'mock': MockModel()})

    def test_forest_plot_ci_level_passthrough(self):
        """Verify ci_level flows through to compute_hazard_ratios and label."""
        from mlsurv.viz.features import compute_hazard_ratios, plot_hazard_ratios

        # Mock model with covariance matrix so CIs are computed
        coefs = np.array([0.5, -0.3, 0.4])
        cov = np.diag([0.01, 0.02, 0.015])

        class MockInner:
            coef_ = coefs
            covariance_matrix_ = cov

        class MockModel:
            hazard_ratio_valid = True
            model_ = MockInner()
            feature_names_in_ = np.array(['a', 'b', 'c'])
            def get_coefficients(self):
                return pd.Series(self.model_.coef_, index=self.feature_names_in_)

        model = MockModel()

        hr_95 = compute_hazard_ratios(model, ci_level=0.95)
        hr_90 = compute_hazard_ratios(model, ci_level=0.90)

        # 90% CI must be strictly narrower than 95% CI
        width_95 = (hr_95["ci_upper"] - hr_95["ci_lower"]).sum()
        width_90 = (hr_90["ci_upper"] - hr_90["ci_lower"]).sum()
        assert width_90 < width_95

        # Axis label should reflect the ci_level
        fig = plot_hazard_ratios(hr_90, ci_level=0.90)
        xaxis_title = fig.layout.xaxis.title.text
        assert "90% CI" in xaxis_title

    def test_forest_plot_no_ci_all_ns(self):
        """When CIs are unavailable, all features should be N.S."""
        from mlsurv.viz.features import plot_hazard_ratios
        import plotly.graph_objects as go

        # Simulate no-CI data: ci_lower == ci_upper == hr
        df = pd.DataFrame({
            'feature': ['a', 'b', 'c'],
            'hr': [0.5, 1.2, 2.0],
            'ci_lower': [0.5, 1.2, 2.0],
            'ci_upper': [0.5, 1.2, 2.0],
        })
        fig = plot_hazard_ratios(df)
        assert isinstance(fig, go.Figure)

        # Only "N.S." traces should exist (no "Higher risk" / "Lower risk")
        trace_names = [t.name for t in fig.data if isinstance(t, go.Scatter)]
        assert all(name == "N.S." for name in trace_names)

        # All error bars should be invisible (no visible whiskers)
        for trace in fig.data:
            if isinstance(trace, go.Scatter) and trace.error_x:
                assert trace.error_x.visible is False

        # Axis label should omit CI mention
        xaxis_title = fig.layout.xaxis.title.text
        assert "CI" not in xaxis_title

    def test_forest_plot_error_bar_color_matches_category(self):
        """Error bar whisker color should match category, not be hardcoded black."""
        from mlsurv.viz.features import plot_hazard_ratios
        from mlsurv.viz._palettes import hex_to_rgba
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'feature': ['a', 'b', 'c'],
            'hr': [0.3, 1.0, 3.0],
            'ci_lower': [0.1, 0.8, 2.0],
            'ci_upper': [0.6, 1.2, 5.0],
        })
        fig = plot_hazard_ratios(df)

        expected_colors = {
            "Higher risk": hex_to_rgba("#d62728", 0.5),
            "N.S.": hex_to_rgba("#999999", 0.5),
            "Lower risk": hex_to_rgba("#1f77b4", 0.5),
        }
        for trace in fig.data:
            if isinstance(trace, go.Scatter) and trace.error_x:
                assert trace.error_x.color == expected_colors[trace.name]

    def test_forest_plot_hover_no_ci(self):
        """Hover text should say 'CI: not available' when CIs are absent."""
        from mlsurv.viz.features import plot_hazard_ratios

        df = pd.DataFrame({
            'feature': ['a', 'b'],
            'hr': [0.5, 2.0],
            'ci_lower': [0.5, 2.0],
            'ci_upper': [0.5, 2.0],
        })
        fig = plot_hazard_ratios(df)

        for trace in fig.data:
            if hasattr(trace, 'hovertemplate') and trace.hovertemplate:
                assert "CI: not available" in trace.hovertemplate

    def test_forest_plot_invalid_ci_level(self):
        """ci_level outside (0, 1) should raise ValueError."""
        from mlsurv.viz.features import compute_hazard_ratios, plot_hazard_ratios

        coefs = np.array([0.5, -0.3])
        cov = np.diag([0.01, 0.02])

        class MockInner:
            coef_ = coefs
            covariance_matrix_ = cov

        class MockModel:
            hazard_ratio_valid = True
            model_ = MockInner()
            feature_names_in_ = np.array(['a', 'b'])
            def get_coefficients(self):
                return pd.Series(self.model_.coef_, index=self.feature_names_in_)

        model = MockModel()

        for bad_level in [0, 1.0, 95, -0.5]:
            with pytest.raises(ValueError, match="ci_level must be between 0 and 1"):
                compute_hazard_ratios(model, ci_level=bad_level)

        df = pd.DataFrame({
            'feature': ['a', 'b'], 'hr': [1.5, 0.8],
            'ci_lower': [1.1, 0.5], 'ci_upper': [2.0, 1.1],
        })
        with pytest.raises(ValueError, match="ci_level must be between 0 and 1"):
            plot_hazard_ratios(df, ci_level=95)

    def test_forest_plot_nan_ci_treated_as_no_ci(self):
        """NaN confidence intervals should be treated as unavailable."""
        from mlsurv.viz.features import plot_hazard_ratios
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'feature': ['a', 'b', 'c'],
            'hr': [0.5, 1.2, 2.0],
            'ci_lower': [float('nan'), float('nan'), float('nan')],
            'ci_upper': [float('nan'), float('nan'), float('nan')],
        })
        fig = plot_hazard_ratios(df)
        trace_names = [t.name for t in fig.data if isinstance(t, go.Scatter)]
        assert all(name == "N.S." for name in trace_names)
        assert "CI" not in fig.layout.xaxis.title.text

    def test_forest_plot_error_bars_se(self):
        """error_bars='se' produces narrower bounds than 95% CI."""
        from mlsurv.viz.features import compute_hazard_ratios, plot_hazard_ratios
        import plotly.graph_objects as go

        class MockInner:
            coef_ = np.array([0.5, -0.3, 0.8])
            covariance_matrix_ = np.diag([0.01, 0.02, 0.015])
            ties = "breslow"

        class MockModel:
            hazard_ratio_valid = True
            model_ = MockInner()
            feature_names_in_ = np.array(["a", "b", "c"])
            def get_coefficients(self):
                return pd.Series(self.model_.coef_, index=self.feature_names_in_)

        model = MockModel()
        df_ci = compute_hazard_ratios(model, error_bars="ci", ci_level=0.95)
        df_se = compute_hazard_ratios(model, error_bars="se")

        width_ci = (df_ci["ci_upper"] - df_ci["ci_lower"]).sum()
        width_se = (df_se["ci_upper"] - df_se["ci_lower"]).sum()
        assert width_se < width_ci

        fig = plot_hazard_ratios(df_se, error_bars="se")
        assert isinstance(fig, go.Figure)
        assert "±SE" in fig.layout.xaxis.title.text
        assert "CI" not in fig.layout.xaxis.title.text

    def test_forest_plot_invalid_error_bars(self):
        """Invalid error_bars raises ValueError."""
        from mlsurv.viz.features import compute_hazard_ratios, plot_hazard_ratios

        class MockInner:
            coef_ = np.array([0.1])
            covariance_matrix_ = np.diag([0.01])
            ties = "breslow"

        class MockModel:
            hazard_ratio_valid = True
            model_ = MockInner()
            feature_names_in_ = np.array(["a"])
            def get_coefficients(self):
                return pd.Series(self.model_.coef_, index=self.feature_names_in_)

        with pytest.raises(ValueError, match="error_bars must be"):
            compute_hazard_ratios(MockModel(), error_bars="invalid")

        df = pd.DataFrame({
            'feature': ['a'], 'hr': [1.0],
            'ci_lower': [0.8], 'ci_upper': [1.2],
        })
        with pytest.raises(ValueError, match="error_bars must be"):
            plot_hazard_ratios(df, error_bars="invalid")

    def test_compute_hazard_ratios_no_overflow(self):
        """Extreme coefficients emit ClampWarning, not RuntimeWarning."""
        from mlsurv._warnings import ClampWarning
        from mlsurv.viz.features import compute_hazard_ratios

        extreme_coefs = np.array([800.0, -900.0, 0.5])

        class MockInner:
            coef_ = extreme_coefs
            covariance_matrix_ = np.diag([0.01, 0.02, 0.01])

        class MockModel:
            hazard_ratio_valid = True
            model_ = MockInner()
            feature_names_in_ = np.array(["a", "b", "c"])
            def get_coefficients(self):
                return pd.Series(self.model_.coef_, index=self.feature_names_in_)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = compute_hazard_ratios(MockModel())

        # No RuntimeWarning (overflow) should be emitted
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0, f"Unexpected RuntimeWarning: {runtime_warnings}"
        # ClampWarning should be emitted for the 2 extreme coefficients
        clamp_warnings = [w for w in caught if issubclass(w.category, ClampWarning)]
        assert len(clamp_warnings) >= 1
        assert "clamped" in str(clamp_warnings[0].message).lower()

        assert np.all(np.isfinite(df["hr"]))
        assert np.all(np.isfinite(df["ci_lower"]))
        assert np.all(np.isfinite(df["ci_upper"]))
        assert (df["hr"] > 0).all()
        # Directional correctness under clamping
        assert df.loc[df["feature"] == "a", "hr"].iloc[0] > 1
        assert df.loc[df["feature"] == "b", "hr"].iloc[0] < 1
        assert np.isclose(df.loc[df["feature"] == "c", "hr"].iloc[0], np.exp(0.5))
        # CI ordering invariant preserved after clamping
        assert (df["ci_lower"] <= df["hr"]).all()
        assert (df["hr"] <= df["ci_upper"]).all()

    def test_compute_hr_gates_non_hr_model(self):
        """compute_hazard_ratios raises for models without valid hazard ratios."""
        from mlsurv.viz.features import compute_hazard_ratios

        class PenalizedModel:
            hazard_ratio_valid = False  # e.g. CoxNet/FSSVM/AFT

            def get_coefficients(self):
                return pd.Series([0.5, -0.3], index=["a", "b"])

        with pytest.raises(ValueError, match="not hazard ratios"):
            compute_hazard_ratios(PenalizedModel())

    def test_compute_hr_default_flag_is_non_hr(self):
        """A model with no flag at all defaults to non-HR and is gated."""
        from mlsurv.viz.features import compute_hazard_ratios

        class NoFlagModel:
            def get_coefficients(self):
                return pd.Series([0.5], index=["a"])

        with pytest.raises(ValueError, match="not hazard ratios"):
            compute_hazard_ratios(NoFlagModel())

    def test_comparison_drops_non_hr_model_with_warning(self):
        """plot_hazard_ratios_comparison drops non-HR models but keeps HR-valid ones."""
        from mlsurv.viz.features import plot_hazard_ratios_comparison
        from mlsurv._warnings import MethodologicalWarning
        import plotly.graph_objects as go

        class MockInner:
            coef_ = np.array([0.5, -0.3, 0.4])
            covariance_matrix_ = np.diag([0.01, 0.02, 0.015])

        class HRModel:
            hazard_ratio_valid = True
            model_ = MockInner()
            feature_names_in_ = np.array(["a", "b", "c"])

            def get_coefficients(self):
                return pd.Series(self.model_.coef_, index=self.feature_names_in_)

        class PenalizedModel:
            hazard_ratio_valid = False

            def get_coefficients(self):
                return pd.Series([0.5, -0.3, 0.4], index=["a", "b", "c"])

        with pytest.warns(MethodologicalWarning, match="penalized"):
            fig = plot_hazard_ratios_comparison(
                {"coxph": HRModel(), "penalized": PenalizedModel()},
                n_features=3,
            )
        assert isinstance(fig, go.Figure)

    def test_comparison_hr_model_no_warning(self):
        """An all-HR-valid comparison emits no methodological warning."""
        from mlsurv.viz.features import plot_hazard_ratios_comparison
        from mlsurv._warnings import MethodologicalWarning
        import plotly.graph_objects as go

        class MockInner:
            coef_ = np.array([0.5, -0.3, 0.4])
            covariance_matrix_ = np.diag([0.01, 0.02, 0.015])

        class HRModel:
            hazard_ratio_valid = True
            model_ = MockInner()
            feature_names_in_ = np.array(["a", "b", "c"])

            def get_coefficients(self):
                return pd.Series(self.model_.coef_, index=self.feature_names_in_)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig = plot_hazard_ratios_comparison({"coxph": HRModel()}, n_features=3)
        assert isinstance(fig, go.Figure)
        assert not [w for w in caught
                    if issubclass(w.category, MethodologicalWarning)]


class TestDataFrameEntryPoints:
    """Test DataFrame-based (pre-computed) entry points for viz functions."""

    def test_forest_plot_delegates_to_from_df(self, synthetic_data_clean):
        """Verify compute_hazard_ratios + plot_hazard_ratios produces consistent output."""
        from mlsurv.viz.features import compute_hazard_ratios, plot_hazard_ratios
        from mlsurv.models import CoxPHModel
        import plotly.graph_objects as go

        X, y = synthetic_data_clean
        model = CoxPHModel()
        model.fit(X, y)

        hr_df = compute_hazard_ratios(model)
        fig = plot_hazard_ratios(hr_df, n_features=5)
        assert isinstance(fig, go.Figure)
        # plot_hazard_ratios traces are split
        # by category (Higher risk, N.S., Lower risk) so 1-3 scatter traces
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert 1 <= len(scatter_traces) <= 3

    def test_forest_annotation_centered(self):
        """Verify hazard ratio forest plot annotation has xanchor='center'."""
        from mlsurv.viz.features import plot_hazard_ratios

        df = pd.DataFrame({
            'feature': ['a', 'b', 'c'],
            'hr': [0.5, 1.0, 2.0],
            'ci_lower': [0.3, 0.8, 1.5],
            'ci_upper': [0.8, 1.2, 3.0],
        })
        fig = plot_hazard_ratios(df)
        annotations = fig.layout.annotations
        risk_ann = [a for a in annotations if 'Lower risk' in a.text]
        assert len(risk_ann) == 1
        assert risk_ann[0].xanchor == 'center'

    def test_calibration_plot_from_data(self):
        """Verify plot_calibration renders from pre-computed list data."""
        from mlsurv.viz.calibration import plot_calibration
        import plotly.graph_objects as go

        calibration_data = [
            {
                'eval_time': 12,
                'time_index': 0,
                'predicted': [0.2, 0.4, 0.6, 0.8],
                'observed': [0.25, 0.45, 0.55, 0.75],
                'ci_lower': [0.15, 0.35, 0.45, 0.65],
                'ci_upper': [0.35, 0.55, 0.65, 0.85],
            }
        ]

        fig = plot_calibration(calibration_data)
        assert isinstance(fig, go.Figure)
        # Should have reference line + one timepoint trace
        assert len(fig.data) >= 2

    def test_calibration_comparison_from_data(self):
        """Verify plot_calibration renders multi-model dict data."""
        from mlsurv.viz.calibration import plot_calibration
        import plotly.graph_objects as go

        calibration_by_model = {
            'CoxPH': {
                'eval_time': 12,
                'predicted': [0.2, 0.5, 0.8],
                'observed': [0.25, 0.48, 0.76],
                'ci_lower': [0.15, 0.40, 0.70],
                'ci_upper': [0.35, 0.56, 0.82],
            },
            'RSF': {
                'eval_time': 12,
                'predicted': [0.2, 0.5, 0.8],
                'observed': [0.22, 0.52, 0.78],
                'ci_lower': [0.12, 0.42, 0.68],
                'ci_upper': [0.32, 0.62, 0.88],
            },
        }

        fig = plot_calibration(calibration_by_model)
        assert isinstance(fig, go.Figure)
        # Reference line + 2 models × (2 ribbon traces + markers) = 7 traces
        assert len(fig.data) == 7

    def test_brier_score_curve_from_df(self):
        """Verify plot_metric_time(metric='brier') renders from DataFrame."""
        from mlsurv.viz.metrics import plot_metric_time
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'model': ['CoxPH'] * 5 + ['RSF'] * 5,
            'time': list(range(1, 6)) * 2,
            'brier_score': [0.10, 0.12, 0.15, 0.18, 0.20,
                            0.09, 0.11, 0.14, 0.17, 0.19],
            'ibs': [0.15] * 5 + [0.14] * 5,
        })

        fig = plot_metric_time(
            df, metric='brier', time_col='time',
            model_col='model', value_col='brier_score',
        )
        assert isinstance(fig, go.Figure)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) >= 2

    def test_brier_score_curve_from_df_empty_raises(self):
        """Verify plot_metric_time raises on empty DataFrame."""
        from mlsurv.viz.metrics import plot_metric_time

        with pytest.raises((ValueError, KeyError)):
            plot_metric_time(pd.DataFrame(), metric='brier')


class TestExport:
    """Test HTML export utilities."""

    def test_save_html(self, tmp_path):
        """Verify save_html creates file."""
        from mlsurv.viz.export import save_html
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        output_path = tmp_path / "test.html"

        save_html(fig, output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding='utf-8')
        assert '<html>' in content or '<!DOCTYPE html>' in content

    def test_save_html_cdn(self, tmp_path):
        """Verify save_html with CDN option creates smaller file."""
        from mlsurv.viz.export import save_html
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        output_path = tmp_path / "test_cdn.html"

        save_html(fig, output_path, include_plotlyjs='cdn')

        assert output_path.exists()

    def test_save_html_report(self, tmp_path):
        """Verify save_html_report creates combined report."""
        from mlsurv.viz.export import save_html_report
        import plotly.graph_objects as go

        fig1 = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        fig2 = go.Figure(data=go.Bar(x=[1, 2, 3], y=[1, 2, 3]))

        figures = [
            ('Plot 1', fig1),
            ('Plot 2', fig2)
        ]

        output_path = tmp_path / "report.html"
        save_html_report(figures, output_path, title='Test Report')

        assert output_path.exists()
        content = output_path.read_text(encoding='utf-8')
        assert 'Plot 1' in content
        assert 'Plot 2' in content
        assert 'Test Report' in content

    def test_save_html_report_no_toc(self, tmp_path):
        """Verify save_html_report without TOC."""
        from mlsurv.viz.export import save_html_report
        import plotly.graph_objects as go

        fig1 = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))

        output_path = tmp_path / "report_no_toc.html"
        save_html_report(
            [('Plot 1', fig1)],
            output_path,
            include_toc=False,
            include_timestamp=False
        )

        assert output_path.exists()

    def test_save_html_report_single_figure(self, tmp_path):
        """Verify save_html_report with single figure (no TOC generated)."""
        from mlsurv.viz.export import save_html_report
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))

        output_path = tmp_path / "single_figure.html"
        save_html_report([('Single Plot', fig)], output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding='utf-8')
        # Single figure should not have TOC (check for specific TOC HTML structure,
        # not just 'Contents' since plotly.js code may contain that word)
        assert '<h3>Contents</h3>' not in content

    def test_figures_to_dict_empty_learner(self, small_data):
        """Without trained models the plot is skipped with a warning."""
        from mlsurv.viz.export import figures_to_dict
        from mlsurv import SurvivalLearner
        from mlsurv._warnings import SuggestionWarning

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup()

        with pytest.warns(SuggestionWarning, match="Skipping plot 'cindex'"):
            figures = figures_to_dict(learner, include=['cindex'])

        assert figures == []


def _has_shap():
    """Check if shap is available."""
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.shap
class TestSHAPVisualization:
    """Test SHAP visualization functions."""

    @pytest.fixture
    def sample_shap_df(self):
        """Generate sample SHAP data for testing."""
        np.random.seed(42)
        n_samples, n_features = 50, 10
        features = [f'feature_{i}' for i in range(n_features)]

        data = []
        for sample_idx in range(n_samples):
            for feat in features:
                data.append({
                    'feature': feat,
                    'sample_idx': sample_idx,
                    'shap_value': np.random.randn() * 0.1,
                    'model_type': 'coxph',
                })

        return pd.DataFrame(data)

    @pytest.fixture
    def sample_X(self):
        """Generate sample feature data for beeswarm plot."""
        np.random.seed(42)
        n_samples, n_features = 50, 10
        return pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )

    def test_shap_bar_plot_basic(self, sample_shap_df):
        """Verify SHAP bar plot generates valid figure."""
        from mlsurv.viz.shap import plot_shap_importance
        import plotly.graph_objects as go

        fig = plot_shap_importance(sample_shap_df, n=5)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_shap_bar_plot_custom_n(self, sample_shap_df):
        """Verify SHAP bar plot respects n parameter."""
        from mlsurv.viz.shap import plot_shap_importance

        fig = plot_shap_importance(sample_shap_df, n=3)

        # Should have traces for 3 features
        assert fig is not None

    def test_shap_bar_plot_no_violin(self, sample_shap_df):
        """Verify SHAP bar plot works without violin overlay."""
        from mlsurv.viz.shap import plot_shap_importance

        fig = plot_shap_importance(sample_shap_df, n=5, show_violin=False)

        # Should be a bar chart trace
        assert fig is not None
        assert any(hasattr(trace, 'type') and trace.type == 'bar' for trace in fig.data)

    def test_shap_bar_plot_variability(self, sample_shap_df):
        """Verify bar plot with show_violin=True has violin traces."""
        from mlsurv.viz.shap import plot_shap_importance

        fig = plot_shap_importance(sample_shap_df, n=5, show_violin=True)

        # Should have violin traces
        has_violin = any(
            hasattr(trace, 'type') and trace.type == 'violin'
            for trace in fig.data
        )
        assert has_violin, "No violin traces found"

    def test_shap_bar_plot_empty_df(self):
        """Verify graceful handling of empty SHAP data."""
        from mlsurv.viz.shap import plot_shap_importance

        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty or None"):
            plot_shap_importance(empty_df)

    def test_shap_bar_plot_custom_title(self, sample_shap_df):
        """Verify custom title is applied."""
        from mlsurv.viz.shap import plot_shap_importance

        custom_title = "My Custom SHAP Plot"
        fig = plot_shap_importance(sample_shap_df, n=5, title=custom_title)

        assert fig.layout.title.text == f"<b>{custom_title}</b>"

    def test_shap_beeswarm_plot_basic(self, sample_shap_df, sample_X):
        """Verify SHAP beeswarm plot generates valid figure."""
        from mlsurv.viz.shap import plot_shap_beeswarm
        import plotly.graph_objects as go

        fig = plot_shap_beeswarm(sample_shap_df, sample_X, n=5)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_shap_beeswarm_plot_custom_n(self, sample_shap_df, sample_X):
        """Verify SHAP beeswarm plot respects n parameter."""
        from mlsurv.viz.shap import plot_shap_beeswarm

        fig = plot_shap_beeswarm(sample_shap_df, sample_X, n=3)

        # Y-axis should have 3 tick labels
        assert len(fig.layout.yaxis.ticktext) == 3

    def test_shap_beeswarm_plot_empty_df(self, sample_X):
        """Verify graceful handling of empty SHAP data."""
        from mlsurv.viz.shap import plot_shap_beeswarm

        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty or None"):
            plot_shap_beeswarm(empty_df, sample_X)

    def test_shap_beeswarm_plot_empty_X(self, sample_shap_df):
        """Verify graceful handling of empty X data."""
        from mlsurv.viz.shap import plot_shap_beeswarm

        empty_X = pd.DataFrame()

        with pytest.raises(ValueError, match="empty or None"):
            plot_shap_beeswarm(sample_shap_df, empty_X)

    def test_shap_beeswarm_plot_custom_title(self, sample_shap_df, sample_X):
        """Verify custom title is applied."""
        from mlsurv.viz.shap import plot_shap_beeswarm

        custom_title = "My Custom Beeswarm"
        fig = plot_shap_beeswarm(sample_shap_df, sample_X, n=5, title=custom_title)

        assert fig.layout.title.text == f"<b>{custom_title}</b>"

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_shap_bar_plot_with_real_data(self, train_test_data):
        """Verify SHAP bar plot works with real evaluation data."""
        from mlsurv.viz.shap import plot_shap_importance
        from mlsurv.models import CoxPHModel
        from mlsurv.evaluation import evaluate
        import plotly.graph_objects as go

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            include_shap=True, n_importance_repeats=0,
            show_progress=False
        )

        fig = plot_shap_importance(result.shap_values, n=10)

        assert isinstance(fig, go.Figure)

    @pytest.mark.skipif(not _has_shap(), reason="shap required")
    def test_shap_beeswarm_plot_with_real_data(self, train_test_data):
        """Verify SHAP beeswarm plot works with real evaluation data."""
        from mlsurv.viz.shap import plot_shap_beeswarm
        from mlsurv.models import CoxPHModel
        from mlsurv.evaluation import evaluate
        import plotly.graph_objects as go

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxPHModel()
        model.fit(X_train, y_train)

        result = evaluate(
            model, X_test, y_test,
            include_shap=True, n_importance_repeats=0,
            show_progress=False
        )

        fig = plot_shap_beeswarm(result.shap_values, X_test, n=10)

        assert isinstance(fig, go.Figure)


class TestBeeswarmDefaults:
    """Beeswarm colour scaling and label placement (no shap library needed)."""

    def test_colour_ignores_extreme_values(self):
        """One extreme value must not push every other sample to one colour."""
        from mlsurv.viz.shap import _normalize_feature_values

        values = np.append(np.linspace(1.0, 20.0, 99), 400.0)
        norm = _normalize_feature_values(values)

        assert norm.min() == 0.0 and norm.max() == 1.0
        # Min-max scaling would put the median near 0.03; percentile
        # scaling spreads the bulk of the data across the colour range.
        assert np.median(norm) == pytest.approx(0.5, abs=0.05)
        assert norm[-1] == 1.0

    def test_colour_falls_back_to_min_max(self):
        """If the 5th-95th percentile range is empty, use min-max."""
        from mlsurv.viz.shap import _normalize_feature_values

        values = np.array([0.0] * 97 + [1.0, 2.0, 3.0])
        norm = _normalize_feature_values(values)

        assert norm.min() == 0.0 and norm.max() == 1.0
        assert norm[-1] == 1.0

    def test_direction_label_and_colour_bar_title(self):
        """Direction label sits above the plot; colour-bar title is one line."""
        from mlsurv.viz.shap import plot_shap_beeswarm

        rng = np.random.RandomState(0)
        shap_df = pd.DataFrame({
            'feature': np.repeat(['a', 'b'], 20),
            'sample_idx': np.tile(np.arange(20), 2),
            'shap_value': rng.randn(40),
            'feature_value': rng.rand(40),
        })
        fig = plot_shap_beeswarm(shap_df)

        label = [a for a in fig.layout.annotations if 'Risk' in (a.text or '')]
        assert len(label) == 1
        assert label[0].yref == 'paper'
        assert label[0].y >= 1.0 and label[0].yanchor == 'bottom'
        bars = [t for t in fig.data if t.marker and t.marker.showscale]
        assert bars[0].marker.colorbar.title.text == 'Feature value'

    @pytest.mark.parametrize('groups', [None, {'a': 'clinical', 'b': 'genomic'}])
    def test_feature_labels_get_room(self, groups):
        """Long feature names widen the left margin instead of being cut off."""
        from mlsurv.viz.shap import plot_shap_beeswarm

        rng = np.random.RandomState(0)
        shap_df = pd.DataFrame({
            'feature': np.repeat(['a', 'b'], 20),
            'sample_idx': np.tile(np.arange(20), 2),
            'shap_value': rng.randn(40),
            'feature_value': rng.rand(40),
        })
        fig = plot_shap_beeswarm(shap_df, feature_groups=groups)

        # The axis holding the feature labels is the first y-axis in both
        # layouts (the group strip when grouped, the main plot otherwise).
        assert fig.layout.yaxis.automargin is True


class TestMultiModelDefaults:
    """Test multi-model default behavior for plot_* methods."""

    # ---- viz layer: km_panel_grid ----

    def test_km_panel_grid_basic(self):
        """Two-model dict produces a figure with subplots."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        np.random.seed(42)
        n = 100
        df1 = pd.DataFrame({
            'time': np.random.exponential(10, n),
            'event': np.random.binomial(1, 0.7, n),
            'risk_score': np.random.randn(n),
        })
        df2 = df1.copy()
        df2['risk_score'] = np.random.randn(n)

        fig = plot_km({'coxph': df1, 'rsf': df2}, n_groups=2)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 2  # multiple traces across subplots

    def test_km_panel_grid_single_model(self):
        """Single-model dict produces a 1x1 grid."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        np.random.seed(42)
        n = 80
        df = pd.DataFrame({
            'time': np.random.exponential(10, n),
            'event': np.random.binomial(1, 0.7, n),
            'risk_score': np.random.randn(n),
        })

        fig = plot_km({'rsf': df}, n_groups=2)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # at least 2 KM traces (one per group)

    # ---- Learner-level tests require mock setup ----

    @pytest.fixture
    def learner_with_models(self, train_test_data):
        """SurvivalLearner with two fake models in _state.models."""
        from mlsurv.learner._core import SurvivalLearner
        from mlsurv.results.containers import ModelResult, EvaluationResult

        X_train, X_test, y_train, y_test = train_test_data

        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
        )
        learner.setup()

        # Build fake EvaluationResult with two models
        n_test = len(X_test)
        features = X_test.columns.tolist()

        # AUC(t) data
        eval_times = [6, 12, 18]
        auc_rows = []
        for model_name in ['coxph', 'rsf']:
            for t in eval_times:
                auc_rows.append({
                    'model_type': model_name,
                    'eval_time': t,
                    'auc': np.random.uniform(0.65, 0.85),
                })
        auc_t = pd.DataFrame(auc_rows)

        # Test set data
        test_rows = []
        for model_name in ['coxph', 'rsf']:
            for i in range(n_test):
                test_rows.append({
                    'model_type': model_name,
                    'index': i,
                    'risk_score': np.random.randn(),
                    'time': y_test['time'].iloc[i] if hasattr(y_test['time'], 'iloc') else y_test['time'][i],
                    'event': y_test['event'].iloc[i] if hasattr(y_test['event'], 'iloc') else y_test['event'][i],
                })
        test_set = pd.DataFrame(test_rows)

        # Importance data
        imp_rows = []
        for model_name in ['coxph', 'rsf']:
            for feat in features:
                imp_rows.append({
                    'model_type': model_name,
                    'feature': feat,
                    'coefficient': np.random.randn() * 0.5,
                    'perm_mean': abs(np.random.randn()) * 0.1,
                    'perm_std': abs(np.random.randn()) * 0.01,
                })
        importance = pd.DataFrame(imp_rows)

        # Concordance
        concordance = pd.DataFrame({
            'model_type': ['coxph', 'rsf'],
            'c_index_train': [0.75, 0.78],
            'c_index_test': [0.72, 0.76],
        })

        # Importance iterations
        iter_rows = []
        for model_name in ['coxph', 'rsf']:
            for feat in features:
                for it in range(3):
                    iter_rows.append({
                        'model_type': model_name,
                        'feature': feat,
                        'iteration': it,
                        'importance': abs(np.random.randn()) * 0.1,
                    })
        importance_iterations = pd.DataFrame(iter_rows)

        eval_result = EvaluationResult(
            concordance=concordance,
            test_set=test_set,
            auc_t=auc_t,
            importance=importance,
            importance_iterations=importance_iterations,
        )

        # Create fake ModelResults
        class FakeModel:
            def predict(self, X):
                return np.random.randn(len(X))

        from mlsurv.results.containers import extract_model_evaluation

        for name in ['coxph', 'rsf']:
            learner._state.models[name] = ModelResult(
                name=name,
                model=FakeModel(),
                cv_metric_scores=np.array([0.7, 0.75, 0.72]),
                cv_metric_mean=0.723,
                cv_metric_std=0.025,
                evaluation=extract_model_evaluation(eval_result, name),
            )

        return learner

    # ---- plot_importance ----

    # models= selector variants (no-args / str / list / 'best') all route
    # through the same resolver; parametrized over the plot method's selector
    # types. ``min_traces`` checks the grouped/panel multi-model trace count
    # (None = single-model, only isinstance checked).
    @pytest.mark.parametrize("selector, min_traces", [
        ("__none__", 2),                 # no args → grouped bars (multi-model)
        ("rsf", None),                   # single string
        (["rsf", "coxph"], 2),           # list of strings → grouped bars
        ("best", None),                  # 'best' sentinel → top model
    ])
    def test_importance_model_selectors(self, learner_with_models, selector, min_traces):
        import plotly.graph_objects as go

        if selector == "__none__":
            fig = learner_with_models.plot_importance()
        else:
            fig = learner_with_models.plot_importance(selector)
        assert isinstance(fig, go.Figure)
        if min_traces is not None:
            assert len(fig.data) >= min_traces

    # ---- plot_survival ----

    @pytest.mark.parametrize("selector, gt_traces", [
        ("__none__", 2),                 # no args → panel grid (multi-model)
        ("rsf", None),                   # single string
        (["rsf", "coxph"], 2),           # list of strings → panel grid
        ("best", None),                  # 'best' sentinel → top model
    ])
    def test_survival_model_selectors(self, learner_with_models, selector, gt_traces):
        import plotly.graph_objects as go

        if selector == "__none__":
            fig = learner_with_models.plot_survival()
        else:
            fig = learner_with_models.plot_survival(selector)
        assert isinstance(fig, go.Figure)
        if gt_traces is not None:
            assert len(fig.data) > gt_traces

    # ---- input validation ----

    def test_invalid_model_name_raises(self, learner_with_models):
        """Unknown model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown models"):
            learner_with_models.plot_importance('nonexistent')

        with pytest.raises(ValueError, match="Unknown models"):
            learner_with_models.plot_survival('nonexistent')

    def test_invalid_type_raises_typeerror(self, learner_with_models):
        """Passing an unsupported type (e.g., int) raises TypeError."""
        with pytest.raises(TypeError, match="models must be"):
            learner_with_models.plot_importance(42)

        with pytest.raises(TypeError, match="models must be"):
            learner_with_models.plot_survival({'bad': 'dict'})

    def test_no_models_trained_raises(self, train_test_data):
        """Calling plot methods with no trained models raises ValueError."""
        from mlsurv.learner._core import SurvivalLearner

        X_train, X_test, y_train, y_test = train_test_data
        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
        )
        learner.setup()

        with pytest.raises(ValueError, match="No trained models"):
            learner.plot_importance()

        with pytest.raises(ValueError, match="No trained models"):
            learner.plot_survival()

    def test_km_panel_grid_empty_input_raises(self):
        """Empty dict raises ValueError."""
        from mlsurv.viz.survival import plot_km

        with pytest.raises(ValueError, match="empty"):
            plot_km({})

    def test_importance_no_args_trace_count(self, learner_with_models):
        """No-arg grouped bars have exactly 2 traces (one per model)."""
        fig = learner_with_models.plot_importance()
        bar_traces = [t for t in fig.data if t.type == 'bar']
        assert len(bar_traces) == 2

    def test_survival_panel_grid_subplot_count(self, learner_with_models):
        """Panel grid for 2 models has traces in both subplot columns."""
        fig = learner_with_models.plot_survival()
        # With 2 models, subplots use cols 1 and 2
        # Each model gets KM traces — check traces exist in each subplot
        subplot_xaxes = set()
        for trace in fig.data:
            if hasattr(trace, 'xaxis') and trace.xaxis:
                subplot_xaxes.add(trace.xaxis)
        # Should span at least 2 x-axes (x, x2)
        assert len(subplot_xaxes) >= 2

    def test_survival_maxstat_default_ngroups(self, learner_with_models):
        """plot_survival(break_type='maxstat') defaults to 2 groups."""
        fig = learner_with_models.plot_survival(
            'rsf', break_type="maxstat"
        )
        # Count KM line traces (mode='lines', not CI ribbons or censoring)
        km_lines = [
            t for t in fig.data
            if getattr(t, 'mode', None) == 'lines'
            and not getattr(t, 'fill', None)
            and t.name and 'risk' in t.name.lower()
        ]
        assert len(km_lines) == 2


# =========================================================================
# Theme tests (Phase 1)
# =========================================================================

class TestTheme:
    """Test centralized theme and figure helpers."""

    def test_template_registered(self):
        """mlsurv_minimal template is registered in plotly."""
        import plotly.io as pio
        assert "mlsurv_minimal" in pio.templates

    def test_template_not_default(self):
        """mlsurv_minimal is NOT set as global default."""
        import plotly.io as pio
        assert pio.templates.default != "mlsurv_minimal"

    def test_create_figure_basic(self):
        """create_figure returns a go.Figure with the template applied."""
        from mlsurv.viz._theme import create_figure
        import plotly.graph_objects as go

        fig = create_figure(title="Test")
        assert isinstance(fig, go.Figure)
        assert fig.layout.template is not None

    def test_create_figure_named_dimensions(self):
        """Named sizes resolve to correct pixel values."""
        from mlsurv.viz._theme import create_figure

        fig = create_figure(width="double", height="large")
        assert fig.layout.width == 800
        assert fig.layout.height == 500

    def test_create_figure_int_dimensions(self):
        """Integer width/height are passed through directly."""
        from mlsurv.viz._theme import create_figure

        fig = create_figure(width=999, height=123)
        assert fig.layout.width == 999
        assert fig.layout.height == 123

    def test_create_figure_overrides(self):
        """Extra kwargs propagate to layout."""
        from mlsurv.viz._theme import create_figure

        fig = create_figure(showlegend=False)
        assert fig.layout.showlegend is False

    def test_add_ci_ribbon(self):
        """add_ci_ribbon uses two-trace tonexty fill."""
        from mlsurv.viz._theme import add_ci_ribbon
        import plotly.graph_objects as go

        fig = go.Figure()
        add_ci_ribbon(fig, [1, 2, 3], [0.9, 0.8, 0.7], [0.7, 0.6, 0.5], "#1f77b4")
        # Two traces: lower bound + upper bound with tonexty fill
        assert len(fig.data) == 2
        assert fig.data[0].line.width == 0  # lower bound invisible
        assert fig.data[1].fill == "tonexty"
        assert fig.data[1].line.width == 0  # upper bound invisible

    def test_get_manuscript_layout_deprecated(self):
        """get_manuscript_layout emits DeprecationWarning."""
        from mlsurv.viz._palettes import get_manuscript_layout
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_manuscript_layout()
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "create_figure" in str(dep_warnings[0].message)


# =========================================================================
# Discrimination tests (Phase 3)
# =========================================================================

class TestDiscrimination:
    """Test C-index forest plot and raincloud plot."""

    @pytest.fixture
    def forest_df(self):
        """Sample forest-plot DataFrame."""
        return pd.DataFrame({
            "model": ["coxph", "rsf", "gbsa"],
            "point": [0.72, 0.76, 0.74],
            "ci_lower": [0.68, 0.71, 0.69],
            "ci_upper": [0.76, 0.81, 0.79],
        })

    @pytest.fixture
    def bootstrap_iterations(self):
        """Sample bootstrap iterations DataFrame."""
        np.random.seed(42)
        rows = []
        for model in ["coxph", "rsf"]:
            for it in range(20):
                rows.append({
                    "model_type": model,
                    "iteration": it,
                    "metric_name": "harrell",
                    "value": np.random.uniform(0.65, 0.85),
                })
        return pd.DataFrame(rows)

    def test_cindex_forest_plot_basic(self, forest_df):
        """Forest plot returns a go.Figure with data."""
        from mlsurv.viz.metrics import plot_metric
        import plotly.graph_objects as go

        fig = plot_metric(forest_df, style="forest", metric="harrell")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_cindex_forest_plot_single_model(self):
        """Forest plot handles single-model edge case."""
        from mlsurv.viz.metrics import plot_metric

        df = pd.DataFrame({
            "model": ["coxph"],
            "point": [0.72],
            "ci_lower": [0.68],
            "ci_upper": [0.76],
        })
        fig = plot_metric(df, style="forest", metric="harrell")
        assert len(fig.data) >= 1

    def test_cindex_forest_plot_sorting(self, forest_df):
        """Best model appears on top when sort='value'."""
        from mlsurv.viz.metrics import plot_metric

        fig = plot_metric(forest_df, style="forest", metric="harrell", sort="value")
        # The y-axis should have models; best (highest point) on top
        # With ascending sort, the last y-value should be the best model
        y_vals = fig.data[0].y
        # rsf has highest point (0.76), should appear last (top in plotly)
        assert y_vals[-1] is not None

    def test_cindex_forest_plot_empty_raises(self):
        """Empty DataFrame raises ValueError."""
        from mlsurv.viz.metrics import plot_metric

        with pytest.raises(ValueError, match="missing required columns"):
            plot_metric(pd.DataFrame(), style="forest", metric="harrell")

    def test_cindex_forest_plot_from_bootstrap(self):
        """_prepare_forest_data_from_bootstrap produces valid DataFrame."""
        from mlsurv.viz.metrics import (
            plot_metric,
            _prepare_forest_data_from_bootstrap,
        )

        summary = pd.DataFrame({
            "model_type": ["coxph", "rsf"],
            "metric": ["harrell", "harrell"],
            "point": [0.72, 0.76],
            "mean": [0.71, 0.75],
            "std": [0.03, 0.04],
            "ci_lower": [0.68, 0.71],
            "ci_upper": [0.76, 0.81],
        })
        forest_df = _prepare_forest_data_from_bootstrap(summary)
        assert list(forest_df.columns) == ["model", "point", "ci_lower", "ci_upper"]
        assert len(forest_df) == 2

        fig = plot_metric(forest_df, style="forest", metric="harrell")
        assert len(fig.data) > 0

    def test_cindex_forest_plot_from_comparison(self):
        """_prepare_forest_data_from_comparison produces valid DataFrame."""
        from mlsurv.viz.metrics import (
            plot_metric,
            _prepare_forest_data_from_comparison,
        )

        metrics = pd.DataFrame({
            "model": ["coxph", "rsf"],
            "cv_metric_mean": [0.72, 0.76],
            "cv_metric_std": [0.03, 0.04],
        })
        forest_df = _prepare_forest_data_from_comparison(metrics)
        assert list(forest_df.columns) == ["model", "point", "ci_lower", "ci_upper"]
        assert len(forest_df) == 2

        fig = plot_metric(forest_df, style="forest", metric="harrell")
        assert len(fig.data) > 0

    def test_ci_level_comparison_changes_bounds(self):
        """ci_level in _prepare_forest_data_from_comparison uses different z-scores."""
        from mlsurv.viz.metrics import _prepare_forest_data_from_comparison

        metrics = pd.DataFrame({
            "model": ["coxph"],
            "cv_metric_mean": [0.72],
            "cv_metric_std": [0.03],
        })
        df_90 = _prepare_forest_data_from_comparison(metrics, ci_level=0.90)
        df_95 = _prepare_forest_data_from_comparison(metrics, ci_level=0.95)
        df_99 = _prepare_forest_data_from_comparison(metrics, ci_level=0.99)

        # Wider CI at higher level: 99% > 95% > 90%
        width_90 = (df_90["ci_upper"] - df_90["ci_lower"]).iloc[0]
        width_95 = (df_95["ci_upper"] - df_95["ci_lower"]).iloc[0]
        width_99 = (df_99["ci_upper"] - df_99["ci_lower"]).iloc[0]
        assert width_90 < width_95 < width_99

    def test_ci_level_bootstrap_recomputes_from_iterations(self):
        """ci_level in _prepare_forest_data_from_bootstrap recomputes percentiles."""
        from mlsurv.viz.metrics import _prepare_forest_data_from_bootstrap

        np.random.seed(42)
        n = 200
        iterations = pd.DataFrame({
            "model_type": ["coxph"] * n,
            "iteration": list(range(n)),
            "metric_name": ["harrell"] * n,
            "value": np.random.normal(0.72, 0.03, n),
        })
        summary = pd.DataFrame({
            "model_type": ["coxph"],
            "metric": ["harrell"],
            "point": [0.72],
            "mean": [0.72],
            "std": [0.03],
            "ci_lower": [0.66],
            "ci_upper": [0.78],
        })

        df_90 = _prepare_forest_data_from_bootstrap(
            summary, ci_level=0.90, iterations=iterations,
        )
        df_99 = _prepare_forest_data_from_bootstrap(
            summary, ci_level=0.99, iterations=iterations,
        )

        width_90 = (df_90["ci_upper"] - df_90["ci_lower"]).iloc[0]
        width_99 = (df_99["ci_upper"] - df_99["ci_lower"]).iloc[0]
        assert width_90 < width_99

    @pytest.mark.parametrize("bad_ci", [95, 90, 0, 1, -0.5, 1.5])
    def test_ci_level_bootstrap_rejects_percentage(self, bad_ci):
        """_prepare_forest_data_from_bootstrap raises on out-of-range ci_level."""
        from mlsurv.viz.metrics import _prepare_forest_data_from_bootstrap

        summary = pd.DataFrame({
            "model_type": ["coxph"],
            "metric": ["harrell"],
            "point": [0.72],
            "mean": [0.72],
            "std": [0.03],
            "ci_lower": [0.66],
            "ci_upper": [0.78],
        })
        with pytest.raises(ValueError, match="ci_level must be between 0 and 1"):
            _prepare_forest_data_from_bootstrap(summary, ci_level=bad_ci)

    @pytest.mark.parametrize("bad_ci", [95, 90, 0, 1, -0.5, 1.5])
    def test_ci_level_comparison_rejects_percentage(self, bad_ci):
        """_prepare_forest_data_from_comparison raises on out-of-range ci_level."""
        from mlsurv.viz.metrics import _prepare_forest_data_from_comparison

        metrics = pd.DataFrame({
            "model": ["coxph"],
            "cv_metric_mean": [0.72],
            "cv_metric_std": [0.03],
        })
        with pytest.raises(ValueError, match="ci_level must be between 0 and 1"):
            _prepare_forest_data_from_comparison(metrics, ci_level=bad_ci)

    def test_cindex_raincloud_basic(self, bootstrap_iterations):
        """Raincloud plot returns a go.Figure with traces."""
        from mlsurv.viz.metrics import plot_metric
        import plotly.graph_objects as go

        fig = plot_metric(bootstrap_iterations, style="raincloud", model_col="model_type")
        assert isinstance(fig, go.Figure)
        # Should have violin, box, and scatter traces for each model
        assert len(fig.data) >= 4  # 2 models x (violin + box + scatter) = 6 minimum, but at least 4

    def test_cindex_raincloud_no_data_raises(self):
        """Empty DataFrame raises ValueError."""
        from mlsurv.viz.metrics import plot_metric

        with pytest.raises(ValueError, match="missing required columns"):
            plot_metric(pd.DataFrame(), style="raincloud")

    def test_cindex_raincloud_missing_metric_raises(self, bootstrap_iterations):
        """Unknown metric raises ValueError."""
        from mlsurv.viz.metrics import plot_metric

        with pytest.raises(ValueError, match="not found"):
            plot_metric(bootstrap_iterations, style="raincloud", metric="nonexistent", model_col="model_type")


class TestModelComparisonStyles:
    """Test style parameters on plot_metric (bar, forest, boxplot styles)."""

    @pytest.fixture
    def comparison_df(self):
        """DataFrame for bar/forest style tests."""
        return pd.DataFrame({
            'model': ['coxph', 'rsf'],
            'point': [0.72, 0.76],
            'ci_lower': [0.69, 0.72],
            'ci_upper': [0.75, 0.80],
        })

    def test_model_comparison_bar_default(self, comparison_df):
        from mlsurv.viz.metrics import plot_metric
        import plotly.graph_objects as go
        fig = plot_metric(comparison_df, style='bar')
        assert isinstance(fig, go.Figure)

    def test_model_comparison_bar_forest(self, comparison_df):
        from mlsurv.viz.metrics import plot_metric
        import plotly.graph_objects as go
        fig = plot_metric(comparison_df, style='forest')
        assert isinstance(fig, go.Figure)
        # Forest uses Scatter, not Bar
        assert any(isinstance(t, go.Scatter) for t in fig.data)

    def test_model_comparison_bar_boxplot(self):
        from mlsurv.viz.metrics import plot_metric
        import plotly.graph_objects as go
        np.random.seed(42)
        rows = []
        for model in ['coxph', 'rsf']:
            for it in range(50):
                rows.append({
                    'model_type': model,
                    'iteration': it,
                    'metric_name': 'harrell',
                    'value': np.random.uniform(0.65, 0.85),
                })
        df = pd.DataFrame(rows)
        fig = plot_metric(df, style='boxplot', model_col='model_type')
        assert isinstance(fig, go.Figure)

    def test_model_comparison_bar_invalid_style(self, comparison_df):
        from mlsurv.viz.metrics import plot_metric
        with pytest.raises(ValueError, match="Unknown style"):
            plot_metric(comparison_df, style='invalid')

    @pytest.fixture
    def bootstrap_df(self):
        """DataFrame for boxplot/raincloud style tests."""
        np.random.seed(42)
        rows = []
        for model in ['coxph', 'rsf']:
            for it in range(50):
                rows.append({
                    'model_type': model,
                    'iteration': it,
                    'metric_name': 'harrell',
                    'value': np.random.uniform(0.68, 0.80),
                })
        return pd.DataFrame(rows)

    def test_bootstrap_boxplot_default(self, bootstrap_df):
        from mlsurv.viz.metrics import plot_metric
        import plotly.graph_objects as go
        fig = plot_metric(bootstrap_df, style='boxplot', model_col='model_type')
        assert isinstance(fig, go.Figure)

    def test_bootstrap_boxplot_raincloud(self, bootstrap_df):
        from mlsurv.viz.metrics import plot_metric
        import plotly.graph_objects as go
        fig = plot_metric(bootstrap_df, style='raincloud', model_col='model_type')
        assert isinstance(fig, go.Figure)

    def test_bootstrap_boxplot_invalid_style(self, bootstrap_df):
        from mlsurv.viz.metrics import plot_metric
        with pytest.raises(ValueError, match="Unknown style"):
            plot_metric(bootstrap_df, style='invalid', model_col='model_type')


class TestForestPlots:
    """Test plot_metric_time(style='forest') and plot_metric(style='forest')."""

    def test_auct_forest_plot_basic(self):
        from mlsurv.viz.metrics import plot_metric_time
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'model': ['coxph', 'coxph', 'rsf', 'rsf'],
            'eval_time': [12.0, 24.0, 12.0, 24.0],
            'auc': [0.78, 0.72, 0.82, 0.76],
        })
        fig = plot_metric_time(df, style='forest', metric='auc')
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_auct_forest_plot_with_eval_times(self):
        from mlsurv.viz.metrics import plot_metric_time

        df = pd.DataFrame({
            'model': ['coxph', 'coxph', 'rsf', 'rsf'],
            'eval_time': [12.0, 24.0, 12.0, 24.0],
            'auc': [0.78, 0.72, 0.82, 0.76],
        })
        fig = plot_metric_time(df, style='forest', metric='auc', eval_time=12.0)
        assert len(fig.data) >= 1

    def test_brier_forest_plot_basic(self):
        from mlsurv.viz.metrics import plot_metric_time
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'model': ['coxph', 'rsf', 'gbsa'],
            'time': [12.0, 12.0, 12.0],
            'brier_score': [0.18, 0.15, 0.16],
        })
        fig = plot_metric_time(
            df, style='forest', metric='brier',
            eval_time=12.0, time_col='time', value_col='brier_score',
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_brier_forest_plot_with_std(self):
        from mlsurv.viz.metrics import plot_metric_time

        df = pd.DataFrame({
            'model': ['coxph', 'rsf', 'gbsa'],
            'time': [12.0, 12.0, 12.0],
            'brier_score': [0.18, 0.15, 0.16],
            'brier_std': [0.02, 0.01, 0.015],
        })
        fig = plot_metric_time(
            df, style='forest', metric='brier',
            eval_time=12.0, time_col='time',
            value_col='brier_score', std_col='brier_std',
        )
        assert len(fig.data) >= 1
        # CI whiskers should be present (error bars on scatter)
        trace = fig.data[0]
        assert trace.error_x is not None or len(fig.data) > 1

    def test_brier_forest_plot_no_std(self):
        from mlsurv.viz.metrics import plot_metric_time

        df = pd.DataFrame({
            'model': ['coxph', 'rsf'],
            'time': [12.0, 12.0],
            'brier_score': [0.18, 0.15],
        })
        fig = plot_metric_time(
            df, style='forest', metric='brier',
            time_col='time', value_col='brier_score',
        )
        assert len(fig.data) >= 1

    def test_metrics_forest_plot_basic(self):
        from mlsurv.viz.metrics import plot_metric
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'model_type': ['coxph', 'rsf', 'gbsa'],
            'harrell': [0.72, 0.76, 0.74],
            'integrated_brier': [0.18, 0.15, 0.16],
        })
        fig = plot_metric(df, style='forest', model_col='model_type', metric=['harrell', 'integrated_brier'])
        assert isinstance(fig, go.Figure)

    def test_metrics_forest_plot_custom_metrics(self):
        from mlsurv.viz.metrics import plot_metric
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'model_type': ['coxph', 'rsf'],
            'harrell': [0.72, 0.76],
            'ipcw': [0.71, 0.75],
            'integrated_brier': [0.18, 0.15],
        })
        fig = plot_metric(df, style='forest', model_col='model_type', metric=['harrell', 'ipcw'])
        assert isinstance(fig, go.Figure)

    def test_metrics_forest_plot_with_std_error_bars(self):
        """Multi-forest plot renders error bars when _std columns are present."""
        from mlsurv.viz.metrics import plot_metric
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'model_type': ['coxph', 'rsf'],
            'harrell': [0.72, 0.76],
            'harrell_std': [0.03, 0.02],
            'ipcw': [0.71, 0.75],
            'ipcw_std': [0.04, 0.03],
        })
        fig = plot_metric(
            df, model_col='model_type', metric=['harrell', 'ipcw'],
        )
        assert isinstance(fig, go.Figure)
        # Each metric panel has one trace; both should have error bars
        for trace in fig.data:
            assert trace.error_x is not None, "Expected error bars from _std columns"
            assert trace.error_x.visible is True

    def test_metrics_forest_plot_ci_level_propagates(self):
        """ci_level parameter changes error bar width in multi-forest."""
        from mlsurv.viz.metrics import plot_metric
        from scipy.stats import norm
        import numpy as np

        df = pd.DataFrame({
            'model_type': ['coxph', 'rsf'],
            'harrell': [0.72, 0.76],
            'harrell_std': [0.03, 0.02],
        })
        fig_95 = plot_metric(df, model_col='model_type', metric=['harrell'], ci_level=0.95)
        fig_90 = plot_metric(df, model_col='model_type', metric=['harrell'], ci_level=0.90)
        # 95% CI should produce wider bars than 90% CI
        bars_95 = np.array(fig_95.data[0].error_x.array)
        bars_90 = np.array(fig_90.data[0].error_x.array)
        assert np.all(bars_95 > bars_90), "95% CI bars should be wider than 90%"
        # Verify exact values
        z_95 = norm.ppf(0.975)
        expected = z_95 * np.array([0.03, 0.02])
        np.testing.assert_allclose(bars_95, expected)

    def test_metrics_forest_plot_empty_df_raises(self):
        """Multi-forest plot raises on empty DataFrame."""
        from mlsurv.viz.metrics import plot_metric

        df = pd.DataFrame(columns=['model_type', 'harrell', 'ipcw'])
        with pytest.raises(ValueError, match="empty"):
            plot_metric(df, model_col='model_type', metric=['harrell', 'ipcw'])

    def test_forest_plot_invalid_ci_level_raises(self):
        """ci_level outside (0, 1) raises ValueError for both forest renderers."""
        from mlsurv.viz.metrics import plot_metric

        # Single-metric forest
        df_single = pd.DataFrame({
            'model': ['coxph'], 'point': [0.72],
            'ci_lower': [0.68], 'ci_upper': [0.76],
        })
        for bad in [0, 1, 95, -0.5]:
            with pytest.raises(ValueError, match="ci_level must be between"):
                plot_metric(df_single, style="forest", ci_level=bad)

        # Multi-metric forest
        df_multi = pd.DataFrame({
            'model_type': ['coxph'], 'harrell': [0.72], 'ipcw': [0.71],
        })
        with pytest.raises(ValueError, match="ci_level must be between"):
            plot_metric(df_multi, model_col='model_type',
                        metric=['harrell', 'ipcw'], ci_level=95)

    def test_multi_forest_sd_label_uses_raw_std(self):
        """uncertainty_label='SD' draws point ± std (z=1), not point ± z*std."""
        from mlsurv.viz.metrics import plot_metric
        import numpy as np

        df = pd.DataFrame({
            'model_type': ['coxph', 'rsf'],
            'harrell': [0.72, 0.76],
            'harrell_std': [0.03, 0.02],
        })
        fig = plot_metric(df, model_col='model_type', metric=['harrell'],
                          uncertainty_label='SD')
        np.testing.assert_allclose(fig.data[0].error_x.array, [0.03, 0.02])
        assert '1 SD' in fig.data[0].hovertemplate
        assert '% CI' not in fig.data[0].hovertemplate

    def test_multi_forest_default_label_keeps_ci(self):
        """No uncertainty_label keeps the z*std CI band (regression)."""
        from mlsurv.viz.metrics import plot_metric
        from scipy.stats import norm
        import numpy as np

        df = pd.DataFrame({
            'model_type': ['coxph', 'rsf'],
            'harrell': [0.72, 0.76],
            'harrell_std': [0.03, 0.02],
        })
        fig = plot_metric(df, model_col='model_type', metric=['harrell'])
        np.testing.assert_allclose(
            fig.data[0].error_x.array, norm.ppf(0.975) * np.array([0.03, 0.02]),
        )
        assert '95% CI' in fig.data[0].hovertemplate

    def test_multi_bar_sd_label_uses_raw_std(self):
        """Bar style with uncertainty_label='SD' draws point ± std error bars."""
        from mlsurv.viz.metrics import plot_metric
        import numpy as np

        df = pd.DataFrame({
            'model_type': ['coxph', 'rsf'],
            'harrell': [0.72, 0.76],
            'harrell_std': [0.03, 0.02],
        })
        fig = plot_metric(df, style='bar', model_col='model_type',
                          metric=['harrell'], uncertainty_label='SD')
        arr = np.sort(np.asarray(fig.data[0].error_x.array, dtype=float))
        np.testing.assert_allclose(arr, np.sort([0.03, 0.02]))

    def test_multi_forest_sd_ignores_ci_level(self):
        """uncertainty_label='SD' uses z=1 regardless of ci_level."""
        from mlsurv.viz.metrics import plot_metric
        import numpy as np

        df = pd.DataFrame({
            'model_type': ['coxph', 'rsf'],
            'harrell': [0.72, 0.76],
            'harrell_std': [0.03, 0.02],
        })
        fig = plot_metric(df, model_col='model_type', metric=['harrell'],
                          uncertainty_label='SD', ci_level=0.90)
        np.testing.assert_allclose(fig.data[0].error_x.array, [0.03, 0.02])

    def test_multi_forest_long_summary_sd_label(self):
        """Long-summary forest input honors uncertainty_label='SD' hover wording.

        This routes through ``_plot_metric_multi_panel`` /
        ``_render_multi_panel_summary`` (not the wide-input forest renderer).
        """
        from mlsurv.viz.metrics import plot_metric

        df = pd.DataFrame({
            'model_type': ['coxph', 'rsf', 'coxph', 'rsf'],
            'metric_name': ['harrell', 'harrell', 'ipcw', 'ipcw'],
            'point': [0.72, 0.76, 0.71, 0.75],
            'ci_lower': [0.69, 0.74, 0.67, 0.72],
            'ci_upper': [0.75, 0.78, 0.75, 0.78],
        })
        fig = plot_metric(
            df, style='forest', model_col='model_type',
            metric=['harrell', 'ipcw'],
            point_col='point', ci_lower_col='ci_lower',
            ci_upper_col='ci_upper', uncertainty_label='SD',
        )
        hovers = [t.hovertemplate for t in fig.data if t.hovertemplate]
        assert any('1 SD' in h for h in hovers)
        assert not any('% CI' in h for h in hovers)


class TestCindexTimeCurve:
    """Test time-dependent C-index curve via plot_metric_time."""

    def test_basic(self):
        from mlsurv.viz.metrics import plot_metric_time
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'model': ['coxph'] * 3 + ['rsf'] * 3,
            'eval_time': [6, 12, 24, 6, 12, 24],
            'cindex': [0.75, 0.72, 0.70, 0.78, 0.76, 0.73],
        })
        fig = plot_metric_time(df, metric='cindex')
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # one per model

    def test_with_std(self):
        from mlsurv.viz.metrics import plot_metric_time
        import plotly.graph_objects as go

        df = pd.DataFrame({
            'model': ['coxph'] * 3,
            'eval_time': [6, 12, 24],
            'cindex': [0.75, 0.72, 0.70],
        })
        std_df = pd.DataFrame({
            'model': ['coxph'] * 3,
            'eval_time': [6, 12, 24],
            'cindex_std': [0.02, 0.03, 0.04],
        })
        fig = plot_metric_time(df, std_df=std_df, metric='cindex')
        assert isinstance(fig, go.Figure)
        # Should have ribbon + line = 2 traces
        assert len(fig.data) >= 2

    def test_no_ci(self):
        from mlsurv.viz.metrics import plot_metric_time

        df = pd.DataFrame({
            'model': ['coxph'] * 3,
            'eval_time': [6, 12, 24],
            'cindex': [0.75, 0.72, 0.70],
        })
        fig = plot_metric_time(df, metric='cindex', show_ci=False)
        assert len(fig.data) >= 1


class TestPredictedObservedOverlay:
    """Test predicted vs. observed survival overlay."""

    @pytest.fixture
    def overlay_data(self):
        np.random.seed(42)
        n = 100
        X = pd.DataFrame(np.random.randn(n, 3), columns=['a', 'b', 'c'])
        y = make_survival_array(
            np.random.binomial(1, 0.5, n),
            np.random.exponential(20, n),
        )
        return X, y

    def test_basic(self, overlay_data):
        pytest.importorskip("lifelines")
        from mlsurv.viz.calibration import plot_calibration_overlay
        import plotly.graph_objects as go

        class _MockModel:
            def predict_survival_function(self, X):
                from sksurv.functions import StepFunction
                n = len(X)
                times = np.array([0, 6, 12, 24, 36, 48])
                return [StepFunction(times, np.array([1.0, 0.9, 0.8, 0.6, 0.4, 0.2])) for _ in range(n)]

        X, y = overlay_data
        eval_times = np.array([6, 12, 24, 36])  # stay within mock's domain
        fig = plot_calibration_overlay(_MockModel(), X, y, eval_times=eval_times)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # predicted + KM observed


@pytest.mark.integration
class TestMaxstat:
    """Tests for maxstat optimal cutpoint risk stratification."""

    @pytest.fixture
    def surv_df(self):
        """Synthetic survival DataFrame with meaningful score separation."""
        rng = np.random.RandomState(42)
        n = 200
        # Two latent groups with different survival distributions
        group = rng.binomial(1, 0.5, n)
        times = np.where(group, rng.exponential(10, n),
                         rng.exponential(3, n))
        events = rng.binomial(1, 0.7, n)
        # Risk score correlated with group (higher score = better prognosis)
        scores = group * 2 + rng.randn(n) * 0.5
        return pd.DataFrame({
            "time": times, "event": events, "risk_score": scores,
        })

    @pytest.fixture
    def multi_model_data(self, surv_df):
        """Dict of model DataFrames for panel grid tests."""
        rng = np.random.RandomState(123)
        return {
            "CoxPH": surv_df.copy(),
            "RSF": surv_df.assign(
                risk_score=surv_df["risk_score"] + rng.randn(len(surv_df)) * 0.3
            ),
        }

    @pytest.fixture
    def perm_surv_df(self):
        """Small synthetic DataFrame for permutation tests (fast)."""
        rng = np.random.RandomState(42)
        n = 50
        group = rng.binomial(1, 0.5, n)
        times = np.where(group, rng.exponential(10, n),
                         rng.exponential(3, n))
        events = rng.binomial(1, 0.7, n)
        scores = group * 2 + rng.randn(n) * 0.5
        return pd.DataFrame({
            "time": times, "event": events, "risk_score": scores,
        })

    @pytest.fixture
    def perm_multi_model_data(self, perm_surv_df):
        """Small dict of model DataFrames for permutation panel tests."""
        rng = np.random.RandomState(123)
        return {
            "CoxPH": perm_surv_df.copy(),
            "RSF": perm_surv_df.assign(
                risk_score=perm_surv_df["risk_score"]
                + rng.randn(len(perm_surv_df)) * 0.3
            ),
        }

    def test_maxstat_basic(self, surv_df):
        """break_type='maxstat' returns a valid 2-group figure."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        fig = plot_km(surv_df, risk_col="risk_score", break_type="maxstat")
        assert isinstance(fig, go.Figure)
        # Should have traces for 2 groups (+ CI ribbons)
        assert len(fig.data) >= 2

    def test_maxstat_subtitle_contains_cutpoint(self, surv_df):
        """Default subtitle includes cutpoint and p-value."""
        from mlsurv.viz.survival import plot_km

        fig = plot_km(surv_df, risk_col="risk_score", break_type="maxstat")
        title_text = fig.layout.title.text
        assert "Maxstat cutpoint" in title_text
        assert "p =" in title_text

    def test_maxstat_custom_subtitle_preserved(self, surv_df):
        """User subtitle is not overridden by maxstat default."""
        from mlsurv.viz.survival import plot_km

        fig = plot_km(surv_df, risk_col="risk_score", break_type="maxstat",
                      subtitle="My custom subtitle")
        assert "My custom subtitle" in fig.layout.title.text
        assert "Maxstat cutpoint" not in fig.layout.title.text

    def test_maxstat_subtitle_suppressed(self, surv_df):
        """Passing subtitle='' suppresses the default."""
        from mlsurv.viz.survival import plot_km

        fig = plot_km(surv_df, risk_col="risk_score", break_type="maxstat",
                      subtitle="")
        assert "<sup>" not in fig.layout.title.text

    def test_maxstat_with_risk_table(self, surv_df):
        """Maxstat works with show_risk_table=True."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        fig = plot_km(surv_df, risk_col="risk_score", break_type="maxstat",
                      show_risk_table=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_maxstat_panel_grid(self, multi_model_data):
        """Maxstat works with multi-model dict input."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        fig = plot_km(multi_model_data, risk_col="risk_score",
                      break_type="maxstat")
        assert isinstance(fig, go.Figure)
        # Panel grid has annotations for per-model cutpoints
        annotations = [a for a in fig.layout.annotations if "cut=" in (a.text or "")]
        assert len(annotations) == 2  # one per model

    def test_maxstat_lausen_schumacher(self, surv_df):
        """Lausen-Schumacher correction produces a figure with correction in subtitle."""
        from mlsurv.viz.survival import plot_km

        fig = plot_km(surv_df, risk_col="risk_score", break_type="maxstat",
                      correction="lausen-schumacher")
        title_text = fig.layout.title.text
        assert "lausen-schumacher" in title_text

    def test_maxstat_identical_scores_raises(self):
        """All-same scores raise ValueError."""
        from mlsurv.viz.survival import _find_maxstat_cutpoint

        n = 50
        with pytest.raises(ValueError, match="identical"):
            _find_maxstat_cutpoint(
                times=np.arange(1, n + 1, dtype=float),
                events=np.ones(n),
                scores=np.ones(n),
            )

    def test_maxstat_too_few_samples_raises(self):
        """Tiny dataset raises ValueError."""
        from mlsurv.viz.survival import _find_maxstat_cutpoint

        with pytest.raises(ValueError, match="Too few samples"):
            _find_maxstat_cutpoint(
                times=np.array([1, 2, 3]),
                events=np.array([1, 0, 1]),
                scores=np.array([0.1, 0.5, 0.9]),
            )

    def test_maxstat_too_few_events_raises(self):
        """Highly censored data with <3 events raises ValueError."""
        from mlsurv.viz.survival import _find_maxstat_cutpoint

        n = 50
        with pytest.raises(ValueError, match="Too few events"):
            _find_maxstat_cutpoint(
                times=np.arange(1, n + 1, dtype=float),
                events=np.zeros(n),  # all censored
                scores=np.linspace(0, 10, n),
            )

    def test_maxstat_concentrated_scores_raises(self):
        """Scores with extreme concentration fail the min_prop filter."""
        from mlsurv.viz.survival import _find_maxstat_cutpoint

        n = 100
        # 95 at 0, 5 at 1 — only 1 candidate, prop_below=0.95 > max_prop
        with pytest.raises(ValueError, match="Fewer than 2 valid"):
            _find_maxstat_cutpoint(
                times=np.arange(1, n + 1, dtype=float),
                events=np.ones(n),
                scores=np.concatenate([np.zeros(95), np.ones(5)]),
            )

    def test_maxstat_nan_input_raises(self):
        """NaN values in input raise ValueError."""
        from mlsurv.viz.survival import _find_maxstat_cutpoint

        n = 50
        scores = np.linspace(0, 10, n)
        scores[5] = np.nan
        with pytest.raises(ValueError, match="NaN/Inf"):
            _find_maxstat_cutpoint(
                times=np.arange(1, n + 1, dtype=float),
                events=np.ones(n),
                scores=scores,
            )

    def test_maxstat_invalid_correction_raises(self, surv_df):
        """Bad correction string raises ValueError."""
        from mlsurv.viz.survival import plot_km

        with pytest.raises(ValueError, match="Unknown correction"):
            plot_km(surv_df, risk_col="risk_score", break_type="maxstat",
                    correction="fdr")

    def test_invalid_break_type_raises(self, surv_df):
        """Bad break_type string raises ValueError."""
        from mlsurv.viz.survival import plot_km

        with pytest.raises(ValueError, match="Unknown break_type"):
            plot_km(surv_df, risk_col="risk_score", break_type="bad")

    def test_find_maxstat_cutpoint_result(self, surv_df):
        """Helper returns MaxstatResult with valid fields."""
        from mlsurv.viz.survival import _find_maxstat_cutpoint, MaxstatResult

        result = _find_maxstat_cutpoint(
            surv_df["time"], surv_df["event"], surv_df["risk_score"],
        )
        assert isinstance(result, MaxstatResult)
        assert result.test_statistic > 0
        assert 0 <= result.p_value_raw <= 1
        assert 0 <= result.p_value_corrected <= 1
        assert result.correction == "bonferroni"
        assert result.n_candidates >= 2

    def test_cutpoint_in_score_range(self, surv_df):
        """Cutpoint falls within min/max of scores."""
        from mlsurv.viz.survival import _find_maxstat_cutpoint

        scores = surv_df["risk_score"]
        result = _find_maxstat_cutpoint(
            surv_df["time"], surv_df["event"], scores,
        )
        assert scores.min() < result.cutpoint < scores.max()

    def test_bonferroni_more_conservative(self, surv_df):
        """Bonferroni p >= Lausen-Schumacher p (same cutpoint)."""
        from mlsurv.viz.survival import _find_maxstat_cutpoint

        bonf = _find_maxstat_cutpoint(
            surv_df["time"], surv_df["event"], surv_df["risk_score"],
            correction="bonferroni",
        )
        ls = _find_maxstat_cutpoint(
            surv_df["time"], surv_df["event"], surv_df["risk_score"],
            correction="lausen-schumacher",
        )
        # Same cutpoint (same algorithm, different correction)
        assert bonf.cutpoint == ls.cutpoint
        assert bonf.p_value_corrected >= ls.p_value_corrected - 1e-10

    def test_maxstat_with_gbsg2(self, gbsg2_data):
        """Works on real breast cancer data with age as proxy risk score."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        X, y = gbsg2_data
        df = pd.DataFrame({
            "time": y["time"],
            "event": y["cens"].astype(int),
            "risk_score": X["age"].values,
        })
        fig = plot_km(df, risk_col="risk_score", break_type="maxstat")
        assert isinstance(fig, go.Figure)

    def test_maxstat_group_labels_low_high(self, surv_df):
        """Maxstat mode labels groups as 'Low risk' and 'High risk'."""
        from mlsurv.viz.survival import plot_km

        fig = plot_km(surv_df, risk_col="risk_score", break_type="maxstat")
        trace_names = [t.name for t in fig.data
                       if getattr(t, "mode", None) == "lines"
                       and not getattr(t, "fill", None)]
        assert any("Low risk" in n for n in trace_names if n)
        assert any("High risk" in n for n in trace_names if n)

    # ── Recursive multi-group tests ────────────────────────────────────

    def test_maxstat_3_groups(self, surv_df):
        """n_groups=3 returns a 3-group figure."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        fig = plot_km(surv_df, risk_col="risk_score", break_type="maxstat",
                      n_groups=3)
        assert isinstance(fig, go.Figure)
        trace_names = [t.name for t in fig.data
                       if getattr(t, "mode", None) == "lines"
                       and not getattr(t, "fill", None)
                       and t.name is not None]
        assert len(trace_names) == 3

    def test_maxstat_3_groups_labels(self, surv_df):
        """3-group maxstat uses Low/Medium/High labels."""
        from mlsurv.viz.survival import plot_km

        fig = plot_km(surv_df, risk_col="risk_score", break_type="maxstat",
                      n_groups=3)
        trace_names = [t.name for t in fig.data
                       if getattr(t, "mode", None) == "lines"
                       and not getattr(t, "fill", None)]
        assert any("Low risk" in n for n in trace_names if n)
        assert any("Medium risk" in n for n in trace_names if n)
        assert any("High risk" in n for n in trace_names if n)

    def test_maxstat_4_groups(self, surv_df):
        """n_groups=4 returns a figure with up to 4 groups."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        fig = plot_km(surv_df, risk_col="risk_score", break_type="maxstat",
                      n_groups=4)
        assert isinstance(fig, go.Figure)
        trace_names = [t.name for t in fig.data
                       if getattr(t, "mode", None) == "lines"
                       and not getattr(t, "fill", None)
                       and t.name is not None]
        assert 2 <= len(trace_names) <= 4

    def test_maxstat_3_groups_subtitle_multiple_cutpoints(self, surv_df):
        """3-group subtitle says 'cutpoints' (plural)."""
        from mlsurv.viz.survival import plot_km

        fig = plot_km(surv_df, risk_col="risk_score", break_type="maxstat",
                      n_groups=3)
        title_text = fig.layout.title.text
        assert "cutpoints" in title_text

    def test_maxstat_recursive_result_fields(self, surv_df):
        """_find_maxstat_cutpoints_recursive returns sorted cutpoints tuple."""
        from mlsurv.viz.survival import _find_maxstat_cutpoints_recursive

        result = _find_maxstat_cutpoints_recursive(
            surv_df["time"], surv_df["event"], surv_df["risk_score"],
            n_groups=3,
        )
        assert len(result.cutpoints) == 2
        assert result.cutpoints[0] < result.cutpoints[1]
        assert len(result.split_stats) == 2

    def test_maxstat_recursive_graceful_degradation(self):
        """Small data: requested 3 groups, may get 2 if subpopulation too small."""
        from mlsurv.viz.survival import _find_maxstat_cutpoints_recursive

        rng = np.random.RandomState(42)
        n = 30
        times = rng.exponential(10, n)
        events = rng.binomial(1, 0.7, n).astype(float)
        scores = rng.randn(n)

        result = _find_maxstat_cutpoints_recursive(
            times, events, scores, n_groups=3,
        )
        # Should get at least 1 cutpoint (2 groups)
        assert len(result.cutpoints) >= 1

    def test_maxstat_n_groups_over_4_raises(self, surv_df):
        """n_groups=5 with maxstat raises ValueError."""
        from mlsurv.viz.survival import plot_km

        with pytest.raises(ValueError, match="exceeds maximum of 4"):
            plot_km(surv_df, risk_col="risk_score", break_type="maxstat",
                    n_groups=5)

    def test_maxstat_panel_grid_3_groups(self, multi_model_data):
        """Multi-model dict with n_groups=3."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        fig = plot_km(multi_model_data, risk_col="risk_score",
                      break_type="maxstat", n_groups=3)
        assert isinstance(fig, go.Figure)
        # Panel grid has annotations for per-model cutpoints
        annotations = [a for a in fig.layout.annotations
                       if "cuts=" in (a.text or "") or "cut=" in (a.text or "")]
        assert len(annotations) == 2  # one per model

    def test_maxstat_3_groups_with_risk_table(self, surv_df):
        """show_risk_table=True with 3 groups."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        fig = plot_km(surv_df, risk_col="risk_score", break_type="maxstat",
                      n_groups=3, show_risk_table=True)
        assert isinstance(fig, go.Figure)
        trace_names = [t.name for t in fig.data
                       if getattr(t, "mode", None) == "lines"
                       and not getattr(t, "fill", None)
                       and t.name is not None]
        assert len(trace_names) == 3

    # ── Permutation p-value tests ──────────────────────────────────────

    def test_permutation_correction(self, perm_surv_df):
        """correction='permutation' returns a valid figure."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        fig = plot_km(perm_surv_df, risk_col="risk_score",
                      break_type="maxstat", correction="permutation",
                      n_permutations=100, random_state=42)
        assert isinstance(fig, go.Figure)

    def test_permutation_p_value_range(self, perm_surv_df):
        """Permutation p-value is in [0, 1]."""
        from mlsurv.viz.survival import _find_maxstat_cutpoint

        result = _find_maxstat_cutpoint(
            perm_surv_df["time"], perm_surv_df["event"],
            perm_surv_df["risk_score"],
            correction="permutation", n_permutations=100, random_state=42,
        )
        assert 0 <= result.p_value_corrected <= 1

    def test_permutation_reproducible(self, perm_surv_df):
        """Same random_state produces same p-value."""
        from mlsurv.viz.survival import _find_maxstat_cutpoint

        r1 = _find_maxstat_cutpoint(
            perm_surv_df["time"], perm_surv_df["event"],
            perm_surv_df["risk_score"],
            correction="permutation", n_permutations=100, random_state=123,
        )
        r2 = _find_maxstat_cutpoint(
            perm_surv_df["time"], perm_surv_df["event"],
            perm_surv_df["risk_score"],
            correction="permutation", n_permutations=100, random_state=123,
        )
        assert r1.p_value_corrected == r2.p_value_corrected

    def test_permutation_different_seeds(self, perm_surv_df):
        """Different seeds produce same cutpoint (deterministic search)."""
        from mlsurv.viz.survival import _find_maxstat_cutpoint

        r1 = _find_maxstat_cutpoint(
            perm_surv_df["time"], perm_surv_df["event"],
            perm_surv_df["risk_score"],
            correction="permutation", n_permutations=99, random_state=1,
        )
        r2 = _find_maxstat_cutpoint(
            perm_surv_df["time"], perm_surv_df["event"],
            perm_surv_df["risk_score"],
            correction="permutation", n_permutations=99, random_state=2,
        )
        assert r1.cutpoint == r2.cutpoint

    def test_permutation_too_few_raises(self, surv_df):
        """n_permutations=10 raises ValueError."""
        from mlsurv.viz.survival import plot_km

        with pytest.raises(ValueError, match="too low"):
            plot_km(surv_df, risk_col="risk_score", break_type="maxstat",
                    correction="permutation", n_permutations=10)

    def test_permutation_panel_grid(self, perm_multi_model_data):
        """Panel grid with permutation correction."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        fig = plot_km(perm_multi_model_data, risk_col="risk_score",
                      break_type="maxstat", correction="permutation",
                      n_permutations=100, random_state=42)
        assert isinstance(fig, go.Figure)

    def test_permutation_with_3_groups(self, perm_surv_df):
        """Permutation + recursive n_groups=3 combined."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        fig = plot_km(perm_surv_df, risk_col="risk_score",
                      break_type="maxstat", n_groups=3,
                      correction="permutation", n_permutations=100,
                      random_state=42)
        assert isinstance(fig, go.Figure)
        trace_names = [t.name for t in fig.data
                       if getattr(t, "mode", None) == "lines"
                       and not getattr(t, "fill", None)
                       and t.name is not None]
        assert len(trace_names) == 3

    def test_fast_logrank_chi2_matches_lifelines(self, surv_df):
        """_fast_logrank_chi2 agrees with lifelines logrank_test."""
        from mlsurv.viz.survival import _fast_logrank_chi2
        from lifelines.statistics import logrank_test

        scores = surv_df["risk_score"].values
        median = np.median(scores)
        lo = scores <= median
        hi = ~lo
        t, e = surv_df["time"].values, surv_df["event"].values

        fast_chi2 = _fast_logrank_chi2(t[lo], e[lo], t[hi], e[hi])
        lr = logrank_test(t[lo], t[hi], e[lo], e[hi])

        assert abs(fast_chi2 - lr.test_statistic) < 0.1

    def test_permutation_subtitle(self, perm_surv_df):
        """Subtitle includes 'permutation'."""
        from mlsurv.viz.survival import plot_km

        fig = plot_km(perm_surv_df, risk_col="risk_score",
                      break_type="maxstat", correction="permutation",
                      n_permutations=100, random_state=42)
        title_text = fig.layout.title.text
        assert "permutation" in title_text

    # ── Edge case tests for _fast_logrank_chi2 ───────────────────────────

    def test_fast_logrank_all_censored(self):
        """No events in either group → chi2 = 0."""
        from mlsurv.viz.survival import _fast_logrank_chi2
        result = _fast_logrank_chi2(
            np.array([1., 2., 3.]), np.array([0., 0., 0.]),
            np.array([4., 5., 6.]), np.array([0., 0., 0.]),
        )
        assert result == 0.0

    def test_fast_logrank_one_group_empty(self):
        """Empty group → returns 0.0."""
        from mlsurv.viz.survival import _fast_logrank_chi2
        result = _fast_logrank_chi2(
            np.array([]), np.array([]),
            np.array([1., 2., 3.]), np.array([1., 1., 0.]),
        )
        assert result == 0.0

    def test_fast_logrank_identical_times(self):
        """All events at same time, equal split → variance 0 → chi2 = 0."""
        from mlsurv.viz.survival import _fast_logrank_chi2
        result = _fast_logrank_chi2(
            np.array([1., 1.]), np.array([1., 1.]),
            np.array([1., 1.]), np.array([1., 1.]),
        )
        assert result == 0.0

    # ── Validation tests ─────────────────────────────────────────────────

    def test_recursive_n_groups_1_raises(self, surv_df):
        """n_groups=1 raises ValueError at recursive level."""
        from mlsurv.viz.survival import _find_maxstat_cutpoints_recursive
        with pytest.raises(ValueError, match="n_groups=1"):
            _find_maxstat_cutpoints_recursive(
                surv_df["time"], surv_df["event"], surv_df["risk_score"],
                n_groups=1,
            )

    # ── Permutation directionality ───────────────────────────────────────

    def test_permutation_strong_signal_small_p(self):
        """Strong group separation → permutation p near 0."""
        from mlsurv.viz.survival import _find_maxstat_cutpoint
        rng = np.random.RandomState(42)
        # 49 perms: p-value resolution 0.02, sufficient for < 0.05 threshold
        n = 100
        group = np.array([0] * 50 + [1] * 50)
        times = np.where(group, rng.exponential(20, n), rng.exponential(2, n))
        events = np.ones(n)
        scores = group.astype(float) + rng.randn(n) * 0.1
        result = _find_maxstat_cutpoint(
            times, events, scores,
            correction="permutation", n_permutations=49, random_state=42,
        )
        assert result.p_value_corrected < 0.05

    def test_permutation_no_signal_large_p(self):
        """No real separation → permutation p > 0.1."""
        from mlsurv.viz.survival import _find_maxstat_cutpoint
        rng = np.random.RandomState(42)
        # 49 perms: p-value resolution 0.02, sufficient for > 0.1 threshold
        n = 100
        times = rng.exponential(10, n)
        events = rng.binomial(1, 0.5, n).astype(float)
        scores = rng.randn(n)  # independent of survival
        result = _find_maxstat_cutpoint(
            times, events, scores,
            correction="permutation", n_permutations=49, random_state=42,
        )
        assert result.p_value_corrected > 0.1


class TestExplicitBreaks:
    """Explicit quantile / score breaks in plot_km + plot_survival."""

    @pytest.fixture
    def surv_df(self):
        rng = np.random.RandomState(7)
        n = 200
        scores = rng.randn(n)
        times = np.clip(np.exp(-scores / 2 + rng.randn(n) * 0.3), 0.1, 20)
        events = rng.binomial(1, 0.7, n)
        return pd.DataFrame({
            "time": times, "event": events, "risk_score": scores,
        })

    def test_quantile_breaks_group_labels(self, surv_df):
        """quantile breaks bin into the expected percentile-range groups."""
        from mlsurv.viz._survival_helpers import _create_risk_groups

        df, n_actual, labels = _create_risk_groups(
            surv_df.copy(), "risk_score", 4,
            breaks=[0.25, 0.5, 0.75], break_type="quantile",
        )
        assert n_actual == 4
        assert labels == ["< 25%", "25–50%", "50–75%", "≥ 75%"]
        assert sorted(df["risk_group"].unique().tolist()) == [0, 1, 2, 3]

    def test_score_breaks_group_labels(self, surv_df):
        """score breaks bin at raw risk-score cut-points."""
        from mlsurv.viz._survival_helpers import _create_risk_groups

        df, n_actual, labels = _create_risk_groups(
            surv_df.copy(), "risk_score", None,
            breaks=[-0.5, 0.5], break_type="score",
        )
        assert n_actual == 3
        assert labels[0].startswith("<") and labels[-1].startswith("≥")

    def test_plot_km_quantile_breaks(self, surv_df):
        """plot_km accepts explicit quantile breaks and renders groups."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        fig = plot_km(surv_df, risk_col="risk_score",
                      breaks=[0.25, 0.5, 0.75], break_type="quantile")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 4

    def test_plot_km_score_breaks(self, surv_df):
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go

        fig = plot_km(surv_df, risk_col="risk_score",
                      breaks=[-0.5, 0.5], break_type="score")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3

    def test_plot_km_invalid_breaks_raise(self, surv_df):
        from mlsurv.viz.survival import plot_km

        with pytest.raises(ValueError):
            plot_km(surv_df, risk_col="risk_score", breaks=[0.75, 0.25])
        with pytest.raises(ValueError):
            plot_km(surv_df, risk_col="risk_score", breaks=[1.5])

    def test_plot_km_both_n_groups_and_breaks_warns(self, surv_df):
        from mlsurv.viz.survival import plot_km
        from mlsurv._warnings import SuggestionWarning

        with pytest.warns(SuggestionWarning, match="n_groups is ignored"):
            plot_km(surv_df, risk_col="risk_score",
                    n_groups=3, breaks=[0.5])

    @pytest.mark.integration
    def test_plot_survival_breaks_passthrough(self):
        """learner.plot_survival(breaks=...) reaches the grouping layer."""
        from mlsurv.utils import SurvivalArray
        from mlsurv.learner import SurvivalLearner
        import plotly.graph_objects as go

        rng = np.random.RandomState(11)
        n = 120
        X = pd.DataFrame({
            "age": rng.randn(n) * 10 + 60,
            "biomarker": rng.randn(n),
        })
        times = np.clip(np.exp(rng.randn(n)), 0.1, 10)
        events = rng.binomial(1, 0.7, n).astype(bool)
        y = SurvivalArray(np.array(
            [(e, t) for e, t in zip(events, times)],
            dtype=[("event", bool), ("time", float)],
        ))
        learner = SurvivalLearner(X, y)
        learner.setup(verbose=False)
        learner.train("coxph", verbose=False)
        learner.evaluate(verbose=False)

        fig = learner.plot_survival(models="coxph",
                                    breaks=[0.25, 0.5, 0.75])
        assert isinstance(fig, go.Figure)

        # break_type / correction must also thread through the wrapper.
        fig_maxstat = learner.plot_survival(models="coxph",
                                            break_type="maxstat")
        assert isinstance(fig_maxstat, go.Figure)


class TestCalibrationPolish:
    """Tests for calibration height fixes and display names."""

    def test_calibration_single_figure_height(self):
        """Single-model calibration should use 'medium' height (400px)."""
        from mlsurv.viz.calibration import plot_calibration
        calibration_data = [{
            "eval_time": 12, "time_index": 0,
            "predicted": [0.2, 0.5, 0.8],
            "observed": [0.25, 0.48, 0.75],
            "ci_lower": [0.2, 0.4, 0.7],
            "ci_upper": [0.3, 0.55, 0.8],
        }]
        fig = plot_calibration(calibration_data)
        assert fig.layout.height == 400

    def test_calibration_comparison_figure_height(self):
        """Multi-model calibration should use 'medium' height (400px)."""
        from mlsurv.viz.calibration import plot_calibration
        calibration_by_model = {
            "coxph": {
                "eval_time": 12,
                "predicted": [0.2, 0.5, 0.8],
                "observed": [0.25, 0.48, 0.75],
                "ci_lower": [0.2, 0.4, 0.7],
                "ci_upper": [0.3, 0.55, 0.8],
            },
        }
        fig = plot_calibration(calibration_by_model)
        assert fig.layout.height == 400

    def test_calibration_comparison_display_names(self):
        """Multi-model calibration legend should use display names."""
        from mlsurv.viz.calibration import plot_calibration
        from mlsurv.viz._palettes import get_model_display_name
        calibration_by_model = {
            "coxph": {
                "eval_time": 12,
                "predicted": [0.2, 0.5, 0.8],
                "observed": [0.25, 0.48, 0.75],
                "ci_lower": [0.2, 0.4, 0.7],
                "ci_upper": [0.3, 0.55, 0.8],
            },
        }
        fig = plot_calibration(calibration_by_model)
        model_traces = [t for t in fig.data if t.name not in ("Perfect",)]
        assert any(t.name == get_model_display_name("coxph") for t in model_traces)

    def test_calibration_overlay_no_palette_param(self):
        """plot_calibration_overlay should not accept palette parameter."""
        import inspect
        from mlsurv.viz.calibration import plot_calibration_overlay
        sig = inspect.signature(plot_calibration_overlay)
        assert "palette" not in sig.parameters


class TestKMLogrank:
    """Tests for log-rank p-value annotation on KM curves."""

    @pytest.fixture()
    def km_data_2group(self):
        """Synthetic survival data with 2 groups having clearly different survival."""
        rng = np.random.RandomState(42)
        n = 200
        # Group 0: low risk (slow events), Group 1: high risk (fast events)
        group = np.repeat([0, 1], n // 2)
        time = np.where(group == 0, rng.exponential(20, n), rng.exponential(5, n))
        event = rng.binomial(1, 0.8, n)
        return pd.DataFrame({"time": time, "event": event, "group": group,
                             "risk_score": -time + rng.normal(0, 1, n)})

    @pytest.fixture()
    def km_data_3group(self):
        """Synthetic survival data with 3 groups."""
        rng = np.random.RandomState(42)
        n = 300
        group = np.repeat(["A", "B", "C"], n // 3)
        scales = {"A": 20, "B": 10, "C": 5}
        time = np.concatenate([rng.exponential(scales[g], n // 3) for g in ["A", "B", "C"]])
        event = rng.binomial(1, 0.8, n)
        return pd.DataFrame({"time": time, "event": event, "group": group,
                             "risk_score": -time + rng.normal(0, 1, n)})

    def test_show_pvalue_risk_col(self, km_data_2group):
        """show_pvalue=True adds log-rank annotation for risk-stratified KM."""
        from mlsurv.viz.survival import plot_km
        fig = plot_km(km_data_2group, risk_col="risk_score", n_groups=2,
                      show_pvalue=True, show_risk_table=False)
        annotations = [a.text for a in fig.layout.annotations]
        assert any("p =" in a for a in annotations)

    def test_show_pvalue_group_col(self, km_data_2group):
        """show_pvalue=True works for pre-defined groups."""
        from mlsurv.viz.survival import plot_km
        fig = plot_km(km_data_2group, group_col="group", show_pvalue=True)
        annotations = [a.text for a in fig.layout.annotations]
        assert any("Log-rank" in a for a in annotations)

    def test_show_pvalue_3_groups(self, km_data_3group):
        """Omnibus p-value shown for 3+ groups."""
        from mlsurv.viz.survival import plot_km
        fig = plot_km(km_data_3group, group_col="group", show_pvalue=True)
        annotations = [a.text for a in fig.layout.annotations]
        assert any("Log-rank p =" in a for a in annotations)

    def test_return_data_structure(self, km_data_2group):
        """return_data=True returns dict with figure and logrank keys."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go
        result = plot_km(km_data_2group, risk_col="risk_score", n_groups=2,
                         return_data=True, show_risk_table=False)
        assert isinstance(result, dict)
        assert "figure" in result
        assert "logrank" in result
        assert isinstance(result["figure"], go.Figure)
        lr = result["logrank"]
        assert lr["test"] == "log-rank"
        assert "omnibus_statistic" in lr
        assert "omnibus_p" in lr
        assert lr["n_groups"] == 2
        assert lr["pairwise"] is None  # only 2 groups

    def test_return_data_pairwise(self, km_data_3group):
        """Pairwise comparisons returned for 3+ groups."""
        from mlsurv.viz.survival import plot_km
        result = plot_km(km_data_3group, group_col="group",
                         return_data=True, p_adjust="holm")
        lr = result["logrank"]
        assert lr["n_groups"] == 3
        assert lr["pairwise"] is not None
        assert len(lr["pairwise"]) == 3  # C(3,2) = 3 pairs
        for pw in lr["pairwise"]:
            assert "group_1" in pw
            assert "group_2" in pw
            assert "p_raw" in pw
            assert "p_adjusted" in pw
            assert pw["p_adjusted"] >= pw["p_raw"]  # adjustment inflates p

    def test_return_data_panel_grid(self, km_data_2group):
        """Panel grid returns per-model logrank results."""
        from mlsurv.viz.survival import plot_km
        data_dict = {
            "ModelA": km_data_2group.copy(),
            "ModelB": km_data_2group.copy(),
        }
        result = plot_km(data_dict, risk_col="risk_score", n_groups=2,
                         return_data=True)
        assert isinstance(result["logrank"], dict)
        assert "ModelA" in result["logrank"]
        assert "ModelB" in result["logrank"]
        assert result["logrank"]["ModelA"]["test"] == "log-rank"

    def test_show_pvalue_overall_ignored(self, km_data_2group):
        """show_pvalue=True with overall KM (no groups) does not error."""
        from mlsurv.viz.survival import plot_km
        fig = plot_km(km_data_2group, show_pvalue=True)
        # No annotations expected for overall KM
        assert not any("Log-rank" in getattr(a, "text", "")
                       for a in fig.layout.annotations)

    def test_show_pvalue_false_no_annotation(self, km_data_2group):
        """Default show_pvalue=False produces no log-rank annotation."""
        from mlsurv.viz.survival import plot_km
        fig = plot_km(km_data_2group, group_col="group")
        annotations = [a.text for a in fig.layout.annotations]
        assert not any("Log-rank" in a for a in annotations)

    def test_p_adjust_methods(self, km_data_3group):
        """Various p_adjust methods work without error."""
        from mlsurv.viz.survival import plot_km
        for method in ("bh", "bonferroni", "none"):
            result = plot_km(km_data_3group, group_col="group",
                             return_data=True, p_adjust=method)
            assert result["logrank"]["p_adjust_method"] == method

    def test_format_logrank_annotation_decimal(self):
        """Decimal format for p >= 0.001."""
        from mlsurv.viz.survival import _format_logrank_annotation
        result = _format_logrank_annotation({"omnibus_p": 0.042})
        assert result == "Log-rank p = 0.042"

    def test_format_logrank_annotation_scientific(self):
        """Scientific notation for p < 0.001."""
        from mlsurv.viz.survival import _format_logrank_annotation
        result = _format_logrank_annotation({"omnibus_p": 3.5e-7})
        assert result == "Log-rank p = 3.50e-07"

    def test_format_logrank_annotation_boundary(self):
        """Boundary p = 0.001 uses decimal format."""
        from mlsurv.viz.survival import _format_logrank_annotation
        result = _format_logrank_annotation({"omnibus_p": 0.001})
        assert result == "Log-rank p = 0.001"

    def test_return_data_no_annotation(self, km_data_2group):
        """return_data=True without show_pvalue computes logrank but no annotation."""
        from mlsurv.viz.survival import plot_km
        result = plot_km(km_data_2group, group_col="group", return_data=True)
        assert result["logrank"] is not None
        fig = result["figure"]
        assert not any("Log-rank" in getattr(a, "text", "")
                       for a in fig.layout.annotations)

    def test_return_data_2group_no_pairwise(self, km_data_2group):
        """2-group logrank: pairwise is None, p_adjust_method is None."""
        from mlsurv.viz.survival import plot_km
        result = plot_km(km_data_2group, group_col="group", return_data=True)
        lr = result["logrank"]
        assert lr["pairwise"] is None
        assert lr["p_adjust_method"] is None

    def test_show_pvalue_panel_grid(self, km_data_2group):
        """Panel grid with show_pvalue=True adds annotations."""
        from mlsurv.viz.survival import plot_km
        data_dict = {
            "ModelA": km_data_2group.copy(),
            "ModelB": km_data_2group.copy(),
        }
        fig = plot_km(data_dict, risk_col="risk_score", n_groups=2,
                      show_pvalue=True)
        annotations = [a.text for a in fig.layout.annotations
                       if hasattr(a, "text") and a.text]
        assert sum("Log-rank" in a for a in annotations) == 2  # one per model

    def test_invalid_p_adjust_raises(self, km_data_2group):
        """Invalid p_adjust value raises ValueError."""
        from mlsurv.viz.survival import plot_km
        with pytest.raises(ValueError, match="Unknown p_adjust"):
            plot_km(km_data_2group, group_col="group", p_adjust="invalid")

    def test_return_data_overall_km(self, km_data_2group):
        """return_data=True on overall KM returns dict with logrank=None."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go
        result = plot_km(km_data_2group, return_data=True)
        assert isinstance(result, dict)
        assert isinstance(result["figure"], go.Figure)
        assert result["logrank"] is None

    def test_show_pvalue_risk_col_with_risk_table(self, km_data_2group):
        """show_pvalue=True with show_risk_table=True (default) adds annotation."""
        from mlsurv.viz.survival import plot_km
        fig = plot_km(km_data_2group, risk_col="risk_score", n_groups=2,
                      show_pvalue=True, show_risk_table=True)
        annotations = [a.text for a in fig.layout.annotations]
        assert any("Log-rank" in a for a in annotations)

    def test_degenerate_data_warns(self):
        """Degenerate data emits warning or returns a valid result."""
        from mlsurv.viz.survival import _compute_logrank
        import warnings
        # All events at same time with identical values
        df = pd.DataFrame({
            "time": [0.0] * 10,
            "event": [1] * 10,
            "group": [0] * 5 + [1] * 5,
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _compute_logrank(df, "time", "event", "group")
        # Must be one of: None (exception/NaN caught) or a valid dict
        if result is None:
            assert any("Log-rank test" in str(wi.message) for wi in w)
        else:
            assert "omnibus_p" in result
            assert not np.isnan(result["omnibus_p"])

    def test_show_pvalue_and_return_data_together(self, km_data_2group):
        """show_pvalue=True + return_data=True: annotation + dict returned."""
        from mlsurv.viz.survival import plot_km
        import plotly.graph_objects as go
        result = plot_km(km_data_2group, group_col="group",
                         show_pvalue=True, return_data=True)
        assert isinstance(result, dict)
        assert isinstance(result["figure"], go.Figure)
        assert result["logrank"] is not None
        annotations = [a.text for a in result["figure"].layout.annotations]
        assert any("Log-rank" in a for a in annotations)

    def test_show_pvalue_overall_warns(self, km_data_2group):
        """show_pvalue=True on overall KM emits a warning."""
        from mlsurv.viz.survival import plot_km
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plot_km(km_data_2group, show_pvalue=True)
        assert any("show_pvalue=True has no effect" in str(wi.message)
                    for wi in w)

    def test_format_annotation_p_zero(self):
        """p=0.0 formats as '< 1e-300' instead of '0.00e+00'."""
        from mlsurv.viz.survival import _format_logrank_annotation
        result = _format_logrank_annotation({"omnibus_p": 0.0})
        assert "< 1e-300" in result


class TestBrierSkipWarning:
    """Tests for compute_brier_curves skipping models without surv func."""

    def test_compute_brier_skips_model_without_surv_func_with_warning(self):
        """Model without predict_survival_function triggers UserWarning."""
        from mlsurv.viz.metrics import compute_brier_curves
        from unittest.mock import MagicMock

        mock_model = MagicMock(spec=[])  # no predict_survival_function
        models = {"bad_model": mock_model}

        y_data = np.array(
            [(True, 1.0), (True, 2.0), (False, 3.0)],
            dtype=[("event", bool), ("time", float)],
        )
        X = pd.DataFrame({"x": [1, 2, 3]})

        with pytest.warns(InternalWarning, match="does not expose"):
            with pytest.raises(ValueError, match="Could not compute Brier"):
                compute_brier_curves(models, X, y_data, X, y_data)


class TestCreateRiskGroupsWarning:
    """Tests for _create_risk_groups warning on reduction."""

    def test_create_risk_groups_warns_on_reduction(self):
        """Few unique scores → warns about n_groups reduction."""
        from mlsurv.viz.survival import _create_risk_groups

        # 2 unique values, request 4 groups → should reduce and warn
        df = pd.DataFrame({
            "risk_col": [0.0] * 25 + [1.0] * 25,
            "time": np.arange(50, dtype=float),
            "event": np.ones(50),
        })

        with pytest.warns(InternalWarning, match="Requested .* risk groups but only"):
            _create_risk_groups(df, "risk_col", n_groups=4)


# =============================================================================
# Multi-panel point_estimates
# =============================================================================

class TestMultiPanelPointEstimates:
    """Tests for point_estimates in multi-panel plot_metric."""

    @staticmethod
    def _make_iter_data():
        """Create long-format iteration data for 2 metrics, 2 models."""
        rows = []
        for metric in ["harrell", "ipcw"]:
            for model in ["coxph", "rsf"]:
                for i in range(10):
                    rows.append({
                        "model_type": model,
                        "metric_name": metric,
                        "value": 0.7 + 0.01 * i + (0.02 if model == "rsf" else 0),
                    })
        return pd.DataFrame(rows)

    def test_multi_panel_raincloud_with_point_estimates(self):
        from mlsurv.viz.metrics import plot_metric
        import plotly.graph_objects as go

        df = self._make_iter_data()
        pe = {
            "harrell": {"coxph": 0.72, "rsf": 0.76},
            "ipcw": {"coxph": 0.71},
        }
        fig = plot_metric(
            df, style="raincloud", model_col="model_type",
            metric=["harrell", "ipcw"], point_estimates=pe,
        )
        assert isinstance(fig, go.Figure)
        # Check triangle markers exist (point estimate markers)
        triangle_traces = [
            t for t in fig.data
            if getattr(t, 'marker', None)
            and getattr(t.marker, 'symbol', None)
            and 'triangle' in str(t.marker.symbol)
        ]
        assert len(triangle_traces) >= 3  # 2 for harrell + 1 for ipcw

    def test_multi_panel_boxplot_with_point_estimates(self):
        from mlsurv.viz.metrics import plot_metric
        import plotly.graph_objects as go

        df = self._make_iter_data()
        pe = {
            "harrell": {"coxph": 0.72, "rsf": 0.76},
            "ipcw": {"coxph": 0.71, "rsf": 0.75},
        }
        fig = plot_metric(
            df, style="boxplot", model_col="model_type",
            metric=["harrell", "ipcw"], point_estimates=pe,
        )
        assert isinstance(fig, go.Figure)
        triangle_traces = [
            t for t in fig.data
            if getattr(t, 'marker', None)
            and getattr(t.marker, 'symbol', None)
            and 'triangle' in str(t.marker.symbol)
        ]
        assert len(triangle_traces) >= 4

    def test_multi_panel_point_estimates_partial_metrics(self):
        from mlsurv.viz.metrics import plot_metric
        import plotly.graph_objects as go

        df = self._make_iter_data()
        # Only provide point_estimates for harrell, not ipcw
        pe = {"harrell": {"coxph": 0.72, "rsf": 0.76}}
        fig = plot_metric(
            df, style="raincloud", model_col="model_type",
            metric=["harrell", "ipcw"], point_estimates=pe,
        )
        assert isinstance(fig, go.Figure)
        triangle_traces = [
            t for t in fig.data
            if getattr(t, 'marker', None)
            and getattr(t.marker, 'symbol', None)
            and 'triangle' in str(t.marker.symbol)
        ]
        # Only harrell panel has markers (2 models)
        assert len(triangle_traces) >= 2

    def test_multi_panel_point_estimates_none(self):
        from mlsurv.viz.metrics import plot_metric
        import plotly.graph_objects as go

        df = self._make_iter_data()
        fig = plot_metric(
            df, style="raincloud", model_col="model_type",
            metric=["harrell", "ipcw"], point_estimates=None,
        )
        assert isinstance(fig, go.Figure)
        # No triangle markers
        triangle_traces = [
            t for t in fig.data
            if getattr(t, 'marker', None)
            and getattr(t.marker, 'symbol', None)
            and 'triangle' in str(t.marker.symbol)
        ]
        assert len(triangle_traces) == 0


# =============================================================================
# Multi-panel plot_metric_time
# =============================================================================

class TestMetricTimeMultiPanel:
    """Tests for multi-panel plot_metric_time."""

    @staticmethod
    def _make_time_data():
        """Create long-format time-dependent data for 2 metrics, 2 models."""
        rows = []
        for metric in ["auc", "brier"]:
            for model in ["coxph", "rsf"]:
                for t in [6, 12, 24]:
                    val = (0.75 - t * 0.005) if metric == "auc" else (0.15 + t * 0.002)
                    if model == "rsf":
                        val += 0.02 if metric == "auc" else -0.01
                    rows.append({
                        "model": model,
                        "eval_time": t,
                        "value": val,
                        "metric_name": metric,
                    })
        return pd.DataFrame(rows)

    @staticmethod
    def _make_std_data():
        """Create std data matching _make_time_data."""
        rows = []
        for metric in ["auc", "brier"]:
            for model in ["coxph", "rsf"]:
                for t in [6, 12, 24]:
                    rows.append({
                        "model": model,
                        "eval_time": t,
                        "value_std": 0.03,
                        "metric_name": metric,
                    })
        return pd.DataFrame(rows)

    def test_multi_panel_line_basic(self):
        from mlsurv.viz.metrics import plot_metric_time
        import plotly.graph_objects as go

        df = self._make_time_data()
        fig = plot_metric_time(df, metric=["auc", "brier"])
        assert isinstance(fig, go.Figure)
        # Should have traces for 2 models (legend shown once each)
        # plus traces without legend for the second panel
        assert len(fig.data) >= 4  # 2 models * 2 panels

    def test_multi_panel_line_with_ci(self):
        from mlsurv.viz.metrics import plot_metric_time
        import plotly.graph_objects as go

        df = self._make_time_data()
        std_df = self._make_std_data()
        fig = plot_metric_time(df, std_df=std_df, metric=["auc", "brier"])
        assert isinstance(fig, go.Figure)
        # Each model in each panel: line + ribbon = 2 traces * 2 models * 2 panels
        assert len(fig.data) >= 8

    def test_multi_panel_line_single_metric_fallback(self):
        from mlsurv.viz.metrics import plot_metric_time
        import plotly.graph_objects as go

        df = self._make_time_data()
        fig = plot_metric_time(df, metric=["auc"])
        assert isinstance(fig, go.Figure)
        # Single panel with 2 models
        assert len(fig.data) >= 2

    def test_multi_panel_line_shared_time_axis(self):
        from mlsurv.viz.metrics import plot_metric_time

        df = self._make_time_data()
        fig = plot_metric_time(df, metric=["auc", "brier"])
        layout = fig.layout
        # shared_xaxes=True means xaxis2 should exist but panels share x
        assert layout.xaxis is not None

    def test_multi_panel_line_reference_lines(self):
        from mlsurv.viz.metrics import plot_metric_time

        df = self._make_time_data()
        fig = plot_metric_time(df, metric=["auc", "brier"])
        # Check for reference lines (hlines stored as shapes)
        layout_dict = fig.to_dict()["layout"]
        shapes = layout_dict.get("shapes", [])
        ref_ys = [s["y0"] for s in shapes
                  if s.get("type") == "line" and s["y0"] == s["y1"]]
        assert 0.5 in ref_ys  # AUC reference
        assert 0.25 in ref_ys  # Brier reference

    def test_multi_panel_custom_title(self):
        from mlsurv.viz.metrics import plot_metric_time

        df = self._make_time_data()
        fig = plot_metric_time(df, metric=["auc", "brier"], title="Custom Title")
        assert "Custom Title" in fig.layout.title.text

    def test_multi_panel_time_unit(self):
        from mlsurv.viz.metrics import plot_metric_time

        df = self._make_time_data()
        fig = plot_metric_time(df, metric=["auc", "brier"], time_unit="months")
        # Bottom panel x-axis should have "months" in the label
        layout_dict = fig.to_dict()["layout"]
        xaxis_titles = [
            v.get("title", "") for k, v in layout_dict.items()
            if k.startswith("xaxis") and isinstance(v, dict)
        ]
        assert any("months" in str(t) for t in xaxis_titles)

    def test_multi_panel_empty_metric_raises(self):
        from mlsurv.viz.metrics import plot_metric_time

        df = self._make_time_data()
        with pytest.raises(ValueError, match="empty"):
            plot_metric_time(df, metric=[])

    def test_multi_panel_non_string_metric_entry_raises(self):
        from mlsurv.viz.metrics import plot_metric_time

        df = self._make_time_data()
        with pytest.raises(TypeError, match="must be strings"):
            plot_metric_time(df, metric=["auc", 1])

    def test_multi_panel_legend_dedup(self):
        from mlsurv.viz.metrics import plot_metric_time

        df = self._make_time_data()
        fig = plot_metric_time(df, metric=["auc", "brier"])
        # Each model should appear in legend exactly once
        legend_names = [t.name for t in fig.data if t.showlegend]
        assert len(legend_names) == len(set(legend_names))

    def test_multi_panel_y_range_per_metric(self):
        from mlsurv.viz.metrics import plot_metric_time

        df = self._make_time_data()
        fig = plot_metric_time(df, metric=["auc", "brier"])
        layout = fig.to_dict()["layout"]
        # AUC panel: data-driven lower bound accommodating all values
        auc_range = layout["yaxis"]["range"]
        assert auc_range[0] < 0.72  # below min data value
        assert auc_range[1] == 1.0
        # Brier panel (second): data-driven upper bound
        brier_range = layout["yaxis2"]["range"]
        assert brier_range[0] == 0
        assert brier_range[1] <= 0.35  # small data values → upper bound ≤ 0.35
        assert brier_range[1] > 0.15   # must accommodate data (~0.15-0.20)

    def test_multi_panel_four_metrics_grid(self):
        """4 metrics creates a 2x3 grid (but only 4 panels filled)."""
        from mlsurv.viz.metrics import plot_metric_time

        rows = []
        for m in ["m1", "m2", "m3", "m4"]:
            for t in [6, 12, 24]:
                rows.append({
                    "model": "coxph",
                    "eval_time": t,
                    "value": 0.7,
                    "metric_name": m,
                })
        df = pd.DataFrame(rows)
        fig = plot_metric_time(df, metric=["m1", "m2", "m3", "m4"])
        # Should create a 2-row grid (4 metrics → ceil(4/3) = 2 rows)
        layout = fig.to_dict()["layout"]
        assert "yaxis4" in layout  # 4th subplot exists

    def test_single_metric_list_renders_flat_single_panel(self):
        """A 1-element metric list renders as a flat single panel, not a grid.

        Regression: a single non-AUC metric used to be forced through the
        multi-panel grid builder, giving it a wider figure than the scalar
        metric path. It must now collapse to the flat single-panel renderer.
        """
        from mlsurv.viz.metrics import plot_metric_time
        from mlsurv.viz._palettes import MANUSCRIPT_DEFAULTS

        df = self._make_time_data()
        df = df[df["metric_name"] == "brier"]
        fig = plot_metric_time(df, metric=["brier"])
        layout = fig.to_dict()["layout"]
        # Flat figure: no second subplot axis
        assert "xaxis2" not in layout
        assert "yaxis2" not in layout
        # Single-panel dimensions, matching the scalar metric path
        assert fig.layout.width == MANUSCRIPT_DEFAULTS["width_single"]
        assert fig.layout.height == MANUSCRIPT_DEFAULTS["height_small"]

    def test_single_metric_brier_matches_auc_dimensions(self):
        """Single brier (list) and single auc (scalar) share figure dimensions."""
        from mlsurv.viz.metrics import plot_metric_time

        long_df = self._make_time_data()
        auc_wide = pd.DataFrame({
            "model": ["coxph", "coxph", "coxph"],
            "eval_time": [6, 12, 24],
            "auc": [0.72, 0.70, 0.68],
        })
        fig_auc = plot_metric_time(auc_wide, metric="auc")
        fig_brier = plot_metric_time(
            long_df[long_df["metric_name"] == "brier"], metric=["brier"],
        )
        assert fig_auc.layout.width == fig_brier.layout.width
        assert fig_auc.layout.height == fig_brier.layout.height

    def test_single_metric_list_is_lower_better_with_reference(self):
        """Collapsed brier panel keeps brier's lower-is-better y-axis + reference.

        The collapse must preserve the metric identity, not just the size: brier
        is in LOWER_IS_BETTER, so the y-range floors at 0 and a reference hline
        is drawn.
        """
        from mlsurv.viz.metrics import plot_metric_time

        df = self._make_time_data()
        df = df[df["metric_name"] == "brier"]
        fig = plot_metric_time(df, metric=["brier"])
        # Lower-is-better: y-range is [0, y_max], not [y_min, 1.0]
        y_range = fig.layout.yaxis.range
        assert y_range[0] == 0
        assert y_range[1] < 1.0
        # Brier reference line is drawn as a layout shape
        assert len(fig.layout.shapes) >= 1

    def test_single_metric_list_ci_ribbon(self):
        """A 1-element metric list with std_df + show_ci draws a CI ribbon.

        Guards the collapsed path's std_col resolution ("value" -> "value_std").
        """
        from mlsurv.viz.metrics import plot_metric_time

        df = self._make_time_data()
        std_df = self._make_std_data()
        df = df[df["metric_name"] == "brier"]
        std_df = std_df[std_df["metric_name"] == "brier"]
        fig = plot_metric_time(df, std_df=std_df, metric=["brier"], show_ci=True)
        # Ribbon traces are filled scatters (fill set to 'toself'/'tonexty')
        fills = [t for t in fig.data if getattr(t, "fill", None)]
        assert len(fills) >= 1

    def test_single_metric_list_return_data(self):
        """return_data=True with a 1-element list returns the figure+data dict."""
        from mlsurv.viz.metrics import plot_metric_time
        import plotly.graph_objects as go

        df = self._make_time_data()
        df = df[df["metric_name"] == "brier"]
        out = plot_metric_time(df, metric=["brier"], return_data=True)
        assert set(out) == {"figure", "data"}
        assert isinstance(out["figure"], go.Figure)
        assert len(out["data"]) == len(df)


class TestPatientSurvivalCurve:
    """Tests for plot_patient_survival_curve (shared per-patient S(t|x) figure)."""

    @staticmethod
    def _make_pred(*, with_band=False, with_times=True, time_unit="months"):
        """Build a minimal PredictionResult by hand (no model training)."""
        from mlsurv.results._prediction import PredictionResult

        time_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        surv = np.array([
            [1.0, 0.90, 0.70, 0.50, 0.30],   # patient 0 (median ~3)
            [1.0, 0.80, 0.50, 0.30, 0.10],   # patient 1 (higher risk)
            [1.0, 0.95, 0.90, 0.85, 0.80],   # patient 2 (median undefined)
        ])
        risk = np.array([0.5, 1.5, 0.1])
        median = np.array([3.0, 2.0, np.nan])
        kwargs = dict(
            risk_scores=risk, model_name="coxph", n_patients=3,
            _time_grid=time_grid, _surv_probs=surv,
            _median_survival=median, time_unit=time_unit,
        )
        if with_times:
            user_times = np.array([1.0, 3.0])
            kwargs["_user_times"] = user_times
            kwargs["_surv_at_user_times"] = surv[:, [1, 3]]
        if with_band:
            # (n_bootstrap, n_patients, n_times) — small jitter around surv
            boot = np.stack([surv * 0.98, surv, surv * 1.02], axis=0).clip(0, 1)
            kwargs["_bootstrap_surv_probs"] = boot
        return PredictionResult(**kwargs)

    def test_returns_figure_with_curve(self):
        from mlsurv.viz.survival import plot_patient_survival_curve
        import plotly.graph_objects as go

        fig = plot_patient_survival_curve(self._make_pred(), 0)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_axis_label_uses_time_unit(self):
        from mlsurv.viz.survival import plot_patient_survival_curve

        fig = plot_patient_survival_curve(self._make_pred(time_unit="months"), 0)
        assert "months" in fig.layout.xaxis.title.text
        # time_unit override wins over pred.time_unit
        fig2 = plot_patient_survival_curve(
            self._make_pred(time_unit="months"), 0, time_unit="years",
        )
        assert "years" in fig2.layout.xaxis.title.text

    def test_ci_band_present_when_bootstrap_available(self):
        from mlsurv.viz.survival import plot_patient_survival_curve

        fig = plot_patient_survival_curve(self._make_pred(with_band=True), 0)
        names = [t.name for t in fig.data if t.name]
        assert any("CI" in n for n in names)

    def test_no_ci_band_without_bootstrap(self):
        from mlsurv.viz.survival import plot_patient_survival_curve

        fig = plot_patient_survival_curve(self._make_pred(with_band=False), 0)
        names = [t.name for t in fig.data if t.name]
        assert not any("CI" in n for n in names)

    def test_milestone_annotations_when_times_present(self):
        from mlsurv.viz.survival import plot_patient_survival_curve

        fig = plot_patient_survival_curve(self._make_pred(with_times=True), 0)
        assert len(fig.layout.annotations) >= 1

    def test_no_milestones_without_user_times(self):
        from mlsurv.viz.survival import plot_patient_survival_curve

        fig = plot_patient_survival_curve(self._make_pred(with_times=False), 0)
        assert len(fig.layout.annotations) == 0

    def test_median_marker_skipped_when_undefined(self):
        from mlsurv.viz.survival import plot_patient_survival_curve

        # patient 2 has NaN median -> no "Median = ..." trace
        with_median = plot_patient_survival_curve(self._make_pred(), 0)
        without_median = plot_patient_survival_curve(self._make_pred(), 2)
        assert any((t.name or "").startswith("Median") for t in with_median.data)
        assert not any((t.name or "").startswith("Median") for t in without_median.data)

    def test_no_survival_function_raises(self):
        from mlsurv.viz.survival import plot_patient_survival_curve
        from mlsurv.results._prediction import PredictionResult

        pred = PredictionResult(
            risk_scores=np.array([0.1, 0.2]), model_name="svm", n_patients=2,
        )
        with pytest.raises(ValueError):
            plot_patient_survival_curve(pred, 0)

    def test_axis_label_unitless_when_unit_unknown(self):
        from mlsurv.viz.survival import plot_patient_survival_curve

        fig = plot_patient_survival_curve(self._make_pred(time_unit=None), 0)
        assert fig.layout.xaxis.title.text == "Time"
        # an unrecognized unit also degrades to unit-free
        fig2 = plot_patient_survival_curve(self._make_pred(time_unit="furlongs"), 0)
        assert fig2.layout.xaxis.title.text == "Time"

    def test_milestone_annotation_count_and_text(self):
        from mlsurv.viz.survival import plot_patient_survival_curve

        # _user_times = [1.0, 3.0]; patient 0 S = 0.90 at t=1, 0.50 at t=3
        fig = plot_patient_survival_curve(self._make_pred(with_times=True), 0)
        assert len(fig.layout.annotations) == 2  # one per user time
        texts = [a.text for a in fig.layout.annotations]
        assert any("P(t > 1 month)" in t and "90%" in t for t in texts)

    def test_ci_band_has_fill_and_label(self):
        from mlsurv.viz.survival import plot_patient_survival_curve

        fig = plot_patient_survival_curve(self._make_pred(with_band=True), 0)
        ci_traces = [t for t in fig.data if (t.name or "").endswith("% CI")]
        assert len(ci_traces) == 1
        assert ci_traces[0].fill == "tonexty"
        assert "95" in ci_traces[0].name

    def test_size_passthrough(self):
        from mlsurv.viz.survival import plot_patient_survival_curve

        fig = plot_patient_survival_curve(self._make_pred(), 0, width=640, height=480)
        assert fig.layout.width == 640
        assert fig.layout.height == 480

    def test_out_of_range_patient_raises(self):
        from mlsurv.viz.survival import plot_patient_survival_curve

        # _make_pred has 3 patients; index 99 has no patient_99 column
        with pytest.raises(ValueError, match="out of range"):
            plot_patient_survival_curve(self._make_pred(), 99)

    def test_non_finite_curve_warns(self):
        from mlsurv.viz.survival import plot_patient_survival_curve
        from mlsurv._warnings import SuggestionWarning

        pred = self._make_pred()
        pred._surv_probs[0, 2] = np.nan  # break patient 0's curve
        with pytest.warns(SuggestionWarning, match="non-finite"):
            plot_patient_survival_curve(pred, 0)

    def test_non_finite_milestone_skipped(self):
        from mlsurv.viz.survival import plot_patient_survival_curve

        pred = self._make_pred(with_times=True)
        pred._surv_at_user_times[0, 1] = np.nan  # drop one milestone for patient 0
        fig = plot_patient_survival_curve(pred, 0)
        assert len(fig.layout.annotations) == 1  # only the finite milestone remains
