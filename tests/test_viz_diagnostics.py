"""Tests for diagnostic residual plots (Schoenfeld + deviance)."""

import numpy as np
import pandas as pd
import pytest

from tests._helpers import make_survival_array

pytest.importorskip("plotly")
pytest.importorskip("lifelines")


@pytest.fixture
def cox_data():
    """Synthetic data suitable for Cox fitting."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        'age': np.random.normal(60, 10, n),
        'bmi': np.random.normal(25, 5, n),
        'crp': np.random.exponential(1, n),
        'albumin': np.random.normal(4, 0.5, n),
        'stage_II': np.random.binomial(1, 0.3, n).astype(float),
        'stage_III': np.random.binomial(1, 0.2, n).astype(float),
    })
    y = make_survival_array(
        np.random.binomial(1, 0.5, n),
        np.random.exponential(24, n),
    )
    return X, y


class TestSchoenfeld:
    """Test Schoenfeld residual plot."""

    def test_basic(self, cox_data):
        from mlsurv.viz.diagnostics import plot_schoenfeld
        import plotly.graph_objects as go
        X, y = cox_data
        fig = plot_schoenfeld(X, y)
        assert isinstance(fig, go.Figure)
        # Should have traces (scatter + optional LOWESS per feature)
        assert len(fig.data) > 0

    def test_feature_selection(self, cox_data):
        from mlsurv.viz.diagnostics import plot_schoenfeld
        import plotly.graph_objects as go
        X, y = cox_data
        fig = plot_schoenfeld(X, y, features=['age', 'bmi'])
        assert isinstance(fig, go.Figure)

    def test_n_features(self, cox_data):
        from mlsurv.viz.diagnostics import plot_schoenfeld
        import plotly.graph_objects as go
        X, y = cox_data
        fig = plot_schoenfeld(X, y, n_features=3)
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, cox_data):
        from mlsurv.viz.diagnostics import plot_schoenfeld
        X, y = cox_data
        fig = plot_schoenfeld(X, y, n_features=2, title='Custom Title')
        assert 'Custom Title' in fig.layout.title.text

    def test_features_filter_no_match_returns_empty(self, cox_data):
        """features=['nonexistent'] → empty figure, not ZeroDivisionError."""
        import plotly.graph_objects as go
        from mlsurv.viz.diagnostics import plot_schoenfeld
        X, y = cox_data
        fig = plot_schoenfeld(X, y, features=['nonexistent'])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0


class TestDeviance:
    """Test deviance residual plot."""

    def test_basic(self, cox_data):
        from mlsurv.viz.diagnostics import plot_deviance
        import plotly.graph_objects as go
        X, y = cox_data
        fig = plot_deviance(X, y)
        assert isinstance(fig, go.Figure)
        # Should have Event + Censored traces
        assert len(fig.data) >= 2

    def test_event_coloring(self, cox_data):
        from mlsurv.viz.diagnostics import plot_deviance
        X, y = cox_data
        fig = plot_deviance(X, y)
        trace_names = [t.name for t in fig.data]
        assert 'Event' in trace_names
        assert 'Censored' in trace_names

    def test_custom_title(self, cox_data):
        from mlsurv.viz.diagnostics import plot_deviance
        X, y = cox_data
        fig = plot_deviance(X, y, title='My Deviance')
        assert 'My Deviance' in fig.layout.title.text

    def test_reference_lines(self, cox_data):
        from mlsurv.viz.diagnostics import plot_deviance
        X, y = cox_data
        fig = plot_deviance(X, y)
        # Should have reference lines at +/-2 and 0
        shapes = fig.layout.shapes or []
        # hlines are stored as shapes in plotly
        assert len(shapes) >= 2


class TestDiagnosticsColors:
    """Verify Okabe-Ito colors are used instead of Tableau 10."""

    def test_schoenfeld_scatter_color(self, cox_data):
        from mlsurv.viz.diagnostics import plot_schoenfeld, _SCATTER_COLOR
        X, y = cox_data
        fig = plot_schoenfeld(X, y, n_features=2)
        scatter_traces = [t for t in fig.data if t.mode == "markers"]
        assert len(scatter_traces) > 0
        assert scatter_traces[0].marker.color == _SCATTER_COLOR

    def test_deviance_event_color(self, cox_data):
        from mlsurv.viz.diagnostics import plot_deviance, _EVENT_COLOR, _CENSORED_COLOR
        X, y = cox_data
        fig = plot_deviance(X, y)
        event_trace = [t for t in fig.data if t.name == "Event"][0]
        censored_trace = [t for t in fig.data if t.name == "Censored"][0]
        assert event_trace.marker.color == _EVENT_COLOR
        assert censored_trace.marker.color == _CENSORED_COLOR

    def test_schoenfeld_has_axis_labels(self, cox_data):
        from mlsurv.viz.diagnostics import plot_schoenfeld
        X, y = cox_data
        fig = plot_schoenfeld(X, y, n_features=3)
        # Bottom-left subplot should have axis labels
        assert fig.layout.xaxis.title.text == "Time" or fig.layout.xaxis3.title.text == "Time"

    def test_schoenfeld_time_unit(self, cox_data):
        from mlsurv.viz.diagnostics import plot_schoenfeld
        X, y = cox_data
        fig = plot_schoenfeld(X, y, n_features=3, time_unit="months")
        has_label = (
            getattr(fig.layout.xaxis.title, 'text', '') == "Time (months)"
            or getattr(fig.layout.xaxis3.title, 'text', '') == "Time (months)"
        )
        assert has_label


class TestSchoenfeldReturnData:
    """Test compute_schoenfeld_data and DataFrame round-trip."""

    def test_return_data_schema(self, cox_data):
        """compute_schoenfeld_data returns DataFrame with expected columns."""
        from mlsurv.viz.diagnostics import compute_schoenfeld_data
        X, y = cox_data
        df = compute_schoenfeld_data(X, y, n_features=3)
        assert isinstance(df, pd.DataFrame)
        for col in ("feature", "time", "residual"):
            assert col in df.columns, f"Missing column: {col}"
        assert len(df) > 0
        assert df["feature"].nunique() == 3

    def test_return_data_round_trip(self, cox_data):
        """compute_schoenfeld_data → plot_schoenfeld(df) produces a figure."""
        from mlsurv.viz.diagnostics import compute_schoenfeld_data, plot_schoenfeld
        import plotly.graph_objects as go
        X, y = cox_data
        df = compute_schoenfeld_data(X, y, n_features=3)
        fig = plot_schoenfeld(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_return_data_specific_features(self, cox_data):
        """compute_schoenfeld_data respects explicit feature list."""
        from mlsurv.viz.diagnostics import compute_schoenfeld_data
        X, y = cox_data
        df = compute_schoenfeld_data(X, y, features=["age", "bmi"])
        assert set(df["feature"].unique()) == {"age", "bmi"}


class TestDevianceReturnData:
    """Test compute_deviance_data and DataFrame round-trip."""

    def test_return_data_schema(self, cox_data):
        """compute_deviance_data returns DataFrame with expected columns."""
        from mlsurv.viz.diagnostics import compute_deviance_data
        X, y = cox_data
        df = compute_deviance_data(X, y)
        assert isinstance(df, pd.DataFrame)
        for col in ("linear_predictor", "deviance", "event"):
            assert col in df.columns, f"Missing column: {col}"
        assert len(df) == len(X)

    def test_return_data_round_trip(self, cox_data):
        """compute_deviance_data → plot_deviance(df) produces a figure."""
        from mlsurv.viz.diagnostics import compute_deviance_data, plot_deviance
        import plotly.graph_objects as go
        X, y = cox_data
        df = compute_deviance_data(X, y)
        fig = plot_deviance(df)
        assert isinstance(fig, go.Figure)
        # Should have Event + Censored traces
        assert len(fig.data) >= 2

    def test_plot_deviance_rejects_wrong_columns(self):
        """plot_deviance raises ValueError for DataFrame with wrong columns."""
        from mlsurv.viz.diagnostics import plot_deviance
        bad_df = pd.DataFrame({"deviance": [0.5], "linear_predictor": [1.0], "wrong": [True]})
        # Has deviance and linear_predictor but missing event
        with pytest.raises(ValueError, match="missing required columns"):
            plot_deviance(bad_df)


# ── _smooth_trend fallback tests ─────────────────────────────────────


class TestSmoothTrend:
    """Test _smooth_trend 3-tier fallback chain."""

    @pytest.fixture
    def xy(self):
        """Simple x/y arrays for smoothing."""
        rng = np.random.RandomState(0)
        x = rng.uniform(0, 10, 50)
        y = np.sin(x) + rng.normal(0, 0.1, 50)
        return x, y

    def test_statsmodels_primary(self, xy):
        """Primary path (statsmodels) returns sorted (x, y) tuple."""
        from mlsurv.viz.diagnostics import _smooth_trend
        x, y = xy
        result = _smooth_trend(x, y)
        assert result is not None
        xs, ys = result
        assert len(xs) == len(x)
        # Verify sorted by x
        assert np.all(np.diff(xs) >= 0)

    def test_scipy_fallback(self, xy, monkeypatch):
        """When statsmodels is missing, scipy uniform_filter1d is used."""
        from mlsurv.viz import diagnostics as diag_mod
        import builtins
        import sys

        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == 'statsmodels.nonparametric.smoothers_lowess':
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', _mock_import)
        # Clear cached module so import actually calls __import__
        monkeypatch.delitem(sys.modules, 'statsmodels.nonparametric.smoothers_lowess', raising=False)

        result = diag_mod._smooth_trend(*xy)
        assert result is not None
        xs, ys = result
        assert len(xs) == len(xy[0])

    def test_both_unavailable_returns_none(self, xy, monkeypatch):
        """When both statsmodels and scipy are missing, returns None."""
        from mlsurv.viz import diagnostics as diag_mod
        import builtins
        import sys

        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name in ('statsmodels.nonparametric.smoothers_lowess',
                        'scipy.ndimage'):
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', _mock_import)
        # Clear cached modules so import actually calls __import__
        monkeypatch.delitem(sys.modules, 'statsmodels.nonparametric.smoothers_lowess', raising=False)
        monkeypatch.delitem(sys.modules, 'scipy.ndimage', raising=False)

        result = diag_mod._smooth_trend(*xy)
        assert result is None

    def test_lowess_error_warns_and_falls_back(self, xy, monkeypatch):
        """ValueError from lowess triggers warning and scipy fallback."""
        pytest.importorskip("statsmodels")
        import warnings
        from mlsurv.viz import diagnostics as diag_mod

        def _bad_lowess(*args, **kwargs):
            raise ValueError("mocked lowess failure")

        monkeypatch.setattr(
            'statsmodels.nonparametric.smoothers_lowess.lowess', _bad_lowess,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = diag_mod._smooth_trend(*xy)

        assert result is not None  # scipy fallback succeeded
        warn_msgs = [str(wi.message) for wi in w]
        assert any("LOWESS smoothing failed" in m for m in warn_msgs)

    def test_schoenfeld_renders_without_trend(self, cox_data, monkeypatch):
        """plot_schoenfeld still works when _smooth_trend returns None."""
        import plotly.graph_objects as go
        from mlsurv.viz import diagnostics as diag_mod

        monkeypatch.setattr(diag_mod, '_smooth_trend', lambda *a, **kw: None)
        X, y = cox_data
        fig = diag_mod.plot_schoenfeld(X, y, n_features=2)
        assert isinstance(fig, go.Figure)
        # All traces should be markers-only (no LOWESS lines)
        for trace in fig.data:
            assert trace.mode == 'markers'


# ── plot_schoenfeld_interactive tests ────────────────────────────────


class TestSchoenfeldInteractive:
    """Test plot_schoenfeld_interactive."""

    def test_basic_returns_figure(self, cox_data):
        """Returns a valid go.Figure with traces."""
        import plotly.graph_objects as go
        from mlsurv.viz.diagnostics import plot_schoenfeld_interactive
        X, y = cox_data
        fig = plot_schoenfeld_interactive(X, y)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_dropdown_buttons_per_feature(self, cox_data):
        """One dropdown with one button per feature (6 features)."""
        from mlsurv.viz.diagnostics import plot_schoenfeld_interactive
        X, y = cox_data
        fig = plot_schoenfeld_interactive(X, y)
        menus = fig.layout.updatemenus
        assert len(menus) == 1
        buttons = menus[0].buttons
        assert len(buttons) == 6  # all 6 cox_data features

    def test_first_feature_visible_by_default(self, cox_data):
        """First feature's traces are visible, rest are hidden."""
        from mlsurv.viz.diagnostics import plot_schoenfeld_interactive
        X, y = cox_data
        fig = plot_schoenfeld_interactive(X, y)
        # At least one visible trace
        visible_traces = [t for t in fig.data if t.visible is True]
        hidden_traces = [t for t in fig.data if t.visible is False]
        assert len(visible_traces) >= 1
        assert len(hidden_traces) >= 1

    def test_dataframe_input(self):
        """Pre-computed DataFrame with 3 features → 3 dropdown buttons."""
        import plotly.graph_objects as go
        from mlsurv.viz.diagnostics import plot_schoenfeld_interactive
        df = pd.DataFrame({
            'feature': ['a'] * 10 + ['b'] * 10 + ['c'] * 10,
            'time': list(range(1, 11)) * 3,
            'residual': np.random.randn(30),
        })
        fig = plot_schoenfeld_interactive(df)
        assert isinstance(fig, go.Figure)
        buttons = fig.layout.updatemenus[0].buttons
        assert len(buttons) == 3

    def test_visibility_arrays_correct_length(self, cox_data):
        """Each button's visibility array matches total trace count."""
        from mlsurv.viz.diagnostics import plot_schoenfeld_interactive
        X, y = cox_data
        fig = plot_schoenfeld_interactive(X, y)
        n_traces = len(fig.data)
        for btn in fig.layout.updatemenus[0].buttons:
            vis = btn.args[0]['visible']
            assert len(vis) == n_traces
            assert any(vis)  # at least one True per button

    def test_custom_title(self, cox_data):
        """Custom title appears in layout."""
        from mlsurv.viz.diagnostics import plot_schoenfeld_interactive
        X, y = cox_data
        fig = plot_schoenfeld_interactive(X, y, title='My PH Check')
        assert 'My PH Check' in fig.layout.title.text

    def test_long_feature_name_truncated_in_dropdown(self):
        """Long feature names get ellipsized in dropdown buttons but
        the underlying title-update args keep the full name."""
        from mlsurv.viz.diagnostics import plot_schoenfeld_interactive
        long_name = "an_extremely_long_feature_name_that_blows_out_the_dropdown"
        df = pd.DataFrame({
            'feature': [long_name] * 10 + ['short_name'] * 10,
            'time': list(range(1, 11)) * 2,
            'residual': np.random.randn(20),
        })
        fig = plot_schoenfeld_interactive(df)
        buttons = fig.layout.updatemenus[0].buttons
        labels = [b.label for b in buttons]
        # One button per feature
        assert len(buttons) == 2
        # Long name truncated, short name unchanged
        long_btn = next(b for b in buttons if b.label.endswith("..."))
        assert len(long_btn.label) <= 30
        # Full name preserved in args (title update)
        assert long_name in long_btn.args[1]["title.text"]
        assert "short_name" in labels

    def test_colliding_long_prefixes_disambiguated_by_title(self):
        """When two long feature names share the first 27 chars, the
        dropdown labels may collide; the full name in the title-update
        args is what disambiguates the selection. This pins that contract."""
        from mlsurv.viz.diagnostics import plot_schoenfeld_interactive
        prefix = "tumor_size_measurement_in_centimeters"
        feat_a = f"{prefix}_baseline"
        feat_b = f"{prefix}_followup"
        df = pd.DataFrame({
            'feature': [feat_a] * 5 + [feat_b] * 5,
            'time': list(range(1, 6)) * 2,
            'residual': np.random.randn(10),
        })
        fig = plot_schoenfeld_interactive(df)
        buttons = fig.layout.updatemenus[0].buttons
        # Title-update args carry the full feature name per button —
        # always unique even when display labels happen to collide.
        title_texts = {b.args[1]["title.text"] for b in buttons}
        assert len(title_texts) == len(buttons)
        assert any(feat_a in t for t in title_texts)
        assert any(feat_b in t for t in title_texts)

    def test_rejects_non_dataframe_without_y(self):
        """TypeError for numpy array with y_train=None."""
        from mlsurv.viz.diagnostics import plot_schoenfeld_interactive
        with pytest.raises(TypeError, match="requires y_train"):
            plot_schoenfeld_interactive(np.array([[1, 2], [3, 4]]))

    def test_rejects_missing_columns(self):
        """ValueError for DataFrame missing required columns."""
        from mlsurv.viz.diagnostics import plot_schoenfeld_interactive
        bad_df = pd.DataFrame({'feature': ['a'], 'time': [1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_schoenfeld_interactive(bad_df)


# ── NaN data tests ──────────────────────────────────────────────────


class TestDiagnosticsNaN:
    """Test NaN handling in all diagnostics."""

    @pytest.fixture
    def cox_data_with_nan(self):
        """Synthetic data with ~10% NaN injected."""
        rng = np.random.RandomState(123)
        n = 200
        X = pd.DataFrame({
            'a': rng.randn(n),
            'b': rng.randn(n),
            'c': rng.randn(n),
        })
        # Inject ~10% NaN
        nan_idx = rng.choice(n, size=20, replace=False)
        X.loc[nan_idx, 'a'] = np.nan

        lp = 0.5 * np.nan_to_num(X['a']) + 0.3 * X['b']
        event_times = rng.exponential(1.0 / np.exp(lp))
        censor_times = rng.exponential(3.0, size=n)
        times = np.minimum(event_times, censor_times)
        events = event_times <= censor_times

        y = make_survival_array(events, times)
        return X, y

    def test_compute_schoenfeld_data_with_nan(self, cox_data_with_nan):
        """compute_schoenfeld_data drops NaN rows and returns valid DF."""
        from mlsurv.viz.diagnostics import compute_schoenfeld_data
        X, y = cox_data_with_nan
        df = compute_schoenfeld_data(X, y, n_features=3)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert not df['residual'].isna().any()

    def test_plot_schoenfeld_with_nan(self, cox_data_with_nan):
        """plot_schoenfeld handles NaN data without crashing."""
        import plotly.graph_objects as go
        from mlsurv.viz.diagnostics import plot_schoenfeld
        X, y = cox_data_with_nan
        fig = plot_schoenfeld(X, y, n_features=3)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_schoenfeld_interactive_with_nan(self, cox_data_with_nan):
        """plot_schoenfeld_interactive handles NaN data without crashing."""
        import plotly.graph_objects as go
        from mlsurv.viz.diagnostics import plot_schoenfeld_interactive
        X, y = cox_data_with_nan
        fig = plot_schoenfeld_interactive(X, y)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_compute_deviance_data_with_nan(self, cox_data_with_nan):
        """compute_deviance_data drops NaN rows and returns valid DF."""
        from mlsurv.viz.diagnostics import compute_deviance_data
        X, y = cox_data_with_nan
        df = compute_deviance_data(X, y)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df) < len(X)  # rows were dropped
        assert not df['deviance'].isna().any()

    def test_plot_deviance_with_nan(self, cox_data_with_nan):
        """plot_deviance handles NaN data without crashing."""
        import plotly.graph_objects as go
        from mlsurv.viz.diagnostics import plot_deviance
        X, y = cox_data_with_nan
        fig = plot_deviance(X, y)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_all_nan_returns_empty_schoenfeld(self):
        """All-NaN input returns empty DataFrame with correct schema."""
        from mlsurv.viz.diagnostics import compute_schoenfeld_data
        X = pd.DataFrame({'a': [np.nan, np.nan], 'b': [np.nan, np.nan]})
        y = make_survival_array([True, False], [1.0, 2.0])
        result = compute_schoenfeld_data(X, y)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ['feature', 'time', 'residual']

    def test_all_nan_returns_empty_deviance(self):
        """All-NaN input returns empty DataFrame with correct schema."""
        from mlsurv.viz.diagnostics import compute_deviance_data
        X = pd.DataFrame({'a': [np.nan, np.nan], 'b': [np.nan, np.nan]})
        y = make_survival_array([True, False], [1.0, 2.0])
        result = compute_deviance_data(X, y)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ['linear_predictor', 'deviance', 'event']

    def test_all_nan_plot_schoenfeld_returns_empty_figure(self):
        """plot_schoenfeld returns empty figure for all-NaN data."""
        import plotly.graph_objects as go
        from mlsurv.viz.diagnostics import plot_schoenfeld
        X = pd.DataFrame({'a': [np.nan, np.nan], 'b': [np.nan, np.nan]})
        y = make_survival_array([True, False], [1.0, 2.0])
        fig = plot_schoenfeld(X, y)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_all_nan_plot_deviance_returns_empty_figure(self):
        """plot_deviance returns empty figure for all-NaN data."""
        import plotly.graph_objects as go
        from mlsurv.viz.diagnostics import plot_deviance
        X = pd.DataFrame({'a': [np.nan, np.nan], 'b': [np.nan, np.nan]})
        y = make_survival_array([True, False], [1.0, 2.0])
        fig = plot_deviance(X, y)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_plot_diagnostics_learner_with_nan(self, cox_data_with_nan):
        """plot_diagnostics(kind='schoenfeld') works via learner when X_train has NaN."""
        import plotly.graph_objects as go
        from mlsurv import SurvivalLearner

        X, y = cox_data_with_nan
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('rsf')

        fig = learner.plot_diagnostics(kind='schoenfeld')
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_diagnostics_deviance_with_nan(self, cox_data_with_nan):
        """plot_diagnostics(kind='deviance') works via learner when X_train has NaN."""
        import plotly.graph_objects as go
        from mlsurv import SurvivalLearner

        X, y = cox_data_with_nan
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3)
        learner.train('rsf')

        fig = learner.plot_diagnostics(kind='deviance')
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_all_nan_plot_interactive_returns_empty_figure(self):
        """plot_schoenfeld_interactive returns empty figure for all-NaN data."""
        import plotly.graph_objects as go
        from mlsurv.viz.diagnostics import plot_schoenfeld_interactive
        X = pd.DataFrame({'a': [np.nan, np.nan], 'b': [np.nan, np.nan]})
        y = make_survival_array([True, False], [1.0, 2.0])
        fig = plot_schoenfeld_interactive(X, y)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0


# ---------------------------------------------------------------------------
# Martingale residual plot
# ---------------------------------------------------------------------------


def _quadratic_data(n=300, seed=0):
    """x0 enters the log-hazard quadratically; x1 linearly."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, 3)),
        columns=[f"x{i}" for i in range(3)],
    )
    X["binary"] = rng.integers(0, 2, size=n).astype(float)
    lp = 0.8 * (X["x0"] ** 2 - 1) - 0.3 * X["x1"]
    time = rng.exponential(scale=np.exp(-lp))
    event = rng.random(n) < 0.7
    y = np.array(
        [(bool(e), max(t, 0.01)) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


class TestComputeMartingaleData:
    def test_shape_and_columns(self):
        from mlsurv.viz.diagnostics import compute_martingale_data
        X, y = _quadratic_data()
        df = compute_martingale_data(X, y)
        assert list(df.columns) == ["feature", "x_value", "residual"]
        # 3 continuous features × 300 obs (binary excluded)
        assert len(df) == 3 * 300

    def test_excludes_binary_features(self):
        from mlsurv.viz.diagnostics import compute_martingale_data
        X, y = _quadratic_data()
        df = compute_martingale_data(X, y)
        assert "binary" not in df["feature"].unique().tolist()

    def test_residuals_mean_is_approximately_zero(self):
        """Null-model martingale residuals sum to ~0 across subjects."""
        from mlsurv.viz.diagnostics import compute_martingale_data
        X, y = _quadratic_data()
        df = compute_martingale_data(X, y)
        # Extract one feature's residuals (any will do — the residual
        # vector is tiled identically across all features)
        r = df[df["feature"] == "x0"]["residual"].to_numpy()
        assert abs(r.mean()) < 0.01, (
            f"null-model martingales should mean ~0, got {r.mean():.4f}"
        )

    def test_reveals_quadratic_shape(self):
        """LOWESS trend of x0 should be U-shaped (tails above middle)."""
        from mlsurv.viz.diagnostics import compute_martingale_data
        X, y = _quadratic_data()
        df = compute_martingale_data(X, y)
        x0 = df[df["feature"] == "x0"].sort_values("x_value")
        left = x0.iloc[:30]["residual"].mean()
        mid = x0.iloc[len(x0) // 2 - 15: len(x0) // 2 + 15]["residual"].mean()
        right = x0.iloc[-30:]["residual"].mean()
        # Quadratic: extreme |x0| → more hazard → shorter time → smaller H0
        # → residual = event - H0 is closer to event (larger)
        assert left > mid, f"left tail {left:.3f} should exceed middle {mid:.3f}"
        assert right > mid, f"right tail {right:.3f} should exceed middle {mid:.3f}"

    def test_feature_subset(self):
        from mlsurv.viz.diagnostics import compute_martingale_data
        X, y = _quadratic_data()
        df = compute_martingale_data(X, y, features=["x0"])
        assert df["feature"].unique().tolist() == ["x0"]

    def test_no_continuous_features_returns_empty(self):
        from mlsurv.viz.diagnostics import compute_martingale_data
        rng = np.random.default_rng(0)
        X = pd.DataFrame({
            "b0": rng.integers(0, 2, 100).astype(float),
            "b1": rng.integers(0, 2, 100).astype(float),
        })
        y = np.array(
            [(bool(e), max(t, 0.01)) for e, t in
             zip(rng.random(100) < 0.7, rng.exponential(1, 100))],
            dtype=[("event", bool), ("time", float)],
        )
        df = compute_martingale_data(X, y)
        assert df.empty
        assert list(df.columns) == ["feature", "x_value", "residual"]


class TestPlotMartingaleInteractive:
    def test_returns_figure_with_dropdown(self):
        import plotly.graph_objects as go
        from mlsurv.viz.diagnostics import plot_martingale_interactive
        X, y = _quadratic_data()
        fig = plot_martingale_interactive(X, y)
        assert isinstance(fig, go.Figure)
        # One dropdown with a button per continuous feature (3)
        assert fig.layout.updatemenus is not None
        assert len(fig.layout.updatemenus) == 1
        assert len(fig.layout.updatemenus[0].buttons) == 3

    def test_trace_count_at_least_one_per_feature(self):
        """At minimum, one scatter trace per continuous feature is drawn.

        We intentionally don't upper-bound the trace count — any future
        change that adds reference lines / overlays as traces (instead of
        ``add_hline``) shouldn't break the contract that every feature
        gets represented.
        """
        from mlsurv.viz.diagnostics import plot_martingale_interactive
        X, y = _quadratic_data()
        fig = plot_martingale_interactive(X, y)
        # 3 continuous features (x0, x1, x2) — expect ≥ 3 traces total
        assert len(fig.data) >= 3

    def test_precomputed_dataframe_path(self):
        """Passing a pre-computed DataFrame (no y_train) also works."""
        import plotly.graph_objects as go
        from mlsurv.viz.diagnostics import (
            compute_martingale_data, plot_martingale_interactive,
        )
        X, y = _quadratic_data()
        df = compute_martingale_data(X, y)
        fig = plot_martingale_interactive(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3

    def test_missing_columns_raises(self):
        """Pre-computed DataFrame missing required columns → ValueError."""
        from mlsurv.viz.diagnostics import plot_martingale_interactive
        bad = pd.DataFrame({"wrong": [1, 2], "cols": [3, 4]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_martingale_interactive(bad)

    def test_no_y_train_with_non_dataframe_raises(self):
        from mlsurv.viz.diagnostics import plot_martingale_interactive
        with pytest.raises(TypeError, match="y_train"):
            plot_martingale_interactive("not a dataframe")

    def test_long_feature_name_truncated_in_dropdown(self):
        """Long feature names get ellipsized in martingale dropdown."""
        from mlsurv.viz.diagnostics import (
            compute_martingale_data, plot_martingale_interactive,
        )
        X, y = _quadratic_data()
        long_name = "ridiculously_long_continuous_covariate_name_that_overflows"
        X = X.rename(columns={X.columns[0]: long_name})
        df = compute_martingale_data(X, y)
        fig = plot_martingale_interactive(df)
        buttons = fig.layout.updatemenus[0].buttons
        long_btn = next(b for b in buttons if b.label.endswith("..."))
        assert len(long_btn.label) <= 30
        assert long_name in long_btn.args[1]["title.text"]

    def test_all_nan_returns_empty_figure(self):
        import plotly.graph_objects as go
        from mlsurv.viz.diagnostics import plot_martingale_interactive
        X = pd.DataFrame({"a": [np.nan] * 20, "b": [np.nan] * 20})
        y = np.array(
            [(bool(i % 2), float(i + 1)) for i in range(20)],
            dtype=[("event", bool), ("time", float)],
        )
        fig = plot_martingale_interactive(X, y)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0


class TestPlotDiagnosticsMartingale:
    """``learner.plot_diagnostics(kind='martingale')`` dispatch."""

    def test_dispatch_returns_figure(self):
        import plotly.graph_objects as go
        from mlsurv import SurvivalLearner
        X, y = _quadratic_data()
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3, verbose=False)
        fig = learner.plot_diagnostics(kind="martingale")
        assert isinstance(fig, go.Figure)

    def test_dispatch_return_data(self):
        from mlsurv import SurvivalLearner
        X, y = _quadratic_data()
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3, verbose=False)
        df = learner.plot_diagnostics(kind="martingale", return_data=True)
        assert list(df.columns) == ["feature", "x_value", "residual"]

    def test_unknown_kind_mentions_martingale(self):
        from mlsurv import SurvivalLearner
        X, y = _quadratic_data()
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42, cv_folds=3, verbose=False)
        with pytest.raises(ValueError, match="martingale"):
            learner.plot_diagnostics(kind="bogus")
