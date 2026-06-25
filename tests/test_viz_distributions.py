"""Tests for mlsurv.viz.distributions — feature distribution plots."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _make_numeric_df(n=50, rng=None):
    """Synthetic numeric-only DataFrame pair (train/test)."""
    if rng is None:
        rng = np.random.default_rng(42)
    X_train = pd.DataFrame({
        "age": rng.normal(60, 10, n),
        "bmi": rng.normal(25, 5, n),
        "bp": rng.normal(120, 15, n),
    })
    X_test = pd.DataFrame({
        "age": rng.normal(60, 10, n // 2),
        "bmi": rng.normal(25, 5, n // 2),
        "bp": rng.normal(120, 15, n // 2),
    })
    return X_train, X_test


def _make_binary_df(n=50, rng=None):
    """Synthetic binary-only DataFrame pair."""
    if rng is None:
        rng = np.random.default_rng(42)
    X_train = pd.DataFrame({
        "treat_A": rng.choice([0, 1], n),
        "treat_B": rng.choice([0, 1], n),
    })
    X_test = pd.DataFrame({
        "treat_A": rng.choice([0, 1], n // 2),
        "treat_B": rng.choice([0, 1], n // 2),
    })
    return X_train, X_test


def _make_onehot_df(n=60, rng=None):
    """Synthetic DataFrame with one-hot encoded categorical group."""
    if rng is None:
        rng = np.random.default_rng(42)
    categories = rng.choice(["I", "II", "III"], n)
    X_train = pd.DataFrame({
        "age": rng.normal(60, 10, n),
        "stage_I": (categories == "I").astype(float),
        "stage_II": (categories == "II").astype(float),
        "stage_III": (categories == "III").astype(float),
    })
    categories_test = rng.choice(["I", "II", "III"], n // 2)
    X_test = pd.DataFrame({
        "age": rng.normal(60, 10, n // 2),
        "stage_I": (categories_test == "I").astype(float),
        "stage_II": (categories_test == "II").astype(float),
        "stage_III": (categories_test == "III").astype(float),
    })
    return X_train, X_test


class TestPlotFeatureDistributions:
    """Basic plotting tests."""

    def test_basic_numeric_only(self):
        from mlsurv.viz.distributions import plot_feature_distributions

        X_train, X_test = _make_numeric_df()
        fig = plot_feature_distributions(X_train, X_test)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_basic_binary_only(self):
        from mlsurv.viz.distributions import plot_feature_distributions

        X_train, X_test = _make_binary_df()
        fig = plot_feature_distributions(X_train, X_test)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_mixed_features(self):
        from mlsurv.viz.distributions import plot_feature_distributions

        X_train, X_test = _make_onehot_df()
        fig = plot_feature_distributions(X_train, X_test)
        assert isinstance(fig, go.Figure)
        # Grouped columns (stage_I/II/III) should NOT appear individually
        buttons = fig.layout.updatemenus[0].buttons
        labels = [b.label for b in buttons]
        assert "age" in labels
        assert "stage (group)" in labels
        assert "stage_I" not in labels
        assert "stage_II" not in labels
        assert "stage_III" not in labels

    def test_dropdown_button_count(self):
        from mlsurv.viz.distributions import plot_feature_distributions

        X_train, X_test = _make_onehot_df()
        fig = plot_feature_distributions(X_train, X_test)
        buttons = fig.layout.updatemenus[0].buttons
        # 1 numeric (age) + 1 group (stage) = 2
        # (stage_I/II/III are grouped, not shown individually)
        assert len(buttons) == 2

    def test_long_feature_name_truncated_in_dropdown(self):
        """Dropdown button labels truncate long feature names so the menu
        doesn't overflow the figure, while underlying state keeps the full
        name (e.g., title annotation in args)."""
        from mlsurv.viz.distributions import plot_feature_distributions

        rng = np.random.default_rng(0)
        long_name = "a_really_extremely_long_feature_name_that_overflows_dropdown"
        X_train = pd.DataFrame({long_name: rng.normal(0, 1, 50)})
        X_test = pd.DataFrame({long_name: rng.normal(0, 1, 25)})
        fig = plot_feature_distributions(X_train, X_test)
        button = fig.layout.updatemenus[0].buttons[0]
        assert len(button.label) <= 30
        assert button.label.endswith("...")
        # Full name preserved in the layout-update args (title text)
        title_text = button.args[1]["title.text"]
        assert long_name in title_text

    def test_first_feature_visible(self):
        from mlsurv.viz.distributions import plot_feature_distributions

        X_train, X_test = _make_numeric_df()
        fig = plot_feature_distributions(X_train, X_test)
        # First feature's traces should be visible
        has_visible = any(t.visible is True or t.visible is None for t in fig.data)
        assert has_visible

    def test_visibility_array_length(self):
        from mlsurv.viz.distributions import plot_feature_distributions

        X_train, X_test = _make_onehot_df()
        fig = plot_feature_distributions(X_train, X_test)
        n_traces = len(fig.data)
        for button in fig.layout.updatemenus[0].buttons:
            vis = button.args[0]["visible"]
            assert len(vis) == n_traces

    def test_empty_dataframe(self):
        from mlsurv.viz.distributions import plot_feature_distributions

        X_train = pd.DataFrame()
        X_test = pd.DataFrame()
        fig = plot_feature_distributions(X_train, X_test)
        assert isinstance(fig, go.Figure)
        assert "No data" in fig.layout.title.text

    def test_single_feature(self):
        from mlsurv.viz.distributions import plot_feature_distributions

        rng = np.random.default_rng(42)
        X_train = pd.DataFrame({"age": rng.normal(60, 10, 50)})
        X_test = pd.DataFrame({"age": rng.normal(60, 10, 25)})
        fig = plot_feature_distributions(X_train, X_test)
        assert isinstance(fig, go.Figure)
        buttons = fig.layout.updatemenus[0].buttons
        assert len(buttons) == 1

    def test_nan_handling(self):
        from mlsurv.viz.distributions import plot_feature_distributions

        rng = np.random.default_rng(42)
        train_data = rng.normal(60, 10, 50)
        train_data[0:5] = np.nan
        X_train = pd.DataFrame({"age": train_data})
        X_test = pd.DataFrame({"age": rng.normal(60, 10, 25)})
        fig = plot_feature_distributions(X_train, X_test)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_inf_handling(self):
        from mlsurv.viz.distributions import plot_feature_distributions

        rng = np.random.default_rng(42)
        train_data = rng.normal(60, 10, 50)
        train_data[0] = np.inf
        train_data[1] = -np.inf
        X_train = pd.DataFrame({"age": train_data})
        X_test = pd.DataFrame({"age": rng.normal(60, 10, 25)})
        fig = plot_feature_distributions(X_train, X_test)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_categorical_group_detection(self):
        from mlsurv.viz.distributions import plot_feature_distributions

        X_train, X_test = _make_onehot_df()
        fig = plot_feature_distributions(X_train, X_test)
        buttons = fig.layout.updatemenus[0].buttons
        group_labels = [b.label for b in buttons if "(group)" in b.label]
        assert len(group_labels) == 1
        assert group_labels[0] == "stage (group)"
        # Grouped columns should not appear as individual entries
        labels = [b.label for b in buttons]
        assert "stage_I" not in labels

    def test_explicit_categorical_groups(self):
        from mlsurv.viz.distributions import plot_feature_distributions

        X_train, X_test = _make_onehot_df()
        groups = {"stage": ["stage_I", "stage_II", "stage_III"]}
        fig = plot_feature_distributions(X_train, X_test, categorical_groups=groups)
        buttons = fig.layout.updatemenus[0].buttons
        labels = [b.label for b in buttons]
        assert "stage (group)" in labels
        # Only age + group = 2 entries (grouped cols excluded)
        assert len(buttons) == 2

    def test_visibility_mutual_exclusivity(self):
        """Each button's visible traces are disjoint and cover all traces."""
        from mlsurv.viz.distributions import plot_feature_distributions

        X_train, X_test = _make_onehot_df()
        fig = plot_feature_distributions(X_train, X_test)
        buttons = fig.layout.updatemenus[0].buttons
        n_traces = len(fig.data)

        all_true_indices = set()
        for button in buttons:
            vis = button.args[0]["visible"]
            true_indices = {i for i, v in enumerate(vis) if v}
            # No overlap with previously seen indices
            assert not true_indices & all_true_indices, (
                f"Button '{button.label}' shares traces with another button"
            )
            all_true_indices.update(true_indices)

        # All traces are covered
        assert all_true_indices == set(range(n_traces))

    def test_trace_count_numeric(self):
        """Numeric features produce 6 traces each (2x violin + 2x box + 2x scatter)."""
        from mlsurv.viz.distributions import plot_feature_distributions

        X_train, X_test = _make_numeric_df()
        fig = plot_feature_distributions(X_train, X_test)
        # 3 features x 6 traces = 18
        assert len(fig.data) == 18

    def test_non_first_features_hidden(self):
        """Non-first feature traces should be explicitly hidden."""
        from mlsurv.viz.distributions import plot_feature_distributions

        X_train, X_test = _make_numeric_df()
        fig = plot_feature_distributions(X_train, X_test)
        # First 6 traces (first feature) are visible
        for trace in fig.data[:6]:
            assert trace.visible is True
        # Remaining traces are hidden
        for trace in fig.data[6:]:
            assert trace.visible is False

    def test_categorical_all_values_shown(self):
        """Low-cardinality features show bars for all unique values, not just 0/1."""
        from mlsurv.viz.distributions import plot_feature_distributions

        rng = np.random.default_rng(42)
        # Feature with values 0, 1, 2 (3 unique, below threshold)
        X_train = pd.DataFrame({"grade": rng.choice([0, 1, 2], 50)})
        X_test = pd.DataFrame({"grade": rng.choice([0, 1, 2], 25)})
        fig = plot_feature_distributions(X_train, X_test)
        # Should have 3 bar traces (one per unique value), not 2
        assert len(fig.data) == 3

    def test_custom_title(self):
        from mlsurv.viz.distributions import plot_feature_distributions

        X_train, X_test = _make_numeric_df()
        fig = plot_feature_distributions(X_train, X_test, title="My Title")
        assert "My Title" in fig.layout.title.text
