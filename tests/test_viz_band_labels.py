"""Error-bar bands are named in the plot subtitle (KM, SHAP, hazard ratios).

When a plot draws a CI ribbon / SD whiskers, the band must be described in a
visible subtitle, with consistent wording, and only when error bars are
actually drawn. A user-supplied title/subtitle is always respected.
"""

import numpy as np
import pandas as pd

from mlsurv.viz.features import plot_hazard_ratios
from mlsurv.viz.shap import plot_shap_importance
from mlsurv.viz.survival import plot_km


def _km_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "time": rng.exponential(10, n),
        "event": rng.integers(0, 2, n).astype(bool),
        "risk_score": rng.normal(0, 1, n),
    })


def _shap_df(n_feat=5, per_feat=40, seed=0):
    rng = np.random.default_rng(seed)
    feats = [f"f{i}" for i in range(n_feat)]
    return pd.DataFrame({
        "feature": np.repeat(feats, per_feat),
        "shap_value": rng.normal(0, 1, n_feat * per_feat),
    })


def _hr_df(n_feat=5):
    feats = [f"f{i}" for i in range(n_feat)]
    return pd.DataFrame({
        "feature": feats,
        "hr": [1.5, 0.7, 1.1, 2.0, 0.5][:n_feat],
        "ci_lower": [1.2, 0.5, 0.9, 1.6, 0.3][:n_feat],
        "ci_upper": [1.9, 0.95, 1.4, 2.5, 0.8][:n_feat],
        "ci_available": [True] * n_feat,
    })


class TestKMBandSubtitle:
    def test_ci_band_named_when_show_ci(self):
        title = plot_km(_km_df(), show_ci=True).layout.title.text
        assert "95% CI" in title

    def test_no_band_when_show_ci_false(self):
        title = plot_km(_km_df(), show_ci=False).layout.title.text
        assert "95% CI" not in title

    def test_user_subtitle_preserved_over_ci_fallback(self):
        title = plot_km(_km_df(), show_ci=True, subtitle="my note").layout.title.text
        assert "my note" in title
        assert "95% CI" not in title

    def test_empty_subtitle_suppresses(self):
        # subtitle="" is explicit suppression — no subtitle line at all.
        title = plot_km(_km_df(), show_ci=True, subtitle="").layout.title.text
        assert "<sup>" not in title
        assert "95% CI" not in title

    def test_more_specific_subtitle_wins(self):
        # Risk-group stratification fills the subtitle, so the CI fallback
        # must not overwrite it even with show_ci=True.
        title = plot_km(
            _km_df(), risk_col="risk_score", n_groups=2, show_ci=True,
        ).layout.title.text
        assert "Risk group stratification" in title
        assert "95% CI" not in title


class TestSHAPBandSubtitle:
    def test_sd_band_named_on_bar_path(self):
        title = plot_shap_importance(_shap_df(), show_violin=False).layout.title.text
        assert "±SD" in title

    def test_no_band_on_violin_path(self):
        # The violin path shows the full distribution, not ±SD error bars.
        title = plot_shap_importance(_shap_df(), show_violin=True).layout.title.text
        assert "±SD" not in title

    def test_user_title_preserved(self):
        title = plot_shap_importance(
            _shap_df(), show_violin=False, title="My SHAP",
        ).layout.title.text
        assert "My SHAP" in title
        assert "±SD" in title

    def test_violin_with_user_title_has_no_band(self):
        # show_violin=True + a user title: the title passes through with no ±SD.
        title = plot_shap_importance(
            _shap_df(), show_violin=True, title="My Violin",
        ).layout.title.text
        assert "My Violin" in title
        assert "±SD" not in title


class TestKMBandSubtitleGroupedPaths:
    """The CI fallback and subtitle precedence must hold across all four KM
    render paths, not just the overall one. by_group and panel-grid (dict input)
    use a different layout builder, so they need their own coverage.
    """

    def _grouped_df(self):
        df = _km_df()
        df["grp"] = (df["risk_score"] > 0).astype(int)
        return df

    def test_by_group_ci_band(self):
        df = self._grouped_df()
        assert "95% CI" in plot_km(df, group_col="grp", show_ci=True).layout.title.text
        assert "95% CI" not in plot_km(df, group_col="grp", show_ci=False).layout.title.text

    def test_by_group_user_subtitle_preserved(self):
        # A user subtitle must be respected on the grouped path too.
        df = self._grouped_df()
        title = plot_km(df, group_col="grp", show_ci=True, subtitle="grp note").layout.title.text
        assert "grp note" in title
        assert "95% CI" not in title

    def test_panel_grid_risk_subtitle_wins(self):
        # A dict (panel grid) is always risk-stratified, so the stratification
        # subtitle outranks the lowest-priority CI fallback regardless of show_ci.
        data = {"m1": _km_df(seed=1), "m2": _km_df(seed=2)}
        for show_ci in (True, False):
            title = plot_km(data, risk_col="risk_score", show_ci=show_ci).layout.title.text
            assert "Risk group stratification" in title
            assert "95% CI" not in title

    def test_rmst_subtitle_wins_over_ci(self):
        # RMST sets the subtitle in-renderer right before the CI fallback; the
        # RMST text (not a bare CI label) must survive. (Its own text contains a
        # "95% CI" for the RMST difference, so assert on the RMST marker.)
        title = plot_km(
            _km_df(), risk_col="risk_score", n_groups=2, show_rmst=True, show_ci=True,
        ).layout.title.text
        assert "RMST(" in title

    def test_maxstat_subtitle_wins_over_ci(self):
        title = plot_km(
            _km_df(), risk_col="risk_score", break_type="maxstat", show_ci=True,
            n_permutations=20, random_state=0,
        ).layout.title.text
        assert "Maxstat" in title
        assert "95% CI" not in title


class TestHazardRatioBandSubtitle:
    def test_ci_band_named_by_default(self):
        title = plot_hazard_ratios(_hr_df()).layout.title.text
        assert "95% CI" in title

    def test_se_band_named(self):
        title = plot_hazard_ratios(_hr_df(), error_bars="se").layout.title.text
        assert "±SE" in title

    def test_band_named_in_grouped_figure(self):
        feats = _hr_df()["feature"].tolist()
        groups = {f: ("A" if i < 3 else "B") for i, f in enumerate(feats)}
        title = plot_hazard_ratios(
            _hr_df(), feature_groups=groups,
        ).layout.title.text
        assert "95% CI" in title

    def test_no_band_without_ci(self):
        df = _hr_df()
        df["ci_available"] = False
        df["ci_lower"] = df["hr"]
        df["ci_upper"] = df["hr"]
        title = plot_hazard_ratios(df).layout.title.text
        assert "95% CI" not in title
        assert "±SE" not in title

    def _groups(self):
        feats = _hr_df()["feature"].tolist()
        return {f: ("A" if i < 3 else "B") for i, f in enumerate(feats)}

    def test_grouped_se_band(self):
        # The grouped figure path is a different title branch than ungrouped.
        title = plot_hazard_ratios(
            _hr_df(), feature_groups=self._groups(), error_bars="se",
        ).layout.title.text
        assert "±SE" in title

    def test_grouped_no_band_without_ci(self):
        df = _hr_df()
        df["ci_available"] = False
        df["ci_lower"] = df["hr"]
        df["ci_upper"] = df["hr"]
        title = plot_hazard_ratios(
            df, feature_groups=self._groups(),
        ).layout.title.text
        assert "95% CI" not in title
        assert "±SE" not in title
