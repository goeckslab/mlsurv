"""Tests for viz._data helper bug fixes."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import SuggestionWarning
from mlsurv.viz._data_metrics import (
    _bootstrap_subtitle,
    _compose_metric_title,
    _metric_title_parts,
    _multi_metric_default_title,
    _multi_metric_title_parts,
    build_cv_fold_iterations,
    build_metric_data_test,
    build_multi_metric_test,
)


# ── SuggestionWarning when models excluded from CV plots ────────────


class TestCvExclusionWarnings:
    """Models excluded from CV plots should emit SuggestionWarning, not
    InternalWarning (which is suppressed by default)."""

    def test_no_cv_data_emits_suggestion_warning(self):
        """Model with no cv_metric_scores emits SuggestionWarning."""
        mock_result_good = MagicMock()
        mock_result_good.cv_metric_scores = np.array([0.7, 0.8])
        mock_result_good.scoring_metric = "harrell"
        mock_result_good.cv_results_df.return_value = pd.DataFrame(
            {"harrell": [0.7, 0.8]},
        )

        mock_result_bad = MagicMock()
        mock_result_bad.cv_metric_scores = None

        models = {"coxph": mock_result_good, "rsf": mock_result_bad}

        with pytest.warns(SuggestionWarning, match="rsf.*no CV data"):
            build_cv_fold_iterations(models, "harrell", None)

    def test_no_fold_data_for_metric_emits_suggestion_warning(self):
        """Model without fold data for the requested metric emits
        SuggestionWarning."""
        mock_result = MagicMock()
        mock_result.cv_metric_scores = np.array([0.7, 0.8])
        mock_result.scoring_metric = "harrell"
        # cv_results_df has 'harrell' but not 'ipcw'
        mock_result.cv_results_df.return_value = pd.DataFrame(
            {"harrell": [0.7, 0.8]},
        )

        models = {"coxph": mock_result}

        with pytest.raises(ValueError, match="No CV fold data"):
            # Only one model and it doesn't have ipcw — raises after warning
            with pytest.warns(SuggestionWarning, match="no CV fold data"):
                build_cv_fold_iterations(models, "ipcw", None)


# ── _bootstrap_subtitle null guard on boot.iterations ───────────────


class TestBootstrapSubtitleNullGuard:
    """_bootstrap_subtitle must not crash when boot.iterations is None."""

    def test_none_iterations(self):
        """boot exists but iterations is None → n=0, no crash."""
        boot = SimpleNamespace(iterations=None)
        result = _bootstrap_subtitle(lambda: boot, True, 95, "forest")
        assert "n=0" in result

    def test_empty_iterations(self):
        """boot exists but iterations is empty DataFrame → n=0."""
        boot = SimpleNamespace(iterations=pd.DataFrame())
        result = _bootstrap_subtitle(lambda: boot, False, 95, "raincloud")
        assert "n=0" in result

    def test_none_boot(self):
        """boot is None → n=0."""
        result = _bootstrap_subtitle(lambda: None, True, 95, "bar")
        assert "n=0" in result

    def test_valid_iterations(self):
        """Normal case: boot with iterations → correct n."""
        iters = pd.DataFrame({"iteration": [0, 0, 1, 1, 2, 2]})
        boot = SimpleNamespace(iterations=iters)
        result = _bootstrap_subtitle(lambda: boot, True, 95, "forest")
        assert "n=3" in result


# ── _multi_metric_default_title: source-aware header + band subtitle ─


class TestMultiMetricDefaultTitle:
    """Multi-metric panels get a descriptive header plus a subtitle naming the
    band actually drawn (or a point-estimate note), mirroring the single-metric
    wording. Previously only source='cv' built a subtitle; bootstrap/test/train
    fell through to the bare 'Model Performance Comparison'.
    """

    def _boot(self, n_iter=3):
        iters = pd.DataFrame({"iteration": list(range(n_iter)) * 2})
        return SimpleNamespace(iterations=iters)

    def _state(self, cv_folds=5, cv_repeats=1):
        return SimpleNamespace(cv_folds=cv_folds, cv_repeats=cv_repeats)

    def _title(self, source, **kw):
        defaults = dict(
            state=self._state(),
            find_bootstrap_fn=lambda: self._boot(),
            source=source,
            effective_style="forest",
            is_iteration_style=False,
            use_sd=False,
            ci_level=0.95,
            uncertainty_label="95% CI",
        )
        defaults.update(kw)
        return _multi_metric_default_title(**defaults)

    def test_bootstrap_forest_ci_subtitle(self):
        """Bootstrap forest: 'Bootstrap Performance' header + CI subtitle with n."""
        title = self._title("bootstrap")
        assert "Bootstrap Performance" in title
        assert "bootstrap 95% CI (n=3)" in title

    def test_bootstrap_sd_subtitle(self):
        """uncertainty='sd' is reflected as '± bootstrap SD'."""
        title = self._title("bootstrap", use_sd=True, uncertainty_label="SD")
        assert "± bootstrap SD (n=3)" in title
        assert "CI" not in title

    def test_bootstrap_ci_level_reflected(self):
        """An explicit ci_level surfaces in the subtitle (e.g. 90%)."""
        title = self._title("bootstrap", ci_level=0.90, uncertainty_label="90% CI")
        assert "bootstrap 90% CI" in title

    def test_bootstrap_iteration_distribution_title(self):
        """raincloud/boxplot bootstrap shows a distribution title (no band)."""
        title = self._title(
            "bootstrap", effective_style="raincloud", is_iteration_style=True,
            find_bootstrap_fn=lambda: self._boot(n_iter=7),
        )
        # n is read from the data, not hardcoded.
        assert title == "Bootstrap Distribution (n=7)"

    def test_bootstrap_iteration_no_bootstrap_n_zero(self):
        """No bootstrap on the iteration path → n=0, no crash (null-guard)."""
        title = self._title(
            "bootstrap", effective_style="raincloud", is_iteration_style=True,
            find_bootstrap_fn=lambda: None,
        )
        assert title == "Bootstrap Distribution (n=0)"

    def test_test_point_estimate_note(self):
        """source='test' notes a point estimate (no interval)."""
        title = self._title("test")
        assert "Test-Set Performance" in title
        assert "Test-set estimate" in title

    def test_train_point_estimate_note(self):
        """source='train' notes the apparent (train-set) estimate."""
        title = self._title("train")
        assert "Train-Set Performance" in title
        assert "Apparent (train-set) estimate" in title

    def test_cv_forest_unchanged(self):
        """CV forest subtitle is unchanged ('CV mean ± ... (k-fold)')."""
        title = self._title("cv", uncertainty_label="SD", use_sd=True)
        assert "Cross-Validation Performance" in title
        assert "CV mean ± SD (5-fold)" in title

    def test_cv_repeats_in_subtitle(self):
        """Repeated CV reports 'r× k-fold'."""
        title = self._title(
            "cv", state=self._state(cv_folds=5, cv_repeats=3),
            uncertainty_label="SD", use_sd=True,
        )
        assert "3× 5-fold" in title

    def test_cv_iteration_distribution_title(self):
        """CV raincloud/boxplot shows the fold-distribution title."""
        title = self._title(
            "cv", effective_style="raincloud", is_iteration_style=True,
        )
        assert title == "CV Fold Distribution (5-fold)"

    def test_unknown_source_returns_none(self):
        """An unrecognised source yields None (generic fallback downstream)."""
        assert self._title("bogus") is None


# ── custom title keeps the band subtitle ───────────────────────────


class TestCustomTitleKeepsSubtitle:
    """A caller-supplied ``title`` replaces only the header line; the band/source
    subtitle is still appended so the plot keeps describing its error bars.
    Mirrors the single-metric and multi-metric paths through the shared
    ``_compose_metric_title`` + ``*_title_parts`` split.
    """

    def _state(self, cv_folds=5, cv_repeats=1):
        return SimpleNamespace(cv_folds=cv_folds, cv_repeats=cv_repeats)

    def _boot(self, n_iter=3):
        iters = pd.DataFrame({"iteration": list(range(n_iter)) * 2})
        return SimpleNamespace(iterations=iters)

    def test_compose_appends_subtitle(self):
        out = _compose_metric_title("My Title", "95% CI")
        assert out.startswith("My Title<br>")
        assert "95% CI" in out

    def test_compose_header_only_when_no_subtitle(self):
        assert _compose_metric_title("My Title", None) == "My Title"

    def test_compose_none_header_returns_none(self):
        # Unknown source with no user title → generic fallback downstream.
        assert _compose_metric_title(None, "ignored") is None

    def test_single_metric_custom_header_keeps_band(self):
        # Swap the metric display name for a user title; subtitle must remain.
        _, subtitle = _metric_title_parts(
            self._state(), lambda: self._boot(), "harrell", "cv",
            use_sd=True, ci_level=0.95, style="forest",
        )
        title = _compose_metric_title("CV C-Index with Tuned GBSA", subtitle)
        assert "CV C-Index with Tuned GBSA" in title
        assert "CV mean ± SD (5-fold)" in title

    def test_multi_metric_custom_header_keeps_band(self):
        header, subtitle = _multi_metric_title_parts(
            self._state(), lambda: self._boot(), "bootstrap",
            effective_style="forest", is_iteration_style=False,
            use_sd=False, ci_level=0.95, uncertainty_label="95% CI",
        )
        assert header == "Bootstrap Performance"
        title = _compose_metric_title("My Bootstrap Comparison", subtitle)
        assert "My Bootstrap Comparison" in title
        assert "bootstrap 95% CI (n=3)" in title

    def test_default_title_matches_compose_of_parts(self):
        # The default (no user title) path is exactly compose(parts).
        parts = _multi_metric_title_parts(
            self._state(), lambda: self._boot(), "cv",
            effective_style="forest", is_iteration_style=False,
            use_sd=True, ci_level=0.95, uncertainty_label="SD",
        )
        default = _multi_metric_default_title(
            self._state(), lambda: self._boot(), "cv",
            effective_style="forest", is_iteration_style=False,
            use_sd=True, ci_level=0.95, uncertainty_label="SD",
        )
        assert default == _compose_metric_title(*parts)


# ── _title_layout: two-line titles get headroom, one-line unchanged ─


class TestTitleLayout:
    """Multi-panel titles with a `<br>` subtitle need extra top room: the bold
    header would otherwise clip at the figure's top edge (the subplot titles sit
    at the top of the plot area). One-line titles must keep the original
    `y=0.98` anchor and no padding.
    """

    def test_single_line_title_unchanged(self):
        from mlsurv.viz._metrics_multi import _title_layout

        title_dict, pad, height = _title_layout("Bootstrap Distribution (n=5)", 240)
        assert pad == 0
        assert height == 240
        assert title_dict["y"] == 0.98
        assert title_dict["yanchor"] == "top"

    def test_two_line_title_gets_pad_and_lower_y(self):
        from mlsurv.viz._metrics_multi import (
            _SUBTITLE_TITLE_PAD,
            _SUBTITLE_TITLE_HEADROOM,
            _title_layout,
        )

        title = "Bootstrap Performance<br><span>Test-set estimate</span>"
        title_dict, pad, height = _title_layout(title, 240)
        # Height and top margin both grow by the pad (plot area keeps its size).
        assert pad == _SUBTITLE_TITLE_PAD
        assert height == 240 + _SUBTITLE_TITLE_PAD
        # Title is lowered into the enlarged top margin so the header clears the
        # top edge: y leaves exactly the headroom px above it.
        assert title_dict["yanchor"] == "top"
        assert title_dict["y"] < 0.98
        expected_y = 1 - _SUBTITLE_TITLE_HEADROOM / height
        assert title_dict["y"] == pytest.approx(expected_y)


# ── build_multi_metric_test null guard on get_metric_info ───────────


class TestBuildMultiMetricTestUnknownMetric:
    """build_multi_metric_test must raise ValueError for unknown metrics,
    not AttributeError from None.concordance_column."""

    def test_unknown_metric_raises_valueerror(self):
        """Unknown metric name → clear ValueError, not AttributeError."""
        from unittest.mock import patch

        conc = pd.DataFrame({
            "model_type": ["coxph"],
            "c_index_test": [0.7],
        })
        eval_result = SimpleNamespace(concordance=conc)

        # Patch resolve_eval_result to skip EvaluationResult.combine
        with patch(
            "mlsurv.viz._data_metrics.resolve_eval_result",
            return_value=eval_result,
        ):
            state = SimpleNamespace()
            with pytest.raises(ValueError, match="Unknown metric.*nonexistent"):
                build_multi_metric_test(state, ["nonexistent"], None)


class TestBuildMetricDataTestUnknownMetric:
    """build_metric_data_test must raise ValueError for unknown metrics."""

    def test_unknown_metric_raises_valueerror(self):
        """Unknown metric in single-metric path → clear ValueError."""
        conc = pd.DataFrame({
            "model_type": ["coxph"],
            "c_index_test": [0.7],
        })
        eval_result = SimpleNamespace(concordance=conc)

        with pytest.raises(ValueError, match="Unknown metric.*bogus"):
            build_metric_data_test(lambda: eval_result, "bogus", None)


# ── single source of truth for metric direction ───────────────────


class TestIsLowerBetter:
    """`_is_lower_better` is consolidated to one definition in
    `_metrics_helpers`; it covers registry keys *plus* the non-registry
    short-name aliases ``brier``/``brier_score`` and guards falsy input."""

    @pytest.mark.parametrize(
        "metric, expected",
        [
            ("integrated_brier", True),   # registry key
            ("cal_intercept", True),      # registry key
            ("brier", True),              # short-name alias (not a registry key)
            ("brier_score", True),        # short-name alias (not a registry key)
            ("harrell", False),           # higher-is-better registry key
            ("ipcw", False),              # higher-is-better registry key
            ("ctd", False),               # higher-is-better registry key
            ("cal_slope", False),         # calibration key, but higher-is-better
            ("gnd", False),               # higher-is-better registry key
            (None, False),                # guard: None
            ("", False),                  # guard: empty string
        ],
    )
    def test_direction(self, metric, expected):
        from mlsurv.viz._metrics_helpers import _is_lower_better

        assert _is_lower_better(metric) is expected

    def test_metrics_facade_reuses_same_object(self):
        """The name re-exported from `mlsurv.viz.metrics` must be the *same*
        object — locks in the consolidation so re-divergence fails CI."""
        from mlsurv.viz import metrics
        from mlsurv.viz._metrics_helpers import _is_lower_better

        assert metrics._is_lower_better is _is_lower_better

    def test_heatmap_zscore_flips_brier_alias_column(self):
        """Heatmap call site: consolidating onto the superset means a column
        literally named ``brier``/``brier_score`` now gets its z-score sign
        flipped (lower-is-better), where the old registry-only helper left it
        unflipped. Documents the one real behavior change of consolidating
        onto the superset direction map."""
        from mlsurv.viz.metrics import _zscore_with_direction_flip

        # Ascending raw values: model B is "worse" on a lower-is-better metric.
        z = pd.DataFrame({"brier": [0.1, 0.3]}, index=["model_a", "model_b"])
        nan_mask = z.isna()
        out = _zscore_with_direction_flip(z.copy(), ["brier"], nan_mask)
        # After the flip, the lower raw value (model_a) has the higher z-score.
        assert out.loc["model_a", "brier"] > out.loc["model_b", "brier"]


# ── renderer/multi sorting routed through the canonical helper ──────


class TestDirectionRoutedThroughHelper:
    """`_metrics_renderers` and `_metrics_multi` now decide metric direction
    via the canonical `_is_lower_better` helper instead of raw
    `LOWER_IS_BETTER` set membership. Behavior is unchanged for registry-key
    metrics; the previously-latent `brier`/`brier_score` short-name aliases
    now sort lower-is-better in these paths too."""

    def _boxplot_iterations(self):
        """Two models: 'a' has the lower median value, 'b' the higher."""
        return pd.DataFrame({
            "model": ["a", "a", "b", "b"],
            "value": [0.10, 0.12, 0.30, 0.32],
        })

    def _summary_frame(self):
        """One row per model: 'a' lower point estimate, 'b' higher."""
        return pd.DataFrame({"model": ["a", "b"], "point": [0.10, 0.30]})

    @pytest.mark.parametrize(
        "metric, lower_first",
        [
            ("integrated_brier", True),   # registry key — lower is better
            ("brier", True),              # short-name alias — the alias-sorting case
            ("harrell", False),           # higher is better
        ],
    )
    def test_boxplot_prep_sort_direction(self, metric, lower_first):
        """`_prepare_boxplot_data` sorts models by median ascending when the
        metric is lower-is-better (best/lowest first), descending otherwise."""
        from mlsurv.viz._metrics_renderers import _prepare_boxplot_data

        _df, models, _names = _prepare_boxplot_data(
            self._boxplot_iterations(),
            model_col="model", value_col="value",
            metric_col="metric_name",  # absent → no metric filtering
            metric=metric,
        )
        # 'a' is the lower-median model.
        expected = ["a", "b"] if lower_first else ["b", "a"]
        assert models == expected

    @pytest.mark.parametrize(
        "metric, lower_first",
        [
            ("integrated_brier", True),
            ("brier", True),
            ("harrell", False),
        ],
    )
    def test_summary_models_sort_direction(self, metric, lower_first):
        """`_resolve_summary_models` (multi-panel forest) orders models by
        point estimate ascending for lower-is-better metrics."""
        from mlsurv.viz._metrics_multi import _resolve_summary_models

        models, _names, _colors = _resolve_summary_models(
            self._summary_frame(),
            model_col="model", point_col="point",
            sort="value", palette=None, first_metric=metric,
        )
        # 'a' is the lower-point model.
        expected = ["a", "b"] if lower_first else ["b", "a"]
        assert models == expected

    def test_modules_no_longer_reference_raw_set(self):
        """Source-of-truth guard: neither module keeps the raw
        `LOWER_IS_BETTER` symbol, and both reuse the *same* canonical helper
        object — so any future re-divergence fails CI (cf.
        `test_metrics_facade_reuses_same_object`)."""
        import mlsurv.viz._metrics_multi as multi
        import mlsurv.viz._metrics_renderers as renderers
        from mlsurv.viz._metrics_helpers import _is_lower_better

        assert "LOWER_IS_BETTER" not in vars(renderers)
        assert "LOWER_IS_BETTER" not in vars(multi)
        assert renderers._is_lower_better is _is_lower_better
        assert multi._is_lower_better is _is_lower_better


class TestBoxplotJitterRngIsolation:
    """Rendering a boxplot must not disturb the caller's global NumPy RNG.

    The jitter previously called ``np.random.seed(42 + idx)``, which reseeds the
    process-global generator and silently changes any random draws the caller
    makes afterward. The fix uses a local ``RandomState`` instead.
    """

    def _boxplot_iterations(self):
        return pd.DataFrame({
            "model": ["a", "a", "a", "b", "b", "b"],
            "value": [0.60, 0.62, 0.64, 0.70, 0.72, 0.74],
        })

    def test_global_rng_state_unchanged(self):
        from mlsurv.viz._metrics_renderers import _plot_metric_boxplot

        np.random.seed(123)
        before = np.random.get_state()

        _plot_metric_boxplot(
            self._boxplot_iterations(),
            model_col="model", metric_col="metric_name", value_col="value",
            metric="harrell", title="t", show_reference_line=False,
            orientation="horizontal",
        )

        after = np.random.get_state()
        # State tuple: (algo, key-array, pos, has_gauss, cached_gauss)
        assert before[0] == after[0]
        assert np.array_equal(before[1], after[1])
        assert before[2] == after[2]
