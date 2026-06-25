"""Tests for mlsurv.constants metric registry and helper functions.

Covers MetricInfo dataclass, METRIC_REGISTRY completeness, and the
get_metric_* helper functions including fallback paths for unknown metrics.
The module is at 92% — missing lines are in the fallback paths of
get_metric_short_name() and get_metric_table_name() for unknown metrics.
"""

import pytest

from mlsurv.constants import (
    ARTIFACT_VERSION,
    MetricInfo,
    METRIC_REGISTRY,
    get_metric_display_name,
    get_metric_info,
    get_metric_short_name,
    get_metric_table_name,
    get_concordance_column,
)


class TestMetricRegistry:
    """Tests for METRIC_REGISTRY completeness and consistency."""

    def test_metric_info_is_frozen(self):
        """MetricInfo dataclass should be immutable."""
        info = METRIC_REGISTRY['harrell']
        with pytest.raises(AttributeError):
            info.key = "modified"

    def test_each_metric_has_concordance_column(self):
        """Every metric should have a non-empty concordance_column."""
        for key, info in METRIC_REGISTRY.items():
            assert info.concordance_column, f"{key} missing concordance_column"


class TestGetMetricDisplayName:
    """Tests for get_metric_display_name() including fallback."""

    def test_known_metric(self):
        assert get_metric_display_name('harrell') == "Harrell's C-index"
        assert get_metric_display_name('integrated_brier') == "Integrated Brier Score"

    def test_unknown_metric_fallback(self):
        """Unknown metric should return title-cased name."""
        assert get_metric_display_name('custom_metric') == "Custom_Metric"


class TestGetMetricInfo:
    """Tests for get_metric_info() including None fallback."""

    def test_known_metric(self):
        info = get_metric_info('harrell')
        assert isinstance(info, MetricInfo)
        assert info.key == 'harrell'

    def test_unknown_metric_returns_none(self):
        """Unknown metric should return None."""
        assert get_metric_info('nonexistent') is None


class TestGetMetricShortName:
    """Tests for get_metric_short_name() with asterisk marking."""

    def test_known_metric(self):
        assert get_metric_short_name('harrell') == "Harrell's C"

    def test_lower_is_better_asterisk(self):
        """IBS should have asterisk marking."""
        name = get_metric_short_name('integrated_brier', with_asterisk=True)
        assert name.endswith(" *")

    def test_ideal_value_annotation(self):
        """Calibration slope should show ideal value annotation."""
        name = get_metric_short_name('cal_slope', with_asterisk=True)
        assert "(=1)" in name
        name = get_metric_short_name('cal_intercept', with_asterisk=True)
        assert "(=0)" in name

    def test_no_asterisk(self):
        """with_asterisk=False should omit asterisk."""
        name = get_metric_short_name('integrated_brier', with_asterisk=False)
        assert not name.endswith(" *")

    def test_unknown_metric_fallback(self):
        """Unknown metric should return title-cased name."""
        assert get_metric_short_name('unknown') == "Unknown"


class TestGetMetricTableName:
    """Tests for get_metric_table_name() compact names."""

    def test_known_metric(self):
        assert get_metric_table_name('harrell') == "Harrell C"

    def test_lower_is_better_asterisk(self):
        name = get_metric_table_name('integrated_brier', with_asterisk=True)
        assert name.endswith(" *")

    def test_unknown_metric_fallback(self):
        """Unknown metric should return title-cased name."""
        assert get_metric_table_name('custom') == "Custom"


class TestGetConcordanceColumn:
    """Tests for get_concordance_column() column name resolution."""

    def test_known_metric(self):
        assert get_concordance_column('harrell') == 'c_index_test'
        assert get_concordance_column('integrated_brier') == 'integrated_brier_test'
        assert get_concordance_column('cal_slope') == 'cal_slope_test'
        assert get_concordance_column('cal_intercept') == 'cal_intercept_test'
        assert get_concordance_column('gnd') == 'gnd_test'

    def test_unknown_metric_returns_none(self):
        assert get_concordance_column('nonexistent') is None


class TestArtifactVersion:
    """Tests for the ARTIFACT_VERSION constant."""

    def test_artifact_version_is_positive_int(self):
        """ARTIFACT_VERSION should be a positive integer."""
        assert isinstance(ARTIFACT_VERSION, int)
        assert ARTIFACT_VERSION >= 1
