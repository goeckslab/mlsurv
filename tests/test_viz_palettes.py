"""Tests for mlsurv.viz.palettes color and layout utilities.

Covers get_manuscript_layout(), get_model_color(), get_model_display_name(),
get_disease_color(), get_risk_colors(), hex_to_rgba(), the _LazyDict
auto-population mechanism, and the palette override API (set_palette,
reset_palette, NAMED_PALETTES, per-call palette).
"""

import pytest

from mlsurv.viz._palettes import (
    MANUSCRIPT_DEFAULTS,
    NAMED_PALETTES,
    REFERENCE_PALETTE,
    build_benchmark_palette,
    get_manuscript_layout,
    get_model_color,
    get_model_display_name,
    get_disease_color,
    get_reference_color,
    get_risk_colors,
    hex_to_rgba,
    set_palette,
    reset_palette,
    _resolve_palette,
    MODEL_COLORS,
    DISEASE_COLORS,
    RISK_QUARTILE_COLORS,
    _LazyDict,
)
import mlsurv.viz._palettes as _palettes


class TestGetManuscriptLayout:
    """Tests for get_manuscript_layout() figure sizing.

    Correct layout ensures publication-quality figures. Wrong dimensions
    cause figures to be rejected by journals.
    """

    def test_single_column(self):
        """Single column should use width_single."""
        layout = get_manuscript_layout(width="single")
        assert layout['width'] == MANUSCRIPT_DEFAULTS['width_single']

    def test_double_column(self):
        """Double column should use width_double."""
        layout = get_manuscript_layout(width="double")
        assert layout['width'] == MANUSCRIPT_DEFAULTS['width_double']

    def test_height_variants(self):
        """All three height options should map correctly."""
        small = get_manuscript_layout(height="small")
        medium = get_manuscript_layout(height="medium")
        large = get_manuscript_layout(height="large")

        assert small['height'] < medium['height'] < large['height']

    def test_title_included(self):
        """Title should be included when provided."""
        layout = get_manuscript_layout(title="Test Title")
        assert layout['title']['text'] == "Test Title"

    def test_no_title(self):
        """No title param should result in no title key."""
        layout = get_manuscript_layout()
        assert 'title' not in layout

    def test_kwargs_override(self):
        """Extra kwargs should override defaults."""
        layout = get_manuscript_layout(width="single", template="plotly_white")
        assert layout['template'] == "plotly_white"

    def test_has_margin(self):
        """Layout should include margin configuration."""
        layout = get_manuscript_layout()
        assert 'margin' in layout
        assert 'l' in layout['margin']

    def test_has_font(self):
        """Layout should include font configuration."""
        layout = get_manuscript_layout()
        assert 'font' in layout

    def test_unknown_height_defaults_to_medium(self):
        """Unknown height string should fall back to medium."""
        layout = get_manuscript_layout(height="unknown")
        expected = get_manuscript_layout(height="medium")
        assert layout['height'] == expected['height']


class TestLazyDict:
    """Tests for _LazyDict lazy-loading mechanism."""

    def test_loads_on_first_access(self):
        """Loader should be called on first access."""
        d = _LazyDict(lambda: {'a': 1, 'b': 2})
        assert d['a'] == 1
        assert d._initialized is True

    def test_contains_triggers_load(self):
        """'in' operator should trigger lazy load."""
        d = _LazyDict(lambda: {'key': 'value'})
        assert 'key' in d

    def test_get_with_default(self):
        """get() should work with defaults."""
        d = _LazyDict(lambda: {'a': 1})
        assert d.get('a') == 1
        assert d.get('missing', 'default') == 'default'

    def test_items_keys_values(self):
        """dict view methods should work."""
        d = _LazyDict(lambda: {'x': 10, 'y': 20})
        assert list(d.keys()) == ['x', 'y']
        assert list(d.values()) == [10, 20]
        assert list(d.items()) == [('x', 10), ('y', 20)]

    def test_iter(self):
        """Iteration should work."""
        d = _LazyDict(lambda: {'a': 1, 'b': 2})
        assert set(d) == {'a', 'b'}

    def test_len(self):
        """len() should work."""
        d = _LazyDict(lambda: {'a': 1, 'b': 2, 'c': 3})
        assert len(d) == 3


class TestGetModelColor:
    """Tests for get_model_color() color lookup."""

    def test_known_model(self):
        """Known model should return its registered color."""
        color = get_model_color('coxph')
        assert color.startswith('#')

    def test_unknown_model_fallback(self):
        """Unknown model should return fallback gray."""
        color = get_model_color('totally_unknown_model_xyz')
        assert isinstance(color, str)
        assert color.startswith('#')

    def test_case_insensitive(self):
        """Lookup should be case-insensitive."""
        color_lower = get_model_color('rsf')
        color_upper = get_model_color('RSF')
        assert color_lower == color_upper

    def test_model_colors_registry_has_core_models(self):
        """MODEL_COLORS registry contains the core algorithm keys."""
        for key in ('coxph', 'rsf', 'coxnet'):
            assert key in MODEL_COLORS


class TestGetModelDisplayName:
    """Tests for get_model_display_name() formatting."""

    def test_known_model(self):
        """Known model should return formatted name."""
        name = get_model_display_name('coxph')
        assert name  # Non-empty

    def test_unknown_model_uppercased(self):
        """Unknown model should return uppercased name."""
        name = get_model_display_name('mymodel')
        assert name == 'MYMODEL'

    def test_resolved_tuned_label_passthrough(self):
        """An already-resolved 'RSF (tuned)' label passes through unchanged."""
        assert get_model_display_name('RSF (tuned)') == 'RSF (tuned)'


class TestGetDiseaseColor:
    """Tests for get_disease_color() color lookup."""

    def test_known_disease(self):
        color = get_disease_color('breast')
        assert color == DISEASE_COLORS['breast']

    def test_partial_match(self):
        """Partial matches should work (e.g., 'breast cancer' matches 'breast')."""
        color = get_disease_color('breast cancer')
        assert color == DISEASE_COLORS['breast']

    def test_unknown_disease_fallback(self):
        """Unknown disease should return default color."""
        color = get_disease_color('unknown_disease')
        assert color == DISEASE_COLORS['default']

    def test_disease_colors_registry_has_expected_keys(self):
        """DISEASE_COLORS registry contains expected disease keys + default."""
        for key in ('breast', 'lung', 'default'):
            assert key in DISEASE_COLORS


class TestGetRiskColors:
    """Tests for get_risk_colors() risk group color generation."""

    def test_two_groups(self):
        colors = get_risk_colors(2)
        assert len(colors) == 2

    def test_three_groups(self):
        colors = get_risk_colors(3)
        assert len(colors) == 3

    def test_four_groups(self):
        colors = get_risk_colors(4)
        assert len(colors) == 4
        assert colors == list(RISK_QUARTILE_COLORS.values())

    def test_five_groups_gradient(self):
        """Non-standard group count should generate gradient."""
        colors = get_risk_colors(5)
        assert len(colors) == 5
        assert all(c.startswith('#') for c in colors)
        # Pin the anchor positions so the gradient's roundtrip through the
        # extracted _hex_to_rgb/_rgb_to_hex helpers can't silently swap a
        # channel after the refactor. Endpoints + midpoint are the three anchors.
        assert colors[0] == '#2166ac'   # low-risk anchor
        assert colors[2] == '#fee090'   # midpoint anchor
        assert colors[4] == '#b2182b'   # high-risk anchor


class TestHexToRgba:
    """Tests for hex_to_rgba() color conversion."""

    def test_basic_conversion(self):
        result = hex_to_rgba('#1f77b4')
        assert result == 'rgba(31, 119, 180, 1.0)'

    def test_with_alpha(self):
        result = hex_to_rgba('#1f77b4', alpha=0.5)
        assert result == 'rgba(31, 119, 180, 0.5)'

    def test_black(self):
        result = hex_to_rgba('#000000')
        assert result == 'rgba(0, 0, 0, 1.0)'

    def test_white(self):
        result = hex_to_rgba('#ffffff')
        assert result == 'rgba(255, 255, 255, 1.0)'


class TestNamedPalettes:
    """Tests for NAMED_PALETTES registry."""

    def test_okabe_ito_present(self):
        """Okabe-Ito palette should be in NAMED_PALETTES."""
        assert "okabe_ito" in NAMED_PALETTES

    def test_tableau10_present(self):
        """Tableau 10 palette should be in NAMED_PALETTES."""
        assert "tableau10" in NAMED_PALETTES

    def test_all_models_covered(self):
        """Both named palettes should cover the standard model set."""
        expected = {
            "coxph", "coxnet", "rsf", "gbsa", "fssvm",
            "deepsurv", "deephit", "weibull_aft", "xgbcox",
            "ensemble", "baseline",
        }
        for name, palette in NAMED_PALETTES.items():
            assert expected.issubset(palette.keys()), (
                f"Palette '{name}' missing keys: {expected - palette.keys()}"
            )


class TestSetPalette:
    """Tests for set_palette() / reset_palette() global override."""

    def setup_method(self):
        """Ensure clean state before each test."""
        reset_palette()

    def teardown_method(self):
        """Ensure clean state after each test."""
        reset_palette()

    def test_set_named_palette(self):
        """set_palette('okabe_ito') should activate the named palette."""
        set_palette("okabe_ito")
        assert get_model_color("coxph") == "#0072B2"

    def test_set_custom_dict(self):
        """set_palette(dict) should use custom overrides."""
        set_palette({"coxph": "#FF0000"})
        assert get_model_color("coxph") == "#FF0000"

    def test_reset_palette(self):
        """reset_palette() should revert to MODEL_COLORS defaults."""
        original = get_model_color("coxph")
        set_palette({"coxph": "#FF0000"})
        assert get_model_color("coxph") == "#FF0000"
        reset_palette()
        assert get_model_color("coxph") == original

    def test_invalid_palette_name_raises(self):
        """Unknown palette name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown palette"):
            set_palette("nonexistent_palette")

    def test_set_empty_dict(self):
        """set_palette({}) should be accepted; get_model_color falls through."""
        set_palette({})
        assert _palettes._active_palette == {}
        # Should still get default colors since empty dict has no overrides
        assert get_model_color("coxph") == MODEL_COLORS["coxph"]

    def test_set_palette_copies_named(self):
        """set_palette should copy the named palette, not alias it."""
        set_palette("okabe_ito")
        pal = _palettes._active_palette
        pal["coxph"] = "#FFFFFF"
        # Original NAMED_PALETTES should be unmodified
        assert NAMED_PALETTES["okabe_ito"]["coxph"] == "#0072B2"


class TestPerCallPalette:
    """Tests for per-call palette parameter on get_model_color()."""

    def setup_method(self):
        reset_palette()

    def teardown_method(self):
        reset_palette()

    def test_per_call_dict_override(self):
        """Per-call dict palette should override defaults."""
        color = get_model_color("coxph", palette={"coxph": "#ABCDEF"})
        assert color == "#ABCDEF"

    def test_per_call_named_override(self):
        """Per-call named palette string should work."""
        color = get_model_color("coxph", palette="tableau10")
        assert color == "#1f77b4"

    def test_per_call_does_not_affect_global(self):
        """Per-call palette should not change module-level state."""
        get_model_color("coxph", palette={"coxph": "#FF0000"})
        assert _palettes._active_palette is None

    def test_per_call_overrides_global(self):
        """Per-call palette takes priority over set_palette()."""
        set_palette({"coxph": "#111111"})
        color = get_model_color("coxph", palette={"coxph": "#222222"})
        assert color == "#222222"

    def test_per_call_invalid_name_raises(self):
        """Invalid palette name in per-call should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown palette"):
            get_model_color("coxph", palette="bad_name")

    def test_per_call_falls_through_to_defaults(self):
        """If per-call palette doesn't have the model, fall through."""
        color = get_model_color("coxph", palette={"rsf": "#FF0000"})
        # Should return the MODEL_COLORS default for coxph
        assert color == MODEL_COLORS["coxph"]

    def test_per_call_partial_skips_global(self):
        """Per-call palette missing a model should skip global, fall to MODEL_COLORS."""
        set_palette({"coxph": "#111111", "rsf": "#222222"})
        # Per-call palette only has rsf; coxph should come from MODEL_COLORS, NOT global
        color = get_model_color("coxph", palette={"rsf": "#333333"})
        assert color == MODEL_COLORS["coxph"]  # NOT "#111111"


class TestDefaultColorsAreOkabeIto:
    """Verify that model class defaults now use the Okabe-Ito palette."""

    def test_coxph_default_is_okabe_ito(self):
        """CoxPH default should be Okabe-Ito blue."""
        assert get_model_color("coxph") == "#0072B2"

    def test_rsf_default_is_okabe_ito(self):
        """RSF default should be Okabe-Ito vermillion."""
        assert get_model_color("rsf") == "#D55E00"

    def test_ensemble_default(self):
        """Ensemble color should be teal."""
        assert get_model_color("ensemble") == "#44AA99"

    def test_baseline_default(self):
        """Baseline color should be gray."""
        assert get_model_color("baseline") == "#999999"

    def test_model_defaults_match_okabe_ito_palette(self):
        """MODEL_COLORS (from model classes) should match NAMED_PALETTES['okabe_ito']."""
        okabe = NAMED_PALETTES["okabe_ito"]
        for model_name, expected_color in okabe.items():
            actual = MODEL_COLORS.get(model_name)
            assert actual == expected_color, (
                f"MODEL_COLORS['{model_name}'] is {actual}, "
                f"expected {expected_color} from okabe_ito palette"
            )


class TestPaletteAPIHardening:
    """Tests for palette copy-safety and type guards."""

    def setup_method(self):
        reset_palette()

    def teardown_method(self):
        reset_palette()

    def test_resolve_palette_returns_copy(self):
        """Mutating _resolve_palette return must not affect NAMED_PALETTES."""
        result = _resolve_palette("okabe_ito")
        original_coxph = NAMED_PALETTES["okabe_ito"]["coxph"]
        result["coxph"] = "#FFFFFF"
        assert NAMED_PALETTES["okabe_ito"]["coxph"] == original_coxph

    def test_resolve_palette_active_returns_copy(self):
        """Mutating _resolve_palette(None) return must not affect module state."""
        set_palette("okabe_ito")
        result = _resolve_palette(None)
        result["coxph"] = "#FFFFFF"
        # _resolve_palette(None) returns its own copy, so check via get_model_color
        assert get_model_color("coxph") != "#FFFFFF"

    def test_set_palette_rejects_non_dict_non_str(self):
        """set_palette(42) should raise TypeError."""
        with pytest.raises(TypeError, match="str or dict"):
            set_palette(42)

    def test_set_palette_rejects_list(self):
        """set_palette([...]) should raise TypeError."""
        with pytest.raises(TypeError, match="str or dict"):
            set_palette(["#FF0000", "#00FF00"])

    def test_resolve_palette_rejects_non_dict_non_str(self):
        """_resolve_palette(42) should raise TypeError."""
        with pytest.raises(TypeError, match="str or dict"):
            _resolve_palette(42)

    def test_case_normalization_of_custom_keys(self):
        """Custom palette with mixed-case keys should normalize to lowercase."""
        color = get_model_color("coxph", palette={"CoxPH": "#FF0000"})
        assert color == "#FF0000"


class TestBuildBenchmarkPalette:
    """Tests for build_benchmark_palette()."""

    def test_models_get_model_colors(self):
        """Each model name maps through get_model_color."""
        palette = build_benchmark_palette(["coxph", "rsf"], [])
        assert palette["coxph"] == get_model_color("coxph")
        assert palette["rsf"] == get_model_color("rsf")

    def test_references_get_indexed_colors(self):
        """Each reference name maps through get_reference_color with index."""
        refs = ["Ref A", "Ref B"]
        palette = build_benchmark_palette([], refs)
        assert palette["Ref A"] == get_reference_color("Ref A", 0)
        assert palette["Ref B"] == get_reference_color("Ref B", 1)

    def test_empty_lists(self):
        """Empty inputs produce empty dict."""
        assert build_benchmark_palette([], []) == {}

    def test_models_only(self):
        """Empty reference list returns only model colors."""
        palette = build_benchmark_palette(["coxph"], [])
        assert len(palette) == 1
        assert "coxph" in palette

    def test_references_only(self):
        """Empty model list returns only reference colors."""
        palette = build_benchmark_palette([], ["Ref A"])
        assert len(palette) == 1
        assert "Ref A" in palette

    def test_reference_index_cycling(self):
        """More references than REFERENCE_PALETTE length cycles correctly."""
        n = len(REFERENCE_PALETTE) + 2
        refs = [f"Ref {i}" for i in range(n)]
        palette = build_benchmark_palette([], refs)
        assert len(palette) == n
        # First ref wraps around after palette length
        assert palette["Ref 0"] == palette[f"Ref {len(REFERENCE_PALETTE)}"]

    def test_overlapping_name_reference_wins(self):
        """When a name appears in both lists, reference color takes precedence."""
        palette = build_benchmark_palette(["coxph"], ["coxph"])
        assert palette["coxph"] == get_reference_color("coxph", 0)
        assert len(palette) == 1

    def test_returns_fresh_dict(self):
        """Each call returns a new dict object."""
        p1 = build_benchmark_palette(["coxph"], ["Ref A"])
        p2 = build_benchmark_palette(["coxph"], ["Ref A"])
        assert p1 == p2
        assert p1 is not p2
