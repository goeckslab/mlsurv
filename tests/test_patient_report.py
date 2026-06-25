"""Tests for SurvivalLearner.patient_report() — HTML content."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv.utils import SurvivalArray


def _tiny_data():
    rng = np.random.RandomState(0)
    n = 60
    X = pd.DataFrame({
        "age": rng.uniform(30, 80, n),
        "stage": rng.randint(1, 5, n).astype(float),
        "biomarker": rng.randn(n),
    })
    times = rng.exponential(2.0, n).clip(0.1, 10)
    events = rng.binomial(1, 0.7, n).astype(bool)
    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[("event", bool), ("time", float)],
    ))
    return X, y


@pytest.fixture(scope="module")
def trained_for_report():
    X, y = _tiny_data()
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, random_state=0, time_unit="months")
    learner.train("coxph")
    return learner, X


@pytest.mark.requires_shap
def test_patient_report_includes_ci_methodology_when_bootstrap_run(
    trained_for_report,
):
    learner, X = trained_for_report
    html = learner.patient_report(
        X.head(2), patient=0, ci=True, ci_band=True,
        n_bootstrap=4, random_state=0,
    )
    assert 'class="methodology"' in html
    assert "percentile-bootstrap" in html
    assert "4 bootstrap iterations" in html
    assert "95%" in html
    assert "training-data variation" in html


@pytest.mark.requires_shap
def test_patient_report_omits_methodology_when_ci_disabled(trained_for_report):
    learner, X = trained_for_report
    html = learner.patient_report(
        X.head(2), patient=0, ci=False,
    )
    assert 'class="methodology"' not in html
    assert "percentile-bootstrap" not in html


@pytest.mark.requires_shap
def test_patient_report_methodology_drops_band_phrase_when_ci_band_off(
    trained_for_report,
):
    learner, X = trained_for_report
    html = learner.patient_report(
        X.head(2), patient=0, ci=True, ci_band=False,
        n_bootstrap=4, random_state=0,
    )
    assert 'class="methodology"' in html
    assert "percentile-bootstrap" in html
    # Brackets exist (median CI), band does not -> wording should reflect that.
    assert "Shaded band" not in html
    assert "[lo&ndash;hi] brackets show" in html


def test_ci_methodology_html_renders_non_default_ci_level():
    """Non-default ci_level should change the percentage and quantile bounds.

    patient_report() does not expose ci_level, so exercise the helper
    directly with a stub PredictionResult-like object.
    """
    from mlsurv.learner._patient_report import _ci_methodology_html

    class _Stub:
        _bootstrap_medians = np.zeros((7, 1))
        _bootstrap_surv_probs = None
        _bootstrap_rmst = None
        _ci_level = 0.80
        _ci_pct = "80"  # mirrors PredictionResult._ci_pct
        survival_function_ci = None

    html = _ci_methodology_html(_Stub())
    assert "80%" in html
    assert "0.100" in html
    assert "0.900" in html
    assert "7 bootstrap iterations" in html


def test_train_percentile_cell_omits_nan_for_missing_value():
    """NaN patient value falls back to bare range."""
    from mlsurv.learner._patient_report_html import _train_percentile_cell

    training_X = pd.DataFrame({"flag": [0.0, 1.0, 0.0, 1.0, 0.0]})

    nan_cell = _train_percentile_cell(
        pval_num=float("nan"), lo=0, hi=1,
        training_X=training_X, feat="flag",
    )
    assert nan_cell == "[0, 1]"
    assert "nan" not in nan_cell.lower()
    assert "%" not in nan_cell

    for non_finite in (float("inf"), float("-inf")):
        cell = _train_percentile_cell(
            pval_num=non_finite, lo=0, hi=1,
            training_X=training_X, feat="flag",
        )
        assert cell == "[0, 1]"

    ok_cell = _train_percentile_cell(
        pval_num=1.0, lo=0, hi=1,
        training_X=training_X, feat="flag",
    )
    assert "%" in ok_cell
    assert "[0, 1]" in ok_cell

    # All-NaN training column: percentileofscore returns NaN, which must
    # also fall back to the bare range rather than render "nan% [...]".
    all_nan = pd.DataFrame({"flag": [float("nan")] * 4})
    nan_col_cell = _train_percentile_cell(
        pval_num=0.5, lo=0, hi=1,
        training_X=all_nan, feat="flag",
    )
    assert nan_col_cell == "[0, 1]"


def test_feature_table_html_no_nan_percent_for_imputed_row():
    """End-to-end: feature table HTML never contains the literal 'nan%'."""
    from mlsurv.learner._patient_report_html import _feature_table_html

    class _Expl:
        shap_values = pd.DataFrame({
            "feature": ["flag", "age"],
            "feature_value": [float("nan"), 55.0],
            "shap_value": [0.4, -0.2],
            "abs_shap_value": [0.4, 0.2],
        })

    class _Pred:
        feature_bounds = {"flag": (0, 1), "age": (30.0, 80.0)}
        input_X = pd.DataFrame({"flag": [float("nan")], "age": [55.0]})
        training_X = pd.DataFrame({
            "flag": [0.0, 1.0, 0.0, 1.0, 0.0],
            "age": [40.0, 50.0, 60.0, 70.0, 45.0],
        })
        imputation_mask = pd.DataFrame({"flag": [True], "age": [False]})

    html = _feature_table_html(_Pred(), _Expl(), patient=0)
    assert "nan%" not in html.lower()
    assert "[0, 1]" in html
    # The Patient value cell for an imputed feature must not render the
    # literal string "nan"; it should show the missing-value sentinel instead.
    assert "<td>nan</td>" not in html.lower()
    assert "—" in html


def test_feature_table_html_hides_zero_shap_features():
    """Selector-dropped raw features (abs_shap_value == 0) are hidden
    from the table, but real contributors remain."""
    from mlsurv.learner._patient_report_html import _feature_table_html

    class _Expl:
        shap_values = pd.DataFrame({
            "feature": ["age", "biomarker", "dropped1", "dropped2"],
            "feature_value": [55.0, 1.2, 0.0, 0.0],
            "shap_value": [0.4, -0.2, 0.0, 0.0],
            "abs_shap_value": [0.4, 0.2, 0.0, 0.0],
        })

    class _Pred:
        feature_bounds = {}
        input_X = pd.DataFrame({
            "age": [55.0], "biomarker": [1.2],
            "dropped1": [0.0], "dropped2": [0.0],
        })
        training_X = input_X
        imputation_mask = pd.DataFrame({
            "age": [False], "biomarker": [False],
            "dropped1": [False], "dropped2": [False],
        })

    html = _feature_table_html(_Pred(), _Expl(), patient=0)
    assert "<td>age</td>" in html
    assert "<td>biomarker</td>" in html
    assert "<td>dropped1</td>" not in html
    assert "<td>dropped2</td>" not in html


def test_feature_table_html_all_zero_falls_back():
    """If every feature is zero-SHAP, the table still renders rows
    rather than collapsing to an empty body."""
    from mlsurv.learner._patient_report_html import _feature_table_html

    class _Expl:
        shap_values = pd.DataFrame({
            "feature": ["age", "biomarker"],
            "feature_value": [55.0, 1.2],
            "shap_value": [0.0, 0.0],
            "abs_shap_value": [0.0, 0.0],
        })

    class _Pred:
        feature_bounds = {}
        input_X = pd.DataFrame({"age": [55.0], "biomarker": [1.2]})
        training_X = input_X
        imputation_mask = pd.DataFrame({"age": [False], "biomarker": [False]})

    html = _feature_table_html(_Pred(), _Expl(), patient=0)
    assert "<td>age</td>" in html
    assert "<td>biomarker</td>" in html


def test_safe_float_str_renders_nan_as_missing_sentinel():
    """``_safe_float_str`` returns the missing-value sentinel for non-finite
    inputs so the Patient value cell never renders 'nan'/'inf'."""
    from mlsurv.learner._patient_report_html import _safe_float_str

    num, disp = _safe_float_str(float("nan"))
    assert num is None
    assert disp == "—"

    for non_finite in (float("inf"), float("-inf")):
        num_nf, disp_nf = _safe_float_str(non_finite)
        assert num_nf is None
        assert disp_nf == "—"

    num_ok, disp_ok = _safe_float_str(1.5)
    assert num_ok == 1.5
    assert disp_ok == "1.5"

    # Non-coercible strings still get the escaped repr, not the sentinel.
    num_str, disp_str = _safe_float_str("category_a")
    assert num_str is None
    assert disp_str == "category_a"


def test_resolve_patient_value_returns_sentinel_for_missing_inputs():
    """``_resolve_patient_value`` normalizes every missing-input branch to
    the em-dash sentinel, never the literal "nan"."""
    from mlsurv.learner._patient_report_html import _resolve_patient_value

    nan = float("nan")
    input_X = pd.DataFrame({
        "x_num": [nan],
        "x_cat": ["category_a"],
    })

    # raw NaN, feat_val NaN: don't fall back to feat_val (would render "nan").
    num, disp = _resolve_patient_value("x_num", nan, input_X, patient=0)
    assert num is None
    assert disp == "—"

    # raw NaN, feat_val None: still the sentinel.
    num, disp = _resolve_patient_value("x_num", None, input_X, patient=0)
    assert num is None
    assert disp == "—"

    # Non-coercible raw + NaN feat_val: the fallback also normalizes NaN.
    num, disp = _resolve_patient_value("x_cat", nan, input_X, patient=0)
    assert num is None
    assert disp == "—"

    # Non-coercible raw + numeric feat_val: fallback to formatted feat_val.
    num, disp = _resolve_patient_value("x_cat", 7.0, input_X, patient=0)
    assert num is None
    assert disp == "7"

    # No input_X column for the feature, NaN feat_val: still the sentinel.
    num, disp = _resolve_patient_value("missing_feat", nan, input_X, patient=0)
    assert num is None
    assert disp == "—"

    # Finite raw passes through unchanged.
    finite_X = pd.DataFrame({"age": [55.0]})
    num, disp = _resolve_patient_value("age", None, finite_X, patient=0)
    assert num == 55.0
    assert disp == "55"


def test_plot_waterfall_labels_nan_feature_value_explicitly():
    """Waterfall bar label flags missing/imputed values rather than showing
    the bare feature name."""
    from mlsurv.viz.explain import plot_waterfall

    contributions = pd.DataFrame({
        "feature": ["opioid_use", "age"],
        "feature_value": [float("nan"), 62.0],
        "shap_value": [0.5, -0.3],
        "abs_shap_value": [0.5, 0.3],
    })
    fig = plot_waterfall(contributions, risk_score=0.2)

    ticktext = list(fig.layout.yaxis.ticktext or [])
    label_blob = " | ".join(ticktext)
    assert "opioid_use = NaN (imputed)" in label_blob
    assert not any(t == "opioid_use" for t in ticktext)


# ---------------------------------------------------------------------------
# Percentile string shortcut for `patient=`
# ---------------------------------------------------------------------------


def _normalize_report_html(html: str) -> str:
    """Strip embedded timestamp + Plotly UUIDs and round floats so two reports
    for the same patient compare equal.

    Serialized Plotly coordinates (SHAP-waterfall bar edges, etc.) carry
    full float64 repr precision, so machine-epsilon accumulation noise (~1e-16,
    order- and BLAS-thread-dependent) made byte-equality flaky between the two
    ways of naming the same patient.  Rounding every float to 6 significant
    figures absorbs that noise while still catching a genuinely different
    patient (whose SHAP values / survival probabilities differ by far more).
    """
    import re
    # Timestamp like '2026-05-21 14:32'.
    html = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", "TS", html)
    # Plotly per-figure UUIDs.
    html = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "UUID",
        html,
    )
    # Round floats (decimals / scientific) to 6 significant figures.
    html = re.sub(
        r"-?\d+\.\d+(?:[eE][-+]?\d+)?",
        lambda m: f"{float(m.group()):.6g}",
        html,
    )
    return html


@pytest.mark.integration
def test_patient_report_low_risk_string_matches_manual_argsort(
    trained_for_report,
):
    learner, X = trained_for_report
    pipeline = learner.best_model.model
    order = np.argsort(pipeline.predict(X))
    expected_idx = int(order[int(0.10 * (len(order) - 1))])

    html_string = learner.patient_report(
        X, patient="low_risk", ci=False, n_bootstrap=2, random_state=0,
    )
    html_int = learner.patient_report(
        X, patient=expected_idx, ci=False, n_bootstrap=2, random_state=0,
    )
    assert _normalize_report_html(html_string) == _normalize_report_html(
        html_int
    )


@pytest.mark.integration
def test_patient_report_high_risk_string_matches_manual_argsort(
    trained_for_report,
):
    learner, X = trained_for_report
    pipeline = learner.best_model.model
    order = np.argsort(pipeline.predict(X))
    expected_idx = int(order[int(0.90 * (len(order) - 1))])

    html_string = learner.patient_report(
        X, patient="high_risk", ci=False, n_bootstrap=2, random_state=0,
    )
    html_int = learner.patient_report(
        X, patient=expected_idx, ci=False, n_bootstrap=2, random_state=0,
    )
    assert _normalize_report_html(html_string) == _normalize_report_html(
        html_int
    )


@pytest.mark.integration
def test_patient_report_p50_equals_median(trained_for_report):
    learner, X = trained_for_report
    html_p50 = learner.patient_report(
        X, patient="p50", ci=False, n_bootstrap=2, random_state=0,
    )
    html_median = learner.patient_report(
        X, patient="median", ci=False, n_bootstrap=2, random_state=0,
    )
    assert _normalize_report_html(html_p50) == _normalize_report_html(
        html_median
    )


@pytest.mark.integration
def test_patient_report_integer_patient_unchanged(trained_for_report):
    """Integer-path regression: `patient=<int>` still works."""
    learner, X = trained_for_report
    html = learner.patient_report(
        X, patient=0, ci=False, n_bootstrap=2, random_state=0,
    )
    assert len(html) > 1000


def test_patient_report_unrecognized_string_raises(trained_for_report):
    learner, X = trained_for_report
    with pytest.raises(ValueError, match="not recognized"):
        learner.patient_report(X, patient="bogus", ci=False)


def test_patient_report_bool_rejected(trained_for_report):
    learner, X = trained_for_report
    with pytest.raises(TypeError, match="bool"):
        learner.patient_report(X, patient=True, ci=False)


def test_stat_grid_nan_median_says_not_reached():
    """When median survival is NaN (S(t) > 0.5 throughout follow-up),
    the stat grid must use the "Not reached" convention, not "undefined"
    (which reads like a computation error to a clinician)."""
    from mlsurv.learner._patient_report_html import _stat_grid_items
    from mlsurv.results._prediction import PredictionResult

    pred = PredictionResult(
        risk_scores=np.array([0.5]),
        model_name="coxph",
        n_patients=1,
        _median_survival=np.array([np.nan]),
    )
    items = dict(_stat_grid_items(pred, 0, canonical_unit="months"))
    assert items["Median survival"] == (
        "Not reached (S(t) > 0.5 throughout follow-up)"
    )
    assert "undefined" not in items["Median survival"]


# ---------------------------------------------------------------------------
# Figure accessibility (role="img" + aria-label for screen readers)
# ---------------------------------------------------------------------------


def test_aria_figure_wraps_only_its_argument():
    """``_aria_figure`` wraps exactly the figure HTML it is given, with a
    role="img" and an escaped aria-label — nothing else is pulled inside."""
    from mlsurv.learner._patient_report import _aria_figure

    out = _aria_figure("<PLOT/>", "Kaplan-Meier survival overlay")
    assert out == (
        '<div role="img" aria-label="Kaplan-Meier survival overlay">'
        "<PLOT/></div>"
    )


def test_aria_figure_escapes_label():
    """A label with quotes/markup is HTML-attribute-escaped."""
    from mlsurv.learner._patient_report import _aria_figure

    out = _aria_figure("<PLOT/>", 'Bad "x" & <y>')
    assert 'aria-label="Bad &quot;x&quot; &amp; &lt;y&gt;"' in out


@pytest.mark.requires_shap
def test_patient_report_figures_have_img_role_and_labels(trained_for_report):
    """The patient report's three always-present figures each expose a
    non-empty accessible name via role="img"."""
    learner, X = trained_for_report
    html = learner.patient_report(X.head(2), patient=0, ci=False)

    for label in (
        "Individualized predicted survival curve",
        "Cohort risk-score distribution",
        "Kaplan-Meier survival overlay",
    ):
        assert f'role="img" aria-label="{label}"' in html


@pytest.mark.requires_shap
def test_patient_report_methodology_stays_outside_figure_region(
    trained_for_report,
):
    """The CI methodology note is a sibling of the survival-curve figure, not
    inside its role="img" region (which would hide it from screen readers)."""
    learner, X = trained_for_report
    html = learner.patient_report(
        X.head(2), patient=0, ci=True, ci_band=True,
        n_bootstrap=4, random_state=0,
    )
    assert 'class="methodology"' in html
    # The survival-curve helper returns _aria_figure(fig) + methodology, so the
    # methodology block is concatenated AFTER the figure's closed role="img"
    # div — i.e. it appears later in the document than the figure's opening
    # wrapper. (test_aria_figure_wraps_only_its_argument independently proves
    # _aria_figure wraps nothing but the figure, so "later" means "sibling".)
    marker = 'aria-label="Individualized predicted survival curve">'
    fig_start = html.index(marker) + len(marker)
    methodology_at = html.index('class="methodology"')
    assert methodology_at > fig_start


def test_shap_waterfall_html_wraps_figure_with_aria():
    """The SHAP figure routes through _aria_figure like the other three.

    Uses a stub ``expl`` so the assertion does not depend on the optional
    ``shap`` library being installed (the real waterfall figure does)."""
    from mlsurv.learner._patient_report import _shap_waterfall_html

    class _FakeFig:
        def to_html(self, **kwargs):
            return "<SHAP_PLOT/>"

    class _Expl:
        figure = _FakeFig()

    out = _shap_waterfall_html(_Expl(), include_js=False)
    assert out == (
        '<div role="img" aria-label="SHAP feature contribution waterfall">'
        "<SHAP_PLOT/></div>"
    )


def test_shap_waterfall_html_unavailable_is_not_wrapped():
    """When no SHAP figure exists, the fallback message stays unwrapped —
    there is no figure to label."""
    from mlsurv.learner._patient_report import _shap_waterfall_html

    class _Expl:
        figure = None

    out = _shap_waterfall_html(_Expl(), include_js=False)
    assert 'role="img"' not in out
    assert "SHAP waterfall unavailable" in out
