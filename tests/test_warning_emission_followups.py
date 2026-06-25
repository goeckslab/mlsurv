"""Follow-up tests for overflow regex / non-finite warnings and
calibration NaN warnings escaping joblib workers."""

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import MethodologicalWarning
from mlsurv.evaluation.evaluate import _emit_pending_calibration_warnings
from mlsurv.evaluation.metrics import (
    _calibration_warned,
    calibration_slope_intercept,
    greenwood_nam_dagostino,
    hosmer_lemeshow,
)
from mlsurv.models._overflow import (
    _OVERFLOW_RE,
    _count_nonfinite,
    _nonfinite_warned,
    breslow_call_with_clamp,
    predict_with_clamp,
    reemit_non_overflow,
)


def _make_y(n=20, all_censored_before=None):
    times = np.linspace(0.1, 5.0, n)
    events = np.ones(n, dtype=bool)
    if all_censored_before is not None:
        events[:] = False
        times[:] = all_censored_before / 2.0
    return np.array(list(zip(events, times)),
                    dtype=[('event', bool), ('time', float)])


# =============================================================================
# warnings_out channel and parent-side emission helper
# =============================================================================

class TestWarningsOutChannel:
    """Each metric helper accepts a ``warnings_out`` list. When provided,
    NaN-return warnings are appended for parent-side emission instead of
    being passed to ``warnings.warn`` (which joblib/loky workers silence)."""

    def setup_method(self):
        # Isolate from other tests' dedup state — we expect the warnings_out
        # path to bypass dedup entirely on each call.
        _calibration_warned.clear()

    def test_cal_slope_appends_no_warn(self):
        # informative<10 path; with warnings_out provided, warnings.warn must
        # not fire even though the metric returns NaN.
        y = _make_y(n=8, all_censored_before=10.0)
        out = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            slope, intercept = calibration_slope_intercept(
                y, np.full(8, 0.5), eval_time=5.0,
                model_name="testmodel", warnings_out=out,
            )
        assert np.isnan(slope) and np.isnan(intercept)
        assert [w for w in caught
                if issubclass(w.category, MethodologicalWarning)] == []
        assert len(out) == 1
        metric_key, model_name, msg = out[0]
        assert metric_key == "cal_slope"
        assert model_name == "testmodel"
        assert "informative" in msg

    def test_hl_appends_no_warn(self):
        y = _make_y(n=5)
        out = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hosmer_lemeshow(y, np.full(5, 0.5), eval_time=2.0,
                            model_name="m", warnings_out=out)
        assert [w for w in caught
                if issubclass(w.category, MethodologicalWarning)] == []
        assert len(out) == 1 and out[0][0] == "hl"

    def test_gnd_appends_no_warn(self):
        y = _make_y(n=5)
        out = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            greenwood_nam_dagostino(y, np.full(5, 0.5), eval_time=2.0,
                                     model_name="m", warnings_out=out)
        assert [w for w in caught
                if issubclass(w.category, MethodologicalWarning)] == []
        # Sample-size early-exit uses the distinct ``gnd_sample_size``
        # metric_key so the multi-model coalescer can group the
        # cohort-keyed shortage separately from the generic ``gnd``
        # exception/bin-failure branch.
        assert len(out) == 1 and out[0][0] == "gnd_sample_size"

    def test_legacy_path_still_warns(self):
        # warnings_out=None preserves direct-caller behavior — tests outside
        # this file (test_calibration_metrics.py) rely on this.
        y = _make_y(n=8, all_censored_before=10.0)
        with pytest.warns(MethodologicalWarning, match="informative"):
            calibration_slope_intercept(
                y, np.full(8, 0.5), eval_time=5.0, model_name="m",
            )


class TestEmitPendingCalibrationWarnings:
    def setup_method(self):
        _calibration_warned.clear()

    def test_dedupes_per_metric_model(self):
        pending = [
            ("cal_slope", "coxph", "msg cal_slope coxph"),
            ("cal_slope", "coxph", "msg cal_slope coxph dup"),
            ("cal_slope", "rsf", "msg cal_slope rsf"),
            ("hl", "coxph", "msg hl coxph"),
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_pending_calibration_warnings(pending, stacklevel=2)
        meth = [w for w in caught
                if issubclass(w.category, MethodologicalWarning)]
        # 3 unique (metric, model) keys; the duplicate cal_slope+coxph dropped.
        assert len(meth) == 3
        msgs = {str(w.message) for w in meth}
        assert "msg cal_slope coxph" in msgs
        assert "msg cal_slope rsf" in msgs
        assert "msg hl coxph" in msgs

    def test_empty_is_noop(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_pending_calibration_warnings([], stacklevel=2)
        assert caught == []

    def test_cross_call_dedup(self):
        # Calling the emitter twice with the same key must not double-emit.
        # Mirrors the earlier module-level dedup that direct-call worker
        # warnings already had via _calibration_warned.
        pending = [("cal_slope", "coxph", "msg")]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_pending_calibration_warnings(pending, stacklevel=2)
            _emit_pending_calibration_warnings(pending, stacklevel=2)
        meth = [w for w in caught
                if issubclass(w.category, MethodologicalWarning)]
        assert len(meth) == 1

    def test_shares_dedup_set_with_legacy_path(self):
        # Pre-populating _calibration_warned (as a direct-call legacy path
        # would do) must suppress a subsequent parent-side emit for the
        # same key — pins the cross-path parity claim in the docstring.
        _calibration_warned.add(("cal_slope", "coxph"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_pending_calibration_warnings(
                [("cal_slope", "coxph", "msg")], stacklevel=2,
            )
        assert [w for w in caught
                if issubclass(w.category, MethodologicalWarning)] == []


class TestComputeConcordanceWiring:
    """End-to-end test that ``pending_warnings`` is correctly threaded
    through ``_compute_concordance`` → ``_add_calibration_metrics`` →
    metric helpers — the easiest part to silently break."""

    def setup_method(self):
        _calibration_warned.clear()

    def test_pending_warnings_populated_on_degenerate_input(self):
        from mlsurv.evaluation._concordance import _compute_concordance
        from mlsurv.models.coxph import CoxPHModel

        # Tiny dataset: 6 samples, 2 features, all events. cal_slope
        # informative<10 path will trigger when surv_probs are computed.
        rng = np.random.default_rng(0)
        n = 6
        X = pd.DataFrame(rng.standard_normal((n, 2)),
                         columns=['a', 'b'])
        y = np.array(
            list(zip(np.ones(n, dtype=bool), np.linspace(0.1, 5.0, n))),
            dtype=[('event', bool), ('time', float)],
        )
        model = CoxPHModel().fit(X, y)
        pending: list = []
        eval_times = np.linspace(0.5, 4.5, 4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _compute_concordance(
                model, X, y, X_train=X, y_train=y,
                eval_times=eval_times, model_name="coxph",
                pending_warnings=pending,
            )
        # Calibration NaN warnings appended to pending (informative<10),
        # NOT emitted via warnings.warn — that's the joblib-worker safety.
        assert any(item[0] == "cal_slope" and item[1] == "coxph"
                   for item in pending), pending
        cal_warns = [w for w in caught
                     if issubclass(w.category, MethodologicalWarning)
                     and "informative" in str(w.message)]
        assert cal_warns == []


# =============================================================================
# overflow regex tightening + non-finite predict warnings
# =============================================================================

class TestOverflowRegex:
    def test_regex_matches_only_overflow(self):
        assert _OVERFLOW_RE.search("overflow encountered in exp")
        assert not _OVERFLOW_RE.search("divide by zero encountered in log")
        assert not _OVERFLOW_RE.search("invalid value encountered in true_divide")
        # NumPy emits these messages lowercase; pin case-sensitivity so a
        # capitalized variant from a future sksurv internal isn't silently
        # swallowed.
        assert not _OVERFLOW_RE.search("Overflow encountered in exp")

    def test_divide_by_zero_no_longer_swallowed(self):
        # Construct a captured RuntimeWarning matching the divide-by-zero text
        # and verify reemit_non_overflow lets it through.
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            _w.warn("divide by zero encountered in log", RuntimeWarning)
        assert len(caught) == 1
        with _w.catch_warnings(record=True) as out:
            _w.simplefilter("always")
            reemit_non_overflow(caught)
        assert len(out) == 1
        assert "divide by zero" in str(out[0].message)

    def test_overflow_still_swallowed(self):
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            _w.warn("overflow encountered in exp", RuntimeWarning)
        with _w.catch_warnings(record=True) as out:
            _w.simplefilter("always")
            reemit_non_overflow(caught)
        assert out == []


class TestCountNonfinite:
    def test_ndarray_finite(self):
        assert _count_nonfinite(np.array([0.1, 0.5, 0.9])) == 0

    def test_ndarray_with_nan_inf(self):
        assert _count_nonfinite(
            np.array([0.1, np.nan, np.inf, -np.inf, 0.5])
        ) == 3

    def test_step_function_array(self):
        from sksurv.functions import StepFunction
        sf_good = StepFunction(np.array([0.0, 1.0]), np.array([1.0, 0.5]))
        sf_bad = StepFunction(np.array([0.0, 1.0]), np.array([1.0, np.nan]))
        arr = np.array([sf_good, sf_bad], dtype=object)
        assert _count_nonfinite(arr) == 1

    def test_unknown_type_returns_zero(self):
        assert _count_nonfinite("not an array") == 0


class _StubModel:
    """Minimal sksurv-like stub: ``predict`` is monkey-patched by
    ``predict_with_clamp``; ``predict_survival_function`` returns whatever
    the test injects so we can exercise the non-finite path."""

    def __init__(self, predict_out, surv_out):
        self._predict_out = predict_out
        self._surv_out = surv_out
        self.predict = lambda X, **k: self._predict_out
        self.predict_survival_function = self._psf
        self.predict_cumulative_hazard_function = self._psf

    def _psf(self, X, return_array=False):
        # Force the monkey-patched ``self.predict`` to run so
        # ClampWarning behavior also exercises.
        _ = self.predict(X)
        return self._surv_out


class TestPredictWithClampNonFinite:
    def setup_method(self):
        _nonfinite_warned.clear()

    def test_nan_in_ndarray_emits_warning(self):
        n = 5
        X = np.zeros((n, 2))
        # Survival probs ndarray with one NaN — simulates degenerate Breslow.
        surv = np.array([[0.9, 0.5], [np.nan, 0.4], [0.8, 0.3], [0.7, 0.2], [0.6, 0.1]])
        model = _StubModel(predict_out=np.zeros(n), surv_out=surv)
        with pytest.warns(MethodologicalWarning, match="non-finite"):
            predict_with_clamp(
                model, "predict_survival_function", X, return_array=True,
                model_display_name="StubCox",
            )

    def test_finite_output_silent(self):
        n = 5
        X = np.zeros((n, 2))
        surv = np.full((n, 2), 0.5)
        model = _StubModel(predict_out=np.zeros(n), surv_out=surv)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predict_with_clamp(
                model, "predict_survival_function", X, return_array=True,
                model_display_name="StubCox",
            )
        assert [w for w in caught
                if issubclass(w.category, MethodologicalWarning)
                and "non-finite" in str(w.message)] == []

    def test_repeated_calls_deduped(self):
        # Bootstrap (200 iters) and CV folds call predict_with_clamp many
        # times — without dedup, each non-finite output would emit its own
        # warning and drown the signal.
        n = 5
        X = np.zeros((n, 2))
        surv = np.array([[0.9, 0.5], [np.nan, 0.4], [0.8, 0.3], [0.7, 0.2], [0.6, 0.1]])
        model = _StubModel(predict_out=np.zeros(n), surv_out=surv)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(5):
                predict_with_clamp(
                    model, "predict_survival_function", X, return_array=True,
                    model_display_name="StubCox",
                )
        nf_warns = [w for w in caught
                    if issubclass(w.category, MethodologicalWarning)
                    and "non-finite" in str(w.message)]
        assert len(nf_warns) == 1

    def test_dedup_includes_func_name(self):
        # Survival-function and cumulative-hazard paths produce different
        # output shapes — silencing the second by accident (key collapsing
        # to model-only) would hide a CHF-specific NaN.
        n = 5
        X = np.zeros((n, 2))
        surv = np.array([[0.9, 0.5], [np.nan, 0.4], [0.8, 0.3], [0.7, 0.2], [0.6, 0.1]])
        model = _StubModel(predict_out=np.zeros(n), surv_out=surv)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predict_with_clamp(
                model, "predict_survival_function", X, return_array=True,
                model_display_name="StubCox",
            )
            predict_with_clamp(
                model, "predict_cumulative_hazard_function", X,
                return_array=True, model_display_name="StubCox",
            )
        # Stub returns the same surv_out for both methods — what matters
        # is that the dedup key changed so both fired.
        nf_warns = [w for w in caught
                    if issubclass(w.category, MethodologicalWarning)
                    and "non-finite" in str(w.message)]
        assert len(nf_warns) == 2


class TestBreslowCallWithClampNonFinite:
    def setup_method(self):
        _nonfinite_warned.clear()

    def test_nan_output_emits_warning(self):
        risk = np.zeros(5)

        def fake_breslow(scores):
            return np.array([0.9, 0.5, np.nan, 0.3, 0.1])

        with pytest.warns(MethodologicalWarning, match="non-finite"):
            breslow_call_with_clamp(
                risk, fake_breslow, model_name="StubBreslow",
            )

    def test_finite_output_silent(self):
        risk = np.zeros(5)

        def fake_breslow(scores):
            return np.array([0.9, 0.5, 0.4, 0.3, 0.1])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            breslow_call_with_clamp(
                risk, fake_breslow, model_name="StubBreslow",
            )
        assert [w for w in caught
                if issubclass(w.category, MethodologicalWarning)
                and "non-finite" in str(w.message)] == []
