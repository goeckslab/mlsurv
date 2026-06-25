"""Coalesce per-model calibration-drift warnings.

Multi-model ``evaluate()`` / ``run()`` should emit exactly one
``MethodologicalWarning`` listing every flagged model and a single
``train(calibrate=...)`` suggestion. Single-model ``_warn_poor_calibration``
keeps the inline per-model wording. The calibration **slope** is the sole
trigger (outside [0.5, 1.5]); the GND p-value is context only.
"""

import warnings

import pandas as pd
import pytest

from mlsurv._warnings import MethodologicalWarning
from mlsurv.evaluation.evaluate import (
    CalibrationDriftRecord,
    _emit_coalesced_calibration_warning,
    _format_coalesced_drift_warning,
    _format_single_drift_warning,
    _warn_poor_calibration,
)
from tests._helpers import make_survival_array


def _frame(slope=1.0, gnd=0.5):
    return pd.DataFrame({'cal_slope_test': [slope], 'gnd_test': [gnd]})


class _StubModel:
    """Plain model stub. ``score_based`` routes the drift branch."""

    def __init__(self, score_based=False):
        self.score_based = score_based


def _calibration_warns(caught):
    return [w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "calibration drift" in str(w.message)]


class TestCollectorDefersEmission:
    """When ``collector`` is passed, warnings.warn is suppressed and the
    record is appended instead. Single-model path (no collector) keeps
    the inline warning."""

    def test_collector_appends_record_no_warning(self):
        collector = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_poor_calibration(
                _StubModel(), _frame(slope=2.5, gnd=1e-6),
                "coxph", _warn_stacklevel=2, collector=collector,
            )
        assert _calibration_warns(caught) == []
        assert len(collector) == 1
        rec = collector[0]
        assert rec.label == "coxph"
        assert rec.branch == "default"
        assert rec.slope == pytest.approx(2.5)

    def test_no_collector_emits_inline(self):
        with pytest.warns(MethodologicalWarning, match="calibration drift"):
            _warn_poor_calibration(
                _StubModel(), _frame(slope=2.5, gnd=1e-6),
                "coxph", _warn_stacklevel=2,
            )

    def test_collector_skipped_when_silent(self):
        # In-band slope → no record appended (collector stays empty).
        collector = []
        _warn_poor_calibration(
            _StubModel(), _frame(slope=1.0, gnd=1e-6),
            "coxph", _warn_stacklevel=2, collector=collector,
        )
        assert collector == []


class TestSingleWording:
    """Single-model message: concise, recommends train(calibrate=...),
    never mentions recalibrate()."""

    def test_default_branch_recommends_train_calibrate(self):
        rec = CalibrationDriftRecord("coxph", 1.7, 0.33, "default")
        msg = _format_single_drift_warning(rec)
        assert len(msg) < 250
        assert "calibration drift" in msg
        assert "train(calibrate=" in msg
        assert "recalibrate(" not in msg

    def test_score_based_branch_recommends_breslow_slope(self):
        rec = CalibrationDriftRecord("fssvm", 2.5, 1e-6, "score_based")
        msg = _format_single_drift_warning(rec)
        assert "train(calibrate='breslow_slope')" in msg
        assert "van Houwelingen" in msg
        assert "recalibrate(" not in msg

    def test_residual_branch_suggests_other_method(self):
        rec = CalibrationDriftRecord("deephit", 1.9, 0.5,
                                     "residual_after_calibration")
        msg = _format_single_drift_warning(rec)
        assert "residual calibration drift after calibration" in msg
        assert "different calibration method" in msg
        assert "recalibrate(" not in msg


class TestCoalescedFormatter:
    def test_lists_each_model_and_one_train_calibrate_sentence(self):
        records = [
            CalibrationDriftRecord("coxph", 1.7, 0.33, "default"),
            CalibrationDriftRecord("rsf", 0.4, 0.04, "default"),
            CalibrationDriftRecord("xgbcox", 1.8, 0.18, "default"),
        ]
        msg = _format_coalesced_drift_warning(records, n_total=6)
        assert "3 of 6 models show calibration drift" in msg
        for name in ("coxph", "rsf", "xgbcox"):
            assert f"- {name}:" in msg
        assert "train(calibrate=" in msg
        assert "recalibrate(" not in msg

    def test_residual_models_get_separate_sentence(self):
        records = [
            CalibrationDriftRecord("coxph", 1.7, 0.2, "default"),
            CalibrationDriftRecord("deephit", 1.9, 0.5,
                                   "residual_after_calibration"),
        ]
        msg = _format_coalesced_drift_warning(records, n_total=4)
        assert "Consider calibrating S(t|x) for coxph" in msg
        assert "deephit: residual drift after calibration" in msg
        assert "[calibrator attached]" in msg

    def test_emit_no_op_on_empty(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_coalesced_calibration_warning([], n_total=4, stacklevel=2)
        assert _calibration_warns(caught) == []

    def test_emit_fires_exactly_one_warning(self):
        records = [
            CalibrationDriftRecord("coxph", 1.7, 0.33, "default"),
            CalibrationDriftRecord("rsf", 0.4, 0.04, "default"),
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_coalesced_calibration_warning(records, n_total=4, stacklevel=2)
        cw = _calibration_warns(caught)
        assert len(cw) == 1
        body = str(cw[0].message)
        assert "2 of 4 models show calibration drift" in body
        assert "coxph" in body and "rsf" in body


class TestNoStandaloneGndTrigger:
    def test_nan_slope_stays_silent(self):
        # NaN slope -> no record regardless of GND.
        collector = []
        _warn_poor_calibration(
            _StubModel(), _frame(slope=float("nan"), gnd=1e-9),
            "coxph", _warn_stacklevel=2, collector=collector,
        )
        assert collector == []

    def test_significant_gnd_in_band_slope_stays_silent(self):
        collector = []
        _warn_poor_calibration(
            _StubModel(), _frame(slope=1.1, gnd=1e-9),
            "coxph", _warn_stacklevel=2, collector=collector,
        )
        assert collector == []


class TestMultiModelEvaluateCoalesces:
    """Drive ``_evaluate_multiple`` directly with stub models that force
    drift — deterministic and fast.

    Acceptance criterion: a 4-model run where 3 models are flagged emits
    exactly one MethodologicalWarning whose body names all 3.
    """

    def test_three_flagged_of_four_emits_one_warning_naming_all_three(
        self, monkeypatch,
    ):
        import sys
        ev = sys.modules['mlsurv.evaluation.evaluate']

        flagged = {
            "coxph": _frame(slope=1.7, gnd=0.188),
            "rsf": _frame(slope=0.4, gnd=0.04),
            "xgbcox": _frame(slope=1.8, gnd=0.18),
        }
        ok = {"weibull_aft": _frame(slope=1.0, gnd=0.5)}
        all_concordance = {**flagged, **ok}

        def fake_eval_single(model, X_test, y_test, *, model_name,
                             _drift_collector=None, **kw):
            ev._warn_poor_calibration(
                _StubModel(), all_concordance[model_name], model_name,
                _warn_stacklevel=2, collector=_drift_collector,
            )
            return object()

        monkeypatch.setattr(ev, "_evaluate_single", fake_eval_single)
        monkeypatch.setattr(
            ev, "_aggregate_evaluation_results", lambda results: results,
        )

        models = {name: object() for name in all_concordance}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ev._evaluate_multiple(
                models, X_test=pd.DataFrame({'x': [0.0]}),
                y_test=make_survival_array([True], [1.0]),
                show_progress=False,
            )

        cal = _calibration_warns(caught)
        assert len(cal) == 1
        body = str(cal[0].message)
        assert "3 of 4 models show calibration drift" in body
        for name in flagged:
            assert f"- {name}:" in body, f"missing {name} in: {body}"
        assert "weibull_aft" not in body
        assert "train(calibrate=" in body

    def test_no_drift_emits_no_warning(self, monkeypatch):
        import sys
        ev = sys.modules['mlsurv.evaluation.evaluate']

        def fake_eval_single(model, X_test, y_test, *, model_name,
                             _drift_collector=None, **kw):
            ev._warn_poor_calibration(
                _StubModel(), _frame(slope=1.0, gnd=0.5), model_name,
                _warn_stacklevel=2, collector=_drift_collector,
            )
            return object()

        monkeypatch.setattr(ev, "_evaluate_single", fake_eval_single)
        monkeypatch.setattr(
            ev, "_aggregate_evaluation_results", lambda results: results,
        )

        models = {"coxph": object(), "rsf": object()}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ev._evaluate_multiple(
                models, X_test=pd.DataFrame({'x': [0.0]}),
                y_test=make_survival_array([True], [1.0]),
                show_progress=False,
            )
        assert _calibration_warns(caught) == []
