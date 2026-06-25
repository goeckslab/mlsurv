"""Coalesce per-model GND sample-size warnings.

When the cohort is smaller than the GND minimum (default 50 = 10 bins x
5/bin), every model in a multi-model ``evaluate()`` previously emitted the
same ``"GND calibration: N samples < M required..."``
``MethodologicalWarning`` — N lines of identical cohort math. The
sample-size shortage is a cohort property, not a per-model property, so
multi-model evaluate now coalesces them into one warning naming all
affected models. Single-model evaluate keeps the inline per-model
wording (no collector path → unchanged behavior).
"""

import importlib
import warnings

import pytest

from mlsurv._warnings import MethodologicalWarning
from mlsurv.evaluation.evaluate import (
    _emit_coalesced_calibration_nan_warning,
    _emit_pending_calibration_warnings,
    _format_coalesced_calibration_nan_warning,
)
from tests._helpers import make_survival_array

# ``mlsurv.evaluation.evaluate`` is shadowed by the ``evaluate`` function
# in the package namespace; reach the module via importlib so monkeypatch
# targets the real module attribute.
eval_mod = importlib.import_module("mlsurv.evaluation.evaluate")


@pytest.fixture(autouse=True)
def _reset_calibration_dedup():
    """Clear module-level dedup set so each test starts clean."""
    from mlsurv.evaluation.metrics import _calibration_warned
    _calibration_warned.clear()
    yield
    _calibration_warned.clear()


def _gnd_shortage(model_name, n=35, min_samples=50, n_bins=10, min_per_bin=5):
    """Build a tuple matching the producer's exact wording."""
    prefix = f"{model_name}: " if model_name else ""
    return (
        "gnd_sample_size", model_name,
        f"{prefix}GND calibration: {n} samples < {min_samples} required "
        f"for {n_bins} bins of {min_per_bin}; returning NaN",
    )


class TestCollectorRouting:
    """``_emit_pending_calibration_warnings`` routes ``gnd_sample_size``
    messages to the collector even though they say ``"returning NaN"``
    (the cohort math is identical across models and can be coalesced)."""

    def test_gnd_sample_size_routed_to_collector(self):
        pending = [_gnd_shortage("coxph"), _gnd_shortage("rsf")]
        collector = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_pending_calibration_warnings(
                pending, stacklevel=2, collector=collector,
            )
        assert len(collector) == 2
        assert {entry[1] for entry in collector} == {"coxph", "rsf"}
        # No inline emission when the collector accepts the records.
        assert [w for w in caught
                if issubclass(w.category, MethodologicalWarning)] == []

    def test_no_collector_emits_inline(self):
        """Single-model (no collector) keeps inline per-model wording, and
        appends the actionable workaround hint as a second warning."""
        pending = [_gnd_shortage("coxph")]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_pending_calibration_warnings(pending, stacklevel=2)
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, MethodologicalWarning)]
        # Per-model cohort-math line + hint line.
        assert len(msgs) == 2
        assert "coxph" in msgs[0]
        assert "GND calibration:" in msgs[0]
        assert "35 samples < 50" in msgs[0]
        # Hint suggests the largest workable bin count (35 // 5 = 7).
        assert "gnd_n_bins=7" in msgs[1]

    # The (metric_key, model_name) dedup contract is metric-key-agnostic and is
    # covered by the identical test in test_coalesced_calibration_nan.py
    # (TestCollectorRouting.test_dedup_applies_before_routing) — not duplicated
    # here.


class TestFormatCoalesced:
    """``_format_coalesced_calibration_nan_warning`` emits the cohort math
    once for ``gnd_sample_size`` and lists all affected models."""

    def test_six_models_one_line(self):
        models = ["coxph", "coxnet", "rsf", "xgbcox", "xgbaft", "fssvm"]
        collected = [_gnd_shortage(m) for m in models]
        lines = _format_coalesced_calibration_nan_warning(
            collected, n_total=6,
        )
        # Cohort-math line + workaround hint line.
        assert len(lines) == 2
        line = lines[0]
        # Cohort math present exactly once, with all expected pieces.
        assert "GND calibration:" in line
        assert "35 samples < 50" in line
        assert "10 bins of 5" in line
        # Model names listed.
        for m in models:
            assert m in line
        assert "6/6 models" in line
        # Hint suggests the largest workable bin count (35 // 5 = 7).
        assert "gnd_n_bins=7" in lines[1]

    def test_order_preserved_by_first_occurrence(self):
        collected = [
            _gnd_shortage("rsf"),
            _gnd_shortage("coxph"),
            _gnd_shortage("rsf"),  # duplicate
            _gnd_shortage("xgbaft"),
        ]
        lines = _format_coalesced_calibration_nan_warning(
            collected, n_total=3,
        )
        line = lines[0]
        assert line.index("rsf") < line.index("coxph") < line.index("xgbaft")

    def test_partial_failures_subset_count(self):
        """3 of 6 models hit the shortage — line says ``3/6 models``."""
        collected = [_gnd_shortage(m) for m in ("a", "b", "c")]
        lines = _format_coalesced_calibration_nan_warning(
            collected, n_total=6,
        )
        # Cohort-math line + workaround hint line.
        assert len(lines) == 2
        assert "3/6 models" in lines[0]

    def test_mixed_with_slope_nan(self):
        """``gnd_sample_size`` + ``cal_slope`` returned-NaN coexist: one
        cohort-math line and one per-metric line, in insertion order."""
        collected = [
            _gnd_shortage("coxph"),
            _gnd_shortage("rsf"),
            (
                "cal_slope", "coxph",
                "coxph: Calibration slope/intercept returned NaN "
                "(enable debug logging for details)",
            ),
        ]
        lines = _format_coalesced_calibration_nan_warning(
            collected, n_total=2,
        )
        # gnd cohort-math line + hint line + slope line.
        assert len(lines) == 3
        # Output order = first-occurrence of metric_key (insertion order
        # via dict.setdefault). gnd_sample_size was inserted first, so
        # its cohort-math line + hint appear before the slope line.
        # Pinning this guards against a refactor that silently reorders
        # user-visible output.
        assert "GND calibration:" in lines[0]
        assert "gnd_n_bins=" in lines[1]
        assert "slope/intercept" in lines[2]
        gnd_line, slope_line = lines[0], lines[2]
        assert "2/2 models" in gnd_line
        assert "coxph" in gnd_line and "rsf" in gnd_line
        assert "1/2 models" in slope_line

    def test_prefix_anchor_locates_cohort_math(self):
        """The prefix stripper anchors on ``"GND calibration:"`` so the
        cohort math survives even when the leading ``<name>: `` token
        is absent (defensive: producer may someday pass ``model_name``
        that omits the standard prefix)."""
        from mlsurv.evaluation.evaluate import _strip_model_prefix

        # With prefix.
        msg_with = (
            "coxph: GND calibration: 35 samples < 50 required for "
            "10 bins of 5; returning NaN"
        )
        assert _strip_model_prefix(msg_with).startswith("GND calibration:")
        # Without prefix.
        msg_without = (
            "GND calibration: 35 samples < 50 required for "
            "10 bins of 5; returning NaN"
        )
        assert _strip_model_prefix(msg_without) == msg_without

    def test_anchor_missing_warns_and_returns_verbatim(self):
        """If a future producer reword removes the ``"GND calibration:"``
        anchor, the stripper falls back to returning the original
        message verbatim AND emits an ``InternalWarning`` so the silent
        degradation is observable when InternalWarning is enabled."""
        from mlsurv._warnings import InternalWarning
        from mlsurv.evaluation.evaluate import _strip_model_prefix

        odd_msg = "coxph: Something else entirely: 35 < 50"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", InternalWarning)
            result = _strip_model_prefix(odd_msg)
        # Verbatim fallback preserves information.
        assert result == odd_msg
        internal = [w for w in caught
                    if issubclass(w.category, InternalWarning)]
        assert len(internal) == 1
        assert "anchor" in str(internal[0].message)


class TestEmitCoalesced:
    """``_emit_coalesced_calibration_nan_warning`` emits one warning for
    a ``gnd_sample_size``-only collector and falls back to per-record
    emission on formatter failure."""

    def test_emits_one_warning(self):
        collected = [_gnd_shortage("a"), _gnd_shortage("b")]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_coalesced_calibration_nan_warning(
                collected, n_total=2, stacklevel=2,
            )
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, MethodologicalWarning)]
        # Cohort-math warning + workaround-hint warning.
        assert len(msgs) == 2
        assert "GND calibration:" in msgs[0]
        assert "gnd_n_bins=" in msgs[1]

    def test_formatter_failure_falls_back_to_per_record(self, monkeypatch):
        """If the formatter raises, every original per-model message is
        re-emitted so no warning is silently dropped."""
        def _boom(*_args, **_kw):
            raise ValueError("synthetic formatter failure")

        monkeypatch.setattr(
            eval_mod, "_format_coalesced_calibration_nan_warning", _boom,
        )
        collected = [_gnd_shortage("a"), _gnd_shortage("b")]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_coalesced_calibration_nan_warning(
                collected, n_total=2, stacklevel=2,
            )
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, MethodologicalWarning)]
        # One fallback notice + one per-record re-emit per record.
        assert any("Failed to format" in m for m in msgs)
        per_record = [m for m in msgs if "Failed to format" not in m]
        assert len(per_record) == 2
        # Original cohort math preserved on the fallback path.
        for m in per_record:
            assert "35 samples < 50" in m


class TestMultiModelIntegration:
    """End-to-end through ``_evaluate_multiple``: a small cohort produces
    one coalesced GND-sample-size warning instead of N per-model warnings."""

    def test_multi_model_evaluate_via_pending_injection(self):
        """Mirrors the multi-model coalescer wiring test. We feed ``_evaluate_multiple`` a
        controlled set of pending GND-shortage warnings via a
        monkeypatched ``_compute_concordance`` and assert exactly one
        coalesced warning surfaces with all model names."""
        import importlib

        import numpy as np
        import pandas as pd

        eval_module = importlib.import_module("mlsurv.evaluation.evaluate")
        from mlsurv.evaluation import _concordance as conc_mod

        class _StubModel:
            def predict(self, X):
                return np.zeros(len(X))

            def predict_survival_function(self, X, **_kwargs):
                return np.full((len(X), 2), 0.5)

        def _fake_compute_concordance(
            model, X_test, y_test, X_train, y_train, eval_times,
            custom_metrics, risk_scores=None, ipcw_tau=None,
            ipcw_y_train=None, model_name=None, pending_warnings=None,
            surv_probs_transform=None, gnd_n_bins=10,
        ):
            if pending_warnings is not None:
                pending_warnings.append((
                    "gnd_sample_size", model_name,
                    f"{model_name}: GND calibration: 5 samples < 50 "
                    f"required for 10 bins of 5; returning NaN",
                ))
            df = pd.DataFrame({
                'c_index_test': [0.5],
                'gnd_test': [np.nan],
            })
            return (df, np.full((len(X_test), 2), 0.5),
                    np.array([1.0, 2.0]), None, None)

        X_test = pd.DataFrame({'f1': [1.0, 2.0, 3.0, 4.0, 5.0]})
        y_test = make_survival_array(
            [True, False, True, False, True], [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        models = {
            "coxph": _StubModel(), "rsf": _StubModel(),
            "coxnet": _StubModel(), "xgbcox": _StubModel(),
        }

        from mlsurv.evaluation.metrics import _calibration_warned
        _calibration_warned.clear()

        orig = conc_mod._compute_concordance
        eval_attr = getattr(eval_module, "_compute_concordance", None)
        conc_mod._compute_concordance = _fake_compute_concordance
        if eval_attr is not None:
            eval_module._compute_concordance = _fake_compute_concordance
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                eval_module._evaluate_multiple(
                    models, X_test, y_test,
                    eval_times=np.array([1.0, 2.0]),
                    n_importance_repeats=0,
                    show_progress=False,
                    bootstrap=False,
                )
        finally:
            conc_mod._compute_concordance = orig
            if eval_attr is not None:
                eval_module._compute_concordance = eval_attr

        gnd_warns = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "GND calibration:" in str(w.message)
        ]
        # Exactly one warning, naming every model, with cohort math once.
        assert len(gnd_warns) == 1, (
            f"Expected 1 coalesced GND warning; got "
            f"{[str(w.message) for w in gnd_warns]}"
        )
        msg = str(gnd_warns[0].message)
        for n in models:
            assert n in msg
        assert "4/4 models" in msg
        assert "5 samples < 50" in msg


class TestUpstreamWordingContract:
    """The coalescer relies on (a) the metric_key ``"gnd_sample_size"``
    being used at the producer's sample-size early-exit and (b) the
    message including ``"GND calibration:"`` followed by the cohort math.
    Lock both so a future reword in ``mlsurv/evaluation/metrics.py``
    breaks loudly rather than silently disabling coalescing."""

    def test_gnd_generic_exception_branch_keeps_gnd_key(self):
        """The non-sample-size GND failure branches (``_assign_bins``
        failure, downstream exception) must keep the generic
        ``"gnd"`` metric_key. If a future copy-edit flipped them to
        ``"gnd_sample_size"``, every exception-branch message
        (``"GND calibration returned NaN ..."``) would route through
        the cohort-math line shape — and since those messages contain
        no sample/bin numbers, ``_strip_model_prefix`` would hit the
        anchor-missing path and produce nonsense output."""
        import numpy as np

        from mlsurv.evaluation.metrics import greenwood_nam_dagostino

        # Cohort above the sample-size gate but with constant predictions
        # → ``_assign_bins`` returns None → emits the "returned NaN"
        # exception-branch message tagged ``"gnd"``.
        n = 100
        y = make_survival_array([False] * n, np.linspace(1, 10, n))
        pred_probs = np.full(n, 0.5)  # constant → qcut fails

        warnings_out: list = []
        chi2, _p, _df = greenwood_nam_dagostino(
            y, pred_probs, eval_time=5.0, model_name="m",
            warnings_out=warnings_out,
        )
        assert np.isnan(chi2)
        keys = {mk for mk, _mn, _msg in warnings_out}
        assert "gnd" in keys, (
            f"Non-sample-size GND failure branches must keep metric_key "
            f"'gnd' so the coalescer routing is unchanged; got {keys!r}"
        )
        assert "gnd_sample_size" not in keys

    def test_gnd_sample_size_early_exit_uses_distinct_key(self):
        """N below the default minimum triggers the early-exit; the
        emitted tuple must carry ``metric_key == 'gnd_sample_size'``
        and the canonical ``GND calibration:`` cohort-math wording."""
        import numpy as np

        from mlsurv.evaluation.metrics import greenwood_nam_dagostino

        n = 20  # < 50 (10 bins * 5 per bin)
        y = make_survival_array([True] * n, np.linspace(1, 10, n))
        pred_probs = np.linspace(0.1, 0.9, n)

        warnings_out: list = []
        chi2, p_value, df = greenwood_nam_dagostino(
            y, pred_probs, eval_time=5.0, model_name="m",
            warnings_out=warnings_out,
        )
        assert np.isnan(chi2)
        assert len(warnings_out) == 1
        metric_key, model_name, msg = warnings_out[0]
        assert metric_key == "gnd_sample_size", (
            f"Producer must use 'gnd_sample_size' for the sample-size "
            f"early-exit so multi-model coalescing fires; got {metric_key!r}"
        )
        assert model_name == "m"
        assert "GND calibration:" in msg
        assert "20 samples < 50" in msg
        assert "returning NaN" in msg
