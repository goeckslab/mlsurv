"""Plumb ``gnd_n_bins`` through evaluate()/validate().

The Greenwood-Nam-D'Agostino (GND) calibration test needs >=5 samples per
bin and defaults to 10 bins, so any cohort with N < 50 gets NaN with no
recourse — the test statistic was hard-wired to 10 bins. This wires a
``gnd_n_bins`` parameter from ``SurvivalLearner.evaluate()`` (and, via
``**kwargs``, ``validate()``) down to ``greenwood_nam_dagostino``, and lets
the coalesced sample-size warning append an actionable workaround hint.
"""

import importlib
import warnings

import numpy as np
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import MethodologicalWarning
from mlsurv.evaluation.evaluate import (
    _format_coalesced_calibration_nan_warning,
    _gnd_workaround_hint,
)
from mlsurv.evaluation.metrics import greenwood_nam_dagostino

from tests._helpers import make_survival_array

eval_mod = importlib.import_module("mlsurv.evaluation.evaluate")
conc_mod = importlib.import_module("mlsurv.evaluation._concordance")


@pytest.fixture(autouse=True)
def _reset_calibration_dedup():
    """Clear module-level dedup set so each test starts clean."""
    from mlsurv.evaluation.metrics import _calibration_warned
    _calibration_warned.clear()
    yield
    _calibration_warned.clear()


def _gnd_shortage(model_name, n=35, n_bins=10, min_per_bin=5):
    """Build a tuple matching the producer's exact wording."""
    prefix = f"{model_name}: " if model_name else ""
    return (
        "gnd_sample_size", model_name,
        f"{prefix}GND calibration: {n} samples < {n_bins * min_per_bin} "
        f"required for {n_bins} bins of {min_per_bin}; returning NaN",
    )


class TestParameterReachesMetric:
    """``gnd_n_bins`` controls the >=5-per-bin gate in the GND metric."""

    def test_lower_n_bins_clears_sample_size_gate(self):
        """At N=35 the default 10 bins fails (needs 50) but 7 bins (needs
        35) clears the gate and computes a finite statistic."""
        n = 35
        y = make_survival_array(
            [bool(i % 2) for i in range(n)], np.linspace(1, 10, n)
        )
        pred_probs = np.linspace(0.05, 0.95, n)

        # Default: NaN + one gnd_sample_size warning record.
        warns_default: list = []
        chi2_default, _, _ = greenwood_nam_dagostino(
            y, pred_probs, eval_time=5.0, model_name="m",
            warnings_out=warns_default,
        )
        assert np.isnan(chi2_default)
        assert {mk for mk, _, _ in warns_default} == {"gnd_sample_size"}

        # gnd_n_bins=7: gate clears (35 >= 7*5), finite statistic, no warning.
        warns_seven: list = []
        chi2_seven, _, _ = greenwood_nam_dagostino(
            y, pred_probs, eval_time=5.0, n_bins=7, model_name="m",
            warnings_out=warns_seven,
        )
        assert np.isfinite(chi2_seven)
        assert warns_seven == []

    @pytest.mark.integration
    def test_learner_evaluate_threads_gnd_n_bins(self, small_data):
        """End-to-end: learner.evaluate(gnd_n_bins=7) reaches the metric
        dispatch with n_bins=7."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train("coxph")

        seen = {}
        orig = conc_mod.calibration_test

        def _spy(*args, **kwargs):
            seen['n_bins'] = kwargs.get('n_bins')
            return orig(*args, **kwargs)

        conc_mod.calibration_test = _spy
        try:
            learner.evaluate(gnd_n_bins=7, n_importance_repeats=0,
                             show_progress=False, verbose=False)
        finally:
            conc_mod.calibration_test = orig

        assert seen.get('n_bins') == 7

    @pytest.mark.integration
    def test_multi_model_evaluate_threads_gnd_n_bins(self, small_data):
        """Multi-model path: evaluate() -> _evaluate_multiple -> per-model
        _evaluate_single must forward gnd_n_bins to every model's metric."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.run(["coxph", "coxnet"])

        seen = []
        orig = conc_mod.calibration_test

        def _spy(*args, **kwargs):
            seen.append(kwargs.get('n_bins'))
            return orig(*args, **kwargs)

        conc_mod.calibration_test = _spy
        try:
            learner.evaluate(n_importance_repeats=0, show_progress=False,
                             verbose=False, gnd_n_bins=6)
        finally:
            conc_mod.calibration_test = orig

        # Both models evaluated; every metric call saw n_bins=6.
        assert len(seen) >= 2
        assert set(seen) == {6}

    @pytest.mark.integration
    def test_learner_validate_forwards_gnd_n_bins(self, small_data):
        """validate() forwards gnd_n_bins through **kwargs to evaluate()."""
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        learner.train("coxph")

        seen = {}
        orig = conc_mod.calibration_test

        def _spy(*args, **kwargs):
            seen['n_bins'] = kwargs.get('n_bins')
            return orig(*args, **kwargs)

        conc_mod.calibration_test = _spy
        try:
            learner.validate(
                learner.X_test, learner.y_test, name="ext",
                gnd_n_bins=4, bootstrap=False, verbose=False,
            )
        finally:
            conc_mod.calibration_test = orig

        assert seen.get('n_bins') == 4


class TestBoundaryValidation:
    """``gnd_n_bins`` is validated at the public boundary: a GND test needs
    >=2 bins, and the value must be integer. Sub-2 / non-integer raise a
    clear ValueError rather than a deep-stack crash or a silent NaN."""

    @pytest.fixture
    def setup_learner(self, small_data):
        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)
        return learner

    @pytest.mark.parametrize("bad", [0, 1, -3])
    def test_below_two_raises_valueerror(self, setup_learner, bad):
        with pytest.raises(ValueError, match="gnd_n_bins must be an integer"):
            setup_learner.evaluate(gnd_n_bins=bad, show_progress=False,
                                   verbose=False)

    def test_non_integer_raises_valueerror(self, setup_learner):
        with pytest.raises(ValueError, match="gnd_n_bins must be an integer"):
            setup_learner.evaluate(gnd_n_bins=2.5, show_progress=False,
                                   verbose=False)


class TestWorkaroundHint:
    """``_gnd_workaround_hint`` derives the largest workable bin count from
    the producer's cohort-math message."""

    def test_suggests_floor_n_over_per_bin(self):
        cohort_math = (
            "GND calibration: 35 samples < 50 required for 10 bins of 5; "
            "returning NaN"
        )
        hint = _gnd_workaround_hint(cohort_math)
        assert hint is not None
        assert "gnd_n_bins=7" in hint  # 35 // 5
        assert "<= N/5" in hint

    def test_returns_none_and_warns_when_unparseable(self):
        """Producer wording drift → no hint (rather than a garbled one) AND
        an InternalWarning so the silent loss is observable (mirrors the
        sibling _strip_model_prefix anchor-miss behavior)."""
        from mlsurv._warnings import InternalWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", InternalWarning)
            result = _gnd_workaround_hint("Some other message entirely")
        assert result is None
        internal = [w for w in caught
                    if issubclass(w.category, InternalWarning)]
        assert len(internal) == 1
        assert "drift" in str(internal[0].message)

    def test_returns_none_when_suggestion_below_two(self):
        """floor(N/P) < 2 → only a degenerate single bin is possible, so no
        useful workaround → no hint. N=8, P=5 → 1 bin."""
        cohort_math = (
            "GND calibration: 8 samples < 50 required for 10 bins of 5; "
            "returning NaN"
        )
        assert _gnd_workaround_hint(cohort_math) is None

    def test_suggests_two_at_boundary(self):
        """floor(N/P) == 2 is the smallest meaningful suggestion. N=10, P=5."""
        cohort_math = (
            "GND calibration: 10 samples < 50 required for 10 bins of 5; "
            "returning NaN"
        )
        hint = _gnd_workaround_hint(cohort_math)
        assert hint is not None
        assert "gnd_n_bins=2" in hint


class TestHintInCoalescedWarning:
    """The coalesced GND-sample-size summary appends the hint line.

    The "hint appears after the model list" and "single-model inline path
    appends the hint" cases are already pinned by the coalesced-warning tests
    ``TestFormatCoalesced.test_six_models_one_line`` and
    ``TestCollectorRouting.test_no_collector_emits_inline`` (both assert
    ``gnd_n_bins=7``); only the no-hint-when-too-small tail case is unique
    here.
    """

    def test_no_hint_when_cohort_too_small(self):
        """Tiny cohort where no bin count works → only the cohort-math line."""
        collected = [_gnd_shortage("coxph", n=3)]
        lines = _format_coalesced_calibration_nan_warning(
            collected, n_total=1,
        )
        assert len(lines) == 1
        assert "GND calibration:" in lines[0]


class TestNanRecoveredEndToEnd:
    """With gnd_n_bins <= N/5 the metric returns a finite GND p-value and
    fires no sample-size warning at a cohort size where the default would."""

    def test_finite_gnd_no_warning_below_threshold(self):
        n = 35
        y = make_survival_array(
            [bool(i % 2) for i in range(n)], np.linspace(1, 10, n)
        )
        pred_probs = np.linspace(0.05, 0.95, n)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _chi2, p_value, _df = greenwood_nam_dagostino(
                y, pred_probs, eval_time=5.0, n_bins=7, model_name="m",
            )
        assert np.isfinite(p_value)
        gnd_warns = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "GND calibration:" in str(w.message)
        ]
        assert gnd_warns == []
