"""Bootstrap calibration slope/intercept cache thread-safety.

The old code cached the (slope, intercept) pair in two module-level globals
(`_cal_cache_ref`, `_cal_cache_result`). Under the joblib threading backend
(RSF/XGBCox/XGBAFT with precomputed predictions) concurrent iterations shared
those globals and did an unsynchronized read-then-write, so one iteration could
silently return another iteration's calibration value.

The fix replaces the globals with a fresh per-iteration ``cal_cache`` dict
created in ``_bootstrap_single_iteration`` and threaded through the metric
dispatch. These tests cover:

- the cache still avoids a redundant logistic fit within one iteration,
- separate per-iteration caches never cross-contaminate,
- the shared module globals are gone (regression guard),
- threading vs. loky backends give identical per-iteration calibration columns.
"""

import numpy as np
import pytest

from mlsurv.evaluation import _bootstrap_iteration as bi
from tests._helpers import make_survival_array


def _cal_inputs(seed, n=80):
    """Build (y, eval_times, surv_probs) for a direct cache call.

    Passing a precomputed ``surv_probs`` lets ``_get_cal_slope_intercept`` run
    without a model or X — it reads the column nearest the median eval_time.
    """
    rng = np.random.RandomState(seed)
    times = rng.exponential(5, n)
    events = (rng.uniform(size=n) < 0.7)
    y = make_survival_array(events, times)
    eval_times = np.linspace(times.min() + 0.1, np.median(times), 4)
    # surv_probs: n samples x len(eval_times), values in (0, 1)
    surv_probs = np.clip(rng.uniform(0.05, 0.95, size=(n, len(eval_times))), 0.01, 0.99)
    return y, eval_times, surv_probs


class TestPerIterationCache:
    """Direct unit tests for the per-iteration cache in _get_cal_slope_intercept."""

    def test_same_y_same_cache_fits_once(self, monkeypatch):
        """Second call on the same y + same cache reuses the result (no refit)."""
        calls = {"n": 0}
        real = bi.calibration_slope_intercept

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(bi, "calibration_slope_intercept", counting)

        y, eval_times, surv_probs = _cal_inputs(seed=1)
        cache = {}
        first = bi._get_cal_slope_intercept(y, eval_times, surv_probs, None, None, cache)
        second = bi._get_cal_slope_intercept(y, eval_times, surv_probs, None, None, cache)

        assert calls["n"] == 1, "logistic fit should run exactly once for a cache hit"
        assert first == second
        assert cache["ref"] is y
        assert cache["result"] == first

    def test_different_y_same_cache_recomputes(self, monkeypatch):
        """A different y array misses the identity guard and recomputes."""
        calls = {"n": 0}
        real = bi.calibration_slope_intercept

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(bi, "calibration_slope_intercept", counting)

        y_a, et_a, sp_a = _cal_inputs(seed=2)
        y_b, et_b, sp_b = _cal_inputs(seed=3)
        cache = {}
        bi._get_cal_slope_intercept(y_a, et_a, sp_a, None, None, cache)
        bi._get_cal_slope_intercept(y_b, et_b, sp_b, None, None, cache)

        assert calls["n"] == 2, "a different y must trigger a fresh fit"
        assert cache["ref"] is y_b

    def test_separate_caches_independent(self):
        """Two per-iteration caches never share state."""
        y_a, et_a, sp_a = _cal_inputs(seed=4)
        y_b, et_b, sp_b = _cal_inputs(seed=5)
        cache_a, cache_b = {}, {}

        res_a = bi._get_cal_slope_intercept(y_a, et_a, sp_a, None, None, cache_a)
        res_b = bi._get_cal_slope_intercept(y_b, et_b, sp_b, None, None, cache_b)

        # Each cache holds only its own iteration's array/result.
        assert cache_a["ref"] is y_a and cache_a["result"] == res_a
        assert cache_b["ref"] is y_b and cache_b["result"] == res_b
        # Distinct inputs should give distinct results (sanity that no bleed-over).
        assert res_a != res_b

    def test_subpopulation_subset_recomputes_on_shared_cache(self, monkeypatch):
        """A subpopulation subset (y_boot[mask]) on the same cache must recompute.

        Mirrors the real dispatch: ``_bootstrap_single_iteration`` threads ONE
        cal_cache into both the main-metric loop (on y_boot) and the
        subpopulation loop (on each sub_y = y_boot[mask]). Each subset is a new
        array object, so the identity guard must miss and recompute rather than
        return the parent y_boot's cached result.
        """
        calls = {"n": 0}
        real = bi.calibration_slope_intercept

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(bi, "calibration_slope_intercept", counting)

        y, eval_times, surv_probs = _cal_inputs(seed=7)
        cache = {}
        parent = bi._get_cal_slope_intercept(y, eval_times, surv_probs, None, None, cache)

        # Build a subpopulation subset of the SAME parent, as the bootstrap does.
        mask = np.zeros(len(y), dtype=bool)
        mask[: len(y) // 2] = True
        sub_y = y[mask]
        sub_surv = surv_probs[mask, :]
        subset = bi._get_cal_slope_intercept(sub_y, eval_times, sub_surv, None, None, cache)

        assert calls["n"] == 2, "subpopulation subset must trigger a fresh fit"
        assert cache["ref"] is sub_y, "cache must advance to the subset array"
        # A second call on the subset now hits (same object, same cache).
        again = bi._get_cal_slope_intercept(sub_y, eval_times, sub_surv, None, None, cache)
        assert calls["n"] == 2 and again == subset
        # The subset result is its own value, not the parent's leaked back.
        assert subset != parent

    def test_cache_none_always_recomputes(self, monkeypatch):
        """cache=None must not crash and must recompute every call."""
        calls = {"n": 0}
        real = bi.calibration_slope_intercept

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(bi, "calibration_slope_intercept", counting)

        y, eval_times, surv_probs = _cal_inputs(seed=6)
        r1 = bi._get_cal_slope_intercept(y, eval_times, surv_probs, None, None, None)
        r2 = bi._get_cal_slope_intercept(y, eval_times, surv_probs, None, None, None)

        assert calls["n"] == 2
        assert r1 == r2  # deterministic inputs → same value, just recomputed


def test_no_module_level_cache_globals():
    """Regression guard: the shared module globals must stay removed."""
    assert not hasattr(bi, "_cal_cache_ref")
    assert not hasattr(bi, "_cal_cache_result")


@pytest.mark.integration
def test_threading_vs_loky_calibration_columns_match(gbsg2_data):
    """RSF threading backend must give the same per-iteration cal columns as sequential.

    Per-iteration seeds make every y_boot deterministic, so once the shared-cache
    race is removed the cal_slope/cal_intercept iteration values are identical
    whether the iterations run sequentially or concurrently on threads. The old
    global-cache race could corrupt the concurrent run.

    Calls the internal ``bootstrap()`` with ``force_parallel=True`` so the
    threading backend (selected for RSF + precomputed predictions) genuinely
    engages even at a small iteration count — otherwise the heavy-model
    parallel threshold (100 iterations) would silently fall back to sequential.

    ``integrated_brier`` is included to also exercise the brier precompute path
    alongside calibration. (It is no longer *required* for the
    calibration metrics to precompute — ``_precompute_predictions`` now builds
    surv_probs for the full ``SURV_FUNC_METRICS`` set, not just brier-family
    metrics — but keeping it here is harmless and broadens coverage.)
    """
    import pandas as pd

    from mlsurv import SurvivalLearner
    from mlsurv.evaluation.bootstrap import bootstrap as _bootstrap
    from mlsurv.evaluation.helpers import _has_internal_parallelism

    X_raw, y_raw = gbsg2_data
    # GBSG2 ships y as ('cens', 'time'); mlsurv wants ('event', 'time').
    y = np.array([(bool(r["cens"]), r["time"]) for r in y_raw],
                 dtype=[("event", bool), ("time", float)])
    X = pd.get_dummies(X_raw, drop_first=True)

    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, n_jobs=4)
    result = learner.train("rsf")
    fitted = result.model
    # The threading backend is only chosen for models with internal parallelism.
    assert _has_internal_parallelism(fitted)

    X_test = learner._state.X_test
    y_test = learner._state.y_test
    y_train = learner._state.y_train
    eval_times = np.quantile(y_test["time"], [0.25, 0.5, 0.75])
    cal_metrics = ["cal_slope", "cal_intercept"]
    metrics = ["integrated_brier", *cal_metrics]

    def run(n_jobs):
        boot = _bootstrap(
            fitted, X_test, y_test, y_train=y_train, n_bootstrap=12,
            metrics=metrics, eval_times=eval_times, random_state=123,
            n_jobs=n_jobs, force_parallel=True, show_progress=False,
            model_name="rsf",
        )
        df = boot.iterations
        df = df[df["metric_name"].isin(cal_metrics)]
        return df.sort_values(["metric_name", "iteration"]).reset_index(drop=True)

    serial = run(1)        # loky/sequential reference
    threaded = run(4)      # threading backend, concurrent iterations

    # Guard against a vacuous all-NaN comparison: calibration must actually compute.
    assert np.isfinite(serial["value"].to_numpy()).any(), \
        "calibration values are all NaN — surv_probs precompute did not engage"

    assert list(serial["metric_name"]) == list(threaded["metric_name"])
    np.testing.assert_allclose(
        serial["value"].to_numpy(), threaded["value"].to_numpy(),
        rtol=1e-9, atol=1e-9, equal_nan=True,
    )
