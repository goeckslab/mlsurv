"""Tests for bootstrap evaluation correctness fixes."""
import numpy as np
import pandas as pd
import pytest

from tests._helpers import make_y


# ── Bug 1: _compute_auc_t no longer accepts y_test_ipcw ────────────────


class TestAucTNoYTestIpcw:
    """Verify _compute_auc_t always uses internally-computed IPCW data."""

    def test_rejects_y_test_ipcw_kwarg(self):
        """_compute_auc_t should raise TypeError if y_test_ipcw is passed."""
        from mlsurv.evaluation._concordance import _compute_auc_t

        y_train = make_y([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y_test = make_y([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        risk = np.random.default_rng(42).standard_normal(10)
        eval_times = np.array([2.0, 5.0, 8.0])

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            _compute_auc_t(y_test, y_train, eval_times, risk,
                           y_test_ipcw=y_test)

    def test_auc_uses_internal_clipped_data(self):
        """AUC computation should produce valid results without caller IPCW."""
        from mlsurv.evaluation._concordance import _compute_auc_t

        rng = np.random.default_rng(42)
        n = 50
        y_train = make_y(
            rng.uniform(0.5, 20, n),
            rng.random(n) > 0.3,
        )
        y_test = make_y(
            rng.uniform(0.5, 15, n),
            rng.random(n) > 0.3,
        )
        risk = rng.standard_normal(n)
        eval_times = np.array([2.0, 5.0, 8.0, 12.0])

        result = _compute_auc_t(y_test, y_train, eval_times, risk)

        assert len(result) == len(eval_times)
        # At least some timepoints should have valid AUC
        valid = result['auc'].dropna()
        assert len(valid) > 0
        assert all(0 <= v <= 1 for v in valid)

    def test_concordance_no_y_test_ipcw_param(self):
        """_compute_concordance should not accept y_test_ipcw."""
        from mlsurv.evaluation._concordance import _compute_concordance
        import inspect

        sig = inspect.signature(_compute_concordance)
        assert "y_test_ipcw" not in sig.parameters

    def test_add_brier_no_y_test_ipcw_param(self):
        """_add_brier_metrics should not accept y_test_ipcw."""
        from mlsurv.evaluation._concordance import _add_brier_metrics
        import inspect

        sig = inspect.signature(_add_brier_metrics)
        assert "y_test_ipcw" not in sig.parameters


# ── Bug 2: _extract_mean_auc consistent timepoint filtering ────────────


def _make_bootstrap_result(models, n_bootstrap=200, auc_t_data=None):
    """Build a BootstrapResult from {model: {metric: values}} dict."""
    from mlsurv.results.containers import BootstrapResult

    iter_rows = []
    summary_rows = []
    for model_name, metrics in models.items():
        for metric_name, values in metrics.items():
            for i, v in enumerate(values):
                iter_rows.append({
                    "model_type": model_name, "iteration": i,
                    "metric_name": metric_name, "value": v,
                })
            summary_rows.append({
                "model_type": model_name, "metric": metric_name,
                "point": float(np.mean(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "ci_lower": float(np.percentile(values, 2.5)),
                "ci_upper": float(np.percentile(values, 97.5)),
            })

    auc_t = None
    if auc_t_data is not None:
        auc_rows = []
        for model_name, timepoints in auc_t_data.items():
            for eval_time, values in timepoints.items():
                for i, v in enumerate(values):
                    auc_rows.append({
                        "model_type": model_name, "iteration": i,
                        "eval_time": eval_time, "auc": v,
                    })
        auc_t = pd.DataFrame(auc_rows)

    return BootstrapResult(
        summary=pd.DataFrame(summary_rows),
        iterations=pd.DataFrame(iter_rows),
        auc_t=auc_t,
    )


class TestMeanAucConsistentTimepoints:
    """Verify _extract_mean_auc averages over consistent timepoint sets."""

    def test_filters_to_universally_valid_timepoints(self):
        """Only timepoints valid in ALL iterations should contribute to mean."""
        from mlsurv.evaluation._paired_comparison import _extract_mean_auc
        from mlsurv._warnings import MethodologicalWarning

        n = 10
        rng = np.random.RandomState(42)
        # t=1.0 valid in all, t=2.0 valid in all, t=3.0 has NaN in some
        auc_data = {
            "coxph": {
                1.0: rng.normal(0.7, 0.01, n),
                2.0: rng.normal(0.8, 0.01, n),
                3.0: np.array([0.6, 0.65, np.nan, 0.62, np.nan,
                               0.63, 0.64, np.nan, 0.61, 0.66]),
            }
        }
        br = _make_bootstrap_result(
            {"coxph": {"harrell": rng.normal(0.7, 0.02, n)}},
            n_bootstrap=n, auc_t_data=auc_data,
        )
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dist = _extract_mean_auc(br, "coxph")
        assert len(dist) == n
        # Mean should be average of t=1.0 and t=2.0 only (t=3.0 dropped)
        expected = (auc_data["coxph"][1.0] + auc_data["coxph"][2.0]) / 2
        np.testing.assert_allclose(dist, expected, atol=1e-8)
        # Should emit MethodologicalWarning about dropped timepoints
        meth_warns = [x for x in w if issubclass(x.category, MethodologicalWarning)]
        assert len(meth_warns) == 1
        assert "dropped 1/3 timepoints" in str(meth_warns[0].message)

    def test_all_timepoints_valid_uses_all(self):
        """When all timepoints are valid in all iterations, all are used."""
        from mlsurv.evaluation._paired_comparison import _extract_mean_auc

        n = 10
        rng = np.random.RandomState(42)
        auc_data = {
            "coxph": {
                1.0: rng.normal(0.7, 0.01, n),
                2.0: rng.normal(0.8, 0.01, n),
            }
        }
        br = _make_bootstrap_result(
            {"coxph": {"harrell": rng.normal(0.7, 0.02, n)}},
            n_bootstrap=n, auc_t_data=auc_data,
        )
        dist = _extract_mean_auc(br, "coxph")
        expected = (auc_data["coxph"][1.0] + auc_data["coxph"][2.0]) / 2
        np.testing.assert_allclose(dist, expected, atol=1e-8)

    def test_raises_when_no_universally_valid_timepoints(self):
        """Should raise ValueError when no timepoint is valid in all iterations."""
        from mlsurv.evaluation._paired_comparison import _extract_mean_auc

        n = 4
        # t=1.0 has NaN in iter 0, t=2.0 has NaN in iter 1 → no universal
        auc_data = {
            "coxph": {
                1.0: np.array([np.nan, 0.7, 0.7, 0.7]),
                2.0: np.array([0.8, np.nan, 0.8, 0.8]),
            }
        }
        rng = np.random.RandomState(42)
        br = _make_bootstrap_result(
            {"coxph": {"harrell": rng.normal(0.7, 0.02, n)}},
            n_bootstrap=n, auc_t_data=auc_data,
        )
        with pytest.raises(ValueError, match="No eval_time is valid"):
            _extract_mean_auc(br, "coxph")

    def test_old_behavior_was_inconsistent(self):
        """Demonstrate the bug: without the fix, iterations average different sets.

        Before the fix, iteration 0 would average over {t=1, t=2} (2 points)
        while iteration 1 averages over {t=1} (1 point), producing incomparable
        means for the paired test.
        """
        from mlsurv.evaluation._paired_comparison import _extract_mean_auc

        n = 4
        # t=1.0 valid everywhere, t=2.0 has NaN in iterations 1 and 3
        vals_t1 = np.array([0.70, 0.72, 0.71, 0.73])
        vals_t2 = np.array([0.80, np.nan, 0.82, np.nan])
        auc_data = {"coxph": {1.0: vals_t1, 2.0: vals_t2}}
        rng = np.random.RandomState(42)
        br = _make_bootstrap_result(
            {"coxph": {"harrell": rng.normal(0.7, 0.02, n)}},
            n_bootstrap=n, auc_t_data=auc_data,
        )
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            dist = _extract_mean_auc(br, "coxph")
        # With the fix, t=2.0 is dropped (NaN in iters 1,3), so mean = t=1.0 only
        np.testing.assert_allclose(dist, vals_t1, atol=1e-8)


# ── Bug 3: Bootstrap must freeze internally-selected hyperparameters ──


class TestBootstrapFreezesTunedHyperparameters:
    """CoxNet's ``fit()`` recomputes the regularization path on every call,
    so a bootstrap resample with ``alpha=<int path index>`` (Optuna's tuned
    form) or ``alpha='auto'`` would re-select a different actual alpha on
    each iteration — instability that propagates into the bootstrap CI band
    on ``S(t|x)``. ``freeze_for_resampling()`` pins ``alpha``
    to the fitted ``selected_alpha_`` (a float) so resample refits stay on
    the ``_fit_final_model`` path with a fixed alpha.
    """

    def _make_data(self, n=80, n_features=5, seed=0):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(
            rng.normal(size=(n, n_features)),
            columns=[f"f{i}" for i in range(n_features)],
        )
        coef = rng.normal(size=n_features) * 0.5
        log_h = X.values @ coef
        u = rng.uniform(size=n)
        t = -np.log(u) / np.exp(log_h - log_h.mean())
        c = rng.uniform(0.1, 3.0, size=n)
        y = make_y(np.minimum(t, c), events=t < c)
        return X, y

    @pytest.mark.parametrize("alpha_init", [
        10,                  # plain int (path index)
        np.int64(10),        # Optuna's actual tuned type
        'auto',              # K-fold CV selection
        'min',               # smallest-alpha shortcut
    ])
    def test_coxnet_freeze_returns_unfitted_clone_with_pinned_alpha(
        self, alpha_init,
    ):
        from mlsurv.models.coxnet import CoxNetModel

        X, y = self._make_data()
        model = CoxNetModel(l1_ratio=0.5, alpha=alpha_init, cv_folds=3)
        model.fit(X, y)
        frozen = model.freeze_for_resampling()
        # alpha must be the resolved float, not the init form
        assert isinstance(frozen.alpha, float)
        assert frozen.alpha == float(model.selected_alpha_)
        # Frozen clone is unfitted — bootstrap will call .fit() on it
        assert not getattr(frozen, "is_fitted_", False)

    def test_coxnet_resample_uses_same_alpha(self):
        """End-to-end: fitting the frozen clone on a different resample must
        keep ``selected_alpha_`` constant (no path recomputation drift)."""
        from sklearn.base import clone
        from mlsurv.models.coxnet import CoxNetModel

        X, y = self._make_data()
        model = CoxNetModel(l1_ratio=0.5, alpha='auto', cv_folds=3)
        model.fit(X, y)
        pinned = float(model.selected_alpha_)

        rng = np.random.default_rng(1)
        seen = []
        for _ in range(5):
            idx = rng.integers(0, len(X), size=len(X))
            X_boot = X.iloc[idx].reset_index(drop=True)
            y_boot = y[idx]
            resample_model = clone(model.freeze_for_resampling())
            resample_model.fit(X_boot, y_boot)
            seen.append(float(resample_model.selected_alpha_))
        assert all(a == pinned for a in seen), (
            f"resample alphas drifted from pinned {pinned}: {seen}"
        )

    @pytest.mark.integration
    def test_run_prediction_bootstrap_uses_frozen_alpha(self):
        """End-to-end regression net: the public bootstrap call path must
        invoke ``freeze_for_resampling`` so per-iteration CoxNet refits
        share a single alpha. Without this test, removing the
        ``freeze_for_resampling()`` call in ``run_prediction_bootstrap`` —
        the actual integration point — would not be caught by the unit
        tests above.
        """
        from mlsurv import SurvivalLearner
        from mlsurv.models.coxnet import CoxNetModel

        X, y = self._make_data(n=120, n_features=4, seed=0)
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector=None, random_state=0, n_jobs=1)
        # Use untuned CoxNet — ``alpha='auto'`` is the default and triggers
        # the same path-recomputation leak as Optuna's ``alpha=<int>``.
        learner.train("coxnet")

        original_fit = CoxNetModel.fit
        recorded_alphas: list[float] = []

        def spying_fit(self, X_, y_):
            result = original_fit(self, X_, y_)
            recorded_alphas.append(float(self.selected_alpha_))
            return result

        CoxNetModel.fit = spying_fit
        try:
            learner.predict(
                learner._state.X_test.iloc[:3],
                models="coxnet",
                ci=True, ci_band=True,
                n_bootstrap=4, random_state=0, n_jobs=1,
                verbose=False, use_recalibrated=False,
            )
        finally:
            CoxNetModel.fit = original_fit

        # First call is the point-estimate fit (already happened in train),
        # the remaining calls are bootstrap iterations — all must share the
        # same selected_alpha_.
        bootstrap_alphas = recorded_alphas[-4:]
        assert len(set(bootstrap_alphas)) == 1, (
            f"alpha drifted across resamples: {bootstrap_alphas}"
        )

    def test_base_default_freeze_is_clone(self):
        """Wrappers without internal selection (default impl) get a plain
        unfitted clone — confirms the hook doesn't disturb clean wrappers."""
        from sklearn.base import clone as sk_clone
        from mlsurv.models.rsf import RSFModel

        model = RSFModel(n_estimators=5, random_state=0)
        frozen = model.freeze_for_resampling()
        # get_params must match clone()'s — i.e., the default hook returned
        # an equivalent unfitted clone
        assert frozen.get_params() == sk_clone(model).get_params()
        assert not getattr(frozen, "is_fitted_", False)
