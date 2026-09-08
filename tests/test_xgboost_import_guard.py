"""The xgboost import guard must catch a broken install, not just a missing one.

An xgboost wheel whose OpenMP runtime is absent raises ``XGBoostError`` on
import, and that class inherits from ``ValueError``. A guard written as
``except ImportError`` therefore lets it through, the cryptic ``dlopen``
message escapes as a bad-input error, and inside ``run()`` the model is
dropped from the comparison without the reason ever being legible.
"""

import builtins

import numpy as np
import pandas as pd
import pytest

from mlsurv._xgboost import import_xgboost


class _FakeXGBoostError(ValueError):
    """Stands in for ``xgboost._c_api.XGBoostError``.

    The real class is only importable when xgboost imports, which is exactly
    what these tests simulate failing. Its MRO is what matters here, and
    ``XGBoostError -> ValueError -> Exception`` is what this reproduces.
    """


@pytest.fixture
def failing_xgboost_import(monkeypatch):
    """Make ``import xgboost`` raise whatever the test asks for."""
    def _install(exc):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "xgboost":
                raise exc
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
    return _install


class TestImportXgboost:
    def test_returns_module_when_importable(self):
        xgb = pytest.importorskip("xgboost")
        assert import_xgboost("test") is xgb

    def test_missing_package_raises_importerror_with_install_hint(
        self, failing_xgboost_import
    ):
        failing_xgboost_import(ImportError("No module named 'xgboost'"))
        with pytest.raises(ImportError, match="pip install xgboost"):
            import_xgboost("XGBAFTModel")

    def test_broken_install_is_reported_as_importerror(
        self, failing_xgboost_import
    ):
        """The regression: a ValueError subclass must not escape as one."""
        failing_xgboost_import(
            _FakeXGBoostError("dlopen(...libxgboost.dylib): Library not loaded")
        )
        with pytest.raises(ImportError) as excinfo:
            import_xgboost("XGBAFTModel")
        # Not merely "an exception was raised": a bare `except ImportError`
        # at the call site has to catch it, which a ValueError would not.
        assert not isinstance(excinfo.value, ValueError)

    def test_broken_install_message_names_the_remedy(
        self, failing_xgboost_import
    ):
        failing_xgboost_import(_FakeXGBoostError("Library not loaded: libomp"))
        with pytest.raises(ImportError, match="brew install libomp"):
            import_xgboost("XGBAFTModel")

    def test_broken_install_chains_the_original_exception(
        self, failing_xgboost_import
    ):
        original = _FakeXGBoostError("Library not loaded: libomp.dylib")
        failing_xgboost_import(original)
        with pytest.raises(ImportError) as excinfo:
            import_xgboost("XGBAFTModel")
        assert excinfo.value.__cause__ is original

    def test_message_names_the_caller(self, failing_xgboost_import):
        failing_xgboost_import(ImportError("nope"))
        with pytest.raises(ImportError, match="XGBCoxModel"):
            import_xgboost("XGBCoxModel")

    def test_original_error_text_survives(self, failing_xgboost_import):
        """The loader's own message is the searchable part; keep it."""
        failing_xgboost_import(_FakeXGBoostError("Library not loaded: libomp"))
        with pytest.raises(ImportError, match="Library not loaded"):
            import_xgboost("XGBAFTModel")

    @pytest.mark.parametrize("exc_type", [MemoryError, RecursionError])
    def test_process_level_errors_propagate_unchanged(
        self, failing_xgboost_import, exc_type
    ):
        """These say the process is in trouble, not that a dependency is."""
        failing_xgboost_import(exc_type("out of resources"))
        with pytest.raises(exc_type):
            import_xgboost("XGBAFTModel")

    def test_non_valueerror_failures_are_also_caught(
        self, failing_xgboost_import
    ):
        """A bad build can fail with OSError rather than XGBoostError."""
        failing_xgboost_import(OSError("cannot open shared object file"))
        with pytest.raises(ImportError, match="cannot open shared object"):
            import_xgboost("XGBAFTModel")


class TestTunedRunReportsTheReason:
    """The tuned path is the one a default ``run()`` actually takes.

    ``run()`` defaults to ``tune=True``, so a macOS box without libomp goes
    through Optuna. ``study.optimize`` runs with ``catch=(Exception,)``, which
    swallows the ``ImportError`` the guard raises; unless the trial failure is
    recorded on the way past, the study ends with nothing recorded and the
    caller gets a bare "Record does not exist." instead of the remedy.
    """

    def test_libomp_reason_survives_tuning(self, monkeypatch):
        import warnings as _warnings

        import mlsurv.models.xgbcox as xgbcox_mod
        from mlsurv import SurvivalLearner

        rng = np.random.default_rng(0)
        n = 80
        X = pd.DataFrame({
            'a': rng.normal(size=n),
            'b': rng.normal(size=n),
            'c': rng.normal(size=n),
        })
        y = np.array(
            [(bool(e), float(t)) for e, t in
             zip(rng.integers(0, 2, n), rng.uniform(1, 100, n))],
            dtype=[("event", bool), ("time", float)],
        )

        def broken(context="xgboost"):
            raise ImportError(
                "XGBoost is required for XGBCoxModel, and importing it failed "
                "with XGBoostError: Library not loaded: @rpath/libomp.dylib. "
                "On macOS: brew install libomp."
            )

        monkeypatch.setattr(xgbcox_mod, "import_xgboost", broken)

        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector=None, random_state=42, cv_folds=2)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            models = learner.run(
                ['coxph', 'xgbcox'], tune=True, n_trials=2,
                n_bootstrap=2, n_importance_repeats=0,
            )

        assert 'xgbcox' not in models, "the broken model should be dropped"
        dropped = " ".join(
            str(w.message) for w in caught
            if "dropped" in str(w.message).lower()
        )
        assert "xgbcox" in dropped
        assert "libomp" in dropped, (
            "the remedy must survive Optuna's catch=(Exception,); "
            f"got: {dropped[:300]}"
        )
        assert "Record does not exist" not in dropped
