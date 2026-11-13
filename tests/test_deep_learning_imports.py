"""torch loads only for deep models, and a broken torch is reported clearly.

Three behaviors are guarded here:

- Listing models and running ``setup()`` must not import torch. torch loads
  its own copy of the OpenMP runtime, and on macOS, once it is in the process,
  a later XGBoost fit can hang or crash.
- When torch is installed but fails to load (e.g. a blocked DLL), building a
  deep model raises a clear ``ImportError`` that says so, and a multi-model
  ``run()`` drops the deep model with a warning instead of crashing.
- On macOS, an XGBoost fit warns when torch is already loaded.
"""

import subprocess
import sys
import textwrap
import types
import warnings

import pytest

from mlsurv.models import _torch
from mlsurv.models._torch import (
    DEEP_LEARNING_PACKAGES,
    import_deep_learning,
    is_deep_learning_installed,
)
from mlsurv._warnings import MethodologicalWarning


def _broken_importer(exc, broken="torch"):
    """Stand-in for ``import_module`` that raises *exc* for one package.

    The other deep-learning packages come back as empty stand-in modules, so
    the tests behave the same whether or not torch is really installed.
    """
    from importlib import import_module as real_import_module

    def import_module(name, *args, **kwargs):
        if name == broken:
            raise exc
        if name in _torch.DEEP_LEARNING_PACKAGES:
            return types.ModuleType(name)
        return real_import_module(name, *args, **kwargs)

    return import_module


@pytest.fixture
def broken_torch(monkeypatch):
    """Make the deep-learning import guard see a torch that fails with OSError."""
    monkeypatch.setattr(
        _torch, "import_module",
        _broken_importer(OSError("[WinError 4551] blocked by policy")),
    )


# ---------------------------------------------------------------------------
# No torch import without a deep model
# ---------------------------------------------------------------------------

def test_setup_and_list_models_do_not_import_torch():
    """A fresh process that lists models, runs setup(), and trains coxph
    never loads torch.

    Runs in a subprocess because torch may already be loaded in the pytest
    process by other tests.
    """
    script = textwrap.dedent("""
        import sys
        import numpy as np
        import pandas as pd
        from mlsurv import SurvivalLearner
        from mlsurv.models import list_models
        from mlsurv.utils import to_structured_array

        list_models()
        list_models(include_unavailable=True)

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
        y = to_structured_array(
            rng.exponential(2, 60) + 0.1,
            rng.binomial(1, 0.6, 60).astype(bool),
        )
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=0, verbose=False)
        learner.train("coxph")

        loaded = [m for m in ("torch", "torchtuples", "pycox") if m in sys.modules]
        print("LOADED=" + ",".join(loaded))
    """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=240,
    )
    assert proc.returncode == 0, proc.stderr
    loaded_line = [
        line for line in proc.stdout.splitlines() if line.startswith("LOADED=")
    ]
    assert loaded_line == ["LOADED="], f"torch stack was imported: {proc.stdout!r}"


def test_is_deep_learning_installed_does_not_import(monkeypatch):
    """The availability check does not import the packages."""
    for name in DEEP_LEARNING_PACKAGES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    is_deep_learning_installed()
    for name in DEEP_LEARNING_PACKAGES:
        assert name not in sys.modules


def test_is_deep_learning_installed_false_when_package_missing(monkeypatch):
    """A package marked unavailable (None in sys.modules) means not installed."""
    monkeypatch.setitem(sys.modules, "pycox", None)
    assert is_deep_learning_installed() is False


def test_list_models_hides_deep_models_when_missing(monkeypatch):
    from mlsurv.models import list_models

    monkeypatch.setitem(sys.modules, "torch", None)
    models = list_models()
    assert "deepsurv" not in models
    assert "deephit" not in models
    assert "deepsurv" in list_models(include_unavailable=True)


# ---------------------------------------------------------------------------
# A torch that is installed but fails to load
# ---------------------------------------------------------------------------

def test_import_deep_learning_reports_oserror_as_importerror(broken_torch):
    with pytest.raises(ImportError, match="OSError") as info:
        import_deep_learning("DeepSurvModel")
    msg = str(info.value)
    assert "DeepSurvModel" in msg
    assert "blocked by policy" in msg  # the loader's own text survives
    assert "torch is installed but failed to load" in msg
    assert "Install with" not in msg
    assert isinstance(info.value.__cause__, OSError)


def test_import_deep_learning_missing_package(monkeypatch):
    monkeypatch.setattr(
        _torch, "import_module",
        _broken_importer(
            ModuleNotFoundError("No module named 'torch'", name="torch")
        ),
    )
    with pytest.raises(ImportError, match=r"pip install mlsurv\[deep\]"):
        import_deep_learning("DeepHitModel")


def test_import_deep_learning_dll_importerror_is_a_load_failure(monkeypatch):
    """A broken torch often raises ImportError itself; don't say "install"."""
    monkeypatch.setattr(
        _torch, "import_module",
        _broken_importer(ImportError("DLL load failed while importing _C")),
    )
    with pytest.raises(ImportError, match="failed to load") as info:
        import_deep_learning("DeepSurvModel")
    msg = str(info.value)
    assert "DLL load failed" in msg
    assert "Install with" not in msg


def test_import_deep_learning_missing_submodule_is_a_load_failure(monkeypatch):
    """torch found, but a module it imports is missing: a broken install."""
    exc = ModuleNotFoundError("No module named 'torch._C'", name="torch._C")
    monkeypatch.setattr(_torch, "import_module", _broken_importer(exc))
    with pytest.raises(ImportError, match="failed to load"):
        import_deep_learning()


@pytest.mark.parametrize("broken", ["torchtuples", "pycox"])
def test_import_deep_learning_names_the_failing_package(monkeypatch, broken):
    monkeypatch.setattr(
        _torch, "import_module",
        _broken_importer(AttributeError("np.float removed"), broken=broken),
    )
    with pytest.raises(ImportError) as info:
        import_deep_learning()
    msg = str(info.value)
    assert f"{broken} is installed but failed to load" in msg
    assert "Reinstalling torch" not in msg  # torch advice only for torch
    assert "exclude=" in msg


def test_import_deep_learning_reraises_memory_error(monkeypatch):
    monkeypatch.setattr(_torch, "import_module", _broken_importer(MemoryError()))
    with pytest.raises(MemoryError):
        import_deep_learning()


@pytest.mark.parametrize("model_name", ["deepsurv", "deephit"])
def test_deep_model_fit_raises_importerror_on_broken_torch(
    broken_torch, small_data, model_name,
):
    from mlsurv.models import get_model

    X, y = small_data
    model = get_model(model_name)
    with pytest.raises(ImportError, match="OSError"):
        model.fit(X.values, y)


def test_run_drops_deep_model_with_warning_on_broken_torch(
    broken_torch, small_data,
):
    """run() keeps the working models and names the broken torch in a warning."""
    from mlsurv import SurvivalLearner

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=0, verbose=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = learner.run(["coxph", "deepsurv"], tune=False, verbose=False)

    assert list(results) == ["coxph"]
    dropped = [
        w for w in caught
        if issubclass(w.category, MethodologicalWarning)
        and "deepsurv" in str(w.message)
    ]
    assert dropped, [str(w.message) for w in caught]
    assert "OSError" in str(dropped[0].message)


# ---------------------------------------------------------------------------
# macOS: warn before an XGBoost fit once torch is loaded
# ---------------------------------------------------------------------------

def _xgb_torch_warnings(fn):
    from mlsurv._warnings import SuggestionWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [
        w for w in caught
        if issubclass(w.category, SuggestionWarning)
        and "OMP_NUM_THREADS=1" in str(w.message)
    ]


@pytest.fixture
def macos_with_torch(monkeypatch):
    """Make mlsurv._xgboost see macOS with torch loaded.

    Patches ``sys`` and ``os`` inside the warning module. The native OpenMP
    runtime still sees the real ``OMP_NUM_THREADS`` setting during XGBoost fits.
    """
    import mlsurv._xgboost as xgb_guard

    fake_sys = types.SimpleNamespace(
        platform="darwin",
        modules={**sys.modules, "torch": types.ModuleType("torch")},
    )
    monkeypatch.setattr(xgb_guard, "sys", fake_sys)
    monkeypatch.setattr(xgb_guard, "os", types.SimpleNamespace(environ={}))
    return fake_sys


def test_macos_torch_warning_fires(macos_with_torch):
    from mlsurv._xgboost import warn_if_torch_loaded_on_macos

    assert len(_xgb_torch_warnings(warn_if_torch_loaded_on_macos)) == 1


@pytest.mark.parametrize("case", ["not_macos", "no_torch", "omp_one"])
def test_macos_torch_warning_silent(macos_with_torch, monkeypatch, case):
    import mlsurv._xgboost as xgb_guard

    if case == "not_macos":
        macos_with_torch.platform = "linux"
    elif case == "no_torch":
        del macos_with_torch.modules["torch"]
    else:
        monkeypatch.setitem(xgb_guard.os.environ, "OMP_NUM_THREADS", "1")
    assert _xgb_torch_warnings(xgb_guard.warn_if_torch_loaded_on_macos) == []


@pytest.mark.parametrize("model_name", ["xgbcox", "xgbaft"])
def test_xgboost_fit_warns_on_macos_with_torch(
    macos_with_torch, small_data, model_name,
):
    pytest.importorskip("xgboost")
    from mlsurv.models import get_model

    X, y = small_data
    model = get_model(model_name, n_estimators=5)
    caught = _xgb_torch_warnings(lambda: model.fit(X.values, y))
    assert len(caught) == 1
