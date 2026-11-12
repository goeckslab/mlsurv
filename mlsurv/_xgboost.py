"""One guarded way to import xgboost.

Importing xgboost fails in two different ways, and only one of them is an
``ImportError``. The package can be absent, which is the ordinary optional
dependency case. Or it can be installed and still refuse to load, because
the wheel's ``libxgboost`` links against an OpenMP runtime that the machine
does not have: on macOS ``libomp`` is not shipped with the OS, so a plain
``pip install xgboost`` leaves a package that raises ``XGBoostError`` on
import. That class inherits from ``ValueError``, not ``ImportError``.

The distinction matters because a bare ``except ImportError`` guard misses
the second case entirely. The raw ``dlopen`` failure then escapes as a
``ValueError``, which reads as a bad-input error rather than a missing
runtime, and inside a multi-model ``run()`` it becomes a pruned trial and
the model is dropped from the results. Both failures mean the same thing to
a caller (xgboost is unusable here), so both are reported as ``ImportError``
with a message naming the actual remedy.

The module also holds ``warn_if_torch_loaded_on_macos``, the warning given
before an XGBoost fit on macOS once PyTorch is in the process.
"""

import os
import sys
import warnings

_INSTALL_HINT = "Install with: pip install xgboost>=1.6.0"

_MACOS_TORCH_MESSAGE = (
    "XGBoost is running in a process that has already loaded PyTorch. On "
    "macOS, PyTorch loads its own copy of the OpenMP runtime, and once it "
    "is in the process an XGBoost fit can hang or crash. To avoid this, set "
    "OMP_NUM_THREADS=1 before starting Python (setting it inside a running "
    "session is too late), or fit the deep models (deepsurv, deephit) and "
    "the XGBoost models (xgbcox, xgbaft) in separate sessions."
)

_RUNTIME_HINT = (
    "xgboost is installed but failed to load, which usually means its "
    "OpenMP runtime is missing. On macOS: brew install libomp. On Linux, "
    "install libgomp via your package manager."
)


def import_xgboost(context: str = "this feature"):
    """Import and return the ``xgboost`` module, or raise a clear ImportError.

    Parameters
    ----------
    context : str, default="this feature"
        What needed xgboost, named in the error message (e.g. an estimator
        class name), so the reader knows which part of the call wanted it.

    Returns
    -------
    module
        The imported ``xgboost`` module.

    Raises
    ------
    ImportError
        If xgboost is not installed, or is installed but cannot be loaded.
        The second case is reported as ``ImportError`` too, even though the
        underlying exception is not one, so a single ``except ImportError``
        at the call site covers both. The original exception is chained.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            f"XGBoost is required for {context}. {_INSTALL_HINT}"
        ) from exc
    except (MemoryError, RecursionError):
        # Not an xgboost problem, and re-raising matches how the rest of the
        # package treats these: they say the process is in trouble, so
        # relabelling them as a missing dependency would mislead.
        raise
    except Exception as exc:
        # Deliberately broad. Anything else raised while importing means
        # xgboost cannot be used, whatever class the loader chose for it
        # (XGBoostError subclasses ValueError; a bad build can raise OSError).
        # Guessing at the exact set would put us back where we started.
        raise ImportError(
            f"XGBoost is required for {context}, and importing it failed "
            f"with {type(exc).__name__}: {exc}. {_RUNTIME_HINT}"
        ) from exc
    return xgb


def warn_if_torch_loaded_on_macos() -> None:
    """Warn before an XGBoost fit on macOS when PyTorch is already loaded.

    Once PyTorch's bundled libomp is in the process, an XGBoost fit on macOS
    can hang or crash, even with XGBoost limited to one thread. Setting
    ``OMP_NUM_THREADS=1`` before the process starts avoids it, and it is too
    late to set it here, so all this can do is say so before the fit.

    Silent on other platforms, when torch has not been imported, and when
    ``OMP_NUM_THREADS=1`` is in the environment. That last check assumes the
    variable was set before Python started; it cannot tell otherwise.

    A warning raised inside a joblib worker does not reach the user. For
    ``train()`` and ``run()`` that does not matter: deep models train only in
    the main process (their CV is sequential), and so does every final fit.
    ``parameter_sweep(n_jobs>1)`` is a known gap: a worker can train a deep
    model and then an XGBoost model, and its warning is not shown.
    """
    if sys.platform != "darwin" or "torch" not in sys.modules:
        return
    if os.environ.get("OMP_NUM_THREADS", "").strip() == "1":
        return
    from mlsurv._warnings import SuggestionWarning

    warnings.warn(_MACOS_TORCH_MESSAGE, SuggestionWarning, stacklevel=3)
