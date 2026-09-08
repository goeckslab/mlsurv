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
"""

_INSTALL_HINT = "Install with: pip install xgboost>=1.6.0"

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
