"""Check for and import the deep-learning stack (torch, torchtuples, pycox).

Two rules live here.

First, checking whether the stack is installed must not import it. PyTorch
loads its own copy of the OpenMP runtime, and on macOS, once that copy is in
the process, a later XGBoost fit can hang or crash. So listing models or
running ``setup()`` must not load torch; only training or loading a deep model
should. ``is_deep_learning_installed`` only asks the import system whether the
packages can be found.

Second, importing the stack can fail in two ways: the packages can be absent,
or they can be installed and still refuse to load (a blocked or missing DLL
on Windows, a failed ``dlopen`` on macOS, a version clash with numpy). A load
failure can arrive as ``ImportError``, ``OSError``, or something else.
``import_deep_learning`` reports every case as ``ImportError``, and tells the
two apart so the message names the right remedy and keeps the loader's text.
"""

from importlib import import_module
from importlib.util import find_spec
import sys

#: Packages a deep model needs. pycox depends on torchtuples, which wraps torch.
DEEP_LEARNING_PACKAGES = ("torch", "torchtuples", "pycox")

_INSTALL_HINT = "Install with: pip install mlsurv[deep]"

_LOAD_HINTS = {
    "torch": (
        "Reinstalling torch for this platform usually fixes it "
        "(see https://pytorch.org/get-started/locally/)."
    ),
}
_DEFAULT_LOAD_HINT = (
    "Its version may not match the installed torch, numpy, or pandas. "
    "Reinstalling with pip install --force-reinstall mlsurv[deep] usually "
    "fixes it."
)
_SKIP_HINT = (
    "To leave the deep models out of run(), pass "
    "exclude=['deepsurv', 'deephit']."
)


def _is_installed(name: str) -> bool:
    """True if *name* can be imported, without importing it."""
    if name in sys.modules:
        # Already imported, or set to None to mark it unavailable. Checked
        # first because find_spec raises ValueError for an imported module
        # whose __spec__ is None.
        return sys.modules[name] is not None
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def is_deep_learning_installed() -> bool:
    """True if torch, torchtuples, and pycox are all installed.

    Does not import them. A True result does not promise that they will
    load; ``import_deep_learning`` reports that when a deep model is trained.
    """
    return all(_is_installed(name) for name in DEEP_LEARNING_PACKAGES)


def _load_failure(context: str, name: str, exc: BaseException) -> ImportError:
    hint = _LOAD_HINTS.get(name, _DEFAULT_LOAD_HINT)
    return ImportError(
        f"{context} requires torch, torchtuples, and pycox. {name} is "
        f"installed but failed to load ({type(exc).__name__}: {exc}). "
        f"{hint} {_SKIP_HINT}"
    )


def import_deep_learning(context: str = "deep learning models") -> None:
    """Import torch, torchtuples, and pycox, or raise a clear ImportError.

    Parameters
    ----------
    context : str, default="deep learning models"
        What needed the stack, named in the error message (e.g. a model
        class name).

    Raises
    ------
    ImportError
        If a package is missing, or is installed but fails to load. The
        second case is reported as ``ImportError`` too, so one
        ``except ImportError`` at the call site covers both. The original
        exception is chained. ``MemoryError`` and ``RecursionError`` are
        re-raised unchanged.
    """
    for name in DEEP_LEARNING_PACKAGES:
        try:
            import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name == name:
                raise ImportError(
                    f"{context} requires torch, torchtuples, and pycox. "
                    f"{_INSTALL_HINT}"
                ) from exc
            # The package was found, but something it imports is missing.
            raise _load_failure(context, name, exc) from exc
        except (MemoryError, RecursionError):
            # The process is in trouble; calling this a broken install
            # would mislead.
            raise
        except Exception as exc:
            # Deliberately broad. Once the package is found, any error while
            # importing it (ImportError "DLL load failed", OSError, an
            # AttributeError from a version clash) means it cannot be used.
            raise _load_failure(context, name, exc) from exc
