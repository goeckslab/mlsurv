"""
mlsurv: Low-code survival machine learning package.
"""

import os
import tempfile
import warnings
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

# Single source of truth: read the version from the installed package metadata
# (pyproject.toml) so it can never drift from the distribution version.
try:
    __version__ = _pkg_version("mlsurv")
except PackageNotFoundError:  # running from an uninstalled source checkout
    __version__ = "0.0.0+unknown"


def _configure_matplotlib_cache_dir() -> None:
    """Set a writable MPLCONFIGDIR when HOME is not writable."""
    if os.environ.get("MPLCONFIGDIR"):
        return

    home_dir = Path.home()
    mpl_dir = home_dir / ".matplotlib"

    # If an existing cache dir is writable, keep matplotlib defaults.
    if mpl_dir.exists() and os.access(mpl_dir, os.W_OK):
        return

    # If HOME itself is writable, matplotlib can create ~/.matplotlib.
    if os.access(home_dir, os.W_OK):
        return

    try:
        fallback = Path(tempfile.gettempdir()) / "mlsurv-matplotlib"
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(fallback)
    except OSError:
        from mlsurv._warnings import InternalWarning
        warnings.warn(
            "mlsurv could not create a writable matplotlib cache directory. "
            "Set the MPLCONFIGDIR environment variable to a writable path "
            "to avoid potential plotting errors.",
            InternalWarning,
            stacklevel=1,
        )


# Register mlsurv's tiered warning filters — must be before any code
# that may emit warnings (e.g., matplotlib cache setup).
from mlsurv._warnings import configure_default_filters
configure_default_filters()


def _check_stdout_encoding() -> None:
    """Warn if stdout encoding cannot represent Unicode used by mlsurv."""
    import codecs
    import sys
    encoding = getattr(sys.stdout, "encoding", None)
    if not encoding:
        return
    try:
        canonical = codecs.lookup(encoding).name
    except LookupError:
        return
    if canonical in ("cp1252", "ascii", "iso8859-1", "latin-1"):
        from mlsurv._warnings import SuggestionWarning
        warnings.warn(
            f"stdout encoding is '{encoding}', which cannot represent all "
            "characters used by mlsurv. Set the environment variable "
            "PYTHONIOENCODING=utf-8 to avoid UnicodeEncodeError crashes.",
            SuggestionWarning,
            stacklevel=1,
        )


_check_stdout_encoding()
_configure_matplotlib_cache_dir()

# Suppress loky resource_tracker KeyError tracebacks on Windows.
# Loky's tracker subprocess has a race condition where multiple workers
# UNREGISTER the same temp folder. We swap the module-level `main`
# reference so the subprocess imports our wrapper (mlsurv._loky_patch),
# which installs a selective sys.excepthook before delegating to the
# real main. Also disable memmapping globally since it provides
# negligible benefit for typical survival analysis datasets.
import joblib
joblib.parallel_config(max_nbytes=None)
from joblib.externals.loky.backend import resource_tracker as _rt_mod
from mlsurv._loky_patch import main as _patched_rt_main
_rt_mod.main = _patched_rt_main

# Suppress NumPy 2.x deprecation warning from sksurv calling np.trapz.
# Cannot fix upstream; narrow filter matches only sksurv.metrics subpackage.
warnings.filterwarnings(
    "ignore",
    message=r"`trapz` is deprecated",
    category=DeprecationWarning,
    module=r"sksurv\.metrics",
)

# Suppress lifelines emitting pandas-4.0 keyword-only deprecation warnings
# from internal Series.var/mean/std calls during fit.  On lifelines
# v0.30.3 + pandas 3.0.2 this is ``pandas.errors.Pandas4Warning`` (a
# ``DeprecationWarning`` subclass), but earlier versions of lifelines have
# emitted equivalent messages as ``UserWarning``.  Use ``category=Warning``
# (the common base) so the filter is robust across versions; the message +
# module narrowing keeps legitimate lifelines warnings (convergence,
# ill-conditioned design, etc.) flowing through.
warnings.filterwarnings(
    "ignore",
    message=r"Starting with pandas version 4\.0",
    category=Warning,
    module=r"lifelines\..*",
)

from mlsurv.utils import (
    to_structured_array, load_survival_data,
    compute_survival_time, SurvivalArray,
)

# Model wrappers (imported after definition)
from mlsurv.models import (
    BaseSurvivalModel,
    CoxPHModel,
    CoxNetModel,
    RSFModel,
    GBSAModel,
    FSSVMModel,
    XGBCoxModel,
    XGBAFTModel,
    WeibullAFTModel,
)

# Result containers
from mlsurv.results import (
    EvaluationResult,
    BootstrapResult,
    ModelResult,
    ValidationResult,
    BenchmarkResult,
)

# Preprocessing
from mlsurv.preprocessing import (
    TopKCoxSelector,
)

# Guidance
from mlsurv.guidance import recommend_models, RecommendedModels

# Main API
from mlsurv.learner import SurvivalLearner
load_learner = SurvivalLearner.load_learner

# Reports
from mlsurv.reports import ReportGenerator

__all__ = [
    # Version
    "__version__",
    # Main API
    "SurvivalLearner",
    "load_learner",
    # Reports
    "ReportGenerator",
    # Utilities
    "recommend_models",
    "RecommendedModels",
    "to_structured_array",
    "load_survival_data",
    "compute_survival_time",
    "SurvivalArray",
    # Models
    "BaseSurvivalModel",
    "CoxPHModel",
    "CoxNetModel",
    "RSFModel",
    "GBSAModel",
    "FSSVMModel",
    "XGBCoxModel",
    "XGBAFTModel",
    "WeibullAFTModel",
    # Results
    "EvaluationResult",
    "BootstrapResult",
    "ModelResult",
    "ValidationResult",
    "BenchmarkResult",
    # Preprocessing
    "TopKCoxSelector",
]

# Lazy import for DeepSurv (requires torch)
def __getattr__(name):
    if name == "DeepSurvModel":
        try:
            from mlsurv.models.deepsurv import DeepSurvModel
            return DeepSurvModel
        except ImportError:
            raise ImportError(
                "DeepSurvModel requires torch and pycox. "
                "Install with: pip install mlsurv[deep]"
            )
    if name == "DeepHitModel":
        try:
            from mlsurv.models.deephit import DeepHitModel
            return DeepHitModel
        except ImportError:
            raise ImportError(
                "DeepHitModel requires torch and pycox. "
                "Install with: pip install mlsurv[deep]"
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
