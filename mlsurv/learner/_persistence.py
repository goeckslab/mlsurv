"""PersistenceMixin - model artifact and learner save/load."""

from __future__ import annotations

import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

from ..constants import (
    ARTIFACT_METADATA_FILE, ARTIFACT_MODEL_FILE,
    ARTIFACT_RECALIBRATOR_FILE, ARTIFACT_VERSION,
    LEARNER_FORMAT_VERSION,
)
from .._warnings import InternalWarning, SuggestionWarning

if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner


def _gpu_state_hint(exc, path):
    """Return a targeted message when *exc* is a cross-device unpickling error.

    Torch tensors carry the device they were pickled from. Files written by an
    older mlsurv on a GPU host hold CUDA tensors outright, and no amount of
    care at load time can map them onto a machine without that device. Say so,
    instead of letting it read as file corruption.

    Returns ``None`` when *exc* is not that failure.
    """
    # Anchor on torch's own wording, not the bare words 'CUDA' or 'deserialize'.
    # Either alone also matches unrelated failures — a truncated pickle, a
    # version skew, an out-of-memory error on a host that does have a GPU — and
    # would turn an honestly uncertain message into a confident wrong diagnosis.
    # The full phrase covers torch's CUDA and MPS variants and nothing else.
    if 'deserialize object on' not in str(exc):
        return None
    return (
        f"Failed to load {path}: it holds model state saved on a GPU device "
        f"that is not available here. Files written by a version of mlsurv "
        f"that stored GPU tensors can only be opened where that device exists. "
        f"Load it there, then re-save it with this version, which stores "
        f"CPU-portable model state. "
        f"Original error: {type(exc).__name__}: {exc}"
    )


def _environment_fingerprint():
    """
    Capture an environment fingerprint for artifact metadata.

    Returns
    -------
    dict
        Dictionary with version strings for mlsurv, Python, and platform.
        Core scientific dependencies (scikit-survival, scikit-learn, numpy)
        are included with their version strings.  The value is
        ``'not installed'`` if a dependency is unexpectedly absent.
    """
    from importlib import import_module

    import mlsurv

    env = {
        'mlsurv_version': mlsurv.__version__,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
    }
    _OPTIONAL_PACKAGES = [
        ('sksurv', 'scikit_survival_version'),
        ('sklearn', 'scikit_learn_version'),
        ('numpy', 'numpy_version'),
    ]
    for pkg, key in _OPTIONAL_PACKAGES:
        try:
            env[key] = import_module(pkg).__version__
        except (ImportError, AttributeError):
            env[key] = 'not installed'
    return env


class _ArtifactEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types, NaN/inf, and Paths."""

    def default(self, obj):
        import numpy as np

        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

    def encode(self, o):
        """Override to convert Python float NaN/inf to null."""
        return super().encode(self._sanitize(o))

    def _sanitize(self, obj):
        import math
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]
        return obj


class PersistenceMixin:
    """Mixin for model-artifact and full-learner save/load."""

    def save_artifact(
        self: SurvivalLearner,
        model,
        path: Union[str, Path],
        *,
        step: str,
        time_col: str = 'time',
        event_col: str = 'event',
    ):
        """
        Save a model artifact directory with metadata and fitted model.

        Parameters
        ----------
        model : ModelResult, BaseSurvivalModel, Pipeline, or None
            Trained model or result to save. None for setup-only artifacts.
        path : str or Path
            Directory to create for the artifact.
        step : str
            CLI step that produced this artifact (e.g. 'train', 'tune', 'run').
        time_col : str
            Name of the time column in the source data.
        event_col : str
            Name of the event column in the source data.

        Raises
        ------
        ValueError
            If no feature data is available (``learner.X`` is ``None`` or
            missing).  Call ``setup()`` before saving.

        Notes
        -----
        Creates the metadata file (``ARTIFACT_METADATA_FILE``, default
        ``artifact.json``) and optionally the model file
        (``ARTIFACT_MODEL_FILE``, default ``model.joblib``) inside the
        target directory.  When ``model`` is None (setup-only artifact),
        only the metadata file is written.
        """
        import joblib

        if self.X is None:
            raise ValueError("Learner must be set up before saving an artifact.")

        artifact_dir = Path(path)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Extract fitted model and training metrics from ModelResult
        fitted_model = None
        model_name = None
        results = {}
        if model is not None:
            fitted_model = model
            model_name = step
            if hasattr(model, 'model'):
                fitted_model = model.model
                model_name = getattr(model, 'name', step)
                results = {
                    'cv_metric_mean': getattr(model, 'cv_metric_mean', None),
                    'cv_metric_std': getattr(model, 'cv_metric_std', None),
                    'cv_ci_lower': getattr(model, 'cv_ci_lower', None),
                    'cv_ci_upper': getattr(model, 'cv_ci_upper', None),
                    'cv_ci_level': getattr(model, 'cv_ci_level', None),
                    'is_tuned': getattr(model, 'is_tuned', False),
                    'best_params': getattr(model, 'best_params_', None),
                    'cv_metric_scores': getattr(model, 'cv_metric_scores', None),
                    'scoring_metric': getattr(model, 'scoring_metric', None),
                }

        # Bundle the attached calibrator (if any) for this model so external
        # prediction with the artifact reproduces the learner's calibrated
        # S(t|x). The SurvivalCalibrator pickles cleanly.
        recalibrator = None
        recalibration_meta = None
        if model_name is not None and self._has_calibrator(model_name):
            _mr = self._state.models.get(model_name)
            calibrator = getattr(_mr, "calibrator", None)
            if calibrator is not None:
                recalibrator = calibrator
                recalibration_meta = {
                    'method': calibrator.method,
                    'horizons': list(calibrator.horizons),
                }

        metadata = {
            'artifact_version': ARTIFACT_VERSION,
            'step': step,
            'model_name': model_name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'environment': _environment_fingerprint(),
            'config': self.config,
            'data_contract': {
                'feature_names': list(self.X.columns),
                'time_col': time_col,
                'event_col': event_col,
                'time_unit': self._state.time_unit,
            },
            'results': results,
            'recalibration': recalibration_meta,
        }

        with open(artifact_dir / ARTIFACT_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2, cls=_ArtifactEncoder)

        if fitted_model is not None:
            joblib.dump(fitted_model, artifact_dir / ARTIFACT_MODEL_FILE)
        if recalibrator:
            joblib.dump(recalibrator, artifact_dir / ARTIFACT_RECALIBRATOR_FILE)

        self._update_artifact_path(str(artifact_dir.resolve()))

    @staticmethod
    def load_artifact(
        path: Union[str, Path],
    ) -> Tuple[dict, Optional[object], Optional[dict]]:
        """
        Load a saved artifact directory.

        Parameters
        ----------
        path : str or Path
            Path to the artifact directory.

        Returns
        -------
        tuple of (dict, object or None, dict or None)
            ``(metadata, model, recalibrator)``. ``model`` is ``None`` for
            setup-only artifacts. ``recalibrator`` is the per-horizon
            ``{horizon: IsotonicRegression}`` mapping when the artifact was
            saved with a fitted recalibrator, otherwise ``None``.

        Raises
        ------
        FileNotFoundError
            If the artifact directory or metadata file is missing.
        ValueError
            If the artifact metadata is corrupt (unparseable JSON) or the
            artifact version is unsupported.
        RuntimeError
            If the model file exists but cannot be deserialized (e.g.,
            missing dependency, version incompatibility, or file corruption).

        Warns
        -----
        UserWarning
            If the artifact has no ``artifact_version`` key (pre-versioned).
            The metadata is treated as version 0 and loading continues.
        """
        import joblib

        artifact_dir = Path(path)
        metadata_path = artifact_dir / ARTIFACT_METADATA_FILE
        model_path = artifact_dir / ARTIFACT_MODEL_FILE
        recalibrator_path = artifact_dir / ARTIFACT_RECALIBRATOR_FILE

        if not artifact_dir.is_dir():
            raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"{ARTIFACT_METADATA_FILE} not found in: {artifact_dir}"
            )

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Corrupt artifact metadata at {metadata_path}: {exc}"
            ) from exc

        if 'artifact_version' not in metadata:
            warnings.warn(
                "Artifact has no 'artifact_version' key — treating as "
                "pre-versioned (v0). Consider running "
                "'mlsurv artifact-upgrade' to bring it up to date.",
                SuggestionWarning,
                stacklevel=2,
            )
            metadata['artifact_version'] = 0
        else:
            version = metadata['artifact_version']
            if not isinstance(version, int) or version < 1 or version > ARTIFACT_VERSION:
                raise ValueError(
                    f"Unsupported artifact version: {version} "
                    f"(this mlsurv supports versions 1-{ARTIFACT_VERSION})"
                )

        model = None
        if model_path.exists():
            try:
                model = joblib.load(model_path)
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError(
                    f"Failed to load model from {model_path}: missing "
                    f"dependency. Install the required package and retry: {exc}"
                ) from exc
            except Exception as exc:
                hint = _gpu_state_hint(exc, model_path)
                if hint:
                    raise RuntimeError(hint) from exc
                raise RuntimeError(
                    f"Failed to load model from {model_path}. This may "
                    f"indicate version incompatibility or file corruption: {exc}"
                ) from exc

        recalibrator = None
        if recalibrator_path.exists():
            try:
                recalibrator = joblib.load(recalibrator_path)
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError(
                    f"Failed to load recalibrator from {recalibrator_path}: "
                    f"missing dependency. Install the required package and "
                    f"retry: {exc}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load recalibrator from {recalibrator_path}. "
                    f"This may indicate version incompatibility or file "
                    f"corruption: {exc}"
                ) from exc

        return metadata, model, recalibrator

    def _update_artifact_path(self: SurvivalLearner, new_path: str) -> None:
        """Set artifact_path, warning if it was already set to a different value."""
        old = self._state.artifact_path
        if old is not None and old != new_path:
            warnings.warn(
                f"artifact_path is being overwritten: "
                f"{old!r} -> {new_path!r}. "
                f"Only the latest path is retained.",
                InternalWarning,
                stacklevel=3,
            )
        self._state.artifact_path = new_path

    def save_learner(self: SurvivalLearner, path: Union[str, Path], *, compress: int = 3):
        """Save the entire learner state to a single file.

        Serializes all learner state (data, models, evaluation results,
        bootstrap CIs, CV metrics) so a session can be resumed later
        without re-running anything.

        Parameters
        ----------
        path : str or Path
            Output file path (e.g. ``'my_analysis.learner'``).
        compress : int
            Compression level for joblib (0-9). Default 3 (zlib).

        Raises
        ------
        ValueError
            If the learner has not been set up yet.
        """
        import joblib

        if not self._state.is_setup:
            raise ValueError(
                "Cannot save: learner has not been set up. Call setup() first."
            )

        import mlsurv

        envelope = {
            '_learner_version': LEARNER_FORMAT_VERSION,
            '_timestamp': datetime.now(timezone.utc).isoformat(),
            '_mlsurv_version': mlsurv.__version__,
            'state': self._state,
        }
        joblib.dump(envelope, path, compress=compress)
        self._update_artifact_path(str(Path(path).resolve()))

    @classmethod
    def load_learner(cls, path: Union[str, Path]) -> SurvivalLearner:
        """Load a saved learner from disk.

        Parameters
        ----------
        path : str or Path
            Path to a file created by :meth:`save_learner`.

        Returns
        -------
        SurvivalLearner
            Fully restored learner instance.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file is not a valid learner file, or the version is
            not in the supported range (1 to ``LEARNER_FORMAT_VERSION``).
        RuntimeError
            If the file cannot be deserialized (e.g., missing dependency,
            file corruption, or version incompatibility).

        Warns
        -----
        UserWarning
            If the file was saved with a different mlsurv version.

        .. warning::

            ``load_learner`` uses pickle internally (via joblib). **Do not
            load files from untrusted sources** — arbitrary code execution
            is possible.
        """
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Learner file not found: {path}")

        try:
            envelope = joblib.load(path)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                f"Failed to load learner from {path}: missing dependency. "
                f"Install the required package and retry: {exc}"
            ) from exc
        except Exception as exc:
            hint = _gpu_state_hint(exc, path)
            if hint:
                raise RuntimeError(hint) from exc
            raise RuntimeError(
                f"Failed to load learner from {path}. The file may be "
                f"corrupt or was not created by save_learner(): {exc}"
            ) from exc

        if not isinstance(envelope, dict):
            raise ValueError(
                f"File at {path} is not a valid learner file (expected dict, "
                f"got {type(envelope).__name__}). Was this file created by "
                f"save_learner()? For model artifacts, use load_artifact()."
            )

        version = envelope.get('_learner_version')
        if not isinstance(version, int) or version < 1 or version > LEARNER_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported learner format version: {version} "
                f"(this mlsurv supports versions 1-{LEARNER_FORMAT_VERSION})"
            )

        state = envelope.get('state')
        if state is None:
            raise ValueError(
                f"Learner file at {path} is missing the 'state' key. "
                f"The file may be corrupt or incompatible."
            )

        saved_version = envelope.get('_mlsurv_version')
        if saved_version is not None:
            import mlsurv
            if saved_version != mlsurv.__version__:
                warnings.warn(
                    f"Learner was saved with mlsurv {saved_version}, "
                    f"but current version is {mlsurv.__version__}. "
                    f"Some features may behave differently.",
                    SuggestionWarning,
                    stacklevel=2,
                )

        learner = cls.__new__(cls)
        learner._state = state
        learner._state.artifact_path = str(path.resolve())
        return learner
