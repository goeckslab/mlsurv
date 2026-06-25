"""Optuna storage configuration and connection helpers."""

import os
from pathlib import Path

import optuna

from ..logging_utils import get_logger

logger = get_logger("tuning")

# Default base directory for Optuna SQLite storage
DEFAULT_STORAGE_BASE = Path('./optuna_studies')

# Environment variable for Optuna storage URL
OPTUNA_STORAGE_ENV_VAR = 'MLSURV_OPTUNA_STORAGE'


def get_storage_url(storage=None):
    """Resolve Optuna storage URL from explicit parameter or environment variable.

    Priority: explicit ``storage`` > ``MLSURV_OPTUNA_STORAGE`` env var > None.

    Parameters
    ----------
    storage : str, optional
        Explicit storage URL (e.g., 'postgresql://user:pass@host:port/db').

    Returns
    -------
    str or None
        Resolved storage URL, or None if no storage configured.
    """
    # Priority 1: Explicit parameter
    if storage is not None:
        return storage

    # Priority 2: Environment variable
    env_storage = os.environ.get(OPTUNA_STORAGE_ENV_VAR)
    if env_storage:
        return env_storage

    # No storage configured — caller handles SQLite fallback when appropriate
    return None


def validate_storage_connection(storage):
    """
    Test connection to an Optuna storage backend.

    Parameters
    ----------
    storage : str
        Storage URL (e.g., 'postgresql://user:pass@host:port/db' or
        'sqlite:///path/to/db.sqlite').

    Returns
    -------
    dict
        Connection status with keys:
        - valid : bool - True if connection succeeded
        - backend : str - Storage backend type ('postgresql', 'sqlite', etc.)
        - message : str - Human-readable status message
        - studies : list - List of study names if connection succeeded

    Raises
    ------
    MemoryError
        Re-raised if the computation exhausts available memory; resource
        errors are never swallowed by the broad failure handler.
    RecursionError
        Re-raised if the computation exceeds the recursion limit; resource
        errors are never swallowed by the broad failure handler.

    Examples
    --------
    >>> result = validate_storage_connection('postgresql://localhost/optuna')
    >>> if result['valid']:
    ...     print(f"Connected! Found {len(result['studies'])} studies.")
    """
    result = {
        'valid': False,
        'backend': 'unknown',
        'message': '',
        'studies': [],
    }

    if storage.startswith(('postgresql://', 'postgres://')):
        result['backend'] = 'postgresql'
    elif storage.startswith('sqlite:///'):
        result['backend'] = 'sqlite'
    elif storage.startswith('mysql://'):
        result['backend'] = 'mysql'

    try:
        storage_obj = optuna.storages.RDBStorage(storage)
        studies = optuna.study.get_all_study_summaries(storage=storage_obj)
        result['studies'] = [s.study_name for s in studies]
        result['valid'] = True
        result['message'] = f"Connected to {result['backend']} storage. Found {len(studies)} studies."

    except (MemoryError, RecursionError):
        raise
    except Exception as e:
        logger.debug("Storage connection failed: %s: %s", type(e).__name__, e)
        result['message'] = f"Connection failed ({type(e).__name__}): {e}"

    return result
