"""Shared helper functions for evaluation module."""

from ..tuning import auto_n_bootstrap
from ..utils import unwrap_pipeline


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds for progress display.

    Parameters
    ----------
    seconds : float
        Elapsed time in seconds.

    Returns
    -------
    str
        Formatted string: ``"3.2s"`` when < 60s, ``"2m05.3s"`` when >= 60s.
    """
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes}m{remaining:04.1f}s"


def resolve_n_bootstrap(n_bootstrap, n_samples, show_progress=True, bootstrap=True):
    """
    Resolve 'auto' n_bootstrap and optionally print message.

    Parameters
    ----------
    n_bootstrap : int or "auto"
        Number of bootstrap iterations or "auto".
    n_samples : int
        Number of samples in test set.
    show_progress : bool
        If True and bootstrap is True, print the resolved value.
    bootstrap : bool
        If True and show_progress, print resolution message.

    Returns
    -------
    int
        Resolved number of bootstrap iterations.
    """
    if n_bootstrap == "auto":
        n_bootstrap = auto_n_bootstrap(n_samples)
        if show_progress and bootstrap:
            print(f"Bootstrap iterations automatically set to {n_bootstrap} "
                  f"(formula: min(500, max(100, 10*sqrt({n_samples}))))")
    return n_bootstrap


def _has_internal_parallelism(model):
    """Check if model has internal parallelism, unwrapping Pipelines if needed.

    Used by both evaluate.py (importance strategy) and bootstrap.py (backend selection).

    Parameters
    ----------
    model : object
        Model instance (BaseSurvivalModel or Pipeline).

    Returns
    -------
    bool
        True if the model has internal n_jobs parallelism.
    """
    inner = unwrap_pipeline(model)
    return getattr(inner, 'has_internal_parallelism', False)


def resolve_model_name(model):
    """
    Resolve model name from a model object.

    Handles:
    - Models with a 'name' attribute
    - Pipelines (gets name from last step)
    - Falls back to class name

    Parameters
    ----------
    model : object
        Model object to get name from.

    Returns
    -------
    str
        Model name.
    """
    # Check for direct name attribute
    name = getattr(model, 'name', None)
    if name is not None:
        return name

    # Check if it's a Pipeline and get the last step's name
    if hasattr(model, 'named_steps'):
        last_step = list(model.named_steps.values())[-1]
        name = getattr(last_step, 'name', None)
        if name is not None:
            return name
        return last_step.__class__.__name__

    # Fall back to class name
    return model.__class__.__name__
