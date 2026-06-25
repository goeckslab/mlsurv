"""Structured logging utilities for mlsurv.

Provides lazy logger creation controlled by the MLSURV_LOG_LEVEL environment
variable (default: WARNING).  The ``log_exception`` and ``log_metric_failure``
helpers standardise how caught exceptions are recorded: a configurable-level
message (default ERROR) for the summary, a DEBUG-level message with the full
traceback, and an optional ``warnings.warn`` for user-visible feedback.  When a
*visible* tiered warning is emitted for the same failure, the summary is
downgraded to DEBUG so it is not duplicated on the default stderr stream.
"""

from __future__ import annotations

import logging
import os
import traceback
import warnings

# Keyword fragment in sksurv's ValueError when evaluation times fall outside
# the test fold's follow-up window — an expected condition in nested/temporal CV.
_FOLD_BOUNDARY_KEYWORD = "all times must be within follow-up time"

_root_configured = False

# Keys that have already been logged at their designated level; subsequent
# occurrences are silently downgraded to DEBUG.
_seen_log_once_keys: set[str] = set()

# Keys that have already triggered a user-visible warnings.warn(); subsequent
# occurrences are silent (the logger record still fires per _log_once_level).
_seen_warn_once_keys: set[str] = set()


def _warn_once_emit(key: str) -> bool:
    """Return True the first time *key* is seen, False thereafter."""
    if key in _seen_warn_once_keys:
        return False
    _seen_warn_once_keys.add(key)
    return True


def _is_fold_boundary_error(exc: BaseException) -> bool:
    """Return True if *exc* is the sksurv fold-boundary ValueError."""
    return (
        isinstance(exc, ValueError)
        and _FOLD_BOUNDARY_KEYWORD in str(exc).lower()
    )


def _fold_boundary_message(metric_name: str) -> str:
    """User-facing wording for partial fold-boundary skips in CV.

    Use this at call sites that have already classified the exception as a
    fold-boundary skip (via :func:`_is_fold_boundary_error`): at least one
    fold's eval-times fell outside that fold's follow-up window, but other
    folds succeed and the CV summary remains finite. The default
    ``log_metric_failure`` wording (``"<metric> computation failed: ..."``)
    misleads users into thinking the metric was wholly unavailable.
    """
    return (
        f"{metric_name} skipped on at least one CV fold because eval-times "
        f"fell outside that fold's follow-up window. Affected folds returned "
        f"NaN; the CV summary is averaged over the successful folds."
    )


def _log_once_level(key: str, level: int = logging.WARNING) -> int:
    """Return *level* the first time *key* is seen, ``DEBUG`` thereafter."""
    if key in _seen_log_once_keys:
        return logging.DEBUG
    _seen_log_once_keys.add(key)
    return level


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``mlsurv.*`` namespace.

    On first call, the ``mlsurv`` root logger is configured with a
    ``StreamHandler`` at the level given by the ``MLSURV_LOG_LEVEL``
    environment variable (default ``WARNING``).
    """
    global _root_configured
    if not _root_configured:
        level_name = os.environ.get("MLSURV_LOG_LEVEL", "WARNING").upper()
        level = getattr(logging, level_name, logging.WARNING)
        root = logging.getLogger("mlsurv")
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(name)s [%(levelname)s] %(message)s")
            )
            root.addHandler(handler)
        root.setLevel(level)
        _root_configured = True

    return logging.getLogger(f"mlsurv.{name}")


def log_exception(
    logger: logging.Logger,
    exception: BaseException,
    context: str,
    *,
    level: int = logging.ERROR,
    warn_user: bool = False,
    user_message: str | None = None,
    category: type | None = None,
) -> None:
    """Log a caught exception with optional user warning.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance (from ``get_logger``).
    exception : BaseException
        The caught exception.
    context : str
        Free-form description of what was being attempted.
    level : int
        Log level for the summary message (default ``logging.ERROR``).
        Downgraded to ``DEBUG`` when a *visible* warning is emitted for the
        same failure (i.e. *warn_user* is *True* and the resolved *category*
        is not ``InternalWarning`` or a subclass), so the failure is not
        reported twice.
    warn_user : bool
        If *True*, also emit ``warnings.warn`` with *user_message* (or a
        default built from *context* and *exception*).
    user_message : str or None
        Custom message for ``warnings.warn``.  Ignored when *warn_user* is
        *False*.
    category : type of Warning or None
        Warning class for the user warning. Defaults to ``InternalWarning``
        when *None*. Ignored when *warn_user* is *False*.
    """
    from ._warnings import InternalWarning
    if warn_user and category is None:
        category = InternalWarning

    # When a user-visible tiered warning carries this same failure to the user,
    # drop the summary to DEBUG so it doesn't duplicate on the default stderr
    # stream (the warning is the user-facing channel). InternalWarning is
    # suppressed by default, so in that case the log line is the *only*
    # user-facing channel and must stay at `level`.
    summary_level = level
    if warn_user and not issubclass(category, InternalWarning):
        summary_level = logging.DEBUG

    logger.log(
        summary_level,
        "%s: %s: %s (set MLSURV_LOG_LEVEL=DEBUG for full traceback)",
        context, type(exception).__name__, exception,
    )
    logger.debug(
        "%s — full traceback:\n%s",
        context,
        "".join(traceback.format_exception(type(exception), exception, exception.__traceback__)),
    )
    if warn_user:
        msg = user_message or f"{context}: {exception}"
        warnings.warn(msg, category, stacklevel=3)


def log_metric_failure(
    logger: logging.Logger,
    metric_name: str,
    exception: BaseException,
    *,
    level: int = logging.ERROR,
    fold: int | str | None = None,
    iteration: int | str | None = None,
    model_name: str | None = None,
    warn_user: bool = False,
    warn_once_key: str | None = None,
    warn_category: type | None = None,
    user_message: str | None = None,
) -> None:
    """Log a metric-computation failure with structured context.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    metric_name : str
        Name of the metric that failed (e.g. ``"brier"``, ``"ipcw"``).
    exception : BaseException
        The caught exception.
    level : int
        Log level for the summary message (default ``logging.ERROR``).
    fold : int, str, or None
        Fold context tag included in the log message when provided.
    iteration : int, str, or None
        Iteration context tag included in the log message when provided.
    model_name : str or None
        Model-name context tag included in the log message when provided.
    warn_user : bool
        If *True*, always emit ``warnings.warn``.
    warn_once_key : str or None
        If provided, emit ``warnings.warn`` only on the first occurrence of
        *warn_once_key* per process (subsequent calls are silent at the
        warnings layer; the logger record is unaffected). Mirrors
        ``_log_once_level`` so expected recurring failures (e.g. fold-boundary
        errors in temporal CV) still satisfy the No-Silent-Data-Loss rule.
        Note: the dedup set is process-global, so warnings emitted inside
        ``joblib`` worker processes never reach the parent. The
        first call exists for ``n_jobs=1`` and parent-process retry paths.
    warn_category : type or None
        Category for ``warnings.warn``. Defaults to ``MetricFailureWarning``
        when *warn_once_key* fires, otherwise to ``InternalWarning`` (handled
        by ``log_exception``).
    user_message : str or None
        Custom wording for ``warnings.warn``. When *None* (default), the
        message is built as ``f"{metric_name} computation failed: {exception}"``.
        Use this when the call site wants to clarify a partial failure mode
        (e.g. fold-boundary skips where other folds still succeeded) — see
        :func:`_fold_boundary_message`.
    """
    parts = [f"metric={metric_name}"]
    if model_name is not None:
        parts.append(f"model={model_name}")
    if fold is not None:
        parts.append(f"fold={fold}")
    if iteration is not None:
        parts.append(f"iteration={iteration}")
    context = ", ".join(parts)

    # Always consume the once-token when warn_once_key is set so that a later
    # warn_once_key-only caller doesn't re-warn for the same key. warn_user=True
    # forces emission unconditionally; warn_once_key gates emission on first
    # occurrence; the two together emit and consume the token.
    once_emit = warn_once_key is not None and _warn_once_emit(warn_once_key)
    emit_warning = warn_user or once_emit
    category = warn_category
    if emit_warning and not warn_user and category is None:
        from ._warnings import MetricFailureWarning
        category = MetricFailureWarning

    if emit_warning:
        resolved_user_message = (
            user_message
            if user_message is not None
            else f"{metric_name} computation failed: {exception}"
        )
    else:
        resolved_user_message = None

    log_exception(
        logger,
        exception,
        context,
        level=level,
        warn_user=emit_warning,
        user_message=resolved_user_message,
        category=category,
    )
