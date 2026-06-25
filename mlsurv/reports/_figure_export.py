"""Bounded static-image export for Plotly figures.

``fig.write_image()`` uses kaleido, which (in 1.x) drives a headless Chrome
subprocess. On some CI runners that browser can deadlock, so the call never
returns and wedges whatever is generating the report. This module runs the
export in a daemon thread with a join timeout so a stalled browser raises
``FigureExportTimeout`` instead of hanging forever. A daemon thread is used so
an abandoned (still-hung) export can never block process exit, and because it
works on every platform (``signal.alarm`` is POSIX-only, and the hang we saw is
on Windows).
"""

import os
import threading

DEFAULT_EXPORT_TIMEOUT = 60.0


def _resolve_default_timeout() -> float:
    """Read the default export timeout, honoring an env override.

    Returns
    -------
    float
        Seconds to wait for a single ``write_image`` call. Falls back to
        ``DEFAULT_EXPORT_TIMEOUT`` when ``MLSURV_FIGURE_EXPORT_TIMEOUT`` is
        unset or not a positive number.
    """
    raw = os.environ.get("MLSURV_FIGURE_EXPORT_TIMEOUT")
    if raw is None:
        return DEFAULT_EXPORT_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        value = None
    if value is None or value <= 0:
        import warnings

        from mlsurv._warnings import SuggestionWarning
        warnings.warn(
            f"Ignoring invalid MLSURV_FIGURE_EXPORT_TIMEOUT={raw!r}; "
            f"using default {DEFAULT_EXPORT_TIMEOUT:g}s.",
            SuggestionWarning, stacklevel=2,
        )
        return DEFAULT_EXPORT_TIMEOUT
    return value


class FigureExportTimeout(RuntimeError):
    """Raised when a static-image export exceeds its timeout."""


def write_image_safe(fig, filepath, *, timeout: float | None = None, **kwargs) -> None:
    """Export *fig* to *filepath* with a bounded timeout.

    Runs ``fig.write_image(str(filepath), **kwargs)`` in a daemon thread. Any
    exception raised by ``write_image`` is re-raised on the caller side, so the
    usual ``ValueError`` / ``OSError`` / ``ImportError`` handling still applies.
    If the export does not finish within *timeout* seconds, raises
    ``FigureExportTimeout`` and leaves the worker thread to be reaped at process
    exit.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to export.
    filepath : str or pathlib.Path
        Destination path. The extension selects the format (kaleido handles
        png/svg/pdf/jpeg/webp).
    timeout : float, optional
        Seconds to wait before giving up. Defaults to the value from
        ``MLSURV_FIGURE_EXPORT_TIMEOUT`` or ``DEFAULT_EXPORT_TIMEOUT``.
    **kwargs
        Forwarded to ``fig.write_image`` (e.g. ``scale``, ``width``).

    Raises
    ------
    FigureExportTimeout
        If the export does not complete within *timeout* seconds.
    """
    if timeout is None:
        timeout = _resolve_default_timeout()

    error: list[BaseException] = []

    def _export() -> None:
        try:
            fig.write_image(str(filepath), **kwargs)
        except BaseException as exc:  # re-raised on the caller side below
            error.append(exc)

    thread = threading.Thread(target=_export, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise FigureExportTimeout(
            f"Static-image export of {filepath} did not finish within "
            f"{timeout:g}s (kaleido/Chrome may be unresponsive)."
        )

    if error:
        raise error[0]
