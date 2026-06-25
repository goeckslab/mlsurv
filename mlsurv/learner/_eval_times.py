"""EvalTimesMixin — register additive "report horizons" via add_eval_times().

``add_eval_times([...])`` registers user report horizons (e.g. 12mo, 60mo) that
flow ADDITIVELY into FUTURE time-dependent metrics (AUC(t), Brier(t),
calibration, bootstrap CIs). The registered set is unioned with — not a
replacement for — the effective base grid; the union is clipped to the
censoring-stable window when the metric is computed.

Registration is LAZY: it triggers no recompute. Already-computed results show
NA at newly added times until the user re-runs ``evaluate()``/``bootstrap()``.
A requested time outside the censoring-stable window warns
(``MethodologicalWarning``) but is kept, not silently dropped — it renders as
honest NA.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from .._warnings import MethodologicalWarning
from ._evaluation_helpers import _censoring_window, _validate_eval_times_array

if TYPE_CHECKING:
    from ._core import SurvivalLearner


class EvalTimesMixin:
    """Public API for registering additive report horizons."""

    def add_eval_times(self: SurvivalLearner, times) -> np.ndarray:
        """Register additive report horizons for time-dependent metrics.

        The registered horizons are unioned with (not a replacement for) the
        effective ``eval_times`` grid the next time ``evaluate()`` or
        ``bootstrap()`` runs, then clipped to the censoring-stable window.
        Calling this does **not** recompute anything: already-computed results
        keep their existing grid and show NA at the new horizons (in plots and
        ``return_data``) until you re-run the relevant step.

        A horizon outside the censoring-stable window ``[t_min, t_max)`` of the
        training data warns (:class:`MethodologicalWarning`) because
        IPCW-weighted metrics are not estimable there, but the horizon is still
        registered and renders as NA rather than being silently dropped.

        Parameters
        ----------
        times : array-like of float
            One or more positive, finite timepoints to register. Duplicates
            (with the already-registered set) are collapsed. An empty array-like
            is a no-op.

        Returns
        -------
        np.ndarray
            The full sorted set of registered report horizons after this call.

        Notes
        -----
        Report horizons flow into the cohort-level curves (the learner's own
        ``plot_metric_time`` / ``plot_calibration`` and validation cohorts) and
        into the per-subpopulation AUC(t) facets.

        See Also
        --------
        report_eval_times : Read back the registered horizons.
        evaluate : Recompute metrics on the extended grid.

        Examples
        --------
        >>> learner.setup()
        >>> learner.add_eval_times([12, 60])
        >>> learner.evaluate()        # AUC(t)/Brier(t) now include t=12, 60
        """
        self._ensure_setup()
        requested = _validate_eval_times_array(times, param_name="times")
        if requested.size == 0:
            return self.report_eval_times

        # Warn (but keep) horizons outside the censoring-stable window. IPCW
        # metrics cannot be estimated there, so they render as honest NA.
        t_min, t_max = _censoring_window(self.y_train)
        if t_min is not None:
            out_of_window = requested[(requested < t_min) | (requested >= t_max)]
            if out_of_window.size:
                listed = ", ".join(f"{t:g}" for t in np.sort(out_of_window))
                warnings.warn(
                    f"Requested report horizon(s) [{listed}] fall outside the "
                    f"censoring-stable window [{t_min:g}, {t_max:g}) of the "
                    f"training data. IPCW-weighted metrics (Brier(t), IBS, "
                    f"AUC(t)) are not estimable there and will render as NA. "
                    f"The horizon(s) are kept, not dropped.",
                    MethodologicalWarning,
                    stacklevel=2,
                )

        existing = self._state.user_extra_eval_times
        if existing is None:
            merged = requested
        else:
            merged = np.concatenate([np.asarray(existing, dtype=float), requested])
        self._state.user_extra_eval_times = np.unique(merged)

        # The registered set changed, so a previously emitted "re-run to fill
        # NA" hint may now be incomplete — re-arm it for the next plot.
        self._state.extra_eval_times_render_warned = False
        return self.report_eval_times

    @property
    def report_eval_times(self: SurvivalLearner) -> np.ndarray:
        """Sorted report horizons registered via :meth:`add_eval_times`.

        Returns an empty float array when none are registered.
        """
        extra = self._state.user_extra_eval_times
        if extra is None:
            return np.array([], dtype=float)
        return np.asarray(extra, dtype=float)
