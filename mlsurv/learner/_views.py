"""View proxies exposing plot methods for alternative data sources.

This module hosts :class:`ValidationView` and :class:`SubpopulationView`.
A view is a lightweight proxy returned by ``learner.validation(cohort)``
/ ``learner.subpopulation(group)`` that exposes the same ``plot_*``
methods as the learner itself but sourced from a cohort's
:class:`~mlsurv.results.containers.ValidationResult` or a subpopulation
variable's :class:`~mlsurv.results.containers.SubpopulationResult`
instead of the main test split.

:class:`ValidationView` exposes ``plot_metric``, ``plot_metric_time``,
``plot_survival``, ``plot_calibration``, and ``plot_roc`` so validation
cohorts reach parity with the main-path plot surface area.
:class:`SubpopulationView` exposes the same set, rendering
``plot_calibration`` and ``plot_roc`` as faceted multi-panel grids.
The main-path refactor remains a future follow-up
— the architecture is designed so each addition is a thin forwarder plus
a new ``_plot_X_impl`` source branch, not a new method on
:class:`SurvivalLearner`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlsurv.viz._context import PlotContext, SubpopulationAxis
from mlsurv.viz._plot_impl import (
    _plot_calibration_impl,
    _plot_metric_impl,
    _plot_metric_time_impl,
    _plot_roc_impl,
    _plot_survival_impl,
)

if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner
    from mlsurv.results.containers import (
        SubpopulationResult,
        ValidationResult,
    )


class ValidationView:
    """Plot-method proxy bound to one validation cohort.

    Returned by :meth:`SurvivalLearner.validation`. The proxy forwards
    its plot calls to shared ``_plot_X_impl`` functions that read from
    the cohort's ``ValidationResult`` instead of the learner's main test
    split. Fitted models and the IPCW training baseline (``y_train``) are
    always inherited from the main learner state — validation never
    refits, and the weighting distribution is fixed across cohorts.

    Parameters
    ----------
    learner : SurvivalLearner
        The learner that owns the cohort. The view keeps a reference so
        repeated plot calls observe any intervening state changes.
    cohort : str
        Cohort name, must be a key in
        ``learner._state.validation_results``.

    Raises
    ------
    KeyError
        If *cohort* is not present in the learner's validation results.

    Examples
    --------
    >>> learner.validate(X_val, y_val, name="Holdout")
    >>> view = learner.validation("Holdout")
    >>> fig_auc = view.plot_metric_time()
    >>> fig_cal = view.plot_calibration(eval_times=24.0)
    >>> fig_roc = view.plot_roc(t=24.0)
    """

    def __init__(self, learner: "SurvivalLearner", cohort: str):
        if cohort not in learner._state.validation_results:
            available = sorted(learner._state.validation_results.keys())
            raise KeyError(
                f"Validation cohort {cohort!r} not found. "
                f"Available: {available or 'none'}. "
                f"Call learner.validate(X_val, y_val, name=...) first."
            )
        self._learner = learner
        self._cohort = cohort

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cohort(self) -> str:
        """The cohort name this view is bound to."""
        return self._cohort

    @property
    def result(self) -> "ValidationResult":
        """The underlying :class:`ValidationResult` for this cohort."""
        return self._learner._state.validation_results[self._cohort]

    def __repr__(self) -> str:
        return f"ValidationView(cohort={self._cohort!r})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ctx(self) -> PlotContext:
        """Build a fresh :class:`PlotContext` for this cohort on every call.

        Not cached, so a user mutating the learner between view
        construction and a plot call sees the latest state.
        """
        return PlotContext.from_validation(self._learner, self.result)

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------

    def plot_metric_time(self, **kwargs):
        """Plot time-dependent metric curves for this validation cohort.

        Parameters
        ----------
        **kwargs
            Forwarded to
            :func:`mlsurv.viz._plot_impl._plot_metric_time_impl`. Common
            options: ``models=``, ``metric=['auc', 'brier']`` (multi-panel),
            ``return_data=``. When ``metric=`` is a list, Brier curves are
            computed on the cohort using the main learner's training set as
            the IPCW baseline (no refit).

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        return _plot_metric_time_impl(self._ctx(), **kwargs)

    def plot_metric(self, **kwargs):
        """Plot a discrimination/Brier metric for this validation cohort.

        Parameters
        ----------
        **kwargs
            Forwarded to
            :func:`mlsurv.viz._plot_impl._plot_metric_impl`. Common
            options: ``models=``, ``metric=``, ``source=`` (``"auto"``,
            ``"bootstrap"``, or ``"test"``; ``"cv"`` is rejected for
            validation cohorts), ``style=``, ``return_data=``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        return _plot_metric_impl(self._ctx(), **kwargs)

    def plot_survival(self, **kwargs):
        """Plot KM curves stratified by risk score for this validation cohort.

        Parameters
        ----------
        **kwargs
            Forwarded to
            :func:`mlsurv.viz._plot_impl._plot_survival_impl`. Common
            options: ``models=``, ``n_groups=``, ``time_limit=``,
            ``show_risk_table=``, ``return_data=``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        return _plot_survival_impl(self._ctx(), **kwargs)

    def plot_calibration(self, **kwargs):
        """Plot calibration curve(s) for this validation cohort.

        Scopes automatically to validated models that support
        ``predict_survival_function``.

        Parameters
        ----------
        **kwargs
            Forwarded to
            :func:`mlsurv.viz._plot_impl._plot_calibration_impl`. Common
            options: ``models=``, ``eval_times=`` (a single time or a list;
            the comparison uses one horizon), ``n_bins=``,
            ``bin_strategy=`` (``"width"`` or ``"quantile"``), ``return_data=``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        return _plot_calibration_impl(self._ctx(), **kwargs)

    def plot_roc(self, **kwargs):
        """Plot time-dependent ROC curves for this validation cohort.

        IPCW weights are derived from the *main* training outcome
        (``y_train``), not the cohort outcomes — the censoring
        distribution used for weighting must match training.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`mlsurv.viz._plot_impl._plot_roc_impl`.
            Common options: ``models=``, ``t=``, ``show_auc=``,
            ``show_ci=``, ``return_data=``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        return _plot_roc_impl(self._ctx(), **kwargs)


class SubpopulationView:
    """Plot-method proxy bound to one subpopulation variable.

    Returned by :meth:`SurvivalLearner.subpopulation`. The proxy forwards
    its plot calls to shared ``_plot_X_impl`` functions that read from
    the learner's stored :class:`~mlsurv.results.containers.SubpopulationResult`
    (via a :class:`~mlsurv.viz._context.SubpopulationAxis`) instead of the
    main test split. Metrics are pre-aggregated at evaluation time; no
    re-prediction or re-fitting happens here.

    Parameters
    ----------
    learner : SurvivalLearner
        The learner that owns the subpopulation results. The view keeps a
        reference so repeated plot calls observe any intervening state
        changes.
    group : str
        Subpopulation variable name. Must be a key in
        ``learner._state.subpopulation_results``.

    Raises
    ------
    KeyError
        If *group* is not present in the learner's subpopulation results.

    Examples
    --------
    >>> learner.setup(subpopulations={"menostat": menostat_labels})
    >>> learner.run(["coxph", "rsf"])
    >>> view = learner.subpopulation("menostat")
    >>> fig_forest = view.plot_metric(style="forest")
    >>> fig_auct = view.plot_metric_time()
    >>> fig_km = view.plot_survival()
    """

    def __init__(self, learner: "SurvivalLearner", group: str):
        # LearnerState.subpopulation_results is always a dict (possibly
        # empty) via field(default_factory=dict); no defensive getattr
        # needed per CONTRIBUTING.md state-access rules.
        if group not in learner._state.subpopulation_results:
            available = sorted(learner._state.subpopulation_results.keys())
            raise KeyError(
                f"Subpopulation variable {group!r} not found. "
                f"Available: {available or 'none'}. "
                f"Call learner.setup(subpopulations={{...}}) first."
            )
        self._learner = learner
        self._group = group

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def group(self) -> str:
        """The subpopulation variable name this view is bound to."""
        return self._group

    @property
    def result(self) -> "SubpopulationResult":
        """The underlying :class:`SubpopulationResult` for this variable."""
        return self._learner._state.subpopulation_results[self._group]

    def __repr__(self) -> str:
        return f"SubpopulationView(group={self._group!r})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ctx(self) -> PlotContext:
        """Build a fresh :class:`PlotContext` on every call.

        Not cached, so a user mutating the learner between view
        construction and a plot call sees the latest state.
        """
        return PlotContext.from_subpopulation(self._learner, self._group)

    def _axis(self) -> SubpopulationAxis:
        """Build a fresh :class:`SubpopulationAxis` on every call."""
        return SubpopulationAxis.from_result(self.result)

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------

    def plot_metric(self, **kwargs):
        """Plot a grouped forest of discrimination metrics per subpopulation level.

        One row per ``(model, subpopulation level)`` pair (or one row
        per model if ``level=`` is set), with bootstrap confidence
        intervals when ``SubpopulationResult.bootstrap_summary`` is
        populated.

        Parameters
        ----------
        **kwargs
            Forwarded to
            :func:`mlsurv.viz._plot_impl._plot_metric_impl`. Only
            ``style='forest'`` / ``'auto'`` is supported in v1. Common
            options: ``models=``, ``metric=``,
            ``level=`` (single-level shorthand), ``return_data=``.

        Returns
        -------
        plotly.graph_objects.Figure, pandas.DataFrame, or None
            Returns ``None`` on the edge case where the underlying
            metrics frame has no usable point estimates for the
            selected metric (all-NaN after filtering) — matching the
            ``plot_grouped_forest`` contract. Report sections treat a
            ``None`` return as "no figure available, skip".
        """
        return _plot_metric_impl(
            self._ctx(), group_axis=self._axis(), **kwargs,
        )

    def plot_metric_time(self, **kwargs):
        """Plot faceted time-dependent metric curves per subpopulation level.

        Default layout: one panel per subpopulation level showing AUC(t)
        (or a single panel if ``level=`` is set). Pass
        ``metric=['auc', 'brier']`` for a 2D grid with one row per
        metric and one column per subgroup; Brier curves are computed
        per subgroup on-the-fly using the main learner's training set as
        the IPCW baseline. ``level=`` is not supported with multi-metric
        mode (``metric=[...]`` or a non-AUC metric) in v1 — call once per
        metric instead.

        Single-metric AUC ribbons show ±1 SD from bootstrap resampling
        when ``SubpopulationResult.auc_t_bootstrap`` is populated.

        Parameters
        ----------
        **kwargs
            Forwarded to
            :func:`mlsurv.viz._plot_impl._plot_metric_time_impl`. Common
            options: ``models=``, ``metric=['auc', 'brier']``,
            ``level=`` (single-level shorthand; single-metric only),
            ``return_data=``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
            Always returns a Figure (or a DataFrame when
            ``return_data=True``) in v1. The underlying
            :func:`plot_grouped_metric_time` ``None`` branch for
            empty / missing ``auc_t`` is shadowed by an impl-layer
            ``ValueError`` guard, so this method never returns
            ``None``. A future refactor that removes the impl guard
            should update this docstring before the ``None`` path
            becomes reachable.
        """
        return _plot_metric_time_impl(
            self._ctx(), group_axis=self._axis(), **kwargs,
        )

    def plot_survival(self, **kwargs):
        """Plot Kaplan-Meier curves stratified by subpopulation level.

        Uses ``learner._state.y_test`` stratified on
        ``learner._state.subpopulations_test[group]``. Model-free — the
        curves are a property of the test outcomes, not the fitted
        estimators. Pass ``level=`` to render the KM curve for a single
        level only.

        Parameters
        ----------
        **kwargs
            Forwarded to
            :func:`mlsurv.viz._plot_impl._plot_survival_impl`. Common
            options: ``time_limit=``, ``show_risk_table=``,
            ``level=`` (single-level shorthand), ``return_data=``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        return _plot_survival_impl(
            self._ctx(), group_axis=self._axis(), **kwargs,
        )

    def plot_calibration(self, **kwargs):
        """Plot calibration curves faceted by subpopulation level.

        One panel per subpopulation level (or a single panel if
        ``level=`` is set). Calibration data is computed at plot time
        by calling ``predict_survival_function()`` on each fitted model
        for each level's filtered test set.

        Parameters
        ----------
        **kwargs
            Forwarded to
            :func:`mlsurv.viz._plot_impl._plot_calibration_impl`. Common
            options: ``models=``, ``eval_times=`` (one horizon used for the
            facets), ``n_bins=``, ``bin_strategy=`` (``"width"`` or
            ``"quantile"``), ``level=`` (single-level shorthand),
            ``return_data=``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        return _plot_calibration_impl(
            self._ctx(), group_axis=self._axis(), **kwargs,
        )

    def plot_roc(self, **kwargs):
        """Plot time-dependent ROC curves faceted by subpopulation level.

        One panel per subpopulation level (or a single panel if
        ``level=`` is set). Risk scores and IPCW-weighted ROC curves
        are computed at plot time. IPCW weights always use the *main*
        training outcome (``y_train``), not per-level outcomes.

        Parameters
        ----------
        **kwargs
            Forwarded to
            :func:`mlsurv.viz._plot_impl._plot_roc_impl`. Common
            options: ``models=``, ``t=``, ``show_auc=``,
            ``level=`` (single-level shorthand), ``return_data=``.

        .. note::
            Bootstrap CI bands are not available on the subpopulation ROC
            surface in v1 (no per-level bootstrap data). ``show_ci="auto"``
            (the default) draws the point curve silently; an explicit
            ``show_ci=True`` also draws the point curve but emits a
            ``SuggestionWarning``; ``show_ci=False`` raises
            ``NotImplementedError``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        return _plot_roc_impl(
            self._ctx(), group_axis=self._axis(), **kwargs,
        )
