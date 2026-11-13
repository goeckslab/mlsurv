"""VisualizationMixin - All plot_* methods.

Thin facade that resolves ``self._state`` and delegates to submodule
functions. The actual logic lives in:

- ``mlsurv.viz._resolve``      — shared resolution utilities
- ``mlsurv.viz._data``         — domain helpers (features, survival, time,
  calibration, tuning, benchmarking)
- ``mlsurv.viz._data_metrics`` — metrics domain helpers

These live in :mod:`mlsurv.viz` (the viz data-prep layer); this mixin imports
them downward, keeping the ``learner -> viz`` dependency direction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ..logging_utils import get_logger

from mlsurv.viz import _data_metrics as _vim
from mlsurv.viz import _resolve as _vh

logger = get_logger("learner")

if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner


class VisualizationMixin:
    """Mixin for visualization methods (plot_*)."""

    # ------------------------------------------------------------------
    # View proxies (alternate data sources)
    # ------------------------------------------------------------------

    def validation(self: SurvivalLearner, cohort=None):
        """Return a :class:`ValidationView` bound to one validation cohort.

        The view exposes ``plot_*`` methods that read from the cohort's
        :class:`~mlsurv.results.containers.ValidationResult` instead of
        the learner's main test split.

        Parameters
        ----------
        cohort : str, optional
            Cohort name. If omitted and exactly one cohort is registered,
            that cohort is selected automatically. If omitted and 0 or 2+
            cohorts exist, ``ValueError`` is raised.

        Returns
        -------
        ValidationView

        Raises
        ------
        ValueError
            If *cohort* is ``None`` and the number of registered cohorts
            is not exactly 1, or if no cohorts are registered at all.
        KeyError
            If *cohort* is not a registered cohort name.

        Examples
        --------
        >>> learner.validate(X_val, y_val, name="Holdout")
        >>> fig = learner.validation("Holdout").plot_metric_time()
        >>> fig = learner.validation().plot_metric_time()  # single-cohort shorthand
        """
        cohorts = list(self.validation_results)
        if cohort is None:
            if len(cohorts) == 0:
                raise ValueError(
                    "No validation cohorts registered. Call "
                    "learner.validate(X_val, y_val, name=...) first."
                )
            if len(cohorts) > 1:
                raise ValueError(
                    f"Multiple validation cohorts {cohorts!r}; "
                    f"pass cohort= to select one."
                )
            cohort = cohorts[0]
        from mlsurv.learner._views import ValidationView
        return ValidationView(self, cohort)

    def subpopulation(self: SurvivalLearner, group=None):
        """Return a :class:`SubpopulationView` bound to one subpopulation variable.

        The view exposes ``plot_*`` methods that read from the stored
        :class:`~mlsurv.results.containers.SubpopulationResult` (pre-aggregated
        at evaluation / benchmark time) instead of the learner's main
        test-set risk scores.

        Parameters
        ----------
        group : str, optional
            Subpopulation variable name. If omitted and exactly one
            variable is configured, that variable is selected
            automatically. If omitted and 0 or 2+ variables exist,
            ``ValueError`` is raised — mirroring
            :meth:`SurvivalLearner.validation`.

        Returns
        -------
        SubpopulationView

        Raises
        ------
        ValueError
            If *group* is ``None`` and the number of configured
            subpopulation variables is not exactly 1, or if none are
            configured at all.
        KeyError
            If *group* is not a configured subpopulation variable name.

        Examples
        --------
        >>> learner.setup(subpopulations={"menostat": menostat_labels})
        >>> learner.run(["coxph", "rsf"])
        >>> fig = learner.subpopulation("menostat").plot_metric(style="forest")
        >>> fig = learner.subpopulation().plot_metric_time()  # single-variable shorthand
        """
        groups = list(self.subpopulation_results)
        if group is None:
            if len(groups) == 0:
                raise ValueError(
                    "No subpopulation variables configured. Call "
                    "learner.setup(subpopulations={...}) first."
                )
            if len(groups) > 1:
                raise ValueError(
                    f"Multiple subpopulation variables {groups!r}; "
                    f"pass group= to select one."
                )
            group = groups[0]
        from mlsurv.learner._views import SubpopulationView
        return SubpopulationView(self, group)

    # ------------------------------------------------------------------
    # Shared helpers (thin wrappers)
    # ------------------------------------------------------------------

    def _resolve_best(self: SurvivalLearner, model):
        """Map the ``'best'`` sentinel to ``best_model.name``; else pass through.

        Scoped to the singular ``model=`` plot methods so ``'best'`` works
        there too without leaking into the shared ``_resolve_single_model``
        (used by evaluate/bootstrap/etc.). The plural ``models=`` methods
        resolve ``'best'`` in ``viz._plot_impl_shared._resolve_models_arg``.

        Parameters
        ----------
        model : None, str, or ModelResult
            A model name, ``'best'`` sentinel, ``None``, or a ``ModelResult``.

        Returns
        -------
        None, str, or ModelResult
            ``best_model.name`` when *model* is ``'best'``; otherwise *model*
            unchanged.

        Raises
        ------
        ValueError
            If *model* is ``'best'`` but no models have been trained.
        """
        if model == "best":
            best = self.best_model
            if best is None:
                raise ValueError(
                    "No trained models to resolve model='best'. "
                    "Run train() or run() first."
                )
            return best.name
        return model

    def _resolve_eval_result(self: SurvivalLearner, model_names=None):
        """Get an EvaluationResult, optionally filtered to *model_names*."""
        return _vh.resolve_eval_result(self._state, model_names)

    # Backward compat: tests import VisualizationMixin._filter_eval_result
    _filter_eval_result = staticmethod(_vh.filter_eval_result)

    # Backward compat: tests import VisualizationMixin._validate_risk_scores
    _validate_risk_scores = staticmethod(_vh.validate_risk_scores)

    _SOURCE_LABELS = _vim.SOURCE_LABELS

    # ==================================================================
    # plot_metric
    # ==================================================================

    def plot_metric(self: SurvivalLearner, models=None, style="auto",
                    metric=None, source="auto",
                    uncertainty="auto",
                    orientation="horizontal",
                    return_data: bool = False, references=None, **kwargs):
        """Plot a metric visualization with automatic source selection.

        Source priority: bootstrap > test > CV.

        Parameters
        ----------
        models : None, str, list[str], or 'best'
            Models to plot. ``None`` (default) uses all trained models;
            ``'best'`` resolves to the top model by CV score
            (``learner.best_model.name``).
        style : str
            ``"auto"``, ``"forest"``, ``"bar"``, ``"raincloud"``, ``"boxplot"``.
        metric : str or list of str, optional
            Metric name(s). A single name (e.g. ``"harrell"``, ``"uno"``) or a
            1-element list draws one panel; ``None`` uses the learner's scoring
            metric. A list of two or more names (e.g. ``["harrell", "ipcw"]``)
            draws a multi-panel plot. Multi-panel works with any source
            (``"bootstrap"``, ``"test"``, ``"train"``, or ``"cv"``); CV tracks
            several metrics per fold, so all of them are available. Iteration
            styles (``"raincloud"``, ``"boxplot"``) need per-iteration data, so
            they require ``source="bootstrap"`` or ``source="cv"`` -- they are
            unsupported with ``source="test"`` or ``source="train"`` (both are
            point estimates with no distribution).
        source : str
            ``"auto"`` (default), ``"bootstrap"``, ``"test"``, ``"train"``, or
            ``"cv"``. ``"train"`` is an opt-in diagnostic: the four apparent
            (resubstitution) train-set metrics (Harrell, Uno, Antolini, IBS)
            as bare point bars (no error bars), shown only when explicitly
            requested -- never selected by ``"auto"`` (priority is
            bootstrap > test > CV). The gap between train and test/CV is an
            optimism/overfitting signal.
        uncertainty : str or int
            ``"auto"`` (default) picks SD for CV, 95% CI for bootstrap.
            ``"sd"`` shows +/- 1 SD whiskers/error bars.
            A float (e.g. ``0.95``) shows that CI level instead.
        orientation : str
            ``"horizontal"`` (default) or ``"vertical"``.
        return_data : bool, default=False
            If ``True``, return the underlying DataFrame instead of a figure.
        references : None, ``"all"``, or list of str
            Include references from ``benchmark()`` results alongside models.
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Notes
        -----
        For a model with an attached calibrator (``train(calibrate=...)``) the
        Brier/IBS bars already reflect the calibrated ``S(t|x)``, because
        evaluate()/bootstrap() apply the calibrator inline — there is no
        plot-level calibrate/raw switch here.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl import _plot_metric_impl

        ctx = PlotContext.from_main(self)
        return _plot_metric_impl(
            ctx, models=models, style=style, metric=metric,
            source=source, uncertainty=uncertainty,
            orientation=orientation, return_data=return_data,
            references=references, **kwargs,
        )

    # ==================================================================
    # plot_survival
    # ==================================================================

    def plot_survival(self: SurvivalLearner, models=None, n_groups=None,
                      show_risk_table=None, time_limit=None,
                      return_data: bool = False, references=None,
                      breaks=None, break_type: str = "quantile",
                      correction: str = "bonferroni", **kwargs):
        """
        Plot Kaplan-Meier curves stratified by risk score.

        Parameters
        ----------
        models : None, str, list[str], or 'best'
            Models to plot. ``None`` shows a panel grid of all trained
            models; ``'best'`` resolves to the top model by CV score
            (``learner.best_model.name``).
        n_groups : int, optional
            Number of equal-frequency risk groups. Ignored when ``breaks``
            is given.
        show_risk_table : bool, optional
            Show number-at-risk table.
        time_limit : float, optional
            Maximum time to display.
        return_data : bool, default=False
            If ``True``, return a tidy ``pd.DataFrame``.
        references : None, ``"all"``, or list of str
            Include references from ``benchmark()`` results alongside models.
        breaks : list, optional
            Custom group boundaries (overrides ``n_groups``). Same vocabulary
            as ``rmst_test``/``logrank_test``.

            - ``break_type="quantile"`` (default): fractions in (0, 1).
              E.g. ``[0.25, 0.5, 0.75]`` -> four groups at 0-25 / 25-50 /
              50-75 / 75-100 percentiles of the risk score.
            - ``break_type="score"``: raw risk-score cut-points.
              E.g. ``[3, 20, 35]`` -> four groups ``< 3``, ``3-20``,
              ``20-35``, ``>= 35``.
        break_type : {"quantile", "score", "maxstat"}, default="quantile"
            Interpretation of ``breaks``. ``"maxstat"`` finds the cutpoint(s)
            that maximally separate the curves via the log-rank test
            (ignores ``breaks``).
        correction : {"bonferroni", "lausen-schumacher", "permutation"}, \
default="bonferroni"
            Multiple-testing correction for ``break_type="maxstat"``.
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl import _plot_survival_impl

        self._ensure_setup()

        ctx = PlotContext.from_main(self)
        return _plot_survival_impl(
            ctx, models=models, n_groups=n_groups,
            show_risk_table=show_risk_table, time_limit=time_limit,
            return_data=return_data, references=references,
            breaks=breaks, break_type=break_type, correction=correction,
            **kwargs,
        )

    # ==================================================================
    # plot_importance
    # ==================================================================

    def plot_importance(self: SurvivalLearner, models=None, n=20,
                        style: str = "auto",
                        return_data: bool = False, **kwargs):
        """
        Plot feature importance.

        Parameters
        ----------
        models : None, str, list[str], or 'best'
            Models to plot. ``'best'`` resolves to the top model by CV
            score (``learner.best_model.name``).
        n : int, default=20
            Number of top features to display.
        style : str, default="auto"
            ``"auto"``, ``"bar"``, ``"forest"``, ``"raincloud"``,
            ``"boxplot"``, ``"heatmap"``.
            ``"raincloud"`` / ``"boxplot"`` show the per-iteration importance
            distribution; with several models they draw one violin/box per model
            grouped within each feature (the distribution analogue of the grouped
            bar) and fall back to ``"bar"`` when per-iteration data is
            unavailable for the plotted models.
            ``"heatmap"`` renders a feature x model importance matrix.
        return_data : bool, default=False
            If ``True``, return the underlying importance DataFrame.
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame

        Raises
        ------
        ValueError
            If ``style`` is not one of the supported styles.
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl import _plot_importance_impl

        _valid = {"auto", "bar", "forest", "raincloud", "boxplot", "heatmap"}
        if style not in _valid:
            raise ValueError(
                f"style={style!r} is not supported. "
                f"Must be one of: {', '.join(sorted(_valid))}"
            )

        ctx = PlotContext.from_main(self)
        return _plot_importance_impl(
            ctx, models=models, n=n, style=style,
            return_data=return_data, **kwargs,
        )

    # ==================================================================
    # plot_calibration
    # ==================================================================

    def plot_calibration(self: SurvivalLearner, models=None, eval_times=None,
                         n_bins=10, return_data: bool = False,
                         raw: bool = False, bin_strategy: str = "width",
                         **kwargs):
        """
        Plot calibration curve.

        Returns an interactive figure with a dropdown to switch the
        evaluation horizon. With one model, the title names the model (so
        ``models='best'`` shows which model won); with several models, it is
        a "Calibration Comparison". The dropdown lists the available
        horizons, and the selected time is shown in the title. The curve is
        computed at whatever time you ask for, so any ``eval_times`` within
        the observed follow-up is valid.

        Parameters
        ----------
        models : None, str, list[str], or 'best'
            Models to plot. ``'best'`` resolves to the top model by CV
            score (``learner.best_model.name``).
        eval_times : float or list of float, optional
            Evaluation horizon(s) that seed the dropdown. Accepts a single
            time or a list. When omitted, defaults to the same evaluation grid
            used for AUC(t)/Brier(t)/ROC: the configured ``eval_times`` from
            ``evaluate()``/``run()`` (or ``setup()``) when set, otherwise an
            auto grid derived from the data, in both cases unioned with any
            horizons registered via ``add_eval_times``.
        n_bins : int, default=10
            Number of bins.
        bin_strategy : {"width", "quantile"}, default="width"
            How predicted probabilities are grouped into bins. ``"width"``
            uses ``n_bins`` equal-width bins across [0, 1]. ``"quantile"``
            uses ``n_bins`` equal-count bins by predicted-probability
            quantiles, which lets a model with a narrow spread of predictions
            (e.g. GBSA) show a full curve instead of collapsing into a single
            bin. Quantile binning gives an uneven x-axis, and the realized
            point count can still be fewer than ``n_bins`` (from tied
            predictions or bins below the ~5-sample floor); a fully constant
            predictor collapses to one bin with a warning.
        return_data : bool, default=False
            If ``True``, return a tidy ``pd.DataFrame`` spanning every
            horizon instead of a figure.
        raw : bool, default=False
            By default a model with an attached calibrator
            (``train(calibrate=...)``) is plotted on the calibrated ``S(t|x)``
            scale, matching its evaluation metrics. Pass ``raw=True`` to plot
            the uncalibrated predictions instead (a debugging view). Models
            with no calibrator are unaffected.
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame

        Raises
        ------
        ValueError
            If the supplied model does not support
            ``predict_survival_function()``, or if ``bin_strategy`` is not one
            of ``"width"`` or ``"quantile"``.
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl import _plot_calibration_impl

        self._ensure_setup()

        ctx = PlotContext.from_main(self)
        return _plot_calibration_impl(
            ctx, models=models, eval_times=eval_times,
            n_bins=n_bins,
            return_data=return_data, raw=raw,
            bin_strategy=bin_strategy,
            **kwargs,
        )

    # ==================================================================
    # plot_roc
    # ==================================================================

    def plot_roc(self: SurvivalLearner, models=None, t=None,
                 show_auc: bool = True, show_ci="auto",
                 return_data: bool = False, references=None, **kwargs):
        """
        Plot time-dependent ROC curve (IPCW-weighted).

        Parameters
        ----------
        models : None, str, list[str], or 'best'
            Models to include. ``'best'`` resolves to the top model by CV
            score (``learner.best_model.name``).
        t : float, optional
            Evaluation time point.  When omitted and evaluation data
            exists, all evaluation timepoints are shown in an
            interactive dropdown; otherwise the median test event time
            is used.
        show_auc : bool, default=True
            Annotate each curve with its AUC in the legend.
        show_ci : {"auto", True, False}, default="auto"
            Controls the bootstrap confidence-interval ribbon.

            - ``"auto"`` (default) — draw the ribbon when bootstrap
              predictions are available, otherwise the point curve.
              Silent either way.
            - ``True`` — explicitly request the ribbon. When bootstrap
              predictions are missing, draw the point curve and emit a
              ``SuggestionWarning`` naming the ``bootstrap(...)`` flag to
              re-run.
            - ``False`` — never draw the ribbon.
        return_data : bool, default=False
            If ``True``, return the underlying DataFrame.
        references : None, ``"all"``, or list of str
            Include references from ``benchmark()`` results alongside
            models. Reference curves never show CI ribbons because
            benchmark references are not bootstrapped.
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame

        Raises
        ------
        TypeError
            If a ``ModelResult`` is passed instead of a model name; pass
            its ``.name`` or the ``'best'`` sentinel.
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl import _plot_roc_impl

        self._ensure_setup()

        ctx = PlotContext.from_main(self)
        return _plot_roc_impl(
            ctx, models=models, t=t,
            show_auc=show_auc, show_ci=show_ci,
            return_data=return_data, references=references, **kwargs,
        )

    # ==================================================================
    # plot_diagnostics
    # ==================================================================

    def plot_diagnostics(self: SurvivalLearner, kind='schoenfeld', features=None,
                         return_data: bool = False, **kwargs):
        """
        Plot diagnostic residual plots.

        Parameters
        ----------
        kind : {'schoenfeld', 'deviance', 'martingale'}
            Type of residual plot.

            - ``'schoenfeld'`` — scaled Schoenfeld residuals vs. time,
              for assessing the proportional hazards assumption.
            - ``'deviance'`` — deviance residuals vs. linear predictor,
              for outlier detection.
            - ``'martingale'`` — null-model martingale residuals vs.
              continuous covariate, for assessing log-hazard linearity
              (Therneau-Grambsch-Fleming 1990).  Only continuous
              features (numeric, nunique >= 6) are shown.
        features : list of str, optional
            Specific features to plot (Schoenfeld and martingale only).
        return_data : bool, default=False
            If ``True``, return the underlying DataFrame.
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl_secondary import _plot_diagnostics_impl

        self._ensure_setup()
        X_imputed = self._get_imputed_X(self.X_train, raise_on_error=True)
        ctx = PlotContext.from_main(self)
        return _plot_diagnostics_impl(
            ctx, X_imputed=X_imputed, kind=kind, features=features,
            return_data=return_data, **kwargs,
        )

    # ==================================================================
    # plot_metric_time
    # ==================================================================

    def plot_metric_time(
        self: SurvivalLearner,
        models=None,
        metric=None,
        return_data: bool = False,
        references=None,
        raw: bool = False,
        **kwargs,
    ):
        """Plot time-dependent metric curves.

        Parameters
        ----------
        models : None, str, list[str], or 'best'
            Model(s) to include. ``'best'`` resolves to the top model by CV
            score (``learner.best_model.name``).
        metric : str or list of str, optional
            Metric name(s). ``None`` / ``"auc"`` / ``["auc"]`` draw the
            single-panel AUC(t) curve (the default). A single non-AUC name
            (e.g. ``"brier"``) draws that one metric's curve in a single panel.
            A list of 2+ names draws a multi-panel facet, one panel per metric.
            Supported keys: ``"auc"``, ``"brier"``.
        return_data : bool
            If True, return the combined DataFrame.
        references : None, ``"all"``, or list of str
            Include references from ``benchmark()`` results.
        raw : bool, default=False
            By default the **Brier(t)** panel is on the calibrated ``S(t|x)``
            scale for models with an attached calibrator (``train(calibrate=...)``),
            matching the deployed model's evaluation metrics; the AUC(t) panel is
            rank-based and unaffected. Pass ``raw=True`` to score Brier(t) on the
            uncalibrated predictions instead. Models with no calibrator are
            unaffected.
        show_ci : {"auto", True, False}, default="auto"
            Controls the bootstrap confidence-interval ribbon (passed via
            ``**kwargs``).

            - ``"auto"`` (default) — draw the ribbon when the time-resolved
              bootstrap data is available, otherwise the point curve. Silent
              either way.
            - ``True`` — explicitly request the ribbon. When the data is
              missing, draw the point curve and emit a ``SuggestionWarning``
              naming the ``bootstrap(...)`` flag to re-run.
            - ``False`` — never draw the ribbon.
        **kwargs
            Forwarded to :func:`~mlsurv.viz.metrics.plot_metric_time`.

        Returns
        -------
        plotly.Figure or DataFrame
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl import _plot_metric_time_impl

        ctx = PlotContext.from_main(self)
        return _plot_metric_time_impl(
            ctx, models=models, metric=metric,
            return_data=return_data, references=references,
            raw=raw, **kwargs,
        )

    # ==================================================================
    # plot_benchmark_incremental
    # ==================================================================

    def plot_benchmark_incremental(
        self: SurvivalLearner,
        return_data: bool = False,
        **kwargs,
    ):
        """Forest plot of incremental metrics (delta-C, NRI, IDI).

        Parameters
        ----------
        return_data : bool
            If True, return the incremental DataFrame.
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.Figure or DataFrame
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl_secondary import (
            _plot_benchmark_incremental_impl,
        )

        ctx = PlotContext.from_main(self)
        return _plot_benchmark_incremental_impl(
            ctx, return_data=return_data, **kwargs,
        )

    # ==================================================================
    # plot_benchmark_scatter
    # ==================================================================

    def plot_benchmark_scatter(
        self: SurvivalLearner,
        return_data: bool = False,
        **kwargs,
    ):
        """Scatter of model risk score vs. each reference score.

        One subplot per reference; a dropdown switches the active model.
        Complements the Spearman/Pearson matrices rendered by the
        benchmarking report.

        Requires :meth:`benchmark` to have been called with raw scores
        persisted on the resulting :class:`BenchmarkResult`.

        Parameters
        ----------
        return_data : bool
            If True, return a long-form DataFrame with columns
            ``model``, ``reference``, ``model_score``, ``reference_score``
            instead of the figure.
        **kwargs
            Forwarded to
            :func:`~mlsurv.viz.benchmarking_scatter.plot_benchmark_scatter`.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame

        Raises
        ------
        RuntimeError
            If :meth:`benchmark` has not been called, or its result is
            missing the raw model/reference score arrays.
        """
        import pandas as pd

        from mlsurv.viz.benchmarking_scatter import (
            plot_benchmark_scatter as _plot_benchmark_scatter_impl,
        )

        br = self._state.benchmark_result
        if br is None:
            raise RuntimeError(
                "No benchmark result is available. Call learner.benchmark(...) "
                "before plot_benchmark_scatter()."
            )
        if not br.model_risk_scores or not br.reference_arrays:
            raise RuntimeError(
                "BenchmarkResult is missing raw model/reference arrays. "
                "Re-run learner.benchmark(...) with the current mlsurv version."
            )

        if return_data:
            rows = []
            for model, m_arr in br.model_risk_scores.items():
                for ref, r_arr in br.reference_arrays.items():
                    rows.extend(
                        {
                            "model": model,
                            "reference": ref,
                            "model_score": float(m),
                            "reference_score": float(r),
                        }
                        for m, r in zip(m_arr, r_arr)
                    )
            return pd.DataFrame(rows)

        return _plot_benchmark_scatter_impl(
            br.model_risk_scores,
            br.reference_arrays,
            br.correlation,
            **kwargs,
        )

    # ==================================================================
    # plot_coefficients
    # ==================================================================

    def plot_coefficients(self: SurvivalLearner, models=None,
                          n_features=20,
                          return_data: bool = False, **kwargs):
        """Plot feature coefficients from one or more Cox models.

        Parameters
        ----------
        models : None, str, list[str], or 'best'
            Which models to include. ``'best'`` resolves to the top model
            by CV score (``learner.best_model.name``).
        n_features : int, default=20
            Number of top features.
        return_data : bool, default=False
            If ``True``, return the coefficient DataFrame.
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl_secondary import _plot_coefficients_impl

        self._ensure_setup()
        ctx = PlotContext.from_main(self)
        return _plot_coefficients_impl(
            ctx, models=models, n_features=n_features,
            return_data=return_data, **kwargs,
        )

    # ==================================================================
    # plot_coefficient_comparison
    # ==================================================================

    def plot_coefficient_comparison(self: SurvivalLearner, models=None,
                                    n_features=20,
                                    return_data: bool = False, **kwargs):
        """Scatter plot comparing coefficients between Cox models.

        Parameters
        ----------
        models : None, str, list[str], or 'best'
            Cox models to compare. ``'best'`` resolves to the top model by
            CV score (``learner.best_model.name``).
        n_features : int, default=20
            Maximum number of feature name annotations.
        return_data : bool, default=False
            If ``True``, return the comparison DataFrame.
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl_secondary import (
            _plot_coefficient_comparison_impl,
        )

        self._ensure_setup()
        ctx = PlotContext.from_main(self)
        return _plot_coefficient_comparison_impl(
            ctx, models=models, n_features=n_features,
            return_data=return_data, **kwargs,
        )

    # ==================================================================
    # plot_hazard_ratios
    # ==================================================================

    def plot_hazard_ratios(self: SurvivalLearner, kind='univariate',
                           n_features=20, return_data: bool = False,
                           **kwargs):
        """Plot hazard ratios (univariate or multivariate).

        Parameters
        ----------
        kind : {'univariate', 'multivariate'}
        n_features : int, default=20
        return_data : bool, default=False
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl_secondary import _plot_hazard_ratios_impl

        self._ensure_setup()
        ctx = PlotContext.from_main(self)
        return _plot_hazard_ratios_impl(
            ctx, kind=kind, n_features=n_features,
            return_data=return_data, **kwargs,
        )

    # ==================================================================
    # plot_hazard_ratios_comparison
    # ==================================================================

    def plot_hazard_ratios_comparison(self: SurvivalLearner,
                                      n_features=20,
                                      return_data: bool = False, **kwargs):
        """Compare univariate vs multivariate CoxPH hazard ratios.

        Parameters
        ----------
        n_features : int, default=20
        return_data : bool, default=False
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl_secondary import (
            _plot_hazard_ratios_comparison_impl,
        )

        self._ensure_setup()
        ctx = PlotContext.from_main(self)
        return _plot_hazard_ratios_comparison_impl(
            ctx, n_features=n_features,
            return_data=return_data, **kwargs,
        )

    # ==================================================================
    # plot_feature_correlation
    # ==================================================================

    def plot_feature_correlation(self: SurvivalLearner, threshold=0.8,
                                 return_data: bool = False, **kwargs):
        """Plot feature correlation matrix.

        Parameters
        ----------
        threshold : float, default=0.8
        return_data : bool, default=False
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl_secondary import (
            _plot_feature_correlation_impl,
        )

        self._ensure_setup()
        ctx = PlotContext.from_main(self)
        return _plot_feature_correlation_impl(
            ctx, threshold=threshold,
            return_data=return_data, **kwargs,
        )

    # ==================================================================
    # plot_missing
    # ==================================================================

    def plot_missing(self: SurvivalLearner, return_data: bool = False,
                     **kwargs):
        """Plot missing data heatmap.

        Parameters
        ----------
        return_data : bool, default=False
            If True, return a boolean DataFrame (True = missing) instead of
            a figure.
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl_secondary import _plot_missing_impl

        self._ensure_setup()
        ctx = PlotContext.from_main(self)
        return _plot_missing_impl(ctx, return_data=return_data, **kwargs)

    # ==================================================================
    # plot_shap
    # ==================================================================

    def plot_shap(self: SurvivalLearner, models=None, kind='importance',
                  n=20, return_data: bool = False, **kwargs):
        """Plot SHAP values (importance or beeswarm).

        Parameters
        ----------
        models : None, str, list[str], or 'best'
            Models to plot. ``'best'`` resolves to the top model by CV
            score (``learner.best_model.name``).
        kind : {'importance', 'beeswarm'}
        n : int, default=20
        return_data : bool, default=False
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``. The beeswarm
            also takes ``xlim=(xmin, xmax)`` to fix the SHAP-axis range.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl_secondary import _plot_shap_impl

        self._ensure_setup()
        ctx = PlotContext.from_main(self)
        return _plot_shap_impl(
            ctx, models=models, kind=kind, n=n,
            return_data=return_data, **kwargs,
        )

    # ==================================================================
    # plot_interactions
    # ==================================================================

    def plot_interactions(self: SurvivalLearner, model=None,
                          n=10, return_data: bool = False, **kwargs):
        """Plot feature interaction heatmap (H-statistic).

        Parameters
        ----------
        model : None, str, or 'best'
            Model to plot. Uses first model if ``None``; ``'best'`` resolves
            to the top model by CV score (``learner.best_model.name``).
        n : int, default=10
            Maximum number of top pairs to show.
        return_data : bool, default=False
            If True, return the interactions DataFrame instead of a figure.
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl_secondary import _plot_interactions_impl

        self._ensure_setup()
        ctx = PlotContext.from_main(self)
        return _plot_interactions_impl(
            ctx, model=model, n=n,
            return_data=return_data, **kwargs,
        )

    # ==================================================================
    # plot_interaction_pdp
    # ==================================================================

    def plot_interaction_pdp(
        self: SurvivalLearner,
        feature_1: str,
        feature_2: str,
        model=None,
        max_samples: int = 100,
        grid_resolution: int = 25,
        return_data: bool = False,
        xscale: str = 'auto',
        yscale: str = 'auto',
        **kwargs,
    ):
        """Plot 2D partial dependence for a feature pair.

        Visualizes how predicted risk varies as two features change
        jointly, marginalizing over all other features.  The rendering is
        chosen from the pair's types: a contour for numeric × numeric
        (non-parallel contours indicate interaction), one risk curve per
        category level for numeric × categorical (diverging curves indicate
        interaction), and a heatmap for categorical × categorical.  Binary
        and low-cardinality columns are treated as categorical automatically;
        control this with ``setup(categorical_features=...)``.  Uses cached
        PDP surfaces when available (from ``interactions()``), otherwise
        computes on-the-fly.

        Parameters
        ----------
        feature_1 : str
            First feature name (must be a numeric column in the training data).
        feature_2 : str
            Second feature name (must be a numeric column in the training data).
        model : None, str, or 'best'
            Model to plot. Uses the first model if ``None``; ``'best'``
            resolves to the top model by CV score
            (``learner.best_model.name``).
        max_samples : int, default=100
            Maximum samples for PDP grid when computing on-the-fly.
        grid_resolution : int, default=25
            Maximum grid points per feature.
        return_data : bool, default=False
            If True, return a tidy DataFrame instead of a figure.
        xscale : str, default='auto'
            Grid spacing and axis type for the x-axis (*feature_2*): one of
            ``'auto'``, ``'linear'``, ``'log'``, or ``'log1p'``.  ``'auto'``
            (the default, matching the report) picks the spacing from the
            feature's right-skew, so dense regions of right-skewed data stay
            legible.  ``'log'`` spaces the grid on a base-10 log scale and sets
            the Plotly axis ``type='log'`` (zeros floored to the smallest
            positive value); ``'log1p'`` uses a ``log(1 + x)`` axis that keeps
            exact zeros.  Explicit ``'log'``/``'log1p'`` fall back to linear
            (with a warning) for categorical, negative-valued, or (for
            ``'log'``) all-non-positive features.  A cached surface
            (from ``interactions()``) is reused only when its stored spacing
            matches the requested scale; any mismatch forces an on-the-fly
            recompute, so ``'linear'`` always yields a literal linear axis.
        yscale : str, default='auto'
            Same as *xscale* but for the y-axis (*feature_1*).
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame

        Examples
        --------
        >>> learner.plot_interaction_pdp(
        ...     'pnodes', 'progrec', model='rsf', xscale='log')
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl_secondary import (
            _plot_interaction_pdp_impl,
        )

        self._ensure_setup()
        model = self._resolve_best(model)
        model_name = (
            model if isinstance(model, str) or model is None
            else getattr(model, "name", None)
        )
        mr = self._resolve_single_model(model_name)
        ctx = PlotContext.from_main(self)
        return _plot_interaction_pdp_impl(
            ctx, model_result=mr, feature_1=feature_1,
            feature_2=feature_2, max_samples=max_samples,
            grid_resolution=grid_resolution,
            return_data=return_data, xscale=xscale, yscale=yscale,
            **kwargs,
        )

    # ==================================================================
    # plot_calibration_overlay
    # ==================================================================

    def plot_calibration_overlay(self: SurvivalLearner, models=None,
                                 eval_times=None,
                                 return_data: bool = False,
                                 raw: bool = False, **kwargs):
        """Overlay predicted vs. observed survival curves.

        Parameters
        ----------
        models : None, str, list[str], or 'best'
            Models to include. ``'best'`` resolves to the top model by CV
            score (``learner.best_model.name``).
        eval_times : array-like, optional
        return_data : bool, default=False
        raw : bool, default=False
            By default a model with an attached calibrator
            (``train(calibrate=...)``) has its predicted ``S(t|x)`` curve
            overlaid on the calibrated scale, matching the deployed model. Pass
            ``raw=True`` to overlay the uncalibrated curve instead. Models with
            no calibrator are unaffected.
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl_secondary import (
            _plot_calibration_overlay_impl,
        )

        self._ensure_setup()
        ctx = PlotContext.from_main(self)
        return _plot_calibration_overlay_impl(
            ctx, models=models, eval_times=eval_times,
            return_data=return_data, raw=raw,
            **kwargs,
        )

    # ==================================================================
    # plot_feature_distributions
    # ==================================================================

    def plot_feature_distributions(self: SurvivalLearner, features=None,
                                   return_data: bool = False, **kwargs):
        """Plot feature distributions for train and test sets.

        Parameters
        ----------
        features : list[str], optional
            Subset of feature columns to include. None = all features.
        return_data : bool, default=False
            If True, return a long-format DataFrame instead of a figure.
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl_secondary import (
            _plot_feature_distributions_impl,
        )

        self._ensure_setup()
        ctx = PlotContext.from_main(self)
        return _plot_feature_distributions_impl(
            ctx, features=features,
            return_data=return_data, **kwargs,
        )

    # ==================================================================
    # plot_metric_heatmap
    # ==================================================================

    def plot_metric_heatmap(self: SurvivalLearner, models=None,
                            metrics=None, return_data: bool = False,
                            **kwargs):
        """Plot multi-metric heatmap across models.

        Parameters
        ----------
        models : None, str, list[str], or 'best'
            Models to include. ``None`` = all evaluated models; ``'best'``
            resolves to the top model by CV score
            (``learner.best_model.name``).
        metrics : list[str], optional
            Metric columns to include. None = auto-detect.
        return_data : bool, default=False
            If True, return the heatmap DataFrame instead of a figure.
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Notes
        -----
        For a model with an attached calibrator (``train(calibrate=...)``) the
        Brier/IBS cells already reflect the calibrated ``S(t|x)``, because
        evaluate() applies the calibrator inline — there is no plot-level
        calibrate/raw switch here.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl_secondary import _plot_metric_heatmap_impl

        self._ensure_setup()
        ctx = PlotContext.from_main(self)
        return _plot_metric_heatmap_impl(
            ctx, models=models, metrics=metrics,
            return_data=return_data,
            **kwargs,
        )

    # ==================================================================
    # plot_validation_heatmap
    # ==================================================================

    def plot_validation_heatmap(self: SurvivalLearner, models=None,
                                metrics=None, return_data: bool = False,
                                **kwargs):
        """Plot a faceted cross-cohort validation metric heatmap.

        One panel per external-validation cohort (registered via
        ``validate()``), showing a model x metric heatmap with metrics
        z-scored across cohorts (shared colorbar, red = better). Distinct
        from :meth:`plot_metric_heatmap`, which is a single-panel test-set
        heatmap of the main-state models.

        Requires at least 2 validation cohorts, each with at least 2 metrics;
        otherwise raises ``ValueError``.

        Parameters
        ----------
        models : None, str, list[str], or 'best'
            Models to include. ``None`` = every model in each cohort's results
            (no cross-cohort intersection); ``'best'`` resolves to the top
            model by CV score (``learner.best_model.name``), which collapses
            the heatmap to a single model row per cohort.
        metrics : list[str], optional
            Metric columns for the figure (must name at least 2). None =
            auto-detect from the first cohort's metric columns. Applies to the
            figure only; ``return_data`` returns every metric regardless.
        return_data : bool, default=False
            If True, return the tidy cohort x model x metric DataFrame
            instead of a figure. The ``models`` filter is honored;
            ``metrics`` is not (the frame carries all metrics).
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.graph_objects.Figure or pandas.DataFrame
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl_secondary import (
            _plot_validation_heatmap_impl,
        )

        self._ensure_setup()
        ctx = PlotContext.from_main(self)
        return _plot_validation_heatmap_impl(
            ctx, models=models, metrics=metrics,
            return_data=return_data, **kwargs,
        )

    # ==================================================================
    # plot_optuna
    # ==================================================================

    def plot_optuna(self: SurvivalLearner, models=None,
                    return_data: bool = False, **kwargs):
        """Plot Optuna hyperparameter optimization results.

        Shows parameter importance and optimization history side by side
        for each tuned model.

        Parameters
        ----------
        models : None, str, list[str], or 'best'
            Which tuned models to include. ``None`` = all tuned models;
            ``'best'`` resolves to the top model by CV score
            (``learner.best_model.name``).
        return_data : bool, default=False
            If True, return a dict of DataFrames instead of a figure.
        **kwargs
            Styling options forwarded to the underlying plot function,
            e.g. ``width``, ``height``, ``title``, ``palette``.

        Returns
        -------
        plotly.graph_objects.Figure or dict[str, pandas.DataFrame]
            If ``return_data=True``, returns
            ``{"tuning_history": DataFrame, "param_importance": DataFrame}``.
        """
        from mlsurv.viz._context import PlotContext
        from mlsurv.viz._plot_impl_secondary import _plot_optuna_impl

        self._ensure_setup()
        ctx = PlotContext.from_main(self)
        return _plot_optuna_impl(
            ctx, models=models,
            return_data=return_data, **kwargs,
        )

    # ``plot_subpopulation_survival`` was removed; use
    # ``learner.subpopulation(group).plot_survival()`` instead.
