"""InteractionsMixin — feature importance, interactions, and statistical helpers.

Methods: ``importance`` (public front door, dispatching to the private
``_permutation_importance`` and ``_shap`` backends), ``interactions``,
``importance_test``, ``ph_test``, ``linearity_test``, ``rmst_test``,
``logrank_test`` plus the ``_get_test_risk_groups`` and
``_importance_test_bootstrap`` helpers. Each test stores its result table on
the registered ``ModelResult`` (``result.pimp_importance_test`` /
``bootstrap_importance_test`` / ``rmst_test`` / ``logrank_test``), so it
dies with the model on overwrite.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from .._warnings import InternalWarning, MethodologicalWarning, SuggestionWarning
from ..logging_utils import get_logger

logger = get_logger("learner")

if TYPE_CHECKING:
    import pandas as pd
    from mlsurv.learner._core import SurvivalLearner


# Failure-sentinel schemas for the risk-group tests (0-row tables, stored so
# the absent-vs-empty distinction is preserved). ``model_type`` is prepended
# at build time so failed and computed tables stack uniformly.
_RMST_EMPTY_COLUMNS = [
    'group_1', 'group_2', 'diff', 'diff_se',
    'ci_lower', 'ci_upper', 'p_value', 'tau',
]
_LOGRANK_EMPTY_COLUMNS = ['comparison', 'statistic', 'p_value']

# Schema of the permutation-importance summary table, used as the 0-row
# sentinel when nothing could be computed.
_PERM_IMPORTANCE_COLUMNS = [
    'model_type', 'feature', 'coefficient', 'perm_mean', 'perm_std',
]


def _make_group_labeller(labels):
    """Return a function mapping an integer group id to its label string.

    ``labels[i]`` describes group id ``i``. A group id outside
    ``range(len(labels))`` is returned unchanged (the explicit bound check
    also blocks Python's negative-index wraparound from silently returning a
    plausible-but-wrong label).
    """
    def _label(group_id):
        idx = int(group_id)
        return labels[idx] if 0 <= idx < len(labels) else group_id

    return _label


class InteractionsMixin:
    """Mixin for feature interactions and statistical hypothesis tests."""

    def importance_test(
        self: SurvivalLearner,
        models=None,
        method: str = "pimp",
        n_permutations: int = 100,
        n_importance_repeats: int = 10,
        p_adjust: str = "bh",
        random_state=None,
        show_progress: bool = True,
        verbose: bool = True,
        n_jobs=None,
    ) -> "pd.DataFrame":
        """Compute feature importance p-values.

        Parameters
        ----------
        models : str, ModelResult, or None
            Model(s) to test: a trained model name, a ModelResult, or None
            (best trained model). Raw pipelines/estimators are not accepted;
            use ``compute_pimp()`` for models not trained through the learner.
        method : str, default="pimp"
            Method for computing p-values:

            - ``"pimp"`` (default): Permutation Importance with Permuted
              response (Altmann et al. 2010). Trains null models on permuted
              response data. Tests H₀: feature has no association with
              response. Gold standard but computationally expensive.
            - ``"bootstrap"``: Uses existing bootstrap importance data.
              Requires prior ``learner.bootstrap(include_importance=True)``.
              Tests H₀: feature importance ≤ 0. Fast but weaker null.

        n_permutations : int, default=100
            Number of response permutations (PIMP only).
        n_importance_repeats : int, default=10
            Permutation importance repeats per model (PIMP only).
        p_adjust : str, default="bh"
            Multiplicity correction: "bh" (Benjamini-Hochberg FDR),
            "holm" (FWER), "bonferroni", or "none".
        random_state : int, optional
            Random seed. Falls back to learner's random_state.
        show_progress : bool, default=True
            Print progress updates (PIMP only).
        verbose : bool, default=True
            Print results summary.
        n_jobs : int, optional
            Parallel jobs (PIMP only). Falls back to learner's n_jobs.

        Returns
        -------
        DataFrame
            For PIMP: feature, observed_importance, null_mean, null_std,
            p_value, p_adjusted.
            For bootstrap: feature, mean, std, ci_lower, ci_upper,
            p_value, p_adjusted.

            The result is also stored per model, keyed by model name, in
            ``pimp_importance_test_results`` (PIMP) or
            ``bootstrap_importance_test_results`` (bootstrap). Re-running a
            method on a model replaces only that model's entry. The
            parameters that shaped the result are recorded in
            ``result.attrs``.

        Method Comparison
        -----------------
        +-----------------+------------------------------+------------------------+
        | Aspect          | PIMP (default)               | Bootstrap              |
        +-----------------+------------------------------+------------------------+
        | Null hypothesis | Feature-response independence | Importance <= 0        |
        | Computation     | s x (refit + importance)      | Uses existing bootstrap |
        | Prerequisite    | None (trains null models)    | bootstrap(importance=True)|
        | Reference       | Altmann et al. (2010)        | Empirical resampling   |
        | Recommended for | Definitive feature selection | Quick screening        |
        +-----------------+------------------------------+------------------------+

        Raises
        ------
        ValueError
            If ``method`` is not ``'pimp'`` or ``'bootstrap'``, or if
            ``models`` is a raw pipeline/estimator not trained through the
            learner (train it through the learner, or use ``compute_pimp()``).

        Notes
        -----
        PIMP always uses the learner's stored train/test splits. To run
        PIMP on external validation data, use ``compute_pimp()`` directly.

        See Also
        --------
        BootstrapResult.importance_summary : Bootstrap-based importance p-values.
        compute_pimp : Standalone PIMP function for custom train/test data.
        """
        self._ensure_setup()

        if method not in ("pimp", "bootstrap"):
            raise ValueError(
                f"method must be 'pimp' or 'bootstrap', got {method!r}"
            )

        if method == "bootstrap":
            return self._importance_test_bootstrap(models, p_adjust, verbose)

        # PIMP method
        random_state = random_state if random_state is not None else self.random_state
        n_jobs = n_jobs if n_jobs is not None else self.n_jobs

        resolved, _, orig = self._resolve_model_input(models)
        if isinstance(resolved, dict):
            # Use first model for PIMP (single-model method)
            pimp_model_name = next(iter(resolved))
            pipeline = resolved[pimp_model_name]
            warnings.warn(
                f"PIMP is a single-model method. Using '{pimp_model_name}'. "
                f"Pass a specific model to analyze a different one. "
                f"Note: PIMP requires n_permutations full model refits — "
                f"for quick exploration, consider reducing n_permutations "
                f"(e.g., n_permutations=20) or using method='bootstrap'.",
                SuggestionWarning, stacklevel=2,
            )
        else:
            if orig is None:
                # importance_test results live in per-model state keyed by
                # registered model names — an unregistered object has no
                # lifecycle (never invalidated on retrain, never read by
                # reports), so reject it.
                raise ValueError(
                    "importance_test requires a model trained through the "
                    f"learner; got an unregistered {type(resolved).__name__}. "
                    "Train it through the learner (learner.train(...)) so "
                    "results are stored and invalidated with the model, or "
                    "call mlsurv.evaluation.compute_pimp() directly for a "
                    "one-off analysis."
                )
            pipeline = resolved
            pimp_model_name = orig.name

        from ..evaluation.importance import compute_pimp
        result = compute_pimp(
            pipeline,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            n_permutations=n_permutations,
            n_importance_repeats=n_importance_repeats,
            p_adjust=p_adjust,
            random_state=random_state,
            n_jobs=n_jobs,
            show_progress=show_progress,
            categorical_groups=self._state.categorical_groups,
        )

        # Stamp every result-affecting parameter so the stored table is
        # self-describing. random_state is the resolved value.
        result.attrs['n_permutations'] = n_permutations
        result.attrs['n_importance_repeats'] = n_importance_repeats
        result.attrs['random_state'] = random_state
        result.attrs['p_adjust'] = p_adjust
        result.attrs['model_name'] = pimp_model_name
        # A pipeline that is not the registered model but resolves to its
        # name (e.g. a modified copy of a wrapper that carries name='coxph')
        # would silently replace the registered model's entry and be
        # misattributed in reports — surface that.
        registered = self._state.models.get(pimp_model_name)
        if registered is None:
            # The non-dict branch already rejected unregistered models
            # above; the dict branch keys off the caller's dict, which need
            # not name a registered model. Reject it the same way rather than
            # KeyError on the store below.
            raise ValueError(
                "importance_test requires a model trained through the "
                f"learner; '{pimp_model_name}' is not a registered model. "
                "Train it through the learner (learner.train(...)), or call "
                "mlsurv.evaluation.compute_pimp() directly for a one-off "
                "analysis."
            )
        if registered.model is not pipeline:
            warnings.warn(
                f"importance_test result stored under '{pimp_model_name}' "
                "replaces the entry for the registered model of that name, "
                "but was computed on a different pipeline object. Reports "
                "and exports will attribute these p-values to the "
                "registered model.",
                SuggestionWarning, stacklevel=2,
            )
        # The table rides on the registered ModelResult so it dies with the
        # model on overwrite.
        self._state.models[pimp_model_name].pimp_importance_test = result

        if verbose:
            print("\nImportance P-Values (PIMP method)")
            print("=" * 50)
            print(result.to_string(index=False))
        return result

    def _importance_test_bootstrap(self: SurvivalLearner, model, p_adjust, verbose):
        """Compute importance p-values using bootstrap method."""
        boot_result = self.bootstrap_result
        if boot_result is None or boot_result.importance is None or boot_result.importance.empty:
            raise ValueError(
                "No bootstrap importance data available. Run "
                "learner.bootstrap(include_importance=True) first, "
                "or use method='pimp' for PIMP-based p-values."
            )
        model_type = None
        if model is not None:
            resolved, _, orig = self._resolve_model_input(model)
            if isinstance(resolved, dict):
                first_name = next(iter(resolved))
                model_type = first_name
                if len(resolved) > 1:
                    warnings.warn(
                        f"Multiple models resolved. Using '{first_name}' "
                        f"for bootstrap importance summary.",
                        InternalWarning, stacklevel=3,
                    )
            elif orig is not None:
                model_type = orig.name
        result = boot_result.importance_summary(
            model_type=model_type, p_adjust=p_adjust,
        )
        if model_type is None:
            # importance_summary already raised on ambiguity (multiple
            # model types with model_type=None), so exactly one remains.
            model_type = str(boot_result.importance['model_type'].unique()[0])
        result.attrs['p_adjust'] = p_adjust
        result.attrs['model_name'] = model_type
        # Rides on the registered ModelResult (model_type comes from the
        # bootstrap importance table, whose model types are all registered).
        self._state.models[model_type].bootstrap_importance_test = result
        if verbose:
            print("\nImportance P-Values (bootstrap method)")
            print("=" * 50)
            print(result.to_string(index=False))
        return result

    def ph_test(
        self: SurvivalLearner,
        time_transform: str = 'rank',
        verbose: bool = True,
    ) -> "pd.DataFrame":
        """Run proportional hazards test and persist result on state.

        Parameters
        ----------
        time_transform : str, default='rank'
            Time transform for the test: 'rank', 'km', 'identity', 'log'.
        verbose : bool, default=True
            If True, print summary of results.

        Returns
        -------
        DataFrame
            Per-feature rows + GLOBAL row with columns: feature,
            test_statistic, p_value, p_adjusted, significant.
            Empty DataFrame on computation failure (with warning).
            ``df.attrs`` records ``time_transform`` and
            ``source='method'``.

        Raises
        ------
        MemoryError
            Re-raised if the computation exhausts available memory; resource
            errors are never swallowed by the broad failure handler.
        RecursionError
            Re-raised if the computation exceeds the recursion limit; resource
            errors are never swallowed by the broad failure handler.
        """
        import pandas as pd

        self._ensure_setup()

        from ..evaluation.metrics import ph_test as _ph_test

        caught_exception = False
        try:
            X_imp = self._get_imputed_X(self.X_train, raise_on_error=True)
            df = _ph_test(X_imp, self.y_train, time_transform=time_transform)
        except (MemoryError, RecursionError):
            raise
        except (ValueError, KeyError, TypeError, RuntimeError,
                ImportError, np.linalg.LinAlgError) as exc:
            logger.warning(
                "ph_test() failed (%s: %s)", type(exc).__name__, exc,
            )
            warnings.warn(
                f"Proportional hazards test failed ({type(exc).__name__}: "
                f"{exc}). Returning empty result.",
                MethodologicalWarning,
                stacklevel=2,
            )
            df = pd.DataFrame(columns=[
                'feature', 'test_statistic', 'p_value',
                'p_adjusted', 'significant',
            ])
            caught_exception = True

        # Stamp the result-affecting params so the stored DataFrame is
        # self-describing — matches importance_test's attrs stamping.
        df.attrs['time_transform'] = time_transform
        df.attrs['source'] = 'method'
        self._state.ph_test_result = df

        if df.empty and not caught_exception:
            warnings.warn(
                "Proportional hazards test returned no results. This may "
                "indicate that the Cox model could not be fit on the "
                "training data (e.g., constant features, singular matrix).",
                MethodologicalWarning,
                stacklevel=2,
            )

        if verbose and not df.empty:
            print("\nProportional Hazards Test")
            print("=" * 50)
            print(df.to_string(index=False))

        return df

    def linearity_test(
        self: SurvivalLearner,
        n_knots: int = 4,
        verbose: bool = True,
    ) -> "pd.DataFrame":
        """Run log-hazard linearity test and persist result on state.

        Implements Harrell's single-fit Wald test on a stacked
        restricted-cubic-spline design — the same approach as R's
        ``rms::anova.cph``. See
        :func:`mlsurv.evaluation.metrics.linearity_test` for details on
        the statistical method and references.

        Parameters
        ----------
        n_knots : int, default=4
            Number of RCS knots (Harrell recommends 4 for most datasets;
            supported: 3, 4, 5, 6, 7).
        verbose : bool, default=True
            If True, print summary of results.

        Returns
        -------
        DataFrame
            Per-continuous-feature rows + GLOBAL row with columns:
            feature, test_statistic, p_value, p_adjusted, significant.
            Empty DataFrame on computation failure (with warning).
            ``df.attrs`` records ``n_knots`` and ``source='method'``.

        Raises
        ------
        MemoryError
            Re-raised if the computation exhausts available memory; resource
            errors are never swallowed by the broad failure handler.
        RecursionError
            Re-raised if the computation exceeds the recursion limit; resource
            errors are never swallowed by the broad failure handler.
        """
        import pandas as pd

        self._ensure_setup()

        from ..evaluation.metrics import linearity_test as _linearity_test

        try:
            X_imp = self._get_imputed_X(self.X_train, raise_on_error=True)
            df = _linearity_test(X_imp, self.y_train, n_knots=n_knots)
        except (MemoryError, RecursionError):
            raise
        except (ValueError, KeyError, TypeError, RuntimeError,
                ImportError, np.linalg.LinAlgError) as exc:
            logger.warning(
                "linearity_test() failed (%s: %s)", type(exc).__name__, exc,
            )
            warnings.warn(
                f"Linearity test failed ({type(exc).__name__}: "
                f"{exc}). Returning empty result.",
                MethodologicalWarning,
                stacklevel=2,
            )
            df = pd.DataFrame(columns=[
                'feature', 'test_statistic', 'p_value',
                'p_adjusted', 'significant',
            ])

        # Stamp the result-affecting params so the stored DataFrame is
        # self-describing — matches importance_test's attrs stamping.
        df.attrs['n_knots'] = n_knots
        df.attrs['source'] = 'method'
        self._state.linearity_test_result = df

        # Note: when ``df`` is empty, the inner ``linearity_test`` function
        # has already emitted a tiered warning (Methodological for genuine
        # data problems, Suggestion for benign skips like "no continuous
        # features"). We deliberately do NOT re-warn here — that would
        # double-classify a benign skip as a methodological issue.

        if verbose and not df.empty:
            print("\nLog-Hazard Linearity Test")
            print("=" * 50)
            print(df.to_string(index=False))

        return df

    def _get_test_risk_groups(
        self: SurvivalLearner,
        model_result,
        n_groups=None,
        breaks=None,
        break_type: str = "quantile",
        correction: str = "bonferroni",
        n_permutations: int = 1000,
        random_state=None,
    ) -> "tuple[pd.DataFrame, list]":
        """Create risk groups from one model's test-set risk scores.

        Reuses the viz grouping helpers (``_create_risk_groups``, and
        ``_resolve_maxstat`` which is a no-op unless ``break_type="maxstat"``)
        so the test path and ``plot_survival`` form groups identically.

        Parameters
        ----------
        model_result : ModelResult
            The registered model to use (resolved once by the public
            ``rmst_test``/``logrank_test`` boundary).
        n_groups : int, optional
            Number of equal-frequency risk groups. Defaults to 2 (median
            split) when ``breaks`` is None; ignored when ``breaks`` is given.
        breaks : list, optional
            Explicit group boundaries. ``break_type="quantile"``: fractions
            in (0, 1); ``break_type="score"``: raw risk-score cut-points.
        break_type : {"quantile", "score", "maxstat"}, default="quantile"
            Interpretation of ``breaks``. ``"maxstat"`` finds cutpoints that
            maximally separate the curves (ignores ``breaks``).
        correction : {"bonferroni", "lausen-schumacher", "permutation"}, \
default="bonferroni"
            Multiple-testing correction for ``break_type="maxstat"``.
        n_permutations : int, default=1000
            Permutations when ``correction="permutation"``.
        random_state : int, optional
            Seed for permutation reproducibility.

        Returns
        -------
        tuple of (DataFrame, list)
            DataFrame with columns time, event, risk_score, risk_group
            (integer group ids), and the ordered list of human-readable
            group labels (label ``i`` describes group id ``i``).

        Raises
        ------
        ValueError
            If ``evaluate()`` has not been called, the inputs are invalid,
            or risk groups cannot be created (fewer than 2 unique scores).
        """
        from ..viz._survival_helpers import (
            _MAXSTAT_LABELS, _create_risk_groups, _validate_breaks,
        )
        from ..viz._survival_maxstat import _resolve_maxstat

        _validate_breaks(breaks, break_type)

        eval_result = self.evaluation_result
        if eval_result is None:
            raise ValueError(
                "No evaluation result. Call evaluate() before "
                "rmst_test() or logrank_test()."
            )

        model_name = model_result.name
        test_set = eval_result.test_set
        test_data = test_set[test_set['model_type'] == model_name].copy()

        if test_data.empty:
            raise ValueError(
                f"No test-set data found for model '{model_name}'. "
                f"Call evaluate('{model_name}') first."
            )

        n_unique = test_data['risk_score'].nunique()
        if n_unique < 2:
            raise ValueError(
                f"Cannot create risk groups: only {n_unique} unique "
                "risk score(s) in test set."
            )

        # Resolve the equal-frequency default; maxstat keeps the same 2.
        resolved_n = 2 if n_groups is None else n_groups
        was_maxstat = break_type == "maxstat"
        # maxstat ignores breaks and targets resolved_n, so the requested count
        # comes from n_groups there, not from the (discarded) breaks list.
        if was_maxstat:
            requested_groups = resolved_n
        elif breaks is not None:
            requested_groups = len(breaks) + 1
        else:
            requested_groups = resolved_n

        # maxstat finds its own cutpoints and rewrites break_type -> "score".
        resolved_n, breaks, break_type, _ = _resolve_maxstat(
            test_data, 'time', 'event', 'risk_score',
            resolved_n, breaks, break_type, correction,
            n_permutations=n_permutations, random_state=random_state,
        )
        test_data, n_actual, labels = _create_risk_groups(
            test_data, 'risk_score', resolved_n,
            breaks=breaks, break_type=break_type,
        )
        if was_maxstat:
            labels = list(_MAXSTAT_LABELS.get(n_actual, labels))

        # The viz helpers warn about collapsed groups only via InternalWarning
        # (suppressed by default). On the test path a collapse changes which
        # statistic is computed, so surface it with a user-visible tier.
        if n_actual < requested_groups:
            warnings.warn(
                f"Requested {requested_groups} risk groups but only "
                f"{n_actual} could be formed from the test-set risk scores; "
                f"the statistical test is computed on {n_actual} groups.",
                MethodologicalWarning, stacklevel=3,
            )

        rg = test_data[['time', 'event', 'risk_score', 'risk_group']]
        return rg, labels

    def _stamp_and_store_risk_test(
        self: SurvivalLearner, model_result, df, attr_name, params,
    ) -> "pd.DataFrame":
        """Stamp provenance + result-affecting params on ``df.attrs``; store on the model.

        Empty/failed tables are stored too, so the absent-vs-empty
        distinction is preserved. Rides on the registered ``ModelResult``,
        so the table dies with the model on overwrite.
        """
        df.attrs['model_name'] = model_result.name
        for key, value in params.items():
            df.attrs[key] = value
        setattr(model_result, attr_name, df)
        return df

    def _run_risk_test_over_models(
        self: SurvivalLearner, resolved, single_form, attr_params,
        run_single, attr_name, empty_columns, method_label, print_title,
        verbose,
    ) -> "pd.DataFrame":
        """Drive a risk-group test (``rmst_test``/``logrank_test``) over a model selection.

        Shared scaffolding for the two parallel public methods. ``run_single``
        is the per-test worker (``_rmst_test_single``/``_logrank_test_single``)
        already bound to its parameters; it takes one ``ModelResult`` and
        returns its stored table. Risk-group formation raises ``ValueError``:
        a single-model selection surfaces it directly, a multi-model
        selection skips that model with a warning and stores an empty table
        so the absent-vs-empty distinction is preserved. The remaining
        arguments are per-test labels/columns, not behaviour flags.
        """
        import pandas as pd

        frames = []
        for mr in resolved:
            try:
                df = run_single(mr)
            except ValueError as exc:
                if single_form:
                    raise
                warnings.warn(
                    f"{method_label}() skipped model '{mr.name}': {exc}",
                    MethodologicalWarning, stacklevel=3,
                )
                empty = pd.DataFrame(columns=empty_columns)
                empty.insert(0, 'model_type', mr.name)
                df = self._stamp_and_store_risk_test(
                    mr, empty, attr_name, attr_params,
                )
            frames.append(df)

        if single_form:
            combined = frames[0]
        else:
            # Empty (failed/skipped) tables contribute no rows; dropping
            # them from the concat also avoids pandas' all-NA-entry
            # concat deprecation.
            non_empty = [f for f in frames if not f.empty]
            if not non_empty:
                warnings.warn(
                    f"{method_label}() produced no results for any of the "
                    f"{len(resolved)} selected models; returning an empty "
                    "table. See the preceding warnings for the causes.",
                    MethodologicalWarning, stacklevel=3,
                )
            combined = (
                pd.concat(non_empty, ignore_index=True) if non_empty
                else pd.DataFrame(columns=frames[0].columns)
            )
            for key, value in attr_params.items():
                combined.attrs[key] = value

        if verbose and not combined.empty:
            print(f"\n{print_title}")
            print("=" * 50)
            print(combined.to_string(index=False))

        return combined

    def _rmst_test_single(
        self: SurvivalLearner,
        model_result,
        *,
        n_groups=None,
        tau=None,
        p_adjust: str = 'holm',
        breaks=None,
        break_type: str = "quantile",
        correction: str = "bonferroni",
        n_permutations: int = 1000,
        random_state=None,
        attr_params=None,
    ) -> "pd.DataFrame":
        """Run the RMST comparison for one resolved model.

        Forms risk groups from the model's test-set risk scores, computes
        the two-group or pairwise comparison, prepends the ``model_type``
        column, stamps ``df.attrs``, and stores the table on
        ``model_result.rmst_test``. Computation failures degrade to an
        empty-schema table with a ``MethodologicalWarning``; risk-group
        formation errors propagate as ``ValueError`` for the public
        boundary to handle per selection arity.
        """
        import pandas as pd
        from ..evaluation.metrics import (
            rmst_test as _rmst_test,
            rmst_test_pairwise as _rmst_test_pairwise,
        )

        rg, labels = self._get_test_risk_groups(
            model_result, n_groups, breaks, break_type, correction,
            n_permutations, random_state,
        )
        _label = _make_group_labeller(labels)

        try:
            if rg['risk_group'].nunique() == 2:
                result = _rmst_test(
                    rg['time'].values, rg['event'].values,
                    rg['risk_group'].values, tau=tau,
                )
                groups = sorted(result.rmst.keys())
                df = pd.DataFrame([{
                    'group_1': _label(groups[0]), 'group_2': _label(groups[1]),
                    'rmst_0': result.rmst[groups[0]],
                    'rmst_1': result.rmst[groups[1]],
                    'diff': result.diff,
                    'diff_se': result.diff_se,
                    'ci_lower': result.ci_lower,
                    'ci_upper': result.ci_upper,
                    'z_stat': result.z_stat,
                    'p_value': result.p_value,
                    'tau': result.tau,
                }])
            else:
                result = _rmst_test_pairwise(
                    rg['time'].values, rg['event'].values,
                    rg['risk_group'].values, tau=tau, p_adjust=p_adjust,
                )
                rows = []
                for pw in result.pairwise:
                    rows.append({
                        'group_1': _label(pw['groups'][0]),
                        'group_2': _label(pw['groups'][1]),
                        'diff': pw['diff'],
                        'diff_se': pw['diff_se'],
                        'ci_lower': pw['ci_lower'],
                        'ci_upper': pw['ci_upper'],
                        'p_value': pw['raw_p'],
                        'p_adjusted': pw['adjusted_p'],
                        'tau': result.tau,
                    })
                df = pd.DataFrame(rows)
        except (MemoryError, RecursionError):
            raise
        except (ValueError, KeyError, TypeError, RuntimeError,
                np.linalg.LinAlgError) as exc:
            logger.warning(
                "rmst_test() failed for model '%s' (%s: %s)",
                model_result.name, type(exc).__name__, exc,
            )
            warnings.warn(
                f"RMST comparison failed for model '{model_result.name}' "
                f"({type(exc).__name__}: {exc}). Returning empty result.",
                MethodologicalWarning,
                stacklevel=3,
            )
            df = pd.DataFrame(columns=_RMST_EMPTY_COLUMNS)

        df.insert(0, 'model_type', model_result.name)
        return self._stamp_and_store_risk_test(
            model_result, df, 'rmst_test', attr_params or {},
        )

    def rmst_test(
        self: SurvivalLearner,
        models=None,
        n_groups=None,
        tau=None,
        p_adjust: str = 'holm',
        verbose: bool = True,
        breaks=None,
        break_type: str = "quantile",
        correction: str = "bonferroni",
        n_permutations: int = 1000,
        random_state=None,
    ) -> "pd.DataFrame":
        """Compare RMST across risk groups and persist result on state.

        Parameters
        ----------
        models : str, ModelResult, list, dict, or None
            Model selection for the risk scores. ``None`` (default) runs the
            test for every trained model; a list/dict selects a subset. The
            result stacks one table per model (``model_type`` column). With
            a multi-model selection, a model whose risk groups cannot be
            formed is skipped with a warning and stored as an empty table.
            A ``ModelResult`` is resolved by name to the currently
            registered model; a stale reference (e.g. held from before
            ``tune()``) triggers a ``SuggestionWarning`` and the
            registered model is used.
        n_groups : int, optional
            Number of equal-frequency risk groups. Defaults to 2 (median
            split). Ignored when ``breaks`` is given.
        tau : float, optional
            Time horizon for RMST. Auto-computed if None.
        p_adjust : str, default='holm'
            Multiplicity correction for 3+ groups.
        verbose : bool, default=True
            If True, print summary.
        breaks : list, optional
            Custom group boundaries (overrides ``n_groups``). Mirrors
            ``plot_survival``/``plot_km``.

            - ``break_type="quantile"`` (default): fractions in (0, 1).
              E.g. ``[0.25, 0.75]`` -> three groups at 0-25 / 25-75 /
              75-100 percentiles of the risk score.
            - ``break_type="score"``: raw risk-score cut-points.
              E.g. ``[3, 20]`` -> three groups ``< 3``, ``3-20``, ``>= 20``.
        break_type : {"quantile", "score", "maxstat"}, default="quantile"
            Interpretation of ``breaks``. ``"maxstat"`` finds the cutpoint(s)
            that maximally separate the curves via the log-rank test
            (ignores ``breaks``).
        correction : {"bonferroni", "lausen-schumacher", "permutation"}, \
default="bonferroni"
            Multiple-testing correction for ``break_type="maxstat"``.
        n_permutations : int, default=1000
            Permutations when ``correction="permutation"``.
        random_state : int, optional
            Seed for permutation reproducibility.

        Returns
        -------
        DataFrame
            RMST comparison results, one block of rows per selected model
            (``model_type`` is the first column).  The ``group_1``/``group_2``
            columns hold human-readable group labels (e.g. ``"< 25%"``,
            ``"3-20"``).  A model whose comparison fails contributes no rows
            (empty result, with warning).

        Raises
        ------
        ValueError
            If ``evaluate()`` has not been called, risk groups cannot be
            formed for a single-model selection, the selection names an
            unknown model, or the list/dict selection is empty.
        TypeError
            If ``models`` is a raw model/pipeline (or other unsupported
            type) — pass a trained model name instead.
        MemoryError
            Re-raised if the computation exhausts available memory; resource
            errors are never swallowed by the broad failure handler.
        RecursionError
            Re-raised if the computation exceeds the recursion limit; resource
            errors are never swallowed by the broad failure handler.
        """
        self._ensure_setup()

        from ..viz._survival_helpers import _warn_breaks_precedence

        _warn_breaks_precedence(n_groups, breaks, break_type)

        # Precondition: evaluate() must have been called — checked once
        # here so multi-model selections fail fast instead of per model.
        if self.evaluation_result is None:
            raise ValueError(
                "No evaluation result. Call evaluate() before rmst_test()."
            )

        resolved, single_form = self._resolve_models_multi(models)
        # Result-affecting parameters stamped on every stored/returned
        # table so it is self-describing.
        attr_params = {
            'n_groups': n_groups, 'tau': tau, 'p_adjust': p_adjust,
            'break_type': break_type, 'breaks': breaks,
            'correction': correction,
        }

        return self._run_risk_test_over_models(
            resolved, single_form, attr_params,
            run_single=lambda mr: self._rmst_test_single(
                mr, n_groups=n_groups, tau=tau, p_adjust=p_adjust,
                breaks=breaks, break_type=break_type,
                correction=correction, n_permutations=n_permutations,
                random_state=random_state, attr_params=attr_params,
            ),
            attr_name='rmst_test',
            empty_columns=_RMST_EMPTY_COLUMNS,
            method_label='rmst_test',
            print_title='RMST Comparison',
            verbose=verbose,
        )

    def _logrank_test_single(
        self: SurvivalLearner,
        model_result,
        *,
        n_groups=None,
        p_adjust: str = 'holm',
        breaks=None,
        break_type: str = "quantile",
        correction: str = "bonferroni",
        n_permutations: int = 1000,
        random_state=None,
        attr_params=None,
    ) -> "pd.DataFrame":
        """Run the log-rank test for one resolved model.

        Forms risk groups from the model's test-set risk scores, computes
        the omnibus (and pairwise) statistics, prepends the ``model_type``
        column, stamps ``df.attrs``, and stores the table on
        ``model_result.logrank_test``. Computation failures degrade to an
        empty-schema table with a ``MethodologicalWarning``; risk-group
        formation errors propagate as ``ValueError`` for the public
        boundary to handle per selection arity.
        """
        import pandas as pd
        from ..viz.survival import _compute_logrank

        rg, labels = self._get_test_risk_groups(
            model_result, n_groups, breaks, break_type, correction,
            n_permutations, random_state,
        )
        _label = _make_group_labeller(labels)

        try:
            result = _compute_logrank(rg, 'time', 'event', 'risk_group', p_adjust)
            if result is None:
                raise ValueError(
                    "Log-rank test failed. Check that risk groups have "
                    "sufficient events."
                )

            rows = [{
                'comparison': 'omnibus',
                'statistic': result['omnibus_statistic'],
                'p_value': result['omnibus_p'],
                'n_groups': result['n_groups'],
            }]

            if result['pairwise'] is not None:
                for pw in result['pairwise']:
                    rows.append({
                        'comparison':
                            f"{_label(pw['group_1'])} vs {_label(pw['group_2'])}",
                        'statistic': pw.get('statistic'),
                        'p_value': pw.get('p_raw'),
                        'p_adjusted': pw.get('p_adjusted'),
                    })

            df = pd.DataFrame(rows)
        except (MemoryError, RecursionError):
            raise
        except (ValueError, KeyError, TypeError, RuntimeError,
                np.linalg.LinAlgError) as exc:
            logger.warning(
                "logrank_test() failed for model '%s' (%s: %s)",
                model_result.name, type(exc).__name__, exc,
            )
            warnings.warn(
                f"Log-rank test failed for model '{model_result.name}' "
                f"({type(exc).__name__}: {exc}). Returning empty result.",
                MethodologicalWarning,
                stacklevel=3,
            )
            df = pd.DataFrame(columns=_LOGRANK_EMPTY_COLUMNS)

        df.insert(0, 'model_type', model_result.name)
        return self._stamp_and_store_risk_test(
            model_result, df, 'logrank_test', attr_params or {},
        )

    def logrank_test(
        self: SurvivalLearner,
        models=None,
        n_groups=None,
        p_adjust: str = 'holm',
        verbose: bool = True,
        breaks=None,
        break_type: str = "quantile",
        correction: str = "bonferroni",
        n_permutations: int = 1000,
        random_state=None,
    ) -> "pd.DataFrame":
        """Run log-rank test across risk groups and persist result on state.

        Parameters
        ----------
        models : str, ModelResult, list, dict, or None
            Model selection for the risk scores. ``None`` (default) runs the
            test for every trained model; a list/dict selects a subset. The
            result stacks one table per model (``model_type`` column). With
            a multi-model selection, a model whose risk groups cannot be
            formed is skipped with a warning and stored as an empty table.
            A ``ModelResult`` is resolved by name to the currently
            registered model; a stale reference (e.g. held from before
            ``tune()``) triggers a ``SuggestionWarning`` and the
            registered model is used.
        n_groups : int, optional
            Number of equal-frequency risk groups. Defaults to 2 (median
            split). Ignored when ``breaks`` is given.
        p_adjust : str, default='holm'
            Multiplicity correction for 3+ groups.
        verbose : bool, default=True
            If True, print summary.
        breaks : list, optional
            Custom group boundaries (overrides ``n_groups``). Mirrors
            ``plot_survival``/``plot_km``.

            - ``break_type="quantile"`` (default): fractions in (0, 1).
              E.g. ``[0.25, 0.75]`` -> three groups at 0-25 / 25-75 /
              75-100 percentiles of the risk score.
            - ``break_type="score"``: raw risk-score cut-points.
              E.g. ``[3, 20]`` -> three groups ``< 3``, ``3-20``, ``>= 20``.
        break_type : {"quantile", "score", "maxstat"}, default="quantile"
            Interpretation of ``breaks``. ``"maxstat"`` finds the cutpoint(s)
            that maximally separate the curves via the log-rank test
            (ignores ``breaks``).
        correction : {"bonferroni", "lausen-schumacher", "permutation"}, \
default="bonferroni"
            Multiple-testing correction for ``break_type="maxstat"``.
        n_permutations : int, default=1000
            Permutations when ``correction="permutation"``.
        random_state : int, optional
            Seed for permutation reproducibility.

        Returns
        -------
        DataFrame
            Log-rank test results, one block of rows per selected model
            (``model_type`` is the first column), each with an omnibus row
            (and pairwise rows for 3+ groups).  Pairwise ``comparison``
            labels use the human-readable group names (e.g.
            ``"< 25%" vs "25-75%"``).  A model whose test fails contributes
            no rows (empty result, with warning).

        Raises
        ------
        ValueError
            If ``evaluate()`` has not been called, risk groups cannot be
            formed for a single-model selection, the selection names an
            unknown model, or the list/dict selection is empty.
        TypeError
            If ``models`` is a raw model/pipeline (or other unsupported
            type) — pass a trained model name instead.
        MemoryError
            Re-raised if the computation exhausts available memory; resource
            errors are never swallowed by the broad failure handler.
        RecursionError
            Re-raised if the computation exceeds the recursion limit; resource
            errors are never swallowed by the broad failure handler.
        """
        self._ensure_setup()

        from ..viz._survival_helpers import _warn_breaks_precedence

        _warn_breaks_precedence(n_groups, breaks, break_type)

        # Precondition: evaluate() must have been called — checked once
        # here so multi-model selections fail fast instead of per model.
        if self.evaluation_result is None:
            raise ValueError(
                "No evaluation result. Call evaluate() before logrank_test()."
            )

        resolved, single_form = self._resolve_models_multi(models)
        # Result-affecting parameters stamped on every stored/returned
        # table so it is self-describing.
        attr_params = {
            'n_groups': n_groups, 'p_adjust': p_adjust,
            'break_type': break_type, 'breaks': breaks,
            'correction': correction,
        }

        return self._run_risk_test_over_models(
            resolved, single_form, attr_params,
            run_single=lambda mr: self._logrank_test_single(
                mr, n_groups=n_groups, p_adjust=p_adjust,
                breaks=breaks, break_type=break_type,
                correction=correction, n_permutations=n_permutations,
                random_state=random_state, attr_params=attr_params,
            ),
            attr_name='logrank_test',
            empty_columns=_LOGRANK_EMPTY_COLUMNS,
            method_label='logrank_test',
            print_title='Log-Rank Test',
            verbose=verbose,
        )

    def interactions(
        self: SurvivalLearner,
        model=None,
        method: str = 'hstat',
        top_k: int = 5,
        max_samples: int = 100,
        random_state=None,
        show_progress: bool = True,
        compute_surfaces: bool = True,
        grid_resolution: int = 25,
    ) -> "pd.DataFrame":
        """Compute pairwise feature interactions.

        Identifies which feature pairs jointly affect predictions more
        than their individual contributions.  Requires ``evaluate()`` to
        have been called first (uses permutation importance to select top
        features).

        Parameters
        ----------
        model : str, ModelResult, or None
            Model to analyse.  If ``None`` and only one model is trained,
            uses it automatically.
        method : str, default='hstat'
            Interaction detection method:

            - ``'hstat'`` — Friedman's H-statistic (model-agnostic, PDP-based).
              Works with any nonlinear model.
            - ``'shap'`` — SHAP interaction values via TreeExplainer.
              Only works with tree-based models (xgbcox, xgbaft, gbsa, rsf).
              Returns per-sample interaction matrix stored on
              ``model.evaluation.interaction_values_raw``. These values are
              path-dependent (true-to-the-data), a different reference than
              the interventional cohort SHAP beeswarm; the two can diverge
              when features are correlated.
        top_k : int, default=5
            Number of top features (by permutation importance) to test.
            Produces ``C(top_k, 2)`` pairs.
        max_samples : int, default=100
            Maximum samples for computation (subsampled from X_test).
        random_state : int, optional
            Seed for reproducible subsampling.  Falls back to the learner's
            ``random_state`` when ``None``.
        show_progress : bool, default=True
            Print progress messages per pair.
        compute_surfaces : bool, default=True
            If True, also compute and cache 2D partial dependence surfaces
            for each pair.  These are used by ``plot_interaction_pdp()``
            and the HTML report.
        grid_resolution : int, default=25
            Maximum grid points per feature for PDP surfaces.

        Returns
        -------
        DataFrame
            For ``method='hstat'``: columns
            ``[model_type, feature_1, feature_2, h_statistic, method]``.
            For ``method='shap'``: columns
            ``[model_type, feature_1, feature_2, shap_interaction, method]``.

        Raises
        ------
        ValueError
            If ``evaluate()`` has not been called, fewer than 2 features
            are available, or ``method`` is invalid.
        TypeError
            If ``method='shap'`` is used with a model that is not tree-based.
        """
        self._ensure_setup()

        _VALID_METHODS = ('hstat', 'shap')
        if method not in _VALID_METHODS:
            raise ValueError(
                f"Invalid interaction method '{method}'. "
                f"Choose from: {_VALID_METHODS}."
            )

        import pandas as pd
        from ..evaluation._interactions import (
            _compute_h_statistic,
            _compute_shap_interactions,
            _is_linear_model,
            _select_interaction_features,
        )

        # Resolve model
        mr = self._resolve_single_model(
            model if isinstance(model, str) or model is None
            else getattr(model, 'name', None)
        )
        if mr.evaluation is None:
            raise ValueError(
                f"Model '{mr.name}' has no evaluation result.  "
                "Call evaluate() before interactions()."
            )

        random_state = random_state if random_state is not None else self.random_state
        X_test = self.X_test
        cat_groups = self._state.categorical_groups or {}
        cat_features = (
            set(self._state.categorical_features)
            if self._state.categorical_features is not None else None
        )

        top_features = _select_interaction_features(
            mr.evaluation.importance, top_k, X_test.columns,
            categorical_groups=cat_groups or None,
        )
        if len(top_features) < 2:
            raise ValueError(
                f"Need at least 2 features for interaction testing, "
                f"got {len(top_features)}."
            )

        if method == 'hstat':
            if _is_linear_model(mr.model):
                raise ValueError(
                    f"Model '{mr.name}' is linear — its H-statistic is always "
                    "zero by construction. Use a nonlinear model (RSF, GBSA, "
                    "XGBoost, DeepSurv, DeepHit) for interaction detection."
                )

            h_result = _compute_h_statistic(
                mr.model, X_test, top_features,
                max_samples=max_samples,
                random_state=random_state,
                show_progress=show_progress,
                model_name=mr.name,
                compute_surfaces=compute_surfaces,
                grid_resolution=grid_resolution,
                categorical_groups=cat_groups or None,
                categorical_features=cat_features,
            )

            if compute_surfaces:
                result, surfaces = h_result
                # Tag surfaces with model type
                for s in surfaces:
                    s['model_type'] = mr.name
                mr.evaluation.interaction_surfaces = surfaces
            else:
                result = h_result
            mr.evaluation.interaction_method = 'hstat'
            mr.evaluation._interaction_raw_cache = None

        else:  # method == 'shap'
            result, raw_array, feat_names = _compute_shap_interactions(
                mr.model, X_test, top_features,
                max_samples=max_samples,
                random_state=random_state,
                show_progress=show_progress,
                model_name=mr.name,
                categorical_groups=cat_groups or None,
            )
            mr.evaluation.interaction_values_raw = raw_array
            mr.evaluation.interaction_feature_names = feat_names
            mr.evaluation.interaction_method = 'shap'
            mr.evaluation._interaction_raw_cache = {mr.name: (raw_array, feat_names)}

        result['model_type'] = mr.name
        # Reorder to have model_type first
        cols = ['model_type'] + [c for c in result.columns if c != 'model_type']
        result = result[cols]

        # Merge with existing interactions from the other method (if any).
        # Same-method rows are replaced; other-method rows are preserved.
        existing = mr.evaluation.interactions
        if existing is not None and 'method' in existing.columns:
            other_method = existing[existing['method'] != method]
            if len(other_method) > 0:
                result = pd.concat([other_method, result], ignore_index=True)
                mr.evaluation.interaction_method = 'both'

        # Store on the model's evaluation so plot_interactions() can find it
        mr.evaluation.interactions = result

        return result

    def _shap(
        self: SurvivalLearner,
        model_result,
        max_explain: int = 50,
        max_background: int = 50,
        random_state=None,
        show_progress: bool = True,
    ) -> "pd.DataFrame":
        """Compute whole-cohort SHAP values for a trained model.

        Internal backend behind :meth:`importance` with ``method='shap'``
        (the public entry point) — not called directly.

        Requires ``evaluate()`` to have been called on the model first.
        Attaches ``shap_values`` (and ``shap_values_per_column`` when
        ``categorical_groups`` is configured) to the existing
        ``ModelResult.evaluation`` in place; returns the per-sample SHAP
        DataFrame, which :meth:`importance` aggregates into the global summary.
        When the model already had SHAP values (from ``evaluate(include_shap=
        True)`` or a prior call), the progress line notes that they are being
        replaced.

        Parameters
        ----------
        model_result : ModelResult
            The registered model to explain (resolved once by the public
            :meth:`importance` boundary).
        max_explain : int, default=50
            Maximum samples to explain (subsampled from ``X_test``).
        max_background : int, default=50
            Background dataset size for the explainer.
        random_state : int, optional
            Seed for reproducible subsampling.  Falls back to the learner's
            ``random_state`` when ``None``.
        show_progress : bool, default=True
            Print a one-line progress message.

        Returns
        -------
        DataFrame
            Long-format SHAP values with columns
            ``[model_type, feature, sample_idx, shap_value, feature_value]``.

        Raises
        ------
        ValueError
            If ``evaluate()`` has not been called on this model.
        """
        self._ensure_setup()

        from ..evaluation._shap import _compute_shap, _group_shap_values
        from ..evaluation.importance import _validate_categorical_groups

        mr = model_result
        if mr.evaluation is None:
            raise ValueError(
                f"Model '{mr.name}' has no evaluation result.  "
                "Call evaluate() before importance(method='shap')."
            )

        rs = random_state if random_state is not None else self.random_state
        X_test = self.X_test
        X_train = self.X_train
        cat_groups = self._state.categorical_groups or {}

        had_prior = mr.evaluation.shap_values is not None
        if show_progress:
            replacing = " (replacing previous results)" if had_prior else ""
            print(
                f"Computing SHAP for {mr.name}{replacing}...",
                end="", flush=True,
            )

        shap_df = _compute_shap(
            mr.model, X_test, X_background=X_train,
            max_background=max_background,
            max_explain=max_explain,
            random_state=rs,
        )
        shap_df['model_type'] = mr.name

        shap_per_column_df = None
        if cat_groups:
            valid_groups = _validate_categorical_groups(
                cat_groups, shap_df['feature'].unique().tolist(),
            )
            if valid_groups:
                shap_per_column_df = shap_df.copy()
                shap_df = _group_shap_values(shap_df, valid_groups)

        mr.evaluation.shap_values = shap_df
        mr.evaluation.shap_values_per_column = shap_per_column_df

        if show_progress:
            print(" done.")

        return shap_df

    def _permutation_importance(
        self: SurvivalLearner,
        model_result,
        n_importance_repeats: int = 10,
        importance_max_samples: int | None = None,
        random_state=None,
        show_progress: bool = True,
        n_jobs=None,
    ) -> "pd.DataFrame":
        """Compute permutation feature importance for a trained model.

        Internal backend behind :meth:`importance` with
        ``method='permutation'`` (the public entry point) — not called
        directly.

        Requires ``evaluate()`` to have been called on the model first.
        Recomputes permutation importance on ``X_test`` and overwrites the
        ``importance`` / ``importance_iterations`` (and
        ``importance_per_column`` when ``categorical_groups`` is configured)
        fields already populated by ``evaluate()``, attaching the fresh
        results to the existing ``ModelResult.evaluation`` in place; returns
        the summary DataFrame for direct use.

        Use ``importance(method='permutation')`` to recompute importance at a
        different ``n_importance_repeats`` or ``random_state`` without
        re-running the full ``evaluate()`` metric suite.  When the model
        already had importance results, the progress line notes that they are
        being replaced.

        Parameters
        ----------
        model_result : ModelResult
            The registered model to explain (resolved once by the public
            :meth:`importance` boundary).
        n_importance_repeats : int, default=10
            Number of permutation repeats per feature.
        importance_max_samples : int, optional
            Cap on the test rows used. ``None`` (the default) uses every
            row; a value at or above the cohort size is a no-op, and the
            minimum is 2. Mirrors ``evaluate(importance_max_samples=...)``.
        random_state : int, optional
            Seed for reproducible permutations.  Falls back to the learner's
            ``random_state`` when ``None``.
        show_progress : bool, default=True
            Print a one-line progress message.
        n_jobs : int, optional
            Parallel jobs across features.  Falls back to the learner's
            ``n_jobs`` when ``None``.

        Returns
        -------
        DataFrame
            Importance summary with columns
            ``[model_type, feature, coefficient, perm_mean, perm_std]``.
            When ``categorical_groups`` is configured, group names replace
            their encoded columns.  If the recomputation fails, a
            ``SuggestionWarning`` is emitted and the existing importance from
            ``evaluate()`` is preserved and returned unchanged; with no prior
            importance to keep (none stored, or only an empty table), an
            empty 0-row table with the same schema is returned instead.

        Raises
        ------
        ValueError
            If ``evaluate()`` has not been called on this model.

        See Also
        --------
        importance : Public front door; ``method='shap'`` gives the analogous
            SHAP explainer.
        importance_test : Significance tests for feature importance.
        """
        self._ensure_setup()

        from ..evaluation.importance import _compute_permutation_importance

        mr = model_result
        if mr.evaluation is None:
            raise ValueError(
                f"Model '{mr.name}' has no evaluation result.  "
                "Call evaluate() before importance(method='permutation')."
            )

        rs = random_state if random_state is not None else self.random_state
        nj = n_jobs if n_jobs is not None else self.n_jobs
        cat_groups = self._state.categorical_groups or {}

        prior = mr.evaluation.importance
        had_prior = prior is not None and not prior.empty

        summary_df, iterations_df, per_column_df = _compute_permutation_importance(
            mr.model, self.X_test, self.y_test,
            n_importance_repeats, rs,
            show_progress=show_progress, model_name=mr.name,
            n_jobs=nj, categorical_groups=cat_groups,
            replacing=had_prior,
            max_samples=importance_max_samples,
        )

        # On failure the helper returns an empty sentinel (and has already
        # warned via log_metric_failure).  Don't clobber the good importance
        # results from evaluate() with empties — preserve and return them;
        # with no prior to keep (None, or an empty table from
        # n_importance_repeats=0), return an empty table with the standard
        # schema instead.
        if summary_df.empty:
            import pandas as pd

            kept = (
                "the existing importance from evaluate() was kept unchanged"
                if had_prior
                else "no prior importance exists, so an empty table is "
                     "returned"
            )
            warnings.warn(
                f"importance('{mr.name}', method='permutation') computed no "
                f"results; {kept}.  See the preceding warning for the cause.",
                SuggestionWarning, stacklevel=2,
            )
            if not had_prior:
                return pd.DataFrame(columns=_PERM_IMPORTANCE_COLUMNS)
            return prior

        summary_df.insert(0, 'model_type', mr.name)
        iterations_df.insert(0, 'model_type', mr.name)
        if per_column_df is not None:
            per_column_df.insert(0, 'model_type', mr.name)

        mr.evaluation.importance = summary_df
        mr.evaluation.importance_iterations = iterations_df
        mr.evaluation.importance_per_column = per_column_df

        return summary_df

    def importance(
        self: SurvivalLearner,
        models=None,
        method: str = "permutation",
        *,
        n_importance_repeats: int = 10,
        importance_max_samples: int | None = None,
        max_explain: int = 50,
        max_background: int = 50,
        random_state=None,
        show_progress: bool = True,
        n_jobs=None,
    ) -> "pd.DataFrame":
        """Compute feature importance for trained models — one entry point.

        Convenience wrapper that dispatches to the underlying importance
        methods so callers don't have to remember which is named what.
        Requires ``evaluate()`` to have been called on the model(s) first
        (the wrapped methods enforce this).

        - ``method="permutation"`` (default) computes permutation importance
          via the internal ``_permutation_importance`` backend.
        - ``method="shap"`` computes whole-cohort SHAP values via the internal
          ``_shap`` backend and aggregates its per-sample values into a
          per-feature summary.  The per-sample values are stored on
          ``result.evaluation.shap_values`` for ``plot_shap`` / beeswarm.

        Both methods return one row per (model, feature), so their outputs
        line up for comparison — but the columns differ (see Returns).

        Parameters
        ----------
        models : str, ModelResult, list, dict, or None
            Model selection. ``None`` (default) computes importance for
            every trained model; a list/dict selects a subset. The result
            stacks one block of rows per model (``model_type`` column).
            With a multi-model selection, a model without an evaluation
            result is skipped with a warning instead of raising. A
            ``ModelResult`` is resolved by name to the currently
            registered model; a stale reference (e.g. held from before
            ``tune()``) triggers a ``SuggestionWarning`` and the
            registered model is used.
        method : {'permutation', 'shap'}, default='permutation'
            Which importance to compute.
        n_importance_repeats : int, default=10
            Permutation repeats per feature.  Ignored when ``method='shap'``.
        importance_max_samples : int, optional
            Cap on the test rows used.  ``None`` (the default) uses every
            row; the minimum is 2.  Ignored when ``method='shap'`` (use
            ``max_explain``).
        max_explain : int, default=50
            Maximum samples to explain.  Used only when ``method='shap'``.
        max_background : int, default=50
            Background dataset size for the explainer.  Used only when
            ``method='shap'``.
        random_state : int, optional
            Seed for reproducibility.  Falls back to the learner's
            ``random_state`` when ``None``.
        show_progress : bool, default=True
            Print a one-line progress message.
        n_jobs : int, optional
            Parallel jobs across features.  Used only when
            ``method='permutation'``; falls back to the learner's ``n_jobs``.

        Returns
        -------
        DataFrame
            One row per (model, feature), one block of rows per selected
            model.  For ``method='permutation'`` the columns are
            ``[model_type, feature, coefficient, perm_mean, perm_std]``.  For
            ``method='shap'`` they are ``[model_type, feature, mean_abs_shap]``
            (mean absolute SHAP value), sorted by ``mean_abs_shap`` descending
            within each model block.  The full per-sample SHAP values remain
            available on ``result.evaluation.shap_values``.

        Raises
        ------
        ValueError
            If ``method`` is not ``'permutation'`` or ``'shap'``, if
            ``evaluate()`` has not been called on a single-model selection,
            if the selection names an unknown model, or if the list/dict
            selection is empty.
        TypeError
            If ``models`` is a raw model/pipeline (or other unsupported
            type) — pass a trained model name instead.

        See Also
        --------
        importance_test : Significance tests for feature importance.
        """
        import pandas as pd

        if method not in ("permutation", "shap"):
            raise ValueError(
                f"Unknown importance method {method!r}; "
                "expected 'permutation' or 'shap'."
            )

        self._ensure_setup()
        resolved, single_form = self._resolve_models_multi(models)

        frames = []
        for mr in resolved:
            if not single_form and mr.evaluation is None:
                # Single-model selections keep the backends' ValueError;
                # in a multi-model selection a missing evaluation only
                # drops that model from the stack.
                warnings.warn(
                    f"Model '{mr.name}' has no evaluation result; skipped. "
                    f"Call evaluate('{mr.name}') to include it.",
                    SuggestionWarning, stacklevel=2,
                )
                continue
            if method == "permutation":
                df = self._permutation_importance(
                    mr,
                    n_importance_repeats=n_importance_repeats,
                    importance_max_samples=importance_max_samples,
                    random_state=random_state,
                    show_progress=show_progress,
                    n_jobs=n_jobs,
                )
            else:
                shap_df = self._shap(
                    mr,
                    max_explain=max_explain,
                    max_background=max_background,
                    random_state=random_state,
                    show_progress=show_progress,
                )
                df = (
                    shap_df.groupby(
                        ["model_type", "feature"], sort=False
                    )["shap_value"]
                    .apply(lambda s: s.abs().mean())
                    .reset_index(name="mean_abs_shap")
                    .sort_values(
                        "mean_abs_shap", ascending=False, ignore_index=True
                    )
                )
            if single_form:
                return df
            if df is None or df.empty:
                # The permutation backend preserves the prior importance on
                # recomputation failure (warning already emitted); with
                # nothing to keep it has no rows to contribute.
                continue
            if 'model_type' not in df.columns:
                # A prior table preserved from a single-model evaluate() may
                # predate the model_type column; label it (on a copy — the
                # stored table stays untouched) so the stack stays typed.
                df = df.copy()
                df.insert(0, 'model_type', mr.name)
            frames.append(df)

        if not frames:
            warnings.warn(
                f"importance() produced no results for any of the "
                f"{len(resolved)} selected models; returning an empty "
                "table. See the preceding warnings for the causes.",
                MethodologicalWarning, stacklevel=2,
            )
            columns = (
                _PERM_IMPORTANCE_COLUMNS
                if method == "permutation"
                else ["model_type", "feature", "mean_abs_shap"]
            )
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True)
