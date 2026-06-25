"""Shared resolution utilities for the viz data-prep layer.

Resolves flexible ``models=``/``references=`` arguments and assembles/filters
``EvaluationResult`` objects. Imported downward (as ``mlsurv.viz._resolve``) by
both ``learner``'s ``VisualizationMixin`` and the ``viz/_plot_impl_*`` renderers
(relocated from ``learner/_visualization_helpers``)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .._warnings import InternalWarning, SuggestionWarning


def tidy_model_col(df: pd.DataFrame) -> pd.DataFrame:
    """Rename 'model_type' -> 'model' for user-facing return_data output."""
    if "model_type" in df.columns and "model" not in df.columns:
        return df.rename(columns={"model_type": "model"})
    return df


def _cal_entry_to_rows(entry, mname):
    """Convert a single calibration entry dict to a list of row dicts."""
    ci_lower = entry.get("ci_lower", [])
    ci_upper = entry.get("ci_upper", [])
    rows = []
    for i, (p, o) in enumerate(
        zip(entry["predicted"], entry["observed"])
    ):
        row = {
            "model": mname,
            "eval_time": entry["eval_time"],
            "predicted": p,
            "observed": o,
        }
        if len(ci_lower) > i:
            row["ci_lower"] = ci_lower[i]
        if len(ci_upper) > i:
            row["ci_upper"] = ci_upper[i]
        rows.append(row)
    return rows


def calibration_to_tidy(cal_data, model_name=None):
    """Convert calibration compute output to tidy DataFrame.

    Handles both single-model (list[dict]) and multi-model (dict[str, dict])
    calibration data structures.
    """
    rows = []
    if isinstance(cal_data, list):
        # Single-model: list[dict] with array values per eval_time
        for entry in cal_data:
            rows.extend(_cal_entry_to_rows(entry, model_name or "model"))
    elif isinstance(cal_data, dict):
        # Multi-model: dict[model_name, dict] at one eval_time
        for mname, mdata in cal_data.items():
            rows.extend(_cal_entry_to_rows(mdata, mname))
    else:
        raise TypeError(
            f"calibration_to_tidy expected list or dict, got "
            f"{type(cal_data).__name__}"
        )
    return pd.DataFrame(rows)


def filter_eval_result(eval_result, model_names):
    """Return a filtered copy of *eval_result* with DataFrames subset to *model_names*."""
    from mlsurv.results.containers import EvaluationResult, BootstrapResult

    def _filt(df):
        if df is None:
            return None
        if 'model_type' not in df.columns:
            return df
        return df[df['model_type'].isin(model_names)].reset_index(drop=True)

    # Filter bootstrap result if present
    boot = eval_result.bootstrap
    if boot is not None:
        boot = BootstrapResult(
            summary=_filt(boot.summary),
            iterations=_filt(boot.iterations),
            predictions=_filt(boot.predictions),
            auc_t=_filt(boot.auc_t),
            brier_t=_filt(boot.brier_t),
            subpopulation_auc_t=_filt(boot.subpopulation_auc_t),
            importance=_filt(boot.importance),
            lower_is_better=boot.lower_is_better,
            ci_level=boot.ci_level,
        )

    # Filter surv_probs cache to matching models
    surv_cache = None
    if eval_result._surv_probs_cache:
        filtered = {k: v for k, v in eval_result._surv_probs_cache.items()
                    if k in model_names}
        surv_cache = filtered or None

    # Filter interaction raw cache to matching models
    interaction_raw_cache = None
    if eval_result._interaction_raw_cache:
        filtered_irc = {k: v for k, v in eval_result._interaction_raw_cache.items()
                        if k in model_names}
        interaction_raw_cache = filtered_irc or None

    # Filter interaction_surfaces (list-of-dicts, not DataFrame)
    filtered_surfaces = None
    if eval_result.interaction_surfaces:
        filtered_surfaces = [
            s for s in eval_result.interaction_surfaces
            if s.get('model_type') in model_names
        ] or None

    return EvaluationResult(
        concordance=_filt(eval_result.concordance),
        test_set=_filt(eval_result.test_set),
        auc_t=_filt(eval_result.auc_t),
        importance=_filt(eval_result.importance),
        importance_iterations=_filt(eval_result.importance_iterations),
        importance_per_column=_filt(eval_result.importance_per_column),
        shap_values=_filt(eval_result.shap_values),
        shap_values_per_column=_filt(eval_result.shap_values_per_column),
        interactions=_filt(eval_result.interactions),
        interaction_surfaces=filtered_surfaces,
        interaction_values_raw=eval_result.interaction_values_raw,
        interaction_feature_names=eval_result.interaction_feature_names,
        interaction_method=eval_result.interaction_method,
        bootstrap=boot,
        model_order=eval_result.model_order,
        _surv_probs_cache=surv_cache,
        _interaction_raw_cache=interaction_raw_cache,
    )


def compose_models_with_refs(models, ref_names):
    """Extend a ``models=`` kwarg with benchmark reference names.

    Used by ``plot_*(models=..., references=...)`` to preserve the
    user's ``models=`` filter intent while still including injected
    benchmark refs in the recursive call. Without this helper the
    recursive call drops either the user's ``models=`` (if it forwards
    ``None``) or the refs (if it forwards the original list).

    Parameters
    ----------
    models : None, str, or list[str]
        The user's ``models=`` argument.
    ref_names : list[str]
        Resolved reference names from ``resolve_references_param``.

    Returns
    -------
    None or list[str]
        ``None`` (meaning "all models, including injected refs") when
        ``models is None``; otherwise a concrete list containing the
        user-requested models plus all reference names, order-preserving
        and de-duplicated so a name appearing in both inputs (rare
        ref/trained-model name clash) renders exactly once.
    """
    if models is None:
        return None
    if isinstance(models, str):
        models = [models]
    # Order-preserving dedupe: ``dict.fromkeys`` keeps first-seen order.
    return list(dict.fromkeys([*models, *ref_names]))


def resolve_eval_result(state, model_names=None):
    """Get an EvaluationResult, optionally filtered to *model_names*.

    Assembles a combined EvaluationResult from per-model ModelResult data.
    """
    from ..results.containers import EvaluationResult

    evaluated = {
        n: m for n, m in state.models.items()
        if m.evaluation is not None
    }
    eval_result = EvaluationResult.combine(evaluated) if evaluated else None
    if eval_result is None:
        raise ValueError("No evaluation results. Run evaluate() first.")

    if model_names is not None:
        eval_result = filter_eval_result(eval_result, model_names)

    return eval_result


def resolve_references_param(benchmark_result, references):
    """Resolve the ``references`` parameter to a list of names.

    Parameters
    ----------
    benchmark_result : object or None
        The learner's ``benchmark_result``.
    references : None, ``"all"``, or list of str
        Reference selection.

    Returns
    -------
    list[str] or None
        Resolved reference names, or None if ``references`` is None.

    Raises
    ------
    TypeError
        If *references* is a single string rather than ``None``, ``"all"``,
        or a list of names.
    ValueError
        If *references* is requested but no benchmark results exist, or if
        any requested reference name is unknown.
    """
    if references is None:
        return None
    br = benchmark_result
    if br is None:
        raise ValueError(
            "No benchmark results. Call benchmark() before using "
            "references parameter."
        )
    if references == "all":
        return list(br.reference_names)
    if isinstance(references, str):
        raise TypeError(
            f"references must be None, 'all', or a list of str, "
            f"got string {references!r}. "
            f"Did you mean references=[{references!r}]?"
        )
    ref_names = list(references)
    unknown = set(ref_names) - set(br.reference_names)
    if unknown:
        raise ValueError(
            f"Unknown references: {unknown}. "
            f"Available: {br.reference_names}"
        )
    return ref_names


def build_reference_palette(benchmark_result, ref_names, kwargs):
    """Build merged model + reference palette, mutating *kwargs* in place."""
    from mlsurv.viz._palettes import build_benchmark_palette

    br = benchmark_result
    user_palette = kwargs.pop("palette", {}) or {}
    # Build for all references so each keeps its stable positional color.
    base = build_benchmark_palette(br.model_names, br.reference_names)
    # Keep only models + requested references.
    keep = set(br.model_names) | set(ref_names)
    base = {k: v for k, v in base.items() if k in keep}
    # User overrides win.
    base.update(user_palette)
    kwargs["palette"] = base


class _swap_benchmark_eval:
    """Context manager: inject benchmark reference models into state.models.

    Temporarily adds synthetic ``ModelResult`` entries for each reference
    model so that the per-model computed properties (``evaluation_result``,
    ``bootstrap_result``) include the benchmark reference data.  Entries
    are removed on exit.

    Used by ``plot_metric`` and ``plot_metric_time`` for reference inclusion.
    """

    def __init__(self, state, benchmark_result, ref_names, kwargs):
        br = benchmark_result
        keep = set(br.model_names) | set(ref_names)
        self._state = state
        self._ref_names = ref_names
        self._injected_keys = []
        # Reference-capture: store (name, installed_eval_object) so
        # __exit__ can restore only the entries whose current
        # ``.evaluation`` is still the exact object we installed. If
        # code inside the ``with`` block replaced the evaluation with a
        # different object (e.g., a legitimate ``evaluate()`` call), we
        # leave the new value alone rather than silently wiping it.
        # NOTE: this check is identity-based, so in-place mutation of
        # the installed ``EvaluationResult`` would still match. In
        # practice ``EvaluationResult`` is a frozen dataclass and no
        # code path mutates its fields after construction, so the
        # identity check is equivalent to "nothing changed this slot".
        self._backfilled_evals = []
        self._filtered_eval = filter_eval_result(br.evaluation, keep)
        build_reference_palette(benchmark_result, ref_names, kwargs)

    def _backfill_validate(self, name, single_eval) -> str | None:
        """Return a user-facing reason string when *single_eval* cannot
        drive a trained-model plot, or ``None`` when it can.

        Called before overwriting ``state.models[name].evaluation``: if
        validation fails, the caller skips the backfill and emits a
        single ``SuggestionWarning`` naming the exact cause so users
        diagnose the issue in one hop instead of tracing through
        downstream "non-finite risk score" warnings.

        Three failure modes are covered:

        1. ``test_set`` is absent or empty — benchmark config skipped
           per-model test predictions for this name.
        2. ``test_set`` is missing the ``time``/``event``/``risk_score``
           columns that ``_get_survival_test_data`` requires — would
           otherwise trip a deep ``KeyError`` inside
           ``_data._extract_from_eval_result``.
        3. ``risk_score`` values are all non-finite — would silently
           drop the model later via
           ``validate_risk_scores`` with a generic "non-finite" message
           that doesn't name the benchmark backfill as the cause.

        Returns
        -------
        str or None
            Human-readable reason string suitable for embedding in a
            ``SuggestionWarning``, or ``None`` if the backfill is safe.
        """
        # ``extract_model_evaluation`` always produces a DataFrame
        # (possibly empty) via ``_filter_model_type``, so ``test_set is
        # None`` is defensively unreachable here. Kept for safety in
        # case that contract ever changes.
        test_set = getattr(single_eval, "test_set", None)
        if test_set is None or test_set.empty:
            return (
                f"benchmark evaluation has no test_set rows for '{name}'"
            )
        required = {"time", "event", "risk_score"}
        missing = required - set(test_set.columns)
        if missing:
            return (
                f"benchmark test_set for '{name}' is missing required "
                f"column(s): {sorted(missing)}"
            )
        scores = test_set["risk_score"].values
        if len(scores) == 0 or not np.any(np.isfinite(scores)):
            return (
                f"benchmark test_set for '{name}' has no finite "
                f"risk_score values"
            )
        return None

    def __enter__(self):
        from ..results.containers import ModelResult, _get_model_names
        from ..results.containers import extract_model_evaluation

        ref_model_names = _get_model_names(self._filtered_eval.concordance)
        try:
            for name in ref_model_names:
                if name not in self._state.models:
                    single_eval = extract_model_evaluation(
                        self._filtered_eval, name,
                    )
                    mr = ModelResult(
                        name=name,
                        model=None,
                        cv_metric_scores=np.array([]),
                        cv_metric_mean=np.nan,
                        cv_metric_std=np.nan,
                        evaluation=single_eval,
                        is_synthetic=True,
                    )
                    self._injected_keys.append(name)
                    self._state.models[name] = mr
                elif name in self._ref_names:
                    warnings.warn(
                        f"Benchmark reference '{name}' has the same name "
                        f"as a trained model. The trained model's data "
                        f"will be shown instead of the benchmark reference.",
                        SuggestionWarning,
                        # __enter__ -> _plot_X_impl -> plot_X wrapper -> user
                        stacklevel=5,
                    )
                elif self._state.models[name].evaluation is None:
                    # Trained model has no per-model eval — benchmark()
                    # populated ``benchmark_result.evaluation`` but never
                    # wrote back to ``state.models[name].evaluation``.
                    # Backfill from the filtered benchmark eval so
                    # ``_get_survival_test_data`` (and any other caller
                    # reading per-model evaluation) sees the trained
                    # model alongside the injected refs. Mirrors the
                    # pattern in ``_evaluation.py:_store_per_model_evaluations``.
                    single_eval = extract_model_evaluation(
                        self._filtered_eval, name,
                    )
                    reason = self._backfill_validate(name, single_eval)
                    if reason is not None:
                        # Defensive skip: surface a single, specific
                        # reason rather than silently installing a bad
                        # eval (which would either drop the model from
                        # plots, trip a KeyError downstream, or emit a
                        # generic "non-finite risk score" warning two
                        # hops away from the real cause).
                        warnings.warn(
                            f"Cannot backfill trained model '{name}' "
                            f"from benchmark_result: {reason}. Call "
                            f"learner.evaluate() before plotting to see "
                            f"this model alongside references.",
                            SuggestionWarning,
                            # __enter__ -> _plot_X_impl -> plot_X wrapper -> user
                            stacklevel=5,
                        )
                        continue
                    self._state.models[name].evaluation = single_eval
                    self._backfilled_evals.append((name, single_eval))
        except Exception:
            # Clean up partially injected/backfilled entries before re-raising
            for key in self._injected_keys:
                self._state.models.pop(key, None)
            for key, installed in self._backfilled_evals:
                mr = self._state.models.get(key)
                if mr is not None and mr.evaluation is installed:
                    mr.evaluation = None
            self._injected_keys.clear()
            self._backfilled_evals.clear()
            raise
        return self

    def __exit__(self, *exc):
        for key in self._injected_keys:
            self._state.models.pop(key, None)
        for key, installed in self._backfilled_evals:
            # Restore to ``None`` only when the current eval is still
            # the exact object we installed — identity check, not
            # equality. If something inside the ``with`` block replaced
            # ``.evaluation`` with a different object, we leave that new
            # value alone rather than silently wiping it.
            mr = self._state.models.get(key)
            if mr is not None and mr.evaluation is installed:
                mr.evaluation = None
        return False


def validate_risk_scores(result: dict) -> dict:
    """Exclude models with non-finite risk scores."""
    validated = {}
    for name, df in result.items():
        scores = df['risk_score'].values
        if not np.all(np.isfinite(scores)):
            n_bad = int((~np.isfinite(scores)).sum())
            warnings.warn(
                f"Model '{name}' produced {n_bad} non-finite risk "
                f"score(s). Excluding from visualization.",
                InternalWarning,
                stacklevel=3,
            )
            continue
        validated[name] = df
    return validated


# ------------------------------------------------------------------
# Phase 2: CC-reduction helpers
# ------------------------------------------------------------------

def collect_reference_survival_data(state, ref_names):
    """Collect risk-score DataFrames for benchmark reference models.

    Returns ``{ref_name: DataFrame}`` with columns ``time, event, risk_score``.
    """
    ref_data = {}
    if ref_names is None:
        return ref_data
    br = state.benchmark_result
    test_set = br.evaluation.test_set
    if test_set is None:
        return ref_data
    for rname in ref_names:
        rsub = test_set[test_set["model_type"] == rname]
        if not rsub.empty:
            ref_data[rname] = rsub[["time", "event", "risk_score"]].copy()
    return ref_data


def survival_test_data_to_tidy(test_data):
    """Convert ``{name: DataFrame}`` to a single tidy DataFrame."""
    if not test_data:
        raise ValueError(
            "No survival test data available. Ensure models "
            "have been trained and evaluated."
        )
    frames = []
    for name, df in test_data.items():
        frame = df.copy()
        frame.insert(0, "model", name)
        frames.append(frame)
    import pandas as pd
    return pd.concat(frames, ignore_index=True)
