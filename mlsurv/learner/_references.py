"""Shared helpers for scoring reference biomarkers in benchmark() and validate().

Reference biomarkers (LORIS, TMB, PD-L1, etc.) are external scalar scores
attached per-patient. ``benchmark()`` compares them against trained models on
the internal test set; ``validate()`` compares them on external
cohorts. Both call into the same machinery here so the row layout, NaN
handling, and metric filtering stay identical.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..constants import ALL_METRICS, SURV_FUNC_METRICS
from ..evaluation import evaluate
from ..models._reference_adapter import ReferenceAdapter
from ..utils import supports_survival_function
from .._warnings import MethodologicalWarning


# Bootstrap metrics to attempt for reference adapters — IBS / C^td / calibration
# require predict_survival_function which references structurally lack.
# Attempting + silencing produces unactionable noise; filter at dispatch instead.
_REFERENCE_BOOTSTRAP_METRICS = [m for m in ALL_METRICS if m not in SURV_FUNC_METRICS]


def _reference_bootstrap_metrics(adapter):
    """Return the bootstrap metric list appropriate for a reference adapter.

    References that declare ``supports_survival_function = False`` skip the
    surv-func-dependent metrics; any future adapter that does support them
    gets the full default list.
    """
    if supports_survival_function(adapter):
        return None  # default — all metrics
    return _REFERENCE_BOOTSTRAP_METRICS


def _resolve_higher_is_worse(higher_is_worse):
    """Return (per-reference map, fallback-for-unlisted).

    For a dict argument, the map is the dict and unlisted references fall back
    to ``True``. For a bool, the map is empty and the bool is the universal
    value.
    """
    if isinstance(higher_is_worse, dict):
        return higher_is_worse, True
    return {}, higher_is_worse


def _build_reference_adapters(reference_arrays, hiw_map, hiw_default):
    """Wrap each reference array in a ReferenceAdapter for evaluate()."""
    return {
        rname: ReferenceAdapter(
            scores, name=rname,
            higher_is_worse=hiw_map.get(rname, hiw_default),
        )
        for rname, scores in reference_arrays.items()
    }


def _evaluate_references_on(
    X, y, *,
    reference_arrays, adapters,
    hiw_map, hiw_default,
    y_train, eval_times, sg_labels,
    bootstrap, n_bootstrap, bootstrap_ci, random_state,
    _warn_stacklevel,
):
    """Evaluate each reference on (X, y), filtering NaN scores per-reference.

    Returns a list of ``(rname, EvaluationResult)`` tuples ready to feed into
    ``_aggregate_evaluation_results`` together with the model evaluation.
    """
    ref_parts = []
    for rname, scores in reference_arrays.items():
        valid = np.isfinite(scores)
        if valid.all():
            ref_adapter = adapters[rname]
            ref_X, ref_y, ref_sg = X, y, sg_labels
        else:
            ref_adapter = ReferenceAdapter(
                scores[valid],
                name=rname,
                higher_is_worse=hiw_map.get(rname, hiw_default),
            )
            ref_X = X[valid]
            ref_y = y[valid]
            ref_sg = sg_labels[valid] if sg_labels is not None else None
        eval_kwargs = dict(
            y_train=y_train,
            eval_times=eval_times,
            n_importance_repeats=0,
            bootstrap=bootstrap,
            n_bootstrap=n_bootstrap,
            bootstrap_ci=bootstrap_ci,
            include_predictions=bootstrap,
            random_state=random_state,
            show_progress=False,
            _warn_stacklevel=_warn_stacklevel,
            subpopulation_labels=ref_sg,
        )
        # Only pass bootstrap_metrics when bootstrap=True — otherwise a non-None
        # filtered list would trip evaluate()'s "unused bootstrap params"
        # SuggestionWarning. Passing None (the default) is silent but pointless.
        if bootstrap:
            eval_kwargs['bootstrap_metrics'] = _reference_bootstrap_metrics(ref_adapter)
        ref_eval = evaluate({rname: ref_adapter}, ref_X, ref_y, **eval_kwargs)
        ref_parts.append((rname, ref_eval))
    return ref_parts


def _resolve_validation_reference_arrays(references, X_val):
    """Resolve & validate reference arrays for ``validate(references=...)``.

    External cohorts are unrelated to training X, so we only accept arrays
    aligned with ``X_val`` — no full-X positional extraction (unlike
    ``benchmark()``'s resolver). NaN/Inf handling mirrors benchmark: count
    non-finite values per reference, raise if a reference is entirely
    non-finite, warn on any Inf values.

    Returns
    -------
    resolved : dict of {name: array}
        Reference arrays aligned with X_val (may contain NaN).
    nonfinite_counts : dict of {name: int}
        Number of non-finite values per reference (0 if clean).
    """
    n_val = len(X_val)
    resolved = {}
    nonfinite_counts = {}

    for name, arr in references.items():
        arr = np.asarray(arr, dtype=np.float64).ravel()
        if len(arr) != n_val:
            raise ValueError(
                f"Reference '{name}' has length {len(arr)}, expected "
                f"{n_val} (validation cohort size). External-cohort "
                f"references must be aligned with X_val."
            )

        n_inf = int(np.sum(np.isinf(arr)))
        n_nonfinite = int(np.sum(~np.isfinite(arr)))

        # All-NaN/Inf — drop the reference for this cohort and warn.
        # External cohorts may legitimately lack a biomarker; a single
        # missing column shouldn't be fatal when the caller passes a dict
        # assembled across heterogeneous cohorts.
        if n_nonfinite == len(arr):
            n_nan = n_nonfinite - n_inf
            warnings.warn(
                f"Reference '{name}': all {len(arr)} values are non-finite "
                f"({n_nan} NaN, {n_inf} Inf). Dropping from this cohort's "
                f"concordance table.",
                MethodologicalWarning,
                stacklevel=4,
            )
            continue

        resolved[name] = arr
        nonfinite_counts[name] = n_nonfinite

        # Partial NaN/Inf — user-visible warning per project's "No Silent Data
        # Loss" rule (CONTRIBUTING.md). Reference metrics will be computed on a
        # smaller cohort than the model rows; CI widths are not comparable
        # across rows of the same concordance table.
        if 0 < n_nonfinite < len(arr):
            n_nan = n_nonfinite - n_inf
            warnings.warn(
                f"Reference '{name}': {n_nan} NaN + {n_inf} Inf out of "
                f"{len(arr)} observations excluded; metrics computed on the "
                f"remaining {len(arr) - n_nonfinite}. Reference CIs and "
                f"effective N differ from model rows.",
                MethodologicalWarning,
                stacklevel=4,
            )

    return resolved, nonfinite_counts


def _report_nonfinite_references(reference_arrays, nonfinite_counts):
    """Print summary of non-finite reference values."""
    for rname, n_bad in nonfinite_counts.items():
        if n_bad > 0:
            n_total = len(reference_arrays[rname])
            n_valid = n_total - n_bad
            print(
                f"Reference '{rname}': {n_bad}/{n_total} observations "
                f"have non-finite values -- computing metrics on "
                f"{n_valid} valid observations."
            )
