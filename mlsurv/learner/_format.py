"""Display/formatting helpers for banner/box output.

These are module-level helpers with no ``self``/instance-state access: most take
primitives (or an ``EvalContext``) and return a display string. The two
exceptions are ``_print_param_box``, the generic renderer, which prints the
assembled box rather than returning it, and ``_format_subpopulation_summary``,
which returns a list of (label, value) rows. They live here so the mixins that
print parameter-summary boxes — ``_setup``, ``_tuning``, ``_evaluation_bootstrap``,
``_evaluation_core`` — share one formatting home instead of reaching into
``_setup`` for it.

``_print_param_box`` is the generic Unicode box renderer; everything else builds
a row (or set of rows) for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..tuning import auto_n_trials, auto_n_bootstrap, DEFAULT_STORAGE_BASE
from ..evaluation.evaluate import DEFAULT_N_IMPORTANCE_REPEATS
from ..utils import unwrap_pipeline
from ._evaluation_helpers import _format_auto_eval_times

if TYPE_CHECKING:
    from ._eval_context import EvalContext


# ---------------------------------------------------------------------------
# Generic box renderer
# ---------------------------------------------------------------------------

def _print_param_box(title: str, rows: list[tuple[str, str]]):
    """Print a Unicode-bordered parameter summary box."""
    if not rows:
        return
    label_width = max(len(label) for label, _ in rows)
    lines = [f"{label:<{label_width}} {value}" for label, value in rows]
    max_width = max(len(line) for line in lines)
    header = f"─ {title} "
    print(f"┌{header}" + "─" * (max_width + 2 - len(header)) + "┐")
    for line in lines:
        print(f"│ {line:<{max_width}} │")
    print("└" + "─" * (max_width + 2) + "┘", flush=True)


# ---------------------------------------------------------------------------
# Shared row builders (used by more than one mixin)
# ---------------------------------------------------------------------------

def _format_scoring_name(scoring) -> str:
    """Return a human-readable name for a scoring metric (str or callable)."""
    if isinstance(scoring, str):
        return scoring
    return getattr(scoring, '__name__', type(scoring).__name__)


def _format_model_display(resolved_model) -> str:
    """Return a display string for a model, Pipeline, or dict of models.

    For single models, prefers .display_name, then .name, falling back to
    the class name. For sklearn Pipelines, extracts display_name from the
    'model' step. For dicts, maps keys to display names using
    get_model_display_names().
    """
    if isinstance(resolved_model, dict):
        from ..models import get_model_display_names
        display_names = get_model_display_names()
        names = [display_names.get(k, k) for k in resolved_model]
        if len(names) <= 3:
            return ", ".join(names)
        return f"all ({len(names)} models)"
    # Single model or Pipeline
    obj = unwrap_pipeline(resolved_model)
    display = getattr(obj, 'display_name', None)
    if display:
        return display
    return getattr(obj, 'name', None) or type(obj).__name__


def _format_state_overrides(
    random_state, n_jobs, orig_random_state, orig_n_jobs, setup_random_state, setup_n_jobs,
) -> str:
    """Build a state string with override annotations when values differ from setup.

    Shows "(setup: X)" when the user explicitly passed a value that differs from
    the learner's setup default. No annotation when the user passed None (fallback)
    or passed the same value as setup.
    """
    seed_str = f"seed={random_state}"
    if orig_random_state is not None and orig_random_state != setup_random_state:
        seed_str = f"seed={random_state} (setup: {setup_random_state})"
    njobs_str = f"n_jobs={n_jobs}"
    if orig_n_jobs is not None and orig_n_jobs != setup_n_jobs:
        njobs_str = f"n_jobs={n_jobs} (setup: {setup_n_jobs})"
    return f"{seed_str}, {njobs_str}"


# ---------------------------------------------------------------------------
# setup() box rows
# ---------------------------------------------------------------------------

def _format_pipeline_str(imputer, scaler, feature_selector):
    """Format pipeline components into a display string."""
    steps = [
        (imputer, lambda s: type(s).__name__),
        (scaler, lambda s: type(s).__name__),
        (feature_selector, str),
    ]
    components = [fmt(obj) for obj, fmt in steps if obj is not None]
    return " → ".join(components) if components else "(none)"


def _format_per_model_preprocessing(model_preprocessing):
    """Format per-model preprocessing overrides for display."""
    parts = []
    for model_name, overrides in sorted(model_preprocessing.items()):
        comps = []
        for key in ('imputer', 'scaler', 'feature_selector'):
            if key in overrides:
                val = overrides[key]
                comps.append(f"{key}={type(val).__name__ if val is not None else 'None'}")
        parts.append(f"{model_name}: {', '.join(comps)}")
    return " | ".join(parts)


def _format_data_str(y_train, y_test, X_train):
    """Format data summary into a display string."""
    n_train = len(y_train)
    n_test = len(y_test)
    n_features = X_train.shape[1]
    n_events = int(y_train['event'].sum())
    epv = n_events / n_features if n_features > 0 else float('inf')
    test_pct = round(n_test / (n_train + n_test) * 100)
    return (
        f"{n_train} train / {n_test} test ({test_pct}%), "
        f"{n_features} features, {n_events} events (EPV={epv:.1f})"
    )


def _format_cv_str(cv_strategy, cv_folds, cv_repeats, scoring, cv_ci):
    """Format CV configuration into a display string."""
    cv_name = (cv_strategy if isinstance(cv_strategy, str)
               else type(cv_strategy).__name__)
    scoring_name = _format_scoring_name(scoring)
    cv_parts = [f"{cv_name}, k={cv_folds}"]
    if cv_repeats > 1:
        cv_parts.append(f"{cv_repeats} repeats")
    cv_parts.append(f"scoring={scoring_name}")
    cv_ci_pct = cv_ci * 100
    cv_parts.append(f"ci={cv_ci_pct:g}%")
    return ", ".join(cv_parts)


def _format_subpopulation_summary(subpopulations_test):
    """Format subpopulation summary rows for the setup box.

    Returns
    -------
    list of tuple
        List of (label, value) pairs for subpopulation rows.
    """
    if not subpopulations_test:
        return []
    import pandas as _pd
    rows = []
    for gvar, labels in subpopulations_test.items():
        unique, counts = np.unique(
            labels[~_pd.isna(labels)], return_counts=True
        )
        parts = [f"{u}={c}" for u, c in zip(unique, counts)]
        rows.append(("Subpopulations:", f"{gvar}: {', '.join(parts)}"))
    return rows


# ---------------------------------------------------------------------------
# tune() box rows
# ---------------------------------------------------------------------------

def _format_trials_display(n_trials, models_list, n_features, display_list=None):
    """Format the trials row for the tuning summary box."""
    if n_trials == 'auto':
        if len(models_list) == 1:
            resolved = auto_n_trials(models_list[0], n_features)
            label = display_list[0] if display_list else models_list[0]
            return f"auto (→{resolved} for {label})"
        return "auto (per-model)"
    if isinstance(n_trials, dict):
        parts = []
        for i, m in enumerate(models_list):
            val = n_trials.get(m, 'auto')
            label = display_list[i] if display_list else m
            if val == 'auto':
                resolved = auto_n_trials(m, n_features)
                parts.append(f"{label}={resolved}")
            else:
                parts.append(f"{label}={val}")
        return f"per-model ({', '.join(parts)})"
    return str(n_trials)


def _format_scoring_display(scoring, learner_scoring):
    """Format the scoring row for the tuning summary box."""
    effective_scoring = scoring if scoring is not None else learner_scoring
    scoring_name = _format_scoring_name(effective_scoring)
    if scoring is not None and scoring != learner_scoring:
        return f"{scoring_name} (setup: {_format_scoring_name(learner_scoring)})"
    return scoring_name


def _format_search_display(search_spaces):
    """Format the search spaces row for the tuning summary box."""
    if search_spaces is None:
        return "default"
    if isinstance(search_spaces, dict) and any(
        isinstance(v, dict) for v in search_spaces.values()
    ):
        custom_models = [k for k, v in search_spaces.items()
                         if isinstance(v, dict) and v]
        return f"custom ({', '.join(custom_models)})"
    return "custom"


def _format_storage_display(storage, experiment_name, state_experiment_name):
    """Format the storage row for the tuning summary box."""
    if storage is not None:
        return storage
    eff_exp = experiment_name if experiment_name is not None else state_experiment_name
    if eff_exp is not None:
        return f"./{(DEFAULT_STORAGE_BASE / eff_exp).as_posix()}/"
    return "(auto-generated)"


# ---------------------------------------------------------------------------
# evaluate() box rows
# ---------------------------------------------------------------------------

def _format_eval_times_row(eval_times, eval_times_was_auto):
    """Format the 'Eval times' row for the evaluate summary box."""
    if eval_times is None:
        # Defensive — evaluate() pre-resolves, so this should not occur.
        return "auto"
    if eval_times_was_auto:
        return _format_auto_eval_times(eval_times)
    n_pts = len(eval_times) if hasattr(eval_times, '__len__') else '?'
    return f"user-specified ({n_pts} points)"


def _format_importance_row(ctx: EvalContext) -> str:
    """Format the 'Importance' row for the evaluate summary box."""
    n = ctx.n_importance_repeats
    if isinstance(n, dict):
        parts = [f"n_repeats=per-model ({len(n)} entries)"]
    elif n == "auto":
        parts = [f"n_repeats={DEFAULT_N_IMPORTANCE_REPEATS}"]
    else:
        parts = [f"n_repeats={n}"]
    if ctx.include_shap:
        parts.append(
            f"SHAP=on (bg={ctx.shap_max_background}, explain={ctx.shap_max_explain})"
        )
    return ", ".join(parts)


def _format_bootstrap_row(ctx: EvalContext, n_samples: int) -> str:
    """Format the 'Bootstrap' row for the evaluate summary box."""
    if not ctx.bootstrap:
        return "off"
    pct = f"{ctx.bootstrap_ci * 100:g}"
    if ctx.n_bootstrap == "auto":
        resolved_n = auto_n_bootstrap(n_samples)
        return f"n=auto (→{resolved_n}), CI={pct}%"
    return f"n={ctx.n_bootstrap}, CI={pct}%"
