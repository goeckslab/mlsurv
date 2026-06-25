"""Print-method delegates for result containers.

Extracted from ``containers.py`` to reduce module complexity.
Each function takes the result object as its first parameter;
the class methods in containers.py delegate to these with one-line calls.
"""

from typing import TYPE_CHECKING, Dict, Optional

import pandas as pd

from mlsurv.results._containers_helpers import (
    _count_bootstrap_iterations,
    _get_model_names,
    _get_printing_functions,
    _resolve_display,
)

if TYPE_CHECKING:
    from mlsurv.results.containers import (
        BenchmarkResult,
        BootstrapResult,
        EvaluationResult,
    )


def _dispatch_multi_bootstrap(
    printers: Dict,
    summary: pd.DataFrame,
    concordance: pd.DataFrame,
    iterations_df: pd.DataFrame,
    ci_level: float,
    precision: int,
    model_order: list,
    delta_baseline=None,
) -> None:
    """Dispatch the multi-model bootstrap printer with the standard arg shape.

    Centralizes the display-name map, the n_bootstrap derivation, and the
    keyword shape expected by
    :func:`print_multi_model_bootstrap_table`. The empty-concordance fallback
    contract is the caller's responsibility — pass
    ``pd.DataFrame(columns=['model_type'])`` when no concordance is available.

    If ``iterations_df`` is None or lacks an ``iteration`` column, ``n_bootstrap``
    is reported as 0 (no error) — see :func:`_count_bootstrap_iterations`.
    """
    from mlsurv.models import get_model_display_names
    printers['multi_bootstrap'](
        summary, concordance,
        n_bootstrap=_count_bootstrap_iterations(iterations_df),
        ci_level=ci_level,
        precision=precision,
        model_order=model_order,
        display_names=get_model_display_names(),
        delta_baseline=delta_baseline,
    )


# ---------------------------------------------------------------------------
# EvaluationResult printing
# ---------------------------------------------------------------------------


def print_eval_metrics(
    eval_result: 'EvaluationResult',
    ci_level: float = 0.95,
    precision: int = 3,
    delta_baseline=None,
) -> None:
    """Print metrics table to console.

    Prints bootstrap table if available, otherwise prints point estimates.
    When bootstrap is available, displays point estimates from the full
    test set with bootstrap confidence intervals (best practice).

    For multi-model results, prints a per-model comparison table. Pairwise
    statistical comparisons (paired bootstrap, Kang z-test) must be requested
    explicitly via ``BootstrapResult.compare()``,
    ``BootstrapResult.print_comparison()``, or
    ``mlsurv.evaluation.compare_bootstrap`` / ``compare_kang``.

    Parameters
    ----------
    eval_result : EvaluationResult
        Evaluation result to print.
    ci_level : float, default=0.95
        Confidence level for bootstrap intervals.
    precision : int, default=3
        Decimal precision for formatting.
    delta_baseline : DataFrame, optional
        Held-out-test concordance (``model_type`` + per-metric columns). When
        given, each cell is annotated with ``(Δ ...)`` = its change from the
        baseline, plus a footnote. ``validate()`` passes this to show the
        generalization gap; plain ``evaluate()`` leaves it ``None``.
    """
    printers = _get_printing_functions()
    model_names = _get_model_names(eval_result.concordance)

    if len(model_names) <= 1:
        # Single model case
        model_name = model_names[0] if model_names else None
        display_name = _resolve_display(model_name) if model_name else None
        _print_single_model(
            eval_result, printers, model_name, display_name, ci_level, precision,
            delta_baseline,
        )
    else:
        # Multi-model case: print comparison table
        # Use CV-sorted order if available
        display_order = eval_result.model_order if eval_result.model_order else model_names
        if eval_result.bootstrap is not None:
            _dispatch_multi_bootstrap(
                printers,
                eval_result.bootstrap.summary,
                eval_result.concordance,
                eval_result.bootstrap.iterations,
                ci_level, precision, display_order,
                delta_baseline,
            )
        else:
            from mlsurv.models import get_model_display_names
            # _warn_stacklevel=4: warn -> print_multi_model_evaluation_table
            #   -> print_eval_metrics -> print_metrics (containers.py delegate)
            #   -> user. stacklevel=N points to frame N up from the warn() call.
            printers['multi_evaluation'](
                eval_result.concordance,
                precision=precision,
                model_order=display_order,
                display_names=get_model_display_names(),
                _warn_stacklevel=4,
                delta_baseline=delta_baseline,
            )


def _print_single_model(
    eval_result: 'EvaluationResult',
    printers: Dict,
    model_name: Optional[str],
    display_name: Optional[str],
    ci_level: float,
    precision: int,
    delta_baseline=None,
) -> None:
    """Print metrics for a single model.

    Parameters
    ----------
    model_name : str or None
        Raw internal key for DataFrame filtering.
    display_name : str or None
        Human-readable name for table headers.
    delta_baseline : DataFrame, optional
        Held-out-test concordance passed through to annotate ``(Δ ...)`` cells.
    """
    if eval_result.bootstrap is not None:
        # Extract point estimates for this specific model (uses raw key)
        point_estimates = _extract_point_estimates(eval_result, model_name)

        # Print bootstrap table with point estimates
        n_bootstrap = _count_bootstrap_iterations(eval_result.bootstrap.iterations)
        # _warn_stacklevel=6: user -> print_metrics (containers.py delegate)
        #   -> print_eval_metrics -> _print_single_model
        #   -> print_bootstrap_table -> _filter_by_model_type -> warnings.warn
        printers['bootstrap'](
            eval_result.bootstrap.summary,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            precision=precision,
            model_name=model_name,
            point_estimates=point_estimates,
            display_name=display_name,
            _warn_stacklevel=6,
            delta_baseline=delta_baseline,
        )
    else:
        # _warn_stacklevel=6: user -> print_metrics (containers.py delegate)
        #   -> print_eval_metrics -> _print_single_model
        #   -> print_evaluation_table -> _filter_by_model_type -> warnings.warn
        printers['evaluation'](
            eval_result.concordance,
            model_name=model_name,
            precision=precision,
            display_name=display_name,
            _warn_stacklevel=6,
            delta_baseline=delta_baseline,
        )


def _extract_point_estimates(
    eval_result: 'EvaluationResult',
    model_name: Optional[str] = None,
) -> Dict[str, float]:
    """Extract point estimates from concordance DataFrame for bootstrap display.

    Parameters
    ----------
    eval_result : EvaluationResult
        Evaluation result containing concordance data.
    model_name : str, optional
        If provided, extract estimates for this specific model.
        If None, uses the first row's estimates (intended for single-model
        results; for multi-model results, pass model_name explicitly).
    """
    if eval_result.concordance is None:
        return {}

    concordance = eval_result.concordance
    if model_name is not None and 'model_type' in concordance.columns:
        concordance = concordance[concordance['model_type'] == model_name]

    if concordance.empty:
        return {}

    estimates = {}

    from mlsurv.constants import get_concordance_column, ALL_METRICS, DIAGNOSTIC_METRICS

    for metric in ALL_METRICS + DIAGNOSTIC_METRICS:
        col = get_concordance_column(metric)
        if col in concordance.columns:
            value = concordance[col].iloc[0]
            if pd.notna(value):
                estimates[metric] = float(value)

    return estimates


# ---------------------------------------------------------------------------
# BootstrapResult printing
# ---------------------------------------------------------------------------


def print_bootstrap_summary(
    boot_result: 'BootstrapResult',
    ci_level: float = 0.95,
    precision: int = 3,
    point_estimates: Optional[Dict[str, float]] = None,
) -> None:
    """Print bootstrap metrics table to console.

    Parameters
    ----------
    boot_result : BootstrapResult
        Bootstrap result to print.
    ci_level : float, default=0.95
        Confidence level used to label the intervals. This only labels the
        precomputed bounds; callers should pass the level the intervals were
        computed at (BootstrapResult.print_summary resolves this from
        ``boot_result.ci_level`` automatically).
    precision : int, default=3
        Decimal precision for formatting.
    point_estimates : dict, optional
        External point estimates that take priority over the 'point' column
        in the summary DataFrame. When neither is available, falls back to
        bootstrap mean. For best practice, use EvaluationResult.print_metrics()
        which automatically provides point estimates.
    """
    printers = _get_printing_functions()
    n_bootstrap = _count_bootstrap_iterations(boot_result.iterations)

    model_names = (
        boot_result.summary['model_type'].unique().tolist()
        if 'model_type' in boot_result.summary.columns
        else []
    )

    if len(model_names) > 1:
        # Multi-model: route to consolidated table (mirrors print_eval_metrics).
        # No concordance is available on the standalone bootstrap path; pass an
        # empty DataFrame. _resolve_point_value falls back to summary['point'],
        # which bootstrap() populates via _compute_point_estimates.
        _dispatch_multi_bootstrap(
            printers,
            boot_result.summary,
            pd.DataFrame(columns=['model_type']),
            boot_result.iterations,
            ci_level, precision, model_names,
        )
        return

    model_name = model_names[0] if model_names else None
    display_name = _resolve_display(model_name) if model_name else None
    printers['bootstrap'](
        boot_result.summary,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        precision=precision,
        model_name=model_name,
        point_estimates=point_estimates,
        display_name=display_name,
    )


def print_bootstrap_comparison(
    boot_result: 'BootstrapResult',
    metric: str = "harrell",
    ci: float = 0.95,
    p_adjust: str = "holm",
    precision: int = 3,
) -> pd.DataFrame:
    """Print and return paired model comparison table.

    Parameters
    ----------
    boot_result : BootstrapResult
        Bootstrap result containing multi-model data.
    metric : str, default="harrell"
        Metric to compare.
    ci : float, default=0.95
        Confidence level.
    p_adjust : str, default="holm"
        Multiplicity correction.
    precision : int, default=3
        Decimal precision.

    Returns
    -------
    pd.DataFrame
        Comparison results (also printed to console).
    """
    df = boot_result.compare(metric=metric, ci=ci, p_adjust=p_adjust)
    from mlsurv.results.printing import print_paired_comparison
    print_paired_comparison(df, precision=precision, ci_level=ci)
    return df


# ---------------------------------------------------------------------------
# BenchmarkResult printing
# ---------------------------------------------------------------------------


def print_bench_summary(
    bench_result: 'BenchmarkResult',
    ci_level: Optional[float] = None,
) -> None:
    """Print human-readable benchmark summary to stdout.

    ``ci_level=None`` lets ``EvaluationResult.print_metrics`` resolve the
    label from the attached bootstrap's actual level.
    """
    from mlsurv.results.printing import print_benchmark_summary
    print_benchmark_summary(bench_result, ci_level=ci_level)
