"""Custom warning categories for mlsurv's tiered warning system.

Tier 1 — MethodologicalWarning: always fires at least once per concern (methodological risk)
Tier 2 — SuggestionWarning: fires once per session (actionable suggestion)
Tier 3 — InternalWarning: suppressed by default (internal fallbacks)

IterativeWarning — mixin for warnings that repeat across bootstrap/tuning/CV
iterations. Entry points filter with ``filterwarnings("once", category=IterativeWarning)``.
"""

import logging
import warnings


logger = logging.getLogger(__name__)

# Stdlib warning formatter captured once at import. Used as the fallback for
# warnings that aren't mlsurv's own — third-party/upstream warnings keep their
# default file/line format, which is useful for diagnosing real upstream issues.
_ORIGINAL_FORMATWARNING = warnings.formatwarning


# ---------------------------------------------------------------------------
# Base severity tiers
# ---------------------------------------------------------------------------

class MethodologicalWarning(UserWarning):
    """Indicates a real risk to result validity (e.g., low EPV, data leakage).

    Always fires at least once per concern. Deduplication across lifecycle
    phases (e.g., setup vs tuning) prevents redundant warnings for the
    same underlying issue.
    """


class SuggestionWarning(UserWarning):
    """Actionable suggestion that is helpful but not critical.

    Fires once per unique message by default.
    """


class ExportSplitWarning(SuggestionWarning):
    """An export's output is not all where the caller asked for it.

    Covers tables written as CSV beside the workbook because they exceed
    Excel's worksheet limits, and files an earlier export left in the same
    directory. Fires every time rather than once per session: the message
    describes what a particular run put where, so suppressing repeats would
    hide the second export's layout behind the first one's.
    """


class InternalWarning(UserWarning):
    """Internal fallback or diagnostic detail.

    Suppressed by default. Enable with
    :func:`enable_internal_warnings` or::

        warnings.simplefilter("default", InternalWarning)
    """


# ---------------------------------------------------------------------------
# Iterative mixin + combined subclasses
# ---------------------------------------------------------------------------

class IterativeWarning(UserWarning):
    """Base mixin for warnings that repeat across bootstrap/tuning/CV iterations.

    Entry points filter with ``filterwarnings("once", category=IterativeWarning)``
    to deduplicate per unique message text.
    """


class ClampWarning(MethodologicalWarning, IterativeWarning):
    """Linear predictors clamped to prevent overflow.

    Custom category so users can filter with
    ``warnings.filterwarnings("once", category=ClampWarning)``.
    """


class FitFailureWarning(MethodologicalWarning, IterativeWarning):
    """CoxPH/CoxNet instability or fallback during model fitting.

    Fires per-fold in CV and tuning contexts.
    """


class MetricFailureWarning(MethodologicalWarning, IterativeWarning):
    """Metric computation failure during bootstrap (NaN, IPCW edge cases).

    Fires once per coalesced ``(metric, exc_type)`` failure group across
    affected models. Every failed iteration represents user-visible data
    that was dropped from the bootstrap aggregate, so the warning fires
    unconditionally; ``IterativeWarning`` dedup keeps repeated identical
    summaries to a single emission per session.
    """


class SmallSubpopulationWarning(MethodologicalWarning, IterativeWarning):
    """Subpopulation has fewer events than the minimum threshold.

    Fires at ``setup()`` for early detection, and per-subpopulation
    per-bootstrap-iteration during evaluation.
    """


class RegressionModeWarning(MethodologicalWarning, IterativeWarning):
    """FSSVM regression-mode Breslow survival curves may be invalid.

    ``rank_ratio < 1.0`` makes FSSVM regress survival time rather than rank
    risk, so its Breslow-based ``S(t|x)`` and Brier scores are approximate.
    The caveat is fixed at fit time (``rank_ratio`` doesn't change), so it is
    invariant per fitted model. ``IterativeWarning`` dedup keeps it to one
    emission per session across CV/bootstrap/evaluation iterations.
    """


# Lookup dict for restoring warning categories from serialized fold results.
# Worker processes (CV folds, recalibration OOF collection) serialize warning
# categories as strings (for picklability); this maps them back to the
# original class on the main process side.
_WARNING_CLASSES = {cls.__name__: cls for cls in (
    MethodologicalWarning, SuggestionWarning, InternalWarning,
    ClampWarning, FitFailureWarning, MetricFailureWarning, SmallSubpopulationWarning,
    RegressionModeWarning, ExportSplitWarning,
)}

# Base tiers that identify a warning as mlsurv's own. Every concrete mlsurv
# warning subclasses at least one of these, so an ``issubclass`` check against
# this tuple precisely selects our warnings for the clean formatter below
# without catching third-party ``UserWarning`` subclasses.
_MLSURV_WARNING_BASES = (
    MethodologicalWarning, SuggestionWarning, InternalWarning, IterativeWarning,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def enable_internal_warnings():
    """Re-enable display of :class:`InternalWarning` for debugging.

    Example::

        from mlsurv._warnings import enable_internal_warnings
        enable_internal_warnings()
    """
    warnings.simplefilter("default", InternalWarning)


def _mlsurv_formatwarning(message, category, filename, lineno, line=None):
    """Format mlsurv's own warnings as clean, one-line guidance.

    PI feedback on the quickstart vignette: the default Python format prefixes
    each warning with a file path, line number, the category name, and an echoed
    ``warnings.warn(...)`` source line, which makes a methodological or data
    caution look like a code bug. For mlsurv's own warning tiers we drop that
    scaffolding and print just ``ClassName: message``. Everything else
    (third-party/upstream warnings) keeps the stdlib format, where file/line
    still helps diagnose real upstream issues.
    """
    if issubclass(category, _MLSURV_WARNING_BASES):
        return f"{category.__name__}: {message}\n"
    return _ORIGINAL_FORMATWARNING(message, category, filename, lineno, line)


def configure_default_filters():
    """Register mlsurv's default warning filters and clean formatter.

    Called once at import time from ``mlsurv/__init__.py``. Safe to call more
    than once: ``simplefilter`` replaces any matching filter rather than
    stacking duplicates, and reassigning ``warnings.formatwarning`` just points
    it at the same function again.
    """
    warnings.simplefilter("once", IterativeWarning)
    warnings.simplefilter("once", SuggestionWarning)
    # Registered after the SuggestionWarning filter so the more specific
    # category wins: each export that splits its output must say so.
    warnings.simplefilter("always", ExportSplitWarning)
    warnings.simplefilter("ignore", InternalWarning)
    warnings.formatwarning = _mlsurv_formatwarning


def _warn_eval_times_failure(exc, site, tail_msg, *, stacklevel=2, verb="auto_eval_times"):
    """Emit the standard log+SuggestionWarning pair for an
    ``auto_eval_times`` / ``clip_eval_times`` failure during ``site``.

    The helper adds 1 to ``stacklevel`` so ``warning.filename`` resolves to
    the caller's file, not this module. Pass ``verb="clip_eval_times"`` for
    sites whose existing message begins with ``"clip_eval_times failed …"``.
    """
    msg = f"{verb} failed {site}: {exc}. {tail_msg}"
    logger.warning(msg)
    warnings.warn(msg, SuggestionWarning, stacklevel=stacklevel + 1)


def warn_missing_time_resolved_ci(metric_label, *, stacklevel):
    """Warn that a requested time-resolved CI ribbon has no bootstrap data.

    Fired only when the caller explicitly set ``show_ci=True`` on
    ``plot_metric_time`` and the bootstrap did not collect the time-resolved
    distribution the ribbon needs. ``metric_label`` is ``"AUC"`` or
    ``"Brier"``. The caller passes the exact ``stacklevel`` so the warning
    attributes to user code; this helper forwards it unchanged.
    """
    warnings.warn(
        f"No bootstrap CI data for the {metric_label}(t) ribbon. Re-run "
        f"`bootstrap(include_time_resolved=True)` to collect it; showing "
        f"point estimates only.",
        SuggestionWarning,
        stacklevel=stacklevel,
    )


def warn_missing_roc_ci(*, stacklevel):
    """Warn that a requested ROC CI band has no bootstrap predictions.

    Fired only when the caller explicitly set ``show_ci=True`` on
    ``plot_roc`` and the bootstrap did not store per-iteration predictions.
    The caller passes the exact ``stacklevel`` (forwarded unchanged).
    """
    warnings.warn(
        "No bootstrap predictions for ROC CI bands. Re-run "
        "`bootstrap(include_predictions=True)`; showing the point ROC "
        "curve only.",
        SuggestionWarning,
        stacklevel=stacklevel,
    )


def _dedup_warning_entries(entries):
    """Dedup a flat sequence of ``(category, message, filename, lineno)`` tuples.

    Returns
    -------
    dict
        Mapping of (category, message) -> (count, filename, lineno).
    """
    seen_warnings = {}
    malformed_count = 0
    for entry in entries:
        try:
            cat, msg, fname, lineno = entry
        except (ValueError, TypeError):
            malformed_count += 1
            logger.debug("Skipping malformed warning entry: %r", entry)
            continue
        # Skip known upstream warnings that can't be fixed
        if "`trapz` is deprecated" in msg:
            continue
        # CoxNet zero-coefficients warning is expected during CV folds;
        # the final model is checked separately.
        if "all coefficients are zero" in msg:
            continue
        key = (cat, msg)
        if key in seen_warnings:
            seen_warnings[key] = (seen_warnings[key][0] + 1, fname, lineno)
        else:
            seen_warnings[key] = (1, fname, lineno)

    if malformed_count:
        logger.warning(
            "Dropped %d malformed warning entries "
            "(possible joblib serialization issue)", malformed_count,
        )
    return seen_warnings


def _reemit_deduped_warnings(seen_warnings, total_iterations, scope_label="CV folds",
                             seen_messages=None):
    """Re-emit deduplicated warnings with occurrence-count suffix.

    Singletons emit verbatim; aggregated warnings get a
    ``"(N occurrences across K {scope_label})"`` suffix.

    When ``seen_messages`` is provided (a set of ``(category, message)``
    keys kept on ``LearnerState``), keys already in the set are skipped
    and new ones are added — so the same underlying warning surfaces at
    most once per learner across CV training, final-fit, and calibration.
    """
    for (cat, msg), (count, fname, lineno) in seen_warnings.items():
        if seen_messages is not None:
            if (cat, msg) in seen_messages:
                continue
            seen_messages.add((cat, msg))
        full_msg = (
            f"{msg} ({count} occurrences across {total_iterations} {scope_label})"
            if count > 1 else msg
        )
        warnings.warn_explicit(full_msg, _WARNING_CLASSES.get(cat, UserWarning), fname, lineno)
