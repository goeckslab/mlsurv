"""Module-level helpers shared by the evaluate/bootstrap/interactions mixins.

Lives outside the mixin classes so all three sibling mixins
(``CoreEvaluationMixin``, ``BootstrapMixin``, ``InteractionsMixin``) can
import from here without violating the cross-mixin rule that sibling
mixins must not import each other.
"""

from __future__ import annotations

import warnings

import numpy as np

from .._shared import try_verbose_print
from .._warnings import (
    InternalWarning,
    MethodologicalWarning,
    SuggestionWarning,
    _warn_eval_times_failure,
)
from ..evaluation.metrics import (
    HIGH_CENSORING_THRESHOLD,
    check_ipcw_support,
    compute_safe_ipcw_tau,
)
from ..results.containers import _get_model_names
from ..logging_utils import get_logger

logger = get_logger("learner")


_SUBPOP_LABELS_UNSET = object()
_IPCW_TAU_AUTO = "auto"
_IPCW_GHAT_SOURCES = ("train", "test")


_INHERITED_GRID_UPPER_EVENT_FRAC = 0.10


def _inherited_grid_risky(grid, y) -> bool:
    """Whether the inherited eval_times grid is misaligned with custom y.

    Returns True (warn) when more than 10% of validation *events* lie above
    ``grid.max()`` — the dominant risk for IPCW-weighted metrics, which
    administratively censor observations beyond the grid.

    A grid that starts above the earliest validation time is not flagged:
    auto-grids typically span event-time quantiles, so grid.min() exceeds
    y.time.min() on the common-case correct usage path.

    When ``y`` is None (X-only path) or has no usable time field, returns True
    so the original warning still fires — we can't assess risk without times.
    """
    if y is None:
        return True
    try:
        times = np.asarray(y['time'], dtype=float)
        events = np.asarray(y['event']).astype(bool)
    except (KeyError, ValueError, IndexError, TypeError):
        return True
    grid_arr = np.asarray(grid, dtype=float)
    if grid_arr.size == 0 or times.size == 0:
        return True
    event_times = times[events]
    if event_times.size == 0:
        return False
    upper_tail_frac = float(np.mean(event_times > grid_arr.max()))
    return upper_tail_frac > _INHERITED_GRID_UPPER_EVENT_FRAC


def resolve_state_eval_times(
    state,
    eval_times,
    *,
    custom_data: bool = False,
    y_for_risk=None,
    persist: bool = True,
    stacklevel: int = 4,
):
    """Canonical 3-tier ``eval_times`` resolution shared by every consumer.

    Resolution order: ``"auto"`` string (clear inherited, defer to auto) >
    explicit array > persisted ``state.user_eval_times`` > ``None`` (defer to
    the downstream data-driven auto-compute). This is the single source of
    truth so the point grid and the bootstrap CI band land on the same
    ``eval_time`` set by construction.

    Parameters
    ----------
    state : LearnerState
        Learner state holding ``user_eval_times``.
    eval_times : array-like, "auto", or None
        The caller's ``eval_times`` argument.
    custom_data : bool, default False
        Whether the caller passed custom X/y instead of the stored test
        cohort. Only then is an inherited grid checked for misalignment.
    y_for_risk : SurvivalArray, optional
        The custom ``y`` used to assess inherited-grid risk.
    persist : bool, default True
        When True, write the resolution back to ``state.user_eval_times``
        (``"auto"`` clears it, an explicit array stores it). When False,
        perform no state writes — used when resolving calibration horizons,
        which are conceptually separate from the eval_times grid.
    stacklevel : int, default 4
        Stack level for the inherited-grid ``SuggestionWarning``.

    Returns
    -------
    np.ndarray or None
        The resolved 1-D grid, or ``None`` meaning "no explicit grid — let the
        downstream compute one from the data."
    """
    resolved = eval_times
    if isinstance(resolved, str) and resolved == "auto":
        if persist:
            state.user_eval_times = None
        return None
    if resolved is None and state.user_eval_times is not None:
        resolved = state.user_eval_times
        if custom_data and _inherited_grid_risky(resolved, y_for_risk):
            warnings.warn(
                "Using inherited eval_times from a prior evaluate()/run() "
                "call, but custom test data was provided. The inherited "
                "eval_times may not be appropriate for this data. Pass "
                "eval_times explicitly or eval_times='auto' to suppress "
                "this warning.",
                SuggestionWarning,
                stacklevel=stacklevel,
            )
    if resolved is not None:
        eval_times_arr = _validate_eval_times_array(resolved)
        if persist:
            state.user_eval_times = eval_times_arr
        return eval_times_arr
    return None


def _validate_eval_times_array(eval_times, *, param_name: str = "eval_times"):
    """Validate and coerce an eval_times array-like to a 1-D float ndarray.

    The single canonical validation shared by ``resolve_state_eval_times`` and
    ``add_eval_times`` so a grid is checked the same way everywhere: scalar →
    ``TypeError`` (with a ``[t]`` hint), non-1-D → ``ValueError``, and any
    NaN/Inf/non-positive timepoint → ``ValueError``. We reject these loudly
    rather than letting them slip through to ``clip_eval_times``, which drops
    NaN silently (NaN comparisons are False) and never names the cause. Empty
    arrays pass (vacuously finite/positive) so consumers keep their own size
    checks.

    Parameters
    ----------
    eval_times : array-like
        The timepoints to validate.
    param_name : str, default "eval_times"
        Name used in error messages (so ``add_eval_times`` can say ``times``).

    Returns
    -------
    np.ndarray
        The validated 1-D float array.
    """
    eval_times_arr = np.asarray(eval_times, dtype=float)
    if eval_times_arr.ndim == 0:
        raise TypeError(
            f"{param_name} must be an array-like of timepoints, "
            f"got scalar {eval_times!r}. Use {param_name}=[{eval_times}]."
        )
    if eval_times_arr.ndim != 1:
        raise ValueError(
            f"{param_name} must be 1-dimensional, "
            f"got shape {eval_times_arr.shape}."
        )
    if eval_times_arr.size and (
        not np.all(np.isfinite(eval_times_arr))
        or np.any(eval_times_arr <= 0)
    ):
        raise ValueError(
            f"{param_name} must be positive and finite; got {eval_times!r}."
        )
    return eval_times_arr


def _censoring_window(y_train):
    """Return the censoring-stable window ``[t_min, t_max)`` from ``y_train``.

    Matches :func:`mlsurv.utils.clip_eval_times` semantics: IPCW-weighted
    metrics (Brier, IBS, AUC(t)) are estimable only for times in
    ``[t_min, t_max)`` where ``t_min``/``t_max`` are the min/max training event
    times (strict upper — sksurv requires ``time < t_max``). Returns
    ``(None, None)`` when ``y_train`` is None or degenerate (all training times
    near-identical), so callers skip the window check but still register the
    horizon; the degenerate case raises later in ``clip_eval_times``.

    Parameters
    ----------
    y_train : structured SurvivalArray or None
        The training survival outcomes.

    Returns
    -------
    tuple
        ``(t_min, t_max)`` floats, or ``(None, None)`` when unavailable.
    """
    if y_train is None:
        return None, None
    try:
        train_times = np.asarray(y_train['time'], dtype=float)
    except (KeyError, ValueError, IndexError, TypeError):
        return None, None
    if train_times.size == 0:
        return None, None
    t_min = float(train_times.min())
    t_max = float(train_times.max())
    if np.isclose(t_min, t_max):
        return None, None
    return t_min, t_max


def _resolve_eval_model_names(learner, model) -> list:
    """Return the names of models ``evaluate(models=...)`` will iterate.

    Mirrors the resolution that ``_resolve_model_input`` performs internally,
    surfacing the names so callers can act on the exact model set before
    kicking off ``evaluate()``/``bootstrap()`` — ``validate()`` uses it to
    decide per-model recalibration eligibility, and the overwrite guard uses
    it to warn only about the models a call will actually write.
    """
    if isinstance(model, dict):
        return list(model.keys())
    if isinstance(model, (list, tuple)):
        # Share _resolve_model_input's element validation so both resolvers
        # report the same model set. Duplicate names collapse to one entry
        # (the resolved dict keys by name), matching _resolve_model_input.
        names = [
            learner._resolve_model_list_element(element).name
            for element in model
        ]
        return list(dict.fromkeys(names))
    if model is None:
        if not learner._state.models:
            return []
        from .._shared import cv_sorted_model_keys
        return cv_sorted_model_keys(
            learner._state.models, model_keys=list(learner._state.models.keys()),
        )
    if isinstance(model, str):
        mr = learner._lookup_model_by_name(model)
        return [mr.name]
    name = getattr(model, 'name', None)
    return [name] if name else []


def _resolve_ipcw_g_source(source, y_train, y_test):
    """Pick which cohort's outcomes feed the Uno IPCW G_hat estimator.

    Training G_hat is correct only under exchangeability between
    train and test cohorts (internal evaluation). For external validation
    (`validate()`), the validation cohort's own censoring distribution is the
    statistically correct choice — Uno's `1/G(T)^2` weight is consistent for
    the population whose pairs are evaluated. Using `y_test` as the G_hat
    fitting set introduces no leakage (G uses censoring times only).

    Parameters
    ----------
    source : {'train', 'test'}
        ``'train'`` returns ``y_train`` (default for `evaluate()`/`bootstrap()`).
        ``'test'`` returns ``y_test`` (default for `validate()`).
    y_train, y_test : SurvivalArray
        Candidate cohorts.

    Returns
    -------
    SurvivalArray
        The resolved cohort to pass downstream as the IPCW ``y_train`` arg.
    """
    if source not in _IPCW_GHAT_SOURCES:
        raise ValueError(
            f"ipcw_g_hat_source must be one of {_IPCW_GHAT_SOURCES}, "
            f"got {source!r}."
        )
    return y_train if source == "train" else y_test


def _emit_ipcw_truncation_warning(
    state,
    cohort_name,
    y_train,
    y_test,
    *,
    tau=None,
    stacklevel=3,
):
    """Emit a once-per-cohort IPCW truncation warning.

    Surfaces the training-range cap on Uno C / AUC(t) / IBS at the moment
    a cohort's metrics are produced (top of evaluate/bootstrap/validate).
    At most one warning is raised per ``cohort_name`` per session; the
    dedup state lives on ``state.ipcw_truncation_warned``.

    Parameters
    ----------
    state : LearnerState
        The learner's state — its ``ipcw_truncation_warned`` dict tracks
        which cohorts have already been warned in this session.
    cohort_name : str
        Cohort identifier (``"main"`` for evaluate/bootstrap on the test
        split; the validation cohort name for ``validate()``).
    y_train, y_test : SurvivalArray or None
        Training and evaluation outcomes. ``None`` short-circuits to a
        no-op return.
    tau : float or None, optional
        Resolved IPCW tau. When supplied (and finite), the warning gains
        a ``Using tau=<value> for IPCW.`` clause so truncation behavior
        and tau are reported in a single consolidated message.
    stacklevel : int, default 3
        ``stacklevel`` passed to :func:`warnings.warn`.

    Returns
    -------
    tuple of (t_max, n_beyond)
        ``t_max`` is ``y_train['time'].max()`` (or ``None`` if no
        truncation applies); ``n_beyond`` is the count of test obs whose
        time exceeds ``t_max`` (``0`` when no truncation).
    """
    if y_train is None or y_test is None:
        return None, 0
    t_max = float(y_train['time'].max())
    n_beyond = int((y_test['time'] > t_max).sum())
    if n_beyond == 0:
        return None, 0
    if not state.ipcw_truncation_warned.get(cohort_name):
        if tau is not None and np.isfinite(tau):
            tau_clause = (
                f" Using tau={float(tau):.1f} for IPCW (override with "
                f"ipcw_tau=<float> or disable with ipcw_tau=None)."
            )
        else:
            tau_clause = ""
        warnings.warn(
            f"IPCW-weighted metrics (Uno C, AUC(t), IBS) for cohort "
            f"{cohort_name!r} are computed on y_test truncated at "
            f"t_max={t_max:.1f}; {n_beyond} observation(s) beyond the "
            f"training range are administratively censored. Harrell's C, "
            f"calibration plots, and survival predictions are unaffected."
            f"{tau_clause} See docs/evaluation.md#training-range-cap.",
            MethodologicalWarning,
            stacklevel=stacklevel,
        )
        state.ipcw_truncation_warned[cohort_name] = True
    return t_max, n_beyond


def _emit_heavy_censoring_warning(
    state,
    cohort_name,
    y_test,
    *,
    threshold=HIGH_CENSORING_THRESHOLD,
    stacklevel=3,
):
    """Emit a once-per-cohort warning when the eval cohort is heavily censored.

    Harrell's C-index (the headline ``c_index_test``) is upward-biased under
    heavy censoring; this points the user to ``c_index_ipcw_test`` (Uno's IPCW
    C). Censoring is measured on ``y_test`` — the cohort the headline metric is
    computed on — unlike the report's high-censoring flag, which pools
    train+test as a general dataset caveat. At most one warning is raised per
    ``cohort_name`` per session; the dedup state lives on
    ``state.heavy_censoring_warned``.

    Parameters
    ----------
    state : LearnerState
        The learner's state — its ``heavy_censoring_warned`` dict tracks which
        cohorts have already been warned in this session.
    cohort_name : str
        Cohort identifier (``"main"`` for evaluate on the test split).
    y_test : SurvivalArray or None
        Evaluation outcomes. ``None`` short-circuits to a no-op return.
    threshold : float, default ``HIGH_CENSORING_THRESHOLD``
        Censoring fraction above which the warning fires (strict ``>``).
    stacklevel : int, default 3
        ``stacklevel`` passed to :func:`warnings.warn`.

    Returns
    -------
    float or None
        The censoring fraction when it is above ``threshold`` (whether or not
        the warning text fired this call, since the per-cohort dedup may
        suppress a repeat), else ``None``.
    """
    if y_test is None:
        return None
    # The SurvivalArray contract guarantees ``event`` is a boolean field
    # (validated at load), so ``mean()`` gives the event rate directly — no
    # coercion, which would otherwise silently mask a malformed cohort.
    events = np.asarray(y_test['event'])
    if len(events) == 0:
        return None
    censoring = 1.0 - float(events.mean())
    if censoring <= threshold:
        return None
    if not state.heavy_censoring_warned.get(cohort_name):
        warnings.warn(
            f"Evaluation cohort {cohort_name!r} is {censoring:.1%} censored "
            f"(> {threshold:.0%}). Harrell's concordance (c_index_test) is "
            f"upward-biased under heavy censoring; prefer Uno's IPCW "
            f"concordance (c_index_ipcw_test). See docs/evaluation.md.",
            MethodologicalWarning,
            stacklevel=stacklevel,
        )
        state.heavy_censoring_warned[cohort_name] = True
    return censoring


def _record_ipcw_support_flag(state, cohort_name, flag, *, is_test_set):
    """Store one cohort's IPCW support flag on the learner state.

    ``flag`` is ``None`` when the cohort raised no violation, which clears any
    stale flag from an earlier call on the same cohort. ``is_test_set`` also
    updates ``state.ipcw_support_flag``, the test-set flag the evaluation
    report reads; validate() passes ``False`` so an external cohort never
    overwrites it.
    """
    if flag is None:
        state.ipcw_support_flags.pop(cohort_name, None)
    else:
        state.ipcw_support_flags[cohort_name] = flag
    if is_test_set:
        state.ipcw_support_flag = flag


def _resolve_ipcw_tau(
    y_train,
    y_test,
    user_tau,
    *,
    metrics=None,
    context="evaluate",
    min_ipcw_support=10,
    _warn_stacklevel=3,
):
    """Resolve the user-provided ``ipcw_tau`` to a concrete float/None and
    emit a :class:`MethodologicalWarning` on support violations.

    Parameters
    ----------
    y_train, y_test : SurvivalArray
        G_hat-fitting cohort outcomes (``y_train``) and test outcomes.
    user_tau : float, None, or 'auto'
        ``'auto'`` → compute safe default via
        :func:`compute_safe_ipcw_tau` and warn on violation.
        ``None`` → preserve legacy unbounded sksurv default, still warn.
        float → use provided value, warn only if support check fires.
    metrics : list or None
        When provided, skip the warning if ``'ipcw'`` is not among the
        requested metrics (the user doesn't care about IPCW stability).
    context : str
        ``'evaluate'`` or ``'bootstrap'`` — used in the warning message.
    min_ipcw_support : int, default 10
        Minimum events and censored the G_hat-fitting cohort needs;
        forwarded to :func:`check_ipcw_support`.

    Returns
    -------
    tuple of (tau, flag)
        ``tau`` is the resolved float or None (None means "pass through
        as-is to sksurv"). ``flag`` is the support-violation dict
        (or None if no violation fires). May emit up to two
        ``MethodologicalWarning``s per call — one for numerical
        instability / overhang, one for cohort composition (G_hat not
        estimable) — since these are distinct concerns.
    """
    if y_train is None or y_test is None:
        return (user_tau if user_tau != _IPCW_TAU_AUTO else None), None

    ipcw_requested = metrics is None or 'ipcw' in set(metrics)

    if user_tau == _IPCW_TAU_AUTO:
        resolved = compute_safe_ipcw_tau(y_train, y_test)
    else:
        resolved = user_tau

    flag = check_ipcw_support(
        y_train, y_test, tau=resolved, min_ipcw_support=min_ipcw_support,
    )
    if flag is not None:
        flag["tau_used"] = resolved
        flag["tau_mode"] = "auto" if user_tau == _IPCW_TAU_AUTO else ("user" if user_tau is not None else "legacy")

    if flag is not None and ipcw_requested:
        n_beyond = flag["n_beyond"]
        min_g = flag["min_g_hat"]
        t_max = flag["t_max_train"]
        n_test = int(len(y_test))
        # Reserve the "may be unstable" wording for material support
        # violations: real numerical instability OR a non-trivial overhang
        # (>1% of test cohort). Trivial overhangs are already covered by
        # the consolidated truncation notice from
        # ``_emit_ipcw_truncation_warning``.
        g_unstable = np.isfinite(min_g) and min_g < 0.05
        overhang_material = n_test > 0 and (n_beyond / n_test) > 0.01
        if g_unstable or overhang_material:
            tau_str = f"{resolved:.1f}" if resolved is not None else "None (unbounded)"
            parts = []
            if n_beyond > 0:
                parts.append(f"{n_beyond} test obs extend beyond training t_max={t_max:.1f}")
            if g_unstable:
                parts.append(f"min(G_hat)={min_g:.3f} on [0, tau]")
            reason = "; ".join(parts) if parts else "support violation"
            warnings.warn(
                f"Uno's IPCW concordance may be unstable on this {context} "
                f"cohort: {reason}. Using tau={tau_str}. Pass ipcw_tau=<float> "
                f"to override or ipcw_tau=None to disable truncation (legacy).",
                MethodologicalWarning,
                stacklevel=_warn_stacklevel,
            )

        # Distinct concern: the G_hat-fitting cohort is too thin (in events or
        # censored) to estimate the censoring distribution. Unlike instability,
        # this makes IPCW silently reduce to its unweighted form. Use
        # ``is not None`` because a count of 0 is a real (the worst) trigger.
        n_ev = flag.get("n_g_events")
        n_cen = flag.get("n_g_censored")
        floor = flag.get("min_ipcw_support", min_ipcw_support)
        if n_ev is not None and n_cen is not None and min(n_ev, n_cen) < floor:
            thin, count = ("censored", n_cen) if n_cen <= n_ev else ("events", n_ev)
            warnings.warn(
                f"This {context} cohort has only {count} {thin} observation(s) "
                f"-- too few to estimate the censoring distribution G_hat "
                f"(need >= {floor} of each). IPCW weighting (Uno's C-index, "
                f"AUC(t), Brier, IBS) is inactive/unreliable and reduces toward "
                f"the unweighted forms. Verify this reflects true follow-up and "
                f"not dropped or missing survivors.",
                MethodologicalWarning,
                stacklevel=_warn_stacklevel,
            )

    return resolved, flag


def _resolve_subpopulation_labels(subpopulations_test, *, bootstrap=True,
                                  y_test=None,
                                  override=_SUBPOP_LABELS_UNSET,
                                  _warn_stacklevel=3):
    """Resolve subpopulation labels for bootstrap (first variable only).

    Parameters
    ----------
    subpopulations_test : dict[str, array-like] or None
        Cohort-aligned subpopulation labels keyed by variable name.
    bootstrap : bool, default True
        If False, returns None (subpop labels only matter when bootstrapping).
    y_test : array-like, optional
        Evaluation-scope outcomes. Used to detect stale training-scope labels
        whose length disagrees with the cohort being evaluated.
    override : object, optional
        Caller-supplied labels for the current evaluation cohort. ``None``
        means "skip bootstrap subpop CIs entirely"; an array replaces the
        dict lookup. Defaults to the module sentinel (use the dict).
    _warn_stacklevel : int, default 3
        ``stacklevel`` used for warnings emitted here. Each ``→`` below is
        one frame (one ``stacklevel`` increment above the ``warnings.warn``
        call). Default 3 is correct when the resolver is called directly
        from an ``EvaluationMixin`` public method
        (``warn → _resolve_… → public method → user`` = 3 arrows). Deeper
        chains such as ``warn → _resolve_… → evaluate → validate → user``
        must pass 4 so ``warnings.filterwarnings(module="user_code")``
        filters still match. Any new call site must count arrows from the
        ``warn`` call to the user and pass the matching value; the two
        current chains are pinned by
        ``tests/test_phase3_learner.py::TestValidateSubpopulationsSetupBootstrap``.

    Returns
    -------
    array-like or None
        Labels aligned with the evaluation cohort (length matches ``y_test``),
        or ``None`` when bootstrap subpop CIs should be skipped.
    """
    if not bootstrap:
        return None
    if override is not _SUBPOP_LABELS_UNSET:
        labels = override
    elif subpopulations_test:
        first_key = next(iter(subpopulations_test))
        labels = subpopulations_test[first_key]
        if len(subpopulations_test) > 1:
            other_keys = list(subpopulations_test.keys())[1:]
            warnings.warn(
                f"Bootstrap subpopulation CIs are computed only for the first "
                f"subpopulation variable ('{first_key}'). Other variables "
                f"({', '.join(other_keys)}) will have point estimates only.",
                SuggestionWarning,
                stacklevel=_warn_stacklevel,
            )
    else:
        return None
    if labels is None:
        return None
    # Guard against stale training-scope labels leaking into a different
    # cohort. Without this, _compute_subpopulation_iteration_metrics would
    # index subpopulation_labels[idx] out of range. For the validate()
    # path the override is pre-validated in _resolve_validate_subpopulation_override
    # (raises ValueError); this net catches the direct evaluate(X=..., y=...)
    # path and any other callers. Callers pass _warn_stacklevel matching
    # their depth so the warning attributes to the user's frame.
    if y_test is not None and len(labels) != len(y_test):
        warnings.warn(
            f"Subpopulation labels length ({len(labels)}) does not match "
            f"evaluation data length ({len(y_test)}); skipping bootstrap "
            f"subpopulation CIs. Pass cohort-aligned labels via "
            f"validate(subpopulations=...) to enable them.",
            SuggestionWarning,
            stacklevel=_warn_stacklevel,
        )
        return None
    return labels


def _resolve_validate_subpopulation_override(subpopulations, y_val, state):
    """Turn validate()'s ``subpopulations=`` kwarg into an evaluate() override.

    Three cases:

    * User passed a non-empty dict → validate the first variable (cohort-scope
      array of length ``len(y_val)``, 1-D) and return the array. Warn if the
      dict has multiple variables (only the first gets bootstrap CIs).
    * User passed ``None`` → return ``None`` so evaluate() skips bootstrap
      subpop CIs. If setup() had configured subpopulations, warn explaining
      why they won't be used on the validation cohort.
    * User passed an empty dict → ``ValueError`` (almost certainly a typo).

    Raising ValueErrors here (rather than letting the safety net in
    ``_resolve_subpopulation_labels`` silently skip) is intentional: in
    ``validate()`` the user explicitly asked for subgroup analysis on this
    cohort, so mismatched input is an error, not a soft warning.
    """
    if subpopulations is None:
        if state.subpopulations_test:
            warnings.warn(
                f"setup() configured subpopulations "
                f"({list(state.subpopulations_test)}), but validate() was "
                f"called without cohort-aligned labels. Training-scope labels "
                f"cannot be applied to the validation cohort, so no "
                f"per-subpopulation validation metrics will be computed. "
                f"Pass validate(subpopulations={{'<name>': <array of length "
                f"{len(y_val)}>}}) to enable them (including bootstrap CIs "
                f"when bootstrap=True).",
                SuggestionWarning,
                stacklevel=3,
            )
        return None
    if not subpopulations:
        raise ValueError(
            "validate(subpopulations=...) was given an empty dict. Pass "
            "a dict of {name: cohort-aligned labels} or omit the argument."
        )
    first_key = next(iter(subpopulations))
    arr = np.asarray(subpopulations[first_key])
    if arr.ndim != 1:
        raise ValueError(
            f"subpopulations[{first_key!r}] must be 1-D, got shape {arr.shape}."
        )
    if len(arr) != len(y_val):
        raise ValueError(
            f"subpopulations[{first_key!r}] has length {len(arr)} but "
            f"y_val has length {len(y_val)}. Provide cohort-aligned labels."
        )
    if len(subpopulations) > 1:
        other = ", ".join(list(subpopulations)[1:])
        warnings.warn(
            f"Bootstrap subpopulation CIs on validation data are computed "
            f"only for the first subpopulation variable ('{first_key}'). "
            f"Other variables ({other}) will have point estimates only. "
            f"(Mirrors the single-variable limit in "
            f"_resolve_subpopulation_labels for the evaluate() path.)",
            SuggestionWarning,
            stacklevel=3,
        )
    return arr


def _compute_and_store_subpopulation_metrics(state, result, y_test, ipcw_y_train=None):
    """Compute subpopulation metrics and attach bootstrap CIs to state.

    ``ipcw_y_train`` is the IPCW G_hat fitting cohort override.
    When None, subpopulation IPCW falls back to ``state.y_train``.
    """
    try:
        from ..evaluation._subpopulation import (
            _attach_per_model_bootstrap_cis,
            compute_subpopulation_metrics,
        )
        from ..utils import auto_eval_times, union_extra_eval_times

        sg_eval_times = state.user_eval_times  # Use inherited if available
        if sg_eval_times is None and state.y_train is not None:
            try:
                sg_eval_times = auto_eval_times(
                    y_test, y_train=state.y_train, _quiet=True,
                )
            except (ValueError, TypeError) as exc:
                _warn_eval_times_failure(
                    exc, "for subpopulation metrics",
                    "AUC(t) subpopulation metrics will be unavailable.",
                    stacklevel=3,
                )

        # Union additive report horizons (add_eval_times) into the grid so a
        # re-run fills them in the per-subpopulation AUC(t) facets, mirroring
        # the cohort engine path. In-window only; out-of-window extras render
        # as honest NA. No-op when none are registered, and a no-op (returns
        # None) when sg_eval_times could not be resolved above.
        sg_eval_times = union_extra_eval_times(
            sg_eval_times, state.user_extra_eval_times, state.y_train,
        )

        sg_results = compute_subpopulation_metrics(
            result, state.subpopulations_test, y_test,
            y_train=state.y_train,
            ipcw_y_train=ipcw_y_train,
            eval_times=sg_eval_times,
            surv_probs_by_model=result._surv_probs_cache,
            dense_surv_probs_by_model=result._ibs_surv_probs_cache,
            ipcw_tau=(state.ipcw_support_flag.get('tau_used')
                      if state.ipcw_support_flag else None),
            _warn_stacklevel=4,
            # setup() already warned on the same test subpops;
            # SubpopulationResult.warnings stays populated for reports.
            _suppress_aggregate_warning=state.subpop_test_warned,
        )

        # Attach bootstrap CIs if available
        _attach_per_model_bootstrap_cis(sg_results, result.bootstrap)

    except (MemoryError, RecursionError):
        raise
    # Broad catch is intentional: compute_subpopulation_metrics calls
    # sksurv/numpy/pandas which can raise many exception types.
    # MemoryError/RecursionError are re-raised above.
    except Exception as exc:
        logger.error("Subpopulation metric update failed: %s", exc, exc_info=True)
        warnings.warn(
            f"Subpopulation metric update failed "
            f"({type(exc).__name__}: {exc}). Any prior subpopulation "
            f"results are preserved; the failed update was discarded.",
            SuggestionWarning,
            stacklevel=3,
        )
        return

    # Merge new results into existing subpopulation data so that a single-model
    # re-evaluate doesn't discard other models' data or bootstrap CIs.
    # Runs OUTSIDE the try-except so merge bugs surface loudly.
    from ..evaluation._subpopulation import _merge_subpopulation_results_dict
    state.subpopulation_results = _merge_subpopulation_results_dict(
        state.subpopulation_results, sg_results,
    )


def _print_evaluation_output(result, state, bootstrap_ci, scoring, delta_baseline=None):
    """Print evaluation metrics and subpopulation summary.

    ``delta_baseline`` (a held-out-test concordance DataFrame) is forwarded so
    each metric cell shows its change from the development metrics; ``validate()``
    sets it, plain ``evaluate()`` leaves it ``None``. We call ``print_eval_metrics``
    directly (rather than ``result.print_metrics``) to thread the baseline without
    widening the public ``EvaluationResult.print_metrics`` signature.
    """
    from mlsurv.results._containers_printing import print_eval_metrics
    print_eval_metrics(result, ci_level=bootstrap_ci, delta_baseline=delta_baseline)

    if state.subpopulation_results:
        try_verbose_print(
            _print_subpopulation_summary,
            state.subpopulation_results,
            scoring_metric=scoring or 'harrell',
            # Skip per-label "! Subpopulation 'X' has only N events"
            # lines when setup() already issued an aggregate warning for the
            # same test subpops. When False (e.g., validate/benchmark or any
            # path where setup didn't warn), keep showing the detail.
            include_warnings=not state.subpop_test_warned,
            context="Subpopulation summary",
            warn=False,
        )


def _update_kang_cache(result, state, bootstrap, y_test, auto_kang_fn):
    """Populate state.kang_comparison_result for 2+ models without bootstrap.

    Decoupled from print path so the cache is available to HTML reports
    and export_results regardless of `verbose`.
    """
    if bootstrap or result.concordance is None:
        return
    model_names = _get_model_names(result.concordance)
    if len(model_names) < 2:
        return
    try:
        kang_df = auto_kang_fn(result, y_test, model_names)
    except (ValueError, KeyError) as e:
        # User-visible: kang_comparison_result stays None so the HTML report's
        # paired-comparison subsection will omit Kang results — surface why.
        warnings.warn(
            f"Kang z-test paired comparison could not be computed for "
            f"models {model_names}: {e}. The cached "
            f"`learner.kang_comparison_result` stays None and the HTML "
            f"report's paired-comparison subsection will omit Kang results.",
            SuggestionWarning,
            stacklevel=3,
        )
        return
    if kang_df is not None:
        state.kang_comparison_result = kang_df


def _is_nan(val) -> bool:
    """Return True if *val* is NaN, False for non-float types."""
    try:
        return val != val  # NaN != NaN
    except (TypeError, ValueError):
        return False


def _format_auto_eval_times(eval_times) -> str:
    """Render an auto-resolved eval_times array for the verbose box."""
    arr = np.asarray(eval_times)
    if arr.size == 0:
        return "auto"
    if np.all(arr == np.round(arr)):
        def fmt(x):
            return f"{int(round(x))}"
    else:
        def fmt(x):
            return f"{x:.1f}"
    if arr.size <= 5:
        return f"auto → [{', '.join(fmt(t) for t in arr)}]"
    return f"auto → {arr.size} times in [{fmt(arr.min())}, {fmt(arr.max())}]"


def _print_subpopulation_summary(
    subpopulation_results, scoring_metric='harrell', *, include_warnings=True,
) -> None:
    """Print a concise per-subpopulation metrics table to stdout.

    When ``include_warnings=False``, the per-label
    ``! Subpopulation 'X' has only N events`` lines are omitted:
    setup() already issued a single aggregate warning, so the detail would
    be a duplicate. ``SubpopulationResult.warnings`` is still populated and
    consumed by report sections regardless of this flag.
    """
    from ..viz._palettes import get_model_display_name

    for group_var, sg_result in subpopulation_results.items():
        # Show scoring metric if available, else harrell
        metrics = sg_result.metrics
        available_metrics = sorted(metrics['metric'].unique())
        display_metric = scoring_metric if scoring_metric in available_metrics else 'harrell'

        metric_label = display_metric.replace('_', ' ').title()
        print(f"\n  Subpopulation {metric_label} ({group_var}):")
        boot = sg_result.bootstrap_summary

        metric_df = metrics[metrics['metric'] == display_metric]
        for _, row in metric_df.iterrows():
            sg = str(row['subpopulation'])
            model = get_model_display_name(row['model_type'])
            val = row['value']
            if boot is not None:
                ci_mask = (
                    (boot['subpopulation'] == row['subpopulation']) &
                    (boot['model_type'] == row['model_type']) &
                    (boot['metric'] == display_metric)
                )
                ci_rows = boot[ci_mask]
                if len(ci_rows) > 0:
                    lo = ci_rows.iloc[0].get('ci_lower', float('nan'))
                    hi = ci_rows.iloc[0].get('ci_upper', float('nan'))
                    if not (_is_nan(lo) or _is_nan(hi)):
                        print(f"    {sg:>12s} | {model:<12s} | {val:.3f} ({lo:.3f}-{hi:.3f})")
                        continue
            if not _is_nan(val):
                print(f"    {sg:>12s} | {model:<12s} | {val:.3f}")
            else:
                print(f"    {sg:>12s} | {model:<12s} | N/A")

        # Per-label "! Subpopulation 'X' has only N events" lines — printed
        # when include_warnings=True (e.g., validate/benchmark paths where
        # setup() did not already warn). Suppressed when setup() already
        # issued the aggregate SmallSubpopulationWarning.
        if include_warnings:
            for w in sg_result.warnings:
                print(f"    ! {w}")


def _align_columns(X, train_cols, *, context="Data", stacklevel=3):
    """Validate and align DataFrame columns to training column order.

    Parameters
    ----------
    X : DataFrame
        Data to align.
    train_cols : list[str]
        Expected column names in the correct order.
    context : str
        Label for error/warning messages (e.g. "Validation data").
    stacklevel : int
        Stack depth for warnings.warn.

    Returns
    -------
    DataFrame
        ``X`` with extra columns dropped and remaining columns reordered.

    Raises
    ------
    ValueError
        If ``X`` is missing one or more training columns.
    """
    val_cols = set(X.columns)
    train_set = set(train_cols)

    missing = train_set - val_cols
    extra = val_cols - train_set

    if missing:
        raise ValueError(
            f"{context} is missing {len(missing)} training column(s): "
            f"{sorted(missing)}"
        )
    if extra:
        warnings.warn(
            f"{context} has {len(extra)} extra column(s) not in training "
            f"data (dropped): {sorted(extra)}",
            SuggestionWarning,
            stacklevel=stacklevel,
        )
        X = X.drop(columns=sorted(extra))
    if list(X.columns) != train_cols:
        warnings.warn(
            f"{context} columns were reordered to match training data. "
            "Pipeline transformers (e.g. KNNImputer, scalers) are "
            "sensitive to column order.",
            InternalWarning,
            stacklevel=stacklevel,
        )
        X = X[train_cols]
    return X
