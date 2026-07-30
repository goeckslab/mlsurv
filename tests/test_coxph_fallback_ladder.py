"""CoxPH numerical-failure classification and the escalating alpha fallback.

Two behaviors are covered:

1. Which ``ValueError`` messages count as "the solver failed numerically" and
   therefore route into the fallback. The regression that motivated this: scipy
   raises ``ValueError("LAPACK reported an illegal value in 5-th argument.")``
   from ``_solve_check`` when LAPACK returns ``info < 0``, and that wording used
   to escape the fallback entirely, so CoxPH was dropped from CV and tuning on
   near-singular designs.
2. The escalating ladder itself: 0.01 -> 0.1 -> 1.0, floored at the user's
   alpha so a supplied penalty is never weakened.
"""

import pickle
import warnings
from unittest.mock import patch

import numpy as np
import pytest

from mlsurv._warnings import (
    FitFailureWarning, MethodologicalWarning, SuggestionWarning,
)
from mlsurv.models import CoxPHModel
from mlsurv.models._overflow import _CALLER_ERROR_RE, _SOLVER_FAILURE_RE
from mlsurv.models.coxph import _FALLBACK_ALPHAS, _fallback_ladder

FAILED = (True, False, False)    # optimizer failed, no model
CLEAN = (False, False, False)    # fit succeeded, no instability
WARNED = (False, True, False)    # fit succeeded but warned


def _side_effect_winning_at(rung):
    """Fail the initial fit, then every rung below `rung`, then succeed.

    Derived from _FALLBACK_ALPHAS rather than hardcoded, so inserting a rung
    keeps these tests meaningful instead of exhausting the mock and raising a
    confusing StopIteration from inside fit().
    """
    return [FAILED] * (1 + _FALLBACK_ALPHAS.index(rung)) + [CLEAN]


# ---------------------------------------------------------------------------
# 1. message classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    # scipy _solve_check, info < 0: the reported regression
    "LAPACK reported an illegal value in 5-th argument.",
    "LAPACK reported an illegal value in 1-th argument.",
    # scipy _solve_check, info > 0 (numpy LinAlgError subclasses ValueError)
    "Matrix is singular.",
    # sksurv's own guard
    "search direction contains NaN or infinite values",
    # scipy LinAlgWarning promoted to an error by warnings-as-errors
    "An ill-conditioned matrix detected: rcond=1e-20",
])
def test_solver_failures_are_classified(message):
    """Numerical solver failures route into the alpha fallback."""
    assert _CALLER_ERROR_RE.search(message) is None
    assert _SOLVER_FAILURE_RE.search(message) is not None


@pytest.mark.parametrize("message", [
    # sksurv parameter validation
    "alpha must be positive, but was -0.5",
    "alpha must be positive, but was array([-1.])",
    "Length alphas (3) must match number of features (5).",
    # sklearn parameter validation, including the nan form, which the solver
    # pattern matches on its own and only the caller check excludes
    "The 'alpha' parameter of CoxPHSurvivalAnalysis must be a float in the "
    "range [0, inf) or an array-like. Got -0.5 instead.",
    "The 'alpha' parameter of CoxPHSurvivalAnalysis must be a float in the "
    "range [0.0, inf) or an array-like. Got nan instead.",
    # sklearn check_array: no penalty fixes a NaN cell, and its message names
    # the offending input and links to the imputation docs
    "Input X contains NaN. CoxPHSurvivalAnalysis does not accept missing "
    "values encoded as NaN natively.",
    "Input contains infinity or a value too large for dtype('float64').",
])
def test_caller_errors_take_precedence_over_the_solver_pattern(message):
    """Bad input and bad arguments must reach the user with their own text."""
    assert _CALLER_ERROR_RE.search(message) is not None


@pytest.mark.parametrize("message", [
    "something unrelated",
    # word-boundary guards: these contain 'nan' only as a substring
    "maintenance window elapsed",
    "took 5 nanoseconds",
])
def test_unrelated_messages_are_not_classified(message):
    assert _SOLVER_FAILURE_RE.search(message) is None
    assert _CALLER_ERROR_RE.search(message) is None


def test_lapack_valueerror_routes_to_fallback(train_test_data):
    """The LAPACK wording reaches the fallback instead of aborting the fit.

    This is the direct regression guard: before the fix, ``_fit_and_check``
    re-raised this ValueError, which propagated out of ``fit()`` and got the
    model dropped from CV/tuning.
    """
    X_train, _, y_train, _ = train_test_data
    model = CoxPHModel(alpha=0)
    model.model_ = _RaisingModel(
        ValueError("LAPACK reported an illegal value in 5-th argument.")
    )

    assert model._fit_and_check(X_train, y_train) == FAILED


class _RaisingModel:
    """Minimal stand-in for CoxPHSurvivalAnalysis whose fit() always raises."""

    def __init__(self, exc):
        self._exc = exc

    def fit(self, X, y):
        raise self._exc


# ---------------------------------------------------------------------------
# 2. ladder construction
# ---------------------------------------------------------------------------

def test_ladder_from_unregularized_default():
    """alpha=0 is the documented default, not 'unset', and it is falsy.

    Pins the truthiness hazard: the guard must be ``alpha > 0``, never
    ``if alpha``.
    """
    assert _fallback_ladder(0) == [0.01, 0.1, 1.0]
    assert _fallback_ladder(0.0) == [0.01, 0.1, 1.0]


@pytest.mark.parametrize("alpha,expected", [
    (0.005, [0.01, 0.1, 1.0]),
    (0.01, [0.1, 1.0]),
    (0.05, [0.1, 1.0]),
    (0.5, [1.0]),
])
def test_ladder_never_weakens_user_alpha(alpha, expected):
    ladder = _fallback_ladder(alpha)
    assert ladder == expected
    assert all(rung > alpha for rung in ladder)


@pytest.mark.parametrize("alpha", [1.0, 5, 100.0])
def test_ladder_empty_at_or_above_top_rung(alpha):
    """Nothing stronger to escalate to, and weakening is never automatic."""
    assert _fallback_ladder(alpha) == []


def test_ladder_never_downgrades_a_strong_alpha():
    """CoxPHModel(alpha=5) must never be silently re-fit at 0.01."""
    assert 0.01 not in _fallback_ladder(5)


# ---------------------------------------------------------------------------
# 3. ladder execution
# ---------------------------------------------------------------------------

def _fit_with(side_effect, train_test_data, alpha=0):
    X_train, _, y_train, _ = train_test_data
    model = CoxPHModel(alpha=alpha)
    with patch.object(CoxPHModel, "_fit_and_check",
                      side_effect=side_effect) as mock:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train, y_train)
    return model, mock, caught


def _fit_failures(caught):
    return [w for w in caught if issubclass(w.category, FitFailureWarning)]


def test_escalates_to_second_rung(train_test_data):
    model, _, caught = _fit_with(_side_effect_winning_at(0.1), train_test_data)

    assert model.alpha_ == 0.1
    warned = _fit_failures(caught)
    assert len(warned) == 1
    assert "alpha=0.1" in str(warned[0].message)
    assert "escalated past 0.01" in str(warned[0].message)


def test_escalates_to_third_rung(train_test_data):
    model, _, caught = _fit_with(_side_effect_winning_at(1.0), train_test_data)

    assert model.alpha_ == 1.0
    assert "escalated past 0.01, 0.1" in str(_fit_failures(caught)[0].message)


def test_caller_error_from_a_rung_propagates(train_test_data):
    """_fit_and_check re-raises caller errors; the ladder must not bury them.

    Folding a bad argument into "data may be too degenerate" would undo the
    classification one frame below.
    """
    X_train, _, y_train, _ = train_test_data
    model = CoxPHModel(alpha=0)

    with patch.object(
        CoxPHModel, "_fit_and_check",
        side_effect=[FAILED, ValueError("alpha must be positive, but was -0.5")],
    ):
        with pytest.raises(ValueError, match="alpha must be positive"):
            model.fit(X_train, y_train)


def test_rung_that_succeeds_with_warnings_wins(train_test_data):
    """The ladder stops at the first rung that does not fail outright.

    A rung that fits but still warns is a usable model, so it wins and the
    second warning reports the residual instability rather than escalating.
    """
    model, _, caught = _fit_with([FAILED, WARNED], train_test_data)

    assert model.alpha_ == 0.01
    warned = _fit_failures(caught)
    assert len(warned) == 2
    assert "alpha=0.01" in str(warned[1].message)
    assert "may still be unreliable" in str(warned[1].message)


def test_all_rungs_fail_raises_runtime_error(train_test_data):
    X_train, _, y_train, _ = train_test_data
    model = CoxPHModel(alpha=0)

    with patch.object(CoxPHModel, "_fit_and_check",
                      side_effect=[FAILED] * (1 + len(_FALLBACK_ALPHAS))):
        with pytest.raises(RuntimeError) as excinfo:
            model.fit(X_train, y_train)

    msg = str(excinfo.value)
    for rung in _FALLBACK_ALPHAS:
        assert f"alpha={rung:g}" in msg
    assert "collinear" in msg


def test_empty_ladder_raises_only_when_there_is_no_model(train_test_data):
    """alpha at the top rung and the fit failed: nothing to keep, fail fast."""
    X_train, _, y_train, _ = train_test_data
    model = CoxPHModel(alpha=5)

    with patch.object(CoxPHModel, "_fit_and_check",
                      side_effect=[FAILED]) as mock:
        with pytest.warns(MethodologicalWarning, match="ridge penalty"):
            with pytest.raises(RuntimeError, match="already at or above"):
                model.fit(X_train, y_train)

    assert mock.call_count == 1


def test_empty_ladder_keeps_a_fit_that_only_warned(train_test_data):
    """A usable model is kept rather than thrown away.

    Raising here would fail the CV fold and could drop CoxPH from the results,
    which is the outcome the fallback exists to prevent. This is the same
    trade-off force=True makes.
    """
    X_train, _, y_train, _ = train_test_data
    model = CoxPHModel(alpha=2.0)

    with patch.object(CoxPHModel, "_fit_and_check", side_effect=[WARNED]):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train, y_train)

    assert model.is_fitted_ is True
    assert model.alpha_ == 2.0
    assert "Keeping the original fit" in str(_fit_failures(caught)[0].message)


def test_memory_error_propagates(train_test_data):
    """The ladder catches nothing, so MemoryError escapes untouched."""
    X_train, _, y_train, _ = train_test_data
    model = CoxPHModel(alpha=0)

    with patch.object(CoxPHModel, "_fit_and_check",
                      side_effect=[FAILED, MemoryError()]):
        with pytest.raises(MemoryError):
            model.fit(X_train, y_train)


def test_exactly_one_warning_regardless_of_rungs_tried(train_test_data):
    """The ladder emits one wrapper warning, not one per rung."""
    _, _, caught = _fit_with([FAILED, FAILED, FAILED, CLEAN], train_test_data)

    assert len(_fit_failures(caught)) == 1


def test_alpha_and_model_stay_in_sync_after_escalation(train_test_data):
    """_attach_covariance builds its penalty vector from alpha_.

    If alpha_ and model_ ever drift apart, the covariance matrix (and so the
    Wald CIs on hazard-ratio plots) would be silently wrong.
    """
    model, _, _ = _fit_with(_side_effect_winning_at(0.1), train_test_data)

    assert model.alpha_ == model.model_.alpha


def test_winning_rung_that_did_not_converge_still_warns(train_test_data):
    """The ladder must not be quieter about non-convergence than a plain fit.

    Without this the coefficients of a non-converged fit would be published
    with no caveat, while the same condition outside the ladder warns.
    """
    model, _, caught = _fit_with(
        [FAILED, (False, False, True)], train_test_data
    )

    assert model.alpha_ == 0.01
    assert any("did not converge" in str(w.message)
               for w in _fit_failures(caught))


def test_ladder_error_quotes_the_real_solver_cause(train_test_data):
    """The boolean return throws the message away; the ladder keeps a copy.

    Without it the user gets three alphas and no hint of why any of them
    failed.
    """
    X_train, _, y_train, _ = train_test_data
    model = CoxPHModel(alpha=0)

    def always_lapack(self, X, y):
        self._last_solver_error_ = "LAPACK reported an illegal value in 5-th argument."
        return FAILED

    with patch.object(CoxPHModel, "_fit_and_check", always_lapack):
        with pytest.raises(RuntimeError, match="Last solver error"):
            model.fit(X_train, y_train)


# ---------------------------------------------------------------------------
# 4. hazard-ratio inference is suppressed past the first rung
# ---------------------------------------------------------------------------

def test_first_rung_keeps_hazard_ratio_inference(train_test_data):
    """Shrinkage at 0.01 is negligible, and this was the only prior fallback."""
    model, _, _ = _fit_with(_side_effect_winning_at(0.01), train_test_data)

    assert model.alpha_ == 0.01
    assert model.hazard_ratio_valid is True


@pytest.mark.parametrize("rung", [0.1, 1.0])
def test_escalation_past_the_first_rung_suppresses_hazard_ratios(
    rung, train_test_data
):
    """Penalized coefficients must not be published as unbiased estimates.

    hazard_ratio_valid gates the coefficient table and the forest plot, so
    leaving it True here would present ridge estimates with Wald CIs and
    p-values as if they were unpenalized multivariable hazard ratios.
    """
    model, _, caught = _fit_with(_side_effect_winning_at(rung), train_test_data)

    assert model.alpha_ == rung
    assert model.hazard_ratio_valid is False
    caveats = [w for w in caught
               if issubclass(w.category, MethodologicalWarning)
               and "shrunk toward zero" in str(w.message)]
    assert len(caveats) == 1


def test_escalation_records_a_reason_for_the_report(train_test_data):
    """The fit-time warning never reaches someone reading the report later."""
    model, _, _ = _fit_with(_side_effect_winning_at(1.0), train_test_data)

    assert "alpha=1" in model.hr_suppressed_reason_
    assert "C-index and survival probabilities are unaffected" in \
        model.hr_suppressed_reason_


def test_no_suppression_reason_when_inference_survives(train_test_data):
    model, _, _ = _fit_with(_side_effect_winning_at(0.01), train_test_data)

    assert model.hr_suppressed_reason_ is None


def test_hazard_ratio_valid_before_fit_is_unchanged():
    """The property is read before fit() in some code paths."""
    assert CoxPHModel(alpha=0).hazard_ratio_valid is True
    assert CoxPHModel(alpha=0.5).hazard_ratio_valid is False


def test_user_alpha_records_a_reason_naming_that_alpha(train_test_data):
    """The other route to suppression, which used to explain nothing.

    A user-supplied alpha turns hazard_ratio_valid False just as an escalation
    does, so the report needs the same explanation instead of an unexplained
    gap where the table and forest plot were.
    """
    model, _, _ = _fit_with([CLEAN], train_test_data, alpha=0.1)

    assert model.hazard_ratio_valid is False
    assert "alpha=0.1" in model.hr_suppressed_reason_
    assert "C-index and survival probabilities are unaffected" in \
        model.hr_suppressed_reason_
    # No ladder ran, so claiming an escalation would invent one.
    assert "escalated" not in model.hr_suppressed_reason_


def test_clean_unpenalized_fit_records_no_reason(train_test_data):
    """The common case: nothing suppressed, so nothing to explain."""
    model, _, _ = _fit_with([CLEAN], train_test_data)

    assert model.hazard_ratio_valid is True
    assert model.hr_suppressed_reason_ is None


def test_user_alpha_reason_mentions_an_escalation_on_top_of_it(train_test_data):
    """alpha=0.5 escalates to 1.0, so the reason must name both penalties.

    Naming only the user's alpha would understate how much the coefficients
    were shrunk; naming only the escalation would read as if the user had
    asked for an unpenalized fit.
    """
    model, _, _ = _fit_with([FAILED, CLEAN], train_test_data, alpha=0.5)

    assert model.alpha_ == 1.0
    assert "alpha=0.5" in model.hr_suppressed_reason_
    assert "escalated to alpha=1 for numerical stability" in \
        model.hr_suppressed_reason_


def test_reason_does_not_promise_usable_probabilities_after_instability(
    train_test_data
):
    """"Unaffected" would contradict the warning the same fit emitted.

    An alpha at the top of the ladder with instability keeps a fit whose
    survival probabilities fit() called unreliable, so the suppression note
    must not reassure the reader in the next paragraph.
    """
    model, _, _ = _fit_with([WARNED], train_test_data, alpha=2.0)

    assert model.overflow_detected_ is True
    assert "survival probabilities may be unreliable" in \
        model.hr_suppressed_reason_
    assert "survival probabilities are unaffected" not in \
        model.hr_suppressed_reason_


def test_resolved_instability_still_promises_usable_probabilities(
    train_test_data
):
    """Escalating is supposed to fix it, so saying otherwise would mislead.

    overflow_detected_ is True on every escalation, so keying the caveat off
    that flag would put "may be unreliable" on the successful case the ladder
    exists to produce.
    """
    model, _, _ = _fit_with(_side_effect_winning_at(1.0), train_test_data)

    assert model.overflow_detected_ is True
    assert model.unstable_fit_kept_ is False
    assert "survival probabilities are unaffected" in \
        model.hr_suppressed_reason_


def test_escalation_that_did_not_resolve_instability_is_not_called_clean(
    train_test_data
):
    """The winning rung warned at the very penalty meant to fix it.

    (alpha, alpha_) alone cannot see this: it looks identical to a clean
    escalation. fit() warns "may still be unreliable" here, so the recorded
    flag has to carry that through to the report.
    """
    model, _, caught = _fit_with(
        [FAILED, FAILED, WARNED], train_test_data
    )

    assert model.alpha_ == 0.1
    assert model.unstable_fit_kept_ is True
    assert any("may still be unreliable" in str(w.message)
               for w in _fit_failures(caught))
    assert "survival probabilities may be unreliable" in \
        model.hr_suppressed_reason_


def test_reason_survives_a_model_that_never_stored_it(train_test_data):
    """A learner saved before this existed must still explain the gap.

    hr_suppressed_reason_ is computed from (alpha, alpha_) rather than stored
    at fit time, so an older pickle, which has no such attribute, still gets
    an explanation instead of the unexplained gap this fix removes.
    """
    model, _, _ = _fit_with([CLEAN], train_test_data, alpha=0.1)
    restored = pickle.loads(pickle.dumps(model))

    assert "hr_suppressed_reason_" not in vars(restored)
    assert "alpha=0.1" in restored.hr_suppressed_reason_


# ---------------------------------------------------------------------------
# 4b. convergence is recorded, not inferred from the absence of overflow
# ---------------------------------------------------------------------------

NOT_CONVERGED = (False, False, True)   # fit succeeded, hit the iteration limit
UNSTABLE_NOT_CONVERGED = (False, True, True)   # ... and warned about overflow

# (label, side_effect, alpha, force) for each of the four success returns in
# fit(). Paired below so both convergence outcomes are covered on every path.
SUCCESS_PATHS = [
    ("plain", [CLEAN], [NOT_CONVERGED], 0, False),
    ("force keeps unstable fit", [WARNED], [UNSTABLE_NOT_CONVERGED], 0, True),
    ("no rung above alpha", [WARNED], [UNSTABLE_NOT_CONVERGED], 2.0, False),
    ("ladder rung won", [FAILED, CLEAN], [FAILED, NOT_CONVERGED], 0, False),
]


@pytest.mark.parametrize(
    "label, converged_effect, failed_effect, alpha, force", SUCCESS_PATHS
)
def test_converged_is_recorded_on_every_success_path(
    label, converged_effect, failed_effect, alpha, force, train_test_data
):
    """Both outcomes, on all four paths that can return a fitted model.

    A path that forgets leaves converged_ at None, which the report renders as
    "not recorded". Covering the False case matters most on the two paths that
    keep an unstable fit: the report line is the only place non-convergence
    surfaces if the console warning is ever dropped.
    """
    def fit_and_record(side_effect):
        model = CoxPHModel(alpha=alpha, force=force)
        with patch.object(CoxPHModel, "_fit_and_check", side_effect=side_effect):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(X_train, y_train)
        return model, caught

    X_train, _, y_train, _ = train_test_data

    model, _ = fit_and_record(converged_effect)
    assert model.converged_ is True, label

    model, caught = fit_and_record(failed_effect)
    assert model.converged_ is False, label
    # The report must never be the only place a non-converged fit is named.
    assert any("did not converge" in str(w.message)
               for w in _fit_failures(caught)), label


def test_non_convergence_and_suppression_are_reported_together(train_test_data):
    """The worst-case model: unreliable coefficients AND no hazard ratios.

    Each caveat is covered alone elsewhere; this pins that neither masks the
    other, since they are computed by different code and rendered as separate
    lines.
    """
    model, _, _ = _fit_with([FAILED, FAILED, NOT_CONVERGED], train_test_data)

    assert model.alpha_ == 0.1
    assert model.converged_ is False
    assert model.hazard_ratio_valid is False
    assert "alpha=0.1" in model.hr_suppressed_reason_


def test_converged_is_none_before_a_fit():
    """None means "not known", which the report must not read as False."""
    assert CoxPHModel().converged_ is None


def test_a_failed_refit_does_not_keep_the_previous_outcome(train_test_data):
    """Both flags describe one fit, so neither may outlive it.

    Without the reset, hr_suppressed_reason_ would go on describing a fit
    that no longer exists as having kept an unstable solution.
    """
    X_train, _, y_train, _ = train_test_data
    model = CoxPHModel(alpha=2.0)

    with patch.object(CoxPHModel, "_fit_and_check", side_effect=[WARNED]):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            model.fit(X_train, y_train)
    assert model.unstable_fit_kept_ is True

    with patch.object(CoxPHModel, "_fit_and_check", side_effect=[FAILED]):
        with pytest.raises(RuntimeError):
            model.fit(X_train, y_train)

    assert model.converged_ is None
    assert model.unstable_fit_kept_ is False


def test_sksurv_convergence_warning_is_classified(train_test_data):
    """Guard the regex against an upstream reword.

    converged_ is only as good as _CONVERGENCE_RE matching what sksurv
    actually emits. If the wording drifts, convergence_failed silently goes
    False and the report says "converged normally" for a fit that did not,
    which is the bug this change removes arriving by another route. Build the
    warning with sksurv's own class and message rather than a paraphrase.
    """
    # sksurv raises sklearn's ConvergenceWarning, not one of its own.
    from sklearn.exceptions import ConvergenceWarning

    X_train, _, y_train, _ = train_test_data
    model = CoxPHModel(alpha=0)
    model.model_ = _WarningModel(ConvergenceWarning(
        "Optimization did not converge: Maximum number of iterations "
        "has been exceeded."
    ))

    assert model._fit_and_check(X_train, y_train) == NOT_CONVERGED


class _WarningModel:
    """Stand-in for CoxPHSurvivalAnalysis whose fit() warns instead of raising."""

    def __init__(self, warning):
        self._warning = warning

    def fit(self, X, y):
        warnings.warn(self._warning)
        return self


# ---------------------------------------------------------------------------
# 5. end to end, through the real solver seam
# ---------------------------------------------------------------------------

def test_real_lapack_valueerror_from_the_solver_lands_on_a_rung(
    train_test_data, monkeypatch
):
    """The regression, driven through the real fit() rather than a mock.

    sksurv calls scipy's solve() inside its Newton loop, so patching that seam
    reproduces the reported failure end to end: no mocking of our own
    classification, and a real fitted model comes out the other side.
    """
    import sksurv.linear_model.coxph as sksurv_coxph

    real_solve = sksurv_coxph.solve
    state = {"calls": 0}

    def flaky_solve(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] <= 2:
            raise ValueError("LAPACK reported an illegal value in 5-th argument.")
        return real_solve(*args, **kwargs)

    monkeypatch.setattr(sksurv_coxph, "solve", flaky_solve)

    X_train, _, y_train, _ = train_test_data
    model = CoxPHModel(alpha=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)

    assert model.is_fitted_ is True
    assert model.alpha_ in _FALLBACK_ALPHAS
    assert model.overflow_detected_ is True
    assert np.all(np.isfinite(model.predict(X_train)))
    assert "LAPACK solve failure" in str(_fit_failures(caught)[0].message)


# ---------------------------------------------------------------------------
# 6. a failed covariance says why, instead of silently costing every CI
# ---------------------------------------------------------------------------

def _fit_with_singular_hessian(train_test_data):
    """Fit for real, but make the covariance inversion raise.

    scipy's inv does not raise on the degenerate designs a caller can actually
    build (duplicate columns, p>n, constant columns all return garbage rather
    than failing), so the seam has to be patched to reach this branch at all.
    """
    X_train, _, y_train, _ = train_test_data
    model = CoxPHModel(alpha=0)
    singular = np.linalg.LinAlgError("Matrix is singular.")

    with patch("mlsurv.models.coxph._inv", side_effect=singular), \
            warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)
    return model, caught


def test_no_reason_recorded_when_the_covariance_works(train_test_data):
    """The common case: intervals exist, so there is nothing to explain."""
    X_train, _, y_train, _ = train_test_data
    model = CoxPHModel(alpha=0).fit(X_train, y_train)

    assert hasattr(model.model_, "covariance_matrix_")
    assert model.covariance_unavailable_reason_ is None


def test_singular_hessian_records_why_the_intervals_are_gone(train_test_data):
    """A fit-time warning does not reach whoever reads the report later.

    The hazard ratios survive a failed inversion; only the intervals around
    them are lost. Without this, that costs every CI and p-value on the report
    with nothing on the page saying so.
    """
    model, caught = _fit_with_singular_hessian(train_test_data)

    assert model.is_fitted_ is True
    assert not hasattr(model.model_, "covariance_matrix_")
    assert "is singular" in model.covariance_unavailable_reason_
    assert "Matrix is singular." in model.covariance_unavailable_reason_
    # The point estimates are unaffected, and the reason must say so.
    assert model.hazard_ratio_valid is True
    assert "still valid point estimates" in model.covariance_unavailable_reason_
    assert any(issubclass(w.category, SuggestionWarning)
               and "Could not compute covariance matrix" in str(w.message)
               for w in caught)


def test_covariance_reason_is_cleared_by_a_later_fit(train_test_data):
    """The reason describes one fit, so it must not outlive it."""
    X_train, _, y_train, _ = train_test_data
    model, _ = _fit_with_singular_hessian(train_test_data)
    assert model.covariance_unavailable_reason_ is not None

    model.fit(X_train, y_train)

    assert model.covariance_unavailable_reason_ is None


def test_covariance_reason_is_none_before_a_fit():
    assert CoxPHModel().covariance_unavailable_reason_ is None
