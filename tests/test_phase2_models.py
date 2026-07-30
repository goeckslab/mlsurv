"""Test model wrappers for Phase 2."""

import warnings

import pytest
import numpy as np
import pandas as pd

from mlsurv._warnings import InternalWarning, SuggestionWarning


def _has_torch_pycox():
    """Check if torch and pycox are available."""
    try:
        import torch  # noqa: F401
        import pycox  # noqa: F401
        import torchtuples  # noqa: F401
        return True
    except ImportError:
        return False


class TestBaseSurvivalModel:
    """Test BaseSurvivalModel abstract class."""

    def test_cannot_instantiate(self):
        """Verify BaseSurvivalModel cannot be instantiated."""
        from mlsurv.models.base import BaseSurvivalModel

        with pytest.raises(TypeError):
            BaseSurvivalModel()

    def test_subclass_requirements(self):
        """Verify subclass must implement fit and predict."""
        from mlsurv.models.base import BaseSurvivalModel

        class IncompleteModel(BaseSurvivalModel):
            pass

        with pytest.raises(TypeError):
            IncompleteModel()


class TestCoxPHModel:
    """Test CoxPHModel wrapper."""

    def test_fit_predict(self, train_test_data):
        """Verify fit and predict work correctly."""
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxPHModel(alpha=0.01)
        model.fit(X_train, y_train)

        assert model.is_fitted_
        assert model.n_features_in_ == X_train.shape[1]

        risk_scores = model.predict(X_test)
        assert len(risk_scores) == len(X_test)
        assert not np.isnan(risk_scores).any()

    def test_score(self, train_test_data):
        """Verify score returns valid C-index."""
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxPHModel()
        model.fit(X_train, y_train)

        c_index = model.score(X_test, y_test)
        assert 0.0 <= c_index <= 1.0

    def test_predict_survival_function(self, train_test_data):
        """Verify survival function prediction."""
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxPHModel()
        model.fit(X_train, y_train)

        surv_funcs = model.predict_survival_function(X_test)
        assert len(surv_funcs) == len(X_test)

    def test_get_coefficients(self, train_test_data):
        """Verify coefficient extraction."""
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxPHModel()
        model.fit(X_train, y_train)

        coef = model.get_coefficients()
        assert isinstance(coef, pd.Series)
        assert len(coef) == X_train.shape[1]

    def test_not_fitted_error(self):
        """Verify error when predicting before fit."""
        from sklearn.exceptions import NotFittedError
        from mlsurv.models import CoxPHModel

        model = CoxPHModel()

        with pytest.raises(NotFittedError, match="not fitted"):
            model.predict(np.random.randn(10, 5))


class TestHazardRatioValidFlag:
    """Test the hazard_ratio_valid capability flag."""

    def test_coxph_is_valid(self):
        """Unpenalized CoxPH yields valid hazard ratios with Wald inference.

        The flag is now alpha-dependent (a per-instance property), so read it
        off an instance: the default alpha=0 baseline is HR-valid.
        """
        from mlsurv.models import CoxPHModel
        assert CoxPHModel().hazard_ratio_valid is True

    def test_penalized_coxph_not_valid(self):
        """A ridge-penalized CoxPH (alpha>0) opts out of HR validity."""
        from mlsurv.models import CoxPHModel
        assert CoxPHModel(alpha=0.5).hazard_ratio_valid is False

    def test_other_models_not_valid(self):
        """Penalized, ranking, AFT, tree, and base models opt out."""
        from mlsurv.models import CoxNetModel, FSSVMModel, RSFModel
        from mlsurv.models import WeibullAFTModel
        from mlsurv.models.base import BaseSurvivalModel

        for cls in (CoxNetModel, FSSVMModel, RSFModel,
                    WeibullAFTModel, BaseSurvivalModel):
            assert cls.hazard_ratio_valid is False, cls.__name__


class _StubCoxPH:
    """The attributes _add_coxph_details reads, without fitting anything.

    Rendering is keyed on the recorded outcome, so a stub covers the state
    combinations far more cheaply than provoking each one from a real solver.
    ``test_real_fitted_model_renders_its_recorded_state`` keeps the stub
    honest: a rename on CoxPHModel would leave every stub test green while
    real reports silently degraded to "status not recorded".
    """

    def __init__(self, converged=True, overflow=False, alpha_=0, alpha=0,
                 unstable_kept=False, hr_suppressed_reason_=None,
                 covariance_unavailable_reason_=None):
        self.converged_ = converged
        self.overflow_detected_ = overflow
        self.alpha_ = alpha_
        self.alpha = alpha
        self.unstable_fit_kept_ = unstable_kept
        self.hr_suppressed_reason_ = hr_suppressed_reason_
        self.covariance_unavailable_reason_ = covariance_unavailable_reason_


def _render_coxph_details(model):
    """Render one model's CoxPH detail block to a single string."""
    from mlsurv.reports.sections._trained_details import _TrainedDetailsSubsection

    md_lines: list = []
    _TrainedDetailsSubsection(section=None)._add_coxph_details(md_lines, model)
    return "\n".join(md_lines)


class TestCoxPHTrainedDetails:
    """_add_coxph_details must not claim more than the model recorded."""

    def test_clean_fit_reports_convergence(self):
        assert "converged normally" in _render_coxph_details(_StubCoxPH())

    def test_overflow_reports_the_penalty_it_settled_on(self):
        rendered = _render_coxph_details(
            _StubCoxPH(overflow=True, alpha=0, alpha_=0.1)
        )

        assert "Overflow detected" in rendered
        assert "Penalizer alpha=0.1 was applied" in rendered

    def test_no_penalizer_is_claimed_when_none_was_applied(self):
        """force=True kept an unpenalized fit, so alpha_ stays 0.

        "penalizer alpha=0 was applied" is not just useless, it is false: 0 is
        the unregularized fit. It would also replace the real caveat (the
        unstable fit was kept) with a reassurance that it was handled.
        """
        rendered = _render_coxph_details(
            _StubCoxPH(overflow=True, alpha=0, alpha_=0)
        )

        assert "was applied" not in rendered
        assert "kept rather than penalized" in rendered
        assert "may be unreliable" in rendered

    def test_a_user_penalty_is_not_described_as_no_penalty(self):
        """alpha_ == alpha here because no penalty was added ON TOP.

        The fit was penalized, at the user's own alpha. Saying it was "kept
        rather than penalized" would contradict the hazard-ratio line in the
        same block, which names that alpha.
        """
        rendered = _render_coxph_details(
            _StubCoxPH(overflow=True, alpha=2.0, alpha_=2.0)
        )

        assert "kept rather than penalized" not in rendered
        assert "No penalty was added on top of the alpha=2 you passed" in rendered
        assert "may be unreliable" in rendered

    def test_penalty_that_did_not_resolve_instability_says_so(self):
        """A penalizer was added, so naming it is right; stopping there is not.

        Without the second sentence this renders identically to a clean
        escalation, while the console said probabilities may be unreliable.
        """
        rendered = _render_coxph_details(_StubCoxPH(
            overflow=True, alpha=0, alpha_=0.1, unstable_kept=True
        ))

        assert "Penalizer alpha=0.1 was applied" in rendered
        assert "remained unstable even so" in rendered

    def test_resolved_instability_does_not_warn_about_probabilities(self):
        """The successful escalation, which is the common case."""
        rendered = _render_coxph_details(
            _StubCoxPH(overflow=True, alpha=0, alpha_=0.01)
        )

        assert "Penalizer alpha=0.01 was applied" in rendered
        assert "unreliable" not in rendered

    def test_failed_convergence_is_not_reported_as_converged(self):
        """The regression: overflow is False, so the old code claimed success.

        fit() warns "did not converge" on this path, which left the console
        and the report contradicting each other about the same run.
        """
        rendered = _render_coxph_details(_StubCoxPH(converged=False))

        assert "did not converge" in rendered
        assert "converged normally" not in rendered

    def test_failed_convergence_with_overflow_reports_both(self):
        """The two are independent, so one must not mask the other."""
        rendered = _render_coxph_details(
            _StubCoxPH(converged=False, overflow=True, alpha=0, alpha_=1.0)
        )

        assert "did not converge" in rendered
        assert "Penalizer alpha=1 was applied" in rendered

    def test_worst_case_model_carries_every_caveat(self):
        """Unreliable coefficients, a penalty, and no hazard ratios at once.

        Each caveat is rendered by a different branch, so this pins that none
        of them swallows another in the state a reader most needs warned about.
        """
        rendered = _render_coxph_details(_StubCoxPH(
            converged=False, overflow=True, alpha=0, alpha_=0.1,
            hr_suppressed_reason_="Suppressed. The fit needed alpha=0.1.",
        ))

        assert "did not converge" in rendered
        assert "Penalizer alpha=0.1 was applied" in rendered
        assert "**Hazard ratios:** Suppressed." in rendered

    def test_unrecorded_convergence_claims_nothing(self):
        """A model pickled before converged_ existed knows nothing either way.

        Falling back to "converged normally" here would reintroduce exactly
        the inference this fix removed.
        """
        model = _StubCoxPH()
        del model.converged_

        rendered = _render_coxph_details(model)
        assert "not recorded" in rendered
        assert "converged normally" not in rendered

    def test_suppression_reason_is_rendered(self):
        """The renderer prints whatever reason the model recorded.

        Being route-agnostic is the point: neither suppression route needs its
        own branch here, so a third one would be covered without a change.
        """
        rendered = _render_coxph_details(
            _StubCoxPH(hr_suppressed_reason_="Suppressed. You passed alpha=0.1.")
        )

        assert "**Hazard ratios:** Suppressed. You passed alpha=0.1." in rendered

    def test_no_suppression_line_when_no_reason_was_recorded(self):
        assert "Hazard ratios" not in _render_coxph_details(_StubCoxPH())

    def test_missing_confidence_intervals_are_explained(self):
        """A separate gap from the hazard ratios, with a separate cause."""
        rendered = _render_coxph_details(_StubCoxPH(
            covariance_unavailable_reason_="The matrix would not invert."
        ))

        assert "**Confidence intervals:** The matrix would not invert." in rendered

    def test_no_confidence_interval_line_when_they_exist(self):
        assert "Confidence intervals" not in _render_coxph_details(_StubCoxPH())

    def test_suppressed_hazard_ratios_swallow_the_interval_note(self):
        """No hazard ratios on the page means no intervals to explain.

        The recorded covariance reason vouches for the point estimates, so
        printing it under a line saying those estimates are withheld would put
        two contradictory sentences one after the other.
        """
        rendered = _render_coxph_details(_StubCoxPH(
            hr_suppressed_reason_="Suppressed. You passed alpha=0.1.",
            covariance_unavailable_reason_=(
                "The matrix is singular. The hazard ratios themselves are "
                "still valid point estimates."
            ),
        ))

        assert "**Hazard ratios:** Suppressed." in rendered
        assert "**Confidence intervals:**" not in rendered
        assert "still valid point estimates" not in rendered

    def test_real_fitted_model_renders_its_recorded_state(self, train_test_data):
        """Bind the stub tests to the real class.

        Every other test here uses a hand-maintained stub. If converged_ were
        renamed on CoxPHModel, or _unwrap_pipeline started handing over the
        sksurv estimator instead of the wrapper, all of them would stay green
        while real reports silently fell back to "status not recorded".
        """
        from mlsurv.models import CoxPHModel

        X_train, _, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0).fit(X_train, y_train)

        rendered = _render_coxph_details(model)
        assert "converged normally" in rendered
        assert "not recorded" not in rendered
        assert "Hazard ratios" not in rendered

    def test_real_penalized_model_explains_the_suppression(self, train_test_data):
        """The end-to-end path for the gap this fix closes."""
        from mlsurv.models import CoxPHModel

        X_train, _, y_train, _ = train_test_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CoxPHModel(alpha=0.1).fit(X_train, y_train)

        rendered = _render_coxph_details(model)
        assert "**Hazard ratios:**" in rendered
        assert "alpha=0.1" in rendered


class TestWeibullAFTCoefficients:
    """Test WeibullAFT.get_coefficients."""

    def test_returns_named_series_without_intercept(self, train_test_data):
        """AFT scale coefficients are returned indexed by feature, no Intercept."""
        from mlsurv.models import WeibullAFTModel

        X_train, X_test, y_train, y_test = train_test_data

        model = WeibullAFTModel(alpha=0.01)
        model.fit(X_train, y_train)

        coef = model.get_coefficients()
        assert isinstance(coef, pd.Series)
        assert len(coef) == X_train.shape[1]
        assert 'Intercept' not in coef.index
        # lifelines orders params_ alphabetically, not by column order;
        # the contract is the same set of features, correctly labeled.
        assert set(coef.index) == set(X_train.columns)

    def test_get_params(self):
        """Verify sklearn get_params compatibility."""
        from mlsurv.models import CoxPHModel

        model = CoxPHModel(alpha=0.1)
        params = model.get_params()

        assert 'alpha' in params
        assert params['alpha'] == 0.1

    def test_set_params(self):
        """Verify sklearn set_params compatibility."""
        from mlsurv.models import CoxPHModel

        model = CoxPHModel(alpha=0.1)
        model.set_params(alpha=0.5)

        assert model.alpha == 0.5

    # --- Overflow / fallback behavior tests ---

    def test_stable_fit_no_fallback(self, train_test_data):
        """Stable fit keeps original alpha and sets overflow_detected_=False."""
        from mlsurv.models import CoxPHModel

        X_train, _, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0.01)
        model.fit(X_train, y_train)

        assert model.alpha_ == 0.01
        assert model.overflow_detected_ is False

    def test_overflow_warning_triggers_fallback(self, train_test_data):
        """Overflow warnings trigger fallback to alpha=0.01."""
        from unittest.mock import patch
        from mlsurv.models import CoxPHModel
        from mlsurv.models.coxph import _FALLBACK_ALPHA

        X_train, _, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0)

        with patch.object(
            CoxPHModel, "_fit_and_check",
            side_effect=[(False, True, False), (False, False, False)],
        ):
            model.fit(X_train, y_train)

        assert model.alpha_ == _FALLBACK_ALPHA
        assert model.overflow_detected_ is True

    def test_valueerror_triggers_fallback(self, train_test_data):
        """ValueError (NaN search direction) triggers fallback."""
        from unittest.mock import patch
        from mlsurv.models import CoxPHModel
        from mlsurv.models.coxph import _FALLBACK_ALPHA

        X_train, _, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0)

        with patch.object(
            CoxPHModel, "_fit_and_check",
            side_effect=[(True, False, False), (False, False, False)],
        ):
            model.fit(X_train, y_train)

        assert model.alpha_ == _FALLBACK_ALPHA
        assert model.overflow_detected_ is True

    def test_unrelated_valueerror_propagates(self, train_test_data):
        """ValueError without NaN/infinite/singular message re-raises."""
        from unittest.mock import MagicMock
        from mlsurv.models import CoxPHModel

        X_train, _, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0)
        model.model_ = MagicMock()
        model.model_.fit.side_effect = ValueError("something unrelated")

        with pytest.raises(ValueError, match="something unrelated"):
            model._fit_and_check(X_train, y_train)

    def test_force_true_keeps_fit_on_warnings(self, train_test_data):
        """force=True keeps original fit when only warnings detected."""
        from unittest.mock import patch
        from mlsurv.models import CoxPHModel

        X_train, _, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0.05, force=True)

        with patch.object(
            CoxPHModel, "_fit_and_check", return_value=(False, True, False)
        ):
            model.fit(X_train, y_train)

        assert model.alpha_ == 0.05  # unchanged
        assert model.overflow_detected_ is True

    def test_force_true_ignored_when_fit_fails(self, train_test_data):
        """force=True is ignored when fit failed — fallback always happens."""
        from unittest.mock import patch
        from mlsurv.models import CoxPHModel
        from mlsurv.models.coxph import _FALLBACK_ALPHA

        X_train, _, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0, force=True)

        with patch.object(
            CoxPHModel, "_fit_and_check",
            side_effect=[(True, False, False), (False, False, False)],
        ):
            model.fit(X_train, y_train)

        assert model.alpha_ == _FALLBACK_ALPHA
        assert model.overflow_detected_ is True

    def test_get_params_includes_force(self):
        """get_params returns both alpha and force (sklearn contract)."""
        from mlsurv.models import CoxPHModel

        model = CoxPHModel(alpha=0.1, force=True)
        params = model.get_params()

        assert "force" in params
        assert params["force"] is True
        assert params["alpha"] == 0.1

    def test_non_overflow_warnings_reemitted(self, train_test_data):
        """Non-overflow warnings (e.g. ConvergenceWarning) pass through."""
        import warnings as _warnings
        from unittest.mock import MagicMock
        from mlsurv.models import CoxPHModel

        X_train, _, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0.01)
        model._validate_data(X_train, y_train)
        model.model_ = MagicMock()

        # Make model_.fit emit a FutureWarning (non-overflow)
        def fake_fit(X, y):
            _warnings.warn("did not converge", FutureWarning)

        model.model_.fit.side_effect = fake_fit

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            fit_failed, has_warnings, convergence_failed = model._fit_and_check(X_train, y_train)

        assert fit_failed is False
        assert has_warnings is False
        assert convergence_failed is False
        # The FutureWarning should have been re-emitted
        reemitted = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(reemitted) == 1
        assert "did not converge" in str(reemitted[0].message)

    def test_ill_conditioned_warning_triggers_fallback(self, train_test_data):
        """scipy LinAlgWarning 'ill-conditioned matrix' triggers the same
        fallback path as overflow RuntimeWarnings — wrapper consumes
        the raw warning and flags `has_warnings=True`.
        """
        import warnings as _warnings
        from unittest.mock import MagicMock
        from scipy.linalg import LinAlgWarning
        from mlsurv.models import CoxPHModel

        X_train, _, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0)
        model._validate_data(X_train, y_train)

        # Stub sksurv's fit to emit exactly the LinAlgWarning that scipy
        # emits from solve() on rank-deficient designs at sksurv coxph.py:492.
        def fake_fit(X, y):
            _warnings.warn_explicit(
                "An ill-conditioned matrix detected: slice 0 has rcond = 1.55e-16.",
                LinAlgWarning,
                "sksurv/linear_model/coxph.py",
                492,
            )

        sk_stub = MagicMock()
        sk_stub.fit.side_effect = fake_fit
        model.model_ = sk_stub

        # First call: detects ill-conditioned, returns has_warnings=True.
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            fit_failed, has_warnings, convergence_failed = model._fit_and_check(X_train, y_train)

        assert fit_failed is False
        assert has_warnings is True
        assert convergence_failed is False
        # The raw scipy LinAlgWarning must NOT have been re-emitted.
        leaked = [w for w in caught
                  if "ill-conditioned" in str(w.message)]
        assert leaked == []

    def test_ill_conditioned_endtoend_fallback(self, train_test_data):
        """End-to-end: ill-conditioned warning during fit() routes through
        the fallback re-fit, lands at alpha=_FALLBACK_ALPHA, and emits exactly
        one wrapper-level FitFailureWarning (not N raw scipy lines).
        """
        import warnings as _warnings
        from unittest.mock import patch
        from mlsurv.models import CoxPHModel
        from mlsurv._warnings import FitFailureWarning
        from mlsurv.models.coxph import _FALLBACK_ALPHA

        X_train, _, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0)

        # Patch _fit_and_check: first call reports ill-conditioned, fallback
        # call succeeds cleanly.
        with patch.object(
            CoxPHModel, "_fit_and_check",
            side_effect=[(False, True, False), (False, False, False)],
        ):
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                model.fit(X_train, y_train)

        assert model.alpha_ == _FALLBACK_ALPHA
        assert model.overflow_detected_ is True
        # Exactly one wrapper-level FitFailureWarning, no raw leak.
        fit_failures = [w for w in caught
                        if issubclass(w.category, FitFailureWarning)]
        assert len(fit_failures) == 1
        leaked = [w for w in caught
                  if "ill-conditioned" in str(w.message)
                  and not issubclass(w.category, FitFailureWarning)]
        assert leaked == []

    def test_reemit_non_overflow_skips_ill_conditioned(self):
        """reemit_non_overflow() suppresses scipy LinAlgWarning
        ill-conditioned messages alongside overflow RuntimeWarnings."""
        import warnings as _warnings
        from scipy.linalg import LinAlgWarning
        from mlsurv.models._overflow import reemit_non_overflow

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            _warnings.warn_explicit(
                "An ill-conditioned matrix detected: slice 0 has rcond = 1e-18.",
                LinAlgWarning,
                "sksurv/linear_model/coxph.py",
                492,
            )

        # Now hand them to reemit_non_overflow and verify nothing propagates.
        with _warnings.catch_warnings(record=True) as out:
            _warnings.simplefilter("always")
            reemit_non_overflow(caught)
        leaked = [w for w in out
                  if "ill-conditioned" in str(w.message)]
        assert leaked == []

    def test_convergence_warning_suppressed(self, train_test_data):
        """sksurv ConvergenceWarning ('Optimization did not converge: ...')
        is consumed by _fit_and_check and surfaced as `convergence_failed=True`.
        The raw warning must NOT leak.
        """
        import warnings as _warnings
        from unittest.mock import MagicMock
        from sklearn.exceptions import ConvergenceWarning
        from mlsurv.models import CoxPHModel

        X_train, _, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0)
        model._validate_data(X_train, y_train)

        def fake_fit(X, y):
            _warnings.warn_explicit(
                "Optimization did not converge: Maximum number of "
                "iterations has been exceeded.",
                ConvergenceWarning,
                "sksurv/linear_model/coxph.py",
                485,
            )

        sk_stub = MagicMock()
        sk_stub.fit.side_effect = fake_fit
        model.model_ = sk_stub

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            fit_failed, has_warnings, convergence_failed = (
                model._fit_and_check(X_train, y_train)
            )

        assert fit_failed is False
        assert has_warnings is False   # NOT routed to alpha-fallback path
        assert convergence_failed is True
        leaked = [w for w in caught
                  if "did not converge" in str(w.message)]
        assert leaked == []

    def test_convergence_endtoend_no_fallback(self, train_test_data):
        """End-to-end: ConvergenceWarning during fit() emits exactly one
        FitFailureWarning, keeps the user's original alpha (NO fallback
        re-fit), and does not leak the raw sksurv warning.
        """
        import warnings as _warnings
        from unittest.mock import patch
        from mlsurv.models import CoxPHModel
        from mlsurv._warnings import FitFailureWarning

        X_train, _, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0)

        with patch.object(
            CoxPHModel, "_fit_and_check",
            return_value=(False, False, True),
        ):
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                model.fit(X_train, y_train)

        # No alpha fallback — coefficients keep their original meaning.
        assert model.alpha_ == 0
        assert model.overflow_detected_ is False
        assert model.is_fitted_ is True

        fit_failures = [w for w in caught
                        if issubclass(w.category, FitFailureWarning)]
        assert len(fit_failures) == 1
        msg = str(fit_failures[0].message)
        assert "did not converge" in msg
        assert "n_iter" in msg  # actionable guidance present

    def test_reemit_non_overflow_skips_convergence(self):
        """reemit_non_overflow() suppresses sksurv ConvergenceWarning
        ('Optimization did not converge: ...') alongside RuntimeWarning
        instability messages. ConvergenceWarning is a UserWarning,
        not RuntimeWarning, so it needs its own message-text check.
        """
        import warnings as _warnings
        from sklearn.exceptions import ConvergenceWarning
        from mlsurv.models._overflow import reemit_non_overflow

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            _warnings.warn_explicit(
                "Optimization did not converge: Maximum number of "
                "iterations has been exceeded.",
                ConvergenceWarning,
                "sksurv/linear_model/coxph.py",
                485,
            )

        with _warnings.catch_warnings(record=True) as out:
            _warnings.simplefilter("always")
            reemit_non_overflow(caught)
        leaked = [w for w in out
                  if "did not converge" in str(w.message)]
        assert leaked == []

    def test_no_false_positive_on_wellconditioned_design(self, train_test_data):
        """Well-conditioned designs MUST NOT trip the new ill-conditioned
        detector — alpha stays at user value, no FitFailureWarning is emitted."""
        import warnings as _warnings
        from mlsurv.models import CoxPHModel
        from mlsurv._warnings import FitFailureWarning

        X_train, _, y_train, _ = train_test_data

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            model = CoxPHModel(alpha=0.01)
            model.fit(X_train, y_train)

        assert model.overflow_detected_ is False
        assert model.alpha_ == 0.01
        fit_failures = [w for w in caught
                        if issubclass(w.category, FitFailureWarning)]
        assert fit_failures == []

    def test_real_rankdeficient_design_routes_through_fallback(
        self, train_test_data
    ):
        """Real rank-deficient X (no mocking) makes scipy emit the actual
        LinAlgWarning inside sksurv's solve(); the wrapper consumes it and
        lands on a rung of the fallback ladder. Guards against scipy/sksurv
        message-text drift that mock-based tests would silently miss."""
        import warnings as _warnings
        from mlsurv.models import CoxPHModel
        from mlsurv._warnings import FitFailureWarning
        from mlsurv.models.coxph import _FALLBACK_ALPHAS

        X_train, _, y_train, _ = train_test_data
        # Force exact rank deficiency: append a column that is the sum of
        # two existing columns. scipy.linalg.solve() will emit
        # LinAlgWarning("An ill-conditioned matrix detected: ...") inside
        # sksurv's optimizer loop.
        X_rd = X_train.copy()
        cols = list(X_rd.columns)
        X_rd["redundant"] = X_rd[cols[0]] + X_rd[cols[1]]

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            model = CoxPHModel(alpha=0)
            model.fit(X_rd, y_train)

        assert model.overflow_detected_ is True
        # Which rung wins depends on the local BLAS, so assert membership
        # rather than pinning the exact penalty.
        assert model.alpha_ in _FALLBACK_ALPHAS
        # No raw ill-conditioned warning should reach the caller.
        leaked = [w for w in caught
                  if "ill-conditioned" in str(w.message)
                  and not issubclass(w.category, FitFailureWarning)]
        assert leaked == []
        # At least one wrapper-level FitFailureWarning should fire.
        fit_failures = [w for w in caught
                        if issubclass(w.category, FitFailureWarning)]
        assert len(fit_failures) >= 1

    def test_double_failure_raises_runtime_error(self, train_test_data):
        """When every fallback rung also fails, RuntimeError is raised."""
        from unittest.mock import patch
        from mlsurv.models import CoxPHModel
        from mlsurv.models.coxph import _FALLBACK_ALPHAS

        X_train, _, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0)

        # First call: instability detected. Then one call per ladder rung,
        # all reporting failure, so the ladder is exhausted.
        with patch.object(
            CoxPHModel,
            "_fit_and_check",
            side_effect=[(True, False, False)] * (1 + len(_FALLBACK_ALPHAS)),
        ):
            with pytest.raises(RuntimeError, match="also failed"):
                model.fit(X_train, y_train)

    # --- Prediction overflow clamping tests ---

    def _fit_and_inject_extreme_predict(self, train_test_data):
        """Helper: fit a CoxPH model and replace predict with extreme LPs."""
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0.01)
        model.fit(X_train, y_train)

        # Replace the sksurv model's predict with one that injects extreme
        # linear predictor values.  _predict_with_clamp wraps whatever
        # predict it finds, so extreme values will be clipped before
        # reaching the Breslow estimator.
        real_predict = model.model_.predict

        def _extreme_predict(X):
            lp = real_predict(X)
            lp[0] = 1000.0   # way beyond exp() range
            lp[1] = -1000.0
            return lp

        model.model_.predict = _extreme_predict
        return model, X_test

    def test_predict_survival_clamps_extreme_lp(self, train_test_data):
        """predict_survival_function clamps extreme LPs without overflow."""
        import warnings as _warnings
        from mlsurv.models._overflow import _OVERFLOW_RE

        model, X_test = self._fit_and_inject_extreme_predict(train_test_data)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            surv = model.predict_survival_function(X_test, return_array=True)

        # No overflow RuntimeWarnings should have escaped
        overflow_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and _OVERFLOW_RE.search(str(w.message))
        ]
        assert overflow_warnings == [], (
            f"Overflow warnings leaked: {[str(w.message) for w in overflow_warnings]}"
        )

        # The clamping UserWarning should have been emitted
        clamp_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "clamped" in str(w.message)
        ]
        assert len(clamp_warnings) == 1
        assert "2 linear predictor(s)" in str(clamp_warnings[0].message)

        # Output must be finite (no NaN / inf)
        assert np.all(np.isfinite(surv)), "Survival predictions contain NaN or inf"
        # Survival values must be in [0, 1]
        assert surv.min() >= 0.0
        assert surv.max() <= 1.0

    def test_predict_cumulative_hazard_clamps_extreme_lp(self, train_test_data):
        """predict_cumulative_hazard_function clamps extreme LPs without overflow."""
        import warnings as _warnings
        from mlsurv.models._overflow import _OVERFLOW_RE

        model, X_test = self._fit_and_inject_extreme_predict(train_test_data)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            chf = model.predict_cumulative_hazard_function(
                X_test, return_array=True
            )

        overflow_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and _OVERFLOW_RE.search(str(w.message))
        ]
        assert overflow_warnings == [], (
            f"Overflow warnings leaked: {[str(w.message) for w in overflow_warnings]}"
        )

        # The clamping UserWarning should have been emitted
        clamp_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "clamped" in str(w.message)
        ]
        assert len(clamp_warnings) == 1
        assert "2 linear predictor(s)" in str(clamp_warnings[0].message)

        assert np.all(np.isfinite(chf)), "CHF predictions contain NaN or inf"
        # Cumulative hazard is non-negative
        assert chf.min() >= 0.0

    def test_predict_with_clamp_restores_predict_on_error(self, train_test_data):
        """model_.predict is restored even when the inner call raises."""
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0.01)
        model.fit(X_train, y_train)

        # Capture normal predict output before the error scenario.
        expected_lp = model.model_.predict(X_test).copy()

        # Force the inner sksurv call to raise
        real_sf = model.model_.predict_survival_function
        def _raise_on_sf(*args, **kwargs):
            raise TypeError("injected error")
        model.model_.predict_survival_function = _raise_on_sf

        with pytest.raises(TypeError, match="injected error"):
            model.predict_survival_function(X_test)

        # predict must be restored — output should match pre-error values.
        # If the clamped wrapper leaked, values would be clipped to [-20, 20].
        restored_lp = model.model_.predict(X_test)
        np.testing.assert_array_equal(restored_lp, expected_lp)

        # Restore for clean teardown
        model.model_.predict_survival_function = real_sf

    def test_predict_with_clamp_reemits_warnings_on_error(self, train_test_data):
        """Non-overflow warnings captured before an exception are re-emitted."""
        import warnings
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0.01)
        model.fit(X_train, y_train)

        real_sf = model.model_.predict_survival_function

        def _warn_then_raise(*args, **kwargs):
            warnings.warn("convergence issue", InternalWarning)
            raise TypeError("injected error")

        model.model_.predict_survival_function = _warn_then_raise

        with pytest.warns(InternalWarning, match="convergence issue"):
            with pytest.raises(TypeError, match="injected error"):
                model.predict_survival_function(X_test)

        model.model_.predict_survival_function = real_sf


class TestCoxNetModel:
    """Test CoxNetModel wrapper."""

    def test_fit_predict(self, train_test_data):
        """Verify fit and predict work correctly."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxNetModel(l1_ratio=0.5, fit_baseline_model=True)
        model.fit(X_train, y_train)

        assert model.is_fitted_

        risk_scores = model.predict(X_test)
        assert len(risk_scores) == len(X_test)

    def test_survival_function_requires_baseline(self, train_test_data):
        """Verify survival function requires fit_baseline_model."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxNetModel(fit_baseline_model=False, alpha='min')
        model.fit(X_train, y_train)

        with pytest.raises(RuntimeError, match="fit_baseline_model"):
            model.predict_survival_function(X_test)

    def test_get_coefficients(self, train_test_data):
        """Verify coefficient extraction with L1 penalty."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxNetModel(l1_ratio=0.9)  # High L1 for sparsity
        model.fit(X_train, y_train)

        coef = model.get_coefficients()
        assert isinstance(coef, pd.Series)
        assert len(coef) == X_train.shape[1]

    def test_alpha_auto_selects_via_cv(self, train_test_data):
        """Verify alpha='auto' uses CV to select alpha."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxNetModel(l1_ratio=0.5, alpha='auto', cv_folds=3)
        model.fit(X_train, y_train)

        assert model.is_fitted_
        assert model.selected_alpha_ is not None
        assert isinstance(model.selected_alpha_, float)
        assert model.selected_alpha_ > 0

    def test_alpha_min_uses_smallest_alpha(self, train_test_data):
        """Verify alpha='min' uses the smallest alpha (least regularization)."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxNetModel(l1_ratio=0.5, alpha='min')
        model.fit(X_train, y_train)

        assert model.is_fitted_
        assert model.selected_alpha_ is not None
        # The selected alpha should be the smallest in the path

    def test_alpha_float_uses_specific_value(self, train_test_data):
        """Verify alpha=<float> uses that specific value."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data

        specific_alpha = 0.1
        model = CoxNetModel(l1_ratio=0.5, alpha=specific_alpha)
        model.fit(X_train, y_train)

        assert model.is_fitted_
        assert model.selected_alpha_ == specific_alpha
        assert model.alpha_ == specific_alpha  # Backwards compatibility

    def test_alpha_int_uses_path_index(self, train_test_data):
        """Verify alpha=<int> picks the alpha at that path index."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxNetModel(l1_ratio=0.5, alpha=50)
        model.fit(X_train, y_train)

        assert model.is_fitted_
        assert model.selected_alpha_ is not None
        assert isinstance(model.selected_alpha_, float)
        assert model.selected_alpha_ > 0

        # Verify it picked the right index from the path
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        path_model = CoxnetSurvivalAnalysis(
            l1_ratio=0.5, alpha_min_ratio=0.01, n_alphas=100,
            fit_baseline_model=False,
        )
        path_model.fit(X_train, y_train)
        expected_alpha = path_model.alphas_[50]
        assert model.selected_alpha_ == pytest.approx(expected_alpha)

        # Verify predictions work correctly
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert np.all(np.isfinite(predictions))

    def test_alpha_int_clamps_to_valid_range(self, train_test_data):
        """Verify alpha=999 clamps to last index in the path."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxNetModel(l1_ratio=0.5, alpha=999)
        model.fit(X_train, y_train)

        assert model.is_fitted_
        assert model.selected_alpha_ is not None
        assert model.selected_alpha_ > 0

    def test_alpha_int_zero_is_max_regularization(self, train_test_data):
        """Verify alpha=0 picks alphas_[0] (strongest regularization)."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxNetModel(l1_ratio=0.5, alpha=0)
        model.fit(X_train, y_train)

        assert model.is_fitted_

        # Verify it picked the first (strongest) alpha from the path
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        path_model = CoxnetSurvivalAnalysis(
            l1_ratio=0.5, alpha_min_ratio=0.01, n_alphas=100,
            fit_baseline_model=False,
        )
        path_model.fit(X_train, y_train)
        expected_alpha = path_model.alphas_[0]
        assert model.selected_alpha_ == pytest.approx(expected_alpha)

    def test_alpha_int_negative_raises(self, train_test_data):
        """Verify negative integer alpha raises ValueError."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxNetModel(l1_ratio=0.5, alpha=-1)
        with pytest.raises(ValueError, match="non-negative"):
            model.fit(X_train, y_train)

    def test_alpha_bool_raises_not_int(self, train_test_data):
        """Verify alpha=True/False is not treated as int path index."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data

        for val in (True, False):
            model = CoxNetModel(l1_ratio=0.5, alpha=val)
            with pytest.raises(ValueError, match="alpha must be"):
                model.fit(X_train, y_train)

    def test_alpha_int_clamp_warns(self, train_test_data):
        """Verify over-range alpha_index emits a warning when clamped."""
        import warnings as _warnings
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxNetModel(l1_ratio=0.5, alpha=999)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            model.fit(X_train, y_train)

        clamp_warnings = [w for w in caught if "Clamping" in str(w.message)]
        assert len(clamp_warnings) == 1

    def test_alpha_auto_differs_from_min(self, train_test_data):
        """Verify alpha='auto' may select different alpha than 'min'."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data

        # Fit with auto
        model_auto = CoxNetModel(l1_ratio=0.9, alpha='auto', cv_folds=3)
        model_auto.fit(X_train, y_train)

        # Fit with min
        model_min = CoxNetModel(l1_ratio=0.9, alpha='min')
        model_min.fit(X_train, y_train)

        # Both should be valid alphas
        assert model_auto.selected_alpha_ > 0
        assert model_min.selected_alpha_ > 0

        # Note: They may or may not be different depending on data
        # The key is that both work correctly

    def test_predict_and_coefficients_use_same_alpha(self, train_test_data):
        """Verify predict() and get_coefficients() use consistent alpha."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxNetModel(l1_ratio=0.5, alpha='auto', cv_folds=3)
        model.fit(X_train, y_train)

        # Get coefficients
        coef = model.get_coefficients()

        # Manually compute risk scores
        manual_scores = X_test.values @ coef.values

        # Model predictions should be perfectly correlated with manual scores
        # (Cox models can have arbitrary offset, so we check correlation not equality)
        model_scores = model.predict(X_test)
        correlation = np.corrcoef(manual_scores, model_scores)[0, 1]
        assert correlation > 0.9999, f"Correlation should be ~1.0, got {correlation}"

    def test_alpha_invalid_raises_error(self, train_test_data):
        """Verify invalid alpha values raise errors."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxNetModel(l1_ratio=0.5, alpha='invalid')

        with pytest.raises(ValueError, match="alpha must be"):
            model.fit(X_train, y_train)

    def test_alpha_auto_fallback_small_dataset(self):
        """Verify alpha='auto' falls back to 'min' for small datasets."""
        import warnings
        from mlsurv.models import CoxNetModel

        # Create a very small dataset (3 samples, less than default cv_folds=5)
        X_small = pd.DataFrame({'x1': [1.0, 2.0, 3.0], 'x2': [0.5, 1.5, 2.5]})
        y_small = np.array([(True, 10.0), (False, 20.0), (True, 15.0)],
                           dtype=[('event', '?'), ('time', '<f8')])

        model = CoxNetModel(l1_ratio=0.5, alpha='auto', cv_folds=5)

        # Should warn and fall back to alpha='min' instead of raising
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(X_small, y_small)

            # Check that a warning was issued
            assert len(w) >= 1
            assert any("less than cv_folds" in str(warning.message) for warning in w)

        # Model should still be fitted
        assert model.is_fitted_
        assert model.selected_alpha_ is not None

    def test_alpha_np_integer_uses_path_index(self, train_test_data):
        """np.int64 alpha (as returned by Optuna) hits the path-index branch."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxNetModel(l1_ratio=0.5, alpha=np.int64(50))
        model.fit(X_train, y_train)

        assert model.is_fitted_
        assert isinstance(model.selected_alpha_, float)
        assert model.selected_alpha_ > 0

    def test_alpha_np_floating_uses_specific_value(self, train_test_data):
        """np.float64 alpha hits the specific-float branch."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxNetModel(l1_ratio=0.5, alpha=np.float64(0.1))
        model.fit(X_train, y_train)

        assert model.is_fitted_
        assert model.selected_alpha_ == pytest.approx(0.1)

    def test_alpha_auto_warns_on_no_valid_scores(self, train_test_data):
        """SuggestionWarning when all CV scoring fails."""
        from unittest.mock import patch
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxNetModel(l1_ratio=0.5, alpha='auto', cv_folds=3)

        # Mock concordance_index_censored to always fail during CV,
        # forcing the no-valid-scores fallback path. The scoring loop lives
        # in the shared helper (mlsurv._shared), so patch it there.
        with patch(
            'mlsurv._shared.concordance_index_censored',
            side_effect=ValueError("all tied"),
        ), pytest.warns(SuggestionWarning, match="no valid C-index scores"):
            model.fit(X_train, y_train)

        assert model.is_fitted_
        assert model.selected_alpha_ is not None

        # Verify fallback uses same alpha as alpha='min'
        model_min = CoxNetModel(l1_ratio=0.5, alpha='min')
        model_min.fit(X_train, y_train)
        assert model.selected_alpha_ == model_min.selected_alpha_

    def test_cv_fold_fit_failure_skips_fold(self, train_test_data, caplog):
        """Model still fits and logs debug when one CV fold's fit() raises."""
        import logging
        from unittest.mock import patch
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxNetModel(l1_ratio=0.5, alpha='auto', cv_folds=3)

        original_fit = CoxnetSurvivalAnalysis.fit
        call_count = [0]

        def fit_fails_first_fold(self_inner, X, y):
            call_count[0] += 1
            # Fail on the first CV fold fit (call 2; call 1 is _fit_regularization_path)
            if call_count[0] == 2:
                raise ValueError("singular matrix in fold")
            return original_fit(self_inner, X, y)

        # The fold-scoring loop runs in the shared helper, which logs under
        # the "mlsurv.shared" logger with a unified message.
        with caplog.at_level(logging.DEBUG, logger="mlsurv.shared"):
            with patch.object(
                CoxnetSurvivalAnalysis, 'fit', fit_fails_first_fold,
            ):
                model.fit(X_train, y_train)

        assert model.is_fitted_
        assert model.selected_alpha_ is not None
        assert any(
            "Penalized Cox CV fold fit failed" in msg and "singular matrix" in msg
            for msg in caplog.messages
        )

    def test_cv_fold_memory_error_propagates(self, train_test_data):
        """MemoryError in CV fold is never caught."""
        from unittest.mock import patch
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxNetModel(l1_ratio=0.5, alpha='auto', cv_folds=3)

        original_fit = CoxnetSurvivalAnalysis.fit
        call_count = [0]

        def fit_raises_memory_error(self_inner, X, y):
            call_count[0] += 1
            if call_count[0] == 2:
                raise MemoryError("out of memory in fold")
            return original_fit(self_inner, X, y)

        with patch.object(CoxnetSurvivalAnalysis, 'fit', fit_raises_memory_error):
            with pytest.raises(MemoryError, match="out of memory"):
                model.fit(X_train, y_train)

    def test_cv_scoring_failure_logs_debug(self, train_test_data, caplog):
        """Debug log emitted when C-index scoring fails."""
        import logging
        from unittest.mock import patch
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, y_test = train_test_data
        model = CoxNetModel(l1_ratio=0.5, alpha='auto', cv_folds=3)

        with caplog.at_level(logging.DEBUG, logger="mlsurv.shared"):
            with patch(
                'mlsurv._shared.concordance_index_censored',
                side_effect=ValueError("all tied"),
            ):
                model.fit(X_train, y_train)

        assert any(
            "C-index scoring failed" in msg and "all tied" in msg
            for msg in caplog.messages
        )

    # --- Prediction overflow clamping tests ---

    def _fit_and_inject_extreme_predict(self, train_test_data):
        """Helper: fit a CoxNet model and replace predict with extreme LPs."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, _ = train_test_data
        model = CoxNetModel(l1_ratio=0.5, fit_baseline_model=True)
        model.fit(X_train, y_train)

        real_predict = model.model_.predict

        def _extreme_predict(X, **kwargs):
            lp = real_predict(X, **kwargs)
            lp[0] = 1000.0   # way beyond exp() range
            lp[1] = -1000.0
            return lp

        model.model_.predict = _extreme_predict
        return model, X_test

    def test_predict_survival_clamps_extreme_lp(self, train_test_data):
        """predict_survival_function clamps extreme LPs without overflow."""
        import warnings as _warnings
        from mlsurv.models._overflow import _OVERFLOW_RE

        model, X_test = self._fit_and_inject_extreme_predict(train_test_data)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            surv = model.predict_survival_function(X_test, return_array=True)

        overflow_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and _OVERFLOW_RE.search(str(w.message))
        ]
        assert overflow_warnings == [], (
            f"Overflow warnings leaked: {[str(w.message) for w in overflow_warnings]}"
        )

        clamp_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "clamped" in str(w.message)
        ]
        assert len(clamp_warnings) == 1
        assert "2 linear predictor(s)" in str(clamp_warnings[0].message)
        assert "CoxNet" in str(clamp_warnings[0].message)

        assert np.all(np.isfinite(surv)), "Survival predictions contain NaN or inf"
        assert surv.min() >= 0.0
        assert surv.max() <= 1.0

    def test_predict_cumulative_hazard_clamps_extreme_lp(self, train_test_data):
        """predict_cumulative_hazard_function clamps extreme LPs without overflow."""
        import warnings as _warnings
        from mlsurv.models._overflow import _OVERFLOW_RE

        model, X_test = self._fit_and_inject_extreme_predict(train_test_data)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            chf = model.predict_cumulative_hazard_function(
                X_test, return_array=True
            )

        overflow_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and _OVERFLOW_RE.search(str(w.message))
        ]
        assert overflow_warnings == [], (
            f"Overflow warnings leaked: {[str(w.message) for w in overflow_warnings]}"
        )

        clamp_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "clamped" in str(w.message)
        ]
        assert len(clamp_warnings) == 1
        assert "2 linear predictor(s)" in str(clamp_warnings[0].message)
        assert "CoxNet" in str(clamp_warnings[0].message)

        assert np.all(np.isfinite(chf)), "CHF predictions contain NaN or inf"
        assert chf.min() >= 0.0

    def test_predict_with_clamp_restores_predict_on_error(self, train_test_data):
        """model_.predict is restored even when the inner call raises."""
        from mlsurv.models import CoxNetModel

        X_train, X_test, y_train, _ = train_test_data
        model = CoxNetModel(l1_ratio=0.5, fit_baseline_model=True)
        model.fit(X_train, y_train)

        expected_lp = model.model_.predict(X_test).copy()

        real_sf = model.model_.predict_survival_function
        def _raise_on_sf(*args, **kwargs):
            raise TypeError("injected error")
        model.model_.predict_survival_function = _raise_on_sf

        with pytest.raises(TypeError, match="injected error"):
            model.predict_survival_function(X_test)

        restored_lp = model.model_.predict(X_test)
        np.testing.assert_array_equal(restored_lp, expected_lp)

        model.model_.predict_survival_function = real_sf


class TestRSFModel:
    """Test RSFModel wrapper."""

    def test_fit_predict(self, train_test_data):
        """Verify fit and predict work correctly."""
        from mlsurv.models import RSFModel

        X_train, X_test, y_train, y_test = train_test_data

        model = RSFModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        assert model.is_fitted_

        risk_scores = model.predict(X_test)
        assert len(risk_scores) == len(X_test)

    def test_feature_importances_deprecated(self, train_test_data):
        """Verify impurity-based importance raises NotImplementedError (deprecated in sksurv>=0.26)."""
        from mlsurv.models import RSFModel

        X_train, X_test, y_train, y_test = train_test_data

        model = RSFModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # scikit-survival >= 0.26 deprecated impurity-based importance
        with pytest.raises(NotImplementedError):
            _ = model.feature_importances_

    def test_get_feature_importance_permutation(self, train_test_data):
        """Verify permutation-based feature importance works."""
        from mlsurv.models import RSFModel

        X_train, X_test, y_train, y_test = train_test_data

        model = RSFModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Use permutation importance (recommended approach)
        importance_df = model.get_feature_importance(
            method='permutation', X=X_test, y=y_test, n_repeats=5
        )
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns

    def test_oob_score_default_off(self, train_test_data):
        """Default oob_score=False — fitted inner model has no oob_score_."""
        from mlsurv.models import RSFModel

        X_train, _, y_train, _ = train_test_data
        model = RSFModel(n_estimators=20, random_state=42).fit(X_train, y_train)

        # sksurv does not populate oob_score_ when oob_score=False
        assert not hasattr(model.model_, 'oob_score_')

    def test_oob_score_enabled(self, train_test_data):
        """oob_score=True exposes oob_score_ as a finite float in [0, 1]."""
        import math

        from mlsurv.models import RSFModel

        X_train, _, y_train, _ = train_test_data
        model = RSFModel(n_estimators=30, oob_score=True, random_state=42).fit(
            X_train, y_train
        )

        oob = model.model_.oob_score_
        assert isinstance(oob, float)
        assert math.isfinite(oob)
        assert 0.0 <= oob <= 1.0

    def test_oob_score_without_bootstrap_raises(self, train_test_data):
        """oob_score=True with bootstrap=False raises a clear ValueError."""
        from mlsurv.models import RSFModel

        X_train, _, y_train, _ = train_test_data
        model = RSFModel(
            n_estimators=10,
            oob_score=True,
            bootstrap=False,
            random_state=42,
        )
        with pytest.raises(
            ValueError, match=r"oob_score=True requires bootstrap=True"
        ):
            model.fit(X_train, y_train)

    def test_oob_score_in_trained_details_report(self, train_test_data):
        """_add_rsf_details renders 'OOB concordance' when oob_score=True."""
        from mlsurv.models import RSFModel
        from mlsurv.reports.sections._trained_details import _TrainedDetailsSubsection

        X_train, _, y_train, _ = train_test_data
        model = RSFModel(n_estimators=30, oob_score=True, random_state=42).fit(
            X_train, y_train
        )

        md_lines: list = []
        _TrainedDetailsSubsection._add_rsf_details(md_lines, model)

        rendered = "\n".join(md_lines)
        assert "OOB concordance" in rendered
        assert "trees" in rendered

    def test_oob_score_via_get_model_factory(self):
        """get_model('rsf', oob_score=True) forwards the flag through **kwargs."""
        from mlsurv.models import get_model

        model = get_model('rsf', oob_score=True, n_estimators=10, random_state=42)
        assert model.oob_score is True
        assert model.get_params()['oob_score'] is True

    def test_oob_score_via_survival_learner(self, train_test_data):
        """SurvivalLearner.train('rsf', oob_score=True) exposes oob_score_ end-to-end."""
        import math

        from mlsurv import SurvivalLearner

        X_train, _, y_train, _ = train_test_data
        learner = SurvivalLearner(X_train, y_train)
        learner.setup(
            feature_selector=None,
            test_fraction=0.25,
            cv_folds=3,
            random_state=42,
            n_jobs=1,
        )
        result = learner.train(
            'rsf', tune=False, n_estimators=20, oob_score=True
        )

        # Unwrap Pipeline -> RSFModel -> underlying RandomSurvivalForest
        rsf_wrapper = result.model.named_steps['model']
        assert rsf_wrapper.oob_score is True
        oob = rsf_wrapper.model_.oob_score_
        assert math.isfinite(oob)
        assert 0.0 <= oob <= 1.0

    def test_oob_score_n_estimators_one_boundary(self, train_test_data):
        """Boundary: n_estimators=1 + oob_score=True must not crash silently.

        Per CONTRIBUTING.md boundary value testing rule. With a single tree,
        sksurv still computes an OOB score (on samples not in that tree's
        bootstrap sample). Verify it either yields a finite score or raises
        cleanly — never a silent NaN that leaks downstream.
        """
        import math

        from mlsurv.models import RSFModel

        X_train, _, y_train, _ = train_test_data
        model = RSFModel(
            n_estimators=1, oob_score=True, random_state=42
        ).fit(X_train, y_train)

        # sksurv populates oob_score_ even with a single tree
        assert hasattr(model.model_, 'oob_score_')
        oob = model.model_.oob_score_
        # Must be finite — NaN would be a silent data loss that leaks into
        # the Trained Model Details report
        assert math.isfinite(oob), (
            f"n_estimators=1 produced non-finite oob_score_={oob!r}; "
            "downstream report rendering would display a NaN silently."
        )
        assert 0.0 <= oob <= 1.0


class TestGBSAModel:
    """Test GBSAModel wrapper."""

    def test_fit_predict(self, train_test_data):
        """Verify fit and predict work correctly."""
        from mlsurv.models import GBSAModel

        X_train, X_test, y_train, y_test = train_test_data

        model = GBSAModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        assert model.is_fitted_

        risk_scores = model.predict(X_test)
        assert len(risk_scores) == len(X_test)

    def test_different_loss_functions(self, train_test_data):
        """Verify different loss functions work."""
        from mlsurv.models import GBSAModel

        X_train, X_test, y_train, y_test = train_test_data

        # Note: 'ipcwls' loss has strict data requirements (IPC weights calculation)
        # and may fail on synthetic data, so we test only coxph and squared
        for loss in ['coxph', 'squared']:
            model = GBSAModel(loss=loss, n_estimators=5, random_state=42)
            model.fit(X_train, y_train)
            risk_scores = model.predict(X_test)
            assert len(risk_scores) == len(X_test)

    def test_early_stopping_params_passed_through(self, train_test_data):
        """Early stopping params are forwarded to the sksurv model."""
        from mlsurv.models import GBSAModel

        X_train, _, y_train, _ = train_test_data

        model = GBSAModel(
            n_estimators=50, n_iter_no_change=10,
            validation_fraction=0.1, random_state=42,
        )
        model.fit(X_train, y_train)

        assert model.model_.n_iter_no_change == 10
        assert model.model_.validation_fraction == 0.1

    def test_early_stopping_disabled_when_none(self, train_test_data):
        """n_iter_no_change=None disables early stopping."""
        from mlsurv.models import GBSAModel

        X_train, _, y_train, _ = train_test_data

        model = GBSAModel(
            n_estimators=10, n_iter_no_change=None, random_state=42,
        )
        model.fit(X_train, y_train)

        # sksurv default when not passed: None
        assert model.model_.n_iter_no_change is None

    def test_early_stopping_skipped_on_small_data(self):
        """Early stopping should be disabled with a warning on very small data."""
        import warnings as _warnings
        from mlsurv.models import GBSAModel

        rng = np.random.RandomState(42)
        X_tiny = pd.DataFrame(rng.randn(10, 3), columns=['a', 'b', 'c'])
        y_tiny = np.array(
            [(True, t) for t in rng.exponential(5, 6)]
            + [(False, t) for t in rng.exponential(5, 4)],
            dtype=[('event', bool), ('time', float)],
        )

        model = GBSAModel(
            n_estimators=10, n_iter_no_change=5, random_state=42,
        )
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            model.fit(X_tiny, y_tiny)
            es_warnings = [x for x in w if "too small" in str(x.message)]
            assert len(es_warnings) == 1
        # Early stopping should have been disabled
        assert model.model_.n_iter_no_change is None

    def test_invalid_validation_fraction_raises(self, train_test_data):
        """Out-of-range validation_fraction should raise ValueError."""
        from mlsurv.models import GBSAModel

        X_train, _, y_train, _ = train_test_data

        for bad_vf in [0.0, 1.0, -0.1, 1.5]:
            model = GBSAModel(
                n_estimators=10, n_iter_no_change=5,
                validation_fraction=bad_vf, random_state=42,
            )
            with pytest.raises(ValueError, match="validation_fraction must be in"):
                model.fit(X_train, y_train)


@pytest.mark.integration
class TestFSSVMModel:
    """Test FSSVMModel wrapper."""

    def test_fit_predict(self, train_test_data):
        """Verify fit and predict work correctly."""
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        model = FSSVMModel(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)

        assert model.is_fitted_

        risk_scores = model.predict(X_test)
        assert len(risk_scores) == len(X_test)

    def test_rank_ratio_output_normalization(self, train_test_data):
        """Verify output normalization for different rank_ratios."""
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, y_test = train_test_data

        # Pure ranking mode
        model_rank = FSSVMModel(rank_ratio=1.0, random_state=42)
        model_rank.fit(X_train, y_train)
        scores_rank = model_rank.predict(X_test)

        # Mixed mode (should negate output)
        model_mixed = FSSVMModel(rank_ratio=0.5, random_state=42)
        model_mixed.fit(X_train, y_train)
        scores_mixed = model_mixed.predict(X_test)

        # Both should produce valid risk scores
        assert not np.isnan(scores_rank).any()
        assert not np.isnan(scores_mixed).any()


class TestFSSVMReturnArray:
    """Test FSSVM predict_survival_function / predict_cumulative_hazard_function return_array."""

    def test_predict_survival_return_array(self, train_test_data):
        """return_array=True produces (n_samples, n_times) ndarray."""
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, _ = train_test_data
        model = FSSVMModel(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)

        surv = model.predict_survival_function(X_test, return_array=True)
        assert isinstance(surv, np.ndarray)
        assert surv.shape[0] == len(X_test)
        assert surv.ndim == 2

    def test_predict_cumulative_hazard_return_array(self, train_test_data):
        """return_array=True produces (n_samples, n_times) ndarray for CHF."""
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, _ = train_test_data
        model = FSSVMModel(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)

        chf = model.predict_cumulative_hazard_function(X_test, return_array=True)
        assert isinstance(chf, np.ndarray)
        assert chf.shape[0] == len(X_test)
        assert chf.ndim == 2


class TestXGBCoxOverflow:
    """Test XGBCox Breslow overflow protection."""

    def _fit_and_inject_extreme_scores(self, train_test_data):
        """Helper: fit an XGBCox model and inject extreme risk scores."""
        from mlsurv.models import XGBCoxModel

        X_train, X_test, y_train, _ = train_test_data
        model = XGBCoxModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        real_predict = model.model_.predict

        def _extreme_predict(X, **kwargs):
            scores = real_predict(X, **kwargs)
            scores[0] = 1000.0
            scores[1] = -1000.0
            return scores

        model.model_.predict = _extreme_predict
        return model, X_test

    def test_predict_survival_clamps_extreme_scores(self, train_test_data):
        """predict_survival_function clamps extreme risk scores without overflow."""
        import warnings as _warnings
        from mlsurv.models._overflow import _OVERFLOW_RE

        model, X_test = self._fit_and_inject_extreme_scores(train_test_data)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            model.predict_survival_function(X_test)

        overflow_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and _OVERFLOW_RE.search(str(w.message))
        ]
        assert overflow_warnings == [], (
            f"Overflow warnings leaked: {[str(w.message) for w in overflow_warnings]}"
        )

        clamp_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "clamped" in str(w.message)
        ]
        assert len(clamp_warnings) == 1
        assert "XGBCox" in str(clamp_warnings[0].message)

    def test_predict_cumulative_hazard_clamps_extreme_scores(self, train_test_data):
        """predict_cumulative_hazard_function clamps extreme risk scores."""
        import warnings as _warnings
        from mlsurv.models._overflow import _OVERFLOW_RE

        model, X_test = self._fit_and_inject_extreme_scores(train_test_data)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            model.predict_cumulative_hazard_function(X_test)

        overflow_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and _OVERFLOW_RE.search(str(w.message))
        ]
        assert overflow_warnings == [], (
            f"Overflow warnings leaked: {[str(w.message) for w in overflow_warnings]}"
        )

        clamp_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "clamped" in str(w.message)
        ]
        assert len(clamp_warnings) == 1
        assert "XGBCox" in str(clamp_warnings[0].message)


@pytest.mark.integration
class TestFSSVMOverflow:
    """Test FSSVM Breslow overflow protection."""

    def _fit_and_inject_extreme_scores(self, train_test_data):
        """Helper: fit an FSSVM model and inject extreme risk scores."""
        from mlsurv.models import FSSVMModel

        X_train, X_test, y_train, _ = train_test_data
        model = FSSVMModel(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)

        real_predict = model.predict

        def _extreme_predict(X):
            scores = real_predict(X)
            scores[0] = 1000.0
            scores[1] = -1000.0
            return scores

        model.predict = _extreme_predict
        return model, X_test

    def test_predict_survival_clamps_extreme_scores(self, train_test_data):
        """predict_survival_function clamps extreme risk scores without overflow."""
        import warnings as _warnings
        from mlsurv.models._overflow import _OVERFLOW_RE

        model, X_test = self._fit_and_inject_extreme_scores(train_test_data)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            model.predict_survival_function(X_test)

        overflow_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and _OVERFLOW_RE.search(str(w.message))
        ]
        assert overflow_warnings == [], (
            f"Overflow warnings leaked: {[str(w.message) for w in overflow_warnings]}"
        )

        clamp_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "clamped" in str(w.message)
        ]
        assert len(clamp_warnings) == 1
        assert "FSSVM" in str(clamp_warnings[0].message)

    def test_predict_cumulative_hazard_clamps_extreme_scores(self, train_test_data):
        """predict_cumulative_hazard_function clamps extreme risk scores."""
        import warnings as _warnings
        from mlsurv.models._overflow import _OVERFLOW_RE

        model, X_test = self._fit_and_inject_extreme_scores(train_test_data)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            model.predict_cumulative_hazard_function(X_test)

        overflow_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and _OVERFLOW_RE.search(str(w.message))
        ]
        assert overflow_warnings == [], (
            f"Overflow warnings leaked: {[str(w.message) for w in overflow_warnings]}"
        )

        clamp_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "clamped" in str(w.message)
        ]
        assert len(clamp_warnings) == 1
        assert "FSSVM" in str(clamp_warnings[0].message)


class TestBreslowCallWithClamp:
    """Test breslow_call_with_clamp helper directly."""

    def test_emits_warning_when_clamping(self):
        """breslow_call_with_clamp emits UserWarning when values are clamped."""
        import warnings as _warnings
        from mlsurv.models._overflow import breslow_call_with_clamp

        extreme_scores = np.array([1.0, 2.0, 1000.0, -1000.0])

        def fake_breslow_fn(scores):
            return scores  # identity

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            result = breslow_call_with_clamp(
                extreme_scores, fake_breslow_fn, model_name="TestModel"
            )

        clamp_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "clamped" in str(w.message)
        ]
        assert len(clamp_warnings) == 1
        assert "2 risk score(s)" in str(clamp_warnings[0].message)
        assert "TestModel" in str(clamp_warnings[0].message)

        # Verify scores were actually clamped
        assert result[2] == 20.0
        assert result[3] == -20.0
        # Non-extreme scores unchanged
        assert result[0] == 1.0
        assert result[1] == 2.0

    def test_no_warning_when_no_clamping(self):
        """breslow_call_with_clamp emits no warning when values are in range."""
        import warnings as _warnings
        from mlsurv.models._overflow import breslow_call_with_clamp

        normal_scores = np.array([1.0, -2.0, 15.0])

        def fake_breslow_fn(scores):
            return scores

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            result = breslow_call_with_clamp(
                normal_scores, fake_breslow_fn, model_name="TestModel"
            )

        clamp_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "clamped" in str(w.message)
        ]
        assert len(clamp_warnings) == 0
        np.testing.assert_array_equal(result, normal_scores)

    def test_suppresses_overflow_warnings(self):
        """breslow_call_with_clamp suppresses overflow RuntimeWarnings."""
        import warnings as _warnings
        from mlsurv.models._overflow import breslow_call_with_clamp, _OVERFLOW_RE

        scores = np.array([1.0, 2.0])

        def fake_breslow_fn(s):
            _warnings.warn("overflow encountered in exp", RuntimeWarning)
            return s

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            breslow_call_with_clamp(scores, fake_breslow_fn, model_name="Test")

        overflow_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and _OVERFLOW_RE.search(str(w.message))
        ]
        assert overflow_warnings == []

    def test_reemits_warnings_on_exception(self):
        """Non-overflow warnings captured before an exception are re-emitted."""
        import warnings as _warnings
        from mlsurv.models._overflow import breslow_call_with_clamp

        scores = np.array([1.0, 2.0])

        def _warn_then_raise(s):
            _warnings.warn("convergence issue", UserWarning)
            raise ValueError("fit failed")

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            with pytest.raises(ValueError, match="fit failed"):
                breslow_call_with_clamp(
                    scores, _warn_then_raise, model_name="Test"
                )

        reemitted = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "convergence" in str(w.message)
        ]
        assert len(reemitted) == 1, (
            f"Expected 1 re-emitted warning, got {len(reemitted)}"
        )

    def test_forwards_args_and_kwargs(self):
        """breslow_call_with_clamp correctly forwards positional and keyword args."""
        from mlsurv.models._overflow import breslow_call_with_clamp

        received = {}

        def fake_breslow_fn(scores, event, time, extra=None):
            received['scores'] = scores
            received['event'] = event
            received['time'] = time
            received['extra'] = extra
            return scores

        scores = np.array([1.0, 2.0])
        event = np.array([True, False])
        time = np.array([10.0, 20.0])

        breslow_call_with_clamp(
            scores, fake_breslow_fn, event, time,
            model_name="Test", extra="kwarg_val",
        )

        np.testing.assert_array_equal(received['scores'], scores)
        np.testing.assert_array_equal(received['event'], event)
        np.testing.assert_array_equal(received['time'], time)
        assert received['extra'] == "kwarg_val"

    def test_reemits_non_overflow_warnings(self):
        """Non-overflow warnings from breslow_fn pass through on success."""
        import warnings as _warnings
        from mlsurv.models._overflow import breslow_call_with_clamp, _OVERFLOW_RE

        scores = np.array([1.0, 2.0])

        def fake_breslow_fn(s):
            _warnings.warn("overflow encountered in exp", RuntimeWarning)
            _warnings.warn("convergence issue", FutureWarning)
            return s

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            breslow_call_with_clamp(scores, fake_breslow_fn, model_name="Test")

        # Overflow warning should be suppressed
        overflow = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and _OVERFLOW_RE.search(str(w.message))
        ]
        assert overflow == []

        # Non-overflow FutureWarning should be re-emitted
        future = [
            w for w in caught
            if issubclass(w.category, FutureWarning)
            and "convergence" in str(w.message)
        ]
        assert len(future) == 1

    def test_errstate_suppresses_real_numpy_overflow(self):
        """np.errstate prevents real numpy-level overflow warnings from leaking."""
        import warnings as _warnings
        from mlsurv.models._overflow import breslow_call_with_clamp

        scores = np.array([1.0, 2.0])

        def breslow_fn_with_real_overflow(s):
            # Trigger a real numpy overflow (not a manual warnings.warn)
            np.exp(np.float64(1000.0))
            return s

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            breslow_call_with_clamp(
                scores, breslow_fn_with_real_overflow, model_name="Test"
            )

        overflow_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "overflow" in str(w.message).lower()
        ]
        assert overflow_warnings == [], (
            f"numpy overflow warning leaked: "
            f"{[str(w.message) for w in overflow_warnings]}"
        )


class TestLpClipValue:
    """Canary test for _LP_CLIP constant."""

    def test_lp_clip_value(self):
        """_LP_CLIP must be 20 to ensure Breslow numerical stability."""
        from mlsurv.models._overflow import _LP_CLIP

        assert _LP_CLIP == 20


class TestModelRegistry:
    """Test model registry functions."""

    def test_get_model(self):
        """Verify get_model returns correct model type."""
        from mlsurv.models import get_model, CoxPHModel, RSFModel

        model = get_model('coxph', alpha=0.1)
        assert isinstance(model, CoxPHModel)
        assert model.alpha == 0.1

        model = get_model('rsf', n_estimators=50)
        assert isinstance(model, RSFModel)
        assert model.n_estimators == 50

    def test_get_model_unknown(self):
        """Verify get_model raises error for unknown model."""
        from mlsurv.models import get_model

        with pytest.raises(ValueError, match="Unknown model"):
            get_model('unknown_model')

    def test_get_model_new_models(self):
        """get_model works for the newer models (xgbcox/xgbaft/weibull_aft/
        deephit). Folded in from test_new_models.py::TestNewModelsIntegration.
        """
        from mlsurv.models import get_model

        for name in ('xgbcox', 'xgbaft', 'weibull_aft'):
            model = get_model(name)
            assert model.name == name

        # DeepHit needs torch/pycox — tolerate its absence.
        try:
            model = get_model('deephit')
            assert model.name == 'deephit'
        except ImportError:
            pass

    def test_list_models(self):
        """Verify list_models can include optional deep models."""
        from mlsurv.models import list_models

        models = list_models(include_unavailable=True)
        assert 'coxph' in models
        assert 'coxnet' in models
        assert 'rsf' in models
        assert 'gbsa' in models
        assert 'fssvm' in models
        assert 'deepsurv' in models
        # Newer models — folded in from test_new_models.py.
        assert 'xgbcox' in models
        assert 'xgbaft' in models
        assert 'weibull_aft' in models
        assert 'deephit' in models


class TestSklearnCompatibility:
    """Test sklearn compatibility of model wrappers."""

    def test_clone(self, train_test_data):
        """Verify models can be cloned."""
        from sklearn.base import clone
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data

        model = CoxPHModel(alpha=0.1)
        model.fit(X_train, y_train)

        model_clone = clone(model)
        assert not getattr(model_clone, 'is_fitted_', False)
        assert model_clone.alpha == model.alpha

    def test_pipeline_integration(self, train_test_data):
        """Verify models work in sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from mlsurv.models import CoxPHModel

        X_train, X_test, y_train, y_test = train_test_data

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', CoxPHModel())
        ])

        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)

        assert 0.0 <= score <= 1.0


class TestDeepSurvModel:
    """Test DeepSurvModel wrapper (requires torch/pycox)."""

    @pytest.mark.skipif(
        not _has_torch_pycox(),
        reason="torch, pycox, and torchtuples required"
    )
    def test_fit_predict(self, train_test_data):
        """Verify fit and predict work correctly."""
        from mlsurv.models.deepsurv import DeepSurvModel

        X_train, X_test, y_train, y_test = train_test_data

        model = DeepSurvModel(
            layers=[32, 32],
            epochs=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        assert model.is_fitted_

        risk_scores = model.predict(X_test)
        assert len(risk_scores) == len(X_test)
        assert not np.isnan(risk_scores).any()

    @pytest.mark.skipif(
        not _has_torch_pycox(),
        reason="torch, pycox, and torchtuples required"
    )
    def test_predict_survival_function(self, train_test_data):
        """Verify survival function prediction."""
        from mlsurv.models.deepsurv import DeepSurvModel

        X_train, X_test, y_train, y_test = train_test_data

        model = DeepSurvModel(
            layers=[32, 32],
            epochs=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        surv_df = model.predict_survival_function(X_test)

        assert surv_df is not None
        assert surv_df.shape[1] == len(X_test)
        # Survival probabilities should be between 0 and 1
        assert (surv_df.values >= 0).all()
        assert (surv_df.values <= 1).all()

    @pytest.mark.skipif(
        not _has_torch_pycox(),
        reason="torch, pycox, and torchtuples required"
    )
    def test_predict_cumulative_hazard_function(self, train_test_data):
        """Verify cumulative hazard function prediction."""
        from mlsurv.models.deepsurv import DeepSurvModel

        X_train, X_test, y_train, y_test = train_test_data

        model = DeepSurvModel(
            layers=[32, 32],
            epochs=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        cum_haz_df = model.predict_cumulative_hazard_function(X_test)

        assert cum_haz_df is not None
        assert cum_haz_df.shape[1] == len(X_test)
        # Cumulative hazard should be non-negative
        assert (cum_haz_df.values >= 0).all()

    @pytest.mark.skipif(
        not _has_torch_pycox(),
        reason="torch, pycox, and torchtuples required"
    )
    def test_score(self, train_test_data):
        """Verify score returns valid C-index."""
        from mlsurv.models.deepsurv import DeepSurvModel

        X_train, X_test, y_train, y_test = train_test_data

        model = DeepSurvModel(
            layers=[32, 32],
            epochs=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        c_index = model.score(X_test, y_test)
        assert 0.0 <= c_index <= 1.0

    @pytest.mark.skipif(
        not _has_torch_pycox(),
        reason="torch, pycox, and torchtuples required"
    )
    def test_hyperparameters(self, train_test_data):
        """Verify hyperparameters are respected."""
        from mlsurv.models.deepsurv import DeepSurvModel

        X_train, X_test, y_train, y_test = train_test_data

        model = DeepSurvModel(
            layers=[64, 32, 16],
            dropout=0.2,
            batch_norm=True,
            learning_rate=0.001,
            batch_size=128,
            epochs=5,
            random_state=42
        )

        assert model.layers == [64, 32, 16]
        assert model.dropout == 0.2
        assert model.batch_norm is True
        assert model.learning_rate == 0.001
        assert model.batch_size == 128
        assert model.epochs == 5

        model.fit(X_train, y_train)
        assert model.is_fitted_

    def test_import_error_without_dependencies(self):
        """Verify helpful error when torch/pycox missing."""
        from mlsurv.models.deepsurv import DeepSurvModel

        model = DeepSurvModel()

        if not _has_torch_pycox():
            # Should raise ImportError with helpful message
            with pytest.raises(ImportError, match="torch|pycox"):
                model._check_dependencies()
        else:
            # If dependencies exist, check should pass
            model._check_dependencies()

    @pytest.mark.skipif(
        not _has_torch_pycox(),
        reason="torch, pycox, and torchtuples required"
    )
    def test_not_fitted_error(self):
        """Verify error when predicting before fit."""
        from sklearn.exceptions import NotFittedError
        from mlsurv.models.deepsurv import DeepSurvModel

        model = DeepSurvModel()

        with pytest.raises(NotFittedError, match="not fitted"):
            model.predict(np.random.randn(10, 5))

    @pytest.mark.skipif(
        not _has_torch_pycox(),
        reason="torch, pycox, and torchtuples required"
    )
    def test_early_stopping_uses_validation(self, train_test_data):
        """DeepSurv with patience splits data and produces valid predictions."""
        from mlsurv.models.deepsurv import DeepSurvModel

        X_train, X_test, y_train, y_test = train_test_data

        model = DeepSurvModel(
            layers=[16], epochs=200, patience=5,
            val_fraction=0.2, random_state=42,
        )
        model.fit(X_train, y_train)

        assert model.is_fitted_
        risk_scores = model.predict(X_test)
        assert len(risk_scores) == len(X_test)
        assert not np.any(np.isnan(risk_scores))

        # Verify survival function works with training-split baseline hazard
        surv_df = model.predict_survival_function(X_test)
        assert surv_df.shape[1] == len(X_test)
        assert (surv_df.values >= 0).all()
        assert (surv_df.values <= 1).all()

        # Verify early stopping actually fired (trained < max epochs)
        log = model.model_.log.to_pandas()
        assert len(log) < 200, (
            f"Expected early stopping before epoch 200, "
            f"but trained for {len(log)} epochs"
        )

    @pytest.mark.skipif(
        not _has_torch_pycox(),
        reason="torch, pycox, and torchtuples required"
    )
    def test_no_val_split_when_patience_none(self, train_test_data):
        """patience=None disables early stopping and val split entirely."""
        from mlsurv.models.deepsurv import DeepSurvModel

        X_train, X_test, y_train, y_test = train_test_data

        model = DeepSurvModel(
            layers=[16], epochs=10, patience=None, random_state=42,
        )
        model.fit(X_train, y_train)

        assert model.is_fitted_
        risk_scores = model.predict(X_test)
        assert not np.any(np.isnan(risk_scores))

    @pytest.mark.skipif(
        not _has_torch_pycox(),
        reason="torch, pycox, and torchtuples required"
    )
    def test_small_dataset_disables_early_stopping(self):
        """Tiny dataset emits warning and disables early stopping."""
        from mlsurv.models.deepsurv import DeepSurvModel

        # 8 samples, val_fraction=0.2 → n_val=1 (<2) → warning
        # Need 5+ events so StratifiedSurvivalSampler doesn't error
        rng = np.random.RandomState(42)
        X_tiny = rng.randn(8, 3)
        y_tiny = np.array(
            [(True, 1.0), (True, 2.0), (True, 3.0), (True, 4.0),
             (True, 5.0), (False, 6.0), (False, 7.0), (False, 8.0)],
            dtype=[('event', bool), ('time', float)],
        )

        model = DeepSurvModel(
            layers=[8], epochs=5, patience=3,
            val_fraction=0.2, random_state=42,
        )

        with pytest.warns(SuggestionWarning, match="Dataset too small"):
            model.fit(X_tiny, y_tiny)

        assert model.is_fitted_

    @pytest.mark.skipif(
        not _has_torch_pycox(),
        reason="torch, pycox, and torchtuples required"
    )
    def test_val_fraction_invalid_raises(self):
        """val_fraction outside (0.0, 1.0) raises ValueError."""
        from mlsurv.models.deepsurv import DeepSurvModel

        rng = np.random.RandomState(0)
        X = rng.randn(50, 3)
        y = np.array(
            [(bool(i % 2), float(i + 1)) for i in range(50)],
            dtype=[('event', bool), ('time', float)],
        )

        for bad_frac in (0.0, 1.0, -0.5, 1.5):
            model = DeepSurvModel(
                layers=[8], epochs=5, patience=3,
                val_fraction=bad_frac,
            )
            with pytest.raises(ValueError, match="val_fraction must be in"):
                model.fit(X, y)

    @pytest.mark.skipif(
        not _has_torch_pycox(),
        reason="torch, pycox, and torchtuples required"
    )
    def test_stratified_split_failure_fallback(self, train_test_data):
        """Stratified split ValueError falls back to no early stopping."""
        from unittest.mock import patch
        from mlsurv.models.deepsurv import DeepSurvModel

        X_train, X_test, y_train, y_test = train_test_data

        model = DeepSurvModel(
            layers=[8], epochs=5, patience=3,
            val_fraction=0.2, random_state=42,
        )

        with patch(
            'sklearn.model_selection.train_test_split',
            side_effect=ValueError(
                "The least populated class in y has only 1 member"
            ),
        ):
            with pytest.warns(SuggestionWarning, match="Stratified val split failed"):
                model.fit(X_train, y_train)

        assert model.is_fitted_
        risk_scores = model.predict(X_test)
        assert not np.any(np.isnan(risk_scores))

    @pytest.mark.skipif(
        not _has_torch_pycox(),
        reason="torch, pycox, and torchtuples required"
    )
    def test_non_stratification_valueerror_propagates(self, train_test_data):
        """Non-stratification ValueError from train_test_split is re-raised."""
        from unittest.mock import patch
        from mlsurv.models.deepsurv import DeepSurvModel

        X_train, X_test, y_train, y_test = train_test_data

        model = DeepSurvModel(
            layers=[8], epochs=5, patience=3,
            val_fraction=0.2, random_state=42,
        )

        with patch(
            'sklearn.model_selection.train_test_split',
            side_effect=ValueError("some completely different error"),
        ):
            with pytest.raises(ValueError, match="some completely different error"):
                model.fit(X_train, y_train)

    @pytest.mark.skipif(
        not _has_torch_pycox(),
        reason="torch, pycox, and torchtuples required"
    )
    def test_checkpoint_cleanup_after_fit(self, train_test_data):
        """Checkpoint .pt files are cleaned up after fit with early stopping."""
        import tempfile
        from pathlib import Path
        from mlsurv.models.deepsurv import DeepSurvModel

        X_train, X_test, y_train, y_test = train_test_data

        ckpt_dir = Path(tempfile.gettempdir()) / "mlsurv_checkpoints"
        before = set(ckpt_dir.glob("deepsurv_*.pt")) if ckpt_dir.exists() else set()

        model = DeepSurvModel(
            layers=[32, 32], epochs=10, patience=3,
            val_fraction=0.2, random_state=42,
        )
        model.fit(X_train, y_train)

        after = set(ckpt_dir.glob("deepsurv_*.pt")) if ckpt_dir.exists() else set()
        new_files = after - before
        assert new_files == set(), f"Orphan checkpoint files: {new_files}"

    @pytest.mark.skipif(
        not _has_torch_pycox(),
        reason="torch, pycox, and torchtuples required"
    )
    def test_checkpoint_cleanup_on_training_failure(self, train_test_data):
        """Checkpoint .pt files are cleaned up even when fit() raises."""
        from unittest.mock import patch
        import tempfile
        from pathlib import Path
        from mlsurv.models.deepsurv import DeepSurvModel

        X_train, X_test, y_train, y_test = train_test_data

        ckpt_dir = Path(tempfile.gettempdir()) / "mlsurv_checkpoints"
        before = set(ckpt_dir.glob("deepsurv_*.pt")) if ckpt_dir.exists() else set()

        model = DeepSurvModel(
            layers=[32, 32], epochs=10, patience=3,
            val_fraction=0.2, random_state=42,
        )

        with patch(
            'pycox.models.CoxPH.fit_dataloader',
            side_effect=RuntimeError("simulated crash"),
        ):
            with pytest.raises(RuntimeError, match="simulated crash"):
                model.fit(X_train, y_train)

        after = set(ckpt_dir.glob("deepsurv_*.pt")) if ckpt_dir.exists() else set()
        new_files = after - before
        assert new_files == set(), f"Orphan checkpoint files: {new_files}"

    def test_stratified_sampler_negative_censored(self):
        """StratifiedSurvivalSampler handles min_events > batch_size gracefully."""
        from mlsurv.models.deepsurv import StratifiedSurvivalSampler

        # 20 events, 10 censored — min_events_per_batch=5 > batch_size=4
        events = np.array([1]*20 + [0]*10)
        sampler = StratifiedSurvivalSampler(
            events, batch_size=4, min_events_per_batch=5, random_state=42
        )

        assert sampler.censored_per_batch >= 0, (
            f"censored_per_batch should be >= 0, got {sampler.censored_per_batch}"
        )

        # Iteration should not crash
        batches = list(sampler)
        assert len(batches) > 0


class TestDeepSurvReturnArray:
    """Test DeepSurv return_array parameter for survival/CHF predictions."""

    @pytest.mark.skipif(
        not _has_torch_pycox(),
        reason="torch, pycox, and torchtuples required"
    )
    def test_predict_survival_return_array(self, train_test_data):
        """return_array=True produces (n_samples, n_times) ndarray."""
        from mlsurv.models.deepsurv import DeepSurvModel

        X_train, X_test, y_train, _ = train_test_data
        model = DeepSurvModel(layers=[16], epochs=5, random_state=42)
        model.fit(X_train, y_train)

        surv = model.predict_survival_function(X_test, return_array=True)
        assert isinstance(surv, np.ndarray)
        assert surv.shape[0] == len(X_test)
        assert surv.ndim == 2
        assert np.all((surv >= 0) & (surv <= 1))

    @pytest.mark.skipif(
        not _has_torch_pycox(),
        reason="torch, pycox, and torchtuples required"
    )
    def test_predict_cumulative_hazard_return_array(self, train_test_data):
        """return_array=True produces (n_samples, n_times) ndarray for CHF."""
        from mlsurv.models.deepsurv import DeepSurvModel

        X_train, X_test, y_train, _ = train_test_data
        model = DeepSurvModel(layers=[16], epochs=5, random_state=42)
        model.fit(X_train, y_train)

        chf = model.predict_cumulative_hazard_function(X_test, return_array=True)
        assert isinstance(chf, np.ndarray)
        assert chf.shape[0] == len(X_test)
        assert chf.ndim == 2
        assert np.all(chf >= 0)


class TestModelMetadata:
    """Test model metadata helper functions."""

    def test_get_models_with_random_state(self):
        """Verify get_models_with_random_state returns correct set."""
        from mlsurv.models import get_models_with_random_state

        result = get_models_with_random_state()

        # Should include models that support random_state
        assert 'rsf' in result
        assert 'gbsa' in result
        assert 'fssvm' in result
        assert 'xgbcox' in result
        assert 'deepsurv' in result
        assert 'deephit' in result
        # coxnet seeds its internal alpha-selection K-fold CV
        assert 'coxnet' in result

        # Should NOT include models without random_state
        assert 'coxph' not in result
        assert 'weibull_aft' not in result

    def test_get_sequential_cv_models(self):
        """Verify get_sequential_cv_models returns correct set."""
        from mlsurv.models import get_sequential_cv_models

        result = get_sequential_cv_models()

        # Should include fast/GPU models that prefer sequential CV
        assert 'coxph' in result
        assert 'coxnet' in result
        assert 'deepsurv' in result
        assert 'deephit' in result

        # Should NOT include parallelizable models
        assert 'rsf' not in result
        assert 'gbsa' not in result
        assert 'fssvm' not in result

    def test_get_model_colors(self):
        """Verify get_model_colors returns all models with valid colors."""
        from mlsurv.models import get_model_colors

        colors = get_model_colors()

        # Should include all models
        assert 'coxph' in colors
        assert 'rsf' in colors
        assert 'deepsurv' in colors
        assert 'ensemble' in colors
        assert 'baseline' in colors

        # Colors should be valid hex codes
        for name, color in colors.items():
            assert color.startswith('#'), f"{name} color should be hex"
            assert len(color) == 7, f"{name} color should be #RRGGBB format"

    def test_get_model_display_names(self):
        """Verify get_model_display_names returns proper formatting."""
        from mlsurv.models import get_model_display_names

        names = get_model_display_names()

        # Should include all models with proper display names
        assert names['coxph'] == 'CoxPH'
        assert names['coxnet'] == 'CoxNet'
        assert names['rsf'] == 'RSF'
        assert names['xgbcox'] == 'XGBoost-Cox'  # Using survival:cox objective
        assert names['weibull_aft'] == 'Weibull-AFT'
        assert names['deepsurv'] == 'DeepSurv'
        assert names['deephit'] == 'DeepHit'

    def test_get_model_metadata(self):
        """Verify get_model_metadata returns correct dict."""
        from mlsurv.models import get_model_metadata

        # Test registered model
        meta = get_model_metadata('rsf')
        assert meta['display_name'] == 'RSF'
        assert meta['color'] == '#D55E00'
        assert meta['supports_random_state'] is True
        assert meta['prefer_sequential_cv'] is False

        # Test lazy-loaded model
        meta = get_model_metadata('deepsurv')
        assert meta['display_name'] == 'DeepSurv'
        assert meta['supports_random_state'] is True
        assert meta['prefer_sequential_cv'] is True

        # Test unknown model (should return defaults with InternalWarning)
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            meta = get_model_metadata('unknown_model')
        assert meta['display_name'] == 'UNKNOWN_MODEL'
        assert meta['color'] == '#7f7f7f'
        assert meta['supports_random_state'] is False
        assert meta['prefer_sequential_cv'] is False
        internal_warns = [x for x in w if issubclass(x.category, InternalWarning)]
        assert len(internal_warns) == 1
        assert "unknown_model" in str(internal_warns[0].message)

    def test_model_class_metadata_matches_lazy(self):
        """Verify class metadata matches _LAZY_MODEL_METADATA for deep models.

        Both the model classes and __init__._LAZY_MODEL_METADATA now read from
        the shared _metadata.py module, so drift should be impossible. This test
        is kept as a regression safeguard.
        """
        if not _has_torch_pycox():
            pytest.skip("torch/pycox required to import model classes")

        from mlsurv.models.deepsurv import DeepSurvModel
        from mlsurv.models.deephit import DeepHitModel
        from mlsurv.models._metadata import DEEP_MODEL_METADATA

        for model_cls, key in [(DeepSurvModel, 'deepsurv'), (DeepHitModel, 'deephit')]:
            meta = DEEP_MODEL_METADATA[key]
            assert model_cls.display_name == meta['display_name']
            assert model_cls.color == meta['color']
            assert model_cls.supports_random_state == meta['supports_random_state']
            assert model_cls.prefer_sequential_cv == meta['prefer_sequential_cv']

    def test_registry_precedence_over_lazy(self):
        """Verify MODEL_REGISTRY metadata takes precedence over _LAZY_MODEL_METADATA."""
        from mlsurv.models import get_model_metadata, MODEL_REGISTRY

        # For any model in MODEL_REGISTRY, class metadata should be used
        for name, model_cls in MODEL_REGISTRY.items():
            meta = get_model_metadata(name)
            expected_display = model_cls.display_name or model_cls.name.upper()
            assert meta['display_name'] == expected_display
            assert meta['color'] == model_cls.color
            assert meta['supports_random_state'] == model_cls.supports_random_state
            assert meta['prefer_sequential_cv'] == model_cls.prefer_sequential_cv
