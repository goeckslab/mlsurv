"""Losing every confidence interval must not be a silent event.

When ``CoxPHModel._attach_covariance`` cannot invert the observed information
matrix, the hazard ratios survive but every CI and Wald p-value around them is
lost. Every consumer handles that correctly but used to report it either behind
``InternalWarning`` (ignored by default) or not at all, so the user saw NaN
columns, zero-width error bars, and an ``'N/A'`` CI column with no explanation.

The assertions here run under mlsurv's **default** filters on purpose. Asserting
the warning class alone would still pass if the tier were reverted to
``InternalWarning``, which is exactly the regression worth guarding.
"""

import warnings
from unittest.mock import patch

import numpy as np
import pytest

from mlsurv._warnings import SuggestionWarning, configure_default_filters
from mlsurv.models import CoxPHModel


def _fit_with_failing_inverse(train_test_data, exc):
    """Fit a real CoxPH, but make the covariance inversion raise *exc*.

    scipy's ``inv`` returns garbage rather than raising on the degenerate
    designs a caller can actually construct, so the seam is patched instead of
    fed bad data.
    """
    X_train, _, y_train, _ = train_test_data
    model = CoxPHModel(alpha=0)
    with patch("mlsurv.models.coxph._inv", side_effect=exc), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    return model


@pytest.fixture
def singular_model(train_test_data):
    """A real fitted CoxPH whose covariance inversion was made to fail."""
    return _fit_with_failing_inverse(
        train_test_data, np.linalg.LinAlgError("Matrix is singular.")
    )


def _warns_by_default(call):
    """Run *call* under mlsurv's real default filters; return the warnings.

    ``catch_warnings`` restores the filter list on exit, so
    ``configure_default_filters`` is re-applied inside the block to reproduce
    what a plain ``import mlsurv`` leaves in place.
    """
    configure_default_filters()
    with warnings.catch_warnings(record=True) as caught:
        configure_default_filters()
        result = call()
    return result, [w for w in caught
                    if issubclass(w.category, SuggestionWarning)]


class TestVisibleUnderDefaultFilters:
    """Each consumer of a missing covariance says so without opt-in."""

    def test_wald_inference_warns(self, singular_model):
        """Coefficient table: NaN p-values and CIs used to be silent."""
        from mlsurv.evaluation._coefficients import compute_wald_inference

        (p, lo, hi), warned = _warns_by_default(
            lambda: compute_wald_inference(
                singular_model, singular_model.get_coefficients().values
            )
        )

        assert np.isnan(p).all() and np.isnan(lo).all() and np.isnan(hi).all()
        assert warned, "NaN CIs and p-values reached the user unannounced"

    def test_error_bounds_warn_and_name_the_real_cause(self, singular_model):
        """Forest plot: whiskers collapse to zero width.

        The old message said the model "does not provide a covariance matrix",
        which is wrong for CoxPH: it tried and failed, and knows why.
        """
        from mlsurv.viz.features import compute_hazard_ratios

        df, warned = _warns_by_default(
            lambda: compute_hazard_ratios(singular_model)
        )

        assert not df["ci_available"].any()
        assert warned
        text = str(warned[0].message)
        assert "is singular" in text
        assert "does not provide a covariance matrix" not in text

    def test_hazard_ratio_table_warns(self, singular_model):
        """Public table: 'N/A' is honest but was mute about the cause."""
        from mlsurv.results.tables import format_hazard_ratios

        df, warned = _warns_by_default(
            lambda: format_hazard_ratios(singular_model)
        )

        assert (df["95% CI"] == "N/A").all()
        assert warned
        assert "is singular" in str(warned[0].message)

    def test_table_keeps_its_public_columns(self, singular_model):
        """The warning is additive; the returned shape is a public contract."""
        from mlsurv.results.tables import format_hazard_ratios

        df, _ = _warns_by_default(lambda: format_hazard_ratios(singular_model))

        assert list(df.columns) == ["Feature", "HR", "95% CI"]
        # Without this the test passes on a healthy model too, pinning nothing.
        assert (df["95% CI"] == "N/A").all()


class TestWordingWhenNoReasonWasRecorded:
    """The fallback branch: a model that lost its covariance without saying why.

    That is what a learner pickled before ``covariance_unavailable_reason_``
    existed looks like. Every other test here uses a model that recorded one,
    so these branches compose their messages from a different half of the code
    and had no wording coverage at all.
    """

    @pytest.fixture
    def mute_model(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0).fit(X_train, y_train)
        del model.model_.covariance_matrix_
        model.covariance_unavailable_reason_ = None
        return model

    def test_table_fallback_reads_as_prose(self, mute_model):
        """*cause* lands after a full stop, so it has to be capitalized."""
        from mlsurv.results.tables import format_hazard_ratios

        _, warned = _warns_by_default(lambda: format_hazard_ratios(mute_model))

        text = str(warned[0].message)
        assert "'N/A'. No covariance matrix was attached" in text
        assert "'N/A'. no covariance" not in text

    def test_plot_fallback_does_not_blame_the_model_type(self, mute_model):
        """CoxPH does normally provide a covariance matrix.

        Saying it "does not provide" one describes the class when the truth is
        about this one fit, which is the same kind of unearned claim the rest
        of this file exists to prevent.
        """
        from mlsurv.viz.features import compute_hazard_ratios

        _, warned = _warns_by_default(lambda: compute_hazard_ratios(mute_model))

        text = str(warned[0].message)
        assert "does not provide a covariance matrix" not in text
        assert "No covariance matrix is available" in text


class TestMalformedCovarianceIsAlsoVisible:
    """The other half of each consumer: a covariance that exists but is junk.

    ``cov is None`` and "cov is present but unusable" are separate branches at
    every site, and both were promoted from ``InternalWarning``. Only the first
    is covered above.
    """

    @pytest.fixture
    def junk_covariance_model(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0).fit(X_train, y_train)
        model.model_.covariance_matrix_ = object()   # not array-like
        return model

    def test_wald_inference_warns(self, junk_covariance_model):
        from mlsurv.evaluation._coefficients import compute_wald_inference

        coef = junk_covariance_model.get_coefficients().values
        (p, lo, hi), warned = _warns_by_default(
            lambda: compute_wald_inference(junk_covariance_model, coef)
        )

        assert np.isnan(p).all()
        assert warned and "Wald inference" in str(warned[0].message)

    def test_error_bounds_warn(self, junk_covariance_model):
        from mlsurv.viz.features import compute_hazard_ratios

        df, warned = _warns_by_default(
            lambda: compute_hazard_ratios(junk_covariance_model)
        )

        assert not df["ci_available"].any()
        assert warned and "error bounds" in str(warned[0].message)


class TestTheOtherFailureBranch:
    """``_attach_covariance`` gives up before the inversion when there are no
    coefficients to build an information matrix from. That branch records its
    own reason and was untested."""

    def test_missing_coefficients_records_its_own_reason(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0).fit(X_train, y_train)
        model.covariance_unavailable_reason_ = None
        del model.model_.coef_

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model._attach_covariance(X_train, y_train)

        assert "no fitted coefficients" in model.covariance_unavailable_reason_
        assert any(issubclass(w.category, SuggestionWarning) for w in caught)


class TestReasonWordingMatchesTheFailure:
    """A singular matrix and a blown-up fit are not the same news."""

    def _reason_for(self, train_test_data, exc):
        model = _fit_with_failing_inverse(train_test_data, exc)
        return model.covariance_unavailable_reason_

    def test_singular_matrix_vouches_for_the_point_estimates(self, train_test_data):
        reason = self._reason_for(
            train_test_data, np.linalg.LinAlgError("Matrix is singular.")
        )

        assert "singular" in reason
        assert "still valid point estimates" in reason

    def test_non_finite_hessian_does_not_vouch_for_them(self, train_test_data):
        """scipy's inv has check_finite=True, so an inf/NaN Hessian lands here.

        That means the fit itself blew up, which is exactly when the
        coefficients should not be called valid, so the reason must not
        reassure the reader about them.
        """
        reason = self._reason_for(
            train_test_data, ValueError("array must not contain infs or NaNs")
        )

        assert "still valid point estimates" not in reason
        assert "caution" in reason


class TestQuietWhenNothingIsWrong:
    """The common case must not gain a warning."""

    def test_healthy_fit_warns_about_nothing(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        model = CoxPHModel(alpha=0).fit(X_train, y_train)

        from mlsurv.results.tables import format_hazard_ratios
        from mlsurv.viz.features import compute_hazard_ratios

        df, warned_table = _warns_by_default(
            lambda: format_hazard_ratios(model))
        hr, warned_plot = _warns_by_default(
            lambda: compute_hazard_ratios(model))

        assert not (df["95% CI"] == "N/A").any()
        assert hr["ci_available"].all()
        assert not warned_table and not warned_plot
