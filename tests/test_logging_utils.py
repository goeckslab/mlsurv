"""Tests for mlsurv.logging_utils."""

import logging
import warnings

import pytest

from mlsurv.logging_utils import (
    get_logger, log_exception, log_metric_failure,
    _is_fold_boundary_error, _log_once_level, _fold_boundary_message,
)


@pytest.fixture(autouse=True)
def _reset_logging_state(monkeypatch):
    """Reset the root-configured flag and env var between tests."""
    import mlsurv.logging_utils as mod
    mod._root_configured = False
    mod._seen_log_once_keys.clear()
    mod._seen_warn_once_keys.clear()
    monkeypatch.delenv("MLSURV_LOG_LEVEL", raising=False)
    # Remove any existing handlers on the mlsurv root logger
    root = logging.getLogger("mlsurv")
    root.handlers.clear()
    yield


class TestGetLogger:
    def test_returns_namespaced_logger(self):
        lg = get_logger("foo")
        assert lg.name == "mlsurv.foo"

    def test_default_level_is_warning(self):
        get_logger("bar")
        root = logging.getLogger("mlsurv")
        assert root.level == logging.WARNING

    def test_env_var_controls_level(self, monkeypatch):
        monkeypatch.setenv("MLSURV_LOG_LEVEL", "DEBUG")
        get_logger("baz")
        root = logging.getLogger("mlsurv")
        assert root.level == logging.DEBUG

    def test_invalid_env_var_falls_back_to_warning(self, monkeypatch):
        monkeypatch.setenv("MLSURV_LOG_LEVEL", "NOTAVALIDLEVEL")
        get_logger("qux")
        root = logging.getLogger("mlsurv")
        assert root.level == logging.WARNING


class TestLogException:
    def test_logs_error_message(self, caplog):
        lg = get_logger("test_exc")
        exc = ValueError("boom")
        with caplog.at_level(logging.ERROR, logger="mlsurv.test_exc"):
            log_exception(lg, exc, "doing stuff")
        assert "doing stuff" in caplog.text
        assert "ValueError" in caplog.text
        assert "boom" in caplog.text
        assert "MLSURV_LOG_LEVEL=DEBUG" in caplog.text

    def test_warn_user_emits_internal_warning_by_default(self):
        from mlsurv._warnings import InternalWarning
        lg = get_logger("test_warn")
        exc = RuntimeError("oops")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log_exception(lg, exc, "ctx", warn_user=True)
        assert len(w) == 1
        assert "ctx: oops" in str(w[0].message)
        assert w[0].category is InternalWarning

    def test_warn_user_explicit_category_override(self):
        from mlsurv._warnings import SuggestionWarning
        lg = get_logger("test_warn_override")
        exc = RuntimeError("oops")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log_exception(lg, exc, "ctx", warn_user=True, category=SuggestionWarning)
        assert len(w) == 1
        assert w[0].category is SuggestionWarning

    def test_custom_user_message(self):
        lg = get_logger("test_custom")
        exc = RuntimeError("oops")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log_exception(lg, exc, "ctx", warn_user=True, user_message="custom msg")
        assert "custom msg" in str(w[0].message)

    def test_no_warn_by_default(self):
        lg = get_logger("test_nowarn")
        exc = RuntimeError("oops")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log_exception(lg, exc, "ctx")
        assert len(w) == 0

    def test_custom_level_warning(self, caplog):
        lg = get_logger("test_level")
        exc = ValueError("expected")
        with caplog.at_level(logging.WARNING, logger="mlsurv.test_level"):
            log_exception(lg, exc, "fold boundary", level=logging.WARNING)
        assert "fold boundary" in caplog.text
        assert caplog.records[0].levelno == logging.WARNING

    def test_default_level_is_error(self, caplog):
        lg = get_logger("test_default_level")
        exc = ValueError("boom")
        with caplog.at_level(logging.ERROR, logger="mlsurv.test_default_level"):
            log_exception(lg, exc, "something failed")
        assert caplog.records[0].levelno == logging.ERROR


class TestLogMetricFailure:
    def test_builds_context_string(self, caplog):
        lg = get_logger("test_mf")
        exc = ValueError("bad data")
        with caplog.at_level(logging.ERROR, logger="mlsurv.test_mf"):
            log_metric_failure(lg, "brier", exc, fold=2, model_name="rsf")
        assert "metric=brier" in caplog.text
        assert "model=rsf" in caplog.text
        assert "fold=2" in caplog.text

    def test_iteration_in_context(self, caplog):
        lg = get_logger("test_mf2")
        exc = ZeroDivisionError("div0")
        with caplog.at_level(logging.ERROR, logger="mlsurv.test_mf2"):
            log_metric_failure(lg, "auc_t", exc, iteration=5)
        assert "iteration=5" in caplog.text

    def test_warn_user_emits_warning(self):
        lg = get_logger("test_mf_warn")
        exc = ValueError("edge case")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log_metric_failure(lg, "ipcw", exc, warn_user=True)
        assert len(w) == 1
        assert "ipcw computation failed" in str(w[0].message)

    def test_no_warn_by_default(self):
        lg = get_logger("test_mf_nowarn")
        exc = ValueError("expected")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log_metric_failure(lg, "brier", exc)
        assert len(w) == 0

    def test_level_parameter_passed_through(self, caplog):
        lg = get_logger("test_mf_level")
        exc = ValueError("all times must be within follow-up time of test data")
        with caplog.at_level(logging.WARNING, logger="mlsurv.test_mf_level"):
            log_metric_failure(lg, "ibs", exc, fold=3, level=logging.WARNING)
        assert "metric=ibs" in caplog.text
        assert caplog.records[0].levelno == logging.WARNING


class TestIsFoldBoundaryError:
    def test_detects_sksurv_message(self):
        exc = ValueError(
            "all times must be within follow-up time of test data: [281.0; 2539.0["
        )
        assert _is_fold_boundary_error(exc)

    def test_rejects_other_value_error(self):
        assert not _is_fold_boundary_error(ValueError("negative value encountered"))

    def test_rejects_non_value_error(self):
        exc = ZeroDivisionError("all times must be within follow-up time")
        assert not _is_fold_boundary_error(exc)

    def test_case_insensitive(self):
        exc = ValueError("All Times Must Be Within Follow-Up Time of test data")
        assert _is_fold_boundary_error(exc)


class TestWarnOnceKey:
    def test_emits_once_per_key(self, caplog):
        from mlsurv._warnings import MetricFailureWarning
        lg = get_logger("test_warn_once")
        exc = ValueError("all times must be within follow-up time of test data")
        with warnings.catch_warnings(record=True) as w, \
                caplog.at_level(logging.DEBUG, logger="mlsurv.test_warn_once"):
            warnings.simplefilter("always")
            log_metric_failure(
                lg, "ibs", exc,
                level=_log_once_level("fold_boundary"),
                warn_once_key="fold_boundary_warn",
            )
            log_metric_failure(
                lg, "ibs", exc,
                level=_log_once_level("fold_boundary"),
                warn_once_key="fold_boundary_warn",
            )
        # Exactly one user-visible warning across two failures.
        assert len(w) == 1
        assert issubclass(w[0].category, MetricFailureWarning)
        # The visible MetricFailureWarning is the user-facing channel, so the
        # summary log is downgraded to an audit-trail-only DEBUG record on both
        # calls — never WARNING (which would duplicate the warning on stderr).
        summary_levels = [
            r.levelno for r in caplog.records if "metric=ibs" in r.getMessage()
        ]
        assert summary_levels  # the summary record is still produced (at DEBUG)
        assert logging.WARNING not in summary_levels
        assert set(summary_levels) == {logging.DEBUG}

    def test_visible_warning_downgrades_summary_to_debug(self, caplog):
        """A visible tiered warning downgrades the summary log to DEBUG so the
        same failure is not reported twice (issue: tiered-warning stragglers)."""
        from mlsurv._warnings import MetricFailureWarning
        lg = get_logger("test_downgrade")
        exc = ValueError("boom")
        with warnings.catch_warnings(record=True) as w, \
                caplog.at_level(logging.DEBUG, logger="mlsurv.test_downgrade"):
            warnings.simplefilter("always")
            log_exception(
                lg, exc, "metric=ibs", level=logging.WARNING,
                warn_user=True, category=MetricFailureWarning,
            )
        assert len(w) == 1
        assert issubclass(w[0].category, MetricFailureWarning)
        summary = [r for r in caplog.records if "metric=ibs" in r.getMessage()]
        assert summary and all(r.levelno == logging.DEBUG for r in summary)
        # Downgrading the level must not drop the audit-trail content: the
        # DEBUG summary still names the exception and the DEBUG hint.
        text = summary[0].getMessage()
        assert "ValueError" in text
        assert "MLSURV_LOG_LEVEL=DEBUG" in text

    def test_fold_boundary_first_occurrence_summary_is_debug(self, caplog):
        """Straggler-2 path through the real entry point: a fold-boundary
        failure routed via ``log_metric_failure`` resolves to the visible
        ``MetricFailureWarning``, so even the FIRST call (whose
        ``_log_once_level`` would otherwise be WARNING) logs its summary at
        DEBUG — the warning is the user-facing channel."""
        from mlsurv._warnings import MetricFailureWarning
        lg = get_logger("test_fold_boundary_first")
        exc = ValueError("all times must be within follow-up time of test data")
        with warnings.catch_warnings(record=True) as w, \
                caplog.at_level(logging.DEBUG, logger="mlsurv.test_fold_boundary_first"):
            warnings.simplefilter("always")
            log_metric_failure(
                lg, "integrated_brier", exc,
                level=logging.WARNING,
                warn_once_key="fold_boundary_warn",
                user_message=_fold_boundary_message("integrated_brier"),
            )
        assert len(w) == 1
        assert issubclass(w[0].category, MetricFailureWarning)
        summary = [
            r for r in caplog.records if "metric=integrated_brier" in r.getMessage()
        ]
        assert summary and all(r.levelno == logging.DEBUG for r in summary)

    def test_internal_warning_keeps_summary_at_level(self, caplog):
        """When the warning is the default ``InternalWarning`` (suppressed by
        default), the summary log stays at its level — it is the only
        user-facing channel and must not be silenced."""
        lg = get_logger("test_internal_keeps")
        exc = ValueError("boom")
        with warnings.catch_warnings(record=True), \
                caplog.at_level(logging.DEBUG, logger="mlsurv.test_internal_keeps"):
            warnings.simplefilter("always")
            log_metric_failure(lg, "ibs", exc, warn_user=True, level=logging.ERROR)
        summary = [r for r in caplog.records if "metric=ibs" in r.getMessage()]
        assert summary
        assert any(r.levelno == logging.ERROR for r in summary)

    def test_default_category_is_metric_failure(self):
        from mlsurv._warnings import MetricFailureWarning
        lg = get_logger("test_warn_default_cat")
        exc = ValueError("eval times outside follow-up")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log_metric_failure(
                lg, "ibs", exc, warn_once_key="some_key",
            )
        assert len(w) == 1
        assert w[0].category is MetricFailureWarning

    def test_explicit_category_overrides_default(self):
        from mlsurv._warnings import SuggestionWarning
        lg = get_logger("test_warn_cat_override")
        exc = ValueError("eval times outside follow-up")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log_metric_failure(
                lg, "ibs", exc,
                warn_once_key="some_key",
                warn_category=SuggestionWarning,
            )
        assert len(w) == 1
        assert w[0].category is SuggestionWarning

    def test_warn_user_independent_of_dedup(self):
        """warn_user=True keeps emitting per call (no dedup)."""
        lg = get_logger("test_warn_user_no_dedup")
        exc = ValueError("oops")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log_metric_failure(lg, "ibs", exc, warn_user=True)
            log_metric_failure(lg, "ibs", exc, warn_user=True)
        assert len(w) == 2

    def test_warn_user_consumes_once_token(self):
        """warn_user=True with warn_once_key still consumes the once-token,
        so a later warn_once_key-only caller does not re-warn."""
        lg = get_logger("test_consume_token")
        exc = ValueError("oops")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log_metric_failure(
                lg, "ibs", exc, warn_user=True, warn_once_key="shared_key",
            )
            log_metric_failure(lg, "ibs", exc, warn_once_key="shared_key")
        # First call emits (warn_user=True), second is suppressed by dedup.
        assert len(w) == 1

    def test_independent_keys(self):
        lg = get_logger("test_warn_independent_keys")
        exc = ValueError("oops")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log_metric_failure(lg, "ibs", exc, warn_once_key="key_a")
            log_metric_failure(lg, "ibs", exc, warn_once_key="key_b")
        assert len(w) == 2

    def test_default_user_message(self):
        """Without user_message, the warning text is the default '<metric>
        computation failed: <exception>' wording."""
        lg = get_logger("test_default_msg")
        exc = ValueError("eval times outside follow-up")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log_metric_failure(
                lg, "integrated_brier", exc, warn_once_key="default_msg_key",
            )
        assert len(w) == 1
        assert str(w[0].message) == (
            "integrated_brier computation failed: eval times outside follow-up"
        )

    def test_custom_user_message(self):
        """user_message overrides the default wording for partial-failure
        modes like fold-boundary skips."""
        from mlsurv.logging_utils import _fold_boundary_message
        lg = get_logger("test_custom_msg")
        exc = ValueError("all times must be within follow-up time of test data")
        custom = _fold_boundary_message("integrated_brier")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log_metric_failure(
                lg, "integrated_brier", exc,
                warn_once_key="custom_msg_key",
                user_message=custom,
            )
        assert len(w) == 1
        msg = str(w[0].message)
        assert msg == custom
        assert "skipped on at least one CV fold" in msg
        assert "computation failed" not in msg


class TestLogOnceLevel:
    def test_first_call_returns_requested_level(self):
        assert _log_once_level("test_key_a") == logging.WARNING

    def test_second_call_returns_debug(self):
        _log_once_level("test_key_b")
        assert _log_once_level("test_key_b") == logging.DEBUG

    def test_different_keys_are_independent(self):
        assert _log_once_level("test_key_c") == logging.WARNING
        assert _log_once_level("test_key_d") == logging.WARNING

    def test_custom_level(self):
        assert _log_once_level("test_key_e", logging.INFO) == logging.INFO
        assert _log_once_level("test_key_e", logging.INFO) == logging.DEBUG
