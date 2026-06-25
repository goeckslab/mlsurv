"""Static signature-drift guards for the tune -> evaluate forwarding contract.

A family of bugs all stemmed from one root cause: ``TuneConfig`` and
``_run_post_tune_evaluation`` must be manually kept in sync with
``evaluate()``'s signature, and there was no guard. When a kwarg was added
to ``evaluate()`` and not mirrored on ``TuneConfig``, it silently no-op'd on
the tune path.

These tests are pure ``inspect`` / ``ast`` introspection — no model fitting.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import fields

from mlsurv.learner._evaluation import EvaluationMixin
from mlsurv.learner._tuning import TuneConfig, TuningMixin


# Kwargs of ``evaluate()`` that are intentionally NOT mirrored on TuneConfig.
# Each entry needs a one-line justification so future readers can audit.
_NON_FORWARDED_EVAL_KWARGS = {
    # Not user kwargs: receiver / data positional inputs.
    "self",
    "models",
    "X",
    "y",
    # TuneConfig renames this to ``show_eval_progress`` to disambiguate from
    # ``show_progress_bar`` (which controls the Optuna trial bar).
    # ``_run_post_tune_evaluation`` forwards ``show_progress=config.show_eval_progress``.
    "show_progress",
    # Not exposed via tune: ``optuna_verbose`` and ``show_eval_progress`` already
    # cover the user-visible verbosity surface during the tune path.
    "verbose",
    # Intentionally NOT forwarded. TuneConfig.n_jobs controls Optuna study
    # parallelism; evaluate() is left to fall back to ``learner.n_jobs`` so the
    # two paths are tuned independently. Do not "fix" the asymmetry without
    # auditing the call sites — the same field name has different semantics.
    "n_jobs",
    # Sourced from learner state, not duplicated on TuneConfig.
    "random_state",
    # Auto-derived in _run_post_tune_evaluation as ``include_predictions=config.bootstrap``;
    # we always want predictions when bootstrapping, never separately configurable here.
    "include_predictions",
    # Not exposed via tune: defaults are reasonable for post-tune eval; users
    # wanting non-default values can call evaluate() separately after tune().
    "bootstrap_ci",
    "bootstrap_stratify",
    "custom_metrics",
    "ipcw_tau",
    "ipcw_g_hat_source",
    # Not exposed via tune: the default 10 bins is reasonable for the
    # post-tune evaluation; a user on a small cohort who needs a lower GND bin
    # count calls evaluate(gnd_n_bins=...) separately after tune().
    "gnd_n_bins",
}

# TuneConfig fields that mirror an evaluate() kwarg under a different name.
# Maps tune_field -> evaluate_kwarg.
_RENAMED_MIRRORS = {
    # ``show_eval_progress`` disambiguates from ``show_progress_bar`` (Optuna).
    "show_eval_progress": "show_progress",
}


def _public_evaluate_kwargs():
    """Public, named ``evaluate()`` kwargs. Filters out ``_``-prefixed and ``**kwargs``."""
    sig = inspect.signature(EvaluationMixin.evaluate)
    return {
        name
        for name, p in sig.parameters.items()
        if not name.startswith("_")
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


def _evaluate_call_forwarded_kwargs():
    """Inspect _run_post_tune_evaluation's body to extract the kwargs of its
    ``self.evaluate(...)`` call.

    Returns ``{eval_kwarg: tune_field_name}`` for every keyword whose value is a
    ``config.<attr>`` access. Comments, log lines, and stray references are
    ignored — only kwargs that actually appear in the call.
    """
    src = textwrap.dedent(inspect.getsource(TuningMixin._run_post_tune_evaluation))
    tree = ast.parse(src)
    forwarded = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "evaluate"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            for kw in node.keywords:
                if kw.arg is None:  # **kwargs unpacking — ignore
                    continue
                value = kw.value
                if (
                    isinstance(value, ast.Attribute)
                    and isinstance(value.value, ast.Name)
                    and value.value.id == "config"
                ):
                    forwarded[kw.arg] = value.attr
    return forwarded


def test_tune_config_mirrors_evaluate_signature():
    """Every public evaluate() kwarg must be a TuneConfig field, or in the allowlist."""
    expected = _public_evaluate_kwargs() - _NON_FORWARDED_EVAL_KWARGS
    tune_fields = {f.name for f in fields(TuneConfig)}
    # Allow renamed mirrors (e.g. evaluate(show_progress) -> TuneConfig.show_eval_progress).
    tune_fields_with_renames = tune_fields | set(_RENAMED_MIRRORS.values())
    missing = expected - tune_fields_with_renames
    assert not missing, (
        f"TuneConfig is missing fields that evaluate() accepts: {sorted(missing)}. "
        "Either add them to TuneConfig and forward them in "
        "_run_post_tune_evaluation, or extend _NON_FORWARDED_EVAL_KWARGS in this "
        "test with a one-line justification."
    )


def test_run_post_tune_evaluation_forwards_all_eval_mirror_fields():
    """Every TuneConfig field that mirrors an evaluate() kwarg must be passed
    to ``self.evaluate(...)`` inside ``_run_post_tune_evaluation``.

    Catches dead fields and verifies the actual call (AST-level), so a forgotten
    forward, a commented-out kwarg, or a refactor that drops the kwarg from the
    call but leaves a logging line will all fail the test.
    """
    eval_kwargs = _public_evaluate_kwargs() - _NON_FORWARDED_EVAL_KWARGS
    eval_mirror_fields = {f.name for f in fields(TuneConfig) if f.name in eval_kwargs}
    eval_mirror_fields |= set(_RENAMED_MIRRORS.keys())

    forwarded = _evaluate_call_forwarded_kwargs()
    forwarded_tune_fields = set(forwarded.values())
    unread = sorted(eval_mirror_fields - forwarded_tune_fields)
    assert not unread, (
        "TuneConfig fields that mirror evaluate() kwargs but are not passed to "
        f"self.evaluate(...) in _run_post_tune_evaluation: {unread}. "
        "Forward them or remove the field."
    )


def test_renamed_mirrors_wired_correctly():
    """Renamed mirrors must be wired as ``<eval_kwarg>=config.<tune_field>``.

    Without this, a future refactor could rename one side without the other and
    the previous tests would still pass (because both names exist somewhere).
    """
    forwarded = _evaluate_call_forwarded_kwargs()
    for tune_field, eval_kwarg in _RENAMED_MIRRORS.items():
        actual = forwarded.get(eval_kwarg)
        assert actual == tune_field, (
            f"Expected ``{eval_kwarg}=config.{tune_field}`` in the evaluate() "
            f"call inside _run_post_tune_evaluation, but found "
            f"{eval_kwarg}=config.{actual!r}. Rename drift in _RENAMED_MIRRORS "
            "or in the call site."
        )


class TestResolvePerModel:
    """Unit tests for the ``_resolve_per_model`` per-model kwarg resolver.

    Relocated from test_integration.py: pure dict-lookup logic, no model
    fitting, so it belongs with the kwarg-forwarding unit tests.
    """

    def test_scalar_returns_as_is(self):
        from mlsurv.learner._training import _resolve_per_model

        assert _resolve_per_model(42, 'rsf', default=0) == 42
        assert _resolve_per_model('auto', 'rsf', default=0) == 'auto'

    def test_dict_returns_model_value(self):
        from mlsurv.learner._training import _resolve_per_model

        d = {'rsf': 100, 'coxph': 50}
        assert _resolve_per_model(d, 'rsf', default=0) == 100
        assert _resolve_per_model(d, 'coxph', default=0) == 50

    def test_dict_missing_key_returns_default(self):
        from mlsurv.learner._training import _resolve_per_model

        d = {'rsf': 100}
        assert _resolve_per_model(d, 'coxph', default='auto') == 'auto'
