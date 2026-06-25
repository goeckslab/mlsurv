"""Resolver-level tests for model selection.

``_available_model_names`` and ``_resolve_models_arg`` must accept
registry keys that exist in ``_state.models`` / ``ctx.fitted_models`` even
when no ``.evaluation`` has been attached (e.g. a model trained but tuned
with ``evaluate=False``), because ``EvaluationResult.combine()`` skips
un-evaluated models and they therefore never reach ``concordance``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlsurv.results.containers import EvaluationResult
from mlsurv.viz._context import PlotContext
from mlsurv.viz._plot_impl import (
    _available_model_names,
    _resolve_models_arg,
)


def _make_ctx(*, concordance_models=None, fitted_model_keys=None, source="main"):
    eval_result = None
    if concordance_models is not None:
        empty = pd.DataFrame()
        eval_result = EvaluationResult(
            concordance=pd.DataFrame({"model_type": list(concordance_models)}),
            test_set=empty,
            auc_t=empty,
            importance=empty,
            importance_iterations=empty,
        )
    fitted_models = (
        {k: object() for k in fitted_model_keys}
        if fitted_model_keys is not None
        else None
    )
    kwargs = dict(
        name=source,
        source=source,
        eval_result=eval_result,
        fitted_models=fitted_models,
    )
    if source == "validation" and eval_result is None:
        empty = pd.DataFrame()
        kwargs["eval_result"] = EvaluationResult(
            concordance=pd.DataFrame({"model_type": []}),
            test_set=empty, auc_t=empty, importance=empty,
            importance_iterations=empty,
        )
    return PlotContext(**kwargs)


class TestAvailableModelNames:
    def test_union_of_fitted_and_concordance(self):
        ctx = _make_ctx(
            concordance_models=["coxnet", "coxph"],
            fitted_model_keys=["coxnet", "coxph", "gbsa"],
        )
        assert set(_available_model_names(ctx)) == {
            "coxnet", "coxph", "gbsa",
        }

    def test_fitted_only_when_no_eval(self):
        ctx = _make_ctx(fitted_model_keys=["coxnet", "gbsa"])
        assert set(_available_model_names(ctx)) == {"coxnet", "gbsa"}

    def test_concordance_only_when_no_fitted(self):
        ctx = _make_ctx(concordance_models=["coxnet"])
        assert set(_available_model_names(ctx)) == {"coxnet"}

    def test_empty_when_both_absent(self):
        ctx = _make_ctx()
        assert _available_model_names(ctx) == []

    def test_validation_source_uses_concordance_only(self):
        # In validation/subpopulation contexts, fitted_models is copied
        # from the main learner and includes models not evaluated against
        # this cohort. Availability must stay cohort-scoped.
        ctx = _make_ctx(
            concordance_models=["coxph"],
            fitted_model_keys=["coxph", "rsf"],
            source="validation",
        )
        assert set(_available_model_names(ctx)) == {"coxph"}


class TestResolveModelsArg:
    def test_accepts_model_absent_from_concordance(self):
        # Core repro: a model trained but tuned with evaluate=False is in
        # fitted_models but not in concordance. Resolver must accept it.
        ctx = _make_ctx(
            concordance_models=["coxnet", "coxph"],
            fitted_model_keys=["coxnet", "coxph", "gbsa"],
        )
        assert _resolve_models_arg(ctx, ["gbsa"]) == ["gbsa"]
        assert _resolve_models_arg(ctx, ["coxnet", "gbsa"]) == [
            "coxnet", "gbsa",
        ]

    def test_none_passes_through(self):
        ctx = _make_ctx(fitted_model_keys=["coxnet"])
        assert _resolve_models_arg(ctx, None) is None

    def test_string_coerced_to_list(self):
        ctx = _make_ctx(fitted_model_keys=["coxnet", "gbsa"])
        assert _resolve_models_arg(ctx, "gbsa") == ["gbsa"]

    def test_unknown_model_raises(self):
        ctx = _make_ctx(fitted_model_keys=["coxnet", "coxph"])
        with pytest.raises(ValueError, match="Unknown models"):
            _resolve_models_arg(ctx, ["mystery"])

    def test_typeerror_on_wrong_arg_type(self):
        ctx = _make_ctx(fitted_model_keys=["coxph"])
        with pytest.raises(TypeError, match="models must be None, str, or list"):
            _resolve_models_arg(ctx, 42)  # type: ignore[arg-type]
