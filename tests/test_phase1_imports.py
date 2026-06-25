"""Test module imports for Phase 1."""

import importlib
import warnings

import pytest


# (module, public symbols) pairs covering the core import surface.
_CORE_IMPORTS = [
    ("mlsurv", ["__version__"]),
    ("mlsurv.utils", ["to_structured_array", "load_survival_data"]),
    ("mlsurv.results.containers",
     ["ModelResult", "ValidationResult", "EvaluationResult", "BootstrapResult"]),
    ("mlsurv.models.base", ["BaseSurvivalModel"]),
    ("mlsurv.models",
     ["CoxPHModel", "CoxNetModel", "RSFModel", "GBSAModel", "FSSVMModel",
      "MODEL_REGISTRY", "get_model", "list_models"]),
    ("mlsurv.preprocessing", ["TopKCoxSelector"]),
    ("mlsurv.exceptions",
     ["MethodologicalWarning", "SuggestionWarning", "InternalWarning",
      "IterativeWarning", "ClampWarning", "FitFailureWarning",
      "MetricFailureWarning", "SmallSubpopulationWarning",
      "enable_internal_warnings"]),
]

# Names that must be re-exported from the top-level mlsurv package.
_TOP_LEVEL_EXPORTS = [
    "CoxPHModel", "CoxNetModel", "RSFModel", "GBSAModel", "FSSVMModel",
    "XGBCoxModel", "XGBAFTModel", "WeibullAFTModel",
    "to_structured_array", "load_survival_data", "SurvivalArray",
    "EvaluationResult", "BootstrapResult", "ModelResult",
    "TopKCoxSelector",
]


@pytest.mark.parametrize("module, symbols", _CORE_IMPORTS,
                         ids=[m for m, _ in _CORE_IMPORTS])
def test_core_module_importable(module, symbols):
    """Core modules import and expose their public symbols."""
    mod = importlib.import_module(module)
    for symbol in symbols:
        assert getattr(mod, symbol, None) is not None, f"{module}.{symbol} missing"


@pytest.mark.parametrize("name", _TOP_LEVEL_EXPORTS)
def test_top_level_export(name):
    """Key classes/utilities are re-exported from the mlsurv package."""
    import mlsurv
    assert hasattr(mlsurv, name)


def test_exceptions_category_filters():
    """A published category from mlsurv.exceptions actually filters via stdlib.

    The whole point of publishing the categories is that users can pass them
    to ``warnings.filterwarnings`` — so verify the filter takes effect for the
    exact class re-exported from the public module.
    """
    from mlsurv.exceptions import MethodologicalWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=MethodologicalWarning)
        warnings.warn("filtered", MethodologicalWarning)
        warnings.warn("kept", UserWarning)
    assert not any(issubclass(w.category, MethodologicalWarning) for w in caught)
    assert len(caught) == 1  # the unrelated UserWarning still came through


def test_exceptions_are_the_same_objects_as_internal():
    """Public re-exports are identity-equal to the private definitions, so a
    filter registered on one governs warnings emitted via the other."""
    from mlsurv import exceptions
    from mlsurv import _warnings

    for name in exceptions.__all__:
        assert getattr(exceptions, name) is getattr(_warnings, name)


def test_exceptions_all_matches_public_namespace():
    """``__all__`` lists exactly the public names the module exposes.

    Guards against silent drift: a re-export added to the import block but
    left out of ``__all__`` (or vice versa) would desync the documented
    surface from what's actually importable.
    """
    import mlsurv.exceptions as exc
    public = {n for n in vars(exc) if not n.startswith("_")}
    assert public == set(exc.__all__)


def test_public_filter_suppresses_internal_emission():
    """A filter registered on the public category suppresses a warning emitted
    by mlsurv's own internal code — the actual user contract.

    ``warn_missing_roc_ci`` is a zero-cost internal emitter (no model
    training) that fires a ``SuggestionWarning`` unconditionally.
    """
    from mlsurv.exceptions import SuggestionWarning
    from mlsurv._warnings import warn_missing_roc_ci

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=SuggestionWarning)
        warn_missing_roc_ci(stacklevel=2)
    assert caught == []


def test_lazy_deephit_export_if_dependencies_available():
    """DeepHit should be top-level importable when optional deps exist."""
    pytest.importorskip("torch")
    pytest.importorskip("pycox")
    import mlsurv
    assert hasattr(mlsurv, "DeepHitModel")


class TestLazyModelMetadata:
    """Test that _LAZY_MODEL_METADATA stays in sync with actual class attributes."""

    def test_lazy_metadata_matches_class_attributes(self):
        """Verify _LAZY_MODEL_METADATA stays in sync with actual class attributes."""
        # Only test if torch is available (otherwise classes can't be imported)
        pytest.importorskip("torch")

        from mlsurv.models import _LAZY_MODEL_METADATA
        from mlsurv.models.deepsurv import DeepSurvModel
        from mlsurv.models.deephit import DeepHitModel

        for name, cls in [("deepsurv", DeepSurvModel), ("deephit", DeepHitModel)]:
            lazy = _LAZY_MODEL_METADATA[name]
            assert lazy["display_name"] == cls.display_name, f"{name} display_name mismatch"
            assert lazy["color"] == cls.color, f"{name} color mismatch"
            assert lazy["supports_random_state"] == cls.supports_random_state, f"{name} supports_random_state mismatch"
            assert lazy["prefer_sequential_cv"] == cls.prefer_sequential_cv, f"{name} prefer_sequential_cv mismatch"
