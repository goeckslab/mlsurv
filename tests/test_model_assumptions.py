"""Pin the PH/linearity assumption sets derived from model class attributes.

These assertions live next to the model wrappers because the sets are
now derived automatically — an accidental `assumes_ph = True` flip on a
new model would otherwise surface first in a manuscript figure's
"Affected trained models" sentence rather than in CI.
"""

from mlsurv.constants import (
    LINEARITY_ASSUMPTION_MODELS,
    MODEL_DISPLAY_NAMES,
    PH_ASSUMPTION_MODELS,
)
from mlsurv.models import MODEL_REGISTRY


def test_linearity_is_subset_of_ph():
    """Every linear-hazard model in mlsurv also fits a Cox partial likelihood."""
    assert LINEARITY_ASSUMPTION_MODELS <= PH_ASSUMPTION_MODELS


def test_every_registered_model_declares_bool_assumptions():
    for name, model_cls in MODEL_REGISTRY.items():
        assert isinstance(model_cls.assumes_ph, bool), (
            f"{name}.assumes_ph must be a bool, got {type(model_cls.assumes_ph)}"
        )
        assert isinstance(model_cls.assumes_linearity, bool), (
            f"{name}.assumes_linearity must be a bool, "
            f"got {type(model_cls.assumes_linearity)}"
        )


def test_display_names_cover_every_registered_model():
    for name in MODEL_REGISTRY:
        assert name in MODEL_DISPLAY_NAMES, (
            f"{name} missing from MODEL_DISPLAY_NAMES"
        )
