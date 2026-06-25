"""Fallback metadata for deep learning models when dependencies are missing.

Used by __init__.py for lazy-loaded models (deepsurv, deephit) when torch/pycox
are not installed and the actual model classes cannot be imported. The model
classes themselves define their metadata as plain class attributes.

No torch/pycox imports here — safe to import unconditionally.
"""

DEEP_MODEL_METADATA = {
    "deepsurv": {
        "display_name": "DeepSurv",
        "color": "#56B4E9",
        "supports_random_state": True,
        "prefer_sequential_cv": True,
        "has_internal_parallelism": False,
        "complexity": "deep_learning",
        # Optimizes Cox partial likelihood (pycox.models.CoxPH) — same loss
        # as XGBoost-Cox. The PH assumption is at the loss level even though
        # the network can partially absorb non-PH effects via non-linear
        # capacity (Katzman et al. 2018).
        "assumes_ph": True,
        "assumes_linearity": False,
    },
    "deephit": {
        "display_name": "DeepHit",
        "color": "#F0E442",
        "supports_random_state": True,
        "prefer_sequential_cv": True,
        "has_internal_parallelism": False,
        "complexity": "deep_learning",
        "assumes_ph": False,
        "assumes_linearity": False,
    },
}
