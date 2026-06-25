"""Per-model Optuna tuner classes for survival models."""

from ..exceptions import NoTunerError
from .coxnet import CoxNetTuner
from .coxph import CoxPHTuner
from .custom import CustomModelTuner
from .deephit import DeepHitTuner
from .deepsurv import DeepSurvTuner
from .fssvm import FSSVMTuner
from .gbsa import GBSATuner
from .rsf import RSFTuner
from .weibull_aft import WeibullAFTTuner
from .xgbcox import XGBCoxTuner
from .xgbaft import XGBAFTTuner

# Tuner registry
TUNER_REGISTRY = {
    'coxph': CoxPHTuner,
    'coxnet': CoxNetTuner,
    'rsf': RSFTuner,
    'gbsa': GBSATuner,
    'fssvm': FSSVMTuner,
    'deepsurv': DeepSurvTuner,
    'xgbcox': XGBCoxTuner,
    'xgbaft': XGBAFTTuner,
    'deephit': DeepHitTuner,
    'weibull_aft': WeibullAFTTuner,
}


def get_tuner(model_name, **kwargs):
    """Instantiate the registered tuner for a model.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g., 'rsf', 'coxnet').
    **kwargs :
        Arguments passed to the tuner constructor.

    Returns
    -------
    BaseTuner
        Instantiated tuner.

    Raises
    ------
    NoTunerError
        If no tuner is registered for the model name.
    """
    tuner_class = TUNER_REGISTRY.get(model_name)
    if tuner_class is None:
        raise NoTunerError(model_name, available=list(TUNER_REGISTRY.keys()))

    return tuner_class(model_name=model_name, **kwargs)


__all__ = [
    "CoxNetTuner",
    "CoxPHTuner",
    "CustomModelTuner",
    "DeepHitTuner",
    "DeepSurvTuner",
    "FSSVMTuner",
    "GBSATuner",
    "RSFTuner",
    "WeibullAFTTuner",
    "XGBCoxTuner",
    "XGBAFTTuner",
    "TUNER_REGISTRY",
    "get_tuner",
]
