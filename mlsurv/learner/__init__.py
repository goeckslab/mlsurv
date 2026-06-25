"""mlsurv.learner - SurvivalLearner package (mixin-based architecture)."""

from mlsurv.learner._core import (
    SurvivalLearner,
    _NOT_SET,
    MODELS_WITH_RANDOM_STATE,
    SEQUENTIAL_CV_MODELS,
    _adjust_model_njobs,
)
from mlsurv.learner._cv_training import _train_cv_fold

__all__ = [
    'SurvivalLearner',
    '_NOT_SET',
    'MODELS_WITH_RANDOM_STATE',
    'SEQUENTIAL_CV_MODELS',
    '_adjust_model_njobs',
    '_train_cv_fold',
]
