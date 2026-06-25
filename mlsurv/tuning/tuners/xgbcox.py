"""Optuna tuner for XGBoost-Cox."""

import warnings

from ..base import BaseTuner
from ..search_spaces import DEFAULT_SPACES


class XGBCoxTuner(BaseTuner):
    """Optuna tuner for XGBoost-Cox."""

    DEFAULT_SPACE = DEFAULT_SPACES['xgbcox']

    def objective(self, trial):
        from ...models import XGBCoxModel

        model_params = self._suggest_component_params(trial, 'model')

        if not model_params:
            # Iterate the resolved search space (same pattern as CoxNetTuner) so
            # every declared parameter is tuned. A hand-maintained dict here drifts
            # out of sync with DEFAULT_SPACES — that omitted max_delta_step.
            model_params = {
                name: self._suggest_param(trial, name)
                for name in self.search_space
            }

        model_params['n_jobs'] = getattr(self, '_model_n_jobs', 1)
        model_params['random_state'] = self.random_state

        # Suppress n_jobs reproducibility warning (auto-computed, not user-chosen)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"^XGBoost with n_jobs != 1", category=UserWarning)
            model = XGBCoxModel(**model_params)
        return self._evaluate_model(trial, model)
