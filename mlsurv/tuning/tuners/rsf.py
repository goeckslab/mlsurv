"""Optuna tuner for Random Survival Forest."""

from ..base import BaseTuner
from ..search_spaces import DEFAULT_SPACES


class RSFTuner(BaseTuner):
    """Optuna tuner for Random Survival Forest."""

    DEFAULT_SPACE = DEFAULT_SPACES['rsf']

    def objective(self, trial):
        from ...models import RSFModel

        model_params = self._suggest_component_params(trial, 'model')

        if not model_params:
            # Iterate the resolved search space (same pattern as CoxNetTuner) so
            # every declared parameter is tuned and the dict cannot drift out of
            # sync with DEFAULT_SPACES.
            model_params = {
                name: self._suggest_param(trial, name)
                for name in self.search_space
            }

        model_params['n_jobs'] = getattr(self, '_model_n_jobs', 1)
        model_params['random_state'] = self.random_state

        model = RSFModel(**model_params)
        return self._evaluate_model(trial, model)
