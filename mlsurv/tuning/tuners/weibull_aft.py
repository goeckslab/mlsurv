"""Optuna tuner for Weibull AFT."""

from ..base import BaseTuner
from ..search_spaces import DEFAULT_SPACES


class WeibullAFTTuner(BaseTuner):
    """Optuna tuner for Weibull AFT."""

    DEFAULT_SPACE = DEFAULT_SPACES['weibull_aft']

    def objective(self, trial):
        from ...models import WeibullAFTModel

        model_params = self._suggest_component_params(trial, 'model')

        if not model_params:
            # Iterate the resolved search space (same pattern as CoxNetTuner) so
            # every declared parameter is tuned and the dict cannot drift out of
            # sync with DEFAULT_SPACES.
            model_params = {
                name: self._suggest_param(trial, name)
                for name in self.search_space
            }

        model = WeibullAFTModel(**model_params)
        return self._evaluate_model(trial, model)
