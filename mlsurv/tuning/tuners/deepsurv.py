"""Optuna tuner for DeepSurv neural network."""

from ...models._auto_defaults import auto_deepsurv_defaults
from ..base import BaseTuner
from ..search_spaces import DEFAULT_SPACES


class DeepSurvTuner(BaseTuner):
    """Optuna tuner for DeepSurv neural network."""

    DEFAULT_SPACE = DEFAULT_SPACES['deepsurv']

    def _get_warm_start_params(self):
        """Return dataset-aware warm-start params from auto_deepsurv_defaults."""
        n_samples = self.X.shape[0]
        n_features = self.X.shape[1]
        n_events = int(self.y['event'].sum())
        defaults = auto_deepsurv_defaults(n_samples, n_features, n_events)
        layers = defaults['layers']
        return {
            'n_layers': len(layers),
            'n_nodes': layers[0],
            'dropout': defaults['dropout'],
            'activation': defaults['activation'],
            'weight_decay': defaults['weight_decay'],
            'batch_size': defaults['batch_size'],
            'lr': defaults['learning_rate'],
        }

    def objective(self, trial):
        from ...models.deepsurv import DeepSurvModel

        model_params = self._suggest_component_params(trial, 'model')

        if not model_params:
            # Iterate the resolved search space (same pattern as CoxNetTuner) so
            # every declared parameter is tuned and the dict cannot drift out of
            # sync with DEFAULT_SPACES.
            model_params = {
                name: self._suggest_param(trial, name)
                for name in self.search_space
            }

        # Map search-space names to model constructor args (applies to both the
        # default and custom paths): n_layers + n_nodes -> layers, lr -> learning_rate.
        if 'n_layers' in model_params and 'n_nodes' in model_params:
            n_layers = model_params.pop('n_layers')
            n_nodes = model_params.pop('n_nodes')
            model_params['layers'] = [n_nodes] * n_layers
        if 'lr' in model_params:
            model_params['learning_rate'] = model_params.pop('lr')

        model_params['random_state'] = self.random_state

        model = DeepSurvModel(**model_params)
        return self._evaluate_model(trial, model)
