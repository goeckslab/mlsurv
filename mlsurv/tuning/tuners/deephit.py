"""Optuna tuner for DeepHit neural network."""

from ...models._auto_defaults import auto_deephit_defaults
from ..base import BaseTuner
from ..search_spaces import DEFAULT_SPACES


class DeepHitTuner(BaseTuner):
    """Optuna tuner for DeepHit neural network."""

    DEFAULT_SPACE = DEFAULT_SPACES['deephit']

    def _get_warm_start_params(self):
        """Return dataset-aware warm-start params from auto_deephit_defaults."""
        n_samples = self.X.shape[0]
        n_features = self.X.shape[1]
        n_events = int(self.y['event'].sum())
        defaults = auto_deephit_defaults(n_samples, n_features, n_events)
        layers = defaults['layers']
        return {
            'n_layers': len(layers),
            'n_nodes': layers[0],
            'dropout': defaults['dropout'],
            'activation': defaults['activation'],
            'alpha': defaults['alpha'],
            'sigma': defaults['sigma'],
            'learning_rate': defaults['learning_rate'],
            'weight_decay': defaults['weight_decay'],
            'batch_size': defaults['batch_size'],
            'num_durations': defaults['num_durations'],
        }

    def objective(self, trial):
        from ...models.deephit import DeepHitModel

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
        # default and custom paths): n_layers + n_nodes -> layers.
        if 'n_layers' in model_params and 'n_nodes' in model_params:
            n_layers = model_params.pop('n_layers')
            n_nodes = model_params.pop('n_nodes')
            model_params['layers'] = [n_nodes] * n_layers

        model_params['random_state'] = self.random_state

        model = DeepHitModel(**model_params)
        return self._evaluate_model(trial, model)
