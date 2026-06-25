"""Optuna tuner for CoxPH preprocessing (and optional ridge penalty)."""

from ..base import BaseTuner


class CoxPHTuner(BaseTuner):
    """Optuna tuner for CoxPH preprocessing (and optional ridge penalty).

    CoxPH has no default model hyperparameters, so plain ``tune('coxph')``
    tunes only the preprocessing pipeline (feature selector k, imputer,
    scaler). The model's L2 ridge penalty ``alpha`` is tunable on an opt-in
    basis by supplying an explicit model space::

        learner.tune('coxph', search_spaces={'model': {'alpha': (1e-4, 10.0, 'log')}})

    ``alpha`` is the only model key CoxPH applies; any other model-space key
    is unsupported and the BaseTuner no-silent-data-loss guard warns about it
    (see ``_consumed_model_keys``). A tuned ``alpha > 0`` regularizes the fit,
    which the model surfaces as an inference caveat (penalized coefficients are
    not unbiased multivariable hazard ratios) via a ``MethodologicalWarning``
    and by setting ``hazard_ratio_valid`` to False. Users who want a
    sparse/regularized Cox model are still steered toward CoxNet.
    """

    DEFAULT_SPACE = {}  # No default model hyperparameters (alpha is opt-in)

    def _consumed_model_keys(self) -> set:
        # CoxPH's only tunable model hyperparameter is the L2 ridge penalty
        # (alpha). Any other key in a model space is unsupported, so the
        # BaseTuner guard warns about it instead of dropping it silently.
        return set(self.search_space) & {'alpha'}

    def objective(self, trial):
        from ...models import CoxPHModel

        # Apply only the consumed key(s); BaseTuner already warned about any
        # others. Use the 'model__'-prefixed Optuna name for consistency with
        # the other model tuners.
        model_params = {}
        for name in sorted(self._consumed_model_keys()):
            key, value = self._suggest_component_param(
                trial, 'model', None, name, self.search_space[name])
            model_params[key] = value

        model = CoxPHModel(**model_params)
        return self._evaluate_model(trial, model)
