"""Optuna tuner for user-provided custom model instances."""

from sklearn.base import clone

from ..base import BaseTuner


class CustomModelTuner(BaseTuner):
    """Optuna tuner for a custom ``BaseSurvivalModel`` instance.

    Built-in tuners hard-import their model class (e.g. ``RSFModel(**params)``)
    and look the search space up by registered name. A custom model is unknown
    to the registry, so this tuner instead:

    - stores the passed-in instance and ``clone()`` + ``set_params()`` it each
      trial (the same opaque-estimator pattern the per-trial fit path already
      uses), and
    - takes its model search space explicitly from the constructor (the space
      declared by the model's ``get_search_space()`` hook, merged with any
      per-call ``search_spaces={'model': {...}}`` overrides).

    Not registered in ``TUNER_REGISTRY``: it is constructed directly by the
    learner, never resolved by name via ``get_tuner()``.
    """

    def __init__(self, custom_model, search_space=None, **kwargs):
        self._custom_model = custom_model
        # Shadow the class-level DEFAULT_SPACE with the model-declared space so
        # BaseTuner.__init__ uses it as the base, then merges the per-call
        # search_spaces['model'] on top (user wins). See base.py:347-357: the
        # custom model_name is not in DEFAULT_SPACES, so it takes the
        # ``self.DEFAULT_SPACE.copy()`` branch.
        self.DEFAULT_SPACE = dict(search_space or {})
        super().__init__(**kwargs)

    def objective(self, trial):
        # Suggest every parameter in the merged model search space. Unlike
        # built-in tuners (which enumerate their known params), the custom
        # model's param names are only known from self.search_space.
        model_params = {
            name: self._suggest_param(trial, name) for name in self.search_space
        }
        model = clone(self._custom_model)
        if model_params:
            model.set_params(**model_params)
        # Inject the learner seed, mirroring the built-in tuners (e.g. rsf.py
        # sets model_params['random_state'] = self.random_state). clone()
        # preserves whatever seed the user baked into the instance, so without
        # this a custom model defaulting to random_state=None tunes unseeded.
        # Guard on get_params() so we never call set_params(random_state=...)
        # on a model that has no random_state — that call would be a silent
        # no-op at best, or an error on a strict set_params.
        if "random_state" in model.get_params():
            model.set_params(random_state=self.random_state)
        return self._evaluate_model(trial, model)
