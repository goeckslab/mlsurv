"""Optuna tuner for CoxNet (elastic net penalized Cox)."""

import optuna

from ..base import BaseTuner
from ..search_spaces import DEFAULT_SPACES


class CoxNetTuner(BaseTuner):
    """Optuna tuner for CoxNet (elastic net penalized Cox)."""

    DEFAULT_SPACE = DEFAULT_SPACES['coxnet']
    _SUPPRESSED_WARNING_PATTERNS = ("all coefficients are zero", "alpha_index=")

    def _validate_fold_scores(self, scores):
        """Prune if ALL folds returned c-index ~0.5 (all-zero coefficients).

        High regularization can produce all-zero CoxNet coefficients, yielding
        tied predictions with c-index = 0.5.  Use epsilon comparison for
        robustness against floating-point imprecision.

        Inherits all-NaN pruning from ``BaseTuner._validate_fold_scores``.
        Empty score lists pass through (``_evaluate_model`` raises
        ``TrialPruned`` for empty lists after this hook returns).
        """
        scores = super()._validate_fold_scores(scores)
        if scores and all(abs(s - 0.5) < 1e-10 for s in scores):
            raise optuna.TrialPruned()
        return scores

    def objective(self, trial):
        from ...models import CoxNetModel

        model_params = self._suggest_component_params(trial, 'model')

        if not model_params:
            model_params = {
                name: self._suggest_param(trial, name)
                for name in self.search_space
            }

        # Translate alpha_index → alpha (integer = path index, bypasses inner CV)
        alpha_index = model_params.pop('alpha_index', None)
        if alpha_index is not None:
            model_params['alpha'] = alpha_index

        model_params['fit_baseline_model'] = False
        model_params['random_state'] = self.random_state
        model = CoxNetModel(**model_params)
        return self._evaluate_model(trial, model)
