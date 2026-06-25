"""Optuna tuner for XGBoost AFT."""

import warnings

from ..base import BaseTuner
from ..search_spaces import DEFAULT_SPACES


class XGBAFTTuner(BaseTuner):
    """Optuna tuner for XGBoost AFT.

    Calibration parameters (``aft_loss_distribution_scale``,
    ``aft_loss_distribution``) are not in the default search space —
    sigma is MLE-fit on training residuals inside ``XGBAFTModel.fit()``
    and the family is fixed at the model default. Override via a custom
    search space if you specifically need to tune them, but note that
    rank-based scoring metrics (Harrell, IPCW, Ctd) provide no signal
    on these parameters.
    """

    DEFAULT_SPACE = DEFAULT_SPACES['xgbaft']

    def objective(self, trial):
        from ...models import XGBAFTModel

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

        # Suppress n_jobs reproducibility warning (auto-computed, not user-chosen)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"^XGBoost with n_jobs != 1", category=UserWarning)
            model = XGBAFTModel(**model_params)
        return self._evaluate_model(trial, model)
