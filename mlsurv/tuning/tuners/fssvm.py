"""Optuna tuner for Fast Survival SVM."""

import sys

from ..base import BaseTuner
from ..search_spaces import DEFAULT_SPACES

# Optimizers that segfault on Windows (scikit-survival 0.26.0 C extension bug)
_WINDOWS_UNSAFE_OPTIMIZERS = frozenset({'PRSVM', 'simple'})


class FSSVMTuner(BaseTuner):
    """Optuna tuner for Fast Survival SVM.

    Tunes alpha, max_iter, tol, and rank_ratio. The optimizer is hardcoded
    to 'avltree' (O(n log n), all platforms) because all optimizers solve
    the same convex problem and produce identical solutions. fit_intercept
    is hardcoded to False because True causes Breslow survival curve
    collapse due to prediction scale mismatch.

    Custom search_spaces can still override these via _suggest_component_params.
    Invalid combos are validated before model fitting so they raise ValueError
    directly (not silently pruned by Optuna).
    """

    DEFAULT_SPACE = DEFAULT_SPACES['fssvm']

    @staticmethod
    def _validate_custom_params(params):
        """Validate custom search_space params before model fitting.

        Raises ValueError directly so Optuna treats these as study-level
        errors, not silently-pruned trials.
        """
        opt = params.get('optimizer')
        rr = params.get('rank_ratio')
        fi = params.get('fit_intercept')

        if sys.platform == "win32" and opt in _WINDOWS_UNSAFE_OPTIMIZERS:
            raise ValueError(
                f"optimizer='{opt}' in search_spaces is not supported on "
                "Windows (segfault). Remove PRSVM/simple from search_spaces."
            )
        if fi and rr == 1.0:
            raise ValueError(
                "search_spaces includes fit_intercept=True with "
                "rank_ratio=1.0, which is invalid (pure ranking mode)."
            )
        if opt in ('PRSVM', 'simple') and rr is not None and rr != 1.0:
            raise ValueError(
                f"search_spaces includes optimizer='{opt}' with "
                f"rank_ratio={rr}, but PRSVM/simple require rank_ratio=1.0."
            )

    def objective(self, trial):
        from ...models import FSSVMModel

        model_params = self._suggest_component_params(trial, 'model')

        if model_params:
            self._validate_custom_params(model_params)
        else:
            # Iterate the resolved search space (same pattern as CoxNetTuner) so
            # every declared parameter is tuned and the dict cannot drift out of
            # sync with DEFAULT_SPACES. optimizer and fit_intercept
            # are fixed on this default path; a custom search space can still
            # tune them via the _validate_custom_params branch above.
            model_params = {
                name: self._suggest_param(trial, name)
                for name in self.search_space
            }
            model_params['optimizer'] = 'avltree'
            model_params['fit_intercept'] = False

        model_params['random_state'] = self.random_state

        model = FSSVMModel(**model_params)
        return self._evaluate_model(trial, model)
