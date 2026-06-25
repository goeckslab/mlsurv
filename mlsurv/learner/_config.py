"""ConfigMixin - the current-configuration snapshot."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlsurv.learner._core import SurvivalLearner


def _groups_display_label(groups) -> str:
    """Render the resolved groups state as a stable display string."""
    if groups is None:
        return "none"
    name = getattr(groups, "name", None)
    if isinstance(name, str) and name:
        return name
    return "custom group array"


def _str_or_none(value):
    """Stringify *value* for config display, preserving None."""
    return None if value is None else str(value)


class ConfigMixin:
    """Mixin exposing the current configuration as a dictionary."""

    @property
    def config(self: SurvivalLearner):
        """
        Current configuration as a dictionary.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        s = self._state
        return {
            'test_fraction': s.test_fraction,
            'cv_folds': s.cv_folds,
            'cv_repeats': s.cv_repeats,
            'cv_ci': s.cv_ci,
            'cv_strategy': str(s.cv_strategy),
            'stratify': s.stratify_spec,
            'groups': _groups_display_label(s.groups),
            'random_state': s.random_state,
            'n_jobs': s.n_jobs,
            'presplit': s.presplit,
            'scoring': s.scoring,
            'imputer': _str_or_none(s.imputer),
            'scaler': _str_or_none(s.scaler),
            'feature_selector': _str_or_none(s.feature_selector),
            'n_train': len(s.X_train) if s.X_train is not None else None,
            'n_test': len(s.X_test) if s.X_test is not None else None,
            'n_features': len(s.X.columns) if s.X is not None else None,
            'n_events_train': int(s.y_train['event'].sum()) if s.y_train is not None else None,
            'tuned_models': [k for k, v in s.models.items() if v.is_tuned],
            'experiment_name': str(s.experiment_name) if s.experiment_name else None,
            'device': s.device,
            'time_unit': s.time_unit,
            'outcome_name': s.outcome_name,
            'report_eval_times': (
                s.user_extra_eval_times.tolist()
                if s.user_extra_eval_times is not None else []
            ),
            'check_leakage': s.check_leakage,
            'ph_test': s.ph_test_enabled,
            'linearity_test': s.linearity_test_enabled,
            'stratify_subpopulations': s.stratify_subpopulations,
            'subpopulations_declared': bool(s.subpopulations or s.subpopulations_train),
            'categorical_groups_declared': bool(s.categorical_groups),
            'categorical_features_declared': s.categorical_features is not None,
            'feature_groups_declared': s.feature_groups is not None,
        }
