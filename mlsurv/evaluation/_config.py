"""Configuration dataclasses for evaluation and bootstrap."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapConfig:
    """Bundled configuration for bootstrap iterations.

    Groups the "what to compute" parameters that are constant across
    all bootstrap iterations, reducing parameter passing from ~14 args
    to a single config object.
    """
    metrics: list
    custom_metrics: dict = field(default_factory=dict)
    eval_times: Optional[np.ndarray] = None
    ci: float = 0.95
    include_predictions: bool = False
    include_auc: bool = False
    include_brier_t: bool = False
    include_importance: bool = False
    n_importance_repeats: int = 10
    categorical_groups: Optional[Dict[str, List[str]]] = None
    stratify: bool = False
    subpopulation_labels: Optional[np.ndarray] = None
    subpopulation_unique: Optional[List[str]] = None
    ipcw_tau: Optional[float] = None
    # IPCW G_hat fitting cohort override. When None, IPCW falls back
    # to the regular y_train; when set, this cohort feeds the G_hat estimator
    # in `concordance_index_ipcw(...)`. Used by validate() to select the
    # validation cohort itself for external-validation Uno IPCW.
    ipcw_y_train: Optional[np.ndarray] = None

    def __post_init__(self):
        # Auto-derive subpopulation_unique from subpopulation_labels when not provided
        if self.subpopulation_labels is not None and self.subpopulation_unique is None:
            unique = sorted(set(
                self.subpopulation_labels[~pd.isna(self.subpopulation_labels)]
            ))
            object.__setattr__(self, 'subpopulation_unique', unique)


@dataclass(frozen=True)
class EvalConfig:
    """Bundled configuration for model evaluation.

    Groups repeated eval parameters that are passed through multiple
    call layers unchanged.
    """
    n_importance_repeats: int = 10
    # Cap on the test rows used for permutation importance. None = all rows.
    importance_max_samples: Optional[int] = None
    eval_times: Optional[np.ndarray] = None
    random_state: Optional[int] = None
    include_shap: bool = False
    shap_max_background: int = 50
    shap_max_explain: int = 50
    show_progress: bool = True
    n_jobs: int = 1
    categorical_groups: Optional[Dict[str, List[str]]] = None
    categorical_features: Optional[List[str]] = None
    include_interactions: bool = False
    interaction_top_k: int = 5
    interaction_max_samples: int = 100
    interaction_method: str = 'hstat'
    # PDP surface axis spacing for the report's 2D interaction gallery:
    # 'auto' picks linear/log/log1p per axis from each feature's skew; pass
    # 'linear' to disable, or 'log'/'log1p' to force a spacing everywhere.
    interaction_scale: str = 'auto'
    ipcw_tau: Optional[float] = None
    # IPCW G_hat fitting cohort override. When None, IPCW falls back
    # to the regular y_train; when set, this cohort feeds the G_hat estimator
    # in `concordance_index_ipcw(...)`. Used by validate() to select the
    # validation cohort itself for external-validation Uno IPCW.
    ipcw_y_train: Optional[np.ndarray] = None
    # GND calibration test bin count. Threaded to
    # greenwood_nam_dagostino so small cohorts can lower it (<= N/5) to
    # clear the >=5-samples-per-bin gate instead of returning NaN.
    gnd_n_bins: int = 10

    def __post_init__(self):
        if self.importance_max_samples is not None:
            if (isinstance(self.importance_max_samples, bool)
                    or not isinstance(self.importance_max_samples,
                                      (int, np.integer))):
                raise ValueError(
                    f"importance_max_samples must be an int or None, got "
                    f"{type(self.importance_max_samples).__name__}."
                )
            # Lower bound of 2 explained in _subsample_for_importance.
            if self.importance_max_samples < 2:
                raise ValueError(
                    f"importance_max_samples must be >= 2 or None, got "
                    f"{self.importance_max_samples}."
                )
        if self.include_shap:
            if self.shap_max_background < 1:
                raise ValueError(
                    f"shap_max_background must be >= 1, got {self.shap_max_background}."
                )
            if self.shap_max_explain < 1:
                raise ValueError(
                    f"shap_max_explain must be >= 1, got {self.shap_max_explain}."
                )
        if self.include_interactions:
            if self.interaction_top_k < 2:
                raise ValueError(
                    f"interaction_top_k must be >= 2, got {self.interaction_top_k}."
                )
            if self.interaction_max_samples < 2:
                raise ValueError(
                    f"interaction_max_samples must be >= 2, got {self.interaction_max_samples}."
                )
            valid_methods = ('hstat', 'shap')
            if self.interaction_method not in valid_methods:
                raise ValueError(
                    f"interaction_method must be one of {valid_methods}, "
                    f"got '{self.interaction_method}'."
                )
            valid_scales = ('auto', 'linear', 'log', 'log1p')
            if self.interaction_scale not in valid_scales:
                raise ValueError(
                    f"interaction_scale must be one of {valid_scales}, "
                    f"got '{self.interaction_scale}'."
                )
