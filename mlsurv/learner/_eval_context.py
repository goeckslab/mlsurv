"""EvalContext — bundled config for evaluate() and bootstrap() (internal use).

Mirrors the SetupConfig / TuneConfig precedents: a frozen dataclass built once
at the public-API boundary that gives helpers a single, named handle to the
~20 parameters the evaluation pipeline threads through. Callers continue to
use the existing kwarg-rich method signatures; the dataclass is internal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ._evaluation_helpers import _IPCW_TAU_AUTO


@dataclass(frozen=True)
class EvalContext:
    """Bundled config shared by ``evaluate()``, ``bootstrap()``, and helpers.

    Built once at the public API boundary. ``evaluate()`` populates the
    full set; ``bootstrap()`` populates the bootstrap-only fields and leaves
    SHAP/interaction fields at their defaults. Field names mirror the
    ``evaluate()`` signature; ``bootstrap()`` translates its differently
    named params (``metrics`` → ``bootstrap_metrics`` etc.) at construction.
    """

    # Evaluation knobs
    eval_times: Any = None
    # Additive report horizons registered via add_eval_times(); unioned with
    # the resolved base grid and clipped at the engine materialization site.
    extra_eval_times: Any = None
    n_importance_repeats: Any = "auto"
    importance_max_samples: Optional[int] = None
    include_shap: bool = False
    shap_max_background: int = 50
    shap_max_explain: int = 50
    custom_metrics: Optional[dict] = None

    # Bootstrap knobs
    bootstrap: bool = False
    n_bootstrap: Any = "auto"
    bootstrap_ci: float = 0.95
    bootstrap_metrics: Optional[Sequence[str]] = None
    bootstrap_stratify: bool = False
    bootstrap_lower_is_better: Any = None  # standalone bootstrap() only
    bootstrap_include_time_resolved: bool = False  # standalone bootstrap(); gates Brier(t) only — AUC(t) is always collected
    bootstrap_include_importance: bool = False  # standalone bootstrap() only

    # Predictions / interactions
    include_predictions: bool = False
    include_interactions: bool = False
    interaction_top_k: int = 5
    interaction_max_samples: int = 100
    interaction_method: str = "hstat"
    interaction_scale: str = "auto"

    # IPCW knobs
    ipcw_tau: Any = _IPCW_TAU_AUTO
    ipcw_g_hat_source: str = "train"

    # Runtime knobs
    random_state: Optional[int] = None
    n_jobs: Optional[int] = None
    show_progress: bool = True
    verbose: bool = True
