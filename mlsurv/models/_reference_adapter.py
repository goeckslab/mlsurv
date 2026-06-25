"""Internal adapter wrapping raw reference score arrays as model-like objects.

ReferenceAdapter implements the minimal interface required by ``evaluate()``
and ``bootstrap()`` — specifically ``predict(X)`` — without being a real
``BaseSurvivalModel``.  It is NOT registered in the model registry and is
not part of the public API.
"""

from __future__ import annotations

import numpy as np


class ReferenceAdapter:
    """Wrap a pre-computed score array so it quacks like a fitted model.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Raw reference values aligned with the test set.
    name : str
        Short identifier (e.g. ``'psa'``).
    display_name : str, optional
        Human-readable label (e.g. ``'PSA'``).  Defaults to *name*.
    higher_is_worse : bool, default True
        If True (default), higher values indicate higher risk
        (sksurv convention).  If False, scores are negated internally so
        that ``predict()`` returns higher-is-worse values.
    """

    # Attributes expected by evaluate / bootstrap pipelines
    is_fitted_ = True
    supports_random_state = False
    prefer_sequential_cv = True
    has_internal_parallelism = False
    # Raw reference scores have no survival curve — metrics that require
    # predict_survival_function (IBS, C^td, calibration) must be skipped
    # rather than attempted and silenced at the failure handler.
    supports_survival_function = False

    def __init__(
        self,
        scores,
        name: str,
        display_name: str | None = None,
        higher_is_worse: bool = True,
    ):
        self._scores = np.asarray(scores, dtype=np.float64).ravel()
        self.name = name
        self.display_name = display_name or name
        self.higher_is_worse = higher_is_worse
        # Pre-compute the sign-corrected output
        self._risk_scores = self._scores if higher_is_worse else -self._scores

    # ------------------------------------------------------------------
    # Model interface required by evaluate() / bootstrap()
    # ------------------------------------------------------------------

    def predict(self, X=None):
        """Return pre-stored risk scores.

        When ``X`` has the same length as the stored scores, return them
        directly.  When ``X`` has a different length (e.g. training set),
        return NaN — reference values are only available for the test set,
        so training-set metrics will naturally be NaN.

        Parameters
        ----------
        X : DataFrame or array-like, optional
            If provided and length differs from stored scores, returns NaN.

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        if X is not None and hasattr(X, '__len__') and len(X) != len(self._risk_scores):
            return np.full(len(X), np.nan)
        return self._risk_scores

    def predict_survival_function(self, X=None, **kwargs):
        """Not supported for raw reference scores."""
        raise NotImplementedError(
            f"Reference '{self.name}' does not support survival function "
            "prediction. Metrics requiring survival curves (Brier score, "
            "calibration) will return NaN for references."
        )

    def predict_cumulative_hazard_function(self, X=None, **kwargs):
        """Not supported for raw reference scores."""
        raise NotImplementedError(
            f"Reference '{self.name}' does not support cumulative hazard "
            "prediction."
        )

    # ------------------------------------------------------------------
    # sklearn-compatible parameter interface (required by Pipeline / clone)
    # ------------------------------------------------------------------

    def get_params(self, deep=True):
        return {
            "name": self.name,
            "display_name": self.display_name,
            "higher_is_worse": self.higher_is_worse,
        }

    def set_params(self, **params):
        valid_keys = self.get_params().keys()
        for key, value in params.items():
            if key not in valid_keys:
                raise ValueError(
                    f"Invalid parameter '{key}' for ReferenceAdapter. "
                    f"Valid parameters: {sorted(valid_keys)}"
                )
            setattr(self, key, value)
        # Recompute derived state if sign convention changed
        if "higher_is_worse" in params:
            self._risk_scores = (
                self._scores if self.higher_is_worse else -self._scores
            )
        return self

    def __repr__(self):
        direction = "higher=worse" if self.higher_is_worse else "higher=better"
        return (
            f"ReferenceAdapter(name={self.name!r}, n={len(self._scores)}, "
            f"{direction})"
        )
