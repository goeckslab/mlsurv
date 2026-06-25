"""Prediction result container for new-patient survival predictions."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

from ..evaluation._prediction_helpers import compute_percentile_ci
from ._containers_helpers import DictExportMixin, _nan_safe

if TYPE_CHECKING:
    import plotly.graph_objects as go


def _time_col(prefix: str, t: float) -> str:
    """Format a time value as a clean column name, e.g. ``t_12`` or ``t_12.5``."""
    return f"{prefix}_{t:g}"


def _percentile_ci_quiet(bootstrap_values, ci_level):
    """Wrap :func:`compute_percentile_ci` and suppress numpy's
    ``RuntimeWarning: All-NaN slice encountered``.

    Per-patient bootstrap arrays for median / RMST / S(t|x)-at-times can have
    patient columns that are entirely NaN — exactly the patients whose median
    was undefined. ``MethodologicalWarning("Median survival undefined for N
    patient(s)")`` already surfaces that condition upstream, so the numpy
    plumbing warning is pure noise here.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        return compute_percentile_ci(bootstrap_values, ci_level)


@dataclass
class PredictionResult(DictExportMixin):
    """Result from ``learner.predict()`` — per-patient survival predictions.

    Attributes
    ----------
    risk_scores : ndarray
        Predicted risk scores, shape ``(n_patients,)``.
    model_name : str
        Name of the model used for prediction.
    n_patients : int
        Number of patients in the prediction.

    Notes
    -----
    Survival-derived attributes (``survival_function``, ``median_survival``,
    ``rmst``, etc.) are ``None`` when the model does not support
    ``predict_survival_function()``.  Bootstrap CI attributes are ``None``
    when ``ci=False``.
    """

    # -- Core (always populated) ---------------------------------------------
    risk_scores: np.ndarray
    model_name: str
    n_patients: int

    # -- Survival function data (None if unsupported) ------------------------
    _time_grid: Optional[np.ndarray] = field(default=None, repr=False)
    _surv_probs: Optional[np.ndarray] = field(default=None, repr=False)
    _median_survival: Optional[np.ndarray] = field(default=None, repr=False)
    _rmst_values: Optional[np.ndarray] = field(default=None, repr=False)
    _rmst_default_tau: Optional[float] = field(default=None, repr=False)

    # -- User-requested times ------------------------------------------------
    _user_times: Optional[np.ndarray] = field(default=None, repr=False)
    _surv_at_user_times: Optional[np.ndarray] = field(default=None, repr=False)
    _rmst_at_user_times: Optional[np.ndarray] = field(default=None, repr=False)

    # -- Bootstrap CI data ---------------------------------------------------
    _bootstrap_medians: Optional[np.ndarray] = field(default=None, repr=False)
    _bootstrap_surv_at_times: Optional[np.ndarray] = field(default=None, repr=False)
    _bootstrap_rmst: Optional[np.ndarray] = field(default=None, repr=False)
    _bootstrap_rmst_at_times: Optional[np.ndarray] = field(default=None, repr=False)
    # Full-grid bootstrap survival probabilities:
    # shape (n_bootstrap, n_patients, n_times). Populated only when the
    # caller requested ``ci_band=True`` on ``predict()``; otherwise ``None``
    # to keep memory flat in batch usage.
    _bootstrap_surv_probs: Optional[np.ndarray] = field(default=None, repr=False)
    _ci_level: float = field(default=0.95, repr=False)

    # -- Training cohort context (populated by learner.predict; None otherwise) --
    # These fields let summary() describe how the patient compares to the training
    # cohort (percentile rank, OOD feature warnings, KNN anchor) and power
    # learner.patient_report() without re-querying the learner.
    training_cohort_size: Optional[int] = None
    test_c_index: Optional[float] = None
    time_unit: Optional[str] = None
    feature_names: Optional[list] = None
    feature_bounds: Optional[dict] = field(default=None, repr=False)
    training_risk_scores: Optional[np.ndarray] = field(default=None, repr=False)
    training_X: Optional[pd.DataFrame] = field(default=None, repr=False)
    training_y: Optional[np.ndarray] = field(default=None, repr=False)
    imputation_mask: Optional[pd.DataFrame] = field(default=None, repr=False)
    input_X: Optional[pd.DataFrame] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate critical invariants at construction time."""
        if len(self.risk_scores) != self.n_patients:
            raise ValueError(
                f"risk_scores length ({len(self.risk_scores)}) != "
                f"n_patients ({self.n_patients})"
            )
        if (self._time_grid is None) != (self._surv_probs is None):
            raise ValueError(
                "_time_grid and _surv_probs must both be set or both None."
            )
        if not (0 < self._ci_level < 1):
            raise ValueError(
                f"_ci_level must be in (0, 1), got {self._ci_level}"
            )

    # ========================================================================
    # Public properties
    # ========================================================================

    @property
    def survival_function_ci(self) -> Optional[dict]:
        """Bootstrap CI band on S(t|x) over the full time grid.

        Returns ``None`` unless ``predict(ci=True, ci_band=True)`` was called
        and the survival function is defined for this model.

        Returns
        -------
        dict or None
            ``{"lower": DataFrame, "upper": DataFrame}`` where each DataFrame
            has the same index (time grid) and columns (``patient_<i>``)
            as :attr:`survival_function`.  Quantiles are taken at
            ``(1 - ci_level) / 2`` and ``1 - (1 - ci_level) / 2``.
        """
        if self._bootstrap_surv_probs is None or self._time_grid is None:
            return None
        alpha = (1 - self._ci_level) / 2
        # Patients with all-NaN bootstrap draws (median undefined across every
        # iteration) trigger `RuntimeWarning: All-NaN slice encountered`. The
        # higher-tier MethodologicalWarning already covers the condition.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="All-NaN slice encountered",
                category=RuntimeWarning,
            )
            lo_arr = np.nanquantile(self._bootstrap_surv_probs, alpha, axis=0)
            hi_arr = np.nanquantile(self._bootstrap_surv_probs, 1 - alpha, axis=0)
        cols = [f"patient_{i}" for i in range(self.n_patients)]
        index = pd.Index(self._time_grid, name="time")
        return {
            "lower": pd.DataFrame(lo_arr.T, index=index, columns=cols),
            "upper": pd.DataFrame(hi_arr.T, index=index, columns=cols),
        }

    @property
    def survival_function(self) -> Optional[pd.DataFrame]:
        """Survival function S(t|x) as DataFrame (times as index, patients as columns).

        Returns ``None`` if the model does not support survival functions.
        """
        if self._surv_probs is None or self._time_grid is None:
            return None
        cols = [f"patient_{i}" for i in range(self.n_patients)]
        return pd.DataFrame(
            self._surv_probs.T,
            index=pd.Index(self._time_grid, name="time"),
            columns=cols,
        )

    @property
    def median_survival(self) -> Optional[np.ndarray]:
        """Median survival time per patient. ``NaN`` where undefined."""
        return self._median_survival

    @property
    def rmst(self):
        """Restricted mean survival time per patient.

        Returns
        -------
        ndarray
            When ``times`` was not specified: RMST at the default horizon
            (max training time), shape ``(n_patients,)``.
        DataFrame
            When ``times`` was specified: RMST at each horizon, with
            columns named by the time values, shape ``(n_patients, n_times)``.
        None
            If the model does not support survival functions.
        """
        if self._user_times is not None and self._rmst_at_user_times is not None:
            return pd.DataFrame(
                self._rmst_at_user_times,
                columns=[_time_col("tau", t) for t in self._user_times],
                index=pd.RangeIndex(self.n_patients, name="patient"),
            )
        return self._rmst_values

    @property
    def survival_at_times(self) -> Optional[pd.DataFrame]:
        """S(t|x) at user-specified times.

        Returns ``None`` if ``times`` was not provided or model lacks
        survival function support.
        """
        if self._surv_at_user_times is None or self._user_times is None:
            return None
        return pd.DataFrame(
            self._surv_at_user_times,
            columns=[_time_col("t", t) for t in self._user_times],
            index=pd.RangeIndex(self.n_patients, name="patient"),
        )

    # -- CI properties -------------------------------------------------------

    @property
    def median_survival_ci(self) -> Optional[pd.DataFrame]:
        """Median survival with bootstrap CIs.

        Returns ``None`` if ``ci=False`` or survival functions unavailable.
        """
        if self._bootstrap_medians is None or self._median_survival is None:
            return None
        lower, upper = _percentile_ci_quiet(self._bootstrap_medians, self._ci_level)
        return pd.DataFrame({
            "median": self._median_survival,
            "ci_lower": lower,
            "ci_upper": upper,
        }, index=pd.RangeIndex(self.n_patients, name="patient"))

    @property
    def survival_at_times_ci(self) -> Optional[pd.DataFrame]:
        """S(t|x) at user-specified times with bootstrap CIs.

        Returns a DataFrame with columns ``t_<time>``, ``t_<time>_ci_lower``,
        ``t_<time>_ci_upper`` for each requested time.
        """
        if (self._bootstrap_surv_at_times is None
                or self._surv_at_user_times is None
                or self._user_times is None):
            return None

        lower, upper = _percentile_ci_quiet(
            self._bootstrap_surv_at_times, self._ci_level,
        )
        data = {}
        for j, t in enumerate(self._user_times):
            col = _time_col("t", t)
            data[col] = self._surv_at_user_times[:, j]
            data[f"{col}_ci_lower"] = lower[:, j]
            data[f"{col}_ci_upper"] = upper[:, j]
        return pd.DataFrame(data,
                            index=pd.RangeIndex(self.n_patients, name="patient"))

    @property
    def rmst_default_ci(self) -> Optional[pd.DataFrame]:
        """Default-tau RMST with bootstrap CIs.

        Always available when bootstrap was run with survival functions,
        regardless of whether user times were requested.
        Columns: ``[rmst, ci_lower, ci_upper]``.
        """
        if self._bootstrap_rmst is None or self._rmst_values is None:
            return None
        lower, upper = _percentile_ci_quiet(self._bootstrap_rmst, self._ci_level)
        return pd.DataFrame({
            "rmst": self._rmst_values,
            "ci_lower": lower,
            "ci_upper": upper,
        }, index=pd.RangeIndex(self.n_patients, name="patient"))

    @property
    def rmst_at_times_ci(self) -> Optional[pd.DataFrame]:
        """RMST at user-specified time horizons with bootstrap CIs.

        Columns: ``[tau_<t>, tau_<t>_ci_lower, tau_<t>_ci_upper, ...]``
        per requested horizon.
        """
        if (self._bootstrap_rmst_at_times is None
                or self._rmst_at_user_times is None
                or self._user_times is None):
            return None
        lower, upper = _percentile_ci_quiet(
            self._bootstrap_rmst_at_times, self._ci_level,
        )
        data = {}
        for j, t in enumerate(self._user_times):
            col = _time_col("tau", t)
            data[col] = self._rmst_at_user_times[:, j]
            data[f"{col}_ci_lower"] = lower[:, j]
            data[f"{col}_ci_upper"] = upper[:, j]
        return pd.DataFrame(data,
                            index=pd.RangeIndex(self.n_patients, name="patient"))

    @property
    def rmst_ci(self) -> Optional[pd.DataFrame]:
        """RMST with bootstrap CIs (backward-compatible).

        Prefers per-time CIs when available, else default-tau CI.
        For both simultaneously, use :attr:`rmst_default_ci` and
        :attr:`rmst_at_times_ci`.
        """
        at_times = self.rmst_at_times_ci
        if at_times is not None:
            return at_times
        return self.rmst_default_ci

    # ========================================================================
    # Training-cohort context helpers
    # ========================================================================

    def risk_percentile(self, patient: int) -> Optional[float]:
        """Risk-score percentile rank of ``patient`` within the training cohort.

        Returns ``None`` when ``training_risk_scores`` was not provided.
        Percentile is in [0, 100]; uses the "mean" ranking convention so ties
        map to their average rank.
        """
        if self.training_risk_scores is None:
            return None
        from scipy.stats import percentileofscore
        return float(percentileofscore(
            self.training_risk_scores,
            self.risk_scores[patient],
            kind="mean",
        ))

    def ood_features(self, patient: int) -> list:
        """Features whose value for ``patient`` falls outside the training range.

        Returns a list of ``(feature_name, value, train_min, train_max)`` tuples,
        or an empty list if ``input_X`` / ``feature_bounds`` were not provided
        or all features are in-range.  Boundary values (equal to min or max) are
        treated as in-range.
        """
        if self.input_X is None or self.feature_bounds is None:
            return []
        row = self.input_X.iloc[patient]
        out = []
        for name, (lo, hi) in self.feature_bounds.items():
            if name not in row.index:
                continue
            val = row[name]
            if pd.isna(val):
                continue
            try:
                num = float(val)
            except (TypeError, ValueError):
                continue
            if num < lo or num > hi:
                out.append((name, num, float(lo), float(hi)))
        return out

    def nearest_neighbors(
        self, patient: int, k: int = 5,
    ) -> Optional[pd.DataFrame]:
        """K training patients whose predicted risk is closest to ``patient``.

        Returns a DataFrame with columns ``[train_index, risk_score, time,
        event]`` (``time`` and ``event`` only included when ``training_y`` is
        available).  Returns ``None`` if ``training_risk_scores`` is unavailable.
        """
        if self.training_risk_scores is None:
            return None
        target = self.risk_scores[patient]
        diffs = np.abs(self.training_risk_scores - target)
        k_eff = min(k, len(diffs))
        idx = np.argsort(diffs, kind="mergesort")[:k_eff]
        data = {
            "train_index": idx,
            "risk_score": self.training_risk_scores[idx],
        }
        if self.training_y is not None:
            try:
                data["time"] = np.asarray(self.training_y["time"])[idx]
                data["event"] = np.asarray(self.training_y["event"])[idx]
            except (ValueError, KeyError, IndexError):
                pass
        return pd.DataFrame(data)

    def nearest_neighbors_feature_space(
        self, patient: int, k: int = 5,
    ) -> Optional[pd.DataFrame]:
        """K training patients closest to ``patient`` in feature space.

        Uses Euclidean distance over z-scored features (mean/std computed on
        the training set) so features on different scales contribute
        comparably. Non-numeric or all-NaN features are dropped before
        distance computation. Returns ``None`` when ``training_X`` or
        ``input_X`` is unavailable.

        Returns a DataFrame with columns ``[train_index, distance,
        risk_score, time, event]`` — ``risk_score`` / ``time`` / ``event`` are
        only included when the corresponding training fields were attached.
        """
        if self.training_X is None or self.input_X is None:
            return None
        X_train = self.training_X
        x_patient = self.input_X.iloc[patient]

        numeric_cols = []
        for col in X_train.columns:
            if col not in x_patient.index:
                continue
            try:
                col_vals = X_train[col].to_numpy(dtype=float)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(col_vals).any():
                continue
            numeric_cols.append(col)
        if not numeric_cols:
            return None

        mu = X_train[numeric_cols].mean().to_numpy(dtype=float)
        sigma = X_train[numeric_cols].std(ddof=0).to_numpy(dtype=float)
        sigma = np.where(sigma > 0, sigma, 1.0)  # avoid /0 for constants

        train_z = (X_train[numeric_cols].to_numpy(dtype=float) - mu) / sigma
        try:
            patient_vals = np.asarray(
                x_patient[numeric_cols].to_numpy(dtype=float),
            )
        except (TypeError, ValueError):
            return None
        patient_z = (patient_vals - mu) / sigma
        # Replace any remaining NaN (missing patient features) with 0, matching
        # "average" — so missing values don't dominate the distance.
        patient_z = np.where(np.isfinite(patient_z), patient_z, 0.0)
        train_z = np.where(np.isfinite(train_z), train_z, 0.0)

        diffs = train_z - patient_z[np.newaxis, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        k_eff = min(k, len(dists))
        idx = np.argsort(dists, kind="mergesort")[:k_eff]
        data = {
            "train_index": idx,
            "distance": dists[idx],
        }
        if self.training_risk_scores is not None:
            data["risk_score"] = self.training_risk_scores[idx]
        if self.training_y is not None:
            try:
                data["time"] = np.asarray(self.training_y["time"])[idx]
                data["event"] = np.asarray(self.training_y["event"])[idx]
            except (ValueError, KeyError, IndexError):
                pass
        return pd.DataFrame(data)

    # ========================================================================
    # summary()
    # ========================================================================

    def summary(self, patient: int = 0) -> str:
        """Plain-language clinical summary for a single patient.

        Parameters
        ----------
        patient : int, default 0
            Patient index (0-based).

        Returns
        -------
        str

        Raises
        ------
        IndexError
            If the patient index is outside the valid range of patients.
        """
        if patient < 0 or patient >= self.n_patients:
            raise IndexError(
                f"Patient index {patient} out of range "
                f"(0..{self.n_patients - 1})."
            )

        lines = [f"Patient {patient} Prediction ({self.model_name} model)"]

        # Provenance header (only when cohort context is present)
        provenance = self._provenance_line()
        if provenance is not None:
            lines.append(f"  {provenance}")

        risk_line = f"  Risk score: {self.risk_scores[patient]:.4f}"
        if self.training_risk_scores is not None:
            lo = float(np.min(self.training_risk_scores))
            hi = float(np.max(self.training_risk_scores))
            risk_line += (
                f" (training range [{lo:.3f}, {hi:.3f}]; "
                "higher = higher risk)"
            )
        else:
            risk_line += " (higher = higher risk)"
        lines.append(risk_line)

        # Risk percentile within training cohort
        pct = self.risk_percentile(patient)
        if pct is not None:
            n_train = self.training_cohort_size or len(self.training_risk_scores)
            lines.append(
                f"  Risk percentile: {int(round(pct))}% of the "
                f"training cohort (N={n_train})"
            )

        unit_suffix = f" {self.time_unit}" if self.time_unit else ""

        # Median survival
        if self._median_survival is not None:
            med = self._median_survival[patient]
            if np.isnan(med):
                med_str = "Not reached (S(t) > 0.5 throughout follow-up)"
            else:
                med_str = f"{med:.1f}{unit_suffix}"
            ci_str = ""
            med_ci = self.median_survival_ci
            if med_ci is not None and not np.isnan(med):
                ci_str = (f" [{self._ci_pct}% CI: "
                          f"{med_ci['ci_lower'].iloc[patient]:.1f}-"
                          f"{med_ci['ci_upper'].iloc[patient]:.1f}"
                          f"{unit_suffix}]")
            lines.append(f"  Median survival: {med_str}{ci_str}")

        # S(t) at user-specified times
        sat_ci = self.survival_at_times_ci
        if self._surv_at_user_times is not None and self._user_times is not None:
            for j, t in enumerate(self._user_times):
                prob = self._surv_at_user_times[patient, j]
                prob_pct = f"{prob * 100:.0f}%"
                ci_str = ""
                if sat_ci is not None:
                    col = _time_col("t", t)
                    lo = sat_ci[f"{col}_ci_lower"].iloc[patient]
                    hi = sat_ci[f"{col}_ci_upper"].iloc[patient]
                    ci_str = (f" [{self._ci_pct}% CI: "
                              f"{lo * 100:.0f}%-{hi * 100:.0f}%]")
                t_label = (
                    f"{t:g} {self.time_unit}" if self.time_unit else f"{t:g}"
                )
                lines.append(f"  P(survive past {t_label}): {prob_pct}{ci_str}")

        # RMST at default horizon
        if self._rmst_values is not None and self._rmst_default_tau is not None:
            rmst_val = self._rmst_values[patient]
            ci_str = ""
            r_ci = self.rmst_default_ci
            if r_ci is not None and "ci_lower" in r_ci.columns:
                ci_str = (f" [{self._ci_pct}% CI: "
                          f"{r_ci['ci_lower'].iloc[patient]:.1f}-"
                          f"{r_ci['ci_upper'].iloc[patient]:.1f}"
                          f"{unit_suffix}]")
            lines.append(
                f"  RMST (to {self._rmst_default_tau:.0f}{unit_suffix}): "
                f"{rmst_val:.1f}{unit_suffix}"
            )

        # Out-of-distribution feature warnings
        ood = self.ood_features(patient)
        if ood:
            lines.append("")
            lines.append("  Out-of-distribution features (outside training range):")
            for name, val, lo, hi in ood:
                lines.append(
                    f"    - {name} = {val:g} (training range: [{lo:g}, {hi:g}])"
                )

        # Per-patient neighbor summary is intentionally omitted: with small k
        # it would be unstable, and any derived statistic mixing event times
        # with censoring is misleading. Callers who want the neighbors can
        # use :meth:`nearest_neighbors` directly (the HTML patient report
        # renders the per-neighbor table).

        lines.append(
            "\n  Note: These estimates reflect model uncertainty. Individual "
            "outcomes\n  can vary significantly -- a healthcare provider can "
            "offer a more\n  precise assessment considering additional "
            "clinical factors."
        )
        return "\n".join(lines)

    def _provenance_line(self) -> Optional[str]:
        """One-line model provenance summary.

        Returns ``None`` when no provenance fields are populated.  Includes
        only the fields that are available.
        """
        parts = [f"Model: {self.model_name}"]
        has_any = False
        if self.training_cohort_size is not None:
            parts.append(f"training cohort N={self.training_cohort_size}")
            has_any = True
        if self.test_c_index is not None:
            parts.append(f"test C-index {self.test_c_index:.2f}")
            has_any = True
        if self.feature_names is not None:
            parts.append(f"{len(self.feature_names)} features")
            has_any = True
        if not has_any:
            return None
        return " | ".join(parts)

    @property
    def _ci_pct(self) -> str:
        """CI level as a percent string (e.g. '95', '97.5')."""
        return f"{self._ci_level * 100:g}"

    # ========================================================================
    # Plotting
    # ========================================================================

    def plot(
        self,
        patient=0,
        *,
        time_unit: Optional[str] = None,
        width: Union[str, int] = "double",
        height: Union[str, int] = "medium",
    ) -> "go.Figure":
        """Plot one patient's predicted survival curve S(t|x).

        Mirrors the survival-curve panel of ``learner.patient_report()`` as a
        standalone, returnable Plotly figure. Renders the point-estimate curve
        plus whatever this result holds: a bootstrap CI band (when produced
        with ``predict(ci=True, ci_band=True)``), a median-survival marker at
        S=0.5, and milestone annotations ``P(t > t_i)`` (when ``times`` was
        passed to ``predict()``). Axis labels and captions use
        ``self.time_unit`` unless ``time_unit`` is given.

        Parameters
        ----------
        patient : int or str, default 0
            Row index into the predicted patients, or a risk-percentile spec
            evaluated against this result's ``risk_scores``: ``'low_risk'``
            (= ``'p10'``), ``'high_risk'`` (= ``'p90'``), ``'median'``
            (= ``'p50'``), or ``'pNN'`` for NN in 0..100.
        time_unit : str, optional
            Override the time unit for the x-axis label and milestone
            captions. Defaults to ``self.time_unit``.
        width, height : str or int
            Figure size (named size or pixels).

        Returns
        -------
        plotly.graph_objects.Figure

        Raises
        ------
        ValueError
            If this model produced no survival function, or if ``patient`` is
            an out-of-range index or an unrecognized spec.
        TypeError
            If ``patient`` is neither int nor str.
        """
        if self._surv_probs is None:
            raise ValueError(
                f"Model '{self.model_name}' produced no survival function, so "
                "there is no S(t|x) curve to plot. Use a model that supports "
                "predict_survival_function()."
            )
        from mlsurv._patient_select import resolve_index_from_scores
        from mlsurv.viz.survival import plot_patient_survival_curve

        idx = resolve_index_from_scores(patient, self.risk_scores)
        return plot_patient_survival_curve(
            self, idx, time_unit=time_unit, width=width, height=height,
        )

    # ========================================================================
    # Export & display
    # ========================================================================

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Export all available results as a dict of DataFrames."""
        result = {
            "risk_scores": pd.DataFrame({
                "risk_score": self.risk_scores,
            }, index=pd.RangeIndex(self.n_patients, name="patient")),
        }
        if self.survival_function is not None:
            result["survival_function"] = self.survival_function
        if self.median_survival is not None:
            result["median_survival"] = pd.DataFrame({
                "median_survival": self.median_survival,
            }, index=pd.RangeIndex(self.n_patients, name="patient"))
        if self.survival_at_times is not None:
            result["survival_at_times"] = self.survival_at_times
        rmst = self.rmst
        if rmst is not None:
            if isinstance(rmst, pd.DataFrame):
                result["rmst"] = rmst
            else:
                result["rmst"] = pd.DataFrame({
                    "rmst": rmst,
                }, index=pd.RangeIndex(self.n_patients, name="patient"))

        # CI tables
        if self.median_survival_ci is not None:
            result["median_survival_ci"] = self.median_survival_ci
        if self.survival_at_times_ci is not None:
            result["survival_at_times_ci"] = self.survival_at_times_ci
        if self.rmst_ci is not None:
            result["rmst_ci"] = self.rmst_ci
        return result

    def _summary_dict(self) -> dict:
        """Headline scalars: model identity, cohort size, test C-index."""
        return {
            "model_name": self.model_name,
            "n_patients": self.n_patients,
            "test_c_index": _nan_safe(self.test_c_index),
            "has_survival_function": self._surv_probs is not None,
        }

    def __repr__(self) -> str:
        parts = [
            f"PredictionResult(model={self.model_name!r}",
            f"n_patients={self.n_patients}",
        ]
        has_surv = self._surv_probs is not None
        parts.append(f"has_survival_function={has_surv}")
        if has_surv:
            n_times = self._time_grid.shape[0] if self._time_grid is not None else 0
            parts.append(f"n_times={n_times}")
        has_ci = self._bootstrap_medians is not None
        parts.append(f"has_ci={has_ci}")
        if self._user_times is not None:
            parts.append(f"times={list(self._user_times)}")
        return ", ".join(parts) + ")"
