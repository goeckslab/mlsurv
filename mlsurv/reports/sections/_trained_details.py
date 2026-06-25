"""Trained Model Details subsection — convergence, feature selection."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from mlsurv.reports.sections.base import ALL_EXCEPTIONS
from mlsurv.utils import unwrap_pipeline, propagate_feature_names
from mlsurv.viz._palettes import get_model_display_name

if TYPE_CHECKING:
    from mlsurv.reports.sections.model_development import ModelDevelopmentSection


class _TrainedDetailsSubsection:
    """Per-model metadata: convergence, feature selection."""

    def __init__(self, section: ModelDevelopmentSection) -> None:
        self._s = section

    @staticmethod
    def _unwrap_pipeline(model):
        """Unwrap Pipeline and propagate feature names to the inner model."""
        inner = unwrap_pipeline(model)
        propagate_feature_names(inner, model)
        return inner

    def is_available(self) -> bool:
        return len(self._s._get_all_models()) > 0

    # Model name -> detail method (dispatched in add()).
    # Deep learning models share a handler and are resolved separately.
    _MODEL_HANDLERS = {
        'coxph': '_add_coxph_details',
        'coxnet': '_add_coxnet_details',
        'xgbcox': '_add_xgb_details',
        'xgbaft': '_add_xgbaft_details',
        'rsf': '_add_rsf_details',
        'gbsa': '_add_gbsa_details',
        'fssvm': '_add_fssvm_details',
        'weibull_aft': '_add_weibull_aft_details',
    }

    def add(self, md_lines: list, figures: list, figure_info: list) -> None:
        trained = self._s._get_all_models()

        label = getattr(self._s, '_model_label', None)
        for name, result in trained.items():
            display_name = label(name) if label else get_model_display_name(name)
            md_lines.append(f"#### {display_name}")
            md_lines.append("")

            model = result.model
            actual_model = self._unwrap_pipeline(model)

            # Use the algorithm identity to dispatch to the model-specific
            # handler.
            base_name = result.base_name

            if base_name in ('deepsurv', 'deephit'):
                self._add_deep_learning_details(md_lines, actual_model, display_name)
            elif base_name in self._MODEL_HANDLERS:
                getattr(self, self._MODEL_HANDLERS[base_name])(md_lines, actual_model)
            else:
                md_lines.append(
                    f"*No model-specific training details available for "
                    f"{display_name}.*"
                )
                md_lines.append("")

            # Feature selection info (all models)
            self._add_feature_selection_info(md_lines, model)

            # Internal S(t|x) calibrator (train(calibrate=...)), if any.
            self._add_calibration_info(md_lines, result)


    # Friendly descriptions for the internal calibrators (train(calibrate=...)).
    _CAL_DESCRIPTIONS = {
        "breslow_slope": (
            "Breslow-slope (van Houwelingen) — S(t|x) recomputed from a Breslow "
            "baseline refit on the model's risk score"
        ),
        "lowdof": "low-DoF (2-parameter cloglog) calibration of S(t|x)",
        "isotonic": "per-horizon isotonic calibration of S(t|x)",
    }

    def _add_calibration_info(self, md_lines, result) -> None:
        """Report the internal calibrator attached by ``train(calibrate=...)``.

        Emits nothing for an uncalibrated model. For a calibrated one, names the
        method (fit on held-out training folds) and, when available, the
        before/after test calibration slope.
        """
        cal = getattr(result, "calibrator", None)
        if cal is None:
            return
        method = getattr(cal, "method", "?")
        desc = self._CAL_DESCRIPTIONS.get(method, f"{method} calibration of S(t|x)")
        md_lines.append(
            f"**Calibration:** {desc} (`{method}`), fit on held-out training "
            f"folds. Applied to the reported metrics, predict(), and bootstrap; "
            f"the C-index is unchanged (monotone)."
        )
        post = self._test_cal_slope(result)
        pre = getattr(result, "precalibration_slope", None)
        if (pre is not None and post is not None
                and math.isfinite(pre) and math.isfinite(post)):
            md_lines.append("")
            md_lines.append(
                f"Test calibration slope: {pre:.2f} (raw) -> {post:.2f} "
                f"(calibrated; target 1.0)."
            )
        md_lines.append("")

    @staticmethod
    def _test_cal_slope(result):
        """Calibrated test calibration slope from the evaluation, or None."""
        ev = getattr(result, "evaluation", None)
        concordance = getattr(ev, "concordance", None) if ev is not None else None
        if concordance is None:
            return None
        try:
            if "cal_slope_test" not in concordance.columns:
                return None
            val = concordance["cal_slope_test"].iloc[0]
            val = float(val)
            return val if math.isfinite(val) else None
        except ALL_EXCEPTIONS:
            return None

    # ------------------------------------------------------------------
    # Per-model detail methods
    # ------------------------------------------------------------------

    def _add_coxph_details(self, md_lines, model):
        """CoxPH: convergence info."""
        overflow = getattr(model, 'overflow_detected_', False)
        alpha = getattr(model, 'alpha_', None)
        if overflow:
            md_lines.append(
                "**Convergence:** Overflow detected during fitting; "
                f"penalizer alpha={alpha} was applied for numerical stability."
            )
        else:
            md_lines.append("**Convergence:** Model converged normally.")
        md_lines.append("")

    def _add_coxnet_details(self, md_lines, model):
        """CoxNet: effective sparsity from L1 regularization."""
        coefs = None
        if hasattr(model, 'get_coefficients'):
            try:
                coefs = model.get_coefficients()
            except ValueError:
                pass
        if coefs is not None:
            n_nonzero = int((coefs != 0).sum())
            n_total = len(coefs)
            md_lines.append(
                f"**Effective sparsity:** {n_nonzero} of {n_total} features "
                f"have non-zero coefficients after L1 regularization."
            )
            md_lines.append("")

    def _add_deep_learning_details(self, md_lines, model, display_name):
        """DeepSurv/DeepHit: epochs, convergence."""
        actual_epochs = getattr(model, 'actual_epochs_', None)
        max_epochs = getattr(model, 'epochs', None)
        converged = getattr(model, 'converged_', None)

        if actual_epochs is not None and max_epochs is not None:
            if converged is True:
                md_lines.append(
                    f"**Training:** Early stopping triggered at epoch {actual_epochs} "
                    f"of {max_epochs} maximum epochs (converged)."
                )
            else:
                md_lines.append(
                    f"**Training:** Ran all {max_epochs} epochs "
                    f"(early stopping not triggered)."
                )
            md_lines.append("")
        else:
            md_lines.append(
                f"*Training details for {display_name} are not available.*"
            )
            md_lines.append("")

    def _add_xgb_details(self, md_lines, model):
        """XGBoost-Cox: boosting rounds (trimmed when early stopping fired)."""
        inner = getattr(model, 'model_', model)
        requested = getattr(model, 'n_estimators', None)
        best_iter = getattr(inner, 'best_iteration', None)
        # best_iteration is 0-indexed; rounds actually used = best_iter + 1.
        actual = best_iter + 1 if best_iter is not None else None
        self._add_boosting_rounds(md_lines, actual, requested)

    def _add_xgbaft_details(self, md_lines, model):
        """XGBoost-AFT: boosting rounds (trimmed when early stopping fired)."""
        booster = getattr(model, 'booster_', None)
        actual = None
        if booster is not None:
            num_rounds_fn = getattr(booster, 'num_boosted_rounds', None)
            try:
                actual = num_rounds_fn() if callable(num_rounds_fn) else None
            except ALL_EXCEPTIONS:
                actual = None
        self._add_boosting_rounds(md_lines, actual, getattr(model, 'n_estimators', None))

    @staticmethod
    def _add_boosting_rounds(md_lines, actual, requested):
        """Render a boosting-round count, flagging early stopping when it fired."""
        if actual is not None and requested is not None and actual < requested:
            md_lines.append(
                f"**Training:** Early stopping: {actual} of {requested} "
                f"boosting rounds used."
            )
            md_lines.append("")
        elif actual is not None:
            md_lines.append(f"**Training:** {actual} boosting rounds.")
            md_lines.append("")
        elif requested is not None:
            md_lines.append(f"**Training:** {requested} boosting rounds.")
            md_lines.append("")

    @staticmethod
    def _add_rsf_details(md_lines, model):
        """RSF: number of trees, OOB score if available."""
        parts = []
        n_trees = getattr(model, 'n_estimators', None)
        if n_trees is not None:
            parts.append(f"{n_trees} trees")
        inner = getattr(model, 'model_', None)
        oob = getattr(inner, 'oob_score_', None) if inner is not None else None
        if oob is not None:
            parts.append(f"OOB concordance {oob:.4f}")
        if parts:
            md_lines.append(f"**Ensemble:** {', '.join(parts)}.")
            md_lines.append("")

    def _add_gbsa_details(self, md_lines, model):
        """GBSA: actual estimators (may differ if early stopping active)."""
        inner = getattr(model, 'model_', None)
        actual_n = getattr(inner, 'n_estimators_', None) if inner else None
        requested_n = getattr(model, 'n_estimators', None)
        if actual_n is not None and requested_n is not None and actual_n < requested_n:
            md_lines.append(
                f"**Ensemble:** Early stopping — {actual_n} of "
                f"{requested_n} estimators used."
            )
            md_lines.append("")
        elif requested_n is not None:
            md_lines.append(f"**Ensemble:** {requested_n} estimators.")
            md_lines.append("")

    def _add_fssvm_details(self, md_lines, model):
        """FSSVM: convergence status and iterations."""
        iters = getattr(model, 'actual_iterations_', None)
        converged = getattr(model, 'converged_', None)
        if iters is not None:
            if converged is True:
                status = "converged"
            elif converged is False:
                status = "did not converge"
            else:
                status = "convergence unknown"
            md_lines.append(
                f"**Optimization:** {status} in {iters} iterations."
            )
            md_lines.append("")
        # S(t|x) calibration for FSSVM is attached at the learner level via
        # train(calibrate=...) (breslow_slope by default); see the per-model
        # calibration row in Environment & Reproducibility.

    def _add_weibull_aft_details(self, md_lines, model):
        """Weibull-AFT: AIC/BIC from lifelines."""
        inner = getattr(model, 'model_', None)
        if inner is None:
            return
        parts = []
        aic = getattr(inner, 'AIC_', None)
        if aic is not None:
            parts.append(f"AIC={aic:.1f}")
        bic = getattr(inner, 'BIC_', None)
        if bic is not None:
            parts.append(f"BIC={bic:.1f}")
        if parts:
            md_lines.append(f"**Model fit:** {', '.join(parts)}.")
            md_lines.append("")

    # ------------------------------------------------------------------
    # Feature selection (all models)
    # ------------------------------------------------------------------

    def _add_feature_selection_info(self, md_lines, model):
        """Report selected/removed features from pipeline selector."""
        if not hasattr(model, 'named_steps') or 'selector' not in model.named_steps:
            return

        selector = model.named_steps['selector']
        selected = getattr(selector, 'selected_features_', None)
        removed = getattr(selector, 'removed_features_', None)

        if selected is None and removed is None:
            return

        if selected is not None:
            md_lines.append(f"**Selected features ({len(selected)}):** {', '.join(str(f) for f in selected)}")
            md_lines.append("")
        if removed is not None and len(removed) > 0:
            md_lines.append(f"**Removed features ({len(removed)}):** {', '.join(str(f) for f in removed)}")
            md_lines.append("")
