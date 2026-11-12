"""XGBoost Accelerated Failure Time (AFT) model wrapper."""

import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

from mlsurv._warnings import MethodologicalWarning, SuggestionWarning
from mlsurv._xgboost import import_xgboost, warn_if_torch_loaded_on_macos
from mlsurv.models.base import BaseSurvivalModel, SurvivalArray


_VALID_DISTRIBUTIONS = ('normal', 'logistic', 'extreme')


class XGBAFTModel(BaseSurvivalModel):
    """
    XGBoost Accelerated Failure Time (AFT) model wrapper.

    Wraps XGBoost's ``survival:aft`` objective for parametric survival
    analysis.  Unlike the Cox objective (``XGBCoxModel``), AFT directly
    models the log survival time as a location-scale distribution, yielding
    fully parametric survival curves without a Breslow estimator.

    Parameters
    ----------
    n_estimators : int, default=300
        Number of boosting rounds (trees). With early stopping on by default,
        the actual round count is typically trimmed well below this cap.
    max_depth : int, default=3
        Maximum tree depth. Shallow trees suit low-EPV survival data and match
        scikit-survival's gradient-boosting default.
    learning_rate : float, default=0.10
        Step size shrinkage (eta).
    subsample : float, default=1.0
        Fraction of samples for each tree (0, 1].
    colsample_bytree : float, default=1.0
        Fraction of features for each tree (0, 1].
    gamma : float, default=0
        Minimum loss reduction for split.
    min_child_weight : int, default=1
        Minimum sum of instance weight in child.
    aft_loss_distribution : str, default='normal'
        Distribution for the AFT loss function.  One of:

        - ``'normal'`` — log-normal AFT (most common)
        - ``'logistic'`` — log-logistic AFT
        - ``'extreme'`` — Weibull AFT (Gumbel minimum)
    aft_loss_distribution_scale : float or None, default=None
        Scale parameter (sigma) for the AFT distribution. ``None`` (default)
        triggers censoring-aware MLE on out-of-fit residuals from a small
        internal hold-out, then refits the production booster on the
        full data with the calibrated sigma. Pass an explicit float to
        fix sigma and skip MLE.
    early_stopping_rounds : int or None, default=10
        Stop boosting when the held-out ``aft-nloglik`` has not improved for
        this many rounds. ``None`` disables early stopping (train the full
        ``n_estimators`` rounds). When set, a fit-fold booster finds the
        optimal round count on a held-out fold, then the production booster
        trains on the full data for exactly that many rounds. With auto-MLE
        sigma (the default) that fold is the sigma-calibration hold-out; with
        a user-supplied sigma it is a separate ``validation_fraction`` split.
        Auto-disables with a warning when the data is too small to hold out a
        validation fold.
    validation_fraction : float, default=0.1
        Fraction held out for early stopping when sigma is user-supplied
        (so no calibration hold-out is carved). When sigma is auto-MLE'd,
        early stopping reuses the calibration hold-out instead and this value
        is unused. Ignored when ``early_stopping_rounds`` is ``None``.
    n_jobs : int, default=1
        Number of parallel threads for tree building.

        .. warning::
           When n_jobs != 1, results may not be perfectly reproducible even
           with the same random_state due to floating-point operation ordering
           in parallel execution. For fully reproducible results, use n_jobs=1.

    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    booster_ : xgboost.Booster
        The fitted XGBoost booster (low-level API).
    sigma_ : float
        Fitted scale parameter used for ``predict_survival_function``. Set
        by MLE on training residuals when ``aft_loss_distribution_scale``
        is ``None`` (default), otherwise mirrors the user-supplied value.
    is_fitted_ : bool
        Whether the model has been fitted.
    train_times_ : ndarray
        Observed event/censoring times from training data (for default
        survival function time grids).
    n_rounds_ : int
        Number of boosting rounds the production booster was trained for
        (``best_iteration + 1`` when early stopping trimmed it, otherwise
        ``n_estimators``). Named to avoid confusion with XGBoost's 0-indexed
        ``best_iteration``, which this is the count form of.

    Notes
    -----
    - Uses ``survival:aft`` objective for accelerated failure time modelling
    - Label encoding: lower/upper bounds via DMatrix (not negative labels)
    - ``predict()`` returns negated log predicted survival time (higher = higher risk)
    - ``predict_survival_function()`` uses parametric S(t) from the chosen distribution
    - Sigma is auto-MLE'd by default on residuals from a small internal
      hold-out (so over-fit trees don't compress sigma toward zero).
      The production booster is re-trained on the full data with the
      calibrated sigma. Pass an explicit ``aft_loss_distribution_scale``
      to bypass.

    Examples
    --------
    >>> from mlsurv.models import XGBAFTModel
    >>> model = XGBAFTModel(n_estimators=200, aft_loss_distribution='logistic')
    >>> model.fit(X_train, y_train)
    >>> risk_scores = model.predict(X_test)

    References
    ----------
    - Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
    - Barnwal, A., et al. (2022). Survival regression with accelerated failure
      time model in XGBoost. Journal of Computational and Graphical Statistics.
    """

    name = "xgbaft"
    display_name = "XGBoost-AFT"
    color = "#AA4499"
    supports_random_state = True
    prefer_sequential_cv = False
    has_internal_parallelism = True
    complexity = "tree_ensemble"

    def __init__(
        self,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.10,
        subsample=1.0,
        colsample_bytree=1.0,
        gamma=0,
        min_child_weight=1,
        aft_loss_distribution='normal',
        aft_loss_distribution_scale=None,
        early_stopping_rounds=10,
        validation_fraction=0.1,
        n_jobs=1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.aft_loss_distribution = aft_loss_distribution
        self.aft_loss_distribution_scale = aft_loss_distribution_scale
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.booster_ = None
        self.train_times_ = None
        self.sigma_ = None
        self.n_rounds_ = None

    def fit(self, X, y: SurvivalArray):
        """
        Fit the XGBoost AFT model.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Training features.
        y : SurvivalArray
            Survival outcomes with dtype=[('event', bool), ('time', float)].

        Returns
        -------
        self : XGBAFTModel
            Fitted model.

        Raises
        ------
        ImportError
            If XGBoost is not installed, or is installed but fails to load
            (a missing OpenMP runtime, most often on macOS).
        ValueError
            If the AFT loss distribution is unrecognized, the distribution
            scale is non-positive, or any survival time is non-positive.

        Warns
        -----
        SuggestionWarning
            On macOS, when PyTorch is already loaded, since the fit can hang.
        """
        xgb = import_xgboost("XGBAFTModel")
        warn_if_torch_loaded_on_macos()

        if self.aft_loss_distribution not in _VALID_DISTRIBUTIONS:
            raise ValueError(
                f"aft_loss_distribution must be one of {_VALID_DISTRIBUTIONS}, "
                f"got {self.aft_loss_distribution!r}"
            )

        if (
            self.aft_loss_distribution_scale is not None
            and self.aft_loss_distribution_scale <= 0
        ):
            raise ValueError(
                f"aft_loss_distribution_scale must be strictly positive or "
                f"None (auto-MLE), got {self.aft_loss_distribution_scale!r}. "
                f"This is the scale parameter (sigma) of the AFT distribution."
            )

        self._validate_data(X, y)

        # AFT operates in log-time space; times must be strictly positive.
        times = y['time'].astype(np.float64)
        if np.any(times <= 0):
            n_bad = int(np.sum(times <= 0))
            raise ValueError(
                f"AFT model requires strictly positive survival times "
                f"(log-time space), but {n_bad} observation(s) have "
                f"time <= 0 (min={float(times.min()):.6g}). "
                f"Filter or impute these before fitting."
            )

        # Resolve sigma and the production round count together.
        #   - sigma: user-supplied (skip MLE) or auto-MLE'd on out-of-fit
        #     residuals from a small hold-out (avoids over-fit-driven sigma
        #     compression from in-sample residuals).
        #   - n_rounds: n_estimators, unless early stopping on a held-out
        #     fold trims it. When sigma is auto-MLE'd, early stopping reuses
        #     the same calibration hold-out (so the two stay consistent).
        # Dispatch on whether sigma is user-supplied. With auto-MLE sigma the
        # calibration hold-out doubles as the early-stopping fold; with a
        # fixed sigma a separate validation_fraction split drives early
        # stopping. Both keep _calibrate_sigma_holdout one frame below fit()
        # so its warnings point at the caller.
        if self.aft_loss_distribution_scale is not None:
            self.sigma_ = float(self.aft_loss_distribution_scale)
            if self.early_stopping_rounds is not None:
                n_rounds = self._rounds_for_fixed_sigma(X, y, self.sigma_, xgb)
            else:
                n_rounds = self.n_estimators
        else:
            self.sigma_, n_rounds = self._calibrate_sigma_holdout(
                X, y, times, xgb,
            )

        # Train the production booster on the full data using the calibrated
        # sigma and the resolved round count.
        self.booster_ = self._train_booster(
            X, y, self.sigma_, xgb, num_boost_round=n_rounds,
        )
        self.n_rounds_ = n_rounds
        self.train_times_ = y['time'].copy()

        self.is_fitted_ = True
        return self

    def _rounds_for_fixed_sigma(self, X, y, sigma, xgb):
        """Early-stopped round count when sigma is user-supplied.

        Carves a stratified hold-out and trains a fit-fold booster with
        early stopping using the fixed sigma. Returns the optimal round
        count, or ``n_estimators`` when no hold-out is viable (the shared
        split helper warns in that case).
        """
        fit_idx, val_idx = self._stratified_es_split(
            y, self.validation_fraction,
        )
        if fit_idx is None:
            return self.n_estimators
        d_fit = self._build_dmatrix(self._take_rows(X, fit_idx), y[fit_idx], xgb)
        d_val = self._build_dmatrix(self._take_rows(X, val_idx), y[val_idx], xgb)
        booster = self._early_stopped_booster(d_fit, d_val, sigma, xgb)
        return booster.best_iteration + 1

    def _early_stopped_booster(self, d_fit, d_val, sigma, xgb):
        """Train a fit-fold booster with early stopping on ``d_val``.

        Monitors ``aft-nloglik`` on the held-out fold. Returns the booster;
        read ``booster.best_iteration`` for the optimal round count.
        """
        return xgb.train(
            self._booster_params(sigma), d_fit,
            num_boost_round=self.n_estimators,
            evals=[(d_val, 'validation')],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )

    def _warn_early_stopping_skipped(self):
        """Warn that a non-viable sigma hold-out also disabled early stopping.

        The sigma-calibration fallbacks reuse their hold-out for early
        stopping; when that hold-out is not viable, early stopping is dropped
        too. The user asked for it (``early_stopping_rounds`` is set), so say
        so explicitly rather than silently training the full round count.
        """
        if self.early_stopping_rounds is not None:
            warnings.warn(
                "XGBoost-AFT: early stopping also disabled because no viable "
                "sigma-calibration hold-out is available; training the full "
                "n_estimators rounds.",
                SuggestionWarning,
                # One frame deeper than the sibling warnings in
                # _calibrate_sigma_holdout (this is a helper it calls).
                stacklevel=4,
            )

    def _build_dmatrix(self, X, y, xgb):
        """Construct a DMatrix with AFT lower/upper label bounds."""
        dmat = xgb.DMatrix(X)
        lower = y['time'].astype(np.float64)
        upper = np.where(y['event'], y['time'].astype(np.float64), np.inf)
        dmat.set_float_info('label_lower_bound', lower)
        dmat.set_float_info('label_upper_bound', upper)
        return dmat

    def _booster_params(self, sigma):
        """Assemble the ``xgb.train`` params dict for the given sigma."""
        params = {
            'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'aft_loss_distribution': self.aft_loss_distribution,
            'aft_loss_distribution_scale': float(sigma),
            'max_depth': self.max_depth,
            'eta': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'gamma': self.gamma,
            'min_child_weight': self.min_child_weight,
            'nthread': self.n_jobs,
            'tree_method': 'hist',
            'verbosity': 0,
        }
        if self.random_state is not None:
            params['seed'] = self.random_state
        return params

    def _train_booster(self, X, y, sigma, xgb, num_boost_round=None):
        """Train an XGBoost AFT booster with the given sigma.

        ``num_boost_round`` defaults to ``n_estimators``; pass a smaller
        value to train the production booster for an early-stopped count.
        """
        dtrain = self._build_dmatrix(X, y, xgb)
        if num_boost_round is None:
            num_boost_round = self.n_estimators
        return xgb.train(
            self._booster_params(sigma), dtrain,
            num_boost_round=num_boost_round,
        )

    def _calibrate_sigma_holdout(
        self, X, y, times, xgb, holdout_frac=0.2, min_holdout_events=5,
    ):
        """Estimate sigma via MLE on a held-out fraction; resolve round count.

        Trains a temporary booster on (1 - holdout_frac) of the rows
        and MLE-fits sigma on residuals from the held-out rows. The
        held-out residuals are out-of-fit, so sigma reflects
        generalisation error rather than training memorisation. The
        production booster is then re-trained on the full data with
        this sigma.

        When early stopping is enabled, the same hold-out drives it: the
        calibration booster monitors ``aft-nloglik`` on the held-out fold,
        and the resulting ``best_iteration`` becomes the production round
        count (sigma is MLE'd at that same iteration so the two stay
        consistent).

        Falls back to in-sample residuals (with a warning) and the full
        ``n_estimators`` round count if the holdout would be too small to
        give a reliable MLE.

        Returns
        -------
        sigma : float
        n_rounds : int
            Boosting rounds for the production booster.
        """
        from sklearn.model_selection import train_test_split

        events = y['event'].astype(bool)
        n_total = len(y)
        n_holdout = int(np.ceil(holdout_frac * n_total))
        n_events_total = int(events.sum())

        # All-censored: MLE is undefined; skip booster training entirely
        # and return the documented fallback. Methodological warning:
        # without events the user should know calibration is degenerate.
        if n_events_total == 0:
            warnings.warn(
                "XGBoost-AFT: zero events in training data; sigma MLE "
                "is undefined. Falling back to sigma=1.0; predicted "
                "survival functions are not data-calibrated. "
                "Provide an explicit aft_loss_distribution_scale to "
                "silence this warning.",
                MethodologicalWarning,
                stacklevel=3,
            )
            self._warn_early_stopping_skipped()
            return 1.0, self.n_estimators

        # Holdout viability: enough total samples and enough events to
        # support both fit and MLE on the held-out portion.
        can_holdout = (
            n_holdout >= 30
            and n_total - n_holdout >= 50
            and n_events_total >= 2 * min_holdout_events
        )
        if not can_holdout:
            warnings.warn(
                "XGBoost-AFT: dataset too small for hold-out sigma "
                f"calibration (n={n_total}, events={n_events_total}); "
                "falling back to in-sample residuals (sigma may be "
                "biased toward zero on over-fit configurations, which "
                "produces over-confident survival probabilities).",
                MethodologicalWarning,
                stacklevel=3,
            )
            self._warn_early_stopping_skipped()
            sigma = self._sigma_from_in_sample(X, y, times, events, xgb)
            return sigma, self.n_estimators

        # Stratify on event so both folds have events/censorings.
        idx = np.arange(n_total)
        try:
            fit_idx, calib_idx = train_test_split(
                idx,
                test_size=holdout_frac,
                random_state=self.random_state,
                stratify=events,
            )
        except ValueError:
            # Stratification can fail if a class has too few members.
            # Warn so the user knows the calibration split is degraded.
            warnings.warn(
                "XGBoost-AFT: stratified hold-out split failed "
                "(likely class imbalance); using non-stratified split. "
                "Sigma calibration may be noisier than usual.",
                SuggestionWarning,
                stacklevel=3,
            )
            fit_idx, calib_idx = train_test_split(
                idx, test_size=holdout_frac, random_state=self.random_state,
            )

        X_fit = self._take_rows(X, fit_idx)
        X_calib = self._take_rows(X, calib_idx)
        y_fit = y[fit_idx]
        y_calib = y[calib_idx]
        times_calib = times[calib_idx]

        if int(y_calib['event'].sum()) < min_holdout_events:
            warnings.warn(
                "XGBoost-AFT: hold-out fold has too few events for "
                "sigma MLE; falling back to in-sample residuals "
                "(sigma may be biased toward zero on over-fit "
                "configurations).",
                MethodologicalWarning,
                stacklevel=3,
            )
            self._warn_early_stopping_skipped()
            sigma = self._sigma_from_in_sample(X, y, times, events, xgb)
            return sigma, self.n_estimators

        # Train the calibration booster on the fit fold. With early stopping,
        # monitor aft-nloglik on the held-out fold to find the optimal round
        # count, and predict residuals at that iteration so the MLE'd sigma
        # matches the early-stopped model. The calibration booster uses
        # sigma=1.0 (sigma is MLE'd from its residuals, not learned).
        d_calib = self._build_dmatrix(X_calib, y_calib, xgb)
        if self.early_stopping_rounds is not None:
            d_fit = self._build_dmatrix(X_fit, y_fit, xgb)
            calib_booster = self._early_stopped_booster(
                d_fit, d_calib, 1.0, xgb,
            )
            n_rounds = calib_booster.best_iteration + 1
            mu_calib = calib_booster.predict(
                d_calib, output_margin=True, iteration_range=(0, n_rounds),
            )
        else:
            calib_booster = self._train_booster(X_fit, y_fit, 1.0, xgb)
            n_rounds = self.n_estimators
            mu_calib = calib_booster.predict(d_calib, output_margin=True)

        sigma = self._mle_sigma(
            np.log(times_calib), mu_calib, y_calib['event'],
        )
        return sigma, n_rounds

    def _sigma_from_in_sample(self, X, y, times, events, xgb):
        """In-sample residual MLE used as the hold-out fallback path."""
        booster = self._train_booster(X, y, 1.0, xgb)
        mu = booster.predict(self._build_dmatrix(X, y, xgb), output_margin=True)
        return self._mle_sigma(np.log(times), mu, events)

    def predict(self, X):
        """
        Predict risk scores.

        Returns the negated log predicted survival time so that higher
        values indicate higher risk, consistent with the convention used
        by all other mlsurv models.

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        risk_scores : ndarray
            Risk scores (higher = higher risk = shorter predicted survival).
        """
        xgb = import_xgboost("XGBAFTModel")

        self._check_is_fitted()
        # output_margin=True returns the raw margin = log(predicted_time).
        # Default predict() returns exp(margin) = predicted_time, but we
        # need log-scale for meaningful risk scores and sign inversion.
        log_time = self.booster_.predict(xgb.DMatrix(X), output_margin=True)
        # Negate: higher log_time → longer survival → lower risk.
        return -log_time

    def _distribution_log_pdf(self, z):
        """Log density of the configured AFT distribution at standardized z."""
        if self.aft_loss_distribution == 'normal':
            return stats.norm.logpdf(z)
        elif self.aft_loss_distribution == 'logistic':
            return stats.logistic.logpdf(z)
        elif self.aft_loss_distribution == 'extreme':
            # Gumbel-min log-density: log f(z) = z - exp(z)
            return z - np.exp(z)
        else:
            raise ValueError(
                f"Unknown AFT distribution: {self.aft_loss_distribution!r}. "
                f"Expected one of {_VALID_DISTRIBUTIONS}."
            )

    _SIGMA_BOUNDS = (1e-3, 50.0)

    def _mle_sigma(self, log_t, mu, event, fallback=1.0):
        """Censoring-aware MLE for sigma on training residuals.

        Maximises the AFT log-likelihood
            uncensored: log f((log t - mu)/sigma) - log sigma
            censored:   log(1 - F((log t - mu)/sigma))
        over sigma in ``_SIGMA_BOUNDS`` using bounded scalar
        optimisation. Emits a ``MethodologicalWarning`` and returns
        ``fallback`` for any path where the data-driven estimate is
        unavailable (zero events, optimizer failure, optimizer pinned
        at a boundary) — the returned sigma directly drives
        ``predict_survival_function`` so users must be able to see
        when calibration fell through.
        """
        event = np.asarray(event, dtype=bool)
        n_events = int(event.sum())
        lo, hi = self._SIGMA_BOUNDS
        if n_events == 0:
            warnings.warn(
                "XGBoost-AFT cannot MLE-fit sigma with zero events; "
                f"falling back to sigma={fallback}. Provide an explicit "
                "aft_loss_distribution_scale to silence this warning.",
                MethodologicalWarning,
                stacklevel=3,
            )
            return float(fallback)

        resid = log_t - mu

        def neg_log_lik(sigma):
            z = resid / sigma
            log_pdf = self._distribution_log_pdf(z)
            cdf = self._distribution_cdf(z)
            log_surv = np.log(np.clip(1.0 - cdf, 1e-300, 1.0))
            ll = np.where(event, log_pdf - np.log(sigma), log_surv)
            total = float(np.sum(ll))
            # Treat non-finite contributions (e.g., extreme-distribution
            # exp(z) overflow at extreme sigma) as the worst point so
            # the optimizer steers away.
            return -total if np.isfinite(total) else np.inf

        result = optimize.minimize_scalar(
            neg_log_lik, bounds=self._SIGMA_BOUNDS, method='bounded',
            options={'xatol': 1e-4},
        )
        if not result.success or not np.isfinite(result.x) or result.x <= 0:
            warnings.warn(
                f"XGBoost-AFT sigma MLE failed to converge "
                f"({result.message!r}); falling back to sigma={fallback}.",
                MethodologicalWarning,
                stacklevel=3,
            )
            return float(fallback)
        # Boundary hits indicate the data wants a sigma outside the
        # admissible range — a methodological concern (model misspecified
        # or numerical instability), not just a diagnostic.
        sigma = float(result.x)
        if abs(sigma - lo) < 1e-3 or abs(sigma - hi) < 1e-3:
            warnings.warn(
                f"XGBoost-AFT sigma MLE pinned at the optimizer boundary "
                f"(sigma={sigma:.4f}, bounds={self._SIGMA_BOUNDS}). "
                "This usually signals model mis-specification "
                "(wrong distribution family) or numerical instability "
                "in the residuals.",
                MethodologicalWarning,
                stacklevel=3,
            )
        return sigma

    def _distribution_cdf(self, z):
        """Compute CDF of the configured AFT distribution.

        Parameters
        ----------
        z : ndarray
            Standardized values ``(log(t) - mu) / sigma``.

        Returns
        -------
        ndarray
            CDF values.
        """
        if self.aft_loss_distribution == 'normal':
            return stats.norm.cdf(z)
        elif self.aft_loss_distribution == 'logistic':
            return stats.logistic.cdf(z)
        elif self.aft_loss_distribution == 'extreme':
            return 1.0 - np.exp(-np.exp(z))
        else:
            raise ValueError(
                f"Unknown AFT distribution: {self.aft_loss_distribution!r}. "
                f"Expected one of {_VALID_DISTRIBUTIONS}."
            )

    def _compute_survival_probs(self, X, times):
        """Compute parametric survival probabilities.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        times : ndarray
            Time points at which to evaluate.

        Returns
        -------
        probs : ndarray of shape (n_samples, n_times)
            Survival probabilities.
        """
        xgb = import_xgboost("XGBAFTModel")

        # mu = log(predicted_survival_time) for each sample
        # output_margin=True returns the raw margin in log-time space.
        mu = self.booster_.predict(xgb.DMatrix(X), output_margin=True)
        sigma = self.sigma_

        times = np.asarray(times, dtype=np.float64)

        # S(t | x) = 1 - F((log(t) - mu) / sigma)
        # Handle t <= 0: S(0) = 1.0 (fixed below after CDF computation)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_t = np.log(times)  # -inf for t=0, nan for t<0

        # z shape: (n_samples, n_times)
        z = (log_t[np.newaxis, :] - mu[:, np.newaxis]) / sigma

        surv = 1.0 - self._distribution_cdf(z)

        # Fix edge cases: t <= 0 → S = 1.0
        surv[:, times <= 0] = 1.0
        # Clip to [0, 1] for numerical safety
        np.clip(surv, 0.0, 1.0, out=surv)

        return surv

    def _default_time_grid(self, n_points=100):
        """Generate a default time grid from training times."""
        return np.linspace(
            max(self.train_times_.min(), 1e-6),
            self.train_times_.max(),
            n_points,
        )

    def predict_survival_function(self, X, times=None, return_array=False):
        """
        Predict survival function for each sample.

        Uses the parametric AFT survival function:
        ``S(t | x) = 1 - F((log(t) - mu) / sigma)``
        where F is the CDF of the configured distribution.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        times : array-like or None, default=None
            Time points at which to evaluate. If provided, returns ndarray
            of shape (n_samples, n_times). If None, uses a default grid
            from training times.
        return_array : bool, default=False
            If True and ``times`` is None, return ndarray instead of
            DataFrame.

        Returns
        -------
        pd.DataFrame or ndarray
            Survival probabilities. DataFrame has times as index and
            samples as columns. When ``times`` is provided, always
            returns ndarray (n_samples, n_times).
        """
        self._check_is_fitted()

        if times is not None:
            return self._compute_survival_probs(X, np.asarray(times))

        t_grid = self._default_time_grid()
        probs = self._compute_survival_probs(X, t_grid)

        if return_array:
            self._warn_if_large_array(probs)
            return probs

        # Return DataFrame (times × samples), matching WeibullAFT convention
        return pd.DataFrame(probs.T, index=t_grid)

    def predict_cumulative_hazard_function(self, X, times=None, return_array=False):
        """
        Predict cumulative hazard function for each sample.

        Computed as ``H(t) = -log(S(t))``.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        times : array-like or None, default=None
            Time points at which to evaluate. If provided, returns ndarray
            of shape (n_samples, n_times).
        return_array : bool, default=False
            If True and ``times`` is None, return ndarray instead of
            DataFrame.

        Returns
        -------
        pd.DataFrame or ndarray
            Cumulative hazard values.
        """
        self._check_is_fitted()

        if times is not None:
            surv = self._compute_survival_probs(X, np.asarray(times))
            # H(t) = -log(S(t)), guard against log(0)
            return -np.log(np.clip(surv, 1e-15, 1.0))

        t_grid = self._default_time_grid()
        surv = self._compute_survival_probs(X, t_grid)
        chf = -np.log(np.clip(surv, 1e-15, 1.0))

        if return_array:
            self._warn_if_large_array(chf)
            return chf

        return pd.DataFrame(chf.T, index=t_grid)
