"""DeepHit neural network model wrapper."""

import tempfile
import uuid
import warnings
from pathlib import Path

import numpy as np

from mlsurv.models.base import BaseSurvivalModel, PycoxSerializationMixin, SurvivalArray
from mlsurv.utils import trapz
from mlsurv._warnings import SuggestionWarning


class DeepHitModel(PycoxSerializationMixin, BaseSurvivalModel):
    """
    DeepHit neural network model wrapper.

    Wraps pycox's DeepHitSingle for discrete-time survival analysis
    (single event type). Uses a deep neural network to model probability
    mass functions over discretized time intervals.

    This is a lazy wrapper that requires torch and pycox to be installed.
    Install with: pip install mlsurv[deep]

    Parameters
    ----------
    layers : list or None, default=None
        Hidden layer sizes. If None, defaults to [64]. Simpler architectures
        often work better on small survival datasets.
    dropout : float, default=0.1
        Dropout rate for regularization.
    activation : str, default='selu'
        Activation function: 'relu' or 'selu'. SELU often outperforms ReLU
        on survival datasets (Katzman et al., 2018).
    alpha : float, default=0.0
        Balance between log-likelihood and ranking loss in pycox's
        ``DeepHitSingleLoss``: ``loss = alpha * NLL + (1 - alpha) * rank_loss``.
        ``alpha=0.0`` is pure ranking (default; best Antolini's C^td); ``alpha=1.0``
        is pure NLL (best calibration in isolation, but worse C^td). DeepHit
        is treated as a rank-based model (``rank_based = True``); raw
        ``S(t|x)`` is miscalibrated at low ``alpha``. Apply post-hoc isotonic
        calibration with ``train('deephit', calibrate='isotonic')`` to fix the
        probability scale without sacrificing C^td (per-horizon isotonic is
        monotone within horizons, so ranking is preserved).
    sigma : float, default=0.1
        Smoothing parameter for the ranking loss term. Active at the default
        ``alpha=0.0``; has no effect when ``alpha=1.0`` (ranking term is
        zeroed out in ``alpha * NLL + (1 - alpha) * rank_loss``).
    learning_rate : float, default=0.005
        Adam optimizer learning rate.
    weight_decay : float, default=0.01
        L2 regularization (weight decay) for Adam optimizer.
    gradient_clip : float or None, default=1.0
        Maximum gradient norm for gradient clipping. Helps stabilize training
        with high learning rates. Set to None to disable.
    batch_size : int, default=256
        Training batch size.
    epochs : int, default=100
        Maximum training epochs.
    num_durations : int, default=100
        Number of discrete time intervals for discretization.
    patience : int or None, default=10
        Early stopping patience. If None, early stopping is disabled.
    val_fraction : float, default=0.2
        Fraction of training data to hold out for early stopping validation.
        Only used when patience is not None. Must be in (0.0, 1.0).
    random_state : int or None, default=None
        Random seed for reproducibility.
    device : str, default='auto'
        Device for computation: 'auto' (GPU if available, else CPU),
        'cpu', 'cuda', 'cuda:0', etc.

    Attributes
    ----------
    model_ : pycox.models.DeepHitSingle
        The fitted pycox model.
    net_ : torch.nn.Module
        The underlying neural network.
    labtrans_ : pycox.preprocessing.LabTransDiscreteTime
        Label transformer for time discretization.
    is_fitted_ : bool
        Whether the model has been fitted.

    Notes
    -----
    - Requires torch and pycox: pip install mlsurv[deep]
    - Time is automatically discretized into num_durations intervals
    - predict() returns risk scores (negative expected survival time)
    - predict_survival_function() returns survival probabilities at discrete times
    - alpha controls likelihood vs. ranking loss balance (not competing risks)
    - **Rank-based model** (``rank_based = True``): the default fit optimizes
      C^td via the ranking loss; raw S(t|x) is miscalibrated. Call
      ``train('deephit', calibrate='isotonic')`` to fit per-horizon isotonic
      calibrators on out-of-fold predictions; downstream ``predict()``,
      ``patient_report()``, and ``validate(use_recalibrated=True)`` will use
      the calibrated S(t|x) while raw evaluate() metrics are preserved.

    Examples
    --------
    >>> from mlsurv.models import DeepHitModel
    >>> model = DeepHitModel(layers=[32], dropout=0.1)
    >>> model.fit(X_train, y_train)
    >>> risk_scores = model.predict(X_test)

    References
    ----------
    - Lee, C., et al. (2018). DeepHit: A deep learning approach to survival
      analysis with competing risks.
    - pycox: https://github.com/havakv/pycox
    """

    name = "deephit"
    display_name = "DeepHit"
    color = "#F0E442"
    supports_random_state = True
    prefer_sequential_cv = True
    complexity = "deep_learning"
    # Rank-based: at the default alpha=0.0 the loss is pure ranking, so raw
    # S(t|x) is well-ordered but miscalibrated. Pair with
    # train('deephit', calibrate='isotonic') for valid probabilities.
    rank_based = True

    def __init__(
        self,
        layers=None,
        dropout=0.1,
        activation='selu',
        alpha=0.0,
        sigma=0.1,
        learning_rate=0.005,
        weight_decay=0.01,
        gradient_clip=1.0,
        batch_size=256,
        epochs=100,
        num_durations=100,
        patience=10,
        val_fraction=0.2,
        random_state=None,
        device='auto',
    ):
        self.layers = layers if layers is not None else [64]
        self.dropout = dropout
        self.activation = activation
        self.alpha = alpha
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_durations = num_durations
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.device = device
        self.model_ = None
        self.net_ = None
        self.labtrans_ = None
        self.actual_epochs_ = None
        self.converged_ = None

    def fit(self, X, y: SurvivalArray):
        """
        Fit the DeepHit model.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Training features.
        y : SurvivalArray
            Survival outcomes with dtype=[('event', bool), ('time', float)].

        Returns
        -------
        self : DeepHitModel
            Fitted model.

        Raises
        ------
        ImportError
            If PyTorch and pycox are not installed.
        ValueError
            If the pycox hyperparameters are invalid or the configured
            activation function is unsupported.
        """
        try:
            import torch
            import torchtuples as tt
            from pycox.models import DeepHitSingle
            from pycox.preprocessing.label_transforms import LabTransDiscreteTime
        except ImportError as exc:
            raise ImportError(
                "PyTorch and pycox are required for DeepHitModel. "
                "Install with: pip install mlsurv[deep]"
            ) from exc

        self._validate_data(X, y)
        self._validate_pycox_params()

        # Resolve device
        device = self._resolve_device(self.device)

        # Set random seeds for reproducibility
        self._set_torch_seed(self.random_state, device)

        # Convert X to numpy array once (handles both DataFrame and array-like)
        X_np = np.asarray(X, dtype=np.float32)

        # Resolve early stopping and split data
        use_early_stopping, X_train, y_train_split, X_val, y_val_split = (
            self._resolve_early_stopping(X_np, y)
        )

        # Convert training features to tensor on target device
        X_tensor = torch.tensor(X_train, device=device)

        # Discretize time into intervals (fit on training data only)
        self.labtrans_ = LabTransDiscreteTime(cuts=self.num_durations)
        y_discrete = self.labtrans_.fit_transform(
            y_train_split['time'], y_train_split['event']
        )
        y_discrete_tensor = (
            torch.tensor(y_discrete[0], device=device),
            torch.tensor(y_discrete[1], device=device),
        )

        # Transform validation data using the already-fit label transformer
        val_data = None
        if use_early_stopping:
            y_val_discrete = self.labtrans_.transform(
                y_val_split['time'], y_val_split['event']
            )
            val_data = (
                torch.tensor(X_val, device=device),
                (torch.tensor(y_val_discrete[0], device=device),
                 torch.tensor(y_val_discrete[1], device=device)),
            )

        # Build neural network
        in_features = X_tensor.shape[1]
        out_features = self.labtrans_.out_features

        # Resolve activation function
        if self.activation == 'relu':
            activation_fn = torch.nn.ReLU
        elif self.activation == 'selu':
            activation_fn = torch.nn.SELU
        else:
            raise ValueError(f"Unknown activation: {self.activation}. Use 'relu' or 'selu'.")

        self.net_ = tt.practical.MLPVanilla(
            in_features=in_features,
            num_nodes=self.layers,
            out_features=out_features,
            batch_norm=True,
            dropout=self.dropout,
            activation=activation_fn,
        )
        self.net_.to(device)

        # Create model with weight decay for L2 regularization
        self.model_ = DeepHitSingle(
            self.net_,
            tt.optim.Adam(lr=self.learning_rate, weight_decay=self.weight_decay),
            device=device,
            alpha=self.alpha,
            sigma=self.sigma,
            duration_index=self.labtrans_.cuts
        )

        # Train model
        callbacks = []

        # Apply gradient clipping if specified (helps with high learning rates)
        if self.gradient_clip is not None:
            callbacks.append(
                tt.callbacks.ClipGradNorm(self.net_, max_norm=self.gradient_clip)
            )

        ckpt_file = None
        if use_early_stopping:
            ckpt_path = Path(tempfile.gettempdir()) / "mlsurv_checkpoints"
            ckpt_path.mkdir(exist_ok=True)
            ckpt_file = ckpt_path / f"deephit_{uuid.uuid4().hex[:8]}.pt"
            callbacks.append(tt.callbacks.EarlyStopping(
                patience=self.patience,
                file_path=str(ckpt_file),
            ))

        # Adjust batch_size to avoid a last mini-batch of size 1, which
        # crashes BatchNorm (requires >1 sample per batch for running stats).
        n_train = X_tensor.shape[0]
        batch_size = self.batch_size
        if n_train > batch_size and n_train % batch_size == 1:
            batch_size += 1

        try:
            self.model_.fit(
                X_tensor,
                y_discrete_tensor,
                batch_size=batch_size,
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=False,
                val_data=val_data,
            )
        finally:
            if ckpt_file is not None:
                try:
                    ckpt_file.unlink(missing_ok=True)
                except OSError:
                    pass  # Best-effort; don't mask training errors

        self.actual_epochs_ = getattr(self.model_.log, 'epoch', self.epochs) if hasattr(self.model_, 'log') else self.epochs
        self.converged_ = self.actual_epochs_ < self.epochs
        self.is_fitted_ = True
        return self

    def _resolve_early_stopping(self, X_np, y):
        """Decide whether early stopping is viable and split data if so.

        Returns (use_early_stopping, X_train, y_train, X_val, y_val).
        X_val and y_val are None when early stopping is disabled.
        """
        if self.patience is None:
            return False, X_np, y, None, None

        viable, info = self._validate_early_stopping_viability(
            y, self.val_fraction,
        )
        if not viable:
            warnings.warn(
                f"Dataset too small (n={info['n']}, events={info['n_events']}) "
                f"for early stopping with val_fraction="
                f"{self.val_fraction}. Disabling early stopping.",
                SuggestionWarning,
                stacklevel=3,
            )
            return False, X_np, y, None, None

        from sklearn.model_selection import train_test_split
        try:
            idx_train, idx_val = train_test_split(
                np.arange(len(y)),
                test_size=self.val_fraction,
                stratify=y['event'],
                random_state=self.random_state,
            )
        except ValueError as exc:
            if "least populated class" not in str(exc):
                raise
            warnings.warn(
                f"Stratified val split failed (n={info['n']}, "
                f"events={info['n_events']}). "
                f"Disabling early stopping.",
                SuggestionWarning,
                stacklevel=3,
            )
            return False, X_np, y, None, None

        return (
            True,
            X_np[idx_train], y[idx_train],
            X_np[idx_val], y[idx_val],
        )

    def predict(self, X):
        """
        Predict risk scores.

        Returns negative expected survival time (integral of S(t)) as risk
        score (higher = higher risk). Unlike median survival time which is
        discretized to ``num_durations`` bins, expected survival time is
        continuous, producing fewer tied risk scores and better concordance.

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        risk_scores : ndarray
            Predicted risk scores.

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required. Install with: pip install torch") from exc

        self._check_is_fitted()

        # Pass numpy array — pycox handles device placement internally
        X_np = np.asarray(X, dtype=np.float32)

        # Get survival predictions (no gradients needed for inference)
        with torch.no_grad():
            surv = self.model_.predict_surv_df(X_np)

        # Compute expected survival time = integral of S(t) over discrete times
        # This yields continuous values (unlike median, which is discretized)
        times = surv.index.values
        surv_values = surv.values  # shape (n_times, n_samples)
        expected_time = trapz(surv_values, x=times, axis=0)

        # Return negative expected time as risk score (higher = higher risk)
        return -expected_time

    def predict_survival_function(self, X, return_array=False):
        """
        Predict survival function for each sample.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        return_array : bool, default=False
            If True, return array instead of DataFrame.

        Returns
        -------
        pd.DataFrame or ndarray
            Survival probabilities at discrete time points.
            If DataFrame: rows=times, columns=samples
            If array: rows=samples, columns=times

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required. Install with: pip install torch") from exc

        self._check_is_fitted()

        # Pass numpy array — pycox handles device placement internally
        X_np = np.asarray(X, dtype=np.float32)

        # Get survival predictions (no gradients needed for inference)
        with torch.no_grad():
            surv_df = self.model_.predict_surv_df(X_np)

        if return_array:
            result = surv_df.T.values
            self._warn_if_large_array(result)
            return result

        return surv_df

    # Serialization is handled by PycoxSerializationMixin
