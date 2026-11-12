"""DeepSurv neural network model wrapper."""

import tempfile
import uuid
import warnings
from pathlib import Path

import numpy as np
from sksurv.metrics import concordance_index_censored

from mlsurv.models.base import BaseSurvivalModel, PycoxSerializationMixin, SurvivalArray
from mlsurv._warnings import SuggestionWarning


class StratifiedSurvivalSampler:
    """
    Stratified batch sampler for survival data.

    Ensures each batch contains both events and censored observations,
    preventing NaN loss from batches with zero events.

    Parameters
    ----------
    events : array-like
        Binary event indicators (1=event, 0=censored).
    batch_size : int
        Desired batch size.
    min_events_per_batch : int, default=5
        Minimum number of events to include in each batch.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(self, events, batch_size, min_events_per_batch=5, random_state=None):
        self.events = np.asarray(events)
        self.batch_size = batch_size
        self.min_events_per_batch = min_events_per_batch
        self.rng = np.random.RandomState(random_state)

        # Separate indices by event status
        self.event_idx = np.where(self.events == 1)[0]
        self.censored_idx = np.where(self.events == 0)[0]

        self.n_events = len(self.event_idx)
        self.n_censored = len(self.censored_idx)
        self.n_samples = len(self.events)

        # Validate: must have at least min_events_per_batch events
        if self.n_events < self.min_events_per_batch:
            raise ValueError(
                f"Dataset has {self.n_events} events but requires at least "
                f"{self.min_events_per_batch} events for stratified batching. "
                f"Consider reducing min_events_per_batch or using a dataset with more events."
            )

        # Calculate events per batch to maintain proportions while ensuring minimum
        event_rate = self.n_events / self.n_samples
        self.events_per_batch = max(
            self.min_events_per_batch,
            int(self.batch_size * event_rate)
        )
        self.censored_per_batch = max(0, self.batch_size - self.events_per_batch)

        # Ensure we don't request more than available
        self.events_per_batch = min(self.events_per_batch, self.n_events)
        self.censored_per_batch = min(self.censored_per_batch, self.n_censored)

    def __iter__(self):
        # Shuffle indices
        event_idx_shuffled = self.rng.permutation(self.event_idx)
        censored_idx_shuffled = self.rng.permutation(self.censored_idx)

        # Calculate number of complete batches
        n_event_batches = self.n_events // self.events_per_batch
        n_censored_batches = self.n_censored // max(1, self.censored_per_batch)
        n_batches = min(n_event_batches, n_censored_batches)
        n_batches = max(1, n_batches)  # At least one batch

        for i in range(n_batches):
            # Get event indices for this batch
            e_start = i * self.events_per_batch
            e_end = e_start + self.events_per_batch
            batch_events = event_idx_shuffled[e_start:e_end]

            # Get censored indices for this batch
            c_start = i * self.censored_per_batch
            c_end = c_start + self.censored_per_batch
            batch_censored = censored_idx_shuffled[c_start:c_end]

            # Combine and shuffle within batch
            batch_idx = np.concatenate([batch_events, batch_censored])
            self.rng.shuffle(batch_idx)

            yield batch_idx.tolist()

        # Handle remaining samples in a final batch (if any)
        remaining_events = event_idx_shuffled[n_batches * self.events_per_batch:]
        remaining_censored = censored_idx_shuffled[n_batches * self.censored_per_batch:]

        if len(remaining_events) > 0 or len(remaining_censored) > 0:
            # Only yield if we have at least some events
            if len(remaining_events) >= self.min_events_per_batch:
                remaining = np.concatenate([remaining_events, remaining_censored])
                self.rng.shuffle(remaining)
                yield remaining.tolist()

    def __len__(self):
        n_event_batches = self.n_events // self.events_per_batch
        n_censored_batches = self.n_censored // max(1, self.censored_per_batch)
        n_batches = min(n_event_batches, n_censored_batches)
        n_batches = max(1, n_batches)

        # Add 1 for remainder batch if applicable
        remaining_events = self.n_events - n_batches * self.events_per_batch
        if remaining_events >= self.min_events_per_batch:
            n_batches += 1

        return n_batches


class _SurvivalDataset:
    """Dataset for survival data that supports indexing by list of indices.

    Uses duck typing (``__len__`` + ``__getitem__``) instead of inheriting
    ``torch.utils.data.Dataset`` so the class is picklable without torch
    being importable at module load time.
    """

    def __init__(self, X, times, events, device=None):
        import torch

        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.times = torch.tensor(times, dtype=torch.float32, device=device)
        self.events = torch.tensor(events, dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], (self.times[idx], self.events[idx])


def _collate_survival(batch):
    """Custom collate function for survival data tuple structure."""
    import torch

    X_batch = torch.stack([b[0] for b in batch])
    times_batch = torch.stack([b[1][0] for b in batch])
    events_batch = torch.stack([b[1][1] for b in batch])
    return X_batch, (times_batch, events_batch)


class DeepSurvModel(PycoxSerializationMixin, BaseSurvivalModel):
    """
    DeepSurv neural network model wrapper.

    Wraps pycox's CoxPH neural network with sklearn-compatible interface.
    Uses a deep neural network to model the log partial hazard.

    This is a lazy wrapper that requires torch and pycox to be installed.
    Install with: pip install mlsurv[deep]

    Parameters
    ----------
    layers : list, default=[8]
        Hidden layer sizes. Simpler architectures (1 layer, 8-32 nodes) often
        outperform deeper networks on small survival datasets.
    dropout : float, default=0.1
        Dropout rate for regularization.
    batch_norm : bool, default=True
        Whether to use batch normalization.
    activation : str, default='selu'
        Activation function: 'relu' or 'selu'. The original DeepSurv paper
        (Katzman et al., 2018) found SELU optimal for most datasets.
    learning_rate : float, default=0.1
        Adam optimizer learning rate. Higher learning rates (0.1-0.15) with
        fewer epochs often work better than low LR with many epochs.
    weight_decay : float, default=0.01
        L2 regularization (weight decay) for Adam optimizer.
    gradient_clip : float or None, default=1.0
        Maximum gradient norm for gradient clipping. Helps stabilize training
        with high learning rates. Set to None to disable.
    batch_size : int, default=256
        Training batch size. Avoid very small batch sizes (<32) which can
        cause training instability.
    epochs : int, default=100
        Maximum training epochs.
    patience : int or None, default=10
        Early stopping patience. If None, early stopping is disabled.
    val_fraction : float, default=0.2
        Fraction of training data to hold out for early stopping validation.
        Only used when patience is not None. Must be in (0.0, 1.0).
    random_state : int or None, default=None
        Random seed.
    device : str, default='auto'
        Device for computation: 'auto' (GPU if available, else CPU),
        'cpu', 'cuda', 'cuda:0', etc.

    Attributes
    ----------
    model_ : pycox.models.CoxPH
        The fitted pycox model.
    net_ : torch.nn.Module
        The underlying neural network.
    is_fitted_ : bool
        Whether the model has been fitted.

    Notes
    -----
    - Requires torch and pycox: pip install mlsurv[deep]
    - Data is automatically converted to float32 for PyTorch
    - Early stopping is enabled by default (patience=10) with a 20% validation split
    - predict() returns log partial hazard (risk scores)
    - predict_survival_function() uses baseline hazards computed during fit()
    - For small datasets (<1000 samples), simpler architectures work better.
      The original DeepSurv paper found 1 layer with 8 nodes optimal for GBSG.

    Examples
    --------
    >>> from mlsurv.models import DeepSurvModel
    >>> model = DeepSurvModel(layers=[32], dropout=0.1)
    >>> model.fit(X_train, y_train)
    >>> risk_scores = model.predict(X_test)

    References
    ----------
    - Katzman, J.L., et al. (2018). DeepSurv: personalized treatment
      recommender system using a Cox proportional hazards deep neural network.
    - pycox: https://github.com/havakv/pycox
    """

    name = "deepsurv"
    display_name = "DeepSurv"
    color = "#56B4E9"
    supports_random_state = True
    prefer_sequential_cv = True
    complexity = "deep_learning"
    # Optimizes Cox partial likelihood (pycox.models.CoxPH) — same loss as
    # XGBoost-Cox. The PH assumption is at the loss level; the network can
    # partially absorb non-PH effects via non-linear capacity, but the
    # recommend_models() tier logic flags it like other Cox-loss models.
    assumes_ph = True
    score_based = True  # Cox-loss net → single linear predictor → Breslow baseline

    def __init__(
        self,
        layers=None,
        dropout=0.1,
        batch_norm=True,
        activation='selu',
        learning_rate=0.1,
        weight_decay=0.01,
        gradient_clip=1.0,
        batch_size=256,
        epochs=100,
        patience=10,
        val_fraction=0.2,
        random_state=None,
        device='auto',
    ):
        self.layers = layers if layers is not None else [8]
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.device = device
        self.model_ = None
        self.net_ = None
        self.actual_epochs_ = None
        self.converged_ = None

    def _check_dependencies(self):
        """Import torch, torchtuples, and pycox, or raise a clear ImportError."""
        from mlsurv.models._torch import import_deep_learning
        import_deep_learning("DeepSurvModel")

    def _should_use_early_stopping(self, y: SurvivalArray) -> bool:
        """Check whether the dataset supports early stopping.

        Returns True if patience is set and the dataset is large enough
        for a stratified train/val split. Raises ValueError for invalid
        val_fraction. Emits a SuggestionWarning and returns False when the
        dataset is too small.
        """
        if self.patience is None:
            return False

        viable, info = self._validate_early_stopping_viability(
            y, self.val_fraction, min_train_events=5,
        )
        if viable:
            return True

        warnings.warn(
            f"Dataset too small (n={info['n']}, events={info['n_events']}) for "
            f"early stopping with val_fraction={self.val_fraction}. "
            f"Disabling early stopping.",
            SuggestionWarning,
            stacklevel=3,
        )
        return False

    def fit(self, X, y: SurvivalArray):
        """
        Fit the DeepSurv neural network.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Training features.
        y : SurvivalArray
            Survival outcomes with 'event' and 'time' fields.

        Returns
        -------
        self : DeepSurvModel
            Fitted model.

        Raises
        ------
        ImportError
            If torch, torchtuples, or pycox is missing, or is installed but
            fails to load.
        ValueError
            If the configured activation is not a supported option, or if
            an unrelated stratification failure surfaces during the
            validation split.
        """
        self._check_dependencies()
        self._validate_data(X, y)
        self._validate_pycox_params()

        import torch
        import torchtuples as tt
        from pycox.models import CoxPH

        # Resolve device
        device = self._resolve_device(self.device)

        # Set random seeds for reproducibility
        self._set_torch_seed(self.random_state, device)

        # Determine whether early stopping is viable
        use_early_stopping = self._should_use_early_stopping(y)

        # Convert X to numpy array once
        X_np = np.asarray(X, dtype=np.float32)

        # Split off validation data for early stopping
        if use_early_stopping:
            from sklearn.model_selection import train_test_split
            try:
                idx_train, idx_val = train_test_split(
                    np.arange(len(y)),
                    test_size=self.val_fraction,
                    stratify=y['event'],
                    random_state=self.random_state,
                )
            except ValueError as exc:
                # sklearn raises ValueError with "least populated class"
                # when stratification is impossible. Match on message
                # string because sklearn has no typed exception for this.
                if "least populated class" not in str(exc):
                    raise
                warnings.warn(
                    f"Stratified val split failed (n={len(y)}, "
                    f"events={int(y['event'].sum())}). "
                    f"Disabling early stopping.",
                    SuggestionWarning,
                    stacklevel=2,
                )
                use_early_stopping = False

        if use_early_stopping:
            X_train = X_np[idx_train]
            y_train_split = y[idx_train]
            X_val = X_np[idx_val]
            y_val_split = y[idx_val]
        else:
            X_train = X_np
            y_train_split = y

        # Extract times and events from the training split
        times = y_train_split['time'].astype(np.float32)
        events = y_train_split['event'].astype(np.float32)

        # Build network
        in_features = X_train.shape[1]
        out_features = 1

        # Get activation function
        if self.activation == 'relu':
            activation_fn = torch.nn.ReLU
        elif self.activation == 'selu':
            activation_fn = torch.nn.SELU
        else:
            raise ValueError(f"Unknown activation: {self.activation}. Use 'relu' or 'selu'.")

        # Build network using torchtuples
        self.net_ = tt.practical.MLPVanilla(
            in_features,
            self.layers,
            out_features,
            self.batch_norm,
            self.dropout,
            activation=activation_fn
        )
        self.net_.to(device)

        # Create CoxPH model with Adam optimizer (including weight decay for L2 reg)
        # Pass device so pycox's internal device tracking matches our tensors
        optimizer = tt.optim.Adam(lr=self.learning_rate, weight_decay=self.weight_decay)
        self.model_ = CoxPH(self.net_, optimizer, device=device)

        # Build callbacks list
        callbacks = []

        # Apply gradient clipping if specified (helps with high learning rates)
        if self.gradient_clip is not None:
            callbacks.append(
                tt.callbacks.ClipGradNorm(self.net_, max_norm=self.gradient_clip)
            )

        # Early stopping callback (only if early stopping is viable)
        ckpt_file = None
        if use_early_stopping:
            ckpt_path = Path(tempfile.gettempdir()) / "mlsurv_checkpoints"
            ckpt_path.mkdir(exist_ok=True)
            ckpt_file = ckpt_path / f"deepsurv_{uuid.uuid4().hex[:8]}.pt"
            callbacks.append(tt.callbacks.EarlyStopping(
                patience=self.patience,
                file_path=str(ckpt_file),
            ))

        # Create stratified batch sampler to ensure events in each batch
        # This prevents NaN loss from batches with zero events
        batch_sampler = StratifiedSurvivalSampler(
            events=events,
            batch_size=self.batch_size,
            min_events_per_batch=5,
            random_state=self.random_state
        )

        dataset = _SurvivalDataset(X_train, times, events, device=device)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=_collate_survival,
        )

        # Build val dataloader if applicable.
        # pycox's fit_dataloader() only accepts val_dataloader= (not val_data=);
        # passing val_data= is silently ignored, making early stopping ineffective.
        val_dataloader = None
        if use_early_stopping:
            val_dataset = _SurvivalDataset(
                X_val,
                y_val_split['time'].astype(np.float32),
                y_val_split['event'].astype(np.float32),
                device=device,
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=len(val_dataset),
                shuffle=False,
                collate_fn=_collate_survival,
            )

        # Fit model using dataloader
        try:
            self.model_.fit_dataloader(
                dataloader,
                self.epochs,
                callbacks=callbacks,
                verbose=False,
                val_dataloader=val_dataloader,
            )

            # Compute baseline hazards on training split only (not full data)
            # to avoid val leakage. Explicit data needed because fit_dataloader()
            # does not store references to training data.
            _ = self.model_.compute_baseline_hazards(X_train, (times, events))
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

    def predict(self, X):
        """
        Predict risk scores (log partial hazard).

        Higher scores indicate higher risk (shorter survival).

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        risk_scores : ndarray
            Predicted log partial hazard values.
        """
        import torch
        self._check_is_fitted()
        X_np = np.asarray(X, dtype=np.float32)
        # Pass numpy array — pycox handles conversion to device tensor internally
        with torch.no_grad():
            return self.model_.predict(X_np).ravel()

    def predict_survival_function(self, X, times=None, return_array=False):
        """
        Predict survival function for each sample.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        times : array-like, optional
            Time points at which to evaluate. If None, uses training times.
        return_array : bool, default=False
            If True, return ndarray of shape (n_samples, n_times) instead of
            DataFrame.

        Returns
        -------
        pd.DataFrame or ndarray
            Survival probabilities with time points as index (DataFrame) or
            shape (n_samples, n_times) (ndarray).
        """
        import torch
        self._check_is_fitted()
        X_np = np.asarray(X, dtype=np.float32)

        # Get survival function DataFrame
        with torch.no_grad():
            surv_df = self.model_.predict_surv_df(X_np)

        if times is not None:
            combined_index = surv_df.index.union(times).sort_values()
            first_time = surv_df.index.min()
            surv_df = surv_df.reindex(combined_index)
            surv_df = surv_df.ffill()
            surv_df.loc[surv_df.index < first_time] = 1.0
            surv_df = surv_df.bfill()
            surv_df = surv_df.loc[times]

        if return_array:
            result = surv_df.T.values
            self._warn_if_large_array(result)
            return result

        return surv_df

    def predict_cumulative_hazard_function(self, X, times=None, return_array=False):
        """
        Predict cumulative hazard function for each sample.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        times : array-like, optional
            Time points at which to evaluate.
        return_array : bool, default=False
            If True, return ndarray of shape (n_samples, n_times).

        Returns
        -------
        pd.DataFrame or ndarray
            Cumulative hazard values with time points as index.
        """
        self._check_is_fitted()
        surv_df = self.predict_survival_function(X, times)
        chf = -np.log(surv_df.clip(lower=1e-10))
        if return_array:
            result = chf.T.values
            self._warn_if_large_array(result)
            return result
        return chf

    def score(self, X, y: SurvivalArray):
        """
        Return the concordance index on the given test data.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : SurvivalArray
            Survival outcomes.

        Returns
        -------
        float
            Concordance index.
        """
        risk_scores = self.predict(X)
        return concordance_index_censored(
            y['event'].astype(bool),
            y['time'],
            risk_scores
        )[0]

    # Serialization is handled by PycoxSerializationMixin
