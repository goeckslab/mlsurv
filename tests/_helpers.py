"""Shared survival-data factories for the mlsurv test suite.

Consolidates the ~20 per-file ``_make_survival_array`` / ``_make_y`` helpers that
had silently diverged in argument order, event distribution, time scale, and
return type. All factories return a :data:`~mlsurv.utils.SurvivalArray` (a
``NewType`` alias for the structured ``np.ndarray`` with dtype
``[('event', bool), ('time', float)]``).

Two explicit-array factories exist deliberately — :func:`make_y` (times-first,
matching :func:`mlsurv.utils.to_structured_array` and the ``_make_y`` cluster)
and :func:`make_survival_array` (events-first, matching the
``_make_survival_array(events, times)`` cluster). They are NOT a single
overloaded function (ambiguous overloading is forbidden); the events-first
form is a thin wrapper so both share one validation path.

:func:`make_random_survival_array` reproduces the legacy ``RandomState``-based
generators byte-for-byte: it draws events first, then times, and exposes
``event_dist`` (``"binomial"`` vs ``"uniform"``) and ``time_scale`` to cover the
binomial+exp(5) and uniform+exp(10)/exp(5) variants, plus an injectable ``rng``
for call sites that interleave further draws.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from mlsurv.utils import SurvivalArray, to_structured_array

ArrayLike = Union[list, np.ndarray, pd.Series]


def make_y(times: ArrayLike, events: Optional[ArrayLike] = None) -> SurvivalArray:
    """Build a ``SurvivalArray`` from times (events default to all-True).

    Times-first to match :func:`mlsurv.utils.to_structured_array` and the
    ``_make_y`` cluster. Validation (positive times, no NaN/inf, length match)
    is delegated to ``to_structured_array`` — non-positive or NaN/inf times
    raise ``ValueError`` by construction. To build a deliberately degenerate
    fixture (e.g. a zero time), build a valid array first and mutate it
    afterwards.

    Parameters
    ----------
    times : array-like
        Survival times (must be strictly positive).
    events : array-like, optional
        Event indicators. Defaults to all events observed (all ``True``).
    """
    times = np.asarray(times, dtype=float)
    if events is None:
        events = np.ones(len(times), dtype=bool)
    return to_structured_array(times, events)


def make_survival_array(events: ArrayLike, times: ArrayLike) -> SurvivalArray:
    """Build a ``SurvivalArray`` from events-first arrays.

    Events-first wrapper over :func:`make_y` for the
    ``_make_survival_array(events, times)`` cluster.
    """
    return make_y(times, events)


def make_random_survival_array(
    n: int,
    *,
    event_rate: float = 0.7,
    time_scale: float = 5.0,
    event_dist: str = "binomial",
    seed: int = 42,
    rng: Optional[np.random.RandomState] = None,
) -> SurvivalArray:
    """Generate a random ``SurvivalArray``, reproducing the legacy generators.

    Draws events first, then times (matching the historical helpers byte-for-byte
    for a given seed). Pass an external ``rng`` for call sites that interleave
    further draws — its state advances by exactly one event draw and one time
    draw, leaving subsequent ``rng`` calls unchanged.

    Parameters
    ----------
    n : int
        Number of observations.
    event_rate : float, default 0.7
        Probability an observation is an event (uncensored).
    time_scale : float, default 5.0
        Scale of the exponential time distribution.
    event_dist : {"binomial", "uniform"}, default "binomial"
        ``"binomial"`` → ``rng.binomial(1, event_rate, n)``;
        ``"uniform"``  → ``rng.random(n) < event_rate`` (identical stream to
        the legacy ``rng.rand(n) < event_rate``).
    seed : int, default 42
        Seed for the internal ``RandomState`` when ``rng`` is not supplied.
    rng : numpy.random.RandomState, optional
        Externally supplied legacy ``RandomState``. When given, ``seed`` is
        ignored.
    """
    if rng is None:
        rng = np.random.RandomState(seed)
    if event_dist == "binomial":
        events = rng.binomial(1, event_rate, n).astype(bool)
    elif event_dist == "uniform":
        events = rng.random(n) < event_rate
    else:
        raise ValueError(
            f"event_dist must be 'binomial' or 'uniform', got {event_dist!r}."
        )
    times = rng.exponential(time_scale, n)
    return SurvivalArray(
        np.array(
            list(zip(events, times)),
            dtype=[("event", bool), ("time", float)],
        )
    )


def make_feature_dataframe(
    n: int,
    p: int = 5,
    *,
    seed: int = 42,
    prefix: str = "f",
) -> pd.DataFrame:
    """Build a random Gaussian feature matrix.

    Column names are ``f"{prefix}{i}"``. The default ``prefix="f"`` matches the
    most common local ``_make_X`` helpers; pass ``prefix="feat_"`` to reproduce
    files that used that naming.
    """
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        rng.randn(n, p), columns=[f"{prefix}{i}" for i in range(p)]
    )


def make_synthetic_data(
    n: int = 200,
    p: int = 5,
    *,
    event_rate: float = 0.7,
    time_scale: float = 5.0,
    event_dist: str = "binomial",
    seed: int = 42,
    prefix: str = "f",
) -> tuple[pd.DataFrame, SurvivalArray]:
    """Build an ``(X, y)`` pair for tests that need both together.

    Convenience over :func:`make_feature_dataframe` + :func:`make_random_survival_array`
    for the common ``X``/``y`` idiom; ``X`` and ``y`` each construct a separate
    ``RandomState`` from the same ``seed`` (so the feature stream and the
    event/time stream both start from identical generator state).
    """
    X = make_feature_dataframe(n, p, seed=seed, prefix=prefix)
    y = make_random_survival_array(
        n,
        event_rate=event_rate,
        time_scale=time_scale,
        event_dist=event_dist,
        seed=seed,
    )
    return X, y
