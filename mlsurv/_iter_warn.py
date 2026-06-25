"""Coalescer for repeated IterativeWarnings.

Per-iteration loops emit warnings whose text varies per iteration (e.g.,
``seed=...`` for failed bootstrap iterations, or ``label='X'`` for
small-subpopulation notices). Python's ``simplefilter("once", ...)`` cannot
deduplicate when message text differs, so stderr fills with hundreds of
near-identical lines.

Two modes are supported:

* ``mode="failures"`` (default): each record carries a caught
  exception. Examples are formatted as
  ``"{summary_label} iteration failed{prefix}: {message}"``; the aggregate
  summary describes count + error-type diversity.

* ``mode="notices"``: each record carries a pre-rendered message
  (no exception). Examples are emitted verbatim; the aggregate summary
  reads ``"{n}/{n_total} ({pct}%) {summary_label}s {notice_phrase}: {keys}.{tail}"``,
  where each ``key`` is the notice's ``display`` text (or ``group_key`` as
  a fallback). ``group_key`` is reserved as a true identifier for future
  grouping/dedup logic; ``display`` is the short summary-list text.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Type

from ._warnings import IterativeWarning


@dataclass(frozen=True)
class IterativeFailure:
    """A single per-iteration failure to be reported by the coalescer."""

    exc_type_name: str
    message: str
    context_prefix: str = ""

    @classmethod
    def from_exception(
        cls, exc: BaseException, *, prefix: str = ""
    ) -> "IterativeFailure":
        """Build a failure record from a caught exception."""
        return cls(
            exc_type_name=type(exc).__name__,
            message=str(exc),
            context_prefix=prefix,
        )


@dataclass(frozen=True)
class IterativeNotice:
    """A single per-iteration notice (non-failure) to be coalesced.

    Fields:

    * ``message`` — full per-iteration text, emitted verbatim when
      ``max_emit > 0``.
    * ``group_key`` — true identifier for the iteration (e.g. a label,
      seed, or bucket name). Reserved for future grouping/dedup logic
      inside the coalescer.
    * ``display`` — short text used in the aggregate summary's
      "affected items" list. Falls back to ``group_key`` when empty,
      so callers that only need an identifier can keep passing
      ``group_key=`` alone.
    """

    message: str
    group_key: str = ""
    display: str = ""


def coalesce_failures(
    failures: Iterable[IterativeFailure],
    *,
    n_total: int,
    summary_label: str,
    max_emit: int = 3,
    category: Type[Warning] = IterativeWarning,
    stacklevel: int = 3,
    tail_msg: str = "",
) -> int:
    """Emit warnings for ``failures`` according to the coalescer policy.

    Policy:
      1. Group by ``exc_type_name`` (encounter order).
      2. Emit up to ``max_emit`` distinct error-type examples verbatim.
      3. Emit one aggregate summary describing total count and type diversity.

    Empty ``failures`` emits nothing.

    Parameters
    ----------
    failures : iterable of IterativeFailure
        Per-iteration failure records to coalesce.
    n_total : int
        Total number of iterations attempted (denominator for the summary).
    summary_label : str
        Human-readable label for the iteration kind (e.g. ``"bootstrap"``).
    max_emit : int, default=3
        Maximum number of distinct error-type examples emitted verbatim
        before the aggregate summary.
    category : type of Warning, default=IterativeWarning
        Warning class to emit.
    stacklevel : int, default=3
        ``stacklevel`` forwarded to :func:`warnings.warn`.
    tail_msg : str, default=""
        Optional text appended to the aggregate summary message.

    Returns
    -------
    int
        Number of failures, for caller convenience.

    Raises
    ------
    ValueError
        If ``n_total`` is less than 1, if ``max_emit`` is negative, or if
        the number of failures exceeds ``n_total``.
    """
    if n_total < 1:
        raise ValueError(f"n_total must be >= 1; got {n_total}")
    if max_emit < 0:
        raise ValueError(f"max_emit must be >= 0; got {max_emit}")
    failures = list(failures)
    n_failed = len(failures)
    if n_failed == 0:
        return 0
    if n_failed > n_total:
        raise ValueError(
            f"len(failures)={n_failed} exceeds n_total={n_total}"
        )

    grouped: "OrderedDict[str, list[IterativeFailure]]" = OrderedDict()
    for f in failures:
        grouped.setdefault(f.exc_type_name, []).append(f)

    for exc_type_name, group in list(grouped.items())[:max_emit]:
        example = group[0]
        prefix = f" ({example.context_prefix})" if example.context_prefix else ""
        warnings.warn(
            f"{summary_label} iteration failed{prefix}: {example.message}",
            category,
            stacklevel=stacklevel,
        )

    pct = n_failed / n_total * 100
    tail = f" {tail_msg}" if tail_msg else ""
    if len(grouped) == 1:
        only_type = next(iter(grouped))
        summary = (
            f"{n_failed}/{n_total} ({pct:.0f}%) {summary_label} iterations failed; "
            f"all shared the same error type ({only_type}).{tail}"
        )
    else:
        breakdown = ", ".join(
            f"{name} (n={len(group)})" for name, group in grouped.items()
        )
        summary = (
            f"{n_failed}/{n_total} ({pct:.0f}%) {summary_label} iterations failed "
            f"across {len(grouped)} distinct error types: {breakdown}.{tail}"
        )
    warnings.warn(summary, category, stacklevel=stacklevel)
    return n_failed


def coalesce_notices(
    notices: Iterable[IterativeNotice],
    *,
    n_total: int,
    summary_label: str,
    notice_phrase: str,
    max_emit: int = 3,
    category: Type[Warning] = IterativeWarning,
    stacklevel: int = 3,
    tail_msg: str = "",
) -> int:
    """Emit warnings for ``notices`` (non-failure repeated warnings).

    Policy:
      1. Emit up to ``max_emit`` notice messages verbatim (encounter order).
      2. Emit one aggregate summary listing the affected ``group_key`` values.

    Empty ``notices`` emits nothing.

    Parameters
    ----------
    notices : iterable of IterativeNotice
        Per-iteration notice records to coalesce.
    n_total : int
        Total number of iterations attempted (denominator for the summary).
    summary_label : str
        Human-readable label for the iteration kind (e.g. ``"bootstrap"``).
    notice_phrase : str
        Phrase describing the notice condition, used in the aggregate summary.
    max_emit : int, default=3
        Maximum number of notice messages emitted verbatim before the
        aggregate summary.
    category : type of Warning, default=IterativeWarning
        Warning class to emit.
    stacklevel : int, default=3
        ``stacklevel`` forwarded to :func:`warnings.warn`.
    tail_msg : str, default=""
        Optional text appended to the aggregate summary message.

    Returns
    -------
    int
        Number of notices, for caller convenience.

    Raises
    ------
    ValueError
        If ``n_total`` is less than 1, if ``max_emit`` is negative, or if
        the number of notices exceeds ``n_total``.
    """
    if n_total < 1:
        raise ValueError(f"n_total must be >= 1; got {n_total}")
    if max_emit < 0:
        raise ValueError(f"max_emit must be >= 0; got {max_emit}")
    notices = list(notices)
    n = len(notices)
    if n == 0:
        return 0
    if n > n_total:
        raise ValueError(f"len(notices)={n} exceeds n_total={n_total}")

    for notice in notices[:max_emit]:
        warnings.warn(notice.message, category, stacklevel=stacklevel)

    pct = n / n_total * 100
    tail = f" {tail_msg}" if tail_msg else ""
    keys = ", ".join(
        (nt.display or nt.group_key)
        for nt in notices
        if (nt.display or nt.group_key)
    )
    keys_part = f": {keys}" if keys else ""
    summary = (
        f"{n}/{n_total} ({pct:.0f}%) {summary_label}s {notice_phrase}"
        f"{keys_part}.{tail}"
    )
    warnings.warn(summary, category, stacklevel=stacklevel)
    return n


class IterativeWarningCoalescer:
    """Context-manager form of the coalescer functions.

    Two modes:

    * ``mode="failures"`` (default): call ``.record(exc, prefix=...)`` for
      each caught exception. Dispatches to :func:`coalesce_failures`.
    * ``mode="notices"``: call ``.record_notice(message, group_key=...)``
      for each pre-rendered notice. Requires ``notice_phrase``. Dispatches
      to :func:`coalesce_notices`.

    Calling the wrong ``record*`` method for the active mode raises
    ``RuntimeError``.

    Example (failures)::

        with IterativeWarningCoalescer(summary_label="bootstrap",
                                       n_total=n_bootstrap) as cz:
            for seed in seeds:
                try:
                    ...
                except (ValueError, RuntimeError) as exc:
                    cz.record(exc, prefix=f"seed={seed}")

    Example (notices)::

        with IterativeWarningCoalescer(
            mode="notices",
            summary_label="'stage' subpopulation",
            notice_phrase="have insufficient events (< 30)",
            n_total=len(labels),
            category=SmallSubpopulationWarning,
        ) as cz:
            for label in labels:
                if n_events < min_events:
                    cz.record_notice(msg, group_key=str(label))
    """

    def __init__(
        self,
        *,
        summary_label: str,
        n_total: int,
        max_emit: int = 3,
        category: Type[Warning] = IterativeWarning,
        stacklevel: int = 3,
        tail_msg: str = "",
        mode: Literal["failures", "notices"] = "failures",
        notice_phrase: str = "",
    ):
        if n_total < 1:
            raise ValueError(f"n_total must be >= 1; got {n_total}")
        if max_emit < 0:
            raise ValueError(f"max_emit must be >= 0; got {max_emit}")
        if mode not in ("failures", "notices"):
            raise ValueError(
                f"mode must be 'failures' or 'notices'; got {mode!r}"
            )
        if mode == "notices" and not notice_phrase:
            raise ValueError("notice_phrase is required when mode='notices'")
        self.summary_label = summary_label
        self.n_total = n_total
        self.max_emit = max_emit
        self.category = category
        self.stacklevel = stacklevel
        self.tail_msg = tail_msg
        self.mode = mode
        self.notice_phrase = notice_phrase
        self._failures: list[IterativeFailure] = []
        self._notices: list[IterativeNotice] = []
        self._summarized = False

    def record(self, exc: Exception, *, prefix: str = "") -> None:
        if self.mode != "failures":
            raise RuntimeError(
                f"record() requires mode='failures'; coalescer is in "
                f"mode={self.mode!r}. Use record_notice() instead."
            )
        self._failures.append(IterativeFailure.from_exception(exc, prefix=prefix))

    def record_failure(self, failure: IterativeFailure) -> None:
        if self.mode != "failures":
            raise RuntimeError(
                f"record_failure() requires mode='failures'; coalescer is in "
                f"mode={self.mode!r}. Use record_notice() instead."
            )
        self._failures.append(failure)

    def record_notice(
        self, message: str, *, group_key: str = "", display: str = ""
    ) -> None:
        if self.mode != "notices":
            raise RuntimeError(
                f"record_notice() requires mode='notices'; coalescer is in "
                f"mode={self.mode!r}. Use record() instead."
            )
        self._notices.append(
            IterativeNotice(message=message, group_key=group_key, display=display)
        )

    def summarize(self) -> int:
        if self._summarized:
            return len(self._failures) + len(self._notices)
        self._summarized = True
        # Bump stacklevel by +2 to account for the summarize() and __exit__()
        # frames the coalesce_* function will see above its own warnings.warn
        # call. This way callers pass the same stacklevel they would use for a
        # direct coalesce_*() call, and the warning still attributes to the
        # caller of compute_subpopulation_metrics() (or whichever site uses
        # the context-manager form).
        bumped_stacklevel = self.stacklevel + 2
        if self.mode == "failures":
            return coalesce_failures(
                self._failures,
                n_total=self.n_total,
                summary_label=self.summary_label,
                max_emit=self.max_emit,
                category=self.category,
                stacklevel=bumped_stacklevel,
                tail_msg=self.tail_msg,
            )
        return coalesce_notices(
            self._notices,
            n_total=self.n_total,
            summary_label=self.summary_label,
            notice_phrase=self.notice_phrase,
            max_emit=self.max_emit,
            category=self.category,
            stacklevel=bumped_stacklevel,
            tail_msg=self.tail_msg,
        )

    def __enter__(self) -> "IterativeWarningCoalescer":
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        # Always attempt to summarize so a propagating exception in the
        # with-block does not silently discard previously recorded
        # records. If summarize() itself raises while a user exception
        # is propagating, prefer the user exception — masking it with
        # our own bookkeeping failure is the worse outcome.
        try:
            self.summarize()
        except Exception:
            if exc_type is None:
                raise
        return None
