"""I/O helpers for the mlsurv CLI."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path


def _write_csv(df, output_path: str, label: str, *, quiet: bool = False) -> None:
    """Write a DataFrame to CSV and print confirmation."""
    try:
        df.to_csv(output_path, index=False)
    except OSError as exc:
        print(
            f"Warning: Failed to write {label} to {output_path}: {exc}",
            file=sys.stderr,
        )
        return
    if not quiet:
        print(f"\n{label} saved to: {output_path}")


def _write_json(
    payload: dict, output_path: str, label: str, *, quiet: bool = False,
) -> None:
    """Write dict payload to JSON and print confirmation."""
    try:
        with open(output_path, 'w') as f:
            json.dump(payload, f, indent=2)
    except (OSError, TypeError) as exc:
        print(
            f"Warning: Failed to write {label} to {output_path}: {exc}",
            file=sys.stderr,
        )
        return
    if not quiet:
        print(f"\n{label} saved to: {output_path}")


def _collect_report_kwargs(args) -> dict:
    """Extract report-customization keyword arguments from parsed CLI args."""
    kw: dict = {}
    sections = getattr(args, 'sections', None)
    if sections:
        kw['sections'] = sections
    outcome_name = getattr(args, 'outcome_name', None)
    if outcome_name is not None:
        kw['outcome_name'] = outcome_name
    if getattr(args, 'manuscript', False):
        kw['manuscript'] = True
    feature_groups_file = getattr(args, 'feature_groups_file', None)
    if feature_groups_file is not None:
        kw['feature_groups'] = _load_json_file(
            feature_groups_file, label="Feature groups file",
        )
    return kw


def _write_report(
    learner,
    report_path: str | None,
    *,
    eval_result=None,
    bootstrap_results=None,
    quiet: bool = False,
    **report_kwargs,
) -> None:
    """Write an optional HTML/Markdown report from learner state/results."""
    if not report_path:
        return
    try:
        learner.generate_report(
            report_path,
            eval_result=eval_result,
            bootstrap_results=bootstrap_results,
            **report_kwargs,
        )
    except OSError as exc:
        print(
            f"Warning: Failed to write report to {report_path}: {exc}",
            file=sys.stderr,
        )
        return
    if not quiet:
        print(f"\nReport saved to: {report_path}")


def _print_loaded_data(args, X, y) -> None:
    """Print basic dataset diagnostics when --verbose is set."""
    import sys as _sys

    if not args.verbose:
        return
    # When --format json/csv, send verbose diagnostics to stderr
    # to avoid contaminating machine-readable stdout.
    out = (
        _sys.stderr
        if getattr(args, 'output_format', 'table') != 'table'
        else _sys.stdout
    )
    print(f"Loaded data: {len(X)} samples, {len(X.columns)} features", file=out)
    print(f"Events: {y['event'].sum()} ({y['event'].mean():.1%})", file=out)


def _load_json_file(path: str | None, *, label: str = "JSON file") -> dict | None:
    """Load JSON object from disk if path is provided."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    with p.open() as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {label} ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object/dict")
    return payload


def _invoke_with_supported_kwargs(func, *args, **kwargs):
    """Call a function while dropping kwargs unsupported by its signature."""
    import warnings

    from .._warnings import SuggestionWarning

    signature = inspect.signature(func)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return func(*args, **kwargs)
    supported = set(signature.parameters)
    dropped = sorted(k for k in kwargs if k not in supported)
    if dropped:
        warnings.warn(
            f"{func.__qualname__} does not accept: {dropped}. "
            f"These arguments were ignored.",
            SuggestionWarning,
            stacklevel=2,
        )
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return func(*args, **filtered)
