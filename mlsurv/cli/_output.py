"""Structured output formatting for the mlsurv CLI."""

from __future__ import annotations

import csv
import io
import json
import sys

import pandas as pd


def format_dataframe(df: pd.DataFrame, fmt: str) -> str:
    """Format a DataFrame for stdout output.

    Parameters
    ----------
    df : pd.DataFrame
    fmt : str
        One of ``'table'``, ``'json'``, ``'csv'``.

    Returns
    -------
    str
        The DataFrame rendered in the requested format.
    """
    if fmt == 'json':
        return df.to_json(orient='records', indent=2)
    if fmt == 'csv':
        return df.to_csv(index=False).rstrip('\n')
    return df.to_string(index=False)


def format_dict(data: dict, fmt: str) -> str | None:
    """Format a dict for stdout output.

    Returns *None* for ``'table'`` so the caller can fall through to its
    existing key-value print logic.
    """
    if fmt == 'json':
        return json.dumps(data, indent=2)
    if fmt == 'csv':
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(['key', 'value'])
        for k, v in data.items():
            writer.writerow([k, v])
        return buf.getvalue().rstrip('\n')
    return None


def format_list(items: list, fmt: str) -> str:
    """Format a list for stdout output."""
    if fmt == 'json':
        return json.dumps(items, indent=2)
    if fmt == 'csv':
        return '\n'.join(str(item) for item in items)
    return '\n'.join(str(item) for item in items)


def emit(text: str) -> None:
    """Print formatted output, handling broken pipe gracefully."""
    try:
        print(text)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass
