"""Tests for scripts/report_dev.py ReportReader.

Covers nested inline tags (<strong>, <em>) inside an <li> that caused
the '- ' bullet prefix to be emitted on every text chunk instead of once
per list item.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "scripts" / "report_dev.py"


def _load_report_dev():
    spec = importlib.util.spec_from_file_location("report_dev", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ReportReader():
    return _load_report_dev().ReportReader


def _parse(ReportReader, body_html: str, anchor: str = "x") -> str:
    """Wrap a body fragment in a report-section and return extracted text."""
    html = (
        f'<html><body><section class="report-section" id="{anchor}">'
        f"{body_html}"
        f"</section></body></html>"
    )
    reader = ReportReader()
    reader.feed(html)
    return reader.sections.get(anchor, "")


def test_li_with_nested_inline_tags_emits_single_prefix(ReportReader):
    text = _parse(
        ReportReader,
        "<ul><li><strong>X</strong> <em>y</em></li></ul>",
    )
    assert "- X y" in text
    assert "- X- y" not in text


def test_multiple_li_each_get_one_prefix(ReportReader):
    text = _parse(
        ReportReader,
        "<ul>"
        "<li><strong>Best model:</strong> <em>XGBoost-AFT</em></li>"
        "<li><strong>Sample size:</strong> 1903 patients</li>"
        "</ul>",
    )
    assert text.count("- ") == 2
    assert "- Best model: XGBoost-AFT" in text
    assert "- Sample size: 1903 patients" in text


def test_ordered_list_with_nested_tags(ReportReader):
    text = _parse(
        ReportReader,
        "<ol>"
        "<li><strong>A</strong> b</li>"
        "<li><strong>C</strong> d</li>"
        "</ol>",
    )
    assert "1. A b" in text
    assert "2. C d" in text
    assert "1. A- b" not in text


def test_plain_li_no_nested_tags_regression(ReportReader):
    text = _parse(
        ReportReader,
        "<ul><li>plain</li></ul>",
    )
    assert "- plain" in text


def test_verbatim_pathological_strings_absent(ReportReader):
    """Pin the exact bug fingerprint so a regression
    producing 'Best model:- XGBoost-AFT- ...' fails loudly."""
    text = _parse(
        ReportReader,
        "<ul>"
        "<li><strong>Best model:</strong> <em>XGBoost-AFT</em></li>"
        "</ul>",
    )
    assert "Best model:-" not in text
    assert "XGBoost-AFT-" not in text


def test_nested_list_outer_and_inner_each_get_prefix(ReportReader):
    """Nested <ul> inside an <li>: both outer and inner items must receive
    bullet prefixes (no inner item silently merges into the outer line)."""
    text = _parse(
        ReportReader,
        "<ul>"
        "<li>outer<ul><li>inner</li></ul></li>"
        "</ul>",
    )
    assert "- outer" in text
    assert "- inner" in text


def test_empty_li_does_not_break_sibling_prefix(ReportReader):
    text = _parse(
        ReportReader,
        "<ul><li></li><li>x</li></ul>",
    )
    assert "- x" in text


def test_li_with_leading_text_then_inline_tag(ReportReader):
    text = _parse(
        ReportReader,
        "<ul><li>prefix <strong>bold</strong> suffix</li></ul>",
    )
    assert "- prefix bold suffix" in text
