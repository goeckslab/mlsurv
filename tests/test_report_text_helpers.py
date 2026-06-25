"""Tests for mlsurv.reports._text helpers."""

import pytest

from mlsurv.reports._text import pluralize, title_if_lower


@pytest.mark.parametrize(
    "word, expected",
    [
        ("event", "events"),
        ("death", "deaths"),
        ("metastasis", "metastases"),
        ("loss", "losses"),
        ("study", "studies"),
        ("delay", "delays"),
        ("OS", "OS"),
        ("PFS", "PFS"),
        ("Metastasis", "Metastases"),
        ("relapse", "relapses"),
        ("match", "matches"),
        ("fox", "foxes"),
        ("crash", "crashes"),
        ("", ""),
    ],
)
def test_pluralize(word, expected):
    assert pluralize(word) == expected


@pytest.mark.parametrize(
    "word, expected",
    [
        ("event", "Event"),
        ("death", "Death"),
        ("OS", "OS"),
        ("PFS", "PFS"),
        ("mOS", "mOS"),
        ("Death", "Death"),
        ("disease progression", "Disease Progression"),
        ("", ""),
    ],
)
def test_title_if_lower(word, expected):
    assert title_if_lower(word) == expected
