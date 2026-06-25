"""Unit tests for pure display/formatting helpers in ``mlsurv.learner._format``.

These are fast, training-free tests of the box/banner string builders. They live
in their own module (not ``test_integration.py``, which carries a file-level
``@pytest.mark.integration`` marker) so they run on every fast-tier invocation.
"""

from mlsurv.learner._format import _format_trials_display


def test_format_trials_display_dict():
    """_format_trials_display handles dict n_trials."""
    result = _format_trials_display(
        {'rsf': 100, 'coxph': 20},
        ['rsf', 'coxph'],
        n_features=10,
    )
    assert "per-model" in result
    assert "100" in result
    assert "20" in result


def test_format_trials_display_dict_auto_fallback():
    """_format_trials_display dict with missing model falls back to auto."""
    result = _format_trials_display(
        {'rsf': 100},  # coxph not in dict -> auto
        ['rsf', 'coxph'],
        n_features=10,
    )
    assert "per-model" in result
    assert "100" in result
    # coxph should show auto-resolved value, not 'auto' string
    assert "auto" not in result.lower()
