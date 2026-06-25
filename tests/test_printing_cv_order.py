"""Regression tests for print_multi_model_cv_table row ordering.

Rows must be ordered best-first by ``cv_metric_mean``.
``cv_metric_mean`` is always stored in higher-is-better space —
``score_survival_model`` negates lower-is-better scorers (e.g. integrated
Brier returns ``-ibs``) before fold scores are aggregated. The table
sort is therefore unconditionally descending; the ``lower_is_better``
flag in ``METRIC_REGISTRY`` describes the raw-metric direction shown in
the table cells, not the ``cv_metric_mean`` direction.
"""

from unittest.mock import MagicMock

import numpy as np

from mlsurv.results.printing import print_multi_model_cv_table


def _make_result(mean, std=0.01, metric='harrell'):
    r = MagicMock(spec=['cv_metrics', 'cv_metric_mean', 'cv_metric_std'])
    r.cv_metrics = {metric: {'mean': mean, 'std': std, 'scores': []}}
    r.cv_metric_mean = mean
    r.cv_metric_std = std
    return r


def _row_order(output, names):
    """Return ``names`` ordered by their first appearance in ``output``."""
    positions = [(output.find(n), n) for n in names]
    assert all(p >= 0 for p, _ in positions), f"missing models in output: {output}"
    return [n for _, n in sorted(positions)]


class TestCvTableRowOrder:
    """Best-first row ordering for print_multi_model_cv_table."""

    def test_higher_is_better_metric_orders_largest_first(self, capsys):
        # Harrell C: cv_metric_mean is the raw C-index (higher = better).
        results = {
            'rsf': _make_result(0.7),
            'coxph': _make_result(0.6),
        }
        print_multi_model_cv_table(results, n_folds=5)
        out = capsys.readouterr().out
        assert _row_order(out, ['rsf', 'coxph']) == ['rsf', 'coxph']

    def test_lower_is_better_metric_orders_least_negative_first(self, capsys):
        # Integrated Brier: cv_metric_mean = -ibs (negated to higher-is-better).
        # Insertion order is deliberately worst-first so the legacy
        # ``lower_is_better=True``-aware sort would have put the worse
        # model on top. With the fix, ``-0.15`` (raw IBS 0.15, better)
        # must lead.
        results = {
            'coxph': _make_result(-0.25, metric='integrated_brier'),  # raw IBS 0.25
            'rsf': _make_result(-0.15, metric='integrated_brier'),    # raw IBS 0.15
        }
        print_multi_model_cv_table(results, n_folds=5)
        out = capsys.readouterr().out
        assert _row_order(out, ['rsf', 'coxph']) == ['rsf', 'coxph']

    def test_missing_cv_placed_last(self, capsys):
        results = {
            'broken': _make_result(None),
            'rsf': _make_result(0.7),
            'nan_model': _make_result(np.nan),
            'coxph': _make_result(0.6),
        }
        print_multi_model_cv_table(results, n_folds=5)
        out = capsys.readouterr().out
        order = _row_order(out, ['rsf', 'coxph', 'broken', 'nan_model'])
        assert order[:2] == ['rsf', 'coxph']
        assert set(order[2:]) == {'broken', 'nan_model'}
