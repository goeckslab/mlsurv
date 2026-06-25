"""CLI tests for the ``mlsurv patient-report`` subcommand.

``patient-report`` wraps ``SurvivalLearner.patient_report()`` to write a
standalone single-patient HTML report. The ``--patient`` path selects a row
(index or risk-percentile spec) from the learner's test set; ``--patient-data``
supplies an external CSV cohort.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Parser-level
# ---------------------------------------------------------------------------


class TestPatientReportParser:
    def test_subcommand_registered(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'patient-report', '--load-learner', 'foo.learner',
            '--patient', '0', '--output', 'report.html',
        ])
        assert args.command == 'patient-report'
        assert args.load_learner == 'foo.learner'
        assert args.output == 'report.html'

    def test_patient_kept_as_string(self):
        """A percentile spec must survive parsing as a string."""
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'patient-report', '--load-learner', 'f.learner',
            '--patient', 'high_risk', '--output', 'r.html',
        ])
        assert args.patient == 'high_risk'

    def test_ci_defaults_and_toggles(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        base = ['patient-report', '--load-learner', 'f.learner',
                '--patient', '0', '--output', 'r.html']

        args = parser.parse_args(base)
        assert args.ci is True
        assert args.ci_band is True
        assert args.use_recalibrated is True
        assert args.offline is False

        args = parser.parse_args(
            base + ['--no-ci', '--no-ci-band', '--no-recalibrated', '--offline']
        )
        assert args.ci is False
        assert args.ci_band is False
        assert args.use_recalibrated is False
        assert args.offline is True

    def test_numeric_defaults(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'patient-report', '--load-learner', 'f.learner',
            '--patient', '0', '--output', 'r.html',
        ])
        assert args.n_neighbors == 10
        assert args.n_bootstrap == 200
        assert args.random_state is None
        assert args.times is None
        assert args.time_unit is None

    def test_output_required(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                'patient-report', '--load-learner', 'f.learner',
                '--patient', '0',
            ])

    def test_patient_selector_required(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                'patient-report', '--load-learner', 'f.learner',
                '--output', 'r.html',
            ])

    def test_patient_and_patient_data_mutually_exclusive(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                'patient-report', '--load-learner', 'f.learner',
                '--patient', '0', '--patient-data', 'p.csv',
                '--output', 'r.html',
            ])


# ---------------------------------------------------------------------------
# End-to-end CLI dispatch
# ---------------------------------------------------------------------------


def _make_data(seed=0, n=120, p=5):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)],
    )
    score = X["x0"].values + 0.5 * X["x1"].values
    times = np.maximum(0.1, rng.exponential(scale=np.exp(-score)))
    events = rng.binomial(1, 0.7, size=n).astype(bool)
    y = np.array(
        list(zip(events, times)),
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


@pytest.fixture(scope="module")
def saved_learner_single(tmp_path_factory):
    """One trained model (coxph) — model auto-resolves."""
    from mlsurv import SurvivalLearner

    X, y = _make_data()
    learner = SurvivalLearner(X, y)
    learner.setup(
        verbose=False, ph_test=False, linearity_test=False,
        check_leakage=False, cv_folds=3, random_state=42,
    )
    learner.run(
        models=["coxph"], tune=False, n_importance_repeats=0,
        bootstrap=False, verbose=False, show_eval_progress=False,
    )
    out = tmp_path_factory.mktemp("learner_single") / "session.learner"
    learner.save_learner(str(out))
    return str(out)


@pytest.fixture(scope="module")
def saved_learner_multi(tmp_path_factory):
    """Two trained models — omitting --model must error."""
    from mlsurv import SurvivalLearner

    X, y = _make_data(seed=1)
    learner = SurvivalLearner(X, y)
    learner.setup(
        verbose=False, ph_test=False, linearity_test=False,
        check_leakage=False, cv_folds=3, random_state=42,
    )
    learner.run(
        models=["coxph", "rsf"], tune=False, n_importance_repeats=0,
        bootstrap=False, verbose=False, show_eval_progress=False,
    )
    out = tmp_path_factory.mktemp("learner_multi") / "session.learner"
    learner.save_learner(str(out))
    return str(out)


@pytest.mark.integration
class TestPatientReportCommand:
    def test_patient_data_happy_path(self, saved_learner_single, tmp_path):
        """An external 1-row CSV produces a valid HTML report."""
        from mlsurv import SurvivalLearner
        from mlsurv.cli import main

        learner = SurvivalLearner.load_learner(saved_learner_single)
        row_csv = tmp_path / "row.csv"
        learner._state.X_test.iloc[[0]].to_csv(row_csv, index=False)

        out_html = tmp_path / "report.html"
        rc = main([
            'patient-report',
            '--load-learner', saved_learner_single,
            '--patient-data', str(row_csv),
            '--output', str(out_html),
            '--no-ci',
        ])
        assert rc == 0
        assert out_html.exists()
        assert out_html.stat().st_size > 0
        assert '<html' in out_html.read_text(encoding='utf-8').lower()

    def test_patient_index(self, saved_learner_single, tmp_path):
        from mlsurv.cli import main

        out_html = tmp_path / "report.html"
        rc = main([
            'patient-report',
            '--load-learner', saved_learner_single,
            '--patient', '0',
            '--output', str(out_html),
            '--no-ci',
        ])
        assert rc == 0
        assert out_html.stat().st_size > 0

    def test_percentile_spec(self, saved_learner_single, tmp_path):
        """Risk-percentile selection against the saved learner's data."""
        from mlsurv.cli import main

        out_html = tmp_path / "report.html"
        rc = main([
            'patient-report',
            '--load-learner', saved_learner_single,
            '--patient', 'high_risk',
            '--output', str(out_html),
            '--no-ci',
        ])
        assert rc == 0
        assert out_html.stat().st_size > 0

    def test_missing_model_with_multiple_errors(
        self, saved_learner_multi, tmp_path, capsys
    ):
        from mlsurv.cli import main

        out_html = tmp_path / "report.html"
        rc = main([
            'patient-report',
            '--load-learner', saved_learner_multi,
            '--patient', '0',
            '--output', str(out_html),
            '--no-ci',
        ])
        # The "multiple models" ValueError maps to EXIT_COMPUTATION (4).
        assert rc == 4
        assert 'multiple models' in capsys.readouterr().err.lower()
        assert not out_html.exists()


# ---------------------------------------------------------------------------
# Handler branches (fast tier, no model training — fake learner)
# ---------------------------------------------------------------------------


class _FakeState:
    def __init__(self, X_test):
        self.X_test = X_test


class _FakeLearner:
    """Stand-in that records patient_report kwargs or raises a set error."""

    def __init__(self, X_test=None, raise_exc=None):
        self._state = _FakeState(X_test)
        self.captured = None
        self._raise = raise_exc

    def patient_report(self, X, **kwargs):
        if self._raise is not None:
            raise self._raise
        self.captured = {'X': X, **kwargs}
        return "<html></html>"


def _patch_learner(monkeypatch, fake):
    from mlsurv.cli import _commands

    monkeypatch.setattr(
        _commands, '_load_learner_if_requested', lambda args: fake
    )


class TestPatientReportHandler:
    def test_all_kwargs_forwarded(self, monkeypatch, tmp_path):
        """Every flag must reach learner.patient_report() by the right name."""
        from mlsurv.cli import main

        X_test = pd.DataFrame({'x0': [0.1, 0.2, 0.3]})
        fake = _FakeLearner(X_test=X_test)
        _patch_learner(monkeypatch, fake)

        out = tmp_path / 'r.html'
        rc = main([
            'patient-report', '--load-learner', 'x.learner',
            '--patient', '2', '--output', str(out), '--model', 'coxph',
            '--times', '1', '2', '--time-unit', 'months',
            '--n-neighbors', '5', '--n-bootstrap', '7', '--random-state', '3',
            '--no-ci', '--no-ci-band', '--no-recalibrated', '--offline',
        ])
        assert rc == 0
        c = fake.captured
        assert c['patient'] == 2
        assert c['model'] == 'coxph'
        assert c['output_path'] == str(out)
        assert c['times'] == [1.0, 2.0]
        assert c['time_unit'] == 'months'
        assert c['n_neighbors'] == 5
        assert c['n_bootstrap'] == 7
        assert c['random_state'] == 3
        assert c['ci'] is False
        assert c['ci_band'] is False
        assert c['use_recalibrated'] is False
        assert c['offline'] is True

    def test_patient_data_sets_patient_zero(self, monkeypatch, tmp_path):
        from mlsurv.cli import main

        fake = _FakeLearner(X_test=pd.DataFrame({'x0': [0.1]}))
        _patch_learner(monkeypatch, fake)

        csv = tmp_path / 'p.csv'
        pd.DataFrame({'x0': [0.5, 0.6]}).to_csv(csv, index=False)
        rc = main([
            'patient-report', '--load-learner', 'x.learner',
            '--patient-data', str(csv), '--output', str(tmp_path / 'r.html'),
        ])
        assert rc == 0
        assert fake.captured['patient'] == 0
        # The CSV cohort, not X_test, is forwarded.
        assert len(fake.captured['X']) == 2

    def test_negative_index_converts_to_int(self, monkeypatch, tmp_path):
        from mlsurv.cli import main

        fake = _FakeLearner(X_test=pd.DataFrame({'x0': [0.1, 0.2]}))
        _patch_learner(monkeypatch, fake)

        # Leading-dash value must use the --patient=-1 form for argparse.
        rc = main([
            'patient-report', '--load-learner', 'x.learner',
            '--patient=-1', '--output', str(tmp_path / 'r.html'),
        ])
        assert rc == 0
        assert fake.captured['patient'] == -1

    def test_percentile_spec_passes_through(self, monkeypatch, tmp_path):
        from mlsurv.cli import main

        fake = _FakeLearner(X_test=pd.DataFrame({'x0': [0.1, 0.2]}))
        _patch_learner(monkeypatch, fake)

        rc = main([
            'patient-report', '--load-learner', 'x.learner',
            '--patient', 'high_risk', '--output', str(tmp_path / 'r.html'),
        ])
        assert rc == 0
        assert fake.captured['patient'] == 'high_risk'

    def test_empty_csv_errors(self, monkeypatch, tmp_path):
        from mlsurv.cli import main

        fake = _FakeLearner(X_test=pd.DataFrame({'x0': [0.1]}))
        _patch_learner(monkeypatch, fake)

        csv = tmp_path / 'empty.csv'
        csv.write_text('x0\n', encoding='utf-8')  # header only, no rows
        rc = main([
            'patient-report', '--load-learner', 'x.learner',
            '--patient-data', str(csv), '--output', str(tmp_path / 'r.html'),
        ])
        assert rc == 3  # EXIT_DATA
        assert fake.captured is None  # patient_report never called

    def test_missing_csv_errors(self, monkeypatch, tmp_path):
        from mlsurv.cli import main

        fake = _FakeLearner(X_test=pd.DataFrame({'x0': [0.1]}))
        _patch_learner(monkeypatch, fake)

        rc = main([
            'patient-report', '--load-learner', 'x.learner',
            '--patient-data', str(tmp_path / 'nope.csv'),
            '--output', str(tmp_path / 'r.html'),
        ])
        assert rc == 3  # EXIT_DATA

    def test_no_test_set_errors(self, monkeypatch, tmp_path, capsys):
        from mlsurv.cli import main

        fake = _FakeLearner(X_test=None)
        _patch_learner(monkeypatch, fake)

        rc = main([
            'patient-report', '--load-learner', 'x.learner',
            '--patient', '0', '--output', str(tmp_path / 'r.html'),
        ])
        assert rc == 3  # EXIT_DATA
        assert '--patient-data' in capsys.readouterr().err

    def test_value_error_maps_to_computation(self, monkeypatch, tmp_path):
        from mlsurv.cli import main

        fake = _FakeLearner(
            X_test=pd.DataFrame({'x0': [0.1]}),
            raise_exc=ValueError("patient='pNNN' not recognized"),
        )
        _patch_learner(monkeypatch, fake)

        rc = main([
            'patient-report', '--load-learner', 'x.learner',
            '--patient', 'pNNN', '--output', str(tmp_path / 'r.html'),
        ])
        assert rc == 4  # EXIT_COMPUTATION

    def test_runtime_error_maps_to_computation(self, monkeypatch, tmp_path):
        from mlsurv.cli import main

        fake = _FakeLearner(
            X_test=pd.DataFrame({'x0': [0.1]}),
            raise_exc=RuntimeError("predict() failed for all models"),
        )
        _patch_learner(monkeypatch, fake)

        rc = main([
            'patient-report', '--load-learner', 'x.learner',
            '--patient', '0', '--output', str(tmp_path / 'r.html'),
        ])
        assert rc == 4  # EXIT_COMPUTATION
