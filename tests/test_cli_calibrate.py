"""CLI tests for the ``mlsurv calibrate`` subcommand.

``calibrate`` wraps the same internal ``_attach_calibrators()`` path that
``train(calibrate=...)`` uses, so it supports all three methods (``isotonic``,
``lowdof``, ``breslow_slope``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Parser-level
# ---------------------------------------------------------------------------


class TestCalibrateParser:
    def test_subcommand_registered(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'calibrate', '--load-learner', 'foo.learner',
            '--models', 'rsf', '--method', 'isotonic',
        ])
        assert args.command == 'calibrate'
        assert args.models == ['rsf']
        assert args.method == 'isotonic'
        assert args.load_learner == 'foo.learner'

    def test_method_required(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        # --method has no default; argparse exits when it is omitted.
        with pytest.raises(SystemExit):
            parser.parse_args(['calibrate'])

    def test_method_choices_enforced(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['calibrate', '--method', 'bogus'])

    def test_all_methods_accepted(self):
        from mlsurv.cli import create_parser

        parser = create_parser()
        for method in ('isotonic', 'lowdof', 'breslow_slope'):
            args = parser.parse_args(['calibrate', '--method', method])
            assert args.method == method
            assert args.models is None
        # The modern path resolves horizons internally; no --horizons flag.
        assert not hasattr(args, 'horizons')


# ---------------------------------------------------------------------------
# End-to-end CLI dispatch
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _saved_learner_template(tmp_path_factory):
    """Train a score-based (coxph) and a non-score (rsf) model, save to disk."""
    from mlsurv import SurvivalLearner

    rng = np.random.default_rng(0)
    n, p = 200, 5
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

    learner = SurvivalLearner(X, y)
    learner.setup(
        verbose=False, ph_test=False, linearity_test=False,
        check_leakage=False, cv_folds=3,
    )
    learner.run(
        models=["coxph", "rsf"],
        tune=False, n_importance_repeats=0,
        bootstrap=False, verbose=False, show_eval_progress=False,
    )

    out = tmp_path_factory.mktemp("learner") / "session.learner"
    learner.save_learner(str(out))
    return Path(out)


@pytest.fixture
def saved_learner(_saved_learner_template, tmp_path):
    """Per-test copy so save_learner output is isolated."""
    import shutil

    dst = tmp_path / "session.learner"
    shutil.copy(_saved_learner_template, dst)
    return dst


@pytest.mark.integration
class TestCalibrateCommand:
    def test_isotonic_populates_state(self, saved_learner, tmp_path):
        from mlsurv.cli import main

        out_path = tmp_path / "session_cal.learner"
        rc = main([
            'calibrate',
            '--load-learner', str(saved_learner),
            '--models', 'coxph',
            '--method', 'isotonic',
            '--save-learner', str(out_path),
        ])
        assert rc == 0
        assert out_path.exists()

        # Reload and verify the attached calibrator round-trips.
        from mlsurv import SurvivalLearner
        reloaded = SurvivalLearner.load_learner(str(out_path))
        assert reloaded._has_calibrator("coxph")
        assert reloaded._state.models["coxph"].calibrator.method == "isotonic"
        # The untargeted model stays raw.
        assert not reloaded._has_calibrator("rsf")

    def test_all_trained_default(self, saved_learner, tmp_path):
        """Omitting --models calibrates every trained model."""
        from mlsurv.cli import main

        out_path = tmp_path / "session_all.learner"
        rc = main([
            'calibrate',
            '--load-learner', str(saved_learner),
            '--method', 'isotonic',
            '--save-learner', str(out_path),
        ])
        assert rc == 0

        from mlsurv import SurvivalLearner
        reloaded = SurvivalLearner.load_learner(str(out_path))
        # Both models calibrated, since --models was omitted.
        assert reloaded._has_calibrator("coxph")
        assert reloaded._has_calibrator("rsf")

    def test_output_csv_summary(self, saved_learner, tmp_path):
        """--output writes a one-row-per-calibrated-model CSV."""
        from mlsurv.cli import main

        csv_out = tmp_path / "summary.csv"
        rc = main([
            'calibrate',
            '--load-learner', str(saved_learner),
            '--models', 'coxph',
            '--method', 'isotonic',
            '--output', str(csv_out),
        ])
        assert rc == 0
        assert csv_out.exists()

        df = pd.read_csv(csv_out)
        assert list(df['model']) == ['coxph']
        assert list(df['method']) == ['isotonic']
        assert 'n_horizons' in df.columns

    def test_breslow_slope_mixed_batch_rejects_all(self, saved_learner, tmp_path):
        """breslow_slope over all-trained (coxph + non-score rsf) rejects the
        whole batch before attaching anything."""
        from mlsurv.cli import main

        out_path = tmp_path / "session_mixed.learner"
        rc = main([
            'calibrate',
            '--load-learner', str(saved_learner),
            '--method', 'breslow_slope',
            '--save-learner', str(out_path),
        ])
        # resolve_calibration_spec rejects the batch -> EXIT_USAGE (2),
        # before any calibrator is attached or the learner re-saved.
        assert rc == 2
        assert not out_path.exists()

    def test_no_load_trains_then_calibrates(self, tmp_path):
        """Without --load-learner, named models are trained then calibrated."""
        from mlsurv.cli import main

        csv_path = tmp_path / "tiny.csv"
        rng = np.random.default_rng(1)
        n = 80
        lines = ["x0,x1,time,event"]
        for i in range(n):
            x0, x1 = rng.normal(), rng.normal()
            t = max(0.1, rng.exponential(scale=2.0))
            e = int(rng.binomial(1, 0.7))
            lines.append(f"{x0:.4f},{x1:.4f},{t:.4f},{e}")
        csv_path.write_text("\n".join(lines) + "\n")

        rc = main([
            'calibrate', str(csv_path),
            '--models', 'coxph',
            '--method', 'isotonic',
        ])
        assert rc == 0

    def test_lowdof_method(self, saved_learner, tmp_path):
        from mlsurv.cli import main

        out_path = tmp_path / "session_lowdof.learner"
        rc = main([
            'calibrate',
            '--load-learner', str(saved_learner),
            '--models', 'coxph',
            '--method', 'lowdof',
            '--save-learner', str(out_path),
        ])
        assert rc == 0

        from mlsurv import SurvivalLearner
        reloaded = SurvivalLearner.load_learner(str(out_path))
        assert reloaded._state.models["coxph"].calibrator.method == "lowdof"

    def test_breslow_slope_on_score_model(self, saved_learner, tmp_path):
        from mlsurv.cli import main

        out_path = tmp_path / "session_breslow.learner"
        rc = main([
            'calibrate',
            '--load-learner', str(saved_learner),
            '--models', 'coxph',
            '--method', 'breslow_slope',
            '--save-learner', str(out_path),
        ])
        assert rc == 0

        from mlsurv import SurvivalLearner
        reloaded = SurvivalLearner.load_learner(str(out_path))
        assert reloaded._state.models["coxph"].calibrator.method == "breslow_slope"

    def test_breslow_slope_on_non_score_model_errors(self, saved_learner):
        from mlsurv.cli import main

        # rsf is not score-based; breslow_slope is invalid for it.
        rc = main([
            'calibrate',
            '--load-learner', str(saved_learner),
            '--models', 'rsf',
            '--method', 'breslow_slope',
        ])
        # CLIUsageError -> EXIT_USAGE (2)
        assert rc == 2

    def test_unknown_model_errors(self, saved_learner):
        from mlsurv.cli import main

        rc = main([
            'calibrate',
            '--load-learner', str(saved_learner),
            '--models', 'doesnotexist',
            '--method', 'isotonic',
        ])
        # CLIUsageError -> EXIT_USAGE (2)
        assert rc == 2

    def test_no_data_no_load_requires_models(self, tmp_path):
        """The no-load-learner branch refuses to run without --models."""
        from mlsurv.cli import main

        # Provide a tiny CSV so data load succeeds; the failure must come
        # from the missing --models guard, not from a load error.
        csv_path = tmp_path / "tiny.csv"
        csv_path.write_text(
            "x0,time,event\n"
            + "\n".join(
                f"{i * 0.1},{1.0 + i * 0.2},{i % 2}" for i in range(20)
            )
            + "\n"
        )
        rc = main(['calibrate', str(csv_path), '--method', 'isotonic'])
        assert rc == 2  # CLIUsageError -> EXIT_USAGE
