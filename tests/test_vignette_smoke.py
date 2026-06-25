"""In-process smoke tests for key vignette workflows.

These tests exercise the same API paths as the vignette scripts but run
in-process (no subprocess overhead) with minimal iterations for speed.
Marked ``integration`` so they are excluded from the default (fast) tier.

Target: the 3 integration-tier smoke tests complete in < 2 minutes total
(the integration_heavy test runs separately).
"""

import warnings
import pytest

pytestmark = pytest.mark.integration

# Minimal settings for speed — just enough to exercise the API path.
_CV_FOLDS = 3
_N_BOOTSTRAP = 2


def _prepare_gbsg2():
    """Load and prepare GBSG2 data for vignette smoke tests.

    Returns (X, y) with categorical columns converted to numeric.
    Skips the test if scikit-survival is not installed.
    """
    pytest.importorskip("sksurv", reason="scikit-survival required")
    from sksurv.datasets import load_gbsg2
    from mlsurv.utils import to_structured_array

    X, y_raw = load_gbsg2()
    X = X.copy()
    X["horTh"] = (X["horTh"] == "yes").astype(int)
    X["menostat"] = (X["menostat"] == "Post").astype(int)
    X["tgrade"] = X["tgrade"].map({"I": 1, "II": 2, "III": 3}).astype(int)
    y = to_structured_array(y_raw["time"], y_raw["cens"])
    return X, y


class TestVignetteSmoke01Quickstart:
    """Smoke test mirroring vignette_01_quickstart.ipynb (METABRIC)."""

    def test_quickstart_workflow(self, tmp_path):
        """setup -> run(coxph, xgbcox) -> summary -> plot_survival -> report."""
        from pathlib import Path
        from mlsurv import SurvivalLearner
        from mlsurv.utils import load_survival_data

        csv_path = Path(__file__).parent.parent / "examples" / "data" / "metabric.csv"
        X, y = load_survival_data(str(csv_path), time_col="duration", event_col="event")
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.25, cv_folds=_CV_FOLDS, random_state=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            comparison = learner.run(
                models=["coxph", "xgbcox"],
                verbose=False,
                n_bootstrap=_N_BOOTSTRAP,
            )

        assert comparison is not None
        assert len(comparison) == 2
        assert learner.best_model is not None
        assert learner.best_model.name in ("coxph", "xgbcox")

        # Summary table
        summary_df = learner.summary()
        assert not summary_df.empty

        # KM survival plot — plot_* select by name (or the 'best' sentinel),
        # not by ModelResult object.
        fig = learner.plot_survival('best')
        assert fig is not None

        report_path = tmp_path / "quickstart_report.html"
        learner.generate_report(str(report_path))
        assert report_path.exists()
        assert report_path.stat().st_size > 500


class TestVignetteSmoke02WorkflowTour:
    """Smoke test mirroring vignette_02_workflow_tour.py."""

    def test_workflow_tour_workflow(self, tmp_path):
        """setup -> run -> tune(all) -> evaluate -> plots -> report."""
        from mlsurv import SurvivalLearner

        X, y = _prepare_gbsg2()
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=_CV_FOLDS, random_state=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Compare models (baseline, no tuning)
            comparison = learner.run(
                models=["coxph", "rsf"],
                tune=False,
                verbose=False,
                n_importance_repeats=2,
                n_bootstrap=_N_BOOTSTRAP,
            )
            assert comparison is not None
            assert len(comparison) == 2

            # Tune all models — CoxPH has no tunable hyperparameters
            # (search_spaces.DEFAULT_SPACES['coxph'] is empty), so only
            # RSF returns a tuned variant.
            tuned = learner.tune(n_trials=1)
            assert isinstance(tuned, dict)
            assert len(tuned) == 1

            # Evaluate all tuned models
            eval_result = learner.evaluate(tuned, n_importance_repeats=2)
            assert eval_result is not None

            # Plots
            fig_metric = learner.plot_metric(style="forest")
            assert fig_metric is not None

            fig_surv = learner.plot_survival(list(tuned), n_groups=3)
            assert fig_surv is not None

        # Report
        report_path = tmp_path / "workflow_tour_report.html"
        learner.generate_report(str(report_path))
        assert report_path.exists()


class TestVignetteSmoke04Benchmark:
    """Smoke test mirroring vignette_04_benchmark_sksurv.py."""

    def test_benchmark_cindex_above_random(self):
        """setup -> run(coxnet) -> assert C-index > 0.5."""
        from mlsurv import SurvivalLearner

        X, y = _prepare_gbsg2()
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=_CV_FOLDS, random_state=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = learner.run(
                models=["coxnet"],
                verbose=False,
                n_trials=1,
                n_bootstrap=_N_BOOTSTRAP,
            )

        assert result is not None
        c_index = result["coxnet"].cv_metric_mean
        assert c_index > 0.5, f"C-index {c_index:.3f} not above random"


class TestVignetteSmoke07Immunotherapy:
    """Smoke test mirroring vignette_07_immunotherapy_case_study.ipynb."""

    def test_immunotherapy_workflow(self):
        """Load LORIS data, run coxph, benchmark vs LORIS, validate on MSK1."""
        import sys
        from pathlib import Path
        import numpy as np
        from mlsurv import SurvivalLearner
        from mlsurv.utils import to_structured_array

        data_dir = Path(__file__).parent.parent / "examples" / "data" / "loris"
        if not (data_dir / "Chowell_train.csv").exists():
            pytest.skip("LORIS data not available in examples/data/loris/")

        import pandas as pd

        df_train = pd.read_csv(data_dir / "Chowell_train.csv")
        df_test = pd.read_csv(data_dir / "Chowell_test.csv")

        # Feature engineering (same as vignette)
        clinical = ["TMB", "Age", "Systemic_therapy_history", "Albumin", "NLR"]
        X_train = df_train[clinical].copy()
        ct_train = pd.get_dummies(df_train["CancerType"], prefix="CancerType")
        X_train = pd.concat([X_train, ct_train], axis=1)

        X_test = df_test[clinical].copy()
        ct_test = pd.get_dummies(df_test["CancerType"], prefix="CancerType")
        X_test = pd.concat([X_test, ct_test], axis=1)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        y_train = to_structured_array(df_train["OS_Months"], df_train["OS_Event"])
        y_test = to_structured_array(df_test["OS_Months"], df_test["OS_Event"])

        learner = SurvivalLearner(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
        )
        learner.setup(cv_folds=_CV_FOLDS, random_state=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = learner.run(
                models=["coxph"],
                verbose=False,
                n_bootstrap=_N_BOOTSTRAP,
                n_trials=0,
            )

        assert results is not None
        assert "coxph" in results

        # Verify LORIS calculator works with string CancerType
        scripts_dir = Path(__file__).parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        from loris_score_calculator import calculate_loris
        loris_scores = calculate_loris(df_test).values
        assert loris_scores.shape == (len(df_test),)
        assert np.all(np.isfinite(loris_scores))

        # Validate on MSK1
        df_msk1 = pd.read_csv(data_dir / "MSK1.csv")
        X_msk1 = df_msk1[clinical].copy()
        ct_msk1 = pd.get_dummies(df_msk1["CancerType"], prefix="CancerType")
        X_msk1 = pd.concat([X_msk1, ct_msk1], axis=1)
        X_msk1 = X_msk1.reindex(columns=X_train.columns, fill_value=0)
        y_msk1 = to_structured_array(df_msk1["OS_Months"], df_msk1["OS_Event"])

        learner.validate(X_msk1, y_msk1, name="MSK1")


@pytest.mark.integration_heavy
class TestVignetteSmoke05Comparison:
    """Smoke test mirroring vignette_05_model_comparison.py.

    Marked integration_heavy: 3-model comparison with tuning is expensive.
    """

    def test_multi_model_comparison(self, tmp_path):
        """setup -> run(coxph, coxnet, rsf) -> report.

        GBSA excluded: its auto-tuning (55 trials) on GBSG2 takes >15 min,
        exceeding reasonable test budgets.
        """
        from mlsurv import SurvivalLearner

        X, y = _prepare_gbsg2()
        learner = SurvivalLearner(X, y)
        learner.setup(test_fraction=0.3, cv_folds=_CV_FOLDS, random_state=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = learner.run(
                models=["coxph", "coxnet", "rsf"],
                verbose=False,
                n_bootstrap=_N_BOOTSTRAP,
            )

        assert result is not None
        assert len(result) == 3
        assert learner.best_model is not None

        report_path = tmp_path / "comparison_report.html"
        learner.generate_report(str(report_path))
        assert report_path.exists()
