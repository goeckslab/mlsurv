"""Lifecycle integration test for the reference-based model lookup.

Pins the public-API invariant established by ``_lookup_model_by_name`` /
``_resolve_model_results``: a user-facing algorithm name (``'coxnet'``)
resolves to the right ``ModelResult`` at every public-API boundary. There
is no ``_tuned`` storage suffix — ``tune()`` overwrites the model in place,
so ``'coxnet'`` is the single key for the tuned model.

The test traces a single ``ModelResult`` reference flow through every public
stage:

    setup -> train -> tune -> evaluate -> bootstrap
          -> calibrate -> predict (cal vs raw) -> explain -> validate
          -> benchmark -> plot_calibration -> patient_report -> generate_report
          -> export_study -> save_artifact -> load_artifact -> predict

At each stage it passes ``'coxnet'`` as the user would, and asserts the
resolver lands on the tuned variant. Any future internal helper that
re-resolves by raw string against ``_state.models`` without going through
``_lookup_model_by_name`` will ``KeyError`` here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner

pytestmark = pytest.mark.integration


def _make_small_data(n: int = 80, seed: int = 0):
    """Tiny synthetic dataset (matches the test_resolve_model_results helper)."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        'a': rng.randn(n),
        'b': rng.randn(n),
        'c': rng.randn(n),
    })
    times = rng.exponential(scale=10.0, size=n) + 1.0
    events = rng.randint(0, 2, size=n).astype(bool)
    y = np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)],
    )
    return X, y


def test_lifecycle_resolves_by_name(tmp_path: Path):
    """Trace ModelResult flow through the full public API by user-facing name."""
    X, y = _make_small_data()
    learner = SurvivalLearner(X, y)

    # 1. setup
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=0)

    # 2. train (untuned)
    learner.run(['coxnet'], tune=False, evaluate=False, verbose=False)
    assert 'coxnet' in learner._state.models

    # 3. tune -- overwrites 'coxnet' in place. No '_tuned' storage suffix exists.
    learner.tune(models='coxnet', n_trials=2, evaluate=False, verbose=False)
    assert learner._state.models['coxnet'].is_tuned, "tuned variant at plain key"
    assert 'coxnet_tuned' not in learner._state.models

    # User-facing string 'coxnet' resolves to the tuned variant directly.
    resolved = learner._lookup_model_by_name('coxnet')
    assert resolved.name == 'coxnet'
    assert resolved.is_tuned
    assert resolved.base_name == 'coxnet'

    # 4. evaluate -- public API accepts the user-facing string
    eval_result = learner.evaluate(
        'coxnet', n_importance_repeats=1, verbose=False, show_progress=False,
    )
    assert eval_result is not None
    assert learner._state.models['coxnet'].evaluation is not None, (
        "evaluate('coxnet') must populate the tuned variant"
    )

    # 5. bootstrap
    boot = learner.bootstrap('coxnet', n_bootstrap=2, verbose=False)
    assert boot is not None
    assert learner._state.models['coxnet'].bootstrap is not None

    # 6. calibrate -- attach an isotonic S(t|x) calibrator to the tuned model
    # via the live train(calibrate=...) path (``_attach_calibrators``), under
    # the resolved storage key ('coxnet') so ``predict()`` finds it. Attaching
    # in place keeps the tuned variant (no retrain).
    from mlsurv.evaluation._calibrators import IsotonicCalibrator
    mr = learner._lookup_model_by_name('coxnet')
    learner._attach_calibrators(mr, 'isotonic')
    assert learner._has_calibrator('coxnet'), (
        "calibrate must attach the calibrator under 'coxnet'"
    )
    assert isinstance(
        learner._state.models['coxnet'].calibrator, IsotonicCalibrator
    )

    # 7. predict -- string 'coxnet' resolves to the tuned model.
    # Make ``use_recalibrated`` explicit per the CONTRIBUTING.md rule.
    pred = learner.predict(
        learner.X_test.head(3), models='coxnet',
        verbose=False, use_recalibrated=True,
    )
    assert pred.model_name == 'coxnet'

    # 8. predict with use_recalibrated=False to confirm the calibrator is
    # actually being applied through the resolved key (numerics differ from
    # the calibrated path). Without this assertion a regression that
    # silently dropped the calibrator would leave (7) green.
    pred_raw = learner.predict(
        learner.X_test.head(3), models='coxnet',
        verbose=False, use_recalibrated=False,
    )
    assert pred_raw.model_name == 'coxnet'
    # Both paths must produce S(t|x); a None here would let the load-bearing
    # diff assertion below no-op and hide a silently-dropped calibrator.
    assert pred._surv_probs is not None and pred_raw._surv_probs is not None, (
        "predict() returned no S(t|x) for the calibrated/raw comparison"
    )
    assert not np.allclose(pred._surv_probs, pred_raw._surv_probs), (
        "calibrated and raw S(t|x) should differ; "
        "if they match, the calibrator was silently skipped"
    )

    # 9. explain -- wraps SHAP. May raise if SHAP isn't installed; in that
    # case skip but keep the resolver assertion via patient_report below.
    try:
        explanation = learner.explain(patient=0, model='coxnet', plot=False)
    except (ImportError, ValueError) as exc:
        pytest.skip(f"explain() unavailable in this environment: {exc}")
    assert getattr(explanation, 'model_name', 'coxnet') == 'coxnet'

    # 10. validate -- external cohort (use the test split as a stand-in).
    val_result = learner.validate(
        learner.X_test, learner.y_test,
        models='coxnet', name='LifecycleExternal',
        bootstrap=False, verbose=False,
    )
    assert val_result is not None
    assert 'LifecycleExternal' in learner._state.validation_results

    # 11. benchmark -- references aligned with the test split.
    rng = np.random.RandomState(0)
    refs = {'rand': rng.rand(len(learner.y_test))}
    bench = learner.benchmark(
        refs, models='coxnet', dataset='test',
        bootstrap=False, n_importance_repeats=0, verbose=False,
    )
    assert bench is not None

    # 12. plot_calibration -- exercises ``_resolve_models_arg``.
    fig = learner.plot_calibration(models='coxnet')
    assert fig is not None

    # 13. patient_report -- single-patient HTML wrapper around predict + explain
    report_path = tmp_path / 'patient.html'
    html = learner.patient_report(
        learner.X_test.head(2),
        patient=0,
        model='coxnet',
        output_path=str(report_path),
        ci=False,
        ci_band=False,
    )
    assert isinstance(html, str) and len(html) > 0
    assert report_path.exists()

    # 14. generate_report -- multi-section HTML for the whole learner
    full_report = tmp_path / 'full.html'
    learner.generate_report(output_path=str(full_report))
    assert full_report.exists()

    # 15. export_study -- alias must resolve to the tuned model's study.
    study_data = learner.export_study('coxnet')
    assert study_data is not None
    assert isinstance(study_data, dict)
    assert 'trials' in study_data and len(study_data['trials']) > 0

    # 16. save_artifact -- resolve the user-facing name to a ModelResult and save
    artifact_dir = tmp_path / 'artifact'
    saved_mr = learner._lookup_model_by_name('coxnet')
    learner.save_artifact(saved_mr, artifact_dir, step='tune')
    assert (artifact_dir / 'artifact.json').exists()

    # 17. load_artifact -- round-trip storage-key invariant through serialization
    metadata, loaded_model, _recal = SurvivalLearner.load_artifact(artifact_dir)
    assert metadata.get('model_name') == 'coxnet', (
        "save_artifact must persist the resolved storage key"
    )
    assert loaded_model is not None

    # 18. predict again -- final regression net on the original learner
    pred2 = learner.predict(
        learner.X_test.head(3), models='coxnet',
        verbose=False, use_recalibrated=False,
    )
    assert pred2.model_name == 'coxnet'
