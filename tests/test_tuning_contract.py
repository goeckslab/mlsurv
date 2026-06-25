"""Thematic regression home for the ``tune()`` contract.

Consolidates three single-topic regression files into one tuning-contract suite
(the pilot for folding regression clusters into thematic
homes). The three lineages folded here:

- **Overwrite & no ``_tuned`` suffix.** ``tune()`` overwrites the untuned
  model at its plain key; ``base_name`` equals ``name``;
  the ``_tuned`` storage suffix no longer exists, so ``predict(ci=True)`` and
  ``generate_report()`` work on tuned models and no module string-strips ``_tuned``.
  (Includes a search-space sub-check: the tuned model's search-space cell renders the
  actual DEFAULT_SPACES ranges via ``base_name``, not a fallback ``'-'``.)
- **Interaction-kwarg forwarding.** ``tune()`` wires the four
  ``include_interactions`` / ``interaction_*`` kwargs through ``TuneConfig`` →
  ``_run_post_tune_evaluation`` → ``evaluate(...)``.
- **``run(tune=True)`` interaction forwarding & preservation.** The sibling
  of the ``tune()`` forwarding case: the ``run(...)`` dispatch path (``_run_model_comparison`` →
  ``_tune_and_train`` → ``_post_run_evaluation``) that ``tune()``'s path does not
  reach. Covers per-model interaction population (single/multi-model, with and
  without bootstrap) and preservation across a subsequent ``evaluate()`` — the
  vignette_07 ``plot_interactions`` failure mode.
- **GBSG2 baseline/CI contract + bootstrap-warning surfacing.** run → tune
  → calibrate → ``patient_report(ci=True)`` succeeds; the all-bootstrap-failed
  warning surfaces a shared exception when every iteration dies the same way.

Synthetic tune-contract tests share ``make_synthetic_data`` from
``tests/_helpers.py``. This replaces the former per-file ``_make_small_data`` /
``_make_data`` helpers; it is a deliberate switch to a different-but-valid synthetic
dataset, not a byte-for-byte reproduction (binomial events instead of ``randint``,
independent RNG streams for X and y, and no ``+1.0`` time floor). The folded tests
assert on storage-key / CI-band / report-cell / interaction behavior that does not
depend on the exact draw, and all pass under the new data. The GBSG2 cases keep a
local ``_gbsg2_learner`` helper — its categorical encoding is specific to these
tests and absent from ``_helpers``.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import IterativeWarning
from mlsurv.utils import to_structured_array
from tests._helpers import make_synthetic_data

# Tiny RSF forest for the interaction-forwarding tests: their assertions only
# need *a* tree ensemble, and both fit and predict-side cost (permutation
# importance, H-stat partial dependence) scale with n_estimators. Partial
# spaces merge per-param with the defaults, so the unlisted RSF params keep
# their default ranges; only the cost-driving ones are narrowed here.
_SMALL_RSF_SPACE = {'n_estimators': (10, 20), 'max_depth': (3, 5)}


def _gbsg2_learner():
    """GBSG2 learner with categorical columns encoded (local to baseline/CI cases)."""
    from sksurv.datasets import load_gbsg2

    X, y_raw = load_gbsg2()
    X = X.copy()
    X['horTh'] = (X['horTh'] == 'yes').astype(int)
    X['menostat'] = (X['menostat'] == 'Post').astype(int)
    X['tgrade'] = X['tgrade'].map({'I': 1, 'II': 2, 'III': 3}).astype(int)
    y = to_structured_array(y_raw['time'], y_raw['cens'])
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, feature_selector='topk',
                  random_state=42, n_jobs=1)
    return learner, X


# ============================================================================
# Overwrite & no ``_tuned`` suffix
# ============================================================================

@pytest.mark.integration
def test_base_name_equals_name_after_overwrite():
    """``tune()`` overwrites the untuned model at the plain key; ``base_name``
    equals ``name`` and there is no ``_tuned`` suffix.
    """
    X, y = make_synthetic_data(n=80, p=3, event_rate=0.5, time_scale=10.0, seed=0)
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=0)
    learner.run(['coxnet'], tune=False, evaluate=False, verbose=False)
    learner.tune(models='coxnet', n_trials=2, evaluate=False, verbose=False)

    models = learner._state.models
    # tune() overwrites in place: a single 'coxnet' key, no '_tuned' suffix.
    assert set(models) == {'coxnet'}
    assert 'coxnet_tuned' not in models

    # Storage-key invariants
    assert models['coxnet'].name == 'coxnet'
    assert models['coxnet'].is_tuned

    # Algorithm identity: base_name equals name
    assert models['coxnet'].base_name == 'coxnet'


def test_tune_rejects_removed_keep_baseline_kwarg():
    """The removed ``keep_baseline`` kwarg must raise, not be silently accepted.

    Guards against a future ``tune()`` refactor that adds ``**kwargs`` and would
    otherwise swallow ``keep_baseline=True``, silently regressing the removal.
    The unexpected-kwarg ``TypeError`` fires at call binding, before any tuning
    work, so this stays a fast (setup-only) test.
    """
    X, y = make_synthetic_data(n=60, p=3, event_rate=0.5, time_scale=10.0, seed=0)
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=2, random_state=0)
    with pytest.raises(TypeError, match="keep_baseline"):
        learner.tune('coxnet', keep_baseline=True, n_trials=1, verbose=False)


@pytest.mark.integration
def test_predict_ci_works_after_tuning():
    """The original repro: tuned model + predict(ci=True) must produce
    real CI bands, not silently drop them via 100% bootstrap failure.
    """
    X, y = make_synthetic_data(n=80, p=3, event_rate=0.5, time_scale=10.0, seed=0)
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=0)
    learner.run(['coxnet'], tune=False, evaluate=False, verbose=False)
    learner.tune(models='coxnet', n_trials=2, evaluate=False, verbose=False)

    # Sanity: tuned model overwrote the baseline at the plain key.
    assert learner._state.models['coxnet'].is_tuned
    assert 'coxnet_tuned' not in learner._state.models

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', IterativeWarning)
        pred = learner.predict(
            X.iloc[:3], models='coxnet',
            ci=True, n_bootstrap=4, n_jobs=1, verbose=False,
        )

    iter_failures = [
        w for w in caught
        if issubclass(w.category, IterativeWarning)
        and 'Bootstrap iteration failed' in str(w.message)
    ]
    assert iter_failures == [], (
        f"Tuned-model bootstrap should not fail per-iteration; got: "
        f"{[str(w.message) for w in iter_failures]}"
    )

    # CI fields should be populated on the prediction result.
    # The exact attribute layout is in PredictionResult; if any of these
    # is None/empty we have the silent-loss regression back.
    assert pred is not None
    has_ci = any(
        getattr(pred, attr, None) is not None
        for attr in ('ci_lower', 'survival_ci', 'rmst_ci', 'median_ci')
    )
    assert has_ci, (
        "predict(ci=True) on a tuned model must produce CI fields — "
        "all known CI attributes were None."
    )


@pytest.mark.integration
def test_generate_report_works_on_tuned_model(tmp_path):
    """``generate_report()`` succeeds on a tuned model and renders its
    search-space cell from DEFAULT_SPACES (via ``ModelResult.base_name``).
    """
    X, y = make_synthetic_data(n=80, p=3, event_rate=0.5, time_scale=10.0, seed=0)
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=0)
    learner.run(['coxnet'], tune=False, evaluate=True, verbose=False)
    learner.tune(models='coxnet', n_trials=2, evaluate=True, verbose=False)

    assert set(learner._state.models) == {'coxnet'}

    out = tmp_path / "report.html"
    learner.generate_report(output_path=str(out))

    assert out.exists(), "generate_report did not write the HTML"
    body = out.read_text(encoding='utf-8', errors='ignore')
    assert 'coxnet' in body.lower(), \
        "Report missing the coxnet model"
    assert 'tuned' in body.lower(), \
        "Report missing the tuned-model section/text"

    # The tuned model's search-space cell must render the actual
    # parameter ranges from DEFAULT_SPACES, not a bare '-'. A '-' here
    # means base_name lookup silently failed and the cell got the
    # fallback. Verify directly via the helper to avoid HTML parsing.
    from mlsurv.reports.sections.data_preprocessing import DataPreprocessingSection
    section = DataPreprocessingSection(learner)
    tuned_result = learner._state.models['coxnet']
    search_space = section._format_search_space(tuned_result)
    assert search_space != '-', (
        "Search-space cell for the tuned coxnet silently fell through to '-' — "
        "base_name resolution may be broken."
    )
    assert any(token in search_space for token in ('alpha', 'l1_ratio')), (
        f"Search-space cell missing expected coxnet parameters: {search_space!r}"
    )
    # Also confirm the rendered HTML actually carries the search-space content,
    # so a wiring regression (is_tuned detection broken, table row dropped, etc.)
    # would surface even though the helper alone is correct.
    assert search_space in body, (
        "Search-space content rendered by the helper is absent from the report body — "
        "wiring between the section main loop and _format_search_space may be broken."
    )


def test_no_tuned_suffix_string_stripping_anywhere():
    """Grep invariant: the ``_tuned`` storage suffix no longer exists,
    so no module should string-strip it. Catches PRs that re-introduce the
    dual-namespace workaround.
    """
    mlsurv_root = Path(__file__).resolve().parent.parent / 'mlsurv'
    assert mlsurv_root.is_dir(), f"mlsurv package not found at {mlsurv_root}"

    pattern = re.compile(
        r"removesuffix\(['\"]_tuned['\"]\)"
        r"|replace\(['\"]_tuned['\"]"
        r"|split\(['\"]_tuned['\"]"
    )

    offenders = []
    for path in mlsurv_root.rglob('*.py'):
        text = path.read_text(encoding='utf-8')
        if pattern.search(text):
            offenders.append(str(path.relative_to(mlsurv_root)))

    assert not offenders, (
        "The ``_tuned`` storage suffix was removed — do not strip it. "
        "Use ``ModelResult.base_name`` for algorithm identity. Offending "
        "files:\n  - " + "\n  - ".join(offenders)
    )


# ============================================================================
# Interaction-kwarg forwarding through tune()
# ============================================================================

@pytest.mark.integration
def test_tune_evaluate_propagates_include_interactions():
    """tune(include_interactions=True, evaluate=True) populates interactions."""
    X, y = make_synthetic_data(n=80, p=4, event_rate=0.6, time_scale=10.0, seed=0)
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=42, n_jobs=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.tune(
            models='rsf',
            n_trials=2,
            search_spaces=_SMALL_RSF_SPACE,
            evaluate=True,
            include_interactions=True,
            interaction_top_k=3,
            n_importance_repeats=1,
            show_eval_progress=False,
            verbose=False,
            show_progress_bar=False,
        )

    rsf = learner._lookup_model_by_name('rsf')
    assert rsf.is_tuned, "rsf should be marked tuned"
    assert rsf.evaluation is not None, "evaluate=True must populate evaluation"
    inter = rsf.evaluation.interactions
    assert inter is not None, "include_interactions=True must populate interactions"
    assert not inter.empty, "interactions df should have rows for rsf"

    # Plotting must not raise — this was the reported failure mode from the
    # interactions vignette: plot_interactions() after a tuned+evaluated
    # include_interactions run.
    learner.plot_interactions(model='rsf', n=3)


@pytest.mark.integration
def test_tune_propagates_interaction_method():
    """``interaction_method='shap'`` (non-default) must reach evaluate().

    Without this assertion a refactor could drop the field from
    ``TuneConfig`` and the default-``hstat`` test still passes.
    """
    pytest.importorskip('shap')
    X, y = make_synthetic_data(n=80, p=4, event_rate=0.6, time_scale=10.0, seed=0)
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=42, n_jobs=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.tune(
            models='rsf',
            n_trials=2,
            search_spaces=_SMALL_RSF_SPACE,
            evaluate=True,
            include_interactions=True,
            interaction_top_k=3,
            interaction_method='shap',
            n_importance_repeats=1,
            show_eval_progress=False,
            verbose=False,
            show_progress_bar=False,
        )

    rsf = learner._lookup_model_by_name('rsf')
    inter = rsf.evaluation.interactions
    assert inter is not None and not inter.empty
    assert (inter['method'] == 'shap').all(), \
        f"interaction_method='shap' was not forwarded; got {inter['method'].unique()}"


@pytest.mark.integration
def test_tune_default_does_not_compute_interactions():
    """Without include_interactions, interactions stays None — defaults are off."""
    X, y = make_synthetic_data(n=80, p=4, event_rate=0.6, time_scale=10.0, seed=0)
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=42, n_jobs=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Must stay a tree model: linear models skip the interactions phase
        # regardless of the flag, so a default-off guard on coxph/coxnet
        # could never fail even if the default flipped on. The tiny search
        # space keeps the rsf cost down without weakening the guard.
        learner.tune(
            models='rsf',
            n_trials=2,
            search_spaces=_SMALL_RSF_SPACE,
            evaluate=True,
            n_importance_repeats=1,
            show_eval_progress=False,
            verbose=False,
            show_progress_bar=False,
        )

    rsf = learner._lookup_model_by_name('rsf')
    assert rsf.evaluation is not None
    assert rsf.evaluation.interactions is None


# ============================================================================
# run(tune=True) interaction forwarding & preservation
# ============================================================================
# Sibling of the tune() forwarding cases above: these exercise the ``run(tune=True)`` dispatch
# path (``_run_model_comparison`` -> ``_tune_and_train`` -> ``_post_run_evaluation``),
# which the public ``tune()`` entrypoint (``_run_post_tune_evaluation``) does not
# reach. The bug: ``run(models=[...], n_trials>0, include_interactions=True)``
# returned successfully but every per-model ``EvaluationResult.interactions`` was
# ``None``, so ``plot_interactions(...)`` raised ``ValueError("No interaction
# data. ...")`` — the exact vignette_07 cell-29 failure mode.

@pytest.mark.integration
def test_run_tune_propagates_include_interactions_single_model():
    """``run(models=['rsf'], n_trials=2, include_interactions=True)`` populates
    interactions on the per-model ``ModelResult.evaluation``.

    This mirrors the minimal repro — the exact failure mode of
    vignette_07_immunotherapy_case_study.ipynb cell 29 (``plot_interactions``).
    """
    X, y = make_synthetic_data(n=120, p=5, event_rate=0.6, time_scale=10.0, seed=0)
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=42, n_jobs=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.run(
            models=['rsf'],
            n_trials=2,
            search_spaces={'rsf': _SMALL_RSF_SPACE},
            include_interactions=True,
            interaction_top_k=3,
            n_importance_repeats=1,
            bootstrap=False,
            show_eval_progress=False,
            verbose=False,
            show_progress_bar=False,
        )

    rsf = learner._lookup_model_by_name('rsf')
    assert rsf.is_tuned, "rsf should be marked tuned after run(n_trials>0)"
    assert rsf.evaluation is not None, \
        "run(evaluate=True default) must populate per-model evaluation"
    inter = rsf.evaluation.interactions
    assert inter is not None, \
        "run(include_interactions=True) must populate per-model interactions"
    assert not inter.empty, "interactions DataFrame should have rows for rsf"

    # Plotting must not raise — the exact reported failure mode.
    learner.plot_interactions(model='rsf', n=3)


@pytest.mark.integration
def test_run_tune_propagates_include_interactions_multi_model():
    """Multi-model ``run()`` populates interactions on every tree-based model.

    Linear models (coxph) correctly yield an empty interactions slice; only
    non-linear models compute H-statistic by default.
    """
    X, y = make_synthetic_data(n=120, p=5, event_rate=0.6, time_scale=10.0, seed=0)
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=42, n_jobs=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.run(
            models=['coxph', 'rsf'],
            n_trials=2,
            search_spaces={'rsf': _SMALL_RSF_SPACE},
            include_interactions=True,
            interaction_top_k=3,
            n_importance_repeats=1,
            bootstrap=False,
            show_eval_progress=False,
            verbose=False,
            show_progress_bar=False,
        )

    rsf = learner._lookup_model_by_name('rsf')
    assert rsf.evaluation.interactions is not None
    assert not rsf.evaluation.interactions.empty, \
        "RSF interactions should populate under multi-model dispatch"

    # plot_interactions resolves through the same lookup that failed before.
    learner.plot_interactions(model='rsf', n=3)


@pytest.mark.integration
def test_subsequent_evaluate_preserves_interactions():
    """A subsequent ``evaluate()`` without ``include_interactions=True`` must
    NOT silently drop interactions populated by an earlier
    ``run(..., include_interactions=True)``.

    This is the actual root cause — vignette_07 calls
    ``learner.evaluate(verbose=False, show_progress=False)`` after attaching a
    calibrator to refresh metrics, and the fresh evaluation's
    ``interactions=None`` previously overwrote the run()-computed interactions,
    making ``plot_interactions()`` raise ValueError in the next cell.
    """
    X, y = make_synthetic_data(n=120, p=5, event_rate=0.6, time_scale=10.0, seed=0)
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=42, n_jobs=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.run(
            models=['rsf'],
            n_trials=2,
            search_spaces={'rsf': _SMALL_RSF_SPACE},
            include_interactions=True,
            interaction_top_k=3,
            n_importance_repeats=1,
            bootstrap=False,
            show_eval_progress=False,
            verbose=False,
            show_progress_bar=False,
        )

    rsf = learner._lookup_model_by_name('rsf')
    pre = rsf.evaluation.interactions
    assert pre is not None and not pre.empty

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.evaluate(verbose=False, show_progress=False)

    rsf = learner._lookup_model_by_name('rsf')
    post = rsf.evaluation.interactions
    assert post is not None, \
        "evaluate() without include_interactions=True must preserve existing interactions"
    assert not post.empty
    # plot_interactions on the re-evaluated learner — the vig_07 cell 29 path.
    learner.plot_interactions(model='rsf', n=3)


@pytest.mark.integration
def test_subsequent_evaluate_single_model_preserves_interactions():
    """The single-model write-back path in ``_dispatch_evaluate`` must also
    preserve interactions.

    The default ``learner.evaluate()`` call (model=None) resolves through the
    multi-model branch of ``_store_per_model_evaluations``; ``evaluate(model='rsf')``
    instead hits the single-model branch — separate fix call site.
    """
    X, y = make_synthetic_data(n=120, p=5, event_rate=0.6, time_scale=10.0, seed=0)
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=42, n_jobs=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.run(
            models=['rsf'],
            n_trials=2,
            search_spaces={'rsf': _SMALL_RSF_SPACE},
            include_interactions=True,
            interaction_top_k=3,
            n_importance_repeats=1,
            bootstrap=False,
            show_eval_progress=False,
            verbose=False,
            show_progress_bar=False,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.evaluate(models='rsf', verbose=False, show_progress=False)

    rsf = learner._lookup_model_by_name('rsf')
    assert rsf.evaluation.interactions is not None
    assert not rsf.evaluation.interactions.empty


@pytest.mark.integration
def test_explicit_include_interactions_overrides_stale(monkeypatch):
    """When the caller explicitly requests ``include_interactions=True`` and
    the fresh computation legitimately returns no interactions (e.g. SHAP on
    a non-tree model, <2 features after group collapse, or a SHAP exception),
    preservation must NOT paste stale interactions over the fresh empty result.

    Without intent-based gating, the user would see stale interactions from a
    prior run after explicitly requesting fresh ones — a silent data-correctness
    failure of the opposite kind from the preservation case itself.
    """
    import sys
    import mlsurv.evaluation.evaluate  # noqa: F401 — populate sys.modules
    # __init__.py re-exports the evaluate() function as mlsurv.evaluation.evaluate,
    # shadowing the submodule attribute on the parent package. Grab the module
    # via sys.modules to monkeypatch the private helper.
    evaluate_mod = sys.modules['mlsurv.evaluation.evaluate']

    X, y = make_synthetic_data(n=120, p=5, event_rate=0.6, time_scale=10.0, seed=0)
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=42, n_jobs=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.run(
            models=['rsf'],
            n_trials=2,
            search_spaces={'rsf': _SMALL_RSF_SPACE},
            include_interactions=True,
            interaction_top_k=3,
            n_importance_repeats=1,
            bootstrap=False,
            show_eval_progress=False,
            verbose=False,
            show_progress_bar=False,
        )

    rsf = learner._lookup_model_by_name('rsf')
    stale = rsf.evaluation.interactions
    assert stale is not None and not stale.empty

    # Force the interactions phase to return _NONE_RESULT — mimics any of the
    # legitimate empty paths (linear model + hstat, SHAP on non-tree, <2 features
    # after categorical-group collapse, SHAP exception). The user asked for fresh
    # interactions explicitly; the preservation helper must not fire.
    _NONE = {
        'interactions': None,
        'interaction_surfaces': None,
        'interaction_values_raw': None,
        'interaction_feature_names': None,
        'interaction_method': None,
    }
    monkeypatch.setattr(
        evaluate_mod, '_run_interactions_phase', lambda *a, **kw: _NONE,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.evaluate(
            include_interactions=True,
            interaction_top_k=3,
            n_importance_repeats=1,
            verbose=False,
            show_progress=False,
        )

    rsf = learner._lookup_model_by_name('rsf')
    assert rsf.evaluation.interactions is None, (
        "Explicit include_interactions=True must not preserve stale data when "
        "the fresh evaluate legitimately returns None."
    )


@pytest.mark.integration
def test_run_tune_propagates_include_interactions_with_bootstrap():
    """The vig_07 path uses ``bootstrap=True`` (default) — ensure the bootstrap
    branch through ``_post_run_evaluation`` does not drop interactions.
    """
    X, y = make_synthetic_data(n=120, p=5, event_rate=0.6, time_scale=10.0, seed=0)
    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, cv_folds=3, random_state=42, n_jobs=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.run(
            models=['rsf'],
            n_trials=2,
            search_spaces={'rsf': _SMALL_RSF_SPACE},
            include_interactions=True,
            interaction_top_k=3,
            n_importance_repeats=1,
            bootstrap=True,
            n_bootstrap=2,
            show_eval_progress=False,
            verbose=False,
            show_progress_bar=False,
        )

    rsf = learner._lookup_model_by_name('rsf')
    assert rsf.evaluation.interactions is not None
    assert not rsf.evaluation.interactions.empty
    learner.plot_interactions(model='rsf', n=3)


# ============================================================================
# GBSG2 baseline/CI contract + bootstrap-warning surfacing
# ============================================================================

@pytest.mark.integration
def test_patient_report_ci_after_tune_calibrate(tmp_path):
    """Exact repro: run -> tune -> calibrate -> patient_report(ci=True)."""
    learner, X = _gbsg2_learner()
    learner.run(['coxnet'], tune=False, verbose=False)
    learner.tune('coxnet', n_trials=2, verbose=False)
    assert 'coxnet' in learner.models
    assert 'coxnet_tuned' not in learner.models
    learner._attach_calibrators(learner.models['coxnet'], 'isotonic')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = learner.patient_report(
            X.iloc[:5], patient=0, model='coxnet',
            output_path=str(tmp_path / 'p.html'),
            use_recalibrated=True, ci=True, n_bootstrap=4,
        )

    msgs = [str(w.message).lower() for w in caught]
    assert not any('bootstrap iteration failed' in m for m in msgs), msgs
    assert not any('all bootstrap iterations failed' in m for m in msgs), msgs
    assert not any('unknown model' in m for m in msgs), msgs
    assert result is not None


@pytest.mark.integration
def test_bootstrap_warning_surfaces_shared_exception(tmp_path, monkeypatch):
    """When every iteration dies the same way, the MethodologicalWarning body
    surfaces that exact exception instead of generic 'data instability' copy."""
    from mlsurv.evaluation import _prediction_bootstrap as pb
    from mlsurv._iter_warn import IterativeFailure
    from mlsurv._warnings import MethodologicalWarning

    learner, X = _gbsg2_learner()
    learner.run(['coxnet'], tune=False, verbose=False)
    learner.evaluate('coxnet', n_importance_repeats=1, show_progress=False)

    def _always_fail(*args, **kwargs):
        return {'__failure__': IterativeFailure(
            exc_type_name='ValueError',
            message='deterministic boom',
            context_prefix=f"seed={kwargs.get('seed')}",
        )}

    monkeypatch.setattr(pb, '_prediction_bootstrap_iteration', _always_fail)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        learner.predict(X.iloc[:3], models='coxnet', ci=True, n_bootstrap=4)

    methodological = [
        str(w.message) for w in caught
        if issubclass(w.category, MethodologicalWarning)
    ]
    surfaced = [m for m in methodological
                if 'ValueError' in m and 'deterministic boom' in m]
    assert surfaced, f"expected surfaced exception in: {methodological}"


@pytest.mark.integration
def test_bootstrap_warning_generic_when_failures_differ(tmp_path, monkeypatch):
    """When iterations fail with *different* exceptions, the all-failed warning
    keeps the generic 'Inspect the IterativeWarning' copy (not a single
    surfaced exception)."""
    from mlsurv.evaluation import _prediction_bootstrap as pb
    from mlsurv._iter_warn import IterativeFailure
    from mlsurv._warnings import MethodologicalWarning

    learner, X = _gbsg2_learner()
    learner.run(['coxnet'], tune=False, verbose=False)
    learner.evaluate('coxnet', n_importance_repeats=1, show_progress=False)

    calls = {'n': 0}

    def _fail_differently(*args, **kwargs):
        calls['n'] += 1
        etype = 'ValueError' if calls['n'] % 2 else 'RuntimeError'
        return {'__failure__': IterativeFailure(
            exc_type_name=etype, message=f"boom {calls['n']}",
            context_prefix=f"seed={kwargs.get('seed')}",
        )}

    monkeypatch.setattr(pb, '_prediction_bootstrap_iteration', _fail_differently)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        learner.predict(X.iloc[:3], models='coxnet', ci=True, n_bootstrap=4)

    methodological = [
        str(w.message) for w in caught
        if issubclass(w.category, MethodologicalWarning)
        and 'bootstrap iteration' in str(w.message).lower()
    ]
    assert any('Inspect the IterativeWarning' in m for m in methodological), (
        f"expected generic copy in: {methodological}"
    )
