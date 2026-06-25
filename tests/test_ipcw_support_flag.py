"""Safe IPCW tau default, MethodologicalWarning, report flag.

Extends: validate() defaults Uno IPCW G_hat to the validation
cohort's own censoring distribution, not training's.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import MethodologicalWarning

from tests._helpers import make_survival_array, make_y


def _make_external_validation_data(seed=42):
    """Training cohort with shorter follow-up than the test cohort.

    Simulates the Rotterdam (test) evaluated against GBSG2 (train) scenario
    where test events/follow-up extend beyond training t_max,
    so Uno's IPCW weights explode near the tail.
    """
    rng = np.random.default_rng(seed)
    n = 120

    X_train = pd.DataFrame({
        'x1': rng.normal(size=n),
        'x2': rng.normal(size=n),
        'x3': rng.normal(size=n),
    })
    y_train = make_survival_array(
        rng.binomial(1, 0.4, size=n),
        rng.uniform(0.1, 10.0, size=n),
    )

    X_test = pd.DataFrame({
        'x1': rng.normal(size=n),
        'x2': rng.normal(size=n),
        'x3': rng.normal(size=n),
    })
    # Test follow-up extends past training t_max (~10) out to 25.
    y_test = make_survival_array(
        rng.binomial(1, 0.4, size=n),
        rng.uniform(0.1, 25.0, size=n),
    )
    return X_train, y_train, X_test, y_test


@pytest.mark.integration
def test_evaluate_emits_methodological_warning_on_support_violation():
    """evaluate() warns once via MethodologicalWarning when test follow-up
    exceeds training support."""
    X_train, y_train, X_test, y_test = _make_external_validation_data()
    learner = SurvivalLearner(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    )
    learner.setup(random_state=42)
    learner.train('coxph')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = learner.evaluate(X=X_test, y=y_test, verbose=False)

    msgs = [str(w.message) for w in caught if issubclass(w.category, MethodologicalWarning)]
    assert any("IPCW" in m and "tau" in m for m in msgs), \
        f"Expected IPCW tau warning, got: {msgs}"

    # Flag is attached to the evaluation result.
    flag = getattr(result, 'ipcw_support_flag', None)
    assert flag is not None
    assert flag['n_beyond'] > 0


@pytest.mark.integration
def test_evaluate_silent_without_violation():
    """No IPCW support warning when test follow-up is within training support."""
    rng = np.random.default_rng(7)
    n = 120
    X = pd.DataFrame({'x1': rng.normal(size=n), 'x2': rng.normal(size=n)})
    y = make_survival_array(
        rng.binomial(1, 0.4, size=n),
        rng.uniform(0.1, 10.0, size=n),
    )
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42)
    learner.train('coxph')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = learner.evaluate(verbose=False)

    msgs = [str(w.message) for w in caught if issubclass(w.category, MethodologicalWarning)]
    assert not any("IPCW" in m and "tau" in m for m in msgs), \
        f"Unexpected IPCW tau warning: {msgs}"
    assert getattr(result, 'ipcw_support_flag', None) is None


@pytest.mark.integration
def test_evaluate_clears_stale_ipcw_flag():
    """A subsequent in-support evaluate() must clear a stale flag from a prior
    violating call (review finding)."""
    X_train, y_train, X_test_violating, y_test_violating = _make_external_validation_data()
    rng = np.random.default_rng(3)
    n = 120
    X_test_safe = pd.DataFrame({
        'x1': rng.normal(size=n), 'x2': rng.normal(size=n), 'x3': rng.normal(size=n),
    })
    y_test_safe = make_survival_array(
        rng.binomial(1, 0.4, size=n),
        rng.uniform(0.1, 5.0, size=n),  # strictly within training support
    )

    learner = SurvivalLearner(
        X_train=X_train, X_test=X_test_violating, y_train=y_train, y_test=y_test_violating,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(random_state=42)
        learner.train('coxph')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.evaluate(X=X_test_violating, y=y_test_violating, verbose=False)
    assert learner._state.ipcw_support_flag is not None

    # Follow up with a safe cohort — flag must be cleared on state.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.evaluate(X=X_test_safe, y=y_test_safe, verbose=False)
    assert learner._state.ipcw_support_flag is None


@pytest.mark.integration
def test_report_caveat_does_not_resurrect_stale_flag():
    """A clean evaluate() after a violating one must clear the report caveat
    even when an old ModelResult still carries the stale flag attribute
    (round-2 review)."""
    from mlsurv.reports.sections.limitations import LimitationsSection

    X_train, y_train, X_test_violating, y_test_violating = _make_external_validation_data()
    rng = np.random.default_rng(11)
    n = 120
    X_test_safe = pd.DataFrame({
        'x1': rng.normal(size=n), 'x2': rng.normal(size=n), 'x3': rng.normal(size=n),
    })
    y_test_safe = make_survival_array(
        rng.binomial(1, 0.4, size=n),
        rng.uniform(0.1, 5.0, size=n),
    )

    learner = SurvivalLearner(
        X_train=X_train, X_test=X_test_violating, y_train=y_train, y_test=y_test_violating,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(random_state=42)
        learner.train('coxph')
        learner.evaluate(X=X_test_violating, y=y_test_violating, verbose=False)
        # Even though the prior ModelResult.evaluation may still carry the
        # flag attribute, the report must read state (which is now None).
        learner.evaluate(X=X_test_safe, y=y_test_safe, verbose=False)

    section = LimitationsSection(learner)
    result = section.generate()
    assert "IPCW Training-Support Violation" not in result["markdown"]


@pytest.mark.integration
def test_report_flag_prose_lists_all_ipcw_metrics():
    """When the IPCW support flag triggers, the prose must name
    AUC(t), Brier, and IBS in addition to Uno's C-index, since all four
    metrics are weighted by 1/G_hat and equally affected by support violations.
    """
    from mlsurv.reports.sections.limitations import LimitationsSection

    X_train, y_train, X_test, y_test = _make_external_validation_data()
    learner = SurvivalLearner(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(random_state=42)
        learner.train('coxph')
        learner.evaluate(X=X_test, y=y_test, verbose=False)

    section = LimitationsSection(learner)
    result = section.generate()
    md = result["markdown"]

    # Renamed flag header (no longer "Uno IPCW...").
    assert "IPCW Training-Support Violation" in md
    assert "Uno IPCW Training-Support Violation" not in md

    # All four IPCW-weighted metrics must be named in the broadened prose.
    for metric in ("Uno", "AUC(t)", "Brier", "IBS"):
        assert metric in md, f"Expected {metric!r} in flag prose, got:\n{md}"


def test_performance_caveat_banner_lists_all_ipcw_metrics():
    """The companion banner in the performance subsection
    (`_add_ipcw_support_caveat`) must use the renamed flag and list all
    four IPCW-weighted metrics in its caveat text.

    Uses a stub state instead of training a real learner — the helper only
    reads `self._s.learner._state.ipcw_support_flag`, so we can exercise it
    directly to keep this test in the fast tier.
    """
    from types import SimpleNamespace

    from mlsurv.reports.sections.evaluation._performance import _PerformanceSubsection

    flag = {'tau_used': 12.0, 'g_hat_source': 'test', 'n_beyond': 5, 'min_g_hat': 0.04}
    stub_section = SimpleNamespace(
        learner=SimpleNamespace(_state=SimpleNamespace(ipcw_support_flag=flag)),
    )
    sub = _PerformanceSubsection.__new__(_PerformanceSubsection)
    sub._s = stub_section

    md_lines: list[str] = []
    sub._add_ipcw_support_caveat(md_lines)
    banner = "\n".join(md_lines)

    assert "IPCW caveat" in banner
    assert "Uno IPCW caveat" not in banner
    # Cross-reference anchor matches the renamed limitations flag.
    assert "IPCW Training-Support Violation" in banner
    assert "Uno IPCW Training-Support Violation" not in banner
    # All four IPCW-weighted metrics named in the banner body.
    for metric in ("Uno", "AUC(t)", "Brier", "IBS"):
        assert metric in banner, f"Expected {metric!r} in banner:\n{banner}"


@pytest.mark.integration
@pytest.mark.bootstrap_heavy
def test_bootstrap_ipcw_tau_reduces_ci_width():
    """With the safe tau default, Uno's bootstrap CI width is no longer
    orders of magnitude wider than Harrell's."""
    X_train, y_train, X_test, y_test = _make_external_validation_data()
    learner = SurvivalLearner(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    )
    learner.setup(random_state=42)
    learner.train('coxph')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = learner.bootstrap(
            X=X_test, y=y_test, n_bootstrap=20, metrics=['harrell', 'ipcw'],
            random_state=0, verbose=False, show_progress=False,
        )

    summary = result.summary
    h = summary[summary['metric'] == 'harrell'].iloc[0]
    u = summary[summary['metric'] == 'ipcw'].iloc[0]
    harrell_width = float(h['ci_upper']) - float(h['ci_lower'])
    uno_width = float(u['ci_upper']) - float(u['ci_lower'])
    # With the safe tau default, Uno's CI should be within ~3x Harrell's
    # (pre-fix it was ~13x on the vignette-2 external validation).
    assert uno_width < max(0.2, 3.0 * harrell_width), \
        f"Uno CI width {uno_width:.3f} still dominates Harrell {harrell_width:.3f}"


# ---------------------------------------------------------------------------
# validate() defaults to validation-cohort G_hat
# ---------------------------------------------------------------------------


def _train_learner_for_validate(seed=42):
    """Build a trained learner whose y_train sits inside a smaller follow-up
    window than the validation cohort it'll be evaluated against."""
    X_train, y_train, X_val, y_val = _make_external_validation_data(seed=seed)
    learner = SurvivalLearner(X_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(random_state=seed)
        learner.train('coxph')
    return learner, X_val, y_val


def test_resolve_ipcw_g_source_dispatch():
    """Helper returns the right cohort and rejects unknown sources."""
    from mlsurv.learner._evaluation import _resolve_ipcw_g_source

    yt = make_survival_array([True], [1.0])
    ye = make_survival_array([False], [2.0])
    assert _resolve_ipcw_g_source('train', yt, ye) is yt
    assert _resolve_ipcw_g_source('test', yt, ye) is ye
    with pytest.raises(ValueError, match="ipcw_g_hat_source"):
        _resolve_ipcw_g_source('other', yt, ye)


@pytest.mark.integration
def test_evaluate_default_g_hat_source_is_train():
    """evaluate() default behavior unchanged (back-compat): when a
    support violation fires, the flag records g_hat_source='train'."""
    learner, X_val, y_val = _train_learner_for_validate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = learner.evaluate(X=X_val, y=y_val, verbose=False)
    flag = getattr(result, 'ipcw_support_flag', None)
    assert flag is not None
    assert flag.get('g_hat_source') == 'train'


@pytest.mark.integration
def test_validate_defaults_to_test_g_hat():
    """validate() defaults Uno IPCW G_hat to the validation cohort.
    With test-G, the support violation against training t_max no longer fires."""
    learner, X_val, y_val = _train_learner_for_validate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.validate(X_val, y_val, name="ext", n_bootstrap=4, verbose=False,
                         show_progress=False)
    flag = learner._state.ipcw_support_flag
    # Either no violation (because G is well-defined on its own cohort) OR
    # a g_hat_floor violation only — but never the n_beyond branch with
    # g_source='train'. Whichever case, when the flag exists it must say 'test'.
    if flag is not None:
        assert flag.get('g_hat_source') == 'test'
        assert flag.get('n_beyond', 0) == 0


@pytest.mark.integration
def test_validate_explicit_train_source_reproduces_legacy():
    """Opting back into training G_hat reproduces the legacy behavior:
    n_beyond > 0 and g_hat_source recorded as 'train'."""
    learner, X_val, y_val = _train_learner_for_validate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.validate(
            X_val, y_val, name="ext_legacy",
            n_bootstrap=4, verbose=False, show_progress=False,
            ipcw_g_hat_source='train',
        )
    flag = learner._state.ipcw_support_flag
    assert flag is not None
    assert flag.get('g_hat_source') == 'train'
    assert flag.get('n_beyond', 0) > 0


@pytest.mark.integration
def test_composition_warning_follows_train_source_for_evaluate():
    """evaluate() checks the training split: an all-events training cohort
    cannot estimate G_hat, so the composition MethodologicalWarning fires and
    the flag records the training-side counts and g_hat_source='train'."""
    rng = np.random.default_rng(11)
    n = 60
    cols = lambda: {  # noqa: E731
        'x1': rng.normal(size=n),
        'x2': rng.normal(size=n),
        'x3': rng.normal(size=n),
    }
    X_train = pd.DataFrame(cols())
    # Training cohort is ALL events -> G_hat never steps down (not estimable).
    y_train = make_survival_array(
        np.ones(n, dtype=int), rng.uniform(0.5, 9.5, size=n),
    )
    X_test = pd.DataFrame(cols())
    # Test strictly within training support so no n_beyond overhang.
    y_test = make_survival_array(
        rng.binomial(1, 0.5, size=n), rng.uniform(0.5, 8.0, size=n),
    )
    learner = SurvivalLearner(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(random_state=11)
        learner.train('coxph')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = learner.evaluate(X=X_test, y=y_test, verbose=False)

    msgs = [str(w.message) for w in caught
            if issubclass(w.category, MethodologicalWarning)]
    assert any("too few to estimate" in m for m in msgs), \
        f"Expected composition warning, got: {msgs}"
    flag = getattr(result, 'ipcw_support_flag', None)
    assert flag is not None
    assert flag['n_g_censored'] == 0
    assert flag['g_hat_source'] == 'train'


@pytest.mark.integration
def test_composition_warning_follows_test_source_for_validate():
    """validate() fits G_hat on the validation cohort, so the composition
    check follows the validation cohort: an all-events validation set fires
    the warning keyed on the test side with g_hat_source='test'."""
    rng = np.random.default_rng(13)
    n = 120
    X_train = pd.DataFrame({
        'x1': rng.normal(size=n),
        'x2': rng.normal(size=n),
        'x3': rng.normal(size=n),
    })
    # Healthy training cohort: ample events and censored.
    y_train = make_survival_array(
        rng.binomial(1, 0.5, size=n), rng.uniform(0.5, 12.0, size=n),
    )
    learner = SurvivalLearner(X_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(random_state=13)
        learner.train('coxph')

    m = 60
    X_val = pd.DataFrame({
        'x1': rng.normal(size=m),
        'x2': rng.normal(size=m),
        'x3': rng.normal(size=m),
    })
    # Validation cohort is ALL events -> G_hat not estimable on it.
    y_val = make_survival_array(
        np.ones(m, dtype=int), rng.uniform(0.5, 8.0, size=m),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        learner.validate(
            X_val, y_val, name="ext_thin",
            n_bootstrap=4, verbose=False, show_progress=False,
        )

    msgs = [str(w.message) for w in caught
            if issubclass(w.category, MethodologicalWarning)]
    assert any("too few to estimate" in m for m in msgs), \
        f"Expected composition warning, got: {msgs}"
    flag = learner._state.ipcw_support_flag
    assert flag is not None
    assert flag.get('g_hat_source') == 'test'
    assert flag.get('n_g_censored') == 0


@pytest.mark.integration
def test_healthy_cohort_no_composition_warning():
    """A cohort with ample events and censored on both sides does not fire the
    composition warning."""
    rng = np.random.default_rng(17)
    n = 150
    X = pd.DataFrame({
        'x1': rng.normal(size=n),
        'x2': rng.normal(size=n),
        'x3': rng.normal(size=n),
    })
    y = make_survival_array(
        rng.binomial(1, 0.5, size=n), rng.uniform(0.5, 10.0, size=n),
    )
    learner = SurvivalLearner(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(random_state=17)
        learner.train('coxph')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = learner.evaluate(verbose=False)

    msgs = [str(w.message) for w in caught
            if issubclass(w.category, MethodologicalWarning)]
    assert not any("too few to estimate" in m for m in msgs), \
        f"Unexpected composition warning: {msgs}"
    # Flag may still be None or carry only an instability/overhang violation,
    # but never the composition counts below the floor.
    flag = getattr(result, 'ipcw_support_flag', None)
    if flag is not None:
        n_ev = flag.get('n_g_events')
        n_cen = flag.get('n_g_censored')
        if n_ev is not None and n_cen is not None:
            assert min(n_ev, n_cen) >= 10


def _render_perf_caveat(flag):
    """Drive the performance-table IPCW caveat with a hand-built flag.

    ``_add_ipcw_support_caveat`` only reads
    ``self._s.learner._state.ipcw_support_flag`` and appends to ``md_lines``,
    so a duck-typed stub exercises the composition-vs-instability branch
    without building a full report.
    """
    import types

    from mlsurv.reports.sections.evaluation._performance import (
        _PerformanceSubsection,
    )

    stub = types.SimpleNamespace(
        _s=types.SimpleNamespace(
            learner=types.SimpleNamespace(
                _state=types.SimpleNamespace(ipcw_support_flag=flag),
            ),
        ),
    )
    md_lines = []
    _PerformanceSubsection._add_ipcw_support_caveat(stub, md_lines)
    return "\n".join(md_lines)


def test_perf_caveat_composition_uses_correction_off_wording():
    """Composition trigger -> caveat says the IPCW correction is effectively
    off, not the 'wide CIs = instability' framing."""
    html = _render_perf_caveat({
        "tau_used": 20.0,
        "g_hat_source": "train",
        "n_g_events": 40,
        "n_g_censored": 0,
        "min_ipcw_support": 10,
    })
    assert "effectively off" in html
    assert "reduce toward their unweighted forms" in html
    assert "weight instability" not in html


def test_perf_caveat_instability_uses_instability_wording():
    """Healthy-composition cohort -> caveat keeps the instability framing."""
    html = _render_perf_caveat({
        "tau_used": 8.0,
        "g_hat_source": "train",
        "n_g_events": 40,
        "n_g_censored": 40,
        "min_ipcw_support": 10,
    })
    assert "weight instability" in html
    assert "effectively off" not in html


@pytest.mark.integration
def test_validate_test_g_hat_closes_uno_harrell_gap():
    """test-cohort G_hat removes the bias in Uno's IPCW
    point estimate when train and test censoring distributions differ."""
    rng = np.random.default_rng(0)
    n_train, n_val = 300, 300
    X_train = pd.DataFrame({'x1': rng.normal(size=n_train), 'x2': rng.normal(size=n_train)})
    risk_train = X_train['x1'].to_numpy() + 0.5 * X_train['x2'].to_numpy()
    t_train = rng.exponential(scale=np.exp(-risk_train), size=n_train) * 5
    c_train = rng.uniform(0.5, 4.0, size=n_train)  # heavy censoring early
    obs_t = np.minimum(t_train, c_train)
    ev_t = (t_train <= c_train)
    y_train = make_survival_array(ev_t, obs_t)

    X_val = pd.DataFrame({'x1': rng.normal(size=n_val), 'x2': rng.normal(size=n_val)})
    risk_val = X_val['x1'].to_numpy() + 0.5 * X_val['x2'].to_numpy()
    t_val = rng.exponential(scale=np.exp(-risk_val), size=n_val) * 5
    c_val = rng.uniform(8.0, 20.0, size=n_val)  # validation barely censors
    obs_v = np.minimum(t_val, c_val)
    ev_v = (t_val <= c_val)
    y_val = make_survival_array(ev_v, obs_v)

    learner = SurvivalLearner(X_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(random_state=0)
        learner.train('coxph')

        # Default: validation G_hat — Uno should track Harrell.
        learner.validate(X_val, y_val, name="default", n_bootstrap=4,
                         verbose=False, show_progress=False)
        ev_default = learner._state.validation_results['default'].evaluation
        h_default = float(ev_default.concordance.iloc[0]['c_index_test'])
        u_default = float(ev_default.concordance.iloc[0]['c_index_ipcw_test'])

        # Legacy: training G_hat — Uno biased downward.
        learner.validate(X_val, y_val, name="legacy", n_bootstrap=4,
                         verbose=False, show_progress=False,
                         ipcw_g_hat_source='train')
        ev_legacy = learner._state.validation_results['legacy'].evaluation
        u_legacy = float(ev_legacy.concordance.iloc[0]['c_index_ipcw_test'])

    # With test-G, Uno should be close to Harrell.
    assert abs(h_default - u_default) < 0.05, \
        f"test-G Uno ({u_default:.3f}) should track Harrell ({h_default:.3f})"
    # With train-G the gap should be larger (the bug we're fixing).
    assert abs(h_default - u_legacy) > abs(h_default - u_default), \
        f"train-G Uno {u_legacy:.3f} should be further from Harrell than test-G {u_default:.3f}"


@pytest.mark.integration
def test_report_caveat_names_g_source():
    """When validate() fires a support flag, the report caveat names the
    validation-cohort G_hat (and limitations section says the same)."""
    from mlsurv.reports.sections.limitations import LimitationsSection

    # Force a g_hat floor violation by making validation events span a long
    # tail relative to validation censoring (so KM(c) drops well below 0.05).
    rng = np.random.default_rng(2)
    n_train, n_val = 80, 80
    X_train = pd.DataFrame({'x1': rng.normal(size=n_train), 'x2': rng.normal(size=n_train)})
    y_train = make_survival_array(
        rng.binomial(1, 0.5, size=n_train),
        rng.uniform(0.1, 8.0, size=n_train),
    )
    X_val = pd.DataFrame({'x1': rng.normal(size=n_val), 'x2': rng.normal(size=n_val)})
    # Validation: nearly all censored, with one big event time at the tail.
    times = np.r_[rng.uniform(0.1, 2.0, size=n_val - 1), [50.0]]
    events = np.r_[np.zeros(n_val - 1, dtype=bool), [True]]
    y_val = make_survival_array(events, times)

    learner = SurvivalLearner(X_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(random_state=2)
        learner.train('coxph')
        learner.validate(X_val, y_val, name="ext", n_bootstrap=4,
                         verbose=False, show_progress=False)

    flag = learner._state.ipcw_support_flag
    if flag is None:
        pytest.skip("This synthetic setup did not trigger a g_hat floor violation.")

    section = LimitationsSection(learner)
    out = section.generate()
    md = out["markdown"]
    assert "IPCW Training-Support Violation" in md
    assert "validation-cohort" in md


@pytest.mark.integration
def test_evaluate_with_subpopulations_does_not_crash():
    """Regression: evaluate() with stored subpopulations must not raise
    TypeError on the new ipcw_y_train threading path (review finding).

    The auto-subpopulation branch in evaluate() forwards the resolved IPCW
    cohort to _compute_and_store_subpopulation_metrics; a kwarg-name
    mismatch there crashes ordinary evaluate() callers, not just validate().
    """
    rng = np.random.default_rng(13)
    n = 100
    X = pd.DataFrame({'x1': rng.normal(size=n), 'x2': rng.normal(size=n)})
    y = make_survival_array(
        rng.binomial(1, 0.5, size=n),
        rng.uniform(0.5, 6.0, size=n),
    )
    groups = pd.Series(rng.choice(['A', 'B'], size=n), name='grp')

    learner = SurvivalLearner(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(random_state=13, subpopulations={'grp': groups})
        learner.train('coxph')
        # Default ipcw_g_hat_source='train' must reach the subpop helper.
        learner.evaluate(verbose=False)


@pytest.mark.integration
def test_bootstrap_direct_call_with_test_g_source():
    """Direct learner.bootstrap(ipcw_g_hat_source='test') threads ipcw_y_train
    end-to-end (validate() reaches bootstrap iterations only via evaluate())."""
    X_train, y_train, X_val, y_val = _make_external_validation_data(seed=21)
    learner = SurvivalLearner(X_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.setup(random_state=21)
        learner.train('coxph')
        result = learner.bootstrap(
            X=X_val, y=y_val,
            n_bootstrap=4, metrics=['harrell', 'ipcw'],
            ipcw_g_hat_source='test',
            verbose=False, show_progress=False,
        )

    flag = learner._state.ipcw_support_flag
    if flag is not None:
        assert flag.get('g_hat_source') == 'test'
        assert flag.get('n_beyond', 0) == 0  # cohort cannot extend past itself

    s = result.summary
    u = s[s['metric'] == 'ipcw']
    assert not u.empty and not np.isnan(float(u.iloc[0]['point']))


# ---------------------------------------------------------------------------
# Gating of "may be unstable" warning by materiality
# ---------------------------------------------------------------------------


class TestMaterialityGating:
    """`_resolve_ipcw_tau` should only emit its "may be unstable" warning
    for material support violations: either real numerical instability
    (min_g_hat < 0.05) or a non-trivial overhang (>1% of test cohort).
    Trivial overhangs are covered by the consolidated truncation notice.
    """

    def _y(self, times, events=None):
        return make_y(times, events)

    def test_trivial_overhang_skips_unstable_warning(self):
        from mlsurv.learner._evaluation_helpers import _resolve_ipcw_tau

        # 1 obs in 200 test → 0.5% < 1% threshold.
        rng = np.random.default_rng(0)
        y_train = self._y(rng.uniform(1.0, 10.0, 100))
        y_test = self._y(np.concatenate([
            rng.uniform(1.0, 9.5, 199),
            np.array([15.0]),
        ]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tau, flag = _resolve_ipcw_tau(y_train, y_test, "auto")
        unstable = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "may be unstable" in str(w.message)
        ]
        assert unstable == [], (
            "Trivial overhang must not trigger the 'may be unstable' "
            "warning — the consolidated truncation notice covers it."
        )
        # Flag itself is still produced (downstream report still shows it).
        assert flag is not None
        assert flag["n_beyond"] == 1

    def test_material_overhang_still_warns(self):
        from mlsurv.learner._evaluation_helpers import _resolve_ipcw_tau

        # 5 obs in 30 test → 16.7% > 1% threshold.
        rng = np.random.default_rng(0)
        y_train = self._y(rng.uniform(1.0, 10.0, 100))
        y_test = self._y(np.concatenate([
            rng.uniform(1.0, 9.5, 25),
            rng.uniform(15.0, 25.0, 5),
        ]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _resolve_ipcw_tau(y_train, y_test, "auto")
        unstable = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "may be unstable" in str(w.message)
        ]
        assert len(unstable) == 1
        # Material overhang must surface the "extend beyond" reason clause.
        assert "extend beyond training t_max" in str(unstable[0].message)

    @pytest.mark.parametrize(
        "n_beyond, n_test, should_warn",
        [
            (1, 100, False),  # exactly 1.0% — strict `>` means suppressed
            (2, 100, True),   # 2.0% — fires
            (1, 99, True),    # 1.01% — fires (just over the strict threshold)
        ],
    )
    def test_one_percent_boundary(self, n_beyond, n_test, should_warn, monkeypatch):
        """The materiality gate uses strict `>` 0.01. Lock down behavior at
        and just past the boundary so a future change to `>=` is caught.
        """
        from mlsurv.learner import _evaluation_helpers as h

        # Hand-craft the support flag so we test the gate, not check_ipcw_support.
        def fake_flag(y_train, y_test, tau=None, min_ipcw_support=10):
            return {
                "n_beyond": n_beyond,
                "min_g_hat": 0.5,  # well above the 0.05 threshold
                "t_max_train": 10.0,
            }

        monkeypatch.setattr(h, "check_ipcw_support", fake_flag)
        monkeypatch.setattr(h, "compute_safe_ipcw_tau", lambda y_train, y_test: 8.0)

        y_train = self._y(np.linspace(1, 10, 50))
        y_test = self._y(np.linspace(1, 9, n_test))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            h._resolve_ipcw_tau(y_train, y_test, "auto")
        unstable = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "may be unstable" in str(w.message)
        ]
        if should_warn:
            assert len(unstable) == 1
        else:
            assert unstable == []

    def test_g_hat_instability_without_overhang(self, monkeypatch):
        """`min_g_hat < 0.05` alone (no overhang) must still trigger the
        "may be unstable" warning. Guards against a refactor that
        accidentally couples `g_unstable` to `n_beyond > 0`.
        """
        from mlsurv.learner import _evaluation_helpers as h

        def fake_flag(y_train, y_test, tau=None, min_ipcw_support=10):
            return {
                "n_beyond": 0,
                "min_g_hat": 0.02,  # < 0.05 → instability
                "t_max_train": 10.0,
            }

        monkeypatch.setattr(h, "check_ipcw_support", fake_flag)
        monkeypatch.setattr(h, "compute_safe_ipcw_tau", lambda y_train, y_test: 8.0)

        y_train = self._y(np.linspace(1, 10, 50))
        y_test = self._y(np.linspace(1, 9, 100))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            h._resolve_ipcw_tau(y_train, y_test, "auto")
        unstable = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "may be unstable" in str(w.message)
        ]
        assert len(unstable) == 1
        msg = str(unstable[0].message)
        assert "min(G_hat)" in msg
        # No overhang → no "extend beyond" clause.
        assert "extend beyond" not in msg

    def test_min_g_hat_exactly_threshold_no_warning(self, monkeypatch):
        """Boundary: `min_g_hat == 0.05` is NOT unstable (strict `<`)."""
        from mlsurv.learner import _evaluation_helpers as h

        def fake_flag(y_train, y_test, tau=None, min_ipcw_support=10):
            return {"n_beyond": 0, "min_g_hat": 0.05, "t_max_train": 10.0}

        monkeypatch.setattr(h, "check_ipcw_support", fake_flag)
        monkeypatch.setattr(h, "compute_safe_ipcw_tau", lambda y_train, y_test: 8.0)

        y_train = self._y(np.linspace(1, 10, 50))
        y_test = self._y(np.linspace(1, 9, 100))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            h._resolve_ipcw_tau(y_train, y_test, "auto")
        unstable = [
            w for w in caught
            if issubclass(w.category, MethodologicalWarning)
            and "may be unstable" in str(w.message)
        ]
        assert unstable == []
