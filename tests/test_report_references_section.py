"""Tests for ReferencesSection."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv.reports.sections.references import ReferencesSection


def test_references_always_available(small_data):
    """is_available() should always return True."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = ReferencesSection(learner)
    assert section.is_available() is True


def test_references_generates_markdown(small_data):
    """generate() should return dict with 'markdown'."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = ReferencesSection(learner)
    result = section.generate()
    assert 'markdown' in result
    assert isinstance(result['markdown'], str)
    assert len(result['markdown']) > 0


def test_references_no_figures(small_data):
    """figures list should be empty."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = ReferencesSection(learner)
    result = section.generate()
    assert result['figures'] == []


@pytest.mark.parametrize("citation_fragment", [
    'TRIPOD+AI',        # reporting guideline
    'scikit-survival',  # sksurv
    'Davidson-Pilon',   # lifelines
])
def test_references_always_includes_standard_citation(small_data, citation_fragment):
    """Standard citations (TRIPOD+AI, scikit-survival, lifelines) are always present."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = ReferencesSection(learner)
    result = section.generate()
    assert citation_fragment in result['markdown']


def test_references_model_citation_conditional(small_data):
    """Cox (1972) should be present when CoxPH is trained."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    # Add a mock CoxPH model
    mock_result = MagicMock()
    mock_result.shap_values = None
    learner._state.models['coxph'] = mock_result

    section = ReferencesSection(learner)
    result = section.generate()
    assert 'Cox, D. R. (1972)' in result['markdown']


def test_references_model_citation_absent(small_data):
    """XGBoost citation should be absent when no XGBCox model."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = ReferencesSection(learner)
    result = section.generate()
    # Chen (2016) = XGBoost
    assert 'Chen, T.' not in result['markdown']


def test_references_method_citation_bootstrap(small_data):
    """Efron (1993) should be present when a model has bootstrap data."""
    import numpy as np
    from mlsurv.results.containers import ModelResult, BootstrapResult
    import pandas as pd

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    # Create a model with bootstrap data
    boot = BootstrapResult(
        summary=pd.DataFrame({'model_type': ['coxph'], 'metric': ['harrell'],
                              'point': [0.7], 'mean': [0.7], 'std': [0.02],
                              'ci_lower': [0.6], 'ci_upper': [0.8]}),
        iterations=pd.DataFrame({'model_type': ['coxph'], 'iteration': [0],
                                 'metric_name': ['harrell'], 'value': [0.7]}),
    )
    mr = ModelResult(name='coxph', model=None,
                     cv_metric_scores=np.array([0.7]),
                     cv_metric_mean=0.7, cv_metric_std=0.0,
                     bootstrap=boot)
    learner._state.models['coxph'] = mr

    section = ReferencesSection(learner)
    result = section.generate()
    assert 'Efron, B.' in result['markdown']


def test_references_method_citation_pimp(small_data):
    """PIMP cites Altmann (2010), not Breiman.

    The PIMP significance test is distinct from plain permutation importance:
    a stored PIMP result (with no perm_mean importance) cites only Altmann.
    """
    import numpy as np
    from mlsurv.results.containers import ModelResult

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    mr = ModelResult(name='coxph', model=None, cv_metric_scores=np.array([0.7]),
                     cv_metric_mean=0.7, cv_metric_std=0.0)
    mr.pimp_importance_test = MagicMock()
    learner._state.models['coxph'] = mr

    section = ReferencesSection(learner)
    result = section.generate()
    assert 'Altmann, A.' in result['markdown']
    assert 'Breiman' not in result['markdown']


def test_references_no_pimp_citation_when_empty(small_data):
    """An empty PIMP dict (no test ran) must not cite Altmann."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = ReferencesSection(learner)
    result = section.generate()
    assert 'Altmann, A.' not in result['markdown']


def test_references_updated_author(small_data):
    """'Goecks Lab' should appear in the citation block."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = ReferencesSection(learner)
    result = section.generate()
    assert 'Goecks Lab' in result['markdown']


def test_references_acknowledgments_lifelines(small_data):
    """'lifelines' should appear in the acknowledgments."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = ReferencesSection(learner)
    result = section.generate()
    assert 'lifelines' in result['markdown']


def test_references_bibtex_collapsible(small_data):
    """BibTeX should be wrapped in <details> blocks."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = ReferencesSection(learner)
    result = section.generate()
    assert '<details>' in result['markdown']
    assert 'BibTeX' in result['markdown']


def test_references_data_citations(small_data):
    """section.data['citations'] should be a DataFrame with expected columns."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = ReferencesSection(learner)
    section.generate()

    assert 'citations' in section.data
    citations_df = section.data['citations']
    assert isinstance(citations_df, pd.DataFrame)
    assert 'key' in citations_df.columns
    assert 'citation' in citations_df.columns
    assert 'type' in citations_df.columns


def test_references_no_state_no_crash(small_data):
    """Section should work with no models trained (setup-only learner)."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    section = ReferencesSection(learner)
    result = section.generate()
    assert isinstance(result['markdown'], str)
    assert len(result['markdown']) > 0

    # Should still have always-included citations
    citations_df = section.data['citations']
    keys = citations_df['key'].tolist()
    assert 'collins_2024' in keys
    assert 'polsterl_2020' in keys
    assert 'akiba_2019' in keys
    assert 'davidson_pilon_2019' in keys


def test_references_method_citation_evaluation(small_data):
    """A full evaluation cites every metric method whose column carries a value.

    Harrell (C-index), KM, and log-rank are always cited when an evaluation
    exists; Uno (IPCW C), Antolini (C^td), Graf (IBS), Steyerberg (calibration
    slope), Demler (GND), and Uno-2007 (AUC(t)) are gated on their respective
    concordance columns / auc_t table being populated.
    """
    import numpy as np
    from mlsurv.results.containers import ModelResult, EvaluationResult
    import pandas as pd

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    # Concordance carrying every metric column the report can render.
    concordance = pd.DataFrame({
        'model_type': ['coxph'],
        'c_index_test': [0.70],
        'c_index_ipcw_test': [0.69],
        'c_index_td_test': [0.68],
        'integrated_brier_test': [0.18],
        'cal_slope_test': [0.95],
        'cal_intercept_test': [0.02],
        'gnd_test': [0.30],
    })
    auc_t = pd.DataFrame({'model_type': ['coxph'], 'eval_time': [12.0], 'auc': [0.71]})
    importance = pd.DataFrame({
        'model_type': ['coxph'], 'feature': ['age'],
        'perm_mean': [0.05], 'perm_std': [0.01],
    })
    eval_r = EvaluationResult(
        concordance=concordance,
        test_set=pd.DataFrame(), auc_t=auc_t,
        importance=importance, importance_iterations=pd.DataFrame(),
    )
    mr = ModelResult(name='coxph', model=None,
                     cv_metric_scores=np.array([0.7]),
                     cv_metric_mean=0.7, cv_metric_std=0.0,
                     evaluation=eval_r)
    learner._state.models['coxph'] = mr

    section = ReferencesSection(learner)
    result = section.generate()
    md = result['markdown']
    # Always-cited when evaluation exists.
    assert 'Harrell' in md
    assert 'Kaplan' in md
    assert 'Mantel' in md
    # Gated metric citations — present because their columns are populated.
    assert 'Uno, H., Cai, T., Pencina' in md    # Uno (2011), IPCW C
    assert 'Antolini' in md                       # Antolini (2005), C^td
    assert 'Graf' in md                           # Graf (1999), IBS
    assert 'Steyerberg' in md                     # Steyerberg (2010), calibration
    assert 'Demler' in md                         # Demler (2015), GND
    assert 'Uno, H., Cai, T., Tian' in md         # Uno (2007), AUC(t)
    assert 'Breiman' in md                         # Breiman (2001), permutation importance


def test_references_metric_citation_gated_on_columns(small_data):
    """Metric citations are absent when their concordance column is missing.

    A discrimination-only evaluation (no IBS/calibration columns, no auc_t)
    must cite Harrell but not Graf, Steyerberg, Demler, or Uno-2007.
    """
    import numpy as np
    from mlsurv.results.containers import ModelResult, EvaluationResult
    import pandas as pd

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    eval_r = EvaluationResult(
        concordance=pd.DataFrame({'model_type': ['coxph'], 'c_index_test': [0.7]}),
        test_set=pd.DataFrame(), auc_t=pd.DataFrame(),
        importance=pd.DataFrame(), importance_iterations=pd.DataFrame(),
    )
    mr = ModelResult(name='coxph', model=None,
                     cv_metric_scores=np.array([0.7]),
                     cv_metric_mean=0.7, cv_metric_std=0.0,
                     evaluation=eval_r)
    learner._state.models['coxph'] = mr

    md = ReferencesSection(learner).generate()['markdown']
    assert 'Harrell' in md          # always present with an evaluation
    assert 'Graf' not in md         # no integrated_brier_test column
    assert 'Steyerberg' not in md   # no calibration columns
    assert 'Demler' not in md       # no gnd_test column
    assert 'Antolini' not in md     # no c_index_td_test column
    assert 'Breiman' not in md      # no perm_mean importance column


def test_references_method_citation_shap(small_data):
    """Lundberg (2017) should be present when shap_values exist on a model."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    mock_result = MagicMock()
    mock_result.shap_values = MagicMock()  # Non-None
    learner._state.models['rsf'] = mock_result

    section = ReferencesSection(learner)
    result = section.generate()
    assert 'Lundberg' in result['markdown']


def test_references_shap_absent_when_none(small_data):
    """Lundberg should NOT appear when model has shap_values=None."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    mock_result = MagicMock()
    mock_result.shap_values = None
    learner._state.models['rsf'] = mock_result

    section = ReferencesSection(learner)
    result = section.generate()
    assert 'Lundberg' not in result['markdown']


def test_references_dl_acknowledgment(small_data):
    """PyTorch/pycox acknowledgment should appear when DL models exist."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    mock_result = MagicMock()
    mock_result.shap_values = None
    learner._state.models['deepsurv'] = mock_result

    section = ReferencesSection(learner)
    result = section.generate()
    assert 'PyTorch' in result['markdown']
    assert 'pycox' in result['markdown']


def test_references_multiple_models(small_data):
    """Multiple model citations should all be present."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    for key in ('coxph', 'rsf'):
        mock = MagicMock()
        mock.shap_values = None
        learner._state.models[key] = mock

    section = ReferencesSection(learner)
    result = section.generate()
    md = result['markdown']
    assert 'Cox, D. R. (1972)' in md
    assert 'Ishwaran' in md


def test_references_detect_citations_no_state(small_data):
    """_detect_used_citations should return only always-included when _state is missing."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    # Remove _state to trigger the AttributeError guard
    del learner._state

    section = ReferencesSection(learner)
    result = section.generate()
    assert isinstance(result['markdown'], str)
    # Should still have always-included citations
    keys = section.data['citations']['key'].tolist()
    assert 'collins_2024' in keys
    assert 'polsterl_2020' in keys


@pytest.mark.parametrize("model_key,author_fragment", [
    ('gbsa', 'Hothorn'),            # GBSA -> Hothorn et al. (2006)
    ('xgbaft', 'Barnwal'),          # XGB-AFT -> Barnwal et al. (2022)
    ('fssvm', 'Polsterl, S., Navab'),  # FSSVM -> Pölsterl et al. (2015)
    ('weibull_aft', 'Kalbfleisch'),  # Weibull AFT -> Kalbfleisch & Prentice (2002)
])
def test_references_model_citation_matches_manuscript(small_data, model_key, author_fragment):
    """Model citations match the manuscript model table (Table 2)."""
    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    mock = MagicMock()
    mock.shap_values = None
    learner._state.models[model_key] = mock

    md = ReferencesSection(learner).generate()['markdown']
    assert author_fragment in md


def test_references_every_citation_has_doi_or_stable_url():
    """Every citation in the database carries a resolvable DOI or stable URL.

    DOIs are tested separately; here we only assert each APA string ends with a
    locator so no citation ships without one. Two works have no DOI (Mantel
    1966, scikit-survival JMLR) and fall back to a stable URL.
    """
    from mlsurv.reports.sections.references import _CITATION_DB

    no_doi = {'mantel_1966', 'polsterl_2020'}
    for key, (apa, bibtex) in _CITATION_DB.items():
        if key in no_doi:
            assert 'https://' in apa, f"{key}: APA missing a stable URL"
            assert 'url={http' in bibtex, f"{key}: BibTeX missing url field"
        else:
            assert 'https://doi.org/' in apa, f"{key}: APA missing DOI link"
            assert 'doi={' in bibtex, f"{key}: BibTeX missing doi field"


def test_eval_gating_helpers_handle_missing_or_bad_concordance():
    """`_eval_has_metric` / `_eval_has_auc` fall back to False, never raise.

    Covers the None-concordance branch and the except (AttributeError,
    TypeError) safety net for mock/placeholder evaluations.
    """
    from types import SimpleNamespace
    from mlsurv.reports.sections.references import _eval_has_metric, _eval_has_auc

    # evaluation present but concordance/auc_t is None
    none_eval = {'m': SimpleNamespace(evaluation=SimpleNamespace(
        concordance=None, auc_t=None))}
    assert _eval_has_metric(none_eval, 'c_index_ipcw_test') is False
    assert _eval_has_auc(none_eval) is False

    # concordance is not a DataFrame -> `column in concordance.columns` raises,
    # caught and treated as "metric absent"
    bad_eval = {'m': SimpleNamespace(evaluation=SimpleNamespace(
        concordance=object(), auc_t=object()))}
    assert _eval_has_metric(bad_eval, 'gnd_test') is False
    assert _eval_has_auc(bad_eval) is False

    # no evaluation at all
    no_eval = {'m': SimpleNamespace(evaluation=None)}
    assert _eval_has_metric(no_eval, 'c_index_td_test') is False
    assert _eval_has_auc(no_eval) is False


def test_references_db_maps_are_consistent():
    """Every DB key has a type; every mapped citation exists in the DB."""
    from mlsurv.reports.sections.references import (
        _CITATION_DB, _CITATION_TYPES, _MODEL_CITATIONS, _ALWAYS_INCLUDED,
    )

    assert set(_CITATION_DB) == set(_CITATION_TYPES), (
        "Every citation must have exactly one type classification"
    )
    for model_key, cite_key in _MODEL_CITATIONS.items():
        assert cite_key in _CITATION_DB, f"{model_key} -> missing citation {cite_key}"
    for cite_key in _ALWAYS_INCLUDED:
        assert cite_key in _CITATION_DB, f"always-included {cite_key} not in DB"


def test_every_citation_is_reachable(small_data):
    """Every citation in the DB must be reachable through detection.

    The consistency test above checks the DB <-> types bijection but not
    reachability: a citation can sit in the DB with no branch in
    _detect_used_citations() that ever adds it (a dead citation). This test
    builds a maximal learner state that exercises every detection branch and
    asserts that the full DB is detected, so any future orphan citation fails
    here and is named in the message.
    """
    import numpy as np
    from mlsurv.reports.sections.references import _CITATION_DB, _MODEL_CITATIONS
    from mlsurv.results.containers import (
        BootstrapResult, EvaluationResult, ModelResult,
    )

    X, y = small_data
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, cv_folds=3)

    # A bare model per model key — covers every 'Model' citation.
    for model_key in _MODEL_CITATIONS:
        learner._state.models[model_key] = ModelResult(
            name=model_key, model=None, cv_metric_scores=np.array([0.7]),
            cv_metric_mean=0.7, cv_metric_std=0.0,
        )

    # Enrich one model so every gated 'Method' branch fires: a full evaluation
    # (all metric columns + auc_t + perm importance), bootstrap, SHAP, and PIMP.
    concordance = pd.DataFrame({
        'model_type': ['coxph'],
        'c_index_test': [0.70],
        'c_index_ipcw_test': [0.69],     # Uno (2011)
        'c_index_td_test': [0.68],       # Antolini (2005)
        'integrated_brier_test': [0.18],  # Graf (1999)
        'cal_slope_test': [0.95],        # Steyerberg (2010)
        'cal_intercept_test': [0.02],
        'gnd_test': [0.30],              # Demler (2015)
    })
    auc_t = pd.DataFrame({'model_type': ['coxph'], 'eval_time': [12.0], 'auc': [0.71]})  # Uno (2007)
    importance = pd.DataFrame({
        'model_type': ['coxph'], 'feature': ['age'],
        'perm_mean': [0.05], 'perm_std': [0.01],  # Breiman (2001)
    })
    boot = BootstrapResult(  # Efron (1993)
        summary=pd.DataFrame({'model_type': ['coxph'], 'metric': ['harrell'],
                              'point': [0.7], 'mean': [0.7], 'std': [0.02],
                              'ci_lower': [0.6], 'ci_upper': [0.8]}),
        iterations=pd.DataFrame({'model_type': ['coxph'], 'iteration': [0],
                                 'metric_name': ['harrell'], 'value': [0.7]}),
    )
    rich = ModelResult(
        name='coxph', model=None, cv_metric_scores=np.array([0.7]),
        cv_metric_mean=0.7, cv_metric_std=0.0,
        evaluation=EvaluationResult(
            concordance=concordance, test_set=pd.DataFrame(), auc_t=auc_t,
            importance=importance, importance_iterations=pd.DataFrame(),
        ),
        bootstrap=boot,
    )
    rich.shap_values = MagicMock()             # Lundberg (2017)
    rich.pimp_importance_test = MagicMock()    # Altmann (2010)
    learner._state.models['coxph'] = rich

    section = ReferencesSection(learner)
    section.generate()
    detected = set(section.data['citations']['key'].tolist())
    assert set(_CITATION_DB) == detected, (
        "Unreachable citations (in DB, no detection branch reaches them): "
        f"{set(_CITATION_DB) - detected}"
    )
