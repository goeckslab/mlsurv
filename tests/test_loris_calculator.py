"""Guard the LORIS calculator used by Vignette 07 and its LORIS-metrics table.

scripts/loris_score_calculator.py once carried wrong cancer-type terms; the
values stayed finite, so the vignette smoke test could not catch it. These
tests pin the calculator to the official LLR6 formula from the LORIS GitHub
README (https://github.com/rootchang/LORIS) and to a published AUC from the
LORIS paper (Chang et al., Nature Cancer 2024, Fig 2a).

They also cover the fixed-time labelling in
scripts/figures/make_vignette_07_loris_metrics_table.py, which follows the
LORIS paper's own code (died before t = 0, followed to t = 1, censored before
t dropped).

All tests are fast: no models are trained.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "examples" / "data" / "loris"


def _load(module_name, relpath):
    """Load a module by file path -- scripts/ is not an importable package."""
    spec = importlib.util.spec_from_file_location(module_name, _REPO / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


calc = _load("loris_score_calculator", "scripts/loris_score_calculator.py")

# The table script imports the calculator by name from scripts/, so make that
# importable before loading it.
sys.path.insert(0, str(_REPO / "scripts"))
table = _load("make_vignette_07_loris_metrics_table",
              "scripts/figures/make_vignette_07_loris_metrics_table.py")

# Official LLR6 cancer-type terms for the CancerType1..16 indicator columns
# in the LORIS data, in order (LORIS GitHub README).
_OFFICIAL_CT = [-0.3323, -0.3323, -0.102, -0.0079, 0.55, 0.2306, 0.0678, -0.1189,
                -0.0086, 0.1255, 0.0008, -0.052, -1.1169, 0.5451, 0.0542, -0.0033]


def _official_loris(df):
    """LORIS from the README formula, using the data's CancerType1..16 columns."""
    s = (0.0371 * df.TMB.clip(upper=50) - 0.8775 * df.Systemic_therapy_history
         + 0.5382 * df.Albumin - 0.033 * df.NLR.clip(upper=25)
         + 0.0049 * df.Age.clip(upper=85)
         + sum(c * df[f"CancerType{k + 1}"] for k, c in enumerate(_OFFICIAL_CT))
         - 2.0886)
    return 1 / (1 + np.exp(-s))


_COHORTS = sorted(p.stem for p in _DATA.glob("*.csv"))


def test_cohort_files_found():
    """The parametrized checks below must not silently run on zero cohorts."""
    assert len(_COHORTS) == 10


@pytest.mark.parametrize("cohort", _COHORTS)
def test_matches_official_formula(cohort):
    """Same score as the README formula on every cohort.

    Rows with a missing input are skipped: the calculator imputes them, while
    the official formula leaves them undefined (the LORIS paper drops them).
    """
    df = pd.read_csv(_DATA / f"{cohort}.csv")
    ours = calc.calculate_loris(df)
    official = _official_loris(df)
    ok = official.notna()
    assert ok.sum() > 0
    np.testing.assert_allclose(ours[ok], official[ok], rtol=0, atol=1e-12)


def test_reproduces_published_chowell_test_auc():
    """Response AUC on Chowell test matches LORIS Fig 2a (0.75)."""
    df = pd.read_csv(_DATA / "Chowell_test.csv")
    auc = roc_auc_score(df.Response, calc.calculate_loris(df))
    assert auc == pytest.approx(0.754, abs=0.001)


def test_every_data_cancer_type_has_a_term():
    """Each cancer type in the ICB training data maps to its own term, not 'Other'."""
    df = pd.read_csv(_DATA / "Chowell_train.csv")
    mapped = calc.map_cancer_type(df)
    assert (mapped == df.CancerType).all()


@pytest.mark.parametrize("cancer_type", ["CNS", "Unknown primary", "Not a real type"])
def test_types_outside_the_model_get_no_term(cancer_type):
    """Types LORIS has no indicator for (e.g. MSK2's CNS) add 0 to the score."""
    assert calc.map_cancer_type(pd.DataFrame({"CancerType": [cancer_type]})).iloc[0] == "Other"
    assert calc.CANCER_TYPE_CALIBRATION["Other"] == 0.0


def test_fixed_time_labels_follow_loris_paper():
    """Died before t -> 0, followed to t or beyond -> 1, censored before t -> dropped."""
    d = pd.DataFrame({
        "time":  [2.0, 3.0, 8.0, 12.0, 15.0],
        "event": [1,   0,   1,   0,    1],
        # Risk ordering puts both deaths above both survivors -> perfect AUC.
        "score": [0.9, 0.5, 0.8, 0.1,  0.2],
    })
    # t = 6: patient 0 died (0); patient 1 censored at 3 (dropped); 2-4 followed (1).
    assert table.loris_method_auc(d, "score", 6) == pytest.approx(1.0)
    # Reverse the scores: survivors now look riskier -> AUC 0.
    d["score"] = -d["score"]
    assert table.loris_method_auc(d, "score", 6) == pytest.approx(0.0)


def test_auc_refuses_single_class():
    """A bootstrap resample with one class must raise, not leak NaN into the CI."""
    with pytest.raises(ValueError):
        table._auc(pd.Series([1, 1, 1]), pd.Series([0.1, 0.2, 0.3]))
