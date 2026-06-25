"""Guard the baked-in scikit-survival targets in Vignette 03.

The vignette's sksurv `target` column is the mean of each scikit-survival recipe
over a seed list, produced by scripts/benchmarks/vignette_03_sksurv_targets.py.
Those means are only trustworthy if the recipes still reproduce the exact values
the scikit-survival docs print. This test re-runs the anchor recipes and checks
they still land on the documented numbers. If a scikit-survival upgrade quietly
changes an estimator's output, this fails instead of letting a stale target slip
through unnoticed.

It also covers the two pieces of new plain logic the rewrite added: the
multiseed-target aggregation contract, and the supp-table's mean ± std formatter.

Tolerances are loose enough to ride out a minor library bump but tight enough to
catch a real recipe break.
"""

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _load(module_name, relpath):
    """Load a module by file path -- scripts/ is not an importable package."""
    spec = importlib.util.spec_from_file_location(module_name, _REPO / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


targets_mod = _load("vignette_03_sksurv_targets",
                    "scripts/benchmarks/vignette_03_sksurv_targets.py")
supp_mod = _load("make_vignette_03_supp_table",
                 "scripts/figures/make_vignette_03_supp_table.py")


def test_gbsg2_coxnet_anchor_reproduces_doc_value():
    """CoxNet on a stratified 75/25 split, seed 1, reproduces the doc's 0.688."""
    Xt, y = targets_mod.gbsg2_onehot()
    assert targets_mod.sk_gbsg2_coxnet(1, Xt, y) == pytest.approx(0.688, abs=0.01)


def test_gbsg2_rsf_anchor_reproduces_doc_value():
    """RSF on a stratified 75/25 split, seed 1, reproduces the doc's 0.692."""
    Xt, y = targets_mod.gbsg2_onehot()
    assert targets_mod.sk_gbsg2_rsf(1, Xt, y) == pytest.approx(0.692, abs=0.01)


def test_fmt_mean_std_branches():
    """The supp-table formatter shows 'mean ± std', a bare mean, or 'N/A'."""
    assert supp_mod.fmt_mean_std(0.669, 0.024) == "0.669 ± 0.024"
    # FSSVM target has no std -> bare mean, no ' ± '.
    assert supp_mod.fmt_mean_std(0.720, None) == "0.720"
    assert supp_mod.fmt_mean_std(float("nan"), 0.02) == "N/A"


@pytest.mark.integration
def test_veterans_fssvm_anchor_reproduces_doc_value():
    """FSSVM 100-split grid search reproduces the doc's 0.720 (slow)."""
    assert targets_mod.sk_veterans_fssvm() == pytest.approx(0.720, abs=0.01)


@pytest.mark.integration
def test_multiseed_targets_contract():
    """multiseed_targets returns (mean, std) per model; FSSVM std is None.

    Checks the structure the vignette and supp-table depend on, on a 2-seed run
    (values are covered by the anchor tests, so only the shape matters here).
    """
    targets = targets_mod.multiseed_targets(seeds=range(2))
    assert set(targets) == {"GBSG2/coxnet", "GBSG2/rsf", "Veterans/fssvm"}
    for key in ("GBSG2/coxnet", "GBSG2/rsf"):
        mean, std = targets[key]
        assert isinstance(mean, float) and isinstance(std, float)
    fssvm_mean, fssvm_std = targets["Veterans/fssvm"]
    assert isinstance(fssvm_mean, float)
    assert fssvm_std is None
