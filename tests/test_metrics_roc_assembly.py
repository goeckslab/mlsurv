"""Direct unit tests for ``assemble_roc_curves_with_references``.

These cover the helper's contract that is not reachable through the
call-site integration tests (plot_roc / benchmarking / evaluation report):
the ``RocAssembly`` field set, ``missing_ref_arrays`` / ``mislength_refs``
detection, per-reference NaN-masking, ``dropped_by_ref`` /
``dropped_model_horizons`` bookkeeping with the ``on_error`` callback, and
the ``catch`` exception contract. ``compute_roc_curves`` is monkeypatched so
the tests are fast and deterministic (numerics are covered in
``test_viz_roc.py``).
"""
import numpy as np
import pandas as pd
import pytest

from mlsurv.viz import _metrics_roc
from mlsurv.viz._metrics_roc import (
    RocAssembly,
    assemble_roc_curves_with_references,
)


def _y(n):
    """Minimal all-event structured survival array of length *n*."""
    return np.array(
        [(True, float(i + 1)) for i in range(n)],
        dtype=[("event", bool), ("time", float)],
    )


def _fake_roc(y_train, y_test, risk_scores, t, **kwargs):
    """Stand-in for compute_roc_curves: one row per model, records y_test len."""
    names = list(risk_scores)
    return pd.DataFrame({
        "model": names,
        "fpr": [0.0] * len(names),
        "tpr": [1.0] * len(names),
        "auc": [0.9] * len(names),
        "eval_time": [t] * len(names),
        "_n": [len(y_test)] * len(names),
    })


def test_returns_rocassembly_with_expected_fields(monkeypatch):
    monkeypatch.setattr(_metrics_roc, "compute_roc_curves", _fake_roc)
    y = _y(10)
    res = assemble_roc_curves_with_references(
        y, y, {"coxph": np.arange(10.0)}, [1.0, 2.0],
    )
    assert isinstance(res, RocAssembly)
    assert res._fields == (
        "roc_data", "all_roc_dfs", "dropped_by_ref",
        "dropped_model_horizons", "missing_ref_arrays", "mislength_refs",
    )
    assert set(res.roc_data) == {1.0, 2.0}
    # all_roc_dfs is co-derived from roc_data's values.
    assert len(res.all_roc_dfs) == len(res.roc_data) == 2
    assert res.dropped_by_ref == {}
    assert res.dropped_model_horizons == []
    assert res.missing_ref_arrays == []
    assert res.mislength_refs == []


def test_reference_nan_masking_uses_finite_subset(monkeypatch):
    monkeypatch.setattr(_metrics_roc, "compute_roc_curves", _fake_roc)
    y = _y(10)
    ref_raw = np.arange(10.0)
    ref_raw[:3] = np.nan  # 7 finite entries
    res = assemble_roc_curves_with_references(
        y, y,
        {"coxph": np.arange(10.0), "ref": np.arange(7.0)},
        [1.0],
        reference_names=["ref"],
        reference_arrays={"ref": ref_raw},
    )
    df = res.roc_data[1.0]
    assert df.loc[df["model"] == "coxph", "_n"].iloc[0] == 10  # full y_data
    assert df.loc[df["model"] == "ref", "_n"].iloc[0] == 7  # masked subset
    assert res.missing_ref_arrays == []
    assert res.mislength_refs == []


def test_missing_ref_arrays_flagged_and_falls_back_to_unmasked(monkeypatch):
    monkeypatch.setattr(_metrics_roc, "compute_roc_curves", _fake_roc)
    y = _y(10)
    res = assemble_roc_curves_with_references(
        y, y, {"coxph": np.arange(10.0), "ref": np.arange(10.0)}, [1.0],
        reference_names=["ref"],
        reference_arrays={},  # ref structurally absent
    )
    assert res.missing_ref_arrays == ["ref"]
    assert res.mislength_refs == []
    df = res.roc_data[1.0]
    assert df.loc[df["model"] == "ref", "_n"].iloc[0] == 10  # unmasked


def test_mislength_refs_flagged_and_falls_back_to_unmasked(monkeypatch):
    monkeypatch.setattr(_metrics_roc, "compute_roc_curves", _fake_roc)
    y = _y(10)
    res = assemble_roc_curves_with_references(
        y, y, {"ref": np.arange(10.0)}, [1.0],
        reference_names=["ref"],
        reference_arrays={"ref": np.ones(13)},  # length != len(y_data)
    )
    assert res.mislength_refs == ["ref"]
    assert res.missing_ref_arrays == []
    df = res.roc_data[1.0]
    assert df.loc[df["model"] == "ref", "_n"].iloc[0] == 10  # unmasked


def test_dropped_by_ref_records_failed_ref_and_fires_on_error(monkeypatch):
    y = _y(10)
    calls = []

    def flaky(y_train, y_test, risk_scores, t, **kwargs):
        if "bad" in risk_scores:
            raise ValueError("degenerate subset")
        return _fake_roc(y_train, y_test, risk_scores, t)

    monkeypatch.setattr(_metrics_roc, "compute_roc_curves", flaky)
    res = assemble_roc_curves_with_references(
        y, y,
        {"coxph": np.arange(10.0), "bad": np.arange(10.0)},
        [1.0, 2.0],
        reference_names=["bad"],
        reference_arrays={"bad": np.arange(10.0)},
        on_error=lambda exc, kind, name, t: calls.append((kind, name, t)),
    )
    assert res.dropped_by_ref == {"bad": [1.0, 2.0]}
    assert res.dropped_model_horizons == []
    assert calls == [
        ("reference", "bad", 1.0),
        ("reference", "bad", 2.0),
    ]
    # The surviving model is still present at every horizon.
    assert all(
        "coxph" in df["model"].values for df in res.roc_data.values()
    )


def test_dropped_model_horizons_records_and_leaves_empty_results(monkeypatch):
    y = _y(10)

    def always_fail(y_train, y_test, risk_scores, t, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(_metrics_roc, "compute_roc_curves", always_fail)
    res = assemble_roc_curves_with_references(
        y, y, {"coxph": np.arange(10.0)}, [1.0, 2.0],
        catch=(ValueError, RuntimeError),
    )
    assert res.dropped_model_horizons == [1.0, 2.0]
    assert res.roc_data == {}
    assert res.all_roc_dfs == []


def test_catch_does_not_swallow_unlisted_exception(monkeypatch):
    y = _y(10)

    def raises_keyerror(y_train, y_test, risk_scores, t, **kwargs):
        raise KeyError("programming error")

    monkeypatch.setattr(_metrics_roc, "compute_roc_curves", raises_keyerror)
    # KeyError is not in the default catch tuple -> must propagate.
    with pytest.raises(KeyError):
        assemble_roc_curves_with_references(
            y, y, {"coxph": np.arange(10.0)}, [1.0],
            catch=(ValueError, RuntimeError),
        )
