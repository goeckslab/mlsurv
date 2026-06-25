"""Tests for ``learner.validate(references=...)``.

References (LORIS, TMB, PD-L1, ...) ride alongside models through the same
``_aggregate_evaluation_results`` machinery used by ``benchmark()``; these
tests pin the contract: reference rows in the concordance table, NaN
filtering per-reference, name/length validation, bootstrap CI propagation,
and CLI parity.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv.utils import SurvivalArray


def _make_small_data():
    """50-sample synthetic survival dataset."""
    rng = np.random.RandomState(42)
    n = 50
    X = pd.DataFrame({
        'age': rng.uniform(30, 80, n),
        'stage': rng.randint(1, 5, n),
        'grade': rng.uniform(1, 4, n),
        'tumor_size': rng.uniform(0.5, 10, n),
        'biomarker': rng.randn(n),
    })
    times = rng.exponential(2, n).clip(0.1, 10)
    events = rng.binomial(1, 0.6, n).astype(bool)
    y = SurvivalArray(np.array(
        [(e, t) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)],
    ))
    return X, y


@pytest.fixture(scope="module")
def _trained_template():
    X, y = _make_small_data()
    learner = SurvivalLearner(X, y)
    learner.setup(random_state=42, feature_selector=None)
    learner.run(["coxph"], tune=False)
    return learner


@pytest.fixture
def trained_learner(_trained_template):
    return copy.deepcopy(_trained_template)


def _val_cohort(learner):
    """Half-cohort drawn from training X/y — cheap and aligned."""
    X = learner._state.X
    y = learner._state.y
    half = len(X) // 2
    return X.iloc[:half].copy(), y[:half]


def _refs_for(X_val, seed=0):
    rng = np.random.RandomState(seed)
    return {
        "rand_a": rng.randn(len(X_val)),
        "rand_b": rng.randn(len(X_val)),
    }


@pytest.mark.integration
def test_validate_with_references_basic(trained_learner):
    """Reference rows appear in concordance with finite c-index."""
    X_val, y_val = _val_cohort(trained_learner)
    refs = _refs_for(X_val)

    result = trained_learner.validate(
        X_val, y_val, name="C1", references=refs,
        bootstrap=False, verbose=False,
    )

    conc = result.evaluation.concordance
    models_in = set(conc['model_type'])
    assert {"coxph", "rand_a", "rand_b"} <= models_in
    ref_rows = conc[conc['model_type'].isin(["rand_a", "rand_b"])]
    assert np.isfinite(ref_rows['c_index_test']).all()
    assert result.reference_names == ("rand_a", "rand_b")
    assert set(result.reference_arrays.keys()) == {"rand_a", "rand_b"}


@pytest.mark.integration
def test_validate_references_higher_is_worse_dict(trained_learner):
    """Per-reference ``higher_is_worse`` dict flips the score sign."""
    X_val, y_val = _val_cohort(trained_learner)
    rng = np.random.RandomState(1)
    score = rng.randn(len(X_val))
    refs = {"r": score}

    res_worse = trained_learner.validate(
        X_val, y_val, name="HiW", references=refs,
        higher_is_worse={"r": True}, bootstrap=False, verbose=False,
    )
    res_better = trained_learner.validate(
        X_val, y_val, name="HiB", references=refs,
        higher_is_worse={"r": False}, bootstrap=False, verbose=False,
    )

    c_worse = res_worse.evaluation.concordance.set_index('model_type').loc['r', 'c_index_test']
    c_better = res_better.evaluation.concordance.set_index('model_type').loc['r', 'c_index_test']
    assert np.isclose(c_worse + c_better, 1.0, atol=1e-6), (
        f"c({c_worse}) + c({c_better}) should equal 1 under sign flip"
    )


@pytest.mark.integration
def test_validate_references_nan_handling(trained_learner):
    """NaN values in a reference filter to the finite subset only."""
    X_val, y_val = _val_cohort(trained_learner)
    rng = np.random.RandomState(2)
    score = rng.randn(len(X_val))
    score[:5] = np.nan
    refs = {"with_nan": score}

    result = trained_learner.validate(
        X_val, y_val, name="NaN", references=refs,
        bootstrap=False, verbose=False,
    )
    conc = result.evaluation.concordance
    nan_row = conc[conc['model_type'] == 'with_nan'].iloc[0]
    assert np.isfinite(nan_row['c_index_test'])


def test_validate_references_length_mismatch(trained_learner):
    X_val, y_val = _val_cohort(trained_learner)
    bad = {"r": np.zeros(len(X_val) + 3)}
    with pytest.raises(ValueError, match="length"):
        trained_learner.validate(
            X_val, y_val, name="Bad", references=bad,
            bootstrap=False, verbose=False,
        )


@pytest.mark.integration
def test_validate_references_all_nonfinite_warns_and_drops(trained_learner):
    """All-NaN reference is dropped with a MethodologicalWarning.

    External cohorts may legitimately lack a biomarker — a column present
    but 100% NaN must not crash ``validate()`` when other references are
    finite.
    """
    from mlsurv._warnings import MethodologicalWarning

    X_val, y_val = _val_cohort(trained_learner)
    rng = np.random.RandomState(3)
    n = len(X_val)
    # Mixed NaN + Inf — all non-finite — exercises both counts in the warning.
    mixed = np.empty(n)
    half = n // 2
    mixed[:half] = np.nan
    mixed[half:] = np.inf
    refs = {
        "all_nan": np.full(n, np.nan),
        "all_mixed": mixed,
        "good": rng.randn(n),
    }

    with pytest.warns(MethodologicalWarning, match="all .* values are non-finite"):
        result = trained_learner.validate(
            X_val, y_val, name="AllNaN", references=refs,
            bootstrap=False, verbose=False,
        )

    conc = result.evaluation.concordance
    model_types = set(conc['model_type'])
    assert "all_nan" not in model_types
    assert "all_mixed" not in model_types
    assert "good" in model_types
    assert "all_nan" not in result.reference_names
    assert "all_mixed" not in result.reference_names
    assert "good" in result.reference_arrays


def test_validate_references_name_collision(trained_learner):
    """Reference named after a trained model raises."""
    X_val, y_val = _val_cohort(trained_learner)
    refs = {"coxph": np.zeros(len(X_val))}
    with pytest.raises(ValueError, match="overlap with model names"):
        trained_learner.validate(
            X_val, y_val, name="Collide", references=refs,
            bootstrap=False, verbose=False,
        )


@pytest.mark.integration
def test_validate_references_bootstrap_ci(trained_learner):
    """Bootstrap CIs propagate to reference rows when enabled."""
    X_val, y_val = _val_cohort(trained_learner)
    refs = _refs_for(X_val, seed=3)

    result = trained_learner.validate(
        X_val, y_val, name="BS", references=refs,
        bootstrap=True, n_bootstrap=5, verbose=False,
    )
    assert result.evaluation.bootstrap is not None
    summary = result.evaluation.bootstrap.summary
    assert summary is not None and 'model_type' in summary.columns
    boot_models = set(summary['model_type'].unique())
    assert {"rand_a", "rand_b"} <= boot_models
    # CI columns populated for at least one reference metric
    ref_summary = summary[summary['model_type'] == 'rand_a']
    assert ref_summary['ci_lower'].notna().any()
    assert ref_summary['ci_upper'].notna().any()


@pytest.mark.integration
def test_validate_references_in_to_dataframes(trained_learner):
    """Reference arrays land in to_dataframes() output."""
    X_val, y_val = _val_cohort(trained_learner)
    refs = _refs_for(X_val, seed=4)

    result = trained_learner.validate(
        X_val, y_val, name="Export", references=refs,
        bootstrap=False, verbose=False,
    )
    dfs = result.to_dataframes()
    assert 'reference_scores' in dfs
    assert list(dfs['reference_scores'].columns) == ["rand_a", "rand_b"]
    assert len(dfs['reference_scores']) == len(X_val)


def test_validate_references_empty_dict(trained_learner):
    """Empty references dict behaves the same as None — pins the truthy check."""
    X_val, y_val = _val_cohort(trained_learner)
    res_none = trained_learner.validate(
        X_val, y_val, name="None", references=None,
        bootstrap=False, verbose=False,
    )
    res_empty = trained_learner.validate(
        X_val, y_val, name="Empty", references={},
        bootstrap=False, verbose=False,
    )
    assert res_none.reference_names == ()
    assert res_empty.reference_names == ()
    assert res_none.reference_arrays is None
    assert res_empty.reference_arrays is None


@pytest.mark.integration
def test_validate_references_feature_overlap_warns(trained_learner):
    """Reference name matching an X_val column triggers data-leakage warning."""
    import warnings as _warnings
    from mlsurv._warnings import SuggestionWarning

    X_val, y_val = _val_cohort(trained_learner)
    # 'age' is a column in the synthetic training X — reusing it as a
    # reference name is a classic leakage footgun.
    refs = {"age": np.random.RandomState(7).randn(len(X_val))}

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        trained_learner.validate(
            X_val, y_val, name="Leak", references=refs,
            bootstrap=False, verbose=False,
        )
    leak_msgs = [
        rec for rec in w
        if issubclass(rec.category, SuggestionWarning)
        and 'data leakage' in str(rec.message)
    ]
    assert leak_msgs, "Expected SuggestionWarning about data leakage"


@pytest.mark.integration
def test_validate_references_partial_nan_warns(trained_learner):
    """Partial-NaN references emit MethodologicalWarning per No-Silent-Data-Loss."""
    import warnings as _warnings
    from mlsurv._warnings import MethodologicalWarning

    X_val, y_val = _val_cohort(trained_learner)
    score = np.random.RandomState(8).randn(len(X_val))
    score[:3] = np.nan
    refs = {"partial": score}

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        trained_learner.validate(
            X_val, y_val, name="PartNaN", references=refs,
            bootstrap=False, verbose=False,
        )
    msgs = [
        rec for rec in w
        if issubclass(rec.category, MethodologicalWarning)
        and 'partial' in str(rec.message).lower()
           or "NaN" in str(rec.message)
    ]
    assert msgs, "Expected MethodologicalWarning reporting dropped NaN rows"


@pytest.mark.integration
def test_cli_validate_with_references(tmp_path):
    """CLI --references-file round-trips reference rows into the metrics CSV."""
    import io
    import sys
    from mlsurv.cli._parsers import create_parser
    from mlsurv.cli._commands import cmd_validate

    X, y = _make_small_data()
    # Training CSV
    train_df = X.copy()
    train_df['time'] = y['time']
    train_df['event'] = y['event'].astype(int)
    train_path = tmp_path / "train.csv"
    train_df.to_csv(train_path, index=False)
    # Validation CSV (use first half as a stand-in cohort)
    half = len(X) // 2
    val_df = train_df.iloc[:half].copy()
    val_path = tmp_path / "val.csv"
    val_df.to_csv(val_path, index=False)
    # References CSV aligned with val_path rows
    rng = np.random.RandomState(9)
    refs_df = pd.DataFrame({
        'ref_x': rng.randn(half),
        'ref_y': rng.randn(half),
    })
    refs_path = tmp_path / "refs.csv"
    refs_df.to_csv(refs_path, index=False)

    output_path = tmp_path / "out.csv"

    parser = create_parser()
    args = parser.parse_args([
        '--quiet',
        'validate',
        str(train_path),
        '--validation-data', str(val_path),
        '--model', 'coxph',
        '--references-file', str(refs_path),
        '--no-bootstrap',
        '--output', str(output_path),
    ])
    stdout_save, stderr_save = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        exit_code = cmd_validate(args)
    finally:
        sys.stdout, sys.stderr = stdout_save, stderr_save

    assert exit_code == 0
    out_df = pd.read_csv(output_path)
    assert {'ref_x', 'ref_y'} <= set(out_df['model_type'])
