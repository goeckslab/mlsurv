"""Tests for the list form of ``setup(stratify=...)``.

``stratify=['event', 'stage']`` balances the train/test split and the CV folds
on every named variable at once. These tests cover resolution, the
high-cardinality binning rule, rare-cell collapse, precedence over declared
subpopulations, and that the scalar/array forms are untouched.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv import SurvivalLearner
from mlsurv._warnings import MethodologicalWarning, SuggestionWarning


# Enough distinct values to exceed MAX_STRATIFY_LEVELS while leaving the
# column overwhelmingly tied at a single value, so quantile binning collapses.
MAX_LEVELS_TAIL = 25


def _make_data(n=200, n_levels=3, seed=0):
    """Balanced synthetic cohort with a categorical column to stratify on."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        'age': rng.uniform(40, 80, n),
        'biomarker': rng.normal(0, 1, n),
        'stage': rng.randint(0, n_levels, n).astype(float),
    })
    events = rng.rand(n) < 0.5
    times = rng.uniform(1, 100, n)
    y = np.array(list(zip(events, times)),
                 dtype=[('event', bool), ('time', float)])
    return X, y


def _setup(X, y, **kwargs):
    """setup() with the noise turned down; returns the learner."""
    learner = SurvivalLearner(X, y)
    learner.setup(verbose=False, random_state=42, **kwargs)
    return learner


def _cell_shares(events, levels):
    """Fraction of rows in each (event, level) cell."""
    total = len(events)
    counts = pd.crosstab(np.asarray(events).astype(int), np.asarray(levels))
    return counts / total


# ---------------------------------------------------------------------------
# Resolution and composite keys
# ---------------------------------------------------------------------------

class TestStratifyListResolution:
    def test_two_columns_build_composite_keys(self):
        X, y = _make_data()
        learner = _setup(X, y, stratify=['event', 'stage'])

        keys = np.asarray(learner._state.stratify_array)
        assert keys.dtype.kind in 'UO'
        # 2 event values x 3 stage levels
        assert len(set(keys.tolist())) == 6

    def test_three_entries_supported(self):
        X, y = _make_data(n=400)
        X['site'] = (np.arange(len(X)) % 2).astype(float)
        learner = _setup(X, y, stratify=['event', 'stage', 'site'])

        keys = np.asarray(learner._state.stratify_array)
        assert len(set(keys.tolist())) == 12
        assert learner._state.stratify_spec == 'event × stage × site'

    def test_array_entry_mixed_with_column_name(self):
        X, y = _make_data()
        site = pd.Series((np.arange(len(X)) % 2).astype(float), name='site')
        learner = _setup(X, y, stratify=['event', site])

        assert learner._state.stratify_spec == 'event × site'
        assert len(set(np.asarray(learner._state.stratify_array).tolist())) == 4

    def test_unnamed_array_entry_gets_generic_label(self):
        X, y = _make_data()
        arr = (np.arange(len(X)) % 2).astype(float)
        learner = _setup(X, y, stratify=['event', arr])

        assert learner._state.stratify_spec == 'event × custom stratify array'

    def test_single_entry_list_splits_like_the_scalar_form(self):
        """One entry groups the same rows as the scalar form.

        The keys are rendered to strings by the list path (that is what puts
        NaN in one group), so the arrays are not identical — but the grouping,
        and therefore the split, must be.
        """
        X, y = _make_data()
        by_list = _setup(X, y, stratify=['stage'])
        by_scalar = _setup(X, y, stratify='stage')

        pd.testing.assert_index_equal(
            by_list.X_train.index, by_scalar.X_train.index
        )
        # Same partition of rows into groups, whatever the labels are called.
        list_keys = np.asarray(by_list._state.stratify_array)
        scalar_keys = np.asarray(by_scalar._state.stratify_array)
        assert (
            pd.crosstab(list_keys, scalar_keys).astype(bool).sum().tolist()
            == [1] * len(set(scalar_keys.tolist()))
        )

    def test_unknown_column_raises_naming_it(self):
        X, y = _make_data()
        with pytest.raises(ValueError, match="stage_typo"):
            _setup(X, y, stratify=['event', 'stage_typo'])

    def test_wrong_length_array_entry_raises(self):
        X, y = _make_data()
        with pytest.raises(ValueError, match="one label per sample"):
            _setup(X, y, stratify=['event', np.zeros(7)])

    def test_empty_list_raises_with_an_explanation(self):
        """`stratify=[]` used to fall through and die inside sklearn."""
        X, y = _make_data()
        with pytest.raises(ValueError, match="no usable stratification"):
            _setup(X, y, stratify=[])

    def test_duplicate_entry_is_ignored_with_a_warning(self):
        """Joining a variable with itself adds no groups.

        It used to reach the collapse helper, where the structurally empty
        off-diagonal cells were reported as rare data.
        """
        X, y = _make_data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, stratify=['event', 'event'])

        assert any(
            "listed more than once" in str(rec.message) for rec in caught
        ), f"expected a duplicate warning, got: {[str(r.message) for r in caught]}"
        assert not any("__other__" in str(rec.message) for rec in caught)
        assert learner._state.stratify_spec == 'event'

    def test_two_unnamed_arrays_are_both_kept(self):
        """Distinct arrays share a label but are not duplicates.

        Deduplicating on the label alone silently dropped the second array,
        losing a variable the user asked to balance on.
        """
        X, y = _make_data(n=400)
        a = (np.arange(len(X)) % 2).astype(float)
        b = (np.arange(len(X)) // 200).astype(float)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, stratify=['event', a, b])

        assert not any("listed more than once" in str(rec.message) for rec in caught)
        assert len(learner._state.stratify_components) == 3
        assert learner._state.stratify_spec == (
            'event × custom stratify array × custom stratify array 2'
        )
        # 2 event x 2 x 2 = 8 distinct keys, so both arrays reached the keys.
        assert len(set(np.asarray(learner._state.stratify_array).tolist())) == 8

    def test_tuple_spec_works_like_a_list(self):
        X, y = _make_data()
        learner = _setup(X, y, stratify=('event', 'stage'))
        assert learner._state.stratify_spec == 'event × stage'

    def test_too_many_keys_for_the_test_fraction_backs_off(self):
        """More keys than rows on the smaller side is a hard sklearn error.

        `train_test_split` needs room for one member of every class on both
        sides and raises a message naming neither mlsurv nor `stratify=`.
        """
        X, y = _make_data(n=60, n_levels=8)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, stratify=['event', 'stage'], test_fraction=0.15)

        backed_off = [
            str(rec.message) for rec in caught
            if issubclass(rec.category, MethodologicalWarning)
            and "smaller side of the split" in str(rec.message)
        ]
        assert backed_off, f"expected a capacity warning, got: {[str(r.message) for r in caught]}"
        assert "test_fraction" in backed_off[0], "the message should name the remedy"
        assert learner._state.is_setup
        assert learner._state.stratify_spec == 'event'

    def test_capacity_guard_fires_when_the_group_holdout_declines(self):
        """Supplying `groups` does not guarantee a group-aware split.

        `_group_aware_holdout` declines with fewer than two distinct groups, and
        the code then falls through to a real stratified `train_test_split`.
        Gating the guard on the intent rather than the outcome skipped it in
        exactly that case, letting the opaque sklearn error back through.
        """
        X, y = _make_data(n=80, n_levels=20)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, stratify=['event', 'stage'],
                             groups=np.zeros(len(X)), test_fraction=0.1)

        assert learner._state.is_setup, "setup should not die inside sklearn"
        assert not learner._state.group_aware_holdout
        assert any(
            "smaller side of the split" in str(rec.message) for rec in caught
        ), f"the capacity guard should have fired: {[str(r.message) for r in caught]}"

    def test_repeated_setup_clears_stale_components(self):
        """A second setup() must not inherit the first one's components."""
        X, y = _make_data()
        learner = SurvivalLearner(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner.setup(verbose=False, random_state=42,
                          stratify=['event', 'stage'])
            assert learner._state.stratify_components is not None
            learner.setup(verbose=False, random_state=42, stratify='event')

        assert learner._state.stratify_components is None
        assert learner._state.stratify_spec == 'event'


# ---------------------------------------------------------------------------
# The scalar / array forms must not change
# ---------------------------------------------------------------------------

class TestScalarFormUnchanged:
    def test_default_split_matches_sklearn_directly(self):
        """Pin the default split against sklearn, not against itself.

        Comparing `stratify='event'` to the default would compare the default
        with the default — `'event'` IS the signature default, so such a test
        cannot detect a regression. The golden reference is what
        `train_test_split` does with the event indicator.
        """
        from sklearn.model_selection import train_test_split

        X, y = _make_data()
        learner = _setup(X, y, stratify='event', test_fraction=0.3)

        expected_train, _ = train_test_split(
            X, test_size=0.3, stratify=y['event'], random_state=42,
        )
        pd.testing.assert_index_equal(learner.X_train.index, expected_train.index)
        assert np.asarray(learner._state.stratify_array).dtype == np.dtype(bool)
        assert learner._state.stratify_components is None

    def test_column_form_matches_sklearn_directly(self):
        """Same golden check for the scalar column form."""
        from sklearn.model_selection import train_test_split

        X, y = _make_data()
        learner = _setup(X, y, stratify='stage', test_fraction=0.3)

        expected_train, _ = train_test_split(
            X, test_size=0.3, stratify=X['stage'], random_state=42,
        )
        pd.testing.assert_index_equal(learner.X_train.index, expected_train.index)

    def test_plain_label_list_still_reads_as_an_array(self):
        """A list of per-sample labels is data, not a spec."""
        X, y = _make_data()
        labels = list((np.arange(len(X)) % 2).astype(int))
        learner = _setup(X, y, stratify=labels)

        assert learner._state.stratify_components is None
        assert learner._state.stratify_spec == 'custom stratify array'

    def test_none_disables_stratification(self):
        X, y = _make_data()
        learner = _setup(X, y, stratify=None)

        assert learner._state.stratify_array is None
        assert learner._state.stratify_spec == 'none'
        # _get_cv_folds falls back to the event indicator, so the folds are
        # still event-stratified and the report should say so.
        assert learner._state.cv_stratify_spec == 'event'


# ---------------------------------------------------------------------------
# Balance actually achieved
# ---------------------------------------------------------------------------

class TestStratifyListBalance:
    """Tolerances here are deliberately tight.

    A composite-stratified split holds the cell shares to ~0.001; an
    UNstratified split on this data drifts 0.018-0.031. A 0.05 tolerance would
    therefore pass even with stratification removed entirely, which makes the
    test worthless. 0.005 sits well above the stratified noise floor and well
    below anything an unstratified split produces.
    """

    def test_split_preserves_event_by_level_shares(self):
        X, y = _make_data(n=400)
        learner = _setup(X, y, stratify=['event', 'stage'], test_fraction=0.3)

        full = _cell_shares(y['event'], X['stage'])
        train_idx = X.index.get_indexer(learner.X_train.index)
        train = _cell_shares(y['event'][train_idx], X['stage'].values[train_idx])

        assert (full - train).abs().max().max() < 0.005

    def test_every_cv_fold_preserves_shares(self):
        X, y = _make_data(n=400)
        learner = _setup(X, y, stratify=['event', 'stage'], cv_folds=4)

        train_idx = X.index.get_indexer(learner.X_train.index)
        ev = y['event'][train_idx]
        lv = X['stage'].values[train_idx]
        overall = _cell_shares(ev, lv)

        folds = list(learner._state.cv_splitter.split(
            learner.X_train, learner._state.stratify_array
        ))
        assert len(folds) == 4
        for _, test_rows in folds:
            fold = _cell_shares(ev[test_rows], lv[test_rows])
            # Composite keys hold this to ~0.011; event-only drifts ~0.093 and
            # plain KFold ~0.079, so 0.02 discriminates a silent fallback.
            assert (overall - fold).abs().max().max() < 0.02

    def test_auto_cv_folds_resolves_once(self):
        """`cv_folds='auto'` must not resolve twice and warn twice.

        The rebuild needs the fold count to threshold its cells, and so does the
        splitter. Resolving it in both places repeated the resolver's warning.
        """
        X, y = _make_data(n=60, seed=3)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, stratify=['event', 'stage'], cv_folds='auto')

        fold_warnings = [
            str(rec.message) for rec in caught
            if "CV folds" in str(rec.message)
        ]
        assert len(fold_warnings) == len(set(fold_warnings)), (
            f"the fold resolver warned more than once: {fold_warnings}"
        )
        assert learner._state.is_setup

    def test_cv_repeats_keeps_composite_keys(self):
        X, y = _make_data(n=400)
        learner = _setup(X, y, stratify=['event', 'stage'], cv_folds=4, cv_repeats=3)

        folds = list(learner._state.cv_splitter.split(
            learner.X_train, learner._state.stratify_array
        ))
        assert len(folds) == 12
        assert learner._state.cv_stratify_spec == 'event × stage'


class TestTwoStageCollapse:
    """The split and the folds collapse rare cells at different thresholds.

    The split uses 2 on the full cohort; the folds use `cv_folds` on the
    training rows only. Without a test that pins the divergence, the entire CV
    rebuild can be deleted with the suite still green (verified by mutation).
    """

    @staticmethod
    def _rare_level_data(n=300, n_rare=3, seed=0):
        """A stage level with `n_rare` members per event class."""
        X, y = _make_data(n=n, seed=seed)
        stage = np.zeros(len(X))
        event_rows = np.flatnonzero(y['event'])
        censored_rows = np.flatnonzero(~y['event'])
        stage[event_rows[:n_rare]] = 1.0
        stage[censored_rows[:n_rare]] = 1.0
        X['stage'] = stage
        return X, y

    def test_split_keeps_what_cv_backs_out_of(self):
        """3 per class clears the split's threshold of 2 but not cv_folds=5."""
        X, y = self._rare_level_data(n=300, n_rare=3)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, stratify=['event', 'stage'], cv_folds=5)

        assert learner._state.stratify_spec == 'event × stage', (
            "the split should have kept the composite keys"
        )
        assert learner._state.cv_stratify_spec == 'event', (
            "the folds should have backed off to event alone"
        )
        # The folds really are event-only, not just labelled that way. The
        # fallback renders the event indicator to '0'/'1'.
        assert set(np.asarray(learner._state.stratify_array).tolist()) == {'0', '1'}
        assert any(
            issubclass(rec.category, MethodologicalWarning)
            and "Falling back to stratifying on 'event' alone" in str(rec.message)
            for rec in caught
        ), f"expected a fallback warning, got: {[str(r.message) for r in caught]}"

    def test_cv_merges_where_split_did_not(self):
        """A level can merge for the folds while the split keeps it whole."""
        X, y = _make_data(n=400)
        stage = np.zeros(len(X))
        event_rows = np.flatnonzero(y['event'])
        censored_rows = np.flatnonzero(~y['event'])
        # Two levels at 4 per class: fine for the split (>= 2), and for the
        # folds each falls short of 6 but the merged bucket clears it.
        for i, level in enumerate((1.0, 2.0)):
            stage[event_rows[i * 4:(i + 1) * 4]] = level
            stage[censored_rows[i * 4:(i + 1) * 4]] = level
        X['stage'] = stage

        learner = _setup(X, y, stratify=['event', 'stage'], cv_folds=6)

        assert learner._state.stratify_spec == 'event × stage'
        assert learner._state.cv_stratify_spec == 'event × stage (rare levels merged)'
        assert any(
            '__other__' in str(k)
            for k in np.asarray(learner._state.stratify_array).tolist()
        )

    def test_every_level_merging_is_reported_as_no_separation(self):
        """All levels merging is a total merge, not a partial one.

        The keys collapse back to the base component, so the extra variable
        contributed nothing. Labelling that "(rare levels merged)" would claim a
        finer stratification than actually happened, and the report would then
        show a CV row implying the folds balanced on `stage`.
        """
        X, y = _make_data(n=160)
        # Eight levels of 20, alternating event so each (event x level) cell
        # holds ~10 — under cv_folds=12, so every level must merge.
        X['stage'] = (np.arange(len(X)) % 8).astype(float)
        events = (np.arange(len(X)) % 2).astype(bool)
        y = np.array(list(zip(events, y['time'])),
                     dtype=[('event', bool), ('time', float)])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, stratify=['event', 'stage'], cv_folds=12)

        assert any(
            "no longer separates anything" in str(rec.message) for rec in caught
        ), f"expected a total-merge warning, got: {[str(r.message) for r in caught]}"
        assert learner._state.cv_stratify_spec == 'event', (
            "the label must not claim stage still separates the folds"
        )
        # Two event classes times one surviving bucket.
        assert len(set(np.asarray(learner._state.stratify_array).tolist())) == 2

    def test_cv_rebuild_uses_the_train_slice_not_the_full_cohort(self):
        """Rebuilding on the full cohort would not back off; on train it does.

        With 3 rare rows per class in 300, the full cohort has 3 per cell but
        the ~70% training slice has ~2, under cv_folds=3. A rebuild that forgot
        to slice would keep the composite keys.
        """
        X, y = self._rare_level_data(n=300, n_rare=3)
        learner = _setup(X, y, stratify=['event', 'stage'], cv_folds=3)

        components = learner._state.stratify_components
        assert components is not None
        # Components are sliced to the training rows, in X_train order.
        train_idx = X.index.get_indexer(learner.X_train.index)
        assert len(components[0][1]) == len(learner.X_train)
        np.testing.assert_array_equal(
            components[1][1], X['stage'].values[train_idx],
        )


# ---------------------------------------------------------------------------
# High-cardinality columns
# ---------------------------------------------------------------------------

class TestHighCardinalityHandling:
    def test_numeric_column_is_quantile_binned(self):
        X, y = _make_data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, stratify=['event', 'age'])

        binned = [
            str(rec.message) for rec in caught
            if issubclass(rec.category, SuggestionWarning)
            and "quantile groups" in str(rec.message)
        ]
        assert binned, f"expected a binning warning, got: {[str(r.message) for r in caught]}"
        assert "'age'" in binned[0]
        assert "cut points:" in binned[0]
        assert learner._state.stratify_spec == 'event × age (4 quantile bins)'
        # 2 event values x 4 bins
        assert len(set(np.asarray(learner._state.stratify_array).tolist())) == 8

    def test_non_numeric_column_never_reaches_the_binning_rule(self):
        """X is numeric by contract, so binning always has an ordering.

        setup() rejects a non-numeric X before stratification is resolved.
        This pins that ordering, since the binning rule relies on it.
        """
        X, y = _make_data()
        X['patient_id'] = [f"p{i}" for i in range(len(X))]
        with pytest.raises(ValueError, match="non-numeric columns"):
            _setup(X, y, stratify=['event', 'patient_id'])

    def test_tied_numeric_column_is_dropped_not_fatal(self):
        """A column too tied to bin drops out; the other keys still work."""
        X, y = _make_data()
        # Mostly one value, with enough distinct tail values to look wide.
        tied = np.zeros(len(X))
        tied[:MAX_LEVELS_TAIL] = np.arange(1, MAX_LEVELS_TAIL + 1)
        X['tied'] = tied

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, stratify=['event', 'tied'])

        dropped = [
            str(rec.message) for rec in caught
            # A dropped entry is a MethodologicalWarning, not a suggestion: the
            # user asked to balance on it and the split will not.
            if issubclass(rec.category, MethodologicalWarning)
            and "stratify entry 'tied' was dropped" in str(rec.message)
        ]
        assert dropped, f"expected a drop warning, got: {[str(r.message) for r in caught]}"
        assert learner._state.stratify_spec == 'event'

    def test_constant_column_is_dropped_not_silently_accepted(self):
        """A constant column has few levels, so it skips the binning branch.

        It must still be rejected: letting it through replaces real
        stratification with a single all-inclusive class, and the report would
        then name a variable that balanced nothing.
        """
        X, y = _make_data()
        X['const'] = 1.0

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, stratify=['event', 'const'])

        assert any(
            issubclass(rec.category, MethodologicalWarning)
            and "'const' was dropped" in str(rec.message)
            for rec in caught
        ), f"expected a drop warning, got: {[str(r.message) for r in caught]}"
        assert learner._state.stratify_spec == 'event'
        assert len(set(np.asarray(learner._state.stratify_array).tolist())) == 2

    def test_constant_column_alone_raises(self):
        X, y = _make_data()
        X['const'] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="no usable stratification"):
                _setup(X, y, stratify=['const'])

    def test_all_entries_dropped_raises(self):
        X, y = _make_data()
        tied = np.zeros(len(X))
        tied[:MAX_LEVELS_TAIL] = np.arange(1, MAX_LEVELS_TAIL + 1)
        X['tied'] = tied

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="no usable stratification"):
                _setup(X, y, stratify=['tied'])

    def test_missing_values_form_one_stratum_as_promised(self):
        """The NaN warning promises one shared stratum — check it holds.

        Asserting only that the warning fired would certify the promise without
        testing it. NaN != NaN, so this only works because the values are
        rendered to strings before the keys are built.
        """
        X, y = _make_data(n=400)
        X.loc[X.index[:20], 'stage'] = np.nan

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, stratify=['event', 'stage'], cv_folds=3)

        assert any(
            "missing values" in str(rec.message) and "'stage'" in str(rec.message)
            for rec in caught
        )
        keys = [str(k) for k in np.asarray(learner._state.stratify_array).tolist()]
        nan_keys = {k for k in keys if 'nan' in k}
        # One key per event class, not one per missing row.
        assert len(nan_keys) == 2, f"NaN rows scattered across {nan_keys}"
        assert 'stage' in learner._state.stratify_spec

    def test_single_entry_with_missing_values_does_not_crash(self):
        """A one-entry list has no base axis, and once bypassed the rendering.

        Unrendered NaN reached `train_test_split` and raised
        "Input y contains NaN" — after warning that NaN rows would simply form
        their own stratum.
        """
        X, y = _make_data(n=400)
        X.loc[X.index[:20], 'stage'] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner = _setup(X, y, stratify=['stage'])

        assert learner._state.is_setup
        keys = [str(k) for k in np.asarray(learner._state.stratify_array).tolist()]
        assert 'nan' in keys

    def test_single_entry_rare_level_folds_into_the_modal_group(self):
        """No base axis means no __other__ bucket worth having.

        An `__other__` holding only the rare rows is still under the threshold,
        so stratification would be abandoned over a handful of patients. They
        join the largest group instead.
        """
        X, y = _make_data(n=400)
        stage = np.zeros(len(X))
        stage[:150] = 1.0
        stage[150] = 2.0  # a single member: too rare to stand alone
        X['stage'] = stage

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, stratify=['stage'], cv_strategy='kfold')

        assert any(
            "into the most common 'stage' group" in str(rec.message)
            for rec in caught
        ), f"expected a modal-merge warning, got: {[str(r.message) for r in caught]}"
        # The other two levels still stratify.
        assert len(set(np.asarray(learner._state.stratify_array).tolist())) == 2


# ---------------------------------------------------------------------------
# Rare-cell collapse
# ---------------------------------------------------------------------------

class TestRareCellCollapse:
    def test_rare_level_merges_into_other(self):
        """Two one-per-class levels are below the split threshold alone.

        Merged they reach it exactly, so stratification survives with an
        ``__other__`` bucket instead of backing out. ``cv_strategy='kfold'``
        keeps the CV rebuild out of the way so the split keys are what state
        holds at the end.
        """
        X, y = _make_data(n=400)
        stage = np.zeros(len(X))
        event_rows = np.flatnonzero(y['event'])
        censored_rows = np.flatnonzero(~y['event'])
        for i, level in enumerate((1.0, 2.0)):
            stage[event_rows[i]] = level
            stage[censored_rows[i]] = level
        X['stage'] = stage

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, stratify=['event', 'stage'],
                             cv_strategy='kfold')

        merged = [
            str(rec.message) for rec in caught
            if issubclass(rec.category, SuggestionWarning)
            and "__other__" in str(rec.message)
        ]
        assert merged, f"expected a merge warning, got: {[str(r.message) for r in caught]}"
        # Whole-number levels render without a trailing ".0" in the keys.
        assert "'1'" in merged[0] and "'2'" in merged[0]
        keys = [str(k) for k in np.asarray(learner._state.stratify_array).tolist()]
        assert any('__other__' in k for k in keys)
        # The label records the merge, so a merge at one stage and not the other
        # stays visible in the report instead of looking identical.
        assert learner._state.stratify_spec == 'event × stage (rare levels merged)'

    def test_fallback_base_is_itself_made_safe(self):
        """The entry fallen back to must survive sklearn, not just exist.

        The base was never checked against the threshold, so a singleton level
        in the FIRST entry reached `train_test_split` and raised "the least
        populated class has only 1 member" — the opaque error the fallback
        exists to prevent. Existing coverage missed it because the first entry
        was always `'event'`, which is never sparse.
        """
        X, y = _make_data(n=120)
        stage = X['stage'].values.copy()
        stage[0] = 9.0  # a level of exactly one patient, in the first entry
        X['stage'] = stage

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, stratify=['stage', 'event'])

        assert learner._state.is_setup
        counts = pd.Series(
            np.asarray(learner._state.stratify_array)
        ).value_counts()
        assert counts.min() >= 2, f"sklearn would reject these keys: {counts.to_dict()}"
        assert any(
            "folded into its most common level" in str(rec.message)
            for rec in caught
        ), f"the extra fold should be stated: {[str(r.message) for r in caught]}"

    def test_unsalvageable_cell_falls_back_to_first_entry(self):
        X, y = _make_data(n=200)
        stage = np.zeros(len(X))
        stage[0] = 1.0  # a single row: even the merged bucket cannot reach 2
        X['stage'] = stage

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, stratify=['event', 'stage'])

        fallback = [
            str(rec.message) for rec in caught
            if issubclass(rec.category, MethodologicalWarning)
            and "Falling back to stratifying on 'event' alone" in str(rec.message)
        ]
        assert fallback, f"expected a fallback warning, got: {[str(r.message) for r in caught]}"
        assert learner._state.stratify_spec == 'event'
        assert learner._state.is_setup


# ---------------------------------------------------------------------------
# Precedence over subpopulations
# ---------------------------------------------------------------------------

class TestSubpopulationPrecedence:
    def test_list_wins_and_warns(self):
        X, y = _make_data(n=400)
        site = (np.arange(len(X)) % 2).astype(float)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(
                X, y, stratify=['event', 'stage'],
                subpopulations={'site': site}, stratify_subpopulations=True,
            )

        precedence = [
            str(rec.message) for rec in caught
            if "names the stratification variables explicitly" in str(rec.message)
        ]
        assert precedence, f"expected a precedence warning, got: {[str(r.message) for r in caught]}"
        # 2 event x 3 stage, NOT multiplied by site
        assert len(set(np.asarray(learner._state.stratify_array).tolist())) == 6
        assert learner._state.stratify_spec == 'event × stage'
        # Subpopulation analysis still works, it just does not drive the folds
        assert learner._state.subpopulations_test is not None

    def test_subpopulation_augmentation_still_works_without_a_list(self):
        """The pre-existing path is untouched when stratify stays scalar."""
        X, y = _make_data(n=400)
        site = (np.arange(len(X)) % 2).astype(float)
        learner = _setup(X, y, subpopulations={'site': site})

        assert learner._state.stratify_spec == 'event × subpopulation(site)'
        assert learner._state.cv_stratify_spec == 'event × subpopulation(site)'


# ---------------------------------------------------------------------------
# Pre-split data
# ---------------------------------------------------------------------------

class TestPresplitIgnoresStratify:
    def test_list_is_reported_as_ignored(self):
        X, y = _make_data(n=200)
        split = len(X) // 2
        learner = SurvivalLearner(
            X_train=X.iloc[:split], y_train=y[:split],
            X_test=X.iloc[split:], y_test=y[split:],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner.setup(verbose=False, random_state=42,
                          stratify=['event', 'stage'])

        ignored = [
            str(rec.message) for rec in caught
            if "ignored when using pre-split data" in str(rec.message)
        ]
        assert ignored
        assert "'stage'" in ignored[0]
        # The label must say what actually happened, not what was requested.
        assert learner._state.stratify_spec == 'event'
        assert learner._state.stratify_components is None


# ---------------------------------------------------------------------------
# Non-stratified CV strategies
# ---------------------------------------------------------------------------

class TestNonStratifiedCv:
    def test_kfold_records_no_cv_label(self):
        X, y = _make_data()
        learner = _setup(X, y, stratify=['event', 'stage'], cv_strategy='kfold')

        assert learner._state.cv_stratify_spec is None
        # The split is still stratified; only the folds are not.
        assert learner._state.stratify_spec == 'event × stage'


# ---------------------------------------------------------------------------
# Split strategies that ignore stratification entirely
# ---------------------------------------------------------------------------

class TestSplitsThatIgnoreStratify:
    """Two split strategies never stratify. The report must not claim they did.

    A wrong label here is worse than the old generic one: `event × stage` is
    specific enough that a reader will believe it.
    """

    @staticmethod
    def _split_row(learner):
        from mlsurv.reports.sections.data_preprocessing import (
            _summary_stratification_rows,
        )
        rows = _summary_stratification_rows(learner)
        return next(r[1] for r in rows if r[0] == "Split stratification")

    def test_group_aware_holdout_reported_as_unstratified(self):
        X, y = _make_data(n=200)
        X['sid'] = (np.arange(len(X)) // 4).astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner = _setup(X, y, stratify=['event', 'stage'], groups='sid')

        assert learner._state.group_aware_holdout
        assert "None (group-aware holdout" in self._split_row(learner)

    def test_group_aware_holdout_skips_the_capacity_guard(self):
        """The guard protects `train_test_split`, which never runs here.

        A group-aware holdout does its own splitting, so degrading the keys for
        it would report a problem that does not exist — and a
        MethodologicalWarning is reserved for problems mlsurv actually detected.
        """
        X, y = _make_data(n=120)
        X['sid'] = (np.arange(len(X)) // 3).astype(float)
        X['site'] = (np.arange(len(X)) % 2).astype(float)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, stratify=['event', 'stage', 'site'],
                             groups='sid', test_fraction=0.05)

        assert learner._state.group_aware_holdout
        assert not any(
            "smaller side of the split" in str(rec.message) for rec in caught
        ), "the capacity guard should not fire for a group-aware holdout"
        assert learner._state.stratify_spec == 'event × stage × site'

    def test_group_aware_holdout_still_reports_the_cv_row(self):
        """The folds ARE stratified even when the holdout is not.

        The CV row is dropped only when it would restate the split row. Here
        the split row says "None (group-aware holdout)", so suppressing the CV
        row would hide a balance that was genuinely applied.
        """
        from mlsurv.reports.sections.data_preprocessing import (
            _summary_stratification_rows,
        )

        X, y = _make_data(n=200)
        X['sid'] = (np.arange(len(X)) // 4).astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner = _setup(X, y, stratify=['event', 'stage'], groups='sid')

        rows = dict(_summary_stratification_rows(learner))
        assert "CV stratification" in rows, (
            f"the folds are stratified but no CV row was shown: {rows}"
        )
        assert rows["CV stratification"] == "Event × stage"

    def test_timeseries_reported_as_unstratified_and_warns_about_nothing(self):
        """A temporal split is ordered, so resolution is skipped entirely.

        Otherwise binning and rare-cell warnings fire about work that never
        happens, and a stale spec list reaches the train-slicing code.
        """
        X, y = _make_data(n=200)
        X['wide'] = np.arange(len(X)).astype(float)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, stratify=['event', 'wide'],
                             cv_strategy='timeseries')

        assert learner._state.is_setup
        assert learner._state.stratify_array is None
        assert learner._state.cv_stratify_spec is None
        assert "None (temporal split" in self._split_row(learner)
        assert not any(
            "quantile groups" in str(rec.message) for rec in caught
        ), "binning should not run for a split that is never stratified"


# ---------------------------------------------------------------------------
# Subpopulation augmentation: the split-side fallback label
# ---------------------------------------------------------------------------

class TestSubpopulationSplitFallbackLabel:
    def test_split_label_drops_subpopulation_after_fallback(self):
        """When the augmentation backs out, the label must back out too.

        This is the same "report claims more than it knows" bug the CV side
        already had; only the success path was covered.
        """
        X, y = _make_data(n=200)
        # A single-member subpopulation: even the merged bucket cannot reach 2.
        site = np.array(['common'] * len(X), dtype=object)
        site[0] = 'rare'

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = _setup(X, y, subpopulations={'site': site})

        assert any(
            issubclass(rec.category, MethodologicalWarning)
            and "Falling back to event-only" in str(rec.message)
            for rec in caught
        ), f"expected a fallback warning, got: {[str(r.message) for r in caught]}"
        assert learner._state.stratify_spec == 'event', (
            "the split label must not claim the subpopulation was used"
        )
