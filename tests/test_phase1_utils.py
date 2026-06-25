"""Test utility functions for Phase 1."""

import unittest.mock
import warnings

import pytest
import numpy as np
import pandas as pd

from mlsurv._warnings import MethodologicalWarning, SuggestionWarning
from tests._helpers import make_y


class TestToStructuredArray:
    """Test to_structured_array function."""

    def test_basic_conversion(self):
        """Verify basic SurvivalArray creation."""
        from mlsurv.utils import to_structured_array

        times = [1.0, 2.0, 3.0]
        events = [True, False, True]

        y = to_structured_array(times, events)

        assert y.dtype.names == ('event', 'time')
        assert y['event'].tolist() == events
        assert y['time'].tolist() == times

    def test_numpy_input(self):
        """Verify numpy array inputs work."""
        from mlsurv.utils import to_structured_array

        times = np.array([1.5, 2.5, 3.5])
        events = np.array([True, False, True])

        y = to_structured_array(times, events)

        assert y.dtype.names == ('event', 'time')
        assert len(y) == 3

    def test_pandas_input(self):
        """Verify pandas Series inputs work."""
        from mlsurv.utils import to_structured_array

        times = pd.Series([1.0, 2.0, 3.0])
        events = pd.Series([True, False, True])

        y = to_structured_array(times, events)

        assert y.dtype.names == ('event', 'time')
        assert len(y) == 3

    def test_integer_events(self):
        """Verify integer event indicators are converted to bool."""
        from mlsurv.utils import to_structured_array

        times = [1.0, 2.0, 3.0]
        events = [1, 0, 1]

        y = to_structured_array(times, events)

        assert y['event'].dtype == bool
        assert y['event'].tolist() == [True, False, True]

    def test_length_mismatch_error(self):
        """Verify error on mismatched lengths."""
        from mlsurv.utils import to_structured_array

        times = [1.0, 2.0, 3.0]
        events = [True, False]

        with pytest.raises(ValueError, match="same length"):
            to_structured_array(times, events)

    def test_negative_time_error(self):
        """Verify error on negative times with rich message."""
        from mlsurv.utils import to_structured_array

        times = [1.0, -2.0, 3.0]
        events = [True, False, True]

        with pytest.raises(ValueError, match="negative survival time"):
            to_structured_array(times, events)

    def test_competing_risks_warning(self):
        """Warn when event column has >2 unique values."""
        from mlsurv.utils import to_structured_array

        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        events = [0, 1, 2, 0, 1]
        with pytest.warns(MethodologicalWarning, match="unique values"):
            y = to_structured_array(times, events)
        # Still produces valid output (all non-zero → True)
        assert y['event'].tolist() == [False, True, True, False, True]

    def test_no_competing_risks_warning_binary(self):
        """No warning for standard binary events."""
        from mlsurv.utils import to_structured_array

        times = [1.0, 2.0, 3.0, 4.0]
        events = [0, 1, 0, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            to_structured_array(times, events)

    def test_competing_risks_warning_float_events(self):
        """Warn for float event arrays with >2 unique values."""
        from mlsurv.utils import to_structured_array

        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        events = [0.0, 1.0, 2.0, 0.0, 1.0]
        with pytest.warns(MethodologicalWarning, match="competing risks"):
            to_structured_array(times, events)

    def test_no_competing_risks_warning_bool_events(self):
        """Boolean events bypass competing risks check entirely."""
        from mlsurv.utils import to_structured_array

        times = [1.0, 2.0, 3.0]
        events = [True, False, True]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            to_structured_array(times, events)

    def test_ambiguous_binary_numeric_events_raise(self):
        """Two non-zero numeric codes must be disambiguated explicitly."""
        from mlsurv.utils import to_structured_array

        with pytest.raises(ValueError, match="ambiguous binary numeric values"):
            to_structured_array([1.0, 2.0, 3.0], [1, 2, 1])

    def test_zero_and_nonone_numeric_accepted(self):
        """[0, 2] encoding is unambiguous: 0=censored, 2=event."""
        from mlsurv.utils import to_structured_array

        y = to_structured_array([1.0, 2.0, 3.0], [0, 2, 0])
        assert y['event'].tolist() == [False, True, False]

    # --- time=0 handling ---

    def test_zero_time_error_message(self):
        """time=0 raises ValueError with count, indices, and suggestions."""
        from mlsurv.utils import to_structured_array

        times = [1.0, 0.0, 3.0]
        events = [True, False, True]

        with pytest.raises(ValueError, match=r"1 observation.*time=0.*indices: \[1\]"):
            to_structured_array(times, events)

    def test_zero_time_error_shows_options(self):
        """time=0 error message includes all three fix options."""
        from mlsurv.utils import to_structured_array

        with pytest.raises(ValueError, match="drop_nonpositive=True"):
            to_structured_array([0.0, 1.0], [True, True])

    def test_zero_time_error_distinct_from_negative(self):
        """Negative times and zero times produce different error messages."""
        from mlsurv.utils import to_structured_array

        with pytest.raises(ValueError, match="negative survival time"):
            to_structured_array([-1.0, 1.0], [True, True])

        with pytest.raises(ValueError, match="time=0"):
            to_structured_array([0.0, 1.0], [True, True])

    def test_mixed_zero_and_negative_raises_negative_first(self):
        """When both zero and negative are present, negative error fires
        but also mentions zero times."""
        from mlsurv.utils import to_structured_array

        with pytest.raises(ValueError, match="negative survival time"):
            to_structured_array([-1.0, 0.0, 1.0], [True, True, True])

    def test_mixed_zero_and_negative_mentions_both(self):
        """Mixed error mentions zero times alongside negative."""
        from mlsurv.utils import to_structured_array

        with pytest.raises(ValueError, match="Additionally.*time=0"):
            to_structured_array([-1.0, 0.0, 1.0], [True, True, True])

    def test_drop_nonpositive_zeros(self):
        """drop_nonpositive=True drops zeros and warns."""
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.utils import to_structured_array

        with pytest.warns(MethodologicalWarning, match="Dropped 1"):
            y = to_structured_array(
                [0.0, 1.0, 2.0], [True, False, True],
                drop_nonpositive=True,
            )
        assert len(y) == 2
        assert y['time'].tolist() == [1.0, 2.0]
        assert y['event'].tolist() == [False, True]

    def test_drop_nonpositive_negatives(self):
        """drop_nonpositive=True drops negative times and warns."""
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.utils import to_structured_array

        with pytest.warns(MethodologicalWarning, match="1 negative"):
            y = to_structured_array(
                [-1.0, 1.0, 2.0], [True, False, True],
                drop_nonpositive=True,
            )
        assert len(y) == 2
        assert y['time'].tolist() == [1.0, 2.0]

    def test_drop_nonpositive_mixed(self):
        """drop_nonpositive=True drops both zero and negative times."""
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.utils import to_structured_array

        with pytest.warns(MethodologicalWarning, match="Dropped 2"):
            y = to_structured_array(
                [-1.0, 0.0, 1.0, 2.0], [True, True, False, True],
                drop_nonpositive=True,
            )
        assert len(y) == 2
        assert y['time'].tolist() == [1.0, 2.0]
        assert y['event'].tolist() == [False, True]

    def test_drop_nonpositive_all_removed_raises(self):
        """drop_nonpositive=True with all non-positive raises ValueError."""
        from mlsurv.utils import to_structured_array

        with pytest.raises(ValueError, match="No data remains"):
            to_structured_array(
                [0.0, -1.0], [True, True], drop_nonpositive=True,
            )

    def test_drop_nonpositive_still_raises_on_nan(self):
        """drop_nonpositive only drops non-positive times; NaN/inf can't be
        dropped meaningfully, so they still raise."""
        from mlsurv.utils import to_structured_array

        with pytest.raises(ValueError, match="NaN"):
            to_structured_array(
                [np.nan, 1.0, 2.0], [True, False, True], drop_nonpositive=True,
            )

    def test_drop_nonpositive_still_raises_on_inf(self):
        """drop_nonpositive=True with an infinite time still raises."""
        from mlsurv.utils import to_structured_array

        with pytest.raises(ValueError, match="infinite"):
            to_structured_array(
                [np.inf, 1.0, 2.0], [True, False, True], drop_nonpositive=True,
            )

    def test_drop_nonpositive_default_is_false(self):
        """Default behavior raises on time=0, does not filter."""
        from mlsurv.utils import to_structured_array

        with pytest.raises(ValueError):
            to_structured_array([0.0, 1.0], [True, True])

    # --- Single structured-array convention (sksurv field normalizer) ---

    def test_structured_array_basic(self):
        """Single 2-field structured array normalizes to canonical fields."""
        from mlsurv.utils import to_structured_array

        y_struct = np.array(
            [(True, 12.0), (False, 5.3)],
            dtype=[('cens', bool), ('time', '<f8')],
        )
        y = to_structured_array(y_struct)

        assert y.dtype.names == ('event', 'time')
        assert y.dtype['event'] == np.bool_
        assert y['event'].tolist() == [True, False]
        assert y['time'].tolist() == [12.0, 5.3]

    def test_structured_array_field_order_independent(self):
        """Auto-detection works regardless of field order (time field first)."""
        from mlsurv.utils import to_structured_array

        y_struct = np.array(
            [(10.0, True), (20.0, False)],
            dtype=[('Survival_in_days', '<f8'), ('Status', bool)],
        )
        y = to_structured_array(y_struct)

        assert y.dtype.names == ('event', 'time')
        assert y['event'].tolist() == [True, False]
        assert y['time'].tolist() == [10.0, 20.0]

    @pytest.mark.parametrize(
        "event_name,time_name",
        [('cens', 'time'), ('e.tdm', 't.tdm'), ('Status', 'Survival_in_days')],
    )
    def test_structured_array_inconsistent_sksurv_names(self, event_name, time_name):
        """Robust to the inconsistent field names used by sksurv datasets."""
        from mlsurv.utils import to_structured_array

        y_struct = np.array(
            [(True, 3.0), (False, 7.0)],
            dtype=[(event_name, bool), (time_name, '<f8')],
        )
        y = to_structured_array(y_struct)

        assert y.dtype.names == ('event', 'time')
        assert y['event'].tolist() == [True, False]
        assert y['time'].tolist() == [3.0, 7.0]

    def test_structured_array_not_structured_raises(self):
        """A plain (non-structured) ndarray raises TypeError naming the cause."""
        from mlsurv.utils import to_structured_array

        with pytest.raises(TypeError, match="no named fields"):
            to_structured_array(np.array([1.0, 2.0, 3.0]))

    def test_structured_array_plain_list_raises(self):
        """A plain list as the only arg raises TypeError, not a length error."""
        from mlsurv.utils import to_structured_array

        with pytest.raises(TypeError, match="structured numpy array"):
            to_structured_array([1.0, 2.0, 3.0])

    @pytest.mark.parametrize("n_fields", [1, 3])
    def test_structured_array_wrong_field_count_raises(self, n_fields):
        """A structured array without exactly 2 fields raises TypeError."""
        from mlsurv.utils import to_structured_array

        dtype = [('event', bool)] + [(f'f{i}', '<f8') for i in range(n_fields - 1)]
        y_struct = np.zeros(2, dtype=dtype)

        with pytest.raises(TypeError, match="exactly 2 fields"):
            to_structured_array(y_struct)

    def test_structured_array_no_boolean_field_raises(self):
        """Zero boolean fields cannot auto-detect the event field."""
        from mlsurv.utils import to_structured_array

        y_struct = np.array(
            [(1.0, 12.0), (0.0, 5.3)],
            dtype=[('event', '<f8'), ('time', '<f8')],
        )
        with pytest.raises(TypeError, match="exactly one boolean"):
            to_structured_array(y_struct)

    def test_structured_array_two_boolean_fields_raises(self):
        """Two boolean fields are ambiguous and raise TypeError."""
        from mlsurv.utils import to_structured_array

        y_struct = np.array(
            [(True, False), (False, True)],
            dtype=[('a', bool), ('b', bool)],
        )
        with pytest.raises(TypeError, match="exactly one boolean"):
            to_structured_array(y_struct)

    def test_structured_array_shares_time_validation(self):
        """The single-arg path runs the same nonpositive-time validation."""
        from mlsurv.utils import to_structured_array

        y_struct = np.array(
            [(True, 0.0), (False, 5.0)],
            dtype=[('cens', bool), ('time', '<f8')],
        )
        with pytest.raises(ValueError, match="time=0"):
            to_structured_array(y_struct)

    def test_structured_array_shares_drop_nonpositive(self):
        """drop_nonpositive flows through the single-arg path."""
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.utils import to_structured_array

        y_struct = np.array(
            [(True, 0.0), (False, 5.0)],
            dtype=[('cens', bool), ('time', '<f8')],
        )
        with pytest.warns(MethodologicalWarning):
            y = to_structured_array(y_struct, drop_nonpositive=True)
        assert len(y) == 1
        assert y['time'].tolist() == [5.0]


class TestLoadSurvivalData:
    """Test load_survival_data function."""

    def test_basic_csv_load(self, tmp_path):
        """Verify CSV loading with time/event columns."""
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "test_data.csv"
        csv_path.write_text("feat1,feat2,time,event\n1.0,2.0,10.0,1\n3.0,4.0,5.0,0\n")

        X, y = load_survival_data(str(csv_path), time_col='time', event_col='event')

        assert X.shape == (2, 2)
        assert 'feat1' in X.columns
        assert 'feat2' in X.columns
        assert y['time'].tolist() == [10.0, 5.0]
        assert y['event'].tolist() == [True, False]

    def test_string_events(self, tmp_path):
        """Verify string event indicators are converted."""
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "test_data.csv"
        csv_path.write_text("feat1,time,event\n1.0,10.0,DECEASED\n2.0,5.0,LIVING\n")

        X, y = load_survival_data(str(csv_path), time_col='time', event_col='event')

        assert y['event'].tolist() == [True, False]

    def test_missing_time_column(self, tmp_path):
        """Verify error on missing time column."""
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "test_data.csv"
        csv_path.write_text("feat1,event\n1.0,1\n")

        with pytest.raises(ValueError, match="Time column"):
            load_survival_data(str(csv_path), time_col='time', event_col='event')

    def test_missing_event_column(self, tmp_path):
        """Verify error on missing event column."""
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "test_data.csv"
        csv_path.write_text("feat1,time\n1.0,10.0\n")

        with pytest.raises(ValueError, match="Event column"):
            load_survival_data(str(csv_path), time_col='time', event_col='event')

    def test_competing_risks_warning(self, tmp_path):
        """Warn when CSV event column has >2 unique values."""
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "competing.csv"
        csv_path.write_text(
            "feat,time,event\n1.0,10.0,0\n2.0,5.0,1\n3.0,8.0,2\n"
        )
        with pytest.warns(MethodologicalWarning, match="unique values"):
            X, y = load_survival_data(
                str(csv_path), time_col='time', event_col='event'
            )

    def test_competing_risks_no_double_warning(self, tmp_path):
        """load_survival_data emits exactly one warning, not two."""
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "competing.csv"
        csv_path.write_text(
            "feat,time,event\n1.0,10.0,0\n2.0,5.0,1\n3.0,8.0,2\n"
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            load_survival_data(
                str(csv_path), time_col='time', event_col='event'
            )
        competing = [w for w in record if "competing risks" in str(w.message)]
        assert len(competing) == 1

    def test_ambiguous_binary_numeric_requires_event_values(self, tmp_path):
        """1/2-coded status columns must be disambiguated explicitly."""
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "lung_like.csv"
        csv_path.write_text(
            "feat,time,status\n1.0,10.0,1\n2.0,5.0,2\n3.0,8.0,1\n"
        )

        with pytest.raises(ValueError, match="ambiguous binary numeric values"):
            load_survival_data(str(csv_path), time_col='time', event_col='status')

    def test_event_values_handles_1_2_encoding(self, tmp_path):
        """event_values lets Python users load 1/2-coded survival data."""
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "lung_like.csv"
        csv_path.write_text(
            "feat,time,status\n1.0,10.0,1\n2.0,5.0,2\n3.0,8.0,1\n"
        )

        X, y = load_survival_data(
            str(csv_path),
            time_col='time',
            event_col='status',
            event_values=[2],
        )

        assert X.shape == (3, 1)
        assert y['event'].tolist() == [False, True, False]
        assert y['time'].tolist() == [10.0, 5.0, 8.0]

    # --- time=0 handling in load_survival_data ---

    def test_load_data_zero_time_error(self, tmp_path):
        """CSV with time=0 rows produces rich error message."""
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "zero_time.csv"
        csv_path.write_text(
            "feat,time,event\n1.0,0.0,1\n2.0,5.0,0\n3.0,8.0,1\n"
        )

        with pytest.raises(ValueError, match="time=0"):
            load_survival_data(str(csv_path), time_col='time', event_col='event')

    def test_load_data_drop_nonpositive(self, tmp_path):
        """drop_nonpositive=True drops rows and keeps X/y aligned."""
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "zero_time.csv"
        csv_path.write_text(
            "feat,time,event\n1.0,0.0,1\n2.0,5.0,0\n3.0,8.0,1\n"
        )

        with pytest.warns(MethodologicalWarning, match="Dropped 1"):
            X, y = load_survival_data(
                str(csv_path), time_col='time', event_col='event',
                drop_nonpositive=True,
            )

        assert X.shape == (2, 1)
        assert len(y) == 2
        assert y['time'].tolist() == [5.0, 8.0]
        assert X['feat'].tolist() == [2.0, 3.0]

    def test_load_data_drop_nonpositive_nan_still_raises(self, tmp_path):
        """NaN times raise ValueError even with drop_nonpositive=True."""
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "nan_time.csv"
        csv_path.write_text(
            "feat,time,event\n1.0,,1\n2.0,5.0,0\n3.0,8.0,1\n"
        )

        with pytest.raises(ValueError, match="NaN"):
            load_survival_data(
                str(csv_path), time_col='time', event_col='event',
                drop_nonpositive=True,
            )

    def test_load_data_drop_nonpositive_single_warning(self, tmp_path):
        """drop_nonpositive=True emits exactly one MethodologicalWarning."""
        from mlsurv._warnings import MethodologicalWarning
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "zero_time.csv"
        csv_path.write_text(
            "feat,time,event\n1.0,0.0,1\n2.0,5.0,0\n3.0,8.0,1\n"
        )

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            load_survival_data(
                str(csv_path), time_col='time', event_col='event',
                drop_nonpositive=True,
            )
        meth = [w for w in record if issubclass(w.category, MethodologicalWarning)]
        assert len(meth) == 1, f"Expected 1 warning, got {len(meth)}"

    # --- Mode dispatch validation ---

    def test_unpaired_start_col_raises(self, tmp_path):
        """start_col without end_col raises ValueError."""
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "data.csv"
        csv_path.write_text("feat,event\n1.0,1\n")

        with pytest.raises(ValueError, match="together"):
            load_survival_data(str(csv_path), event_col='event',
                               start_col='start')

    def test_unpaired_end_col_raises(self, tmp_path):
        """end_col without start_col raises ValueError."""
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "data.csv"
        csv_path.write_text("feat,event\n1.0,1\n")

        with pytest.raises(ValueError, match="together"):
            load_survival_data(str(csv_path), event_col='event',
                               end_col='end')

    def test_no_mode_specified_raises(self, tmp_path):
        """No time_col/event_col/start_col/feature_names raises ValueError."""
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "data.csv"
        csv_path.write_text("feat\n1.0\n")

        with pytest.raises(ValueError, match="time_col and event_col"):
            load_survival_data(str(csv_path))

    def test_date_mode_missing_event_col_raises(self, tmp_path):
        """Date mode without event_col raises ValueError."""
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "data.csv"
        csv_path.write_text("start,end,feat\n2020-01-01,2020-07-01,1.0\n")

        with pytest.raises(ValueError, match="event_col is required"):
            load_survival_data(str(csv_path),
                               start_col='start', end_col='end')

    def test_prediction_mode_returns_dataframe(self, tmp_path):
        """Prediction mode returns DataFrame, not tuple."""
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "data.csv"
        csv_path.write_text("age,size,extra\n50,2.0,1\n60,3.0,2\n")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_survival_data(str(csv_path),
                                        feature_names=['age', 'size'])

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['age', 'size']
        assert len(result) == 2
        # Extra-column warning
        assert any('extra' in str(wn.message).lower() for wn in w)

    def test_date_mode_with_drop_cols(self, tmp_path):
        """Date mode applies drop_cols to result."""
        from mlsurv.utils import load_survival_data

        csv_path = tmp_path / "data.csv"
        pd.DataFrame({
            'feat': [1.0, 2.0],
            'extra': [10, 20],
            'start': ['2020-01-01', '2020-06-01'],
            'end': ['2020-07-01', '2021-06-01'],
            'event': [1, 0],
        }).to_csv(csv_path, index=False)

        X, y = load_survival_data(
            str(csv_path), event_col='event',
            start_col='start', end_col='end',
            drop_cols=['extra'],
        )
        assert 'extra' not in X.columns
        assert 'feat' in X.columns


class TestAutoEvalTimes:
    """Test auto_eval_times function."""

    def test_basic_eval_times(self, synthetic_data_clean):
        """Verify evaluation times are computed."""
        from mlsurv.utils import auto_eval_times

        X, y = synthetic_data_clean
        times = auto_eval_times(y)

        assert len(times) > 0
        assert times.min() > 0
        assert times.max() <= y['time'].max()

    def test_custom_n_times(self, synthetic_data_clean):
        """Verify n_times parameter works."""
        from mlsurv.utils import auto_eval_times

        X, y = synthetic_data_clean
        times = auto_eval_times(y, n_times=5)

        assert len(times) <= 5

    def test_empty_y_raises_with_name(self):
        """Empty y raises ValueError mentioning 'y'."""
        from mlsurv.utils import auto_eval_times

        y_empty = make_y([])
        with pytest.raises(ValueError, match="y has no observations"):
            auto_eval_times(y_empty)

    def test_empty_y_train_raises_with_name(self):
        """Empty y_train raises ValueError mentioning 'y_train'."""
        from mlsurv.utils import auto_eval_times

        y_ok = make_y([1, 2, 3])
        y_train_empty = make_y([])
        with pytest.raises(ValueError, match="y_train has no observations"):
            auto_eval_times(y_ok, y_train=y_train_empty)

    def test_prefers_y_train_range(self):
        """When y_train is given, eval_times are derived from its range."""
        from mlsurv.utils import auto_eval_times

        y_train = make_y([1, 2, 3, 4, 5])
        y_test = make_y([1, 5, 10, 15, 20])
        # Coverage warning expected (max 20 > 1.5 * 5); tested separately
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            times = auto_eval_times(y_test, y_train=y_train)
        assert times.min() >= 1.0
        assert times.max() < y_train['time'].max()

    def test_y_train_none_backward_compat(self):
        """auto_eval_times(y) == auto_eval_times(y, y_train=None)."""
        from mlsurv.utils import auto_eval_times

        y = make_y([1, 3, 5, 7, 9, 11, 13, 15])
        np.testing.assert_array_equal(
            auto_eval_times(y),
            auto_eval_times(y, y_train=None),
        )

    def test_y_train_makes_clip_noop(self):
        """Auto-computed times from y_train need no clipping."""
        from mlsurv.utils import auto_eval_times, clip_eval_times

        y_train = make_y([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y_test = make_y([1, 5, 10, 15, 20])
        # Coverage warning is expected (y_test max 20 > 1.5 * y_train max 10)
        # but clip_eval_times should NOT warn
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            times = auto_eval_times(y_test, y_train=y_train)
            clipped = clip_eval_times(times, y_train)
        clip_warnings = [w for w in record if "Dropped" in str(w.message)]
        assert len(clip_warnings) == 0
        np.testing.assert_array_equal(times, clipped)

    def test_coverage_warning_when_test_much_longer(self):
        """Warning fires when y_test max > 1.5 * y_train max."""
        from mlsurv.utils import auto_eval_times

        y_train = make_y([1, 2, 3, 4, 5])
        y_test = make_y([1, 5, 10, 15, 20])  # max 20 > 1.5 * 5
        with pytest.warns(MethodologicalWarning, match="IPCW metrics will only reflect"):
            auto_eval_times(y_test, y_train=y_train)

    def test_no_coverage_warning_when_similar_range(self):
        """No warning when y_test and y_train have similar time ranges."""
        from mlsurv.utils import auto_eval_times

        y_train = make_y([1, 2, 5, 8, 10])
        y_test = make_y([1, 3, 6, 9, 12])  # max 12 < 1.5 * 10
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            auto_eval_times(y_test, y_train=y_train)

    def test_no_coverage_warning_when_no_ytrain(self):
        """No coverage warning when y_train is None (backward compat)."""
        from mlsurv.utils import auto_eval_times

        y = make_y([1, 2, 5, 8, 10, 15, 20])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            auto_eval_times(y)

    def test_degenerate_single_unique_time_raises(self):
        """Single unique event time raises ValueError (IPCW needs >= 2)."""
        from mlsurv.utils import auto_eval_times

        y = make_y([5, 5, 5, 5])
        with pytest.raises(ValueError, match="unique event time"):
            auto_eval_times(y)

    def test_degenerate_single_unique_time_ytrain_raises(self):
        """Single unique event time in y_train raises ValueError."""
        from mlsurv.utils import auto_eval_times

        y_test = make_y([1, 2, 3])
        y_train = make_y([5, 5, 5])
        with pytest.raises(ValueError, match="y_train"):
            auto_eval_times(y_test, y_train=y_train)


class TestClipEvalTimes:
    """Test clip_eval_times function."""

    def test_no_clipping(self):
        """Times within range are returned unchanged, no warning issued."""
        from mlsurv.utils import clip_eval_times

        y_train = make_y([1, 5, 10])
        eval_times = np.array([2.0, 5.0, 8.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = clip_eval_times(eval_times, y_train)
        np.testing.assert_array_equal(result, eval_times)

    def test_no_clipping_deduplicates(self):
        """Duplicate eval_times are removed even when no clipping needed."""
        from mlsurv.utils import clip_eval_times

        y_train = make_y([1, 5, 10])
        eval_times = np.array([2.0, 5.0, 5.0, 8.0])
        result = clip_eval_times(eval_times, y_train)
        np.testing.assert_array_equal(result, np.array([2.0, 5.0, 8.0]))

    def test_upper_bound_clipping(self):
        """Times beyond training max are removed with a warning."""
        from mlsurv.utils import clip_eval_times

        y_train = make_y([1, 5, 10])
        eval_times = np.array([2.0, 5.0, 12.0, 15.0])
        with pytest.warns(SuggestionWarning, match="Dropped 2 eval time"):
            result = clip_eval_times(eval_times, y_train)
        np.testing.assert_array_equal(result, np.array([2.0, 5.0]))

    def test_lower_bound_clipping(self):
        """Times below training min are removed with a warning."""
        from mlsurv.utils import clip_eval_times

        y_train = make_y([5, 10, 20])
        eval_times = np.array([1.0, 3.0, 8.0, 15.0])
        with pytest.warns(SuggestionWarning, match="Dropped 2 eval time"):
            result = clip_eval_times(eval_times, y_train)
        np.testing.assert_array_equal(result, np.array([8.0, 15.0]))

    def test_all_out_of_range(self):
        """All times out of range falls back to training percentiles with warnings."""
        from mlsurv.utils import clip_eval_times

        y_train = make_y([1, 5, 10])
        eval_times = np.array([20.0, 30.0])
        with pytest.warns(UserWarning) as record:
            result = clip_eval_times(eval_times, y_train)
        messages = [str(w.message) for w in record]
        assert any("Dropped 2 eval time" in m for m in messages)
        assert any("Fewer than 2 eval_times survived" in m for m in messages)
        # Fallback produces at least 2 time points from training percentiles
        assert len(result) >= 2
        assert result.max() < y_train['time'].max()

    def test_linspace_last_resort(self):
        """When percentile fallback also fails, linspace warning fires."""
        from mlsurv.utils import clip_eval_times

        # Event times cluster at t_max — percentiles produce values at/near
        # t_max which get filtered by clipped < t_max, leaving < 2 points
        # and triggering linspace last resort.
        y_train = make_y([1, 10, 10])
        eval_times = np.array([20.0, 30.0])
        with pytest.warns(UserWarning) as record:
            result = clip_eval_times(eval_times, y_train)
        messages = [str(w.message) for w in record]
        assert any("equally-spaced interior points" in m for m in messages)
        assert len(result) >= 2
        assert result.max() < y_train['time'].max()

    def test_no_ytrain(self):
        """When y_train is None, eval_times are returned unchanged."""
        from mlsurv.utils import clip_eval_times

        eval_times = np.array([2.0, 5.0, 100.0])
        result = clip_eval_times(eval_times, None)
        np.testing.assert_array_equal(result, eval_times)

    def test_exactly_at_tmax_is_dropped(self):
        """t_max itself is excluded (strict < upper bound for sksurv)."""
        from mlsurv.utils import clip_eval_times

        y_train = make_y([1, 5, 10])
        eval_times = np.array([2.0, 5.0, 10.0])
        with pytest.warns(SuggestionWarning, match="Dropped 1 eval time"):
            result = clip_eval_times(eval_times, y_train)
        np.testing.assert_array_equal(result, np.array([2.0, 5.0]))

    def test_degenerate_tmin_equals_tmax(self):
        """When all training times are identical, raise ValueError."""
        from mlsurv.utils import clip_eval_times

        y_train = make_y([5, 5, 5])
        eval_times = np.array([2.0, 5.0, 8.0])
        original = eval_times.copy()
        with pytest.raises(ValueError, match="All training times are identical"):
            clip_eval_times(eval_times, y_train)
        # Input array must not be mutated
        np.testing.assert_array_equal(eval_times, original)

    def test_degenerate_single_training_observation(self):
        """Single training observation triggers degenerate t_min == t_max guard."""
        from mlsurv.utils import clip_eval_times

        y_train = make_y([7])
        eval_times = np.array([3.0, 7.0, 10.0])
        with pytest.raises(ValueError, match="All training times are identical"):
            clip_eval_times(eval_times, y_train)

    def test_degenerate_near_identical(self):
        """Near-identical times (within float tolerance) also raise."""
        from mlsurv.utils import clip_eval_times

        y_train = make_y([5.0, 5.0 + 1e-15, 5.0])
        eval_times = np.array([2.0, 5.0, 8.0])
        with pytest.raises(ValueError, match="All training times are identical"):
            clip_eval_times(eval_times, y_train)


class TestTruncateYToTrainRange:
    """Test truncate_y_to_train_range function."""

    def test_no_truncation_needed(self):
        """When all test times are within training range, return unchanged."""
        from mlsurv.utils import truncate_y_to_train_range

        y_train = make_y([1, 5, 10])
        y_test = make_y([2, 4, 8])
        result = truncate_y_to_train_range(y_test, y_train)
        np.testing.assert_array_equal(result['time'], y_test['time'])
        np.testing.assert_array_equal(result['event'], y_test['event'])

    def test_at_tmax_boundary_not_truncated(self):
        """Observation exactly at t_max is NOT truncated (strict >)."""
        from mlsurv.utils import truncate_y_to_train_range

        y_train = make_y([1, 5, 10])
        y_test = make_y([5, 10], events=[True, True])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = truncate_y_to_train_range(y_test, y_train)
        assert result['time'][1] == 10.0
        assert result['event'][1]  # Event preserved, not censored

    def test_truncates_beyond_max(self):
        """Observations beyond training max are censored at t_max."""
        from mlsurv.utils import truncate_y_to_train_range

        y_train = make_y([1, 5, 10])
        y_test = make_y([3, 12, 15], events=[True, True, False])
        y_test_original = y_test.copy()
        with pytest.warns(MethodologicalWarning, match="Censored 2 y_test"):
            result = truncate_y_to_train_range(y_test, y_train, warn=True)
        assert result['time'][0] == 3.0
        assert result['time'][1] == 10.0  # Clamped to t_max
        assert result['time'][2] == 10.0
        assert result['event'][0]          # Unchanged
        assert not result['event'][1]      # Was event, now censored
        assert not result['event'][2]      # Was censored, still censored
        # Input must not be mutated
        np.testing.assert_array_equal(y_test, y_test_original)

    def test_no_ytrain(self):
        """When y_train is None, y_test is returned unchanged."""
        from mlsurv.utils import truncate_y_to_train_range

        y_test = make_y([3, 12, 15])
        result = truncate_y_to_train_range(y_test, None)
        np.testing.assert_array_equal(result, y_test)

    def test_warning_on_truncation(self):
        """Warning text includes count and t_max."""
        from mlsurv.utils import truncate_y_to_train_range

        y_train = make_y([1, 5, 10])
        y_test = make_y([3, 12, 15], events=[True, True, True])
        with pytest.warns(MethodologicalWarning, match=r"Censored 2 y_test.*10\.0"):
            truncate_y_to_train_range(y_test, y_train, warn=True)

    def test_no_warning_when_no_truncation(self):
        """No warning emitted when all test times are within range."""
        from mlsurv.utils import truncate_y_to_train_range

        y_train = make_y([1, 5, 10])
        y_test = make_y([2, 4, 8])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            truncate_y_to_train_range(y_test, y_train)

    def test_warn_false_suppresses_warning(self):
        """warn=False suppresses the truncation warning."""
        from mlsurv.utils import truncate_y_to_train_range

        y_train = make_y([1, 5, 10])
        y_test = make_y([3, 12, 15], events=[True, True, False])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = truncate_y_to_train_range(y_test, y_train, warn=False)
        # Truncation still happens, just silently
        assert result['time'][1] == 10.0
        assert not result['event'][1]

    def test_warn_false_emits_logger_debug(self, caplog):
        """warn=False emits a logger.debug record instead of a UserWarning."""
        import logging
        from mlsurv.utils import truncate_y_to_train_range

        y_train = make_y([1, 5, 10])
        y_test = make_y([3, 12, 15], events=[True, True, False])
        with caplog.at_level(logging.DEBUG, logger="mlsurv.utils"):
            result = truncate_y_to_train_range(y_test, y_train, warn=False)
        assert result['time'][1] == 10.0
        assert any(
            "Truncated" in r.message and r.levelno == logging.DEBUG
            for r in caplog.records
        )


class TestCheckTruncationNeeded:
    """Test check_truncation_needed utility function."""

    def test_truncation_needed(self):
        """Returns note string when test times exceed training max."""
        from mlsurv.utils import check_truncation_needed

        y_train = make_y([1, 5, 10])
        y_test = make_y([3, 12, 15], events=[True, True, False])
        note = check_truncation_needed(y_test, y_train)
        assert note is not None
        assert "2 y_test obs" in note
        assert "t_max=10.0" in note
        assert "Uno C" in note and "AUC(t)" in note and "IBS" in note

    def test_no_truncation_needed(self):
        """Returns None when all test times are within training range."""
        from mlsurv.utils import check_truncation_needed

        y_train = make_y([1, 5, 10])
        y_test = make_y([2, 4, 8])
        assert check_truncation_needed(y_test, y_train) is None

    def test_y_train_none(self):
        """Returns None when y_train is None."""
        from mlsurv.utils import check_truncation_needed

        y_test = make_y([3, 12, 15])
        assert check_truncation_needed(y_test, None) is None

    def test_at_boundary_no_truncation_note(self):
        """Observation exactly at t_max does NOT trigger a note (strict >)."""
        from mlsurv.utils import check_truncation_needed

        y_train = make_y([1, 5, 10])
        y_test = make_y([5, 10], events=[True, True])
        assert check_truncation_needed(y_test, y_train) is None


class TestEvaluateSummaryIpcwNote:
    """Test IPCW note in Evaluate/Bootstrap boxes (only shown for custom y data)."""

    @staticmethod
    def _make_learner_mock(y_test=None):
        """Build a mock SurvivalLearner with standard y_train and state."""
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.y_train = make_y([1, 5, 10])
        mock.y_test = y_test
        mock.random_state = 42
        mock.n_jobs = 1
        return mock

    @staticmethod
    def _make_ctx(**overrides):
        """Build an EvalContext with sensible defaults for summary-box tests."""
        from mlsurv.learner._eval_context import EvalContext

        defaults = dict(
            n_importance_repeats=10,
            include_shap=False,
            shap_max_background=50,
            shap_max_explain=50,
            bootstrap=False,
            n_bootstrap=0,
            bootstrap_ci=0.95,
            random_state=42,
            n_jobs=1,
        )
        defaults.update(overrides)
        return EvalContext(**defaults)

    @staticmethod
    def _make_model_mock():
        """Build a mock model with a .name attribute."""
        from unittest.mock import MagicMock

        model = MagicMock()
        model.name = "CoxPH"
        return model

    def test_evaluate_summary_no_ipcw_note_for_default_y(self, capsys):
        """Evaluate box omits IPCW note for default y_test (Setup box shows it)."""
        from mlsurv.learner._evaluation import EvaluationMixin

        y_test = make_y([3, 12, 15], events=[True, True, False])
        EvaluationMixin._print_evaluate_summary(
            self._make_learner_mock(y_test=y_test),
            self._make_ctx(),
            resolved_model=self._make_model_mock(),
            eval_times=None,
            eval_times_was_auto=False,
            orig_random_state=42,
            orig_n_jobs=1,
            y_test=y_test,
        )
        captured = capsys.readouterr()
        assert "IPCW note" not in captured.out

    def test_evaluate_summary_shows_ipcw_note_for_custom_y(self, capsys):
        """Evaluate box shows IPCW note when custom y data exceeds training range."""
        from mlsurv.learner._evaluation import EvaluationMixin

        default_y = make_y([2, 4, 8])
        custom_y = make_y([3, 12, 15], events=[True, True, False])
        EvaluationMixin._print_evaluate_summary(
            self._make_learner_mock(y_test=default_y),
            self._make_ctx(),
            resolved_model=self._make_model_mock(),
            eval_times=None,
            eval_times_was_auto=False,
            orig_random_state=42,
            orig_n_jobs=1,
            y_test=custom_y,
        )
        captured = capsys.readouterr()
        assert "IPCW note" in captured.out
        assert "2 y_test obs" in captured.out

    def test_evaluate_summary_no_ipcw_note_when_no_truncation(self, capsys):
        """Boxed evaluate summary omits IPCW note when no truncation needed."""
        from mlsurv.learner._evaluation import EvaluationMixin

        y_test = make_y([2, 4, 8])
        EvaluationMixin._print_evaluate_summary(
            self._make_learner_mock(),
            self._make_ctx(),
            resolved_model=self._make_model_mock(),
            eval_times=None,
            eval_times_was_auto=False,
            orig_random_state=42,
            orig_n_jobs=1,
            y_test=y_test,
        )
        captured = capsys.readouterr()
        assert "IPCW note" not in captured.out

    def test_evaluate_summary_shows_resolved_auto_eval_times(self, capsys):
        """Boxed evaluate summary shows resolved times when eval_times was auto."""
        from mlsurv.learner._evaluation import EvaluationMixin

        y_test = make_y([2, 4, 8])
        eval_times = [365, 730, 1095]
        EvaluationMixin._print_evaluate_summary(
            self._make_learner_mock(y_test=y_test),
            self._make_ctx(),
            resolved_model=self._make_model_mock(),
            eval_times=eval_times,
            eval_times_was_auto=True,
            orig_random_state=42,
            orig_n_jobs=1,
            y_test=y_test,
        )
        out = capsys.readouterr().out
        assert "auto" in out and "365" in out and "1095" in out

    def test_evaluate_summary_formats_float_auto_eval_times(self, capsys):
        """Non-integral auto times render with one decimal place."""
        from mlsurv.learner._evaluation import EvaluationMixin

        y_test = make_y([2, 4, 8])
        eval_times = [1.5, 7.5, 30.5]
        EvaluationMixin._print_evaluate_summary(
            self._make_learner_mock(y_test=y_test),
            self._make_ctx(),
            resolved_model=self._make_model_mock(),
            eval_times=eval_times,
            eval_times_was_auto=True,
            orig_random_state=42,
            orig_n_jobs=1,
            y_test=y_test,
        )
        out = capsys.readouterr().out
        # Float branch: rendered with one decimal, NOT rounded to int.
        assert "1.5" in out and "7.5" in out and "30.5" in out
        # Sanity: not rendered as bare ints (would be "1, 7, 30").
        assert "1, 7, 30" not in out

    def test_evaluate_summary_summarizes_long_auto_eval_times(self, capsys):
        """Long auto-resolved time grids are summarized as 'N times in [tmin, tmax]'."""
        from mlsurv.learner._evaluation import EvaluationMixin

        y_test = make_y([2, 4, 8])
        eval_times = list(range(100, 1101, 100))  # 11 times
        EvaluationMixin._print_evaluate_summary(
            self._make_learner_mock(y_test=y_test),
            self._make_ctx(),
            resolved_model=self._make_model_mock(),
            eval_times=eval_times,
            eval_times_was_auto=True,
            orig_random_state=42,
            orig_n_jobs=1,
            y_test=y_test,
        )
        out = capsys.readouterr().out
        assert "11 times in" in out and "100" in out and "1100" in out

    def test_evaluate_summary_shows_shap_params(self, capsys):
        """Boxed evaluate summary shows SHAP=on (bg=N, explain=N) when enabled."""
        from mlsurv.learner._evaluation import EvaluationMixin

        y_test = make_y([2, 4, 8])
        EvaluationMixin._print_evaluate_summary(
            self._make_learner_mock(),
            self._make_ctx(
                include_shap=True,
                shap_max_background=30,
                shap_max_explain=20,
            ),
            resolved_model=self._make_model_mock(),
            eval_times=None,
            eval_times_was_auto=False,
            orig_random_state=42,
            orig_n_jobs=1,
            y_test=y_test,
        )
        captured = capsys.readouterr()
        assert "SHAP=on (bg=30, explain=20)" in captured.out

    def test_bootstrap_summary_no_ipcw_note_for_default_y(self, capsys):
        """Bootstrap box omits IPCW note for default y_test (Setup box shows it)."""
        from mlsurv.learner._evaluation import EvaluationMixin

        y_test = make_y([3, 12, 15], events=[True, True, False])
        EvaluationMixin._print_bootstrap_summary(
            self._make_learner_mock(y_test=y_test),
            self._make_ctx(
                bootstrap=True,
                n_bootstrap=100,
                bootstrap_metrics=["c_index", "ibs"],
            ),
            resolved_model=self._make_model_mock(),
            orig_random_state=42,
            orig_n_jobs=1,
            y_test=y_test,
        )
        captured = capsys.readouterr()
        assert "IPCW note" not in captured.out

    def test_bootstrap_summary_shows_ipcw_note_for_custom_y(self, capsys):
        """Bootstrap box shows IPCW note when custom y data exceeds training range."""
        from mlsurv.learner._evaluation import EvaluationMixin

        default_y = make_y([2, 4, 8])
        custom_y = make_y([3, 12, 15], events=[True, True, False])
        EvaluationMixin._print_bootstrap_summary(
            self._make_learner_mock(y_test=default_y),
            self._make_ctx(
                bootstrap=True,
                n_bootstrap=100,
                bootstrap_metrics=["c_index", "ibs"],
            ),
            resolved_model=self._make_model_mock(),
            orig_random_state=42,
            orig_n_jobs=1,
            y_test=custom_y,
        )
        captured = capsys.readouterr()
        assert "IPCW note" in captured.out
        assert "2 y_test obs" in captured.out


class TestIpcwCouplingInvariant:
    """Test the IPCW two-constraint invariant (clip + truncate)."""

    def test_external_validation_no_clip_warning(self):
        """Full external-validation scenario: auto_eval_times(y_test, y_train=y_train) + clip produces no clip warning."""
        from mlsurv.utils import auto_eval_times, clip_eval_times

        y_train = make_y(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            events=[True, True, False, True, True, False, True, True, False, True],
        )
        y_test = make_y(
            [2, 5, 12, 18, 25],
            events=[True, False, True, True, False],
        )
        # Coverage warning is expected (y_test max 25 > 1.5 * y_train max 10)
        # but clip_eval_times should NOT warn
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            eval_times = auto_eval_times(y_test, y_train=y_train)
            clipped = clip_eval_times(eval_times, y_train)
        clip_warnings = [w for w in record if "Dropped" in str(w.message)]
        assert len(clip_warnings) == 0
        np.testing.assert_array_equal(eval_times, clipped)


class TestValidateSurvivalData:
    """Test validate_survival_data function."""

    def test_valid_data(self, synthetic_data_clean):
        """Verify valid data passes validation."""
        from mlsurv.utils import validate_survival_data

        X, y = synthetic_data_clean
        validate_survival_data(X, y)  # Should not raise

    def test_non_dataframe_error(self, synthetic_data_clean):
        """Verify error on non-DataFrame X."""
        from mlsurv.utils import validate_survival_data

        X, y = synthetic_data_clean

        with pytest.raises(ValueError, match="DataFrame"):
            validate_survival_data(X.values, y)

    def test_length_mismatch_error(self, synthetic_data_clean):
        """Verify error on mismatched X/y length."""
        from mlsurv.utils import validate_survival_data

        X, y = synthetic_data_clean

        with pytest.raises(ValueError, match="same number"):
            validate_survival_data(X.iloc[:10], y)


class TestDropNanRows:
    """Test drop_nan_rows utility."""

    def test_no_nan_passthrough(self):
        """No NaN → returns same objects unchanged."""
        from mlsurv.utils import drop_nan_rows
        df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        y = make_y([1.0, 2.0], [True, False])
        df_out, y_out = drop_nan_rows(df, y, "test")
        assert df_out is df
        assert y_out is y

    def test_partial_nan_filters_correctly(self):
        """Partial NaN → drops affected rows, keeps alignment."""
        from mlsurv.utils import drop_nan_rows
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, 6.0]})
        y = make_y([10.0, 20.0, 30.0], [True, False, True])
        df_out, y_out = drop_nan_rows(df, y, "test")
        assert len(df_out) == 2
        assert len(y_out) == 2
        assert list(df_out.index) == [0, 1]  # reset_index applied
        np.testing.assert_array_equal(df_out['a'].values, [1.0, 3.0])
        assert y_out[0]['time'] == 10.0
        assert y_out[1]['time'] == 30.0

    def test_all_nan_returns_empty(self):
        """All NaN → returns empty df and y."""
        from mlsurv.utils import drop_nan_rows
        df = pd.DataFrame({'a': [np.nan, np.nan], 'b': [np.nan, np.nan]})
        y = make_y([1.0, 2.0], [True, False])
        df_out, y_out = drop_nan_rows(df, y, "test")
        assert len(df_out) == 0
        assert len(y_out) == 0

    def test_emits_warning_on_drop(self):
        """Dropping rows emits a logger.warning."""
        import logging
        from mlsurv.utils import drop_nan_rows
        df = pd.DataFrame({'a': [1.0, np.nan]})
        y = make_y([1.0, 2.0], [True, False])
        logger = logging.getLogger("mlsurv.utils")
        with unittest.mock.patch.object(logger, 'warning') as mock_warn:
            drop_nan_rows(df, y, "test_caller")
            mock_warn.assert_called_once()
            # Logger uses %-style formatting: first arg is template, second is caller
            assert mock_warn.call_args[0][1] == "test_caller"


class TestTrapzShim:
    """Tests for the centralized trapz/trapezoid shim."""

    def test_callable(self):
        """trapz shim is callable."""
        from mlsurv.utils import trapz
        assert callable(trapz)

    def test_correct_result(self):
        """trapz shim computes correct integral."""
        from mlsurv.utils import trapz
        y = np.array([0.0, 1.0, 2.0])
        x = np.array([0.0, 1.0, 2.0])
        assert trapz(y, x) == pytest.approx(2.0)

    def test_matches_numpy(self):
        """trapz shim matches whichever numpy function is available."""
        from mlsurv.utils import trapz
        np_trapz = getattr(np, 'trapezoid', None) or np.trapz
        y = np.array([1.0, 4.0, 9.0, 16.0])
        x = np.array([0.0, 1.0, 2.0, 3.0])
        assert trapz(y, x) == pytest.approx(np_trapz(y, x))


class TestParseEventValues:
    """Test the extracted _parse_event_values helper."""

    def test_explicit_event_values(self):
        """event_values list maps via isin()."""
        from mlsurv.utils import _parse_event_values
        s = pd.Series(['deceased', 'alive', 'deceased', 'unknown'])
        result = _parse_event_values(s, event_values=['deceased'])
        assert result.tolist() == [True, False, True, False]

    def test_string_auto_detection(self):
        """String column uses known_true/known_false mapping."""
        from mlsurv.utils import _parse_event_values
        s = pd.Series(['DECEASED', 'LIVING', 'dead', 'alive'])
        result = _parse_event_values(s)
        assert result.tolist() == [True, False, True, False]

    def test_numeric_passthrough(self):
        """Numeric column cast to bool."""
        from mlsurv.utils import _parse_event_values
        s = pd.Series([1, 0, 1, 0])
        result = _parse_event_values(s)
        assert result.tolist() == [True, False, True, False]

    def test_unknown_string_warns(self):
        """Unknown string values trigger a warning."""
        from mlsurv.utils import _parse_event_values
        s = pd.Series(['deceased', 'unknown_status'])
        with pytest.warns(MethodologicalWarning, match="Unknown event values"):
            result = _parse_event_values(s)
        assert result.tolist() == [True, False]

    def test_nan_raises(self):
        """NaN in event column raises ValueError."""
        from mlsurv.utils import _parse_event_values
        s = pd.Series([1, 0, None])
        with pytest.raises(ValueError, match="missing.*NaN"):
            _parse_event_values(s)

    def test_competing_risks_warning_numeric(self):
        """Numeric column with >2 unique values warns about competing risks."""
        from mlsurv.utils import _parse_event_values
        s = pd.Series([0, 1, 2, 0, 1])
        with pytest.warns(MethodologicalWarning, match="competing risks"):
            result = _parse_event_values(s)
        assert result.tolist() == [False, True, True, False, True]

    def test_no_warning_binary_numeric(self):
        """Binary numeric column does not warn."""
        from mlsurv.utils import _parse_event_values
        s = pd.Series([0, 1, 0, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _parse_event_values(s)

    def test_ambiguous_binary_numeric_raises(self):
        """Two non-zero numeric codes must be disambiguated explicitly."""
        from mlsurv.utils import _parse_event_values
        s = pd.Series([1, 2, 1, 2])
        with pytest.raises(ValueError, match="ambiguous binary numeric values"):
            _parse_event_values(s)

    def test_ambiguous_binary_float_raises(self):
        """Float 1.0/2.0 (CSV-loaded dtype) is also caught."""
        from mlsurv.utils import _parse_event_values
        s = pd.Series([1.0, 2.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="ambiguous binary numeric values"):
            _parse_event_values(s)

    def test_string_whitespace_stripped(self):
        """Leading/trailing whitespace in string values is stripped."""
        from mlsurv.utils import _parse_event_values
        s = pd.Series(['  deceased  ', ' alive ', 'dead', '  living  '])
        result = _parse_event_values(s)
        assert result.tolist() == [True, False, True, False]

    def test_n_recognized_as_false(self):
        """'n' is treated as censored (false)."""
        from mlsurv.utils import _parse_event_values
        s = pd.Series(['y', 'n', 'Y', 'N'])
        result = _parse_event_values(s)
        assert result.tolist() == [True, False, True, False]

    def test_empty_string_recognized_as_false(self):
        """Empty string is treated as censored (false)."""
        from mlsurv.utils import _parse_event_values
        s = pd.Series(['deceased', '', 'dead', ''])
        result = _parse_event_values(s)
        assert result.tolist() == [True, False, True, False]

    def test_cli_path_uses_consolidated(self, tmp_path):
        """load_survival_data treats 'living' as censored."""
        import csv
        from mlsurv.utils import load_survival_data

        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "event", "feat"])
            writer.writerow([10, "deceased", 1.0])
            writer.writerow([20, "living", 2.0])
            writer.writerow([30, "alive", 3.0])

        X, y = load_survival_data(str(csv_file), "time", "event")
        assert y["event"].tolist() == [True, False, False]

    def test_cli_competing_risks_no_double_warning(self, tmp_path):
        """load_survival_data emits exactly one competing-risks warning."""
        import csv
        from mlsurv.utils import load_survival_data

        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "event", "feat"])
            for val in [0, 1, 2, 0, 1]:
                writer.writerow([10, val, 1.0])

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            load_survival_data(str(csv_file), "time", "event")
        competing = [w for w in record if "competing risks" in str(w.message)]
        assert len(competing) == 1


class TestComputeSurvivalTime:
    """Test compute_survival_time function."""

    @staticmethod
    def _make_df():
        """Create a simple DataFrame with date columns."""
        return pd.DataFrame({
            'feat1': [1.0, 2.0, 3.0, 4.0],
            'start_date': ['2020-01-01', '2020-03-15', '2020-06-01', '2020-09-01'],
            'end_date': ['2020-07-01', '2021-03-15', '2021-06-01', '2021-03-01'],
            'status': ['deceased', 'alive', 'deceased', 'alive'],
        })

    def test_basic_conversion(self):
        """Dates + event → correct (X, y) with days."""
        from mlsurv.utils import compute_survival_time
        df = self._make_df()
        X, y = compute_survival_time(
            df, 'start_date', 'end_date', 'status',
            event_values=['deceased'],
        )
        assert X.shape == (4, 1)
        assert 'feat1' in X.columns
        assert 'start_date' not in X.columns
        assert 'end_date' not in X.columns
        assert 'status' not in X.columns
        assert y['event'].tolist() == [True, False, True, False]
        # 2020-01-01 to 2020-07-01 = 182 days
        assert y['time'][0] == pytest.approx(182.0)

    def test_time_unit_months(self):
        """time_unit='months' produces correct approximate values."""
        from mlsurv.utils import compute_survival_time
        df = self._make_df()
        X, y = compute_survival_time(
            df, 'start_date', 'end_date', 'status',
            time_unit='months', event_values=['deceased'],
        )
        # 182 days / 30.4375 ≈ 5.98 months
        assert y['time'][0] == pytest.approx(5.98, abs=0.1)

    def test_time_unit_years(self):
        """time_unit='years' produces correct approximate values."""
        from mlsurv.utils import compute_survival_time
        df = self._make_df()
        X, y = compute_survival_time(
            df, 'start_date', 'end_date', 'status',
            time_unit='years', event_values=['deceased'],
        )
        # 366 days (2020-03-15 to 2021-03-15, leap year) / 365.25 ≈ 1.002
        assert y['time'][1] == pytest.approx(1.002, abs=0.01)

    def test_event_values_parameter(self):
        """Custom event_values=['deceased'] correctly maps events."""
        from mlsurv.utils import compute_survival_time
        df = self._make_df()
        _, y = compute_survival_time(
            df, 'start_date', 'end_date', 'status',
            event_values=['deceased'],
        )
        assert y['event'].tolist() == [True, False, True, False]

    def test_default_event_parsing(self):
        """String labels like 'deceased'/'alive' auto-detected without event_values."""
        from mlsurv.utils import compute_survival_time
        df = self._make_df()
        _, y = compute_survival_time(df, 'start_date', 'end_date', 'status')
        assert y['event'].tolist() == [True, False, True, False]

    def test_missing_dates_raises(self):
        """NaT dates raise ValueError."""
        from mlsurv.utils import compute_survival_time
        df = self._make_df()
        df.loc[1, 'end_date'] = None
        with pytest.raises(ValueError, match="Missing dates"):
            compute_survival_time(
                df, 'start_date', 'end_date', 'status',
                event_values=['deceased'],
            )

    def test_negative_times_raises(self):
        """end < start raises ValueError."""
        from mlsurv.utils import compute_survival_time
        df = self._make_df()
        df.loc[0, 'end_date'] = '2019-01-01'  # Before start
        with pytest.raises(ValueError, match="negative survival time"):
            compute_survival_time(
                df, 'start_date', 'end_date', 'status',
                event_values=['deceased'],
            )

    def test_zero_followup_warns(self):
        """Same start/end date warns but doesn't error; epsilon applied."""
        from mlsurv.utils import compute_survival_time
        df = self._make_df()
        df.loc[0, 'end_date'] = '2020-01-01'  # Same as start
        with pytest.warns(MethodologicalWarning, match="zero follow-up.*epsilon"):
            X, y = compute_survival_time(
                df, 'start_date', 'end_date', 'status',
                event_values=['deceased'],
            )
        assert len(y) == 4
        # Zero times get a small epsilon to pass to_structured_array validation
        assert y['time'][0] > 0
        assert y['time'][0] < 0.001

    def test_drop_date_cols_true(self):
        """drop_date_cols=True removes date columns from X."""
        from mlsurv.utils import compute_survival_time
        df = self._make_df()
        X, _ = compute_survival_time(
            df, 'start_date', 'end_date', 'status',
            event_values=['deceased'], drop_date_cols=True,
        )
        assert 'start_date' not in X.columns
        assert 'end_date' not in X.columns

    def test_drop_date_cols_false(self):
        """drop_date_cols=False keeps date columns in X."""
        from mlsurv.utils import compute_survival_time
        df = self._make_df()
        X, _ = compute_survival_time(
            df, 'start_date', 'end_date', 'status',
            event_values=['deceased'], drop_date_cols=False,
        )
        assert 'start_date' in X.columns
        assert 'end_date' in X.columns

    def test_unparseable_dates_raises(self):
        """Non-date strings raise ValueError."""
        from mlsurv.utils import compute_survival_time
        df = self._make_df()
        df['start_date'] = ['not-a-date', 'also-bad', 'nope', 'invalid']
        with pytest.raises(ValueError, match="Could not parse start dates"):
            compute_survival_time(
                df, 'start_date', 'end_date', 'status',
                event_values=['deceased'],
            )

    def test_missing_column_raises(self):
        """Missing column raises ValueError."""
        from mlsurv.utils import compute_survival_time
        df = self._make_df()
        with pytest.raises(ValueError, match="Start column 'missing'"):
            compute_survival_time(
                df, 'missing', 'end_date', 'status',
                event_values=['deceased'],
            )

    def test_invalid_time_unit_raises(self):
        """Unknown time_unit raises ValueError."""
        from mlsurv.utils import compute_survival_time
        df = self._make_df()
        with pytest.raises(ValueError, match="Unknown time_unit"):
            compute_survival_time(
                df, 'start_date', 'end_date', 'status',
                time_unit='weeks', event_values=['deceased'],
            )

    def test_input_dataframe_not_mutated(self):
        """Input DataFrame is not modified by compute_survival_time."""
        from mlsurv.utils import compute_survival_time
        df = self._make_df()
        df_original = df.copy()
        compute_survival_time(
            df, 'start_date', 'end_date', 'status',
            event_values=['deceased'],
        )
        pd.testing.assert_frame_equal(df, df_original)

    def test_multi_value_event_values(self):
        """Multiple event_values map correctly via isin()."""
        from mlsurv.utils import compute_survival_time
        df = self._make_df()
        df['status'] = ['deceased', 'alive', 'dead', 'alive']
        _, y = compute_survival_time(
            df, 'start_date', 'end_date', 'status',
            event_values=['deceased', 'dead'],
        )
        assert y['event'].tolist() == [True, False, True, False]
