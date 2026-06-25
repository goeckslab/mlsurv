"""Tests for categorical group permutation importance."""

import numpy as np
import pandas as pd
import pytest

from mlsurv.preprocessing.utils import detect_categorical_groups
from mlsurv._warnings import SuggestionWarning
from mlsurv.evaluation.importance import (
    _validate_categorical_groups,
    _build_permutation_units,
    _grouped_permutation_importance,
)


# ---------------------------------------------------------------------------
# detect_categorical_groups
# ---------------------------------------------------------------------------


class TestDetectCategoricalGroups:
    """Tests for auto-detection of one-hot encoded groups."""

    def test_prefix_detection(self):
        """Standard pd.get_dummies output is detected."""
        df = pd.DataFrame({
            "age": [30, 40, 50, 60],
            "chapter_A": [1, 0, 0, 0],
            "chapter_B": [0, 1, 0, 0],
            "chapter_C": [0, 0, 1, 1],
        })
        result = detect_categorical_groups(df)
        assert "chapter" in result
        assert set(result["chapter"]) == {"chapter_A", "chapter_B", "chapter_C"}

    def test_suffix_detection(self):
        """Suffix-encoded columns (value_feature) are detected."""
        df = pd.DataFrame({
            "age": [30, 40, 50, 60],
            "A_type": [1, 0, 0, 0],
            "B_type": [0, 1, 0, 0],
            "C_type": [0, 0, 1, 0],
        })
        result = detect_categorical_groups(df)
        assert "type" in result
        assert set(result["type"]) == {"A_type", "B_type", "C_type"}

    def test_drop_first_encoding(self):
        """drop_first=True encoding (at most one is 1, rows can be all-zero)."""
        df = pd.DataFrame({
            "age": [30, 40, 50, 60],
            "chapter_B": [1, 0, 0, 0],
            "chapter_C": [0, 1, 0, 0],
            # All-zero rows represent the dropped category A
        })
        result = detect_categorical_groups(df)
        assert "chapter" in result
        assert set(result["chapter"]) == {"chapter_B", "chapter_C"}

    def test_multi_separator(self):
        """Columns with multiple separators use prefix correctly."""
        df = pd.DataFrame({
            "tumor_stage_I": [1, 0, 0, 0],
            "tumor_stage_II": [0, 1, 0, 0],
            "tumor_stage_III": [0, 0, 1, 0],
            "tumor_stage_IV": [0, 0, 0, 1],
        })
        result = detect_categorical_groups(df)
        assert "tumor_stage" in result
        assert len(result["tumor_stage"]) == 4

    def test_no_categoricals(self):
        """No binary column groups -> empty dict."""
        df = pd.DataFrame({
            "age": [30, 40, 50],
            "weight": [70, 80, 90],
            "height": [170, 180, 190],
        })
        assert detect_categorical_groups(df) == {}

    def test_single_binary_not_grouped(self):
        """A single binary column with a prefix but no pair is not grouped."""
        df = pd.DataFrame({
            "age": [30, 40, 50],
            "is_male": [0, 1, 1],
        })
        assert detect_categorical_groups(df) == {}

    def test_mutual_exclusivity_rejection(self):
        """Binary columns sharing prefix but not mutually exclusive are rejected."""
        df = pd.DataFrame({
            "flag_a": [1, 1, 0, 0],
            "flag_b": [0, 1, 0, 1],  # Both flag_a and flag_b can be 1
        })
        result = detect_categorical_groups(df)
        assert "flag" not in result

    def test_non_binary_columns_ignored(self):
        """Non-binary columns sharing a prefix are not grouped."""
        df = pd.DataFrame({
            "score_math": [85, 90, 75],
            "score_english": [70, 80, 95],
        })
        assert detect_categorical_groups(df) == {}

    def test_empty_dataframe(self):
        """Empty DataFrame -> empty dict."""
        df = pd.DataFrame()
        assert detect_categorical_groups(df) == {}

    def test_custom_separator(self):
        """Custom separator works."""
        df = pd.DataFrame({
            "chapter.A": [1, 0, 0],
            "chapter.B": [0, 1, 0],
            "chapter.C": [0, 0, 1],
        })
        result = detect_categorical_groups(df, sep=".")
        assert "chapter" in result

    def test_collision_with_non_binary_column(self):
        """If 'chapter' exists as a non-binary column, its prefix group is excluded."""
        df = pd.DataFrame({
            "chapter": ["A", "B", "C"],  # Non-binary column named 'chapter'
            "chapter_A": [1, 0, 0],
            "chapter_B": [0, 1, 0],
            "chapter_C": [0, 0, 1],
        })
        result = detect_categorical_groups(df)
        assert "chapter" not in result

    def test_multiple_groups(self):
        """Multiple one-hot groups detected simultaneously."""
        df = pd.DataFrame({
            "age": [30, 40, 50, 60],
            "sex_M": [1, 0, 1, 0],
            "sex_F": [0, 1, 0, 1],
            "stage_I": [1, 0, 0, 0],
            "stage_II": [0, 1, 0, 0],
            "stage_III": [0, 0, 1, 1],
        })
        result = detect_categorical_groups(df)
        assert "sex" in result
        assert "stage" in result
        assert len(result["sex"]) == 2
        assert len(result["stage"]) == 3

    def test_with_nan_values(self):
        """NaN values in binary columns don't break detection."""
        df = pd.DataFrame({
            "chapter_A": [1, 0, np.nan, 0],
            "chapter_B": [0, 1, 0, 0],
            "chapter_C": [0, 0, 0, 1],
        })
        result = detect_categorical_groups(df)
        assert "chapter" in result

    def test_boolean_columns(self):
        """Boolean-typed columns treated as binary."""
        df = pd.DataFrame({
            "flag_a": [True, False, False],
            "flag_b": [False, True, False],
            "flag_c": [False, False, True],
        })
        result = detect_categorical_groups(df)
        assert "flag" in result


# ---------------------------------------------------------------------------
# _validate_categorical_groups
# ---------------------------------------------------------------------------


class TestValidateCategoricalGroups:
    """Tests for validation of categorical_groups dict."""

    def test_valid_groups(self):
        cols = ["age", "chapter_A", "chapter_B", "stage_I", "stage_II"]
        groups = {
            "chapter": ["chapter_A", "chapter_B"],
            "stage": ["stage_I", "stage_II"],
        }
        result = _validate_categorical_groups(groups, cols)
        assert result == groups

    def test_not_a_dict(self):
        with pytest.raises(TypeError, match="must be a dict"):
            _validate_categorical_groups(["a", "b"], ["a", "b"])

    def test_column_in_multiple_groups(self):
        with pytest.raises(ValueError, match="multiple"):
            _validate_categorical_groups(
                {"g1": ["a", "b"], "g2": ["b", "c"]},
                ["a", "b", "c"],
            )

    def test_group_name_collides_with_column(self):
        with pytest.raises(ValueError, match="collide"):
            _validate_categorical_groups(
                {"age": ["chapter_A", "chapter_B"]},
                ["age", "chapter_A", "chapter_B"],
            )

    def test_absent_columns_warn(self):
        with pytest.warns(SuggestionWarning, match="absent"):
            result = _validate_categorical_groups(
                {"g": ["a", "b", "c"]},
                ["a", "b"],  # "c" is absent
            )
        assert result == {"g": ["a", "b"]}

    def test_all_absent_drops_group(self):
        with pytest.warns(SuggestionWarning, match="all columns"):
            result = _validate_categorical_groups(
                {"g": ["x", "y"]},
                ["a", "b"],
            )
        assert result == {}

    def test_single_column_remaining(self):
        """Group with only 1 column after filtering is treated as ungrouped."""
        with pytest.warns(SuggestionWarning, match="only 1 column"):
            result = _validate_categorical_groups(
                {"g": ["a", "b"]},
                ["a"],  # only "a" present
            )
        assert result == {}


# ---------------------------------------------------------------------------
# _build_permutation_units
# ---------------------------------------------------------------------------


class TestBuildPermutationUnits:
    """Tests for permutation unit construction."""

    def test_no_groups(self):
        units = _build_permutation_units(
            ["a", "b", "c"], {}
        )
        assert [(n, idx) for n, idx in units] == [
            ("a", [0]), ("b", [1]), ("c", [2]),
        ]

    def test_group_replaces_members(self):
        units = _build_permutation_units(
            ["age", "chapter_A", "chapter_B", "weight"],
            {"chapter": ["chapter_A", "chapter_B"]},
        )
        names = [n for n, _ in units]
        assert names == ["age", "chapter", "weight"]

    def test_group_indices_correct(self):
        units = _build_permutation_units(
            ["age", "chapter_A", "chapter_B", "weight"],
            {"chapter": ["chapter_A", "chapter_B"]},
        )
        chapter_unit = [(n, idx) for n, idx in units if n == "chapter"][0]
        assert chapter_unit[1] == [1, 2]


# ---------------------------------------------------------------------------
# _grouped_permutation_importance
# ---------------------------------------------------------------------------


class TestGroupedPermutationImportance:
    """Tests for the grouped permutation importance computation."""

    @pytest.fixture
    def synthetic_grouped_data(self):
        """Create data with a known one-hot encoded feature."""
        rng = np.random.RandomState(42)
        n = 100
        age = rng.normal(50, 10, n)
        # One-hot encoded 'stage': 3 categories
        stage = rng.choice(3, n)
        stage_I = (stage == 0).astype(float)
        stage_II = (stage == 1).astype(float)
        stage_III = (stage == 2).astype(float)
        noise = rng.normal(0, 1, n)

        X = pd.DataFrame({
            "age": age,
            "stage_I": stage_I,
            "stage_II": stage_II,
            "stage_III": stage_III,
            "noise": noise,
        })

        # Survival outcome depends on age and stage
        risk = 0.05 * age + 0.5 * (stage == 2).astype(float)
        time = rng.exponential(1.0 / np.exp(risk * 0.1))
        event = rng.random(n) < 0.7
        y = np.array(
            [(e, t) for e, t in zip(event, time)],
            dtype=[("event", bool), ("time", float)],
        )
        return X, y

    def test_result_shape(self, synthetic_grouped_data):
        """Result has correct number of features (groups collapsed)."""
        X, y = synthetic_grouped_data
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        model = CoxPHSurvivalAnalysis().fit(X, y)

        groups = {"stage": ["stage_I", "stage_II", "stage_III"]}
        result = _grouped_permutation_importance(
            model, X, y, groups, n_repeats=5, random_state=42,
        )
        # 5 original columns -> 3 result features (age, stage, noise)
        assert len(result.feature_names) == 3
        assert result.importances.shape == (3, 5)
        assert "stage" in result.feature_names
        assert "age" in result.feature_names
        assert "noise" in result.feature_names

    def test_reproducibility(self, synthetic_grouped_data):
        """Same random_state produces identical results."""
        X, y = synthetic_grouped_data
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        model = CoxPHSurvivalAnalysis().fit(X, y)

        groups = {"stage": ["stage_I", "stage_II", "stage_III"]}
        r1 = _grouped_permutation_importance(
            model, X, y, groups, n_repeats=5, random_state=42,
        )
        r2 = _grouped_permutation_importance(
            model, X, y, groups, n_repeats=5, random_state=42,
        )
        np.testing.assert_array_equal(r1.importances, r2.importances)

    def test_no_groups_matches_ungrouped(self, synthetic_grouped_data):
        """With empty groups, result matches standard per-column permutation."""
        X, y = synthetic_grouped_data
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        model = CoxPHSurvivalAnalysis().fit(X, y)

        result = _grouped_permutation_importance(
            model, X, y, {}, n_repeats=5, random_state=42,
        )
        assert len(result.feature_names) == 5
        assert result.feature_names == X.columns.tolist()


# ---------------------------------------------------------------------------
# SHAP value grouping tests
# ---------------------------------------------------------------------------

from mlsurv.evaluation._shap import _group_shap_values
from mlsurv.evaluation._explain import _group_patient_shap


def _make_shap_df(with_model_type=False):
    """Build a synthetic long-format SHAP DataFrame for testing."""
    rows = []
    for sample in range(3):
        # Ungrouped feature
        rows.append({
            'feature': 'age', 'sample_idx': sample,
            'shap_value': 0.1 * (sample + 1), 'feature_value': 40.0 + sample,
        })
        # Dummy columns for "tgrade" group
        rows.append({
            'feature': 'tgrade_II', 'sample_idx': sample,
            'shap_value': 0.05, 'feature_value': 1.0,
        })
        rows.append({
            'feature': 'tgrade_III', 'sample_idx': sample,
            'shap_value': -0.03, 'feature_value': 0.0,
        })
    df = pd.DataFrame(rows)
    if with_model_type:
        df['model_type'] = 'coxph'
    return df


class TestShapGrouping:
    """Tests for _group_shap_values()."""

    groups = {"tgrade": ["tgrade_II", "tgrade_III"]}

    def test_sums_correctly(self):
        """SHAP values for dummy columns are summed per group per sample."""
        df = _make_shap_df()
        result = _group_shap_values(df, self.groups)

        tgrade_rows = result[result['feature'] == 'tgrade']
        assert len(tgrade_rows) == 3  # one per sample
        for sample in range(3):
            row = tgrade_rows[tgrade_rows['sample_idx'] == sample]
            assert len(row) == 1
            np.testing.assert_almost_equal(
                row['shap_value'].iloc[0], 0.05 + (-0.03),
            )

    def test_preserves_ungrouped(self):
        """Ungrouped features pass through unchanged."""
        df = _make_shap_df()
        result = _group_shap_values(df, self.groups)

        age_rows = result[result['feature'] == 'age']
        assert len(age_rows) == 3
        for sample in range(3):
            row = age_rows[age_rows['sample_idx'] == sample]
            expected = 0.1 * (sample + 1)
            np.testing.assert_almost_equal(row['shap_value'].iloc[0], expected)
            np.testing.assert_almost_equal(row['feature_value'].iloc[0], 40.0 + sample)

    def test_feature_value_shows_dummy_breakdown(self):
        """Grouped features show all dummy values as a summary string."""
        df = _make_shap_df()
        result = _group_shap_values(df, self.groups)

        tgrade_rows = result[result['feature'] == 'tgrade']
        # All samples have tgrade_II=1, tgrade_III=0
        for _, row in tgrade_rows.iterrows():
            fv = row['feature_value']
            assert isinstance(fv, str), f"Expected string, got {type(fv)}"
            assert 'tgrade_II' in fv
            assert 'tgrade_III' in fv

    def test_with_model_type(self):
        """Multi-model grouping preserves model_type column."""
        df = _make_shap_df(with_model_type=True)
        result = _group_shap_values(df, self.groups)

        assert 'model_type' in result.columns
        assert (result['model_type'] == 'coxph').all()
        tgrade_rows = result[result['feature'] == 'tgrade']
        assert len(tgrade_rows) == 3

    def test_multi_model_grouped_independently(self):
        """Each model's SHAP values are grouped independently."""
        df1 = _make_shap_df(with_model_type=True)  # model_type='coxph'
        df2 = _make_shap_df(with_model_type=True)
        df2['model_type'] = 'rsf'
        df2['shap_value'] = df2['shap_value'] * 2  # different SHAP values
        combined = pd.concat([df1, df2], ignore_index=True)

        result = _group_shap_values(combined, self.groups)

        # Each model should have its own grouped tgrade rows
        cox_tgrade = result[(result['model_type'] == 'coxph') & (result['feature'] == 'tgrade')]
        rsf_tgrade = result[(result['model_type'] == 'rsf') & (result['feature'] == 'tgrade')]
        assert len(cox_tgrade) == 3
        assert len(rsf_tgrade) == 3

        # Values should differ (rsf = 2x coxph)
        for sample in range(3):
            cox_val = cox_tgrade[cox_tgrade['sample_idx'] == sample]['shap_value'].iloc[0]
            rsf_val = rsf_tgrade[rsf_tgrade['sample_idx'] == sample]['shap_value'].iloc[0]
            np.testing.assert_almost_equal(rsf_val, cox_val * 2)

    def test_no_matching_groups(self):
        """Groups with no matching columns return original DataFrame."""
        df = _make_shap_df()
        result = _group_shap_values(df, {"nonexistent": ["a", "b"]})
        pd.testing.assert_frame_equal(result, df)

    def test_additivity_preserved(self):
        """Total sum of SHAP values is preserved after grouping."""
        df = _make_shap_df()
        result = _group_shap_values(df, self.groups)

        np.testing.assert_almost_equal(
            df['shap_value'].sum(),
            result['shap_value'].sum(),
        )

    def test_dummy_columns_removed(self):
        """Individual dummy column names do not appear in result."""
        df = _make_shap_df()
        result = _group_shap_values(df, self.groups)

        assert 'tgrade_II' not in result['feature'].values
        assert 'tgrade_III' not in result['feature'].values
        assert 'tgrade' in result['feature'].values

    def test_same_columns_as_input(self):
        """Result has exactly the same columns as input."""
        df = _make_shap_df()
        result = _group_shap_values(df, self.groups)
        assert list(result.columns) == list(df.columns)

        df_mt = _make_shap_df(with_model_type=True)
        result_mt = _group_shap_values(df_mt, self.groups)
        assert list(result_mt.columns) == list(df_mt.columns)


class TestPatientShapGrouping:
    """Tests for _group_patient_shap()."""

    groups = {"tgrade": ["tgrade_II", "tgrade_III"]}

    def _make_patient_df(self):
        return pd.DataFrame({
            "feature": ["age", "tgrade_II", "tgrade_III", "size"],
            "feature_value": [55.0, 1.0, 0.0, 3.2],
            "shap_value": [0.5, 0.2, -0.1, -0.3],
            "abs_shap_value": [0.5, 0.2, 0.1, 0.3],
        })

    def test_grouped(self):
        """Patient SHAP contributions are summed for groups."""
        df = self._make_patient_df()
        result = _group_patient_shap(df, self.groups)

        assert 'tgrade' in result['feature'].values
        assert 'tgrade_II' not in result['feature'].values
        assert 'tgrade_III' not in result['feature'].values

        tgrade_row = result[result['feature'] == 'tgrade']
        np.testing.assert_almost_equal(tgrade_row['shap_value'].iloc[0], 0.1)

    def test_abs_recomputed(self):
        """abs_shap_value = abs(summed_shap), not sum of abs."""
        df = self._make_patient_df()
        result = _group_patient_shap(df, self.groups)

        tgrade_row = result[result['feature'] == 'tgrade']
        np.testing.assert_almost_equal(tgrade_row['abs_shap_value'].iloc[0], 0.1)

    def test_feature_value_is_active_category(self):
        """Grouped features show the active category's dummy name."""
        df = self._make_patient_df()
        result = _group_patient_shap(df, self.groups)

        tgrade_row = result[result['feature'] == 'tgrade']
        # tgrade_II has feature_value=1.0 (active), so active category is "tgrade_II"
        assert tgrade_row['feature_value'].iloc[0] == 'tgrade_II'

    def test_ungrouped_preserved(self):
        """Ungrouped features are unchanged."""
        df = self._make_patient_df()
        result = _group_patient_shap(df, self.groups)

        age_row = result[result['feature'] == 'age']
        np.testing.assert_almost_equal(age_row['shap_value'].iloc[0], 0.5)
        np.testing.assert_almost_equal(age_row['feature_value'].iloc[0], 55.0)
