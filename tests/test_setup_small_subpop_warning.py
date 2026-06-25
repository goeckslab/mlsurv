"""Tests for small-subpopulation warning at setup."""

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsurv._warnings import SmallSubpopulationWarning
from mlsurv.learner import SurvivalLearner
from mlsurv.utils import to_structured_array


def _make_data(n_samples=200, n_features=5, event_rate=0.5, seed=42):
    """Create synthetic survival data."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    time = rng.exponential(10, n_samples).astype(np.float64)
    event = rng.binomial(1, event_rate, n_samples).astype(bool)
    y = to_structured_array(times=time, events=event)
    return X, y


class TestSmallSubpopulationWarningAtSetup:
    """SmallSubpopulationWarning fires during setup() for small subpopulations."""

    def test_fires_for_small_subpopulation(self):
        """Warning fires when a subpopulation has < MIN_EVENTS_WARNING events."""
        X, y = _make_data(100)
        labels = np.array(["common"] * 90 + ["rare"] * 10)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = SurvivalLearner(X, y)
            learner.setup(
                subpopulations={"grp": labels},
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )

        small_warns = [
            w for w in caught
            if issubclass(w.category, SmallSubpopulationWarning)
            and "rare" in str(w.message)
        ]
        assert len(small_warns) >= 1, (
            f"Expected SmallSubpopulationWarning for 'rare', got: "
            f"{[str(w.message) for w in caught]}"
        )

    def test_silent_for_large_subpopulations(self):
        """No warning when all subpopulations have >= MIN_EVENTS_WARNING events."""
        X, y = _make_data(400, event_rate=0.7)
        labels = np.array(["A"] * 200 + ["B"] * 200)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = SurvivalLearner(X, y)
            learner.setup(
                subpopulations={"grp": labels},
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )

        small_warns = [
            w for w in caught
            if issubclass(w.category, SmallSubpopulationWarning)
        ]
        assert len(small_warns) == 0, (
            f"Expected no SmallSubpopulationWarning, got: "
            f"{[str(w.message) for w in small_warns]}"
        )

    def test_no_warning_without_subpopulations(self):
        """No SmallSubpopulationWarning when no subpopulations are defined."""
        X, y = _make_data(100)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = SurvivalLearner(X, y)
            learner.setup(
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )

        small_warns = [
            w for w in caught
            if issubclass(w.category, SmallSubpopulationWarning)
        ]
        assert len(small_warns) == 0

    def test_warning_message_includes_group_name(self):
        """Warning message includes both the label and the group variable name."""
        # Use larger dataset so "tiny" group has enough test samples to appear
        # but still fewer than MIN_EVENTS_WARNING events
        X, y = _make_data(400, event_rate=0.5)
        labels = np.array(["common"] * 360 + ["tiny"] * 40)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = SurvivalLearner(X, y)
            learner.setup(
                subpopulations={"cancer_type": labels},
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )

        small_warns = [
            w for w in caught
            if issubclass(w.category, SmallSubpopulationWarning)
            and "tiny" in str(w.message)
        ]
        assert len(small_warns) >= 1, (
            f"Expected SmallSubpopulationWarning for 'tiny', got: "
            f"{[str(w.message) for w in caught if issubclass(w.category, SmallSubpopulationWarning)]}"
        )
        msg = str(small_warns[0].message)
        assert "cancer_type" in msg

    def test_warning_uses_correct_category(self):
        """The warning is SmallSubpopulationWarning (subclass of MethodologicalWarning)."""
        X, y = _make_data(100)
        labels = np.array(["common"] * 95 + ["tiny"] * 5)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = SurvivalLearner(X, y)
            learner.setup(
                subpopulations={"grp": labels},
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )

        small_warns = [
            w for w in caught
            if issubclass(w.category, SmallSubpopulationWarning)
        ]
        assert len(small_warns) >= 1
        assert small_warns[0].category is SmallSubpopulationWarning

    def test_multiple_groups_warn_independently(self):
        """Each group variable with small subpopulations produces its own warning."""
        X, y = _make_data(100)
        labels_a = np.array(["big"] * 95 + ["small_a"] * 5)
        labels_b = np.array(["big"] * 95 + ["small_b"] * 5)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = SurvivalLearner(X, y)
            learner.setup(
                subpopulations={"grp_a": labels_a, "grp_b": labels_b},
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )

        small_warns = [
            w for w in caught
            if issubclass(w.category, SmallSubpopulationWarning)
        ]
        messages = [str(w.message) for w in small_warns]
        assert any("small_a" in m for m in messages), (
            f"Expected warning for 'small_a', got: {messages}"
        )
        assert any("small_b" in m for m in messages), (
            f"Expected warning for 'small_b', got: {messages}"
        )

    def test_presplit_data_also_warns(self):
        """Warning fires for pre-split data with subpopulations_test."""
        X, y = _make_data(200)
        X_train, X_test = X.iloc[:140], X.iloc[140:]
        y_train, y_test = y[:140], y[140:]
        # 60 test samples, "rare" has 5
        test_labels = np.array(["common"] * 55 + ["rare"] * 5)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = SurvivalLearner(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
            )
            learner.setup(
                subpopulations_test={"grp": test_labels},
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )

        small_warns = [
            w for w in caught
            if issubclass(w.category, SmallSubpopulationWarning)
            and "rare" in str(w.message)
        ]
        assert len(small_warns) >= 1, (
            f"Expected SmallSubpopulationWarning for 'rare' (presplit), got: "
            f"{[str(w.message) for w in caught]}"
        )

    @pytest.mark.integration
    def test_run_does_not_duplicate_setup_warning(self):
        """setup() + run() emits SmallSubpopulationWarning exactly once."""
        import io
        import contextlib

        X, y = _make_data(200)
        X_train, X_test = X.iloc[:140], X.iloc[140:]
        y_train, y_test = y[:140], y[140:]
        test_labels = np.array(["common"] * 55 + ["rare"] * 5)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = SurvivalLearner(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
            )
            learner.setup(
                subpopulations_test={"grp": test_labels},
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )
            assert learner._state.subpop_test_warned is True
            # Capture stdout from run() to also assert the verbose per-label
            # printout was dropped.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                learner.run(
                    models=["coxph"],
                    n_trials=0,
                    n_bootstrap=2,
                    n_importance_repeats=0,
                    show_eval_progress=False,
                    verbose=True,
                )

        rare_warns = [
            w for w in caught
            if issubclass(w.category, SmallSubpopulationWarning)
            and "rare" in str(w.message)
        ]
        assert len(rare_warns) == 1, (
            f"Expected exactly one SmallSubpopulationWarning across setup+run, "
            f"got {len(rare_warns)}: {[str(w.message) for w in rare_warns]}"
        )
        # The per-label '! Subpopulation' verbose printout was dropped.
        stdout = buf.getvalue()
        assert "! Subpopulation" not in stdout, (
            f"Verbose per-label printout should be gone; stdout had:\n{stdout}"
        )

    @pytest.mark.integration
    def test_subpopulation_result_warnings_still_populated(self):
        """SubpopulationResult.warnings stays populated even when the
        aggregate warning is suppressed on the evaluate-storage path."""
        X, y = _make_data(200)
        X_train, X_test = X.iloc[:140], X.iloc[140:]
        y_train, y_test = y[:140], y[140:]
        test_labels = np.array(["common"] * 55 + ["rare"] * 5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner = SurvivalLearner(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
            )
            learner.setup(
                subpopulations_test={"grp": test_labels},
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )
            learner.run(
                models=["coxph"],
                n_trials=0,
                n_bootstrap=2,
                n_importance_repeats=0,
                show_eval_progress=False,
                verbose=False,
            )

        sg = learner._state.subpopulation_results
        assert "grp" in sg
        warn_msgs = sg["grp"].warnings
        assert any("rare" in m for m in warn_msgs), (
            f"Expected 'rare' in SubpopulationResult.warnings, got: {warn_msgs}"
        )

    def test_setup_rerun_resets_flag(self):
        """Re-calling setup() must reset subpop_test_warned so a later
        run() does not silently inherit a stale-True flag."""
        # 400 samples at high event rate so 120 test rows can be split into
        # two subpops with >= MIN_EVENTS_WARNING (30) events each.
        X, y = _make_data(400, event_rate=0.95)
        X_train, X_test = X.iloc[:280], X.iloc[280:]
        y_train, y_test = y[:280], y[280:]
        small_labels = np.array(["common"] * 115 + ["rare"] * 5)
        healthy_labels = np.array(["A"] * 60 + ["B"] * 60)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner = SurvivalLearner(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
            )
            # First setup: small subpops -> flag True
            learner.setup(
                subpopulations_test={"grp": small_labels},
                random_state=42, ph_test=False, linearity_test=False,
            )
            assert learner._state.subpop_test_warned is True

            # Re-setup with no subpops -> flag must reset to False
            learner.setup(
                random_state=42, ph_test=False, linearity_test=False,
            )
            assert learner._state.subpop_test_warned is False

            # Re-setup with healthy subpops (>= 30 events each) -> stays False
            learner.setup(
                subpopulations_test={"grp": healthy_labels},
                random_state=42, ph_test=False, linearity_test=False,
            )
            assert learner._state.subpop_test_warned is False

    @pytest.mark.integration
    def test_mixed_groups_dedup_preserves_warnings_field(self):
        """When only some test subpopulation groups have small cells,
        setup() warns for those groups and run() does not re-warn for any
        group, but SubpopulationResult.warnings stays populated for the
        small-cell group."""
        X, y = _make_data(200)
        X_train, X_test = X.iloc[:140], X.iloc[140:]
        y_train, y_test = y[:140], y[140:]
        # Group A has a small "rare" subpop; group B is healthy (one label
        # only, all 60 test rows).
        a_labels = np.array(["common"] * 55 + ["rare"] * 5)
        b_labels = np.array(["healthy"] * 60)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = SurvivalLearner(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
            )
            learner.setup(
                subpopulations_test={"A": a_labels, "B": b_labels},
                random_state=42, ph_test=False, linearity_test=False,
            )
            learner.run(
                models=["coxph"], n_trials=0, n_bootstrap=2,
                n_importance_repeats=0, show_eval_progress=False, verbose=False,
            )

        rare_warns = [
            w for w in caught
            if issubclass(w.category, SmallSubpopulationWarning)
            and "rare" in str(w.message)
        ]
        assert len(rare_warns) == 1, (
            f"Expected exactly one SmallSubpopulationWarning across setup+run "
            f"for the small group, got {len(rare_warns)}: "
            f"{[str(w.message) for w in rare_warns]}"
        )
        sg = learner._state.subpopulation_results
        assert "A" in sg and "B" in sg
        assert any("rare" in m for m in sg["A"].warnings), (
            f"Expected 'rare' in A.warnings, got: {sg['A'].warnings}"
        )

    @pytest.mark.integration
    def test_validate_still_warns_on_external_cohort(self):
        """validate() with small external subpops still emits the
        SmallSubpopulationWarning even after setup+run already ran."""
        # setup with healthy training-side subpops (no test subpops -> setup
        # does not warn and does not set subpop_test_warned)
        X, y = _make_data(400, event_rate=0.7)
        X_train, X_test = X.iloc[:300], X.iloc[300:]
        y_train, y_test = y[:300], y[300:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner = SurvivalLearner(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
            )
            learner.setup(
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )
            learner.run(
                models=["coxph"],
                n_trials=0,
                n_bootstrap=2,
                n_importance_repeats=0,
                show_eval_progress=False,
                verbose=False,
            )

        # External cohort with small subpops
        X_ext, y_ext = _make_data(80, seed=7)
        ext_labels = np.array(["common"] * 75 + ["rare_ext"] * 5)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner.validate(
                X_ext, y_ext,
                name="external",
                subpopulations={"grp": ext_labels},
                verbose=False,
                bootstrap=False,
                show_progress=False,
            )

        ext_warns = [
            w for w in caught
            if issubclass(w.category, SmallSubpopulationWarning)
            and "rare_ext" in str(w.message)
        ]
        assert len(ext_warns) >= 1, (
            f"validate() should still warn for external small subpops, got: "
            f"{[str(w.message) for w in caught if issubclass(w.category, SmallSubpopulationWarning)]}"
        )

    def test_nan_labels_excluded_from_count(self):
        """NaN labels are excluded; warning based on non-NaN events only."""
        X, y = _make_data(100)
        # 80 common + 5 rare + 15 NaN — NaN samples should not count
        labels = np.array(
            ["common"] * 80 + ["rare"] * 5 + [None] * 15, dtype=object,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            learner = SurvivalLearner(X, y)
            learner.setup(
                subpopulations={"grp": labels},
                random_state=42,
                ph_test=False,
                linearity_test=False,
            )

        small_warns = [
            w for w in caught
            if issubclass(w.category, SmallSubpopulationWarning)
            and "rare" in str(w.message)
        ]
        assert len(small_warns) >= 1, (
            f"Expected SmallSubpopulationWarning for 'rare' with NaN labels, "
            f"got: {[str(w.message) for w in caught if issubclass(w.category, SmallSubpopulationWarning)]}"
        )
