"""Tests for tuning auto-configuration and per-model auto-defaults.

Covers auto_n_trials(), auto_n_bootstrap(), auto_cv_folds() (in
mlsurv.tuning.search_spaces) and the per-model auto-defaults
(auto_deepsurv_defaults, auto_rsf_defaults, auto_gbsa_defaults,
auto_deephit_defaults, auto_weibull_aft_defaults,
in mlsurv.models._auto_defaults).
Wrong defaults cause poor model performance or training failures.
"""

import warnings

import pytest

from mlsurv.models._auto_defaults import (
    auto_deepsurv_defaults,
    auto_rsf_defaults,
    auto_gbsa_defaults,
    auto_deephit_defaults,
    auto_weibull_aft_defaults,
)
from mlsurv.tuning.search_spaces import (
    DEFAULT_SPACES,
    auto_n_trials,
    auto_n_bootstrap,
    auto_cv_folds,
    get_search_space,
    get_preprocessing_defaults,
)


class TestAutoNTrials:
    """Tests for auto_n_trials() trial count computation."""

    def test_known_models(self):
        """Each known model should return its base trial count."""
        assert auto_n_trials('coxph') == 20
        assert auto_n_trials('coxnet') == 30
        assert auto_n_trials('rsf') == 50
        assert auto_n_trials('deepsurv') == 60

    def test_deephit_auto_n_trials(self):
        """DeepHit should return 75 trials (10 params)."""
        assert auto_n_trials('deephit') == 75

    def test_coxnet_alpha_index_range_matches_n_alphas(self):
        """alpha_index upper bound should match CoxNetModel.n_alphas default - 1."""
        from mlsurv.models.coxnet import CoxNetModel
        space = DEFAULT_SPACES['coxnet']
        assert 'alpha_index' in space
        low, high = space['alpha_index']
        assert low == 0
        assert high == CoxNetModel().n_alphas - 1

    def test_unknown_model_default(self):
        """Unknown model should return 50."""
        assert auto_n_trials('unknown_model') == 50

    def test_high_dimensional_scaling(self):
        """High-dimensional data (>100 features) should scale up trials."""
        base = auto_n_trials('rsf')
        scaled = auto_n_trials('rsf', n_features=150)
        assert scaled == int(base * 1.5)

    def test_normal_features_no_scaling(self):
        """Normal feature count (<=100) should not scale trials."""
        assert auto_n_trials('rsf', n_features=50) == auto_n_trials('rsf')


class TestAutoNBootstrap:
    """Tests for auto_n_bootstrap() iteration count formula."""

    def test_small_sample_floors_at_100(self):
        assert auto_n_bootstrap(1) == 100

    def test_large_sample_caps_at_500(self):
        assert auto_n_bootstrap(100000) == 500

    def test_medium_sample_scales(self):
        # 10 * sqrt(2500) = 10 * 50 = 500
        assert auto_n_bootstrap(2500) == 500

    def test_returns_int(self):
        result = auto_n_bootstrap(200)
        assert isinstance(result, int)


class TestAutoCvFolds:
    """Tests for auto_cv_folds() fold count based on event count.

    Too many folds with few events leads to empty folds. Too few folds
    with many events wastes statistical power.
    """

    def test_many_events_10_folds(self):
        assert auto_cv_folds(100) == 10
        assert auto_cv_folds(500) == 10

    def test_moderate_events_5_folds(self):
        assert auto_cv_folds(50) == 5
        assert auto_cv_folds(99) == 5

    def test_few_events_3_folds(self):
        assert auto_cv_folds(20) == 3
        assert auto_cv_folds(49) == 3

    def test_very_few_events_warns(self):
        """Very few events should warn about instability."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = auto_cv_folds(10)
            assert result == 3
            assert len(w) == 1
            assert "Only 10 events" in str(w[0].message)


class TestAutoDeepsurvDefaults:
    """Tests for auto_deepsurv_defaults() data-adaptive hyperparameters."""

    def test_small_dataset_low_epv(self):
        """Low EPV dataset should get high dropout and learning rate."""
        # EPV = 30/10 = 3, which is < 10
        defaults = auto_deepsurv_defaults(200, 10, 30)
        assert defaults['dropout'] == 0.6
        assert defaults['learning_rate'] >= 0.05
        assert defaults['activation'] == 'selu'

    def test_large_dataset(self):
        """Large dataset should get lower dropout and learning rate."""
        defaults = auto_deepsurv_defaults(10000, 10, 7000)
        assert defaults['dropout'] <= 0.25
        assert defaults['learning_rate'] <= 0.02

    def test_low_epv_warns(self):
        """EPV < 10 should trigger warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            auto_deepsurv_defaults(100, 20, 50)  # EPV = 2.5
            assert any("EPV" in str(warning.message) for warning in w)

    def test_batch_size_power_of_2(self):
        """Batch size should be a power of 2."""
        defaults = auto_deepsurv_defaults(500, 5, 300)
        batch = defaults['batch_size']
        assert batch & (batch - 1) == 0  # Power of 2 check

    def test_batch_size_within_search_space(self):
        """batch_size must always be a valid DeepSurv search space choice."""
        from mlsurv.tuning.search_spaces import DEFAULT_SPACES

        valid_choices = DEFAULT_SPACES['deepsurv']['batch_size']
        for n_samples in [50, 100, 200, 216, 300, 500, 1000, 5000]:
            defaults = auto_deepsurv_defaults(n_samples, 10, n_samples // 2)
            batch = defaults['batch_size']
            assert batch in valid_choices, (
                f"n_samples={n_samples}: batch_size={batch} not in {valid_choices}"
            )

    def test_returns_all_required_keys(self):
        """Should return all keys needed by DeepSurv model."""
        defaults = auto_deepsurv_defaults(500, 10, 300)
        required = {'layers', 'dropout', 'learning_rate', 'batch_size',
                    'epochs', 'activation', 'weight_decay'}
        assert required.issubset(set(defaults.keys()))

    def test_small_dataset_gbsg_scale(self):
        """GBSG-like data (686, 7, 299): single narrow layer, more epochs."""
        defaults = auto_deepsurv_defaults(686, 7, 299)
        assert len(defaults['layers']) == 1  # 1 layer
        assert defaults['layers'][0] <= 16   # few nodes (paper found 8 for GBSG)
        assert defaults['epochs'] >= 100     # more epochs for small data

    def test_large_dataset_support_scale(self):
        """SUPPORT-like data (8873, 14, 5000): wider layer, fewer epochs."""
        defaults = auto_deepsurv_defaults(8873, 14, 5000)
        assert defaults['layers'][0] >= 20  # more nodes (paper found 44 for SUPPORT)
        assert defaults['epochs'] <= 100    # fewer epochs for large data

    def test_low_epv_warns_exactly_once(self):
        """Low EPV emits exactly one EPV warning plus high-regularization dropout."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            defaults = auto_deepsurv_defaults(200, 10, 50)  # EPV = 5
        assert len(w) == 1
        assert "EPV" in str(w[0].message)
        assert defaults['dropout'] == 0.6

    def test_batch_size_explicit_bounds(self):
        """batch_size stays within the explicit [64, 256] range across sizes."""
        for n_samples in [100, 500, 1000, 5000, 10000]:
            defaults = auto_deepsurv_defaults(n_samples, 10, n_samples // 2)
            assert 64 <= defaults['batch_size'] <= 256


class TestAutoRsfDefaults:
    """Tests for auto_rsf_defaults() data-adaptive hyperparameters."""

    def test_small_features(self):
        """Few features should give n_estimators=100 (floor)."""
        defaults = auto_rsf_defaults(686, 7, 299)
        assert defaults['n_estimators'] == 100  # 7*10 = 70, floor 100

    def test_many_features(self):
        """Many features should cap n_estimators at 300."""
        defaults = auto_rsf_defaults(1000, 50, 600)
        assert defaults['n_estimators'] == 300  # 50*10 = 500, cap 300

    def test_high_epv_unlimited_depth(self):
        """High EPV should allow unlimited depth."""
        defaults = auto_rsf_defaults(1000, 5, 500)  # EPV = 100
        assert defaults['max_depth'] is None

    def test_low_epv_limited_depth(self):
        """Low EPV should limit depth to prevent overfitting."""
        defaults = auto_rsf_defaults(100, 20, 50)  # EPV = 2.5
        assert defaults['max_depth'] == 5

    def test_min_samples_leaf_scaling(self):
        """min_samples_leaf should be >= 3 and ~1% of samples."""
        defaults = auto_rsf_defaults(1000, 10, 600)
        assert defaults['min_samples_leaf'] >= 3
        assert defaults['min_samples_leaf'] == 10  # 1% of 1000


class TestAutoGbsaDefaults:
    """Tests for auto_gbsa_defaults() data-adaptive hyperparameters."""

    def test_small_dataset(self):
        """Small dataset should get learning_rate=0.1."""
        defaults = auto_gbsa_defaults(500, 5, 300)
        assert defaults['learning_rate'] == 0.1

    def test_large_dataset(self):
        """Large dataset should get lower learning rate."""
        defaults = auto_gbsa_defaults(5000, 10, 3500)
        assert defaults['learning_rate'] <= 0.05

    def test_very_large_dataset(self):
        """Very large dataset should get learning_rate=0.01."""
        defaults = auto_gbsa_defaults(10000, 10, 7000)
        assert defaults['learning_rate'] == 0.01

    def test_n_estimators_inversely_proportional(self):
        """More trees when learning rate is lower."""
        small = auto_gbsa_defaults(500, 5, 300)
        large = auto_gbsa_defaults(5000, 10, 3500)
        assert large['n_estimators'] >= small['n_estimators']

    def test_low_epv_shallow_trees(self):
        """Low EPV should limit tree depth."""
        defaults = auto_gbsa_defaults(100, 20, 50)  # EPV = 2.5
        assert defaults['max_depth'] == 2


class TestAutoDeephitDefaults:
    """Tests for auto_deephit_defaults() data-adaptive hyperparameters."""

    def test_small_dataset(self):
        defaults = auto_deephit_defaults(300, 5, 200)
        assert defaults['layers'] == [32]
        assert defaults['num_durations'] == 50

    def test_medium_dataset(self):
        defaults = auto_deephit_defaults(1000, 10, 600)
        assert defaults['layers'] == [64, 64]
        assert defaults['num_durations'] == 100

    def test_large_dataset(self):
        defaults = auto_deephit_defaults(5000, 10, 3500)
        assert defaults['layers'] == [128, 128]
        assert defaults['num_durations'] == 200
        assert defaults['learning_rate'] == 0.005

    def test_low_epv_dropout_floor(self):
        """EPV < 10 should enforce dropout >= 0.3."""
        # n=2000, features=50, events=100 -> EPV=2 (low), but n>=2000 tier
        defaults = auto_deephit_defaults(2000, 50, 100)
        assert defaults['dropout'] >= 0.3

    def test_returns_required_keys(self):
        defaults = auto_deephit_defaults(500, 5, 300)
        required = {'layers', 'dropout', 'learning_rate', 'batch_size',
                    'num_durations', 'alpha', 'sigma'}
        assert required.issubset(set(defaults.keys()))

    def test_deephit_model_constructor_defaults(self):
        """DeepHitModel() default layers and lr should match updated values."""
        from mlsurv.models.deephit import DeepHitModel
        model = DeepHitModel()
        assert model.layers == [64]
        assert model.learning_rate == 0.005

    def test_deephit_search_space_count(self):
        """DEFAULT_SPACES['deephit'] should have 10 parameters."""
        assert len(DEFAULT_SPACES['deephit']) == 10

    def test_deephit_auto_defaults_activation(self):
        """auto_deephit_defaults() returns 'activation' key."""
        defaults = auto_deephit_defaults(500, 5, 300)
        assert 'activation' in defaults
        assert defaults['activation'] == 'selu'

    def test_deephit_auto_defaults_weight_decay(self):
        """auto_deephit_defaults() returns 'weight_decay' key."""
        defaults = auto_deephit_defaults(500, 5, 300)
        assert 'weight_decay' in defaults
        assert defaults['weight_decay'] == 0.01

    def test_deephit_auto_defaults_alpha(self):
        """Warm-start alpha defaults to 0.0.

        The model constructor and the tuning warm-start are independent
        sources of the default; locking both prevents silent desync where
        a future revert touches one but not the other.
        """
        defaults = auto_deephit_defaults(500, 5, 300)
        assert defaults['alpha'] == 0.0


class TestAutoWeibullAftDefaults:
    """Tests for auto_weibull_aft_defaults() data-adaptive hyperparameters."""

    def test_low_epv_more_regularization(self):
        defaults = auto_weibull_aft_defaults(100, 20, 50)  # EPV = 2.5
        assert defaults['alpha'] == 0.1

    def test_medium_epv(self):
        defaults = auto_weibull_aft_defaults(500, 10, 150)  # EPV = 15
        assert defaults['alpha'] == 0.01

    def test_high_epv_no_regularization(self):
        defaults = auto_weibull_aft_defaults(1000, 5, 500)  # EPV = 100
        assert defaults['alpha'] == 0.0


class TestGetSearchSpace:
    """Tests for get_search_space() with defaults and custom overrides."""

    def test_known_model(self):
        """Known model returns its default space."""
        space = get_search_space('rsf')
        assert 'n_estimators' in space
        assert 'max_depth' in space

    def test_unknown_model_raises(self):
        """Unknown model should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_search_space('nonexistent')

    def test_custom_space_merges(self):
        """Custom space should override defaults."""
        space = get_search_space('rsf', custom_space={'n_estimators': (50, 100)})
        assert space['n_estimators'] == (50, 100)

    def test_coxph_empty_space(self):
        """CoxPH has no hyperparameters to tune."""
        space = get_search_space('coxph')
        assert space == {}

    def test_default_spaces_contains_core_models(self):
        """DEFAULT_SPACES must contain every core model (incl. fssvm)."""
        for model in ('coxph', 'coxnet', 'rsf', 'gbsa', 'fssvm'):
            assert model in DEFAULT_SPACES


class TestGetPreprocessingDefaults:
    """Tests for get_preprocessing_defaults().

    Feature-selector defaults and copy semantics are covered more thoroughly
    in ``test_preprocessing_tuning.py :: TestPreprocessingDefaults``;
    only the unknown-component edge case is unique here.
    """

    def test_unknown_component(self):
        """Unknown component returns empty dict."""
        assert get_preprocessing_defaults('nonexistent') == {}


class TestEpvConstrainedSearchSpace:
    """Tests for EPV-aware search space narrowing in get_search_space().

    Verifies that capacity-capping models (RSF, GBSA, XGBCox, DeepSurv,
    DeepHit) and shrinkage-forcing penalized models (CoxNet, FSSVM) have
    their search spaces constrained based on both the events-per-variable
    ratio and the absolute event count, while unconstrained models and user
    overrides are unaffected.
    """

    # --- RSF ---

    def test_rsf_low_epv(self):
        """RSF with EPV=3: aggressive constraints."""
        # EPV = 30/10 = 3
        space = get_search_space('rsf', n_samples=200, n_events=30, n_features=10)
        assert space['bootstrap'] == [True]
        assert space['max_depth'] == (3, 8)
        assert space['min_samples_leaf'] == (5, 30)

    def test_rsf_medium_epv(self):
        """RSF with EPV=15: moderate constraints."""
        # EPV = 150/10 = 15
        space = get_search_space('rsf', n_samples=500, n_events=150, n_features=10)
        assert space['bootstrap'] == [True]
        assert space['max_depth'] == (5, 12)
        assert space['min_samples_leaf'] == (3, 30)

    def test_rsf_high_epv_large_n(self):
        """RSF with EPV=50, n=3000: bootstrap=[True, False] allowed."""
        # EPV = 500/10 = 50
        space = get_search_space('rsf', n_samples=3000, n_events=500, n_features=10)
        assert space['bootstrap'] == [True, False]

    def test_rsf_high_epv_small_n(self):
        """RSF with EPV=50 but n=500: bootstrap still [True] (n < 2000)."""
        # EPV = 250/5 = 50
        space = get_search_space('rsf', n_samples=500, n_events=250, n_features=5)
        assert space['bootstrap'] == [True]

    def test_rsf_bootstrap_n_samples_boundary(self):
        """High tier at exactly n=2000: bootstrap stays [True] (strict > 2000)."""
        # EPV = 100/5 = 20, 100 events -> high tier; n_samples == 2000 is not > 2000
        space = get_search_space('rsf', n_samples=2000, n_events=100, n_features=5)
        assert space['bootstrap'] == [True]

    # --- GBSA ---

    def test_gbsa_low_epv(self):
        """GBSA with low EPV: capped estimators, shallow trees, nonzero dropout."""
        # EPV = 30/10 = 3
        space = get_search_space('gbsa', n_samples=200, n_events=30, n_features=10)
        assert space['n_estimators'] == (100, 200)
        assert space['max_depth'] == (2, 4)
        assert space['min_samples_leaf'] == (5, 30)
        assert space['dropout_rate'] == (0.05, 0.3)

    def test_gbsa_medium_epv(self):
        """GBSA with EPV=15: moderate constraints."""
        # EPV = 150/10 = 15
        space = get_search_space('gbsa', n_samples=500, n_events=150, n_features=10)
        assert space['n_estimators'] == (100, 300)
        assert space['max_depth'] == (2, 5)

    def test_gbsa_epv_exactly_10(self):
        """EPV=10 exactly should be medium tier (not low)."""
        # EPV = 100/10 = 10
        space = get_search_space('gbsa', n_samples=500, n_events=100, n_features=10)
        assert space['n_estimators'] == (100, 300)  # medium, not (100, 200)
        assert space['max_depth'] == (2, 5)          # medium, not (2, 4)

    def test_gbsa_epv_exactly_20(self):
        """EPV=20 exactly should be high tier (defaults preserved)."""
        # EPV = 200/10 = 20
        space = get_search_space('gbsa', n_samples=500, n_events=200, n_features=10)
        default = get_search_space('gbsa')
        assert space == default

    def test_gbsa_default_n_estimators_cap(self):
        """GBSA n_estimators capped at 300 to limit tuning time."""
        assert DEFAULT_SPACES['gbsa']['n_estimators'] == (100, 300)

    # --- XGBCox ---

    def test_xgbcox_low_epv(self):
        """XGBCox with low EPV: depth and min_child_weight constrained."""
        # EPV = 30/10 = 3
        space = get_search_space('xgbcox', n_samples=200, n_events=30, n_features=10)
        assert space['max_depth'] == (3, 6)
        assert space['min_child_weight'] == (3, 10)

    def test_xgbcox_medium_epv(self):
        """XGBCox with EPV=15: moderate constraints."""
        # EPV = 150/10 = 15
        space = get_search_space('xgbcox', n_samples=500, n_events=150, n_features=10)
        assert space['max_depth'] == (3, 8)
        assert space['min_child_weight'] == (2, 10)

    # --- Boundary values ---

    def test_rsf_epv_exactly_10(self):
        """EPV=10 exactly should be medium tier (not low)."""
        # EPV = 100/10 = 10
        space = get_search_space('rsf', n_samples=500, n_events=100, n_features=10)
        assert space['max_depth'] == (5, 12)  # medium, not (3, 8)
        assert space['min_samples_leaf'] == (3, 30)  # medium, not (5, 30)

    def test_rsf_epv_exactly_20(self):
        """EPV=20 exactly should be high tier (defaults preserved)."""
        # EPV = 200/10 = 20
        space = get_search_space('rsf', n_samples=500, n_events=200, n_features=10)
        assert space['max_depth'] == (5, 15)  # default, not (5, 12)
        assert space['min_samples_leaf'] == (1, 30)  # default, not (3, 30)

    # --- High EPV defaults preserved ---

    def test_gbsa_high_epv_unchanged(self):
        """GBSA with EPV=50: defaults preserved."""
        space = get_search_space('gbsa', n_samples=3000, n_events=500, n_features=10)
        default = get_search_space('gbsa')
        assert space == default

    # --- Unconstrained model unchanged ---

    def test_unconstrained_model_unchanged(self):
        """A model with no _EPV_CONSTRAINTS entry is identical with/without EPV params."""
        # weibull_aft is not in _EPV_CONSTRAINTS, so it is never narrowed.
        space_with = get_search_space(
            'weibull_aft', n_samples=200, n_events=30, n_features=10,
        )
        space_without = get_search_space('weibull_aft')
        assert space_with == space_without

    # --- User override wins ---

    def test_user_override_wins(self):
        """Custom space overrides EPV constraint."""
        space = get_search_space(
            'rsf',
            custom_space={'bootstrap': [True, False]},
            n_samples=100, n_events=30, n_features=10,  # EPV=3
        )
        assert space['bootstrap'] == [True, False]

    # --- Neural networks ---

    def test_deepsurv_low_epv(self):
        """DeepSurv with EPV=3: single layer, narrow nodes, high dropout."""
        # EPV = 30/10 = 3
        space = get_search_space('deepsurv', n_samples=200, n_events=30, n_features=10)
        assert space['n_layers'] == (1, 1)
        assert space['n_nodes'] == (8, 24)
        assert space['dropout'] == (0.3, 0.7)

    def test_deepsurv_medium_epv(self):
        """DeepSurv with EPV=15: moderate constraints."""
        # EPV = 150/10 = 15
        space = get_search_space('deepsurv', n_samples=500, n_events=150, n_features=10)
        assert space['n_layers'] == (1, 2)
        assert space['n_nodes'] == (8, 36)
        assert space['dropout'] == (0.2, 0.7)

    def test_deepsurv_high_epv_unchanged(self):
        """DeepSurv with EPV=50: defaults preserved."""
        space = get_search_space('deepsurv', n_samples=3000, n_events=500, n_features=10)
        default = get_search_space('deepsurv')
        assert space == default

    def test_deephit_low_epv(self):
        """DeepHit with EPV=3: single layer, narrow nodes, constrained dropout."""
        # EPV = 30/10 = 3
        space = get_search_space('deephit', n_samples=200, n_events=30, n_features=10)
        assert space['n_layers'] == (1, 1)
        assert space['n_nodes'] == (16, 48)
        assert space['dropout'] == (0.2, 0.5)

    def test_deephit_medium_epv(self):
        """DeepHit with EPV=15: moderate constraints."""
        # EPV = 150/10 = 15
        space = get_search_space('deephit', n_samples=500, n_events=150, n_features=10)
        assert space['n_layers'] == (1, 2)
        assert space['n_nodes'] == (16, 128)
        assert space['dropout'] == (0.1, 0.5)

    def test_deephit_high_epv_unchanged(self):
        """DeepHit with EPV=50: defaults preserved."""
        space = get_search_space('deephit', n_samples=3000, n_events=500, n_features=10)
        default = get_search_space('deephit')
        assert space == default

    # --- Neural network boundary values ---

    def test_deepsurv_epv_exactly_10(self):
        """DeepSurv at EPV=10 exactly should use medium tier."""
        space = get_search_space('deepsurv', n_samples=500, n_events=100, n_features=10)
        assert space['n_layers'] == (1, 2)   # medium, not (1, 1)
        assert space['n_nodes'] == (8, 36)   # medium, not (8, 24)

    def test_deepsurv_epv_exactly_20(self):
        """DeepSurv at EPV=20 exactly should use defaults (high tier)."""
        space = get_search_space('deepsurv', n_samples=500, n_events=200, n_features=10)
        default = get_search_space('deepsurv')
        assert space == default

    # --- User override wins for neural nets ---

    def test_user_override_wins_neural(self):
        """Custom space overrides EPV constraint for DeepSurv."""
        space = get_search_space(
            'deepsurv',
            custom_space={'n_layers': (1, 4)},
            n_samples=200, n_events=30, n_features=10,  # EPV=3
        )
        assert space['n_layers'] == (1, 4)  # custom wins over (1, 1)

    # --- XGBCox high EPV unchanged ---

    def test_xgbcox_high_epv_unchanged(self):
        """XGBCox with EPV=50: defaults preserved."""
        space = get_search_space('xgbcox', n_samples=3000, n_events=500, n_features=10)
        default = get_search_space('xgbcox')
        assert space == default

    # --- Absolute-event gate (EPV ratio alone is insufficient) ---

    def test_high_epv_few_events_demoted_to_low(self):
        """EPV=20 but only 40 events: gated down to the low tier, not defaults."""
        # EPV = 40/2 = 20, but 40 < 100 events and 40 < 50 events -> low tier
        space = get_search_space('rsf', n_samples=120, n_events=40, n_features=2)
        assert space['max_depth'] == (3, 8)          # low tier, not default (5, 15)
        assert space['min_samples_leaf'] == (5, 30)  # low tier

    def test_high_epv_enough_events_keeps_defaults(self):
        """EPV=20 with 100 events: high tier, tree params left at defaults."""
        # EPV = 100/5 = 20 and 100 >= 100 events -> high tier
        # (assert on the tier-driven params; bootstrap depends on n_samples, not tier)
        space = get_search_space('rsf', n_samples=500, n_events=100, n_features=5)
        assert space['max_depth'] == (5, 15)          # default, not narrowed
        assert space['min_samples_leaf'] == (1, 30)   # default, not narrowed

    def test_event_floor_high_tier_boundary(self):
        """n_events exactly 100 unlocks the high tier; 99 stays in mid."""
        # EPV = 99/4 = 24.75 (>=20) but 99 < 100 events -> mid tier
        mid = get_search_space('rsf', n_samples=500, n_events=99, n_features=4)
        assert mid['max_depth'] == (5, 12)           # mid, not default
        # EPV = 100/4 = 25 and 100 >= 100 events -> high tier (defaults)
        high = get_search_space('rsf', n_samples=500, n_events=100, n_features=4)
        assert high['max_depth'] == (5, 15)          # default, not narrowed

    def test_event_floor_mid_tier_boundary(self):
        """n_events exactly 50 unlocks the mid tier; 49 falls to low."""
        # EPV = 49/4 = 12.25 (>=10) but 49 < 50 events -> low tier
        low = get_search_space('gbsa', n_samples=300, n_events=49, n_features=4)
        assert low['max_depth'] == (2, 4)            # low tier
        # EPV = 50/4 = 12.5 and 50 >= 50 events -> mid tier
        mid = get_search_space('gbsa', n_samples=300, n_events=50, n_features=4)
        assert mid['max_depth'] == (2, 5)            # mid tier

    def test_ample_events_low_epv_still_gated_by_ratio(self):
        """Ample events but EPV<20 stays in mid (high requires the ratio too)."""
        # EPV = 120/10 = 12 (<20 blocks high) but 120 >= 100 events -> mid tier
        space = get_search_space('rsf', n_samples=500, n_events=120, n_features=10)
        assert space['max_depth'] == (5, 12)         # mid, not default (5, 15)

    def test_ample_events_very_low_epv_demoted_to_low(self):
        """Ample events but EPV<10 falls to low (mid requires the ratio too)."""
        # EPV = 60/8 = 7.5 (<10) but 60 >= 50 events -> low tier
        space = get_search_space('coxnet', n_samples=400, n_events=60, n_features=8)
        assert space['l1_ratio'] == (0.0, 0.5)       # low tier, not mid (0.0, 0.7)
        assert space['alpha_index'] == (0, 49)

    def test_coxnet_high_epv_few_events_demoted(self):
        """CoxNet: EPV=20 but few events gates down to the low tier."""
        # EPV = 40/2 = 20, but 40 < 50 events -> low tier
        space = get_search_space('coxnet', n_samples=120, n_events=40, n_features=2)
        assert space['l1_ratio'] == (0.0, 0.5)
        assert space['alpha_index'] == (0, 49)

    def test_fssvm_high_epv_few_events_demoted(self):
        """FSSVM: EPV=20 but few events gates down to the low tier."""
        # EPV = 40/2 = 20, but 40 < 50 events -> low tier
        space = get_search_space('fssvm', n_samples=120, n_events=40, n_features=2)
        assert space['alpha'] == (1e-5, 1.0, 'log')

    # --- CoxNet (penalized: more shrinkage at low EPV) ---

    def test_coxnet_low_epv(self):
        """CoxNet with EPV=3: favor L2 and the penalized end of the path."""
        # EPV = 30/10 = 3
        space = get_search_space('coxnet', n_samples=200, n_events=30, n_features=10)
        assert space['l1_ratio'] == (0.0, 0.5)
        assert space['alpha_index'] == (0, 49)

    def test_coxnet_medium_epv(self):
        """CoxNet with EPV=15: moderate shrinkage steering."""
        # EPV = 150/10 = 15
        space = get_search_space('coxnet', n_samples=500, n_events=150, n_features=10)
        assert space['l1_ratio'] == (0.0, 0.7)
        assert space['alpha_index'] == (0, 74)

    def test_coxnet_high_epv_unchanged(self):
        """CoxNet with EPV=50: defaults preserved."""
        space = get_search_space('coxnet', n_samples=3000, n_events=500, n_features=10)
        assert space == get_search_space('coxnet')

    def test_coxnet_user_override_wins(self):
        """Custom l1_ratio overrides the EPV constraint."""
        space = get_search_space(
            'coxnet',
            custom_space={'l1_ratio': (0.0, 1.0)},
            n_samples=200, n_events=30, n_features=10,  # EPV=3
        )
        assert space['l1_ratio'] == (0.0, 1.0)  # custom wins over (0.0, 0.5)

    # --- FSSVM (penalized: larger alpha = less regularization) ---

    def test_fssvm_low_epv(self):
        """FSSVM with EPV=3: cap the weakly-regularized (large-alpha) end."""
        # EPV = 30/10 = 3
        space = get_search_space('fssvm', n_samples=200, n_events=30, n_features=10)
        assert space['alpha'] == (1e-5, 1.0, 'log')
        # rank_ratio / max_iter / tol are intentionally not constrained
        assert space['rank_ratio'] == DEFAULT_SPACES['fssvm']['rank_ratio']
        assert space['max_iter'] == DEFAULT_SPACES['fssvm']['max_iter']
        assert space['tol'] == DEFAULT_SPACES['fssvm']['tol']

    def test_fssvm_medium_epv(self):
        """FSSVM with EPV=15: moderate alpha cap."""
        # EPV = 150/10 = 15
        space = get_search_space('fssvm', n_samples=500, n_events=150, n_features=10)
        assert space['alpha'] == (1e-5, 10.0, 'log')

    def test_fssvm_high_epv_unchanged(self):
        """FSSVM with EPV=50: defaults preserved."""
        space = get_search_space('fssvm', n_samples=3000, n_events=500, n_features=10)
        assert space == get_search_space('fssvm')

    def test_fssvm_user_override_wins(self):
        """Custom alpha overrides the EPV constraint."""
        space = get_search_space(
            'fssvm',
            custom_space={'alpha': (1e-5, 100.0, 'log')},
            n_samples=200, n_events=30, n_features=10,  # EPV=3
        )
        assert space['alpha'] == (1e-5, 100.0, 'log')

    # --- Input validation ---

    def test_zero_features_raises(self):
        """n_features=0 should raise ValueError."""
        with pytest.raises(ValueError, match="Dataset dimensions must be positive"):
            get_search_space('rsf', n_samples=200, n_events=30, n_features=0)

    def test_negative_events_raises(self):
        """n_events=-1 should raise ValueError."""
        with pytest.raises(ValueError, match="Dataset dimensions must be positive"):
            get_search_space('rsf', n_samples=200, n_events=-1, n_features=10)

    def test_zero_events_raises(self):
        """n_events=0 should raise ValueError (degenerate dataset)."""
        with pytest.raises(ValueError, match="Dataset dimensions must be positive"):
            get_search_space('rsf', n_samples=200, n_events=0, n_features=10)

    def test_zero_samples_raises(self):
        """n_samples=0 should raise ValueError."""
        with pytest.raises(ValueError, match="Dataset dimensions must be positive"):
            get_search_space('rsf', n_samples=0, n_events=30, n_features=10)

    def test_negative_samples_raises(self):
        """n_samples=-1 should raise ValueError."""
        with pytest.raises(ValueError, match="Dataset dimensions must be positive"):
            get_search_space('rsf', n_samples=-1, n_events=30, n_features=10)

    def test_negative_features_raises(self):
        """n_features=-1 should raise ValueError."""
        with pytest.raises(ValueError, match="Dataset dimensions must be positive"):
            get_search_space('rsf', n_samples=200, n_events=30, n_features=-1)

    # --- Partial params warning ---

    def test_partial_params_warns(self):
        """Providing only some dataset params should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            space = get_search_space('rsf', n_samples=200, n_events=30)
            assert len(w) == 1
            assert "EPV constraints require all" in str(w[0].message)
            assert "2/3 were provided" in str(w[0].message)
            # Should return unconstrained defaults
            assert space == DEFAULT_SPACES['rsf']

    def test_partial_params_one_of_three_warns(self):
        """Providing only 1 of 3 dataset params should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_search_space('rsf', n_features=10)
            assert len(w) == 1
            assert "1/3 were provided" in str(w[0].message)

    # --- No dataset params = unchanged ---

    def test_no_dataset_params_unchanged(self):
        """get_search_space('rsf') without dataset params returns DEFAULT_SPACES."""
        space = get_search_space('rsf')
        assert space == DEFAULT_SPACES['rsf']


# ---------------------------------------------------------------------------
# Regression test: _create_model_instance passes data characteristics
# ---------------------------------------------------------------------------

class TestCreateModelInstanceDataCharacteristics:
    """All models in _AUTO_DEFAULTS_REGISTRY must receive data characteristics
    when created through _create_model_instance().
    """

    def test_all_registry_models_receive_data_characteristics(self, small_data):
        """Every model in _AUTO_DEFAULTS_REGISTRY gets correct data characteristics."""
        from unittest.mock import patch, MagicMock
        from mlsurv import SurvivalLearner
        from mlsurv.models import _AUTO_DEFAULTS_REGISTRY

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        expected_n_samples = len(learner.X_train)
        expected_n_features = len(learner.X_train.columns)
        expected_n_events = int(learner.y_train['event'].sum())

        for model_name in _AUTO_DEFAULTS_REGISTRY:
            with patch('mlsurv.learner._training.get_model') as mock_get_model:
                mock_get_model.return_value = MagicMock()
                learner._create_model_instance(model_name)
                _, kwargs = mock_get_model.call_args
                assert kwargs['n_samples'] == expected_n_samples, model_name
                assert kwargs['n_features'] == expected_n_features, model_name
                assert kwargs['n_events'] == expected_n_events, model_name

    def test_n_features_uses_pre_selection_count_when_no_selector(self, small_data):
        """n_features equals pre-selection count when no selector is active."""
        from unittest.mock import patch, MagicMock
        from mlsurv import SurvivalLearner
        from mlsurv.models import _AUTO_DEFAULTS_REGISTRY

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        model_name = next(iter(_AUTO_DEFAULTS_REGISTRY))
        with patch('mlsurv.learner._training.get_model') as mock_get_model:
            mock_get_model.return_value = MagicMock()
            learner._create_model_instance(model_name)
            _, kwargs = mock_get_model.call_args
            assert kwargs['n_features'] == len(learner.X_train.columns)

    # xgbcox/xgbaft are deliberately NOT in the registry: their survival
    # defaults live on the constructor, so the untuned path must not inject
    # data-driven values that would shadow them.
    @pytest.mark.parametrize('model_name', ['coxph', 'xgbcox', 'xgbaft'])
    def test_non_registry_models_skip_data_characteristics(self, model_name, small_data):
        """Models NOT in _AUTO_DEFAULTS_REGISTRY should not receive data chars."""
        from unittest.mock import patch, MagicMock
        from mlsurv import SurvivalLearner
        from mlsurv.models import _AUTO_DEFAULTS_REGISTRY

        assert model_name not in _AUTO_DEFAULTS_REGISTRY

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)

        with patch('mlsurv.learner._training.get_model') as mock_get_model:
            mock_get_model.return_value = MagicMock()
            learner._create_model_instance(model_name)
            _, kwargs = mock_get_model.call_args
            assert 'n_samples' not in kwargs
            assert 'n_features' not in kwargs
            assert 'n_events' not in kwargs

    def test_n_features_estimates_topk_auto(self, small_data):
        """With TopKCoxSelector(k='auto'), n_features uses Harrell's rule."""
        from unittest.mock import patch, MagicMock
        from mlsurv import SurvivalLearner
        from mlsurv.models import _AUTO_DEFAULTS_REGISTRY

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector='topk', random_state=42)

        n_features_in = len(learner.X_train.columns)
        n_events = int(learner.y_train['event'].sum())
        expected = min(n_features_in, max(1, n_events // 15))

        model_name = next(iter(_AUTO_DEFAULTS_REGISTRY))
        with patch('mlsurv.learner._training.get_model') as mock_get_model:
            mock_get_model.return_value = MagicMock()
            learner._create_model_instance(model_name)
            _, kwargs = mock_get_model.call_args
            assert kwargs['n_features'] == expected

    def test_n_features_estimates_topk_explicit_k(self, small_data):
        """With TopKCoxSelector(k=3), n_features equals 3."""
        from unittest.mock import patch, MagicMock
        from mlsurv import SurvivalLearner
        from mlsurv.models import _AUTO_DEFAULTS_REGISTRY
        from mlsurv.preprocessing.selection import TopKCoxSelector

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector=TopKCoxSelector(k=3), random_state=42)

        model_name = next(iter(_AUTO_DEFAULTS_REGISTRY))
        with patch('mlsurv.learner._training.get_model') as mock_get_model:
            mock_get_model.return_value = MagicMock()
            learner._create_model_instance(model_name)
            _, kwargs = mock_get_model.call_args
            assert kwargs['n_features'] == 3

    def test_n_features_falls_back_for_vif(self, small_data):
        """VIF selector: falls back to pre-selection count + SuggestionWarning."""
        from unittest.mock import patch, MagicMock
        from mlsurv import SurvivalLearner
        from mlsurv.models import _AUTO_DEFAULTS_REGISTRY
        from mlsurv._warnings import SuggestionWarning

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(feature_selector='vif', random_state=42)

        model_name = next(iter(_AUTO_DEFAULTS_REGISTRY))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch('mlsurv.learner._training.get_model') as mock_get_model:
                mock_get_model.return_value = MagicMock()
                learner._create_model_instance(model_name)
                _, kwargs = mock_get_model.call_args
                assert kwargs['n_features'] == len(learner.X_train.columns)

        suggestion_warnings = [w for w in caught if issubclass(w.category, SuggestionWarning)]
        assert len(suggestion_warnings) >= 1
        assert "VIFSelector" in str(suggestion_warnings[0].message)

    def test_n_features_with_per_model_preprocessing(self, small_data):
        """Per-model preprocessing override is respected for n_features."""
        from unittest.mock import patch, MagicMock
        from mlsurv import SurvivalLearner
        from mlsurv.models import _AUTO_DEFAULTS_REGISTRY
        from mlsurv.preprocessing.selection import TopKCoxSelector

        X, y = small_data
        learner = SurvivalLearner(X, y)
        learner.setup(random_state=42)  # No global selector

        model_name = next(iter(_AUTO_DEFAULTS_REGISTRY))
        preprocessing = {model_name: {'feature_selector': TopKCoxSelector(k=5)}}

        with patch('mlsurv.learner._training.get_model') as mock_get_model:
            mock_get_model.return_value = MagicMock()
            learner._create_model_instance(model_name, _preprocessing=preprocessing)
            _, kwargs = mock_get_model.call_args
            n_features_in = len(learner.X_train.columns)
            assert kwargs['n_features'] == min(5, n_features_in)


# ---------------------------------------------------------------------------
# Unit tests: _estimate_post_selection_features
# ---------------------------------------------------------------------------

class TestEstimatePostSelectionFeatures:
    """Direct unit tests for the estimation helper."""

    def test_none_selector_returns_n_features_in(self):
        from mlsurv.learner._training import _estimate_post_selection_features
        assert _estimate_post_selection_features(None, 50, 300) == 50

    def test_topk_explicit_k(self):
        from mlsurv.learner._training import _estimate_post_selection_features
        from mlsurv.preprocessing.selection import TopKCoxSelector
        sel = TopKCoxSelector(k=5)
        assert _estimate_post_selection_features(sel, 50, 300) == 5

    def test_topk_explicit_k_clamped(self):
        """k > n_features_in should clamp to n_features_in."""
        from mlsurv.learner._training import _estimate_post_selection_features
        from mlsurv.preprocessing.selection import TopKCoxSelector
        sel = TopKCoxSelector(k=100)
        assert _estimate_post_selection_features(sel, 10, 300) == 10

    def test_topk_auto(self):
        from mlsurv.learner._training import _estimate_post_selection_features
        from mlsurv.preprocessing.selection import TopKCoxSelector
        sel = TopKCoxSelector(k='auto')
        # 300 events // 15 = 20, min(50, 20) = 20
        assert _estimate_post_selection_features(sel, 50, 300) == 20

    def test_topk_auto_few_events(self):
        """Very few events: Harrell's rule gives max(1, 0) = 1."""
        from mlsurv.learner._training import _estimate_post_selection_features
        from mlsurv.preprocessing.selection import TopKCoxSelector
        sel = TopKCoxSelector(k='auto')
        # 10 events // 15 = 0, max(1, 0) = 1, min(50, 1) = 1
        assert _estimate_post_selection_features(sel, 50, 10) == 1

    def test_topk_auto_clamped_to_n_features_in(self):
        """Harrell's k exceeding n_features_in should clamp to n_features_in."""
        from mlsurv.learner._training import _estimate_post_selection_features
        from mlsurv.preprocessing.selection import TopKCoxSelector
        sel = TopKCoxSelector(k='auto')
        # 300 events // 15 = 20, min(5, 20) = 5
        assert _estimate_post_selection_features(sel, 5, 300) == 5

    def test_vif_fallback_with_warning(self):
        from mlsurv.learner._training import _estimate_post_selection_features
        from mlsurv.preprocessing.selection import VIFSelector
        from mlsurv._warnings import SuggestionWarning
        sel = VIFSelector()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _estimate_post_selection_features(sel, 50, 300)
        assert result == 50
        assert any(issubclass(w.category, SuggestionWarning) for w in caught)

    def test_correlation_fallback_with_warning(self):
        from mlsurv.learner._training import _estimate_post_selection_features
        from mlsurv.preprocessing.selection import CorrelationSelector
        from mlsurv._warnings import SuggestionWarning
        sel = CorrelationSelector()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _estimate_post_selection_features(sel, 30, 100)
        assert result == 30
        assert any(issubclass(w.category, SuggestionWarning) for w in caught)

    def test_coxlasso_fallback_with_warning(self):
        from mlsurv.learner._training import _estimate_post_selection_features
        from mlsurv.preprocessing.selection import CoxLassoSelector
        from mlsurv._warnings import SuggestionWarning
        sel = CoxLassoSelector()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _estimate_post_selection_features(sel, 40, 200)
        assert result == 40
        assert any(issubclass(w.category, SuggestionWarning) for w in caught)

    def test_coxnet_fallback_with_warning(self):
        from mlsurv.learner._training import _estimate_post_selection_features
        from mlsurv.preprocessing.selection import CoxNetSelector
        from mlsurv._warnings import SuggestionWarning
        sel = CoxNetSelector()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _estimate_post_selection_features(sel, 40, 200)
        assert result == 40
        assert any(issubclass(w.category, SuggestionWarning) for w in caught)

    def test_pipeline_with_topk_last_step(self):
        """Pipeline ending in TopKCoxSelector uses TopK estimation."""
        from sklearn.pipeline import Pipeline
        from mlsurv.learner._training import _estimate_post_selection_features
        from mlsurv.preprocessing.selection import VIFSelector, TopKCoxSelector
        pipe = Pipeline([('vif', VIFSelector()), ('topk', TopKCoxSelector(k=5))])
        assert _estimate_post_selection_features(pipe, 50, 300) == 5

    def test_custom_transformer_warns(self):
        """Unknown custom transformers return n_features_in with warning."""
        from sklearn.preprocessing import StandardScaler
        from mlsurv.learner._training import _estimate_post_selection_features
        from mlsurv._warnings import SuggestionWarning
        sel = StandardScaler()  # Not a known selector
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _estimate_post_selection_features(sel, 50, 300)
        assert result == 50
        suggestion_warnings = [w for w in caught if issubclass(w.category, SuggestionWarning)]
        assert len(suggestion_warnings) >= 1
        assert "StandardScaler" in str(suggestion_warnings[0].message)
