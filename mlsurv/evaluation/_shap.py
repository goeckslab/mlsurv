"""SHAP value computation for model explanations."""

import numpy as np
import pandas as pd

try:
    from shap.utils._exceptions import DimensionError as _ShapDimensionError
except (ImportError, AttributeError):
    class _ShapDimensionError(Exception):
        """Placeholder when shap.utils._exceptions.DimensionError is unavailable."""


def _wrap_predict_with_columns(predict_fn, columns):
    """Coerce ndarray SHAP inputs back into a DataFrame with feature names.

    SHAP explainers perturb the background and pass numpy arrays to the
    model's predict callable. When the underlying estimator (e.g. sksurv
    RSF) was fitted on a DataFrame, that triggers
    ``UserWarning: X does not have valid feature names``. Wrapping here
    keeps column-name identity intact so the warning never fires and
    sklearn can still raise on a true name mismatch.
    """
    columns = list(columns)

    def predict(X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=columns)
        return predict_fn(X)

    return predict


def _numeric_frame_for_shap(df, role):
    """Return a float64 copy of ``df`` so the SHAP masker's numeric ops succeed.

    The explainer works in raw feature space (the full pipeline does its own
    encoding), so the frame handed to ``shap.maskers.Independent`` can mix
    float, int and bool columns (``pd.get_dummies`` returns bool). A DataFrame
    with a mixed bool/float block has an *object* ``.values`` array, and shap's
    masker then calls ``np.isfinite`` on it, which raises
    ``TypeError: ufunc 'isfinite' not supported``. Bool and int one-hot encode
    as 0/1, so casting to float leaves predictions and SHAP values unchanged.

    Columns that are not numeric (a raw string, ``category`` or datetime column)
    can't be encoded at this layer, so reject them up front with a clear message
    rather than coercing them silently (an object column of numeric-looking
    strings would otherwise cast without warning) or surfacing the cryptic
    isfinite error. ``is_numeric_dtype`` counts bool as numeric, so dummy
    columns pass.
    """
    non_numeric = [c for c in df.columns
                   if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(
            f"SHAP explanation needs numeric features, but the {role} has "
            f"non-numeric column(s): {non_numeric}. One-hot or numerically "
            f"encode them before explaining (dummy columns may be bool)."
        )
    return df.astype(np.float64)


def _run_explainer(model, X_background, X_explain, max_background, random_state=None):
    """Run the one explainer: a seeded, model-agnostic PermutationExplainer.

    The engine treats every model the same way — it only calls ``model.predict``,
    so it works for any estimator (CoxPH, RSF, XGBoost, deep models, custom
    wrappers) without inspecting its internals.  In practice the model is always
    the full sklearn ``Pipeline`` (preprocessing + survival model), and a
    Pipeline exposes none of the inner estimator's attributes, so a single
    prediction-only explainer is the only thing that can apply uniformly.

    Reproducibility: ``random_state`` seeds both the background subsample (the
    ``Independent`` masker takes no seed of its own, so we subsample before
    constructing it) and the explainer's permutation order (``seed=`` makes
    PermutationExplainer call ``np.random.seed``).  A fixed ``random_state``
    therefore gives identical SHAP values run to run, even for non-additive
    models like RSF where the permutation order affects the result.

    Parameters
    ----------
    model : object
        Fitted model/pipeline with a ``predict(X)`` method.
    X_background : pd.DataFrame
        Reference data the SHAP background is summarized from (typically the
        training cohort).  Subsampled to ``max_background`` rows.
    X_explain : pd.DataFrame
        Rows to explain.
    max_background : int
        Maximum background samples for the masker.
    random_state : int, optional
        Seed for reproducible background subsampling and permutation order.

    Returns
    -------
    tuple
        ``(shap_values, base_value)`` where ``shap_values`` has shape
        ``(n_explain, n_features)`` and ``base_value`` is the scalar
        ``E[f(X)]`` over the background set (same for every explained
        sample, so a single scalar suffices).  Callers that only need
        SHAP values can discard ``base_value``.
    """
    import shap

    # Pre-subsample background with random_state for reproducibility (Independent
    # masker doesn't accept a seed, so we subsample before constructing it).
    if len(X_background) > max_background:
        bg = shap.sample(X_background, max_background, random_state=random_state)
    else:
        bg = X_background
    # The masker runs np.isfinite on the background, so it must be numeric float:
    # a get_dummies bool + float mix is an object array it can't handle. Both the
    # background and the explained frame go through this. predict still gets the
    # original column names via the wrapper below.
    bg = _numeric_frame_for_shap(bg, "background")
    masker = shap.maskers.Independent(bg, max_samples=len(bg))
    predict_fn = _wrap_predict_with_columns(model.predict, X_explain.columns)
    # seed= makes PermutationExplainer seed np.random, so the random permutation
    # order is reproducible — needed for non-additive models (e.g. RSF) where
    # the order changes the result.
    explainer = shap.Explainer(predict_fn, masker, algorithm="permutation",
                               seed=random_state)
    # silent=True hides shap's own progress bar; mlsurv prints its own progress
    # line, and the bar fills notebooks with dozens of output lines.
    explanation = explainer(
        _numeric_frame_for_shap(X_explain, "explained set"), silent=True,
    )
    base = np.asarray(explanation.base_values).ravel()
    base_value = float(base[0]) if base.size else 0.0
    return explanation.values, base_value


def _validate_shap_inputs(model, X_test, max_background, max_explain):
    """Validate parameters and model interface for SHAP computation.

    Raises TypeError, ValueError, or ImportError on failure.
    """
    if max_background < 1:
        raise ValueError(f"max_background must be >= 1, got {max_background}.")
    if max_explain < 1:
        raise ValueError(f"max_explain must be >= 1, got {max_explain}.")

    if not hasattr(model, 'predict') or not callable(getattr(model, 'predict')):
        raise TypeError(
            "Model must have a callable predict() method for SHAP explanations. "
            "Custom models should implement predict(X) -> np.ndarray of risk scores."
        )

    try:
        test_pred = model.predict(X_test.iloc[:2])
    except Exception as e:
        raise TypeError(
            f"Model predict() failed on test input: {e}. "
            "Ensure predict(X) accepts DataFrame input."
        ) from e

    if not hasattr(test_pred, '__len__') or len(test_pred) != 2:
        raise TypeError(
            f"Model predict() must return array-like with one value per sample. "
            f"Got {type(test_pred)} with unexpected shape."
        )

    try:
        import shap  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'shap' package is required for SHAP explanations. "
            "Install with: pip install shap  or  pip install mlsurv[deep]"
        )


def _shap_array(model, X_explain, X_background=None, max_background=50,
                random_state=None):
    """Shared SHAP core: validate the background, then dispatch the explainer.

    This is the single seam both the cohort path (:func:`_compute_shap`) and the
    single-patient path (``compute_patient_explanation``) route through, so they
    cannot drift apart: same background semantics, same model-agnostic explainer
    (:func:`_run_explainer`), same ``model`` (the full pipeline, raw features).
    Per-row SHAP values are independent of the rest of the explained set, so
    explaining one patient here yields exactly the value that patient gets inside
    a cohort run with the same background and ``random_state``.

    Parameters
    ----------
    model : object
        Fitted model/pipeline with a ``predict(X)`` method.
    X_explain : pd.DataFrame
        Rows to explain (the cohort, or a single patient).
    X_background : pd.DataFrame, optional
        Reference data the SHAP background is summarized from (typically the
        training cohort).  When None, falls back to ``X_explain``.
    max_background : int, default=50
        Maximum background samples for the SHAP masker.
    random_state : int, optional
        Seed for reproducible background subsampling.

    Returns
    -------
    tuple
        ``(shap_array, base_value)`` where ``shap_array`` has shape
        ``(len(X_explain), n_features)`` and ``base_value`` is the scalar
        ``E[f(X)]`` over the background.

    Raises
    ------
    ValueError
        If the background is empty or its columns do not match ``X_explain``.
    """
    # Decouple background from explained set: explain X_explain, but reference
    # the background (typically X_train).  Fall back to X_explain when no
    # separate background is supplied.
    if X_background is None:
        X_background = X_explain

    # The background must be a usable reference: non-empty and column-aligned
    # with the explained set (the masker/explainer assume identical columns in
    # identical order).  Fail loudly here rather than deep inside shap.
    if len(X_background) == 0:
        raise ValueError(
            "X_background (SHAP reference cohort) is empty. Pass a non-empty "
            "background, or omit X_background to reuse the explained set."
        )
    if list(X_background.columns) != list(X_explain.columns):
        raise ValueError(
            "X_background columns must match the explained set's columns (same "
            f"names, same order). Got background={list(X_background.columns)} vs "
            f"explained={list(X_explain.columns)}."
        )

    return _run_explainer(
        model, X_background, X_explain, max_background,
        random_state=random_state,
    )


def _compute_shap(model, X_test, X_background=None, max_background=50,
                  max_explain=50, random_state=None):
    """Compute SHAP values for model explanations.

    Uses one model-agnostic explainer for every model: a seeded
    PermutationExplainer that only calls ``model.predict`` (see
    :func:`_run_explainer`).

    The explained set (``X_test``) and the SHAP background (``X_background``)
    are decoupled: the cohort being explained stays the held-out test set,
    while the background is summarized from a separate reference cohort
    (typically the training set).  The same background is used for every
    model, so attributions are comparable across models.

    Parameters
    ----------
    model : object
        Fitted model with predict(X) method returning risk scores.
        Custom models must implement predict() returning a 1D array
        of risk scores (higher = higher risk).
    X_test : pd.DataFrame
        Test features to explain (the cohort shown in the beeswarm).
    X_background : pd.DataFrame, optional
        Reference data the SHAP background is summarized from (typically the
        training cohort).  When None, falls back to ``X_test`` (background and
        explained set coincide).  Production callers pass the training set.
    max_background : int, default=50
        Maximum number of background samples for the SHAP masker.
    max_explain : int, default=50
        Maximum number of test samples to explain.  When
        ``len(X_test) > max_explain``, a random subset is drawn.
        Mean |SHAP| importance rankings are stable with 50-100 samples.
    random_state : int, optional
        Seed for reproducible subsampling of explained data.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: feature, sample_idx, shap_value,
        feature_value.  The ``feature_value`` column carries the original
        feature value for each (sample, feature) pair so downstream plots
        can colour correctly even when ``max_explain`` subsamples the data.

    Raises
    ------
    TypeError
        If model lacks predict() method or returns invalid format.

    Notes
    -----
    The PermutationExplainer perturbs features against the summarized
    background, so for high-dimensional data SHAP magnitudes are approximate.
    Importance rankings remain stable with 50-100 explained samples.
    """
    # Cohort contract check (predict returns one value per row) — needs >=2
    # rows, so it stays here rather than in the shared core (the single-patient
    # path explains exactly one row).
    _validate_shap_inputs(model, X_test, max_background, max_explain)

    # Resolve the background fallback to the *full* test cohort before
    # subsampling the explained set, so a None background references all of
    # X_test (not the subsample).  _shap_array then receives an explicit
    # background and validates it.
    if X_background is None:
        X_background = X_test

    # Subsample explained data if needed (importance rankings are stable with 50-100)
    if len(X_test) > max_explain:
        X_explain = X_test.sample(n=max_explain, random_state=random_state)
    else:
        X_explain = X_test

    # Run the shared core (background validation + model-agnostic explainer).
    # base_value is discarded here: the cohort-level SHAP DataFrame is used for
    # beeswarm/importance plots, which don't reference the global expected value.
    shap_array, _base_value = _shap_array(
        model, X_explain, X_background, max_background,
        random_state=random_state,
    )

    # Convert to long DataFrame (include feature values for beeswarm coloring)
    data = []
    for sample_idx in range(len(X_explain)):
        for feat_idx, feat in enumerate(X_explain.columns):
            data.append({
                'feature': feat,
                'sample_idx': sample_idx,
                'shap_value': shap_array[sample_idx, feat_idx],
                'feature_value': X_explain.iloc[sample_idx, feat_idx],
            })
    return pd.DataFrame(data)


def _group_shap_values(shap_df, categorical_groups):
    """Sum SHAP values across dummy columns for each categorical group.

    For each group in *categorical_groups*, replaces individual dummy-column
    rows with a single group row whose ``shap_value`` is the sum of the
    dummy SHAP values for the same sample.  ``feature_value`` is set to a
    compact summary string showing each dummy column's value (e.g.
    ``"tgrade_II=1, tgrade_III=0"``), giving full context in beeswarm
    hover text.

    Parameters
    ----------
    shap_df : pd.DataFrame
        Long-format SHAP values with at least columns
        ``[feature, sample_idx, shap_value, feature_value]``.
        May also contain ``model_type``.
    categorical_groups : dict of {str: list of str}
        Mapping from group name to dummy column names.

    Returns
    -------
    pd.DataFrame
        Grouped SHAP DataFrame with the same schema.
    """
    # Reverse mapping: dummy column → group name
    col_to_group = {
        col: group
        for group, cols in categorical_groups.items()
        for col in cols
    }

    grouped_mask = shap_df['feature'].isin(col_to_group)
    ungrouped = shap_df[~grouped_mask]
    to_group = shap_df[grouped_mask].copy()

    if to_group.empty:
        return shap_df

    # Keep original dummy name before replacing with group name
    to_group['_dummy_name'] = to_group['feature']
    to_group['feature'] = to_group['feature'].map(col_to_group)

    # Determine groupby columns (model_type may or may not be present)
    groupby_cols = ['feature', 'sample_idx']
    if 'model_type' in to_group.columns:
        groupby_cols = ['model_type'] + groupby_cols

    # Sum SHAP values; build summary string of all dummy values
    def _agg(g):
        parts = [
            f"{row['_dummy_name']}={row['feature_value']:.3g}"
            for _, row in g.sort_values('_dummy_name').iterrows()
        ]
        return pd.Series({
            'shap_value': g['shap_value'].sum(),
            'feature_value': ', '.join(parts),
        })

    aggregated = to_group.groupby(groupby_cols, sort=False).apply(
        _agg, include_groups=False,
    ).reset_index()

    # Ensure same column order
    result = pd.concat([ungrouped, aggregated], ignore_index=True)
    return result[shap_df.columns]
