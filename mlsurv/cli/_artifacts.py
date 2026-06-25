"""Artifact management for the mlsurv CLI."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from ._exit_codes import EXIT_DATA, EXIT_ERROR

from mlsurv.constants import (
    ARTIFACT_METADATA_FILE,
    ARTIFACT_MODEL_FILE,
    ARTIFACT_VERSION,
)


def _extract_feature_names(
    metadata: dict,
) -> tuple[list[str] | None, str | None]:
    """Extract feature names from artifact metadata.

    Returns ``(feature_names, error_message)``.  On success
    *error_message* is ``None``.
    """
    contract = metadata.get('data_contract')
    if not isinstance(contract, dict):
        return None, "Artifact has no data_contract."
    feature_names = contract.get('feature_names')
    if feature_names is None:
        return None, "Artifact data_contract has no feature_names."
    return list(feature_names), None


def _check_feature_compatibility(metadata: dict, X) -> None:
    """Check artifact feature names against the loaded data columns.

    Comparison is set-based: column order is ignored, only presence/absence
    of feature names is checked.  Warns on stderr when the artifact has no
    ``data_contract`` or no ``feature_names``, then returns without checking.

    Raises
    ------
    ValueError
        If the data is missing features that the artifact model requires.
        Extra features in the data produce a warning instead, since most
        models ignore columns not present at training time.
    """
    artifact_features, error = _extract_feature_names(metadata)
    if artifact_features is None:
        print(
            f"Warning: {error} Skipping feature compatibility check.",
            file=sys.stderr,
        )
        return
    artifact_set = set(artifact_features)
    data_set = set(X.columns)
    if artifact_set == data_set:
        return
    missing = sorted(artifact_set - data_set)
    extra = sorted(data_set - artifact_set)
    if missing:
        raise ValueError(
            f"Data is missing features required by the artifact model: "
            f"{missing}. Ensure your data includes all features the model "
            f"was trained on."
        )
    if extra:
        print(
            f"Warning: Data has extra features not in the artifact: {extra}",
            file=sys.stderr,
        )


def _train_or_load_model(learner, args, *, X):
    """Train a model from scratch, or load a pre-trained artifact.

    When ``--load-artifact`` is supplied, loads the saved model and checks
    feature compatibility. Otherwise trains the model specified by
    ``--model``.

    Parameters
    ----------
    learner : SurvivalLearner
        Configured learner instance (already set up).
    args : argparse.Namespace
        Parsed CLI arguments.  Must have ``model`` and optionally
        ``load_artifact``.
    X : DataFrame
        Feature matrix from the loaded CSV, used for feature-name
        compatibility checking.

    Returns
    -------
    ModelResult
        When training from scratch, a standard ModelResult.  When loading
        an artifact, a ModelResult constructed from the artifact's stored
        metadata (no training occurs).

    Raises
    ------
    ValueError
        If ``--load-artifact`` points to an artifact that contains no model
        (e.g., a setup-only artifact), or if the loaded data is missing
        features required by the artifact model.
    """
    from mlsurv.learner._persistence import PersistenceMixin

    if getattr(args, 'load_artifact', None):
        metadata, loaded_model, _recalibrator = PersistenceMixin.load_artifact(
            args.load_artifact,
        )
        if loaded_model is None:
            raise ValueError(
                f"Artifact at '{args.load_artifact}' does not contain a model "
                f"(step='{metadata.get('step', 'unknown')}'). "
                "Only artifacts from train/tune/run/evaluate/bootstrap "
                "contain loadable models."
            )
        _check_feature_compatibility(metadata, X)

        from mlsurv.results.containers import ModelResult
        model_result = ModelResult.from_artifact(metadata, loaded_model)

        quiet = getattr(args, 'quiet', False)
        if not quiet:
            display_name = model_result.name if model_result.name != 'unknown' else args.model
            has_cv = (
                not math.isnan(model_result.cv_metric_mean)
                and not math.isnan(model_result.cv_metric_std)
            )
            if has_cv:
                tuned_tag = ' (tuned)' if model_result.is_tuned else ''
                print(
                    f"Loaded artifact '{display_name}': "
                    f"CV {model_result.cv_metric_mean:.4f} "
                    f"+/- {model_result.cv_metric_std:.4f}{tuned_tag}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Loaded artifact '{display_name}' (no CV metrics stored)",
                    file=sys.stderr,
                )

        return model_result
    return learner.train(args.model)


def _save_artifact_if_requested(learner, model_result, args, *, step: str) -> None:
    """Save an artifact directory when ``--save-artifact`` is provided.

    Warns on stderr if overwriting an existing artifact.  Catches
    ``OSError`` (e.g., permission denied, disk full) and prints a warning
    instead of crashing — the primary operation (train/tune/evaluate)
    already succeeded, so a save failure is degraded to a non-fatal warning.
    ``ValueError`` from ``save_artifact`` (e.g., setup not called) is *not*
    caught and propagates to ``main()`` as a fatal error.
    """
    save_path = getattr(args, 'save_artifact', None)
    if not save_path:
        return
    if Path(save_path, ARTIFACT_METADATA_FILE).exists():
        print(
            f"Warning: Overwriting existing artifact at: {save_path}",
            file=sys.stderr,
        )
    # Prefer column names stored in learner state (persisted from original
    # training session) over CLI arg defaults.
    state = getattr(learner, '_state', None)
    saved_time = getattr(state, 'time_col', None)
    saved_event = getattr(state, 'event_col', None)
    time_col = saved_time if saved_time is not None else args.time_col
    event_col = saved_event if saved_event is not None else args.event_col
    try:
        learner.save_artifact(
            model_result, save_path,
            step=step, time_col=time_col, event_col=event_col,
        )
    except OSError as exc:
        print(f"Warning: Failed to save artifact to '{save_path}': {exc}", file=sys.stderr)
        return
    quiet = getattr(args, 'quiet', False)
    if not quiet:
        print(f"\nArtifact saved to: {save_path}")


# ---------------------------------------------------------------------------
# Learner persistence helpers
# ---------------------------------------------------------------------------

def _save_learner_if_requested(learner, args) -> None:
    """Save learner state when ``--save-learner`` is provided.

    Follows the same contract as :func:`_save_artifact_if_requested`:
    ``OSError`` is demoted to a non-fatal warning; ``ValueError`` (setup
    not called) propagates to ``main()``.
    """
    save_path = getattr(args, 'save_learner', None)
    if not save_path:
        return
    if Path(save_path).exists():
        print(
            f"Warning: Overwriting existing learner at: {save_path}",
            file=sys.stderr,
        )
    try:
        learner.save_learner(save_path)
    except OSError as exc:
        print(
            f"Warning: Failed to save learner: {exc}. "
            f"The operation results are not persisted.",
            file=sys.stderr,
        )
        return
    quiet = getattr(args, 'quiet', False)
    if not quiet:
        print(f"\nLearner saved to: {save_path}")


def _load_learner_if_requested(args):
    """Load a :class:`SurvivalLearner` from ``--load-learner`` if provided.

    Returns the loaded learner, or ``None`` if the flag was not given.
    Raises :class:`FileNotFoundError`, :class:`ValueError`, or
    :class:`RuntimeError` on failure — all caught by ``main()``'s
    existing except handlers.
    """
    load_path = getattr(args, 'load_learner', None)
    if not load_path:
        return None

    from mlsurv import SurvivalLearner

    learner = SurvivalLearner.load_learner(load_path)

    # Sanity-check the loaded state for required attributes.
    if not getattr(learner, '_state', None) or not learner._state.is_setup:
        raise ValueError(
            f"Loaded learner from '{load_path}' is not in a valid state "
            f"(setup was not completed). Save the learner after setup() "
            f"or train() to get a usable file."
        )

    # Validate that the requested model exists in the loaded learner.
    model_name = getattr(args, 'model', None)
    if model_name and model_name not in learner.models:
        available = sorted(learner.models.keys())
        raise ValueError(
            f"Model '{model_name}' not found in loaded learner. "
            f"Available: {', '.join(available) or '(none trained)'}"
        )

    quiet = getattr(args, 'quiet', False)
    if not quiet:
        n_models = len(learner.models)
        print(
            f"Loaded learner from: {load_path} "
            f"({n_models} model{'s' if n_models != 1 else ''}, "
            f"{len(learner.X_train)} train / {len(learner.X_test)} test)",
            file=sys.stderr,
        )
    return learner


def _load_artifact_metadata(artifact_dir: Path) -> dict | None:
    """Load and return artifact metadata, printing errors to stderr.

    Returns the parsed metadata dict on success, or None on failure
    (after printing a user-facing error message).
    """
    if not artifact_dir.is_dir():
        print(f"Error: Directory not found: {artifact_dir}", file=sys.stderr)
        return None

    metadata_path = artifact_dir / ARTIFACT_METADATA_FILE
    if not metadata_path.exists():
        print(
            f"Error: {ARTIFACT_METADATA_FILE} not found in: {artifact_dir}",
            file=sys.stderr,
        )
        return None

    try:
        with open(metadata_path) as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Error: Corrupt artifact metadata: {exc}", file=sys.stderr)
        return None
    except OSError as exc:
        print(f"Error: Cannot read {metadata_path}: {exc}", file=sys.stderr)
        return None


def _upgrade_artifact_metadata(metadata: dict) -> list[str]:
    """Apply in-place migrations to artifact metadata.

    Returns a list of human-readable change descriptions.
    """
    changes: list[str] = []
    changes.extend(_migrate_artifact_version(metadata))

    # Back-fill required top-level keys with safe defaults.
    _MISSING_KEY_DEFAULTS = [
        ('timestamp', None, "Added timestamp=null (unknown)"),
        ('environment', {}, "Added empty environment stub"),
        ('data_contract', {}, "Added empty data_contract stub"),
        ('results', {}, "Added empty results stub"),
    ]
    for key, default, description in _MISSING_KEY_DEFAULTS:
        if key not in metadata:
            metadata[key] = default
            changes.append(description)

    return changes


def _migrate_artifact_version(metadata: dict) -> list[str]:
    """Migrate the artifact_version field, returning change descriptions."""
    if 'artifact_version' not in metadata:
        metadata['artifact_version'] = ARTIFACT_VERSION
        return [f"Added artifact_version={ARTIFACT_VERSION}"]

    current = metadata['artifact_version']

    if not isinstance(current, int) or current < 1:
        metadata['artifact_version'] = ARTIFACT_VERSION
        return [f"Fixed invalid artifact_version={current!r} -> {ARTIFACT_VERSION}"]

    if current > ARTIFACT_VERSION:
        return [
            f"Warning: artifact_version={current} is newer than this "
            f"mlsurv supports (max {ARTIFACT_VERSION}). "
            f"Cannot downgrade — upgrade mlsurv instead."
        ]

    if current < ARTIFACT_VERSION:
        metadata['artifact_version'] = ARTIFACT_VERSION
        return [f"Bumped artifact_version from {current} to {ARTIFACT_VERSION}"]

    # Already at current version.
    return []


def _print_artifact_summary(artifact_dir: Path, metadata: dict) -> None:
    """Print a human-readable summary of artifact metadata."""
    if 'artifact_version' not in metadata:
        print(
            "Warning: No artifact_version — pre-versioned artifact. "
            "Run 'mlsurv artifact-upgrade' to update.",
            file=sys.stderr,
        )
    version = metadata.get('artifact_version', '(missing)')

    print(f"Artifact: {artifact_dir}")
    print(f"  version:    {version}")
    print(f"  step:       {metadata.get('step', '(missing)')}")
    print(f"  model:      {metadata.get('model_name', '(missing)')}")
    print(f"  timestamp:  {metadata.get('timestamp', '(missing)')}")

    env = metadata.get('environment')
    if isinstance(env, dict):
        print(f"  mlsurv:     {env.get('mlsurv_version', '?')}")
        print(f"  python:     {env.get('python_version', '?')}")

    contract = metadata.get('data_contract')
    if isinstance(contract, dict):
        print(f"  features:   {len(contract.get('feature_names', []))}")
    else:
        print("Warning: No data_contract in artifact.", file=sys.stderr)

    _print_results_section(metadata.get('results'))
    print(f"  has_model:  {(artifact_dir / ARTIFACT_MODEL_FILE).exists()}")


def _print_results_section(results) -> None:
    """Print the results subsection of an artifact summary."""
    if not isinstance(results, dict) or not results:
        return
    mean = results.get('cv_metric_mean')
    std = results.get('cv_metric_std')
    if mean is not None and std is not None:
        try:
            print(f"  cv_metric:  {float(mean):.4f} +/- {float(std):.4f}")
        except (TypeError, ValueError):
            print(f"  cv_metric:  {mean} +/- {std}")
    scoring = results.get('scoring_metric')
    if scoring:
        print(f"  scoring:    {scoring}")


def cmd_artifact_inspect(args) -> int:
    """Inspect artifact metadata without loading the model."""
    artifact_dir = Path(args.dir)
    metadata = _load_artifact_metadata(artifact_dir)
    if metadata is None:
        return EXIT_DATA

    if args.as_json:
        print(json.dumps(metadata, indent=2))
        return 0

    _print_artifact_summary(artifact_dir, metadata)
    return 0


def _resolve_upgrade_output_dir(source_dir: Path, output: str | None):
    """Resolve and prepare the output directory for artifact upgrade.

    Returns (output_dir, error_code) — error_code is None on success.
    """
    import shutil

    if not output:
        return source_dir, None
    output_dir = Path(output)
    if output_dir.exists():
        print(
            f"Error: Output directory already exists: {output_dir}",
            file=sys.stderr,
        )
        return None, EXIT_ERROR
    try:
        shutil.copytree(source_dir, output_dir)
    except OSError as exc:
        print(f"Error: Failed to copy artifact: {exc}", file=sys.stderr)
        return None, EXIT_ERROR
    return output_dir, None


def cmd_artifact_upgrade(args) -> int:
    """Upgrade artifact metadata to the current schema."""
    source_dir = Path(args.dir)
    metadata = _load_artifact_metadata(source_dir)
    if metadata is None:
        return EXIT_DATA

    changes = _upgrade_artifact_metadata(metadata)

    if not changes:
        print("Artifact is already up to date — no changes needed.")
        return 0

    if args.dry_run:
        print("Dry run — the following changes would be applied:")
        for change in changes:
            print(f"  - {change}")
        return 0

    output_dir, err = _resolve_upgrade_output_dir(source_dir, args.output)
    if err is not None:
        return err

    try:
        with open(output_dir / ARTIFACT_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except OSError as exc:
        print(
            f"Error: Failed to write upgraded metadata: {exc}",
            file=sys.stderr,
        )
        if args.output and output_dir.exists():
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
        return EXIT_ERROR

    for change in changes:
        print(f"  - {change}")
    print(f"\nUpgraded artifact written to: {output_dir}")
    return 0
