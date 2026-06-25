"""Command-line interface for mlsurv."""

from __future__ import annotations

from ._artifacts import (
    _check_feature_compatibility,
    _upgrade_artifact_metadata,
    cmd_artifact_inspect,
    cmd_artifact_upgrade,
)
from ._exit_codes import (
    CLIDataError,
    CLIUsageError,
    EXIT_COMPUTATION,
    EXIT_DATA,
    EXIT_ERROR,
    EXIT_INTERRUPT,
    EXIT_USAGE,
)
from ._commands import (
    cmd_benchmark,
    cmd_bootstrap,
    cmd_calibrate,
    cmd_evaluate,
    cmd_explain,
    cmd_export_results,
    cmd_export_study,
    cmd_importance_test,
    cmd_interactions,
    cmd_linearity_test,
    cmd_list_studies,
    cmd_logrank_test,
    cmd_models,
    cmd_patient_report,
    cmd_ph_test,
    cmd_predict,
    cmd_rmst_test,
    cmd_run,
    cmd_setup,
    cmd_sweep,
    cmd_train,
    cmd_tune,
    cmd_validate,
)
from ._plot import cmd_plot
from ._data import (
    _ARG_VALIDATIONS,
    _setup_learner,
    _validate_args,
)
from ._parsers import create_parser

__all__ = [
    'main',
    'create_parser',
    'cmd_export_study',
    '_validate_args',
    '_setup_learner',
    '_check_feature_compatibility',
    '_upgrade_artifact_metadata',
    '_ARG_VALIDATIONS',
]


def main(argv: list | None = None) -> int:
    """Main entry point."""
    import sys

    parser = create_parser()

    # Enable shell completion if argcomplete is installed.
    # Broad catch: completion is optional and must never break the CLI.
    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except (ImportError, TypeError, OSError):
        pass

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    _validate_args(parser, args)

    if getattr(args, 'verbose', False):
        import logging
        import os
        os.environ.setdefault("MLSURV_LOG_LEVEL", "INFO")
        logging.getLogger("mlsurv").setLevel(logging.INFO)

    if getattr(args, 'quiet', False):
        import logging
        logging.getLogger("mlsurv").setLevel(logging.WARNING)

    # Built inside main() so mock.patch on module-level names takes effect.
    _COMMANDS = {
        'run': cmd_run,
        'train': cmd_train,
        'tune': cmd_tune,
        'evaluate': cmd_evaluate,
        'calibrate': cmd_calibrate,
        'bootstrap': cmd_bootstrap,
        'validate': cmd_validate,
        'predict': cmd_predict,
        'benchmark': cmd_benchmark,
        'sweep': cmd_sweep,
        'setup': cmd_setup,
        'models': cmd_models,
        'export-study': cmd_export_study,
        'list-studies': cmd_list_studies,
        'explain': cmd_explain,
        'patient-report': cmd_patient_report,
        'artifact-inspect': cmd_artifact_inspect,
        'artifact-upgrade': cmd_artifact_upgrade,
        'ph-test': cmd_ph_test,
        'logrank-test': cmd_logrank_test,
        'rmst-test': cmd_rmst_test,
        'importance-test': cmd_importance_test,
        'linearity-test': cmd_linearity_test,
        'interactions': cmd_interactions,
        'export-results': cmd_export_results,
        'plot': cmd_plot,
    }

    try:
        handler = _COMMANDS.get(args.command)
        if handler is None:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return EXIT_ERROR
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_INTERRUPT
    except CLIDataError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_DATA
    except CLIUsageError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_USAGE
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_DATA
    except ImportError as e:
        print(
            f"Error: Missing dependency: {e}\n"
            f"Install the required package and retry.",
            file=sys.stderr,
        )
        return EXIT_ERROR
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_DATA
    except (RuntimeError, ArithmeticError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_COMPUTATION
    except OSError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_ERROR
    except (MemoryError, RecursionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_COMPUTATION
    except (TypeError, AttributeError, KeyError, IndexError,
            NotImplementedError) as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        else:
            print("Run with --verbose for the full traceback.", file=sys.stderr)
        return EXIT_ERROR

