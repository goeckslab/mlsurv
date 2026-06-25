"""Semantic exit codes and CLI exceptions for the mlsurv CLI."""

EXIT_SUCCESS = 0
EXIT_ERROR = 1        # general/unexpected error
EXIT_USAGE = 2        # argument/usage error (matches argparse convention)
EXIT_DATA = 3         # file not found, parse failure, validation
EXIT_COMPUTATION = 4  # model training/eval failure, convergence
EXIT_INTERRUPT = 130  # keyboard interrupt


class CLIDataError(Exception):
    """Raised for data-related CLI errors (file not found, parse failure)."""


class CLIUsageError(Exception):
    """Raised for argument/usage errors in CLI command handlers."""
