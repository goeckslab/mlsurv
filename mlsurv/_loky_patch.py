"""Wrapper for loky resource_tracker.main that suppresses KeyError tracebacks.

The loky resource tracker has a race condition on Windows where multiple
workers try to UNREGISTER the same temp folder, causing noisy KeyError
tracebacks on stderr. This wrapper installs a selective sys.excepthook
that suppresses only those specific errors.

Used by monkeypatching loky's resource_tracker.main reference at mlsurv
import time, so the tracker subprocess imports this wrapper instead of
the original.
"""

import sys

# Save the ORIGINAL main before mlsurv.__init__ replaces it.
# This module is imported by __init__.py *before* the replacement happens,
# so _original_main always points to loky's real main function.
from joblib.externals.loky.backend.resource_tracker import (
    main as _original_main,
)


def main(fd, verbose=False):
    _orig_excepthook = sys.excepthook

    def _quiet_excepthook(exc_type, exc_value, exc_tb):
        if exc_type is KeyError and exc_tb is not None:
            filename = getattr(exc_tb.tb_frame.f_code, "co_filename", "")
            if "resource_tracker" in filename:
                return
        _orig_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _quiet_excepthook
    _original_main(fd, verbose)
