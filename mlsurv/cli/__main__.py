"""Allow ``python -m mlsurv.cli``."""
import sys
from . import main
sys.exit(main())
