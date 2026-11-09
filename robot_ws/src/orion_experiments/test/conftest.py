"""Make orion_learning importable when it is not pip-installed (CI: pip install -e learning)."""

import os
import sys

try:
    import orion_learning  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'learning')))
sys.path.insert(0, os.path.dirname(__file__))
