"""gen20.1: absolute log D versus adjacent log SF on identical leakage-safe splits.

Every module imports :mod:`gen201.bootstrap` first (IMPLEMENTATION_PLAN.md §8); importing the
package triggers it, so ``import gen201.<anything>`` is always bootstrapped before a frozen import.
"""
from . import bootstrap  # noqa: F401
