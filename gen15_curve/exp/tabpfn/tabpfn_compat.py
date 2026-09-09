"""Import TabPFN v2 under scikit-learn 1.9.

TabPFN 2.2.1's vendored ``misc/_sklearn_compat.py`` imports the private helper
``sklearn.utils.validation._is_pandas_df``, which scikit-learn removed (1.9 exposes
``is_pandas_df_or_series`` instead).  The shim below restores the old private name -- it is a
pure rename, nothing about the behaviour of either library changes -- and disables the
package's telemetry before anything is imported.

``load()`` returns ``(TabPFNClassifier, TabPFNRegressor)`` or raises the real error.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")


def _patch_sklearn() -> None:
    import sklearn.utils.validation as V
    if not hasattr(V, "_is_pandas_df"):
        def _is_pandas_df(X) -> bool:
            pd = sys.modules.get("pandas")
            return False if pd is None else isinstance(X, pd.DataFrame)
        V._is_pandas_df = _is_pandas_df


def load():
    """Return (TabPFNClassifier, TabPFNRegressor) with the sklearn shim applied."""
    warnings.filterwarnings("ignore")
    _patch_sklearn()
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    return TabPFNClassifier, TabPFNRegressor
