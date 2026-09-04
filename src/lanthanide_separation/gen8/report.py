"""Markdown table rendering without a new dependency.

``DataFrame.to_markdown`` needs ``tabulate``, and this project has already been
burned once by a casual ``pip install`` moving scikit-learn and pandas underneath
a half-finished sweep (see ``runs/gen7_architecture`` on the TabPFN incident).  A
twenty-line renderer is cheaper than a non-comparable leaderboard.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd


def _fmt(value, float_format: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return ""
        return format(float(value), float_format)
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def md_table(frame: pd.DataFrame, *, float_format: str = ".3f",
             columns: Sequence[str] | None = None, index: bool = False) -> str:
    """Render a frame as a GitHub-flavoured markdown table."""
    work = frame if columns is None else frame[list(columns)]
    if index:
        work = work.reset_index()
    header = [str(c) for c in work.columns]
    body = [[_fmt(v, float_format) for v in row] for row in work.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)
