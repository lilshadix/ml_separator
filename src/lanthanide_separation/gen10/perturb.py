"""Building candidate designs that were never measured — the deployment question.

A user proposing an experiment does not hand over rows from this corpus.  They
hand over *conditions*: "I have this extractant, I am going to run a titration at
0.01, 0.03, 0.1 and 0.3 M, what do you expect?"  Some of those points exist in the
corpus and some do not, and the ones that do not are exactly the ones a query-set
consistency study needs — a decoy candidate two decades above the measured window
has no measured value by construction.

So this module synthesises a candidate row: copy an existing row of the same
curve, overwrite one condition, and **recompute every column that is a function of
it**.  That last part is where a careless version would go wrong, because the
design matrix carries the same concentration three times — raw
(``cond__extractant_concentration_M``), logged
(``massact__log10_cond__extractant_concentration_M``) and multiplied by the
denticity and by the acid term (``massact__logL_x_*``).  Overwriting the raw column
alone would produce a row whose logged copy disagrees with it, and the model would
answer a question about a chemically impossible point.

Only the two concentration axes can be synthesised.  A synthetic *metal* point
would need a different element's atomic number, ionic radius, 4f count, radial
basis expansion and one-hot — that is a different experiment, not a different
point on this one, and the lanthanide axis is handled by subsetting only.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pandas as pd

#: Axes a candidate point may be synthesised on, with the raw column each is
#: expressed in.  ``metal`` is deliberately absent; see the module docstring.
SYNTHESISABLE_AXES: dict[str, str] = {
    "cond__extractant_concentration_M": "cond__extractant_concentration_M",
    "cond__acid_concentration_M": "cond__acid_concentration_M",
}

#: Continuous condition columns whose log10 copy lives in the MASSACTION block.
LOGGED_COLUMNS: tuple[str, ...] = (
    "cond__acid_concentration_M", "cond__contact_time_min",
    "cond__extractant_concentration_M", "cond__metal_concentration_mM",
    "cond__temperature_C")


def _synthetic_row_id(source: str, axis: str, value: float) -> str:
    """Deterministic, collision-resistant, and never a real ``row_id``.

    BLAKE2b rather than Python's ``hash``: the builtin is salted per process, and
    gen8 lost cross-run pairing to exactly that.  The ``syn:`` prefix means a
    synthetic row can never be mistaken for a measured one by a join.
    """
    digest = hashlib.blake2b(
        f"{source}|{axis}|{value!r}".encode(), digest_size=8).hexdigest()
    return f"syn:{digest}"


def recompute_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Rebuild every MASSACTION column from the raw condition columns.

    Transcribed from :func:`lanthanide_separation.levels.attach_massaction_block`
    so the two cannot drift; the test suite asserts that recomputing on unmodified
    cohort rows reproduces the stored columns exactly.
    """
    out = frame.copy()
    logs: dict[str, pd.Series] = {}
    for column in LOGGED_COLUMNS:
        target = f"massact__log10_{column}"
        if column not in out.columns or target not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        positive = values.where(values > 0)
        series = np.log10(positive)
        logs[column] = series
        out[target] = series.to_numpy(dtype=float)
    log_l = logs.get("cond__extractant_concentration_M")
    log_h = logs.get("cond__acid_concentration_M")
    if log_l is not None:
        for column in ("DENTATE", "coreCN"):
            target = f"massact__logL_x_{column}"
            if column in out.columns and target in out.columns:
                out[target] = (log_l * pd.to_numeric(out[column], errors="coerce")
                               ).to_numpy(dtype=float)
        if log_h is not None and "massact__logL_x_logH" in out.columns:
            out["massact__logL_x_logH"] = (log_l * log_h).to_numpy(dtype=float)
    return out


def synthesise_points(template: pd.Series, axis: str,
                      values: Sequence[float]) -> pd.DataFrame:
    """New candidate rows on ``axis``, everything else copied from ``template``.

    ``values`` are on the **axis scale** the curve geometry uses — log10 molarity
    for a concentration — because that is the scale the user reasons on and the
    scale a window is defined in.
    """
    if axis not in SYNTHESISABLE_AXES:
        raise ValueError(f"cannot synthesise on {axis!r}; have {sorted(SYNTHESISABLE_AXES)}")
    raw = SYNTHESISABLE_AXES[axis]
    rows = []
    for value in values:
        row = template.copy()
        row[raw] = float(10.0 ** float(value))
        row["row_id"] = _synthetic_row_id(str(template["row_id"]), axis, float(value))
        rows.append(row)
    frame = pd.DataFrame(rows).reset_index(drop=True)
    return recompute_derived(frame)


def is_synthetic(row_ids: Sequence[str]) -> np.ndarray:
    return np.asarray([str(r).startswith("syn:") for r in row_ids])


def assert_no_target(frame: pd.DataFrame, target: str = "log_D") -> None:
    if target in frame.columns:
        raise AssertionError(
            f"a candidate design carries {target!r}; a query is conditions only")
