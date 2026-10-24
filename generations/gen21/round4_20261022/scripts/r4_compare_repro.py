#!/usr/bin/env python
"""Bit-identity of round-4 reproductions with round 3's stored predictions: predictions/R3-<ID>_<design>.parquet
against round 3's predictions/<ID>_<design>.parquet, every row and every shared column (NaN equal to NaN).

    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_compare_repro.py B0 R14 R02
writes results/reproduction.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402

KEY = ["design", "split_seed", "fold", "cell_id", "k", "A", "B"]


def compare(cid: str, design: str) -> dict:
    a = pd.read_parquet(paths.PRED / f"R3-{cid}_{design}.parquet").sort_values(KEY).reset_index(drop=True)
    b = pd.read_parquet(paths.R3DIR / "predictions" / f"{cid}_{design}.parquet").sort_values(KEY).reset_index(drop=True)
    out = {"rows": [len(a), len(b)], "columns_only_r4": sorted(set(a) - set(b)), "columns_only_r3": sorted(set(b) - set(a))}
    if len(a) != len(b):
        return out | {"identical": False}
    diffs = {}
    for c in sorted(set(a) & set(b)):
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind == "f" or y.dtype.kind == "f":
            same = np.array_equal(x.astype(float), y.astype(float), equal_nan=True)
            if not same:
                diffs[c] = float(np.nanmax(np.abs(x.astype(float) - y.astype(float))))
        elif not (x == y).all():
            diffs[c] = "differs"
    return out | {"identical": not diffs and not out["columns_only_r3"], "differences": diffs}


def main() -> int:
    res = {f"{cid}_{d}": compare(cid, d) for cid in sys.argv[1:] for d in ("BP", "BPG")}
    (paths.RES / "reproduction.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(k, "IDENTICAL" if v["identical"] else f"DIFFERS {v}")
    return 0 if all(v["identical"] for v in res.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
