#!/usr/bin/env python
"""Round-3 diagnostics from the stored row-level predictions (no refit, no new look at other data):

  results/diagnostics/logd_unread_<ID>_<design>.csv   log D of UNREAD metals only (brief section 16): pooled
                                                      R2 / MAE / RMSE / bias per seed, then the seed mean,
                                                      overall and per metal (Gd separately)
  results/diagnostics/sse_<ID>_<design>.csv           SSE shares and MAE by chemotype and by pair type, and the
                                                      one-read-end / both-hidden split with counts, candidate vs
                                                      PWL_R on the same rows (seed mean)

    .venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_diagnostics.py R23 R14 R18
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3 import paths  # noqa: E402
from r3.metrics import _r2  # noqa: E402

CAND = {"R23": "COR[Dnc]", "R14": "COR[Dnc]", "R18": "COR[Dnc]", "CTRL23": "COR[Dnc]_SHUFFLEDALL"}


def _stats(g: pd.DataFrame, y: str, p: str) -> dict:
    e = g[p].to_numpy(float) - g[y].to_numpy(float)
    return {"R2": _r2(g[y], g[p]), "MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean()), "SSE": float((e ** 2).sum()), "n": len(g)}


def _seed_mean(t: pd.DataFrame, by, y, arms) -> pd.DataFrame:
    rows = []
    keys = [by] if isinstance(by, str) else list(by)
    for (s, *grp), g in t.groupby(["split_seed"] + keys, dropna=False):
        for a in arms:
            rows.append({"split_seed": s, **dict(zip(keys, grp)), "arm": a, **_stats(g, y, a)})
    d = pd.DataFrame(rows)
    return d.groupby(keys + ["arm"], dropna=False)[["R2", "MAE", "RMSE", "bias", "SSE", "n"]].mean().reset_index()


def logd_unread(cid: str, design: str) -> pd.DataFrame:
    t = pd.read_parquet(paths.PRED / f"{cid}_{design}_logd.parquet")
    t = t[(t.k == 3) & (t.k_max >= 3)]
    w = t.pivot_table(index=["split_seed", "fold", "cell_id", "metal"], columns="arm", values="pred").reset_index()
    y = t.drop_duplicates(["split_seed", "fold", "cell_id", "metal"]).set_index(["split_seed", "fold", "cell_id", "metal"]).y
    w = w.join(y, on=["split_seed", "fold", "cell_id", "metal"])
    arms = [a for a in t.arm.unique()]
    w = w.dropna(subset=arms)
    w["all"] = "all"
    w["Gd"] = np.where(w.metal == "Gd", "Gd", "not Gd")
    out = [_seed_mean(w, "all", "y", arms).rename(columns={"all": "group"}),
           _seed_mean(w, "Gd", "y", arms).rename(columns={"Gd": "group"}),
           _seed_mean(w, "metal", "y", arms).rename(columns={"metal": "group"})]
    return pd.concat(out, ignore_index=True)


def sse(cid: str, design: str) -> pd.DataFrame:
    t = pd.read_parquet(paths.PRED / f"{cid}_{design}.parquet")
    t = t[(t.k == 3) & (t.k_max >= 3)].copy()
    arms = [CAND[cid], "PWL_R"]
    t["pair"] = t.A + "/" + t.B
    t["ends"] = np.where(t.a_read | t.b_read, "one_end_read", "both_hidden")
    t["family"] = np.where(t.chemotype == "sc009", "DGA_sc009", "non_DGA")
    out = []
    for by in ("family", "chemotype", "pair", "ends"):
        d = _seed_mean(t, by, "y", arms).rename(columns={by: "group"})
        tot = d.groupby("arm").SSE.transform("sum")
        d["SSE_share"] = d.SSE / tot
        d.insert(0, "by", by)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def main() -> int:
    out = paths.RES / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    for cid in sys.argv[1:]:
        for design in ("BP", "BPG"):
            logd_unread(cid, design).to_csv(out / f"logd_unread_{cid}_{design}.csv", index=False)
            sse(cid, design).to_csv(out / f"sse_{cid}_{design}.csv", index=False)
            print("written", cid, design)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
