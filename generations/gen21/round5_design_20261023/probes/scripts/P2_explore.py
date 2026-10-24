#!/usr/bin/env python
"""P2 (idea M34) exploration: check the stored round-4 OOF parquets before the replay is written.
Read-only.  Run from the repo root with -B (no bytecode written into the round-4 tree):
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/probes/scripts/P2_explore.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
R4DIR = Path(__file__).resolve().parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import metrics as M4  # noqa: E402
from r4 import paths  # noqa: E402

KEY = ["split_seed", "cell_id", "read_set", "A", "B", "draw", "k", "policy"]

for design in ("BPG", "BP"):
    ref = pd.read_parquet(paths.PRED / f"REF_{design}.parquet")
    print(design, "policies", ref.policy.unique(), "k", sorted(ref.k.unique()), "k_max", sorted(ref.k_max.unique()),
          "draw", sorted(ref.draw.unique())[:5], "data_version", ref.data_version.unique(), "design", ref.design.unique())
    b = ref[(ref.k_max >= 3) & (ref.k == 3)]
    print(" C3 k3 rows per seed", b.groupby("split_seed").size().to_dict())
    print(" cells per seed", b.groupby("split_seed").cell_id.nunique().to_dict())
    print(" dup keys", b.duplicated(KEY).sum())
    # read sets stable across seeds?
    rs = b.groupby(["cell_id", "split_seed"]).read_set.first().unstack()
    print(" cells whose read set differs across seeds", int((rs.nunique(axis=1) > 1).sum()))
    b = M4.add_clusters(b, None)
    one = b[b.split_seed == b.split_seed.iloc[0]]
    for u in ("cluster_cpg", "publication", "chemotype"):
        s = one.groupby(u).size()
        print(f" {u}: n={len(s)} kish={M4.kish(one, u):.2f} max share={s.max() / s.sum():.3f} min rows={s.min()}")
    one = one.assign(pc=one.publication.astype(str) + "|" + one.chemotype.astype(str))
    s = one.groupby("pc").size()
    print(f" pub x chemotype: n={len(s)} kish={(s.sum() ** 2 / (s ** 2).sum()):.2f}")
    # ties B1 == B0 per cluster
    for u in ("cluster_cpg", "publication", "chemotype"):
        g = b.assign(e0=(b.B0 - b.y) ** 2, e1=(b.B1 - b.y) ** 2).groupby(u)[["e0", "e1"]].sum() / b.split_seed.nunique()
        d = g.e0 - g.e1
        print(f"  B1-B0 {u}: win {int((d > 1e-12).sum())} loss {int((d < -1e-12).sum())} tie {int((d.abs() <= 1e-12).sum())}"
              f" mean log ratio {np.log(g.e0 / g.e1).mean():+.4f}")
    for slot in ("S01", "S02", "S05", "S06", "S11"):
        s = pd.read_parquet(paths.PRED / f"{slot}_{design}.parquet")
        s = s[(s.k_max >= 3) & (s.k == 3)]
        m = b[KEY + ["B1", "y"]].merge(s[KEY + ["B1", "y", slot]], on=KEY, how="outer", suffixes=("", "_s"), indicator=True)
        print(f" {slot}: rows {len(s)} merge {m._merge.value_counts().to_dict()} max|B1-B1_s| "
              f"{np.nanmax(np.abs(m.B1 - m.B1_s)):.2e} max|y-y_s| {np.nanmax(np.abs(m.y - m.y_s)):.2e} "
              f"nan slot {int(m[slot].isna().sum())}")
    print(" HIST_D7 nan", int(b.HIST_D7.isna().sum()), "HBK@ML nan", int(b["HBK@ML"].isna().sum()))
