"""Full reproduction of a stored V1 configuration (R02, R14, R23, ...) under the current code: every fold of one design, compared
row by row with the stored predictions.  Usage: .venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_repro.py R23 BP|BPG <out.json>"""
import json
import sys
import time

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r3 import harness as H, paths, registry as RG  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402

cid, design, out = sys.argv[1], sys.argv[2], sys.argv[3]
b = D.load_bench()
cfg = RG.get(cid, b)
assert cfg.version == "V1", "V1 configurations only (V2 needs the datav2 record transform)"
methods = {**RG.base_methods(b, cfg), **cfg.methods}
st = pd.read_parquet(paths.PRED / f"{cid}_{design}.parquet")
key = ["split_seed", "fold", "cell_id", "A", "B", "k"]
cols = [c for c in methods if c in st.columns]
worst, n, nf, t0 = 0.0, 0, 0, time.time()
for f in folds_for(b.frame, design, seeds=list(paths.DEV_SEEDS)):
    rec = C.load(f.design, f.seed, f.fold)
    blk = dict(rec["outer"], Y=b.Y, policy=cfg.policy, _rec=rec)
    blk["fitted"] = RG.fit(cfg, b, rec, b.Y)
    rows = []
    for cell in RG.cells_of(cfg, blk):
        rows += H.cell_rows(rec, blk, cell, methods, policy=cfg.policy, ks=cfg.ks, Y=b.Y, version="V1")
    t = pd.DataFrame(rows)
    old = st[(st.split_seed == f.seed) & (st.fold == f.fold)]
    m = t.merge(old, on=key, suffixes=("", "_old"))
    assert len(m) == len(t) == len(old), (f, len(m), len(t), len(old))
    for c in cols:
        a, o = m[c].to_numpy(float), m[c + "_old"].to_numpy(float)
        assert np.array_equal(np.isnan(a), np.isnan(o))
        worst = max(worst, float(np.nanmax(np.abs(a - o))) if len(a) else 0.0)
    n += len(m)
    nf += 1
    print(design, f.seed, f.fold, len(m), f"{time.time() - t0:.0f}s", flush=True)
res = {"config": cid, "design": design, "folds": nf, "rows": n, "columns": cols, "max_abs_diff": worst,
       "bit_identical": worst == 0.0}
open(out, "w").write(json.dumps(res, indent=1))
print(res)
