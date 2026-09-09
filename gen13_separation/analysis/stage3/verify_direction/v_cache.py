"""Cache the frozen bench inputs so the audit scripts can iterate in seconds."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path("D:/ml_separator_gh/gen13_separation").resolve()))
from gen13sep.amplitude_bench import load_bench, ALL_BLOCKS

OUT = Path("D:/ml_separator_gh/gen13_separation/analysis/stage3/verify_direction")

bench = load_bench("exact", ("radius", "radius_sq"), ALL_BLOCKS)
payload = {
    "frame": bench.frame,
    "Y": bench.Y,
    "basis": bench.basis,
    "coef": bench.coef,
    "n_obs": bench.n_obs,
    "frames": bench.frames,
    "groups": bench.groups,
}
with open(OUT / "bench_cache.pkl", "wb") as fh:
    pickle.dump(payload, fh, protocol=4)

print("cells", len(bench.frame))
print("extractants", bench.frame["extractant"].nunique())
print("chemotypes", bench.frame["chemotype"].nunique())
print("publications", bench.frame["publication_id"].nunique())
print("blocks", {k: v.shape[1] for k, v in bench.frames.items()})
print("coef shape", bench.coef.shape)
print("n_obs distribution", pd.Series(bench.n_obs).value_counts().sort_index().to_dict())
