"""Does the programme's own p-value machinery hold its nominal size on the corpus's clusters?

Generates data with zero mean and a chemotype random effect on the *real* 82-extractant /
40-chemotype membership, then counts how often each test rejects at nominal 5 %.  Every rejection
is a false positive.  Run this before rewriting any published p-value.

Usage:  python generations/gen16_protocol/scripts/g16_size.py [sims] [reps]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen14_direction", ROOT / "generations" / "gen16_protocol"):
    sys.path.insert(0, str(p))

from gen14.dirbench import load                       # noqa: E402
from gen16.clusterboot import css_effective_clusters, size_study   # noqa: E402

SIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 999

bench = load()
frame = bench.frame
rich = frame.n_metals.to_numpy() >= 5
units = frame[rich].drop_duplicates("extractant")
cl = units.chemotype.to_numpy().astype(str)

rows = []
for icc in (0.0, 0.2, 0.4, 0.6, 0.72):
    r = size_study(cl, icc=icc, sims=SIMS, reps=REPS, seed=20260909 + int(icc * 100))
    r.update(css_effective_clusters(cl, rho=icc))
    rows.append(r)
    print(r, flush=True)

out = pd.DataFrame(rows)
out["mc_se"] = (out.size_percentile_bootstrap * (1 - out.size_percentile_bootstrap) / SIMS) ** 0.5
(ROOT / "generations" / "gen16_protocol" / "results").mkdir(parents=True, exist_ok=True)
out.to_csv(ROOT / "generations" / "gen16_protocol" / "results" / "g16_size_study.csv", index=False)
print(out.round(4).to_string(index=False))
