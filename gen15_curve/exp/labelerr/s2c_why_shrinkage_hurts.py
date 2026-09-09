"""Step 2c: the mechanism behind the harm -- what the hierarchical label does to the training target."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from noise import shrink, sigma_table  # noqa: E402
from gen15 import valuebench as V  # noqa: E402
from gen13sep.amplitude_bench import cell_weights  # noqa: E402

OUT = HERE / "results"; OUT.mkdir(exist_ok=True)


def main() -> None:
    bench = V.load(); f = bench.frame
    m = f.n_metals.to_numpy(dtype=float); a = bench.coef[:, 0]
    rich = m >= 5
    ext = f.extractant.astype(str).to_numpy()
    sizes = pd.Series(ext[rich]).value_counts()
    print(f"extractants among the {int(rich.sum())} cells gen14 trains on: {len(sizes)}; "
          f"{int((sizes == 1).sum())} contribute exactly one cell "
          f"({(sizes == 1).mean():.0%}) -- for those the ligand prior is the corpus prior")
    print(sizes.value_counts().sort_index().head(6).to_string())

    se = sigma_table(bench, "resid_const")
    w = cell_weights(bench.groups[rich], bench.n_obs[rich])
    rows = []
    for grp in ("extractant", "chemotype", "none"):
        sh, _ = shrink(bench, np.arange(len(a)), se, group=grp)
        A = sh[:, 0]
        yr, ys = (a[rich] < 0).astype(int), (A[rich] < 0).astype(int)
        rows.append({"group": grp, "n_rich": int(rich.sum()),
                     "flip_rate": float((yr != ys).mean()),
                     "flip_rate_weighted": float(np.average((yr != ys), weights=w)),
                     "frac_heavy_raw": float(np.average(yr, weights=w)),
                     "frac_heavy_shrunk": float(np.average(ys, weights=w)),
                     "flip_rate_small_|a|": float((yr != ys)[np.abs(a[rich]) < 0.1].mean()),
                     "flip_rate_large_|a|": float((yr != ys)[np.abs(a[rich]) >= 0.3].mean())})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "s2c_label_flips.csv", index=False)
    print("\n=== what the hierarchical label does to the direction target on gen14's training cells ===")
    print(d.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
