"""Chemotype-blocked paired bootstrap (gen13's own inference) for named arms against G14 and FLAT.

Reads the per-extractant frames a ``run.py`` design saved, so no model is refitted.

    python contrasts.py <batch> ARM1 ARM2 ...
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "generations" / "gen15_curve"))

from gen15 import valuebench as V     # noqa: E402

OUT = HERE / "results"
DESIGNS = ["B", "BR", "BQ", "A", "BP"]


def main() -> None:
    batch, names = sys.argv[1], sys.argv[2:]
    comps = {}
    for n in names:
        comps[f"{n}_vs_G14"] = ("G14", n)
        comps[f"{n}_vs_FLAT"] = ("FLAT", n)
    parts = []
    for d in DESIGNS:
        f = OUT / f"perext_{batch}_{d}.pkl"
        if not f.exists():
            continue
        with f.open("rb") as fh:
            pe = pickle.load(fh)
        have = set(pe["arm"].unique())
        use = {k: v for k, v in comps.items() if v[0] in have and v[1] in have}
        if not use:
            continue
        c = V.contrasts(pe, use)
        c.insert(0, "design", d)
        parts.append(c)
    C = pd.concat(parts, ignore_index=True)
    C.to_csv(OUT / f"contrasts_{batch}.csv", index=False)
    print(C.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
