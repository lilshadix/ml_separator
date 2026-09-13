"""Build and cache the pair table of the five reference arms under all five designs.

Writes ``tables/pairs_<design>.parquet`` (one row per split seed x held-out cell x metal pair,
with a prediction column per arm) and ``tables/g14_conf.csv`` (gen14's own held-out probability of
being heavy-selective, per split seed x fold x cell).  Everything downstream reads these.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "generations" / "gen15_curve"))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from gen15 import valuebench as V   # noqa: E402
import dec_arms as D                # noqa: E402

DESIGNS = ("B", "BR", "BQ", "A", "BP")
OUT = HERE / "tables"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bench = V.load()
    probe = D.G14Probe()
    arms = {"FLAT": D.flat, "MEAN_CURVE": D.mean_curve, "G14": probe,
            "G13_FULL": D.g13_full, "O_BOTH": D.o_both}
    boards = []
    for d in DESIGNS:
        t0 = time.time()
        probe.records.clear()
        tab = V.run_arms(bench, arms, d)
        tab.to_parquet(OUT / f"pairs_{d}.parquet", index=False)
        pd.DataFrame(probe.records).to_csv(OUT / f"g14_conf_{d}.csv", index=False)
        pe, bd = V.board(tab, list(arms))
        bd.insert(0, "design", d)
        boards.append(bd)
        print(f"[{d}] {len(tab)} pairs in {time.time() - t0:.0f}s", flush=True)
        print(bd[["arm", "macro_mae_extractant", "macro_sign_acc_strong",
                  "macro_pair_spearman"]].round(4).to_string(index=False), flush=True)
    B = pd.concat(boards, ignore_index=True)
    B.to_csv(OUT / "mae_board.csv", index=False)
    print(V.wide(B).round(4).to_string())


if __name__ == "__main__":
    main()
