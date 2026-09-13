"""Permutation null: shuffle ligand identity inside the training fold and re-run.

For every replicate and every design, the 137 ligand-derived columns of the *training* cells are
permuted between extractants (conditions, mass-action columns, weights, publication structure and
targets all stay attached to their own cell), the two learning arms are refitted, and every
headline decision number is recomputed.  FLAT / MEAN_CURVE / O_BOTH are invariant under this
permutation by construction and are not re-run; replicate 0 checks that claim once.

Results are appended to ``perm/null_<design>.csv`` after every replicate, so a partial run is
still usable.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "generations" / "gen15_curve"))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from gen15 import valuebench as V        # noqa: E402
from gen13sep.metals import LANTHANIDES  # noqa: E402
import dec_arms as D                     # noqa: E402
import decmetrics as M                   # noqa: E402

DESIGNS = ("B", "BR", "BQ", "A", "BP")
N_REP = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = HERE / "perm"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bench = V.load()
    lig = D.ligand_columns(bench)
    print(f"[perm] {len(lig)} ligand columns permuted, {N_REP} replicates", flush=True)
    for rep in range(N_REP):
        for d in DESIGNS:
            path = OUT / f"null_{d}.csv"
            if path.exists() and rep in set(pd.read_csv(path)["rep"]):
                continue
            t0 = time.time()
            arms = {"G14_PERM": D.permuted(D.G14Probe(), rep, lig),
                    "G13_PERM": D.permuted(D.g13_full, rep, lig)}
            if rep == 0:
                arms["FLAT_PERM"] = D.permuted(D.flat, rep, lig)
                arms["MEAN_CURVE_PERM"] = D.permuted(D.mean_curve, rep, lig)
                arms["O_BOTH_PERM"] = D.permuted(D.o_both, rep, lig)
            tab = M.prepare(V.run_arms(bench, arms, d, verbose=False), bench.basis, LANTHANIDES)
            rows = []
            for name in arms:
                h = M.headline(tab, name)
                h.update({"rep": rep, "design": d})
                rows.append(h)
            df = pd.DataFrame(rows)
            df.to_csv(path, mode="a", header=not path.exists(), index=False)
            print(f"[perm] rep {rep} design {d} in {time.time() - t0:.0f}s  "
                  + "  ".join(f"{r['arm']} mae {r['macro_mae']:.3f} sign {r['sign_acc']:.3f}"
                              for r in rows), flush=True)


if __name__ == "__main__":
    main()
