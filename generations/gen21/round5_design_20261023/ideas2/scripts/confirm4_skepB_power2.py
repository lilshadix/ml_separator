#!/usr/bin/env python
"""Skeptic B for confirm-4, check 4 (EXPLORATORY planning replay): primary power (exact one-sided sign-flip, 0.025,
SH at 4 reads vs B1 at 4) in a non-DGA world.  W1: corpus PRIMARY units shifted to the non-DGA (no C5) unit mean
+0.140 (nondga_units.csv); W2: resample the 5 non-DGA no-C5 units directly (any n_sib; 4 W / 1 L).
Writes only ideas2/confirm4_skepB_out/power_nondga.csv.
"""
from __future__ import annotations
import itertools
import sys
import zlib
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "confirm4_skepB_out"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
R = 4000


def main() -> int:
    U = pd.read_csv(OUT / "units_primary.csv"); N = pd.read_csv(OUT / "nondga_units.csv")
    L4 = U.L_SH4.to_numpy(); nd = N[(~N.dga) & (~N.c5)].L_SH4.to_numpy()
    pools = {"W1_shift_to_nonDGA_mean": L4 - (L4.mean() - nd.mean()), "W2_nonDGA_units": nd}
    rows = []
    for name, pool in pools.items():
        for tau in (0.0, 0.065):
            for G in (6, 7, 8, 10, 12):
                rng = np.random.default_rng(99 + zlib.crc32(f"{name}|{tau}|{G}".encode()))
                X = pool[rng.integers(len(pool), size=(R, G))]
                if tau > 0:
                    s = tau / np.sqrt(2)
                    X = X + 2 * np.log((1 - rng.normal(0, s, X.shape)) / (1 - rng.normal(0, s, X.shape)))
                E = np.array(list(itertools.product((1.0, -1.0), repeat=G)))
                p = ((X @ E.T) >= X.sum(1)[:, None] - 1e-12).sum(1) / E.shape[0]
                kill = (X.mean(1) <= 0) | ((X > 0).mean(1) < 4 / 7)
                rows.append({"world": name, "pool_mean": pool.mean(), "tau": tau, "G": G, "power": (p <= 0.025 + 1e-12).mean(),
                             "kill": kill.mean()})
    r = pd.DataFrame(rows); r.to_csv(OUT / "power_nondga.csv", index=False)
    print(r.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
