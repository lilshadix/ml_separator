"""REFUTER lens B for claim L1NULL.  Statistics, confounds, generalisation.

Writes gen16_leads/results/refutation/L1NULL/B/.  Imports the lead's code, never copies it,
never touches anything under gen13_separation/, gen14_direction/, gen15_curve/, gen16_protocol/,
gen16_anchor/, gen17_pairdiff/ or the lead's own files.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa
from gen16 import l1_cycle as L  # noqa

OUT = L.RESULTS.parent / "refutation" / "L1NULL" / "B"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def main():
    rows, count_cols = L.load_energy_rows()
    log("rows", rows.shape, "finite energy", int(np.isfinite(rows['complex_total_energy_eV']).sum()))
    g_all = rows.copy()
    fitted, coefs = L.fit_models(rows, count_cols)
    log("models fitted")
    sl = L.slope_table(fitted, L.MODELS + ("SPECIES_NFILLCOL",), g_all=g_all)
    df = L.attach_sets(sl)
    df.to_csv(OUT / "repro_slopes.csv", index=False)
    log("slopes", df.shape)
    # headline reproduction
    for m in ("NAIVE", "ELEM", "SPECIES", "SPECIES_CONST", "SPECIES_NFILLCOL"):
        for s in ("S8", "S14", "S3"):
            d = df[(df.model == m) & df[s]]
            r, p, n = L._rho(d["slope"].to_numpy(float), d["a"].to_numpy(float))
            print(f"  {m:18s} {s:4s} n={n:3d} rho={r:+.4f} p={p:.4g}")
    df.to_pickle(OUT / "_df.pkl")
    fitted.to_pickle(OUT / "_fitted.pkl")
    pd.to_pickle({k: v for k, v in coefs.items()}, OUT / "_coefs.pkl")
    pd.to_pickle(count_cols, OUT / "_countcols.pkl")


if __name__ == "__main__":
    main()
