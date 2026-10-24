#!/usr/bin/env python
"""Skeptic A for confirm-1, E-CT leg and two extra C-GD sensitivities.  EXPLORATORY (already-seen C3 cells).

Reads (read-only) the P1 rows that the confirm author rebuilt with the unchanged P1_ct_build.py
(ideas2/scripts/confirm_out/p1/P1_rows.parquet).  Recomputes, independently of confirm_ect_replay.py:
  * PRIMARY stratum (CLEAN, n_clean >= 2, no C5-BTBP), per family (publication x extractant) and per pubchem unit,
    seed-mean SSE of SH at 4 reads (CL_SH_3) vs B1 at 4 reads (B1_3); log ratio; wins / losses.
  * the same after dropping the two suspected bit-identical copy pairs flagged in PROBE_CHECKS (cells
    44012938119033a5 / 5169c377777d282b, d4c1e23a091d0960 / e4ee526bda3920c2) as targets AND as siblings is not
    possible from rows alone, so targets are dropped (a lower bound on the effect of the duplicates).
  * family mechanism class (SMILES) and perturbation class: how many corpus families are cation exchangers.
  * leave-one-publication-out mean log ratio and win share.
  * exact sign-flip power at G = 5..9 by family bootstrap (EMP), and predictive power with the win-rate uncertain.
  * C-GD sensitivity at beta = -0.118 (full-information) and -0.072 (HIST_D7).
Writes only ideas2/confirm1_skA_out/ect_*.csv and ect_summary.json.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "confirm1_skA_out"
R4DIR = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r4 import donor as DN  # noqa: E402

P1 = HERE.parent / "confirm_out" / "p1" / "P1_rows.parquet"
DUPS = {"44012938119033a5", "5169c377777d282b", "d4c1e23a091d0960", "e4ee526bda3920c2"}
R = 4000


def fam_table(t, unit, cand="CL_SH_3", ref="B1_3"):
    t = t.dropna(subset=[cand, ref])
    g = t.assign(ec=(t[cand] - t.y) ** 2, er=(t[ref] - t.y) ** 2)
    a = g.groupby([unit, "split_seed"])[["ec", "er"]].sum().groupby(level=0).mean()
    a["L"] = np.log(a.er / a.ec)
    meta = t.groupby(unit).agg(pub=("publication", "first"), ext=("extractant", "first"), cls=("cls", lambda s: "/".join(sorted(set(s)))),
                               cells=("cell_id", "nunique"))
    return a.join(meta)


def sf_power(L, G, rng, reps=R):
    X = L[rng.integers(len(L), size=(reps, G))]
    E = np.array(list(itertools.product((1.0, -1.0), repeat=G)))
    T = X.sum(1)
    p = ((X @ E.T) >= T[:, None] - 1e-12).sum(1) / E.shape[0]
    return float((p <= 0.025 + 1e-12).mean())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_parquet(P1)
    t = t[t.design == "BPG"].copy()
    t["pubchem"] = t.publication + "|" + t.chemotype
    t["pubext"] = t.publication + "|" + t.extractant
    clean = ~t.t_v2drop & (t.n_clean >= 1)
    prim = clean & (t.n_clean >= 2) & ~t.c5
    res = {}
    rng = np.random.default_rng(7)
    out = []
    for name, mask in (("PRIMARY", prim), ("PRIMARY_noDUPtargets", prim & ~t.cell_id.isin(DUPS))):
        for unit in ("pubchem", "pubext"):
            a = fam_table(t[mask], unit)
            a["dga"] = a.ext.astype(str).map(DN.is_dga)
            rec = {"set": name, "unit": unit, "units": len(a), "pubs": a.pub.nunique(), "wins": int((a.L > 0).sum()),
                   "losses": int((a.L < 0).sum()), "meanL": float(a.L.mean()), "medianL": float(a.L.median()),
                   "dga_units": int(a.dga.sum())}
            # leave one publication out
            lo = []
            for p in a.pub.unique():
                r = a[a.pub != p]
                lo.append((p, float(r.L.mean()), float((r.L > 0).mean())))
            rec["lopo_meanL_min"] = min(x[1] for x in lo)
            rec["lopo_win_min"] = min(x[2] for x in lo)
            for G in (5, 6, 7, 8, 9):
                rec[f"sfpow_G{G}"] = sf_power(a.L.to_numpy(), G, rng)
            out.append(rec)
            if name == "PRIMARY" and unit == "pubext":
                a.to_csv(OUT / "ect_families.csv")
                print(a.round(3).to_string(), flush=True)
    out = pd.DataFrame(out)
    print(out.round(3).to_string(), flush=True)
    out.to_csv(OUT / "ect_recompute.csv", index=False)
    # predictive power for "all G win" type rejection: win prob ~ Beta(w+.5, l+.5); also worlds with lower win rate
    a = fam_table(t[prim], "pubext")
    w, l = int((a.L > 0).sum()), int((a.L < 0).sum())
    pp = {}
    for G in (6, 7, 8):
        ps = rng.beta(w + 0.5, l + 0.5, size=20000)
        pp[f"P_all_win_G{G}_predictive"] = float(np.mean(ps ** G))
        for pr in (0.93, 0.85, 0.75):
            pp[f"P_all_win_G{G}_p{pr}"] = float(pr ** G)
    res["all_win_prob"] = pp
    res["family_count_by_cls"] = a.cls.value_counts().to_dict()
    print(json.dumps(res, indent=1), flush=True)

    # C-GD beta sensitivity on the HS units written by confirm1_recompute_skA.py
    u = pd.read_csv(OUT / "units.csv", index_col=0)
    hs = u[u.cls == "nonDGA_HS"].cw.to_numpy()
    sens = []
    for beta in (-0.0923, -0.118, -0.072):
        for G in (4, 5, 6, 8, 10):
            X = hs[rng.integers(len(hs), size=(R, G))]
            m, se = X.mean(1), X.std(1, ddof=1) / np.sqrt(G)
            tc = SS.t.ppf(0.975, G - 1)
            sens.append({"beta": beta, "G": G, "pow": float((((m - beta) / se) < -tc).mean())})
    sens = pd.DataFrame(sens).pivot(index="G", columns="beta", values="pow")
    print(sens.round(3).to_string(), flush=True)
    sens.to_csv(OUT / "cgd_beta_sensitivity.csv")
    (OUT / "ect_summary.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
