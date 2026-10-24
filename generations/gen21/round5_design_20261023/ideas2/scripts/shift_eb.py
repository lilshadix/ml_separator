#!/usr/bin/env python
"""shift theme, step 6 (EXPLORATORY; the lambda sweep is HINDSIGHT, the empirical-Bayes lambda is label-free with
respect to the target but was written after seeing steps 4-5).

Because the read correction is linear in the template and the common slope cancels in every log SF, a per-metal
slope model with metal-specific deviations scaled by lambda gives, at every read count k, exactly
    X_k(lambda) = ACOPY_k + lambda * (AREG_k - ACOPY_k).
1. lambda sweep 0 ... 1 on A2 (hindsight).
2. Label-free James-Stein lambda per target from its acid-only siblings only: per metal OLS slope b_m with
   residual variance s2_m (dof = n_AS - 2 >= 1), se2_m = s2_m / Sxx; lambda_EB = clip(1 - mean(se2) / var_m(b_m - bbar),
   0, 1); lambda_EB = 0 when dof < 1.  AEB; AEBG = AEB with lambda = 0 when the target acid lies outside the
   siblings' range.  Shrunk to B1 with w = n_AS / (n_AS + 1.78).

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/shift_eb.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sa", HERE / "shift_analyze.py")
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)
OUT = sa.OUT
K = 1.78
FAMS_TRIVIAL = {"contact_time", "metal_conc"}


def main():
    T = pd.read_parquet(OUT / "shift_rows.parquet")
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    cond_cols = [c for c in f.columns if c.startswith("cond__")]
    x = np.log10(f.cond__acid_concentration_M.astype(float).to_numpy())
    pub, ext = f.publication_id.astype(str).to_numpy(), f.extractant.astype(str).to_numpy()

    def fam(c):
        if c.startswith("cond__acid__"):
            return "acid_type"
        if c == "cond__acid_concentration_M":
            return "acid_conc"
        if c.startswith("cond__additive__"):
            return "additive"
        if c.startswith("cond__diluent__"):
            return "diluent"
        return {"cond__contact_time_min": "contact_time", "cond__extractant_concentration_M": "extr_conc",
                "cond__metal_concentration_mM": "metal_conc", "cond__temperature_C": "temperature"}[c]

    def diff(i, j):
        s = set()
        for c in cond_cols:
            a, b = f.at[i, c], f.at[j, c]
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) or pd.isna(b) or not np.isclose(float(a), float(b)):
                s.add(fam(c))
        return s - FAMS_TRIVIAL

    lam, info = {}, []
    for t in sorted(T[T.n_AS >= 1].cell_id.unique()):
        i = pos[t]
        met = [m for m in range(N_LN) if np.isfinite(Y[i, m])]
        cover = np.all(np.isfinite(Y[:, met]), axis=1)
        cover[i] = False
        AS = [j for j in np.where((pub == pub[i]) & (ext == ext[i]) & cover)[0]
              if t not in v2drop and ids[j] not in v2drop and diff(i, j) == {"acid_conc"}]
        xa = x[AS]
        n = len(AS)
        dof = n - 2
        if dof < 1 or np.ptp(xa) < 0.1:
            lam[t] = 0.0
            info.append({"cell": t, "pub": pub[i], "n": n, "lam": 0.0, "reason": "dof<1"})
            continue
        Sxx = ((xa - xa.mean()) ** 2).sum()
        b = np.array([((xa - xa.mean()) * (Y[AS, m] - Y[AS, m].mean())).sum() / Sxx for m in met])
        s2 = np.array([(((Y[AS, m] - Y[AS, m].mean()) - bb * (xa - xa.mean())) ** 2).sum() / dof for m, bb in zip(met, b)])
        dev = b - b.mean()
        noise = (s2 / Sxx).mean() * (1 - 1 / len(met))
        v = (dev ** 2).mean()
        lam[t] = float(np.clip(1 - noise / v, 0, 1)) if v > 0 else 0.0
        info.append({"cell": t, "pub": pub[i], "n": n, "lam": lam[t], "sd_dev": float(np.sqrt(v)),
                     "se_rms": float(np.sqrt(noise)), "reason": "ok"})
    I = pd.DataFrame(info)
    print("lambda_EB per publication (mean, cells):")
    print(I.groupby("pub").agg(cells=("cell", "size"), n_AS=("n", "mean"), lam=("lam", "mean"),
                               sd_dev=("sd_dev", "mean"), se_rms=("se_rms", "mean")).round(3).to_string())

    for design in ("BPG", "BP"):
        t = T[(T.design == design) & (T.n_AS >= 1)].copy()
        print("\n" + "=" * 90 + f"\nDESIGN {design}\n" + "=" * 90)
        t["lam"] = t.cell_id.map(lam)
        t["lamG"] = np.where(t.extrap == True, 0.0, t.lam)                 # noqa: E712
        w = t.n_AS / (t.n_AS + K)
        for k in (0, 1, 2, 3):
            for nm, lc in (("AEB", "lam"), ("AEBG", "lamG")):
                t[f"{nm}_{k}"] = t[f"ACOPY_{k}"] + t[lc] * (t[f"AREG_{k}"] - t[f"ACOPY_{k}"])
                if k:
                    t[f"{nm}SH_{k}"] = w * t[f"{nm}_{k}"] + (1 - w) * t[f"B1_{k}"]
            for L in (0.25, 0.5, 0.75):
                t[f"AL{L}_{k}"] = t[f"ACOPY_{k}"] + L * (t[f"AREG_{k}"] - t[f"ACOPY_{k}"])
        for nm, d in (("A2", t[(t.reg_ok == True) & ~t.c5]), ("A2_withC5", t[t.reg_ok == True]),    # noqa: E712
                      ("A1", t[~t.c5])):
            print(f"\n[lambda sweep, HINDSIGHT] {nm}: {sa.describe(d)}")
            for k, lab in ((0, "0 reads"), (3, "4 reads")):
                print(f"   {lab}: " + "; ".join(f"lam {L}: {sa.r2(d, c):.4f}" for L, c in
                                                ((0, f"ACOPY_{k}"), (0.25, f"AL0.25_{k}"), (0.5, f"AL0.5_{k}"),
                                                 (0.75, f"AL0.75_{k}"), (1, f"AREG_{k}"))))
            sa.ladder(d, ["B1", "SH", "ACOPY", "AREG", "AEB", "AEBG"], nm)
            for cand, ref in (("AEB_0", "ACOPY_0"), ("AEBG_0", "ACOPY_0"), ("AEB_3", "ACOPY_3"), ("AEBSH_3", "SH_3"),
                              ("AEBGSH_3", "SH_3"), ("AEBG_0", "B1_3"), ("AEBGSH_1", "B1_3")):
                sa.infer(d, cand, ref, nm)


if __name__ == "__main__":
    main()
