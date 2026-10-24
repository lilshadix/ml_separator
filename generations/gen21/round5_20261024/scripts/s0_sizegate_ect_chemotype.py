#!/usr/bin/env python
"""Stage-0 fresh size check for the sealed C-ECT fallback (CLAUDE_CODE_PROMPT.md section 4.3 item 2, fallback (ii);
DECISION D3).  Design fixed in this docstring before it was run.  Synthetic nulls only: no bench value, no prediction,
no round-5 score.

Why.  results/stage0/size_gate.json flags C-ECT's laboratory-level exact sign-flip (ECT_SF: 21 of 48 gating rows,
max size 0.0979) and SIGN (11 of 48) under the two-stage (chemotype -> laboratory) nulls.  D3 seals fallback (ii) for
C-ECT: one value per chemotype (the unweighted mean of the laboratory values inside a chemotype; each laboratory value is
the unweighted mean of its families' log(sum SSE_B1@4 / sum SSE_SH) ratios), an exact one-sided sign-flip over chemotype
units, minimum G = 7 chemotypes.  Brief 4.3 item 2 requires a fresh size check of the fallback statistic.

Statistic (one-sided 0.025; r5/confirm.py):
  ECT_CHEM_SF    exact sign-flip over chemotype units, H0: mean chemotype value <= 0 (full enumeration, G_chem <= 13;
                 r5.confirm.signflip_p).  Decidable only at G_chem >= 7.  GATING.
  ECT_CHEM_SIGN  exact binomial sign test on chemotype wins (ties dropped).  Reported beside it, not a sealed test.
  ECT_LAB_SF / ECT_LAB_SIGN  the laboratory-level versions on the same draws (descriptive under D3), for comparison
                 with size_gate.json (different seeds, so not bit-identical).
Flag: the lower limit of the two-sided 95 % Clopper-Pearson interval of the size (one-sided 97.5 %; P2's convention)
> 0.035 (r5.confirm.SIZE_FLAG), as in size_gate.json.

Nulls (R = 20,000 replicates per cell, as size_gate.json; one RNG per cell seeded with
paths.seed_for("s0_sizegate_ect_chem|<cell key>")):
  two_stage / file   the 7 compositions of ideas2/confirm1skepB_out/clustered_null_size.csv (laboratories per
                     chemotype; r5.confirm.load_compositions) x ICC {0.23, 0.3, 0.5, 0.69}; laboratory value
                     b_chem + e_lab (r5.confirm.two_stage), sd_within = 1 (every statistic here is scale-free).
                     Only (4, 3, 2, 1, 1, 1, 1) reaches G_chem = 7; the others are below the floor (reported).
  two_stage / decl   declared supplementary compositions that reach G_chem >= 7, so that the decidable range is
                     covered: (1,)*7, (2,1,1,1,1,1,1), (2,2,1,1,1,1,1), (3,2,1,1,1,1,1), (2,2,2,1,1,1,1),
                     (2,2,2,2,2,2,2), (3,3,2,2,1,1,1,1), (1,)*10, (4,3,2,2,1,1,1,1,1,1), (1,)*13; same ICC grid.
  iid                one value per chemotype, G_chem in {7, 8, 10, 13}: normal; Student t(3); emp_P1 = the 14 seen P1
                     PRIMARY family log ratios (ideas2/confirm4_skepB_out/units_primary.csv, L_SH4) resampled and
                     recentred to mean 0 (sign-flip) or median 0 (SIGN), a right-skewed null (seen, label-based
                     planning input, disclosed); emp_P1_mirror = its mirror image -(V - centre), the left-skewed
                     orientation that is dangerous for a one-sided sign-flip (s0-statistics-2); both skewed rows are
                     descriptive, never gating (brief 4.2: skewed nulls are non-gating [PC P2; P2 3]); swap_P1 = the
                     same values with random signs (exact by construction).
Reading: under the two-stage nulls, chemotype values are independent and symmetric about 0 (heteroscedastic when
chemotypes hold different numbers of laboratories), so the exact sign-flip over chemotypes keeps its size by
construction; the check confirms it numerically under symmetric nulls and measures the skewed emp_P1 null in both
orientations (descriptive).

    D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s0_sizegate_ect_chemotype.py
writes results/stage0/size_gate_ect_chemotype.csv, results/stage0/size_gate_ect_chemotype.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r5 import confirm as CF  # noqa: E402

OUT = paths.RES / "stage0"
G_FLOOR_CHEM = 7
G_IID = (7, 8, 10, 13)
DECLARED = ((1,) * 7, (2, 1, 1, 1, 1, 1, 1), (2, 2, 1, 1, 1, 1, 1), (3, 2, 1, 1, 1, 1, 1), (2, 2, 2, 1, 1, 1, 1),
            (2, 2, 2, 2, 2, 2, 2), (3, 3, 2, 2, 1, 1, 1, 1), (1,) * 10, (4, 3, 2, 2, 1, 1, 1, 1, 1, 1), (1,) * 13)


def chem_means(X: np.ndarray, comp) -> np.ndarray:
    """R x G_lab laboratory values -> R x G_chem chemotype values (unweighted mean of the laboratories in each)."""
    comp = tuple(int(c) for c in comp)
    edges = np.cumsum((0,) + comp)
    return np.stack([X[:, a:b].mean(1) for a, b in zip(edges[:-1], edges[1:])], axis=1)


def row(meta, stat, role, gating, G, hit, R):
    k = int(hit.sum())
    lo, hi = CF.clopper_pearson(k, R)
    return {**meta, "statistic": stat, "role": role, "G": G, "decidable": bool(G >= G_FLOOR_CHEM) if "CHEM" in stat
            else bool(G >= CF.G_FLOOR["C-ECT"]), "gating": bool(gating), "reps": R, "rejections": k, "size": k / R,
            "cp_lo": lo, "cp_hi": hi, "flagged": bool(lo > CF.SIZE_FLAG)}


def cell(rows, X_lab, comp, meta, X_sign_chem=None, gating: bool = True):
    R = X_lab.shape[0]
    Xc = chem_means(X_lab, comp) if comp is not None else X_lab
    Gc = Xc.shape[1]
    Xs = Xc if X_sign_chem is None else X_sign_chem
    a = CF.ALPHA + 1e-12
    rows.append(row(meta, "ECT_CHEM_SF", "sealed fallback (ii), primary" if gating else
                    "sealed fallback (ii), skewed seen-pool null (descriptive)", gating, Gc, CF.signflip_p(Xc) <= a, R))
    rows.append(row(meta, "ECT_CHEM_SIGN", "reported beside it (not a sealed test)", False, Gc,
                    CF.sign_test_p(Xs) <= a, R))
    if comp is not None:
        Gl = X_lab.shape[1]
        rows.append(row(meta, "ECT_LAB_SF", "descriptive under D3", False, Gl, CF.signflip_p(X_lab) <= a, R))
        rows.append(row(meta, "ECT_LAB_SIGN", "descriptive under D3", False, Gl, CF.sign_test_p(X_lab) <= a, R))


def main() -> int:
    from r5 import sealcheck
    sealcheck.refuse_if_sealed("scripts/s0_sizegate_ect_chemotype.py (rewrites sealed Stage-0 inputs)", "--amend" in sys.argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--amend", action="store_true", help="rerun after the seal (an AMENDMENTS.md entry and a reseal follow)")
    ap.add_argument("--reps", type=int, default=20000)
    a = ap.parse_args()
    R = a.reps
    if R < 2000:
        raise SystemExit("section 4.3 item 2 needs >= 2,000 replicates per null cell")
    t0 = time.time()
    comps_file, _, _ = CF.load_compositions()
    L4, _, _ = CF.load_ect_units()
    rows = []
    for src, comps in (("file", comps_file), ("declared", DECLARED)):
        for icc in CF.ICC_GRID:
            for comp in comps:
                key = f"s0_sizegate_ect_chem|two_stage|{src}|icc={icc}|comp={tuple(comp)}"
                X = CF.two_stage(np.random.default_rng(paths.seed_for(key)), R, comp, 1.0, icc)
                meta = {"null_family": "two_stage", "composition_source": src, "null_kind": "normal (scale-free)",
                        "icc": icc, "comp": str(tuple(comp)), "G_lab": int(sum(comp)), "G_chem": len(comp),
                        "seed_key": key}
                cell(rows, X, comp, meta)
        print(f"{time.time() - t0:6.1f}s two-stage {src} done", flush=True)
    for G in G_IID:
        for kind in ("normal", "t3", "emp_P1", "emp_P1_mirror", "swap_P1"):
            key = f"s0_sizegate_ect_chem|iid|{kind}|G={G}"
            rng = np.random.default_rng(paths.seed_for(key))
            Xs = None
            if kind in ("normal", "t3"):
                X = CF.iid(rng, R, G, kind, 1.0)
            elif kind == "emp_P1":
                V = CF.iid(rng, R, G, "pool", pool=L4)
                X, Xs = V - L4.mean(), V - np.median(L4)
            elif kind == "emp_P1_mirror":
                V = CF.iid(rng, R, G, "pool", pool=L4)
                X, Xs = -(V - L4.mean()), -(V - np.median(L4))
            else:
                X = CF.iid(rng, R, G, "swap", pool=L4)
            label = {"emp_P1": "emp_P1 (seen, right-skewed; mean-0 for SF, median-0 for SIGN; descriptive)",
                     "emp_P1_mirror": "emp_P1_mirror (seen, left-skewed mirror; descriptive)",
                     "swap_P1": "swap_P1 (seen, sign-symmetrised)"}.get(kind, kind)
            meta = {"null_family": "iid", "composition_source": "one value per chemotype", "null_kind": label,
                    "icc": np.nan, "comp": f"iid({G})", "G_lab": np.nan, "G_chem": G, "seed_key": key}
            cell(rows, X, None, meta, X_sign_chem=Xs, gating=not kind.startswith("emp_P1"))
    print(f"{time.time() - t0:6.1f}s iid done", flush=True)

    df = pd.DataFrame(rows)
    summ = {}
    for stat, q0 in df.groupby("statistic"):
        q = q0[q0.gating] if q0.gating.any() else q0              # a gating statistic is judged on its gating rows
        dec = q[q.decidable]
        fl = q[q.flagged]
        top = q.loc[q["size"].idxmax()]
        sk = q0[q0.null_kind.str.startswith("emp_P1")]
        summ[stat] = {
            "role": q.role.iloc[0], "gating": bool(q0.gating.any()), "rows": int(len(q0)), "gating_rows": int(q0.gating.sum()),
            "skewed_seen_null_rows_descriptive": {f"{r.null_kind.split(' ')[0]}|G={r.G}": {
                "size": float(r["size"]), "cp_lo": float(r.cp_lo), "would_flag": bool(r.flagged)} for _, r in sk.iterrows()},
            "rows_decidable_G": int(len(dec)), "flagged_rows": int(len(fl)),
            "flagged_rows_decidable_G": int(fl.decidable.sum()),
            "max_size": float(top["size"]), "max_size_cell": f"{top.null_family}/{top.null_kind} icc={top.icc} "
                                                             f"comp={top.comp} G={top.G}",
            "max_cp_lo": float(q.cp_lo.max()),
            "max_size_decidable_G": float(dec["size"].max()) if len(dec) else None,
            "max_size_by_icc_two_stage": {str(k): float(v) for k, v in
                                          q[q.null_family == "two_stage"].groupby("icc")["size"].max().items()},
            "max_size_iid_by_kind": {k: float(v) for k, v in q[q.null_family == "iid"].groupby("null_kind")["size"].max().items()},
            "flagged_cells": [f"{r.null_family}/{r.null_kind} icc={r.icc} comp={r.comp} G={r.G} size={r['size']:.4f} "
                              f"cp_lo={r.cp_lo:.4f}" for _, r in fl.iterrows()]}
    gating = summ["ECT_CHEM_SF"]
    verdict = ("passes the size gate on every cell" if gating["flagged_rows"] == 0 else
               "FLAGGED: fails the size gate")
    code = [Path(__file__), paths.R5 / "r5" / "confirm.py", paths.R5 / "r5" / "paths.py"]
    out = {
        "label": "STAGE-0 FRESH SIZE CHECK of the sealed C-ECT fallback (ii) (synthetic nulls; the emp_P1 pool is a seen, "
                 "label-based planning input; no round-5 score)",
        "brief": "CLAUDE_CODE_PROMPT.md section 4.3 item 2 (fallback (ii): one value per chemotype, declared test, minimum G, "
                 "fresh size check); DECISION D3",
        "statistic": "chemotype value = unweighted mean of its laboratories' values; exact one-sided sign-flip over "
                     "chemotype units (H0 mean <= 0); confirm at p <= 0.025 with G_chem >= 7",
        "rule": f"flagged if the lower limit of the two-sided 95 % Clopper-Pearson interval (one-sided 97.5 %; P2's "
                 f"convention) of the size > {CF.SIZE_FLAG} at nominal one-sided {CF.ALPHA}",
        "reps_per_cell": R, "icc_grid": CF.ICC_GRID, "compositions_file": [str(tuple(c)) for c in comps_file],
        "compositions_declared": [str(c) for c in DECLARED], "G_iid": G_IID, "G_floor_chemotypes": G_FLOOR_CHEM,
        "min_attainable_p": {str(g): 1 / 2 ** g for g in (6, 7, 8, 10, 13)},
        "seed_scheme": "numpy default_rng(paths.seed_for(seed_key)), seed_for(k) = 20261023 + crc32(k); seed_key per row",
        "inputs_sha256_lf": CF.digests([CF.NULL_SIZE_CSV, CF.ECT_UNITS_CSV]),
        "code_sha256_lf": CF.digests(code),
        "verdict_ECT_CHEM_SF": verdict,
        "summary": summ,
        "notes": ["Two-stage chemotype values are independent and symmetric about 0, so the exact sign-flip over "
                  "chemotypes holds its size by construction (exactness under a sharp, exchangeable null; brief 4.2).",
                  "The emp_P1 rows (right-skewed) and their mirror (left-skewed) are descriptive: skewed nulls are "
                  "non-gating (brief 4.2), and under the mirrored pool the one-sided sign-flip exceeds 0.025, as every "
                  "test here does under skewed heterogeneous nulls.",
                  "Compositions with fewer than 7 chemotypes cannot reach p <= 0.025 at the chemotype level "
                  "(min p 1/2^G_chem); they are below the declared floor and reported for completeness.",
                  "The laboratory-level rows (ECT_LAB_*) reproduce the pattern of size_gate.json on fresh draws; they "
                  "are descriptive under D3.",
                  "emp_P1 is a skewed null built from seen corpus family ratios (13 of 14 DGA); it assumes new "
                  "chemotypes resemble corpus units, which the supply does not (P4: 15 of 16 new laboratories on "
                  "absent chemotypes)."],
        }
    OUT.mkdir(parents=True, exist_ok=True)
    paths.write_text_lf(OUT / "size_gate_ect_chemotype.csv", df.to_csv(index=False, lineterminator="\n"))
    paths.write_text_lf(OUT / "size_gate_ect_chemotype.json", json.dumps(out, indent=1, default=float) + "\n")
    pd.set_option("display.width", 220)
    tab = df.groupby(["statistic", "null_family", "composition_source"]).agg(
        cells=("size", "size"), max_size=("size", "max"), max_cp_lo=("cp_lo", "max"), flagged=("flagged", "sum"))
    print(tab.round(4).to_string())
    print("ECT_CHEM_SF:", verdict)
    print(f"wrote {OUT / 'size_gate_ect_chemotype.csv'} ({len(df)} rows)")
    print(f"wrote {OUT / 'size_gate_ect_chemotype.json'}")
    print(f"{time.time() - t0:6.1f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
