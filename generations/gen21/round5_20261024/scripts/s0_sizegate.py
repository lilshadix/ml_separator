#!/usr/bin/env python
"""Stage-0 size gate of the confirmation package (CLAUDE_CODE_PROMPT.md section 4.3 item 2; section 4.2; section 8
"Discriminating power").  Design fixed in this docstring before it was run.

A test is dropped before sealing if its size exceeds 0.035 (the lower limit of the two-sided 95 % Clopper-Pearson
interval, i.e. a one-sided 97.5 % bound; P2's convention) under the i.i.d. nulls or under two-stage (chemotype ->
laboratory) nulls.  This script only measures and flags; the sealed fallback for a
flagged statistic (exclusion list, or one value per family / chemotype with a fresh size check) is a PROTOCOL decision.

Statistics (one-sided 0.025 each; r5/confirm.py):
  CGD_W   C-GD laboratory-mean chord 95 % t-interval, laboratory values winsorised to [-0.40, +0.20]   (primary)
  CGD_R   the same, raw                                                                                  (sensitivity)
  GDT_R   Gd-dip transport (M33) laboratory-mean chord t-interval, raw.  Under the parametric nulls it is the same
          computation on the same draws as CGD_R; it differs in its G floor (10 laboratories) and in the corpus pool
          that gates it (non-DGA: "the non-DGA interval decides").
  rules per chord statistic: confirm (UB < beta), kill (LB > beta), helps (UB < beta/2), harm (LB > beta/2) gate;
          A1 (UB < 0) and LB > 0 are descriptive (reported, never gating).  Size is the rule's hit rate when the
          laboratory-mean chord (the raw population mean) equals the rule's reference; for CGD_W, wins_bias is the
          mean winsorised laboratory mean minus that reference.
  ECT_SF  C-ECT laboratory-level exact one-sided sign-flip (H0: mean laboratory log ratio <= 0; full enumeration, G <= 13)
  ECT_SIGN  C-ECT exact binomial sign test on laboratory wins (H0: median 0; ties dropped)
Nulls (>= 2,000 replicates per cell; default 20,000, as in clustered_null_size.csv; one RNG per cell, seeded with
paths.seed_for(<cell key>)):
  two-stage: laboratory value = mu0 + b_chem + e_lab (normal), the 7 compositions and the sd_within grid
    {0.05, 0.08, 0.10} of ideas2/confirm1skepB_out/clustered_null_size.csv, ICC in {0.23, 0.3, 0.5, 0.69}.
    C-ECT's tests are scale-free, so its two-stage cells use sd_within = 1 (ICC x composition only).
  i.i.d.: normal and Student-t(3) laboratory values (chord: sd 0.10, 0.12, 0.14, 0.16; C-ECT: scale-free), G 6, 8, 10,
    13 (chord) and 6, 7, 8, 10, 13 (C-ECT); and seen corpus pools recentred to the null (label-based Stage-0 planning
    input, disclosed): chord - pub x chemotype unit chords of ideas2/confirm1skepB_out/strata_units.csv, pools
    heavy-selective non-DGA (12 units; gates C-GD; N-donor dominated, the nearest corpus proxy), non-DGA (25; gates
    C-GD and GDT) and all 44 (with the DGA right tail; context, never gating); C-ECT - the 14 P1 PRIMARY family log
    ratios (ideas2/confirm4_skepB_out/units_primary.csv) recentred to mean 0 (sign-flip) or median 0 (SIGN), its
    mirror image (emp_P1_mirror = -(V - centre): left-skewed, the dangerous orientation for a one-sided sign-flip;
    s0-statistics-2), and the sign-symmetrised version (exact by construction).  The two skewed seen-pool nulls of
    C-ECT are reported as descriptive rows and never gate (brief 4.2: no test holds 0.025 under skewed heterogeneous
    nulls; skewed nulls are non-gating [PC P2; P2 3]).
Technical check: CGD_R's confirm-rule size at the three ICC values shared with clustered_null_size.csv (20,000 reps per
cell, beta -0.0923; the raw t is location-invariant) is compared with that file.

    D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s0_sizegate.py
writes results/stage0/size_gate.csv, results/stage0/size_gate.json
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
SD_IID = (0.10, 0.12, 0.14, 0.16)
G_IID_CHORD = (6, 8, 10, 13)
G_IID_ECT = (6, 7, 8, 10, 13)
CHORD_POOLS = {"emp_HS12": (("nonDGA_HS",), ("CGD_W", "CGD_R")),
               "emp_nonDGA25": (("nonDGA_HS", "nonDGA_LS"), ("CGD_W", "CGD_R", "GDT_R")),
               "emp_all44": (("DGA", "nonDGA_HS", "nonDGA_LS"), ())}
ALL_CHORD = tuple(CF.CHORD_STATS)


def _row(meta, claim, sid, role, ref, mu0, tail, rule, G, gating, hit, R, extra=None):
    k = int(hit.sum())
    lo, hi = CF.clopper_pearson(k, R)
    r = {**meta, "claim": claim, "statistic": sid, "role": role, "reference": ref, "ref_value": mu0, "tail": tail,
         "rule": rule, "G": G, "decidable": G >= CF.G_FLOOR[claim], "gating": bool(gating), "reps": R,
         "rejections": k, "size": k / R, "cp_lo": lo, "cp_hi": hi, "flagged": lo > CF.SIZE_FLAG}
    r.update(extra or {})
    return r


def chord_cell(rows, Z, meta, beta, gate_stats):
    """Z: R x G mean-0 draws; laboratory values are mu0 + Z at each rule's reference mu0."""
    R, G = Z.shape
    for ref, mu0 in CF.chord_refs(beta).items():
        X = mu0 + Z
        for sid, (claim, clip, role) in CF.CHORD_STATS.items():
            lo, hi = CF.t_interval(X, clip=clip)
            bias = float(np.clip(X, *clip).mean() - mu0) if clip else 0.0
            for r_ref, tail, rule, gating in CF.CHORD_RULES:
                if r_ref == ref:
                    hit = (hi < mu0) if tail == "lower" else (lo > mu0)
                    rows.append(_row(meta, claim, sid, role, ref, mu0, tail, rule, G, gating and sid in gate_stats,
                                     hit, R, {"wins_bias": bias}))


def ect_cell(rows, X_sf, X_sign, meta, gating: bool = True):
    R, G = X_sf.shape
    p_sf, p_sign = CF.signflip_p(X_sf), CF.sign_test_p(X_sign)
    rows.append(_row(meta, "C-ECT", "ECT_SF", "primary", "zero", 0.0, "upper", "confirm", G, gating,
                     p_sf <= CF.ALPHA + 1e-12, R))
    rows.append(_row(meta, "C-ECT", "ECT_SIGN", "win share", "zero", 0.0, "upper", "win-share", G, gating,
                     p_sign <= CF.ALPHA + 1e-12, R))


def main() -> int:
    from r5 import sealcheck
    sealcheck.refuse_if_sealed("scripts/s0_sizegate.py (rewrites sealed Stage-0 inputs)", "--amend" in sys.argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--amend", action="store_true", help="rerun after the seal (an AMENDMENTS.md entry and a reseal follow)")
    ap.add_argument("--reps", type=int, default=20000)
    a = ap.parse_args()
    R = a.reps
    if R < 2000:
        raise SystemExit("section 4.3 item 2 needs >= 2,000 replicates per null cell")
    t0 = time.time()
    beta = CF.load_beta_frozen()
    B = beta["value"]
    comps, sdw_grid, ref_tab = CF.load_compositions()
    chords = CF.load_unit_chords()
    L4, _, _ = CF.load_ect_units()
    rows = []
    # ---- chord statistics, two-stage nulls
    for sdw in sdw_grid:
        for icc in CF.ICC_GRID:
            for comp in comps:
                key = f"s0_sizegate|chord|two_stage|icc={icc}|sdw={sdw}|comp={comp}"
                Z = CF.two_stage(np.random.default_rng(paths.seed_for(key)), R, comp, sdw, icc)
                sdb = sdw * np.sqrt(icc / (1 - icc))
                meta = {"null_family": "two_stage", "null_kind": "normal", "icc": icc, "sd_within": sdw,
                        "sd_between": sdb, "sd_total": float(np.hypot(sdw, sdb)), "comp": str(comp), "seed_key": key}
                chord_cell(rows, Z, meta, B, ALL_CHORD)
    print(f"{time.time() - t0:6.1f}s chord two-stage done", flush=True)
    # ---- chord statistics, i.i.d. nulls
    for G in G_IID_CHORD:
        for kind in ("normal", "t3"):
            for sd in SD_IID:
                key = f"s0_sizegate|chord|iid|{kind}|sd={sd}|G={G}"
                Z = CF.iid(np.random.default_rng(paths.seed_for(key)), R, G, kind, sd)
                meta = {"null_family": "iid", "null_kind": kind, "icc": np.nan, "sd_within": np.nan,
                        "sd_between": np.nan, "sd_total": sd, "comp": f"iid({G})", "seed_key": key}
                chord_cell(rows, Z, meta, B, ALL_CHORD)
        for pname, (classes, gate_stats) in CHORD_POOLS.items():
            v = chords[chords.cls.isin(classes)].chord.to_numpy(float)
            key = f"s0_sizegate|chord|iid|{pname}|G={G}"
            Z = CF.iid(np.random.default_rng(paths.seed_for(key)), R, G, "pool", pool=v - v.mean())
            meta = {"null_family": "iid", "null_kind": f"{pname} (seen, recentred)", "icc": np.nan,
                    "sd_within": np.nan, "sd_between": np.nan, "sd_total": float(v.std(ddof=1)),
                    "comp": f"iid({G})", "seed_key": key}
            chord_cell(rows, Z, meta, B, gate_stats)
    print(f"{time.time() - t0:6.1f}s chord i.i.d. done", flush=True)
    # ---- C-ECT, two-stage nulls (scale-free)
    for icc in CF.ICC_GRID:
        for comp in comps:
            key = f"s0_sizegate|ect|two_stage|icc={icc}|comp={comp}"
            Z = CF.two_stage(np.random.default_rng(paths.seed_for(key)), R, comp, 1.0, icc)
            meta = {"null_family": "two_stage", "null_kind": "normal (scale-free)", "icc": icc, "sd_within": 1.0,
                    "sd_between": float(np.sqrt(icc / (1 - icc))), "sd_total": np.nan, "comp": str(comp),
                    "seed_key": key}
            ect_cell(rows, Z, Z, meta)
    print(f"{time.time() - t0:6.1f}s C-ECT two-stage done", flush=True)
    # ---- C-ECT, i.i.d. nulls
    for G in G_IID_ECT:
        for kind in ("normal", "t3", "emp_P1", "emp_P1_mirror", "swap_P1"):
            key = f"s0_sizegate|ect|iid|{kind}|G={G}"
            rng = np.random.default_rng(paths.seed_for(key))
            if kind in ("normal", "t3"):
                X_sf = X_sign = CF.iid(rng, R, G, kind, 1.0)
            elif kind == "emp_P1":
                V = CF.iid(rng, R, G, "pool", pool=L4)
                X_sf, X_sign = V - L4.mean(), V - np.median(L4)
            elif kind == "emp_P1_mirror":
                V = CF.iid(rng, R, G, "pool", pool=L4)
                X_sf, X_sign = -(V - L4.mean()), -(V - np.median(L4))
            else:
                X_sf = X_sign = CF.iid(rng, R, G, "swap", pool=L4)
            label = {"emp_P1": "emp_P1 (seen, right-skewed; mean-0 for SF, median-0 for SIGN; descriptive)",
                     "emp_P1_mirror": "emp_P1_mirror (seen, left-skewed mirror; descriptive)",
                     "swap_P1": "swap_P1 (seen, sign-symmetrised)"}.get(kind, kind)
            meta = {"null_family": "iid", "null_kind": label, "icc": np.nan, "sd_within": np.nan,
                    "sd_between": np.nan, "sd_total": np.nan, "comp": f"iid({G})", "seed_key": key}
            ect_cell(rows, X_sf, X_sign, meta, gating=not kind.startswith("emp_P1"))
    print(f"{time.time() - t0:6.1f}s C-ECT i.i.d. done", flush=True)

    df = pd.DataFrame(rows)
    # ---- technical check against the seen clustered_null_size.csv (CGD_R confirm rule = its lower-tail lab t)
    m = df[(df.null_family == "two_stage") & (df.statistic == "CGD_R") & (df.rule == "confirm")]
    ref_tab = ref_tab.assign(comp=ref_tab["comp"].astype(str))
    chk = m.merge(ref_tab, left_on=["sd_within", "icc", "comp"], right_on=["sd_within", "icc", "comp"])
    se = np.sqrt(chk.size_lab_t * (1 - chk.size_lab_t) * (1 / R + 1 / 20000))
    zc = (chk["size"] - chk.size_lab_t) / se
    repro = {"cells": int(len(chk)), "max_abs_diff": float((chk["size"] - chk.size_lab_t).abs().max()),
             "max_abs_z": float(zc.abs().max()), "share_abs_z_gt_2": float((zc.abs() > 2).mean()),
             "note": "CGD_R confirm-rule size vs size_lab_t (20,000 reps, beta -0.0923; raw t is location-invariant)"}

    # ---- summaries
    g = df[df.gating]
    summ = {}
    for sid, q in g.groupby("statistic"):
        fl = q[q.flagged]
        top = q.loc[q["size"].idxmax()]
        summ[sid] = {
            "claim": q.claim.iloc[0], "role": q.role.iloc[0], "gating_rows": int(len(q)),
            "flagged_rows": int(len(fl)), "flagged_rows_decidable_G": int(fl.decidable.sum()),
            "flagged_two_stage": int((fl.null_family == "two_stage").sum()),
            "flagged_iid": int((fl.null_family == "iid").sum()),
            "max_size": float(top["size"]), "max_size_cell": f"{top.null_kind} icc={top.icc} sdw={top.sd_within} "
                                                             f"comp={top.comp} rule={top.rule}",
            "max_cp_lo": float(q.cp_lo.max()),
            "verdict": "FLAGGED: fails the size gate (fallback must be sealed, section 4.3 item 2)" if len(fl)
            else "passes the size gate on every gating cell",
            "max_size_by_rule_two_stage_icc": {
                rule: {str(icc): float(v) for icc, v in qq[qq.null_family == "two_stage"].groupby("icc")["size"].max().items()}
                for rule, qq in q.groupby("rule")},
            "max_size_by_rule_iid": {rule: float(qq[qq.null_family == "iid"]["size"].max()) for rule, qq in q.groupby("rule")},
            "flagged_cells": [f"{r.null_family}/{r.null_kind} icc={r.icc} sdw={r.sd_within} comp={r.comp} G={r.G} "
                              f"rule={r.rule} size={r['size']:.4f} cp_lo={r.cp_lo:.4f}" for _, r in fl.head(60).iterrows()]}
    code = [Path(__file__), paths.R5 / "r5" / "confirm.py", paths.R5 / "r5" / "paths.py"]
    out = {
        "label": "STAGE-0 SIZE GATE (synthetic nulls; corpus pools are seen, label-based Stage-0 planning inputs)",
        "brief": "CLAUDE_CODE_PROMPT.md section 4.3 item 2, section 4.2, section 8",
        "rule": f"flagged if the lower limit of the two-sided 95 % Clopper-Pearson interval (one-sided 97.5 %; P2's "
                 f"convention) of the size > {CF.SIZE_FLAG} at nominal one-sided {CF.ALPHA}",
        "reps_per_cell": R, "clip": CF.CLIP, "icc_grid": CF.ICC_GRID, "sd_within_grid": sdw_grid,
        "compositions": [str(c) for c in comps], "sd_iid": SD_IID, "G_iid_chord": G_IID_CHORD, "G_iid_ect": G_IID_ECT,
        "G_floor": CF.G_FLOOR, "beta": beta, "beta_half": B / 2,
        "seed_scheme": "numpy default_rng(paths.seed_for(seed_key)), seed_for(k) = 20261023 + crc32(k); seed_key per row",
        "chord_pools": {k: {"classes": v[0], "gates": v[1]} for k, v in CHORD_POOLS.items()},
        "inputs_sha256_lf": CF.digests([CF.NULL_SIZE_CSV, CF.STRATA_CSV, CF.ECT_UNITS_CSV]),
        "code_sha256_lf": CF.digests(code),
        "disclosure": "Corpus pools (unit chords, P1 family log ratios) are seen label-based values used only as "
                      "planning null distributions (recentred to the null); they assume new laboratories resemble "
                      "corpus units, which the supply does not (P4: 15 of 16 new laboratories on absent chemotypes).",
        "notes": ["The raw t-interval is location-invariant, so CGD_R / GDT_R sizes do not depend on beta; CGD_W does "
                  "(fixed clip bounds), and must be rerun if beta_frozen differs from the value used.",
                  "Under parametric nulls GDT_R equals CGD_R on identical draws; GDT_R decides only at G >= 10, so only "
                  "the G = 13 composition and the i.i.d. G = 10, 13 cells are decidable for it.",
                  "The C-ECT emp_P1 (right-skewed) and emp_P1_mirror (left-skewed) rows are descriptive: skewed nulls "
                  "are non-gating (brief 4.2), and like every test here the sign-flip does not hold 0.025 under them.",
                  "Two-stage cells put each laboratory in one chemotype (worst case for a Gd-dip laboratory value "
                  "averaged across chemotypes).",
                  "C-ECT sign-flip and SIGN are scale-free; the family-level p (I2 confirm-4 item 6) is not re-run here.",
                  "Not covered: C-GD's secondary per-unit Gd-pair SSE sign-flip and the A3 positive control."],
        "reproduction_check": repro, "summary": summ,
        "skewed_seen_nulls_descriptive": {
            f"{r.statistic}|{r.null_kind.split(' ')[0]}|G={r.G}": {"size": float(r["size"]), "cp_lo": float(r.cp_lo),
                                                                   "would_flag": bool(r.flagged)}
            for _, r in df[(df.claim == "C-ECT") & df.null_kind.str.startswith("emp_P1")].iterrows()}}
    OUT.mkdir(parents=True, exist_ok=True)
    paths.write_text_lf(OUT / "size_gate.csv", df.to_csv(index=False, lineterminator="\n"))
    paths.write_text_lf(OUT / "size_gate.json", json.dumps(out, indent=1, default=float) + "\n")
    pd.set_option("display.width", 220)
    tab = g.groupby(["statistic", "rule", "null_family"]).agg(cells=("size", "size"), max_size=("size", "max"),
                                                              max_cp_lo=("cp_lo", "max"), flagged=("flagged", "sum"))
    print(tab.round(4).to_string())
    print("reproduction check:", repro)
    for sid, s in summ.items():
        print(f"{sid}: {s['verdict']} ({s['flagged_rows']} of {s['gating_rows']} gating rows; "
              f"{s['flagged_rows_decidable_G']} at decidable G)")
    if beta["todo"]:
        print(beta["todo"])
    print(f"wrote {OUT / 'size_gate.csv'} ({len(df)} rows)")
    print(f"wrote {OUT / 'size_gate.json'}")
    print(f"{time.time() - t0:6.1f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
