#!/usr/bin/env python
"""Stage-0 supplements to the confirmation package (CLAUDE_CODE_PROMPT.md section 4.3 items 1 and 9; DECISIONS D2, D3,
D4).  Design fixed in this docstring before it was run.  Planning only: no bench value, no prediction, no round-5
score.  Two outputs:

1. results/stage0/power_cgd_beta_sens.{csv,json} -- C-GD / Gd-dip chord power at every beta reference sealed by D2.
   power_worlds.json reports beta_frozen and the confirm-1 sensitivities (-0.118, -0.072) and the rounded round-4 fold
   mean (-0.094); D2 asks for power at the beta_frozen sensitivities of results/stage0/beta_frozen.json too: design BP
   (-0.10950), lambda fixed at 0.03 (-0.08780), and the round-4 fold means (-0.09230 BPG, -0.09488 BP).  The worlds,
   seeds, statistics and rules are those of scripts/s0_power_worlds.py (imported, not copied): normal worlds
   HDEHP_LIKE (-0.26, PROVISIONAL), P507_LIKE (-0.105, PROVISIONAL), HS_CORPUS (-0.217, seen), NONDGA_CORPUS (-0.142,
   seen), NO_DIP (0), AT_BETA and AT_BETA_HALF (centre = the reference / its half), between-laboratory sd in
   {0.10, 0.12, 0.14, 0.16}; empirical bootstrap worlds of the seen unit chords (HS12, NONDGA25, DGA19, ALL44); G 6-13;
   10,000 replicates per world.  Fixed-centre and empirical worlds use power_worlds' seed keys, so the beta_frozen rows
   must reproduce power_worlds.csv exactly (technical check, asserted).  AT_BETA / AT_BETA_HALF use new keys per
   reference name.

2. results/stage0/supply_reach.json -- probability that each claim reaches its minimum G (brief 4.3 item 1), from the
   lab-by-lab supply of results/stage0/supply_audit.json (metadata only) under the two forecast rate sets the audit
   reports (P4: p_E2 0.5, p_U 0.25; PC P4: p_E2 2/6, p_U 0).  Model: verified (E1) units count 1; every E2-only unit
   converts independently with p_E2 and every U-only unit with p_U (the audit's expected-yield formula is the mean of
   this model, asserted); Gd-dip transport adds SAFE at its maximum (3; U2 unanswered), with SAFE = 0 as a sensitivity.
   FORECASTS: never counted toward the decision supply (D4).  C-GD's ">= 3 extractant families" condition is not
   modelled (the probability is an upper bound).  For C-ECT's fallback (ii) unit (D3) the script also counts chemotype
   units from P4's hand chemotype labels (merged for [A336][NO3] = TOMANO3), with laboratories that span chemotypes
   linking them into one unit (connected components; upper bound on the gen6-map count).

    D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s0_package_supplements.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r5 import confirm as CF  # noqa: E402
from r5 import supply as SU  # noqa: E402

OUT = paths.RES / "stage0"
_spec = importlib.util.spec_from_file_location("s0_power_worlds", Path(__file__).with_name("s0_power_worlds.py"))
PW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PW)


# ------------------------------------------------------------------------------------------------ 1. beta sensitivities
def beta_refs_d2() -> list:
    b = json.loads((OUT / "beta_frozen.json").read_text(encoding="utf-8"))
    s = b["sensitivities"]
    return [("beta_frozen", float(b["beta_frozen"])),
            ("sens_design_BP", float(s["design_BP"]["beta"])),
            ("sens_lambda_0.03", float(s["lambda_fixed_0.03"]["beta"])),
            ("r4_fold_mean_BPG", float(s["round4_fold_mean"]["BPG"])),
            ("r4_fold_mean_BP", float(s["round4_fold_mean"]["BP"]))]


def chord_rows_refs(refs, chords, R) -> pd.DataFrame:
    rows = []
    for G in PW.G_CHORD:
        for name, (cv, label) in PW.CENTRES.items():
            for sd in PW.SD_WORLD:
                fixed = not isinstance(cv, str)
                X0 = None
                if fixed:
                    key = f"s0_power|chord|{name}|sd={sd}|G={G}"
                    X0 = CF.iid(np.random.default_rng(paths.seed_for(key)), R, G, "normal", sd) + cv
                for refname, bref in refs:
                    if fixed:
                        X, mu, k = X0, cv, key
                    else:
                        mu = bref if cv == "beta" else bref / 2
                        k = f"s0_power|chord|{name}|ref={refname}|sd={sd}|G={G}"
                        X = CF.iid(np.random.default_rng(paths.seed_for(k)), R, G, "normal", sd) + mu
                    meta = {"world": name, "world_kind": "normal", "world_note": label, "centre": mu, "sd": sd,
                            "seed_key": k}
                    PW.chord_block(rows, X, meta, refname, bref, mu, sd)
        for name, classes in PW.POOLS.items():
            v = chords[chords.cls.isin(classes)].chord.to_numpy(float)
            key = f"s0_power|chord|{name}|G={G}"
            X = CF.iid(np.random.default_rng(paths.seed_for(key)), R, G, "pool", pool=v)
            meta = {"world": name, "world_kind": "empirical (seen, label-based)",
                    "world_note": f"bootstrap of {len(v)} seen unit chords ({'+'.join(classes)})",
                    "centre": float(v.mean()), "sd": float(v.std(ddof=1)), "seed_key": key}
            for refname, bref in refs:
                PW.chord_block(rows, X, meta, refname, bref)
    return pd.DataFrame(rows)


def repro_vs_power_worlds(C: pd.DataFrame) -> dict:
    pw = pd.read_csv(OUT / "power_worlds.csv")
    pw = pw[(pw.claim != "C-ECT") & (pw.beta_ref_name == "beta_frozen")]
    mine = C[C.beta_ref_name == "beta_frozen"].copy()
    pw = pw.assign(sd_key=pw.sd.round(10))           # empirical-pool sd values round-trip through CSV text
    mine["sd_key"] = mine.sd.round(10)
    keys = ["world", "sd_key", "G", "statistic"]
    m = mine.merge(pw, on=keys, suffixes=("", "_pw"))
    cols = ["P_confirm", "P_kill", "P_helps", "P_harm", "P_A1"]
    d = max(float((m[c] - m[f"{c}_pw"]).abs().max()) for c in cols)
    return {"rows_compared": int(len(m)), "rows_expected": int(len(pw)), "max_abs_diff": d,
            "identical": bool(len(m) == len(pw) and d == 0.0)}


def pivot(df, index, val="P_confirm", G=(6, 8, 10, 13)):
    q = df[df.G.isin(G)]
    p = q.pivot_table(index=index, columns="G", values=val, aggfunc="first")
    return {(" | ".join(map(str, i)) if isinstance(i, tuple) else str(i)): {str(c): round(float(x), 4) for c, x in r.items()}
            for i, r in p.iterrows()}


# ------------------------------------------------------------------------------------------------ 2. supply reach
def p_reach(v: int, n_e2: int, n_u: int, p_e2: float, p_u: float, add: int, need: int) -> float:
    """P(v + add + Bin(n_e2, p_e2) + Bin(n_u, p_u) >= need), exact."""
    k = need - v - add
    if k <= 0:
        return 1.0
    pe = SS.binom.pmf(np.arange(n_e2 + 1), n_e2, p_e2)
    pu = SS.binom.pmf(np.arange(n_u + 1), n_u, p_u)
    tot = np.add.outer(np.arange(n_e2 + 1), np.arange(n_u + 1))
    return float((np.outer(pe, pu) * (tot >= k)).sum())


def ect_chem_units(lk: pd.DataFrame, labs: list) -> dict:
    """C-ECT chemotype units (fallback ii): r5.supply.ect_chem_units (moved there, s0-statistics-3)."""
    return SU.ect_chem_units(lk, labs)


def ect_chem_reach(lk: pd.DataFrame, verified: list, e2_only: list, rates: dict, need: int = 7) -> dict:
    """P(chemotype units >= need) for C-ECT fallback (ii): enumerate which E2-only laboratories convert (each with
    p_E2, independently; the G_ECT supply has no U-only laboratory), count units with and without linking."""
    import itertools
    out = {}
    for name, (pe, _) in rates.items():
        p_link = p_nolink = 0.0
        for k in range(len(e2_only) + 1):
            for sub in itertools.combinations(e2_only, k):
                w = pe ** k * (1 - pe) ** (len(e2_only) - k)
                u = ect_chem_units(lk, sorted(verified + list(sub)))
                p_link += w * (u["chemotype_units_linked"] >= need)
                p_nolink += w * (u["distinct_chemotypes"] >= need)
        out[name] = {"P_reach_7_units_linked": round(p_link, 4), "P_reach_7_distinct_chemotypes": round(p_nolink, 4)}
    return out


def main() -> int:
    from r5 import sealcheck
    sealcheck.refuse_if_sealed("scripts/s0_package_supplements.py (rewrites sealed results/stage0 files)",
                               "--amend" in sys.argv[1:])
    t0 = time.time()
    R = 10000
    refs = beta_refs_d2()
    chords = CF.load_unit_chords()
    C = chord_rows_refs(refs, chords, R)
    C.insert(0, "label", "STAGE-0 PLANNING (synthetic / seen-corpus worlds; D2 beta sensitivities)")
    repro = repro_vs_power_worlds(C)
    if not repro["identical"]:
        raise SystemExit(f"beta_frozen rows do not reproduce power_worlds.csv: {repro}")
    W = ["HDEHP_LIKE", "P507_LIKE", "HS_CORPUS", "NONDGA_CORPUS", "NO_DIP", "AT_BETA", "AT_BETA_HALF", "EMP_HS12",
         "EMP_NONDGA25"]
    summ = {}
    for sid in ("CGD_W", "CGD_R"):
        q = C[(C.statistic == sid) & C.world.isin(W)]
        for val in ("P_confirm", "P_kill", "P_helps", "P_harm"):
            summ[f"{sid}_{val}"] = pivot(q, ["world", "sd", "beta_ref_name"], val)
    q = C[(C.statistic == "GDT_R") & C.world.isin(W)]
    for val in ("P_confirm", "P_helps", "P_harm"):
        summ[f"GDT_R_{val}"] = pivot(q, ["world", "sd", "beta_ref_name"], val, G=(10, 13))
    code = [Path(__file__), Path(__file__).with_name("s0_power_worlds.py"), paths.R5 / "r5" / "confirm.py",
            paths.R5 / "r5" / "paths.py"]
    out1 = {"label": "STAGE-0 PLANNING: C-GD / Gd-dip chord power at the D2 beta references (no round-5 score; corpus "
                     "worlds are seen, label-based inputs)",
            "brief": "CLAUDE_CODE_PROMPT.md section 4.3 items 9-10, 7.5; DECISION D2",
            "reps_per_world": R, "beta_refs": dict(refs), "G_chord": PW.G_CHORD, "sd_world": PW.SD_WORLD,
            "provisional": "HDEHP_LIKE (-0.26) and P507_LIKE (-0.105) are UNVERIFIED planning values (PENDING-U5)",
            "reproduction_vs_power_worlds_csv": repro,
            "inputs_sha256_lf": CF.digests([OUT / "beta_frozen.json", CF.STRATA_CSV, OUT / "power_worlds.csv"]),
            "code_sha256_lf": CF.digests(code), "summary": summ}

    au = json.loads((OUT / "supply_audit.json").read_text(encoding="utf-8"))
    cs = au["claim_supply_lab_by_lab"]
    rates = {"P4_rates_pE2_0.5_pU_0.25": (0.5, 0.25), "PC_P4_rates_pE2_2of6_pU_0": (2 / 6, 0.0)}
    spec = {"C-ECT": ("G_ECT", 7, 0), "C-GD": ("G_HS", 6, 0), "Gd-dip transport (merged SAFE + literature)": ("G_Gd", 10, 3)}
    reach = {}
    for claim, (k, need, safe) in spec.items():
        v, e2, u = (len(cs[k][x]) for x in ("verified", "e1_e2_all_convert", "e1_e2_u_all_convert"))
        n_e2, n_u = e2 - v, u - e2
        r = {"supply_key": k, "min_G": need, "verified": v, "e2_only": n_e2, "u_only": n_u, "safe_added": safe}
        for name, (pe, pu) in rates.items():
            mean = v + safe + pe * n_e2 + pu * n_u
            fc = next(c for c in au["audit"]["claims"] if c["claim"] == claim)["forecasts"][name]["expected"]
            assert abs(round(mean, 2) - fc) < 1e-9, (claim, name, mean, fc)
            r[name] = {"expected": round(mean, 2), "P_reach_min": round(p_reach(v, n_e2, n_u, pe, pu, safe, need), 4)}
            if safe:
                r[name]["P_reach_min_SAFE_0"] = round(p_reach(v, n_e2, n_u, pe, pu, 0, need), 4)
        r["all_convert"] = v + n_e2 + n_u + safe
        reach[claim] = r
    reach["SAFE route (M33 gate)"] = {"min_G": 10, "max_possible": 3, "P_reach_min": 0.0,
                                      "note": "SAFE 0-3 expected (IR M33); 3 < 10 at any rate"}
    lk = pd.read_csv(OUT / "lab_key.csv", dtype=str).fillna("")
    allc = cs["G_ECT"]["e1_e2_u_all_convert"]
    ect_units = {"all_convert": ect_chem_units(lk, allc), "verified": ect_chem_units(lk, cs["G_ECT"]["verified"])}
    assert ect_units["all_convert"]["labs"] == sorted(allc), (ect_units["all_convert"]["labs"], allc)
    ver_ect = list(cs["G_ECT"]["verified"])
    e2_ect = [x for x in cs["G_ECT"]["e1_e2_all_convert"] if x not in ver_ect]
    assert sorted(cs["G_ECT"]["e1_e2_u_all_convert"]) == sorted(cs["G_ECT"]["e1_e2_all_convert"])  # no U-only lab
    ect_units["reach_chemotype_units"] = ect_chem_reach(lk, ver_ect, e2_ect, rates)
    out2 = {"label": "STAGE-0 SUPPLY FORECAST (metadata only; forecasts are reported, never counted: D4)",
            "brief": "CLAUDE_CODE_PROMPT.md section 4.3 item 1 ('each with the probability of reaching its minimum')",
            "model": "verified count 1; E2-only units convert with p_E2, U-only units with p_U, independently; mean = the "
                     "audit's expected-yield formula (asserted); Gd-dip adds SAFE at its maximum 3 (U2 unanswered)",
            "caveats": ["C-GD's '>= 3 extractant families' condition is not modelled: its probability is an upper bound.",
                        "Conversion rates are P4's assumed rates and PC P4's corrected ones, not measured rates.",
                        "C-ECT chemotype units use P4's hand chemotype labels (upper bound on the frozen gen6 map's "
                        "count, P4 notes)."],
            "reach": reach, "ect_fallback_chemotype_units": ect_units,
            "inputs_sha256_lf": CF.digests([OUT / "supply_audit.json", OUT / "lab_key.csv"]),
            "code_sha256_lf": CF.digests([Path(__file__), paths.R5 / "r5" / "supply.py"])}
    paths.write_text_lf(OUT / "power_cgd_beta_sens.csv", C.to_csv(index=False, lineterminator="\n"))
    paths.write_text_lf(OUT / "power_cgd_beta_sens.json", json.dumps(out1, indent=1, default=float) + "\n")
    paths.write_text_lf(OUT / "supply_reach.json", json.dumps(out2, indent=1, default=float) + "\n")
    print("repro vs power_worlds:", repro)
    print(json.dumps(reach, indent=1))
    print(json.dumps(ect_units, indent=1))
    print(f"wrote {OUT / 'power_cgd_beta_sens.csv'} ({len(C)} rows), power_cgd_beta_sens.json, supply_reach.json")
    print(f"{time.time() - t0:6.1f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
