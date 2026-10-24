#!/usr/bin/env python
"""Stage-0 power in worlds for the confirmation claims (CLAUDE_CODE_PROMPT.md section 4.2 "Power is stated in worlds",
section 4.3 items 1, 5, 9, 10, section 7.5).  Design fixed in this docstring before it was run.  Planning only: no
round-5 model is scored; every corpus-based world uses seen, label-based values (allowed as Stage-0 planning input and
disclosed in the JSON), and assumes new laboratories are exchangeable with corpus units, which the supply is not.

C-ECT (laboratory units, one family per laboratory; G = 6..10; tau in {0, 0.065}, P2's per-source noise):
  CORPUS_DGA          resample the 14 P1 PRIMARY family log ratios (L_SH4 with its paired L_SH3; 13 of 14 DGA)
  HALF                every value minus half its arm's corpus mean (effect halved; confirm4_skepB S1)
  NULL_SWAP           corpus values with random signs (sign-symmetrised null; size and P(kill | null))
  NULLLIKE_W1_CATEX   SH@4 values shifted to the non-DGA no-C5 unit mean (+0.140; cation-exchanger-like)
  NULLLIKE_W2_NONDGA  resample the 5 non-DGA no-C5 units (any n_sib)
  MIX25 / MIX50       each family is null-like (random sign) with probability 0.25 / 0.50
  Decision (section 7.5): confirm if the exact one-sided sign-flip p <= 0.025 (G >= 7 decidable); kill if the mean log
  ratio <= 0 or the win share < 0.5 (when not confirmed); else INCONCLUSIVE; SIGN on wins; fixed sequence SH@4 -> SH@3
  (SH@3 only where the world defines it).
Gd chord claims (G = 6..13; laboratory values i.i.d.):
  statistics CGD_W (winsorised [-0.40, +0.20], C-GD primary), CGD_R (raw, sensitivity), GDT_R (Gd-dip transport,
  raw; decides at G >= 10); rules confirm (UB < beta), kill (LB > beta), helps (UB < beta/2), harm (LB > beta/2).
  references beta_frozen (results/stage0/beta_frozen.json, else the round-4 fold mean -0.094 marked TODO), its
  sensitivities -0.118, -0.072, and the round-4 fold mean -0.094 (reference only) when beta_frozen differs from it.
  normal worlds, between-laboratory sd in {0.10 (M33's assumption), 0.12, 0.14, 0.16 (confirm-1's measured range)}:
    HDEHP_LIKE -0.26 and P507_LIKE -0.105 (both UNVERIFIED planning values, PROVISIONAL / PENDING-U5), HS_CORPUS
    -0.217 and NONDGA_CORPUS -0.142 (seen corpus context), AT_BETA (= reference), AT_BETA_HALF, NO_DIP (0).
  empirical worlds: bootstrap of seen pub x chemotype unit chords (strata_units.csv): heavy-selective non-DGA (12),
    non-DGA (25), DGA (19; descriptive only), all (44).
  Raw statistics in normal worlds also carry the analytic non-central t value of each rule.
Replicates: >= 50 per world (default 10,000); one RNG per world cell, seed paths.seed_for(<key>).  Monte Carlo
uncertainty (brief 8): every reported rate (confirm, kill, helps, harm, A1; C-ECT confirm, kill, SIGN reject, SH@3 in
sequence) carries its two-sided 95 % Clopper-Pearson bounds (P_<rule>_cp_lo / _cp_hi); the Monte Carlo SE is at most
sqrt(0.25 / 10,000) = 0.005.

    D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s0_power_worlds.py
writes results/stage0/power_worlds.csv, results/stage0/power_worlds.json
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
from scipy import stats as SS  # noqa: E402

from r5 import confirm as CF  # noqa: E402

OUT = paths.RES / "stage0"
G_ECT = (6, 7, 8, 9, 10)
TAUS = (0.0, 0.065)
ECT_WORLDS = ("CORPUS_DGA", "HALF", "NULL_SWAP", "NULLLIKE_W1_CATEX", "NULLLIKE_W2_NONDGA", "MIX25", "MIX50")
G_CHORD = tuple(range(6, 14))
SD_WORLD = (0.10, 0.12, 0.14, 0.16)
CENTRES = {"HDEHP_LIKE": (CF.HDEHP_LIKE, "PROVISIONAL: UNVERIFIED planning value (I2 confirm-1), PENDING-U5"),
           "P507_LIKE": (CF.P507_LIKE, "PROVISIONAL: UNVERIFIED search-snippet reading (I2 confirm-2), PENDING-U5"),
           "HS_CORPUS": (-0.217, "seen corpus context: slope heavy-selective non-DGA winsorised mean; N-donor dominated"),
           "NONDGA_CORPUS": (-0.142, "seen corpus context: non-DGA pub x chemotype unit mean"),
           "AT_BETA": ("beta", "calibrated world: true mean = reference (null boundary of confirm / kill)"),
           "AT_BETA_HALF": ("beta_half", "true mean = reference / 2 (null boundary of helps / harm)"),
           "NO_DIP": (0.0, "no Gd dip: B1's term harms")}
POOLS = {"EMP_HS12": ("nonDGA_HS",), "EMP_NONDGA25": ("nonDGA_HS", "nonDGA_LS"), "EMP_DGA19": ("DGA",),
         "EMP_ALL44": ("DGA", "nonDGA_HS", "nonDGA_LS")}


def ect_draws(rng, R, G, world, L4, L3, nd):
    idx = rng.integers(len(L4), size=(R, G))
    A4, A3 = L4[idx], L3[idx]
    if world == "HALF":
        A4, A3 = A4 - L4.mean() / 2, A3 - L3.mean() / 2
    elif world == "NULL_SWAP":
        s = rng.choice((-1.0, 1.0), size=(R, G))
        A4, A3 = A4 * s, A3 * s
    elif world == "NULLLIKE_W1_CATEX":
        A4, A3 = A4 - (L4.mean() - nd.mean()), None
    elif world == "NULLLIKE_W2_NONDGA":
        A4, A3 = nd[rng.integers(len(nd), size=(R, G))], None
    elif world in ("MIX25", "MIX50"):
        nul = rng.random((R, G)) < (0.25 if world == "MIX25" else 0.50)
        s = np.where(nul, rng.choice((-1.0, 1.0), size=(R, G)), 1.0)
        A4, A3 = A4 * s, A3 * s
    return A4, A3


def ect_rows(L4, L3, nd, R):
    rows = []
    for world in ECT_WORLDS:
        for tau in TAUS:
            for G in G_ECT:
                key = f"s0_power|ect|{world}|tau={tau}|G={G}"
                rng = np.random.default_rng(paths.seed_for(key))
                A4, A3 = ect_draws(rng, R, G, world, L4, L3, nd)
                A4 = CF.tau_noise(A4, tau, rng)
                A3 = None if A3 is None else CF.tau_noise(A3, tau, rng)
                d = CF.ect_decisions(A4, A3)
                kc, kk = int(d["confirm"].sum()), int(d["kill"].sum())
                (cl, ch), (kl, kh) = CF.clopper_pearson(kc, R), CF.clopper_pearson(kk, R)
                ks = int(d["sign_reject"].sum())
                sl, sh = CF.clopper_pearson(ks, R)
                kq = int(d["seq_SH3"].sum()) if "seq_SH3" in d else None
                ql, qh = CF.clopper_pearson(kq, R) if kq is not None else (np.nan, np.nan)
                rows.append({"claim": "C-ECT", "world": world, "tau": tau, "G": G, "reps": R,
                             "decidable": G >= CF.G_FLOOR["C-ECT"], "mean_lab_value": float(A4.mean()),
                             "P_confirm": kc / R, "P_confirm_cp_lo": cl, "P_confirm_cp_hi": ch,
                             "P_kill": kk / R, "P_kill_cp_lo": kl, "P_kill_cp_hi": kh,
                             "P_inconclusive": float(d["inconclusive"].mean()), "P_overlap": float(d["overlap"].mean()),
                             "P_sign_reject": float(d["sign_reject"].mean()), "P_sign_reject_cp_lo": sl,
                             "P_sign_reject_cp_hi": sh,
                             "P_seq_SH3": float(d["seq_SH3"].mean()) if "seq_SH3" in d else np.nan,
                             "P_seq_SH3_cp_lo": ql, "P_seq_SH3_cp_hi": qh,
                             "min_p": 1 / 2 ** G, "seed_key": key})
    return pd.DataFrame(rows)


def analytic(mu, sd, G, ref, tail):
    """Non-central t value of a raw-interval rule in a normal world (one-sided 0.025)."""
    df, tc = G - 1, SS.t.ppf(0.975, G - 1)
    nc = (mu - ref) * np.sqrt(G) / sd
    return float(SS.nct.cdf(-tc, df, nc)) if tail == "lower" else float(SS.nct.sf(tc, df, nc))


def chord_block(rows, X, meta, refname, bref, mu=None, sd=None):
    R, G = X.shape
    refs = CF.chord_refs(bref)
    for sid, (claim, clip, role) in CF.CHORD_STATS.items():
        lo, hi = CF.t_interval(X, clip=clip)
        h = CF.chord_rule_hits(lo, hi, refs)
        r = {**meta, "claim": claim, "statistic": sid, "role": role, "beta_ref_name": refname, "beta_ref": bref,
             "G": G, "reps": R, "decidable": G >= CF.G_FLOOR[claim]}
        for rule in ("confirm", "kill", "helps", "harm", "A1"):
            k = int(h[rule].sum())
            r[f"P_{rule}"] = k / R
            r[f"P_{rule}_cp_lo"], r[f"P_{rule}_cp_hi"] = CF.clopper_pearson(k, R)      # every reported rate (s0-statistics-11)
        r["P_inconclusive_beta"] = 1 - r["P_confirm"] - r["P_kill"]
        r["P_statusquo_half"] = 1 - r["P_helps"] - r["P_harm"]
        for rule, ref, tail in (("confirm", bref, "lower"), ("kill", bref, "upper"), ("helps", bref / 2, "lower"),
                                ("harm", bref / 2, "upper")):
            r[f"analytic_{rule}"] = analytic(mu, sd, G, ref, tail) if (mu is not None and clip is None) else np.nan
        rows.append(r)


def beta_refs(B):
    """beta_frozen, the confirm-1 sensitivities, and the round-4 fold mean (reference only) when it differs."""
    refs = [("beta_frozen", B)] + [(f"beta_sens_{b}", b) for b in CF.BETA_SENS]
    return refs + ([("beta_r4_fold_mean", CF.BETA_R4_FOLD_MEAN)] if B != CF.BETA_R4_FOLD_MEAN else [])


def chord_rows(B, chords, R):
    refs = beta_refs(B)
    rows = []
    for G in G_CHORD:
        for name, (cv, label) in CENTRES.items():
            for sd in SD_WORLD:
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
                    chord_block(rows, X, meta, refname, bref, mu, sd)
        for name, classes in POOLS.items():
            v = chords[chords.cls.isin(classes)].chord.to_numpy(float)
            key = f"s0_power|chord|{name}|G={G}"
            X = CF.iid(np.random.default_rng(paths.seed_for(key)), R, G, "pool", pool=v)
            meta = {"world": name, "world_kind": "empirical (seen, label-based)",
                    "world_note": f"bootstrap of {len(v)} seen unit chords ({'+'.join(classes)})",
                    "centre": float(v.mean()), "sd": float(v.std(ddof=1)), "seed_key": key}
            for refname, bref in refs:
                chord_block(rows, X, meta, refname, bref)
    return pd.DataFrame(rows)


def pivot(df, index, col="G", val="P_confirm"):
    p = df.pivot_table(index=index, columns=col, values=val, aggfunc="first")
    return {(" | ".join(map(str, i)) if isinstance(i, tuple) else str(i)): {str(c): round(float(x), 4) for c, x in r.items()}
            for i, r in p.iterrows()}


def main() -> int:
    from r5 import sealcheck
    sealcheck.refuse_if_sealed("scripts/s0_power_worlds.py (rewrites sealed Stage-0 inputs)", "--amend" in sys.argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--amend", action="store_true", help="rerun after the seal (an AMENDMENTS.md entry and a reseal follow)")
    ap.add_argument("--reps", type=int, default=10000)
    a = ap.parse_args()
    R = a.reps
    if R < 50:
        raise SystemExit("section 4.2 / 4.5 item 2: >= 50 replicates per world")
    t0 = time.time()
    beta = CF.load_beta_frozen()
    B = beta["value"]
    L4, L3, nd = CF.load_ect_units()
    chords = CF.load_unit_chords()
    E = ect_rows(L4, L3, nd, R)
    print(f"{time.time() - t0:6.1f}s C-ECT worlds done", flush=True)
    C = chord_rows(B, chords, R)
    print(f"{time.time() - t0:6.1f}s chord worlds done", flush=True)
    E.insert(0, "label", "STAGE-0 PLANNING (synthetic / seen-corpus worlds)")
    C.insert(0, "label", "STAGE-0 PLANNING (synthetic / seen-corpus worlds)")
    df = pd.concat([E, C], ignore_index=True, sort=False)

    bf = C[C.beta_ref_name == "beta_frozen"]
    keyG = (6, 8, 10, 13)
    summ = {
        "C-ECT_P_confirm": pivot(E, ["world", "tau"]),
        "C-ECT_P_kill": pivot(E, ["world", "tau"], val="P_kill"),
        "C-ECT_P_seq_SH3": pivot(E[E.P_seq_SH3.notna()], ["world", "tau"], val="P_seq_SH3"),
        "C-ECT_P_sign_reject": pivot(E, ["world", "tau"], val="P_sign_reject"),
        "CGD_W_P_confirm_at_beta_frozen": pivot(bf[(bf.statistic == "CGD_W") & bf.G.isin(keyG)], ["world", "sd"]),
        "CGD_R_P_confirm_at_beta_frozen": pivot(bf[(bf.statistic == "CGD_R") & bf.G.isin(keyG)], ["world", "sd"]),
        "CGD_W_P_confirm_beta_sensitivity": pivot(
            C[(C.statistic == "CGD_W") & C.world.isin(["HDEHP_LIKE", "P507_LIKE", "HS_CORPUS", "EMP_HS12"])
              & C.G.isin(keyG)], ["world", "sd", "beta_ref_name"]),
        "CGD_W_P_kill_at_beta_frozen": pivot(bf[(bf.statistic == "CGD_W") & bf.G.isin(keyG)], ["world", "sd"],
                                             val="P_kill"),
        "CGD_W_P_harm_at_beta_frozen": pivot(bf[(bf.statistic == "CGD_W") & bf.G.isin(keyG)], ["world", "sd"],
                                             val="P_harm"),
        "GDT_R_P_confirm_at_beta_frozen": pivot(bf[(bf.statistic == "GDT_R") & bf.G.isin((10, 13))], ["world", "sd"]),
        "GDT_R_P_helps_at_beta_frozen": pivot(bf[(bf.statistic == "GDT_R") & bf.G.isin((10, 13))], ["world", "sd"],
                                              val="P_helps"),
        "GDT_R_P_harm_at_beta_frozen": pivot(bf[(bf.statistic == "GDT_R") & bf.G.isin((10, 13))], ["world", "sd"],
                                             val="P_harm")}
    raw_n = C[(C.world_kind == "normal") & (C.statistic == "CGD_R")]
    mc_vs_analytic = float((raw_n.P_confirm - raw_n.analytic_confirm).abs().max())
    pv = [SS.binomtest(int(round(p * n)), int(n), float(q)).pvalue for p, n, q in
          zip(raw_n.P_confirm, raw_n.reps, raw_n.analytic_confirm) if 0 < q < 1]
    mc_bonf = float(min(1.0, min(pv) * len(pv)))
    mc_vs_analytic_txt = (f"max |diff| {mc_vs_analytic:.4f}; Bonferroni-adjusted minimum exact binomial p of the Monte "
                          f"Carlo count given the analytic rate {mc_bonf:.3f} over {len(pv)} cells")
    code = [Path(__file__), paths.R5 / "r5" / "confirm.py", paths.R5 / "r5" / "paths.py"]
    out = {
        "label": "STAGE-0 PLANNING: power in worlds (no round-5 score; corpus worlds are seen, label-based inputs)",
        "brief": "CLAUDE_CODE_PROMPT.md section 4.2, 4.3 items 1, 5, 9, 10, section 7.5",
        "reps_per_world": R, "alpha_one_sided": CF.ALPHA, "G_ect": G_ECT, "G_chord": G_CHORD, "taus": TAUS,
        "sd_world": SD_WORLD, "G_floor": CF.G_FLOOR, "clip": CF.CLIP, "beta": beta,
        "beta_refs": dict(beta_refs(B)),
        "centres": {k: {"value": v[0], "note": v[1]} for k, v in CENTRES.items()},
        "provisional": "HDEHP_LIKE (-0.26) and P507_LIKE (-0.105) are UNVERIFIED planning values; PROVISIONAL, to be "
                       "fixed from primary SF tables by append-only amendment before any new value is read (PENDING-U5)",
        "ect_inputs": {"L_SH4_mean": float(L4.mean()), "L_SH3_mean": float(L3.mean()), "L_SH4_wins": int((L4 > 0).sum()),
                       "units": int(len(L4)), "nonDGA_noC5_units": int(len(nd)), "nonDGA_noC5_mean": float(nd.mean())},
        "seed_scheme": "numpy default_rng(paths.seed_for(seed_key)), seed_for(k) = 20261023 + crc32(k); seed_key per row",
        "inputs_sha256_lf": CF.digests([CF.STRATA_CSV, CF.ECT_UNITS_CSV, CF.ECT_NONDGA_CSV]),
        "code_sha256_lf": CF.digests(code),
        "disclosure": "Corpus-based worlds (P1 PRIMARY family log ratios; pub x chemotype unit chords; the -0.217 and "
                      "-0.142 centres) use seen, label-based values as Stage-0 planning distributions only. They are "
                      "upper-side for genuinely new sources (B1's and SH's forms were chosen on these cells) and assume "
                      "exchangeability, which the supply violates (P4: 15 of 16 new laboratories on absent chemotypes).",
        "notes": ["C-ECT: one family per laboratory; the sign-flip is scale-free, so a halved effect is a shift, not a "
                  "rescaling; SH@3 is undefined in the W1 / W2 worlds (NaN).",
                  "C-GD needs >= 3 extractant families among its >= 6 laboratories; that constraint is not simulated.",
                  "Chord worlds are i.i.d. laboratories; chemotype clustering enters through the size gate only.",
                  f"Monte Carlo vs analytic non-central t, CGD_R confirm rule in normal worlds: {mc_vs_analytic_txt}.",
                  "CGD_R and GDT_R are the same computation on the same draws; they differ in the G floor (6 vs 10) "
                  "and in the claim population (cation-exchanger stratum vs all non-DGA laboratories)."],
        "monte_carlo_uncertainty": "10,000 replicates per world; Monte Carlo SE <= 0.005; two-sided 95 % Clopper-Pearson "
                                   "bounds per row in power_worlds.csv (P_<rule>_cp_lo / _cp_hi)",
        "summary": summ}
    OUT.mkdir(parents=True, exist_ok=True)
    paths.write_text_lf(OUT / "power_worlds.csv", df.to_csv(index=False, lineterminator="\n"))
    paths.write_text_lf(OUT / "power_worlds.json", json.dumps(out, indent=1, default=float) + "\n")
    pd.set_option("display.width", 220)
    print("C-ECT P(confirm) by G:")
    print(E.pivot_table(index=["world", "tau"], columns="G", values="P_confirm").round(3).to_string())
    print("C-ECT P(kill) by G:")
    print(E.pivot_table(index=["world", "tau"], columns="G", values="P_kill").round(3).to_string())
    for sid in CF.CHORD_STATS:
        q = bf[(bf.statistic == sid) & bf.G.isin(keyG)]
        print(f"{sid} P(confirm: UB < beta_frozen = {B}) by G:")
        print(q.pivot_table(index=["world", "sd"], columns="G", values="P_confirm").round(3).to_string())
    print(f"MC vs analytic (CGD_R confirm, normal worlds): {mc_vs_analytic_txt}")
    if beta["todo"]:
        print(beta["todo"])
    print(f"wrote {OUT / 'power_worlds.csv'} ({len(df)} rows)")
    print(f"wrote {OUT / 'power_worlds.json'}")
    print(f"{time.time() - t0:6.1f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
