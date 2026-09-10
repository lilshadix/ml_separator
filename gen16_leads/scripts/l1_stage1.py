"""L1 Stage 1 -- the bookkeeping half of the cycle-corrected xTB descriptor.

PRE_REGISTRATION.md section 3, L1.  Runs once, on the pre-specified models and sets, on the
``complex_total_energy_eV`` values already in the repository (no cluster job).  Order:

  A. reproduce gen15's ``exp/phys3d/trap_check.py`` numbers exactly against its stored
     ``trap_check.csv``, and diagnose the 39-vs-41 difference between the report's +0.644 and the
     script's +0.599;
  B. fit NAIVE / ELEM / SPECIES / SPECIES_CONST (+ the exploratory SPECIES_NFILLCOL) on identical
     rows, report gamma_nitrate / gamma_water and the identifiability of delta_ligand;
  C. per-extractant within-series slope, quadratic and rms of every model's residual in the
     standardised Shannon radius, with set membership S8 / S14 / S3;
  D. Spearman with a, |a| and b; LOCO over chemotypes; partial Spearman given n_metals;
     chemotype-blocked bootstrap 95 % CI (2000 reps); family-wise permutation null (2000 reps,
     max |rho| over the 4 registered models x 3 sets);
  E. the registered decision rule on SPECIES / S8.

The MAE arm (G14+PHYSCYC) is Phase 3 and is deliberately NOT run here.

Writes gen16_leads/results/L1/{stage1_reproduction.csv, stage1_bookkeeping.csv,
stage1_coefficients.csv, stage1_identifiability.csv, stage1_slopes.csv, stage1_stats.csv,
stage1_loco.csv, stage1_perm_null.csv, contrasts_stage1.csv, stage1_decision.json}.

Run from the repo root:
    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe gen16_leads/scripts/l1_stage1.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa: E402
from gen16 import l1_cycle as L  # noqa: E402

OUT = L.RESULTS
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(*a) -> None:
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ======================================================================================
# 0. bookkeeping: composition, fill species, charge convention
# ======================================================================================
def bookkeeping(rows: pd.DataFrame, count_cols: list[str], chg: pd.DataFrame) -> pd.DataFrame:
    n = len(rows)
    phantom = [c for c in count_cols if c[2:] not in L.KNOWN_ELEMENTS]
    real = [c for c in count_cols if c[2:] in L.KNOWN_ELEMENTS]
    fin = np.isfinite(rows["complex_total_energy_eV"].to_numpy(dtype=float))
    nitr = rows["fill_ligand"].to_numpy() == "nitrate"
    recs = [
        {"check": "complexes", "value": n},
        {"check": "distinct canonical_smiles", "value": int(rows["canonical_smiles"].nunique())},
        {"check": "distinct metals", "value": int(rows["metal_symbol"].nunique())},
        {"check": "complex_total_energy_eV non-null", "value": int(fin.sum())},
        {"check": "inner_sphere_anion == fill_ligand", "value": int((rows["inner_sphere_anion"] == rows["fill_ligand"]).all())},
        {"check": "decomposes into metal + n_ligs*L + n_NO3*NO3 + n_H2O*H2O", "value": int(rows["fill_ok"].all())},
        {"check": "n_fill == 2*n_NO3 + n_H2O", "value": int((rows["n_fill"] == 2 * rows["n_NO3"] + rows["n_H2O"]).all())},
        {"check": "nitrate series with an odd n_fill (one water present)", "value": int(((rows["n_fill"] % 2 == 1) & nitr).sum())},
        {"check": "water series with n_NO3 > 0", "value": int(((~nitr) & (rows["n_NO3"] > 0)).sum())},
        {"check": "ligands with a non-zero RDKit formal charge", "value": int(sum(
            L.ligand_formal_charge(s) != 0 for s in rows["canonical_smiles"].unique()))},
        {"check": "max n_NO3", "value": int(rows["n_NO3"].max())},
        {"check": "max n_H2O", "value": int(rows["n_H2O"].max())},
        {"check": "total charge = 3 - n_NO3, distinct values", "value": "/".join(
            str(v) for v in sorted(rows["total_charge"].unique()))},
        {"check": "series (smiles||anion) with >= 2 metals", "value": int(
            (rows.groupby("series").metal_symbol.nunique() >= 2).sum())},
        {"check": "series in which n_ligs varies", "value": len(L.varying_series(rows))},
        {"check": "series in which n_NO3 varies", "value": len(L.varying_series(rows, "n_NO3"))},
        {"check": "series in which n_H2O varies", "value": len(L.varying_series(rows, "n_H2O"))},
        {"check": "xyz files with a Properties header", "value": int(chg["has_properties_header"].sum())},
        {"check": "of those, initial_charges sum == 3 - n_NO3", "value": int(np.isclose(
            chg["q_initial_sum"], 3 - rows["n_NO3"], atol=1e-6).sum())},
        {"check": "files carrying a Mulliken charge column", "value": int(chg["q_mulliken_sum"].notna().sum())},
        {"check": "of those, Mulliken sum == 3 - n_NO3 (tol 0.02 e)", "value": int((
            (chg["q_mulliken_sum"] - (3 - rows["n_NO3"])).abs() < 0.02).sum())},
        {"check": "complexes with a non-zero initial_magmoms (Eu 6.0 / Yb 1.0 f-count metadata)",
         "value": int((chg["max_abs_magmom"].fillna(0) > 1e-9).sum())},
        {"check": "gen15 element-count columns that are real elements", "value": "/".join(c[2:] for c in real)},
        {"check": "gen15 element-count columns that are the extxyz comment line (phantom)",
         "value": "/".join(c[2:8] + "..." for c in phantom) or "none"},
    ]
    return pd.DataFrame(recs)


# ======================================================================================
# A. reproduce gen15
# ======================================================================================
def reproduce(rows_all: pd.DataFrame, count_cols: list[str]) -> pd.DataFrame:
    """trap_check.py's construction, line for line, on the imported frozen helpers."""
    g = rows_all.copy()
    E = g["complex_total_energy_eV"].to_numpy(dtype=float)
    ok = np.isfinite(E)
    g.loc[ok, "resid_naive"] = L._two_way_residual(E[ok], g["series"].to_numpy()[ok],
                                                   g["metal_symbol"].to_numpy()[ok])
    mA = g["is_main_block"].to_numpy() & ok
    g.loc[mA, "resid_A"] = L._two_way_residual(E[mA], g["block"].to_numpy()[mA],
                                               g["metal_symbol"].to_numpy()[mA])
    C = g.loc[ok, count_cols].to_numpy(dtype=float)
    C = C - pd.DataFrame(C).groupby(g["series"].to_numpy()[ok]).transform("mean").to_numpy()
    keep = C.std(axis=0) > 1e-9
    g.loc[ok, "resid_B"] = L._two_way_residual(E[ok], g["series"].to_numpy()[ok],
                                               g["metal_symbol"].to_numpy()[ok], cov=C[:, keep])

    recs = []
    for smi, sub in g.groupby("canonical_smiles"):
        cand = sub.groupby("series").metal_symbol.nunique().sort_values(ascending=False)
        ser = cand.index[0]
        s = sub[sub["series"] == ser]
        blk = s[s["is_main_block"]]
        rec = {"extractant": smi, "n_series": s["metal_symbol"].nunique(),
               "n_block": blk["metal_symbol"].nunique(),
               "const_comp": int(s["composition"].nunique() == 1),
               "all_E_series": int(s["complex_total_energy_eV"].notna().all()),
               "all_E_block": int(blk["complex_total_energy_eV"].notna().all()),
               "all_E_any": int(sub["complex_total_energy_eV"].notna().all())}
        for name, frame, col in (("naive", s, "resid_naive"), ("A", blk, "resid_A"), ("B", s, "resid_B")):
            sl, _, _, nfit = L._slope_fit(frame["r"].to_numpy(float), frame[col].to_numpy(float), True)
            rec[name] = sl
            rec[f"n_fit_{name}"] = nfit
        recs.append(rec)
    sl = pd.DataFrame(recs)

    tgt = L.targets().rename(columns={"extractant": "extractant"}).merge(sl, on="extractant", how="inner")
    subsets = {
        "all cohort extractants": np.ones(len(tgt), dtype=bool),
        "complete 14-metal series": tgt["n_series"].to_numpy() == 14,
        "constant-composition series only": tgt["const_comp"].to_numpy() == 1,
        "block >= 7 metals": tgt["n_block"].to_numpy() >= 7,
    }
    out = []
    for label, m in subsets.items():
        t = tgt[m]
        for cons in ("naive", "A", "B"):
            for target in ("a", "abs_a"):
                rho, p, n = L._rho(t[cons].to_numpy(float), t[target].to_numpy(float))
                pr = L._partial_rho(t[cons].to_numpy(float), t[target].to_numpy(float),
                                    t["n_metals"].to_numpy(float))
                out.append({"subset": label, "construction": cons, "target": target,
                            "n": n, "rho": rho, "p": p, "rho_given_n_metals": pr})
    got = pd.DataFrame(out)

    ref = pd.read_csv(L.PHYS3D / "trap_check.csv")
    cmp_ = got.merge(ref, on=["subset", "construction", "target"], how="outer",
                     suffixes=("_gen16", "_gen15"))
    for c in ("n", "rho", "p", "rho_given_n_metals"):
        cmp_[f"d_{c}"] = (cmp_[f"{c}_gen16"] - cmp_[f"{c}_gen15"]).abs()
    cmp_["reproduces"] = (cmp_[["d_n", "d_rho", "d_p"]].max(axis=1) < 1e-9)

    # ---- the 39-vs-41 diagnosis: n_series counts metals in the geometry table, the energy fit
    # only uses metals with a finite complex_total_energy_eV.
    extra = []
    for label, mask in (("complete 14 GEOMETRY metals (trap_check n_series == 14)",
                         tgt["n_series"].to_numpy() == 14),
                        ("complete 14 ENERGY metals (n rows entering the naive fit == 14)",
                         tgt["n_fit_naive"].to_numpy() == 14),
                        ("all cohort extractants, every metal of the chosen series energised",
                         tgt["all_E_series"].to_numpy() == 1),
                        ("all cohort extractants, every metal of the main block energised",
                         tgt["all_E_block"].to_numpy() == 1),
                        ("all cohort extractants, every geometry of the extractant energised",
                         tgt["all_E_any"].to_numpy() == 1)):
        t = tgt[mask]
        for cons in ("naive", "A", "B"):
            for target in ("a", "abs_a"):
                rho, p, n = L._rho(t[cons].to_numpy(float), t[target].to_numpy(float))
                pr = L._partial_rho(t[cons].to_numpy(float), t[target].to_numpy(float),
                                    t["n_metals"].to_numpy(float))
                extra.append({"subset": label, "construction": cons, "target": target,
                              "n": n, "rho": rho, "p": p, "rho_given_n_metals": pr,
                              "reproduces": np.nan})
    extra = pd.DataFrame(extra).rename(columns={"n": "n_gen16", "rho": "rho_gen16", "p": "p_gen16",
                                                "rho_given_n_metals": "rho_given_n_metals_gen16"})
    out_df = pd.concat([cmp_.assign(role="reproduction"), extra.assign(role="39_vs_41_diagnosis")],
                       ignore_index=True)
    return out_df, tgt


# ======================================================================================
def main() -> None:
    log("loading the 1155 accepted geometries and their element counts ...")
    rows_all, count_cols = L.load_energy_rows()
    log(f"  rows={len(rows_all)}  ligands={rows_all.canonical_smiles.nunique()}  "
        f"metals={rows_all.metal_symbol.nunique()}  energies={rows_all.complex_total_energy_eV.notna().sum()}")

    chg = L.xyz_charge_audit(rows_all)
    chg.to_csv(OUT / "stage1_charge_audit.csv", index=False)
    book = bookkeeping(rows_all, count_cols, chg)
    book.to_csv(OUT / "stage1_bookkeeping.csv", index=False)
    print(book.to_string(index=False))

    # ---- the copy of _two_way_residual must reproduce the frozen one ---------------------
    fin = rows_all[np.isfinite(rows_all["complex_total_energy_eV"].to_numpy(dtype=float))].reset_index(drop=True)
    dmax = L.check_two_way_copy(fin, count_cols)
    log(f"two_way_fit vs frozen _two_way_residual: max |diff| = {dmax:.3e}")
    assert dmax < 1e-9, "the two-way copy does not reproduce the frozen residual"

    # ---- A. reproduction -----------------------------------------------------------------
    log("A. reproducing gen15 exp/phys3d/trap_check.py ...")
    rep, tgt_rep = reproduce(rows_all, count_cols)
    rep.to_csv(OUT / "stage1_reproduction.csv", index=False)
    r = rep[rep.role == "reproduction"]
    log(f"  rows compared: {len(r)}   all reproduce: {bool(r['reproduces'].all())}   "
        f"max |d rho| = {r['d_rho'].max():.2e}")
    show = r[(r.target == "a")][["subset", "construction", "n_gen16", "rho_gen16", "rho_gen15", "reproduces"]]
    print(show.to_string(index=False))
    print(rep[rep.role == "39_vs_41_diagnosis"][
        ["subset", "construction", "target", "n_gen16", "rho_gen16", "rho_given_n_metals_gen16"]]
        .to_string(index=False))

    # ---- B. models -----------------------------------------------------------------------
    log("B. fitting NAIVE / ELEM / SPECIES / SPECIES_CONST (+ SPECIES_NFILLCOL) ...")
    models_all = L.MODELS + ("SPECIES_NFILLCOL",)
    rows, coefs = L.fit_models(rows_all, count_cols, models=models_all)
    log(f"  identical rows for NAIVE/ELEM/SPECIES: {int(np.isfinite(rows['resid_NAIVE']).sum())}  "
        f"SPECIES_CONST rows: {int(np.isfinite(rows['resid_SPECIES_CONST']).sum())}")

    ctab = []
    for m, c in coefs.items():
        c = c.copy()
        c["model"] = m
        ctab.append(c)
    ctab = pd.concat(ctab, ignore_index=True)
    ctab["is_gamma"] = ctab["term"].str.startswith("gamma::")
    ctab["is_delta"] = ctab["term"].str.startswith("delta::")
    ctab.to_csv(OUT / "stage1_coefficients.csv", index=False)
    gam = ctab[ctab.is_gamma][["model", "term", "coef", "se"]]
    gam = gam.assign(coef_Ha=gam["coef"] / L.HARTREE_EV)
    print("\nfill-species coefficients (eV per molecule of the species):")
    print(gam.to_string(index=False))

    # a physical yardstick from the corpus itself: per-element energies from a global count fit
    els = [c[2:] for c in count_cols]
    Xg = np.c_[np.ones(len(rows)), rows[count_cols].to_numpy(dtype=float)]
    yg = rows["complex_total_energy_eV"].to_numpy(dtype=float)
    bg, *_ = np.linalg.lstsq(Xg, yg, rcond=None)
    per_el = dict(zip(els, bg[1:]))
    yard = {"nitrate NO3 (N + 3 O)": per_el.get("N", np.nan) + 3 * per_el.get("O", np.nan),
            "water H2O (O + 2 H)": per_el.get("O", np.nan) + 2 * per_el.get("H", np.nan)}
    print("\nper-element energies from a global count regression on the same 1116 rows (eV):")
    print({k: round(v, 2) for k, v in per_el.items()})
    print("implied free-species yardstick (eV): " + ", ".join(f"{k} = {v:.1f}" for k, v in yard.items()))

    ident = L.identifiability(rows)
    ident.to_csv(OUT / "stage1_identifiability.csv", index=False)
    log(f"  series in which n_ligs varies: {len(ident)}")
    if len(ident):
        print(ident.round(3).to_string(index=False))

    # ---- C. per-extractant slopes ---------------------------------------------------------
    log("C. per-extractant within-series slopes ...")
    sl = L.slope_table(rows, models_all, g_all=rows_all)
    sl = L.attach_sets(sl)
    sl.to_csv(OUT / "stage1_slopes.csv", index=False)
    counts = sl[sl.model == "SPECIES"][["S8", "S14", "S3"]].sum().to_dict()
    log(f"  extractants x models: {len(sl)}   SPECIES set sizes {counts}")
    per_model = sl.groupby("model")[["S8", "S14", "S3"]].apply(
        lambda d: pd.Series({s: int(d[s].sum()) for s in ("S8", "S14", "S3")}))
    print(per_model.to_string())
    nvar = sl[(sl.model == "SPECIES") & sl.S8]["n_ligs_varies"].sum()
    log(f"  S8 extractants whose chosen series has a varying n_ligs: {int(nvar)} of {int(counts['S8'])}")

    # ---- D. statistics --------------------------------------------------------------------
    log("D. statistics (bootstrap 2000, permutation 2000) ...")
    st, loco, null, W = L.full_stats(sl, models=models_all, value="slope", family_models=L.MODELS)
    st.to_csv(OUT / "stage1_stats.csv", index=False)
    loco.to_csv(OUT / "stage1_loco.csv", index=False)
    null.to_csv(OUT / "stage1_perm_null.csv", index=False)
    W.to_csv(OUT / "stage1_wide.csv", index=False)
    print("\n=== Spearman of the model slope with a (signed amplitude) ===")
    show = st[st.target == "a"][["model", "set", "n", "rho", "p", "loco_min", "loco_max",
                                 "loco_sign_stable", "partial_rho_n_metals", "ci95_low", "ci95_high",
                                 "ci_excludes_zero"]]
    print(show.round(4).to_string(index=False))
    print("\n=== the other two targets ===")
    print(st[st.target != "a"][["model", "set", "target", "n", "rho", "p", "ci95_low", "ci95_high"]]
          .round(4).to_string(index=False))
    print("\n=== family-wise permutation null (4 registered models x 3 sets, chemotype means) ===")
    print(null.round(4).to_string(index=False))

    # quadratic term as a secondary descriptor (exploratory, written out)
    stq, locoq, nullq, _ = L.full_stats(sl, models=models_all, value="quad", family_models=L.MODELS)
    stq.to_csv(OUT / "stage1_stats_quad.csv", index=False)
    nullq.to_csv(OUT / "stage1_perm_null_quad.csv", index=False)

    # ---- contrasts file --------------------------------------------------------------------
    reg_rows = st[st.model.isin(L.MODELS)]
    exp_rows = st[~st.model.isin(L.MODELS)]
    con = L.contrasts_frame(reg_rows, extra_exploratory=pd.concat([exp_rows, stq], ignore_index=True))
    con.to_csv(OUT / "contrasts_stage1.csv", index=False)
    log(f"  contrasts written: {len(con)}  registered={(con.family == 'registered').sum()}  "
        f"exploratory={(con.family == 'exploratory').sum()}")

    # ---- E. the registered decision rule ----------------------------------------------------
    dec = L.decision(st, "SPECIES", "S8")
    dec_sec = L.decision(st, "SPECIES_CONST", "S8")
    perm_a = null[null.target == "a"].iloc[0].to_dict()
    payload = {"registered": dec, "secondary_SPECIES_CONST": dec_sec,
               "permutation_bar_a": perm_a,
               "sets": {s: int(sl[(sl.model == "SPECIES")][s].sum()) for s in ("S8", "S14", "S3")},
               "n_S8_series_with_varying_n_ligs": int(nvar),
               "two_way_copy_max_abs_diff": dmax,
               "reproduction_all_match": bool(rep[rep.role == "reproduction"]["reproduces"].all()),
               "gamma": {r["term"]: {"model": r["model"], "coef_eV": r["coef"], "se_eV": r["se"]}
                         for _, r in gam.iterrows() if r["model"] == "SPECIES"},
               "yardstick_eV": yard}
    (OUT / "stage1_decision.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print("\n=== REGISTERED DECISION (SPECIES, S8, target a) ===")
    for k, v in dec.items():
        if k != "rule":
            print(f"  {k:32s} {v}")
    print(f"\n  VERDICT: {dec['verdict'].upper()}")
    log("done")


if __name__ == "__main__":
    main()
