"""L4 prospective ranking (PRE_REGISTRATION.md section 3, L4 -- "prospective deliverable, not scored").

Two candidate pools are ranked by the same A-optimal criterion the retrospective simulation uses,
with the FULL frozen cohort as the already-chosen design:

* pool (i)  -- bundle extractants (``dataset with 3D structures/dataset.parquet``) with exactly ONE
               measured lanthanide.  These are the 100 compounds the gen13 cohort's ">= 2 metals per
               cell" guardrail keeps out; L6's audit showed they carry 53 chemotypes absent from the
               cohort, so one extra lanthanide on one compound per absent chemotype is the cheapest
               route from Kish n_eff 11.7 to 27.4.
* pool (ii) -- the 273 external aqueous-logK ligands of ``gen15_curve/exp/external/data/
               side_K_logk.parquet``, whose ``coord__*`` columns already carry the TOPO39 names.

Criterion.  Standardise TOPO39 with the cohort's own median-imputed statistics; the chosen design is
the cohort's well-determined cells, ``P = I*1 + sum_chosen z z^T``.  A candidate contributes one row
``z``; by Sherman-Morrison the drop in the posterior variance of the linear predictor at an
evaluation point ``m`` is ``(z_m^T P^-1 z)^2 / (1 + z^T P^-1 z)``.  ``aopt_reduction`` averages that
over the candidate pool itself (the brief's "pool-averaged"); ``aopt_reduction_cohort_plus_pool``
averages it over the cohort cells and the pool together.  Both are reported; the ranking is by the
first.

This file scores nothing and writes no contrast.  It is a deliverable list, and the caveats printed
into ``L4_REPORT.md`` travel with it.

    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe gen16_leads/scripts/l4_rank.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "gen13_separation", ROOT / "gen14_direction", ROOT / "gen15_curve",
           ROOT / "gen16_leads"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen13sep import paths  # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights  # noqa: E402
from gen13sep.cohort import apply_quarantine, load_bundle  # noqa: E402
from gen13sep.features import DONOR_COLUMNS  # noqa: E402
from gen13sep.metals import LANTHANIDES  # noqa: E402
from gen14.dirbench import load  # noqa: E402
from gen14.models import dir_logistic  # noqa: E402
from gen15.fewshot import centred_residual, pick_support, residual_covariance  # noqa: E402
from gen16 import l4_acq as L  # noqa: E402

LEAD = "L4"
OUT = ROOT / "gen16_leads" / "results" / "L4"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "l4_rank.log"
SIDE_K = ROOT / "gen15_curve" / "exp" / "external" / "data" / "side_K_logk.parquet"
NEW_TANIMOTO = 0.7          # the chemotype clustering's own single-linkage threshold
TOPO_PREFIXES = ("coord__dist__", "coord__arm__")
CENSUS_COORD = ["coord__arm__topicity", "coord__arm__estimated_denticity_per_pocket",
                "coord__arm__n_donor_bearing_branches", "coord__arm__n_amide_branches",
                "coord__dist__n_donor_pairs_within_3", "coord__dist__frac_donor_pairs_within_3",
                "coord__dist__donor_network_diameter"]


def say(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


# --------------------------------------------------------------------------------------
# fingerprints and Tanimoto
# --------------------------------------------------------------------------------------
def tanimoto_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Binary Tanimoto between every row of A and every row of B (0/1 bit matrices)."""
    inter = A @ B.T
    a = A.sum(axis=1)[:, None]
    b = B.sum(axis=1)[None, :]
    union = a + b - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)


def census_string(row: pd.Series, cols: list[str], short: dict[str, str]) -> str:
    bits = []
    for c in cols:
        v = row.get(c, np.nan)
        if pd.isna(v):
            continue
        v = float(v)
        bits.append(f"{short[c]}={v:g}" if v != int(v) else f"{short[c]}={int(v)}")
    return " ".join(bits)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main() -> None:
    LOG.write_text("", encoding="utf-8")
    say("L4 prospective ranking: A-optimal over the full frozen cohort")
    bench = load()
    P = L.prepare(bench)
    fr = bench.frame
    say(f"cohort: {len(fr)} cells, {fr.extractant.nunique()} extractants, {len(P.centroid)} chemotypes")

    # ---- the chosen design: the cohort's well-determined cells, standardised on the cohort ----
    st = L.fit_standardiser(P.topo)
    Zcohort = st(P.topo)
    Zchosen = Zcohort[P.rich]
    say(f"chosen design: {int(P.rich.sum())} well-determined cells x {Zchosen.shape[1]} TOPO39 columns; "
        f"{int(st.constant.sum())} constant/all-missing columns zeroed")

    # ---- the deployed G14 fit on the whole cohort (direction probability + residual covariance) --
    rich_idx = np.flatnonzero(P.rich)
    ext_all = P.ext_of
    amp, cur = bench.coef[:, 0], bench.coef[:, 1]
    w_rich = cell_weights(bench.groups[rich_idx], bench.n_obs[rich_idx])
    w_all = cell_weights(bench.groups, bench.n_obs)
    mean_mag = float(np.average(np.abs(amp[rich_idx]), weights=w_rich))
    mean_cur = float(np.average(cur, weights=w_all))
    say(f"full-cohort G14: mean |amplitude| {mean_mag:.4f}, mean curvature {mean_cur:.4f}")

    fitter = dir_logistic()
    p_cohort = np.asarray(fitter(P.topo[rich_idx], amp[rich_idx], w_rich, bench.groups[rich_idx],
                                 P.topo, 0, ext_all[rich_idx]), dtype=float)
    sign_cohort = np.where(p_cohort >= 0.5, -1.0, 1.0)
    coef_cohort = np.column_stack([sign_cohort * mean_mag, np.full(len(fr), mean_cur)])
    curves = coef_cohort @ bench.basis
    resid = np.array([centred_residual(bench.Y[ci], curves[ci]) for ci in rich_idx])
    cov = residual_covariance(resid)
    ALL_PAIRS = [(a, b) for a in range(len(LANTHANIDES)) for b in range(a + 1, len(LANTHANIDES))]
    dopt3 = pick_support(ALL_PAIRS, 3, "dopt", cov)
    dopt3_names = [f"{LANTHANIDES[a]}-{LANTHANIDES[b]}" for a, b in dopt3]
    say(f"full-cohort residual covariance from {len(resid)} well-determined residual curves; "
        f"first three D-optimal pairs (all 91 available): {', '.join(dopt3_names)}")

    # ---- feature sources for candidates ----
    coord = pd.read_parquet(paths.GEN122_COORDINATION_PARQUET)
    if coord.index.name != "extractant":
        coord = coord.set_index("extractant")
    coord = coord.apply(pd.to_numeric, errors="coerce")
    coord.columns = [c if c.startswith("coord__") else f"coord__{c}" for c in coord.columns]
    # the exact TOPO39 column names, in the exact order of the columns of ``P.topo``
    lean_cols = list(bench.columns(LEAN_BLOCKS))
    topo_cols = [lean_cols[i] for i in P.fs["TOPO39"]]
    assert len(topo_cols) == P.topo.shape[1] == 39, (len(topo_cols), P.topo.shape)
    assert all(c.startswith(TOPO_PREFIXES) for c in topo_cols)
    chem = pd.read_parquet(paths.CHEMISTRY_MAP_PARQUET).set_index("extractant")

    cohort_ext = fr.drop_duplicates("extractant")[["extractant", "extractant_name", "chemotype"]]
    cohort_ext = cohort_ext.reset_index(drop=True)
    Mco = L.morgan_matrix(cohort_ext.extractant.astype(str).tolist())
    cohort_chemos = set(fr.chemotype.astype(str))

    donor_short = {c: c.replace("chem__donor__", "").replace("chem__", "") for c in DONOR_COLUMNS}
    coord_short = {c: c.replace("coord__arm__", "").replace("coord__dist__", "") for c in CENSUS_COORD}

    notes: dict = {"topo39_columns": topo_cols, "dopt3_all_pairs": dopt3_names,
                   "n_residual_curves": int(len(resid)),
                   "chosen_cells": int(P.rich.sum()), "lambda": L.LAMBDA}

    # ==================================================================================
    # pool (i): bundle extractants with exactly one measured lanthanide
    # ==================================================================================
    # the frozen cohort's own bundle path: quarantine applied, gen6 provenance joined
    bundle, _q = apply_quarantine(load_bundle())
    prov = pd.read_parquet(paths.GEN6_PROVENANCE_PARQUET)[["safe_exp_id", "publication_id"]]
    bundle = bundle.merge(prov, on="safe_exp_id", how="left", validate="one_to_one")
    bundle = bundle[bundle["metal"].isin(LANTHANIDES)]
    per = bundle.groupby("canonical_smiles").agg(
        n_lanthanides=("metal", "nunique"),
        metals=("metal", lambda s: ";".join(sorted(set(s)))),
        n_rows=("metal", "size"),
        n_publications=("publication_id", "nunique"),
        name=("extractant_name", "first"))
    single = per[per.n_lanthanides == 1].copy()
    in_cohort = set(fr.extractant.astype(str))
    n_single_in_cohort = int(single.index.astype(str).isin(in_cohort).sum())
    single = single[~single.index.astype(str).isin(in_cohort)]
    say(f"pool (i): {len(per)} bundle extractants with lanthanide rows; {len(single)} have exactly one "
        f"measured lanthanide and are outside the cohort ({n_single_in_cohort} single-metal but in cohort)")

    have = single.index.astype(str).isin(coord.index.astype(str))
    missing_coord = sorted(single.index[~have].astype(str))
    B = single[have].copy()
    Xb = coord.reindex(B.index.astype(str))[topo_cols].to_numpy(dtype=float)
    all_nan = ~np.isfinite(Xb).any(axis=1)
    if all_nan.any():
        missing_coord += sorted(B.index[all_nan].astype(str))
        B, Xb = B[~all_nan], Xb[~all_nan]
    say(f"pool (i): TOPO39 available for {len(B)}; excluded {len(missing_coord)} with no computable "
        f"TOPO39 row in the frozen coordination parquet")
    notes["pool_bundle_excluded_no_topo39"] = len(missing_coord)
    notes["pool_bundle_excluded_smiles"] = missing_coord

    Zb = st(Xb)
    red_pool = L.prospective_aopt(Zchosen, Zb, Zb)
    red_both = L.prospective_aopt(Zchosen, Zb, np.vstack([Zcohort, Zb]))
    Mb = L.morgan_matrix(B.index.astype(str).tolist())
    T = tanimoto_matrix(Mb, Mco)
    nn = T.argmax(axis=1)
    p_b = np.asarray(fitter(P.topo[rich_idx], amp[rich_idx], w_rich, bench.groups[rich_idx],
                            Xb, 0, ext_all[rich_idx]), dtype=float)

    chem_b = chem.reindex(B.index.astype(str))
    rows = []
    for i, smi in enumerate(B.index.astype(str)):
        ch = chem_b.iloc[i]
        cl = str(ch.get("chem__supercluster", "")) if pd.notna(ch.get("chem__supercluster", np.nan)) else ""
        in_co = cl in cohort_chemos
        m = B.metals.iloc[i]
        mi = LANTHANIDES.index(m) if m in LANTHANIDES else None
        restricted = [(a, b) for (a, b) in ALL_PAIRS if mi in (a, b)]
        d3 = pick_support(restricted, 3, "dopt", cov) if restricted else []
        rows.append({
            "rank": 0, "smiles": smi, "name": B.name.iloc[i],
            "aopt_reduction": float(red_pool.aopt_reduction.iloc[i]),
            "aopt_reduction_cohort_plus_pool": float(red_both.aopt_reduction.iloc[i]),
            "own_variance": float(red_pool.own_variance.iloc[i]),
            "measured_lanthanide": m, "n_rows_bundle": int(B.n_rows.iloc[i]),
            "n_publications": int(B.n_publications.iloc[i]),
            "chemotype_frozen": cl, "chemotype_in_cohort": bool(in_co),
            "nn_tanimoto_cohort": float(T[i, nn[i]]),
            "nn_cohort_extractant": cohort_ext.extractant.iloc[nn[i]],
            "nn_cohort_name": cohort_ext.extractant_name.iloc[nn[i]],
            "nn_cohort_chemotype": cohort_ext.chemotype.iloc[nn[i]],
            "label": cl if in_co else "new",
            "is_new_T_lt_0.7": bool(T[i, nn[i]] < NEW_TANIMOTO),
            "p_heavy_G14_full_cohort": float(p_b[i]),
            "direction_G14": "heavy" if p_b[i] >= 0.5 else "light",
            "donor_census": census_string(ch, list(DONOR_COLUMNS), donor_short),
            "donor_census_source": "frozen chemistry map",
            "topo_census": census_string(pd.Series(Xb[i], index=topo_cols), CENSUS_COORD, coord_short),
            "dopt3_pairs_all": "|".join(dopt3_names),
            "dopt3_pairs_with_measured": "|".join(f"{LANTHANIDES[a]}-{LANTHANIDES[b]}" for a, b in d3),
            "pool": "bundle_single_lanthanide", "lead": LEAD,
        })
    RB = pd.DataFrame(rows).sort_values("aopt_reduction", ascending=False).reset_index(drop=True)
    RB["rank"] = np.arange(1, len(RB) + 1)
    RB.to_csv(OUT / "ranking_pool_bundle.csv", index=False)
    say(f"pool (i): wrote {len(RB)} ranked candidates; {int(RB.label.eq('new').sum())} labelled 'new', "
        f"{RB.chemotype_frozen.nunique()} distinct frozen chemotype labels")

    # ==================================================================================
    # pool (ii): the 273 external aqueous-logK ligands
    # ==================================================================================
    K = pd.read_parquet(SIDE_K)
    smi_col = next((c for c in ("canonical_smiles", "smiles", "SMILES", "ligand_smiles")
                    if c in K.columns), None)
    if smi_col is None:
        raise SystemExit(f"no SMILES column in {SIDE_K}; columns: {list(K.columns)[:40]}")
    K = K.drop_duplicates(smi_col).reset_index(drop=True)
    have_cols = [c for c in topo_cols if c in K.columns]
    missing_cols = [c for c in topo_cols if c not in K.columns]
    say(f"pool (ii): {len(K)} ligands; TOPO39 columns present {len(have_cols)}/39"
        + (f"; MISSING {missing_cols}" if missing_cols else ""))
    notes["pool_logk_missing_topo39_columns"] = missing_cols
    Xk = np.full((len(K), len(topo_cols)), np.nan)
    for j, c in enumerate(topo_cols):
        if c in K.columns:
            Xk[:, j] = pd.to_numeric(K[c], errors="coerce").to_numpy(dtype=float)
    all_nan = ~np.isfinite(Xk).any(axis=1)
    n_drop_k = int(all_nan.sum())
    K, Xk = K[~all_nan].reset_index(drop=True), Xk[~all_nan]
    say(f"pool (ii): {len(K)} ligands with a computable TOPO39 row; excluded {n_drop_k}")
    notes["pool_logk_excluded_no_topo39"] = n_drop_k

    Zk = st(Xk)
    redk_pool = L.prospective_aopt(Zchosen, Zk, Zk)
    redk_both = L.prospective_aopt(Zchosen, Zk, np.vstack([Zcohort, Zk]))
    Mk = L.morgan_matrix(K[smi_col].astype(str).tolist())
    Tk = tanimoto_matrix(Mk, Mco)
    nnk = Tk.argmax(axis=1)
    p_k = np.asarray(fitter(P.topo[rich_idx], amp[rich_idx], w_rich, bench.groups[rich_idx],
                            Xk, 0, ext_all[rich_idx]), dtype=float)
    chem_k = chem.reindex(K[smi_col].astype(str))
    kname = next((c for c in ("ligand_name", "name", "extractant_name") if c in K.columns), None)
    kdonor = [c for c in K.columns if c.startswith("coord__donor__")]
    kdonor_short = {c: c.replace("coord__donor__", "") for c in kdonor}

    rows = []
    for i in range(len(K)):
        smi = str(K[smi_col].iloc[i])
        ch = chem_k.iloc[i]
        cl = str(ch.get("chem__supercluster", "")) if pd.notna(ch.get("chem__supercluster", np.nan)) else ""
        in_co = cl in cohort_chemos
        if kdonor:
            census, src = census_string(K.iloc[i], kdonor, kdonor_short), "external coord__donor__ columns"
        elif cl:
            census, src = census_string(ch, list(DONOR_COLUMNS), donor_short), "frozen chemistry map"
        else:
            census, src = "", "unavailable (external ligand, TOPO39 only)"
        rows.append({
            "rank": 0, "smiles": smi, "name": str(K[kname].iloc[i]) if kname else "",
            "aopt_reduction": float(redk_pool.aopt_reduction.iloc[i]),
            "aopt_reduction_cohort_plus_pool": float(redk_both.aopt_reduction.iloc[i]),
            "own_variance": float(redk_pool.own_variance.iloc[i]),
            "chemotype_frozen": cl, "chemotype_in_cohort": bool(in_co),
            "nn_tanimoto_cohort": float(Tk[i, nnk[i]]),
            "nn_cohort_extractant": cohort_ext.extractant.iloc[nnk[i]],
            "nn_cohort_name": cohort_ext.extractant_name.iloc[nnk[i]],
            "nn_cohort_chemotype": cohort_ext.chemotype.iloc[nnk[i]],
            "label": cl if in_co else "new",
            "is_new_T_lt_0.7": bool(Tk[i, nnk[i]] < NEW_TANIMOTO),
            "p_heavy_G14_full_cohort": float(p_k[i]),
            "direction_G14": "heavy" if p_k[i] >= 0.5 else "light",
            "donor_census": census, "donor_census_source": src,
            "topo_census": census_string(pd.Series(Xk[i], index=topo_cols), CENSUS_COORD, coord_short),
            "dopt3_pairs_all": "|".join(dopt3_names),
            "dopt3_pairs_with_measured": "",
            "pool": "external_logk_273", "lead": LEAD,
        })
    RK = pd.DataFrame(rows).sort_values("aopt_reduction", ascending=False).reset_index(drop=True)
    RK["rank"] = np.arange(1, len(RK) + 1)
    RK.to_csv(OUT / "ranking_pool_logk.csv", index=False)
    say(f"pool (ii): wrote {len(RK)} ranked candidates; {int(RK.label.eq('new').sum())} labelled 'new'")

    (OUT / "ranking_notes.json").write_text(json.dumps(notes, indent=2, default=str), encoding="utf-8")

    cols = ["rank", "name", "label", "nn_tanimoto_cohort", "nn_cohort_name", "aopt_reduction",
            "p_heavy_G14_full_cohort", "measured_lanthanide", "dopt3_pairs_with_measured"]
    say("\n=== pool (i) top 10 ===")
    say(RB.head(10)[cols].round(4).to_string(index=False))
    say("\n=== pool (ii) top 10 ===")
    say(RK.head(10)[[c for c in cols if c in RK.columns]].round(4).to_string(index=False))
    say("DONE")


if __name__ == "__main__":
    main()
