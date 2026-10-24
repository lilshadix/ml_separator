#!/usr/bin/env python
"""Stage-0 duplicate and near-copy scan of V1 (CLAUDE_CODE_PROMPT.md section 5.1).  Label-based scan, NO SCORE: it
reads V1 log D (to compare cells) and the round-4 REF read sets (row identity and read_set columns only); it computes
no prediction, residual, R^2, SSE or contrast.

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s0_dupscan.py

Writes results/stage0/dupscan_links.csv (every pair with >= 2 bit-identical metals, or an OFFSET / NEAR_ID flag),
dupscan_quant.csv (per-cell quantisation metrics), dupscan_nearcopy.csv (lab-1 near-copy donor scan) and
dupscan_summary.json (rule, counts, known-item checks, provenance list for U1 / M25).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r5 import dupes as DU  # noqa: E402
from r5 import population as POP  # noqa: E402
from gen20k import data as D  # noqa: E402

OUT = paths.RES / "stage0"
SRC = {"lab1skepB_nearcopy.py": paths.DESIGN / "ideas2" / "scripts" / "lab1skepB_nearcopy.py",
       "P1_ct_build.py": paths.DESIGN / "probes" / "scripts" / "P1_ct_build.py"}


def v2_status(ids) -> dict:
    L = POP.v2_lists()
    keep = set(L["V2-2"].values())
    out = {}
    for c in ids:
        s = []
        if c in L["V2-1"]:
            s.append("V2-1")
        if c in L["V2-2"]:
            s.append("V2-2drop")
        if c in keep:
            s.append("V2-2keep")
        if c in L["V2-3"]:
            s.append("V2-3")
        if c in L["V2-5"]:
            s.append("V2-5floor")
        out[c] = ";".join(s)
    return out


def main() -> int:
    from r5 import sealcheck
    sealcheck.refuse_if_sealed("scripts/s0_dupscan.py (rewrites sealed Stage-0 inputs)", "--amend" in sys.argv[1:])
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    assert tuple(D.LANTHANIDES) == POP.LN, "lanthanide order differs from gen20k"
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f["cell_id"].astype(str).tolist()
    pub = dict(zip(ids, f["publication_id"].astype(str)))
    ext = dict(zip(ids, f["extractant"].astype(str)))
    ename = dict(zip(ids, f["extractant_name"].astype(str)))
    chem = dict(zip(ids, f["chemotype"].astype(str)))
    cfam = dict(zip(ids, f["chem_family"].astype(str)))
    meta = POP.metadata_frame(f)
    fin_meta = POP.finite_from_metals(meta)
    assert np.array_equal(fin_meta, np.isfinite(Y)), "metals metadata disagrees with the finite pattern of V1"
    c3 = set(POP.c3_cells_from_ref())
    v2s = v2_status(ids)
    print(f"bench {len(ids)} cells, C3 {len(c3)}; {time.time() - t0:.1f}s", flush=True)

    # ---- bit identity (frozen) --------------------------------------------------------------------------
    bt = DU.bit_identity(Y, ids)
    flagged = DU.flagged_links(bt)
    print(f"bit identity: {len(bt)} pairs with >= 2 identical metals; {len(flagged)} flagged; {time.time() - t0:.1f}s",
          flush=True)
    # ---- offset / near-identity ---------------------------------------------------------------------------
    off = DU.offset_scan(Y, ids, flagged)
    print(f"offset scan: {int(off.flag_offset.sum())} OFFSET, {int(off.flag_nearid.sum())} NEAR_ID; "
          f"{time.time() - t0:.1f}s", flush=True)
    # ---- quantisation -------------------------------------------------------------------------------------
    q = DU.quant_scan(Y, ids)
    q.insert(1, "publication", q.cell_id.map(pub))
    q.insert(2, "c3", q.cell_id.isin(c3))
    q["v2"] = q.cell_id.map(v2s)
    # ---- lab-1 near-copy donors -------------------------------------------------------------------------
    rs, rsets = {}, {}
    for d in paths.DESIGNS:
        t = pd.read_parquet(paths.R4PRED / f"REF_{d}.parquet", columns=["split_seed", "cell_id", "read_set"])
        g = t.drop_duplicates(["split_seed", "cell_id"])
        if g.duplicated(["split_seed", "cell_id"]).any() or t.groupby(["split_seed", "cell_id"]).read_set.nunique().max() != 1:
            raise AssertionError(f"REF_{d}: read_set not constant within (seed, cell)")
        rsets[d] = {(int(r.split_seed), r.cell_id): r.read_set for r in g.itertuples(index=False)}
        for (s, c), v in rsets[d].items():
            rs[(d, s, c)] = [LN_IDX[m] for m in v.split(",")]
    same_rs = rsets["BPG"] == rsets["BP"]
    nc = DU.near_copy_scan(Y, meta, rs, POP.v2drop_set())
    mcd = DU.metal_conc_duplicates(meta, fin_meta)
    mcl = DU.metal_conc_links(meta, fin_meta)
    mc_drop = {c for g in mcd for c in g[1:]}
    nc["mconc_dup_dropped"] = nc.donor.isin(mc_drop)
    print(f"near-copy scan: {len(nc)} (target, donor) pairs, {int((nc.n_flagged > 0).sum())} flagged in >= 1 read set; "
          f"metal-conc duplicate groups {len(mcd)}; {time.time() - t0:.1f}s", flush=True)

    # ---- merged link table --------------------------------------------------------------------------------
    L = bt.merge(off, on=["cell_a", "cell_b"], how="outer", suffixes=("", "_off"))
    L["n_shared"] = L["n_shared"].fillna(L.pop("n_shared_off"))
    for c in ("n_ident",):
        L[c] = L[c].fillna(0).astype(int)
    L["frac_ident"] = L["n_ident"] / L["n_shared"]
    for c in ("flag_bit", "flag_offset", "flag_nearid", "integer_shift"):
        L[c] = L[c].astype("boolean").fillna(False).astype(bool)
    L["ident_metals"] = L["ident_metals"].fillna("")
    for side in ("a", "b"):
        L[f"pub_{side}"] = L[f"cell_{side}"].map(pub)
        L[f"chem_{side}"] = L[f"cell_{side}"].map(chem)
        L[f"family_{side}"] = L[f"cell_{side}"].map(cfam)
        L[f"ext_name_{side}"] = L[f"cell_{side}"].map(ename)
        L[f"c3_{side}"] = L[f"cell_{side}"].isin(c3)
        L[f"v2_{side}"] = L[f"cell_{side}"].map(v2s)
    L["class"] = [DU.pair_class(a, b, pub, ext) for a, b in zip(L.cell_a, L.cell_b)]
    known = {}
    for k, (a, b, _, _) in DU.KNOWN_BIT.items():
        known[frozenset((DU.resolve(a, ids), DU.resolve(b, ids)))] = k
    for k, (a, b) in DU.KNOWN_NOT_FLAGGED.items():
        known.setdefault(frozenset((DU.resolve(a, ids), DU.resolve(b, ids))), k)

    # ---- named sensitivity items (I2 items 2, 5-11) ------------------------------------------------------
    def item_links(spec) -> pd.DataFrame:
        parts = []
        for a, b in spec.get("pairs", []):
            a, b = DU.resolve(a, ids), DU.resolve(b, ids)
            parts.append(L[((L.cell_a == a) & (L.cell_b == b)) | ((L.cell_a == b) & (L.cell_b == a))])
        for c, pps in spec.get("cell_partner_pubs", {}).items():
            c = DU.resolve(c, ids)
            parts.append(L[((L.cell_a == c) & L.pub_b.isin(pps)) | ((L.cell_b == c) & L.pub_a.isin(pps))])
        for pa, pb in spec.get("pub_pairs", []):
            parts.append(L[((L.pub_a == pa) & (L.pub_b == pb)) | ((L.pub_a == pb) & (L.pub_b == pa))])
        h = pd.concat(parts).drop_duplicates(["cell_a", "cell_b"]) if parts else L.iloc[0:0]
        want = [s for s in spec["scan"].split("|") if s in ("OFFSET", "NEAR_ID")]
        m = np.zeros(len(h), bool)
        if "OFFSET" in want:
            m |= h.flag_offset.to_numpy()
        if "NEAR_ID" in want:
            m |= h.flag_nearid.to_numpy()
        return h[m]
    sens = {}
    for k, spec in DU.KNOWN_SENS.items():
        hits = item_links(spec)
        for a, b in zip(hits.cell_a, hits.cell_b):
            known.setdefault(frozenset((a, b)), k)
        qc = q[q.publication.isin(spec.get("pubs", []))]
        qflags = {fl_: sorted(qc.cell_id[qc[fl_]].tolist()) for fl_ in ("Q_TIES", "Q_ADJEQ", "Q_COARSE", "Q_SMALLINT")
                  if fl_ in spec["scan"]}
        diag = {}
        for a, b in spec.get("pairs", []):
            a, b = DU.resolve(a, ids), DU.resolve(b, ids)
            diag[f"{a}~{b}"] = DU.closest_window(Y, ids.index(a), ids.index(b))
        sens[k] = {"scan": spec["scan"], "n_link_flags": int(len(hits)),
                   "links": hits[["cell_a", "cell_b", "class", "pub_a", "pub_b", "off_segment", "off_mean", "off_sd",
                                  "nearid_maxabs", "flag_offset", "flag_nearid", "flag_bit"]].round(4).pipe(recs),
                   "quant_flags": qflags, "closest_window_diagnostic": diag,
                   "detected": bool(len(hits) or any(qflags.values()))}
    L["known_item"] = [known.get(frozenset((a, b)), "") for a, b in zip(L.cell_a, L.cell_b)]
    # provenance tiers (POST HOC: set after the broad OFFSET tier was seen to flag several hundred links, mostly
    # parallel heavy-end DGA segments; the tiers only choose what is sent to U1 / M25, never an analysis pool)
    tier = np.where(L.flag_bit, "BIT",
                    np.where(L.known_item.str.startswith("item") & (L.flag_offset | L.flag_nearid), "KNOWN_ITEM",
                             np.where(L.flag_nearid & (L["class"] != "within_pub|same_ext"), "NEAR_ID_CROSS",
                                      np.where(L.integer_shift, "INTEGER_SHIFT", ""))))
    L["provenance_tier"] = tier
    L["provenance_list"] = L.provenance_tier != ""
    L = L.sort_values(["flag_bit", "n_ident", "cell_a", "cell_b"], ascending=[False, False, True, True]).reset_index(drop=True)
    cols = ["cell_a", "cell_b", "class", "pub_a", "pub_b", "chem_a", "chem_b", "family_a", "family_b", "ext_name_a",
            "ext_name_b", "c3_a", "c3_b", "v2_a", "v2_b", "n_shared", "n_ident", "frac_ident", "ident_metals", "flag_bit",
            "off_len", "off_segment", "off_mean", "off_sd", "flag_offset", "integer_shift", "nearid_maxabs", "flag_nearid",
            "known_item", "provenance_tier", "provenance_list"]
    L = L[cols]

    # ---- known-item checks ---------------------------------------------------------------------------------
    def row(a, b):
        s = L[((L.cell_a == a) & (L.cell_b == b)) | ((L.cell_a == b) & (L.cell_b == a))]
        return s.iloc[0] if len(s) else None
    checks = {}
    for k, (a, b, ne, ns) in DU.KNOWN_BIT.items():
        a, b = DU.resolve(a, ids), DU.resolve(b, ids)
        r = row(a, b)
        got = (int(r.n_ident), int(r.n_shared)) if r is not None else (0, None)
        checks[k] = {"cells": [a, b], "publication": [pub[a], pub[b]], "extractant_names": [ename[a], ename[b]],
                     "same_extractant": ext[a] == ext[b], "expected_ident_of_shared": [ne, ns],
                     "found_ident_of_shared": list(got), "flagged": bool(r is not None and r.flag_bit),
                     "v2": [v2s[a], v2s[b]], "c3": [a in c3, b in c3],
                     "pass": bool(r is not None and r.flag_bit and got == (ne, ns))}
    for k, (a, b) in DU.KNOWN_NOT_FLAGGED.items():
        a, b = DU.resolve(a, ids), DU.resolve(b, ids)
        r = row(a, b)
        checks[k] = {"cells": [a, b], "found_ident_of_shared": [int(r.n_ident), int(r.n_shared)] if r is not None else [0, None],
                     "flagged": bool(r is not None and r.flag_bit), "pass": bool(r is None or not r.flag_bit)}
    L2 = POP.v2_lists()["V2-2"]
    b9 = {k: {"cells": v["cells"], "V2-2_dropped_member": [c for c in v["cells"] if c in L2],
              "V2-2_kept_member": [c for c in v["cells"] if c in set(L2.values())],
              "is_V2-2_pair": any(L2.get(c) in v["cells"] for c in v["cells"])}
          for k, v in checks.items() if k.startswith("b9_")}
    sub = L[(L.n_ident >= 2) & ~L.flag_bit]

    # ---- summary ---------------------------------------------------------------------------------------------
    fl = L[L.flag_bit]
    off_l, nid_l = L[L.flag_offset], L[L.flag_nearid]
    ncf = nc[nc.n_flagged > 0]
    summ = {
        "label": "STAGE-0 DATA RULE (label-based scan of V1; no prediction, no score)",
        "section": "CLAUDE_CODE_PROMPT.md 5.1",
        "rule_bit_identity_frozen": {"equality": "float64 ==", "min_ident": DU.BIT_MIN_IDENT, "min_frac_of_shared": DU.BIT_MIN_FRAC,
                                     "scope": "all 521 bench cells, every unordered pair, within and across publications and "
                                              "extractants", "action": "remove flagged links (not cells) from E-CT and E-LAB pools"},
        "rule_offset_sensitivity": {"min_len": DU.OFF_MIN_LEN, "sd_max": DU.OFF_SD_MAX, "min_abs_shift": DU.OFF_MIN_SHIFT,
                                    "nearid_maxabs": DU.NEARID_MAXABS, "action": "sensitivity + U1/M25 provenance list only"},
        "rule_quant_sensitivity": {"tie_min": DU.Q_TIE_MIN, "coarse_step": DU.Q_COARSE_STEP, "coarse_half_median": DU.Q_COARSE_HALF,
                                   "smallint_frac": DU.Q_SMALLINT_FRAC, "smallint_pq_max": DU.Q_SMALLINT_PQ},
        "rule_near_copy_lab1": {"same_medium": "acid type, diluent, |dlog10 acid| <= 0.15", "tol_log": DU.NC_TOL,
                                "reads": "the target's 4 DOPT_Q reads per (design, seed) from REF read_set",
                                "donors": "own publication, other extractant, >= 6 finite metals, not V2-1/2/3",
                                "metal_conc_duplicates": "one member kept per group (most finite metals, then smallest id)"},
        "inputs": {"bench": D.assert_matches_manifest(f), "REF_read_sets_identical_BP_BPG": bool(same_rs),
                   "copied_from_LF_sha256": {k: paths.sha256_text_lf(p) for k, p in SRC.items()}},
        "n_cells": len(ids), "n_pairs": len(ids) * (len(ids) - 1) // 2,
        "bit_identity": {
            "n_pairs_ge2_identical": int(len(bt)), "n_flagged": int(len(fl)),
            "flagged_by_class": fl["class"].value_counts().to_dict(),
            "flagged_links": fl[["cell_a", "cell_b", "class", "pub_a", "pub_b", "ext_name_a", "ext_name_b", "n_ident",
                                 "n_shared", "c3_a", "c3_b", "v2_a", "v2_b", "known_item"]].pipe(recs),
            "flagged_involving_c3_both": int((fl.c3_a & fl.c3_b).sum()),
            "flagged_cross_publication": int((fl.pub_a != fl.pub_b).sum()),
            "known_items": checks, "known_items_all_pass": bool(all(v["pass"] for v in checks.values())),
            "b9dfc468cd_further_pairs_V2_2": b9,
            "unflagged_with_ge2_identical": {"n": int(len(sub)), "by_class": sub["class"].value_counts().to_dict(),
                                             "links": sub[["cell_a", "cell_b", "class", "pub_a", "pub_b", "n_ident",
                                                           "n_shared"]].pipe(recs)}},
        "offset": {"n_offset": int(len(off_l)), "by_class": off_l["class"].value_counts().to_dict(),
                   "by_window_length": off_l.off_len.astype(int).value_counts().sort_index().to_dict(),
                   "by_family_pair": (off_l.family_a.where(off_l.family_a <= off_l.family_b, off_l.family_b) + "|"
                                      + off_l.family_b.where(off_l.family_a <= off_l.family_b, off_l.family_a)
                                      ).value_counts().to_dict(),
                   "n_integer_shift": int(off_l.integer_shift.sum()),
                   "n_offset_on_provenance_list": int(off_l.provenance_list.sum()),
                   "reading": "the declared OFFSET tier is not specific: most links are parallel heavy-end segments of "
                              "homologous solvating extractants (chemistry, not copying); only the named items, integer "
                              "shifts and cross near-identities go to the provenance list (POST HOC tiering)"},
        "near_identity": {"n_nearid": int(len(nid_l)), "by_class": nid_l["class"].value_counts().to_dict(),
                          "n_nearid_on_provenance_list": int(nid_l.provenance_list.sum()),
                          "links": nid_l[["cell_a", "cell_b", "class", "pub_a", "pub_b", "ext_name_a", "ext_name_b",
                                          "n_shared", "nearid_maxabs", "known_item"]].round(4).pipe(recs)},
        "quant": {fl_: {"n_cells": int(q[fl_].sum()), "n_c3_cells": int((q[fl_] & q.c3).sum()),
                        "publications": q[q[fl_]].publication.value_counts().to_dict()}
                  for fl_ in ("Q_TIES", "Q_ADJEQ", "Q_COARSE", "Q_SMALLINT", "Q_UNRESOLVED")},
        "known_sensitivity_items": sens,
        "near_copy_lab1": {"n_target_donor_pairs": int(len(nc)), "n_flagged_pairs": int(len(ncf)),
                           "n_targets_with_a_near_copy_donor": int(ncf.target.nunique()),
                           "by_publication_targets": ncf.groupby("publication").target.nunique().to_dict(),
                           "n_pairs_donor_dropped_as_metal_conc_duplicate": int(nc.mconc_dup_dropped.sum()),
                           "flagged_pairs": ncf.pipe(recs)},
        "metal_conc_duplicates": {"rule": "same publication and extractant, cond__ differ only in metal concentration "
                                          "(both recorded), >= 3 shared finite metals; first member kept",
                                  "n_groups": len(mcd), "groups": mcd,
                                  "n_metal_conc_only_pairs": int(len(mcl)),
                                  "pairs_by_n_shared": mcl.n_shared.value_counts().sort_index().to_dict(),
                                  "split_series_pairs_by_publication": mcl[mcl.n_shared == 0].publication.value_counts().to_dict(),
                                  "note": "pairs sharing no metal are complementary halves of one series at two metal "
                                          "loadings (e.g. pub_5a68dc5665 0.0006 / 0.0007 mM), not repeated measurements"},
        "provenance_list_U1_M25": {"n_links": int(L.provenance_list.sum()),
                                   "by_tier": L[L.provenance_list].provenance_tier.value_counts().to_dict(),
                                   "tiering": "BIT > KNOWN_ITEM (an I2-named item flagged by its scan) > NEAR_ID_CROSS "
                                              "(near-identity across publications or extractants) > INTEGER_SHIFT; set "
                                              "POST HOC after the broad OFFSET counts were seen; selects provenance "
                                              "requests only, never an analysis pool",
                                   "publications": sorted(set(L[L.provenance_list].pub_a) | set(L[L.provenance_list].pub_b)),
                                   "quant_flag_publications": sorted(set(q[q.Q_TIES | q.Q_COARSE | q.Q_SMALLINT | q.Q_ADJEQ].publication)),
                                   "component_error_item12": "pub_84113c17ad (metadata, not detectable from values)"},
        "seconds": round(time.time() - t0, 1),
    }
    L.to_csv(OUT / "dupscan_links.csv", index=False, lineterminator="\n")
    q.to_csv(OUT / "dupscan_quant.csv", index=False, lineterminator="\n")
    nc.to_csv(OUT / "dupscan_nearcopy.csv", index=False, lineterminator="\n")
    paths.write_text_lf(OUT / "dupscan_summary.json", json.dumps(summ, indent=1, default=_js) + "\n")
    for k, v in checks.items():
        print(f"  {k:24s} found {v['found_ident_of_shared']} flagged={v['flagged']} pass={v['pass']}")
    for k, v in b9.items():
        print(f"  {k}: V2-2 pair={v['is_V2-2_pair']} dropped={v['V2-2_dropped_member']} kept={v['V2-2_kept_member']}")
    for k, v in sens.items():
        print(f"  {k:7s} [{v['scan']}] detected={v['detected']} links={v['n_link_flags']} quant={v['quant_flags']}")
    print(f"flagged bit links {len(fl)} (cross-publication {summ['bit_identity']['flagged_cross_publication']}); "
          f"provenance-list links {summ['provenance_list_U1_M25']['n_links']}")
    for p in ("dupscan_links.csv", "dupscan_quant.csv", "dupscan_nearcopy.csv", "dupscan_summary.json"):
        print("wrote", OUT / p)
    print(f"done {time.time() - t0:.1f}s")
    return 0


def recs(df: pd.DataFrame) -> list:
    """DataFrame records with NaN as None (strict JSON)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


LN_IDX = {m: k for k, m in enumerate(POP.LN)}

if __name__ == "__main__":
    raise SystemExit(main())
