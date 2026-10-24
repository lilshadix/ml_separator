#!/usr/bin/env python
"""Stage-0 E-CT sibling population (CLAUDE_CODE_PROMPT.md sections 5.1 and 6.1).  COUNTS ONLY, NO SCORE.

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s0_ect_population.py

The pools are built from metadata (r5.population.build_pools: identity, ``metals`` and cond__ fields); V1 log D enters
only through the frozen bit-identity rule (r5.dupes.bit_identity), which removes links, not cells.  Row counts come
from the round-4 REF parquets' row-identity columns (split_seed, cell_id, k).  Nothing here reads y, B1 or any
prediction.

Writes results/stage0/ect_population.json (counts per stratum with and without the bit-identity rule, the
flagged-link list and its effect on each pool, P1's reproduction targets), ect_population_cells.csv (one metadata
row per C3 target) and ect_sibling_links.csv (every V1 sibling link with CLEAN / BIT status).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402

import numpy as np  # noqa: E402

from r5 import dupes as DU  # noqa: E402
from r5 import population as POP  # noqa: E402
from gen20k import data as D  # noqa: E402

OUT = paths.RES / "stage0"
P1_TABLE = {  # P1 section 3 (population table) and PC P1: (cells, publications, units, rows per seed); no bit rule
    "V1_GE1": (149, 19, 22, 1111), "GE1_WITH_C5_NOBIT": (130, 18, 21, 968), "GE1_NOBIT": (115, 17, 20, 820),
    "PRIMARY_NOBIT": (99, 12, 14, 690), "WITH_C5_NOBIT": (114, 13, 15, 838), "EQ1_NOBIT": (16, 8, 9, 130),
    "V1_GE2_NOC5": (112, None, None, 821)}
SENS = {"PRIMARY": "CLEAN + bit rule, n_sib >= 2, without C5-BTBP (primary)",
        "WITH_C5": "as PRIMARY, with C5-BTBP (sensitivity)",
        "GE1": "CLEAN + bit rule, n_sib >= 1, without C5 (sensitivity)",
        "EQ1": "CLEAN + bit rule, n_sib = 1, without C5 (harm rule)",
        "GE1_WITH_C5": "CLEAN + bit rule, n_sib >= 1, with C5",
        "PRIMARY_NOBIT": "P1's CLEAN rule without the bit-identity check (P1 reproduction target; sensitivity)",
        "WITH_C5_NOBIT": "P1 CLEAN, n >= 2, with C5", "GE1_NOBIT": "P1 CLEAN, n >= 1, without C5",
        "GE1_WITH_C5_NOBIT": "P1 CLEAN, n >= 1, with C5", "EQ1_NOBIT": "P1 CLEAN, n = 1",
        "V1_GE1": "V1 pool (every covering sibling), n >= 1 (E6 population)",
        "V1_GE2_NOC5": "P1's V1 stratum: V1 pool, n >= 2, without C5 (bridge to IR's numbers)"}


def main() -> int:
    from r5 import sealcheck
    sealcheck.refuse_if_sealed("scripts/s0_ect_population.py (rewrites sealed Stage-0 inputs)", "--amend" in sys.argv[1:])
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    assert tuple(D.LANTHANIDES) == POP.LN
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    meta = POP.metadata_frame(f)
    finite = POP.finite_from_metals(meta)
    Y = np.asarray(bench.Y, float)                 # used ONLY by the bit-identity rule (and the coverage check)
    assert np.array_equal(finite, np.isfinite(Y)), "metals metadata disagrees with the finite pattern of V1"
    ids = meta["cell_id"].astype(str).tolist()
    c3 = POP.c3_cells_from_ref()
    v2drop = POP.v2drop_set()
    bt = DU.bit_identity(Y, ids)
    flagged = DU.flagged_links(bt)
    del Y
    pools = POP.build_pools(meta, finite, c3, v2drop, flagged)
    S = POP.strata(pools)
    rows = POP.rows_per_seed_from_ref()
    C = {k: POP.counts(v, pools, rows) for k, v in S.items()}
    for k in C:
        C[k]["description"] = SENS.get(k, "")

    # ---- P1 reproduction (counts) ---------------------------------------------------------------------------
    repro = {}
    for k, exp in P1_TABLE.items():
        c = C[k]
        rps = c["rows_per_seed"]["BPG"]["distinct"]
        got = (c["cells"], c["publications"], c["pub_x_chemotype_units"], rps[0] if len(rps) == 1 else rps)
        ok = all(e is None or e == g for e, g in zip(exp, got))
        repro[k] = {"expected_P1": list(exp), "found": list(got), "match": bool(ok)}
    same_bp = all(C[k]["rows_per_seed"]["BPG"] == C[k]["rows_per_seed"]["BP"] for k in C)
    # P1: after V2 removal the cond rule dropped 0 further siblings; the contact-time-only links are the V2-2 pairs
    L = POP.links_table(pools)
    noncl = L[~L.clean & ~L.target_v2drop & ~L.sibling.isin(v2drop)]

    # ---- flagged links and where they bite ------------------------------------------------------------------
    fl = []
    for a, b in flagged:
        r = bt[(bt.cell_a == a) & (bt.cell_b == b)].iloc[0]
        hit = {}
        for t in (a, b):
            s = b if t == a else a
            if t in pools:
                P = pools[t]
                hit[t] = sorted(nm for nm in ("clean", "plc_chem", "plc_xext", "plc_pub", "off_all", "off_pub", "off_xext")
                                if s in P[nm + "_clean" if nm != "clean" else "clean"])
        fl.append({"cell_a": a, "cell_b": b, "n_ident": int(r.n_ident), "n_shared": int(r.n_shared),
                   "publication": [meta.publication_id[ids.index(a)], meta.publication_id[ids.index(b)]],
                   "same_extractant": bool(meta.extractant[ids.index(a)] == meta.extractant[ids.index(b)]),
                   "v2drop": [a in v2drop, b in v2drop], "c3": [a in pools, b in pools],
                   "pools_where_removed_by_target": hit})
    prim, prim0 = C["PRIMARY"], C["PRIMARY_NOBIT"]
    out = {
        "label": "STAGE-0 DATA RULE (population counts only; no prediction, no score)",
        "section": "CLAUDE_CODE_PROMPT.md 6.1 (population) and 5.1 (bit-identity rule)",
        "definitions": {
            "sibling": "any other bench cell (521) with the same publication_id and extractant SMILES and a finite V1 "
                       "log D at every metal of the target (coverage from the 'metals' metadata)",
            "CLEAN": "target and sibling outside V2-1 / V2-2 / V2-3 (r3.datav2); sibling cond__ differs from the target "
                     "in at least one field family other than contact time / metal concentration",
            "bit_rule": "links flagged by r5.dupes.bit_identity (float64-identical V1 log D at >= 3 shared metals and "
                        ">= 50 % of the shared finite metals) removed from every pool; cells kept; n_sib counted after",
            "PRIMARY": "CLEAN + bit rule, n_sib >= 2, without C5-BTBP (pub_6a0ef8528c)",
            "units": "publication x chemotype", "rows_per_seed": "scored strict-adjacent rows of the round-4 REF parquet"},
        "n_c3_targets": len(c3), "n_v2drop_cells": len(v2drop),
        "PRIMARY_with_bit_rule": {k: prim[k] for k in ("cells", "publications", "pub_x_chemotype_units", "rows_per_seed",
                                                        "n_bit_flagged_sibling_links", "bit_flagged_sibling_links",
                                                        "targets_with_a_bit_flagged_sibling", "n_sib_bit")},
        "PRIMARY_without_bit_rule": {k: prim0[k] for k in ("cells", "publications", "pub_x_chemotype_units",
                                                            "rows_per_seed")},
        "PRIMARY_cells_lost_to_bit_rule": sorted(set(S["PRIMARY_NOBIT"]) - set(S["PRIMARY"])),
        "PRIMARY_cells_gained": sorted(set(S["PRIMARY"]) - set(S["PRIMARY_NOBIT"])),
        "strata": C,
        "flagged_links_all": fl, "n_flagged_links_all": len(fl),
        "p1_reproduction_counts": repro, "p1_reproduction_all_match": bool(all(v["match"] for v in repro.values())),
        "rows_identical_BP_BPG": bool(same_bp),
        "cond_rule_extra_drops_after_V2": int(len(noncl)),
        "inputs": {"bench": D.assert_matches_manifest(f),
                   "copied_from_LF_sha256": {"P1_ct_build.py": paths.sha256_text_lf(
                       paths.DESIGN / "probes" / "scripts" / "P1_ct_build.py"),
                       "P1_ct_diag.py": paths.sha256_text_lf(paths.DESIGN / "probes" / "scripts" / "P1_ct_diag.py")}},
        "seconds": round(time.time() - t0, 1),
    }
    paths.write_text_lf(OUT / "ect_population.json", json.dumps(out, indent=1, default=_js) + "\n")
    POP.pools_table(pools, S).to_csv(OUT / "ect_population_cells.csv", index=False, lineterminator="\n")
    L.to_csv(OUT / "ect_sibling_links.csv", index=False, lineterminator="\n")
    for k in ("PRIMARY", "PRIMARY_NOBIT", "WITH_C5", "GE1", "EQ1", "V1_GE2_NOC5"):
        c = C[k]
        print(f"{k:14s} cells {c['cells']:3d} pubs {c['publications']:2d} units {c['pub_x_chemotype_units']:2d} "
              f"rows/seed {c['rows_per_seed']['BPG']['distinct']} bit-flagged CLEAN sibling links {c['n_bit_flagged_sibling_links']}")
    print("P1 reproduction:", {k: v["match"] for k, v in repro.items()}, "| BP rows = BPG rows:", same_bp,
          "| cond-rule extra drops after V2:", len(noncl))
    print(f"flagged links (all bench): {len(fl)}")
    for p in ("ect_population.json", "ect_population_cells.csv", "ect_sibling_links.csv"):
        print("wrote", OUT / p)
    print(f"done {time.time() - t0:.1f}s")
    return 0


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


if __name__ == "__main__":
    raise SystemExit(main())
