#!/usr/bin/env python
"""Gate G-transfer for block C (PROTOCOL sections 7 and 12), evaluated BEFORE any block-C outer score; training side only.

--counts (label-free; pre-seal evidence): per outer fold-fit, the post-exclusion An(III) counts, the S24 changed fraction
  and the S15 matched diversity -> results/blockC_counts.json; and a recomputation of the union exclusions from the held
  cells WITHOUT calling excluded_by (publication, copy group, canonical extractant, linked chemotype; it reuses the sealed
  link map, canonicaliser and copy-group map, which tests/test_slots_c.py rebuilds from scratch) for every outer fold and
  every inner block, with the overlap counts that must all be zero -> results/leakage/isolation_an.csv.
Shard and combine modes run only under the verified seal; every shard row and gate_c.json carry the seal key, and
--combine refuses shards from another seal or not covering exactly the 50 dev fold-fits.
Count part: after the Ln u An union exclusions of every outer fold, the An(III) set keeps >= 5 independent publications
  (gen19 copy groups), >= 10 extractants and >= 100 rows (minimum over the 50 dev fold-fits).
Inner part: in >= 30 of the 50 outer fold-fits, S13's best inner score over POSITIVE An weights is below both S11's
  selected inner score (the minimum over B1 and the Ln-only networks) and B2's best inner score.  Uses only each fold's
  outer-training cells and the An records that survive its exclusions; no held-out cell is predicted.
S24 count gate (label-free): S24 runs only if its within-series metal cycle changes >= 10% of the fold's An rows in
  every one of the 50 fold-fits.

    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_gate_c.py --counts
    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_gate_c.py --shard 0/3
then  r4_gate_c.py --combine 3   writes results/gate_c.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import an_data as AD  # noqa: E402
from r4 import paths  # noqa: E402
from r4 import refs as RF  # noqa: E402
from r4 import slots_c as SC  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.splits import copy_group_map, folds_for  # noqa: E402

OUT = paths.RES / "gate_c"


def _overlaps(bench, held_idx, an) -> dict:
    """Independent recomputation of the four exclusion clauses: overlaps of the kept An set with held cells."""
    f = bench.frame
    cg = copy_group_map()
    pubs = {str(f.publication_id.iat[i]) for i in held_idx}
    links = AD.an_links()
    chems = {str(f.chemotype.iat[i]) for i in held_idx}
    return {"publication": int(an.publication.isin(pubs).sum()),
            "copy_group": int(an.copy_group.isin({cg.get(p, "own:" + p) for p in pubs}).sum()),
            "extractant": int(an.extractant.isin({AD._canon(f.extractant.iat[i]) for i in held_idx}).sum()),
            "linked_chemotype": int(sum(bool(links[e] & chems) for e in an.extractant))}


def counts(bench) -> int:
    rows, iso = [], []
    for d in ("BP", "BPG"):
        for fo in folds_for(bench.frame, d, seeds=list(paths.DEV_SEEDS)):
            rec = C.load(d, fo.seed, fo.fold)
            an = AD.fold_an(bench, rec)
            train = [int(c) for c in rec["outer"]["train_index"]]
            held = sorted(set(range(len(bench.frame))) - set(train))
            iso.append({"design": d, "seed": fo.seed, "fold": fo.fold, "block": "outer", "kept_rows": len(an),
                        **_overlaps(bench, held, an)})
            for j, blk in enumerate(rec["inner"]):
                itr = set(map(int, blk["train_index"]))
                an_j = AD.inner_an(bench, rec, blk, an)
                iso.append({"design": d, "seed": fo.seed, "fold": fo.fold, "block": f"inner{j}", "kept_rows": len(an_j),
                            **_overlaps(bench, sorted(set(held) | (set(train) - itr)), an_j),
                            "subset_of_outer": bool(set(an_j.rid) <= set(an.rid))})
            ok = [c for c in train if np.isfinite(bench.Y[c]).any()]
            _, _, _, e15 = SC._matched_rows(bench, set(train), an, SC.MATCH_SEED)
            rows.append({"design": d, "seed": fo.seed, "fold": fo.fold, **AD.counts(an),
                         "am_share": float((an.metal_symbol == "Am").mean()),
                         "s24_changed_fraction": SC.s24_changed_fraction(an),
                         "train_ln_extractants": int(bench.frame.extractant.iloc[ok].nunique()),
                         "s15_distinct_extractants": len(set(e15))})
    t = pd.DataFrame(rows)
    summ = {k: {"min": float(t[k].min()), "median": float(t[k].median()), "max": float(t[k].max())}
            for k in ("rows", "publications", "copy_groups", "extractants", "am_share", "s24_changed_fraction",
                      "train_ln_extractants", "s15_distinct_extractants")}
    res = {"all_records": AD.counts(AD.an3_records()), "fold_fits": len(rows), "summary": summ,
           "count_gate_passes": bool(t.copy_groups.min() >= 5 and t.extractants.min() >= 10 and t.rows.min() >= 100),
           "S24_count_gate_passes": bool(t.s24_changed_fraction.min() >= 0.10), "folds": rows}
    (paths.RES / "blockC_counts.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    it = pd.DataFrame(iso)
    (paths.RES / "leakage").mkdir(parents=True, exist_ok=True)
    it.to_csv(paths.RES / "leakage" / "isolation_an.csv", index=False, lineterminator="\n")
    viol = int(it[["publication", "copy_group", "extractant", "linked_chemotype"]].to_numpy().sum())
    sub = bool(it.subset_of_outer.dropna().astype(bool).all())
    print({k: v for k, v in res.items() if k != "folds"}, "isolation violations:", viol, "inner subsets ok:", sub,
          "checks:", len(it))
    return 0 if viol == 0 and sub else 1


def _seal() -> str:
    """The gate counts only under the verified seal (as r4_run.seal_tag): shards and gate_c.json carry its key."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import r4_seal
    ok, msg, seal = r4_seal.verify()
    if not ok:
        raise SystemExit(f"seal does not verify: {msg}")
    return seal[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--combine", type=int, default=0)
    ap.add_argument("--counts", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.counts:
        return counts(D.load_bench())
    tag = _seal()
    if a.combine:
        rows = []
        for i in range(a.combine):
            rows += json.loads((OUT / f"shard_{i}of{a.combine}.json").read_text())
        keys = {(r["design"], r["seed"], r["fold"]) for r in rows}
        if {r.get("seal") for r in rows} != {tag} or len(keys) != 50 or len(rows) != 50:
            raise SystemExit(f"refusing to combine: shards must cover 50 distinct fold-fits under seal {tag}")
        wins = sum(r["win"] for r in rows)
        cnt = {k: min(r["an_counts"][k] for r in rows) for k in ("rows", "publications", "copy_groups", "extractants")}
        res = {"seal": tag, "fold_fits": len(rows), "S13_beats_S11_and_B2_inner": wins, "inner_gate_passes": wins >= 30,
               "S13_beats_S11_inner_only": sum(r["win_vs_S11"] for r in rows),
               "min_counts_after_exclusions": cnt,
               "count_gate_passes": cnt["copy_groups"] >= 5 and cnt["extractants"] >= 10 and cnt["rows"] >= 100,
               "S11_best_network": pd.Series([r["S11_best_network"] for r in rows]).value_counts().to_dict(),
               "folds": rows}
        res["G_transfer_passes"] = bool(res["inner_gate_passes"] and res["count_gate_passes"])
        res["S24_min_changed_fraction"] = min(r["s24_changed_fraction"] for r in rows)
        res["S24_count_gate_passes"] = bool(res["S24_min_changed_fraction"] >= 0.10)
        (paths.RES / "gate_c.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
        print({k: v for k, v in res.items() if k != "folds"})
        return 0
    i, n = map(int, a.shard.split("/"))
    bench = D.load_bench()
    jobs = [f for d in ("BP", "BPG") for f in folds_for(bench.frame, d, seeds=list(paths.DEV_SEEDS))][i::n]
    rows, t0 = [], time.time()
    for f in jobs:
        rec = C.load(f.design, f.seed, f.fold)
        b1 = RF.fit_reference(bench, rec, bench.Y, "B1")
        b2 = RF.fit_reference(bench, rec, bench.Y, "B2")
        fit = SC.fit_slot(bench, rec, bench.Y, "S13", b1, gate_only=True)
        sc11, sx = fit["inner_scores"]["S11"], fit["inner_scores"]["S13"]
        best11 = min(sc11.values())
        best13 = min(v for k, v in sx.items() if k not in ("zero", "0.0"))
        best_b2 = min(b2["inner_scores"].values())
        an = AD.fold_an(bench, rec)
        rows.append({"seal": tag, "design": f.design, "seed": f.seed, "fold": f.fold, "S11_best": best11,
                     "S13_best_positive": best13,
                     "B2_best": best_b2, "win_vs_S11": bool(best13 < best11),
                     "win": bool(best13 < best11 and best13 < best_b2), "an_counts": AD.counts(an),
                     "s24_changed_fraction": SC.s24_changed_fraction(an), "S11_best_network": SC.net_key(*fit["best_net"]),
                     "inner_S11": sc11, "inner_S13": sx})
        print(f.design, f.seed, f.fold, "win" if rows[-1]["win"] else "no", f"{time.time() - t0:.0f}s", flush=True)
    (OUT / f"shard_{i}of{n}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
