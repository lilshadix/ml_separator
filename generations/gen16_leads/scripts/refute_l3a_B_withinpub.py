"""Fifth stage of the lens-B refutation of CLAIM L3A.

The lead's own limitation 1 says a within-publication L3a "is the first thing to ask for" and was
not run.  This runs it, plus the within-chemotype version, plus a per-publication leave-one-out
with a full re-run of the headline, the diglycolamide-removed headline re-run, and the coherent
null under a second seed.  Everything is built from the lead's cached pair tables and the lead's
own estimator (`gen16.l3_decision`); nothing is re-implemented.

Usage (repo root):
  PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe \
      generations/gen16_leads/scripts/refute_l3a_B_withinpub.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen16 import bootstrap                      # noqa: E402
from gen16 import l3_decision as L               # noqa: E402
from refute_l3a_B import Board, OUT, ext_meta, block_boot, bh   # noqa: E402
from refute_l3a_B_extra import coherent_null     # noqa: E402

ALT_SEED = 20260912          # a third resampling seed, different from the lead's and from stage 2


def grouped_tasks(tab, design, groupcol, ext, chem_of, min_ext):
    """Pooled tasks built *inside* one value of ``groupcol`` (publication or chemotype)."""
    pos = {e: i for i, e in enumerate(ext)}
    keys = ["split_seed", groupcol, "A", "B"]
    g = tab.groupby(keys + ["extractant"], sort=False)[["y", "G14"]].median().reset_index()
    seeds, As, Bs, idx, obs, pred, groups = [], [], [], [], [], [], []
    n_before = n_drop = 0
    for k, blk in g.groupby(keys, sort=False):
        n_before += 1
        if len(blk) < min_ext:
            n_drop += 1
            continue
        seeds.append(int(k[0])); groups.append(k[1]); As.append(k[2]); Bs.append(k[3])
        idx.append(np.array([pos[e] for e in blk["extractant"]], dtype=int))
        obs.append(blk["y"].to_numpy(float))
        pred.append(blk["G14"].to_numpy(float))
    T = L.Tasks(design=design, arms=["G14"], ext=list(ext), chem_of=np.asarray(chem_of),
                seeds=np.array(seeds, dtype=int), A=np.array(As), B=np.array(Bs),
                dZ=np.zeros(len(seeds), dtype=int), idx=idx, obs=obs, pred={"G14": pred})
    return T, np.array(groups), n_before, n_drop


def evaluate(T, chem, reps_seed=L.BOOT_SEED, perm_seed=ALT_SEED, do_ci=True):
    if T.n == 0:
        return dict(point=np.nan, n_tasks=0)
    calls = [L.calls_of(p) for p in T.pred["G14"]]
    mats = L.l3a_matrices(T, calls)
    one = np.ones((1, len(T.ext)))
    r = L.l3a_saved_weighted(one, mats)
    pt = float(np.atleast_1d(L.seed_macro(r["saved"], T.seeds))[0])
    out = dict(point=pt,
               e_random=float(np.atleast_1d(L.seed_macro(r["e_random"], T.seeds))[0]),
               e_model=float(np.atleast_1d(L.seed_macro(r["e_model"], T.seeds))[0]),
               n_tasks=int(np.isfinite(r["saved"][0]).sum()),
               median_candidates=float(np.median([len(i) for i in T.idx])) if T.n else np.nan,
               frac_tasks_constant_call=float(np.mean([np.ptp(c) == 0 for c in calls])))
    out["saved_frac"] = out["point"] / out["e_random"] if out["e_random"] else np.nan
    if do_ci:
        W = block_boot(chem, reps=L.N_BOOT, seed=reps_seed)
        dr = L.seed_macro(L.l3a_saved_weighted(W, mats)["saved"], T.seeds)
        out.update(ci_low=L.pct(dr, 0.025), ci_high=L.pct(dr, 0.975),
                   p_block=L.two_sided_p(dr))
        perm = L.seed_macro(L.l3a_permutation(T, calls, seed=perm_seed)["saved"], T.seeds)
        out["p_perm"] = float((perm >= pt).mean())
        out["null_sd"] = float(np.std(perm))
    return out


def main() -> int:
    t0 = time.time()
    em = ext_meta()
    meta = pd.read_parquet(OUT / "cell_meta.parquet")[["cell_id", "publication_id"]]
    rows, notes = [], {}

    for d in L.DESIGNS:
        bd = Board(d, em)
        chem = np.asarray(bd.chem)
        ext = bd.T.ext
        tab = pd.read_parquet(OUT / f"tab_{d}.parquet").merge(meta, on="cell_id", how="left")
        assert tab.publication_id.notna().all()

        base = bd.point()
        rows.append(dict(design=d, check="headline_reproduced", **evaluate(bd.T, chem)))

        # ---- within one publication (the lead's limitation 1), and within one chemotype
        for gcol, tag in (("publication_id", "within_publication"), ("chemotype", "within_chemotype")):
            for mn in (5, 4, 3):
                old = L.MIN_EXT
                L.MIN_EXT = mn
                T, groups, n_before, n_drop = grouped_tasks(tab, d, gcol, ext, chem, mn)
                res = evaluate(T, chem, do_ci=(T.n > 0))
                L.MIN_EXT = old
                rows.append(dict(design=d, check=f"{tag}_min{mn}", n_groups=int(len(set(groups))),
                                 n_task_slots_before=n_before, n_dropped=n_drop,
                                 baseline_pooled=base, **res))

        # ---- per-publication leave-one-out, headline re-run (not only the min/max sweep)
        allp = sorted(set(meta.publication_id))
        lo = []
        for p in allp:
            keep = np.array([p not in bd.pubs[i] for i in range(bd.n_ext)])
            if keep.sum() < L.MIN_EXT:
                continue
            v = L.l3a_saved_weighted(keep.astype(float)[None, :], bd.mats)["saved"][0]
            lo.append((p, float(np.atleast_1d(L.seed_macro(v, bd.T.seeds))[0]), int(keep.sum())))
        vals = np.array([x[1] for x in lo])
        j = int(np.nanargmin(vals))
        rows.append(dict(design=d, check="lopo_publication_rerun", point=base,
                         loco_min=float(np.nanmin(vals)), loco_max=float(np.nanmax(vals)),
                         worst_block=lo[j][0], n_groups=len(lo),
                         n_sign_flips=int((vals <= 0).sum())))
        # the interval on the worst leave-one-publication-out re-run
        keep_w = np.array([lo[j][0] not in bd.pubs[i] for i in range(bd.n_ext)])
        dr = L.seed_macro(L.l3a_saved_weighted(block_boot(chem, keep_w), bd.mats)["saved"], bd.T.seeds)
        rows.append(dict(design=d, check="lopo_worst_publication_ci", point=lo[j][1],
                         ci_low=L.pct(dr, 0.025), ci_high=L.pct(dr, 0.975),
                         p_block=L.two_sided_p(dr), worst_block=lo[j][0], n_groups=1))

        # ---- diglycolamides removed, headline re-run with a third resampling seed
        keep = chem != "sc009"
        v = L.l3a_saved_weighted(keep.astype(float)[None, :], bd.mats)["saved"][0]
        pt = float(np.atleast_1d(L.seed_macro(v, bd.T.seeds))[0])
        dr = L.seed_macro(L.l3a_saved_weighted(block_boot(chem, keep, seed=ALT_SEED),
                                               bd.mats)["saved"], bd.T.seeds)
        rows.append(dict(design=d, check="drop_sc009_altseed_ci", point=pt,
                         ci_low=L.pct(dr, 0.025), ci_high=L.pct(dr, 0.975),
                         p_block=L.two_sided_p(dr), n_ext=int(keep.sum()),
                         n_groups=len(set(chem[keep]))))
        # sc009 removed AND per-publication LOO on top of it
        lo2 = []
        for p in allp:
            k2 = keep & np.array([p not in bd.pubs[i] for i in range(bd.n_ext)])
            if k2.sum() < L.MIN_EXT:
                continue
            v2 = L.l3a_saved_weighted(k2.astype(float)[None, :], bd.mats)["saved"][0]
            lo2.append(float(np.atleast_1d(L.seed_macro(v2, bd.T.seeds))[0]))
        rows.append(dict(design=d, check="drop_sc009_then_lopo", point=pt,
                         loco_min=float(np.nanmin(lo2)), loco_max=float(np.nanmax(lo2)),
                         n_groups=len(lo2), n_sign_flips=int((np.array(lo2) <= 0).sum())))

        # ---- the coherent (per-extractant, width-matched) null under a second seed
        cn = coherent_null(bd, seed=ALT_SEED)
        rows.append(dict(design=d, check="coherent_null_altseed", point=base,
                         null_mean=float(np.mean(cn)), null_sd=float(np.std(cn)),
                         p_perm=float((cn >= base).mean())))
        # and the same null restricted to the diglycolamide-free headline
        print(f"[{d}] done t={time.time()-t0:.0f}s", flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(OUT / "refute_l3a_B_withinpub.csv", index=False)

    # ---- comparison family size and BH, recomputed on today's contrast files
    fam = []
    for f in sorted(bootstrap.RESULTS.glob("*/*contrasts*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "family" in df.columns:
            df["__file"] = f.name
            fam.append(df)
    F = pd.concat(fam, ignore_index=True)
    notes["contrast_rows_total"] = int(len(F))
    notes["contrast_rows_by_family"] = {k: int(v) for k, v in F.family.value_counts().items()}
    notes["contrast_files"] = sorted(F.__file.unique().tolist())
    for famname in ("registered", "exploratory"):
        sub = F[F.family == famname].copy()
        for pcol in ("p_registered", "p_two_sided"):
            if pcol in sub.columns:
                sub[f"bh_{pcol}"] = bh(sub[pcol].to_numpy())
        notes[f"bh_{famname}_n"] = int(len(sub))
        l3a = sub[sub.comparison.astype(str).str.fullmatch("G14_saved_vs_0")]
        if len(l3a):
            notes[f"bh_{famname}_L3A"] = {
                r.design: {"p_registered": float(r.p_registered),
                           "bh_p_registered": float(r.bh_p_registered),
                           "p_two_sided": float(r.p_two_sided),
                           "bh_p_two_sided": float(r.bh_p_two_sided)}
                for r in l3a.itertuples()}
    (OUT / "refute_l3a_B_withinpub_notes.json").write_text(
        json.dumps(notes, indent=2, default=str), encoding="utf-8")
    with pd.option_context("display.width", 260, "display.max_columns", 40,
                           "display.max_rows", 400):
        print(D.to_string())
    print(json.dumps(notes, indent=2, default=str))
    print(f"total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
