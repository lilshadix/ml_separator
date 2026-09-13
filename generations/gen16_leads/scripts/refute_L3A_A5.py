"""REFUTER lens A, stage 5: power diagnostics for the publication-stratified null, and the
composition + chemotype-blocked interval of the within-publication L3a.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refute_L3A_A import OUT, DESIGNS, STRONG, MIN_EXT, my_tasks, e_random, e_model, seed_macro  # noqa: E402
from refute_L3A_A3 import within_pub_tasks, saved_from_tasks  # noqa: E402


def main() -> int:
    meta = pd.read_parquet(OUT / "cellmeta.parquet")
    chem_of = meta.drop_duplicates("extractant").set_index("extractant")["chemotype"].astype(str).to_dict()
    pub_of = (meta.groupby("extractant")["publication_id"]
              .agg(lambda s: s.astype(str).value_counts().index[0]).to_dict())
    out = {}
    for d in DESIGNS:
        tab = pd.read_parquet(OUT / f"pairs_don_{d}.parquet")
        tasks, _ = my_tasks(tab)
        seeds = np.array([t["seed"] for t in tasks])
        calls = [np.sign(t["pred"]).astype(int) for t in tasks]
        # --- power of the publication-stratified null
        bs, movable, singleton = [], [], []
        for t in tasks:
            pubs = np.array([pub_of[e] for e in t["ext"]])
            _, cnt = np.unique(pubs, return_counts=True)
            bs.append(cnt)
            movable.append(cnt[cnt >= 2].sum() / cnt.sum())
            singleton.append((cnt == 1).sum() / len(cnt))
        allb = np.concatenate(bs)
        # how many candidates sit in a publication block whose calls are not all equal?
        eff = []
        for i, t in enumerate(tasks):
            pubs = np.array([pub_of[e] for e in t["ext"]])
            k = 0
            for p in np.unique(pubs):
                idx = pubs == p
                if idx.sum() >= 2 and len(set(calls[i][idx].tolist())) > 1:
                    k += idx.sum()
            eff.append(k / len(t["ext"]))
        # --- within-publication composition and chemotype-blocked interval
        wp, wp_drop = within_pub_tasks(tab, meta)
        wseeds = np.array([t["seed"] for t in wp])
        wcalls = [np.sign(t["pred"]).astype(int) for t in wp]
        base_wp = saved_from_tasks(wp, wseeds, wcalls)
        pubs_used = sorted({tuple(sorted({pub_of[e] for e in t["ext"]}))[0] for t in wp})
        chems = sorted({chem_of[e] for t in wp for e in t["ext"]})
        # chemotype-blocked bootstrap over the within-pub tasks (2000 reps, my seed)
        rng = np.random.default_rng(20260913)
        blocks = sorted(set(chem_of.values()))
        picks = rng.integers(0, len(blocks), size=(2000, len(blocks)))
        cnt = np.zeros((2000, len(blocks)))
        for j in range(len(blocks)):
            cnt[:, j] = (picks == j).sum(1)
        bidx = {b: j for j, b in enumerate(blocks)}
        draws = np.full((2000, len(wp)), np.nan)
        for i, t in enumerate(wp):
            cols = np.array([bidx[chem_of[e]] for e in t["ext"]])
            W = cnt[:, cols]                       # (reps, n_cand)
            obs, call = t["obs"], wcalls[i]
            N = W.sum(1)
            tot = np.zeros(2000)
            for dd in (-1, 1):
                succ = (np.sign(obs) == dd) & (np.abs(obs) >= STRONG)
                kept = call != -dd
                K = W @ succ.astype(float)
                Nk = W @ kept.astype(float)
                Kk = W @ (kept & succ).astype(float)
                tot += 0.5 * (e_random(N, K) - e_model(N, K, Nk, Kk))
            draws[:, i] = np.where(N >= MIN_EXT, tot, np.nan)
        bm = np.array([seed_macro(draws[r], wseeds) for r in range(2000)])
        out[d] = {
            "pub_block_mean_size": float(allb.mean()), "pub_block_median_size": float(np.median(allb)),
            "frac_candidates_in_blocks_ge2": float(np.mean(movable)),
            "frac_candidates_in_movable_blocks": float(np.mean(eff)),
            "n_publications_per_task_mean": float(np.mean([len(b) for b in bs])),
            "within_pub_saved": base_wp[0], "within_pub_e_random": base_wp[1],
            "within_pub_n_tasks": len(wp), "within_pub_n_publications": len(pubs_used),
            "within_pub_n_chemotypes": len(chems),
            "within_pub_ci95": [float(np.nanquantile(bm, .025)), float(np.nanquantile(bm, .975))],
            "within_pub_p_two_sided": float(min(1.0, 2 * min((bm <= 0).mean(), (bm >= 0).mean()))),
            "within_pub_per_seed": {int(s): saved_from_tasks(
                [t for t in wp if t["seed"] == s], np.array([s] * sum(t["seed"] == s for t in wp)),
                [c for t, c in zip(wp, wcalls) if t["seed"] == s])[0] for s in np.unique(wseeds)},
        }
        print(f"[{d}] pub blocks: mean size {allb.mean():.2f}, {np.mean(movable):.3f} of candidates "
              f"in blocks>=2, {np.mean(eff):.3f} in blocks whose calls differ; pubs/task "
              f"{np.mean([len(b) for b in bs]):.1f} || within-pub saved {base_wp[0]:+.5f} "
              f"CI [{out[d]['within_pub_ci95'][0]:+.4f}, {out[d]['within_pub_ci95'][1]:+.4f}] "
              f"p {out[d]['within_pub_p_two_sided']:.3f} over {len(wp)} tasks / "
              f"{len(pubs_used)} pubs / {len(chems)} chemotypes", flush=True)
    (OUT / "checks_stage5.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
