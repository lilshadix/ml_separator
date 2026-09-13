"""REFUTER lens A, stage 3: the cheapest *sensible* competitor (gen6 DONORS13 direction call),
the within-publication candidate set, and a determinism re-check in a fresh process.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refute_L3A_A import OUT, DESIGNS, STRONG, MIN_EXT, my_tasks, e_random, e_model, seed_macro  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa: E402
from gen15 import valuebench as V  # noqa: E402
from gen15 import arms as A  # noqa: E402
from gen14.dirbench import load  # noqa: E402


def g14_donors(ctx):
    """G14 with gen6's 13-column donor census in place of TOPO39 -- everything else identical."""
    return A._curve(A._logistic_sign(ctx, "DONORS13") * ctx.train_mean_magnitude(),
                    ctx.train_mean_curvature(), len(ctx.test))


def saved_from_tasks(tasks, seeds, calls):
    sv, er = [], []
    for i, t in enumerate(tasks):
        obs, call = t["obs"], calls[i]
        N = len(obs)
        s_, r_ = 0.0, 0.0
        for d in (-1, 1):
            succ = (np.sign(obs) == d) & (np.abs(obs) >= STRONG)
            kept = call != -d
            K, Nk, Kk = float(succ.sum()), float(kept.sum()), float((kept & succ).sum())
            r = e_random(N, K); m = float(e_model(N, K, Nk, Kk))
            s_ += 0.5 * (r - m); r_ += 0.5 * r
        sv.append(s_); er.append(r_)
    return seed_macro(np.array(sv), seeds), seed_macro(np.array(er), seeds)


def within_pub_tasks(tab, meta):
    """Candidate sets restricted to one publication: (seed, pair, publication)."""
    t = tab.merge(meta[["cell_id", "publication_id"]], on="cell_id", how="left", validate="many_to_one")
    g = (t.groupby(["split_seed", "publication_id", "A", "B", "extractant"], sort=True)[["y", "G14", "DON"]]
         .median().reset_index())
    out, drop = [], 0
    for (seed, pub, a, b), blk in g.groupby(["split_seed", "publication_id", "A", "B"], sort=True):
        if len(blk) < MIN_EXT:
            drop += 1
            continue
        out.append({"seed": int(seed), "A": a, "B": b,
                    "ext": blk["extractant"].to_numpy(),
                    "obs": blk["y"].to_numpy(float),
                    "pred": blk["G14"].to_numpy(float),
                    "don": blk["DON"].to_numpy(float)})
    return out, drop


def main() -> int:
    bench = load()
    meta = pd.read_parquet(OUT / "cellmeta.parquet")
    res = {}
    for d in DESIGNS:
        p = OUT / f"pairs_don_{d}.parquet"
        if not p.exists():
            t0 = time.time()
            tab = V.run_arms(bench, {"G14": A.g14, "DON": g14_donors}, d, verbose=False)
            tab.to_parquet(p)
            print(f"[don] {d}: {len(tab)} rows {time.time()-t0:.0f}s", flush=True)
        tab = pd.read_parquet(p)
        # determinism: this fresh process must reproduce the stage-1 cache exactly
        ref = pd.read_parquet(OUT / f"pairs_{d}.parquet")
        m = tab.merge(ref[["split_seed", "cell_id", "A", "B", "G14"]],
                      on=["split_seed", "cell_id", "A", "B"], suffixes=("", "_ref"), validate="one_to_one")
        dmax = float(np.abs(m["G14"] - m["G14_ref"]).max())

        tasks, _ = my_tasks(tab)
        seeds = np.array([t["seed"] for t in tasks])
        # DON calls need their own median collapse
        g = (tab.groupby(["split_seed", "A", "B", "extractant"], sort=True)[["y", "G14", "DON"]]
             .median().reset_index())
        don_calls, g14_calls = [], []
        for (seed, a, b), blk in g.groupby(["split_seed", "A", "B"], sort=True):
            if len(blk) < MIN_EXT:
                continue
            blk = blk.sort_values("extractant")
            don_calls.append(np.sign(blk["DON"].to_numpy(float)).astype(int))
            g14_calls.append(np.sign(blk["G14"].to_numpy(float)).astype(int))
        s_g14, er = saved_from_tasks(tasks, seeds, g14_calls)
        s_don, _ = saved_from_tasks(tasks, seeds, don_calls)
        agree = float(np.mean(np.concatenate([(a == b) for a, b in zip(g14_calls, don_calls)])))

        # within-publication
        wp, wp_drop = within_pub_tasks(tab, meta)
        if wp:
            wseeds = np.array([t["seed"] for t in wp])
            s_wp, er_wp = saved_from_tasks(wp, wseeds, [np.sign(t["pred"]).astype(int) for t in wp])
            s_wp_don, _ = saved_from_tasks(wp, wseeds, [np.sign(t["don"]).astype(int) for t in wp])
            const = float(np.mean([len(set(np.sign(t["pred"]).astype(int))) == 1 for t in wp]))
        else:
            s_wp = er_wp = s_wp_don = const = float("nan")

        res[d] = {"max_abs_diff_G14_vs_stage1_cache": dmax,
                  "saved_G14": s_g14, "saved_DONORS13": s_don, "e_random": er,
                  "call_agreement_G14_vs_DONORS13": agree,
                  "increment_over_cheapest_sensible": s_g14 - s_don,
                  "within_pub_n_tasks": len(wp), "within_pub_dropped_lt5": wp_drop,
                  "within_pub_saved_G14": s_wp, "within_pub_e_random": er_wp,
                  "within_pub_saved_DONORS13": s_wp_don,
                  "within_pub_frac_tasks_constant_call": const}
        print(f"[{d}] G14 {s_g14:+.4f} | DONORS13 {s_don:+.4f} | increment {s_g14-s_don:+.4f} "
              f"| call agree {agree:.3f} | within-pub tasks {len(wp)} saved {s_wp:+.4f} "
              f"of E_rand {er_wp:.3f} (const-call {const:.3f}) | cache dmax {dmax:.2e}", flush=True)
    (OUT / "checks_stage3.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
