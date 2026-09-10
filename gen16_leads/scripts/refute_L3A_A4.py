"""REFUTER lens A, stage 4: publication decomposition of the L3a saving, the within-publication
detail, the decision-rule recomputation and BH over L3's registered family.  All offline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refute_L3A_A import OUT, DESIGNS, STRONG, MIN_EXT, my_tasks, e_random, e_model, seed_macro  # noqa: E402
from refute_L3A_A2 import perm_null  # noqa: E402
from refute_L3A_A3 import within_pub_tasks, saved_from_tasks  # noqa: E402


def main() -> int:
    meta = pd.read_parquet(OUT / "cellmeta.parquet")
    res = {}
    rng = np.random.default_rng(20260912)
    for d in DESIGNS:
        tab = pd.read_parquet(OUT / f"pairs_don_{d}.parquet")
        tasks, _ = my_tasks(tab)
        seeds = np.array([t["seed"] for t in tasks])
        calls = [np.sign(t["pred"]).astype(int) for t in tasks]
        # publication label per extractant (modal publication of its cells)
        pub_of = (meta.groupby("extractant")["publication_id"]
                  .agg(lambda s: s.astype(str).value_counts().index[0]).to_dict())
        n_pub_of = meta.groupby("extractant")["publication_id"].nunique().to_dict()
        strat_pub = [np.array([pub_of[e] for e in t["ext"]]) for t in tasks]
        base = saved_from_tasks(tasks, seeds, calls)[0]
        n_p = perm_null(tasks, seeds, calls, 500, np.random.default_rng(int(rng.integers(1e9))),
                        strata=strat_pub)
        R = {"saved": base,
             "perm_publication_stratified": {"mean": float(n_p.mean()), "sd": float(n_p.std(ddof=1)),
                                             "max": float(n_p.max()),
                                             "p": float((n_p >= base).mean()),
                                             "real_minus_shuffled": base - float(n_p.mean())},
             "n_ext_in_multiple_pubs": int(sum(1 for e in pub_of if n_pub_of[e] > 1)),
             "n_publications": int(len(set(pub_of.values())))}
        # within-publication detail
        wp, wp_drop = within_pub_tasks(tab, meta)
        wseeds = np.array([t["seed"] for t in wp])
        wcalls = [np.sign(t["pred"]).astype(int) for t in wp]
        R["within_pub"] = {}
        s, er = saved_from_tasks(wp, wseeds, wcalls)
        R["within_pub"]["saved"] = s
        R["within_pub"]["e_random"] = er
        R["within_pub"]["n_tasks"] = len(wp)
        R["within_pub"]["dropped_lt5"] = wp_drop
        nonconst = np.array([len(set(c.tolist())) > 1 for c in wcalls])
        R["within_pub"]["n_nonconstant_call"] = int(nonconst.sum())
        if nonconst.any():
            sub = [t for t, k in zip(wp, nonconst) if k]
            sc = [c for c, k in zip(wcalls, nonconst) if k]
            ss, ser = saved_from_tasks(sub, np.array([t["seed"] for t in sub]), sc)
            R["within_pub"]["saved_nonconstant_only"] = ss
            R["within_pub"]["e_random_nonconstant_only"] = ser
            npw = perm_null(sub, np.array([t["seed"] for t in sub]), sc, 500,
                            np.random.default_rng(int(rng.integers(1e9))))
            R["within_pub"]["perm_mean_nonconstant"] = float(npw.mean())
            R["within_pub"]["perm_p_nonconstant"] = float((npw >= ss).mean())
        # pooled saving on tasks restricted to candidate sets that are *direction-homogeneous*
        # in truth (no partition possible): a sanity descriptor
        homo = []
        for t in tasks:
            succ_h = ((np.sign(t["obs"]) == -1) & (np.abs(t["obs"]) >= STRONG)).sum()
            succ_l = ((np.sign(t["obs"]) == 1) & (np.abs(t["obs"]) >= STRONG)).sum()
            homo.append(min(succ_h, succ_l) == 0)
        R["frac_pooled_tasks_one_direction_only"] = float(np.mean(homo))
        res[d] = R
        print(f"[{d}] saved {base:+.4f} | pub-stratified perm mean {R['perm_publication_stratified']['mean']:+.4f} "
              f"p {R['perm_publication_stratified']['p']:.4f} real-shuf "
              f"{R['perm_publication_stratified']['real_minus_shuffled']:+.4f} | within-pub {s:+.5f} "
              f"(nonconst {int(nonconst.sum())}/{len(wp)} -> "
              f"{R['within_pub'].get('saved_nonconstant_only', float('nan')):+.4f} of "
              f"{R['within_pub'].get('e_random_nonconstant_only', float('nan')):.3f}, perm mean "
              f"{R['within_pub'].get('perm_mean_nonconstant', float('nan')):+.4f} "
              f"p {R['within_pub'].get('perm_p_nonconstant', float('nan')):.3f})", flush=True)

    # --- decision rule recomputed from the lead's own CSVs + BH over L3's registered family
    ca = pd.read_csv(bootstrap_results() / "L3" / "l3a_contrasts.csv")
    cc = pd.read_csv(bootstrap_results() / "L3" / "l3c_contrasts.csv")
    reg = pd.concat([ca[ca.family == "registered"], cc[cc.family == "registered"]], ignore_index=True)
    p = reg["p_registered"].to_numpy(float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        adj[i] = prev
    reg["p_BH_registered_L3"] = adj
    reg["recomputed_pass"] = (reg["point"] > 0) & (reg["p_registered"] < 0.05) & (reg["ci95_low"] > 0)
    keep = ["design", "comparison", "family", "point", "ci95_low", "ci95_high", "p_registered",
            "p_BH_registered_L3", "passes_registered", "recomputed_pass"]
    reg[keep].to_csv(OUT / "registered_family_BH.csv", index=False)
    print(reg[keep].to_string())
    (OUT / "checks_stage4.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return 0


def bootstrap_results():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gen16 import bootstrap
    return bootstrap.RESULTS


if __name__ == "__main__":
    sys.exit(main())
