#!/usr/bin/env python
"""Selection optimism and multiplicity for round 3 (from the stored row-level predictions only; no refit).

Pool: every V1 / C3 / k = 3 configuration scored on outer dev (B0's PWL_R / PWL_Z / HBK@ML, R01-R16, R19, R21-R23)
on identical rows.  One chemotype-blocked bootstrap (gen20k.inference draws, 2000 replicates, shared across all
configurations and seeds) gives, per design:
  * the evaluation-noise part of the winner's curse: E*[max_j theta*_j] - max_j theta_j, P*(argmax = R23 / R14), and
    the mean of theta*_{j*b} - theta_{j*b} for the bootstrap winner j*b (R2 level and dR2 vs PWL_R on the same reads);
  * multiplicity: R23's and R14's dR2 lower bounds at 95 %, 98.75 % and Bonferroni-over-the-pool, and a max-T
    simultaneous one-sided 95 % lower bound over the pool;
  * the 2 x 2 (policy x Gd parameter) interaction and the R23 - R14, R23 - R16, R09 - B0, R14 - B0 contrasts;
  * blocking robustness: the same contrasts and the R2 level with clusters = connected components of
    chemotype U publication instead of chemotype.
The adaptive part of the selection (R23 designed after R15 / R16 / R21 were seen on the same chemotypes) is NOT
captured by any resampling of these rows.

    .venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_selection_optimism.py
writes results/selection_optimism.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3 import paths  # noqa: E402
from gen20k import inference as INF  # noqa: E402

KEY = ["split_seed", "cell_id", "A", "B"]
POOL = {  # name: (prediction file, column)
    "B0:PWL_R": ("B0", "PWL_R"), "B0:PWL_Z": ("B0", "PWL_Z"), "B0:HBK@ML": ("B0", "HBK@ML"),
    "R01": ("R01", "PCHIP_R"), "R02": ("R02", "COR_G"), "R03": ("R03", "COR_GP"), "R04": ("R04", "HBK_TASK_K"),
    "R05": ("R05", "HBK_TASK_KC"), "R06": ("R06", "COR_TASKFS"), "R07": ("R07", "COR_GC"), "R08": ("R08", "COR_GDC"),
    "R09": ("R09", "PWL_R"), "R10": ("R09", "PWL_Z"), "R11": ("R09", "HBK@ML"), "R12": ("R12", "COR[GD]"),
    "R13": ("R13", "STACK"), "R14": ("R14", "COR[Dnc]"), "R15": ("R15", "COR[G]"), "R16": ("R16", "COR[GDnc]"),
    "R19": ("R19", "ROBUST_COR"), "R21": ("R21", "COR_SHUFFLED"), "R22": ("R22", "STACK4"), "R23": ("R23", "COR[Dnc]"),
}
REPS = 2000


def _wide(design: str) -> pd.DataFrame:
    cache = {}

    def load(f):
        if f not in cache:
            t = pd.read_parquet(paths.PRED / f"{f}_{design}.parquet")
            cache[f] = t[(t.k == 3) & (t.k_max >= 3)].sort_values(KEY).reset_index(drop=True)
        return cache[f]
    ref = load("B0")
    w = ref[KEY + ["y", "chemotype", "publication"]].copy()
    for name, (f, col) in POOL.items():
        t = load(f)
        assert (t[KEY].values == ref[KEY].values).all() and np.allclose(t.y.values, ref.y.values)
        w[name] = t[col].values
        w[name + "|PWL"] = t["PWL_R"].values          # PWL_R on the configuration's own reads
    return w


def _draws(w: pd.DataFrame, cols, cluster: str):
    clusters = sorted(w[cluster].astype(str).unique())
    counts = INF._draw_counts(len(clusters), REPS, INF.SEED)
    ps = INF._per_seed_sums(w.assign(**{cluster: w[cluster].astype(str)}), cols, cluster, "split_seed", clusters)
    na = len(cols)
    pt, dr = [], []
    for S in ps.values():
        st = INF._stats_from_sums(S.sum(0, keepdims=True), na)
        pt.append(np.array([st[f"R2_{i}"][0] for i in range(na)]))
        d = INF._stats_from_sums(counts @ S, na)
        dr.append(np.stack([d[f"R2_{i}"] for i in range(na)], 1))
    return np.mean(pt, 0), np.mean(dr, 0), len(clusters)


def _components(w: pd.DataFrame) -> pd.Series:
    par = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for c, p in w[["chemotype", "publication"]].drop_duplicates().itertuples(index=False):
        par[find("c:" + str(c))] = find("p:" + str(p))
    return w.chemotype.map(lambda c: find("c:" + str(c)))


def analyse(design: str) -> dict:
    w = _wide(design)
    arms = list(POOL)
    cols = arms + [a + "|PWL" for a in arms]
    out = {}
    for cluster in ("chemotype", "chemotype_U_publication"):
        if cluster == "chemotype_U_publication":
            w[cluster] = _components(w)
        pt, dr, G = _draws(w, cols, cluster)
        n = len(arms)
        R2, R2d = pt[:n], dr[:, :n]
        D, Dd = pt[:n] - pt[n:], dr[:, :n] - dr[:, n:]
        i = {a: j for j, a in enumerate(arms)}
        res = {"n_clusters": G, "replicates": REPS}
        for nm, v, vd in (("R2_level", R2, R2d), ("dR2_vs_PWL_R_same_reads", D, Dd)):
            j = int(np.argmax(v))
            am = vd.argmax(1)
            res[nm] = {
                "argmax": arms[j], "max": float(v[j]),
                "E_boot_max_minus_max": float(vd.max(1).mean() - v[j]),
                "boot_winner_optimism": float((vd[np.arange(len(vd)), am] - v[am]).mean()),
                "P_argmax": {a: float((np.array(arms)[am] == a).mean()) for a in ("R23", "R14", "R16", "R12")},
            }
        sd = Dd.std(0, ddof=1)
        ok = sd > 0                                   # R09 (PWL_R on its own reads) has dR2 = 0 in every draw
        T = ((Dd[:, ok] - D[ok]) / sd[ok]).max(1)
        c95 = float(np.quantile(T, 0.95))
        # non-parametric min-P (Westfall-Young style) on the percentile scale: no symmetry assumption
        U = (np.argsort(np.argsort(Dd[:, ok], axis=0), axis=0) + 1) / len(Dd)
        alpha_sim = float(np.quantile(U.min(1), 0.05))
        res["lower_bounds_dR2"] = {
            a: {"point": float(D[i[a]]), "lb95": float(np.quantile(Dd[:, i[a]], 0.025)),
                "lb98.75": float(np.quantile(Dd[:, i[a]], 0.0125 / 2)),
                "lb_bonferroni_pool": float(np.quantile(Dd[:, i[a]], 0.05 / len(arms) / 2)),
                "lb_maxT_simultaneous_95": float(D[i[a]] - c95 * sd[i[a]]),
                "lb_minP_simultaneous_95": float(np.quantile(Dd[:, i[a]], alpha_sim)), "minP_alpha": alpha_sim}
            for a in ("R23", "R14", "R16", "R12", "R03")}
        inter = R2d[:, i["R23"]] - R2d[:, i["R14"]] - R2d[:, i["R09"]] + R2d[:, i["B0:PWL_R"]]
        res["interaction_R23-R14-R09+B0"] = {"point": float(R2[i["R23"]] - R2[i["R14"]] - R2[i["R09"]] + R2[i["B0:PWL_R"]]),
                                            "ci95": [float(x) for x in np.quantile(inter, [0.025, 0.975])]}
        res["contrasts"] = {}
        for a, b in (("R23", "R14"), ("R23", "R16"), ("R09", "B0:PWL_R"), ("R14", "B0:PWL_R"), ("R23", "B0:PWL_R"),
                     ("R23", "B0:HBK@ML"), ("R14", "B0:HBK@ML")):
            d = R2d[:, i[a]] - R2d[:, i[b]]
            res["contrasts"][f"{a} - {b}"] = {"point": float(R2[i[a]] - R2[i[b]]),
                                             "ci95": [float(x) for x in np.quantile(d, [0.025, 0.975])]}
        res["levels"] = {a: {"point": float(R2[i[a]]), "ci95": [float(x) for x in np.quantile(R2d[:, i[a]], [0.025, 0.975])]}
                         for a in ("R23", "R14", "B0:PWL_R", "R09")}
        C = np.corrcoef(Dd.T)
        res["corr_dR2_draws_with_R23"] = {a: float(C[i["R23"], i[a]]) for a in ("R14", "R16", "R12", "R09", "R08", "R03")}
        if cluster == "chemotype_U_publication":
            one = w[w.split_seed == w.split_seed.iloc[0]]
            sizes = one.groupby(cluster).size()
            multi = one.groupby("publication").chemotype.nunique()
            res["blocking"] = {"publications": int(multi.size), "publications_spanning_chemotypes": int((multi > 1).sum()),
                               "row_share_of_those": float(one.publication.isin(multi[multi > 1].index).mean()),
                               "kish_effective_clusters": float(sizes.sum() ** 2 / (sizes ** 2).sum())}
        out[cluster] = res
    return out


def main() -> int:
    res = {"pool": list(POOL), "note": __doc__.split("\n\n")[0]}
    for design in ("BP", "BPG"):
        res[design] = analyse(design)
        r = res[design]["chemotype"]
        print(design, "level winner", r["R2_level"]["argmax"], round(r["R2_level"]["E_boot_max_minus_max"], 4),
              "dR2 winner", r["dR2_vs_PWL_R_same_reads"]["argmax"], round(r["dR2_vs_PWL_R_same_reads"]["E_boot_max_minus_max"], 4),
              "R23 maxT lb", round(r["lower_bounds_dR2"]["R23"]["lb_maxT_simultaneous_95"], 4),
              "R14 maxT lb", round(r["lower_bounds_dR2"]["R14"]["lb_maxT_simultaneous_95"], 4))
    (paths.RES / "selection_optimism.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
