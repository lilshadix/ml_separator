#!/usr/bin/env python
"""Power / MDE simulation for the round-4 promotion gate (PROTOCOL section 6 item 8; brief section 9).
Design fixed in this docstring before it was run.

Generative model of a SOURCE-INDEPENDENT confirmation set: G new sources, each a chemotype drawn with replacement
from the 38 C3 chemotypes of REF's BPG out-of-fold rows for dev seed 104729 (cells, pairs, targets y and the
B1 = reference predictions of that chemotype); every drawn source is its own cluster.  A synthetic candidate
moves B1 toward the target:  p_cand = p_B1 + kappa_i (y - p_B1),  kappa_i = kbar * w_i * z_g, with
  w_i  uniform (1)                    or concentrated (|e_B1,i| / mean |e_B1|; large errors improve most),
  z_g  homogeneous (1)                or sparse (1 / 0.3 for a random 30 % of sources, else 0; mean 1),
and kbar solved so that the candidate's dev ΔR² vs B1 (all 1,802 template rows, z = 1) equals delta.
Null (delta = 0): each source is either improved or harmed, kappa_g = +0.1 w.p. q, -0.1 otherwise, with
q = (2 + 0.1) / 4 so that the expected squared error equals B1's (a candidate that differs but is no better).
Each confirmation set is scored exactly as PROTOCOL section 6 item 3: ΔR² and Δbalanced-MAE vs B1 with the
percentile 95 %, BCa 95 % and cluster-jackknife t(G-1) intervals (999 cluster-bootstrap draws), Kish >= 3;
'pass_ci' = all three intervals exclude 0 in the favourable direction for BOTH endpoints and Kish >= 3;
'pass_full' additionally needs the point estimates >= +0.01 R^2 and >= 0.005 balanced MAE.
Grid: delta in {0, 0.01, 0.02, 0.034}, G in {5, 10, 20, 40, 80}, three scenarios, 400 Monte Carlo sets each.
Also reported: the Δbalanced-MAE each delta implies on the dev template (are the two margins consistent?).
Added after the first run (disclosed in RUN_RECORD): the three mechanical scenarios improve every source, so their
power depends on the cluster count only.  Scenario 'empirical_B1' replays the real, heterogeneous structure of a
learned gain: on each drawn source the reference is B0 and the candidate is B0 + s (B1 - B0), with s solved so
that the dev ΔR² equals delta (s = 1 reproduces B1 vs B0 itself; per-source gains vary and can be negative).

    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_power.py
writes results/power/power_grid.csv, results/power/margins_implied.csv, results/power/summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import paths  # noqa: E402

SEED_TEMPLATE = 104729
DELTAS = (0.0, 0.001, 0.01, 0.02, 0.034)
# Run 3 (after the block-A pre-seal review): every non-null scenario also gets an EFFECT-INDEPENDENT per-source
# perturbation kappa_g ~ N(0, TAU) applied to the candidate (pc <- pc + kappa_g (y - pc)); TAU is fixed before the run
# as half the between-chemotype SD of the relative SSE change of two near-equal dev models (HIST_D7 vs B1, template
# seed), since a kappa perturbation changes a source's SSE by about -2 kappa.  A 'boundary' null (dR2 = 0 with a
# positive balanced-MAE gain: an SSE-preserving reshaping of B1's errors, large errors +10 %, small errors shrunk) gives
# the size of the intersection-union rule.  Kish < 3 failures are reported separately.
GS = (5, 10, 20, 40, 80)
SCENARIOS = ("uniform_homog", "conc_sparse", "empirical_B1", "boundary")
REPS, BOOT = 400, 999
NULL_K, SPARSE_P = 0.1, 0.3
OUT = paths.RES / "power"


def template():
    t = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    t = t[(t.split_seed == SEED_TEMPLATE) & (t.k == 3) & (t.k_max >= 3)].reset_index(drop=True)
    e = (t.B1 - t.y).to_numpy()
    w = np.abs(e) / np.abs(e).mean()
    return t, e, w


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def solve_kbar(t, e, w, delta, conc: bool) -> float:
    y, p = t.y.to_numpy(), t.B1.to_numpy()
    ww = w if conc else np.ones_like(w)
    base = r2(y, p)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        k = (lo + hi) / 2
        kk = np.minimum(k * ww, 0.95)
        if r2(y, p - kk * e) - base < delta:
            lo = k
        else:
            hi = k
    return (lo + hi) / 2


def units(t):
    """Per chemotype: arrays of y, B1, B0, extractant codes (for the balanced MAE)."""
    out = []
    for ch, g in t.groupby("chemotype"):
        out.append({"y": g.y.to_numpy(), "p": g.B1.to_numpy(), "p0": g.B0.to_numpy(), "ext": pd.factorize(g.extractant)[0]})
    return out


def solve_s(t, delta) -> float:
    y, p0, d = t.y.to_numpy(), t.B0.to_numpy(), (t.B1 - t.B0).to_numpy()
    base = r2(y, p0)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        s = (lo + hi) / 2
        if r2(y, p0 + s * d) - base < delta:
            lo = s
        else:
            hi = s
    return (lo + hi) / 2


def jk_t(point, jk):
    G = len(jk)
    se = np.sqrt((G - 1) / G * ((jk - jk.mean()) ** 2).sum())
    q = SS.t.ppf(0.975, G - 1)
    return point - q * se, point + q * se


def bca_low(draws, point, jk):
    prop = ((draws < point).sum() + 0.5 * (draws == point).sum()) / len(draws)
    if prop <= 0 or prop >= 1:
        return np.nan
    z0 = SS.norm.ppf(prop)
    m = jk.mean()
    den = 6 * (((m - jk) ** 2).sum()) ** 1.5
    if den <= 0:
        return np.nan
    a = ((m - jk) ** 3).sum() / den
    z = SS.norm.ppf(0.025)
    adj = SS.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    return float(np.quantile(draws, min(max(adj, 0), 1)))


def score(sources, rng):
    """sources: list of dicts with y, pc (candidate), pr (reference), ext.  Returns decision quantities."""
    G = len(sources)
    n = np.array([len(s["y"]) for s in sources], float)
    Sy = np.array([s["y"].sum() for s in sources])
    Sy2 = np.array([(s["y"] ** 2).sum() for s in sources])
    Sc = np.array([((s["y"] - s["pc"]) ** 2).sum() for s in sources])
    Sr = np.array([((s["y"] - s["pr"]) ** 2).sum() for s in sources])
    mc, mr = [], []
    for s in sources:
        ac, ar = np.abs(s["y"] - s["pc"]), np.abs(s["y"] - s["pr"])
        mc.append(np.mean([ac[s["ext"] == x].mean() for x in np.unique(s["ext"])]))
        mr.append(np.mean([ar[s["ext"] == x].mean() for x in np.unique(s["ext"])]))
    mc, mr = np.array(mc), np.array(mr)

    def stats(wt):                                   # wt: (..., G) cluster multiplicities
        N, SY, SY2 = wt @ n, wt @ Sy, wt @ Sy2
        sst = SY2 - SY ** 2 / N
        dr2 = (wt @ Sr - wt @ Sc) / sst
        dmae = (wt @ (mr - mc)) / wt.sum(-1)
        return dr2, dmae
    p_r2, p_mae = stats(np.ones(G))
    W = rng.multinomial(G, np.full(G, 1 / G), size=BOOT).astype(float)
    b_r2, b_mae = stats(W)
    J = 1 - np.eye(G)
    j_r2, j_mae = stats(J)
    kish = n.sum() ** 2 / (n ** 2).sum()
    res = {"kish": kish, "dR2": float(p_r2), "dMAE": float(p_mae)}
    ok = kish >= 3
    for name, pt, b, j in (("R2", p_r2, b_r2, j_r2), ("MAE", p_mae, b_mae, j_mae)):
        perc = np.quantile(b, 0.025)
        bl = bca_low(b, pt, j)
        jl, _ = jk_t(pt, j)
        ex = (perc > 0) and (np.isfinite(bl) and bl > 0) and (jl > 0)
        res[f"excl_{name}"] = bool(ex)
        ok &= ex
    res["pass_ci"] = bool(ok)
    res["pass_full"] = bool(ok and p_r2 >= 0.01 and p_mae >= 0.005)
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t, e, w = template()
    U = units(t)
    # per-row weights on the template, reindexed per unit
    wmap = {}
    for ch, g in t.groupby("chemotype"):
        wmap[ch] = w[g.index.to_numpy()]
    chems = list(t.groupby("chemotype").groups)
    implied, kbars = [], {}
    # TAU: half the between-chemotype SD of (SSE_HIST_D7 - SSE_B1) / SSE_B1 on the template seed
    rel = []
    for ch, g in t.groupby("chemotype"):
        s1 = ((g.B1 - g.y) ** 2).sum()
        if s1 > 0:
            rel.append((((g.HIST_D7 - g.y) ** 2).sum() - s1) / s1)
    tau = float(np.std(rel) / 2)
    for sc in SCENARIOS:
        conc = sc.startswith("conc")
        if sc == "boundary":
            continue
        for d in DELTAS:
            if d == 0:
                continue
            if sc == "empirical_B1":
                kb = solve_s(t, d)
                kbars[(sc, d)] = kb
                pc, pr = (t.B0 + kb * (t.B1 - t.B0)).to_numpy(), t.B0.to_numpy()
            else:
                kb = solve_kbar(t, e, w, d, conc)
                kbars[(sc, d)] = kb
                kk = np.minimum(kb * (w if conc else 1.0), 0.95)
                pc, pr = t.B1.to_numpy() - kk * e, t.B1.to_numpy()
            g = t.assign(ac=np.abs(t.y - pc), ar=np.abs(t.y - pr))
            ch = g.groupby(["chemotype", "extractant"])[["ac", "ar"]].mean().groupby(level=0).mean()
            implied.append({"scenario": sc, "delta_R2": d, "kbar": kb, "implied_dMAE_balanced": float((ch.ar - ch.ac).mean()),
                            "implied_dMAE_rows": float((g.ar - g.ac).mean())})
    pd.DataFrame(implied).to_csv(OUT / "margins_implied.csv", index=False)
    rows = []
    rng = np.random.default_rng(20261022)
    for sc in SCENARIOS:
        conc, sparse = sc.startswith("conc"), sc.endswith("sparse")
        for d in (DELTAS if sc != "boundary" else (0.0,)):
            for G in GS:
                acc = []
                for r in range(REPS):
                    pick = rng.integers(len(U), size=G)
                    srcs = []
                    for u_i in pick:
                        u = U[u_i]
                        wi = wmap[chems[u_i]] if conc else np.ones(len(u["y"]))
                        if sc == "boundary":                 # SSE-preserving reshaping: dR2 = 0, balanced MAE better
                            e = u["p"] - u["y"]          # small errors shrink 30 %, large errors grow to keep the SSE
                            big = np.abs(e) >= np.median(np.abs(e))
                            sl, ss = (e[big] ** 2).sum(), (e[~big] ** 2).sum()
                            grow = np.sqrt(1 + (1 - 0.7 ** 2) * ss / max(sl, 1e-12))
                            e2 = np.where(big, grow * e, 0.7 * e)
                            kap = NULL_K if rng.random() < (2 + NULL_K) / 4 else -NULL_K
                            pc = u["y"] + e2
                            srcs.append({"y": u["y"], "pr": u["p"], "pc": pc + kap * (u["y"] - pc), "ext": u["ext"]})
                            continue
                        if d == 0:
                            kap = NULL_K if rng.random() < (2 + NULL_K) / 4 else -NULL_K
                            kk = np.full(len(u["y"]), kap)
                            srcs.append({"y": u["y"], "pr": u["p"], "pc": u["p"] + kk * (u["y"] - u["p"]), "ext": u["ext"]})
                            continue
                        if sc == "empirical_B1":
                            sv = kbars[(sc, d)]
                            pr, pc = u["p0"], u["p0"] + sv * (u["p"] - u["p0"])
                        else:
                            z = (1 / SPARSE_P if rng.random() < SPARSE_P else 0.0) if sparse else 1.0
                            kk = np.minimum(kbars[(sc, d)] * wi * z, 0.95)
                            pr, pc = u["p"], u["p"] + kk * (u["y"] - u["p"])
                        kg = rng.normal(0, tau)                  # effect-independent per-source perturbation
                        srcs.append({"y": u["y"], "pr": pr, "pc": pc + kg * (u["y"] - pc), "ext": u["ext"]})
                    acc.append(score(srcs, rng))
                a = pd.DataFrame(acc)
                rec = {"scenario": sc, "delta_R2": d, "G": G, "reps": REPS, "tau": tau,
                       "kish_fail": float((a.kish < 3).mean())}
                for c in ("pass_ci", "pass_full", "excl_R2", "excl_MAE"):
                    p = float(a[c].mean())
                    rec[c] = p
                    rec[c + "_mcse"] = float(np.sqrt(p * (1 - p) / REPS))
                rec["kish_median"] = float(a.kish.median())
                rec["dR2_median"], rec["dMAE_median"] = float(a.dR2.median()), float(a.dMAE.median())
                rows.append(rec)
                print(sc, d, G, f"pass_ci {rec['pass_ci']:.3f} pass_full {rec['pass_full']:.3f} kish {rec['kish_median']:.1f}",
                      flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "power_grid.csv", index=False)
    need = {}
    for sc in SCENARIOS:
        if sc == "boundary":
            continue
        for d in DELTAS[1:]:
            g = grid[(grid.scenario == sc) & (grid.delta_R2 == d)]
            ok = g[g.pass_ci >= 0.8]
            need[f"{sc}|{d}"] = int(ok.G.min()) if len(ok) else None
    summary = {"template": f"REF BPG seed {SEED_TEMPLATE}, {len(t)} rows, {len(U)} chemotypes", "tau": tau,
               "joint_null_pass_ci_by_G": grid[(grid.delta_R2 == 0) & (grid.scenario != "boundary")].groupby("G").pass_ci.max().to_dict(),
               "boundary_null_pass_ci_by_G (size)": grid[grid.scenario == "boundary"].set_index("G").pass_ci.to_dict(),
               "sources_needed_for_80pct_pass_ci": need,
               "note": "G = number of new independent sources (each a chemotype-like unit); None = not reached at G <= 80"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
