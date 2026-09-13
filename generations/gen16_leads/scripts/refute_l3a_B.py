"""Refutation of CLAIM L3A (lens B: statistics, confounds, generalisation).

Imports the lead's own construction (``gen16.l3_decision``); no metric, splitter, task builder
or bootstrap is re-implemented.  Stage 'tabs' caches ``run_arms`` output per design (discovery
seeds, never ``seeds=``); stage 'main' runs every check and writes CSVs under
results/refutation/L3A/B/.

Usage (repo root):
  PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe \
      gen16_leads/scripts/refute_l3a_B.py --stage tabs
  ... --stage main
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap                      # noqa: E402
from gen16 import l3_decision as L               # noqa: E402
from gen14.dirbench import load                  # noqa: E402

OUT = bootstrap.RESULTS / "refutation" / "L3A" / "B"
OUT.mkdir(parents=True, exist_ok=True)
DESIGNS = list(L.DESIGNS)
COLS = ["split_seed", "cell_id", "extractant", "chemotype", "A", "B", "y", "G14"]
BOOT2 = 20260911            # a *different* resampling seed from the lead's 8675309
NB = L.N_BOOT


# --------------------------------------------------------------------------- stage 1
def stage_tabs() -> int:
    bench = load()
    f = bench.frame
    f[["cell_id", "extractant", "publication_id", "n_metals", "recipe__DENTATE",
       "recipe__coreCN"]].to_parquet(OUT / "cell_meta.parquet")
    Y, bas = bench.Y, bench.basis
    amp = []
    for i in range(len(f)):
        y = Y[i]
        ok = ~np.isnan(y)
        if ok.sum() < 3:
            amp.append(np.nan)
            continue
        X = np.column_stack([np.ones(int(ok.sum())), bas[0][ok], bas[1][ok]])
        amp.append(float(np.linalg.lstsq(X, y[ok], rcond=None)[0][1]))
    pd.DataFrame({"cell_id": f.cell_id.to_numpy(), "a_lin": amp}).to_parquet(OUT / "amp.parquet")
    for d in DESIGNS:
        t0 = time.time()
        tab = L.run_pair_table(bench, d)
        tab[[c for c in COLS if c in tab.columns]].to_parquet(OUT / f"tab_{d}.parquet")
        print(f"[{d}] {len(tab)} pair rows cached in {time.time()-t0:.0f}s", flush=True)
    return 0


# --------------------------------------------------------------------------- helpers
def ext_meta() -> pd.DataFrame:
    meta = pd.read_parquet(OUT / "cell_meta.parquet")
    amp = pd.read_parquet(OUT / "amp.parquet")
    meta = meta.merge(amp, on="cell_id", how="left")
    g = meta.groupby("extractant")
    out = pd.DataFrame({
        "n_metals": g.n_metals.median(),
        "n_cells": g.size(),
        "dentate": g["recipe__DENTATE"].median(),
        "abs_a": g.a_lin.median().abs(),
        "n_pub": g.publication_id.nunique(),
    })
    out["pubs"] = meta.groupby("extractant").publication_id.agg(lambda s: sorted(set(s)))
    return out.reset_index()


class Board:
    """Everything derived from one design's task set, built with the lead's code."""

    def __init__(self, design: str, em: pd.DataFrame):
        tab = pd.read_parquet(OUT / f"tab_{design}.parquet")
        self.design = design
        self.tab = tab
        self.T = T = L.build_tasks(tab, design, ["G14"])
        self.calls = [L.calls_of(p) for p in T.pred["G14"]]
        self.mats = L.l3a_matrices(T, self.calls)
        self.n_ext = len(T.ext)
        self.one = np.ones((1, self.n_ext))
        self.pt = {k: v[0] for k, v in L.l3a_saved_weighted(self.one, self.mats).items()}
        m = em.set_index("extractant").reindex(T.ext)
        self.nmet = m.n_metals.to_numpy(float)
        self.dent = m.dentate.to_numpy(float)
        self.absa = m.abs_a.to_numpy(float)
        self.pubs = list(m.pubs)
        self.chem = np.asarray(T.chem_of)

    def stat(self, W, key="saved", sel=None):
        return L.seed_macro(L.l3a_saved_weighted(W, self.mats)[key], self.T.seeds, sel)

    def point(self, key="saved", sel=None):
        return L.seed_macro(self.pt[key], self.T.seeds, sel)

    def saved_for_calls(self, calls, W=None, sel=None):
        mats = L.l3a_matrices(self.T, calls)
        W = self.one if W is None else W
        v = L.l3a_saved_weighted(W, mats)["saved"]
        return float(np.atleast_1d(L.seed_macro(v, self.T.seeds, sel))[0])


def block_boot(blocks, keep=None, reps=NB, seed=L.BOOT_SEED):
    """Resample the distinct values of ``blocks`` (one label per extractant) with replacement."""
    blocks = np.asarray(blocks)
    keep = np.ones(len(blocks), bool) if keep is None else np.asarray(keep, bool)
    names = sorted(set(blocks[keep]))
    members = [np.flatnonzero((blocks == b) & keep) for b in names]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(names), size=(reps, len(names)))
    W = np.zeros((reps, len(blocks)))
    for j, idx in enumerate(members):
        W[:, idx] = (picks == j).sum(1)[:, None]
    return W


def pair_boot(bd, values, reps=NB, seed=L.BOOT_SEED):
    """Bootstrap over the distinct unordered metal pairs -- the generalisation unit implied by
    'given a target metal pair'.  All five seeds of a resampled pair move together."""
    T = bd.T
    key = np.array([f"{a}|{b}" for a, b in zip(T.A, T.B)])
    pairs = sorted(set(key))
    lut = {p: i for i, p in enumerate(pairs)}
    pidx = np.array([lut[k] for k in key])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(pairs), size=(reps, len(pairs)))
    M = np.zeros((reps, len(pairs)))
    for j in range(len(pairs)):
        M[:, j] = (picks == j).sum(1)
    w_task = M[:, pidx]
    per_seed = []
    for s in np.unique(T.seeds):
        c = T.seeds == s
        v = values[c]
        ok = np.isfinite(v)
        w = w_task[:, c][:, ok]
        den = w.sum(1)
        per_seed.append(np.where(den > 0, (w @ v[ok]) / np.maximum(den, 1e-12), np.nan))
    return np.nanmean(np.stack(per_seed, 1), 1)


def width_matched(bd, score, sign=+1):
    """Calls that defer exactly as many candidates as G14 does in each task, chosen by a
    per-extractant covariate instead of by the model (a width-matched, model-free competitor)."""
    out = []
    for t in range(bd.T.n):
        idx, c = bd.T.idx[t], bd.calls[t]
        s = sign * score[idx]
        s = np.where(np.isfinite(s), s, -np.inf)
        n_light, n_heavy = int((c == 1).sum()), int((c == -1).sum())
        order = np.argsort(-s, kind="stable")
        new = np.zeros(len(idx), int)
        new[order[:n_light]] = 1
        if n_heavy:
            new[order[len(idx) - n_heavy:]] = -1
        out.append(new)
    return out


def chemo_collapsed(bd):
    """Every candidate takes its chemotype's majority G14 call inside the task."""
    out = []
    for t in range(bd.T.n):
        idx, c = bd.T.idx[t], bd.calls[t].copy()
        ch = bd.chem[idx]
        for g in set(ch):
            m = ch == g
            c[m] = int(np.sign(c[m].sum()))
        out.append(c)
    return out


def perm_subset(bd, keep, seed):
    """The lead's within-task permutation of the calls, restricted to the kept candidates."""
    T = bd.T
    rng = np.random.default_rng(seed)
    out = np.zeros((L.N_PERM, T.n))
    for t in range(T.n):
        m = keep[T.idx[t]]
        obs, call = T.obs[t][m], bd.calls[t][m]
        n = len(obs)
        if n < L.MIN_EXT:
            out[:, t] = np.nan
            continue
        perm = np.argsort(rng.random((L.N_PERM, n)), axis=1)
        cp = call[perm]
        sv = []
        for dd in (-1, 1):
            succ = (np.sign(obs) == dd) & (np.abs(obs) >= L.STRONG)
            K = int(succ.sum())
            kept = cp != -dd
            Nk = kept.sum(1)
            Kk = (kept & succ[None, :]).sum(1)
            sv.append(L.expected_draws(n, K) - L.e_model_from_counts(n, K, Nk, Kk))
        out[:, t] = 0.5 * (sv[0] + sv[1])
    return L.seed_macro(out, T.seeds)


def bh(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    v = p[ok]
    n = len(v)
    o = np.argsort(v)
    adj = np.minimum.accumulate((v[o] * n / np.arange(n, 0, -1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(adj, 1.0)
    q[ok] = out
    return q


# --------------------------------------------------------------------------- main
def main() -> int:
    t_all = time.time()
    em = ext_meta()
    rows, notes = [], {}
    boards = {d: Board(d, em) for d in DESIGNS}
    for d in DESIGNS:
        print(f"[{d}] board built ({boards[d].T.n} tasks)", flush=True)

    # ---- 0. structural: are the 455 tasks 91 pairs x 5 seeds with an identical observed side?
    bd = boards["BP"]
    T = bd.T
    key = [f"{a}|{b}" for a, b in zip(T.A, T.B)]
    by_pair = {}
    for t, k in enumerate(key):
        by_pair.setdefault(k, []).append((tuple(T.idx[t]), tuple(np.round(T.obs[t], 12))))
    notes["n_distinct_pairs"] = len(by_pair)
    notes["observed_side_identical_across_seeds"] = bool(
        all(all(s == v[0] for s in v[1:]) for v in by_pair.values()))
    notes["n_tasks"] = int(T.n)
    notes["e_random_by_design"] = {d: boards[d].point("e_random") for d in DESIGNS}
    notes["e_random_spread"] = float(max(notes["e_random_by_design"].values())
                                     - min(notes["e_random_by_design"].values()))
    # correlation of the call with n_metals inside a task
    sp = []
    for t in range(T.n):
        c, nm = bd.calls[t].astype(float), bd.nmet[T.idx[t]]
        if np.ptp(c) > 0 and np.ptp(nm) > 0:
            sp.append(float(pd.Series(c).corr(pd.Series(nm), method="spearman")))
    notes["mean_within_task_spearman_call_vs_nmetals_BP"] = float(np.mean(sp))
    notes["n_tasks_with_varying_call_BP"] = len(sp)

    for d in DESIGNS:
        bd = boards[d]
        T = bd.T
        pt = bd.pt
        base = bd.point()
        W_boot = L.blocked_weights(T)
        draws = bd.stat(W_boot)
        ci = (L.pct(draws, 0.025), L.pct(draws, 0.975))
        perm1 = L.seed_macro(L.l3a_permutation(T, bd.calls, seed=L.BOOT_SEED)["saved"], T.seeds)
        P2 = L.l3a_permutation(T, bd.calls, seed=BOOT2)["saved"]
        perm2 = L.seed_macro(P2, T.seeds)
        p_seed, sd_seed_null = [], []
        for s in np.unique(T.seeds):
            sel = T.seeds == s
            obs_s = L.seed_macro(pt["saved"], T.seeds, sel)
            null_s = L.seed_macro(P2, T.seeds, sel)
            p_seed.append(float((null_s >= obs_s).mean()))
            sd_seed_null.append(float(np.std(null_s)))
        rows.append(dict(design=d, check="headline", point=base, ci_low=ci[0], ci_high=ci[1],
                         p_two_sided=L.two_sided_p(draws),
                         p_perm_leadseed=float((perm1 >= base).mean()),
                         p_perm_altseed=float((perm2 >= base).mean()),
                         p_perm_per_seed_max=float(max(p_seed)),
                         null_sd_pooled=float(np.std(perm2)),
                         null_sd_single_seed=float(np.mean(sd_seed_null)),
                         n_tasks=int(T.n), n_ext=bd.n_ext, n_blocks=len(set(bd.chem))))
        rows.append(dict(design=d, check="e_random_e_model", point=base,
                         extra1=bd.point("e_random"), extra2=bd.point("e_model"),
                         extra3=base / bd.point("e_random")))
        rows.append(dict(design=d, check="saved_heavy", point=bd.point("saved_heavy")))
        rows.append(dict(design=d, check="saved_light", point=bd.point("saved_light")))

        # ---- check 1: diglycolamides (sc009) removed
        keep = bd.chem != "sc009"
        v_nod = L.l3a_saved_weighted(keep.astype(float)[None, :], bd.mats)["saved"][0]
        pt_nod = L.seed_macro(v_nod, T.seeds)
        dr = L.seed_macro(L.l3a_saved_weighted(block_boot(bd.chem, keep), bd.mats)["saved"], T.seeds)
        pn = perm_subset(bd, keep, BOOT2)
        rows.append(dict(design=d, check="drop_sc009_diglycolamides", point=pt_nod,
                         ci_low=L.pct(dr, 0.025), ci_high=L.pct(dr, 0.975),
                         p_two_sided=L.two_sided_p(dr),
                         p_perm_altseed=float((pn >= pt_nod).mean()),
                         n_tasks=int(np.isfinite(v_nod).sum()), n_ext=int(keep.sum()),
                         n_blocks=len(set(bd.chem[keep]))))
        v_only = L.l3a_saved_weighted((~keep).astype(float)[None, :], bd.mats)["saved"][0]
        rows.append(dict(design=d, check="sc009_only", point=L.seed_macro(v_only, T.seeds),
                         n_tasks=int(np.isfinite(v_only).sum()), n_ext=int((~keep).sum()),
                         n_blocks=1))

        # ---- check 2: n_metals (and two other covariates) as width-matched competitors
        for lab, sc, sg in (("nmetals_hi", bd.nmet, +1), ("nmetals_lo", bd.nmet, -1),
                            ("dentate_hi", bd.dent, +1), ("absa_hi", bd.absa, +1)):
            rows.append(dict(design=d, check=f"width_matched_{lab}",
                             point=bd.saved_for_calls(width_matched(bd, sc, sg)),
                             n_tasks=int(T.n), n_ext=bd.n_ext))
        for lo, hi in ((5, 8), (9, 14)):
            k2 = (bd.nmet >= lo) & (bd.nmet <= hi)
            v = L.l3a_saved_weighted(k2.astype(float)[None, :], bd.mats)["saved"][0]
            rows.append(dict(design=d, check=f"cand_nmetals_{lo}_{hi}",
                             point=L.seed_macro(v, T.seeds),
                             n_tasks=int(np.isfinite(v).sum()), n_ext=int(k2.sum()),
                             n_blocks=len(set(bd.chem[k2]))))
        rows.append(dict(design=d, check="call_collapsed_to_chemotype",
                         point=bd.saved_for_calls(chemo_collapsed(bd)),
                         n_tasks=int(T.n), n_ext=bd.n_ext))

        # ---- LOCO chemotype (lead's) and LOPO publication (new)
        names_c, Wc = L.loco_weights(T)
        lc = bd.stat(Wc)
        rows.append(dict(design=d, check="loco_chemotype", point=base,
                         loco_min=float(np.nanmin(lc)), loco_max=float(np.nanmax(lc)),
                         worst_block=str(names_c[int(np.nanargmin(lc))]), n_blocks=len(names_c)))
        allp = sorted({p for ps in bd.pubs for p in ps})
        Wp = np.ones((len(allp), bd.n_ext))
        for j, p in enumerate(allp):
            Wp[j, [i for i in range(bd.n_ext) if p in bd.pubs[i]]] = 0.0
        lp = bd.stat(Wp)
        rows.append(dict(design=d, check="lopo_publication", point=base,
                         loco_min=float(np.nanmin(lp)), loco_max=float(np.nanmax(lp)),
                         worst_block=str(allp[int(np.nanargmin(lp))]), n_blocks=len(allp)))
        pub_lab = np.array([p[0] for p in bd.pubs])
        dp = bd.stat(block_boot(pub_lab))
        rows.append(dict(design=d, check="publication_blocked_bootstrap", point=base,
                         ci_low=L.pct(dp, 0.025), ci_high=L.pct(dp, 0.975),
                         p_two_sided=L.two_sided_p(dp), n_blocks=len(set(pub_lab))))
        pb = pair_boot(bd, pt["saved"])
        rows.append(dict(design=d, check="pair_bootstrap", point=base,
                         ci_low=float(np.nanquantile(pb, 0.025)),
                         ci_high=float(np.nanquantile(pb, 0.975)),
                         p_two_sided=L.two_sided_p(pb), n_blocks=notes["n_distinct_pairs"]))
        metals = sorted(set(T.A) | set(T.B))
        lm = [L.seed_macro(pt["saved"], T.seeds, (T.A != m) & (T.B != m)) for m in metals]
        rows.append(dict(design=d, check="leave_one_metal_out", point=base,
                         loco_min=float(np.nanmin(lm)), loco_max=float(np.nanmax(lm)),
                         worst_block=str(metals[int(np.nanargmin(lm))]), n_blocks=len(metals)))

        # ---- bands on candidate-set size
        sizes = T.sizes()
        qs = np.quantile(sizes, [0.25, 0.75])
        for lab, sel in (("cands_le_q1", sizes <= qs[0]),
                         ("cands_mid", (sizes > qs[0]) & (sizes <= qs[1])),
                         ("cands_gt_q3", sizes > qs[1])):
            rows.append(dict(design=d, check=f"band_{lab}",
                             point=L.seed_macro(pt["saved"], T.seeds, sel),
                             n_tasks=int(sel.sum())))

        # ---- threshold sensitivity
        old_strong, old_min = L.STRONG, L.MIN_EXT
        for thr in (0.2, 0.3, 0.5, 1.0):
            L.STRONG = thr
            mats = L.l3a_matrices(T, bd.calls)
            v = L.l3a_saved_weighted(bd.one, mats)["saved"][0]
            dr2 = L.seed_macro(L.l3a_saved_weighted(W_boot, mats)["saved"], T.seeds)
            rows.append(dict(design=d, check=f"strong_threshold_{thr}",
                             point=L.seed_macro(v, T.seeds),
                             ci_low=L.pct(dr2, 0.025), ci_high=L.pct(dr2, 0.975)))
        L.STRONG = old_strong
        for mn in (5, 10, 20):
            L.MIN_EXT = mn
            T2 = L.build_tasks(bd.tab, d, ["G14"])
            c2 = [L.calls_of(p) for p in T2.pred["G14"]]
            m2 = L.l3a_matrices(T2, c2)
            v = L.l3a_saved_weighted(np.ones((1, len(T2.ext))), m2)["saved"][0]
            rows.append(dict(design=d, check=f"min_candidates_{mn}",
                             point=L.seed_macro(v, T2.seeds), n_tasks=int(T2.n)))
        L.MIN_EXT = old_min

        for s in np.unique(T.seeds):
            rows.append(dict(design=d, check=f"seed_{s}",
                             point=L.seed_macro(pt["saved"], T.seeds, T.seeds == s),
                             n_tasks=int((T.seeds == s).sum())))
        print(f"[{d}] checks done  t={time.time()-t_all:.0f}s", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "refute_l3a_B_checks.csv", index=False)
    (OUT / "refute_l3a_B_notes.json").write_text(json.dumps(notes, indent=2, default=str),
                                                 encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_rows", 400):
        print(R.to_string())
    print(json.dumps(notes, indent=2, default=str))
    print(f"total {time.time()-t_all:.0f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="main")
    a = ap.parse_args()
    sys.exit(stage_tabs() if a.stage == "tabs" else main())
