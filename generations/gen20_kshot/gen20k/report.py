"""From the long prediction tables to every table and figure in the report.

Nothing here refits anything.  The only inputs are ``predictions/<label>/pairs_<design>.parquet``
(one row per design, seed, fold, cell, measurement model, policy, draw, k, pair; one column per
arm@adapter) and the fold records.  The common-query cohort ``C<K>`` of a measurement model is the
set of (seed, cell) with ``k_max >= K`` and at least one adjacent query; inside it every budget
``k <= K`` is scored on byte-identical query rows, which ``wide_common`` makes explicit by pivoting
the budgets into columns ``arm|k<k>``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import bootstrap
from . import inference as INF
from . import metrics as M
from .design import K_GRID

COHORT_KS = {"metal": (3, 5, 6), "pair": (3, 5, 10)}
KEY = ["design", "split_seed", "fold", "cell_id", "extractant", "chemotype", "publication", "n_metals", "model", "k_max",
       "A", "B", "dZ", "regime", "y"]
NON_ARM = set(KEY + ["policy", "draw", "k", "ia", "ib", "n_read", "lab_known"])
PAIR_ORDER = ["La/Ce", "Ce/Pr", "Pr/Nd", "Nd/Sm", "Sm/Eu", "Eu/Gd", "Gd/Tb", "Tb/Dy", "Dy/Ho", "Ho/Er", "Er/Tm", "Tm/Yb", "Yb/Lu"]
MAIN_ARMS = ("FLAT@BLUP20", "G14@BLUP20", "NNC@SELECTED", "LINE", "PROJ", "RIDGE2", "NN_TANIMOTO@RESCALE", "G14@ZERO")


def arm_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in NON_ARM]


def load_tables(label: str) -> dict[str, pd.DataFrame]:
    d = bootstrap.PREDICTIONS / label
    return {p.stem.split("_", 1)[1]: pd.read_parquet(p) for p in sorted(d.glob("pairs_*.parquet"))}


def kish(counts) -> float:
    c = np.asarray(list(counts), dtype=float)
    return float(c.sum() ** 2 / (c ** 2).sum()) if c.size else float("nan")


def cohort_counts(table: pd.DataFrame, model: str) -> pd.DataFrame:
    """Eligible cells / extractants / chemotypes / publications / query pairs and the pair-weighted
    Kish effective number of chemotypes at each k, per-seed mean."""
    q = table[(table.regime == "Q") & (table.model == model) & (table.policy == "UNIFORM") & (table.k == 0)]
    rows = []
    for k in K_GRID:
        e = q[q.k_max >= k]
        if e.empty:
            rows.append({"model": model, "k": k, "cells": 0}); continue
        per = e.groupby("split_seed").agg(cells=("cell_id", "nunique"), extractants=("extractant", "nunique"),
                                          chemotypes=("chemotype", "nunique"), publications=("publication", "nunique"),
                                          query_pairs=("y", "size"))
        kis = e.groupby("split_seed").apply(lambda g: kish(g.groupby("chemotype").size()), include_groups=False)
        share = e.groupby("split_seed").apply(lambda g: g.chemotype.value_counts(normalize=True).iloc[0], include_groups=False)
        rows.append({"model": model, "k": k, **{c: float(per[c].mean()) for c in per.columns},
                     "kish_chemotypes_pairs": float(kis.mean()), "largest_chemotype_share": float(share.mean())})
    return pd.DataFrame(rows)


def wide_common(table: pd.DataFrame, K: int, policy: str, model: str, *, regime: str = "Q", arms=None,
                lab_known: bool | None = None) -> pd.DataFrame:
    """Rows: query pairs of cells with ``k_max >= K`` (one row per RANDOM draw); columns: KEY + draw +
    ``<arm>|k<k>`` for every budget k <= K.  ``lab_known=False`` keeps only rows whose two metals were
    not both read at that budget (pair model; varies with k, so it is a diagnostic, not a cohort)."""
    t = table[(table.regime == regime) & (table.model == model) & (table.policy == policy) & (table.k_max >= K) & (table.k <= K)]
    arms = list(arms) if arms is not None else arm_columns(table)
    ks = [k for k in K_GRID if k <= K]
    parts = []
    for k in ks:
        tk = t[t.k == k]
        if lab_known is not None:
            tk = tk[tk.lab_known == lab_known]
        tk = tk.set_index(KEY + ["draw"])[arms]
        tk.columns = [f"{a}|k{k}" for a in arms]
        parts.append(tk)
    return pd.concat(parts, axis=1, join="inner").reset_index()


def levels(wide: pd.DataFrame, arms_k: list[str], *, replicates: int = INF.REPLICATES) -> pd.DataFrame:
    lv = INF.pooled_levels(wide, arms_k, replicates=replicates)
    lv = lv.pivot(index="arm", columns="statistic", values=["point", "ci_low", "ci_high", "seed_sd", "per_seed"])
    lv.columns = [f"{s}_{p}" if p != "point" else s for p, s in lv.columns]
    extra = []
    for a in arms_k:
        per = [M.pooled(g.rename(columns={a: "pred"}), "pred") for _, g in wide.groupby("split_seed")]
        mean = {}
        for k in per[0]:
            if isinstance(per[0][k], (int, float)):
                vals = np.array([p[k] for p in per], dtype=float)
                mean[k] = float(vals[np.isfinite(vals)].mean()) if np.isfinite(vals).any() else float("nan")
        mean["arm"] = a
        extra.append(mean)
    ex = pd.DataFrame(extra).set_index("arm")
    out = lv.join(ex[[c for c in ex.columns if c not in lv.columns]])
    out["arm_base"] = [a.split("|")[0] for a in out.index]
    out["k"] = [int(a.split("|k")[1]) for a in out.index]
    out["kish_chemotypes_pairs"] = kish(wide.groupby("chemotype").size())
    return out.reset_index().sort_values(["arm_base", "k"])


def contrasts(wide: pd.DataFrame, comps: dict[str, tuple[str, str]], *, replicates: int = INF.REPLICATES) -> pd.DataFrame:
    return INF.pooled_contrasts(wide, comps, replicates=replicates)


def per_pair(wide: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for s, g in wide.groupby("split_seed"):
        pp = M.per_pair_type(g.rename(columns={col: "pred"}), "pred")
        pp["split_seed"] = s
        rows.append(pp)
    allp = pd.concat(rows)
    agg = allp.groupby("pair").agg(n_rows=("n_rows", "mean"), n_extractants=("n_extractants", "mean"),
                                   R2=("R2", "mean"), R2_sd=("R2", "std"), MAE=("MAE", "mean"), RMSE=("RMSE", "mean"),
                                   y_sd=("y_sd", "mean"), MAE_FLAT=("MAE_FLAT", "mean"), dir_acc_0=("dir_acc_0", "mean"),
                                   n_seeds_defined=("R2", lambda v: int(np.isfinite(v).sum()))).reset_index()
    order = {p: i for i, p in enumerate(PAIR_ORDER)}
    agg["order"] = agg["pair"].map(order)
    return agg.sort_values("order").drop(columns="order")


# --------------------------------------------------------------------------------------
# the standard set of tables for one design
# --------------------------------------------------------------------------------------
def build_design(table: pd.DataFrame, design: str, out: Path, *, replicates: int = INF.REPLICATES, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    arms = arm_columns(table)
    summary = {"design": design, "arms": arms}
    for model in sorted(table.model.unique()):
        cohort_counts(table, model).to_csv(out / f"cohort_counts_{design}_{model}.csv", index=False)
        policies = sorted(table[table.model == model].policy.unique())
        for K in COHORT_KS[model]:
            for pol in policies:
                w = wide_common(table, K, pol, model)
                if w.empty:
                    continue
                cols = [c for c in w.columns if "|k" in c]
                lv = levels(w, cols, replicates=replicates)
                lv.insert(0, "policy", pol); lv.insert(0, "cohort", f"C{K}"); lv.insert(0, "model", model); lv.insert(0, "design", design)
                lv.to_csv(out / f"levels_{design}_{model}_C{K}_{pol}.csv", index=False)
                if pol == "DOPT_Q":
                    for a in ("G14@BLUP20", "NNC@SELECTED", "FLAT@BLUP20", "LINE", "PROJ", "G14@ZERO"):
                        for k in sorted({0, 3, K}):
                            c = f"{a}|k{k}"
                            if c in w.columns:
                                pp = per_pair(w, c); pp.insert(0, "arm_k", c); pp.insert(0, "cohort", f"C{K}"); pp.insert(0, "model", model); pp.insert(0, "design", design)
                                pp.to_csv(out / f"per_pair_{design}_{model}_C{K}_{pol}_{a.replace('@', '-')}_k{k}.csv", index=False)
                    if model == "pair":
                        wl = wide_common(table, K, pol, model, lab_known=False)
                        if not wl.empty:
                            lvl = levels(wl, [c for c in wl.columns if "|k" in c], replicates=min(replicates, 2000))
                            lvl.insert(0, "policy", pol); lvl.insert(0, "cohort", f"C{K}_not_lab_known"); lvl.insert(0, "model", model); lvl.insert(0, "design", design)
                            lvl.to_csv(out / f"levels_{design}_{model}_C{K}_{pol}_not_lab_known.csv", index=False)
                log(f"  {design} {model} C{K} {pol}: {len(w)} rows, {w.cell_id.nunique()} cells")
    # the largest chemotype's hold-out on its own and the rest without it (DOPT_Q, C3, both models)
    for model in sorted(table.model.unique()):
        w = wide_common(table, 3, "DOPT_Q", model)
        if w.empty:
            continue
        big = table.chemotype.value_counts().index[0]
        for name, sub in ((f"{big}_only", w[w.chemotype == big]), (f"without_{big}", w[w.chemotype != big])):
            if len(sub) > 10:
                lv = levels(sub, [c for c in w.columns if "|k" in c], replicates=min(replicates, 2000))
                lv.insert(0, "subset", name); lv.insert(0, "model", model); lv.insert(0, "design", design)
                lv.to_csv(out / f"levels_{design}_{model}_C3_DOPT_Q_{name}.csv", index=False)
    return summary


def decide_mstar(folds_meta: list[dict]) -> dict:
    """PRE_REGISTRATION §4: NNC@SELECTED carries the registered claims only if its inner k = 3
    adjacent MAE beat G14@BLUP20's on a majority of BP (seed, fold) inner selections."""
    wins = total = 0
    for m in folds_meta:
        mi = (m.get("selection") or {}).get("mstar_inner")
        if not mi:
            continue
        total += 1
        wins += int(bool(mi.get("nnc_wins")))
    mstar = "NNC@SELECTED" if total and wins * 2 > total else "G14@BLUP20"
    return {"mstar": mstar, "nnc_wins": wins, "n_selections": total}


def registered_contrasts(table: pd.DataFrame, mstar: str, *, replicates: int = INF.REPLICATES) -> pd.DataFrame:
    """P1-P4 of PRE_REGISTRATION §7: metal model, design BP, policy DOPT_Q.  P1/P2/P4 are
    extractant-macro MAE contrasts (chemotype-blocked intervals at 95 and 98.75 %, wild-cluster-t p);
    P3 is the pooled ΔR² (cluster bootstrap, both levels).  Pooled MAE rows are printed beside."""
    rows = []
    w3 = wide_common(table, 3, "DOPT_Q", "metal")
    w6 = wide_common(table, 6, "DOPT_Q", "metal")
    specs = {"P1": (w3, "LINE|k3", f"{mstar}|k3"), "P2": (w3, f"{mstar}|k0", f"{mstar}|k3"), "P4": (w6, "LINE|k6", f"{mstar}|k6")}
    for name, (w, ref, cand) in specs.items():
        if w.empty or ref not in w.columns or cand not in w.columns:
            rows.append({"name": name, "statistic": "MAE_macro_extractant", "status": "not computable"}); continue
        r = INF.macro_mae_contrast(w, ref, cand, replicates=replicates)
        r.update({"name": name, "registered": True, "cohort": "C3" if name != "P4" else "C6", "n_cells": int(w.cell_id.nunique())})
        rows.append(r)
        pc = INF.pooled_contrasts(w, {name: (ref, cand)}, statistics=("MAE", "RMSE"), replicates=replicates)
        for rec in pc.to_dict(orient="records"):
            rec.update({"registered": False, "cohort": r["cohort"], "note": "pooled, descriptive"})
            rows.append(rec)
    if not w3.empty and f"{mstar}|k0" in w3.columns:
        pc = INF.pooled_contrasts(w3, {"P3": (f"{mstar}|k0", f"{mstar}|k3")}, statistics=("R2",), replicates=replicates)
        for rec in pc.to_dict(orient="records"):
            rec.update({"registered": True, "cohort": "C3", "n_cells": int(w3.cell_id.nunique())})
            rows.append(rec)
    out = pd.DataFrame(rows)
    out["mstar"] = mstar
    out["bonferroni_level"] = 0.9875
    return out


def exploratory_contrasts(table: pd.DataFrame, *, replicates: int = 2000, K: int = 3) -> pd.DataFrame:
    """The counted exploratory family (both models): every arm vs LINE and vs PROJ at each k, k vs 0
    per arm, NNC vs G14, structure after supports, and every policy vs RANDOM on identical cells."""
    rows = []
    arms = arm_columns(table)
    for model in sorted(table.model.unique()):
        for pol in sorted(table[table.model == model].policy.unique()):
            w = wide_common(table, K, pol, model)
            if w.empty:
                continue
            comps = {}
            for a in arms:
                for k in [k for k in K_GRID if 1 <= k <= K]:
                    for base in ("LINE", "PROJ"):
                        if a not in (base, "RIDGE2") and f"{a}|k{k}" in w.columns and f"{base}|k{k}" in w.columns:
                            comps[f"{a}_vs_{base}@k{k}@{pol}@{model}"] = (f"{base}|k{k}", f"{a}|k{k}")
                    if f"{a}|k0" in w.columns and f"{a}|k{k}" in w.columns:
                        comps[f"{a}@k{k}_vs_k0@{pol}@{model}"] = (f"{a}|k0", f"{a}|k{k}")
                for k in [k for k in K_GRID if k <= K]:
                    if a.startswith("NNC@") and f"G14@BLUP20|k{k}" in w.columns and f"{a}|k{k}" in w.columns:
                        comps[f"{a}_vs_G14@BLUP20@k{k}@{pol}@{model}"] = (f"G14@BLUP20|k{k}", f"{a}|k{k}")
                    if a == "G14@BLUP20" and f"FLAT@BLUP20|k{k}" in w.columns:
                        comps[f"G14@BLUP20_vs_FLAT@BLUP20@k{k}@{pol}@{model}"] = (f"FLAT@BLUP20|k{k}", f"{a}|k{k}")
            c = contrasts(w, comps, replicates=replicates)
            c.insert(0, "policy", pol); c.insert(0, "model", model)
            rows.append(c)
        wr = wide_common(table, K, "RANDOM", model)
        for pol in [p for p in sorted(table[table.model == model].policy.unique()) if p != "RANDOM"]:
            wp = wide_common(table, K, pol, model)
            if wp.empty or wr.empty:
                continue
            cols = [c for c in wp.columns if "|k" in c]
            m = wr.merge(wp[KEY + cols].rename(columns={c: c + "#" + pol for c in cols}), on=KEY, how="inner")
            comps = {}
            for a in ("G14@BLUP20", "FLAT@BLUP20", "LINE", "NNC@SELECTED"):
                for k in [k for k in K_GRID if 1 <= k <= K]:
                    if f"{a}|k{k}#{pol}" in m.columns and f"{a}|k{k}" in m.columns:
                        comps[f"{pol}_vs_RANDOM@{a}@k{k}@{model}"] = (f"{a}|k{k}", f"{a}|k{k}#{pol}")
            if comps:
                c = contrasts(m, comps, replicates=replicates)
                c.insert(0, "policy", pol); c.insert(0, "model", model)
                rows.append(c)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["p_bh"] = INF.benjamini_hochberg(out["p_two_sided"].to_numpy())
        out["family_size"] = int(len(out))
    return out


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def figures(label: str, out: Path, design: str = "BP", policy: str = "DOPT_Q", log=print) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    made = []
    path = bootstrap.PREDICTIONS / label / f"pairs_{design}.parquet"
    if not path.exists():
        avail = sorted((bootstrap.PREDICTIONS / label).glob("pairs_*.parquet"))
        if not avail:
            return made
        path = avail[0]; design = path.stem.split("_", 1)[1]
        log(f"figures: design BP absent for {label}; drawing {design}")
    table = pd.read_parquet(path)
    for model in sorted(table.model.unique()):
        for K in COHORT_KS[model]:
            w = wide_common(table, K, policy, model)
            if w.empty:
                continue
            cols = [f"{a}|k{k}" for a in MAIN_ARMS for k in K_GRID if k <= K and f"{a}|k{k}" in w.columns]
            lv = levels(w, cols, replicates=2000)
            for stat in ("R2", "MAE"):
                fig, ax = plt.subplots(figsize=(6.4, 4.2))
                for a in MAIN_ARMS:
                    sub = lv[lv.arm_base == a].sort_values("k")
                    if sub.empty:
                        continue
                    ax.errorbar(sub.k, sub[stat], yerr=[sub[stat] - sub[f"{stat}_ci_low"], sub[f"{stat}_ci_high"] - sub[stat]], marker="o", capsize=3, label=a, lw=1)
                ax.set_xlabel("k measured separation factors"); ax.set_ylabel(f"pooled {stat}, adjacent queries")
                ax.set_title(f"{design}, {model} model, C{K}, {policy}", fontsize=9)
                if stat == "R2":
                    ax.axhline(0, color="k", lw=0.5)
                ax.legend(fontsize=6); fig.tight_layout()
                p = out / f"fig_{stat}_vs_k_{design}_{model}_C{K}_{policy}.png"; fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
            for ax, k in zip(axes, sorted({0, 3, K})):
                c = f"G14@BLUP20|k{k}"
                if c in w.columns:
                    ax.scatter(w.y, w[c], s=4, alpha=0.4); lim = [w.y.min() - 0.1, w.y.max() + 0.1]
                    ax.plot(lim, lim, "k--", lw=0.7); r, _ = M.r2(w.y, w[c]); ax.set_title(f"k={k}, R²={r:.2f}")
                    ax.set_xlabel("observed log SF"); ax.set_ylabel("predicted (G14@BLUP20)")
            fig.tight_layout(); p = out / f"fig_pred_vs_obs_{design}_{model}_C{K}_{policy}.png"; fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))
            fig, ax = plt.subplots(figsize=(6, 4))
            for k in sorted({0, 3, K}):
                c = f"G14@BLUP20|k{k}"
                if c in w.columns:
                    ax.hist((w[c] - w.y), bins=60, histtype="step", label=f"k={k}", density=True)
            ax.set_xlabel("residual (predicted − observed)"); ax.legend(); fig.tight_layout()
            p = out / f"fig_residuals_{design}_{model}_C{K}_{policy}.png"; fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))
            c = "G14@BLUP20|k3" if K >= 3 else "G14@BLUP20|k0"
            if c in w.columns:
                pp = per_pair(w, c)
                fig, ax = plt.subplots(figsize=(8, 3.6))
                ax.bar(pp.pair, pp.MAE, label="MAE, G14@BLUP20 at k=3"); ax.plot(pp.pair, pp.MAE_FLAT, "k_", label="FLAT MAE", markersize=12)
                ax.set_ylabel("log SF"); ax.legend(fontsize=7); ax.tick_params(axis="x", rotation=45); fig.tight_layout()
                p = out / f"fig_per_pair_{design}_{model}_C{K}_{policy}.png"; fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))
    log(f"figures: {len(made)}")
    return made


# --------------------------------------------------------------------------------------
# Setting B (exploratory): condition adaptation inside one extractant, COPY is the null
# --------------------------------------------------------------------------------------
def setting_b_tables(label: str, out: Path, *, replicates: int = 2000, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    made = {}
    for p in sorted((bootstrap.PREDICTIONS / label).glob("settingB_*.parquet")):
        design = p.stem.split("_", 1)[1]
        t = pd.read_parquet(p)
        if t.empty:
            continue
        t = t.rename(columns={"unit": "cell_group"})
        t["model"] = "settingB"
        arms = [c for c in t.columns if c in ("COPY", "LINE") or "@" in c]
        lv = INF.pooled_levels(t, arms, replicates=replicates)
        lv = lv.pivot(index="arm", columns="statistic", values=["point", "ci_low", "ci_high", "seed_sd"])
        lv.columns = [f"{s}_{q}" if q != "point" else s for q, s in lv.columns]
        extra = {}
        for a in arms:
            per = [M.pooled(g.rename(columns={a: "pred"}), "pred") for _, g in t.groupby("split_seed")]
            extra[a] = {k: float(np.nanmean([x[k] for x in per])) for k in per[0] if isinstance(per[0][k], (int, float))}
        ex = pd.DataFrame(extra).T
        lv = lv.join(ex[[c for c in ex.columns if c not in lv.columns]]).reset_index()
        lv.insert(0, "design", design)
        lv["n_units"] = t.cell_group.nunique(); lv["n_extractants"] = t.extractant.nunique(); lv["n_chemotypes"] = t.chemotype.nunique()
        lv["kish_chemotypes_pairs"] = kish(t.groupby("chemotype").size())
        lv.to_csv(out / f"settingB_levels_{design}.csv", index=False)
        comps = {f"{a}_vs_COPY": ("COPY", a) for a in arms if a != "COPY"}
        comps["COPY_vs_FLAT"] = ("FLAT@ZERO", "COPY")
        c = INF.pooled_contrasts(t, comps, replicates=replicates)
        rows = []
        for a in arms:
            if a == "COPY":
                continue
            r = INF.macro_mae_contrast(t, "COPY", a, replicates=replicates); r["name"] = f"{a}_vs_COPY"; rows.append(r)
        mc = pd.DataFrame(rows)
        c.insert(0, "design", design); mc.insert(0, "design", design)
        c.to_csv(out / f"settingB_contrasts_pooled_{design}.csv", index=False)
        mc.to_csv(out / f"settingB_contrasts_macro_{design}.csv", index=False)
        made[design] = {"rows": int(len(t)), "units": int(t.cell_group.nunique())}
        log(f"  setting B {design}: {len(t)} rows, {t.cell_group.nunique()} units")
    return made
