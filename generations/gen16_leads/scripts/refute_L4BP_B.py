"""Refutation of claim L4BP, lens B (statistics, confounds, generalisation).

Stage `stats`  -- no refit.  Reproduces the registered ABC contrast from the lead's own
                 per_extractant_abc.parquet under all five designs; per-seed ABC; leave-one-
                 chemotype-out and leave-one-PUBLICATION-out sweeps; the diglycolamide split;
                 band concentration (n_metals, |a|, cell count, chemotype size); effective number
                 of independent units (Kish, block macro); BH within the registered family and
                 within the whole L4 output; the AOPT-order-vs-chemotype-size correlation.

Stage `size`   -- refit.  Adds a SIZE order (chemotypes in descending well-determined-cell count;
                 no features, no model, no descriptors -- the cheapest sensible competitor to an
                 A-optimal ordering) at the same six budgets, all five designs, the same folds and
                 the same untouched test index, and scores it through the same metric path.
                 Contrasts: AOPT_vs_SIZE, SIZE_vs_RANDOM.

Outputs under gen16_leads/results/refutation/L4BP/B/.  Nothing under gen13_/gen14_/gen15_/gen16_
lead directories is modified; the lead's l4_acq module is imported, not copied.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen14_direction", ROOT / "generations" / "gen15_curve",
           ROOT / "generations" / "gen16_leads"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen13sep.inference import paired_contrasts, _per_unit  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402
from gen13sep.splits import all_folds  # noqa: E402
from gen14.dirbench import load  # noqa: E402
from gen15.valuebench import DESIGNS, MIN_METALS  # noqa: E402
from gen16 import l4_acq as L  # noqa: E402

L4 = ROOT / "generations" / "gen16_leads" / "results" / "L4"
OUT = ROOT / "generations" / "gen16_leads" / "results" / "refutation" / "L4BP" / "B"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "refute_L4BP_B.log"
REG = {"AOPT_vs_RANDOM": ("RANDOM", "AOPT"), "AOPT_vs_MAXMIN": ("MAXMIN", "AOPT"),
       "UNCERT_vs_RANDOM": ("RANDOM", "UNCERT"), "UNCERT_vs_MAXMIN": ("MAXMIN", "UNCERT")}
SHOW = ["design", "comparison", "point", "ci95_low", "ci95_high", "bca_low", "bca_high",
        "p_two_sided", "seeds_positive", "loco_sign_stable", "loco_min", "loco_max",
        "n_units", "block_macro", "passes_P1"]


def say(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(str(msg) + "\n")


def bh(p: pd.Series) -> pd.Series:
    p = p.astype(float)
    ok = p.dropna()
    if not len(ok):
        return p
    adj = (ok * len(ok) / ok.rank(method="first")).clip(upper=1.0)
    return adj.loc[ok.sort_values(ascending=False).index].cummin().reindex(p.index)


def kish(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum())


# ======================================================================================
def stage_stats() -> None:
    LOG.write_text("", encoding="utf-8")
    bench = load()
    fr = bench.frame
    ext_series = fr.extractant.astype(str)
    chemo = fr.chemotype.astype(str)
    dga = chemo.value_counts().idxmax()
    dga_ext = sorted(set(ext_series[chemo == dga]))
    say(f"largest chemotype {dga}: {(chemo == dga).sum()} cells, {len(dga_ext)} extractants")

    ABC = pd.read_parquet(L4 / "per_extractant_abc.parquet")
    say(f"ABC frame {ABC.shape}; n_budgets unique {sorted(ABC.n_budgets.unique())}")

    # ---- 1. reproduce the registered contrasts, all five designs -----------------------
    rows = []
    for d in DESIGNS:
        pe = ABC[ABC.design == d].rename(columns={"order": "arm"})
        c = paired_contrasts(pe, REG, value="mae_all")
        c.insert(0, "design", d)
        rows.append(c)
    R = pd.concat(rows, ignore_index=True)
    R["family"] = "registered"
    R["p_bh_registered20"] = bh(R["p_two_sided"])
    R.to_csv(OUT / "reproduced_registered.csv", index=False)
    say("\n=== 1. reproduced registered ABC contrasts (mae_all), five designs ===")
    say(R[SHOW + ["p_bh_registered20"]].round(4).to_string(index=False))

    lead = pd.read_csv(L4 / "contrasts_abc.csv")
    lead_reg = lead[(lead.family == "registered")][["design", "comparison", "point", "ci95_low",
                                                    "ci95_high", "p_two_sided", "seeds_positive",
                                                    "loco_sign_stable", "passes_P1",
                                                    "p_bh_within_lead_family"]]
    m = R[["design", "comparison", "point", "p_two_sided"]].merge(
        lead_reg, on=["design", "comparison"], suffixes=("_mine", "_lead"))
    m["abs_diff_point"] = (m["point_mine"] - m["point_lead"]).abs()
    say(f"\nmax |point_mine - point_lead| over the 20 registered rows: {m.abs_diff_point.max():.3e}")
    m.to_csv(OUT / "reproduction_diff.csv", index=False)

    # ---- 2. per-seed ABC, individually -------------------------------------------------
    seed_rows = []
    for d in DESIGNS:
        for s in sorted(ABC.split_seed.unique()):
            pe = ABC[(ABC.design == d) & (ABC.split_seed == s)].rename(columns={"order": "arm"})
            t, _ = _per_unit(pe, "mae_all")
            for lab, (ref, cand) in REG.items():
                dd = (t[ref] - t[cand]).dropna()
                seed_rows.append({"design": d, "split_seed": int(s), "comparison": lab,
                                  "abc": float(dd.mean()), "n_units": int(len(dd))})
    S = pd.DataFrame(seed_rows)
    S.to_csv(OUT / "per_seed_abc.csv", index=False)
    say("\n=== 2. per-seed ABC (mae_all), AOPT_vs_RANDOM ===")
    say(S[S.comparison == "AOPT_vs_RANDOM"].pivot_table(
        index="design", columns="split_seed", values="abc").round(4).to_string())

    # ---- 3. diglycolamides ------------------------------------------------------------
    rows = []
    for d in DESIGNS:
        pe = ABC[ABC.design == d].rename(columns={"order": "arm"})
        for tag, sub in (("noDGA", pe[~pe.extractant.isin(dga_ext)]),
                         ("DGAonly", pe[pe.extractant.isin(dga_ext)])):
            c = paired_contrasts(sub, REG, value="mae_all")
            if c.empty:
                continue
            c.insert(0, "design", d)
            c["comparison"] = c["comparison"] + "_" + tag
            rows.append(c)
    D = pd.concat(rows, ignore_index=True)
    D.to_csv(OUT / "dga_split.csv", index=False)
    say("\n=== 3. diglycolamide split (scoring units only; training unchanged) ===")
    say(D[D.comparison.str.startswith("AOPT_vs_RANDOM")][SHOW].round(4).to_string(index=False))

    # ---- 4. leave-one-publication-out --------------------------------------------------
    pubs = (fr.assign(extractant=ext_series, publication_id=fr.publication_id.astype(str))
            .groupby("extractant")["publication_id"].apply(lambda s: sorted(set(s))))
    modal = (fr.assign(extractant=ext_series, publication_id=fr.publication_id.astype(str))
             .groupby(["extractant", "publication_id"]).size().rename("n").reset_index()
             .sort_values(["extractant", "n", "publication_id"], ascending=[True, False, True])
             .drop_duplicates("extractant").set_index("extractant")["publication_id"])
    all_pubs = sorted(set(fr.publication_id.astype(str)))
    say(f"\n=== 4. leave-one-publication-out: {len(all_pubs)} publications, "
        f"{ext_series.nunique()} extractants ===")
    lopo_rows = []
    for d in DESIGNS:
        pe = ABC[ABC.design == d].rename(columns={"order": "arm"})
        t, _ = _per_unit(pe, "mae_all")
        base = (t["RANDOM"] - t["AOPT"]).dropna()
        for p in all_pubs:
            drop_any = {e for e, ps in pubs.items() if p in ps}
            drop_modal = set(modal[modal == p].index)
            for mode, dr in (("any_cell", drop_any), ("modal", drop_modal)):
                keep = base[~base.index.isin(dr)]
                lopo_rows.append({"design": d, "publication_id": p, "mode": mode,
                                  "n_dropped_extractants": len(dr & set(base.index)),
                                  "n_units_left": int(len(keep)),
                                  "abc": float(keep.mean()) if len(keep) else np.nan})
        lopo_rows.append({"design": d, "publication_id": "(none)", "mode": "any_cell",
                          "n_dropped_extractants": 0, "n_units_left": int(len(base)),
                          "abc": float(base.mean())})
    P = pd.DataFrame(lopo_rows)
    P.to_csv(OUT / "lopo_publication.csv", index=False)
    for d in DESIGNS:
        for mode in ("any_cell", "modal"):
            s = P[(P.design == d) & (P["mode"] == mode) & (P.publication_id != "(none)")]
            b = float(P[(P.design == d) & (P.publication_id == "(none)")]["abc"].iloc[0])
            flips = int((np.sign(s.abc) != np.sign(b)).sum())
            worst = s.loc[s.abc.abs().idxmin()] if len(s) else None
            say(f"  {d:3s} {mode:9s} base {b:+.4f}  LOPO min {s.abc.min():+.4f} "
                f"max {s.abc.max():+.4f}  sign flips {flips}/{len(s)}  "
                f"largest single-publication drop -> {s.abc.min():+.4f} "
                f"(pub {s.loc[s.abc.idxmin(),'publication_id']}, "
                f"{int(s.loc[s.abc.idxmin(),'n_dropped_extractants'])} extractants)")

    # ---- 5. band concentration ---------------------------------------------------------
    ecell = (fr.assign(extractant=ext_series)
             .groupby("extractant")
             .agg(n_cells=("cell_id", "size"),
                  n_metals_max=("n_metals", "max"),
                  n_metals_mean=("n_metals", "mean"),
                  n_pubs=("publication_id", "nunique")))
    amp = pd.DataFrame({"extractant": ext_series.to_numpy(), "a": bench.coef[:, 0],
                        "b": bench.coef[:, 1], "n_metals": fr.n_metals.to_numpy()})
    rich_amp = amp[amp.n_metals >= MIN_METALS].groupby("extractant")[["a", "b"]].mean()
    ecell = ecell.join(rich_amp)
    ecell["abs_a"] = ecell["a"].abs()
    chsize = chemo.value_counts().rename("chemotype_cells")
    chext = (fr.assign(extractant=ext_series).groupby("chemotype")["extractant"].nunique()
             .rename("chemotype_extractants"))
    band_rows = []
    for d in DESIGNS:
        pe = ABC[ABC.design == d].rename(columns={"order": "arm"})
        t, blk = _per_unit(pe, "mae_all")
        delta = (t["RANDOM"] - t["AOPT"]).dropna()
        df = pd.DataFrame({"delta": delta})
        df = df.join(ecell, how="left")
        df["chemotype"] = blk.reindex(df.index).astype(str)
        df = df.join(chsize, on="chemotype").join(chext, on="chemotype")
        df.insert(0, "design", d)
        band_rows.append(df.reset_index())
    B = pd.concat(band_rows, ignore_index=True)
    B.to_csv(OUT / "per_extractant_delta_bands.csv", index=False)
    say("\n=== 5. band concentration of the per-extractant ABC delta (RANDOM - AOPT) ===")
    from scipy.stats import spearmanr
    for d in DESIGNS:
        s = B[B.design == d]
        line = [f"  {d:3s} n={len(s)}"]
        for col in ("n_metals_max", "n_cells", "abs_a", "chemotype_cells", "chemotype_extractants",
                    "n_pubs"):
            v = s[[col, "delta"]].dropna()
            rho, pv = spearmanr(v[col], v["delta"])
            line.append(f"{col} rho={rho:+.3f}(p={pv:.3f})")
        say("  ".join(line))
    say("\n  BP tertiles of the delta by band:")
    s = B[B.design == "BP"].copy()
    for col in ("n_metals_max", "n_cells", "abs_a", "chemotype_cells"):
        v = s[[col, "delta"]].dropna().copy()
        v["tertile"] = pd.qcut(v[col].rank(method="first"), 3, labels=["low", "mid", "high"])
        g = v.groupby("tertile", observed=True)["delta"].agg(["mean", "size"])
        say(f"    {col:22s} " + "  ".join(f"{i}: {r['mean']:+.4f} (n={int(r['size'])})"
                                          for i, r in g.iterrows()))

    # ---- 6. effective number of independent units --------------------------------------
    say("\n=== 6. effective number of independent units ===")
    for d in ("BP",):
        pe = ABC[ABC.design == d].rename(columns={"order": "arm"})
        t, blk = _per_unit(pe, "mae_all")
        delta = (t["RANDOM"] - t["AOPT"]).dropna()
        b2 = blk.reindex(delta.index).astype(str)
        per_block = delta.groupby(b2).mean()
        counts = delta.groupby(b2).size()
        say(f"  {d}: {len(delta)} extractant units in {per_block.size} chemotype blocks; "
            f"Kish n_eff over block sizes = {kish(counts.to_numpy()):.2f}")
        say(f"  {d}: extractant-weighted point = {delta.mean():+.4f}; "
            f"chemotype-weighted (block_macro) = {per_block.mean():+.4f}")
        say(f"  {d}: blocks with positive mean delta {int((per_block > 0).sum())}/"
            f"{per_block.size}; largest block {counts.idxmax()} n={int(counts.max())} "
            f"mean delta {per_block[counts.idxmax()]:+.4f}")
        pb = per_block.sort_values()
        say("  {d}: five most negative blocks: ".format(d=d)
            + ", ".join(f"{i}:{v:+.3f}(n={int(counts[i])})" for i, v in pb.head(5).items()))
        say("  {d}: five most positive blocks: ".format(d=d)
            + ", ".join(f"{i}:{v:+.3f}(n={int(counts[i])})" for i, v in pb.tail(5).items()))
        per_block.rename("mean_delta").to_frame().join(counts.rename("n_extractants")).to_csv(
            OUT / "per_chemotype_delta_BP.csv")

    # ---- 7. BH across the whole L4 output ---------------------------------------------
    say("\n=== 7. multiplicity ===")
    files = ["contrasts_abc.csv", "contrasts_per_budget.csv", "contrasts_cellmatched.csv",
             "contrasts_nodga.csv"]
    allc = []
    for f in files:
        c = pd.read_csv(L4 / f)
        c["src"] = f
        allc.append(c)
    AC = pd.concat(allc, ignore_index=True)
    AC["p_bh_all_L4"] = bh(AC["p_two_sided"])
    AC.to_csv(OUT / "all_L4_contrasts_bh.csv", index=False)
    say(f"  rows across the four L4 contrast files: {len(AC)} "
        f"(registered {int((AC.family == 'registered').sum())})")
    tgt = AC[(AC.comparison == "AOPT_vs_RANDOM") & (AC.src == "contrasts_abc.csv")]
    say("  AOPT_vs_RANDOM raw p, BH within registered 20, BH across all L4 rows:")
    for _, r in tgt.iterrows():
        say(f"    {r['design']:3s} p={r['p_two_sided']:.5f} "
            f"BH20={r['p_bh_within_lead_family']:.5f} BH{len(AC)}={r['p_bh_all_L4']:.5f}")

    # ---- 8. does the AOPT order track chemotype size? ----------------------------------
    say("\n=== 8. AOPT pick position vs chemotype size (features-only criterion?) ===")
    orders = pd.read_parquet(L4 / "orders.parquet")
    orders["chemotype"] = orders["chemotype"].astype(str)
    rich = fr.n_metals.to_numpy() >= MIN_METALS
    groups = bench.groups.astype(str)
    rows = []
    for d in DESIGNS:
        folds = all_folds(fr, design=d)
        for f in folds:
            g = groups[f.train_index]
            r = rich[f.train_index]
            sz = pd.Series(g).value_counts().rename("cells")
            szr = pd.Series(g[r]).value_counts().rename("rich_cells")
            for order in ("AOPT", "UNCERT", "MAXMIN", "RANDOM"):
                sub = orders[(orders.design == d) & (orders.split_seed == f.seed)
                             & (orders.fold == f.fold) & (orders.order == order) & (orders.draw == 0)]
                if sub.empty:
                    continue
                v = sub.sort_values("position")
                pos = v["position"].to_numpy()
                cells = sz.reindex(v.chemotype).fillna(0).to_numpy()
                rcells = szr.reindex(v.chemotype).fillna(0).to_numpy()
                rho_c = spearmanr(pos, cells).statistic
                rho_r = spearmanr(pos, rcells).statistic
                rows.append({"design": d, "split_seed": f.seed, "fold": f.fold, "order": order,
                             "spearman_pos_vs_cells": rho_c, "spearman_pos_vs_rich_cells": rho_r})
    O = pd.DataFrame(rows)
    O.to_csv(OUT / "order_vs_size.csv", index=False)
    say(O.groupby(["design", "order"])[["spearman_pos_vs_cells", "spearman_pos_vs_rich_cells"]]
        .mean().round(3).to_string())

    say("\nSTAGE stats DONE")


# ======================================================================================
def order_size(chemos, groups_pool, rich_pool) -> list[str]:
    """Descending well-determined-cell count, ties by descending total cells then by name.

    Uses no features, no descriptors and no model: the cheapest possible ordering that
    exploits training-set volume.
    """
    chemos = sorted(chemos)
    rc = {c: int(((groups_pool == c) & rich_pool).sum()) for c in chemos}
    ac = {c: int((groups_pool == c).sum()) for c in chemos}
    return sorted(chemos, key=lambda c: (-rc[c], -ac[c], c))


def stage_size() -> None:
    bench = load()
    P = L.prepare(bench)
    fr = bench.frame
    groups = bench.groups.astype(str)
    rich = fr.n_metals.to_numpy() >= MIN_METALS
    say(f"\n=== 9. SIZE order (cheapest competitor): refit, all five designs ===")
    pe_parts, curve_rows, order_rows, cellrows = [], [], [], []
    for design in DESIGNS:
        t0 = time.time()
        folds = all_folds(fr, design=design)
        parts = {k: [] for k in L.BUDGETS}
        for f in folds:
            pairs = L.pair_table(P, f)
            if pairs.empty:
                continue
            g = groups[f.train_index]
            r = rich[f.train_index]
            chemos = sorted(set(g))
            o = order_size(chemos, g, r)
            for pos, c in enumerate(o):
                order_rows.append({"design": design, "split_seed": f.seed, "fold": f.fold,
                                   "order": "SIZE", "draw": 0, "position": pos, "chemotype": c})
            base = pairs.drop(columns=["ia", "ib", "cell_local"])
            for k in L.BUDGETS:
                if len(chemos) < k:
                    continue
                thin = L.thin_fold(f, bench.groups, o[:k])
                coef, fb = L.fit_g14_guarded(P, thin, design)
                t = base.copy()
                t["SIZE|0"] = L.pair_predictions(coef, pairs, P)
                parts[k].append(t)
                m = np.isin(g, o[:k])
                cellrows.append({"design": design, "split_seed": f.seed, "fold": f.fold,
                                 "budget": k, "n_train_cells": int(m.sum()),
                                 "n_rich_train_cells": int((m & r).sum()), "flat_fallback": int(fb)})
        for k, plist in parts.items():
            tab = pd.concat(plist, ignore_index=True)
            arms = ["SIZE|0"]
            pe = per_extractant(tab, arms)
            s = summarise(pe, tab, arms).set_index("arm")
            pe = pe.assign(design=design, budget=str(k), order="SIZE", draw=0)
            pe_parts.append(pe)
            curve_rows.append({"design": design, "order": "SIZE", "budget": str(k),
                               "macro_mae": float(s.loc["SIZE|0", "macro_mae_extractant"]),
                               "macro_sign_acc": float(s.loc["SIZE|0", "macro_sign_acc_strong"]),
                               "n_units": int(s.loc["SIZE|0", "n_units_mae_all"])})
        say(f"  [{design}] SIZE done in {time.time() - t0:.0f}s")
    PE = pd.concat(pe_parts, ignore_index=True)
    PE.to_parquet(OUT / "_pe_SIZE.parquet", index=False)
    C = pd.DataFrame(curve_rows)
    C.to_csv(OUT / "curve_SIZE.csv", index=False)
    CE = pd.DataFrame(cellrows)
    CE.to_csv(OUT / "cells_SIZE.csv", index=False)
    say("\n  SIZE learning curve (macro MAE):")
    say(C.pivot_table(index="design", columns="budget", values="macro_mae").round(4).to_string())
    say("\n  SIZE mean training cells per budget:")
    say(CE.groupby(["design", "budget"])[["n_train_cells", "n_rich_train_cells"]].mean()
        .round(1).to_string())
    say(f"  SIZE FLAT fallbacks total: {int(CE.flat_fallback.sum())}")

    # ---- join with the lead's kept per-extractant tables and contrast -------------------
    VALUE_COLS = ["mae_all", "mae_adjacent", "mae_far", "sign_acc_strong", "pair_spearman"]
    keys = ["design", "budget", "order", "split_seed", "extractant", "chemotype"]
    rows = []
    for design in DESIGNS:
        lead_pe = pd.read_parquet(L4 / "_pe_keep" / f"_pe_{design}.parquet")
        lead_pe = lead_pe[lead_pe.budget != "full"]
        mine = PE[PE.design == design]
        both = pd.concat([lead_pe[keys + VALUE_COLS], mine[keys + VALUE_COLS]], ignore_index=True)
        DA = both.groupby(keys)[VALUE_COLS].mean().reset_index()
        abc = (DA.groupby(["order", "split_seed", "extractant", "chemotype"])[VALUE_COLS]
               .mean().reset_index().rename(columns={"order": "arm"}))
        nb = (DA.groupby(["order", "split_seed", "extractant"])["budget"].nunique())
        assert nb.min() == 6 and nb.max() == 6, f"budget set not shared in {design}: {nb.value_counts()}"
        comps = {"AOPT_vs_SIZE": ("SIZE", "AOPT"), "SIZE_vs_RANDOM": ("RANDOM", "SIZE"),
                 "SIZE_vs_MAXMIN": ("MAXMIN", "SIZE"), "AOPT_vs_RANDOM": ("RANDOM", "AOPT")}
        c = paired_contrasts(abc, comps, value="mae_all")
        c.insert(0, "design", design)
        rows.append(c)
        abc.to_parquet(OUT / f"_abc_with_size_{design}.parquet", index=False)
    CS = pd.concat(rows, ignore_index=True)
    CS["family"] = "exploratory"
    CS.to_csv(OUT / "contrasts_size.csv", index=False)
    say("\n=== 9. contrasts against the SIZE order (exploratory) ===")
    say(CS[SHOW].round(4).to_string(index=False))
    say("\nSTAGE size DONE")


# ======================================================================================
def stage_more() -> None:
    """n_metals stratification, the DGA-only delta, AOPT/SIZE order overlap, budget decomposition."""
    from scipy.stats import spearmanr
    bench = load()
    fr = bench.frame
    ext_series = fr.extractant.astype(str)
    chemo = fr.chemotype.astype(str)
    dga = chemo.value_counts().idxmax()
    dga_ext = sorted(set(ext_series[chemo == dga]))
    ABC = pd.read_parquet(L4 / "per_extractant_abc.parquet")
    nm = (fr.assign(extractant=ext_series).groupby("extractant")["n_metals"].max())

    say("\n=== 10. n_metals stratification of AOPT_vs_RANDOM (ABC, mae_all) ===")
    rows = []
    for d in DESIGNS:
        pe = ABC[ABC.design == d].rename(columns={"order": "arm"})
        for lo, hi, lab in ((2, 4, "n_metals<=4"), (5, 7, "n_metals 5-7"), (8, 14, "n_metals>=8")):
            keep = set(nm[(nm >= lo) & (nm <= hi)].index)
            sub = pe[pe.extractant.isin(keep)]
            c = paired_contrasts(sub, {"AOPT_vs_RANDOM": ("RANDOM", "AOPT")}, value="mae_all")
            if c.empty:
                say(f"  {d} {lab}: fewer than 2 chemotype blocks; not computable")
                continue
            c.insert(0, "design", d)
            c["stratum"] = lab
            rows.append(c)
    ST = pd.concat(rows, ignore_index=True)
    ST.to_csv(OUT / "stratified_n_metals.csv", index=False)
    say(ST[["design", "stratum", "point", "ci95_low", "ci95_high", "p_two_sided",
            "seeds_positive", "loco_sign_stable", "n_units", "passes_P1"]].round(4).to_string(index=False))
    # partial Spearman of delta with |a| given n_metals, and with n_metals given |a|
    say("\n  partial Spearman of the per-extractant delta (BP):")
    B = pd.read_csv(OUT / "per_extractant_delta_bands.csv")
    s = B[B.design == "BP"][["delta", "abs_a", "n_metals_max"]].dropna()
    r = s.rank()

    def partial(x, y, z):
        rxz = spearmanr(r[x], r[z]).statistic
        ryz = spearmanr(r[y], r[z]).statistic
        rxy = spearmanr(r[x], r[y]).statistic
        return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    say(f"    delta~|a| given n_metals : {partial('delta', 'abs_a', 'n_metals_max'):+.3f} "
        f"(raw {spearmanr(s.delta, s.abs_a).statistic:+.3f}, n={len(s)})")
    say(f"    delta~n_metals given |a| : {partial('delta', 'n_metals_max', 'abs_a'):+.3f} "
        f"(raw {spearmanr(s.delta, s.n_metals_max).statistic:+.3f}, n={len(s)})")

    say("\n=== 11. diglycolamide-only delta (single block: no bootstrap possible) ===")
    for d in DESIGNS:
        pe = ABC[ABC.design == d].rename(columns={"order": "arm"})
        t, _ = _per_unit(pe, "mae_all")
        delta = (t["RANDOM"] - t["AOPT"]).dropna()
        d_in = delta[delta.index.isin(dga_ext)]
        d_out = delta[~delta.index.isin(dga_ext)]
        say(f"  {d:3s} all {delta.mean():+.4f} (n={len(delta)}) | DGA-only {d_in.mean():+.4f} "
            f"(n={len(d_in)}) | noDGA {d_out.mean():+.4f} (n={len(d_out)})")
    nod = pd.read_csv(L4 / "contrasts_nodga.csv")
    say(f"  lead's contrasts_nodga.csv: {len(nod)} rows, comparisons "
        f"{sorted(set(nod.comparison))[:8]}")
    say(f"  rows whose comparison ends in _DGAonly: {int(nod.comparison.str.endswith('_DGAonly').sum())}")

    say("\n=== 12. AOPT order vs SIZE order: overlap of the first k chemotypes ===")
    orders = pd.read_parquet(L4 / "orders.parquet")
    orders["chemotype"] = orders["chemotype"].astype(str)
    mine = pd.read_csv(OUT / "cells_SIZE.csv")  # existence check only
    groups = bench.groups.astype(str)
    rich = fr.n_metals.to_numpy() >= MIN_METALS
    rows = []
    for d in DESIGNS:
        for f in all_folds(fr, design=d):
            g = groups[f.train_index]
            r = rich[f.train_index]
            chemos = sorted(set(g))
            so = order_size(chemos, g, r)
            ao = (orders[(orders.design == d) & (orders.split_seed == f.seed) & (orders.fold == f.fold)
                         & (orders.order == "AOPT")].sort_values("position")["chemotype"].tolist())
            rec = {"design": d, "split_seed": f.seed, "fold": f.fold,
                   "spearman_rank": spearmanr([so.index(c) for c in chemos],
                                              [ao.index(c) for c in chemos]).statistic}
            for k in L.BUDGETS:
                rec[f"jaccard_k{k}"] = len(set(so[:k]) & set(ao[:k])) / len(set(so[:k]) | set(ao[:k]))
            rows.append(rec)
    OV = pd.DataFrame(rows)
    OV.to_csv(OUT / "order_overlap_aopt_size.csv", index=False)
    say(OV.groupby("design")[["spearman_rank"] + [f"jaccard_k{k}" for k in L.BUDGETS]]
        .mean().round(3).to_string())

    say("\n=== 13. where BP's ABC comes from, by budget ===")
    pb = pd.read_csv(L4 / "contrasts_per_budget.csv")
    q = pb[(pb.design == "BP") & (pb.comparison.str.startswith("AOPT_vs_RANDOM_k"))
           & (pb.comparison.str.endswith("_mae_all"))][["budget", "point", "p_two_sided",
                                                        "seeds_positive", "passes_P1"]]
    q = q.sort_values("budget")
    say(q.round(4).to_string(index=False))
    say(f"  mean over the six budgets = {q.point.mean():+.5f} (registered ABC +0.0501)")
    curve = pd.read_csv(L4 / "curve.csv")
    cb = curve[(curve.design == "BP") & (curve.order == "RANDOM")][["budget", "macro_mae"]]
    say("  BP RANDOM macro MAE vs FLAT = 0.5885: "
        + ", ".join(f"k={int(r.budget)}:{r.macro_mae:.4f}"
                    + ("(worse than FLAT)" if r.macro_mae > 0.5885 else "")
                    for r in cb.itertuples()))
    worse = cb[cb.macro_mae > 0.5885].budget.astype(int).tolist()
    share = q[q.budget.isin(worse)].point.sum() / q.point.sum()
    say(f"  budgets where RANDOM is worse than predicting nothing: {worse}; "
        f"they contribute {100 * share:.0f} % of the registered ABC")
    say("\nSTAGE more DONE")


def stage_null2() -> None:
    """Mandatory check 5: re-run the lead's RANDOM null with a DIFFERENT seed.

    Ten fresh uniform random chemotype orders per fold, drawn from
    ``default_rng(model_seed*1000 + 700_000 + d)`` -- disjoint from the lead's
    ``+d`` (RANDOM) and ``+500_000+d`` (MAXMIN start) streams.  Designs BP (the selecting
    design, where the claim lives) and B (where it reverses).  No ``seeds=`` is passed to any
    bench call; the fold plans are the frozen discovery defaults.
    """
    N2 = 10
    bench = load()
    P = L.prepare(bench)
    fr = bench.frame
    VALUE_COLS = ["mae_all", "mae_adjacent", "mae_far", "sign_acc_strong", "pair_spearman"]
    say(f"\n=== 14. re-seeded RANDOM null (RANDOM2, {N2} fresh draws), designs BP and B ===")
    for design in ("BP", "B"):
        t0 = time.time()
        folds = all_folds(fr, design=design)
        parts = {k: [] for k in L.BUDGETS}
        for f in folds:
            pairs = L.pair_table(P, f)
            if pairs.empty:
                continue
            chemos = sorted(set(bench.groups[f.train_index].astype(str)))
            orders = {}
            for d in range(N2):
                rng = np.random.default_rng(int(f.model_seed) * 1000 + 700_000 + d)
                orders[d] = L.order_random(chemos, rng)
            base = pairs.drop(columns=["ia", "ib", "cell_local"])
            for k in L.BUDGETS:
                t = base.copy()
                for d in range(N2):
                    thin = L.thin_fold(f, bench.groups, orders[d][:k])
                    coef, _fb = L.fit_g14_guarded(P, thin, design)
                    t[f"RANDOM2|{d}"] = L.pair_predictions(coef, pairs, P)
                parts[k].append(t)
        pe_parts = []
        for k, plist in parts.items():
            tab = pd.concat(plist, ignore_index=True)
            arms = [f"RANDOM2|{d}" for d in range(N2)]
            pe = per_extractant(tab, arms)
            pe = pe.assign(design=design, budget=str(k), order="RANDOM2",
                           draw=pe["arm"].str.split("|", regex=False).str[1].astype(int))
            pe_parts.append(pe)
        PE2 = pd.concat(pe_parts, ignore_index=True)
        PE2.to_parquet(OUT / f"_pe_RANDOM2_{design}.parquet", index=False)
        say(f"  [{design}] RANDOM2 done in {time.time() - t0:.0f}s")

        keys = ["design", "budget", "order", "split_seed", "extractant", "chemotype"]
        lead_pe = pd.read_parquet(L4 / "_pe_keep" / f"_pe_{design}.parquet")
        lead_pe = lead_pe[lead_pe.budget != "full"]
        size_pe = pd.read_parquet(OUT / "_pe_SIZE.parquet")
        size_pe = size_pe[size_pe.design == design]
        both = pd.concat([lead_pe[keys + VALUE_COLS], size_pe[keys + VALUE_COLS],
                          PE2[keys + VALUE_COLS]], ignore_index=True)
        DA = both.groupby(keys)[VALUE_COLS].mean().reset_index()
        abc = (DA.groupby(["order", "split_seed", "extractant", "chemotype"])[VALUE_COLS]
               .mean().reset_index().rename(columns={"order": "arm"}))
        comps = {"AOPT_vs_RANDOM2": ("RANDOM2", "AOPT"), "SIZE_vs_RANDOM2": ("RANDOM2", "SIZE"),
                 "MAXMIN_vs_RANDOM2": ("RANDOM2", "MAXMIN"),
                 "AOPT_vs_RANDOM": ("RANDOM", "AOPT")}
        c = paired_contrasts(abc, comps, value="mae_all")
        c.insert(0, "design", design)
        c["family"] = "exploratory"
        say(c[SHOW].round(4).to_string(index=False))
        c.to_csv(OUT / f"contrasts_random2_{design}.csv", index=False)

        # percentile of AOPT / SIZE among the fresh draws, the lead's construction
        s = PE2
        rbar = (s.groupby(["budget", "split_seed", "extractant"])["mae_all"].mean().rename("rbar"))
        lead_all = lead_pe.copy()
        lead_all["budget"] = lead_all["budget"].astype(str)

        def macro_abc(block: pd.DataFrame) -> float:
            j = (block.groupby(["budget", "split_seed", "extractant"])["mae_all"].mean().rename("v"))
            j = pd.concat([j, rbar], axis=1, join="inner")
            per = (j["rbar"] - j["v"]).groupby(level=["split_seed", "extractant"]).mean()
            return float(per.groupby(level="extractant").mean().mean())
        draws = np.array([macro_abc(s[s.draw == i]) for i in range(N2)])
        v_aopt = macro_abc(lead_all[lead_all.order == "AOPT"])
        v_size = macro_abc(size_pe.assign(budget=size_pe.budget.astype(str)))
        say(f"  [{design}] RANDOM2 draw ABC: min {draws.min():+.4f} max {draws.max():+.4f} "
            f"sd {draws.std(ddof=1):.4f}; AOPT {v_aopt:+.4f} "
            f"(percentile {100 * (draws < v_aopt).mean():.0f}); SIZE {v_size:+.4f} "
            f"(percentile {100 * (draws < v_size).mean():.0f})")
    say("\nSTAGE null2 DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stats", "size", "more", "null2"])
    a = ap.parse_args()
    {"stats": stage_stats, "size": stage_size, "more": stage_more, "null2": stage_null2}[a.stage]()
