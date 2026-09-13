"""L4 exploratory diagnostics: the training-set-size confound and the diglycolamide check.

Everything in this file is **family = exploratory**.  It changes no registered endpoint; it measures
the two objections a refuter will raise against the registered AOPT-vs-RANDOM contrast:

1. **Is the comparison arm matched on training-set size?**  A chemotype budget is not a cell budget.
   A greedy A-optimal criterion is drawn to chemotypes that carry many well-determined cells, so at
   the same k the AOPT training set can be much larger than a random one.  ``cells_by_budget.csv``
   reports the cell counts; ``contrasts_cellmatched.csv`` re-runs the contrast against a RANDOM arm
   interpolated to AOPT's own training-cell count, per (extractant, seed, budget).
2. **Is the effect carried by the diglycolamides?**  The largest chemotype holds 375 of 521 cells.
   ``contrasts_nodga.csv`` drops every extractant of that chemotype from the scoring units and
   re-runs the registered comparisons on what is left.

Inputs: ``results/L4/_pe_keep/_pe_<design>.parquet`` (per-extractant scores with the draw dimension
intact, copied out of the main run) and ``results/L4/orders.parquet``.  No model is refitted here.

    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe gen16_leads/scripts/l4_diag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "gen13_separation", ROOT / "gen14_direction", ROOT / "gen15_curve",
           ROOT / "gen16_leads"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen13sep.inference import paired_contrasts  # noqa: E402
from gen13sep.splits import all_folds  # noqa: E402
from gen14.dirbench import load  # noqa: E402
from gen15.valuebench import DESIGNS, MIN_METALS  # noqa: E402
from gen16 import l4_acq as L  # noqa: E402

LEAD = "L4"
OUT = ROOT / "gen16_leads" / "results" / "L4"
KEEP = OUT / "_pe_keep"
LOG = OUT / "l4_diag.log"
REGISTERED = {"AOPT_vs_RANDOM": ("RANDOM", "AOPT"), "AOPT_vs_MAXMIN": ("MAXMIN", "AOPT"),
              "UNCERT_vs_RANDOM": ("RANDOM", "UNCERT"), "UNCERT_vs_MAXMIN": ("MAXMIN", "UNCERT")}
CONTRAST_COLS = ["design", "comparison", "point", "ci95_low", "ci95_high", "bca_low", "bca_high",
                 "p_two_sided", "seeds_positive", "loco_stable", "passes_P1", "family", "lead"]


def say(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def bh(p: pd.Series) -> pd.Series:
    p = p.astype(float)
    ok = p.dropna()
    if not len(ok):
        return p
    adj = (ok * len(ok) / ok.rank(method="first")).clip(upper=1.0)
    return adj.loc[ok.sort_values(ascending=False).index].cummin().reindex(p.index)


def finish(c: pd.DataFrame, design: str, family: str) -> pd.DataFrame:
    c = c.copy()
    c.insert(0, "design", design)
    c["family"] = family
    c["lead"] = LEAD
    c["loco_stable"] = c["loco_sign_stable"]
    return c


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    bench = load()
    fr = bench.frame
    groups = bench.groups.astype(str)
    rich = fr.n_metals.to_numpy() >= MIN_METALS
    ext = fr.extractant.astype(str).to_numpy()
    dga = fr.chemotype.astype(str).value_counts().idxmax()
    dga_ext = sorted(set(ext[fr.chemotype.astype(str) == dga]))
    say(f"largest chemotype {dga}: {int((fr.chemotype.astype(str) == dga).sum())} cells, "
        f"{len(dga_ext)} extractants (the diglycolamides)")

    orders = pd.read_parquet(OUT / "orders.parquet")
    orders["chemotype"] = orders["chemotype"].astype(str)

    # ---------------------------------------------------------------- cells per (order, budget)
    cell_rows, per_cell = [], []
    for design in DESIGNS:
        folds = {(f.seed, f.fold): f for f in all_folds(fr, design=design)}
        od = orders[orders.design == design]
        for (seed, fold), f in folds.items():
            tr = f.train_index
            g_tr = groups[tr]
            r_tr = rich[tr]
            sub = od[(od.split_seed == seed) & (od.fold == fold)]
            for (order, draw), blk in sub.groupby(["order", "draw"]):
                chosen = blk.sort_values("position")["chemotype"].tolist()
                for k in L.BUDGETS:
                    if len(chosen) < k:
                        continue
                    m = np.isin(g_tr, chosen[:k])
                    cell_rows.append({"design": design, "split_seed": seed, "fold": fold,
                                      "order": order, "draw": int(draw), "budget": k,
                                      "n_train_cells": int(m.sum()),
                                      "n_rich_train_cells": int((m & r_tr).sum()),
                                      "n_train_extractants": int(len(set(ext[tr][m])))})
        say(f"  [{design}] cell counts done")
    CELLS = pd.DataFrame(cell_rows)
    agg = (CELLS.groupby(["design", "order", "budget"])
           .agg(n_train_cells_mean=("n_train_cells", "mean"), n_train_cells_sd=("n_train_cells", "std"),
                n_rich_cells_mean=("n_rich_train_cells", "mean"),
                n_extractants_mean=("n_train_extractants", "mean")).reset_index())
    agg["lead"] = LEAD
    agg.to_csv(OUT / "cells_by_budget.csv", index=False)
    say("\n=== mean training cells per (design, order, budget) ===")
    say(agg.pivot_table(index=["design", "order"], columns="budget",
                        values="n_train_cells_mean").round(1).to_string())

    # -------------------------------------------- which extractant is tested in which fold, per seed
    weight_rows = []
    for design in DESIGNS:
        for f in all_folds(fr, design=design):
            for e, n in pd.Series(ext[f.test_index]).value_counts().items():
                weight_rows.append({"design": design, "split_seed": f.seed, "fold": f.fold,
                                    "extractant": e, "n_test_cells": int(n)})
    W = pd.DataFrame(weight_rows)

    # cells that an (extractant, seed) row's own fold(s) saw, per (order, budget)
    CW = CELLS.merge(W, on=["design", "split_seed", "fold"], how="inner")
    CW["wc"] = CW["n_train_cells"] * CW["n_test_cells"]
    EC = (CW.groupby(["design", "order", "draw", "budget", "split_seed", "extractant"])
          .agg(wc=("wc", "sum"), n=("n_test_cells", "sum")).reset_index())
    EC["cells"] = EC["wc"] / EC["n"]
    ECm = EC.groupby(["design", "order", "budget", "split_seed", "extractant"])["cells"].mean().reset_index()

    # ---------------------------------------------------------------- cell-matched RANDOM arm
    rows_cm, rows_dga = [], []
    for design in DESIGNS:
        pe_path = KEEP / f"_pe_{design}.parquet"
        if not pe_path.exists():
            say(f"  [{design}] no kept per-extractant table; cell-matched diagnostic skipped")
            continue
        PE = pd.read_parquet(pe_path)
        PE = PE[PE.budget != "full"].copy()
        PE["budget"] = PE["budget"].astype(int)
        DA = (PE.groupby(["order", "budget", "split_seed", "extractant", "chemotype"])["mae_all"]
              .mean().reset_index())
        cells = ECm[ECm.design == design].drop(columns="design")
        DA = DA.merge(cells, on=["order", "budget", "split_seed", "extractant"], how="left")

        key = ["split_seed", "extractant", "chemotype"]
        rnd = DA[DA.order == "RANDOM"].sort_values("cells")
        out = []
        for order in ("AOPT", "UNCERT", "MAXMIN"):
            cand = DA[DA.order == order]
            if cand.empty:
                continue
            for (seed, e, ch), blk in cand.groupby(key):
                r = rnd[(rnd.split_seed == seed) & (rnd.extractant == e)]
                if r.empty:
                    continue
                xs, ys = r["cells"].to_numpy(), r["mae_all"].to_numpy()
                matched = np.interp(blk["cells"].to_numpy(), xs, ys)
                out.append(pd.DataFrame({"split_seed": seed, "extractant": e, "chemotype": ch,
                                         "order": order, "budget": blk["budget"].to_numpy(),
                                         "mae_all": blk["mae_all"].to_numpy(),
                                         "mae_random_cellmatched": matched,
                                         "cells": blk["cells"].to_numpy()}))
        if not out:
            continue
        M = pd.concat(out, ignore_index=True)
        M.insert(0, "design", design)
        M.to_parquet(OUT / f"_cellmatched_{design}.parquet", index=False)
        agg2 = M.groupby(["order", "split_seed", "extractant", "chemotype"])[
            ["mae_all", "mae_random_cellmatched"]].mean().reset_index()
        frames = []
        for order in agg2.order.unique():
            s = agg2[agg2.order == order]
            frames.append(s.assign(arm=order, mae_all=s["mae_all"])[["arm", "split_seed", "extractant",
                                                                     "chemotype", "mae_all"]])
            frames.append(s.assign(arm=f"RANDOM_CELLMATCHED_{order}",
                                   mae_all=s["mae_random_cellmatched"])[["arm", "split_seed", "extractant",
                                                                         "chemotype", "mae_all"]])
        per_ext = pd.concat(frames, ignore_index=True)
        comps = {f"{o}_vs_RANDOM_CELLMATCHED": (f"RANDOM_CELLMATCHED_{o}", o) for o in agg2.order.unique()}
        c = paired_contrasts(per_ext, comps, value="mae_all")
        if not c.empty:
            rows_cm.append(finish(c, design, "exploratory"))

        # ------------------------------------------------------- diglycolamides removed
        base = DA.groupby(["order", "split_seed", "extractant", "chemotype"])["mae_all"].mean().reset_index()
        base = base.rename(columns={"order": "arm"})
        nod = base[~base.extractant.isin(dga_ext)]
        c = paired_contrasts(nod, REGISTERED, value="mae_all")
        if not c.empty:
            c["comparison"] = c["comparison"] + "_noDGA"
            rows_dga.append(finish(c, design, "exploratory"))
        c = paired_contrasts(base[base.extractant.isin(dga_ext)], REGISTERED, value="mae_all")
        if not c.empty:
            c["comparison"] = c["comparison"] + "_DGAonly"
            rows_dga.append(finish(c, design, "exploratory"))
        say(f"  [{design}] cell-matched and DGA diagnostics done "
            f"({nod.extractant.nunique()} non-DGA extractants of {base.extractant.nunique()})")

    if rows_cm:
        CM = pd.concat(rows_cm, ignore_index=True)
        CM["p_bh_within_lead_family"] = bh(CM["p_two_sided"])
        CM.to_csv(OUT / "contrasts_cellmatched.csv", index=False)
        say("\n=== cell-matched contrasts (exploratory) ===")
        say(CM[CONTRAST_COLS].round(4).to_string(index=False))
    if rows_dga:
        DG = pd.concat(rows_dga, ignore_index=True)
        DG["p_bh_within_lead_family"] = bh(DG["p_two_sided"])
        DG.to_csv(OUT / "contrasts_nodga.csv", index=False)
        say("\n=== diglycolamide split contrasts (exploratory) ===")
        say(DG[CONTRAST_COLS].round(4).to_string(index=False))

    # ---------------------------------------------------------------- what AOPT picks first
    first = (orders[(orders.order.isin(("AOPT", "UNCERT", "MAXMIN"))) & (orders.position < 6)]
             .groupby(["design", "order", "position"])["chemotype"]
             .agg(lambda s: s.value_counts().index[0]).reset_index())
    first["lead"] = LEAD
    first.to_csv(OUT / "first_picks.csv", index=False)
    say("\n=== modal first six picks ===")
    say(first.pivot_table(index=["design", "order"], columns="position", values="chemotype",
                          aggfunc="first").to_string())
    say("DONE")


if __name__ == "__main__":
    main()
