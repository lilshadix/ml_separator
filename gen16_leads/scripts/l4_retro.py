"""L4 retrospective acquisition simulation (PRE_REGISTRATION.md section 3, L4).

For every fold of every design the training chemotypes are ordered by RANDOM (20 draws), MAXMIN
(20 starts), AOPT and UNCERT; at budgets k in (6, 9, 12, 16, 20, 24) the G14 arm is refitted on the
first k chemotypes and the untouched held-out cells are scored through the frozen gen13 metric path.
A fold with fewer than k training chemotypes is skipped at that budget (for every order alike);
a thinned fit with < 10 well-determined cells or one direction class falls back to FLAT.

Primary endpoint: ABC = mean over budgets of [MAE_RANDOM(k) - MAE_ORDER(k)] per extractant, RANDOM
averaged over its draws, chemotype-blocked paired bootstrap (gen13sep.inference.paired_contrasts),
P1 with margin 0.02, five designs.  Registered: AOPT_vs_RANDOM, AOPT_vs_MAXMIN, UNCERT_vs_RANDOM,
UNCERT_vs_MAXMIN.  Everything else written here is family=exploratory.

    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe gen16_leads/scripts/l4_retro.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "gen13_separation", ROOT / "gen14_direction", ROOT / "gen15_curve",
           ROOT / "gen16_leads"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen13sep.inference import paired_contrasts  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402
from gen13sep.splits import all_folds  # noqa: E402
from gen14.dirbench import load  # noqa: E402
from gen15.valuebench import DESIGNS  # noqa: E402
from gen16 import l4_acq as L  # noqa: E402

LEAD = "L4"
OUT = ROOT / "gen16_leads" / "results" / "L4"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "l4_retro.log"
REGISTERED = {"AOPT_vs_RANDOM": ("RANDOM", "AOPT"), "AOPT_vs_MAXMIN": ("MAXMIN", "AOPT"),
              "UNCERT_vs_RANDOM": ("RANDOM", "UNCERT"), "UNCERT_vs_MAXMIN": ("MAXMIN", "UNCERT")}
EXTRA = {"MAXMIN_vs_RANDOM": ("RANDOM", "MAXMIN")}
VALUE_COLS = ["mae_all", "mae_adjacent", "mae_far", "sign_acc_strong", "pair_spearman"]
ANCHOR_BP = 0.5000794414203691


def say(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def bh(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p within one family (information only; the orchestrator adjusts
    across the fleet-wide families)."""
    p = p.astype(float)
    ok = p.dropna()
    n = len(ok)
    if n == 0:
        return p
    rank = ok.rank(method="first")
    adj = (ok * n / rank).clip(upper=1.0)
    # monotone from the largest p downwards, in the order of p
    idx = ok.sort_values(ascending=False).index
    s = adj.loc[idx].cummin()
    return s.reindex(p.index)


def run_design(P: L.Prepared, design: str) -> tuple[pd.DataFrame, pd.DataFrame, list, dict, dict]:
    bench = P.bench
    folds = all_folds(bench.frame, design=design)
    parts: dict = {k: [] for k in list(L.BUDGETS) + ["full"]}
    order_rows, skipped = [], []
    fallbacks: dict = {}
    t0 = time.time()
    for f in folds:
        pairs = L.pair_table(P, f)
        if pairs.empty:
            continue
        orders = L.fold_orders(P, f, n_draws=L.N_DRAWS)
        chemos = sorted(set(str(c) for c in bench.groups[f.train_index]))
        n_chem = len(chemos)
        for (order, draw), o in orders.items():
            for pos, c in enumerate(o):
                order_rows.append({"design": design, "split_seed": f.seed, "fold": f.fold,
                                   "order": order, "draw": draw, "position": pos,
                                   "chemotype": str(c), "n_train_chemotypes": n_chem})
        base = pairs.drop(columns=["ia", "ib", "cell_local"])
        for k in list(L.BUDGETS) + ["full"]:
            if k != "full" and n_chem < k:
                skipped.append({"design": design, "split_seed": f.seed, "fold": f.fold,
                                "budget": k, "n_train_chemotypes": n_chem})
                continue
            t = base.copy()
            if k == "full":
                coef, fb = L.fit_g14_guarded(P, L.thin_fold(f, bench.groups, chemos), design)
                t["FULL|0"] = L.pair_predictions(coef, pairs, P)
                t["FLAT|0"] = 0.0
                fallbacks[(design, "FULL", "full")] = fallbacks.get((design, "FULL", "full"), 0) + int(fb)
            else:
                for (order, draw), o in orders.items():
                    thin = L.thin_fold(f, bench.groups, [str(c) for c in o[:k]])
                    coef, fb = L.fit_g14_guarded(P, thin, design)
                    t[f"{order}|{draw}"] = L.pair_predictions(coef, pairs, P)
                    fallbacks[(design, order, k)] = fallbacks.get((design, order, k), 0) + int(fb)
            t["k_actual"] = n_chem if k == "full" else k
            parts[k].append(t)
    say(f"  [{design}] fits done in {time.time() - t0:.0f}s; skipped (fold, budget): {len(skipped)}")

    pe_parts, curve_rows = [], []
    for k, plist in parts.items():
        if not plist:
            continue
        t0 = time.time()
        tab = pd.concat(plist, ignore_index=True)
        arms = [c for c in tab.columns if "|" in c]
        pe = per_extractant(tab, arms)
        s = summarise(pe, tab, arms).set_index("arm")
        pe = pe.assign(design=design, budget=str(k),
                       order=pe["arm"].str.split("|", regex=False).str[0],
                       draw=pe["arm"].str.split("|", regex=False).str[1].astype(int))
        pe_parts.append(pe)
        n_folds_used = len(plist)
        for order in sorted(set(a.split("|")[0] for a in arms)):
            cols = [a for a in arms if a.startswith(order + "|")]
            mm = s.loc[cols, "macro_mae_extractant"]
            curve_rows.append({
                "design": design, "order": order, "budget": str(k),
                "k_actual": float(tab.groupby(["split_seed", "fold"])["k_actual"].first().mean()),
                "macro_mae": float(mm.mean()), "draw_sd": float(mm.std(ddof=1)) if len(cols) > 1 else np.nan,
                "macro_mae_far": float(s.loc[cols, "macro_mae_far"].mean()),
                "macro_sign_acc": float(s.loc[cols, "macro_sign_acc_strong"].mean()),
                "macro_pair_spearman": float(s.loc[cols, "macro_pair_spearman"].mean()),
                "n_draws": len(cols), "n_folds_used": n_folds_used,
                "n_folds_skipped": int(sum(1 for r in skipped if r["budget"] == k)),
                "n_flat_fallbacks": int(fallbacks.get((design, order, k), 0)),
                "n_units": int(s.loc[cols[0], "n_units_mae_all"]), "n_seeds": int(s.loc[cols[0], "n_seeds"]),
            })
        say(f"  [{design}] budget {k}: {len(arms)} arms scored in {time.time() - t0:.0f}s")
    PE = pd.concat(pe_parts, ignore_index=True)
    return PE, pd.DataFrame(curve_rows), order_rows, {"skipped": skipped}, fallbacks


def draw_average(PE: pd.DataFrame) -> pd.DataFrame:
    """Per (design, budget, order, split_seed, extractant): metrics averaged over draws."""
    keys = ["design", "budget", "order", "split_seed", "extractant", "chemotype"]
    return PE.groupby(keys)[VALUE_COLS].mean().reset_index()


def abc_frame(DA: pd.DataFrame) -> pd.DataFrame:
    """Per (design, order, split_seed, extractant): mean over budgets of each metric.  Skipped
    (fold, budget) pairs are absent for every order alike, so the budget set is shared."""
    sub = DA[DA["budget"] != "full"]
    keys = ["design", "order", "split_seed", "extractant", "chemotype"]
    g = sub.groupby(keys)
    out = g[VALUE_COLS].mean().reset_index()
    out["n_budgets"] = g["budget"].nunique().to_numpy()
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="smoke test: design BP only, 2 draws, results under results/L4/_dry/")
    args = ap.parse_args()
    global OUT, LOG, DESIGNS
    if args.dry:
        OUT = OUT / "_dry"; OUT.mkdir(parents=True, exist_ok=True); LOG = OUT / "l4_retro.log"
        DESIGNS = ("BP",); L.N_DRAWS = 2
    LOG.write_text("", encoding="utf-8")
    say(f"L4 retrospective simulation; designs {DESIGNS}; budgets {L.BUDGETS}; draws {L.N_DRAWS}")
    bench = load()
    P = L.prepare(bench)
    say(f"bench: {len(bench.frame)} cells, {bench.frame.extractant.nunique()} extractants, "
        f"{len(P.centroid)} chemotypes; TOPO39 columns {P.topo.shape[1]}")

    PEs, curves, orders, skipped, fallbacks = [], [], [], [], {}
    for d in DESIGNS:
        t0 = time.time()
        PE, C, O, S, F = run_design(P, d)
        PE.to_parquet(OUT / f"_pe_{d}.parquet", index=False)
        PEs.append(PE); curves.append(C); orders.extend(O); skipped.extend(S["skipped"]); fallbacks.update(F)
        say(f"design {d} done in {time.time() - t0:.0f}s")

    PE = pd.concat(PEs, ignore_index=True)
    curve = pd.concat(curves, ignore_index=True)
    curve.to_csv(OUT / "curve.csv", index=False)
    pd.DataFrame(orders).to_parquet(OUT / "orders.parquet", index=False)
    pd.DataFrame(skipped).to_csv(OUT / "skipped_fold_budgets.csv", index=False)
    pd.DataFrame([{"design": a, "order": b, "budget": str(c), "n_flat_fallbacks": v}
                  for (a, b, c), v in fallbacks.items()]).to_csv(OUT / "flat_fallbacks.csv", index=False)

    # ---- anchor check: the full-corpus refit through this path must be the frozen G14 ----
    full_bp = curve[(curve.design == "BP") & (curve.order == "FULL")]["macro_mae"].iloc[0]
    say(f"anchor: FULL under BP = {full_bp:.10f} (frozen {ANCHOR_BP:.10f})")
    if abs(full_bp - ANCHOR_BP) > 1e-9:
        raise SystemExit("G14 anchor not reproduced through the L4 path; stop")

    DA = draw_average(PE)
    DA.to_parquet(OUT / "per_extractant_by_budget.parquet", index=False)
    ABC = abc_frame(DA)
    ABC.to_parquet(OUT / "per_extractant_abc.parquet", index=False)

    # ---- registered contrasts on ABC (mae_all), plus exploratory companions ----
    rows = []
    for d in DESIGNS:
        pe = ABC[ABC.design == d].rename(columns={"order": "arm"})
        c = paired_contrasts(pe, REGISTERED, value="mae_all")
        c.insert(0, "design", d); c["family"] = "registered"; c["endpoint"] = "ABC_mae_all"
        rows.append(c)
        c = paired_contrasts(pe, EXTRA, value="mae_all")
        c.insert(0, "design", d); c["family"] = "exploratory"; c["endpoint"] = "ABC_mae_all"
        rows.append(c)
        for val in ("sign_acc_strong", "mae_far", "pair_spearman"):
            c = paired_contrasts(pe, {**REGISTERED, **EXTRA}, value=val)
            c.insert(0, "design", d); c["family"] = "exploratory"; c["endpoint"] = f"ABC_{val}"
            c["comparison"] = c["comparison"] + f"_{val}"
            rows.append(c)
    CA = pd.concat(rows, ignore_index=True)
    CA["lead"] = LEAD
    CA["p_bh_within_lead_family"] = np.nan
    for fam in ("registered", "exploratory"):
        m = CA.family == fam
        CA.loc[m, "p_bh_within_lead_family"] = bh(CA.loc[m, "p_two_sided"])
    CA.to_csv(OUT / "contrasts_abc.csv", index=False)

    # ---- per-budget contrasts (exploratory) ----
    rows = []
    for d in DESIGNS:
        for k in L.BUDGETS:
            pe = DA[(DA.design == d) & (DA.budget == str(k))].rename(columns={"order": "arm"})
            if pe.empty:
                continue
            for val in ("mae_all", "sign_acc_strong"):
                c = paired_contrasts(pe, {**REGISTERED, **EXTRA}, value=val)
                if c.empty:
                    continue
                c.insert(0, "design", d); c.insert(1, "budget", k)
                c["family"] = "exploratory"; c["endpoint"] = f"budget{k}_{val}"
                c["comparison"] = c["comparison"] + f"_k{k}_{val}"
                rows.append(c)
    CB = pd.concat(rows, ignore_index=True)
    CB["lead"] = LEAD
    CB["p_bh_within_lead_family"] = bh(CB["p_two_sided"])
    CB.to_csv(OUT / "contrasts_per_budget.csv", index=False)

    # ---- board: macro MAE per (design, arm=order@budget) ----
    board = curve.rename(columns={"macro_mae": "macro_mae_extractant"}).copy()
    board["arm"] = board["order"] + "@k" + board["budget"].astype(str)
    board.to_csv(OUT / "board.csv", index=False)

    # ---- percentile of AOPT's ABC among the 20 random draws (permutation-style null) ----
    prow = []
    sub = PE[PE.budget != "full"]
    for d in DESIGNS:
        s = sub[sub.design == d]
        rbar = (s[s.order == "RANDOM"].groupby(["budget", "split_seed", "extractant"])["mae_all"]
                .mean().rename("rbar"))
        def macro_abc(block: pd.DataFrame) -> float:
            j = block.set_index(["budget", "split_seed", "extractant"])["mae_all"].rename("v")
            j = pd.concat([j, rbar], axis=1, join="inner")
            per = (j["rbar"] - j["v"]).groupby(level=["split_seed", "extractant"]).mean()
            return float(per.groupby(level="extractant").mean().mean())
        draws = np.array([macro_abc(s[(s.order == "RANDOM") & (s.draw == i)]) for i in range(L.N_DRAWS)])
        for order in ("AOPT", "UNCERT"):
            v = macro_abc(s[s.order == order])
            pct = 100.0 * (float((draws < v).mean()) + 0.5 * float((draws == v).mean()))
            prow.append({"design": d, "order": order, "abc_macro": v, "percentile_among_random_draws": pct,
                         "random_draw_abc_min": float(draws.min()), "random_draw_abc_max": float(draws.max()),
                         "random_draw_abc_sd": float(draws.std(ddof=1)), "n_draws": L.N_DRAWS,
                         "family": "exploratory" if d != "BP" or order != "AOPT" else "registered_secondary",
                         "lead": LEAD})
        # maxmin starts against the same random-draw distribution
        starts = np.array([macro_abc(s[(s.order == "MAXMIN") & (s.draw == i)]) for i in range(L.N_DRAWS)])
        prow.append({"design": d, "order": "MAXMIN(mean over starts)", "abc_macro": float(starts.mean()),
                     "percentile_among_random_draws": 100.0 * float((draws < starts.mean()).mean()),
                     "random_draw_abc_min": float(draws.min()), "random_draw_abc_max": float(draws.max()),
                     "random_draw_abc_sd": float(draws.std(ddof=1)), "n_draws": L.N_DRAWS,
                     "family": "exploratory", "lead": LEAD})
    PCT = pd.DataFrame(prow)
    PCT.to_csv(OUT / "percentile_random.csv", index=False)

    summary = {
        "designs": list(DESIGNS), "budgets": list(L.BUDGETS), "n_draws": L.N_DRAWS,
        "anchor_full_bp": full_bp, "n_skipped_fold_budgets": len(skipped),
        "flat_fallbacks_total": int(sum(fallbacks.values())),
        "registered_contrasts": CA[CA.family == "registered"][["design", "comparison", "point", "ci95_low",
                                                              "ci95_high", "p_two_sided", "seeds_positive",
                                                              "loco_sign_stable", "passes_P1"]].to_dict("records"),
        "percentile": PCT.to_dict("records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    for f in OUT.glob("_pe_*.parquet"):
        f.unlink()
    say("\n=== learning curve (macro MAE, mean over draws) ===")
    say(curve.pivot_table(index=["design", "order"], columns="budget", values="macro_mae").round(4).to_string())
    say("\n=== registered contrasts (ABC on mae_all) ===")
    say(CA[CA.family == "registered"][["design", "comparison", "point", "ci95_low", "ci95_high", "p_two_sided",
                                        "seeds_positive", "loco_sign_stable", "passes_P1"]].round(4).to_string(index=False))
    say("\n=== percentile among random draws ===")
    say(PCT.round(4).to_string(index=False))
    say("DONE")


if __name__ == "__main__":
    main()
