"""L3a (measurements saved by the direction call) and L3c (ranking at k = 1), as registered in
PRE_REGISTRATION.md section 3 L3.  Everything is computed on the discovery seeds through the frozen
bench; see gen16/l3_decision.py for the construction.

Usage (from the repository root):
  PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe gen16_leads/scripts/l3_decision.py
      [--designs B,BR,BQ,A,BP] [--check-evaluate BP] [--report-only]

Writes results/L3/{tasks_summary.csv, l3a_saved.csv, l3a_contrasts.csv, l3a_per_task.csv,
l3c_metrics.csv, l3c_contrasts.csv, l3c_per_task.csv, checks.json, L3_REPORT.md}.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa: E402
from gen16 import l3_decision as L  # noqa: E402
from gen14.dirbench import load  # noqa: E402
from gen15 import arms as A  # noqa: E402
from gen15 import fewshot as FS  # noqa: E402

M = L.M
OUT = bootstrap.RESULTS / "L3"
DEC_TABLES = L.DEC / "tables"
warnings.filterwarnings("ignore")


def _py(v):
    return v.item() if hasattr(v, "item") else v


# --------------------------------------------------------------------------------------
# L3a
# --------------------------------------------------------------------------------------
def run_l3a(tab: pd.DataFrame, design: str, checks: dict) -> tuple[pd.DataFrame, list[dict], pd.DataFrame, dict]:
    t0 = time.time()
    T = L.build_tasks(tab, design, ["G14"])
    calls = {"G14": [L.calls_of(p) for p in T.pred["G14"]],
             "HEAVIER_ALWAYS": [np.full(len(o), -1, dtype=int) for o in T.obs]}
    names, W_loco = L.loco_weights(T)
    W_boot = L.blocked_weights(T)
    W_one = np.ones((1, len(T.ext)))
    bands = L.band_masks(T)
    splits = {"all": np.ones(T.n, dtype=bool), **bands}
    board_rows, con_rows = [], []
    per_task = L.per_task_frame(T)
    for arm, cl in calls.items():
        mats = L.l3a_matrices(T, cl)
        point = L.l3a_saved_weighted(W_one, mats)
        direct = L.l3a_per_task_direct(T, cl)
        worst = max(float(np.nanmax(np.abs(point[k][0] - direct[k].to_numpy())))
                    for k in ("saved", "saved_heavy", "saved_light", "e_random", "e_model"))
        checks[f"{design}:{arm}:matrix_vs_direct"] = worst
        if worst > 1e-9:
            raise RuntimeError(f"matrix route disagrees with the direct one: {worst}")
        boot = L.l3a_saved_weighted(W_boot, mats)
        loco = L.l3a_saved_weighted(W_loco, mats)
        perm = L.l3a_permutation(T, cl) if arm == "G14" else None
        if arm == "G14":
            for k in ("saved", "saved_heavy", "saved_light", "e_random", "e_model"):
                per_task[k] = point[k][0]
            per_task["n_no_call"] = direct["n_no_call"].to_numpy()
            per_task["n_called_heavy"] = direct["n_called_heavy"].to_numpy()
            per_task["n_called_light"] = direct["n_called_light"].to_numpy()
        for split, sel in splits.items():
            stats = {}
            for stat in ("saved", "saved_heavy", "saved_light"):
                stats[stat] = {
                    "point": L.seed_macro(point[stat], T.seeds, sel),
                    "block": L.seed_macro(boot[stat], T.seeds, sel),
                    "loco": L.seed_macro(loco[stat], T.seeds, sel),
                    "perm": (L.seed_macro(perm[stat], T.seeds, sel) if perm is not None else None),
                    "per_seed": np.array([L.seed_macro(point[stat], T.seeds, sel & (T.seeds == s))
                                          for s in np.unique(T.seeds)]),
                }
            e_rand = L.seed_macro(point["e_random"], T.seeds, sel)
            e_mod = L.seed_macro(point["e_model"], T.seeds, sel)
            e_rand_b = L.seed_macro(boot["e_random"], T.seeds, sel)
            with np.errstate(invalid="ignore", divide="ignore"):
                frac_task = L.seed_macro(point["saved"][0] / point["e_random"][0], T.seeds, sel)
                frac_boot = stats["saved"]["block"] / e_rand_b
            n_sel = int(sel.sum())
            row = {"design": design, "arm": arm, "split": split, "n_tasks": n_sel,
                   "e_random": e_rand, "e_model": e_mod, "saved": stats["saved"]["point"],
                   "saved_frac": stats["saved"]["point"] / e_rand if e_rand else np.nan,
                   "saved_frac_task_mean": frac_task,
                   "saved_frac_ci_low": L.pct(frac_boot, 0.025), "saved_frac_ci_high": L.pct(frac_boot, 0.975),
                   "saved_heavy": stats["saved_heavy"]["point"], "saved_light": stats["saved_light"]["point"],
                   "e_random_heavy": L.seed_macro(point["e_random_heavy"], T.seeds, sel),
                   "e_random_light": L.seed_macro(point["e_random_light"], T.seeds, sel),
                   "e_model_heavy": L.seed_macro(point["e_model_heavy"], T.seeds, sel),
                   "e_model_light": L.seed_macro(point["e_model_light"], T.seeds, sel),
                   "saved_sd_seed": float(np.std(stats["saved"]["per_seed"], ddof=1)),
                   "saved_ci_low": L.pct(stats["saved"]["block"], 0.025),
                   "saved_ci_high": L.pct(stats["saved"]["block"], 0.975),
                   "p_perm": (float((stats["saved"]["perm"] >= stats["saved"]["point"]).mean())
                              if perm is not None else np.nan),
                   "metric": "measurements saved = E_random - E_model, seed-macro over tasks",
                   "n_seeds": int(len(np.unique(T.seeds)))}
            for s, v in zip(np.unique(T.seeds), stats["saved"]["per_seed"]):
                row[f"saved_seed_{s}"] = float(v)
            board_rows.append(row)
            # contrasts: registered = G14 'all' on saved; everything else exploratory
            for stat, lab in (("saved", ""), ("saved_heavy", "_heavy"), ("saved_light", "_light")):
                if split != "all" and stat != "saved":
                    continue
                if arm == "HEAVIER_ALWAYS" and (split != "all" or stat != "saved"):
                    continue
                st = stats[stat]
                comparison = f"{arm}_saved{lab}_vs_0" + ("" if split == "all" else f"_{split}")
                family = "registered" if (arm == "G14" and split == "all" and stat == "saved") else "exploratory"
                con_rows.append(L.contrast_row(
                    design=design, comparison=comparison, value=stat, reference="random order",
                    candidate=arm, family=family, point=st["point"], block_draws=st["block"],
                    task_draws=None, perm_draws=st["perm"], per_seed=st["per_seed"], loco=st["loco"],
                    n_units=n_sel, n_ext=len(T.ext), n_blocks=len(names),
                    rule="saved > 0, permutation p < 0.05, chemotype-blocked CI excludes 0, all five designs",
                    registered_rule=L.rule_l3a))
    summary = {"design": design, "l3a_n_tasks": T.n, "l3a_n_dropped_lt5": T.n_dropped_lt_min,
               "l3a_n_candidate_slots": int(T.sizes().sum()), "l3a_median_candidates": float(np.median(T.sizes())),
               "l3a_min_candidates": int(T.sizes().min()), "l3a_max_candidates": int(T.sizes().max()),
               "n_extractants": len(T.ext), "n_chemotypes": len(names),
               "l3a_frac_tasks_all_same_call": float(np.mean([len(set(c)) == 1 for c in calls["G14"]])),
               "l3a_n_candidate_slots_no_call": int(sum(int((c == 0).sum()) for c in calls["G14"]))}
    print(f"   [{design}] L3a {T.n} tasks ({T.n_dropped_lt_min} dropped < {L.MIN_EXT}), "
          f"saved {board_rows[0]['saved']:+.4f} of E_random {board_rows[0]['e_random']:.3f}, "
          f"p_perm {board_rows[0]['p_perm']:.4f}, CI [{board_rows[0]['saved_ci_low']:+.4f}, "
          f"{board_rows[0]['saved_ci_high']:+.4f}]  {L.timer(t0)}", flush=True)
    return pd.DataFrame(board_rows), con_rows, per_task, summary


# --------------------------------------------------------------------------------------
# L3c
# --------------------------------------------------------------------------------------
L3C_CONTRASTS = [
    # comparison, value, reference arm, candidate arm, permuted arm, family
    ("G14k1_minus_NAIVE_regret", "regret", "NAIVE_LINE@k1", "G14@k1", "G14@k1", "registered"),
    ("G14k1_minus_NAIVE_spearman", "spearman", "NAIVE_LINE@k1", "G14@k1", "G14@k1", "registered"),
    ("G14k1_minus_NAIVE_top1", "top1", "NAIVE_LINE@k1", "G14@k1", "G14@k1", "exploratory"),
    ("G14k1_minus_G14zs_regret", "regret", "G14", "G14@k1", "G14@k1", "exploratory"),
    ("G14k1_minus_G14zs_spearman", "spearman", "G14", "G14@k1", "G14@k1", "exploratory"),
    ("G14k1_minus_RANDOM_regret", "regret", L.RAND, "G14@k1", "G14@k1", "exploratory"),
    ("G14k1_minus_RANDOM_spearman", "spearman", L.RAND, "G14@k1", "G14@k1", "exploratory"),
    ("NAIVE_minus_RANDOM_regret", "regret", L.RAND, "NAIVE_LINE@k1", "NAIVE_LINE@k1", "exploratory"),
    ("NAIVE_minus_RANDOM_spearman", "spearman", L.RAND, "NAIVE_LINE@k1", "NAIVE_LINE@k1", "exploratory"),
]
HIGHER = {"spearman", "top1"}


def gain(stat: str, ref: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Positive = candidate better: regret is lower-is-better, Spearman / top-1 higher-is-better."""
    return (cand - ref) if stat in HIGHER else (ref - cand)


def run_l3c(k1: pd.DataFrame, tab: pd.DataFrame, design: str, checks: dict) -> tuple[pd.DataFrame, list[dict], pd.DataFrame, dict]:
    t0 = time.time()
    arms = list(L.K1_ARMS)
    summary_df, per_pair = M.cross_extractant(k1, arms, unit="extractant")
    summary_df.insert(0, "design", design)
    T = L.build_tasks(k1, design, arms, drop_ptp0=True)
    n_pp = int(per_pair.groupby(["split_seed", "A", "B"]).ngroups)
    if n_pp != T.n:
        raise RuntimeError(f"task count mismatch: decmetrics {n_pp} vs tasks {T.n}")
    chk = L.check_weighted(T, per_pair)
    checks[f"{design}:l3c"] = chk
    if max(chk.values()) > 1e-9:
        raise RuntimeError(f"weighted statistics disagree with decmetrics: {chk}")
    all_arms = arms + [L.RAND]
    names, W_loco = L.loco_weights(T)
    W_boot = L.blocked_weights(T)
    W_one = np.ones((1, len(T.ext)))
    point = L.l3c_stats_for_weights(T, W_one, all_arms)
    boot = L.l3c_stats_for_weights(T, W_boot, all_arms)
    loco = L.l3c_stats_for_weights(T, W_loco, all_arms)
    perm = {a: L.l3c_permuted_stats(T, a) for a in ("G14@k1", "NAIVE_LINE@k1")}
    # task bootstrap, stratified by split seed (so the seed-macro aggregate is defined per draw)
    rng = np.random.default_rng(L.BOOT_SEED)
    task_idx = np.empty((L.N_BOOT, T.n), dtype=int)
    for s in np.unique(T.seeds):
        cols = np.flatnonzero(T.seeds == s)
        task_idx[:, cols] = cols[rng.integers(0, len(cols), size=(L.N_BOOT, len(cols)))]
    seeds_u = np.unique(T.seeds)
    con_rows = []
    for comparison, stat, ref, cand, permuted, family in L3C_CONTRASTS:
        g_task = gain(stat, point[ref][stat][0], point[cand][stat][0])          # (T,)
        pt = L.seed_macro(g_task, T.seeds)
        block = L.seed_macro(gain(stat, boot[ref][stat], boot[cand][stat]), T.seeds)
        task_draws = L.seed_macro(g_task[task_idx], T.seeds)
        lo = L.seed_macro(gain(stat, loco[ref][stat], loco[cand][stat]), T.seeds)
        per_seed = np.array([L.seed_macro(g_task, T.seeds, T.seeds == s) for s in seeds_u])
        if permuted == cand:
            pdraws = L.seed_macro(gain(stat, point[ref][stat], perm[cand][stat]), T.seeds)
        else:
            pdraws = L.seed_macro(gain(stat, perm[ref][stat], point[cand][stat]), T.seeds)
        con_rows.append(L.contrast_row(
            design=design, comparison=comparison, value=stat, reference=ref, candidate=cand,
            family=family, point=pt, block_draws=block, task_draws=task_draws, perm_draws=pdraws,
            per_seed=per_seed, loco=lo, n_units=T.n, n_ext=len(T.ext), n_blocks=len(names),
            rule="conservative p (max of task bootstrap, blocked bootstrap, permutation) < 0.05 "
                 "with gain > 0 in all five designs",
            registered_rule=L.rule_l3c))
    # candidates dropped because their only measured pair is the target
    key = ["split_seed", "A", "B"]
    a_sets = tab.groupby(key)["extractant"].agg(set)
    c_sets = k1.groupby(key)["extractant"].agg(set)
    Ta = L.build_tasks(tab, design, ["G14"])
    n_dropped_cand = 0
    for s, a, b in zip(Ta.seeds, Ta.A, Ta.B):
        n_dropped_cand += len(a_sets.loc[(s, a, b)] - c_sets.get((s, a, b), set()))
    per_pair.insert(0, "design", design)
    summary = {"design": design, "l3c_n_tasks": T.n, "l3c_n_dropped_lt5": T.n_dropped_lt_min,
               "l3c_n_dropped_ptp0": T.n_dropped_ptp0, "l3c_n_candidate_slots": int(T.sizes().sum()),
               "l3c_median_candidates": float(np.median(T.sizes())),
               "l3c_n_candidates_dropped_only_target": int(n_dropped_cand)}
    reg = [r for r in con_rows if r["comparison"] == "G14k1_minus_NAIVE_regret"][0]
    print(f"   [{design}] L3c {T.n} tasks; regret G14@k1 "
          f"{float(summary_df.loc[summary_df.arm == 'G14@k1', 'regret'].iat[0]):.4f} "
          f"NAIVE {float(summary_df.loc[summary_df.arm == 'NAIVE_LINE@k1', 'regret'].iat[0]):.4f} "
          f"G14 {float(summary_df.loc[summary_df.arm == 'G14', 'regret'].iat[0]):.4f} "
          f"RANDOM {float(summary_df.loc[summary_df.arm == '_RANDOM', 'regret'].iat[0]):.4f}; "
          f"gain {reg['point']:+.4f} p_reg {reg['p_registered']:.4f} "
          f"CI[{reg['ci95_low']:+.4f}, {reg['ci95_high']:+.4f}] ({reg['conservative_interval']}) "
          f"{L.timer(t0)}", flush=True)
    return summary_df, con_rows, per_pair, summary


# --------------------------------------------------------------------------------------
# reproduction checks
# --------------------------------------------------------------------------------------
def check_cached_tables(tab: pd.DataFrame, design: str) -> dict:
    p = DEC_TABLES / f"pairs_{design}.parquet"
    if not p.exists():
        return {"available": False}
    c = pd.read_parquet(p, columns=["split_seed", "cell_id", "A", "B", "y", "G14"])
    m = tab.merge(c, on=["split_seed", "cell_id", "A", "B"], suffixes=("", "_cached"), validate="one_to_one")
    return {"available": True, "n_rows": int(len(tab)), "n_rows_cached": int(len(c)),
            "n_matched": int(len(m)),
            "max_abs_diff_G14": float(np.abs(m["G14"] - m["G14_cached"]).max()),
            "max_abs_diff_y": float(np.abs(m["y"] - m["y_cached"]).max())}


def check_zero_shot(k1: pd.DataFrame, tab: pd.DataFrame) -> dict:
    m = k1[["split_seed", "cell_id", "A", "B", "y", "G14"]].merge(
        tab[["split_seed", "cell_id", "A", "B", "y", "G14"]], on=["split_seed", "cell_id", "A", "B"],
        suffixes=("", "_run"), validate="one_to_one")
    return {"n_rows_k1": int(len(k1)), "n_matched": int(len(m)),
            "max_abs_diff_G14": float(np.abs(m["G14"] - m["G14_run"]).max()),
            "max_abs_diff_y": float(np.abs(m["y"] - m["y_run"]).max())}


def check_against_evaluate(bench, k1: pd.DataFrame, design: str) -> dict:
    """``fewshot.evaluate`` at k = 1 (widest, publication-masked) predicts every non-widest pair
    from the cell's widest pair -- exactly this lead's support whenever the target is not the
    widest pair -- so the two routes must agree there to floating-point precision."""
    t0 = time.time()
    r = FS.evaluate(bench, {"G14": A.g14}, design, ks=(0, 1), how="widest",
                    mask_publication=True, verbose=False)
    e = r.pairs[["split_seed", "cell_id", "A", "B", "G14@k0", "G14@k1", "NAIVE_LINE@k1"]]
    m = k1.merge(e, on=["split_seed", "cell_id", "A", "B"], suffixes=("", "_eval"), validate="one_to_one")
    return {"seconds": time.time() - t0, "n_rows_evaluate": int(len(e)), "n_matched": int(len(m)),
            "max_abs_diff_G14k1": float(np.abs(m["G14@k1"] - m["G14@k1_eval"]).max()),
            "max_abs_diff_NAIVEk1": float(np.abs(m["NAIVE_LINE@k1"] - m["NAIVE_LINE@k1_eval"]).max()),
            "max_abs_diff_G14zs": float(np.abs(m["G14"] - m["G14@k0"]).max()),
            "all_matched_support_is_widest": bool((m["sup_dZ"] >= m["dZ"]).all())}


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", default=",".join(L.DESIGNS))
    ap.add_argument("--check-evaluate", default="BP", help="designs for the fewshot.evaluate check ('' = none)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        from l3_report import write_report
        write_report(OUT)
        return 0
    designs = [d for d in args.designs.split(",") if d]
    check_ev = set(d for d in args.check_evaluate.split(",") if d)
    t_all = time.time()
    bench = load()
    checks: dict = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "designs": designs}
    boards_a, cons, per_a, boards_c, per_c, summaries = [], [], [], [], [], []
    for d in designs:
        t0 = time.time()
        tab = L.run_pair_table(bench, d)
        checks[f"{d}:cached_tables"] = check_cached_tables(tab, d)
        curves, fold_of = L.g14_curves(bench, d)
        cov_for = L.CovFor(bench, curves)
        k1, drop = L.k1_table(bench, curves, fold_of, cov_for)
        checks[f"{d}:zero_shot"] = check_zero_shot(k1, tab)
        if checks[f"{d}:zero_shot"]["max_abs_diff_G14"] > 1e-9:
            raise RuntimeError("curve route disagrees with run_arms")
        checks[f"{d}:cov_fallbacks"] = cov_for.n_fallback
        checks[f"{d}:cov_keys"] = len(cov_for.cache)
        if d in check_ev:
            checks[f"{d}:evaluate"] = check_against_evaluate(bench, k1, d)
            ev = checks[f"{d}:evaluate"]
            print(f"   [{d}] fewshot.evaluate check: {ev['n_matched']} rows, max |dG14@k1| "
                  f"{ev['max_abs_diff_G14k1']:.2e}, |dNAIVE| {ev['max_abs_diff_NAIVEk1']:.2e} "
                  f"({ev['seconds']:.0f}s)", flush=True)
            if ev["max_abs_diff_G14k1"] > 1e-9 or ev["max_abs_diff_NAIVEk1"] > 1e-9:
                raise RuntimeError("k = 1 route disagrees with fewshot.evaluate")
        print(f"   [{d}] tables built: {len(tab)} pairs, {len(k1)} k1 rows, "
              f"{drop['n_cell_seeds_single_pair']} single-pair cell-seeds dropped, "
              f"cached-table max |dG14| {checks[f'{d}:cached_tables'].get('max_abs_diff_G14', float('nan')):.2e} "
              f"{L.timer(t0)}", flush=True)
        ba, ca, pa, sa = run_l3a(tab, d, checks)
        bc, cc, pc, sc = run_l3c(k1, tab, d, checks)
        boards_a.append(ba); cons.extend(ca); per_a.append(pa)
        boards_c.append(bc); cons.extend(cc); per_c.append(pc)
        summaries.append({**sa, **sc, **drop, "n_pair_rows": int(len(tab)), "n_k1_rows": int(len(k1)),
                          "cov_fallbacks": cov_for.n_fallback, "seconds": time.time() - t0})
        # write after every design so a partial run is still usable
        pd.DataFrame(summaries).to_csv(OUT / "tasks_summary.csv", index=False)
        pd.concat(boards_a, ignore_index=True).to_csv(OUT / "l3a_saved.csv", index=False)
        pd.concat(per_a, ignore_index=True).to_csv(OUT / "l3a_per_task.csv", index=False)
        pd.concat(boards_c, ignore_index=True).to_csv(OUT / "l3c_metrics.csv", index=False)
        pd.concat(per_c, ignore_index=True).to_csv(OUT / "l3c_per_task.csv", index=False)
        C = pd.DataFrame(cons)[L.CONTRAST_COLUMNS]
        C[C.comparison.str.contains("saved")].to_csv(OUT / "l3a_contrasts.csv", index=False)
        C[~C.comparison.str.contains("saved")].to_csv(OUT / "l3c_contrasts.csv", index=False)
        checks["seconds_total"] = time.time() - t_all
        (OUT / "checks.json").write_text(json.dumps(checks, indent=2, default=_py), encoding="utf-8")
    print(f"[L3] done in {L.timer(t_all)}; {len(cons)} contrasts written", flush=True)
    from l3_report import write_report
    write_report(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
