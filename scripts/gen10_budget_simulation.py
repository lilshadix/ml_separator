#!/usr/bin/env python
"""PHASE 9 — breadth versus depth, simulated from the rows that already exist.

No new data.  Every unseen ligand in the k-shot study already has a *held-out
adaptation curve*: its macro MAE at k = 0, 1, 2, 3, 5 measurements, each number
computed when the ligand's whole chemotype was held out, averaged over seeds and
over the twelve pool/evaluation draws.  A measurement budget is then a question
about how to spend points across those curves, and it can be answered exactly
without refitting anything:

* the expected reduction in macro MAE from the *first* point on a ligand is
  ``m(0) - m(1)``; from the second, ``m(1) - m(2)``; and so on.  Breadth beats
  depth whenever the first point's gain exceeds the later points' — which is the
  quantity the table ``marginal_gains.csv`` reports, overall and by how far the
  ligand sits from the training chemistry;
* for a budget ``B`` over ``N`` unseen ligands, each strategy is an allocation
  ``k_l`` with ``sum k_l = B``; the resulting macro MAE is ``mean_l m_l(k_l)`` with
  unmeasured ligands at ``m_l(0)``.

Strategies, as the brief lists them:

``A_DEPTH_5``       five points each on ``B/5`` ligands
``A_DEPTH_3``       three points each on ``B/3`` ligands
``B_BREADTH_1``     one point on ``B`` ligands
``C_BREADTH_2``     two points on ``B/2`` ligands
``D_NEW_CHEMOTYPE`` one point each, ligands ordered so every chemotype is covered
                    before any chemotype gets a second ligand, farthest-from-
                    training first (deployable: uses conditions and structure only)
``E_CENTRAL_SPREAD`` one point on every ligand first (central), then the remaining
                    budget spread as second/third points
``F_DIVERSITY``     one point each, ligands chosen by Tanimoto max-min diversity
                    from the candidate pool (deployable)
``ORACLE_ERROR``    one point each, ligands ordered by their true zero-shot error
                    (**not deployable** — labelled, reported as the ceiling)

Random ligand orderings are averaged over draws; ``D``, ``F`` and ``ORACLE`` are
deterministic.  Reported per budget: macro MAE, gain per measurement, and gain
per newly covered chemotype.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

GEN9 = REPO_ROOT / "runs" / "gen9_shape"
GEN7 = REPO_ROOT / "runs" / "gen7_architecture"
OUT = REPO_ROOT / "runs" / "gen10_final" / "budget_simulation"
#: The frontier's deployable arms per k (SHAPE_RECOMPOSED, central-then-spread).
ARM_BY_K = {0: ("ZERO_SHOT", "NONE"), 1: ("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD"),
            2: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD"), 3: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD"),
            5: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD")}
K_STEPS = (0, 1, 2, 3, 5)
RANDOM_DRAWS = 200


def adaptation_curves(detail: pd.DataFrame, model: str) -> pd.DataFrame:
    """One row per ligand: m(0), m(1), m(2), m(3), m(5) and its chemistry metadata."""
    block = detail[detail["global_model"] == model]
    parts = []
    for k, (adapter, policy) in ARM_BY_K.items():
        sub = block[(block["k"] == k) & (block["adapter"].isin([adapter, "ZERO_SHOT_REF"]
                                                               if k == 0 else [adapter]))]
        if k > 0:
            sub = sub[sub["policy"] == policy]
        per = sub.groupby("extractant")["mae"].mean().rename(f"m{k}")
        parts.append(per)
    curves = pd.concat(parts, axis=1).dropna()
    meta = (block.drop_duplicates("extractant")
            .set_index("extractant")[["tanimoto_cluster", "nn_train_tanimoto", "n_rows"]])
    return curves.join(meta, how="left")


def maxmin_order(ligands: list[str], similarity: pd.DataFrame, start: str) -> list[str]:
    """Greedy max-min Tanimoto diversity over the candidate pool."""
    chosen = [start]
    remaining = [l for l in ligands if l != start]
    while remaining:
        best = max(remaining, key=lambda l: (min(1.0 - float(similarity.loc[l, c])
                                                 for c in chosen), l))
        chosen.append(best)
        remaining.remove(best)
    return chosen


def allocate(order: list[str], budget: int, *, per_ligand: int) -> dict[str, int]:
    out: dict[str, int] = {}
    remaining = budget
    for ligand in order:
        if remaining < per_ligand:
            break
        out[ligand] = per_ligand
        remaining -= per_ligand
    return out


def central_then_spread(order: list[str], budget: int) -> dict[str, int]:
    out = {l: 0 for l in order}
    remaining = budget
    for k in (1, 2, 3, 5):
        for ligand in order:
            need = k - out[ligand]
            if need <= 0:
                continue
            if remaining < need:
                return {l: v for l, v in out.items() if v > 0}
            out[ligand] = k
            remaining -= need
    return {l: v for l, v in out.items() if v > 0}


def score(curves: pd.DataFrame, allocation: dict[str, int]) -> float:
    """Macro MAE over every unseen ligand after the allocation; k not in the table
    falls back to the largest measured k below it (never above)."""
    total = 0.0
    for ligand, row in curves.iterrows():
        k = allocation.get(ligand, 0)
        usable = max(s for s in K_STEPS if s <= k)
        total += float(row[f"m{usable}"])
    return total / len(curves)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", type=Path, default=GEN9 / "kshot" / "recomposed_detail.parquet")
    parser.add_argument("--model", default="GEN9_SHAPE_RECOMPOSED")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--draws", type=int, default=RANDOM_DRAWS)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    detail = pd.read_parquet(args.detail, columns=[
        "global_model", "extractant", "tanimoto_cluster", "nn_train_tanimoto", "n_rows",
        "policy", "adapter", "k", "mae"])
    curves = adaptation_curves(detail, args.model)
    curves.to_csv(args.out / "adaptation_curves.csv")
    ligands = list(curves.index)
    n = len(ligands)

    # --- marginal gains ------------------------------------------------------
    gains = pd.DataFrame({
        "first_point": curves["m0"] - curves["m1"],
        "second_point": curves["m1"] - curves["m2"],
        "third_point": curves["m2"] - curves["m3"],
        "fourth_and_fifth_each": (curves["m3"] - curves["m5"]) / 2.0,
    })
    gains["nn_train_tanimoto"] = curves["nn_train_tanimoto"]
    gains["distance_tercile"] = pd.qcut(1.0 - curves["nn_train_tanimoto"], 3,
                                        labels=["near", "mid", "far"])
    gains["m0"] = curves["m0"]
    marginal = pd.concat([
        gains.drop(columns=["distance_tercile", "nn_train_tanimoto"]).agg(["mean", "median"])
        .assign(group="all"),
        gains.groupby("distance_tercile", observed=True)[
            ["first_point", "second_point", "third_point", "fourth_and_fifth_each", "m0"]]
        .mean().reset_index().rename(columns={"distance_tercile": "group"}).set_index("group"),
    ])
    marginal.to_csv(args.out / "marginal_gains.csv")

    # --- orderings -------------------------------------------------------------
    from lanthanide_separation.gen6.chemistry import ChemistryMap
    chemistry = ChemistryMap.from_parquet(GEN7 / "cache" / "chemistry_map.parquet")
    known = [l for l in ligands if l in set(chemistry.extractants)]
    similarity = pd.DataFrame(chemistry.similarity_between(known, known), index=known,
                              columns=known) if known else None
    by_novelty = curves.sort_values(["nn_train_tanimoto", "n_rows"]).index.tolist()
    # D: cover every chemotype once, farthest first, then second ligands per chemotype
    seen: dict[str, int] = {}
    first_pass, later = [], []
    for ligand in by_novelty:
        cluster = str(curves.loc[ligand, "tanimoto_cluster"])
        (first_pass if cluster not in seen else later).append(ligand)
        seen[cluster] = seen.get(cluster, 0) + 1
    new_chemotype_order = first_pass + later
    oracle_order = curves.sort_values("m0", ascending=False).index.tolist()
    if similarity is not None and len(known) == n:
        diversity_order = maxmin_order(ligands, similarity, by_novelty[0])
    else:
        diversity_order = None

    rng = np.random.default_rng(20260821)
    budgets = sorted(set([max(1, n // 10), n // 4, n // 2, n, 2 * n, 3 * n, 5 * n]))
    rows = []
    for budget in budgets:
        random_orders = [list(rng.permutation(ligands)) for _ in range(args.draws)]

        def record(strategy: str, allocation_fn, deployable: bool, orders=None):
            values, covered = [], []
            for order in (orders or [None]):
                allocation = allocation_fn(order)
                values.append(score(curves, allocation))
                covered.append(len({str(curves.loc[l, "tanimoto_cluster"]) for l in allocation}))
            mae = float(np.mean(values))
            base = float(curves["m0"].mean())
            n_measured = sum(allocation.values())
            rows.append({"budget": int(budget), "strategy": strategy, "deployable": deployable,
                         "macro_mae": mae, "zero_shot_macro_mae": base,
                         "reduction": base - mae,
                         "gain_per_measurement": (base - mae) / max(n_measured, 1),
                         "n_measurements_used": int(n_measured),
                         "n_ligands_measured": int(len(allocation)),
                         "n_chemotypes_covered": float(np.mean(covered)),
                         "gain_per_chemotype_covered": (base - mae) / max(np.mean(covered), 1)})

        record("A_DEPTH_5", lambda o: allocate(o, budget, per_ligand=5), True, random_orders)
        record("A_DEPTH_3", lambda o: allocate(o, budget, per_ligand=3), True, random_orders)
        record("B_BREADTH_1", lambda o: allocate(o, budget, per_ligand=1), True, random_orders)
        record("C_BREADTH_2", lambda o: allocate(o, budget, per_ligand=2), True, random_orders)
        record("D_NEW_CHEMOTYPE", lambda o: allocate(new_chemotype_order, budget, per_ligand=1), True)
        record("E_CENTRAL_SPREAD", lambda o: central_then_spread(o, budget), True, random_orders)
        record("E_CENTRAL_SPREAD_NOVELTY", lambda o: central_then_spread(new_chemotype_order, budget),
               True)
        if diversity_order is not None:
            record("F_DIVERSITY", lambda o: allocate(diversity_order, budget, per_ligand=1), True)
        record("ORACLE_ERROR", lambda o: allocate(oracle_order, budget, per_ligand=1), False)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "budget_strategies.csv", index=False)

    summary = {
        "model": args.model, "n_unseen_ligands": n,
        "n_chemotypes": int(curves["tanimoto_cluster"].nunique()),
        "mean_marginal_gain": {c: float(gains[c].mean()) for c in
                               ("first_point", "second_point", "third_point",
                                "fourth_and_fifth_each")},
        "first_point_gain_by_distance_tercile": {
            str(g): float(v) for g, v in gains.groupby("distance_tercile", observed=True)
            ["first_point"].mean().items()},
        "best_deployable_strategy_by_budget": {
            int(b): str(block[block["deployable"]].sort_values("macro_mae").iloc[0]["strategy"])
            for b, block in table.groupby("budget")},
        "diversity_order_available": diversity_order is not None,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(table.pivot(index="budget", columns="strategy", values="macro_mae").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
