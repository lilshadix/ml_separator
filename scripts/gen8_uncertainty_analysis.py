"""gen8 section 20 -- scoring every uncertainty source twice.

(a) calibration: does it rank |residual|?
(b) decision usefulness: does it rank candidates to measure?
(c) mechanism: what does a 1-shot policy actually need to rank?

Reads runs/gen8_architecture/uncertainty/row_uncertainty.parquet (built by
scripts/gen8_uncertainty.py) and the exact 1-shot candidate table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lanthanide_separation.gen8.inference import paired_chemotype_bootstrap  # noqa: E402

OUT = ROOT / "runs" / "gen8_architecture" / "uncertainty"
MIN_ROWS = 5
BOOT_SEED = 8675309

SOURCES: dict[str, str] = {
    "u_tree_sd": "tree-ensemble sd (sibling TREE_MC_ecfp_massaction)",
    "u_tree_sd_mean": "tree-ensemble sd, mean of the 3 TREE_MC arms",
    "u_model_spread": "deep-ensemble spread, sd over 5 finalists",
    "u_cond_nn1": "condition space, 1-NN distance to fold train",
    "u_cond_nn5": "condition space, 5-NN distance to fold train",
    "u_gp_var_design": "RBF-GP LOO variance over the ligand condition axes",
    "u_gp_var_train": "RBF-GP variance vs the fold training design",
    "u_lig_tanimoto": "1 - nn_train_tanimoto (ligand-level)",
    "u_lig_mech": "mechanistic distance to nearest train ligand (ligand-level)",
}
ROW_LEVEL = [s for s in SOURCES if not s.startswith("u_lig_")]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < MIN_ROWS:
        return np.nan
    x, y = x[ok], y[ok]
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def macro_ci(per_cell: pd.DataFrame, value: str, label: str) -> dict:
    """Ligand-macro mean of ``value`` with a chemotype bootstrap CI."""
    frame = per_cell.dropna(subset=[value]).copy()
    if frame.empty:
        return {"comparison": label, "point": np.nan, "n_units": 0}
    long = pd.concat([
        frame.assign(arm="X", mae=frame[value]),
        frame.assign(arm="ZERO", mae=0.0),
    ])
    res = paired_chemotype_bootstrap(
        long, {label: ("X", "ZERO")}, value_column="mae", seed=BOOT_SEED)
    return res.iloc[0].to_dict()


def pick(values: np.ndarray, rng: np.random.Generator, largest: bool) -> int:
    """argmax/argmin with random tie-breaking; NaN -> uniform random pick."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if not ok.any():
        return int(rng.integers(len(v)))
    w = np.where(ok, v, -np.inf if largest else np.inf)
    target = w.max() if largest else w.min()
    hits = np.flatnonzero(w == target)
    return int(hits[rng.integers(len(hits))])


# --------------------------------------------------------------------------- #
# (a) calibration
# --------------------------------------------------------------------------- #
def calibration(unc: pd.DataFrame) -> None:
    rows, cross = [], []
    cells = list(unc.groupby(["split_seed", "extractant"], sort=True))

    for src in SOURCES:
        per_cell = []
        for (seed, lig), block in cells:
            per_cell.append({
                "split_seed": seed, "extractant": lig,
                "tanimoto_cluster": block["tanimoto_cluster"].iloc[0],
                "rho": rho(block[src].to_numpy(), block["abs_residual"].to_numpy()),
                "n": len(block),
            })
        per_cell = pd.DataFrame(per_cell)
        res = macro_ci(per_cell, "rho", f"within-ligand rho: {src}")
        frac = float((per_cell["rho"].dropna() > 0).mean()) if per_cell["rho"].notna().any() else np.nan
        rows.append({
            "source": src, "description": SOURCES[src],
            "within_ligand_rho": res.get("point", np.nan),
            "ci_low": res.get("ci95_low", np.nan), "ci_high": res.get("ci95_high", np.nan),
            "bca_low": res.get("bca_low", np.nan), "bca_high": res.get("bca_high", np.nan),
            "n_ligand_seed_cells": int(per_cell["rho"].notna().sum()),
            "frac_cells_positive": frac,
        })

        # across-ligand: one point per ligand, per seed
        per_seed = []
        for seed, block in unc.groupby("split_seed"):
            agg = block.groupby("extractant").agg(u=(src, "mean"), mae=("abs_residual", "mean"))
            per_seed.append(rho(agg["u"].to_numpy(), agg["mae"].to_numpy()))
        cross.append({
            "source": src, "description": SOURCES[src],
            "across_ligand_rho_mean": float(np.nanmean(per_seed)),
            "across_ligand_rho_min": float(np.nanmin(per_seed)),
            "across_ligand_rho_max": float(np.nanmax(per_seed)),
            "seeds_positive": int(np.sum(np.asarray(per_seed) > 0)), "n_seeds": len(per_seed),
        })

    within = pd.DataFrame(rows)
    acrossf = pd.DataFrame(cross)
    within.to_csv(OUT / "calibration_within_ligand.csv", index=False)
    acrossf.to_csv(OUT / "calibration_across_ligand.csv", index=False)

    # The per-source cell counts above are not equal: a source that is constant
    # inside a cell yields no Spearman there, so u_cond_nn1 is defined on fewer
    # cells than u_tree_sd.  Comparing the columns as printed is therefore an
    # unpaired comparison.  Repeat it on the cells where *every* row-level source
    # is defined, so the ranking cannot be an artefact of differing supports.
    per_cell_all = {}
    for src in ROW_LEVEL:
        per_cell_all[src] = np.array(
            [rho(block[src].to_numpy(), block["abs_residual"].to_numpy()) for _, block in cells])
    frame_all = pd.DataFrame(per_cell_all)
    frame_all["extractant"] = [lig for (_, lig), _ in cells]
    common = frame_all[ROW_LEVEL].notna().all(axis=1)
    common_rows = []
    for src in ROW_LEVEL:
        macro_own = frame_all.groupby("extractant")[src].mean().mean()
        macro_common = frame_all[common].groupby("extractant")[src].mean().mean()
        common_rows.append({
            "source": src, "within_ligand_rho_own_cells": float(macro_own),
            "n_cells_own": int(frame_all[src].notna().sum()),
            "within_ligand_rho_common_cells": float(macro_common),
            "n_cells_common": int(common.sum()),
            "n_cells_possible": int(len(frame_all)),
        })
    pd.DataFrame(common_rows).to_csv(OUT / "calibration_within_ligand_common.csv", index=False)

    # self-consistency: sibling model's sd vs the sibling model's OWN residual
    self_rows = []
    for src, resid_col, tag in [("u_tree_sd", "abs_residual", "frozen model residual"),
                                ("u_tree_sd", "sib_abs_residual", "sibling's own residual")]:
        per_cell = []
        for (seed, lig), block in cells:
            per_cell.append({
                "split_seed": seed, "extractant": lig,
                "tanimoto_cluster": block["tanimoto_cluster"].iloc[0],
                "rho": rho(block[src].to_numpy(), block[resid_col].to_numpy())})
        per_cell = pd.DataFrame(per_cell)
        res = macro_ci(per_cell, "rho", tag)
        per_seed = []
        for seed, block in unc.groupby("split_seed"):
            agg = block.groupby("extractant").agg(u=(src, "mean"), mae=(resid_col, "mean"))
            per_seed.append(rho(agg["u"].to_numpy(), agg["mae"].to_numpy()))
        self_rows.append({"target": tag, "within_ligand_rho": res.get("point", np.nan),
                          "ci_low": res.get("ci95_low", np.nan),
                          "ci_high": res.get("ci95_high", np.nan),
                          "across_ligand_rho": float(np.nanmean(per_seed))})
    pd.DataFrame(self_rows).to_csv(OUT / "calibration_self_consistency.csv", index=False)

    # calibration curves
    pooled, within_curve = [], []
    for src in SOURCES:
        v = unc[[src, "abs_residual", "extractant"]].dropna()
        if v.empty:
            continue
        try:
            v = v.assign(bin=pd.qcut(v[src].rank(method="first"), 10, labels=False))
        except ValueError:
            continue
        g = v.groupby("bin").agg(mean_u=(src, "mean"), mae=("abs_residual", "mean"),
                                 n=("abs_residual", "size"))
        for b, r in g.iterrows():
            pooled.append({"source": src, "decile": int(b) + 1, "mean_u": r["mean_u"],
                           "realised_mae": r["mae"], "n": int(r["n"])})

        # within-ligand quintiles: rank inside each ligand-seed cell, then macro
        recs = []
        for (seed, lig), block in cells:
            b = block[[src, "abs_residual"]].dropna()
            if len(b) < 10 or np.ptp(b[src]) == 0:
                continue
            q = pd.qcut(b[src].rank(method="first"), 5, labels=False)
            tmp = b.assign(q=q, lig=lig)
            base = b["abs_residual"].mean()
            agg = tmp.groupby("q")["abs_residual"].mean().rename("mae").reset_index()
            agg["centred"] = agg["mae"] - base
            agg["extractant"] = lig
            recs.append(agg)
        if recs:
            allr = pd.concat(recs)
            per_lig = allr.groupby(["extractant", "q"])[["mae", "centred"]].mean().reset_index()
            g2 = per_lig.groupby("q")[["mae", "centred"]].mean()
            n_lig = per_lig["extractant"].nunique()
            for q, r in g2.iterrows():
                within_curve.append({"source": src, "quintile": int(q) + 1,
                                     "realised_mae": r["mae"], "centred_mae": r["centred"],
                                     "n_ligands": n_lig})
    pd.DataFrame(pooled).to_csv(OUT / "calibration_curve_pooled.csv", index=False)
    pd.DataFrame(within_curve).to_csv(OUT / "calibration_curve_within_ligand.csv", index=False)
    print("calibration written")


# --------------------------------------------------------------------------- #
# (b) decision usefulness  +  (c) mechanism
# --------------------------------------------------------------------------- #
def decisions(unc: pd.DataFrame) -> None:
    cand = pd.read_parquet(
        ROOT / "runs/gen8_architecture/cross_series/one_shot_candidate_scores.parquet")
    keep = ["split_seed", "row_id", "residual", "abs_residual"] + list(SOURCES)
    merged = cand.merge(unc[keep], on=["split_seed", "row_id"], how="left",
                        validate="one_to_one")
    assert len(merged) == len(cand), "join changed the row count"
    merged.to_parquet(OUT / "candidate_uncertainty.parquet", index=False)

    rng = np.random.default_rng(20260820)
    policy_rows, rho_rows, mech_rows = [], [], []

    for (seed, lig), block in merged.groupby(["split_seed", "extractant"], sort=True):
        block = block.reset_index(drop=True)
        y = block["one_shot_mae"].to_numpy(dtype=float)
        r = block["residual"].to_numpy(dtype=float)
        med = float(np.median(r))
        dev = np.abs(r - med)
        absr = np.abs(r)
        tan = block["tanimoto_cluster"].iloc[0]
        common = {"split_seed": seed, "extractant": lig, "tanimoto_cluster": tan,
                  "n_candidates": len(block)}

        def add(arm: str, idx: int) -> None:
            policy_rows.append({**common, "arm": arm, "mae": float(y[idx]),
                                "picked_residual_pctile": float((r < r[idx]).mean()
                                                                + 0.5 * (r == r[idx]).mean())})

        policy_rows.append({**common, "arm": "RANDOM(expected)", "mae": float(np.mean(y)),
                            "picked_residual_pctile": np.nan})
        policy_rows.append({**common, "arm": "ZERO_SHOT",
                            "mae": float(block["zero_shot_mae"].iloc[0]),
                            "picked_residual_pctile": np.nan})
        policy_rows.append({**common, "arm": "ORACLE_LEVEL(best constant)",
                            "mae": float(block["oracle_level_mae"].iloc[0]),
                            "picked_residual_pctile": np.nan})
        add("ORACLE_1SHOT", int(np.argmin(y)))
        add("WORST_1SHOT", int(np.argmax(y)))
        add("ORACLE_MIN_ABS_RESID", pick(absr, rng, largest=False))
        add("ORACLE_MAX_ABS_RESID", pick(absr, rng, largest=True))
        add("ORACLE_MIN_MEDIAN_DEV", pick(dev, rng, largest=False))
        for src in SOURCES:
            u = block[src].to_numpy(dtype=float)
            add(f"MAX[{src}]", pick(u, rng, largest=True))
            add(f"MIN[{src}]", pick(u, rng, largest=False))
            rho_rows.append({**common, "source": src,
                             "rho_u_vs_one_shot_mae": rho(u, y),
                             "rho_u_vs_median_dev": rho(u, dev),
                             "rho_u_vs_abs_resid": rho(u, absr)})

        mech_rows.append({**common,
                          "one_shot_gap_of_median_dev_pick":
                              float(y[int(np.argmin(dev))] - y.min()),
                          "rho_meddev_vs_one_shot": rho(dev, y),
                          "rho_absresid_vs_one_shot": rho(absr, y),
                          "rho_absresid_vs_meddev": rho(absr, dev),
                          "abs_median_residual": abs(med),
                          "mad_residual": float(np.mean(np.abs(r - med))),
                          "zero_shot": float(block["zero_shot_mae"].iloc[0])})

    policies = pd.DataFrame(policy_rows)
    policies.to_parquet(OUT / "policy_detail.parquet", index=False)

    per_lig = policies.groupby(["arm", "extractant"])["mae"].mean()
    per_seed = policies.groupby(["arm", "split_seed", "extractant"])["mae"].mean() \
        .groupby(["arm", "split_seed"]).mean().unstack("split_seed")
    summary = per_lig.groupby("arm").mean().rename("macro_mae").to_frame()
    summary["macro_pctile"] = policies.groupby(["arm", "extractant"])["picked_residual_pctile"] \
        .mean().groupby("arm").mean()
    summary["n_ligands"] = per_lig.groupby("arm").size()
    summary = summary.join(per_seed.add_prefix("seed_"))
    policies["centrality"] = (policies["picked_residual_pctile"] - 0.5).abs()
    summary["macro_centrality"] = policies.groupby(["arm", "extractant"])["centrality"] \
        .mean().groupby("arm").mean()
    summary = summary.sort_values("macro_mae").reset_index()
    summary.to_csv(OUT / "decision_policies.csv", index=False)

    # how well does "how far from the median residual did you pick" explain the
    # policy leaderboard?  One number over the deployable + oracle picking arms.
    picks = summary.dropna(subset=["macro_centrality"])
    # The oracle arms are *defined* by residual extremity, so including them makes
    # the correlation partly circular.  Report it three ways: all picking arms, the
    # deployable ones only, and the deployable row-level ones only.
    deployable = picks[~picks["arm"].str.startswith(("ORACLE", "WORST"))]
    row_level = deployable[~deployable["arm"].str.contains("u_lig_")]
    pd.DataFrame([{
        "n_arms": len(picks),
        "spearman_centrality_vs_macro_mae": rho(picks["macro_centrality"].to_numpy(),
                                                picks["macro_mae"].to_numpy()),
        "n_arms_deployable": len(deployable),
        "spearman_deployable_only": rho(deployable["macro_centrality"].to_numpy(),
                                        deployable["macro_mae"].to_numpy()),
        "n_arms_deployable_row_level": len(row_level),
        "spearman_deployable_row_level": rho(row_level["macro_centrality"].to_numpy(),
                                             row_level["macro_mae"].to_numpy()),
        "random_pick_expected_centrality": 0.25,
    }]).to_csv(OUT / "centrality_explains_leaderboard.csv", index=False)

    arms = [a for a in policies["arm"].unique() if a != "RANDOM(expected)"]
    comparisons = {f"RANDOM - {a}": ("RANDOM(expected)", a) for a in arms}
    boot = paired_chemotype_bootstrap(policies, comparisons, seed=BOOT_SEED)
    boot = boot.sort_values("point", ascending=False)
    boot.to_csv(OUT / "decision_bootstrap_vs_random.csv", index=False)

    rho_frame = pd.DataFrame(rho_rows)
    rho_frame.to_parquet(OUT / "decision_rho_detail.parquet", index=False)
    out = []
    for src in SOURCES:
        sub = rho_frame[rho_frame["source"] == src]
        row = {"source": src, "description": SOURCES[src]}
        for col, tag in [("rho_u_vs_one_shot_mae", "vs_one_shot_mae"),
                         ("rho_u_vs_median_dev", "vs_median_dev"),
                         ("rho_u_vs_abs_resid", "vs_abs_resid")]:
            res = macro_ci(sub, col, tag)
            row[f"rho_{tag}"] = res.get("point", np.nan)
            row[f"rho_{tag}_lo"] = res.get("ci95_low", np.nan)
            row[f"rho_{tag}_hi"] = res.get("ci95_high", np.nan)
        row["n_cells"] = int(sub["rho_u_vs_one_shot_mae"].notna().sum())
        out.append(row)
    pd.DataFrame(out).to_csv(OUT / "decision_spearman.csv", index=False)

    mech = pd.DataFrame(mech_rows)
    mech.to_parquet(OUT / "mechanism_detail.parquet", index=False)
    mrows = []
    for col, tag in [("rho_meddev_vs_one_shot", "abs(r - median r) ranks candidate quality"),
                     ("rho_absresid_vs_one_shot", "abs(r) ranks candidate quality"),
                     ("rho_absresid_vs_meddev", "abs(r) ranks abs(r - median r)")]:
        res = macro_ci(mech, col, tag)
        mrows.append({"quantity": tag, "within_ligand_rho": res.get("point", np.nan),
                      "ci_low": res.get("ci95_low", np.nan),
                      "ci_high": res.get("ci95_high", np.nan),
                      "n_cells": int(mech[col].notna().sum())})
    pd.DataFrame(mrows).to_csv(OUT / "mechanism_spearman.csv", index=False)

    gap = mech["one_shot_gap_of_median_dev_pick"].to_numpy(dtype=float)
    pd.DataFrame([{
        "n_cells": int(len(gap)),
        "max_abs_gap_vs_exhaustive_oracle": float(np.nanmax(np.abs(gap))),
        "n_cells_gap_above_1e_12": int((np.abs(gap) > 1e-12).sum()),
    }]).to_csv(OUT / "mechanism_oracle_equivalence.csv", index=False)

    lvl = mech.groupby("extractant")[["abs_median_residual", "mad_residual", "zero_shot"]].mean()
    pd.DataFrame([{
        "macro_abs_median_residual (level error)": lvl["abs_median_residual"].mean(),
        "macro_mad_residual (shape error, = oracle level)": lvl["mad_residual"].mean(),
        "macro_zero_shot": lvl["zero_shot"].mean(),
        "n_ligands": len(lvl),
    }]).to_csv(OUT / "level_vs_shape.csv", index=False)
    print("decisions written")


# --------------------------------------------------------------------------- #
# (d) the other decision: WHICH LIGAND to spend the measurement on
# --------------------------------------------------------------------------- #
def triage() -> None:
    """Uncertainty as a triage signal over ligands rather than over rows.

    With a budget of M measurements spread over 143 unseen ligands, ranking the
    ligands by mean uncertainty is a different question from ranking the rows
    inside one ligand, and the calibration numbers say it should work.
    """
    merged = pd.read_parquet(OUT / "candidate_uncertainty.parquet")
    rows = []
    for (seed, lig), block in merged.groupby(["split_seed", "extractant"], sort=True):
        gp = block["u_gp_var_design"].to_numpy(dtype=float)
        y = block["one_shot_mae"].to_numpy(dtype=float)
        ok = np.isfinite(gp)
        deployable = float(y[ok][np.argmin(gp[ok])]) if ok.any() else float(np.mean(y))
        rec = {"split_seed": seed, "extractant": lig,
               "tanimoto_cluster": block["tanimoto_cluster"].iloc[0],
               "zero_shot": float(block["zero_shot_mae"].iloc[0]),
               "one_shot_random": float(np.mean(y)),
               "one_shot_deployable": deployable}
        for src in SOURCES:
            rec[src] = float(np.nanmean(block[src].to_numpy(dtype=float)))
        rows.append(rec)
    lig = pd.DataFrame(rows)
    lig["gain_random"] = lig["zero_shot"] - lig["one_shot_random"]
    lig["gain_deployable"] = lig["zero_shot"] - lig["one_shot_deployable"]
    lig.to_csv(OUT / "triage_ligand_table.csv", index=False)

    out = []
    for src in SOURCES:
        per_seed_gain, per_seed_zero = [], []
        for seed, block in lig.groupby("split_seed"):
            per_seed_gain.append(rho(block[src].to_numpy(), block["gain_deployable"].to_numpy()))
            per_seed_zero.append(rho(block[src].to_numpy(), block["zero_shot"].to_numpy()))
        out.append({"source": src, "description": SOURCES[src],
                    "rho_vs_gain_from_one_shot": float(np.nanmean(per_seed_gain)),
                    "seeds_positive_gain": int(np.sum(np.asarray(per_seed_gain) > 0)),
                    "rho_vs_zero_shot_mae": float(np.nanmean(per_seed_zero)),
                    "seeds_positive_zero": int(np.sum(np.asarray(per_seed_zero) > 0))})
    pd.DataFrame(out).to_csv(OUT / "triage_spearman.csv", index=False)

    # budget curve, macro MAE over all 143 ligands
    budgets = [14, 29, 43, 72, 107]
    curve = []
    for M in budgets:
        for src in list(SOURCES) + ["ORACLE_gain", "RANDOM"]:
            per_seed = []
            for seed, block in lig.groupby("split_seed"):
                b = block.reset_index(drop=True)
                n = len(b)
                if src == "RANDOM":
                    frac = M / n
                    per_seed.append(frac * b["one_shot_deployable"].mean()
                                    + (1 - frac) * b["zero_shot"].mean())
                    continue
                key = b["gain_deployable"] if src == "ORACLE_gain" else b[src]
                order = np.argsort(-key.to_numpy(dtype=float), kind="stable")
                chosen = set(order[:M].tolist())
                vals = [b["one_shot_deployable"][i] if i in chosen else b["zero_shot"][i]
                        for i in range(n)]
                per_seed.append(float(np.mean(vals)))
            curve.append({"budget_M": M, "selector": src,
                          "macro_mae": float(np.mean(per_seed)),
                          "n_ligands": int(lig.groupby("split_seed").size().mean())})
    pd.DataFrame(curve).to_csv(OUT / "triage_budget_curve.csv", index=False)

    # paired chemotype bootstrap on the budget decision itself: per ligand, the
    # realised MAE it contributes when the selector does / does not buy it a point.
    boot_rows = []
    for M in (29, 72):
        long = []
        for seed, block in lig.groupby("split_seed"):
            b = block.reset_index(drop=True)
            n = len(b)
            frac = M / n
            rnd = frac * b["one_shot_deployable"] + (1 - frac) * b["zero_shot"]
            long.append(b.assign(arm="RANDOM_budget", mae=rnd))
            for src in list(SOURCES) + ["ORACLE_gain"]:
                key = b["gain_deployable"] if src == "ORACLE_gain" else b[src]
                order = np.argsort(-key.to_numpy(dtype=float), kind="stable")
                chosen = np.zeros(n, dtype=bool)
                chosen[order[:M]] = True
                mae = np.where(chosen, b["one_shot_deployable"], b["zero_shot"])
                long.append(b.assign(arm=src, mae=mae))
        long = pd.concat(long, ignore_index=True)
        arms = [a for a in long["arm"].unique() if a != "RANDOM_budget"]
        res = paired_chemotype_bootstrap(
            long, {f"RANDOM_budget - {a}": ("RANDOM_budget", a) for a in arms},
            seed=BOOT_SEED)
        res["budget_M"] = M
        boot_rows.append(res)
    pd.concat(boot_rows).sort_values(["budget_M", "point"], ascending=[True, False]) \
        .to_csv(OUT / "triage_bootstrap.csv", index=False)
    print("triage written")


def main() -> None:
    unc = pd.read_parquet(OUT / "row_uncertainty.parquet")
    unc["sib_abs_residual"] = unc["sib_residual"].abs()
    calibration(unc)
    decisions(unc)
    triage()


if __name__ == "__main__":
    main()
