"""The gen8 primary tables: adaptation curve, acquisition value, oracle gap accounting.

Reads the paired detail written by ``scripts/gen8_run.py`` (and any adapter run that
shares the same seeds and repeats, which is automatically paired because the
pool/evaluation split is a deterministic function of seed, repeat, ligand and row
count) and produces:

* **Table A** — maximal coverage: every ligand eligible at each k.
* **Table B** — the common cohort: the *same* ligands at every k, which is the only
  honest longitudinal reading of the adaptation curve (brief §16).
* the §21 primary comparison, with the full metric panel;
* the paired chemotype bootstrap for every contrast that carries a claim;
* the oracle-gap decomposition (brief §23) and the marginal value of each extra
  measurement (brief §24).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen8.inference import paired_chemotype_bootstrap  # noqa: E402
from lanthanide_separation.gen8.report import md_table  # noqa: E402

ROOT = REPO_ROOT / "runs" / "gen8_architecture"
OUT = ROOT / "finalists"
#: A ligand is in the common cohort if, in **every** repeat, its candidate pool held
#: at least this many rows and its evaluation set at least two — so the same ligands
#: are scored at k = 1 and at k = 5 and the curve is longitudinal.
COMMON_MIN_POOL = 5
METRICS = ("mae", "offset", "shape_mae", "spearman", "sign_accuracy", "within_0_5", "within_1_0")


def load(paths) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not Path(path).exists():
            print(f"  (missing: {path})")
            continue
        frame = pd.read_parquet(path)
        frame["source"] = Path(path).stem
        frames.append(frame)
    if not frames:
        raise SystemExit("no detail files found")
    detail = pd.concat(frames, ignore_index=True)
    # Adapters and policies can appear in more than one run; keep one copy of each
    # (arm, unit, repeat) so a duplicated arm cannot double its weight.
    key = ["split_seed", "fold", "extractant", "repeat", "policy", "adapter", "k"]
    return detail.drop_duplicates(key, keep="first")


def arm_label(row) -> str:
    if row["adapter"] == "ZERO_SHOT_REF":
        return "ZERO_SHOT"
    return f"{row['adapter']}@{row['policy']}"


def common_cohort(detail: pd.DataFrame) -> set:
    """Ligands whose pool is large enough for k = 5 in every repeat of every seed."""
    per = detail.groupby(["extractant", "split_seed", "repeat"])[["n_pool", "n_eval"]].first()
    ok = per.groupby("extractant").agg(min_pool=("n_pool", "min"), min_eval=("n_eval", "min"))
    return set(ok[(ok["min_pool"] >= COMMON_MIN_POOL) & (ok["min_eval"] >= 2)].index)


def available(detail: pd.DataFrame) -> list[str]:
    """Metrics actually present.  A detail file written before a metric existed is
    still usable for every metric it does carry, rather than failing the whole
    analysis — which matters because runs are merged across sessions."""
    return [m for m in METRICS if m in detail.columns]


def table(detail: pd.DataFrame, *, ligands: set | None = None) -> pd.DataFrame:
    work = detail if ligands is None else detail[detail["extractant"].isin(ligands)]
    work = work.assign(arm=work.apply(arm_label, axis=1))
    metrics = available(work)
    per_ligand = work.groupby(["arm", "k", "extractant"])[metrics].mean().reset_index()
    out = per_ligand.groupby(["arm", "k"]).agg(
        **{m: (m, "mean") for m in metrics},
        n_ligands=("extractant", "nunique")).reset_index()
    worst = per_ligand.groupby(["arm", "k"])["mae"].quantile(0.75).rename("worst_quartile_mae")
    out = out.merge(worst, on=["arm", "k"])
    hard = work[work["nn_train_tanimoto"] < 0.4]
    if len(hard):
        hard_per = hard.groupby(["arm", "k", "extractant"])["mae"].mean().reset_index()
        hard_out = hard_per.groupby(["arm", "k"]).agg(
            hard_mae=("mae", "mean"), n_hard=("extractant", "nunique")).reset_index()
        out = out.merge(hard_out, on=["arm", "k"], how="left")
    return out.sort_values(["k", "mae"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_detail = [
        str(ROOT / "active_acquisition" / "primary_detail.parquet"),
        str(ROOT / "functional" / "slope_restore_detail.parquet"),
        str(ROOT / "functional" / "finalists_slope_detail.parquet"),
        str(ROOT / "physics_latent" / "physics_latent_detail_allseeds.parquet"),
    ]
    # The CNP and curve-baseline runners shard by split seed; every shard is paired with
    # the primary run because the pool/evaluation split is a pure function of
    # (seed, repeat, ligand, n_rows) — which is asserted in tests/test_gen8_harness.py.
    default_detail += sorted(str(x) for x in (ROOT / "cnp").glob("cnp_detail_final_*.parquet"))
    default_detail += sorted(str(x) for x in (ROOT / "baselines").glob("detail_f*.parquet"))
    parser.add_argument("--detail", nargs="*", default=default_detail)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    detail = load(args.detail)
    print(f"{len(detail):,} records / {detail['extractant'].nunique()} ligands / "
          f"{detail['adapter'].nunique()} adapters / {detail['policy'].nunique()} policies")

    cohort = common_cohort(detail)
    table_a = table(detail)
    table_b = table(detail, ligands=cohort)
    table_a.to_csv(args.out / "table_a_maximal_coverage.csv", index=False)
    table_b.to_csv(args.out / "table_b_common_cohort.csv", index=False)
    print(f"common cohort: {len(cohort)} ligands of {detail['extractant'].nunique()}")

    # ---- leaderboard: best arm per k, on the common cohort ------------------ #
    board = table_b.sort_values(["k", "mae"])
    board.to_csv(args.out / "leaderboard_all.csv", index=False)
    # Also at the run root: the brief names these two as top-level artefacts, and a
    # reader should not have to know which subdirectory produced them.
    board.to_csv(ROOT / "leaderboard_all.csv", index=False)

    # ---- scores by seed ----------------------------------------------------- #
    work = detail.assign(arm=detail.apply(arm_label, axis=1))
    per_seed = work.groupby(["arm", "k", "split_seed", "extractant"])[available(work)].mean()
    per_seed = per_seed.groupby(["arm", "k", "split_seed"]).mean().reset_index()
    per_seed.to_csv(args.out / "scores_by_seed.csv", index=False)
    per_seed.to_csv(ROOT / "scores_by_seed.csv", index=False)

    # ---- the §21 primary comparison ---------------------------------------- #
    def pick(arm: str, k: int, source: pd.DataFrame = table_b):
        row = source[(source["arm"] == arm) & (source["k"] == k)]
        return row.iloc[0] if len(row) else None

    wanted = [
        ("ZERO_SHOT current best", "ZERO_SHOT", 0),
        ("RANDOM_1SHOT (offset correction)", "OFFSET_K1@RANDOM", 1),
        ("RANDOM_2SHOT (offset correction)", "OFFSET_K1@RANDOM", 2),
        ("RANDOM_2SHOT (response coefficients)", "OFFSET_K3@RANDOM", 2),
        ("BEST_FIXED_1SHOT_POLICY (CENTRAL)", "OFFSET_K1@CENTRAL", 1),
        ("ACTIVE_1SHOT (MEDOID)", "OFFSET_K1@MEDOID", 1),
        ("ACTIVE_2SHOT (MEDOID)", "OFFSET_K1@MEDOID", 2),
        ("ACTIVE_2SHOT (MEDOID + K3)", "OFFSET_K3@MEDOID", 2),
        ("BEST_1SHOT (slope repair + medoid)", "SLOPE_L_s1_K1@MEDOID", 1),
        ("BEST_2SHOT (slope repair + K3 + max predictive variance)",
         "SLOPE_L_s1_K3@MAX_PREDICTIVE_VARIANCE", 2),
        ("BEST_3SHOT (slope repair + K3 + central-then-spread)",
         "SLOPE_L_s1_K3@CENTRAL_THEN_SPREAD", 3),
        ("BEST_5SHOT (slope repair + K3 + D-optimal)", "SLOPE_L_s1_K3@D_OPTIMAL", 5),
        ("NO_MODEL_1SHOT (null)", "NO_MODEL@RANDOM", 1),
        ("NO_MODEL_2SHOT (null)", "NO_MODEL@RANDOM", 2),
        ("ORACLE_1SHOT", "OFFSET_K1@ORACLE[OFFSET_K1]", 1),
        ("ORACLE_2SHOT", "OFFSET_K1@ORACLE[OFFSET_K1]", 2),
        ("ORACLE_2SHOT (response coefficients)", "OFFSET_K3@ORACLE[OFFSET_K1]", 2),
    ]
    rows = []
    for label, arm, k in wanted:
        row = pick(arm, k)
        if row is None:
            continue
        rows.append({"comparison": label, "arm": arm, "k": k,
                     **{m: float(row[m]) for m in METRICS if m in row.index},
                     "worst_quartile_mae": float(row["worst_quartile_mae"]),
                     "hard_mae": float(row.get("hard_mae", np.nan)),
                     "n_ligands": int(row["n_ligands"])})
    primary = pd.DataFrame(rows)
    primary.to_csv(args.out / "primary_comparison.csv", index=False)

    # ---- paired bootstrap --------------------------------------------------- #
    contrasts_by_k = {
        1: {"CENTRAL vs RANDOM": ("OFFSET_K1@RANDOM", "OFFSET_K1@CENTRAL"),
            "MEDOID vs RANDOM": ("OFFSET_K1@RANDOM", "OFFSET_K1@MEDOID"),
            "MAX_ENSEMBLE_SD vs RANDOM": ("OFFSET_K1@RANDOM", "OFFSET_K1@MAX_ENSEMBLE_SD"),
            "FARTHEST vs RANDOM": ("OFFSET_K1@RANDOM", "OFFSET_K1@FARTHEST_FROM_EXISTING"),
            "ORACLE vs RANDOM": ("OFFSET_K1@RANDOM", "OFFSET_K1@ORACLE[OFFSET_K1]"),
            "ORACLE vs CENTRAL": ("OFFSET_K1@CENTRAL", "OFFSET_K1@ORACLE[OFFSET_K1]"),
            "one measurement vs zero": ("ZERO_SHOT", "OFFSET_K1@RANDOM"),
            "offset vs no model": ("NO_MODEL@RANDOM", "OFFSET_K1@RANDOM"),
            "slope repair + medoid vs random offset":
                ("OFFSET_K1@RANDOM", "SLOPE_L_s1_K1@MEDOID"),
            "slope repair vs offset, both at medoid":
                ("OFFSET_K1@MEDOID", "SLOPE_L_s1_K1@MEDOID"),
            "physics-latent residual vs offset, both at CENTRAL":
                ("OFFSET_K1@CENTRAL", "PHYS_residual_gbm@CENTRAL"),
            "physics-latent residual vs slope repair, both at CENTRAL":
                ("SLOPE_L_s1_K1@CENTRAL_THEN_SPREAD", "PHYS_residual_gbm@CENTRAL"),
            "physics law on the absolute value vs offset (RANDOM)":
                ("OFFSET_K1@RANDOM", "PHYS_map@RANDOM"),
            "residual CNP vs offset (RANDOM)":
                ("OFFSET_K1@RANDOM", "CNPRES_shrinkage@RANDOM"),
            "absolute CNP vs offset (RANDOM)":
                ("OFFSET_K1@RANDOM", "CNP_conditions_only@RANDOM"),
            "CNP: adding a ligand representation (RANDOM)":
                ("CNPRES_conditions_only@RANDOM", "CNPRES_ligand@RANDOM"),
            "absolute CNP: adding a ligand representation (RANDOM)":
                ("CNP_conditions_only@RANDOM", "CNP_ligand@RANDOM"),
            "mass-action curve baseline vs offset (RANDOM)":
                ("OFFSET_K1@RANDOM", "MASSACTION_K1@RANDOM"),
            "spline residual baseline vs offset (RANDOM)":
                ("OFFSET_K1@RANDOM", "CURVE_SPLINE_K1@RANDOM"),
            "local GP curve vs offset (RANDOM)":
                ("OFFSET_K1@RANDOM", "CURVE_GP@RANDOM")},
        2: {"K3 vs K1 at RANDOM": ("OFFSET_K1@RANDOM", "OFFSET_K3@RANDOM"),
            "K2 vs K1 at RANDOM": ("OFFSET_K1@RANDOM", "OFFSET_K2@RANDOM"),
            "MEDOID vs RANDOM": ("OFFSET_K1@RANDOM", "OFFSET_K1@MEDOID"),
            "FARTHEST vs RANDOM (K3)": ("OFFSET_K3@RANDOM", "OFFSET_K3@FARTHEST_FROM_EXISTING"),
            "MAX_PRED_VAR vs RANDOM (K3)": ("OFFSET_K3@RANDOM", "OFFSET_K3@MAX_PREDICTIVE_VARIANCE"),
            "best deployable vs random offset":
                ("OFFSET_K1@RANDOM", "SLOPE_L_s1_K3@MAX_PREDICTIVE_VARIANCE"),
            "slope repair vs offset, both K3 + max predictive variance":
                ("OFFSET_K3@MAX_PREDICTIVE_VARIANCE", "SLOPE_L_s1_K3@MAX_PREDICTIVE_VARIANCE"),
            "ORACLE vs RANDOM": ("OFFSET_K1@RANDOM", "OFFSET_K1@ORACLE[OFFSET_K1]"),
            "offset vs no model": ("NO_MODEL@RANDOM", "OFFSET_K1@RANDOM")},
        5: {"K3 vs K1 at RANDOM": ("OFFSET_K1@RANDOM", "OFFSET_K3@RANDOM"),
            "offset vs no model": ("NO_MODEL@RANDOM", "OFFSET_K1@RANDOM"),
            "best deployable vs random offset":
                ("OFFSET_K1@RANDOM", "SLOPE_L_s1_K3@D_OPTIMAL"),
            "best deployable vs the 1-shot oracle":
                ("OFFSET_K1@ORACLE[OFFSET_K1]", "SLOPE_L_s1_K3@D_OPTIMAL")},
    }
    boots = []
    for k, contrasts in contrasts_by_k.items():
        sub = work[(work["k"] == k) & (work["extractant"].isin(cohort))]
        zero = work[(work["adapter"] == "ZERO_SHOT_REF") & (work["extractant"].isin(cohort))]
        sub = pd.concat([sub, zero.assign(k=k)], ignore_index=True)
        if sub.empty:
            continue
        for statistic in [s for s in ("mae", "shape_mae", "offset") if s in sub.columns]:
            frame = paired_chemotype_bootstrap(sub, contrasts, arm_column="arm",
                                               value_column=statistic)
            frame.insert(0, "k", k)
            frame.insert(1, "statistic", statistic)
            boots.append(frame)
    bootstrap = pd.concat(boots, ignore_index=True) if boots else pd.DataFrame()
    bootstrap.to_csv(args.out / "primary_bootstrap.csv", index=False)

    # ---- adaptation curve and marginal value -------------------------------- #
    curve_arms = ["OFFSET_K1@RANDOM", "OFFSET_K1@MEDOID", "OFFSET_K3@RANDOM",
                  "OFFSET_K3@MEDOID", "NO_MODEL@RANDOM", "OFFSET_K1@ORACLE[OFFSET_K1]"]
    curve = table_b[table_b["arm"].isin(curve_arms)].pivot_table(
        index="arm", columns="k", values="mae")
    zero = table_b[table_b["arm"] == "ZERO_SHOT"]["mae"]
    if len(zero):
        curve.insert(0, 0, float(zero.iloc[0]))
    marginal = pd.DataFrame({
        "step": ["0->1", "1->2", "2->3", "3->5"],
        **{arm: [curve.loc[arm].get(a, np.nan) - curve.loc[arm].get(b, np.nan)
                 for a, b in ((0, 1), (1, 2), (2, 3), (3, 5))]
           for arm in curve.index}})
    curve.to_csv(args.out / "adaptation_curve.csv")
    marginal.to_csv(args.out / "marginal_value.csv", index=False)

    pd.set_option("display.width", 260)
    print("\n=== §21 PRIMARY COMPARISON (common cohort) ===")
    print(primary.to_string(index=False))
    print("\n=== ADAPTATION CURVE (common cohort) ===")
    print(curve.to_string())
    print("\n=== MARGINAL VALUE ===")
    print(marginal.to_string(index=False))
    if not bootstrap.empty:
        print("\n=== PAIRED CHEMOTYPE BOOTSTRAP (mae) ===")
        print(bootstrap[bootstrap["statistic"] == "mae"].to_string(index=False))

    (args.out / "analysis.json").write_text(json.dumps({
        "n_records": int(len(detail)), "n_ligands": int(detail["extractant"].nunique()),
        "common_cohort_size": len(cohort), "sources": args.detail,
        "common_min_pool": COMMON_MIN_POOL,
    }, indent=2))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
