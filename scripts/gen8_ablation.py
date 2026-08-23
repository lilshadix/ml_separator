"""How many ligand-specific degrees of freedom can safely be adapted at each k? (§7)

gen7 found that an unrestricted affine recalibration at k = 2 is catastrophic — it
scored 3.32 against 0.64 for a plain offset — and concluded that only the level
should move.  That conclusion conflated two things: *how many* parameters move and
*how hard they are held down*.  This ablation separates them, sweeping the ridge
penalty on the non-intercept coefficients across three adaptation modes and every
k, so the answer is a measured surface rather than a rule of thumb.

The intercept is never penalised in any cell of the sweep, which is why the k = 1
column is identical across modes: with one observation and a free intercept the
penalised optimum puts everything into the level and every slope to zero, exactly
as it should.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lanthanide_separation.gen8.adapters import NoModel, RidgeOffset, ZeroShot  # noqa: E402
from lanthanide_separation.gen8.evaluate import evaluate_fewshot, summarise  # noqa: E402
from lanthanide_separation.gen8.inference import paired_chemotype_bootstrap  # noqa: E402
from lanthanide_separation.gen8.report import md_table  # noqa: E402
from gen8_run import DISAGREEMENT_MODELS, SEEDS, make_fold_trainer  # noqa: E402

OUT = REPO_ROOT / "runs" / "gen8_architecture" / "ablations"
PENALTIES = (0.0, 0.25, 1.0, 4.0, 16.0, 64.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--model", default="REC_ecfp_plus_recovered")
    parser.add_argument("--policies", nargs="*", default=["RANDOM", "CENTRAL_THEN_SPREAD"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    seeds = SEEDS[: max(1, min(5, args.seeds))]
    oof = pd.read_parquet(REPO_ROOT / "runs/gen7_architecture/finalists/oof_predictions.parquet")
    disagreement = (oof[oof["model"].isin(DISAGREEMENT_MODELS)]
                    .groupby(["row_id", "split_seed"])["prediction"].std().rename("disagreement"))
    work = oof[(oof["model"] == args.model) & (oof["split_seed"].isin(seeds))].merge(
        disagreement, on=["row_id", "split_seed"], how="left")
    cohort = pd.read_parquet(REPO_ROOT / "runs/gen7_architecture/cache/cohort.parquet")

    adapters = [ZeroShot(), NoModel()]
    for mode in ("K1", "K2", "K3"):
        for penalty in PENALTIES:
            tag = f"{penalty:g}".replace(".", "p")
            adapters.append(RidgeOffset(mode=mode, penalty=penalty, name=f"{mode}_lam{tag}"))
    print(f"{len(adapters)} adapters", flush=True)

    started = time.time()
    detail = evaluate_fewshot(work, cohort, adapters, policies=args.policies,
                              with_oracle_for=(), repeats=args.repeats,
                              fold_trainer=make_fold_trainer(cohort))
    print(f"{len(detail):,} records in {time.time() - started:.0f}s")
    detail.to_parquet(args.out / "dof_ablation_detail.parquet", index=False)

    summary = summarise(detail, keys=("adapter", "policy", "k"))
    summary.to_csv(args.out / "dof_ablation_summary.csv", index=False)
    pd.set_option("display.width", 240)
    for policy in args.policies:
        sub = summary[summary["policy"] == policy]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="adapter", columns="k", values="mae")
        print(f"\n--- {policy}")
        print(pivot.to_string())

    lines = ["# How many degrees of freedom can k measurements safely move?", "",
             "*Ridge penalty on the non-intercept coefficients, swept across three adaptation "
             "modes. `lam0` is the unrestricted least-squares fit gen7 found catastrophic; "
             "the intercept is unpenalised everywhere.*", ""]
    for policy in args.policies:
        sub = summary[summary["policy"] == policy]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="adapter", columns="k", values="mae").reset_index()
        pivot.columns = [str(c) for c in pivot.columns]
        lines += [f"## Selection policy: `{policy}`", "", md_table(pivot), ""]
    (args.out / "degrees_of_freedom.md").write_text("\n".join(lines) + "\n")

    contrasts = {}
    for mode in ("K2", "K3"):
        for penalty in PENALTIES:
            tag = f"{penalty:g}".replace(".", "p")
            contrasts[f"{mode}_lam{tag} vs K1_lam4"] = ("K1_lam4", f"{mode}_lam{tag}")
    boots = []
    for k in (2, 3, 5):
        sub = detail[(detail["k"] == k) & (detail["policy"] == "RANDOM")]
        if sub.empty:
            continue
        frame = paired_chemotype_bootstrap(sub.rename(columns={"adapter": "arm"}), contrasts,
                                           arm_column="arm")
        frame.insert(0, "k", k)
        boots.append(frame)
    if boots:
        allboot = pd.concat(boots, ignore_index=True)
        allboot.to_csv(args.out / "dof_bootstrap.csv", index=False)
        print("\npaired chemotype bootstrap vs K1 at lambda=4 (positive = better):")
        print(allboot.to_string(index=False))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
