#!/usr/bin/env python
"""Read gen8 CNP detail parquets and answer the one question they were run for.

Two tables come out.

*The level table* is ``evaluate.summarise`` — macro MAE per (adapter, k), one
ligand one vote, which is the number the study reports.

*The paired table* is the one that decides anything.  Every adapter was scored on
the identical ligand, repeat, pool and evaluation rows as ``OFFSET_K1``, so the
difference can be taken **per ligand** and the between-ligand spread — which is
several times any method effect here — cancels out.  A macro-MAE column alone
cannot tell a 3% method difference from a 3% difference in which ligands happened
to land in a fold; a paired difference with a bootstrap interval over ligands can.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lanthanide_separation.gen8.evaluate import summarise  # noqa: E402

BASELINE = "OFFSET_K1"


def paired(detail: pd.DataFrame, *, baseline: str = BASELINE, policy: str = "RANDOM",
           draws: int = 2000, seed: int = 20260820) -> pd.DataFrame:
    """Per-ligand paired difference against ``baseline``, bootstrapped over ligands."""
    frame = detail[detail["policy"] == policy]
    keys = ["split_seed", "fold", "extractant", "repeat", "k"]
    reference = frame[frame["adapter"] == baseline][keys + ["mae"]].rename(
        columns={"mae": "baseline_mae"})
    merged = frame.merge(reference, on=keys, how="inner", validate="many_to_one")
    merged["difference"] = merged["mae"] - merged["baseline_mae"]

    per_ligand = merged.groupby(["adapter", "k", "extractant"])[
        ["mae", "baseline_mae", "difference"]].mean().reset_index()
    rng = np.random.default_rng(seed)
    rows = []
    for (adapter, k), block in per_ligand.groupby(["adapter", "k"]):
        values = block["difference"].to_numpy(dtype=float)
        n = len(values)
        boot = values[rng.integers(0, n, size=(draws, n))].mean(axis=1) if n > 1 \
            else np.full(draws, values.mean())
        low, high = np.percentile(boot, [2.5, 97.5])
        rows.append({"adapter": adapter, "k": int(k), "n_ligands": n,
                     "mae": float(block["mae"].mean()),
                     "baseline_mae": float(block["baseline_mae"].mean()),
                     "delta": float(values.mean()), "ci_low": float(low), "ci_high": float(high),
                     "win_rate": float((values < 0).mean()),
                     "beats_baseline": bool(high < 0.0)})
    return pd.DataFrame(rows).sort_values(["k", "delta"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--policy", default="RANDOM")
    parser.add_argument("--baselines", default="OFFSET_K1,OFFSET_K3")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    detail = pd.concat([pd.read_parquet(p) for p in args.paths], ignore_index=True)
    print(f"{len(detail):,} records, seeds {sorted(detail['split_seed'].unique())}, "
          f"{detail['extractant'].nunique()} ligands, policies "
          f"{sorted(detail['policy'].unique())}\n")

    zero = detail[detail["adapter"] == "ZERO_SHOT_REF"]
    per = zero.groupby("extractant")["mae"].mean()
    print(f"zero-shot macro MAE on the evaluation rows: {per.mean():.4f} "
          f"({per.size} ligands)\n")

    level = summarise(detail[detail["policy"].isin([args.policy, "NONE"])])
    grid = level.pivot_table(index="adapter", columns="k", values="mae")
    # A trained adapter's k = 0 number lives on its "@k0" view, which returns the
    # same empty-context prediction whatever k the driver asked for.  Fold it into
    # the k = 0 column of the parent so one table carries k = 0...5 for every arm.
    zero_column = {name[:-3]: row.get(1, np.nan)
                   for name, row in grid.iterrows() if name.endswith("@k0")}
    grid[0] = [zero_column.get(name, grid.loc[name].get(0, np.nan)) for name in grid.index]
    grid = grid.loc[[n for n in grid.index if not n.endswith("@k0")]]
    grid = grid[sorted(grid.columns)].sort_values(1)

    tables = {}
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print("=== macro MAE, one ligand one vote (k=0 taken from the @k0 view) ===")
        print(grid.round(4).to_string())
        for baseline in [b for b in args.baselines.split(",") if b]:
            print(f"\n=== paired per-ligand difference against {baseline} ===")
            table = paired(detail, baseline=baseline, policy=args.policy)
            table = table[~table["adapter"].str.endswith("@k0")]
            tables[baseline] = table
            print(table.round(4).to_string(index=False))
    if args.out:
        pd.concat([t.assign(baseline=b) for b, t in tables.items()]).to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
