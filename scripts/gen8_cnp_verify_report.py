#!/usr/bin/env python
"""Compare verification runs against the reported CNP result.

Three questions, three tables.

1. *Does it reproduce?*  Merge the re-run detail on the record key and compare
   ``mae`` to the stored parquet.  Non-CNP adapters (``OFFSET_*``, ``NO_MODEL``,
   ``NEAREST_OBSERVED``) do not depend on any training and must agree to the bit;
   if they do not, the harness itself differs and nothing else is comparable.
2. *Does poisoning the target column move anything?*  Same merge, honest against
   poisoned.  A single differing record is a leak.
3. *Does the network beat the two-parameter version of its own claim?*  The paired
   per-ligand table, with ``SHRINK_FITTED`` and ``SHRINK_ANALYTIC`` in it.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lanthanide_separation.gen8.evaluate import summarise  # noqa: E402

KEYS = ["split_seed", "fold", "extractant", "repeat", "policy", "adapter", "k"]


def load(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    assert paths, pattern
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def compare(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    merged = left.merge(right, on=KEYS, how="outer", suffixes=("_a", "_b"), indicator=True)
    print(f"\n=== {label} ===")
    print(f"  records: left {len(left):,} right {len(right):,} merged {len(merged):,}")
    only = merged["_merge"].value_counts().to_dict()
    print(f"  key coverage: {only}")
    both = merged[merged["_merge"] == "both"].copy()
    both["d"] = (both["mae_a"] - both["mae_b"]).abs()
    per = both.groupby("adapter")["d"].agg(["max", "mean", "size"])
    per["identical"] = per["max"] == 0.0
    print(per.sort_values("max", ascending=False).to_string())
    print(f"  GLOBAL max |delta mae| = {both['d'].max():.3e} over {len(both):,} records")


def paired(detail: pd.DataFrame, baseline: str, policy: str = "RANDOM",
           draws: int = 4000, seed: int = 20260820) -> pd.DataFrame:
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
        boot = values[rng.integers(0, n, size=(draws, n))].mean(axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
        rows.append({"adapter": adapter, "k": int(k), "n_ligands": n,
                     "mae": float(block["mae"].mean()),
                     "baseline_mae": float(block["baseline_mae"].mean()),
                     "delta": float(values.mean()), "ci_low": float(low), "ci_high": float(high),
                     "win_rate": float((values < 0).mean()),
                     "beats": bool(high < 0.0)})
    return pd.DataFrame(rows).sort_values(["k", "delta"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--honest", default=str(ROOT / "runs/gen8_architecture/cnp_verify/detail_honest_*.parquet"))
    parser.add_argument("--stored", default=str(ROOT / "runs/gen8_architecture/cnp/cnp_detail_final_*.parquet"))
    parser.add_argument("--poison", default=str(ROOT / "runs/gen8_architecture/cnp_verify/detail_poison_104729.parquet"))
    args = parser.parse_args()

    honest = load(args.honest)
    stored = load(args.stored)
    poison = load(args.poison)

    shared = sorted(set(honest["adapter"]) & set(stored["adapter"]))
    compare(honest[honest["adapter"].isin(shared)], stored[stored["adapter"].isin(shared)],
            "REPRODUCTION: re-run vs the reported parquets")
    seed = int(poison["split_seed"].iloc[0])
    compare(honest[(honest["split_seed"] == seed)], poison,
            f"LEAKAGE: honest vs target-poisoned block (seed {seed})")

    print("\n=== macro MAE, one ligand one vote (verification re-run, all seeds) ===")
    level = summarise(honest[honest["policy"].isin(["RANDOM", "NONE"])])
    grid = level.pivot_table(index="adapter", columns="k", values="mae")
    zero = {n[:-3]: r.get(1, np.nan) for n, r in grid.iterrows() if n.endswith("@k0")}
    grid[0] = [zero.get(n, grid.loc[n].get(0, np.nan)) for n in grid.index]
    grid = grid.loc[[n for n in grid.index if not n.endswith("@k0")]]
    grid = grid[sorted(grid.columns)].sort_values(1)
    print(grid.round(4).to_string())
    counts = level.pivot_table(index="adapter", columns="k", values="n_ligands")
    print("\nn_ligands:")
    print(counts.loc[[n for n in counts.index if not n.endswith("@k0")]].to_string())

    for baseline in ("OFFSET_K1", "SHRINK_FITTED", "OFFSET_K3"):
        print(f"\n=== paired per-ligand difference vs {baseline} ===")
        table = paired(honest, baseline)
        table = table[~table["adapter"].str.endswith("@k0")]
        print(table.round(4).to_string(index=False))

    print("\n=== per-seed k=1 delta of each arm vs OFFSET_K1 ===")
    frame = honest[(honest["policy"] == "RANDOM") & (honest["k"] == 1)]
    keys = ["split_seed", "fold", "extractant", "repeat"]
    ref = frame[frame["adapter"] == "OFFSET_K1"][keys + ["mae"]].rename(
        columns={"mae": "base"})
    merged = frame.merge(ref, on=keys, validate="many_to_one")
    merged["d"] = merged["mae"] - merged["base"]
    per = merged.groupby(["adapter", "split_seed", "extractant"])["d"].mean().reset_index()
    table = per.groupby(["adapter", "split_seed"])["d"].mean().unstack()
    interesting = [a for a in table.index
                   if a.startswith(("CNPRES_shrink", "SHRINK", "OFFSET", "CNPRES_conditions"))]
    print(table.loc[interesting].round(4).to_string())

    print("\n=== folds where an arm is bit-identical to OFFSET_K1 at k=1 "
          "(= kept its initialisation) ===")
    pivot = frame.pivot_table(index=keys, columns="adapter", values="mae")
    for adapter in [a for a in pivot.columns if a.startswith("CNPRES") and not a.endswith("@k0")]:
        same = pd.Series(np.isclose(pivot[adapter].to_numpy(),
                                    pivot["OFFSET_K1"].to_numpy(), atol=1e-10),
                         index=pivot.index)
        by_fold = same.groupby(level=[0, 1]).mean()
        print(f"  {adapter:26s} {(by_fold > 0.99).sum():>2d}/{len(by_fold)} folds identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
