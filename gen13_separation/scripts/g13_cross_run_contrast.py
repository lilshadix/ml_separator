"""Paired contrast of arms across two runs that share the frozen fold plan (S5 / H3 and helpers).

    .venv/Scripts/python.exe gen13_separation/scripts/g13_cross_run_contrast.py \
        --candidate B_primary:M_SELECTED --reference B_abl_cond_only:M_SELECTED --name S5_all_blocks_vs_cond_only

Both runs must have identical (split_seed, fold, cell_id, A, B) pair keys; the script asserts it.
Writes bootstrap/cross_run/<name>.csv with the same columns as paired_contrasts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen13sep import paths  # noqa: E402
from gen13sep.inference import paired_contrasts  # noqa: E402
from gen13sep.metrics import per_extractant  # noqa: E402

KEY = ["split_seed", "fold", "cell_id", "extractant", "chemotype", "n_metals", "A", "B", "dZ", "y"]


def load_arm(label: str, arm: str, alias: str) -> pd.DataFrame:
    t = pd.read_parquet(paths.PREDICTION_DIR / label / f"{arm}.parquet")
    return t.rename(columns={"prediction": alias})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="label:arm")
    ap.add_argument("--reference", required=True, help="label:arm")
    ap.add_argument("--name", required=True)
    ap.add_argument("--replicates", type=int, default=10_000)
    args = ap.parse_args()
    cl, ca = args.candidate.split(":"); rl, ra = args.reference.split(":")
    cand = load_arm(cl, ca, "CAND"); ref = load_arm(rl, ra, "REF")
    merged = cand.merge(ref, on=KEY, how="inner", validate="one_to_one")
    if len(merged) != len(cand) or len(merged) != len(ref):
        raise SystemExit(f"pair keys differ between runs: {len(cand)} / {len(ref)} / merged {len(merged)}")
    per_ext = per_extractant(merged, ["CAND", "REF"])
    tables = []
    for value in ("mae_all", "mae_far", "mae_adjacent", "sign_acc_strong", "pair_spearman"):
        t = paired_contrasts(per_ext.dropna(subset=[value]), {args.name: ("REF", "CAND")}, value=value,
                             replicates=args.replicates)
        t["candidate"] = args.candidate; t["reference"] = args.reference
        tables.append(t)
    out = pd.concat(tables, ignore_index=True)
    out_dir = paths.BOOTSTRAP_DIR / "cross_run"; out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"{args.name}.csv", index=False)
    print(out[["comparison", "value", "point", "ci95_low", "ci95_high", "bca_low", "bca_high", "p_two_sided", "mde_80",
               "seeds_positive", "n_seeds", "n_units", "loco_sign_stable", "passes_P1"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
